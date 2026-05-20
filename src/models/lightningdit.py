"""
Lightning DiT's codes are built from original DiT & SiT.
(https://github.com/facebookresearch/DiT; https://github.com/willisma/SiT)
It demonstrates that a advanced DiT together with advanced diffusion skills
could also achieve a very promising result with 1.35 FID on ImageNet 256 generation.

Enjoy everyone, DiT strikes back!

by Maple (Jingfeng Yao) from HUST-VL
"""

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import PatchEmbed, Mlp
from models.swiglu_ffn import SwiGLUFFN 
from models.pos_embed import VisionRotaryEmbeddingFast
from models.rmsnorm import RMSNorm
from einops import rearrange

from utils.mm_logging import MMLogger
import logging

try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func, flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    FLASH_ATTN_AVAILABLE = True
except:
    print('[WARN] flash_attn not available, using naive implementation')
    FLASH_ATTN_AVAILABLE = False

@torch.compile
def modulate(x, shift, scale):
    if shift is None and scale is None:
        return x
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# flashattn 2.7.0 changes unpad_input API... we are overriding it here
@torch._dynamo.disable()
def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.detach().max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def attention_flashattn_qkvpacked(qkv, mask=None, dropout=0, causal=False, window_size=(-1, -1)):
    # qkv: (B, N, 3, H, D)
    # mask_qkv: (B, N)
    # return: (B, N, H, D)

    B, N, _, H, D = qkv.shape

    ### unmasked case (usually inference)
    ### will ignore window_size except flash-attn impl. Only provide the effective window!
    if mask is None:
        return flash_attn_qkvpacked_func(qkv, dropout, causal=causal, window_size=window_size) # [B, N, H, D]
    
    # unpad (gather) input
    # mask: [B, N], first row has N1 1s, second row has N2 1s, ...
    # indices: [Ns,], Ns = N1 + N2 + ...
    # cu_seqlens: [B+1,], (0, N1, N1+N2, ...), cu=cumulative
    # max_len: scalar, max(N1, N2, ...)
    qkv, indices, cu_seqlens, max_len = unpad_input(qkv, mask)
    # call varlen_func
    out = flash_attn_varlen_qkvpacked_func(
        qkv,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_len,
        dropout_p=dropout,
        causal=causal,
        window_size=window_size,
    )

    # pad (put back) output
    out = pad_input(out, indices, B, N)
    return out

def attention_flashattn(q, k, v, mask_q=None, mask_kv=None, dropout=0, causal=False, window_size=(-1, -1)):
    # q: (B, N, H, D)
    # k: (B, M, H, D)
    # v: (B, M, H, D)
    # mask_q: (B, N)
    # mask_kv: (B, M)
    # return: (B, N, H, D)

    B, N, H, D = q.shape
    M = k.shape[1]

    # kiui.lo(q, k, v)

    if causal: 
        assert N == 1 or N == M, 'Causal mask only supports self-attention'

    ### unmasked case (usually inference)
    ### will ignore window_size except flash-attn impl. Only provide the effective window!
    if mask_q is None and mask_kv is None:
        return flash_attn_func(q, k, v, dropout, causal=causal, window_size=window_size) # [B, N, H, D]
    
    ### at least one of q or kv is masked (training)
    ### only support flash-attn for now...
    if mask_q is None:
        mask_q = torch.ones(B, N, dtype=torch.bool, device=q.device)
    elif mask_kv is None:
        mask_kv = torch.ones(B, M, dtype=torch.bool, device=q.device)

    # unpad (gather) input
    # mask_q: [B, N], first row has N1 1s, second row has N2 1s, ...
    # indices: [Ns,], Ns = N1 + N2 + ...
    # cu_seqlens_q: [B+1,], (0, N1, N1+N2, ...), cu=cumulative
    # max_len_q: scalar, max(N1, N2, ...)
    q, indices_q, cu_seqlens_q, max_len_q = unpad_input(q, mask_q)
    k, indices_kv, cu_seqlens_kv, max_len_kv = unpad_input(k, mask_kv)
    v = index_first_axis(v.reshape(-1, H, D), indices_kv) # same indice as k

    # call varlen_func
    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=max_len_q,
        max_seqlen_k=max_len_kv,
        dropout_p=dropout,
        causal=causal,
        window_size=window_size,
    )

    # pad (put back) output
    out = pad_input(out, indices_q, B, N)
    return out

class Attention(nn.Module):
    """
    Attention module of LightningDiT.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        use_rmsnorm: bool = False,
        backend: str = "flash-attn"
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        if use_rmsnorm:
            norm_layer = RMSNorm
            
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm = qk_norm
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.backend = backend
        if self.backend == "flash-attn" and not FLASH_ATTN_AVAILABLE:
            self.backend = 'torch'
            MMLogger.print_log("Warning: flash-attn is not available, using torch backend instead.", "logger", logging.WARNING)
        
    def forward(self, x: torch.Tensor, rope=None, mask=None, causal=False, return_attn=False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.chunk(3, dim=2)
        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, N, self.num_heads, self.head_dim)
        v = v.reshape(B, N, self.num_heads, self.head_dim)
        dropout = self.attn_drop.p if self.training else 0.
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q = rope(q).to(torch.bfloat16)
            k = rope(k).to(torch.bfloat16)
        if self.backend == 'flash-attn' and not return_attn:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            x = attention_flashattn(q, k, v, mask_q=mask, mask_kv=mask,dropout=dropout, causal=causal)
            x = x.reshape(B, N, -1)
        elif self.backend == 'torch' or return_attn:
            attn = q @ k.transpose(-2, -1) * self.scale
            if mask is not None:
                mask = torch.logical_and(mask.unsqueeze(1), mask.unsqueeze(2))
                mask = mask.unsqueeze(1)
            else:
                mask = torch.ones(B, 1, N, N, dtype=torch.bool, device=q.device)
            if causal:
                mask = mask.tril()
            attn = attn.masked_fill(torch.logical_not(mask), -np.inf)
            
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Same as DiT.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        Args:
            t: A 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.
        Returns:
            An (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding
    
    @torch.compile
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    Same as DiT.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    @torch.compile
    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class LightningDiTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including: 
    - ROPE
    - QKNorm 
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False, 
        use_rmsnorm=False,
        wo_shift=False,
        **block_kwargs
    ):
        super().__init__()
        
        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            
        # Initialize attention layer
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )
        
        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )
            
        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift
    @torch.compile
    def forward(self, x, c, feat_rope=None, mask=None):
        if c is not None:
            if self.wo_shift:
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
                shift_msa = None
                shift_mlp = None
            else:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
                
            attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope, mask=mask)
            x = x + gate_msa.unsqueeze(1) * attn_out
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            attn_out = self.attn(self.norm1(x), rope=feat_rope, mask=mask)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of LightningDiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False, no_conditioning=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        if not no_conditioning:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = None
    @torch.compile
    def forward(self, x, c):
        if c is not None:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
            x = self.linear(x)
        else:
            x = self.linear(self.norm_final(x))
        return x


class LightningDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        use_qknorm=False,
        use_swiglu=False,
        use_rope=False,
        use_rmsnorm=False,
        wo_shift=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels if not learn_sigma else in_channels * 2
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_checkpoint = use_checkpoint
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        # self.front_back_embedder = nn.Embedding(2, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None

        self.blocks = nn.ModuleList([
            LightningDiTBlock(hidden_size, 
                     num_heads, 
                     mlp_ratio=mlp_ratio, 
                     use_qknorm=use_qknorm, 
                     use_swiglu=use_swiglu, 
                     use_rmsnorm=use_rmsnorm,
                     wo_shift=wo_shift,
                     ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t=None, y=None, ft_index=None):
        """
        Forward pass of LightningDiT.
        x: (N, C, 2, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        use_checkpoint: boolean to toggle checkpointing
        """
        use_checkpoint = self.use_checkpoint
        # front_back_ids = torch.zeros_like(x).int()
        # front_back_ids[:, :, 1, :, :] = 1
        # front_back_embed = self.front_back_embedder(front_back_ids)
        # print(front_back_embed.shape)
        # x = x + front_back_embed
        # x = x.permute(2, 0, 1, 3, 4).reshape(N*2, C, H, W)
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        # y = self.y_embedder(y.flatten(), self.training)    # (N, D)
        c = t                                # (N, D)

        for i, block in enumerate(self.blocks):
            if use_checkpoint:
                x = checkpoint(block, x, c, self.feat_rope, use_reentrant=True)
            else:
                x = block(x, c, self.feat_rope)
            if ft_index is not None and i == ft_index:
                return x

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, 2, H, W)

        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=None, cfg_interval_start=None):
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        
        if cfg_interval is True:
            timestep = t[0]
            if timestep < cfg_interval_start:
                half_eps = cond_eps

        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                             LightningDiT Configs                              #
#################################################################################

def LightningDiT_XL_1(**kwargs):
    return LightningDiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

def LightningDiT_XL_2(**kwargs):
    return LightningDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def LightningDiT_L_2(**kwargs):
    return LightningDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def LightningDiT_B_1(**kwargs):
    return LightningDiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def LightningDiT_B_2(**kwargs):
    return LightningDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def LightningDiT_1p0B_1(**kwargs):
    return LightningDiT(depth=24, hidden_size=1536, patch_size=1, num_heads=24, **kwargs)

def LightningDiT_1p0B_2(**kwargs):
    return LightningDiT(depth=24, hidden_size=1536, patch_size=2, num_heads=24, **kwargs)

def LightningDiT_1p6B_1(**kwargs):
    return LightningDiT(depth=28, hidden_size=1792, patch_size=1, num_heads=28, **kwargs)

def LightningDiT_1p6B_2(**kwargs):
    return LightningDiT(depth=28, hidden_size=1792, patch_size=2, num_heads=28, **kwargs)

LightningDiT_models = {
    'LightningDiT-B/1': LightningDiT_B_1, 'LightningDiT-B/2': LightningDiT_B_2,
    'LightningDiT-L/2': LightningDiT_L_2,
    'LightningDiT-XL/1': LightningDiT_XL_1, 'LightningDiT-XL/2': LightningDiT_XL_2,
    'LightningDiT-1p0B/1': LightningDiT_1p0B_1, 'LightningDiT-1p0B/2': LightningDiT_1p0B_2,
    'LightningDiT-1p6B/1': LightningDiT_1p6B_1, 'LightningDiT-1p6B/2': LightningDiT_1p6B_2,
}

if __name__ == '__main__':
    block = LightningDiTBlock(
        hidden_size=1152, 
        num_heads=16, 
        mlp_ratio=4.0, 
        use_qknorm=False, 
        use_swiglu=False, 
        use_rmsnorm=False, 
        wo_shift=False,
        backend='torch'
        )
    x = torch.randn(1, 196, 1152)
    c = torch.randn(1, 1152)
    x, attn = block(x, c, feat_rope=None, mask=None, return_attn=True)
    print(x.shape)
    print(attn.shape)