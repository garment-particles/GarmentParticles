"""
Lightning DiT's codes are built from original DiT & SiT.
(https://github.com/facebookresearch/DiT; https://github.com/willisma/SiT)
It demonstrates that a advanced DiT together with advanced diffusion skills
could also achieve a very promising result with 1.35 FID on ImageNet 256 generation.

Enjoy everyone, DiT strikes back!

by Maple (Jingfeng Yao) from HUST-VL
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import einops
import numpy as np

from timm.models.vision_transformer import Mlp
from models.swiglu_ffn import SwiGLUFFN 
from models.pos_embed import RotaryEmbedding
from models.rmsnorm import RMSNorm
from models.lightningdit import LightningDiTBlock, unpad_input, modulate

from utils.mm_logging import MMLogger
import logging

try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func, flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func, flash_attn_kvpacked_func, flash_attn_varlen_kvpacked_func
    from flash_attn.bert_padding import index_first_axis, pad_input  # noqa
    FLASH_ATTN_AVAILABLE = True
except:
    print('[WARN] flash_attn not available, using naive implementation')
    FLASH_ATTN_AVAILABLE = False


def attention_flashattn_kvpacked(kv, q, mask_q=None, mask_kv=None, dropout=0, causal=False, window_size=(-1, -1)):
    # kv: (B, M, 2, H, D)
    # q: (B, N, H, D)
    # mask_kv: (B, M)
    # mask_q: (B, N)
    # return: (B, N, H, D)

    B, N, H, D = q.shape
    M = kv.shape[1]

    ### unmasked case (usually inference)
    ### will ignore window_size except flash-attn impl. Only provide the effective window!
    if mask_q is None and mask_kv is None:
        return flash_attn_kvpacked_func(q, kv, dropout, causal=causal, window_size=window_size) # [B, N, H, D]
    
    
    if mask_q is None:
        mask_q = torch.ones(B, N, dtype=torch.bool, device=q.device)
    if mask_kv is None:
        mask_kv = torch.ones(B, M, dtype=torch.bool, device=q.device)
    # unpad (gather) input
    # mask: [B, N], first row has N1 1s, second row has N2 1s, ...
    # indices: [Ns,], Ns = N1 + N2 + ...
    # cu_seqlens: [B+1,], (0, N1, N1+N2, ...), cu=cumulative
    # max_len: scalar, max(N1, N2, ...)
    q, indices_q, cu_seqlens_q, max_len_q = unpad_input(q, mask_q)
    kv, indices_kv, cu_seqlens_kv, max_len_kv = unpad_input(kv, mask_kv)
    # call varlen_func
    out = flash_attn_varlen_kvpacked_func(
        q, kv,
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

def attention_flashattn(q, k, v, mask_q=None, mask_kv=None, dropout=0, causal=False, window_size=(-1, -1), backend='flash-attn'):
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

class CrossAttention(nn.Module):
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
            
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
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
        
    def forward(self, x: torch.Tensor, ctx: torch.Tensor, rope=None, mask_x=None, mask_ctx=None, causal=False) -> torch.Tensor:
        """
        x: [B, N, C]
        ctx: [B, M, C]
        """
        B, N, C = x.shape
        M = ctx.shape[1]
        kv = self.kv(ctx).reshape(B, M, 2, self.num_heads, self.head_dim).contiguous()
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).contiguous()
        if self.qk_norm:
            q = self.q_norm(q)
            k, v = kv.chunk(2, dim=2)
            k = self.k_norm(k)
            kv = torch.cat([k, v], dim=2).to(torch.bfloat16)
        if rope is not None:
            q = rope(q).to(torch.bfloat16)
        dropout = self.attn_drop.p if self.training else 0.
        if self.backend == 'flash-attn':
            x = attention_flashattn_kvpacked(kv, q, mask_q=mask_x, mask_kv=mask_ctx, dropout=dropout, causal=causal)
        else:
            k, v = kv.unbind(2)
            # [B, N, H, D] @ [B, M, H, D] -> [B, N, M]
            attn = einops.einsum(q, k, 'b n h d, b m h d -> b n m h') * self.scale # [B, N, M]
            if mask_x is None:
                mask_x = torch.ones(B, N, dtype=torch.bool, device=q.device)
            if mask_ctx is None:
                mask_ctx = torch.ones(B, M, dtype=torch.bool, device=q.device)
            mask = torch.logical_and(mask_x.unsqueeze(2), mask_ctx.unsqueeze(1))
            attn = attn.masked_fill(torch.logical_not(mask)[..., None], -np.inf)
            attn = attn.softmax(dim=-2)
            attn = self.attn_drop(attn)
            x = einops.einsum(attn, v, 'b n m h, b m h d -> b n h d') # [B, N, H, D]
        x = x.reshape(B, N, -1)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
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
            
            self.norm_ctx = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            
            self.norm_ctx = RMSNorm(hidden_size)
            
        # Initialize attention layer
        self.attn = CrossAttention(
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

    # @torch.compile
    def forward(self, x, ctx, c=None, feat_rope=None, mask_x=None, mask_ctx=None):
        if c is not None:
            if self.wo_shift:
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
                shift_msa = None
                shift_mlp = None
            else:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            
            attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa), self.norm_ctx(ctx), rope=feat_rope, mask_x=mask_x, mask_ctx=mask_ctx)
            x = x + gate_msa.unsqueeze(1) * attn_out
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            attn_out = self.attn(self.norm1(x), self.norm_ctx(ctx), rope=feat_rope, mask_x=mask_x, mask_ctx=mask_ctx)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of LightningDiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    # @torch.compile
    def forward(self, x, c=None):
        if c is not None:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
        else:
            x = self.norm_final(x)
        x = self.linear(x)
        return x


class CrossAttentionTransformer(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels_ctx=32,
        out_channels=7,
        n_query_tokens=25,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False,
        use_rope=False,
        use_rmsnorm=False,
        wo_shift=False,
        use_checkpoint=False,
        use_loop_loss=False,
        curve_stats=None,
        absolute_coords=False
    ):
        super().__init__()
        self.n_query_tokens = n_query_tokens
        self.in_channels_ctx = in_channels_ctx
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_checkpoint = use_checkpoint
        self.ctx_embedder = nn.Linear(in_channels_ctx, hidden_size, bias=True)
        self.query_embedding = nn.Embedding(n_query_tokens, hidden_size)
        self.use_loop_loss = use_loop_loss
        self.curve_mean, self.curve_std = curve_stats["curve_mean"], curve_stats["curve_std"]
        self.curve_mean = torch.from_numpy(self.curve_mean)
        self.curve_std = torch.from_numpy(self.curve_std)
        self.absolute_coords = absolute_coords
        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            head_dim = hidden_size // num_heads
            self.feat_rope = RotaryEmbedding(
                head_dim=head_dim,
                base = 150000.0,
                dtype = torch.float32,
                initial_context_length = 4096,
                scaling_factor = 32.0,
                ntk_alpha = 1.0,
                ntk_beta = 32.0,
            )
        else:
            self.feat_rope = None

        self.blocks = []
        for d in range(depth // 2):
            self.blocks.append(
                CrossAttentionBlock(hidden_size, 
                     num_heads, 
                     mlp_ratio=mlp_ratio, 
                     use_qknorm=use_qknorm, 
                     use_swiglu=use_swiglu, 
                     use_rmsnorm=use_rmsnorm,
                     wo_shift=wo_shift,
                )
            )
            self.blocks.append(
                LightningDiTBlock(hidden_size, 
                     num_heads, 
                     mlp_ratio=mlp_ratio, 
                     use_qknorm=use_qknorm, 
                     use_swiglu=use_swiglu, 
                     use_rmsnorm=use_rmsnorm,
                     wo_shift=wo_shift,
                )
            )
        self.blocks = nn.ModuleList(self.blocks)
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.query_embedding.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        ww = self.ctx_embedder.weight.data
        nn.init.xavier_uniform_(ww.view([ww.shape[0], -1]))
        nn.init.constant_(self.ctx_embedder.bias, 0)

        # # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, input=None, ctx=None, t=None, y=None, mask_x=None, mask_ctx=None):
        """
        Forward pass of LightningDiT.
        x: (N, T, C) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        use_checkpoint: boolean to toggle checkpointing
        """
        N = ctx.shape[0]
        use_checkpoint = self.use_checkpoint
        x = self.query_embedding.weight.unsqueeze(0).repeat(N, 1, 1) # (N, n_query_tokens, D)
        ctx = self.ctx_embedder(ctx) # (N, M, D)
        c = None                          # (N, D)

        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                if use_checkpoint:
                    x = checkpoint(block, x, ctx, c, self.feat_rope, None, mask_ctx, use_reentrant=True)
                else:
                    x = block(x, ctx, c, self.feat_rope, mask_x=None, mask_ctx=mask_ctx)
            else:
                if use_checkpoint:
                    x = checkpoint(block, x, c, self.feat_rope, None, use_reentrant=True)
                else:
                    x = block(x, c, self.feat_rope, mask=None)
        x = self.final_layer(x, c)                # (N, T, out_channels)
        loss_dict = {"preds": x}
        if input is not None:
            mse_loss = (x - input) ** 2
            mse_loss = mse_loss.mean(dim=(1, 2))
            loss_dict["mse_loss"] = mse_loss
            if self.use_loop_loss:
                vecs = x[..., :2]
                endpoints = torch.cumsum(vecs, dim=1) if not self.absolute_coords else vecs
                gt_curve_lens = mask_x.sum(1) - 1
                endpoints = torch.gather(endpoints, 1, gt_curve_lens.unsqueeze(1).unsqueeze(2).repeat(1, 1, 2))
                gt_zero_vec = -self.curve_mean.to(x) / self.curve_std.to(x)
                loop_loss = (endpoints[..., 0] - gt_zero_vec) ** 2
                loop_loss = loop_loss.mean(dim=1)
                loss_dict["loop_loss"] = loop_loss
        return loss_dict



#################################################################################
#                             LightningDiT Configs                              #
#################################################################################

def CrossAttentionTransformer_XL_1(**kwargs):
    return CrossAttentionTransformer(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def CrossAttentionTransformer_XL_2(**kwargs):
    return CrossAttentionTransformer(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def CrossAttentionTransformer_L_2(**kwargs):
    return CrossAttentionTransformer(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def CrossAttentionTransformer_B_1(**kwargs):
    return CrossAttentionTransformer(depth=12, hidden_size=768, num_heads=12, **kwargs)

def CrossAttentionTransformer_B_2(**kwargs):
    return CrossAttentionTransformer(depth=12, hidden_size=768, num_heads=12, **kwargs)

def CrossAttentionTransformer_1p0B_1(**kwargs):
    return CrossAttentionTransformer(depth=24, hidden_size=1536, num_heads=24, **kwargs)

def CrossAttentionTransformer_1p0B_2(**kwargs):
    return CrossAttentionTransformer(depth=24, hidden_size=1536, num_heads=24, **kwargs)

def CrossAttentionTransformer_1p6B_1(**kwargs):
    return CrossAttentionTransformer(depth=28, hidden_size=1792, num_heads=28, **kwargs)

def CrossAttentionTransformer_1p6B_2(**kwargs):
    return CrossAttentionTransformer(depth=28, hidden_size=1792, num_heads=28, **kwargs)

CrossAttentionTransformer_models = {
    'CrossAttentionTransformer-B/1': CrossAttentionTransformer_B_1, 'CrossAttentionTransformer-B/2': CrossAttentionTransformer_B_2,
    'CrossAttentionTransformer-L/2': CrossAttentionTransformer_L_2,
    'CrossAttentionTransformer-XL/1': CrossAttentionTransformer_XL_1, 'CrossAttentionTransformer-XL/2': CrossAttentionTransformer_XL_2,
    'CrossAttentionTransformer-1p0B/1': CrossAttentionTransformer_1p0B_1, 'CrossAttentionTransformer-1p0B/2': CrossAttentionTransformer_1p0B_2,
    'CrossAttentionTransformer-1p6B/1': CrossAttentionTransformer_1p6B_1, 'CrossAttentionTransformer-1p6B/2': CrossAttentionTransformer_1p6B_2,
}

if __name__ == '__main__':
    x = torch.randn(1, 196, 4, 6).cuda().bfloat16()
    ctx = torch.randn(1, 8192, 2, 4, 6).cuda().bfloat16()
    mask_q = torch.ones(1, 196).cuda().bool()
    mask_kv = torch.ones(1, 8192).cuda().bool()
    mask_q[0, 100:] = False
    mask_kv[0, 8000:] = False
    dgrad = torch.randn((1, 196, 4, 6), dtype=torch.bfloat16).cuda()
    x.requires_grad_()
    ctx.requires_grad_()
    
    out = attention_flashattn_kvpacked(ctx, x, mask_q=mask_q, mask_kv=mask_kv)
    print(out.shape)
    (dgrad * out).sum().backward()
    print(torch.isnan(x.grad).sum())
    print(torch.isnan(ctx.grad).sum())
    print(x.grad.shape)
    print(ctx.grad.shape)
    print(x.grad)
    print(ctx.grad)
