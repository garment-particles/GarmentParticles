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
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.utils.checkpoint import checkpoint

from models.swiglu_ffn import SwiGLUFFN 
from models.lightningdit import TimestepEmbedder, Mlp, unpad_input
from models.rmsnorm import RMSNorm

from transformers import CLIPTextModel

from utils.mm_logging import MMLogger
import logging

try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func, flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func, flash_attn_kvpacked_func, flash_attn_varlen_kvpacked_func
    from flash_attn.bert_padding import index_first_axis, pad_input  # noqa
    FLASH_ATTN_AVAILABLE = True
except:
    print('[WARN] flash_attn not available, using naive implementation')
    FLASH_ATTN_AVAILABLE = False


def freeze_model(model, except_for=None):
    for name, param in model.named_parameters():
        if except_for is not None and except_for in name:
            param.requires_grad = True
            continue
        param.requires_grad = False

def modulate(x, shift, scale):
    if shift is None:
        return x * (1 + scale)
    if scale is None:
        return x + shift
    if shift is None and scale is None:
        return x
    return x * (1 + scale) + shift

def attention_flashattn_qkvpacked_varlen(qkv, cu_lens=None, max_len=None, dropout=0, causal=False, window_size=(-1, -1)):
    # qkv: (N, 3, H, D)
    # mask_qkv: (N)
    # return: (N, H, D)


    out = flash_attn_varlen_qkvpacked_func(
        qkv,
        cu_seqlens=cu_lens,
        max_seqlen=max_len,
        dropout_p=dropout,
        causal=causal,
        window_size=window_size,
    )

    return out

def attention_flashattn_kvpacked_varlen(kv, q, cu_q_lens=None, max_q_len=None, cu_kv_lens=None, max_kv_len=None, dropout=0, causal=False, window_size=(-1, -1)):
    # kv: (M, 2, H, D)
    # q: (N, H, D)
    # cu_q_lens: (N)
    # cu_kv_lens: (M)
    # return: (N, H, D)
    
    # call varlen_func
    out = flash_attn_varlen_kvpacked_func(
        q, kv,
        cu_seqlens_q=cu_q_lens,
        cu_seqlens_k=cu_kv_lens,
        max_seqlen_q=max_q_len,
        max_seqlen_k=max_kv_len,
        dropout_p=dropout,
        causal=causal,
        window_size=window_size,
    )

    return out


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
    def forward(self, x, c, cu_lens=None):
        # c is (N, D)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        if cu_lens is not None:
            lens = cu_lens[1:] - cu_lens[:-1]
            lens = lens.long()
            shift = torch.repeat_interleave(shift, lens, dim=0)
            scale = torch.repeat_interleave(scale, lens, dim=0)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class AttentionVarlen(nn.Module):
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
            MMLogger.print_log("Varlen must use flash-attn backend.", "logger", logging.ERROR)
            raise ValueError("Varlen must use flash-attn.")
        
    def forward(self, x: torch.Tensor, cu_lens=None, max_len=None, causal=False) -> torch.Tensor:
        dropout = self.attn_drop.p if self.training else 0.
        N = x.shape[0]
        qkv = self.qkv(x).reshape(N, 3, self.num_heads, self.head_dim).contiguous()
        if self.qk_norm:
            q, k, v = qkv.chunk(3, dim=1)
            q = self.q_norm(q)
            k = self.k_norm(k)
            qkv = torch.cat([q, k, v], dim=1).to(torch.bfloat16)
        x = attention_flashattn_qkvpacked_varlen(qkv, cu_lens=cu_lens, max_len=max_len, dropout=dropout, causal=causal)
        x = x.reshape(N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionVarlen(nn.Module):
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
            MMLogger.print_log("Varlen must use flash-attn backend.", "logger", logging.ERROR)
            raise ValueError("Varlen must use flash-attn.")
        
    def forward(self, x: torch.Tensor, ctx: torch.Tensor, cu_input_lens=None, max_input_len=None, cu_ctx_lens=None, max_ctx_len=None, causal=False, return_attn=False) -> torch.Tensor:
        """
        x: [N, C]
        ctx: [M, C]
        """
        M, N = ctx.shape[0], x.shape[0]
        dropout = self.attn_drop.p if self.training else 0.
        kv = self.kv(ctx).reshape(M, 2, self.num_heads, self.head_dim).contiguous()
        q = self.q(x).reshape(N, self.num_heads, self.head_dim).contiguous()
        if self.qk_norm:
            q = self.q_norm(q)
            k, v = kv.chunk(2, dim=1)
            k = self.k_norm(k)
            kv = torch.cat([k, v], dim=1).to(torch.bfloat16)
            q = q.to(torch.bfloat16)
        x = attention_flashattn_kvpacked_varlen(kv, q, cu_q_lens=cu_input_lens, max_q_len=max_input_len, cu_kv_lens=cu_ctx_lens, max_kv_len=max_ctx_len, dropout=dropout, causal=causal)
        x = x.reshape(N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LightningDiTCrossAttnVarlenBlock(nn.Module):
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
        no_conditioning=False,
        **block_kwargs
    ):
        super().__init__()
        
        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            self.norm3 = RMSNorm(hidden_size)
            
        # Initialize attention layer
        self.attn = AttentionVarlen(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )
        
        self.cross_attn = CrossAttentionVarlen(
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
    def forward(self, x, c, context, cu_input_lens=None, max_input_len=None, cu_text_lens=None, max_text_len=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa, shift_mlp = None, None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # 3. Apply Attention (Standard Varlen Logic)
        attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa), cu_lens=cu_input_lens, max_len=max_input_len)
        
        # 4. Residuals
        x = x + gate_msa * attn_out
        cross_attn_out = self.cross_attn(self.norm2(x), context, cu_ctx_lens=cu_text_lens, max_ctx_len=max_text_len, cu_input_lens=cu_input_lens, max_input_len=max_input_len)
        x = x + cross_attn_out
        x = x + gate_mlp * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        
        return x, context
    
class LightningDiTCrossAttnVarlenBlockV2(nn.Module):
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
        no_conditioning=False,
        **block_kwargs
    ):
        super().__init__()
        
        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.additional_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            self.norm3 = RMSNorm(hidden_size)
            self.additional_norm = RMSNorm(hidden_size)
        # Initialize attention layer
        self.attn = AttentionVarlen(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )
        
        self.cross_attn = CrossAttentionVarlen(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )
        
        self.additional_cross_attn = CrossAttentionVarlen(
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
    def forward(self, x, c, context, additional_context, cu_input_lens=None, max_input_len=None, cu_text_lens=None, max_text_len=None, cu_additional_lens=None, max_additional_len=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa, shift_mlp = None, None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # 3. Apply Attention (Standard Varlen Logic)
        attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa), cu_lens=cu_input_lens, max_len=max_input_len)
        
        # 4. Residuals
        x = x + gate_msa * attn_out
        cross_attn_out = self.cross_attn(self.norm2(x), context, cu_ctx_lens=cu_text_lens, max_ctx_len=max_text_len, cu_input_lens=cu_input_lens, max_input_len=max_input_len)
        additional_cross_attn_out = self.additional_cross_attn(self.additional_norm(x), additional_context, cu_ctx_lens=cu_additional_lens, max_ctx_len=max_additional_len, cu_input_lens=cu_input_lens, max_input_len=max_input_len)
        x = x + cross_attn_out + additional_cross_attn_out
        
        x = x + gate_mlp * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        
        return x
    

class SparseLightningDiTV3CrossAttnVarlen(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels=32,
        text_encoder_name="openai/clip-vit-large-patch14",
        block_type="v1",
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
        backend="flash-attn",
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_checkpoint = use_checkpoint
        self.x_embedder_proj = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_name)
        self.text_encoder.eval()
        freeze_model(self.text_encoder)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, hidden_size, bias=True)
        block_cls = {
            "v1": LightningDiTCrossAttnVarlenBlock,
            "v2": LightningDiTCrossAttnVarlenBlockV2,
        }[block_type]
        self.block_type = block_type
        self.blocks = nn.ModuleList([block_cls(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_qknorm=use_qknorm, use_swiglu=use_swiglu, use_rmsnorm=use_rmsnorm, wo_shift=wo_shift, backend=backend) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

    def train(self, mode: bool = True):
        super().train(mode)
        # Always keep text encoder in eval mode
        self.text_encoder.eval()
        return self

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
        # Only apply to modules that are NOT the text encoder
        self.x_embedder_proj.apply(_basic_init)
        self.t_embedder.apply(_basic_init)
        self.text_proj.apply(_basic_init)
        self.blocks.apply(_basic_init)
        self.final_layer.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            if self.block_type == "v2":
                nn.init.constant_(block.adaLN_modulation_x[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation_x[-1].bias, 0)
                nn.init.constant_(block.adaLN_modulation_ctx[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation_ctx[-1].bias, 0)
            else:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t=None, text_tokens=None, text_attn_mask=None, cu_input_lens=None, max_input_len=None, **kwargs):
        """
        Forward pass of LightningDiT.
        x: (T, C) tensor of spatial inputs (images or latent representations of images)
        text_tokens: (N, T) tensor of text inputs
        text_attn_mask: (N, T) tensor of text attention mask
        cu_input_lens: (N,) tensor of cumulative input lengths
        t: (T,) tensor of diffusion timesteps
        use_checkpoint: boolean to toggle checkpointing
        """
        
        # print(f"cu_input_lens: {cu_input_lens}")
        # print(f"max_input_len: {max_input_len}")
        # print(f"x.shape[0]: {x.shape[0]}")

        # # Verify the cumulative lengths are valid
        # assert cu_input_lens[0] == 0, "cu_input_lens must start with 0"
        # assert cu_input_lens[-1] == x.shape[0], f"cu_input_lens[-1] ({cu_input_lens[-1]}) must equal total tokens ({x.shape[0]})"
        # assert torch.all(cu_input_lens[1:] > cu_input_lens[:-1]), "cu_input_lens must be strictly increasing"
        
        use_checkpoint = self.use_checkpoint
        cu_input_lens = cu_input_lens.int()
        x = self.x_embedder_proj(x)  # (T, D)
        text_embeds = self.text_proj(self.text_encoder(text_tokens, attention_mask=text_attn_mask).last_hidden_state)  # (N, M, D)
        cu_ctx_lens = torch.ones(text_embeds.shape[0], dtype=torch.int32) * text_embeds.shape[1]
        cu_ctx_lens = F.pad(torch.cumsum(cu_ctx_lens, dim=0), (1, 0)).to(cu_input_lens)
        max_ctx_len = text_embeds.shape[1]
        text_embeds = text_embeds.reshape(-1, self.hidden_size)
        t = self.t_embedder(t)                   # (T, D)
        c = t                                # (T, D)
        context = text_embeds
        
        for i, block in enumerate(self.blocks):
            if use_checkpoint:
                x, context = checkpoint(block, x, c, context, cu_input_lens, max_input_len, cu_ctx_lens, max_ctx_len, use_reentrant=True)
            else:
                x, context = block(x, c, context, cu_input_lens, max_input_len, cu_ctx_lens, max_ctx_len)
        x = self.final_layer(x, c, cu_input_lens if self.block_type == "v2" else None)                # (T, out_channels)
        return {"pred": x}
    
    def forward_with_mask(self, x, t, mask, text_tokens=None, text_attn_mask=None, **kwargs):
        B, N = x.shape[0], x.shape[1]
        x, indices, cu_lens, max_len = unpad_input(x, mask)
        if self.block_type == "v1":
            lens = cu_lens[1:] - cu_lens[:-1]
            t = t.repeat_interleave(lens.int(), dim=0)
        x = self.forward(x, t, text_tokens, text_attn_mask, cu_lens, max_len)["pred"]
        x = pad_input(x, indices, B, N)
        return {"pred": x}
    
    def forward_with_cfg(self, x, t, cfg_scale, text_tokens=None, text_attn_mask=None, mask=None, cfg_interval=None, cfg_interval_start=None, **kwargs):
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward_with_mask(combined, t, text_tokens=text_tokens, text_attn_mask=text_attn_mask, mask=mask)["pred"]
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        
        if cfg_interval is True:
            timestep = t[0]
            if timestep < cfg_interval_start:
                half_eps = cond_eps

        eps = torch.cat([half_eps, half_eps], dim=0)
        return {"pred":eps}

    
    def apply_fsdp2(
        self, 
        device_mesh, 
        mp_policy: MixedPrecisionPolicy, 
        reshard_after_forward: bool = True
    ):
        """
        Advanced FSDP2 application with best practices.
        """
        # Mixed precision policy
        
        # DON'T wrap frozen modules (text_encoder)
        # FSDP2 will skip parameters that require_grad=False anyway
        
        # Wrap embedders separately if they're large
        fully_shard(
            self.x_embedder_proj,
            mesh=device_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
        
        fully_shard(
            self.t_embedder,
            mesh=device_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
        
        fully_shard(
            self.text_proj,
            mesh=device_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
        
        # Wrap each transformer block
        for block in self.blocks:
            # For very deep models, you can wrap sub-components within each block
            # But typically wrapping the entire block is sufficient
            fully_shard(
                block,
                mesh=device_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
            )
        
        # Wrap final layer
        fully_shard(
            self.final_layer,
            mesh=device_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
        
        # Final outer wrap
        fully_shard(
            self,
            mesh=device_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
            ignored_params=[self.text_encoder.parameters()],
        )
        
        return self

if __name__ == "__main__":
    block = LightningDiTCrossAttnVarlenBlock(
        hidden_size=1152, 
        num_heads=16, 
        mlp_ratio=4.0, 
        use_qknorm=False, 
        use_swiglu=True, 
        use_rmsnorm=True, 
        wo_shift=False).cuda()
    x = torch.randn(1024, 1152).cuda().bfloat16()
    c = torch.randn(1024, 1152).cuda().bfloat16()
    context = torch.randn(1, 77, 1152).cuda().bfloat16()
    feat_rope = None
    mask = torch.ones(1, 1024).cuda().bool()
    mask[0, 1000:] = False
    ctx_mask = torch.ones(1, 77).cuda().bool()
    ctx_mask[0, 50:] = False
    # set detect anomaly
    torch.autograd.set_detect_anomaly(True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = block(x, c, context, feat_rope, mask, ctx_mask)
    # unpadded_random_input = torch.randn_like(unpadded_cross_attn_out)
    padded_random_input = torch.randn_like(out)
    # (unpadded_random_input * unpadded_cross_attn_out).sum().backward()
    # x.grad = None
    # c.grad = None
    # context.grad = None
    (padded_random_input * out).sum().backward()
    print(unpadded_cross_attn_out.shape)
    print(unpadded_random_input.shape)
    print(unpadded_cross_attn_out.grad.shape)
    print(unpadded_random_input.grad.shape)
    print(unpadded_cross_attn_out.grad)
    print(unpadded_random_input.grad)
    print(out.shape)
    print(padded_random_input.shape)
    print(out.grad.shape)
    print(padded_random_input.grad.shape)
    print(out.grad)
    print(padded_random_input.grad)