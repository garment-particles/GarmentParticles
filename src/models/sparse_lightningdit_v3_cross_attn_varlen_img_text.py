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
from models.sparse_lightningdit_v3_cross_attn_varlen import LightningDiTCrossAttnVarlenBlock, LightningDiTCrossAttnVarlenBlockV2, FinalLayer, freeze_model

from transformers import AutoModel, CLIPTextModel

from flash_attn.bert_padding import pad_input


class SparseLightningDiTV3CrossAttnVarlenImgText(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels=32,
        image_encoder_name="facebook/dinov2-large",
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
        self.image_encoder = AutoModel.from_pretrained(image_encoder_name)
        self.image_encoder.eval()
        freeze_model(self.image_encoder)
        self.image_proj = nn.Linear(self.image_encoder.config.hidden_size, hidden_size, bias=True)
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
        self.image_encoder.eval()
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
        self.image_proj.apply(_basic_init)
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

    def forward(self, x, t=None, text_tokens=None, text_attn_mask=None, pixel_values=None, cu_input_lens=None, max_input_len=None, **kwargs):
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
        image_embeds = self.image_proj(self.image_encoder(pixel_values).last_hidden_state)  # (N, M, D)
        embeds = torch.cat([text_embeds, image_embeds], dim=1)
        cu_ctx_lens = torch.ones(embeds.shape[0], dtype=torch.int32) * embeds.shape[1]
        cu_ctx_lens = F.pad(torch.cumsum(cu_ctx_lens, dim=0), (1, 0)).to(cu_input_lens)
        max_ctx_len = embeds.shape[1]
        embeds = embeds.reshape(-1, self.hidden_size)
        t = self.t_embedder(t)                   # (T, D)
        c = t                                # (T, D)
        context = embeds
        
        for i, block in enumerate(self.blocks):
            if use_checkpoint:
                x, context = checkpoint(block, x, c, context, cu_input_lens, max_input_len, cu_ctx_lens, max_ctx_len, use_reentrant=True)
            else:
                x, context = block(x, c, context, cu_input_lens, max_input_len, cu_ctx_lens, max_ctx_len)
        x = self.final_layer(x, c, cu_input_lens if self.block_type == "v2" else None)                # (T, out_channels)
        return {"pred": x}
    
    def forward_with_mask(self, x, t, mask, text_tokens=None, text_attn_mask=None, pixel_values=None, **kwargs):
        B, N = x.shape[0], x.shape[1]
        x, indices, cu_lens, max_len = unpad_input(x, mask)
        if self.block_type == "v1":
            lens = cu_lens[1:] - cu_lens[:-1]
            t = t.repeat_interleave(lens.int(), dim=0)
        x = self.forward(x, t, text_tokens, text_attn_mask, pixel_values, cu_lens, max_len)["pred"]
        x = pad_input(x, indices, B, N)
        return {"pred": x}
    
    def forward_with_cfg(self, x, t, cfg_scale, text_tokens=None, text_attn_mask=None, pixel_values=None, mask=None, cfg_interval=None, cfg_interval_start=None, **kwargs):
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward_with_mask(combined, t, text_tokens=text_tokens, text_attn_mask=text_attn_mask, pixel_values=pixel_values, mask=mask)["pred"]
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
            self.image_proj,
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
            ignored_params=[self.image_encoder.parameters()],
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