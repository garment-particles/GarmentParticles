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
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.utils.checkpoint import checkpoint

from models.swiglu_ffn import SwiGLUFFN 
from models.lightningdit import TimestepEmbedder, FinalLayer, Attention, modulate, Mlp
from models.lightningdit_cross_attn import CrossAttention
from models.rmsnorm import RMSNorm

from transformers import CLIPTextModel



def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False



class LightningDiTCrossAttnBlock(nn.Module):
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
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )
        
        self.cross_attn = CrossAttention(
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
        elif no_conditioning:
            self.adaLN_modulation = None
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift
    @torch.compile
    def forward(self, x, c, context, feat_rope=None, mask=None, ctx_mask=None):
        if c is not None:
            if self.wo_shift:
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
                shift_msa = None
                shift_mlp = None
            else:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope, mask=mask)
            x = x + gate_msa.unsqueeze(1) * attn_out
            cross_attn_out = self.cross_attn(self.norm2(x), context, rope=feat_rope, mask_x=mask, mask_ctx=ctx_mask)
            x = x + cross_attn_out
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        else:
            attn_out = self.attn(self.norm1(x), rope=feat_rope, mask=mask)
            cross_attn_out = self.cross_attn(self.norm2(x), context, rope=feat_rope, mask_x=mask, mask_ctx=ctx_mask)
            x = x + cross_attn_out
            x = x + self.mlp(self.norm3(x))
        
        return x

class SparseLightningDiTV3CrossAttn(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels=32,
        text_encoder_name="openai/clip-vit-large-patch14",
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

        self.blocks = nn.ModuleList([
            LightningDiTCrossAttnBlock(hidden_size, 
                num_heads, 
                mlp_ratio=mlp_ratio, 
                use_qknorm=use_qknorm, 
                use_swiglu=use_swiglu, 
                use_rmsnorm=use_rmsnorm,
                wo_shift=wo_shift,
                backend=backend
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

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
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t=None, text_tokens=None, text_attn_mask=None, mask=None, **kwargs):
        """
        Forward pass of LightningDiT.
        x: (N, T, C) tensor of spatial inputs (images or latent representations of images)
        text_tokens: (N, T) tensor of text inputs
        text_attn_mask: (N, T) tensor of text attention mask
        mask: (N, T) 
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        use_checkpoint: boolean to toggle checkpointing
        """
        use_checkpoint = self.use_checkpoint
        x = self.x_embedder_proj(x)  # (N, n_face, n_pts, D)
        text_embeds = self.text_proj(self.text_encoder(text_tokens, attention_mask=text_attn_mask).last_hidden_state)  # (N, M, D)
        t = self.t_embedder(t)                   # (N, D)
        c = t                                # (N, D)
        rope = None
        for i, block in enumerate(self.blocks):
            if use_checkpoint:
                x = checkpoint(block, x, c, text_embeds, rope, mask, text_attn_mask, use_reentrant=True)
            else:
                x = block(x, c, text_embeds, rope, mask, text_attn_mask)

        x = self.final_layer(x, c)                # (N, T, out_channels)
        return {"pred": x}

    def forward_with_mask(self, x, t=None, text_tokens=None, text_attn_mask=None, mask=None, **kwargs):
        return self.forward(x, t, text_tokens, text_attn_mask, mask, **kwargs)

    def forward_with_cfg(self, x, t, cfg_scale, text_tokens=None, text_attn_mask=None, mask=None, cfg_interval=None, cfg_interval_start=None, **kwargs):
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward(combined, t, text_tokens=text_tokens, text_attn_mask=text_attn_mask, mask=mask)["pred"]
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
        )
        
        return self

    
if __name__ == "__main__":
    block = LightningDiTCrossAttnBlock(
        hidden_size=1152, 
        num_heads=16, 
        mlp_ratio=4.0, 
        use_qknorm=False, 
        use_swiglu=True, 
        use_rmsnorm=True, 
        wo_shift=False).cuda()
    x = torch.randn(1, 1024, 1152).cuda().bfloat16()
    c = torch.randn(1, 1152).cuda().bfloat16()
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