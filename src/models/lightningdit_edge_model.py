import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.utils.checkpoint import checkpoint

from models.lightningdit import TimestepEmbedder, FinalLayer
from models.sparse_lightningdit_v3_cross_attn import LightningDiTCrossAttnBlock

from models.pos_embed import RotaryEmbedding


class LightningCrossAttnDiTV3EdgeModel(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels=32,
        in_channels_context=32,
        n_panels=34,
        n_curves=25,
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
        self.point_encoder = nn.Linear(in_channels_context, hidden_size, bias=True)
        self.panel_embedding = nn.Embedding(n_panels, hidden_size)
        self.curve_embedding = nn.Embedding(n_curves, hidden_size)
        
        self.in_channels = in_channels
        self.in_channels_context = in_channels_context
        self.n_panels = n_panels
        self.out_channels = in_channels
        self.n_curves = n_curves
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_checkpoint = use_checkpoint
        self.x_embedder_proj = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
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
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        nn.init.normal_(self.panel_embedding.weight, std=0.02)
        nn.init.normal_(self.curve_embedding.weight, std=0.02)
        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self, 
        x, 
        t=None, 
        panel_points=None, 
        panel_points_mask=None, 
        panel_indices=None, 
        mask=None, 
        **kwargs):
        """
        Forward pass of LightningDiT.
        x: (N, n_face, n_curves, C) tensor of spatial inputs (images or latent representations of images)
        panel_points: (N, T, 3) tensor of panel points
        panel_points_mask: (N, T) tensor of panel points mask
        panel_indices: (N, T) tensor of panel indices
        mask: (N, T) 
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        use_checkpoint: boolean to toggle checkpointing
        """
        B = x.shape[0]
        x = x.reshape(B, self.n_panels, self.n_curves, -1)
        N, n_face, n_curves, C = x.shape
        use_checkpoint = self.use_checkpoint
        x = self.x_embedder_proj(x)  # (N, n_face, n_pts, D)
        ctx = self.point_encoder(panel_points)
        x = x + self.panel_embedding.weight[:, None, :] # (n_panels, 1, D)
        curve_embedding = self.curve_embedding.weight[None, :, :] # (1, n_curves, D)
        x = x + curve_embedding
        x = x.reshape(N, n_face * n_curves, self.hidden_size)
        t = self.t_embedder(t)                   # (N, D)
        c = t                                # (N, D)
        rope = self.feat_rope
        for i, block in enumerate(self.blocks):
            if use_checkpoint:
                x = checkpoint(block, x, c, ctx, rope, mask, panel_points_mask, use_reentrant=True)
            else:
                x = block(x, c, ctx, rope, mask, panel_points_mask)
        x = self.final_layer(x, c)                # (N, T, out_channels)
        return {"pred": x}

    def forward_with_cfg(self, 
        x, 
        t=None, 
        panel_points=None, 
        panel_points_mask=None, 
        panel_indices=None, 
        mask=None, 
        cfg_interval=None, 
        cfg_interval_start=None, 
        **kwargs):
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward(combined, t, panel_points=panel_points, panel_points_mask=panel_points_mask, panel_indices=panel_indices, mask=mask)["pred"]
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
            self.point_encoder,
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