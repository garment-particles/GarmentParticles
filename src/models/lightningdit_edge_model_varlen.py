"""
Varlen version of the edge model for faster inference/training.
Uses flash_attn_varlen kernels to skip padded context tokens.
All operations stay in flat (B*N, D) format to avoid reshaping overhead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.utils.checkpoint import checkpoint

from models.lightningdit import TimestepEmbedder, unpad_input
from models.pos_embed import RotaryEmbedding, _apply_rotary_emb
from models.swiglu_ffn import SwiGLUFFN
from models.rmsnorm import RMSNorm

from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func
from flash_attn.bert_padding import pad_input

@torch.compile
def modulate(x, shift, scale):
    if shift is None and scale is None:
        return x
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


class AttentionVarlenRope(nn.Module):
    """Varlen self-attention with optional RoPE."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False,
                 attn_drop=0., proj_drop=0., use_rmsnorm=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm = qk_norm
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cu_lens, max_len, rope_cos=None, rope_sin=None):
        # x: (total_tokens, C)
        N = x.shape[0]
        qkv = self.qkv(x).reshape(N, 3, self.num_heads, self.head_dim).contiguous()
        if self.qk_norm:
            q, k, v = qkv.unbind(1)
            q = self.q_norm(q)
            k = self.k_norm(k)
            if rope_cos is not None:
                q = _apply_rotary_emb(q, rope_cos, rope_sin)
                k = _apply_rotary_emb(k, rope_cos, rope_sin)
            qkv = torch.stack([q, k, v], dim=1).to(torch.bfloat16)
        elif rope_cos is not None:
            q, k, v = qkv.unbind(1)
            q = _apply_rotary_emb(q, rope_cos, rope_sin)
            k = _apply_rotary_emb(k, rope_cos, rope_sin)
            qkv = torch.stack([q, k, v], dim=1).contiguous()
        dropout = self.attn_drop.p if self.training else 0.
        out = flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens=cu_lens, max_seqlen=max_len, dropout_p=dropout
        )
        out = out.reshape(N, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossAttentionVarlenRope(nn.Module):
    """Varlen cross-attention with optional RoPE on query."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False,
                 attn_drop=0., proj_drop=0., use_rmsnorm=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm = qk_norm
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, ctx, cu_q_lens, max_q_len, cu_kv_lens, max_kv_len,
                rope_cos=None, rope_sin=None):
        # x: (total_q, C), ctx: (total_kv, C)
        N, M = x.shape[0], ctx.shape[0]
        dropout = self.attn_drop.p if self.training else 0.
        kv = self.kv(ctx).reshape(M, 2, self.num_heads, self.head_dim).contiguous()
        q = self.q(x).reshape(N, self.num_heads, self.head_dim).contiguous()
        if self.qk_norm:
            q = self.q_norm(q)
            k, v = kv.unbind(1)
            k = self.k_norm(k)
            kv = torch.stack([k, v], dim=1).to(torch.bfloat16)
            if rope_cos is not None:
                q = _apply_rotary_emb(q, rope_cos, rope_sin)
            q = q.to(torch.bfloat16)
        elif rope_cos is not None:
            q = _apply_rotary_emb(q, rope_cos, rope_sin)
        out = flash_attn_varlen_kvpacked_func(
            q, kv,
            cu_seqlens_q=cu_q_lens, cu_seqlens_k=cu_kv_lens,
            max_seqlen_q=max_q_len, max_seqlen_k=max_kv_len,
            dropout_p=dropout,
        )
        out = out.reshape(N, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class LightningDiTCrossAttnVarlenBlockEdge(nn.Module):
    """Varlen cross-attention block — all ops in flat (B*N, D) for speed."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 use_qknorm=False, use_swiglu=False, use_rmsnorm=False,
                 wo_shift=False, **block_kwargs):
        super().__init__()
        if use_rmsnorm:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            self.norm3 = RMSNorm(hidden_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = AttentionVarlenRope(
            hidden_size, num_heads=num_heads, qkv_bias=True,
            qk_norm=use_qknorm, use_rmsnorm=use_rmsnorm, **block_kwargs
        )
        self.cross_attn = CrossAttentionVarlenRope(
            hidden_size, num_heads=num_heads, qkv_bias=True,
            qk_norm=use_qknorm, use_rmsnorm=use_rmsnorm, **block_kwargs
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            from timm.models.vision_transformer import Mlp
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,
                           act_layer=approx_gelu, drop=0)

        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True))
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.wo_shift = wo_shift

    @torch.compile
    def forward(self, x, c, ctx, cu_x_lens, max_x_len, cu_ctx_lens, max_ctx_len,
                rope_cos=None, rope_sin=None):
        # x: (B*N, D), c: (B*N, D), ctx: (total_ctx, D) — all flat
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa, shift_mlp = None, None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(c).chunk(6, dim=1)

        attn_out = self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            cu_x_lens, max_x_len, rope_cos, rope_sin
        )
        x = x + gate_msa * attn_out

        cross_out = self.cross_attn(
            self.norm2(x), ctx,
            cu_x_lens, max_x_len, cu_ctx_lens, max_ctx_len,
            rope_cos, rope_sin
        )
        x = x + cross_out

        x = x + gate_mlp * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FinalLayerVarlen(nn.Module):
    def __init__(self, hidden_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if use_rmsnorm:
            self.norm_final = RMSNorm(hidden_size)
        else:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    @torch.compile
    def forward(self, x, c):
        # x: (B*N, D), c: (B*N, D) — all flat
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LightningCrossAttnDiTV3EdgeModelVarlen(nn.Module):
    """
    Varlen edge model. Same architecture as LightningCrossAttnDiTV3EdgeModel
    but uses flash_attn_varlen kernels to skip padded context tokens.
    Weights are compatible with the padded version.
    """
    def __init__(
        self,
        in_channels=32,
        in_channels_context=32,
        n_panels=34,
        n_curves=25,
        use_panel_embedding=True,
        point_encoder_type="linear",
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
        self.use_panel_embedding = use_panel_embedding
        self.point_encoder_type = point_encoder_type
        if point_encoder_type == "linear":
            self.point_encoder = nn.Linear(in_channels_context, hidden_size, bias=True)
        else:
            raise ValueError(f"Unsupported point encoder type: {point_encoder_type}")
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
        self.n_edge_tokens = n_panels * n_curves

        self.x_embedder_proj = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if self.use_rope:
            head_dim = hidden_size // num_heads
            self.feat_rope = RotaryEmbedding(
                head_dim=head_dim,
                base=150000.0,
                dtype=torch.float32,
                initial_context_length=4096,
                scaling_factor=32.0,
                ntk_alpha=1.0,
                ntk_beta=32.0,
            )
        else:
            self.feat_rope = None

        self.blocks = nn.ModuleList([
            LightningDiTCrossAttnVarlenBlockEdge(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm, use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm, wo_shift=wo_shift,
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayerVarlen(hidden_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.panel_embedding.weight, std=0.02)
        nn.init.normal_(self.curve_embedding.weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @torch._dynamo.disable
    def _prepare_rope(self, num_tokens, device):
        """Precompute RoPE cos/sin for the given sequence length."""
        if self.feat_rope is None:
            return None, None
        cos, sin = self.feat_rope._compute_cos_sin(num_tokens, device)
        return cos, sin

    def forward(
        self,
        x,
        t=None,
        panel_points=None,
        panel_points_mask=None,
        panel_indices=None,
        mask=None,
        **kwargs
    ):
        B = x.shape[0]
        n_edge = self.n_edge_tokens

        # === Embed edge tokens ===
        x = x.reshape(B, self.n_panels, self.n_curves, -1)
        x = self.x_embedder_proj(x)  # (B, n_panels, n_curves, D)

        # === Embed context (panel points) ===
        ctx = self.point_encoder(panel_points)  # (B, N_padded, D)
        if self.use_panel_embedding and panel_indices is not None:
            panel_emb = self.panel_embedding(panel_indices)
            ctx = ctx + panel_emb

        x = x + self.panel_embedding.weight[:, None, :]  # (n_panels, 1, D)
        curve_emb = self.curve_embedding.weight[None, :, :]  # (1, n_curves, D)
        x = x + curve_emb
        x = x.reshape(B, n_edge, self.hidden_size)

        # === Unpad context using mask ===
        if panel_points_mask is not None:
            ctx_flat, ctx_indices, cu_ctx_lens, max_ctx_len = unpad_input(ctx, panel_points_mask)
        else:
            N_ctx = ctx.shape[1]
            ctx_flat = ctx.reshape(B * N_ctx, self.hidden_size)
            cu_ctx_lens = torch.arange(0, (B + 1) * N_ctx, N_ctx,
                                       dtype=torch.int32, device=x.device)
            max_ctx_len = N_ctx

        # === Flatten edge tokens ===
        x_flat = x.reshape(B * n_edge, self.hidden_size)
        cu_x_lens = torch.arange(0, (B + 1) * n_edge, n_edge,
                                 dtype=torch.int32, device=x.device)
        max_x_len = n_edge

        # === Timestep embedding — repeat per token for flat ops ===
        t_emb = self.t_embedder(t)  # (B, D)
        c = t_emb.repeat_interleave(n_edge, dim=0)  # (B*N, D)

        # === Precompute RoPE ===
        rope_cos, rope_sin = self._prepare_rope(n_edge, x.device)
        if rope_cos is not None:
            rope_cos = rope_cos.repeat(B, 1)
            rope_sin = rope_sin.repeat(B, 1)

        # === Transformer blocks ===
        for block in self.blocks:
            if self.use_checkpoint:
                x_flat = checkpoint(
                    block, x_flat, c, ctx_flat,
                    cu_x_lens, max_x_len, cu_ctx_lens, max_ctx_len,
                    rope_cos, rope_sin,
                    use_reentrant=True
                )
            else:
                x_flat = block(
                    x_flat, c, ctx_flat,
                    cu_x_lens, max_x_len, cu_ctx_lens, max_ctx_len,
                    rope_cos, rope_sin
                )

        # === Final layer ===
        x_flat = self.final_layer(x_flat, c)

        # === Reshape back to (B, n_edge, out_channels) ===
        x = x_flat.reshape(B, n_edge, self.out_channels)
        return {"pred": x}

    def forward_with_cfg(
        self,
        x,
        t=None,
        panel_points=None,
        panel_points_mask=None,
        panel_indices=None,
        mask=None,
        cfg_interval=None,
        cfg_interval_start=None,
        **kwargs
    ):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward(
            combined, t,
            panel_points=panel_points,
            panel_points_mask=panel_points_mask,
            panel_indices=panel_indices,
            mask=mask
        )["pred"]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

        if cfg_interval is True:
            timestep = t[0]
            if timestep < cfg_interval_start:
                half_eps = cond_eps

        eps = torch.cat([half_eps, half_eps], dim=0)
        return {"pred": eps}

    def apply_fsdp2(self, device_mesh, mp_policy: MixedPrecisionPolicy,
                    reshard_after_forward: bool = True):
        fully_shard(self.x_embedder_proj, mesh=device_mesh, mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward)
        fully_shard(self.t_embedder, mesh=device_mesh, mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward)
        fully_shard(self.point_encoder, mesh=device_mesh, mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward)
        for block in self.blocks:
            fully_shard(block, mesh=device_mesh, mp_policy=mp_policy,
                        reshard_after_forward=reshard_after_forward)
        fully_shard(self.final_layer, mesh=device_mesh, mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward)
        fully_shard(self, mesh=device_mesh, mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward)
        return self
