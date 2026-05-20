# --------------------------------------------------------
# EVA-02: A Visual Representation for Neon Genesis
# Github source: https://github.com/baaivision/EVA/EVA02
# Copyright (c) 2023 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Yuxin Fang
#
# Based on https://github.com/lucidrains/rotary-embedding-torch
# --------------------------------------------------------'

from math import pi
import math

import torch
from torch import nn

from einops import rearrange, repeat



def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)



def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)




class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int = 150000.0,
        dtype: torch.dtype = torch.float32,
        initial_context_length: int = 4096,
        scaling_factor: float = 32.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta

    
    @torch._dynamo.disable  # keep this tiny bit out of torch.compile
    def _compute_concentration_and_inv_freq(self, device: torch.device):
        """YaRN-style RoPE with NTK-by-parts; numerically/compile friendly."""
        hd = int(self.head_dim)
        assert hd % 2 == 0, "head_dim must be even (cos/sin pairs)."

        # --- frequencies (keep in fp32, avoid pow) ---
        i = torch.arange(0, hd, 2, dtype=torch.float32, device=device)
        x = i / float(hd)                                   # [0, 1)
        freq = torch.exp(x * math.log(float(self.base)))    # == base ** x, but compiler-friendly

        if float(self.scaling_factor) > 1.0:
            # YaRN concentration
            concentration = 0.1 * math.log(float(self.scaling_factor)) + 1.0

            d_half = hd / 2.0
            log_base = math.log(float(self.base))

            # NTK-by-parts boundaries (compute as Python floats, then clamp/sort)
            low  = d_half * math.log(self.initial_context_length / (self.ntk_beta  * 2 * math.pi)) / log_base
            high = d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / log_base

            # Avoid compile-time symbolic comparisons by sorting & softly clamping
            low, high = (low, high) if low <= high else (high, low)
            low  = max(0.0,            min(low,  d_half - 1.0))
            high = max(low + 1e-3,     min(high, d_half - 1.0))  # keep denom > 0

            # Build ramp/mask in fp32
            j = torch.arange(int(d_half), dtype=torch.float32, device=device)
            ramp = (j - low) / (high - low)
            mask = 1.0 - ramp.clamp(0.0, 1.0)   # ∈ [0,1]

            # Interpolate/extrapolate in fp32, cast later if you need bf16 elsewhere
            inv_freq_interp = 1.0 / (float(self.scaling_factor) * freq)
            inv_freq_extra  = 1.0 / freq
            inv_freq = inv_freq_interp * (1.0 - mask) + inv_freq_extra * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        # Return fp32; cast to bf16 only at the final consumer, if desired
        return torch.tensor(concentration, dtype=torch.float32, device=device), inv_freq

    def _compute_cos_sin(self, num_tokens: int, device: torch.device):
        concentration, inv_freq = self._compute_concentration_and_inv_freq(device)
        t = torch.arange(num_tokens, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, num_tokens, _, head_dim = query.shape
        cos, sin = self._compute_cos_sin(num_tokens, device=query.device)

        query_shape = query.shape
        query = query.view(B, num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_dim)
            key = _apply_rotary_emb(key, cos, sin)
            key = key.reshape(key_shape)
            return query, key
        return query


class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum('..., f -> ... f', t, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)

        freqs_w = torch.einsum('..., f -> ... f', t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

        # print('======== shape of rope freq', self.freqs_cos.shape, '========')

    def forward(self, t, start_index = 0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim = -1)



class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if isinstance(pt_seq_len, list):
            pt_seq_len_h, pt_seq_len_w = pt_seq_len
        else:
            pt_seq_len_h = pt_seq_len
            pt_seq_len_w = pt_seq_len
            
        t_h = torch.arange(pt_seq_len_h) / pt_seq_len_h * pt_seq_len_h
        t_w = torch.arange(pt_seq_len_w) / pt_seq_len_w * pt_seq_len_w
        
        freqs_h = torch.einsum('..., f -> ... f', t_h, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)

        freqs_w = torch.einsum('..., f -> ... f', t_w, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        # print('======== shape of rope freq', self.freqs_cos.shape, '========')

    def forward(self, t): return  t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class UVRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        uv_scale=1,
        theta=10000,
        freqs_for='lang',
        max_freq=10,
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.uv_scale = uv_scale
        
        # Generate frequency basis (same as VisionRotaryEmbeddingFast)
        if freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        else:
            raise ValueError(f'unknown modality {freqs_for}')
            
        self.register_buffer("freqs", freqs)
        
    def forward(self, t, uv_coords):
        """
        Apply UV-based rotary embedding to attention features.
        
        Args:
            t: (B, num_heads, N, head_dim) attention features (q or k)
            uv_coords: (B, N, 2) UV coordinates in [0, 1]
            
        Returns:
            t: (B, num_heads, N, head_dim) rotated features
        """
        B, num_heads, N, head_dim = t.shape
        
        # Scale UV coordinates to match original uv_scale range
        uv_scaled = uv_coords * self.uv_scale  # [0, 1] → [0, uv_scale]
        
        # Create frequency embeddings for U and V separately
        freqs_u = torch.einsum('bn,d->bnd', uv_scaled[..., 0], self.freqs)  # (B, N, dim//2)
        freqs_v = torch.einsum('bn,d->bnd', uv_scaled[..., 1], self.freqs)  # (B, N, dim//2)
        
        # Repeat for sine/cosine pairs
        freqs_u = repeat(freqs_u, 'b n d -> b n (d r)', r=2)
        freqs_v = repeat(freqs_v, 'b n d -> b n (d r)', r=2)
        
        # Combine U and V frequencies (concatenate along feature dim)
        freqs = torch.cat([freqs_u, freqs_v], dim=-1)  # (B, N, dim)
        
        # Expand for attention heads
        freqs = freqs.unsqueeze(1).expand(-1, num_heads, -1, -1)  # (B, num_heads, N, dim)
        
        # Apply rotation
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()
        
        return t * freqs_cos + rotate_half(t) * freqs_sin
    
    
if __name__ == '__main__':
    rand_query = torch.randn(16, 1024, 16, 72).cuda()
    embed = RotaryEmbedding(72, 10000, 4096, 1.0, 1.0, 32.0).cuda()
    query = embed(rand_query)
    print(query.shape)
    