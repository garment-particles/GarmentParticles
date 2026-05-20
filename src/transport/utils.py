import torch as th

class EasyDict:

    def __init__(self, sub_dict):
        for k, v in sub_dict.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

def mean_flat(x, mask=None):
    """
    Take the mean over all non-batch dimensions.
    """
    if mask is None:
        return th.mean(x, dim=list(range(1, len(x.size()))))
    mask = mask.float()
    return th.sum(x * mask, dim=list(range(1, len(x.size())))) / mask.sum(dim=list(range(1, len(x.size()))))

def log_state(state):
    result = []
    
    sorted_state = dict(sorted(state.items()))
    for key, value in sorted_state.items():
        # Check if the value is an instance of a class
        if "<object" in str(value) or "object at" in str(value):
            result.append(f"{key}: [{value.__class__.__name__}]")
        else:
            result.append(f"{key}: {value}")
    
    return '\n'.join(result)

# Sample from categorical distribution for each position using the transition probabilities
def _sample_tokens(probs: th.Tensor) -> th.Tensor:
    """Sample one token per position from probability distribution.
    Args:
        probs: [batch_size, seq_len, vocab_size] transition probabilities
    Returns:
        [batch_size, seq_len] sampled token indices
    """
    batch_size, seq_len, vocab_size = probs.shape
    flat_probs = probs.view(-1, vocab_size)
    samples = th.multinomial(flat_probs, num_samples=1)
    return samples.view(batch_size, seq_len)


def sample_polygon_curves(curves: th.Tensor,
                          num_samples_per_curve: int = 16) -> th.Tensor:
    """
    curves: (..., n_curves, 7)
        [..., i, :] = [endpoint_x, endpoint_y, ctrl0_x, ctrl0_y, ctrl1_x, ctrl1_y, flag]
        flag = 0 -> cubic Bézier
        flag = 1 -> arc, with ctrl0, ctrl1 both equal to the arc midpoint
    Returns:
        pts: (..., n_curves * num_samples_per_curve, 2)
    """
    orig_shape = curves.shape
    if curves.dim() == 2:
        # add batch dim if missing
        curves = curves.unsqueeze(0)  # (1, n_curves, 7)

    B, N, C = curves.shape
    assert C == 7, "Expected last dim = 7"

    device = curves.device
    dtype = curves.dtype

    endpoints = curves[..., :2]      # (B, N, 2)
    endpoints = th.cumsum(endpoints, dim=1)
    ctrl     = curves[..., 2:6]      # (B, N, 4)
    flags    = curves[..., 6]        # (B, N)

    # Define start and end points for each curve:
    P1 = endpoints                             # (B, N, 2): end of each segment
    P0 = th.roll(endpoints, shifts=1, dims=1)  # (B, N, 2): start (previous endpoint)

    # Sample parameter t in [0, 1] along each curve
    t = th.linspace(0.0, 1.0, num_samples_per_curve,
                       device=device, dtype=dtype)  # (K,)
    t = t.view(1, 1, -1, 1)                         # (1, 1, K, 1)
    one_minus_t = 1.0 - t                           # (1, 1, K, 1)

    # Expand P0, P1 for broadcasting
    P0_e = P0.unsqueeze(2)   # (B, N, 1, 2)
    P1_e = P1.unsqueeze(2)   # (B, N, 1, 2)

    # ---- Cubic Bézier for flag = 0 ----
    C1 = ctrl[..., 0:2]      # (B, N, 2)
    C2 = ctrl[..., 2:4]      # (B, N, 2)
    C1_e = C1.unsqueeze(2)   # (B, N, 1, 2)
    C2_e = C2.unsqueeze(2)   # (B, N, 1, 2)

    # B(t) = (1-t)^3 P0 + 3(1-t)^2 t C1 + 3(1-t) t^2 C2 + t^3 P1
    bez_pts = (one_minus_t**3) * P0_e \
              + 3 * (one_minus_t**2) * t * C1_e \
              + 3 * one_minus_t * (t**2) * C2_e \
              + (t**3) * P1_e           # (B, N, K, 2)

    # ---- Arc approximation (quadratic Bézier) for flag = 1 ----
    # A circular arc between P0 and P1 with midpoint M can be
    # approximated by quadratic Bézier through (P0, M, P1).
    M  = ctrl[..., 0:2]      # (B, N, 2) - midpoint
    M_e = M.unsqueeze(2)     # (B, N, 1, 2)

    # Q(t) = (1-t)^2 P0 + 2(1-t)t M + t^2 P1
    arc_pts = (one_minus_t**2) * P0_e \
              + 2 * one_minus_t * t * M_e \
              + (t**2) * P1_e          # (B, N, K, 2)

    # ---- Select between Bézier and Arc using the flag ----
    # flag = 0 → Bézier, flag = 1 → Arc
    w_arc = th.clamp(flags, 0.0, 1.0).unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
    # pts = (1 - flag) * bez + flag * arc
    curve_pts = bez_pts * (1.0 - w_arc) + arc_pts * w_arc             # (B, N, K, 2)

    # Flatten curves into one set of points
    pts = curve_pts.reshape(B, N * num_samples_per_curve, 2)          # (B, NK, 2)

    if orig_shape.dim() == 2:
        pts = pts[0]  # drop batch dim for single example

    return pts