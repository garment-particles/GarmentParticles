import torch as th
import math
from chamferdist import ChamferDistance

def sample_polygon_curves_exact(curves: th.Tensor, flags: th.Tensor,
                          num_samples_per_curve: int = 16) -> th.Tensor:
    """
    curves: (..., n_curves, 7)
        [..., i, :] = [endpoint_x, endpoint_y, ctrl0_x, ctrl0_y, ctrl1_x, ctrl1_y, flag]
        flag = 0 -> cubic Bézier
        flag = 1 -> arc, with ctrl0, ctrl1 both equal to the arc midpoint
    Returns:
        pts: (..., n_curves * num_samples_per_curve, 2)
    """
    def rel_to_abs_2d(start, end, rel_point):
        """
        Converts coordinates expressed in a coordinate frame local
        to the edge [start, end] into edge vertices (global) coordinate frame
        """
        edge = end - start
        edge_perp = th.stack([-edge[..., 1], edge[..., 0]], dim=-1)
        abs_start = start + rel_point[..., :1] * edge
        abs_point = abs_start + rel_point[..., 1:] * edge_perp
        return abs_point

    def sample_arc_polar(P0, P1, M, num_samples):
        """
        Exact arc sampling using polar coordinates.
        Given three points on a circle (P0, M, P1), find center and sample the arc.
        
        P0, P1, M: (B, N, 2)
        Returns: (B, N, K, 2)
        """
        B, N, _ = P0.shape
        device = P0.device
        dtype = P0.dtype
        
        # Find circle center from three points using perpendicular bisectors
        # Midpoint of P0-M and P1-M
        mid_0m = (P0 + M) / 2  # (B, N, 2)
        mid_1m = (P1 + M) / 2  # (B, N, 2)
        
        # Direction vectors
        d_0m = M - P0  # (B, N, 2)
        d_1m = M - P1  # (B, N, 2)
        
        # Perpendicular directions (rotate 90 degrees)
        perp_0m = th.stack([-d_0m[..., 1], d_0m[..., 0]], dim=-1)  # (B, N, 2)
        perp_1m = th.stack([-d_1m[..., 1], d_1m[..., 0]], dim=-1)  # (B, N, 2)
        
        # Find intersection of perpendicular bisectors
        # Line 1: mid_0m + s * perp_0m
        # Line 2: mid_1m + t * perp_1m
        # Solve: mid_0m + s * perp_0m = mid_1m + t * perp_1m
        
        # Using Cramer's rule for 2x2 system:
        # perp_0m * s - perp_1m * t = mid_1m - mid_0m
        diff = mid_1m - mid_0m  # (B, N, 2)
        
        # Determinant: perp_0m.x * (-perp_1m.y) - perp_0m.y * (-perp_1m.x)
        #            = perp_1m.x * perp_0m.y - perp_0m.x * perp_1m.y
        det = perp_1m[..., 0] * perp_0m[..., 1] - perp_0m[..., 0] * perp_1m[..., 1]  # (B, N)
        
        # Handle degenerate case (collinear points - infinite radius)
        eps = 1e-8
        det_safe = th.where(det.abs() < eps, th.ones_like(det) * eps, det)
        
        # s = (diff.x * (-perp_1m.y) - diff.y * (-perp_1m.x)) / det
        #   = (perp_1m.x * diff.y - diff.x * perp_1m.y) / det
        s = (perp_1m[..., 0] * diff[..., 1] - diff[..., 0] * perp_1m[..., 1]) / det_safe  # (B, N)
        
        # Circle center
        center = mid_0m + s.unsqueeze(-1) * perp_0m  # (B, N, 2)
        
        # Radius
        radius = th.norm(P0 - center, dim=-1, keepdim=True)  # (B, N, 1)
        
        # Angles for P0, M, P1 relative to center
        def get_angle(point, center):
            delta = point - center  # (B, N, 2)
            return th.atan2(delta[..., 1], delta[..., 0])  # (B, N)
        
        theta0 = get_angle(P0, center)  # (B, N)
        theta_m = get_angle(M, center)  # (B, N)
        theta1 = get_angle(P1, center)  # (B, N)
        
        # Determine arc direction by checking if M is on the shorter or longer arc
        # Normalize angles to [0, 2pi) relative to theta0
        def normalize_angle(angle):
            return th.remainder(angle, 2 * math.pi)
        
        # Shift angles so theta0 is at 0
        theta_m_rel = normalize_angle(theta_m - theta0)  # (B, N)
        theta1_rel = normalize_angle(theta1 - theta0)    # (B, N)
        
        # Check if going counterclockwise from P0 passes through M before P1
        # If theta_m_rel < theta1_rel, then CCW direction is correct
        # Otherwise, we need to go CW (negative direction)
        go_ccw = theta_m_rel < theta1_rel  # (B, N)
        
        # For CCW: sweep from 0 to theta1_rel
        # For CW: sweep from 0 to (theta1_rel - 2pi), i.e., negative sweep
        sweep_ccw = theta1_rel
        sweep_cw = theta1_rel - 2 * math.pi
        sweep = th.where(go_ccw, sweep_ccw, sweep_cw)  # (B, N)
        
        # Sample parameter t in [0, 1]
        t = th.linspace(0, 1, num_samples, device=device, dtype=dtype)  # (K,)
        t = t.view(1, 1, -1)  # (1, 1, K)
        
        # Interpolate angles
        theta0_e = theta0.unsqueeze(-1)   # (B, N, 1)
        sweep_e = sweep.unsqueeze(-1)     # (B, N, 1)
        theta = theta0_e + t * sweep_e    # (B, N, K)
        
        # Convert polar to Cartesian
        center_e = center.unsqueeze(2)    # (B, N, 1, 2)
        radius_e = radius.unsqueeze(2)    # (B, N, 1, 1)
        
        x = center_e[..., 0] + radius_e[..., 0] * th.cos(theta)  # (B, N, K)
        y = center_e[..., 1] + radius_e[..., 0] * th.sin(theta)  # (B, N, K)
        
        arc_pts = th.stack([x, y], dim=-1)  # (B, N, K, 2)
        
        # Handle degenerate case: if det ≈ 0, fall back to linear interpolation
        is_degenerate = (det.abs() < eps).unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
        P0_e = P0.unsqueeze(2)  # (B, N, 1, 2)
        P1_e = P1.unsqueeze(2)  # (B, N, 1, 2)
        t_lin = t.unsqueeze(-1)  # (1, 1, K, 1)
        linear_pts = P0_e * (1 - t_lin) + P1_e * t_lin  # (B, N, K, 2)
        
        arc_pts = th.where(is_degenerate, linear_pts, arc_pts)
        
        return arc_pts

    orig_shape = curves.shape
    if curves.dim() == 2:
        curves = curves.unsqueeze(0)  # (1, n_curves, 7)

    B, N, C = curves.shape
    assert C == 6, "Expected last dim = 6"

    device = curves.device
    dtype = curves.dtype

    endpoints = curves[..., :2]       # (B, N, 2)
    endpoints = th.cumsum(endpoints, dim=1)
    ctrl1 = curves[..., 2:4]          # (B, N, 2)
    ctrl2 = curves[..., 4:6]          # (B, N, 2)

    # Define start and end points for each curve
    P1 = endpoints                              # (B, N, 2): end of each segment
    P0 = th.roll(endpoints, shifts=1, dims=1)   # (B, N, 2): start (previous endpoint)

    # Sample parameter t in [0, 1] for Bézier curves
    t = th.linspace(0, 1, num_samples_per_curve // 2, device=device, dtype=dtype) ** (1 / 3)
    t = th.cat([t, th.linspace(0, 1, num_samples_per_curve // 2, device=device, dtype=dtype) ** 3], dim=-1)
    t = t.view(1, 1, -1, 1)           # (1, 1, K, 1)
    one_minus_t = 1.0 - t             # (1, 1, K, 1)

    # Expand P0, P1 for broadcasting
    P0_e = P0.unsqueeze(2)   # (B, N, 1, 2)
    P1_e = P1.unsqueeze(2)   # (B, N, 1, 2)

    # ---- Cubic Bézier for flag = 0 ----
    C1 = rel_to_abs_2d(P0, P1, ctrl1)  # (B, N, 2)
    C2 = rel_to_abs_2d(P0, P1, ctrl2)  # (B, N, 2)
    C1_e = C1.unsqueeze(2)   # (B, N, 1, 2)
    C2_e = C2.unsqueeze(2)   # (B, N, 1, 2)

    # B(t) = (1-t)^3 P0 + 3(1-t)^2 t C1 + 3(1-t) t^2 C2 + t^3 P1
    bez_pts = (one_minus_t**3) * P0_e \
              + 3 * (one_minus_t**2) * t * C1_e \
              + 3 * one_minus_t * (t**2) * C2_e \
              + (t**3) * P1_e           # (B, N, K, 2)

    # ---- Exact arc sampling using polar coordinates ----
    M = rel_to_abs_2d(P0, P1, ctrl1)   # (B, N, 2) - midpoint on arc
    arc_pts = sample_arc_polar(P0, P1, M, num_samples_per_curve)  # (B, N, K, 2)

    # ---- Select between Bézier and Arc using the flag ----
    w_arc = th.clamp(flags, 0.0, 1.0).unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
    curve_pts = bez_pts * (1.0 - w_arc) + arc_pts * w_arc          # (B, N, K, 2)

    # Flatten curves into one set of points
    pts = curve_pts.reshape(B, N * num_samples_per_curve, 2)       # (B, NK, 2)

    if len(orig_shape) == 2:
        pts = pts[0]

    return pts


def get_shape_loss(preds: th.Tensor, pred_transformation: th.Tensor, gt_points: th.Tensor, mask: th.Tensor, normalization, gt_flag: th.Tensor) -> th.Tensor:
    edge_std = th.tensor(normalization.edge_std).to(preds.device)
    edge_mean = th.tensor(normalization.edge_mean).to(preds.device)
    transf_std = th.tensor(normalization.transformation_std).to(preds.device)
    transf_mean = th.tensor(normalization.transformation_mean).to(preds.device)
    preds = preds * edge_std + edge_mean
    pred_transformation = pred_transformation * transf_std + transf_mean
    B, n_panels, n_curves = preds.shape[:3]
    preds = preds.reshape(B*n_panels, n_curves, -1)
    gt_flag = gt_flag.reshape(B*n_panels, n_curves)
    pred_points = sample_polygon_curves_exact(preds, gt_flag)
    chamfer_loss = chamfer_loss(pred_points, gt_points)
    square_error = (preds - gt_points) ** 2
    square_error = square_error.sum(dim=-1)
    square_error = square_error.mean()
    return square_error