# --------------------- CDT via Triangle (PSLG input) ----------------------- #
import triangle as tr
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from shapely.geometry import Polygon
import numpy as np

# ---------------------------- Geometry helpers ---------------------------- #

def _ensure_ccw(ring: np.ndarray) -> np.ndarray:
    """Ensure ring (Nx2, open or closed) is CCW; return as OPEN ring."""
    ring = np.asarray(ring, dtype=float)
    if ring.ndim != 2 or ring.shape[1] != 2 or ring.shape[0] < 3:
        raise ValueError("ring must be (N,2) with N >= 3")

    # If closed, make it open for consistent processing
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]

    # Shoelace signed area (positive => CCW)
    x = ring[:, 0]
    y = ring[:, 1]
    area2 = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))

    return ring if area2 > 0 else ring[::-1]


def _ring_segments(indices: List[int]) -> List[Tuple[int, int]]:
    """Return segment index pairs to close the ring."""
    segs = []
    n = len(indices)
    for i in range(n):
        segs.append((indices[i], indices[(i + 1) % n]))
    return segs


def _point_in_ring_centroid(ring: np.ndarray) -> Tuple[float, float]:
    """A point guaranteed inside the hole ring (use polygon centroid)."""
    poly = Polygon(ring)
    c = poly.representative_point()  # always inside
    return (c.x, c.y)


def _dedup_points(points: List[Tuple[float, float]], tol=0.0) -> Tuple[np.ndarray, Dict[Tuple[float, float], int]]:
    """Deduplicate points; return (unique Nx2 array, mapping original->index).
       If tol>0, round coordinates to that many decimals before dedup."""
    idx_map: Dict[Tuple[float, float], int] = {}
    unique = []
    for p in points:
        key = p if tol == 0 else (round(p[0], tol), round(p[1], tol))
        if key not in idx_map:
            idx_map[key] = len(unique)
            unique.append(p)
    return np.asarray(unique, dtype=float), idx_map

@dataclass
class CDTResult:
    vertices: np.ndarray          # (N, 2)
    triangles: np.ndarray         # (T, 3) vertex indices
    segments: np.ndarray          # (M, 2) boundary segment indices (outer + holes)


def constrained_delaunay_triangulation(
    outer_ring: np.ndarray,
    holes: Optional[List[np.ndarray]] = None,
    interior_points: Optional[np.ndarray] = None,
    min_angle: float = 25.0,
    max_area: Optional[float] = None,
) -> CDTResult:
    """
    Build a CDT of a polygon with optional holes and optional interior points.
    Uses the 'triangle' package (Shewchuk's Triangle).

    Parameters
    ----------
    outer_ring : (N,2) ndarray, CCW open or closed
    holes : list of (H_i,2) ndarrays (each hole ring CW or CCW; orientation will be fixed)
    interior_points : (K,2) ndarray of extra points to include as vertices
    min_angle : float, quality constraint (degrees)
    max_area : optional float, area upper bound per triangle

    Returns
    -------
    CDTResult
    """

    outer = _ensure_ccw(np.asarray(outer_ring, dtype=float))
    holes = holes or []
    holes = [_ensure_ccw(np.asarray(h, dtype=float)) for h in holes]

    # Build PSLG: unique vertices + segments
    all_coords: List[Tuple[float, float]] = []
    outer_idx = []
    for p in outer:
        all_coords.append((float(p[0]), float(p[1])))
        outer_idx.append(len(all_coords) - 1)

    hole_indices = []
    for h in holes:
        idxs = []
        for p in h:
            all_coords.append((float(p[0]), float(p[1])))
            idxs.append(len(all_coords) - 1)
        hole_indices.append(idxs)

    # Deduplicate in case rings share points
    V, mapping = _dedup_points(all_coords, tol=12)

    # Rebuild index lists after dedup
    def remap(idxs_raw: List[int]) -> List[int]:
        remapped = []
        for k in idxs_raw:
            key = all_coords[k]
            keyr = (round(key[0], 12), round(key[1], 12))
            remapped.append(mapping[keyr] if keyr in mapping else mapping[key])
        return remapped

    outer_idx = remap(outer_idx)
    holes_idx = [remap(idxs) for idxs in hole_indices]

    segs = _ring_segments(outer_idx)
    for idxs in holes_idx:
        segs += _ring_segments(idxs)

    # Triangle expects "holes" as a point INSIDE each hole polygon
    triangle_holes = []
    for h in holes:
        triangle_holes.append(_point_in_ring_centroid(h))

    # Add interior points (forced vertices) if provided
    if interior_points is not None and len(interior_points) > 0:
        interior_points = np.asarray(interior_points, dtype=float)
        V = np.vstack([V, interior_points])

    A = dict(vertices=V, segments=np.asarray(segs, dtype=int))
    if len(triangle_holes) > 0:
        A["holes"] = np.asarray(triangle_holes, dtype=float)

    # Triangle options:
    #  p : PSLG input       D : constrained Delaunay
    #  qXX : min angle      aXX : max area
    opts = f"p"
    # if max_area is not None and max_area > 0:
    #     opts += f"a{max_area}"

    T = tr.triangulate(A, opts)

    verts = np.asarray(T["vertices"], dtype=float)
    tris = np.asarray(T["triangles"], dtype=int)
    segs_out = np.asarray(T["segments"], dtype=int) if "segments" in T else np.asarray(segs, dtype=int)

    return CDTResult(vertices=verts, triangles=tris, segments=segs_out)