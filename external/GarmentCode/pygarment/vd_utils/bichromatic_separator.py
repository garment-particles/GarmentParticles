import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPoint, LineString, MultiLineString, box
from shapely.ops import unary_union, polygonize, linemerge
from typing import Tuple, List, Optional


# ----------------------------
# Utilities
# ----------------------------
def clip_polygon_to_aabb(poly: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Clip a polygon (Nx2) to an axis-aligned bounding box (xmin, ymin, xmax, ymax).
    Returns possibly empty polygon as (M,2). Works with convex or concave polys.
    """
    xmin, ymin, xmax, ymax = bbox

    def clip_edge(vertices: np.ndarray, edge: str) -> np.ndarray:
        if len(vertices) == 0:
            return vertices
        out = []
        for i in range(len(vertices)):
            A = vertices[i - 1]
            B = vertices[i]

            def inside(P):
                x, y = P
                if edge == "left":   return x >= xmin
                if edge == "right":  return x <= xmax
                if edge == "bottom": return y >= ymin
                if edge == "top":    return y <= ymax
                raise ValueError("bad edge")

            def intersect(A, B):
                x1, y1 = A; x2, y2 = B
                if edge in ("left", "right"):
                    xE = xmin if edge == "left" else xmax
                    if x2 == x1:  # parallel; shouldn't happen for intersection calc
                        t = 0.0
                    else:
                        t = (xE - x1) / (x2 - x1)
                    yE = y1 + t * (y2 - y1)
                    return np.array([xE, yE], float)
                else:  # bottom/top
                    yE = ymin if edge == "bottom" else ymax
                    if y2 == y1:
                        t = 0.0
                    else:
                        t = (yE - y1) / (y2 - y1)
                    xE = x1 + t * (x2 - x1)
                    return np.array([xE, yE], float)

            Ain = inside(A); Bin = inside(B)
            if Ain and Bin:
                out.append(B)
            elif Ain and not Bin:
                out.append(intersect(A, B))
            elif (not Ain) and Bin:
                out.append(intersect(A, B))
                out.append(B)
            # else both out: add nothing
        return np.asarray(out, float)

    poly = clip_edge(poly, "left")
    poly = clip_edge(poly, "right")
    poly = clip_edge(poly, "bottom")
    poly = clip_edge(poly, "top")
    return poly

def _perp(v):
    return np.array([v[1], -v[0]])

def _voronoi_finite_polygons_2d(vor: Voronoi, radius: Optional[float] = None):
    """
    Reconstruct infinite Voronoi regions to finite polygons by extending ridges
    to a large radius. Returns (regions, vertices), where:
      - regions: list of lists of vertex indices (into returned vertices)
      - vertices: (M,2) vertices array (original + appended far points)
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Only 2D points are supported.")

    new_regions: List[List[int]] = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points).max() * 2.0

    # Map from site to its ridges
    all_ridges = {}
    for (p, q), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p, []).append((q, v1, v2))
        all_ridges.setdefault(q, []).append((p, v1, v2))

    for p_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if len(region) == 0:
            new_regions.append([])
            continue

        if all(v >= 0 for v in region):
            # already finite
            new_regions.append(region)
            continue

        # Rebuild non-finite region
        ridges = all_ridges[p_idx]
        new_region = [v for v in region if v >= 0]

        for q_idx, v1, v2 in ridges:
            if v1 >= 0 and v2 >= 0:
                continue  # finite ridge
            # direction perpendicular to the edge between sites
            t = vor.points[q_idx] - vor.points[p_idx]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            # pick direction that points "outwards"
            midpoint = (vor.points[p_idx] + vor.points[q_idx]) / 2
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v1 if v1 >= 0 else v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        # order vertices CCW
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)].tolist()
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)

def _make_bbox(points, scale=1.3):
    pts = np.vstack(points)
    minx, miny = pts.min(axis=0)
    maxx, maxy = pts.max(axis=0)
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    w, h = (maxx - minx), (maxy - miny)
    w = max(w, 1e-6)
    h = max(h, 1e-6)
    # pad box
    w2, h2 = (w * scale) / 2.0, (h * scale) / 2.0
    return box(cx - w2, cy - h2, cx + w2, cy + h2)

# ----------------------------
# Main: bisector & regions
# ----------------------------
def bichromatic_voronoi_separator(
    A: np.ndarray, 
    B: np.ndarray, 
    bbox: Optional[Tuple[float, float, float, float]] = None,
    pad: float = 0.1,
):
    """
    Compute polygons for regions closer to A and closer to B (nearest-neighbor distance),
    and their shared bisector polyline(s) within a bounding window.

    Args:
        A, B: (n,2) numpy arrays
        bbox: optional shapely Polygon for clipping; if None, a padded box is used
        bbox_scale: scale for automatic bbox if bbox is None

    Returns:
        regionA_poly (shapely geometry): polygon/multipolygon - points closer to A than B
        regionB_poly (shapely geometry): polygon/multipolygon - points closer to B than A
        bisector (MultiLineString): lines equidistant to nearest A vs B (the separator)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    assert A.ndim == 2 and B.ndim == 2 and A.shape[1] == 2 and B.shape[1] == 2

    all_pts = np.vstack([A, B])
    labels = np.hstack([np.zeros(len(A), dtype=int), np.ones(len(B), dtype=int)])

    # Voronoi diagram for all points
    vor = Voronoi(all_pts)
    regions, vertices = _voronoi_finite_polygons_2d(vor)
    
    # Default bbox: padded data range
    if bbox is None:
        xmin, ymin = all_pts.min(axis=0)
        xmax, ymax = all_pts.max(axis=0)
        dx = xmax - xmin; dy = ymax - ymin
        xmin -= pad * dx; xmax += pad * dx
        ymin -= pad * dy; ymax += pad * dy
        bbox = (xmin, ymin, xmax, ymax)
    
    # Build clipped region polygon for each site
    polys_A = []
    polys_B = []
    for i, region in enumerate(regions):
        if len(region) == 0:
            continue
        poly = vertices[region]
        poly = clip_polygon_to_aabb(poly, bbox)
        poly = Polygon(poly)
        if poly.is_empty:
            continue
        if labels[i] == 0:
            polys_A.append(poly)
        else:
            polys_B.append(poly)


    box = _make_bbox(np.array(bbox).reshape(2, 2))
    regionA = unary_union(polys_A).intersection(box) if polys_A else Polygon()
    try:
        regionB = unary_union(polys_B).intersection(box) if polys_B else Polygon()
    except Exception as e:
        regionB = Polygon()

    return regionA, regionB

def voronoi_cells(points: np.ndarray,
                  bbox: Optional[Tuple[float, float, float, float]] = None,
                  pad: float = 0.1) -> List[np.ndarray]:
    """
    Compute Voronoi cells (finite polygons) from 2D points.
    Optionally clip to bbox=(xmin, ymin, xmax, ymax). If bbox is None, uses
    a padded bounding box around the data by 'pad' fraction of the range.

    Returns:
        cells: list of polygons (each as (Ni,2) float array), one per input site.
               Empty polygons may occur if degenerate/clipped away.

    Notes:
        - Duplicate points are removed (Voronoi requires unique sites).
        - Requires at least 3 non-collinear points.
    """
    P = np.asarray(points, float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points must be (N,2)")

    # Deduplicate (avoid Qhull errors)
    P_unique = np.unique(np.round(P, 12), axis=0)
    if len(P_unique) < 3:
        raise ValueError("Need at least 3 unique points for a Voronoi diagram.")
    # Check collinearity
    vecs = P_unique - P_unique.mean(axis=0)
    if np.linalg.matrix_rank(vecs) < 2:
        raise ValueError("Points are collinear; Voronoi diagram is undefined in 2D.")

    vor = Voronoi(P_unique)
    regions, vertices = _voronoi_finite_polygons_2d(vor)

    # Default bbox: padded data range
    if bbox is None:
        xmin, ymin = P_unique.min(axis=0)
        xmax, ymax = P_unique.max(axis=0)
        dx = xmax - xmin; dy = ymax - ymin
        xmin -= pad * dx; xmax += pad * dx
        ymin -= pad * dy; ymax += pad * dy
        bbox = (xmin, ymin, xmax, ymax)

    cells: List[np.ndarray] = []
    for region in regions:
        if len(region) == 0:
            print("Empty region")
            cells.append(np.empty((0, 2)))
            continue
        poly = vertices[region]
        poly = clip_polygon_to_aabb(poly, bbox)
        cells.append(poly)
    return cells, P_unique, bbox