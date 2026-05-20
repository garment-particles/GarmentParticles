"""Generic utility functions"""

import numpy as np

def list_to_c(num):
    """Convert 2D list or list of 2D lists into complex number/list of complex numbers"""
    if isinstance(num[0], list) or isinstance(num[0], np.ndarray):
        return [complex(n[0], n[1]) for n in num]
    else: 
        return complex(num[0], num[1])
    
def c_to_np(num):
    """Convert complex number to a numpy array of 2 elements"""
    return np.asarray([num.real, num.imag])

def vector_angle(v1, v2):
    """Find an angle between two 2D vectors"""
    v1, v2 = np.asarray(v1), np.asarray(v2)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos = max(min(cos, 1), -1)  # NOTE: getting rid of numbers like 1.000002 that appear due to numerical instability
    angle = np.arccos(cos) 
    # Cross to indicate correct relative orienataion of v2 w.r.t. v1
    cross = np.cross(v1, v2)
    
    if abs(cross) > 1e-5:
        angle *= np.sign(cross)
    return angle

def c_to_list(num):
    """Convert complex number to a list of 2 elements
        Allows processing of lists of complex numbers
    """

    if isinstance(num, (list, tuple, set, np.ndarray)):
        return [c_to_list(n) for n in num]
    else:
        return [num.real, num.imag]
    
def close_enough(f1, f2=0, tol=1e-4):
    """Compare two floats correctly """
    return abs(f1 - f2) < tol

# Vector local coodinates conversion
def rel_to_abs_2d(start, end, rel_point):
    """
        Converts coordinates expressed in a coordinate frame local
        to the edge [start, end] into edge vertices (global) coordinate frame
    """

    start, end = np.array(start), np.array(end)  # in case inputs are lists/tuples
    edge = end - start
    edge_perp = np.array([-edge[1], edge[0]])

    abs_start = start + rel_point[0] * edge
    abs_point = abs_start + rel_point[1] * edge_perp

    return abs_point

def abs_to_rel_2d(start, end, abs_point, as_vector=False):
    """
        Converts coordinates expressed in a global coordinate frame into 
        a frame local to the edge [start, end] 
    """

    start, end, abs_point = np.array(start), np.array(end), \
        np.array(abs_point)

    rel_point = [None, None]
    edge_vec = end - start
    edge_len = np.linalg.norm(edge_vec)
    point_vec = abs_point if as_vector else abs_point - start  # vector or point
    
    # X
    # project control_vec on edge_vec by dot product properties
    projected_len = edge_vec.dot(point_vec) / edge_len 
    rel_point[0] = projected_len / edge_len
    # Y
    projected = edge_vec * rel_point[0]
    vert_comp = point_vec - projected  
    rel_point[1] = np.linalg.norm(vert_comp) / edge_len

    # Distinguish left&right curvature
    rel_point[1] *= np.sign(np.cross(edge_vec, point_vec))

    return np.asarray(rel_point)

# Arcs converters
def arc_from_three_points(start, end, point_on_arc):
    """Create a circle arc from 3 points (start, end and any point on an arc)
    
        NOTE: Control point specified in the same coord system as start and end
        NOTE: points should not be on the same line
    """

    nstart, nend, npoint_on_arc = np.asarray(start), np.asarray(end), np.asarray(point_on_arc)

    # https://stackoverflow.com/a/28910804
    # Using complex numbers to calculate the center & radius
    x, y, z = list_to_c([start, point_on_arc, end]) 
    w = z - x
    w /= y - x
    c = (x - y)*(w - abs(w)**2)/2j/w.imag - x
    # NOTE center = [c.real, c.imag]
    rad = abs(c + x)

    # Large/small arc
    mid_dist = np.linalg.norm(npoint_on_arc - ((nstart + nend) / 2))

    # Orientation
    angle = vector_angle(npoint_on_arc - nstart, nend - nstart)  # +/-

    return (start, end, rad, mid_dist > rad, angle > 0) 


def sample_path_arclength(path,
                          num_samples: int = None,
                          spacing: float = None,
                          include_endpoint: bool = True) -> np.ndarray:
    """
    Convert an svgpathtools.Path to a set of points sampled uniformly by arc length.

    Args:
        path: svgpathtools.Path (can contain Line, CubicBezier, QuadraticBezier, Arc, etc.)
        num_samples: total number of samples along the whole path (overrides spacing if given).
        spacing: desired spacing between consecutive samples in the same units as the path.
        include_endpoint: whether to include the very end point of the path.

    Returns:
        (N, 2) float64 array of [x, y] points in SVG coordinates (y grows downward).
    """
    # total length
    L = path.length()
    if L == 0:
        # Path with no length (e.g., all moves). Return its start point once.
        z0 = path.start
        return np.array([[z0.real, z0.imag]], dtype=np.float64)

    # Decide sampling distances
    if num_samples is not None and num_samples >= 2:
        s_vals = np.linspace(0.0, L, num_samples, endpoint=include_endpoint)
    else:
        if spacing is None or spacing <= 0:
            spacing = L / 200.0  # sensible default: ~200 samples
        # number of steps (ensure at least 2)
        n = int(np.floor(L / spacing)) + 1
        if include_endpoint:
            s_vals = np.linspace(0.0, L, n, endpoint=True)
        else:
            # avoid hitting the exact end if caller asks so
            s_vals = np.linspace(0.0, L, n, endpoint=False)

    # Precompute per-segment lengths and cumulative sum
    seg_lengths = [seg.length() for seg in path]
    cum = np.cumsum([0.0] + seg_lengths)  # cum[k] is length up to start of segment k

    def locate_segment(s):
        # find the segment index k such that cum[k] <= s <= cum[k+1]
        # handle s == L (end of last segment)
        if s >= cum[-1]:
            return len(path) - 1, seg_lengths[-1]
        k = np.searchsorted(cum, s, side='right') - 1
        return k, s - cum[k]

    pts = []
    for s in s_vals:
        k, local_s = locate_segment(s)
        seg = path[k]
        # invert arc length on the segment to get the local parameter t in [0, 1]
        # ilength returns t such that length(0->t) == local_s
        t = seg.ilength(local_s)
        z = seg.point(t)
        pts.append([z.real, z.imag])

    return np.asarray(pts, dtype=np.float64)