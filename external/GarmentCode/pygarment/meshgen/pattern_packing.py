"""UV-independent packing for semantically arranged sewing-pattern panels.

This module owns the geometry and hierarchy logic used to lay pattern panels
out in a plane.  Texture/UV code can adapt mesh islands to ``PackingPanel``
objects, but the packing algorithm itself does not know about UV coordinates,
texture faces, materials, or image normalization.
"""

from collections import deque
from itertools import combinations
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.ops import unary_union


INDIVIDUAL = "individual"
HIERARCHICAL = "hierarchical"
FINE_TO_COARSE = "fine_to_coarse"
JOINT_OPTIMIZATION = "joint_optimization"
_VALID_STRATEGIES = {
    INDIVIDUAL,
    HIERARCHICAL,
    FINE_TO_COARSE,
    JOINT_OPTIMIZATION,
}
_DEFAULT_PANEL_TREE_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "panel_tree.json"
)


class PanelHierarchyNode:
    """Small dependency-free node used by the semantic packing hierarchy."""

    def __init__(self, name, parent=None):
        self.name = str(name)
        self.children = []
        self._parent = None
        self.parent = parent

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        if self._parent is parent:
            return
        if self._parent is not None:
            self._parent.children.remove(self)
        self._parent = parent
        if parent is not None:
            parent.children.append(self)


def _level_order_iter(root):
    """Yield nodes breadth-first using the small tree protocol we need."""

    pending = deque([root])
    while pending:
        node = pending.popleft()
        yield node
        pending.extend(node.children)


def _level_order_groups(root):
    """Yield one tuple of nodes per tree depth."""

    level = [root]
    while level:
        yield tuple(level)
        level = [
            child
            for node in level
            for child in node.children
        ]


def iter_panel_hierarchy(root):
    """Public breadth-first iterator for a packing hierarchy."""

    return _level_order_iter(root)


def _find_node(root, name):
    return next(
        (node for node in _level_order_iter(root) if node.name == name),
        None,
    )


def _find_child(parent, name):
    return next(
        (child for child in parent.children if child.name == name),
        None,
    )


def _node_depth(node):
    depth = 0
    while node.parent is not None:
        node = node.parent
        depth += 1
    return depth


def _is_ancestor(ancestor, node):
    node = node.parent
    while node is not None:
        if node is ancestor:
            return True
        node = node.parent
    return False


def _lowest_common_ancestor(first, second):
    first_ancestors = []
    node = first
    while node is not None:
        first_ancestors.append(node)
        node = node.parent

    node = second
    while node is not None:
        if any(node is ancestor for ancestor in first_ancestors):
            return node
        node = node.parent
    return None


def _child_below(ancestor, descendant):
    node = descendant
    while node.parent is not ancestor:
        node = node.parent
        if node is None:
            return None
    return node


def panel_in_branch(panel_tree, panel_name, branch_name):
    """Return whether a panel belongs to a named top-level branch."""

    branch = _find_child(panel_tree, branch_name)
    if branch is None:
        return False
    return any(
        node.name == panel_name for node in _level_order_iter(branch)
    )


def load_panel_hierarchy(path=None):
    """Load the semantic packing hierarchy from its JSON definition."""

    if path is None:
        hierarchy_path = _DEFAULT_PANEL_TREE_PATH
        if not hierarchy_path.is_file():
            # Preserve support for installations that keep assets beside the
            # invoking project instead of packaging them with ``pygarment``.
            hierarchy_path = Path("assets") / "panel_tree.json"
    else:
        hierarchy_path = Path(path)
    with hierarchy_path.open("r") as hierarchy_file:
        hierarchy_data = json.load(hierarchy_file)

    def build_node(node_data, parent=None):
        node = PanelHierarchyNode(node_data["name"], parent=parent)
        for child_data in node_data.get("children", []):
            build_node(child_data, parent=node)
        return node

    return build_node(hierarchy_data)


def _nest_existing_chain(panel_tree, panel_names, chain):
    if not all(name in panel_names for name in chain):
        return

    nodes = [_find_node(panel_tree, name) for name in chain]
    if any(node is None for node in nodes):
        return

    for parent, child in zip(nodes, nodes[1:]):
        child.parent = parent


def _nest_indexed_tier_chain(panel_tree, panel_names, base_name):
    """Nest contiguous numbered skirt tiers in their physical stitch order."""

    if base_name not in panel_names:
        return

    prefix = f"{base_name}_"
    indexed_names = []
    for panel_name in panel_names:
        if not panel_name.startswith(prefix):
            continue
        suffix = panel_name[len(prefix) :]
        if suffix.isdigit():
            indexed_names.append((int(suffix), panel_name))
    indexed_names.sort(key=lambda item: item[0])
    contiguous_names = []
    for expected_index, (index, panel_name) in enumerate(indexed_names):
        if index != expected_index:
            break
        contiguous_names.append(panel_name)

    if contiguous_names:
        _nest_existing_chain(
            panel_tree,
            panel_names,
            [base_name, *contiguous_names],
        )


def update_panel_hierarchy(
    panel_tree,
    panel_translations: Mapping[str, Sequence[float]],
):
    """Adapt the base hierarchy to panels generated by a concrete pattern.

    Generated multi-panel skirts are assigned to front/back using their 3D
    placement. Sequential skirt tiers and cuff layers are nested in physical
    stitch order so moving a coarser panel preserves its downstream subtree.
    """

    panel_names = set(panel_translations)
    hierarchy_names = {
        node.name for node in _level_order_iter(panel_tree)
    }
    front = _find_child(panel_tree, "front")
    back = _find_child(panel_tree, "back")

    skirt_panels = sorted(
        (
            panel_name
            for panel_name in panel_names
            if "skirt_panel" in panel_name
        ),
        key=lambda name: int(name.rsplit("_", 1)[-1]),
    )
    for panel_name in skirt_panels:
        if panel_name in hierarchy_names:
            continue
        translation = panel_translations[panel_name]
        parent = front if translation[-1] >= 0 else back
        if parent is not None:
            if isinstance(parent, PanelHierarchyNode):
                PanelHierarchyNode(panel_name, parent=parent)
            else:
                # Preserve compatibility with callers supplying an anytree
                # hierarchy or another node type with the same constructor.
                type(parent)(panel_name, parent=parent)
            hierarchy_names.add(panel_name)

    for side in ("front", "back"):
        _nest_indexed_tier_chain(
            panel_tree,
            panel_names,
            f"skirt_{side}",
        )

    nested_cuff_chains = [
        ("pant_f_r", "pant_r_cuff_f", "pant_r_cuff_skirt_f"),
        ("pant_b_r", "pant_r_cuff_b", "pant_r_cuff_skirt_b"),
        ("pant_f_l", "pant_l_cuff_f", "pant_l_cuff_skirt_f"),
        ("pant_b_l", "pant_l_cuff_b", "pant_l_cuff_skirt_b"),
        (
            "left_sleeve_f",
            "sl_left_cuff_f",
            "sl_left_cuff_skirt_f",
        ),
        (
            "left_sleeve_b",
            "sl_left_cuff_b",
            "sl_left_cuff_skirt_b",
        ),
        (
            "right_sleeve_f",
            "sl_right_cuff_f",
            "sl_right_cuff_skirt_f",
        ),
        (
            "right_sleeve_b",
            "sl_right_cuff_b",
            "sl_right_cuff_skirt_b",
        ),
    ]
    for chain in nested_cuff_chains:
        _nest_existing_chain(panel_tree, panel_names, chain)


class PackingPanel:
    """A named 2D panel and the placement used to initialize its layout.

    ``vertices`` contains all panel vertices. ``boundary_indices`` selects the
    exterior loop used for collision checks. Rotation is expressed in radians.
    Inputs are copied so packing never mutates the source mesh or pattern.
    """

    def __init__(
        self,
        name,
        vertices,
        boundary_indices,
        translation=None,
        rotation=0.0,
        rotation_center=None,
    ):
        self.name = str(name)
        self.vertices = np.asarray(vertices, dtype=float).copy()
        self.boundary_indices = np.asarray(boundary_indices, dtype=int).copy()
        self.translation = np.asarray(
            [0.0, 0.0] if translation is None else translation,
            dtype=float,
        ).copy()
        self.rotation = float(rotation)
        self.rotation_center = np.asarray(
            [0.0, 0.0] if rotation_center is None else rotation_center,
            dtype=float,
        ).copy()
        self._validate()

    def _validate(self):
        if not self.name:
            raise ValueError("PackingPanel name cannot be empty")
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError(
                "PackingPanel vertices must have shape (N, 2); "
                f"got {self.vertices.shape} for {self.name!r}"
            )
        if self.boundary_indices.ndim != 1 or len(self.boundary_indices) < 3:
            raise ValueError(
                "PackingPanel boundary_indices must contain at least three "
                f"indices for {self.name!r}"
            )
        if (
            self.boundary_indices.min() < 0
            or self.boundary_indices.max() >= len(self.vertices)
        ):
            raise ValueError(
                f"PackingPanel boundary index is out of range for {self.name!r}"
            )
        if self.translation.shape != (2,):
            raise ValueError(
                "PackingPanel translation must have shape (2,); "
                f"got {self.translation.shape} for {self.name!r}"
            )
        if self.rotation_center.shape != (2,):
            raise ValueError(
                "PackingPanel rotation_center must have shape (2,); "
                f"got {self.rotation_center.shape} for {self.name!r}"
            )

    def placed_vertices(self):
        """Return vertices after applying the panel's initial 2D placement."""

        cos_angle = np.cos(self.rotation)
        sin_angle = np.sin(self.rotation)
        rotation_matrix = np.array(
            [[cos_angle, -sin_angle], [sin_angle, cos_angle]]
        )
        centered = self.vertices - self.rotation_center
        return (
            (rotation_matrix @ centered[..., None])[..., 0]
            + self.rotation_center
            + self.translation
        )


class PackingResult:
    """Result of semantic panel packing."""

    def __init__(
        self,
        panel_vertices,
        boundaries,
        offsets,
        has_overlap,
        iterations,
    ):
        self.panel_vertices = panel_vertices
        self.boundaries = boundaries
        self.offsets = offsets
        # ``has_overlap`` means physical overlaps remain after packing.
        self.has_overlap = bool(has_overlap)
        self.iterations = int(iterations)


class GarmentPackingPolicy:
    """Direction constraints that preserve garment-layout semantics.

    The hierarchy supplies grouping. This policy supplies the few directional
    rules used by the existing GarmentCode layout: front/back sections separate
    vertically, torso halves and pant legs separate horizontally, and sleeves
    move away from torsos horizontally.
    """

    def sibling_axis(self, parent, first_child, second_child):
        parent_name = str(parent.name)
        first_name = str(first_child.name)
        second_name = str(second_child.name)

        if parent_name in {"front", "back"}:
            return "y"
        if parent_name == "top":
            return "x"
        if "pant" in first_name or "pant" in second_name:
            return "x"
        return None

    def parent_child_axis(self, parent, child):
        parent_name = str(parent.name)
        child_name = str(child.name)
        if "torso" in parent_name and "sleeve" in child_name:
            return "x"
        return None

    def packs_children_as_rigid_sections(self, parent):
        """Return whether sibling branches freeze after local packing."""

        return str(parent.name) in {"front", "back"}


def _unit_direction(delta, axis=None):
    direction = np.asarray(delta, dtype=float).copy()
    if axis == "x":
        direction[1] = 0.0
        fallback = np.array([1.0, 0.0])
    elif axis == "y":
        direction[0] = 0.0
        fallback = np.array([0.0, 1.0])
    elif axis is None:
        fallback = np.array([1.0, 0.0])
    else:
        raise ValueError(f"Unknown packing axis constraint: {axis!r}")

    norm = np.linalg.norm(direction)
    if norm == 0:
        return fallback
    return direction / norm


def _centroid(polygon):
    return np.array([polygon.centroid.x, polygon.centroid.y])


def _subtree_union(node, polygons):
    subtree_polygons = [
        polygons[subtree_node.name]
        for subtree_node in _level_order_iter(node)
        if subtree_node.name in polygons
    ]
    if not subtree_polygons:
        return None
    return unary_union(subtree_polygons)


def _move_subtree(node, move_vector, polygons, buffered_polygons):
    for subtree_node in _level_order_iter(node):
        panel_name = subtree_node.name
        if panel_name not in polygons:
            continue
        polygons[panel_name] = translate(
            polygons[panel_name],
            xoff=move_vector[0],
            yoff=move_vector[1],
        )
        buffered_polygons[panel_name] = translate(
            buffered_polygons[panel_name],
            xoff=move_vector[0],
            yoff=move_vector[1],
        )


def _has_any_overlap(polygons):
    polygon_list = list(polygons.values())
    for first_idx, first_polygon in enumerate(polygon_list):
        for second_polygon in polygon_list[first_idx + 1 :]:
            if first_polygon.intersects(second_polygon):
                return True
    return False


def _resolve_individual_overlaps(
    panel_tree,
    polygons,
    buffered_polygons,
    padding,
):
    had_overlap = False

    # Preserve the behavior of the original non-hierarchical UV routine:
    # compare concrete panels at the same semantic depth.
    for level_nodes in _level_order_groups(panel_tree):
        for first_idx, first_node in enumerate(level_nodes):
            for second_node in level_nodes[first_idx + 1 :]:
                if (
                    first_node.name not in buffered_polygons
                    or second_node.name not in buffered_polygons
                ):
                    continue

                first_polygon = buffered_polygons[first_node.name]
                second_polygon = buffered_polygons[second_node.name]
                if not first_polygon.intersects(second_polygon):
                    continue

                had_overlap = True
                direction = _unit_direction(
                    _centroid(second_polygon) - _centroid(first_polygon)
                )
                move_vector = direction * padding / 4
                _move_subtree(
                    first_node, -move_vector, polygons, buffered_polygons
                )
                _move_subtree(
                    second_node, move_vector, polygons, buffered_polygons
                )

    # Keep directly related concrete panels apart.
    for child in _level_order_iter(panel_tree):
        if child.parent is None:
            continue
        parent = child.parent
        if (
            child.name not in buffered_polygons
            or parent.name not in buffered_polygons
        ):
            continue

        child_polygon = buffered_polygons[child.name]
        parent_polygon = buffered_polygons[parent.name]
        if not child_polygon.intersects(parent_polygon):
            continue

        had_overlap = True
        direction = _unit_direction(
            _centroid(child_polygon) - _centroid(parent_polygon)
        )
        _move_subtree(
            child,
            direction * padding,
            polygons,
            buffered_polygons,
        )

    return had_overlap


def _resolve_hierarchical_overlaps(
    panel_tree,
    polygons,
    buffered_polygons,
    padding,
    policy,
):
    had_overlap = False

    # Compare semantic sibling groups as unions, then move each entire group.
    for parent in _level_order_iter(panel_tree):
        children = list(parent.children)
        for first_idx, first_child in enumerate(children):
            for second_child in children[first_idx + 1 :]:
                first_union = _subtree_union(
                    first_child, buffered_polygons
                )
                second_union = _subtree_union(
                    second_child, buffered_polygons
                )
                if (
                    first_union is None
                    or second_union is None
                    or not first_union.intersects(second_union)
                ):
                    continue

                had_overlap = True
                axis = policy.sibling_axis(
                    parent, first_child, second_child
                )
                direction = _unit_direction(
                    _centroid(second_union) - _centroid(first_union),
                    axis=axis,
                )
                move_vector = direction * padding / 4
                _move_subtree(
                    first_child,
                    -move_vector,
                    polygons,
                    buffered_polygons,
                )
                _move_subtree(
                    second_child,
                    move_vector,
                    polygons,
                    buffered_polygons,
                )

    # Keep a concrete child subtree clear of its concrete parent panel.
    for child in _level_order_iter(panel_tree):
        if child.parent is None:
            continue
        parent = child.parent
        if (
            child.name not in buffered_polygons
            or parent.name not in buffered_polygons
        ):
            continue

        child_union = _subtree_union(child, buffered_polygons)
        parent_polygon = buffered_polygons[parent.name]
        if child_union is None or not child_union.intersects(parent_polygon):
            continue

        had_overlap = True
        axis = policy.parent_child_axis(parent, child)
        direction = _unit_direction(
            _centroid(child_union) - _centroid(parent_polygon),
            axis=axis,
        )
        _move_subtree(
            child,
            direction * padding,
            polygons,
            buffered_polygons,
        )

    return had_overlap


def _fine_to_coarse_axis(first, second, policy):
    """Choose a semantic direction constraint for two concrete panels."""

    if _is_ancestor(first, second):
        branch = _child_below(first, second)
        return policy.parent_child_axis(first, branch)
    if _is_ancestor(second, first):
        branch = _child_below(second, first)
        return policy.parent_child_axis(second, branch)

    common_ancestor = _lowest_common_ancestor(first, second)
    if common_ancestor is None:
        return None
    first_branch = _child_below(common_ancestor, first)
    second_branch = _child_below(common_ancestor, second)
    if first_branch is None or second_branch is None:
        return None
    return policy.sibling_axis(
        common_ancestor,
        first_branch,
        second_branch,
    )


def _packs_children_as_rigid_sections(policy, parent):
    predicate = getattr(
        policy,
        "packs_children_as_rigid_sections",
        None,
    )
    return bool(predicate(parent)) if predicate is not None else False


def _rigid_section_branches(first, second, policy):
    """Return rigid LCA branches for a cross-section concrete pair."""

    common_ancestor = _lowest_common_ancestor(first, second)
    if (
        common_ancestor is None
        or not _packs_children_as_rigid_sections(
            policy,
            common_ancestor,
        )
    ):
        return None

    first_branch = _child_below(common_ancestor, first)
    second_branch = _child_below(common_ancestor, second)
    if (
        first_branch is None
        or second_branch is None
        or first_branch is second_branch
    ):
        return None
    return common_ancestor, first_branch, second_branch


def _resolve_rigid_section_overlaps(
    panel_tree,
    polygons,
    buffered_polygons,
    padding,
    policy,
):
    """Separate locally packed sibling sections as rigid subtree unions."""

    had_overlap = False
    for parent in _level_order_iter(panel_tree):
        if not _packs_children_as_rigid_sections(policy, parent):
            continue

        children = list(parent.children)
        for first_idx, first_child in enumerate(children):
            for second_child in children[first_idx + 1 :]:
                first_union = _subtree_union(
                    first_child,
                    buffered_polygons,
                )
                second_union = _subtree_union(
                    second_child,
                    buffered_polygons,
                )
                if (
                    first_union is None
                    or second_union is None
                    or not first_union.intersects(second_union)
                ):
                    continue

                had_overlap = True
                axis = policy.sibling_axis(
                    parent,
                    first_child,
                    second_child,
                )
                direction = _unit_direction(
                    _centroid(second_union) - _centroid(first_union),
                    axis=axis,
                )
                move_vector = direction * padding / 4
                _move_subtree(
                    first_child,
                    -move_vector,
                    polygons,
                    buffered_polygons,
                )
                _move_subtree(
                    second_child,
                    move_vector,
                    polygons,
                    buffered_polygons,
                )

    return had_overlap


def _resolve_fine_to_coarse_overlaps(
    panel_tree,
    polygons,
    buffered_polygons,
    padding,
    policy,
):
    """Resolve concrete collisions from deepest panels toward their parents.

    Unlike hierarchical union packing, a collision between descendants of two
    branches first moves the concrete descendants. An ancestor subtree moves
    only when its own concrete panel still collides after the deeper panels
    have had a chance to separate.
    """

    concrete_nodes = [
        node
        for node in _level_order_iter(panel_tree)
        if node.name in buffered_polygons
    ]
    depths = {
        id(node): _node_depth(node) for node in concrete_nodes
    }
    node_pairs = [
        (first, second)
        for first_idx, first in enumerate(concrete_nodes)
        for second in concrete_nodes[first_idx + 1 :]
    ]
    node_pairs.sort(
        key=lambda pair: (
            depths[id(pair[0])] + depths[id(pair[1])],
            min(depths[id(pair[0])], depths[id(pair[1])]),
        ),
        reverse=True,
    )

    had_overlap = False
    for first, second in node_pairs:
        # Locally pack each major section first. Concrete pairs that cross a
        # rigid section boundary are handled once as subtree unions below,
        # preventing one descendant pair from tearing apart a solved section.
        if _rigid_section_branches(first, second, policy) is not None:
            continue

        first_polygon = buffered_polygons[first.name]
        second_polygon = buffered_polygons[second.name]
        if not first_polygon.intersects(second_polygon):
            continue

        had_overlap = True
        axis = _fine_to_coarse_axis(first, second, policy)
        direction = _unit_direction(
            _centroid(second_polygon) - _centroid(first_polygon),
            axis=axis,
        )
        move_vector = direction * padding / 4
        first_depth = depths[id(first)]
        second_depth = depths[id(second)]

        if _is_ancestor(first, second) or second_depth > first_depth:
            _move_subtree(
                second,
                move_vector,
                polygons,
                buffered_polygons,
            )
        elif _is_ancestor(second, first) or first_depth > second_depth:
            _move_subtree(
                first,
                -move_vector,
                polygons,
                buffered_polygons,
            )
        else:
            _move_subtree(
                first,
                -move_vector,
                polygons,
                buffered_polygons,
            )
            _move_subtree(
                second,
                move_vector,
                polygons,
                buffered_polygons,
            )

    sections_had_overlap = _resolve_rigid_section_overlaps(
        panel_tree,
        polygons,
        buffered_polygons,
        padding,
        policy,
    )
    return had_overlap or sections_had_overlap


def _concrete_section_names(section_root, polygons):
    """Return concrete panels in stable hierarchy order for one section."""

    return [
        node.name
        for node in _level_order_iter(section_root)
        if node.name in polygons
    ]


def _has_nested_concrete_chain(section_root, panel_names, min_length=3):
    """Return whether a section contains a sufficiently long concrete chain."""

    panel_names = set(panel_names)
    for node in _level_order_iter(section_root):
        if node.name not in panel_names:
            continue

        chain_length = 1
        parent = node.parent
        while parent is not None:
            if parent.name in panel_names:
                chain_length += 1
            if parent is section_root:
                break
            parent = parent.parent
        if chain_length >= min_length:
            return True
    return False


def _joint_optimization_sections(panel_tree, polygons, policy):
    """Find independent major sections containing nested tier chains."""

    sections = []
    has_rigid_section_parent = False
    for parent in _level_order_iter(panel_tree):
        if not _packs_children_as_rigid_sections(policy, parent):
            continue
        has_rigid_section_parent = True
        for child in parent.children:
            # Production garment hierarchies also contain nested sleeve/cuff
            # chains. They are intentionally left to fine-to-coarse packing:
            # the free-2D stage is for bottom/skirt tiers only.
            if str(child.name) != "bottom":
                continue
            panel_names = _concrete_section_names(child, polygons)
            if (
                len(panel_names) >= 3
                and _has_nested_concrete_chain(child, panel_names)
            ):
                sections.append((child, panel_names))

    # Custom hierarchies do not necessarily define front/back rigid sections.
    # In that case, optimize a nested concrete chain at the supplied root.
    if not sections and not has_rigid_section_parent:
        panel_names = _concrete_section_names(panel_tree, polygons)
        if (
            len(panel_names) >= 3
            and _has_nested_concrete_chain(panel_tree, panel_names)
        ):
            sections.append((panel_tree, panel_names))
    return sections


def _nearest_concrete_parent(node, section_root, panel_names):
    """Find the closest concrete ancestor without leaving a section."""

    parent = node.parent
    while parent is not None:
        if parent.name in panel_names:
            return parent
        if parent is section_root:
            break
        parent = parent.parent
    return None


def _joint_semantic_pairs(section_root, panel_names):
    """Return concrete hierarchy neighbors that should remain adjacent."""

    panel_names = set(panel_names)
    pairs = set()
    for node in _level_order_iter(section_root):
        if node.name not in panel_names:
            continue
        parent = _nearest_concrete_parent(
            node,
            section_root,
            panel_names,
        )
        if parent is not None:
            pairs.add(tuple(sorted((parent.name, node.name))))

    # Preserve adjacency among concrete siblings in non-chain custom trees.
    for parent in _level_order_iter(section_root):
        concrete_children = [
            child.name
            for child in parent.children
            if child.name in panel_names
        ]
        for first_name, second_name in combinations(
            concrete_children,
            2,
        ):
            pairs.add(tuple(sorted((first_name, second_name))))
    return pairs


def _translated_geometries(geometries, offsets):
    return [
        translate(geometry, xoff=offset[0], yoff=offset[1])
        for geometry, offset in zip(geometries, offsets)
    ]


def _pair_clearances(geometries):
    return np.asarray(
        [
            geometries[first_idx].distance(geometries[second_idx])
            for first_idx, second_idx in combinations(
                range(len(geometries)),
                2,
            )
        ],
        dtype=float,
    )


def _layout_is_clear(geometries, clearance, tolerance=1e-7):
    clearances = _pair_clearances(geometries)
    return bool(
        len(clearances) == 0
        or np.min(clearances) >= clearance - tolerance
    )


def _expanded_feasible_start(
    proxy_polygons,
    source_centers,
    local_offsets,
    clearance,
):
    """Radially expand a proposed layout into a feasible solver start."""

    center_positions = source_centers + local_offsets
    anchor_center = center_positions[0].copy()
    relative_centers = center_positions - anchor_center

    section_bounds = np.asarray(
        [geometry.bounds for geometry in proxy_polygons],
        dtype=float,
    )
    section_scale = max(
        float(np.ptp(section_bounds[:, [0, 2]])),
        float(np.ptp(section_bounds[:, [1, 3]])),
        clearance,
        1.0,
    )
    for index in range(1, len(relative_centers)):
        if np.linalg.norm(relative_centers[index]) <= 1e-8:
            angle = index * np.pi * (3.0 - np.sqrt(5.0))
            relative_centers[index] = (
                section_scale
                * np.array([np.cos(angle), np.sin(angle)])
            )

    # Rotating only panel centers can create collisions even when the source
    # arrangement was feasible. Uniform expansion supplies SLSQP with a
    # collision-free start while leaving it free to compact in any direction.
    scale = 1.0
    expanded_offsets = local_offsets.copy()
    for _ in range(96):
        expanded_centers = anchor_center + scale * relative_centers
        expanded_offsets = expanded_centers - source_centers
        expanded_offsets -= expanded_offsets[0]
        translated = _translated_geometries(
            proxy_polygons,
            expanded_offsets,
        )
        if _layout_is_clear(translated, clearance):
            return expanded_offsets
        scale *= 1.075
    return expanded_offsets


def _joint_layout_starts(
    proxy_polygons,
    source_centers,
    baseline_local_offsets,
    clearance,
):
    """Build feasible multistarts without imposing an optimization axis."""

    baseline_centers = source_centers + baseline_local_offsets
    baseline_relative = baseline_centers - baseline_centers[0]
    starts = []
    for angle in (0.0, np.pi / 2, -np.pi / 2, np.pi):
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation = np.array(
            [[cos_angle, -sin_angle], [sin_angle, cos_angle]]
        )
        desired_centers = (
            source_centers[0] + baseline_relative @ rotation.T
        )
        local_offsets = desired_centers - source_centers
        local_offsets -= local_offsets[0]
        starts.append(
            _expanded_feasible_start(
                proxy_polygons,
                source_centers,
                local_offsets,
                clearance,
            )
        )

    # A square-turn chain gives elongated tier stacks a compact alternative
    # that is topologically very different from rotated versions of the
    # fine-to-coarse initializer.
    radii = np.asarray(
        [
            0.5 * np.hypot(
                geometry.bounds[2] - geometry.bounds[0],
                geometry.bounds[3] - geometry.bounds[1],
            )
            for geometry in proxy_polygons
        ]
    )
    for turn_sign in (-1.0, 1.0):
        desired_centers = np.empty_like(source_centers)
        desired_centers[0] = source_centers[0]
        for index in range(1, len(source_centers)):
            angle = turn_sign * (index - 1) * np.pi / 2
            direction = np.array([np.cos(angle), np.sin(angle)])
            step = radii[index - 1] + radii[index] + clearance
            desired_centers[index] = (
                desired_centers[index - 1] + direction * step
            )
        local_offsets = desired_centers - source_centers
        local_offsets -= local_offsets[0]
        starts.append(
            _expanded_feasible_start(
                proxy_polygons,
                source_centers,
                local_offsets,
                clearance,
            )
        )

    unique_starts = []
    seen = set()
    for start in starts:
        key = tuple(np.round(start[1:].ravel(), decimals=5))
        if key not in seen:
            seen.add(key)
            unique_starts.append(start)
    return unique_starts


def _optimize_nested_section(
    section_root,
    panel_names,
    source_polygons,
    polygons,
    buffered_polygons,
    padding,
    max_iterations,
):
    """Jointly translate all panels in one nested section.

    The proxy problem is deliberately conservative: simplifying a polygon can
    move its boundary by ``simplification_tolerance``, so proxy polygons are
    kept ``padding + 2 * simplification_tolerance`` apart. The accepted result
    is always checked again with the original full-resolution polygons.
    """

    try:
        from scipy.optimize import minimize
    except ImportError as error:
        raise ImportError(
            "The joint_optimization packing strategy requires scipy"
        ) from error

    source = [source_polygons[name] for name in panel_names]
    current = [polygons[name] for name in panel_names]
    source_centers = np.asarray(
        [_centroid(polygon) for polygon in source]
    )
    baseline_offsets = np.asarray(
        [
            _centroid(current_polygon) - source_center
            for current_polygon, source_center in zip(
                current,
                source_centers,
            )
        ]
    )
    common_offset = baseline_offsets[0].copy()
    baseline_local_offsets = baseline_offsets - common_offset

    simplification_tolerance = min(
        0.5,
        max(0.05, padding / 6.0),
    )
    proxy_polygons = []
    for polygon in source:
        proxy = polygon.simplify(
            simplification_tolerance,
            preserve_topology=True,
        )
        if proxy.is_empty:
            proxy = polygon
        proxy_polygons.append(proxy)
    proxy_clearance = padding + 2.0 * simplification_tolerance

    pair_indices = list(combinations(range(len(panel_names)), 2))
    name_to_index = {
        name: index for index, name in enumerate(panel_names)
    }
    semantic_pairs = [
        (name_to_index[first_name], name_to_index[second_name])
        for first_name, second_name in sorted(
            _joint_semantic_pairs(
                section_root,
                panel_names,
            )
        )
    ]

    all_bounds = np.asarray(
        [polygon.bounds for polygon in source],
        dtype=float,
    )
    section_scale = max(
        float(np.max(all_bounds[:, 2]) - np.min(all_bounds[:, 0])),
        float(np.max(all_bounds[:, 3]) - np.min(all_bounds[:, 1])),
        padding,
        1.0,
    )
    # Hierarchy edges represent sewn or immediately neighboring tiers. Give
    # their target clearance enough weight that footprint compaction cannot
    # create a visually uneven chain merely to shave a few centimeters from
    # the bounding box.
    semantic_weight = 2.0 / max(padding, 1.0)
    displacement_weight = 0.04 / section_scale

    def unpack(variable_offsets):
        local_offsets = np.zeros((len(panel_names), 2), dtype=float)
        local_offsets[1:] = np.reshape(variable_offsets, (-1, 2))
        return local_offsets

    def translated_proxy(variable_offsets):
        return _translated_geometries(
            proxy_polygons,
            unpack(variable_offsets),
        )

    def objective(variable_offsets):
        local_offsets = unpack(variable_offsets)
        translated_bounds = np.asarray(
            [
                (
                    polygon.bounds[0] + offset[0],
                    polygon.bounds[1] + offset[1],
                    polygon.bounds[2] + offset[0],
                    polygon.bounds[3] + offset[1],
                )
                for polygon, offset in zip(
                    proxy_polygons,
                    local_offsets,
                )
            ]
        )
        width = (
            np.max(translated_bounds[:, 2])
            - np.min(translated_bounds[:, 0])
        )
        height = (
            np.max(translated_bounds[:, 3])
            - np.min(translated_bounds[:, 1])
        )
        compactness = np.sqrt(max(width * height, 0.0))
        compactness += 0.25 * (width + height)

        translated = _translated_geometries(
            proxy_polygons,
            local_offsets,
        )
        semantic_penalty = sum(
            (
                translated[first_idx].distance(translated[second_idx])
                - proxy_clearance
            )
            ** 2
            for first_idx, second_idx in semantic_pairs
        )
        displacement_penalty = np.sum(local_offsets[1:] ** 2)
        return (
            compactness
            + semantic_weight * semantic_penalty
            + displacement_weight * displacement_penalty
        )

    def clearance_constraints(variable_offsets):
        translated = translated_proxy(variable_offsets)
        return np.asarray(
            [
                translated[first_idx].distance(translated[second_idx])
                - proxy_clearance
                for first_idx, second_idx in pair_indices
            ],
            dtype=float,
        )

    starts = _joint_layout_starts(
        proxy_polygons,
        source_centers,
        baseline_local_offsets,
        proxy_clearance,
    )
    max_start_extent = max(
        float(np.max(np.abs(start[1:])))
        for start in starts
    )
    offset_limit = max(
        6.0 * section_scale,
        1.5 * max_start_extent,
        20.0 * padding,
    )
    bounds = [
        (-offset_limit, offset_limit)
        for _ in range(2 * (len(panel_names) - 1))
    ]
    constraints = {
        "type": "ineq",
        "fun": clearance_constraints,
    }

    baseline_value = objective(baseline_local_offsets[1:].ravel())
    best_value = baseline_value
    best_local_offsets = None
    solver_iterations = 0
    solver_max_iterations = min(max_iterations, 180)
    for start in starts:
        optimization = minimize(
            objective,
            start[1:].ravel(),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": solver_max_iterations,
                "ftol": 1e-6,
                "disp": False,
            },
        )
        solver_iterations += int(getattr(optimization, "nit", 0))
        if not np.all(np.isfinite(optimization.x)):
            continue
        constraint_values = clearance_constraints(optimization.x)
        if (
            len(constraint_values)
            and np.min(constraint_values) < -1e-4
        ):
            continue
        candidate_value = objective(optimization.x)
        if candidate_value < best_value - 1e-5:
            best_value = candidate_value
            best_local_offsets = unpack(optimization.x)

    if best_local_offsets is None:
        return False, solver_iterations

    candidate_offsets = common_offset + best_local_offsets
    exact_candidate = _translated_geometries(
        source,
        candidate_offsets,
    )
    if not _layout_is_clear(
        exact_candidate,
        padding,
        tolerance=1e-6,
    ):
        return False, solver_iterations

    for name, candidate_polygon in zip(panel_names, exact_candidate):
        polygons[name] = candidate_polygon
        buffered_polygons[name] = candidate_polygon.buffer(padding / 2)
    return True, solver_iterations


def _resolve_joint_optimization(
    panel_tree,
    source_polygons,
    polygons,
    buffered_polygons,
    padding,
    policy,
    max_iterations,
):
    """Initialize hierarchically, then jointly compact nested tier sections."""

    packing_passes = 0
    for packing_passes in range(1, max_iterations + 1):
        had_overlap = _resolve_fine_to_coarse_overlaps(
            panel_tree,
            polygons,
            buffered_polygons,
            padding,
            policy,
        )
        if not had_overlap:
            break

    solver_iterations = 0
    for section_root, panel_names in _joint_optimization_sections(
        panel_tree,
        polygons,
        policy,
    ):
        _, section_iterations = _optimize_nested_section(
            section_root,
            panel_names,
            source_polygons,
            polygons,
            buffered_polygons,
            padding,
            max_iterations,
        )
        solver_iterations += section_iterations

    # A locally compacted bottom may now touch a top or waistband. Keep each
    # solved section rigid while restoring the high-level semantic clearance.
    rigid_passes = 0
    for rigid_passes in range(1, max_iterations + 1):
        had_overlap = _resolve_rigid_section_overlaps(
            panel_tree,
            polygons,
            buffered_polygons,
            padding,
            policy,
        )
        if not had_overlap:
            break

    # ``PackingResult.iterations`` historically counts geometric packing
    # passes. Solver iterations are intentionally not mixed into that value.
    return packing_passes + rigid_passes


def pack_pattern_panels(
    panels: Iterable[PackingPanel],
    panel_tree,
    padding=1.0,
    strategy=HIERARCHICAL,
    max_iterations=500,
    policy=None,
):
    """Pack 2D pattern panels without introducing UV-specific concerns.

    Args:
        panels: Named panel geometry and initial 2D placements.
        panel_tree: Tree-like hierarchy describing garment semantics.
        padding: Required separation between panel boundaries.
        strategy: ``"hierarchical"`` moves semantic groups as unions;
            ``"fine_to_coarse"`` resolves the deepest concrete panels before
            moving parent subtrees; ``"joint_optimization"`` uses that result
            as a feasible initializer and jointly compacts nested tier panels
            with unconstrained 2D translations; ``"individual"`` reproduces
            the older per-panel behavior.
        max_iterations: Maximum overlap-resolution or solver iterations per
            stage.
        policy: Optional direction policy. Defaults to
            :class:`GarmentPackingPolicy`.

    Returns:
        :class:`PackingResult` containing packed vertices, packed boundaries,
        per-panel translation offsets, and convergence information.
    """

    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Unknown packing strategy {strategy!r}; "
            f"expected one of {sorted(_VALID_STRATEGIES)}"
        )
    if padding <= 0:
        raise ValueError("Packing padding must be greater than zero")
    if max_iterations <= 0:
        raise ValueError("Packing max_iterations must be greater than zero")

    panel_list = list(panels)
    panel_by_name: Dict[str, PackingPanel] = {}
    for panel in panel_list:
        if panel.name in panel_by_name:
            raise ValueError(f"Duplicate packing panel name: {panel.name!r}")
        panel_by_name[panel.name] = panel

    if not panel_list:
        return PackingResult({}, {}, {}, False, 0)

    placed_vertices = {
        panel.name: panel.placed_vertices() for panel in panel_list
    }
    polygons = {
        panel.name: Polygon(
            placed_vertices[panel.name][panel.boundary_indices]
        )
        for panel in panel_list
    }
    source_polygons = dict(polygons)
    original_centroids = {
        name: _centroid(polygon) for name, polygon in polygons.items()
    }
    buffered_polygons = {
        name: polygon.buffer(padding / 2)
        for name, polygon in polygons.items()
    }

    if policy is None:
        policy = GarmentPackingPolicy()

    if strategy == JOINT_OPTIMIZATION:
        iterations = _resolve_joint_optimization(
            panel_tree,
            source_polygons,
            polygons,
            buffered_polygons,
            padding,
            policy,
            max_iterations,
        )
    else:
        has_overlap = False
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            if strategy == HIERARCHICAL:
                has_overlap = _resolve_hierarchical_overlaps(
                    panel_tree,
                    polygons,
                    buffered_polygons,
                    padding,
                    policy,
                )
            elif strategy == FINE_TO_COARSE:
                has_overlap = _resolve_fine_to_coarse_overlaps(
                    panel_tree,
                    polygons,
                    buffered_polygons,
                    padding,
                    policy,
                )
            else:
                has_overlap = _resolve_individual_overlaps(
                    panel_tree,
                    polygons,
                    buffered_polygons,
                    padding,
                )
            if not has_overlap:
                break

    # Report physical overlaps even when panels are absent from, or unrelated
    # within, the semantic hierarchy.
    has_overlap = _has_any_overlap(buffered_polygons)

    offsets = {
        name: _centroid(polygon) - original_centroids[name]
        for name, polygon in polygons.items()
    }
    packed_vertices = {
        name: placed_vertices[name] + offsets[name]
        for name in panel_by_name
    }
    packed_boundaries = {
        name: packed_vertices[name][panel_by_name[name].boundary_indices]
        for name in panel_by_name
    }

    return PackingResult(
        packed_vertices,
        packed_boundaries,
        offsets,
        has_overlap,
        iterations,
    )
