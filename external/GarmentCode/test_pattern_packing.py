import importlib
import sys
import types
import unittest
from unittest import mock

import numpy as np
from shapely.geometry import Polygon

from pygarment.meshgen.pattern_packing import (
    FINE_TO_COARSE,
    HIERARCHICAL,
    INDIVIDUAL,
    JOINT_OPTIMIZATION,
    PackingPanel,
    load_panel_hierarchy,
    pack_pattern_panels,
    panel_in_branch,
    update_panel_hierarchy,
)


class TreeNode:
    """Minimal tree implementation exercising the packer's public protocol."""

    def __init__(self, name, parent=None):
        self.name = name
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


def square_panel(name, x=0.0, y=0.0):
    vertices = np.array(
        [
            [x, y],
            [x + 1, y],
            [x + 1, y + 1],
            [x, y + 1],
        ],
        dtype=float,
    )
    return PackingPanel(name, vertices, [0, 1, 2, 3])


class PatternPackingTest(unittest.TestCase):
    def test_default_hierarchy_loads_independently_of_working_directory(self):
        class DictImporterStub:
            def import_(self, data):
                def build_node(node_data, parent=None):
                    node = TreeNode(node_data["name"], parent)
                    for child_data in node_data.get("children", []):
                        build_node(child_data, node)
                    return node

                return build_node(data)

        anytree_stub = types.ModuleType("anytree")
        importer_stub = types.ModuleType("anytree.importer")
        importer_stub.DictImporter = DictImporterStub
        with mock.patch.dict(
            sys.modules,
            {
                "anytree": anytree_stub,
                "anytree.importer": importer_stub,
            },
        ):
            hierarchy = load_panel_hierarchy()

        self.assertTrue(
            panel_in_branch(hierarchy, "left_ftorso", "front")
        )
        self.assertTrue(
            panel_in_branch(hierarchy, "right_btorso", "back")
        )

    def test_panel_branch_lookup_is_owned_by_packing_hierarchy(self):
        root = TreeNode("root")
        front = TreeNode("front", root)
        back = TreeNode("back", root)
        TreeNode("front_panel", front)
        TreeNode("back_panel", back)

        self.assertTrue(
            panel_in_branch(root, "front_panel", "front")
        )
        self.assertFalse(
            panel_in_branch(root, "front_panel", "back")
        )

    def test_hierarchy_update_assigns_skirts_and_nests_panel_chains(self):
        root = TreeNode("root")
        front = TreeNode("front", root)
        back = TreeNode("back", root)
        pant = TreeNode("pant_f_r", front)
        cuff = TreeNode("pant_r_cuff_f", pant)
        cuff_skirt = TreeNode("pant_r_cuff_skirt_f", pant)
        front_skirt = TreeNode("skirt_front", front)
        front_tier_0 = TreeNode("skirt_front_0", front_skirt)
        front_tier_1 = TreeNode("skirt_front_1", front_skirt)
        back_skirt = TreeNode("skirt_back", back)
        back_tier_0 = TreeNode("skirt_back_0", back_skirt)
        back_tier_1 = TreeNode("skirt_back_1", back_skirt)

        anytree_stub = types.ModuleType("anytree")
        anytree_stub.Node = TreeNode
        translations = {
            "pant_f_r": [0, 0, 1],
            "pant_r_cuff_f": [0, 0, 1],
            "pant_r_cuff_skirt_f": [0, 0, 1],
            "skirt_front": [0, 0, 1],
            "skirt_front_1": [0, 0, 1],
            "skirt_front_0": [0, 0, 1],
            "skirt_back": [0, 0, -1],
            "skirt_back_1": [0, 0, -1],
            "skirt_back_0": [0, 0, -1],
            "skirt_panel_0": [0, 0, 1],
            "skirt_panel_1": [0, 0, -1],
        }
        with mock.patch.dict(sys.modules, {"anytree": anytree_stub}):
            update_panel_hierarchy(root, translations)

        self.assertIs(cuff.parent, pant)
        self.assertIs(cuff_skirt.parent, cuff)
        self.assertIs(front_tier_0.parent, front_skirt)
        self.assertIs(front_tier_1.parent, front_tier_0)
        self.assertIs(back_tier_0.parent, back_skirt)
        self.assertIs(back_tier_1.parent, back_tier_0)
        self.assertTrue(
            panel_in_branch(root, "skirt_panel_0", "front")
        )
        self.assertTrue(
            panel_in_branch(root, "skirt_panel_1", "back")
        )

    def test_initial_placement_is_independent_of_uv_data(self):
        panel = PackingPanel(
            "panel",
            [[0, 0], [1, 0], [1, 1], [0, 1]],
            [0, 1, 2, 3],
            translation=[2, 3],
            rotation=np.pi / 2,
            rotation_center=[0, 0],
        )

        np.testing.assert_allclose(
            panel.placed_vertices(),
            [[2, 3], [2, 4], [1, 4], [1, 3]],
            atol=1e-8,
        )

    def test_uv_adapter_only_extracts_boundaries_and_preserves_inputs(self):
        igl_stub = types.ModuleType("igl")
        igl_stub.boundary_loop = lambda faces: np.array([0, 1, 2, 3])
        module_name = "pygarment.meshgen.render.texture_utils"

        with mock.patch.dict(sys.modules, {"igl": igl_stub}):
            sys.modules.pop(module_name, None)
            texture_utils = importlib.import_module(module_name)

            root = TreeNode("root")
            TreeNode("first", root)
            TreeNode("second", root)
            faces = np.array([[0, 1, 2], [0, 2, 3]])
            islands = [
                {
                    "panel_name": name,
                    "uv": square_panel(name).vertices,
                    "face_texture_coords": faces,
                    "translation": [0, 0],
                    "rotation": 0,
                    "rotation_center": [0, 0],
                }
                for name in ("first", "second")
            ]
            original_uvs = [island["uv"].copy() for island in islands]

            packed = texture_utils.pack_uv_islands(
                islands,
                root,
                padding=0.5,
                strategy=INDIVIDUAL,
            )

        self.assertEqual(len(packed), 4)
        for island, original_uv in zip(islands, original_uvs):
            np.testing.assert_array_equal(island["uv"], original_uv)

    def test_individual_packing_separates_panels_without_mutating_input(self):
        root = TreeNode("root")
        TreeNode("first", root)
        TreeNode("second", root)
        first = square_panel("first")
        second = square_panel("second")
        original_first = first.vertices.copy()

        result = pack_pattern_panels(
            [first, second],
            root,
            padding=0.5,
            strategy=INDIVIDUAL,
        )

        self.assertFalse(result.has_overlap)
        np.testing.assert_array_equal(first.vertices, original_first)
        first_polygon = Polygon(result.boundaries["first"])
        second_polygon = Polygon(result.boundaries["second"])
        self.assertGreaterEqual(
            first_polygon.distance(second_polygon),
            0.5 - 1e-8,
        )

    def test_hierarchical_packing_moves_a_semantic_subtree_together(self):
        root = TreeNode("root")
        first_group = TreeNode("first_group", root)
        second_group = TreeNode("second_group", root)
        TreeNode("first", first_group)
        TreeNode("first_detail", first_group)
        TreeNode("second", second_group)

        result = pack_pattern_panels(
            [
                square_panel("first"),
                square_panel("first_detail", x=3),
                square_panel("second", x=0.2),
            ],
            root,
            padding=0.5,
            strategy=HIERARCHICAL,
        )

        self.assertFalse(result.has_overlap)
        np.testing.assert_allclose(
            result.offsets["first"],
            result.offsets["first_detail"],
        )

    def test_garment_policy_keeps_front_sections_vertically_ordered(self):
        root = TreeNode("root")
        front = TreeNode("front", root)
        top = TreeNode("top", front)
        bottom = TreeNode("bottom", front)
        TreeNode("torso", top)
        TreeNode("skirt", bottom)

        result = pack_pattern_panels(
            [square_panel("torso"), square_panel("skirt")],
            root,
            padding=0.5,
            strategy=HIERARCHICAL,
        )

        self.assertFalse(result.has_overlap)
        self.assertAlmostEqual(result.offsets["torso"][0], 0.0)
        self.assertAlmostEqual(result.offsets["skirt"][0], 0.0)
        self.assertLess(result.offsets["torso"][1], 0.0)
        self.assertGreater(result.offsets["skirt"][1], 0.0)

    def test_fine_to_coarse_resolves_hoods_before_moving_torsos(self):
        root = TreeNode("root")
        back = TreeNode("back", root)
        top = TreeNode("top", back)
        right_torso = TreeNode("right_btorso", top)
        left_torso = TreeNode("left_btorso", top)
        TreeNode("right_hood", right_torso)
        TreeNode("left_hood", left_torso)
        sleeve = TreeNode("right_sleeve", right_torso)
        cuff = TreeNode("right_cuff", sleeve)
        TreeNode("right_cuff_skirt", cuff)
        panels = [
            square_panel("right_btorso", x=-1.0),
            square_panel("left_btorso", x=0.0),
            square_panel("right_hood", x=-0.5, y=2.0),
            square_panel("left_hood", x=-0.5, y=2.0),
            square_panel("right_sleeve", x=-2.0),
            square_panel("right_cuff", x=-2.0, y=1.5),
            square_panel("right_cuff_skirt", x=-2.0, y=3.0),
        ]

        hierarchical = pack_pattern_panels(
            panels,
            root,
            padding=0.5,
            strategy=HIERARCHICAL,
        )
        fine_to_coarse = pack_pattern_panels(
            panels,
            root,
            padding=0.5,
            strategy=FINE_TO_COARSE,
        )

        self.assertFalse(fine_to_coarse.has_overlap)
        hierarchical_torso_spread = abs(
            hierarchical.offsets["left_btorso"][0]
            - hierarchical.offsets["right_btorso"][0]
        )
        fine_to_coarse_torso_spread = abs(
            fine_to_coarse.offsets["left_btorso"][0]
            - fine_to_coarse.offsets["right_btorso"][0]
        )
        self.assertLess(
            fine_to_coarse_torso_spread,
            hierarchical_torso_spread,
        )
        self.assertFalse(
            np.allclose(
                fine_to_coarse.offsets["right_hood"],
                fine_to_coarse.offsets["right_btorso"],
            )
        )

    def test_fine_to_coarse_preserves_locally_packed_rigid_section(self):
        root = TreeNode("root")
        back = TreeNode("back", root)
        top = TreeNode("top", back)
        bottom = TreeNode("bottom", back)
        TreeNode("sleeve", top)
        skirt = TreeNode("skirt", bottom)
        TreeNode("skirt_0", skirt)

        result = pack_pattern_panels(
            [
                square_panel("sleeve", y=3.2),
                square_panel("skirt"),
                square_panel("skirt_0", y=3.0),
            ],
            root,
            padding=0.5,
            strategy=FINE_TO_COARSE,
        )

        self.assertFalse(result.has_overlap)
        np.testing.assert_allclose(
            result.offsets["skirt"],
            result.offsets["skirt_0"],
        )

    def test_joint_optimization_compacts_nested_chain_in_two_dimensions(self):
        root = TreeNode("root")
        back = TreeNode("back", root)
        bottom = TreeNode("bottom", back)
        tier = TreeNode("tier", bottom)
        tier_0 = TreeNode("tier_0", tier)
        tier_1 = TreeNode("tier_1", tier_0)
        TreeNode("tier_2", tier_1)
        panel_names = ["tier", "tier_0", "tier_1", "tier_2"]
        panels = [square_panel(name) for name in panel_names]

        fine_to_coarse = pack_pattern_panels(
            panels,
            root,
            padding=0.5,
            strategy=FINE_TO_COARSE,
        )
        optimized = pack_pattern_panels(
            panels,
            root,
            padding=0.5,
            strategy=JOINT_OPTIMIZATION,
        )

        self.assertFalse(optimized.has_overlap)
        fine_bounds = np.asarray(
            [
                Polygon(fine_to_coarse.boundaries[name]).bounds
                for name in panel_names
            ]
        )
        optimized_polygons = [
            Polygon(optimized.boundaries[name])
            for name in panel_names
        ]
        optimized_bounds = np.asarray(
            [polygon.bounds for polygon in optimized_polygons]
        )
        fine_max_extent = max(
            np.max(fine_bounds[:, 2]) - np.min(fine_bounds[:, 0]),
            np.max(fine_bounds[:, 3]) - np.min(fine_bounds[:, 1]),
        )
        optimized_max_extent = max(
            np.max(optimized_bounds[:, 2])
            - np.min(optimized_bounds[:, 0]),
            np.max(optimized_bounds[:, 3])
            - np.min(optimized_bounds[:, 1]),
        )
        self.assertLess(optimized_max_extent, fine_max_extent)
        for first_idx, first_polygon in enumerate(optimized_polygons):
            for second_polygon in optimized_polygons[first_idx + 1 :]:
                self.assertGreaterEqual(
                    first_polygon.distance(second_polygon),
                    0.5 - 1e-6,
                )

    def test_joint_optimization_leaves_non_tier_hierarchy_unchanged(self):
        root = TreeNode("root")
        back = TreeNode("back", root)
        top = TreeNode("top", back)
        right_torso = TreeNode("right_btorso", top)
        left_torso = TreeNode("left_btorso", top)
        TreeNode("right_hood", right_torso)
        TreeNode("left_hood", left_torso)
        panels = [
            square_panel("right_btorso", x=-1.0),
            square_panel("left_btorso", x=0.0),
            square_panel("right_hood", x=-0.5, y=2.0),
            square_panel("left_hood", x=-0.5, y=2.0),
        ]

        fine_to_coarse = pack_pattern_panels(
            panels,
            root,
            padding=0.5,
            strategy=FINE_TO_COARSE,
        )
        optimized = pack_pattern_panels(
            panels,
            root,
            padding=0.5,
            strategy=JOINT_OPTIMIZATION,
        )

        self.assertFalse(optimized.has_overlap)
        for name in fine_to_coarse.offsets:
            np.testing.assert_allclose(
                optimized.offsets[name],
                fine_to_coarse.offsets[name],
            )


if __name__ == "__main__":
    unittest.main()
