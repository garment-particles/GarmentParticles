"""
ImageNet Latent Dataset with safetensors.
"""

import os
import numpy as np
import json
from glob import glob
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset
from safetensors import safe_open
import matplotlib.cm as cm
from typing import List
import io
import h5py
import matplotlib.pyplot as plt
import datasets.pattern_utils as pat_utils
import svgpathtools as svgpath
from scipy.spatial.transform import Rotation as R
from shapely.validation import make_valid
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment
from patterns.pattern_converter import NNSewingPattern
import networkx as nx

def get_optimal_stitching_pairs(stitch_tags):
    G = nx.Graph()
    keys = sorted(stitch_tags.keys())

    # Build graph with distances as edge weights
    for k, key1 in enumerate(keys):
        for l, key2 in enumerate(keys):
            if k < l:  # only add each edge once (undirected)
                dist = np.linalg.norm(stitch_tags[key1] - stitch_tags[key2])
                G.add_edge(key1, key2, weight=dist)

    # Find minimum weight matching - each node appears in exactly one pair
    stitching_pairs = nx.min_weight_matching(G)
    return stitching_pairs

def vector_angle(v1, v2):
    """Find an angle between two 2D vectors"""
    v1, v2 = np.asarray(v1), np.asarray(v2)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cos) 
    # Cross to indicate correct relative orienataion of v2 w.r.t. v1
    cross = np.cross(v1, v2)
    
    if abs(cross) > 1e-5:
        angle *= np.sign(cross)
    return angle

def get_panel_from_vecs(edge_vecs, abs_vecs=False):
    start = np.array([0, 0])
    curves = []
    verts = [start]
    for edge_vec in edge_vecs:
        end = start + edge_vec[:2] if not abs_vecs else edge_vec[:2]
        verts.append(end)
        flag = edge_vec[6] > 0.5
        if flag == 1:
            third_point = pat_utils.rel_to_abs_2d(start, end, edge_vec[2:4]) if not abs_vecs else edge_vec[2:4]
            _, _, radius, large_arc, right = pat_utils.arc_from_three_points(start, end, third_point)
            curve = svgpath.Arc(
                    pat_utils.list_to_c(start), radius + 1j * radius,
                    rotation=0,
                    large_arc=large_arc,
                    sweep=right, #maya: not right
                    end=pat_utils.list_to_c(end)
                )
        else:
            control_points = edge_vec[2:6]
            if np.allclose(control_points, 0, atol=1e-3):
                curve = svgpath.Line(*pat_utils.list_to_c([start, end]))
            else:
                abs_control_points1 = pat_utils.rel_to_abs_2d(start, end, control_points[:2]) if not abs_vecs else control_points[:2]
                abs_control_points2 = pat_utils.rel_to_abs_2d(start, end, control_points[2:]) if not abs_vecs else control_points[2:]
                curve = svgpath.CubicBezier(*pat_utils.list_to_c([start, abs_control_points1, abs_control_points2, end]))
        
        curves.append(curve)
        start = end
    return curves

def rel_to_abs_vecs(edge_vecs):
    start = np.array([0, 0])
    verts = [start]
    abs_vecs = []
    for edge_vec in edge_vecs:
        end = start + edge_vec[:2]
        verts.append(end)
        flag = edge_vec[6]
        if flag == 1:
            third_point = pat_utils.rel_to_abs_2d(start, end, edge_vec[2:4])
            abs_vecs.append(np.concatenate([end, third_point, np.zeros_like(third_point), [flag]]))
        else:
            control_points = edge_vec[2:6]
            abs_control_points1 = pat_utils.rel_to_abs_2d(start, end, control_points[:2])
            abs_control_points2 = pat_utils.rel_to_abs_2d(start, end, control_points[2:])
            abs_vecs.append(np.concatenate([end, abs_control_points1, abs_control_points2, [flag]]))
        start = end
    return np.stack(abs_vecs)


def stitch_as_tag(panel_dict, stitch):
    """For every stitch, assign an approximate identifier (tag) of the stitch to the edges that are part of that stitch
        * tags are calculated as ~3D locations of the stitch when the garment is draped on the body in T-pose
        * It's calculated as average of the participating edges' endpoint -- Although very approximate, this should be enough
        to separate stitches from each other and from free edges
    Return
        * List of stitch tags for every stitch in the panel
    """
    # NOTE stitch tags values are independent from the choice of origin & edge order within a panel
    # iterate over stitches
    edge_tags = np.empty((2, 3))  # two 3D tags per edge
    for side_idx, side in enumerate(stitch[:2]):
        panel = panel_dict[side['panel']]
        edge_endpoints = panel['edges'][side['edge']]['endpoints']
        # get 2D locations of participating vertices -- per panel
        edge_endpoints = np.array([
            panel['vertices'][edge_endpoints[side]] for side in [0, 1]
        ])
        # Get edges midpoints (2D)
        edge_mean = edge_endpoints.mean(axis=0)

        # calculate their 3D locations
        edge_tags[side_idx] = _point_in_3D(edge_mean, panel['rotation'], panel['translation'])

    return edge_tags.mean(axis=0)


def _point_in_3D(local_coord, rotation, translation):
    """Apply 3D transformation to the point given in 2D local coordinated, e.g. on the panel
    * rotation is expected to be given in 'xyz' Euler anges (as in Autodesk Maya) or as 3x3 matrix"""

    # 2D->3D local
    local_coord = np.append(local_coord, 0)

    # Rotate
    rotation = R.from_euler('XYZ', rotation, degrees=True)
    rotated_point = rotation.apply(local_coord)

    # translate
    return rotated_point + translation

class GarmentEdgeDataset(Dataset):
    def __init__(
        self, 
        garment_particle_dir, 
        garment_edge_dir,
        gcd_dir,
        gcd_list_file,
        split_file,
        normalization, 
        n_samples=None,
        n_points=8192,
        n_curves=37,
        n_panels=37,
        jitter_pcd_scale=0,
        jitter_uv_scale=0,
        absolute_coords=False,
        uv_only=False,
        xyz_only=False,
        include_panel_indices=True,
        random_dropout_scale=0,
        order_panels_by="uv",
        predict_rigid_transformations=False,
        predict_attachment_type=False,
        predict_stitch_tags=False,
        predict_valid_mask=False,
        collate_fn=None,
        transformation_postfix="",
        **kwargs
        ):
        assert not (xyz_only and uv_only), "xyz_only and uv_only cannot be True at the same time"
        self.xyz_only = xyz_only
        self.uv_only = uv_only
        self.split_file = split_file
        self.garment_edge_dir = garment_edge_dir
        self.normalization = normalization
        self.absolute_coords = absolute_coords
        self.garment_particle_dir = garment_particle_dir
        self.include_panel_indices = include_panel_indices
        self.transformation_postfix = transformation_postfix
        self.predict_rigid_transformations = predict_rigid_transformations
        self.predict_attachment_type = predict_attachment_type
        self.predict_valid_mask = predict_valid_mask
        self.predict_stitch_tags = predict_stitch_tags
        self.n_edge_params = 8 + 3 * (int(self.predict_attachment_type) + int(self.predict_stitch_tags)) + int(self.predict_valid_mask)
        self.attachment_type_dict = {
            "null": np.zeros(3),
            "lower_interface": np.array([1, 0, 0]),
            "strapless_top": np.array([1, 0, 1]),
            "left_collar": np.array([1, 1, 0]),
            "right_collar": np.array([1, 1, 1]),
        }
        self.attachment_type_dict_inv = {
            0: "lower_interface",
            1: "strapless_top",
            2: "left_collar",
            3: "right_collar",
        }
        gcd_list = open(gcd_list_file, "r").readlines()
        gcd_list = [name.strip() for name in gcd_list]
        garment_names = [os.path.basename(name) for name in gcd_list]
        self.gcd_specs = {garment_name: os.path.join(gcd_dir, name, f"{garment_name}_specification.json") for name, garment_name in zip(gcd_list, garment_names)}
        
        data_names = open(self.split_file).readlines()
        particle_data_names = [name.strip().replace("npz", "h5") for name in data_names]
        edge_data_names = [name.replace("garment_particles", "panel_edge_vecs").replace(".h5", ".npz") for name in particle_data_names]
        self.particle_files = [os.path.join(self.garment_particle_dir, name) for name in particle_data_names]
        self.edge_files = [os.path.join(self.garment_edge_dir, name) for name in edge_data_names]
        # self.rotation_files = [os.path.join("/", *(name.split("/")[:-1]), "rotations.json") for name in self.edge_files]
        self.transformation_files = [os.path.join("/", *(name.split("/")[:-1]), "transformations.json" if self.transformation_postfix == "" else f"transformations_{self.transformation_postfix}.json") for name in self.edge_files]
        if n_samples is not None:
            self.particle_files = self.particle_files[:n_samples]
            self.edge_files = self.edge_files[:n_samples]
        self.n_points = n_points
        self.n_curves = n_curves
        self.n_panels = n_panels
        self.collate_fn = collate_fn
        self.jitter_pcd_scale = jitter_pcd_scale
        self.jitter_uv_scale = jitter_uv_scale
        self.random_dropout_scale = random_dropout_scale
        self.uv_only = uv_only
        self.limits = np.array([self.normalization.xlims, self.normalization.ylims])
        self.order_panels_by = order_panels_by
        assert self.order_panels_by in ["uv", "3d"], "order_panels_by must be either 'uv' or '3d'"
    def __len__(self):
        return len(self.particle_files)
    
    def order_panels(self, pcd):
        n_panels = len(pcd)
        if self.order_panels_by == "uv":
            centers = np.array([np.median(pcd[i][:, :2], axis=0) for i in range(n_panels)])[:, ::-1] # order by x, y
        elif self.order_panels_by == "3d":
            centers = np.array([np.median(pcd[i][:, 2:5], axis=0) for i in range(n_panels)])
            # centers = np.concatenate([centers[..., 2:3], centers[..., 0:1], centers[..., 1:2]], axis=-1) # order by y, x, z
        sorted_ids = np.lexsort(centers.T, axis=0)
        return sorted_ids

    def get_pts(self, particle_data, edge_data, transformations, panel_dict):
        all_pts = []
        all_edges = []
        all_panel_names = []
        all_panel_shifts = []
        for side in ["front", "back"]:
            side_data = particle_data[side]
            _all_pts = []
            _all_edges = []
            for panel_name, panel_data in side_data.items():
                boundary_pts = panel_data['boundary_verts']
                interior_pts = panel_data['interior_verts']
                inside_flag = np.concatenate([np.ones_like(boundary_pts)[:, :1], np.zeros_like(interior_pts)[:, :1]], axis=0)
                pts = np.concatenate([boundary_pts, interior_pts], axis=0)
                
                pts[:, :2] = (pts[:, :2] - np.array(self.limits[:, 0])) / (self.limits[:, 1] - self.limits[:, 0])
                if side == "back":
                    pts[:, 0] += 1
                pts = np.concatenate([pts, inside_flag], axis=1)
                if self.random_dropout_scale > 0:
                    if_add = np.random.rand() < 0.5
                    dropout_num = np.random.randint(0, np.ceil(pts.shape[0] * self.random_dropout_scale))
                    dropout_inds = np.random.choice(pts.shape[0], dropout_num, replace=False)
                    if if_add:
                        pts = np.concatenate([pts, pts[dropout_inds]])
                    else:
                        keep_inds = np.setdiff1d(np.arange(pts.shape[0]), dropout_inds)
                        pts = pts[keep_inds]
                
                
                transformation = transformations[panel_name]
                panel_shift = np.array(transformation[:2])
                rotation = transformation[2:]
                rotation = R.from_euler('XYZ', rotation, degrees=True)   # XYZ
                res = rotation.apply([0, 1, 0])
                flat_rot_angle = -vector_angle([0, 1], res[:2])
                if self.predict_rigid_transformations:
                    translation_3d = panel_dict[panel_name]["translation"]
                    current_rotation = R.from_euler('z', -flat_rot_angle, degrees=False) 
                    diff_rotation = rotation * current_rotation.inv()
                    diff_rotation = diff_rotation.as_euler('XYZ', degrees=False)
                    panel_rigid_transformation = np.concatenate([translation_3d, diff_rotation])
                    panel_shift = np.concatenate([panel_shift, panel_rigid_transformation])
                else:
                    panel_shift = np.concatenate([panel_shift, np.zeros(6)])
                rot_mat = np.array([[np.cos(flat_rot_angle), -np.sin(flat_rot_angle)], [np.sin(flat_rot_angle), np.cos(flat_rot_angle)]])
                edges = edge_data[panel_name]
                edges[:, :2] = (edges[:, :2].reshape(-1, 2) @ rot_mat).reshape(-1, 2)
            
                if self.absolute_coords:
                    edges = rel_to_abs_vecs(edges)
                    if not self.separate_panel_shifts:
                        _edges = edges[:, :-1].reshape(-1, 2) + panel_shift
                        _edges = (_edges - np.array(self.limits[:, 0])) / (self.limits[:, 1] - self.limits[:, 0])
                        edges[:, :-1] = _edges.reshape(-1, 6)
                if self.predict_valid_mask:
                    edges = np.concatenate([edges, np.ones((edges.shape[0], 1))], axis=-1)
                if self.predict_attachment_type:
                    attachment_types = [edge.get("label", "null") for edge in panel_dict[panel_name]["edges"]]
                    attachment_types = [self.attachment_type_dict.get(attachment_type, np.zeros(3)) for attachment_type in attachment_types]
                    attachment_types = np.stack(attachment_types, axis=0)
                    edges = np.concatenate([edges, attachment_types], axis=-1)
                if edges.shape[0] < self.n_curves:
                    edges = np.concatenate([edges, np.zeros((self.n_curves - edges.shape[0], edges.shape[1]))], axis=0)
                
                _all_edges.append(edges)
                _all_pts.append(pts)
                all_panel_names.append(panel_name)
                all_panel_shifts.append(panel_shift)
            all_edges.extend(_all_edges)
            all_pts.extend(_all_pts)
        panel_indices = self.order_panels(all_pts).flatten().tolist()
        return all_pts, all_edges, panel_indices, all_panel_names, all_panel_shifts
    
    
    def get_stitch_info(self, stitches, panel_names, panel_dict):
        stitch_flag = np.zeros((self.n_panels, self.n_curves))
        stitch_tags = np.zeros((self.n_panels, self.n_curves, 3))
        stitch_matrix = np.zeros((self.n_panels, self.n_curves, self.n_panels, self.n_curves))
        for stitch in stitches:
            stitch_tag = stitch_as_tag(panel_dict, stitch)
            panel_name1 = stitch[0]["panel"]
            panel_name2 = stitch[1]["panel"]
            idx1 = panel_names.index(panel_name1)
            idx2 = panel_names.index(panel_name2)
            edge_id1 = stitch[0]["edge"]
            edge_id2 = stitch[1]["edge"]
            stitch_flag[idx1, edge_id1] = 1
            stitch_flag[idx2, edge_id2] = 1
            stitch_tags[idx1, edge_id1] = stitch_tag
            stitch_tags[idx2, edge_id2] = stitch_tag
            stitch_matrix[idx1, edge_id1, idx2, edge_id2] = 1
            stitch_matrix[idx2, edge_id2, idx1, edge_id1] = 1
        return stitch_flag, stitch_matrix, stitch_tags
    
    def __getitem__(self, idx):
        particle_file = self.particle_files[idx]
        edge_file = self.edge_files[idx]
        transformation_file = self.transformation_files[idx]
        garment_name = particle_file.split("/")[-2]
        spec_file = self.gcd_specs[garment_name]
        with open(spec_file, "r") as f:
            spec = json.load(f)
        stitches = spec["pattern"]["stitches"]
        panel_dict = spec["pattern"]["panels"]
        with open(transformation_file, "r") as f:
            transformations = json.load(f)
        particle_data = h5py.File(particle_file, 'r')
        edge_data = np.load(edge_file)
        pcd, edges, panel_indices, panel_names, panel_shifts = self.get_pts(particle_data, edge_data, transformations, panel_dict)
        edges = [edges[i] for i in panel_indices]
        panel_names = [panel_names[i] for i in panel_indices]
        panel_shifts = [panel_shifts[i] for i in panel_indices]
        stitch_flag, stitch_matrix, stitch_tags = self.get_stitch_info(stitches, panel_names, panel_dict)
        edges = np.stack(edges, axis=0)
        
        panel_indices = [i * np.ones(len(pcd[k]), dtype=int) for k, i in enumerate(np.argsort(panel_indices))]
        panel_indices = np.concatenate(panel_indices, axis=0)
        pcd = np.concatenate(pcd, axis=0)
        if edges.shape[0] < self.n_panels:
            edges = np.concatenate([edges, np.zeros((self.n_panels - edges.shape[0], edges.shape[1], edges.shape[2]))], axis=0)
        edges = np.concatenate([edges, stitch_flag[:, :, None]], axis=-1)
        if self.predict_stitch_tags:
            edges = np.concatenate([edges, stitch_tags], axis=-1)
        
        panel_shifts = np.stack(panel_shifts, axis=0)
        panel_shifts = (panel_shifts - np.array(self.normalization.transformation_mean)[:self.n_edge_params]) / np.array(self.normalization.transformation_std)[:self.n_edge_params]    
        if panel_shifts.shape[1] < self.n_edge_params:
            panel_shifts = np.concatenate([panel_shifts, np.zeros((panel_shifts.shape[0], self.n_edge_params - panel_shifts.shape[1]))], axis=1)
        if panel_shifts.shape[0] < self.n_panels:
            panel_shifts = np.concatenate([panel_shifts, np.zeros((self.n_panels - panel_shifts.shape[0], self.n_edge_params))], axis=0)
        edges = (edges - np.array(self.normalization.edge_mean)[:self.n_edge_params]) / np.array(self.normalization.edge_std)[:self.n_edge_params]    
        edges = np.concatenate([panel_shifts[:, None], edges[:, :-1]], axis=1) # [n_panels, n_curves, self.n_edge_params]
        edges = edges.reshape(self.n_panels*self.n_curves, self.n_edge_params)
        pcd = (pcd - np.array(self.normalization.pcd_mean)) / np.array(self.normalization.pcd_std)
        if self.jitter_pcd_scale > 0:
            pcd[..., 2:5] = pcd[..., 2:5] + np.random.randn(pcd.shape[0], 3) * self.jitter_pcd_scale
        if self.jitter_uv_scale > 0:
            pcd[..., :2] = pcd[..., :2] + np.random.randn(pcd.shape[0], 2) * self.jitter_uv_scale
        n = pcd.shape[0]
        while pcd.shape[0] < self.n_points:
            pcd = np.concatenate([pcd, pcd], 0)
            panel_indices = np.concatenate([panel_indices, panel_indices], 0)
        pcd = pcd[:self.n_points]
        panel_indices = panel_indices[:self.n_points]
        if self.uv_only:
            pcd = np.concatenate([pcd[:, :2], pcd[:, -1:]], axis=1)
        if self.xyz_only:
            pcd = pcd[:, 2:5]
        mask = np.ones(self.n_points).astype(bool)
        mask[n:] = False
        pcd = torch.from_numpy(pcd).float()
        edges = torch.from_numpy(edges).float()
        
        data_dict = {
            "panel_points": pcd,
            "input": edges,
            "panel_points_mask": torch.from_numpy(mask).bool(),
            "panel_indices": torch.from_numpy(panel_indices).int() if self.include_panel_indices else None,
            "spec_path": spec_file,
        }
        return data_dict

    

    def reconstruct(
        self, 
        pred, 
        panel_points=None, 
        input=None, 
        panel_points_mask=None, 
        panel_indices=None, 
        spec_path=None,
        **kwargs):
        has_gt = input is not None
        edge_data = pred.reshape(-1, self.n_panels, self.n_curves, self.n_edge_params)
        panel_metadata = edge_data[:, :, 0, :8]
        panel_metadata = panel_metadata * np.array(self.normalization.transformation_std)[:8] + np.array(self.normalization.transformation_mean)[:8]
        if self.predict_rigid_transformations:
            panel_shifts = panel_metadata[:, :, :2]
            panel_translations = panel_metadata[:, :, 2:5]
            panel_rotations = panel_metadata[:, :, 5:8]
        else:
            panel_shifts = panel_metadata[:, :, :2]
            panel_translations = np.zeros((self.n_panels, 3))
            panel_rotations = np.zeros((self.n_panels, 3))
        edge_data = edge_data[:, :, 1:]
        if not self.predict_valid_mask:
            valid_mask = np.logical_not(np.isclose(edge_data[..., :2], 0, atol=1).all(axis=-1))
        else:
            valid_mask = edge_data[..., 7] > 0.5
        n_panels = [valid_mask[i].any(axis=-1).sum() for i in range(valid_mask.shape[0])]
        
        edge_data = edge_data * np.array(self.normalization.edge_std)[:self.n_edge_params] + np.array(self.normalization.edge_mean)[:self.n_edge_params]
        if self.predict_attachment_type:
            attachment_types = edge_data[..., 8:11] > 0.5
            attachment_types = np.sum(attachment_types.astype(int) * np.array([4, 2, 1]), axis=-1)
        stitch_flag = edge_data[..., 11] > 0.5
        if self.predict_stitch_tags:
            stitch_tags = edge_data[..., 12:15]
        edge_data = edge_data[..., :7]
        
        if has_gt:
            input = input.reshape(-1, self.n_panels, self.n_curves, self.n_edge_params)
            gt_panel_metadata = input[:, :, 0, :8]
            gt_panel_metadata = gt_panel_metadata * np.array(self.normalization.transformation_std)[:8] + np.array(self.normalization.transformation_mean)[:8]
            if self.predict_rigid_transformations:
                gt_panel_shifts = gt_panel_metadata[:, :, :2]
                gt_panel_translations = gt_panel_metadata[:, :, 2:5]
                gt_panel_rotations = gt_panel_metadata[:, :, 5:]
            else:
                gt_panel_shifts = gt_panel_metadata[:, :, :2]
                gt_panel_translations = np.zeros((self.n_panels, 3))
                gt_panel_rotations = np.zeros((self.n_panels, 3))
            input = input[:, :, 1:]
            input = input * np.array(self.normalization.edge_std)[:self.n_edge_params] + np.array(self.normalization.edge_mean)[:self.n_edge_params]
            if not self.predict_valid_mask:
                gt_valid_mask = np.logical_not(np.isclose(input, 0, atol=1).all(axis=-1))
            else:
                gt_valid_mask = input[..., 7] > 0.5
            gt_n_panels = [gt_valid_mask[i].any(axis=-1).sum() for i in range(gt_valid_mask.shape[0])]
            
            if self.predict_attachment_type:
                gt_attachment_types = input[..., 8:11] > 0.5
                gt_attachment_types = np.sum(gt_attachment_types.astype(int) * np.array([4, 2, 1]), axis=-1)
            gt_stitch_flag = input[..., 11] > 0.5
            if self.predict_stitch_tags:
                gt_stitch_tags = input[..., 12:15]
            else:
                gt_stitch_tags = None
            input = input[..., :7]
        
        if self.uv_only:
            panel_points = np.concatenate([panel_points[..., :2], np.zeros_like(panel_points), panel_points[..., -1:]], axis=-1)
        if self.xyz_only:
            panel_points = np.concatenate([np.zeros_like(panel_points[..., :2]), panel_points, panel_points[..., :1]], axis=-1)
        
        pcds = panel_points * np.array(self.normalization.pcd_std) + np.array(self.normalization.pcd_mean)
        pcds[..., :2] = pcds[..., :2] * (self.limits[:, 1] - self.limits[:, 0]) + np.array(self.limits[:, 0])
        
        
        

        N = edge_data.shape[0]
        imgs = []
        pts = []
        patterns = []
        for i in range(N):
            # Extract RGB and UV coordinates
            valid_panel_mask = valid_mask[i].any(axis=-1)
            panel_valid_mask = valid_mask[i][valid_panel_mask]
            edges = edge_data[i][valid_panel_mask]
            rotations = panel_rotations[i][valid_panel_mask]
            translations = panel_translations[i][valid_panel_mask]
            panel_shift = panel_shifts[i][valid_panel_mask]
            edges = [edges[j][panel_valid_mask[j]] for j in range(panel_valid_mask.shape[0])]
            stitch_mask = stitch_flag[i][valid_panel_mask]
            stitch_tag = stitch_tags[i][valid_panel_mask]
            stitch_tag = [stitch_tag[j][panel_valid_mask[j]] for j in range(panel_valid_mask.shape[0])]
            stitch_mask = [stitch_mask[j][panel_valid_mask[j]] for j in range(panel_valid_mask.shape[0])]
            attachment_flag = attachment_types[i] >= 4
            attachment_flag = attachment_flag[valid_panel_mask]
            attachment_type = attachment_types[i][valid_panel_mask]
            attachment_type = [attachment_type[j][panel_valid_mask[j]] for j in range(panel_valid_mask.shape[0])]
            attachment_flag = [attachment_flag[j][panel_valid_mask[j]] for j in range(panel_valid_mask.shape[0])]
            pcd = pcds[i][panel_points_mask[i]]
            # pcd -= pcd.min(0)
            pt = pcd[:, 2:6]
            pts.append(pt)
            
            
                
            pattern = NNSewingPattern()
            try:
                pattern.pattern_from_tensors(edges, panel_rotations=rotations, panel_translations=translations, padded=False, valid_mask=[np.ones(edges[j].shape[0]).astype(bool) for j in range(panel_valid_mask.shape[0])])
            except:
                pass
            if self.predict_attachment_type:
                for j, panel_attachment_flag in enumerate(attachment_flag):
                    col_inds = np.where(panel_attachment_flag)[0]
                    for k in col_inds:
                        attachment_type_str = self.attachment_type_dict_inv[attachment_type[j][k] - 4]
                        if f"panel_{j}" not in pattern.pattern["panels"] or k >= len(pattern.pattern["panels"][f"panel_{j}"]["edges"]):
                            import ipdb; ipdb.set_trace()
                        pattern.pattern["panels"][f"panel_{j}"]["edges"][k]["label"] = attachment_type_str
                    
            if self.predict_stitch_tags:
                stitch_tag_dict = {}
                for j, panel_stitch_mask in enumerate(stitch_mask):
                    col_inds = np.where(panel_stitch_mask)[0]
                    for k in col_inds:
                        stitch_tag_dict[(j, k)] = stitch_tag[j][k]
                stitching_pairs = get_optimal_stitching_pairs(stitch_tag_dict)
                for (j, k), (m, n) in stitching_pairs:
                    pattern.pattern["stitches"].append([{"panel": f"panel_{j}", "edge": k.item()}, {"panel": f"panel_{m}", "edge": n.item()}])
                
            patterns.append(pattern)
                
            for j in range(panel_valid_mask.shape[0]):
                panel_edge = edges[j]
                stitches = stitch_mask[j]
                try:
                    paths = get_panel_from_vecs(panel_edge, self.absolute_coords)
                    path = svgpath.Path(*paths)
                    total_length = path.length()
                    for k, seg in enumerate(paths):
                        length = seg.length()
                        num_samples = int(length / total_length * 100)
                        points = pat_utils.path_to_polygon_simple(svgpath.Path(seg), total_points=num_samples)
                        points += panel_shift[j]
                        plt.plot(points[:, 0], points[:, 1], color='blue', linestyle= '--' if stitches[k] else '-', linewidth=2, alpha=1)
                except Exception as e:
                    print("Error in get_panel_from_vecs for reconstruction: ", e)
                    pass
            if has_gt:
                gt_edges = input[i]
                gt_n_panel = gt_n_panels[i]
                for j in range(gt_n_panel):
                    gt_panel_edge = gt_edges[j][gt_valid_mask[i, j]]
                    gt_stitches = gt_stitch_flag[i, j][gt_valid_mask[i, j]]
                    gt_panel_shift = gt_panel_shifts[i, j]
                    
                    try:
                        gt_paths = get_panel_from_vecs(gt_panel_edge, self.absolute_coords)
                        gt_path = svgpath.Path(*gt_paths)
                        total_length = gt_path.length()
                        for k, seg in enumerate(gt_paths):
                            length = seg.length()
                            num_samples = int(length / total_length * 100)
                            points = pat_utils.path_to_polygon_simple(svgpath.Path(seg), total_points=num_samples)
                            points += gt_panel_shift
                            plt.plot(points[:, 0], points[:, 1], color='red', linestyle= '--' if gt_stitches[k] else '-', linewidth=2, alpha=0.5)
                    except Exception as e:
                        print("Error in get_panel_from_vecs for ground truth: ", e)
                        pass
                    
                
            inside_flag = pcd[:, -1] < 0.5
            interior_pcd = pcd[inside_flag]
            boundary_pcd = pcd[~inside_flag]
            plt.scatter(interior_pcd[:, 0], interior_pcd[:, 1], color="green", s=0.5)
            plt.scatter(boundary_pcd[:, 0], boundary_pcd[:, 1], color="black", s=0.5)
            plt.tight_layout()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis("off")
            
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            img = Image.open(buf)
            imgs.append(img)
        panel_data = {
            "preds": edge_data,
            "pred_stitches": stitch_flag,
            "pred_stitch_tags": stitch_tags,
            "pred_shifts": panel_shifts,
            "pred_mask": valid_mask,
            "pcds": pcds,
            "pcd_mask": panel_points_mask,
            "pred_translations": panel_translations,
            "pred_rotations": panel_rotations,
        }
        if has_gt:
            panel_data["gts"] = input
            panel_data["gt_stitches"] = gt_stitch_flag
            panel_data["gt_stitch_tags"] = gt_stitch_tags
            panel_data["gt_shifts"] = gt_panel_shifts
            panel_data["gt_mask"] = gt_valid_mask   
            panel_data["gt_translations"] = gt_panel_translations
            panel_data["gt_rotations"] = gt_panel_rotations
        if spec_path is not None:
            gt_patterns = [NNSewingPattern(path) for path in spec_path]
        else:
            gt_patterns = []
        return imgs, pts, panel_data, [], patterns, gt_patterns
    
    def evaluate_simplified(
        self, 
        pred, 
        input, 
        **kwargs
    ):
        edge_data = pred.reshape(-1, self.n_panels, self.n_curves, self.n_edge_params)
        panel_metadata = edge_data[:, :, 0, :8]
        panel_metadata = panel_metadata * np.array(self.normalization.transformation_std)[:8] + np.array(self.normalization.transformation_mean)[:8]
        panel_shifts = panel_metadata[:, :, :2]
        edge_data = edge_data[:, :, 1:]
        valid_mask = edge_data[..., 7] > 0.5
        
        edge_data = edge_data * np.array(self.normalization.edge_std)[:self.n_edge_params] + np.array(self.normalization.edge_mean)[:self.n_edge_params]
        attachment_types = edge_data[..., 8:11] > 0.5
        attachment_types = np.sum(attachment_types.astype(int) * np.array([4, 2, 1]), axis=-1)
        stitch_flag = edge_data[..., 11] > 0.5
        stitch_tags = edge_data[..., 12:15]
        edge_data = edge_data[..., :7]
        
        input = input.reshape(-1, self.n_panels, self.n_curves, self.n_edge_params)
        gt_panel_metadata = input[:, :, 0, :8]
        gt_panel_metadata = gt_panel_metadata * np.array(self.normalization.transformation_std)[:8] + np.array(self.normalization.transformation_mean)[:8]
        gt_panel_shifts = gt_panel_metadata[:, :, :2]
        input = input[:, :, 1:]
        input = input * np.array(self.normalization.edge_std)[:self.n_edge_params] + np.array(self.normalization.edge_mean)[:self.n_edge_params]
        gt_valid_mask = input[..., 7] > 0.5
        
        gt_attachment_types = input[..., 8:11] > 0.5
        gt_attachment_types = np.sum(gt_attachment_types.astype(int) * np.array([4, 2, 1]), axis=-1)
        gt_stitch_flag = input[..., 11] > 0.5
        gt_stitch_tags = input[..., 12:15]
        input = input[..., :7]
        panel_accs, edge_accs, all_stitch_accs, all_pattern_iou = self.evaluate(
            preds=edge_data, 
            pred_stitches=stitch_flag, 
            pred_stitch_tags=stitch_tags, 
            pred_shifts=panel_shifts, 
            pred_mask=valid_mask, 
            gts=input, 
            gt_stitches=gt_stitch_flag, 
            gt_stitch_tags=gt_stitch_tags, 
            gt_shifts=gt_panel_shifts, 
            gt_mask=gt_valid_mask)
        return panel_accs, edge_accs, all_stitch_accs, all_pattern_iou
    
    def evaluate(
        self, 
        preds, 
        pred_stitches,
        pred_stitch_tags,
        pred_shifts,
        pred_mask, 
        gts, 
        gt_stitches, 
        gt_stitch_tags,
        gt_shifts, 
        gt_mask, 
        **kwargs):
        N = len(preds)
        panel_accs = []
        edge_accs = []
        all_pattern_iou = []
        all_stitch_accs = []
        for i in range(N):
            pred_edge_mask = pred_mask[i]
            pred_panel_mask = pred_edge_mask.any(-1)
            pred_edge_mask = pred_edge_mask[pred_panel_mask]
            pred_edge = preds[i][pred_panel_mask]
            pred_shift = pred_shifts[i][pred_panel_mask]
            pred_stitch = pred_stitches[i][pred_panel_mask]
            pred_stitch_tag = pred_stitch_tags[i][pred_panel_mask]
            pred_edge = [pred_edge[j][pred_edge_mask[j]] for j in range(pred_edge_mask.shape[0])]
            pred_stitch = [pred_stitch[j][pred_edge_mask[j]] for j in range(pred_edge_mask.shape[0])]
            pred_stitch_tag = [pred_stitch_tag[j][pred_edge_mask[j]] for j in range(pred_edge_mask.shape[0])]

            gt_edge_mask = gt_mask[i]
            gt_panel_mask = gt_edge_mask.any(-1)
            gt_edge_mask = gt_edge_mask[gt_panel_mask]
            gt_edge = gts[i][gt_panel_mask]
            gt_shift = gt_shifts[i][gt_panel_mask]
            gt_stitch = gt_stitches[i][gt_panel_mask]
            gt_stitch_tag = gt_stitch_tags[i][gt_panel_mask]
            gt_edge = [gt_edge[j][gt_edge_mask[j]] for j in range(gt_edge_mask.shape[0])]
            gt_stitch = [gt_stitch[j][gt_edge_mask[j]] for j in range(gt_edge_mask.shape[0])]
            gt_stitch_tag = [gt_stitch_tag[j][gt_edge_mask[j]] for j in range(gt_edge_mask.shape[0])]

            gt_n_panels = len(gt_edge)
            pred_n_panels = len(pred_edge)
            if pred_n_panels == gt_n_panels:
                panel_accs.append(1)
            else:
                panel_accs.append(0)
                continue
            n_panels = pred_n_panels
            cost_matrix = np.zeros((n_panels, n_panels))
            for j in range(n_panels):
                for k in range(n_panels):
                    cost_matrix[j, k] = np.linalg.norm(pred_shift[j] - gt_shift[k])
            assignment = linear_sum_assignment(cost_matrix)
            correct_n_panels = 0
            pattern_iou = []
            for j in range(n_panels):
                idx = assignment[0][j]
                gt_idx = assignment[1][j]
                pred_panel_edge = pred_edge[idx]
                gt_panel_edge = gt_edge[gt_idx]
                n_edges = pred_panel_edge.shape[0]
                n_edges_gt = gt_panel_edge.shape[0]
                if n_edges == n_edges_gt:
                    correct_n_panels += 1
                try:
                    paths = get_panel_from_vecs(gt_panel_edge, self.absolute_coords)
                    path = svgpath.Path(*paths)
                    points = pat_utils.path_to_polygon_simple(path)
                    points -= points.min(0)
                    points = np.concatenate([points, points[0:1]])
                    gt_polygon = pat_utils.safe_polygon(points)
                except Exception as e:
                    print("Error in get_panel_from_vecs for ground truth: ", e)
                    continue
                
                try:
                    paths = get_panel_from_vecs(pred_panel_edge, self.absolute_coords)
                    path = svgpath.Path(*paths)
                    points = pat_utils.path_to_polygon_simple(path)
                    points -= points.min(0)
                    points = np.concatenate([points, points[0:1]])
                    polygon = pat_utils.safe_polygon(points)
                except Exception as e:
                    print("Error in get_panel_from_vecs for reconstruction: ", e)
                    continue
                iou = gt_polygon.intersection(polygon).area / (gt_polygon.union(polygon).area + 1e-6)
                pattern_iou.append(iou)
            pattern_iou = sum(pattern_iou) / max(len(pattern_iou), 1)
            
            
            if self.predict_stitch_tags:
                correct_n_stitches = 0
                stitch_tag_dict = {}
                for j, panel_pred_stitch in enumerate(pred_stitch):
                    col_inds = np.where(panel_pred_stitch)[0]
                    for k in col_inds:
                        stitch_tag_dict[(j, k)] = pred_stitch_tag[j][k]
                stitching_pairs = get_optimal_stitching_pairs(stitch_tag_dict)

                gt_stitch_tag_dict = {}
                for j, panel_gt_stitch in enumerate(gt_stitch):
                    col_inds = np.where(panel_gt_stitch)[0]
                    for k in col_inds:
                        gt_stitch_tag_dict[(j, k)] = gt_stitch_tag[j][k]
                stitching_pairs_gt = get_optimal_stitching_pairs(gt_stitch_tag_dict)

                for (j, k), (m, n) in stitching_pairs_gt:
                    j_inv = np.where(assignment[1] == j)[0][0]
                    m_inv = np.where(assignment[1] == m)[0][0]
                    pred_j = assignment[0][j_inv]
                    pred_m = assignment[0][m_inv]
                    if ((pred_j, k), (pred_m, n)) in stitching_pairs or ((pred_m, n), (pred_j, k)) in stitching_pairs:
                        correct_n_stitches += 1
                stitch_acc = correct_n_stitches / max(len(stitching_pairs_gt), 1)
            else:
                stitch_acc = None
            all_stitch_accs.append(stitch_acc)
            all_pattern_iou.append(pattern_iou)
            edge_accs.append(correct_n_panels / n_panels)
        return panel_accs, edge_accs, all_stitch_accs, all_pattern_iou
    
if __name__ == "__main__":
    from omegaconf import DictConfig
    import yaml
    import hydra
    normalization = DictConfig(yaml.load(open("/orion/u/w4756677/garment/InteractGarment/src/configs/dataset/normalization/garment_edges_relative_v2.2.yaml", "r"), Loader=yaml.FullLoader))
    pcd_normalization = DictConfig(yaml.load(open("/orion/u/w4756677/garment/InteractGarment/src/configs/dataset/normalization/garment_particles_v2.2.yaml", "r"), Loader=yaml.FullLoader))
    normalization.pcd_mean = pcd_normalization.pcd_mean
    normalization.pcd_std = pcd_normalization.pcd_std
    normalization.xlims = pcd_normalization.xlims
    normalization.ylims = pcd_normalization.ylims
    
    dataset = GarmentEdgeDataset(
        garment_particle_dir="/orion/u/w4756677/garment/gcdv2/garment_particles_v2.1_11182025",
        garment_edge_dir="/orion/u/w4756677/garment/gcdv2/panel_edge_vecs",
        gcd_dir="/orion/u/w4756677/garment/gcdv2/garmentcodedatav2",
        gcd_list_file="assets/gcd_list.txt",
        split_file="/orion/u/w4756677/garment/InteractGarment/src/assets/garment_particle_v2_filtered_11182025.txt",
        order_panels_by="3d",
        transformation_postfix="11182025",
        absolute_coords=False,
        normalization=normalization,
        n_curves=38,
        n_panels=37,
        jitter_pcd_scale=0,
        jitter_uv_scale=0,
        random_dropout_scale=0,
        predict_rigid_transformations=True,
        predict_attachment_type=True,
        predict_stitch_tags=True,
        predict_valid_mask=True,
    )
    
    print(len(dataset))
    data_dict = dataset[0]
    print(data_dict.keys())
    edges = data_dict["input"]
    print(edges.mean(axis=-1), edges.std(axis=-1))
    edges = edges.numpy()[None]
    plt.scatter(data_dict["panel_points"][data_dict["panel_points_mask"]][:, 0], data_dict["panel_points"][data_dict["panel_points_mask"]][:, 1], color="green", s=0.5)
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("panel_points.png")
    plt.close()
    plt.scatter(data_dict["panel_points"][data_dict["panel_points_mask"]][:, 2], data_dict["panel_points"][data_dict["panel_points_mask"]][:, 3], color="black", s=0.5)
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("panel_points_3d.png")
    plt.close()
    for k in data_dict.keys():
        print(k, data_dict[k].shape)
        data_dict[k] = data_dict[k][None].numpy() if isinstance(data_dict[k], torch.Tensor) else data_dict[k][None]
    img, pts, panel_data, _, patterns = dataset.reconstruct(pred=edges, **data_dict)
    img[0].save("reconstruction.png")
    np.savez_compressed("reconstruction.npz", **panel_data)
    patterns[0].serialize("tmp")
    # all_pattern_verts_l2, all_pattern_iou, edge_acc, panel_acc = dataset.evaluate(x, **data_dict)
    # print(all_pattern_verts_l2, all_pattern_iou, edge_acc, panel_acc)
