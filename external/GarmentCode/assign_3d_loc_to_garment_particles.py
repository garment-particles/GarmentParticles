from pathlib import Path
import trimesh
import numpy as np
from pygarment.meshgen.boxmeshgen import BoxMesh, DegenerateTrianglesError
import igl 
from shapely.geometry import Polygon
from shapely import Point
from matplotlib import pyplot as plt
import torch 
import torch.nn.functional as F 
import torch.optim as optim 
from collections import defaultdict
from pygarment.vd_utils.constrained_delaunay_triangulation import constrained_delaunay_triangulation
from pygarment.vd_utils.bichromatic_separator import bichromatic_voronoi_separator, voronoi_cells
from pygarment.meshgen.render.texture_utils import unwarp_UV
from anytree import LevelOrderIter
import json
import logging
import traceback
import sys
import os

def process_garment_single(pattern_folder, out_folder, panel_offset):
    try:
        garment_name = pattern_folder.split('/')[-1]
        
        out_folder = Path(out_folder) / garment_name
        out_folder.mkdir(parents=True, exist_ok=True)
        
        pattern_spec = Path(pattern_folder) / f'{garment_name}_specification.json'
        
        box_mesh_path = Path(pattern_folder) / f'{garment_name}_boxmesh.ply'
        
        sim_mesh_path = Path(pattern_folder) / f'{garment_name}_sim.ply'
        
        box_mesh = trimesh.load_mesh(box_mesh_path)
        
        sim_mesh = trimesh.load_mesh(sim_mesh_path)
        garment_box_mesh = BoxMesh(pattern_spec)
        garment_box_mesh.load()
        front_islands = [island for island in garment_box_mesh.vertex_texture if any(node.name == island['panel_name'] for node in LevelOrderIter(garment_box_mesh.panel_tree.children[0]))]
        back_islands = [island for island in garment_box_mesh.vertex_texture if any(node.name == island['panel_name'] for node in LevelOrderIter(garment_box_mesh.panel_tree.children[1]))]
        uv_dict = {}
        has_overlap = False
        for side, islands in [('front', front_islands), ('back', back_islands)]:
            panel_name = islands["panel_name"]
            panel = garment_box_mesh.panels[panel_name]
            verts = panel.panel_vertices
            verts = np.array(verts)
            flat_rot_angle = islands['rotation']
            rotation_center = islands['rotation_center']
            rotation_matrix = np.array([[np.cos(flat_rot_angle), -np.sin(flat_rot_angle)], [np.sin(flat_rot_angle), np.cos(flat_rot_angle)]])
            verts = (rotation_matrix @ (verts - rotation_center)[..., None])[..., 0] + rotation_center + islands["translation"]
            verts += offsets[panel_name]
            faces = panel.panel_faces
            faces = np.array(faces)
            all_verts.append(verts)
            all_faces.append(faces + face_offset)
            face_offset += len(verts)
            
    verts = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)

    particles = np.load(f"/orion/u/w4756677/garment/InteractGarment/external/GarmentCode/garment_particles_rerun_rerun/{garment_name}/garment_particles_{garment_name}.npz")
    
# save the garment mesh