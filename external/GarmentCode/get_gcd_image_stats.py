import matplotlib.pyplot as plt 
import numpy as np
import os 
from glob import glob
import cv2
import igl
import h5py
from tqdm import tqdm
from shapely.geometry import Polygon
save_path = "/miele/data/gcd/gcdv2_images_rerun_again/"
folders = glob(os.path.join(save_path, "*", "*.hdf5"))
print(len(folders))
target_min_vert = np.array([-135,   -40])
target_max_vert = np.array([135, 200])
target_min_color = np.array([-63.32013931, -12, -40])
target_max_color = np.array([63.51664658, 172.80234273,  40])
out_folder = "gcd_image_stats_again/"
if not os.path.exists(out_folder):
    os.makedirs(out_folder, exist_ok=True)
if not os.path.exists(os.path.join(out_folder, "has_overlap_list.txt")):
    has_overlap_list = []
else:
    has_overlap_list = open(os.path.join(out_folder, "has_overlap_list.txt"), "r").readlines()
    has_overlap_list = [item.strip() for item in has_overlap_list]
if not os.path.exists(os.path.join(out_folder, "visited_list.txt")):
    visited_list = []
else:
    visited_list = open(os.path.join(out_folder, "visited_list.txt"), "r").readlines()
    visited_list = [item.strip() for item in visited_list]

oversize_list = []
max_verts, min_verts = [], []
max_verts_names, min_verts_names = [], []
max_colors, min_colors = [], []
max_colors_names, min_colors_names = [], []
for texture_file in tqdm(folders):
    data_name = os.path.basename(texture_file).replace("_texture.hdf5", "")
    if texture_file in visited_list:
        continue
    visited_list.append(texture_file)
    with h5py.File(texture_file, 'r') as f:
        has_overlap = f['has_overlap'][()]
        if has_overlap:
            has_overlap_list.append(texture_file)
            continue
        # Initialize lists for front and back panels
        front_verts, front_colors, front_faces = [], [], []
        back_verts, back_colors, back_faces = [], [], []
        
        for panel_name in f.keys():
            if panel_name == 'has_overlap':
                continue
            panel_data = f[panel_name]
            verts = panel_data['vertices'][()]
            faces = panel_data['faces'][()]
            colors = panel_data['colors'][()]
            
            if panel_name.startswith('front_'):
                front_verts.append(verts)
                front_colors.append(colors)
                front_faces.append(faces)
            elif panel_name.startswith('back_'):
                back_verts.append(verts)
                back_colors.append(colors)
                back_faces.append(faces)
                
    
    # Combine panels and render
    overlap = False
    for panel_type, verts, colors, faces in [
        ('front', front_verts, front_colors, front_faces),
        ('back', back_verts, back_colors, back_faces)
    ]:
        if len(verts) == 0:
            continue
        
        # check overlap
        polygons = []
        for panel_verts, panel_faces in zip(verts, faces):
            boundary_faces= igl.boundary_loop(panel_faces.astype(int))
            boundary_verts = panel_verts[boundary_faces]
            polygon = Polygon(boundary_verts)
            polygons.append(polygon)
        if len(polygons) > 1:
            for i in range(len(polygons)):
                for j in range(i+1, len(polygons)):
                    if polygons[i].intersects(polygons[j]):
                        overlap = True
                        break
        if overlap:
            print(f"Overlap found in {texture_file}")
            has_overlap_list.append(texture_file)
            continue
            
        # Combine all panels
        verts = np.concatenate(verts, axis=0)
        colors = np.concatenate(colors, axis=0)
        _min_verts = np.min(verts, axis=0)
        min_verts.append(_min_verts)
        min_verts_names.append(data_name + "_" + panel_type)
        _max_verts = np.max(verts, axis=0)
        if _max_verts[0] > target_max_vert[0] or _min_verts[0] < target_min_vert[0]:
            oversize_list.append(texture_file)
        if _max_verts[1] > target_max_vert[1] or _min_verts[1] < target_min_vert[1]:
            oversize_list.append(texture_file)
        
        max_verts.append(_max_verts)
        max_verts_names.append(data_name + "_" + panel_type)
        _min_colors = np.min(colors, axis=0)
        min_colors.append(_min_colors)
        min_colors_names.append(data_name + "_" + panel_type)
        _max_colors = np.max(colors, axis=0)
        if _max_colors[0] > target_max_color[0] or _min_colors[0] < target_min_color[0]:
            oversize_list.append(texture_file)
        if _max_colors[1] > target_max_color[1] or _min_colors[1] < target_min_color[1]:
            oversize_list.append(texture_file)
        if _max_colors[2] > target_max_color[2] or _min_colors[2] < target_min_color[2]:
            oversize_list.append(texture_file)
        max_colors.append(_max_colors)
        max_colors_names.append(data_name + "_" + panel_type)
if os.path.exists(os.path.join(out_folder, "stats.npz")):
    stats = np.load(os.path.join(out_folder, "stats.npz"))
    max_verts = np.concatenate([stats["max_verts"], np.stack(max_verts)], axis=0)
    min_verts = np.concatenate([stats["min_verts"], np.stack(min_verts)], axis=0)
    max_colors = np.concatenate([stats["max_colors"], np.stack(max_colors)], axis=0)
    min_colors = np.concatenate([stats["min_colors"], np.stack(min_colors)], axis=0)
    max_verts_names = np.concatenate([stats["max_verts_names"], np.stack(max_verts_names)], axis=0)
    min_verts_names = np.concatenate([stats["min_verts_names"], np.stack(min_verts_names)], axis=0)
    max_colors_names = np.concatenate([stats["max_colors_names"], np.stack(max_colors_names)], axis=0)
    min_colors_names = np.concatenate([stats["min_colors_names"], np.stack(min_colors_names)], axis=0)
else:
    max_verts_names = np.stack(max_verts_names)
    min_verts_names = np.stack(min_verts_names)
    max_colors_names = np.stack(max_colors_names)
    min_colors_names = np.stack(min_colors_names)
    max_verts = np.stack(max_verts)
    min_verts = np.stack(min_verts)
    max_colors = np.stack(max_colors)
    min_colors = np.stack(min_colors)
print(max_verts.shape, min_verts.shape, max_colors.shape, min_colors.shape)
print(max_verts_names.shape, min_verts_names.shape, max_colors_names.shape, min_colors_names.shape)
print(len(has_overlap_list), len(visited_list), len(oversize_list))
np.savez_compressed(os.path.join(out_folder, "stats.npz"), 
                    max_verts=max_verts, min_verts=min_verts, max_colors=max_colors, min_colors=min_colors,
                    max_verts_names=max_verts_names, min_verts_names=min_verts_names, max_colors_names=max_colors_names, min_colors_names=min_colors_names)
with open(os.path.join(out_folder, "has_overlap_list.txt"), "w") as f:
    for item in has_overlap_list:
        f.write(item + "\n")
with open(os.path.join(out_folder, "visited_list.txt"), "w") as f:
    for item in visited_list:
        f.write(item + "\n")
with open(os.path.join(out_folder, "oversize_list.txt"), "w") as f:
    for item in oversize_list:
        f.write(item + "\n")


