import numpy as np 
import h5py
import os, glob 
import tqdm
data_folder = "/orion/u/w4756677/garment/gcdv2/garment_particles_v3.1"
data_files = glob.glob(os.path.join(data_folder, "rand_*"))
count_dict = {}
for data_file in tqdm.tqdm(data_files, desc="Processing data files"):
    garment_name = data_file.split("/")[-1].split(".")[0]
    h5_file = os.path.join(data_file, f"garment_particles_{garment_name}.h5")
    if not os.path.exists(h5_file):
        continue
    h5_file = h5py.File(h5_file, 'r')
    n_points = 0
    n_panels = 0
    for panel_name in h5_file.keys():
        n_points += h5_file[panel_name]["boundary_verts"].shape[0]
        n_points += h5_file[panel_name]["interior_verts"].shape[0]
        n_panels += 1
    count_dict[garment_name] = (n_panels, n_points)
with open("total_count.txt", 'w') as f:
    for garment_name, (n_panels, n_points) in count_dict.items():
        f.write(f"{garment_name} {n_panels} {n_points}\n")