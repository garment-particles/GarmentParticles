import os
import argparse
from pathlib import Path
import trimesh
import numpy as np
from pygarment.meshgen.boxmeshgen import BoxMesh
from pygarment.meshgen.simulation import run_sim
import pygarment.data_config as data_config
from pygarment.meshgen.sim_config import PathCofig
import glob
from pygarment.meshgen.pattern_packing import (
    INDIVIDUAL,
    pack_pattern_panels,
)
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import h5py


def get_command_args():
    """command line arguments to control the run"""
    # https://stackoverflow.com/questions/40001892/reading-named-command-arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_path', '-o', 
        help='output path', 
        type=str, 
        default='/miele/data/gcd/gcdv2_texture_rerun/')
    parser.add_argument(
        '--file_list', '-f', 
        help='file list', 
        type=str, 
        default='all_pattern_list.txt')
    parser.add_argument(
        '--is_orion', '-i', 
        help='is orion', 
        type=bool, 
        default=False)


    args = parser.parse_args()
    print('Commandline arguments: ', args)

    return args


def get_garment_texture(pattern_folder, out_path):
    garment_name = pattern_folder.split('/')[-1]
    if os.path.exists(os.path.join(out_path, f'{garment_name}_texture.hdf5')):
        return False
    pattern_spec = Path(pattern_folder) / f'{garment_name}_specification.json'
    box_mesh = Path(pattern_folder) / f'{garment_name}_boxmesh.ply'
    box_mesh = trimesh.load_mesh(box_mesh)
    sim_mesh = Path(pattern_folder) / f'{garment_name}_sim.ply'
    sim_mesh = trimesh.load_mesh(sim_mesh)
    

    garment_box_mesh = BoxMesh(pattern_spec, 1.0)
    garment_box_mesh.load()
    front_islands = [
        island
        for island in garment_box_mesh.vertex_texture
        if garment_box_mesh.is_front(island["panel_name"])
    ]
    back_islands = [
        island
        for island in garment_box_mesh.vertex_texture
        if garment_box_mesh.is_back(island["panel_name"])
    ]
    uv_dict = {}
    has_overlap = False
    for islands in (front_islands, back_islands):
        panel_names = [island['panel_name'] for island in islands]
        packing_result = pack_pattern_panels(
            garment_box_mesh.get_packing_panels(panel_names),
            garment_box_mesh.panel_tree,
            padding=1,
            strategy=INDIVIDUAL,
        )
        has_overlap = has_overlap or packing_result.has_overlap
        for island in islands:
            panel_name = island['panel_name']
            uv_dict[panel_name] = packing_result.panel_vertices[panel_name]
    for vertex_texture in garment_box_mesh.vertex_texture:
        panel_name = vertex_texture['panel_name']
        vertex_texture['uv'] = uv_dict[panel_name]
    panels = {}
    start_face = 0
    for i, panel_dict in enumerate(garment_box_mesh.vertex_texture):
        _panel_dict = {}
        _front_colors = []
        _front_uvs = []
        _back_colors = []
        _back_uvs = []
        panel_name = panel_dict['panel_name']
        is_front = garment_box_mesh.is_front(panel_name)
        uvs_panel = panel_dict['uv']
        _panel_dict['vertices'] = uvs_panel
        _panel_dict['faces'] = panel_dict['face_texture_coords']
        face_colors = []
        faces = garment_box_mesh.faces[start_face:start_face + len(panel_dict['face_texture_coords'])]
        start_face = start_face + len(panel_dict['face_texture_coords'])
        for face in faces:
            id0, id1, id2 = face
            verts = np.array([garment_box_mesh.vertices[id0], garment_box_mesh.vertices[id1], garment_box_mesh.vertices[id2]])
            dist = np.linalg.norm(verts[None, :, :] - box_mesh.vertices[:, None, :], axis=-1)
            min_indices = np.argmin(dist, axis=0)
            sim_vert = sim_mesh.vertices[min_indices]
            face_colors.append(np.mean(sim_vert, axis=0))
        _panel_dict['colors'] = np.stack(face_colors)
        panel_name = "front_" + panel_name if is_front else "back_" + panel_name
        panels[panel_name] = _panel_dict
    panels['has_overlap'] = has_overlap
    # Save panels as HDF5 file
    with h5py.File(os.path.join(out_path, f'{garment_name}_texture.hdf5'), 'w') as f:
        for panel_name, panel_data in panels.items():
            if panel_name != 'has_overlap':
                panel_group = f.create_group(panel_name)
                for key, value in panel_data.items():
                    panel_group.create_dataset(key, data=value)
        f.create_dataset('has_overlap', data=has_overlap)
    print(f'Saved {garment_name} texture')
    return has_overlap
    
    
args = get_command_args()
with open(args.file_list, 'r') as f:
    pattern_folders = f.read().splitlines()
if args.is_orion:
    pattern_folders = [folder.replace('/miele/data/gcd/garmentcodedatav2/', '/orion/u/w4756677/garment/gcdv2/garmentcodedatav2/') for folder in pattern_folders]
out_path = args.out_path
error_list = []
os.makedirs(args.out_path, exist_ok=True)
def process_pattern(pattern_folder):
    try:
        has_overlap = get_garment_texture(pattern_folder, args.out_path)
        if has_overlap:
            error_list.append(pattern_folder)
    except Exception as e:
        print(f"Error processing {pattern_folder}:")
        error_list.append(pattern_folder)
# Use ThreadPoolExecutor to parallelize processing
with ThreadPoolExecutor(max_workers=16) as executor:
    # Submit all pattern folders for processing and wrap with tqdm
    futures = list(tqdm([executor.submit(process_pattern, folder) for folder in pattern_folders], 
                       total=len(pattern_folders),
                       desc="Processing patterns"))
    # Wait for all tasks to complete
    for future in futures:
        future.result()
with open(os.path.join(args.out_path, 'error_list.txt'), 'w') as f:
    for error in error_list:
        f.write(error + '\n')
print(f'Error list: {error_list}')
