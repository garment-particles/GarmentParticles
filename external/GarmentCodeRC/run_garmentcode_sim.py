import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
import glob
# add the path of GarmentCode
from pygarment.meshgen.boxmeshgen import BoxMesh
from pygarment.meshgen.simulation import run_sim
import pygarment.data_config as data_config
from pygarment.meshgen.sim_config import PathCofig

def run_simultion_warp(pattern_spec, sim_config, output_path, easy_texture_path):
    
    props = data_config.Properties(sim_config) 
    props.set_section_stats('sim', fails={}, sim_time={}, spf={}, fin_frame={}, body_collisions={}, self_collisions={})
    props.set_section_stats('render', render_time={})

    spec_path = Path(pattern_spec)
    garment_name, _, _ = spec_path.stem.rpartition('_')  # assuming ending in '_specification'

    paths = PathCofig(
        in_element_path=spec_path.parent,  
        out_path=output_path, 
        in_name=garment_name,
        body_name= 'mean_all', #'mean_all_smplx',    # 'f_smpl_average_A40'
        smpl_body=False,   # NOTE: depends on chosen body model
        add_timestamp=False,
        system_path='system.json',
    )

    # Generate and save garment box mesh (if not existent)
    print(f"Generate box mesh of {garment_name} with resolution {props['sim']['config']['resolution_scale']}...")
    print('\nGarment load: ', paths.in_g_spec)

    garment_box_mesh = BoxMesh(paths.in_g_spec, props['sim']['config']['resolution_scale'])
    garment_box_mesh.load()
    # garment_box_mesh.vertices = [v + np.array([0, 14, 0]) for v in garment_box_mesh.vertices]
    garment_box_mesh.serialize(
        paths, store_panels=False, uv_config=props['render']['config']['uv_texture'])

    props.serialize(paths.element_sim_props)

    run_sim(
        garment_box_mesh.name, 
        props, 
        paths,
        save_v_norms=False,
        store_usd=False,  # NOTE: False for fast simulation!
        optimize_storage=False,   # props['sim']['config']['optimize_storage'],
        verbose=False
    )
    
    props.serialize(paths.element_sim_props)


parser = argparse.ArgumentParser()
parser.add_argument("--all_paths_json", type=str, default='', help="path to the save resules shapenet dataset")
parser.add_argument("--spec_folder", type=str, default='', help="path to the save resules shapenet dataset")
parser.add_argument("--json_spec_file", type=str, default='', help="path to the save resules shapenet dataset")
parser.add_argument("--easy_texture_path", type=str, default='', help="path to the save resules shapenet dataset")
parser.add_argument("--output_path", type=str, default=None, help="path to the save resules shapenet dataset")
args = parser.parse_args()

if args.all_paths_json.endswith('.json'):
    with open(args.all_paths_json) as f:
        garment_json_paths = json.load(f)
elif len(args.all_paths_json) > 1:
    garment_json_path = os.path.join(args.all_paths_json, 'vis_new/all_json_spec_files.json')

    with open(garment_json_path) as f:
        garment_json_paths = json.load(f)

elif args.json_spec_file:
    garment_json_paths = [args.json_spec_file]

if args.spec_folder:
    garment_json_paths = glob.glob(os.path.join(args.spec_folder, '*.json'), recursive=True)

print(len(garment_json_paths))
for json_spec_file in garment_json_paths:
    print(json_spec_file)
    json_spec_file = json_spec_file.replace('validate_garment', 'valid_garment')
    pattern_name = json_spec_file.split('/')[-2]
    saved_folder = os.path.dirname(json_spec_file) if args.output_path is None else os.path.join(args.output_path, pattern_name)
    os.makedirs(saved_folder, exist_ok=True)
    try:
        run_simultion_warp(
                json_spec_file,
                'assets/Sim_props/default_sim_props.yaml',
                    saved_folder,
                    easy_texture_path=args.easy_texture_path
                )
    except Exception as e:
        print(e)
        continue
