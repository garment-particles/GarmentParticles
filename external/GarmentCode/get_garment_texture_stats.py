from pathlib import Path
import numpy as np
from pygarment.meshgen.boxmeshgen import BoxMesh
import pygarment.data_config as data_config
from pygarment.meshgen.sim_config import PathCofig
import glob
from tqdm import tqdm
import trimesh

if __name__ == "__main__":
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    def process_pattern(pattern_folder):
        garment_name = pattern_folder.split('/')[-1]
        boxmesh = Path(pattern_folder) / f'{garment_name}_sim.ply'
        boxmesh = trimesh.load_mesh(boxmesh)
        max_vert, min_vert = boxmesh.vertices.max(0), boxmesh.vertices.min(0)
        return max_vert, min_vert

    max_uv, min_uv = [], []
    pattern_folders = glob.glob("/miele/data/gcd/garmentcodedatav2/*/default_body/*")
    print(len(pattern_folders))
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_pattern = {
            executor.submit(process_pattern, folder): folder 
            for folder in pattern_folders
        }

        results = []
        for future in tqdm(as_completed(future_to_pattern), total=len(pattern_folders)):
            try:
                _max_vert, _min_vert = future.result()
                results.append((_max_vert, _min_vert))
            except Exception as e:
                print(f'Pattern {future_to_pattern[future]} generated an exception: {e}')
                
    max_uv = np.array([r[0] for r in results])
    min_uv = np.array([r[1] for r in results]) 
    print(f'max_uv: {max_uv.max(0)}, min_uv: {min_uv.min(0)}')