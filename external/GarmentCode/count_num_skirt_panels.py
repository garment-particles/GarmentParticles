import numpy as np
import json 
import glob 
import os 
from tqdm import tqdm

gcd_path = "/miele/data/gcd/"
all_patterns = []
for i in range(20):
    all_patterns.extend(open(f"/orion/u/w4756677/garment/InteractGarment/external/GarmentCode/assets/pattern_lists/all_pattern_list_{i}.txt", "r").readlines())
all_patterns = [pattern.strip().replace("/miele/data/gcd/", "/orion/u/w4756677/garment/gcdv2/") for pattern in all_patterns]

def process_pattern(pattern_folder):
    pattern_name = os.path.basename(pattern_folder)
    pattern_file = os.path.join(pattern_folder, f"{pattern_name}_specification.json")
    if not os.path.exists(pattern_file):
        return None
    n_edge_list = []
    with open(pattern_file, "r") as f:
        pattern = json.load(f)
        for panel in pattern["pattern"]["panels"].values():
            n_edges = len(panel["edges"])
            n_edge_list.append(n_edges)
    return n_edge_list, pattern_file

def find_pattern():
    from multiprocessing import Pool, cpu_count
    
    n_edge_list = []
    pattern_files = []
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_pattern, all_patterns), total=len(all_patterns)):
            if result is not None:
                n_edge_list.append(result[0])
                pattern_files.append(result[1])
    with open("n_edge_list.json", "w") as f:
        for n_edges, pattern_file in zip(n_edge_list, pattern_files):
            n_edge_str = " ".join(map(str, n_edges))
            f.write(f"{pattern_file} {n_edge_str}\n")

find_pattern()