import numpy as np
import json 
import glob 
import os 
from tqdm import tqdm

gcd_path = "/miele/data/gcd/garmentcodedata/"
all_patterns = open("/home/w4756677/garment/AIpparel-Code/assets/data_configs/garmentcodedata_list.txt", "r").readlines()
all_patterns = [os.path.join(gcd_path, pattern.strip()) for pattern in all_patterns]


def find_pattern(panel_name):
    for pattern_folder in tqdm(all_patterns):
        pattern_name = os.path.basename(pattern_folder)
        pattern_file = os.path.join(pattern_folder, f"{pattern_name}_specification.json")
        with open(pattern_file, "r") as f: 
            pattern = json.load(f)
            panel_names =pattern["pattern"]["panel_order"]
            if panel_name in panel_names:
                return pattern_file
    return None

print(find_pattern("skirt_front_4"))