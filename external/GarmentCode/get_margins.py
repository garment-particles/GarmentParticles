import numpy as np
import os 
from glob import glob
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
saved_folder = "/orion/u/w4756677/garment/gcdv2/gcdv2_images"
folders = glob(os.path.join(saved_folder, "*"))

def get_margins(mask: np.ndarray):
    # Ensure binary
    mask = mask.astype(bool)
    
    # Sum along rows and columns to find non-zero regions
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # Find top and bottom indices
    top = np.argmax(rows)
    bottom = len(rows) - 1 - np.argmax(rows[::-1])
    
    # Find left and right indices
    left = np.argmax(cols)
    right = len(cols) - 1 - np.argmax(cols[::-1])
    
    # Margins are distances from image borders
    top_margin = top
    bottom_margin = mask.shape[0] - 1 - bottom
    left_margin = left
    right_margin = mask.shape[1] - 1 - right
    
    return left_margin, right_margin, top_margin, bottom_margin

left_margins, right_margins, top_margins, bottom_margins = [], [], [], []
error_list = []
for folder in tqdm(folders):
    _left_margins, _right_margins, _top_margins, _bottom_margins = [], [], [], []
    if not os.path.exists(os.path.join(folder, "front_1024.png")) or not os.path.exists(os.path.join(folder, "back_1024.png")):
        error_list.append(folder)
        continue
    for img_file in ["front_1024.png", "back_1024.png"]:
        
        img = cv2.imread(os.path.join(folder, img_file), cv2.IMREAD_UNCHANGED)
        
        empty_mask = img[...,3] > 0.5
        left_margin, right_margin, top_margin, bottom_margin = get_margins(empty_mask)
        _left_margins.append(left_margin)
        _right_margins.append(right_margin)
        _top_margins.append(top_margin)
        _bottom_margins.append(bottom_margin)
    
    left_margins.append(min(_left_margins))
    right_margins.append(min(_right_margins))
    top_margins.append(min(_top_margins))
    bottom_margins.append(min(_bottom_margins))

left_margins = np.array(left_margins)
right_margins = np.array(right_margins)
top_margins = np.array(top_margins)
bottom_margins = np.array(bottom_margins)

with open("error_list.txt", "w") as f:
    for error in error_list:
        f.write(error + "\n")
# save the margins
np.savez("margins.npz", left_margins=left_margins, right_margins=right_margins, top_margins=top_margins, bottom_margins=bottom_margins)
print(f"Saved {len(left_margins)} margins")
print(f"Error list: {error_list}")
