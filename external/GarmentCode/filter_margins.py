import numpy as np
import os 
from glob import glob
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
saved_folder = "/orion/u/w4756677/garment/gcdv2/gcdv2_images"
to_save_folder = "/orion/u/w4756677/garment/gcdv2/gcdv2_images_cutout"
os.makedirs(to_save_folder, exist_ok=True)
folders = glob(os.path.join(saved_folder, "*"))
target_left_margin = 200
target_right_margin = 200
target_top_margin = 300
target_bottom_margin = 100

for folder in tqdm(folders):
    if not os.path.exists(os.path.join(folder, "front_1024.png")) or not os.path.exists(os.path.join(folder, "back_1024.png")):
        error_list.append(folder)
        continue
    cutout = False
    cutout_imgs = []
    for img_file in ["front_1024.png", "back_1024.png"]:
        
        img = cv2.imread(os.path.join(folder, img_file), cv2.IMREAD_UNCHANGED)
        cutout_imgs.append(img[target_top_margin:-target_bottom_margin, target_left_margin:-target_right_margin, :])
        leftout_img = [img[:target_top_margin, :, 3], img[-target_bottom_margin:, :, 3], img[:, :target_left_margin, 3], img[:, -target_right_margin:, 3]]
        _cutout = any([np.sum(x) > 0 for x in leftout_img])
        cutout = _cutout or cutout
    if not cutout:
        os.makedirs(os.path.join(folder.replace(saved_folder, to_save_folder)), exist_ok=True)
        for img, img_file in zip(cutout_imgs, ["front_1024.png", "back_1024.png"]):
            cv2.imwrite(os.path.join(folder.replace(saved_folder, to_save_folder), img_file), img)
    
