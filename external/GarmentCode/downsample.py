import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
saved_folder = "/miele/data/gcd/gcdv2_images_cutout"
to_save_folder = "/miele/data/gcd/gcdv2_images_cutout_64"
os.makedirs(to_save_folder, exist_ok=True)
folders = glob(os.path.join(saved_folder, "*"))

for folder in tqdm(folders):
    data_name = os.path.basename(folder)
    os.makedirs(os.path.join(to_save_folder, data_name), exist_ok=True)
    for img_file in ["front_1024.png", "back_1024.png"]:
        img = cv2.imread(os.path.join(folder, img_file), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA) / 255.0
        downsampled_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
        img_float32 = (downsampled_img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(to_save_folder, data_name, img_file), img_float32)