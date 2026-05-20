"""
ImageNet Latent Dataset with safetensors.
"""

import os
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset
from safetensors import safe_open
import io
import matplotlib.pyplot as plt
import h5py


def collate_fn(batch):
    pcds, labels = zip(*batch)
    max_n_points = max([pcd.shape[0] for pcd in pcds])
    pcd_tensor = np.zeros((len(pcds), max_n_points, 6), dtype=np.float32)
    mask = np.zeros((len(pcds), max_n_points), dtype=bool)
    for i, pcd in enumerate(pcds):
        pcd_tensor[i, :pcd.shape[0], :] = pcd
        mask[i, :pcd.shape[0]] = True
    pcd_tensor = torch.from_numpy(pcd_tensor)
    mask = torch.from_numpy(mask)
    labels = torch.from_numpy(np.array(labels).astype(int))
    return pcd_tensor, mask, labels
    

class GarmentParticleDataset(Dataset):
    def __init__(
        self, 
        data_dir, 
        normalization,
        n_points=8192,
        multiplier=1.0, 
        split_file=None, 
        n_samples=None,
        fixed_n_points=True,
        get_curve_inds=False,
        collate_fn=None):
        self.data_dir = data_dir
        self.normalization = normalization
        self.multiplier = multiplier
        self.split_file = split_file
        self.get_curve_inds = get_curve_inds

        if self.split_file is not None:
            data_names = open(self.split_file).readlines()
            data_names = [name.strip() for name in data_names]
            self.files = [os.path.join(data_dir, name).replace(".npz", ".h5") for name in data_names]
        else:
            self.files = sorted(glob(os.path.join(data_dir, "*", "*.h5")))
        self.limits = np.array([self.normalization.xlims, self.normalization.ylims])
        self.n_points = n_points
        self.fixed_n_points = fixed_n_points
        self.collate_fn = collate_fn


    def get_normalize_stats(self) -> tuple[np.ndarray, np.ndarray]:
        latent_stats_cache_file = os.path.join(self.data_dir, "garment_particle_normalize_stats.pt")
        if not os.path.exists(latent_stats_cache_file):
            latent_stats = self.compute_normalize_stats()
            torch.save(latent_stats, latent_stats_cache_file)
        else:
            latent_stats = torch.load(latent_stats_cache_file, weights_only=False)
        return latent_stats['mean'], latent_stats['std'], latent_stats['n_points']
    
    def get_front_back_pts(self, data):
        front = []
        back = []
        front_flag = []
        back_flag = []
        for panel_name, panel_data in data["front"].items():
            boundary_pts = panel_data['boundary_verts']
            interior_pts = panel_data['interior_verts']
            inside_flag = np.concatenate([np.ones_like(boundary_pts)[:, :1], np.zeros_like(interior_pts)[:, :1]], axis=0)
            pts = np.concatenate([boundary_pts, interior_pts], axis=0)
            pts[:, :2] = (pts[:, :2] - np.array(self.limits[:, 0])) / (self.limits[:, 1] - self.limits[:, 0])
            front.append(pts)
            front_flag.append(inside_flag)
        for panel_name, panel_data in data["back"].items():
            boundary_pts = panel_data['boundary_verts']
            interior_pts = panel_data['interior_verts']
            inside_flag = np.concatenate([np.ones_like(boundary_pts)[:, :1], np.zeros_like(interior_pts)[:, :1]], axis=0)
            pts = np.concatenate([boundary_pts, interior_pts], axis=0)
            pts[:, :2] = (pts[:, :2] - np.array(self.limits[:, 0])) / (self.limits[:, 1] - self.limits[:, 0])
            pts[:, 0] += 1
            back.append(pts)
            back_flag.append(inside_flag)
        front = np.concatenate(front, axis=0)
        back = np.concatenate(back, axis=0)
        front_flag = np.concatenate(front_flag, axis=0)
        back_flag = np.concatenate(back_flag, axis=0)
        return front, back, front_flag, back_flag
    
    def get_pts(self, data):
        front, back, front_flag, back_flag = self.get_front_back_pts(data)
        front = np.concatenate([front, front_flag], axis=1)
        back = np.concatenate([back, back_flag], axis=1)
        pcd = np.concatenate([front, back], 0)
        return pcd
    
    def compute_normalize_stats(self) -> dict[str, np.ndarray]:
        num_samples = min(10000, len(self.files))
        random_indices = np.random.choice(len(self.files), num_samples, replace=False)
        all_pts = []
        n_points = []
        for idx in tqdm(random_indices):
            file = self.files[idx]
            data = np.load(file)
            pts = self.get_pts(data)[:, :-1]
            n_points.append(pts.shape[0])
            all_pts.append(pts)
        all_pts = np.concatenate(all_pts, axis=0)
        mean = np.mean(all_pts, axis=0, keepdims=True)
        mean = np.concatenate([mean, np.zeros((1, 1))], -1)
        std = np.std(all_pts, axis=0, keepdims=True)
        std = np.concatenate([std, np.ones((1, 1))], -1)
        self.n_point_list = n_points
        self.n_points = max(n_points)
        
        normalize_stats = {'n_points': n_points, 'mean': mean, 'std': std}
        print(normalize_stats)
        print("num points:", self.n_points)
        return normalize_stats

    def __len__(self):
        return len(self.files)
    
    
    def __getitem__(self, idx):
        file = self.files[idx]
        data = h5py.File(file, "r")
        pcd = self.get_pts(data)
        pcd = (pcd - np.array(self.normalization.pcd_mean)) / np.array(self.normalization.pcd_std)
        n = pcd.shape[0]
        while pcd.shape[0] < self.n_points:
            pcd = np.concatenate([pcd, pcd], 0)
        pcd = pcd[:self.n_points]
        mask = None
        if not self.fixed_n_points:
            mask = np.ones(self.n_points).astype(bool)
            mask[n:] = False
        pcd = pcd * self.multiplier
        pcd = torch.from_numpy(pcd).float()
        label = np.array([0]) # Dummy label
        
        data_dict = {
            "input": pcd,
            "y": label
        }
        if mask is not None:
            data_dict["mask"] = torch.from_numpy(mask).bool()
        return data_dict
    
    def reconstruct(self, noise, mask=None, **kwargs):
        img_data = noise * np.array(self.normalization.pcd_std) + np.array(self.normalization.pcd_mean)
        img_data[..., :2] = img_data[..., :2] * (self.limits[:, 1] - self.limits[:, 0]) + self.limits[:, 0]
        N, T, _ = img_data.shape
        imgs = []
        pcds = []
        if mask is not None:
            mask = mask.reshape(N, T).astype(bool)
        for i in range(N):
            # Extract RGB and UV coordinates
            pts = img_data[i]
            if mask is not None:
                pts = pts[mask[i]]
            inside_flag = pts[:, -1] < 0.5
            inside_pts = pts[inside_flag]
            outside_pts = pts[~inside_flag]
            
            plt.scatter(inside_pts[:, 0], inside_pts[:, 1], s=0.2, color='green', label='inside')
            plt.scatter(outside_pts[:, 0], outside_pts[:, 1], s=0.2, color='red', label='outside')
            buf = io.BytesIO()
            plt.axis("equal")
            plt.legend()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            img = Image.open(buf)
            pcd = pts[:, 2:]
            pcd[:, -1] = pcd[:, -1] < 0.5
            imgs.append(img)
            pcds.append(pcd)
        img_data = {"preds": img_data}
        if mask is not None:
            img_data["mask"] = mask
        return imgs, pcds, img_data
    

        
    
    
