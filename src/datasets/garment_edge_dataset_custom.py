"""
ImageNet Latent Dataset with safetensors.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datasets.garment_edge_dataset import GarmentEdgeDataset
import torch

class CustomGarmentEdgeDataset(GarmentEdgeDataset):
    def __init__(
        self, 
        custom_sample_dir, 
        garment_particle_dir, 
        garment_edge_dir,
        gcd_dir,
        gcd_list_file,
        split_file,
        normalization, 
        n_samples=None,
        n_points=8192,
        n_curves=37,
        n_panels=37,
        jitter_pcd_scale=0,
        jitter_uv_scale=0,
        absolute_coords=False,
        uv_only=False,
        xyz_only=False,
        include_panel_indices=True,
        random_dropout_scale=0,
        order_panels_by="uv",
        predict_rigid_transformations=False,
        predict_attachment_type=False,
        predict_stitch_tags=False,
        predict_valid_mask=False,
        collate_fn=None,
        transformation_postfix="",
        **kwargs,
        ):
        super().__init__(
            garment_particle_dir=garment_particle_dir, 
            garment_edge_dir=garment_edge_dir,
            gcd_dir=gcd_dir,
            gcd_list_file=gcd_list_file,
            split_file=split_file,
            normalization=normalization, 
            n_samples=n_samples,
            n_points=n_points,
            n_curves=n_curves,
            n_panels=n_panels,
            jitter_pcd_scale=jitter_pcd_scale,
            jitter_uv_scale=jitter_uv_scale,
            absolute_coords=absolute_coords,
            uv_only=uv_only,
            xyz_only=xyz_only,
            include_panel_indices=include_panel_indices,
            random_dropout_scale=random_dropout_scale,
            order_panels_by=order_panels_by,
            predict_rigid_transformations=predict_rigid_transformations,
            predict_attachment_type=predict_attachment_type,
            predict_stitch_tags=predict_stitch_tags,
            predict_valid_mask=predict_valid_mask,
            collate_fn=collate_fn,
            transformation_postfix=transformation_postfix,
        )
        self.custom_sample_dir = custom_sample_dir
        assert os.path.exists(self.custom_sample_dir) and (self.custom_sample_dir.endswith(".npz") or self.custom_sample_dir.endswith(".npy")), "Custom sample directory must be a .npz or .npy file"
        self.samples = np.load(self.custom_sample_dir)
        if self.custom_sample_dir.endswith(".npz"):
            self.pcds = self.samples["preds"]
            self.pcd_masks = self.samples["mask"]
        else:
            self.pcds = self.samples[None]
            self.pcd_masks = np.ones_like(self.pcds[..., 0]).astype(bool)
        
    def __len__(self):
        return len(self.pcds)

    def __getitem__(self, idx):
        data_dict = super().__getitem__(idx)
        pcd = self.pcds[idx]
        pcd_mask = self.pcd_masks[idx]
        pcd[:,:2] = (pcd[:,:2] - self.limits[:, 0]) / (self.limits[:, 1] - self.limits[:, 0])
        pcd = (pcd - np.array(self.normalization.pcd_mean)) / np.array(self.normalization.pcd_std)
        
        pcd = torch.from_numpy(pcd).float()
        pcd_mask = torch.from_numpy(pcd_mask).bool()
        data_dict["panel_points"] = pcd
        data_dict["panel_points_mask"] = pcd_mask
        return data_dict

    
if __name__ == "__main__":
    from omegaconf import DictConfig
    import yaml
    import hydra
    normalization = DictConfig(yaml.load(open("/orion/u/w4756677/garment/InteractGarment/src/configs/dataset/normalization/garment_edges_relative_v2.2.yaml", "r"), Loader=yaml.FullLoader))
    pcd_normalization = DictConfig(yaml.load(open("/orion/u/w4756677/garment/InteractGarment/src/configs/dataset/normalization/garment_particles_v2.2.yaml", "r"), Loader=yaml.FullLoader))
    normalization.pcd_mean = pcd_normalization.pcd_mean
    normalization.pcd_std = pcd_normalization.pcd_std
    normalization.xlims = pcd_normalization.xlims
    normalization.ylims = pcd_normalization.ylims
    
    dataset = CustomGarmentEdgeDataset(
        custom_sample_dir="/orion/u/w4756677/garment/InteractGarment/src/img_data_000_032.npz",
        absolute_coords=False,
        normalization=normalization,
        n_curves=38,
        n_panels=37,
        jitter_pcd_scale=0.01,
        jitter_uv_scale=0.01,
        random_dropout_scale=0.2,
    )
    
    print(len(dataset))
    data_dict = dataset[0]
    print(data_dict.keys())
    edges = data_dict["input"]
    edges = edges.numpy()[None]
    del data_dict["input"]
    plt.scatter(data_dict["panel_points"][data_dict["panel_points_mask"]][:, 0], data_dict["panel_points"][data_dict["panel_points_mask"]][:, 1], color="green", s=0.5)
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("panel_points.png")
    plt.close()
    plt.scatter(data_dict["panel_points"][data_dict["panel_points_mask"]][:, 2], data_dict["panel_points"][data_dict["panel_points_mask"]][:, 3], color="black", s=0.5)
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("panel_points_3d.png")
    plt.close()
    # all_pattern_verts_l2, all_pattern_iou, edge_acc, panel_acc = dataset.evaluate(x, **data_dict)
    # print(all_pattern_verts_l2, all_pattern_iou, edge_acc, panel_acc)
