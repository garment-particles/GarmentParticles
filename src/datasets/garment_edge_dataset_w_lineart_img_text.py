"""
ImageNet Latent Dataset with safetensors.
"""

import os
import numpy as np
import json
from glob import glob
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset
from safetensors import safe_open
import matplotlib.cm as cm
from typing import List
import io
import h5py
import matplotlib.pyplot as plt
import datasets.pattern_utils as pat_utils
from datasets.garment_edge_dataset import GarmentEdgeDataset
import svgpathtools as svgpath
from scipy.spatial.transform import Rotation as R
from shapely.validation import make_valid
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment
from patterns.pattern_converter import NNSewingPattern
import random
from transformers import AutoImageProcessor, AutoTokenizer

class GarmentEdgeDatasetWithLineArt(GarmentEdgeDataset):
    def __init__(
        self, 
        garment_particle_dir, 
        garment_edge_dir,
        gcd_dir,
        gcd_list_file,
        split_file,
        image_root,
        text_path,
        normalization, 
        n_samples=None,
        n_points=8192,
        n_curves=37,
        n_panels=37,
        image_processor_name="facebook/dinov2-large",
        text_tokenizer_name="openai/clip-vit-large-patch14",
        front_only=False,
        front_back=False,
        text_max_length=None,
        use_all_captions=False,
        img_drop_prob=0.1,
        text_drop_prob=0.1,
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
        **kwargs
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
        self.image_root = image_root
        self.image_processor = AutoImageProcessor.from_pretrained(image_processor_name, use_fast=False)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name, use_fast=False)
        self.text_max_length = text_max_length
        self.use_all_captions = use_all_captions
        self.img_drop_prob = img_drop_prob
        self.text_drop_prob = text_drop_prob
        self.front_only = front_only
        self.front_back = front_back
        self.caption_dict = json.load(open(text_path, "r"))

    def __getitem__(self, idx):
        data_dict = super().__getitem__(idx)
        file = self.particle_files[idx]
        name = file.split("/")[-2]

        img_paths = [os.path.join(self.image_root, f"{name}_render_front.png"), os.path.join(self.image_root, f"{name}_render_back.png")]
        if not self.front_back:
            if self.front_only:
                img_paths = [img_paths[0]]
            else:
                img_paths = [random.choice(img_paths)]
        imgs = [Image.open(img_path) for img_path in img_paths]
        encodings = [self.image_processor(img, return_tensors="pt") for img in imgs]
        img_tokens = []
        for encoding in encodings:
            img_token = encoding["pixel_values"]
            if random.random() < self.img_drop_prob:
                img_token = torch.zeros_like(img_token)
            img_tokens.append(img_token)
        img_tokens = torch.cat(img_tokens, dim=0)
            
        caption_list = self.caption_dict[name]
        if not self.use_all_captions:
            caption_list = random.sample(caption_list, random.randint(1, len(caption_list)))
        caption = ", ".join(caption_list)
        if random.random() < self.text_drop_prob:
            caption = ""
        encoding = self.text_tokenizer(caption, max_length=self.text_max_length, padding="max_length", truncation=True, return_tensors="pt")
        text_tokens = encoding["input_ids"].squeeze(0)
        text_attn_mask = encoding["attention_mask"].squeeze(0).bool()

        data_dict.update({
            "pixel_values": img_tokens,
            "text_tokens": text_tokens,
            "text_attn_mask": text_attn_mask,
            "image_path": ":".join(img_paths)
        })
        return data_dict

    def reconstruct(
        self, 
        pred, 
        panel_points=None, 
        input=None, 
        panel_points_mask=None, 
        panel_indices=None, 
        spec_path=None,
        pixel_values=None, 
        text_tokens=None, 
        text_attn_mask=None, 
        image_path=None, 
        **kwargs
        ):
        img, pcd_reconstructed, panel_datum, _, patterns, gt_patterns = super().reconstruct(pred, panel_points, input, panel_points_mask, panel_indices, spec_path)
        image_conds = []
        for paths in image_path:
            paths = paths.split(":")
            imgs = [np.array(Image.open(path)) for path in paths]
            imgs = np.concatenate(imgs, axis=1)
            image_conds.append(Image.fromarray(imgs))
        N = text_tokens.shape[0]
        captions = []
        for i in range(N):
            caption = self.text_tokenizer.decode(text_tokens[i], skip_special_tokens=True)
            captions.append(caption)
        return img, pcd_reconstructed, panel_datum, patterns, gt_patterns, captions, image_conds


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
    
    dataset = GarmentEdgeDataset(
        garment_particle_dir="/orion/u/w4756677/garment/gcdv2/garment_particles_v2.1_11182025",
        garment_edge_dir="/orion/u/w4756677/garment/gcdv2/panel_edge_vecs",
        gcd_dir="/orion/u/w4756677/garment/gcdv2/garmentcodedatav2",
        gcd_list_file="assets/gcd_list.txt",
        split_file="/orion/u/w4756677/garment/InteractGarment/src/assets/garment_particle_v2_filtered_11182025.txt",
        order_panels_by="3d",
        transformation_postfix="11182025",
        absolute_coords=False,
        normalization=normalization,
        n_curves=38,
        n_panels=37,
        jitter_pcd_scale=0,
        jitter_uv_scale=0,
        random_dropout_scale=0,
        predict_rigid_transformations=True,
        predict_attachment_type=True,
        predict_stitch_tags=True,
        predict_valid_mask=True,
    )
    
    print(len(dataset))
    data_dict = dataset[0]
    print(data_dict.keys())
    edges = data_dict["input"]
    print(edges.mean(axis=-1), edges.std(axis=-1))
    edges = edges.numpy()[None]
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
    for k in data_dict.keys():
        print(k, data_dict[k].shape)
        data_dict[k] = data_dict[k][None].numpy() if isinstance(data_dict[k], torch.Tensor) else data_dict[k][None]
    img, pts, panel_data, _, patterns = dataset.reconstruct(pred=edges, **data_dict)
    img[0].save("reconstruction.png")
    np.savez_compressed("reconstruction.npz", **panel_data)
    patterns[0].serialize("tmp")
    # all_pattern_verts_l2, all_pattern_iou, edge_acc, panel_acc = dataset.evaluate(x, **data_dict)
    # print(all_pattern_verts_l2, all_pattern_iou, edge_acc, panel_acc)
