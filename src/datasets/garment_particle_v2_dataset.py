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

from datasets.garment_particle_dataset import GarmentParticleDataset


class GarmentParticleDatasetV2(GarmentParticleDataset):
    def __init__(
        self, 
        data_dir, 
        normalization,
        n_points=8192,
        multiplier=1.0, 
        split_file=None, 
        n_samples=None,
        fixed_n_points=True,
        collate_fn=None,
        ):
        super().__init__(data_dir, normalization, n_points, multiplier, split_file, n_samples, fixed_n_points, collate_fn=collate_fn)

