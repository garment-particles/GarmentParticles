"""
Training Codes of LightningDiT together with VA-VAE using Hydra for configuration management.
It envolves advanced training methods, sampling methods, 
architecture design methods, computation methods. We achieve
state-of-the-art FID 1.35 on ImageNet 256x256.

by Maple (Jingfeng Yao) from HUST-VL

DDP version for distributed sampling across multiple GPUs.
"""

import torch
import torch.distributed as dist
import torch.backends.cuda
import torch.backends.cudnn
import torch.distributed.checkpoint as dcp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import defaultdict
from copy import deepcopy
import time
import trimesh


import json
import numpy as np
import logging
import os
import matplotlib.cm as cm
import math
from collections import OrderedDict
import hydra
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
from tqdm import tqdm
import wandb as wb
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from transport import Sampler
from datasets.garment_edge_dataset_custom import CustomGarmentEdgeDataset
from utils.train_utils import check_checkpoint_compatibility
from utils.mm_logging import MMLogger


class IndexTrackingDataset(torch.utils.data.Dataset):
    """Wrapper dataset that returns the original dataset index along with data."""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.collate_fn = dataset.collate_fn
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if isinstance(data, dict):
            data["_dataset_idx"] = idx
        return data
    
    @property
    def text_tokenizer(self):
        return self.dataset.text_tokenizer
    

    def reconstruct(self, *args, **kwargs):
        # Remove our added key before passing to original reconstruct
        kwargs.pop("_dataset_idx", None)
        return self.dataset.reconstruct(*args, **kwargs)
    
    def evaluate(self, *args, **kwargs):
        # Remove our added key before passing to original evaluate
        kwargs.pop("_dataset_idx", None)
        return self.dataset.evaluate(*args, **kwargs)


def setup_distributed():
    """
    Initialize distributed process group.
    Returns rank, local_rank, and world_size.
    """
    # Check if we're in a distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        # SLURM environment
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        # Single GPU fallback
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is the main process (rank 0)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_rank():
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get world size."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def synchronize():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def gather_list_to_main(local_list, world_size, device):
    """
    Gather lists from all ranks to the main process.
    Returns the combined list on rank 0, empty list on other ranks.
    """
    if world_size == 1:
        return local_list
    
    # Convert list to tensor for gathering
    local_tensor = torch.tensor(local_list, dtype=torch.float32, device=device)
    local_size = torch.tensor([len(local_list)], dtype=torch.long, device=device)
    
    # Gather sizes first
    if is_main_process():
        all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    else:
        all_sizes = None
    dist.gather(local_size, all_sizes, dst=0)
    
    # Gather actual data
    if is_main_process():
        max_size = max(s.item() for s in all_sizes)
        # Pad local tensors to max size
        padded_tensors = [torch.zeros(max_size, dtype=torch.float32, device=device) for _ in range(world_size)]
    else:
        max_size = torch.tensor([0], dtype=torch.long, device=device)
        padded_tensors = None
    
    # Broadcast max_size to all ranks
    if is_main_process():
        max_size = torch.tensor([max_size], dtype=torch.long, device=device)
    else:
        max_size = torch.tensor([0], dtype=torch.long, device=device)
    dist.broadcast(max_size, src=0)
    
    # Pad local tensor
    padded_local = torch.zeros(max_size.item(), dtype=torch.float32, device=device)
    padded_local[:len(local_list)] = local_tensor
    
    dist.gather(padded_local, padded_tensors, dst=0)
    
    if is_main_process():
        # Unpad and combine
        combined = []
        for tensor, size in zip(padded_tensors, all_sizes):
            combined.extend(tensor[:size.item()].cpu().tolist())
        return combined
    return []


@hydra.main(version_base=None, config_path="../configs", config_name="lightningdit_xl_vavae_f16d32")
def do_sampling(cfg: DictConfig):
    """
    Samples from LightningDiT using Hydra configuration management with DDP support.
    """
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Enable TF32 for better performance on modern GPUs
    torch.backends.cuda.matmul.allow_tf32 = True

    # Setup an experiment folder:
    experiment_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(experiment_dir, f"{timestamp}.log")
    logger = MMLogger.get_instance('logger', log_file=log_file, distributed=world_size>1, log_level=cfg.train.log_level)
    logger.info(f"Experiment directory created at {experiment_dir}")
    logger.info(f"Running with {world_size} GPUs")
    
    # Setup dataset
    base_dataset = hydra.utils.instantiate(cfg.dataset, split_file=cfg.data.val_split)
    
    # Wrap dataset for index tracking
    dataset = IndexTrackingDataset(base_dataset)

    # Use DistributedSampler for multi-GPU sampling
    if world_size > 1:
        sampler_data = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
    else:
        sampler_data = None
    
    valid_loader = DataLoader(
        dataset,
        batch_size=cfg.eval.sample_per_batch,
        shuffle=False,  # DistributedSampler handles this
        sampler=sampler_data,
        num_workers=cfg.data.workers_per_gpu,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        drop_last=False
    )
    
    # Calculate samples per GPU for early stopping
    n_samples_total = cfg.eval.n_samples if cfg.eval.n_samples > 0 else len(dataset)
    max_samples_per_rank = math.ceil(n_samples_total / world_size)
    
    # Create model:
    model = hydra.utils.instantiate(cfg.model)
    requires_grad(model, False)

    edge_model = hydra.utils.instantiate(cfg.edge_model)
    requires_grad(edge_model, False)
    
    model = model.to(device)
    edge_model = edge_model.to(device)

    transport = hydra.utils.instantiate(cfg.transport)
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method=cfg.sample.sampling_method,
        num_steps=cfg.sample.num_sampling_steps,
        atol=cfg.sample.atol,
        rtol=cfg.sample.rtol,
        reverse=cfg.sample.reverse,
        timestep_shift=cfg.sample.timestep_shift,
    )

    # Prepare models for training:
    model.eval()
    edge_model.eval()
    
    # Synchronize before checkpoint loading
    synchronize()

    if cfg.pgf_weight_init is not None:
        logger.info("Loading GPF Weight Initialization")
        init_path = cfg.pgf_weight_init
        if is_main_process():
            logger.info(f"Loading pretrained weights from {init_path}")
        if os.path.exists(init_path) and os.path.isdir(init_path) and os.path.exists(os.path.join(init_path, ".metadata")):
            # DCP checkpoint (directory)
            missing_keys, unexpected_keys, common_keys = check_checkpoint_compatibility(init_path, model, app_name="app.ema")
            # Build nested structure matching AppState.state_dict() format
            model_sd = model.state_dict()
            state_dict = {
                "app": {
                    "ema": {k.replace("app.ema.", ""): model_sd[k.replace("app.ema.", "")].clone() for k in common_keys},
                }
            }
            
            # Load - DCP will automatically unshard
            dcp.load(
                state_dict, 
                checkpoint_id=init_path,
                no_dist=True,  # Key: allows loading without distributed setup
            )
            missing_keys, unexpected_keys = model.load_state_dict(state_dict["app"]["ema"], strict=False)
        elif os.path.exists(init_path) and os.path.isfile(init_path) and init_path.endswith(".pt"):
            missing_keys, unexpected_keys = model.load_state_dict(torch.load(init_path, map_location=device), strict=False)
            
        if is_main_process():
            if len(missing_keys) > 0:
                logger.warning(f"Loading Weight Init: Missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                logger.warning(f"Loading Weight Init: Unexpected keys: {unexpected_keys}")

    if cfg.gpf_ckpt is not None:
        logger.info("Loading PGF Weight Checkpoint")
        if os.path.exists(cfg.gpf_ckpt):
            checkpoint_path = cfg.gpf_ckpt
            if checkpoint_path.endswith('.pt'):
                # Legacy .pt checkpoint
                checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
                model_dict = checkpoint['ema']
                model_dict = {k.replace('module.', ''): v for k, v in model_dict.items()}
                model.load_state_dict(model_dict, strict=False)
            else:
                # DCP checkpoint (directory)
                missing_keys, unexpected_keys, common_keys = check_checkpoint_compatibility(checkpoint_path, model, app_name="app.ema")
                # Build nested structure matching AppState.state_dict() format
                model_sd = model.state_dict()
                state_dict = {
                    "app": {
                        "ema": {k.replace("app.ema.", ""): model_sd[k.replace("app.ema.", "")].clone() for k in common_keys},
                    }
                }

                # Load - DCP will automatically unshard
                dcp.load(
                    state_dict, 
                    checkpoint_id=checkpoint_path,
                    no_dist=True,  # Key: allows loading without distributed setup
                )
                missing_keys, unexpected_keys = model.load_state_dict(state_dict["app"]["ema"], strict=False)
                if is_main_process():
                    if len(missing_keys) > 0:
                        logger.warning(f"Loading PGF checkpoint: Missing keys: {missing_keys}")
                    if len(unexpected_keys) > 0:
                        logger.warning(f"Loading PGF checkpoint: Unexpected keys: {unexpected_keys}")
                train_steps = int(checkpoint_path.rstrip('/').split('/')[-1])
        else:
            logger.error("No checkpoint found. Provide a valid checkpoint path.")
            cleanup_distributed()
            exit()

    if cfg.edge_model_ckpt is not None:
        logger.info("Loading Edge Model Weight Checkpoint")
        if os.path.exists(cfg.edge_model_ckpt):
            checkpoint_path = cfg.edge_model_ckpt
            if checkpoint_path.endswith('.pt'):
                # Legacy .pt checkpoint
                checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
                model_dict = checkpoint['ema']
                model_dict = {k.replace('module.', ''): v for k, v in model_dict.items()}
                edge_model.load_state_dict(model_dict, strict=False)
            else:
                # DCP checkpoint (directory)
                model_sd = edge_model.state_dict()
                
                # Build nested structure matching AppState.state_dict() format
                state_dict = {
                    "app": {
                        "ema": {k: v.clone() for k, v in model_sd.items()},
                    }
                }
                
                # Load - DCP will automatically unshard
                dcp.load(
                    state_dict, 
                    checkpoint_id=checkpoint_path,
                    no_dist=True,  # Key: allows loading without distributed setup
                )
                missing_keys, unexpected_keys = edge_model.load_state_dict(state_dict["app"]["ema"], strict=False)
                if len(missing_keys) > 0:
                    logger.warning(f"Missing keys: {missing_keys}")
                if len(unexpected_keys) > 0:
                    logger.warning(f"Unexpected keys: {unexpected_keys}")
        else:
            logger.error("No checkpoint found. Provide a valid checkpoint path.")
            cleanup_distributed()
            exit()
    
    # Create save directory (only main process creates it, others wait)
    save_dir = os.path.join(experiment_dir, f"samples_{0:07d}")
    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)
    synchronize()  # Wait for directory creation
    
    # Create rank-specific subdirectory to avoid conflicts
    rank_save_dir = os.path.join(save_dir, f"rank_{rank:03d}")
    os.makedirs(rank_save_dir, exist_ok=True)
    
    all_panel_accs = []
    all_edge_accs = []
    all_stitch_accs = []
    all_pattern_ious = []
    
    total_samples = 0
    for batch_idx, data_dict in enumerate(valid_loader):
        # Early stopping if we've collected enough samples
        if total_samples >= max_samples_per_rank:
            break
        
        # Get the actual dataset indices for this batch (before any processing)
        dataset_indices = data_dict.get("_dataset_indices", None)
        
        logger.info(f"Rank {rank}: Sampling batch {batch_idx + 1}, samples so far: {total_samples}/{max_samples_per_rank}...")
        # import ipdb; ipdb.set_trace()
        first_stage_data_dict = {
            "pixel_values": data_dict["pixel_values"],
            "text_tokens": data_dict["text_tokens"],
            "text_attn_mask": data_dict["text_attn_mask"],
            "image_path": data_dict["image_path"],
            "mask": data_dict["panel_points_mask"]
        }
        z = torch.randn_like(data_dict["panel_points"]).to(device).float()

        first_stage_data_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in first_stage_data_dict.items()}
        n = z.shape[0]

        model_fn = model.forward_with_mask
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            samples = sample_fn(z, model_fn, **first_stage_data_dict)

        
        samples = samples[-1].float()
        data_dict["panel_points"] = samples

        data_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}
        edge_z = torch.randn_like(data_dict["input"]).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            samples = sample_fn(edge_z, edge_model.forward, **data_dict)
        stage2_end = time.perf_counter()
        
        samples = samples.cpu().numpy()[-1]
        data_dict = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}
        img, pcd, panel_datum, patterns, gt_patterns, captions, image_conds = dataset.reconstruct(samples, **data_dict)
        panel_datum["noise"] = z.cpu().numpy()
        panel_datum["edge_noise"] = edge_z.cpu().numpy()
        panel_datum["dataset_indices"] = np.array(dataset_indices) if dataset_indices is not None else np.arange(n)
        
        # Save using actual dataset indices
        for j, pattern in enumerate(patterns):
            sample_idx = dataset_indices[j] if dataset_indices else (total_samples + j)
            pattern.serialize(os.path.join(rank_save_dir, f"pattern_{sample_idx:05d}"), empty_ok=True)
        
        for j, pattern in enumerate(gt_patterns):
            sample_idx = dataset_indices[j] if dataset_indices else (total_samples + j)
            pattern.serialize(os.path.join(rank_save_dir, f"gt_pattern_{sample_idx:05d}"), empty_ok=True)
        for k, (img_item, pcd_item, caption, image_cond) in enumerate(zip(img, pcd, captions, image_conds)):
            # Use actual dataset index for filename
            sample_idx = dataset_indices[k] if dataset_indices is not None else (total_samples + k)
            if image_cond is not None:
                image_cond.save(os.path.join(rank_save_dir, f"image_cond_{sample_idx:05d}.png"))
            img_item.save(os.path.join(rank_save_dir, f"img_{sample_idx:05d}.png"))
            with open(os.path.join(rank_save_dir, f"caption_{sample_idx:05d}.txt"), "w") as f:
                f.write(caption)
            inside_flag = pcd_item[:, -1] < 0.5
            color = np.zeros_like(pcd_item[:, :4])
            color[inside_flag] = [0, 255, 0, 255]
            color[~inside_flag] = [255, 0, 0, 255]
            pcd_export = trimesh.PointCloud(pcd_item[:, :3], colors=color)
            pcd_export.export(os.path.join(rank_save_dir, f"pcd_{sample_idx:05d}.ply"))
        
        # Include dataset indices in panel_datum for reference
        start_idx = dataset_indices[0] if dataset_indices else total_samples
        end_idx = dataset_indices[-1] if dataset_indices else (total_samples + n - 1)
        np.savez_compressed(os.path.join(rank_save_dir, f"panel_data_{start_idx:05d}_{end_idx:05d}.npz"), **panel_datum)

        if cfg.eval.evaluate:
            panel_acc, edge_accs, stitch_accs, pattern_ious = dataset.evaluate(**panel_datum)
            all_panel_accs.extend(panel_acc)
            all_edge_accs.extend(edge_accs)
            all_stitch_accs.extend(stitch_accs)
            all_pattern_ious.extend(pattern_ious)

        
        total_samples += n
    
    # Synchronize all processes before aggregating results
    synchronize()
    
    # Gather evaluation metrics from all ranks
    if cfg.eval.evaluate:
        all_panel_accs = gather_list_to_main(all_panel_accs, world_size, device)
        all_edge_accs = gather_list_to_main(all_edge_accs, world_size, device)
        all_stitch_accs = gather_list_to_main(all_stitch_accs, world_size, device)
        all_pattern_ious = gather_list_to_main(all_pattern_ious, world_size, device)
        
        if is_main_process():
            logger.info(f"Panel accuracy: {np.mean(all_panel_accs):.4f}")
            logger.info(f"Edge accuracy: {np.mean(all_edge_accs):.4f}")
            logger.info(f"Stitch accuracy: {np.mean(all_stitch_accs):.4f}")
            logger.info(f"Pattern IOU: {np.mean(all_pattern_ious):.4f}")

    
    synchronize()
    logger.info("Done!")
    cleanup_distributed()


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


if __name__ == "__main__":
    do_sampling()