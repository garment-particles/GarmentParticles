"""
Remote server for point cloud generation.
Receives text prompts and returns generated point clouds.

Usage:
    python remote_server.py --host 0.0.0.0 --port 5000

The server exposes:
    POST /generate  - Generate point cloud from text prompt
    GET  /health    - Health check endpoint
"""
import argparse
import numpy as np
from flask import Flask, request, jsonify, Response
import json
from omegaconf import OmegaConf
import torch
import hydra
from transformers import AutoTokenizer
from transport import Sampler
import torch.distributed.checkpoint as dcp
from functools import partial
import sys
sys.path.append("../external/PyTorchEMD")
from emd import earth_mover_distance
from utils.pyTorchChamferDistance.chamfer_distance import ChamferDistance
from datasets.garment_edge_dataset import get_optimal_stitching_pairs
from patterns.pattern_converter import NNSewingPattern
import matplotlib.pyplot as plt
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def one_way_chamfer(inp, target):
    dist_fn = ChamferDistance()
    dist1, dist2 = dist_fn(inp[None], target[None])
    return dist2.mean()

def chamfer(inp, target):
    dist_fn = ChamferDistance()
    dist1, dist2 = dist_fn(inp[None], target[None])
    return dist2.mean() + dist1.mean()


def torch_camera_from_angles(azimuth, elevation, radius, look_at=torch.zeros(3)):
    """
    Construct camera extrinsics from spherical coordinates (Y-up convention).
    
    azimuth: horizontal angle in degrees (0 = +Z axis, 90 = +X axis)
    elevation: vertical angle in degrees (0 = horizontal, 90 = top-down)
    radius: distance from look_at point
    look_at: [3,] point the camera looks at
    
    Returns: R [3,3], t [3,] (world-to-camera transform)
    """
    az = np.radians(azimuth)
    el = np.radians(elevation)
    
    # Camera position (Y-up spherical coordinates)
    cam_x = radius * np.cos(el) * np.sin(az)
    cam_y = radius * np.sin(el)
    cam_z = radius * np.cos(el) * np.cos(az)
    cam_pos = torch.tensor([cam_x, cam_y, cam_z]).to(look_at.device).float() + look_at
    
    # Camera basis vectors
    forward = look_at - cam_pos
    forward = forward / torch.norm(forward, p=2)
    
    world_up = torch.tensor([0, 1, 0]).to(look_at.device).float()  # Y-up
    
    right = torch.cross(forward, world_up)
    if torch.norm(right, p=2) < 1e-6:  # looking straight up/down
        right = torch.tensor([1, 0, 0]).to(look_at.device).float()
    right = right / torch.norm(right, p=2)
    
    up = torch.cross(right, forward)
    
    # World-to-camera rotation (camera looks down -Z)
    R = torch.stack([
        right,
        up,
        -forward
    ], dim=0)
    
    t = -R @ cam_pos
    
    return R, t


def torch_project_from_angles(points_3d, K, azimuth, elevation, radius, look_at=np.zeros(3)):
    """
    Project 3D points using orthographic projection (PyTorch version).
    
    Args:
        points_3d: (N, 3) tensor of 3D points
        K: Camera intrinsic matrix (not used in orthographic, kept for API compatibility)
        azimuth: Camera azimuth angle in degrees
        elevation: Camera elevation angle in degrees
        radius: Camera distance (not used in orthographic, kept for API compatibility)
        look_at: Point camera looks at
    
    Returns:
        points_2d: (N, 2) tensor of 2D projected points (in world coordinates)
        depths: (N,) tensor of depth values (z-coordinates in camera space)
    """
    R, t = torch_camera_from_angles(azimuth, elevation, radius, look_at)
    
    # Transform to camera space
    points_cam = (R @ points_3d.T).T + t
    depths = points_cam[:, 2]
    
    # Orthographic projection: just take x,y coordinates from camera space
    # Use world coordinates directly without scaling/offset
    # Negate Y to match screen coordinates (Y increases downward)
    
    return points_cam[:, :2], depths


def reconstruct(edge_data, normalization):

    attachment_type_dict_inv = {
        0: "lower_interface",
        1: "strapless_top",
        2: "left_collar",
        3: "right_collar",
    }


    panel_metadata = edge_data[:, :, 0, :8]
    panel_metadata = panel_metadata * np.array(normalization.transformation_std)[:8] + np.array(normalization.transformation_mean)[:8]
    panel_shifts = panel_metadata[:, :, :2]
    panel_translations = panel_metadata[:, :, 2:5]
    panel_rotations = panel_metadata[:, :, 5:8]
    edge_data = edge_data[:, :, 1:]
    valid_mask = edge_data[..., 7] > 0.5
    n_panels = [valid_mask[i].any(axis=-1).sum() for i in range(valid_mask.shape[0])]
    
    edge_data = edge_data * np.array(normalization.edge_std) + np.array(normalization.edge_mean)
    attachment_types = edge_data[..., 8:11] > 0.5
    attachment_types = np.sum(attachment_types.astype(int) * np.array([4, 2, 1]), axis=-1)
    stitch_flag = edge_data[..., 11] > 0.5
    stitch_tags = edge_data[..., 12:15]
    edge_data = edge_data[..., :7]
    

    N = edge_data.shape[0]
    patterns = []
    for i in range(N):
        # Extract RGB and UV coordinates
        valid_panel_mask = valid_mask[i].any(axis=-1)
        panel_valid_mask = valid_mask[i][valid_panel_mask]
        edges = edge_data[i][valid_panel_mask]
        rotations = panel_rotations[i][valid_panel_mask]
        translations = panel_translations[i][valid_panel_mask]
        panel_shift = panel_shifts[i][valid_panel_mask]
        edges = [edges[j][panel_valid_mask[j]] for j in range(panel_valid_mask.shape[0])]
        stitch_mask = stitch_flag[i][valid_panel_mask]
        stitch_tag = stitch_tags[i][valid_panel_mask]
        stitch_tag = [stitch_tag[j][panel_valid_mask[j]] for j in range(panel_valid_mask.shape[0])]
        stitch_mask = [stitch_mask[j][panel_valid_mask[j]] for j in range(panel_valid_mask.shape[0])]
        attachment_flag = attachment_types[i] >= 4
        attachment_flag = attachment_flag[valid_panel_mask]
        attachment_type = attachment_types[i][valid_panel_mask]
        attachment_type = [attachment_type[j][panel_valid_mask[j]] for j in range(panel_valid_mask.shape[0])]
        attachment_flag = [attachment_flag[j][panel_valid_mask[j]] for j in range(panel_valid_mask.shape[0])]
        
        
            
        pattern = NNSewingPattern()
        try:
            pattern.pattern_from_tensors(edges, panel_rotations=rotations, panel_translations=translations, padded=False, valid_mask=[np.ones(edges[j].shape[0]).astype(bool) for j in range(panel_valid_mask.shape[0])])
        except:
            pass
        for j, panel_attachment_flag in enumerate(attachment_flag):
            col_inds = np.where(panel_attachment_flag)[0]
            for k in col_inds:
                attachment_type_str = attachment_type_dict_inv[attachment_type[j][k] - 4]
                if f"panel_{j}" not in pattern.pattern["panels"] or k >= len(pattern.pattern["panels"][f"panel_{j}"]["edges"]):
                    import ipdb; ipdb.set_trace()
                pattern.pattern["panels"][f"panel_{j}"]["edges"][k]["label"] = attachment_type_str
                
        stitch_tag_dict = {}
        for j, panel_stitch_mask in enumerate(stitch_mask):
            col_inds = np.where(panel_stitch_mask)[0]
            for k in col_inds:
                stitch_tag_dict[(j, k)] = stitch_tag[j][k]
        stitching_pairs = get_optimal_stitching_pairs(stitch_tag_dict)
        for (j, k), (m, n) in stitching_pairs:
            pattern.pattern["stitches"].append([{"panel": f"panel_{j}", "edge": k.item()}, {"panel": f"panel_{m}", "edge": n.item()}])
            
        patterns.append(pattern)
            
    panel_data = {
        "preds": edge_data,
        "pred_stitches": stitch_flag,
        "pred_stitch_tags": stitch_tags,
        "pred_shifts": panel_shifts,
        "pred_mask": valid_mask,
        "pred_translations": panel_translations,
        "pred_rotations": panel_rotations,
    }
    
    return panel_data, patterns



def emd_distance(x, y, mask_x=None, mask_y=None):
    """
    Compute the Earth Mover's Distance (EMD) between two point clouds.
    
    Args:
        x: torch.Tensor, shape (N, 3)
        y: torch.Tensor, shape (M, 3)
        mask_x: torch.Tensor, shape (N,)
        mask_y: torch.Tensor, shape (M,)
    Returns:
        float: EMD distance
    """
    if mask_x is not None:
        x = x[mask_x]
    if mask_y is not None:
        y = y[mask_y]
    return earth_mover_distance(x, y, transpose=False)

def initialize_model():
    """
    Initialize the model.
    """
    cfg = OmegaConf.load("configs/pretrained/pgf_text.yaml")
    normalization = cfg.dataset.normalization
    limits = np.array([normalization.xlims, normalization.ylims])
    # Create model:
    model = hydra.utils.instantiate(cfg.model)
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.tokenizer_name)

    transport = hydra.utils.instantiate(cfg.transport)
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method=cfg.sample.sampling_method,
        num_steps=cfg.sample.num_sampling_steps,
        atol=cfg.sample.atol,
        rtol=cfg.sample.rtol,
        reverse=cfg.sample.reverse,
        timestep_shift=cfg.sample.timestep_shift,
        curve_sampling=cfg.sample.curve_sampling,
        stitch_sampling=cfg.sample.stitch_sampling
    )
    model.eval()
    ckpt_path = "checkpoints/pgf_text"
    model_sd = model.state_dict()
                    
    # Build nested structure matching AppState.state_dict() format
    state_dict = {
        "app": {
            "ema": {k: v.clone() for k, v in model_sd.items() if "text_encoder" not in k},
        }
    }

    # Load - DCP will automatically unshard
    dcp.load(
        state_dict, 
        checkpoint_id=ckpt_path,
        no_dist=True,  # Key: allows loading without distributed setup
    )
    missing_keys, unexpected_keys = model.load_state_dict(state_dict["app"]["ema"], strict=False)
    if len(missing_keys) > 0:
        print(f"Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys: {unexpected_keys}")


    edge_cfg = OmegaConf.load("configs/pretrained/edge.yaml")
    normalization = edge_cfg.dataset.normalization
    # Create model:
    edge_model = hydra.utils.instantiate(edge_cfg.model)
    for p in edge_model.parameters():
        p.requires_grad = False
    edge_model.to(device)
    edge_model.eval()
    ckpt_path = "checkpoints/edge"
    model_sd = edge_model.state_dict()
                    
    # Build nested structure matching AppState.state_dict() format
    state_dict = {
        "app": {
            "ema": {k: v.clone() for k, v in model_sd.items() if "text_encoder" not in k},
        }
    }

    # Load - DCP will automatically unshard
    dcp.load(
        state_dict, 
        checkpoint_id=ckpt_path,
        no_dist=True,  # Key: allows loading without distributed setup
    )
    missing_keys, unexpected_keys = edge_model.load_state_dict(state_dict["app"]["ema"], strict=False)
    if len(missing_keys) > 0:
        print(f"Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys: {unexpected_keys}")

    return model, tokenizer, sample_fn, limits, normalization, edge_model

model, tokenizer, sample_fn, limits, normalization, edge_model = initialize_model()

def _generate_pointcloud_from_prompt(prompt, num_points, model, tokenizer, sample_fn, limits, normalization) -> np.ndarray:
    """
    Generate a point cloud based on the text prompt.
    
    This is a placeholder implementation that generates random points.
    Replace this with your actual generation model.
    
    Args:
        prompt: Text description for generation
        num_points: Number of points to generate
    Returns:
        numpy array of shape (N, 3) with point coordinates
    """
    print(f"Generating point cloud for prompt: {prompt} with {num_points} points")
    
    prompt = prompt.strip()
    encoding = tokenizer(prompt, return_tensors="pt", max_length=77, padding="max_length", truncation=True)
    tokens = encoding["input_ids"]
    attn_mask = encoding["attention_mask"].bool()
    x = torch.randn(1, num_points, 6).to(device).float()
    data_dict = {
        "text_tokens": tokens.to(device),
        "text_attn_mask": attn_mask.to(device),
        "mask": torch.ones_like(x[..., 0]).bool(),
    }
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_dict = sample_fn(x, model.forward_with_mask, **data_dict)
    pred = output_dict[-1].float()
    pred = pred[0].detach().cpu().numpy()
    pred = (pred * np.array(normalization.pcd_std) + np.array(normalization.pcd_mean))
    pred[..., :2] = pred[..., :2] * (limits[:, 1] - limits[:, 0]) + limits[:, 0]
    return pred


generate_pointcloud_from_prompt = partial(_generate_pointcloud_from_prompt, model=model, tokenizer=tokenizer, sample_fn=sample_fn, limits=limits, normalization=normalization)

def _generate_from_silouette(prompt, guide, early_stop_t, lr, n_opt_steps, num_points, num_timesteps, K, azimuth, elevation, radius, look_at, model, tokenizer, sample_fn, limits, normalization) -> np.ndarray:
    print(f"Generating point cloud for prompt: {prompt} with {num_points} points with guide")
    print(f"early_stop_t: {early_stop_t}, n_opt_steps: {n_opt_steps}, lr: {lr}, num_timesteps: {num_timesteps}")


    pcd_std = torch.tensor(normalization.pcd_std).to(device)
    pcd_mean = torch.tensor(normalization.pcd_mean).to(device)
    
    # Convert guide to tensor
    guide_world = torch.from_numpy(guide).to(device).float()  # (N, 2) world coordinates
    
    K = torch.from_numpy(K).to(device).float()
    look_at = torch.from_numpy(look_at).to(device).float()
    
    # Compute transformed mean and std after projection
    # Project the 3D mean point to get the 2D mean
    mean_3d = pcd_mean[2:5].unsqueeze(0)  # (1, 3)
    mean_2d_world, _ = torch_project_from_angles(
        mean_3d, K,
        azimuth=azimuth,
        elevation=elevation,
        radius=radius,
        look_at=look_at
    )
    mean_2d = mean_2d_world[0]  # (2,)
    
    # Compute std by projecting points at mean ± std
    # This approximates how std transforms through the projection
    std_3d = pcd_std[2:5]
    
    # Sample points along each axis to estimate projected std
    points_x = torch.stack([mean_3d[0] + torch.tensor([std_3d[0], 0, 0]).to(device),
                            mean_3d[0] + torch.tensor([-std_3d[0], 0, 0]).to(device)], dim=0)
    points_y = torch.stack([mean_3d[0] + torch.tensor([0, std_3d[1], 0]).to(device),
                            mean_3d[0] + torch.tensor([0, -std_3d[1], 0]).to(device)], dim=0)
    points_z = torch.stack([mean_3d[0] + torch.tensor([0, 0, std_3d[2]]).to(device),
                            mean_3d[0] + torch.tensor([0, 0, -std_3d[2]]).to(device)], dim=0)
    
    proj_x, _ = torch_project_from_angles(points_x, K, azimuth, elevation, radius, look_at)
    proj_y, _ = torch_project_from_angles(points_y, K, azimuth, elevation, radius, look_at)
    proj_z, _ = torch_project_from_angles(points_z, K, azimuth, elevation, radius, look_at)
    
    # Compute std as the average deviation from mean after projection
    std_2d = torch.stack([
        (proj_x - mean_2d).abs().mean(dim=0),
        (proj_y - mean_2d).abs().mean(dim=0),
        (proj_z - mean_2d).abs().mean(dim=0)
    ], dim=0).max(dim=0)[0]  # Take max contribution from each 3D axis
    
    # Normalize the guide using transformed statistics
    guide_normalized = (guide_world - mean_2d) / std_2d
    guide_normalized = torch.cat([guide_normalized, torch.zeros_like(guide_normalized[:, :1])], dim=-1).float()

    

    prompt = prompt.strip()
    encoding = tokenizer(prompt, return_tensors="pt", max_length=77, padding="max_length", truncation=True)
    tokens = encoding["input_ids"]
    attn_mask = encoding["attention_mask"].bool()
    x = torch.randn(1, num_points, 6).to(device).float()
    data_dict = {
        "text_tokens": tokens.to(device),
        "text_attn_mask": attn_mask.to(device),
        "mask": torch.ones_like(x[..., 0]).bool(),
    }
    ts = torch.linspace(0, 1, num_timesteps).to(device)
    for t1, t2 in zip(ts[:-1], ts[1:]):
        t1 = torch.ones(x[0].size(0)).to(device) * t1 if isinstance(x, tuple) else torch.ones(x.size(0)).to(device) * t1
        t2 = torch.ones(x[0].size(0)).to(device) * t2 if isinstance(x, tuple) else torch.ones(x.size(0)).to(device) * t2
        
        # Forward pass without gradient
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            drift = model.forward_with_mask(x, t=t1, **data_dict)["pred"]
            x_1_t = x + drift * (1 - t1) # x_1 given x_t
            x_0_t = x - drift * t1
            x_1_t_hat = x_1_t.clone().detach()  # Detach before optimization
        
        if t1.item() < early_stop_t:
            # optimization step
            x_1_t_hat.requires_grad_(True)
            optimizer = torch.optim.Adam([x_1_t_hat], lr=lr)
            for n in range(n_opt_steps): # number of optimization steps can be adjusted. Higher leads to better guidance, but slower.
                # Project the normalized prediction to 2D world space
                inp_3d = x_1_t_hat[0][..., 2:5]
                inp_3d_world = (inp_3d * pcd_std[2:5]) + pcd_mean[2:5]
                inp_2d_world, _ = torch_project_from_angles(
                    inp_3d_world, K, 
                    azimuth=azimuth, 
                    elevation=elevation, 
                    radius=radius,
                    look_at=look_at
                )
                # Normalize using the transformed statistics (mean_2d, std_2d)
                inp_2d_normalized = (inp_2d_world - mean_2d) / std_2d
                inp_normalized = torch.cat([inp_2d_normalized, torch.zeros_like(inp_2d_normalized[:, :1])], dim=-1).float()
                
                loss = emd_distance(inp_normalized, guide_normalized)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("t: ", t1.item(), "n: ", n, "loss: ", loss.item(), "lr: ", optimizer.param_groups[0]["lr"])
            
            x_1_t_hat = x_1_t_hat.detach()  # Detach after optimization
            eps = torch.randn_like(x_0_t)
            x_0_t = torch.sqrt(1 - t2) * x_0_t + torch.sqrt(t2) * eps
        
        x = t2 * x_1_t_hat + (1 - t2) * x_0_t
    pred = x[0].detach().cpu().numpy()
    pred = (pred * np.array(normalization.pcd_std) + np.array(normalization.pcd_mean))
    pred[..., :2] = pred[..., :2] * (limits[1] - limits[0]) + limits[0]
    return pred

generate_from_silouette = partial(_generate_from_silouette, model=model, tokenizer=tokenizer, sample_fn=sample_fn, limits=limits, normalization=normalization)


def _generate_from_silouette_streaming(prompt, guide, early_stop_t, lr, n_opt_steps, num_points, num_timesteps, mode, K, azimuth, elevation, radius, look_at, model, tokenizer, sample_fn, limits, normalization):
    """Generator version that yields progress updates."""
    print(f"Generating point cloud for prompt: {prompt} with {num_points} points with guide (streaming)")
    print(f"early_stop_t: {early_stop_t}, n_opt_steps: {n_opt_steps}, lr: {lr}, num_timesteps: {num_timesteps}")
    print(f"K: {K}, azimuth: {azimuth}, elevation: {elevation}, radius: {radius}, look_at: {look_at}")

    pcd_std = torch.tensor(normalization.pcd_std).to(device)
    pcd_mean = torch.tensor(normalization.pcd_mean).to(device)
    
    # Convert guide to tensor
    guide_world = torch.from_numpy(guide).to(device).float()  # (N, 2) world coordinates
    
    K = torch.from_numpy(K).to(device).float()
    look_at = torch.from_numpy(look_at).to(device).float()
    
    # Compute transformed mean and std after projection
    # Project the 3D mean point to get the 2D mean
    mean_3d = pcd_mean[2:5].unsqueeze(0)  # (1, 3)
    mean_2d_world, _ = torch_project_from_angles(
        mean_3d, K,
        azimuth=azimuth,
        elevation=elevation,
        radius=radius,
        look_at=look_at
    )
    mean_2d = mean_2d_world[0]  # (2,)
    
    # Compute std by projecting points at mean ± std
    # This approximates how std transforms through the projection
    std_3d = pcd_std[2:5]
    
    # Sample points along each axis to estimate projected std
    points_x = torch.stack([mean_3d[0] + torch.tensor([std_3d[0], 0, 0]).to(device),
                            mean_3d[0] + torch.tensor([-std_3d[0], 0, 0]).to(device)], dim=0)
    points_y = torch.stack([mean_3d[0] + torch.tensor([0, std_3d[1], 0]).to(device),
                            mean_3d[0] + torch.tensor([0, -std_3d[1], 0]).to(device)], dim=0)
    points_z = torch.stack([mean_3d[0] + torch.tensor([0, 0, std_3d[2]]).to(device),
                            mean_3d[0] + torch.tensor([0, 0, -std_3d[2]]).to(device)], dim=0)
    
    proj_x, _ = torch_project_from_angles(points_x, K, azimuth, elevation, radius, look_at)
    proj_y, _ = torch_project_from_angles(points_y, K, azimuth, elevation, radius, look_at)
    proj_z, _ = torch_project_from_angles(points_z, K, azimuth, elevation, radius, look_at)
    
    # Compute std as the average deviation from mean after projection
    std_2d = torch.stack([
        (proj_x - mean_2d).abs().mean(dim=0),
        (proj_y - mean_2d).abs().mean(dim=0),
        (proj_z - mean_2d).abs().mean(dim=0)
    ], dim=0).max(dim=0)[0]  # Take max contribution from each 3D axis
    
    # Normalize the guide using transformed statistics
    guide_normalized = (guide_world - mean_2d) / std_2d
    guide_normalized = torch.cat([guide_normalized, torch.zeros_like(guide_normalized[:, :1])], dim=-1).float()

    if guide_normalized.shape[0] > num_points:
        guide_normalized = guide_normalized[torch.randperm(guide_normalized.shape[0])[:num_points].to(device)]


    prompt = prompt.strip()
    encoding = tokenizer(prompt, return_tensors="pt", max_length=77, padding="max_length", truncation=True)
    tokens = encoding["input_ids"]
    attn_mask = encoding["attention_mask"].bool()
    x = torch.randn(1, num_points, 6).to(device).float()
    data_dict = {
        "text_tokens": tokens.to(device),
        "text_attn_mask": attn_mask.to(device),
        "mask": torch.ones_like(x[..., 0]).bool(),
    }
    ts = torch.linspace(0, 1, num_timesteps).to(device)
    total_steps = len(ts) - 1
    
    for step_idx, (t1, t2) in enumerate(zip(ts[:-1], ts[1:])):
        t1 = torch.ones(x[0].size(0)).to(device) * t1 if isinstance(x, tuple) else torch.ones(x.size(0)).to(device) * t1
        t2 = torch.ones(x[0].size(0)).to(device) * t2 if isinstance(x, tuple) else torch.ones(x.size(0)).to(device) * t2
        
        loss_val = None
        
        # Forward pass without gradient
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            drift = model.forward_with_mask(x, t=t1, **data_dict)["pred"]
            x_1_t = x + drift * (1 - t1)
            x_0_t = x - drift * t1
            x_1_t_hat = x_1_t.clone().detach()  # Detach before optimization
        
        if t1.item() < early_stop_t and t1.item() > 0.3:
            # Optimization step with gradient enabled on detached tensor
            x_1_t_hat.requires_grad_(True)
            optimizer = torch.optim.Adam([x_1_t_hat], lr=lr)
            for n in range(n_opt_steps):
                # Project the normalized prediction to 2D world space
                inp_3d = x_1_t_hat[0][..., 2:5]
                inp_3d_world = (inp_3d * pcd_std[2:5]) + pcd_mean[2:5]
                inp_2d_world, _ = torch_project_from_angles(
                    inp_3d_world, K, 
                    azimuth=azimuth, 
                    elevation=elevation, 
                    radius=radius,
                    look_at=look_at
                )
                # Normalize using the transformed statistics (mean_2d, std_2d)
                inp_2d_normalized = (inp_2d_world - mean_2d) / std_2d
                inp_normalized = torch.cat([inp_2d_normalized, torch.zeros_like(inp_2d_normalized[:, :1])], dim=-1).float()
                
                if mode == "completion":
                    loss = one_way_chamfer(inp_normalized, guide_normalized)
                elif mode == "reconstruction":
                    loss = emd_distance(inp_normalized, guide_normalized)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val = loss.item()
            
            x_1_t_hat = x_1_t_hat.detach()  # Detach after optimization
            eps = torch.randn_like(x_0_t)
            x_0_t = torch.sqrt(1 - t2) * x_0_t + torch.sqrt(t2) * eps
        
        x = t2 * x_1_t_hat + (1 - t2) * x_0_t
        
        # Yield progress update
        progress = {
            "type": "progress",
            "step": step_idx + 1,
            "total_steps": total_steps,
            "t": t1.item(),
            "loss": loss_val,
            "percent": int(100 * (step_idx + 1) / total_steps)
        }
        yield json.dumps(progress) + "\n"
    
    pred = x[0].detach().cpu().numpy()
    pred = (pred * np.array(normalization.pcd_std) + np.array(normalization.pcd_mean))
    pred[..., :2] = pred[..., :2] * (limits[:, 1] - limits[:, 0]) + limits[:, 0]
    
    # Yield final result
    result = {
        "type": "result",
        "points": pred.tolist()
    }
    yield json.dumps(result) + "\n"

generate_from_silouette_streaming = partial(_generate_from_silouette_streaming, model=model, tokenizer=tokenizer, sample_fn=sample_fn, limits=limits, normalization=normalization)


def _generate_from_3d(prompt, guide, early_stop_t, lr, n_opt_steps, num_points, num_timesteps, mode, model, tokenizer, sample_fn, limits, normalization) -> np.ndarray:
    print(f"Generating point cloud for prompt: {prompt} with {num_points} points with guide")
    print(f"early_stop_t: {early_stop_t}, n_opt_steps: {n_opt_steps}, lr: {lr}")


    normalized_guide = (guide - np.array(normalization.pcd_mean)[2:5]) / np.array(normalization.pcd_std)[2:5]
    normalized_guide_3d = torch.from_numpy(normalized_guide).to(device).float()

    prompt = prompt.strip()
    encoding = tokenizer(prompt, return_tensors="pt", max_length=77, padding="max_length", truncation=True)
    tokens = encoding["input_ids"]
    attn_mask = encoding["attention_mask"].bool()
    x = torch.randn(1, num_points, 6).to(device).float()
    data_dict = {
        "text_tokens": tokens.to(device),
        "text_attn_mask": attn_mask.to(device),
        "mask": torch.ones_like(x[..., 0]).bool(),
    }
    ts = torch.linspace(0, 1, num_timesteps).to(device)
    for t1, t2 in zip(ts[:-1], ts[1:]):
        t1 = torch.ones(x[0].size(0)).to(device) * t1 if isinstance(x, tuple) else torch.ones(x.size(0)).to(device) * t1
        t2 = torch.ones(x[0].size(0)).to(device) * t2 if isinstance(x, tuple) else torch.ones(x.size(0)).to(device) * t2
        
        # Forward pass without gradient
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            drift = model.forward_with_mask(x, t=t1, **data_dict)["pred"]
            x_1_t = x + drift * (1 - t1) # x_1 given x_t
            x_0_t = x - drift * t1
            x_1_t_hat = x_1_t.clone().detach()  # Detach before optimization
        
        if t1.item() < early_stop_t:
            # optimization step
            x_1_t_hat.requires_grad_(True)
            optimizer = torch.optim.Adam([x_1_t_hat], lr=lr)
            for n in range(n_opt_steps): # number of optimization steps can be adjusted. Higher leads to better guidance, but slower.
                inp_3d = x_1_t_hat[0][..., 2:5]
                target = normalized_guide_3d
                if mode == "completion":
                    loss = one_way_chamfer(inp_3d, target)
                elif mode == "reconstruction":
                    loss = emd_distance(inp_3d, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("t: ", t1.item(), "n: ", n, "loss: ", loss.item(), "lr: ", optimizer.param_groups[0]["lr"])
            
            x_1_t_hat = x_1_t_hat.detach()  # Detach after optimization
            eps = torch.randn_like(x_0_t)
            x_0_t = torch.sqrt(1 - t2) * x_0_t + torch.sqrt(t2) * eps
        
        x = t2 * x_1_t_hat + (1 - t2) * x_0_t
    pred = x[0].detach().cpu().numpy()
    pred = (pred * np.array(normalization.pcd_std) + np.array(normalization.pcd_mean))
    pred[..., :2] = pred[..., :2] * (limits[:, 1] - limits[:, 0]) + limits[:, 0]
    return pred

generate_from_3d = partial(_generate_from_3d, model=model, tokenizer=tokenizer, sample_fn=sample_fn, limits=limits, normalization=normalization)


def _generate_from_pattern(prompt, guide, early_stop_t, lr, n_opt_steps, num_points, num_timesteps, mode, model, tokenizer, sample_fn, limits, normalization) -> np.ndarray:
    print(f"Generating point cloud for prompt: {prompt} with {num_points} points with pattern guide")
    print(f"early_stop_t: {early_stop_t}, n_opt_steps: {n_opt_steps}, lr: {lr}")


    guide[..., :2] = (guide[..., :2] - limits[:, 0]) / (limits[:, 1] - limits[:, 0])
    normalized_guide = (guide - np.array(normalization.pcd_mean)[:2]) / np.array(normalization.pcd_std)[:2]
    normalized_guide = torch.from_numpy(normalized_guide).to(device).float()
    normalized_guide = torch.cat([normalized_guide, torch.zeros_like(normalized_guide[:, :1])], dim=-1).float()

    prompt = prompt.strip()
    encoding = tokenizer(prompt, return_tensors="pt", max_length=77, padding="max_length", truncation=True)
    tokens = encoding["input_ids"]
    attn_mask = encoding["attention_mask"].bool()
    x = torch.randn(1, num_points, 6).to(device).float()
    data_dict = {
        "text_tokens": tokens.to(device),
        "text_attn_mask": attn_mask.to(device),
        "mask": torch.ones_like(x[..., 0]).bool(),
    }
    ts = torch.linspace(0, 1, num_timesteps).to(device)
    for t1, t2 in zip(ts[:-1], ts[1:]):
        t1 = torch.ones(x[0].size(0)).to(device) * t1 if isinstance(x, tuple) else torch.ones(x.size(0)).to(device) * t1
        t2 = torch.ones(x[0].size(0)).to(device) * t2 if isinstance(x, tuple) else torch.ones(x.size(0)).to(device) * t2
        
        # Forward pass without gradient
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            drift = model.forward_with_mask(x, t=t1, **data_dict)["pred"]
            x_1_t = x + drift * (1 - t1) # x_1 given x_t
            x_0_t = x - drift * t1
            x_1_t_hat = x_1_t.clone().detach()  # Detach before optimization
        
        if t1.item() < early_stop_t:
            # optimization step
            x_1_t_hat.requires_grad_(True)
            optimizer = torch.optim.Adam([x_1_t_hat], lr=lr)
            for n in range(n_opt_steps): # number of optimization steps can be adjusted. Higher leads to better guidance, but slower.
                inp_2d = x_1_t_hat[0][..., :2]
                inp_3d = torch.cat([inp_2d, torch.zeros_like(inp_2d[:, :1])], dim=-1).float()
                if mode == "completion":
                    loss = one_way_chamfer(inp_3d, normalized_guide)
                elif mode == "reconstruction":
                    loss = emd_distance(inp_3d, normalized_guide)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("t: ", t1.item(), "n: ", n, "loss: ", loss.item(), "lr: ", optimizer.param_groups[0]["lr"])
            
            x_1_t_hat = x_1_t_hat.detach()  # Detach after optimization
            eps = torch.randn_like(x_0_t)
            x_0_t = torch.sqrt(1 - t2) * x_0_t + torch.sqrt(t2) * eps
        
        x = t2 * x_1_t_hat + (1 - t2) * x_0_t
    pred = x[0].detach().cpu().numpy()
    pred = (pred * np.array(normalization.pcd_std) + np.array(normalization.pcd_mean))
    pred[..., :2] = pred[..., :2] * (limits[:, 1] - limits[:, 0]) + limits[:, 0]
    return pred

generate_from_pattern = partial(_generate_from_pattern, model=model, tokenizer=tokenizer, sample_fn=sample_fn, limits=limits, normalization=normalization)


def _point2pattern(pcd, edge_model, sample_fn, limits, normalization) -> np.ndarray:
    pcd[..., :2] = (pcd[..., :2] - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) 
    pcd = (pcd - np.array(normalization.pcd_mean)) / np.array(normalization.pcd_std)
    pcd = torch.from_numpy(pcd).to(device).float()

    x = torch.randn(1, edge_model.n_curves*edge_model.n_panels, edge_model.in_channels).to(device).float()
    data_dict = {
        "panel_points": pcd,
        "panel_points_mask": torch.ones_like(pcd[..., 0]).bool()
    }
    print(x.shape, pcd.shape)
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_dict = sample_fn(x, edge_model.forward, **data_dict)
    pred = output_dict[-1].float()
    pred = pred.detach().cpu().numpy()
    pred = pred.reshape(1, edge_model.n_panels, edge_model.n_curves, edge_model.in_channels)
    panel_data, patterns = reconstruct(pred, normalization)

    return panel_data, patterns[0]

point2pattern = partial(_point2pattern, edge_model=edge_model, sample_fn=sample_fn, limits=limits, normalization=normalization)


@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate point cloud from text prompt.
    
    Request JSON:
        {"prompt": "your text description"}
        
    Response JSON:
        {"points": [[x1,y1,z1], [x2,y2,z2], ...]}
        or
        {"error": "error message"}
    """
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400
            
        prompt = data['prompt']
        
        if not prompt or not prompt.strip():
            prompt = ""
        
        if "num_points" not in data:
            return jsonify({"error": "Missing 'num_points' in request"}), 400 
        
        num_points = data["num_points"]
        if not isinstance(num_points, int) or num_points < 100 or num_points > 8192:
            return jsonify({"error": "num_points must be between 100 and 8192"}), 400
        
        # Generate point clouds
        points = generate_pointcloud_from_prompt(prompt, num_points)
        
        # Convert to list for JSON serialization
        points_list = points.tolist()
        
        return jsonify({"points": points_list})
        
    except Exception as e:
        print(f"Error in /generate: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict_with_silouette', methods=['POST'])
def predict_with_silouette():
    """
    Generate point cloud from text prompt.
    
    Request JSON:
        {"prompt": "your text description"}
        
    Response JSON:
        {"points": [[x1,y1,z1], [x2,y2,z2], ...]}
        or
        {"error": "error message"}
    """
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400
            
        prompt = data['prompt']
        
        if not prompt or not prompt.strip():
            prompt = ""


        if "guide" not in data:
            return jsonify({"error": "Missing 'guide' in request"}), 400

        guide = data["guide"]
        if not isinstance(guide, list) or any([len(v) != 2 for v in guide]):
            return jsonify({"error": "guide must be a list of 2d vertices"}), 400

        guide = np.array(guide)

        
        if "num_points" not in data:
            return jsonify({"error": "Missing 'num_points' in request"}), 400 
        
        num_points = data["num_points"]
        if not isinstance(num_points, int) or num_points < 100 or num_points > 8192:
            return jsonify({"error": "num_points must be between 100 and 8192"}), 400

        if "lr" not in data:
            return jsonify({"error": "Missing 'lr' in request"}), 400 

        lr = data["lr"]
        if not isinstance(lr, float) or lr <= 0 or lr > 10:
            return jsonify({"error": "lr must be between 0 and 10"}), 400

        if "early_stop_t" not in data:
            return jsonify({"error": "Missing 'early_stop_t' in request"}), 400 

        early_stop_t = data["early_stop_t"]
        if not isinstance(early_stop_t, float) or early_stop_t < 0 or early_stop_t > 1:
            return jsonify({"error": "early_stop_t must be between 0 and 1"}), 400


        if "n_opt_steps" not in data:
            return jsonify({"error": "Missing 'n_opt_steps' in request"}), 400 

        n_opt_steps = data["n_opt_steps"]
        if not isinstance(n_opt_steps, int) or n_opt_steps < 0 or n_opt_steps > 10:
            return jsonify({"error": "n_opt_steps must be between 0 and 10"}), 400

        if "num_timesteps" not in data:
            return jsonify({"error": "Missing 'num_timesteps' in request"}), 400 

        num_timesteps = data["num_timesteps"]
        if not isinstance(num_timesteps, int) or num_timesteps < 10 or num_timesteps > 1000:
            return jsonify({"error": "num_timesteps must be between 10 and 1000"}), 400


        if "mode" not in data:
            return jsonify({"error": "Missing 'mode' in request"}), 400 

        mode = data["mode"]
        if not isinstance(mode, str) or mode not in ["completion", "reconstruction"]:
            return jsonify({"error": "mode must be either completion or reconstruction"}), 400

        # Camera params (optional; defaults match /predict_with_silouette_stream)
        K = np.array(data.get("K", [
            [1.0, 0, 250.0],
            [0, 1.0, 250.0],
            [0, 0, 1],
        ]), dtype=np.float32)
        azimuth = float(data.get("azimuth", 0.0))
        elevation = float(data.get("elevation", 0.0))
        radius = float(data.get("radius", 150.0))
        look_at = np.array(data.get("look_at", [0, 0, 0]), dtype=np.float32)

        # NOTE: _generate_from_silouette does not accept 'mode'; the validation above
        # is kept for parity with /predict_with_silouette_stream but the value is unused.
        points = generate_from_silouette(
            prompt, guide, early_stop_t, lr, n_opt_steps, num_points, num_timesteps,
            K, azimuth, elevation, radius, look_at,
        )

        # Convert to list for JSON serialization
        points_list = points.tolist()

        return jsonify({"points": points_list})

    except Exception as e:
        print(f"Error in /predict_with_silouette: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict_with_silouette_stream', methods=['POST'])
def predict_with_silouette_stream():
    """
    Generate point cloud with streaming progress updates.
    
    Request JSON: Same as /predict_with_silouette plus camera parameters
        
    Response: Server-Sent Events stream with JSON lines
        {"type": "progress", "step": 1, "total_steps": 100, "percent": 1, ...}
        {"type": "result", "points": [[x1,y1,z1], ...]}
    """
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400
            
        prompt = data['prompt']
        if not prompt or not prompt.strip():
            prompt = ""

        if "guide" not in data:
            return jsonify({"error": "Missing 'guide' in request"}), 400

        guide = data["guide"]
        if not isinstance(guide, list) or any([len(v) != 2 for v in guide]):
            return jsonify({"error": "guide must be a list of 2d vertices"}), 400

        guide = np.array(guide)

        if "num_points" not in data:
            return jsonify({"error": "Missing 'num_points' in request"}), 400 
        
        num_points = data["num_points"]
        if not isinstance(num_points, int) or num_points < 100 or num_points > 8192:
            return jsonify({"error": "num_points must be between 100 and 8192"}), 400

        if "lr" not in data:
            return jsonify({"error": "Missing 'lr' in request"}), 400 

        lr = data["lr"]
        if not isinstance(lr, float) or lr <= 0 or lr > 10:
            return jsonify({"error": "lr must be between 0 and 10"}), 400

        if "early_stop_t" not in data:
            return jsonify({"error": "Missing 'early_stop_t' in request"}), 400 

        early_stop_t = data["early_stop_t"]
        if not isinstance(early_stop_t, float) or early_stop_t < 0 or early_stop_t > 1:
            return jsonify({"error": "early_stop_t must be between 0 and 1"}), 400

        if "n_opt_steps" not in data:
            return jsonify({"error": "Missing 'n_opt_steps' in request"}), 400 

        n_opt_steps = data["n_opt_steps"]
        if not isinstance(n_opt_steps, int) or n_opt_steps < 0 or n_opt_steps > 10:
            return jsonify({"error": "n_opt_steps must be between 0 and 10"}), 400

        if "num_timesteps" not in data:
            return jsonify({"error": "Missing 'num_timesteps' in request"}), 400 

        num_timesteps = data["num_timesteps"]
        if not isinstance(num_timesteps, int) or num_timesteps < 10 or num_timesteps > 1000:
            return jsonify({"error": "num_timesteps must be between 10 and 1000"}), 400

        if "mode" not in data:
            return jsonify({"error": "Missing 'mode' in request"}), 400 

        mode = data["mode"]
        if not isinstance(mode, str) or mode not in ["completion", "reconstruction"]:
            return jsonify({"error": "mode must be either completion or reconstruction"}), 400

        # Camera parameters (with defaults for orthographic projection)
        K = np.array(data.get("K", [
            [1.0, 0, 250.0],      # scale_x, 0, offset_x
            [0, 1.0, 250.0],      # 0, scale_y, offset_y
            [0, 0, 1]
        ]), dtype=np.float32)
        
        azimuth = float(data.get("azimuth", 0.0))
        elevation = float(data.get("elevation", 0.0))
        radius = float(data.get("radius", 150.0))
        look_at = np.array(data.get("look_at", [0, 0, 0]), dtype=np.float32)

        def generate():
            try:
                for update in generate_from_silouette_streaming(
                    prompt, guide, early_stop_t, lr, n_opt_steps, num_points, num_timesteps, mode,
                    K, azimuth, elevation, radius, look_at
                ):
                    yield update
            except Exception as e:
                yield json.dumps({"type": "error", "error": str(e)}) + "\n"

        return Response(generate(), mimetype='application/x-ndjson')
        
    except Exception as e:
        print(f"Error in /predict_with_silouette_stream: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict_with_pattern', methods=['POST'])
def predict_with_pattern():
    """
    Generate point cloud from pattern guide (2D UV coordinates).
    
    Request JSON:
        {
            "prompt": "your text description",
            "guide": [[u1, v1], [u2, v2], ...],
            "num_points": 1024,
            "lr": 0.1,
            "early_stop_t": 0.7,
            "n_opt_steps": 2,
            "num_timesteps": 100
        }
        
    Response JSON:
        {"points": [[x1,y1,z1,u1,v1,c1], ...]}
        or
        {"error": "error message"}
    """
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400
            
        prompt = data['prompt']
        
        if not prompt or not prompt.strip():
            prompt = ""

        if "guide" not in data:
            return jsonify({"error": "Missing 'guide' in request"}), 400

        guide = data["guide"]
        if not isinstance(guide, list) or any([len(v) != 2 for v in guide]):
            return jsonify({"error": "guide must be a list of 2d vertices"}), 400

        guide = np.array(guide)

        if "num_points" not in data:
            return jsonify({"error": "Missing 'num_points' in request"}), 400 
        
        num_points = data["num_points"]
        if not isinstance(num_points, int) or num_points < 100 or num_points > 8192:
            return jsonify({"error": "num_points must be between 100 and 8192"}), 400

        if "lr" not in data:
            return jsonify({"error": "Missing 'lr' in request"}), 400 

        lr = data["lr"]
        if not isinstance(lr, float) or lr <= 0 or lr > 10:
            return jsonify({"error": "lr must be between 0 and 10"}), 400

        if "early_stop_t" not in data:
            return jsonify({"error": "Missing 'early_stop_t' in request"}), 400 

        early_stop_t = data["early_stop_t"]
        if not isinstance(early_stop_t, float) or early_stop_t < 0 or early_stop_t > 1:
            return jsonify({"error": "early_stop_t must be between 0 and 1"}), 400

        if "n_opt_steps" not in data:
            return jsonify({"error": "Missing 'n_opt_steps' in request"}), 400 

        n_opt_steps = data["n_opt_steps"]
        if not isinstance(n_opt_steps, int) or n_opt_steps < 0 or n_opt_steps > 10:
            return jsonify({"error": "n_opt_steps must be between 0 and 10"}), 400

        if "num_timesteps" not in data:
            return jsonify({"error": "Missing 'num_timesteps' in request"}), 400 

        num_timesteps = data["num_timesteps"]
        if not isinstance(num_timesteps, int) or num_timesteps < 10 or num_timesteps > 1000:
            return jsonify({"error": "num_timesteps must be between 10 and 1000"}), 400

        if "mode" not in data:
            return jsonify({"error": "Missing 'mode' in request"}), 400 

        mode = data["mode"]
        if not isinstance(mode, str) or mode not in ["completion", "reconstruction"]:
            return jsonify({"error": "mode must be either completion or reconstruction"}), 400

        # Generate point clouds from pattern
        points = generate_from_pattern(prompt, guide, early_stop_t, lr, n_opt_steps, num_points, num_timesteps, mode)
        
        # Convert to list for JSON serialization
        points_list = points.tolist()
        
        return jsonify({"points": points_list})
        
    except Exception as e:
        print(f"Error in /predict_with_pattern: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


def main():
    parser = argparse.ArgumentParser(description='Point Cloud Generation Server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=12345,
                        help='Port to listen on (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    print("Endpoints:")
    print(f"  POST http://{args.host}:{args.port}/generate")
    print(f"  GET  http://{args.host}:{args.port}/health")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
    # import trimesh 
    # tag = "edited_110_104"
    # pcd = trimesh.load(f"/scratch/m000051/george/InteractGarment/src/tools/{tag}.ply").vertices
    # print(pcd.shape)
    # guide = np.array(pcd)
    # prompt = "godet skirt, asymmetric top"
    # num_points = int(pcd.shape[-2] * 2)
    # lr = 0.02
    # n_opt_steps=20
    # early_stop_t = 0.8
    # num_timesteps=1000
    # mode="completion"
    # verts = generate_from_3d(prompt, guide, early_stop_t, lr, n_opt_steps, num_points, num_timesteps, mode)
    # import matplotlib.pyplot as plt 
    # plt.scatter(verts[:, 0], verts[:, 1], c=verts[:, -1], s=1)
    # plt.axis("equal")
    # plt.savefig(f"{tag}.png")
    # out_pcd = trimesh.points.PointCloud(verts[:, 2:5])
    # out_pcd.export(f"{tag}.ply")
    # np.save(f"{tag}.npy", verts)
    # verts = np.load(f"{tag}.npy")

    # _, pattern = point2pattern(verts[None])
    # pattern.serialize(tag)


