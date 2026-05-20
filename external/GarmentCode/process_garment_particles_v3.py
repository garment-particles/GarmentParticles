from pathlib import Path
import trimesh
import numpy as np
from pygarment.meshgen.boxmeshgen import BoxMesh, DegenerateTrianglesError
import igl 
from shapely.geometry import Polygon
from shapely import Point
from matplotlib import pyplot as plt
import torch 
import torch.nn.functional as F 
import torch.optim as optim 
from collections import defaultdict
from pygarment.vd_utils.constrained_delaunay_triangulation import constrained_delaunay_triangulation
from pygarment.vd_utils.bichromatic_separator import bichromatic_voronoi_separator, voronoi_cells
from pygarment.meshgen.render.texture_utils import unwarp_UV, unwarp_UV_hierarchical
from anytree import LevelOrderIter
import json
import logging
import traceback
import sys
import os
import igl
import h5py

def get_line_from_dir(line_dir, p):
    return lambda t: p + t * line_dir

def normalize(v):
    m, M = v.min(), v.max()
    return (v - m) / (M - m)

def dist_to_line(x, normal, point):
    """
    Corrected implementation
    x: [N, 2]
    normal: [N, K, 2] 
    point: [N, K, 2]
    return: [N, K]
    """
    # Compute dot product: (x - point) · normal for each x and each line
    dot_products = torch.sum((x.unsqueeze(1) - point) * normal, dim=-1)
    # Take absolute value and normalize by normal magnitude
    return torch.abs(dot_products) / torch.norm(normal, dim=-1)

def loss_fn(x, in_pts, out_pts):
    """
    x: [N, 2]
    in_pts: [M, 2]
    out_pts: [K, 2]
    """
    dist_to_in = torch.norm(x.unsqueeze(1) - in_pts, dim=2)
    matched_in_ids = torch.argmin(dist_to_in, dim=1)
    matched_in_pts = in_pts[matched_in_ids]
    
    normals = F.normalize(matched_in_pts.unsqueeze(1) - out_pts, dim=2) # [N, K, 2]
    mid_points = (matched_in_pts.unsqueeze(1) + out_pts) / 2 # [N, K, 2]
    dist_to_bisector = dist_to_line(x, normals, mid_points) # [N, K]
    
    loss = dist_to_bisector.amin(1)
    loss = loss.mean()
    return loss 

def sample_points_from_curve(verts, num_points):
    n_curves = verts.shape[0] - 1
    lengths = torch.norm(verts[1:] - verts[:-1], dim=1)
    total_length = torch.sum(lengths)
    lengths = lengths / total_length
    n_points_per_curve = (lengths * num_points).int()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ts_per_curve = [torch.linspace(0, 1, n_points_per_curve[i]).to(device) for i in range(n_curves)]
    points = torch.cat([(1 - ts_per_curve[i].unsqueeze(-1)) * verts[i] + ts_per_curve[i].unsqueeze(-1) * verts[i+1] for i in range(len(ts_per_curve))])
    return points

def optimize_particles(boundary_verts, inside_pts, outside_pts, thresh=0.2):
    try:
        # Check CUDA availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            logging.warning("CUDA not available, using CPU for optimization")
        
        # Validate input arrays
        if boundary_verts.size == 0 or inside_pts.size == 0 or outside_pts.size == 0:
            logging.error("Empty input arrays for optimization")
            return inside_pts, outside_pts
            
        boundary_verts_tensor = torch.from_numpy(boundary_verts).to(device)
        inside_pts_tensor = torch.from_numpy(inside_pts).to(device)
        outside_pts_tensor = torch.from_numpy(outside_pts).to(device)
        inside_pts_tensor.requires_grad = True
        outside_pts_tensor.requires_grad = True
        optimizer = optim.Adam([inside_pts_tensor, outside_pts_tensor], lr=0.001)
        
        n_points = 1000
        n_steps = 200
        for i in range(n_steps):
            try:
                x = sample_points_from_curve(boundary_verts_tensor, n_points)
                x = x.to(device)
                loss = loss_fn(x, inside_pts_tensor, outside_pts_tensor)
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"Invalid loss value at step {i}: {loss.item()}")
                    break
                if loss < thresh:
                    break
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except Exception as e:
                logging.error(f"Error in optimization step {i}: {e}")
                break
                
        print(f"Step: {n_steps}: Loss: {loss.item()}")
        return inside_pts_tensor.detach().cpu().numpy(), outside_pts_tensor.detach().cpu().numpy()
        
    except Exception as e:
        logging.error(f"Error in optimize_particles: {e}")
        # Return original points if optimization fails
        return inside_pts, outside_pts

def judge_orientation(panel_name, is_front):
    if "skirt_panel" in panel_name or "hood" in panel_name or is_front:
        return 1
    return -1

def categorize_failure(error_message, garment_name):
    """Categorize failure types for better statistics tracking"""
    error_lower = error_message.lower()
    
    if "degeneratetriangles" in error_lower or "degenerate triangles" in error_lower:
        return 'degenerate_triangles'
    elif "not found" in error_lower or "missing" in error_lower or "filenotfound" in error_lower:
        return 'missing_files'
    elif "boxmesh" in error_lower and ("error" in error_lower or "panel" in error_lower):
        return 'boxmesh_errors'
    elif any(keyword in error_lower for keyword in ['processing', 'optimization', 'particles', 'uv']):
        return 'processing_errors'
    else:
        return 'other_errors'

def process_garment_single(pattern_folder, out_folder, visualize=False):
    try:
        garment_name = pattern_folder.split('/')[-1]
        
        out_folder = Path(out_folder) / garment_name
        out_folder.mkdir(parents=True, exist_ok=True)
        
        pattern_spec = Path(pattern_folder) / f'{garment_name}_specification.json'
        
        box_mesh_path = Path(pattern_folder) / f'{garment_name}_boxmesh.ply'
        
        sim_mesh_path = Path(pattern_folder) / f'{garment_name}_sim.ply'
        
        box_mesh = trimesh.load_mesh(box_mesh_path)
        
        sim_mesh = trimesh.load_mesh(sim_mesh_path)
        
        try:
            garment_box_mesh = BoxMesh(pattern_spec, 2.5)
            garment_box_mesh.load()
            garment_box_mesh_fine = BoxMesh(pattern_spec, 1.0)
            garment_box_mesh_fine.load()
        except DegenerateTrianglesError as e:
            logging.error(f"DegenerateTrianglesError for {garment_name}: {e}")
            return False
        except Exception as e:
            logging.error(f"BoxMesh loading error for {garment_name}: {e}")
            return False
                
        

        final_interior_verts = {}
        final_boundary_pts = {}
        
        all_verts = []
        all_sim_verts = []
        all_faces = []
        face_offset = 0
        
        for i, (panel_name, panel) in enumerate(garment_box_mesh.panels.items()):
            fine_panel = garment_box_mesh_fine.panels[panel_name]
            
            fine_verts = fine_panel.panel_vertices
            fine_verts = np.array(fine_verts)
            fine_faces = fine_panel.panel_faces
            fine_faces = np.array(fine_faces)
            
            verts = panel.panel_vertices
            verts = np.array(verts)
            faces = panel.panel_faces
            faces = np.array(faces)
            all_verts.append(verts)
            all_faces.append(faces + face_offset)
            face_offset += len(verts)
            
            dist, closest_id, closest_pt = igl.point_mesh_squared_distance(
                np.concatenate([verts, verts[:, :1]*0], -1), 
                np.concatenate([fine_verts, fine_verts[:, :1]*0], -1), 
                fine_faces
            )
            
            matched_fine_faces = fine_faces[closest_id]
            matched_fine_verts = fine_verts[matched_fine_faces]
            matched_fine_verts = np.concatenate([
                matched_fine_verts, matched_fine_verts[..., :1]*0
            ], -1)
            bary_coords = igl.barycentric_coordinates(
                np.concatenate([verts, verts[:, :1]*0], -1),
                matched_fine_verts[:, 0], 
                matched_fine_verts[:, 1], 
                matched_fine_verts[:, 2]
            )
            
            box_mesh_matched_faces = np.array(garment_box_mesh_fine._get_glob_ids(fine_panel, matched_fine_faces.flatten())).reshape(matched_fine_faces.shape)
            box_mesh_matched_verts = np.array(garment_box_mesh_fine.vertices)[box_mesh_matched_faces].reshape(-1, 3)
            dist = np.linalg.norm(box_mesh_matched_verts[None, ] - box_mesh.vertices[:, None], axis=-1)
            min_indices = np.argmin(dist, axis=0).reshape(box_mesh_matched_faces.shape)
            fine_matched_sim_verts = sim_mesh.vertices[min_indices]
            coarse_sim_verts = np.sum(fine_matched_sim_verts * bary_coords[..., None], axis=1)
            all_sim_verts.append(coarse_sim_verts)
            
            
            boundary_ids = igl.boundary_loop(faces)
            verts -= verts.min(0)
            boundary_verts = verts[boundary_ids]
            interior_ids = np.setdiff1d(np.arange(len(verts)), boundary_ids)
            interior_verts = verts[interior_ids]
            interior_sim_verts = coarse_sim_verts[interior_ids]
            boundary_sim_verts = coarse_sim_verts[boundary_ids]
            final_boundary_pts[panel_name] = np.concatenate([boundary_verts, boundary_sim_verts], -1)
            final_interior_verts[panel_name] = np.concatenate([interior_verts, interior_sim_verts], -1)
            
        all_verts = np.concatenate(all_verts, axis=0)
        all_sim_verts = np.concatenate(all_sim_verts, axis=0)
        all_faces = np.concatenate(all_faces, axis=0)


        if visualize:
            coarse_panel_mesh = trimesh.Trimesh(np.concatenate([all_verts, np.zeros_like(all_verts)[:, :1]], -1), all_faces)
            coarse_sim_mesh = trimesh.Trimesh(all_sim_verts, all_faces)
            coarse_panel_mesh.export(out_folder / f'coarse_panel_mesh_{garment_name}.ply')
            coarse_sim_mesh.export(out_folder / f'coarse_sim_mesh_{garment_name}.ply')
            
        dataset = h5py.File(out_folder / f'garment_particles_{garment_name}.h5', 'w')
        for panel_name in garment_box_mesh.panels.keys():
            panel_dataset = dataset.create_group(panel_name)
            panel_dataset.create_dataset("boundary_verts", data=final_boundary_pts[panel_name])
            panel_dataset.create_dataset("interior_verts", data=final_interior_verts[panel_name])
        dataset.close()
        return True
        
    except Exception as e:
        logging.error(f"Unexpected error processing {garment_name}: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    from tqdm import tqdm
    import argparse
    import datetime
    
    # Setup logging
    log_filename = f"process_garment_particles_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern_list", type=str, default="/orion/u/w4756677/garment/InteractGarment/external/GarmentCode/assets/pattern_lists/all_pattern_list.txt")
    parser.add_argument("--output_dir", type=str, default="/orion/u/w4756677/garment/gcdv2/garment_particles_v3.1")
    parser.add_argument("--resume", action="store_true", help="Skip already processed garments")
    parser.add_argument("--vis", action="store_true", help="Visualize panels")
    args = parser.parse_args()
    
    with open(args.pattern_list, 'r') as f:
        pattern_folders = f.readlines()
        
    logging.info(f"Processing {len(pattern_folders)} garments")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    
    failed_folders = []
    processed_count = 0
    skipped_count = 0
    
    # Track failure types for better reporting
    failure_stats = {
        'degenerate_triangles': 0,
        'missing_files': 0,
        'boxmesh_errors': 0,
        'processing_errors': 0,
        'other_errors': 0
    }
    
    for i, pattern_folder in enumerate(tqdm(pattern_folders, desc="Processing garments")):
        pattern_folder = pattern_folder.strip().replace("/miele/data/gcd", "/orion/u/w4756677/garment/gcdv2")
        
        if not pattern_folder:
            continue
            
        garment_name = pattern_folder.split('/')[-1]
        if not garment_name:
            logging.warning(f"Invalid pattern folder path: {pattern_folder}")
            failed_folders.append(pattern_folder)
            continue
            
        # Check if already processed
        output_file = Path(args.output_dir) / garment_name / f'garment_particles_{garment_name}.npz'
        if args.resume and output_file.exists():
            logging.info(f"Skipping already processed garment: {garment_name}")
            skipped_count += 1
            continue
            
        try:
            logging.info(f"Processing ({i+1}/{len(pattern_folders)}): {garment_name}")
            success = process_garment_single(pattern_folder, args.output_dir, args.vis)
            if success:
                processed_count += 1
            else:
                failed_folders.append(pattern_folder)
                logging.warning(f"Failed to process: {garment_name}")
                failure_stats['processing_errors'] += 1
        except KeyboardInterrupt:
            logging.info("Processing interrupted by user")
            break
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Unexpected error processing {pattern_folder}: {error_msg}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            failed_folders.append(pattern_folder)
            
            # Categorize the failure
            failure_type = categorize_failure(error_msg, garment_name)
            failure_stats[failure_type] += 1
            continue
            
    # Save failed folders list
    failed_file = f"failed_folders_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        with open(failed_file, "w") as f:
            for folder in failed_folders:
                f.write(folder + "\n")
        logging.info(f"Failed folders list saved to: {failed_file}")
    except Exception as e:
        logging.error(f"Error saving failed folders list: {e}")
        
    # Summary
    logging.info(f"Processing complete!")
    logging.info(f"Total garments: {len(pattern_folders)}")
    logging.info(f"Successfully processed: {processed_count}")
    logging.info(f"Skipped (already processed): {skipped_count}")
    logging.info(f"Failed: {len(failed_folders)}")
    
    # Detailed failure breakdown
    if len(failed_folders) > 0:
        logging.info("Failure breakdown:")
        for failure_type, count in failure_stats.items():
            if count > 0:
                percentage = (count / len(failed_folders)) * 100
                logging.info(f"  {failure_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        # Specific advice for degenerate triangles
        if failure_stats['degenerate_triangles'] > 0:
            logging.info("Note: Degenerate triangle errors are common with certain garment meshes and typically cannot be fixed automatically.")
            
    logging.info(f"Log file: {log_filename}")
    
    if len(failed_folders) > 0:
        logging.warning(f"Some garments failed to process. Check {failed_file} for details.")
        sys.exit(1)
    else:
        logging.info("All garments processed successfully!")
        sys.exit(0)
