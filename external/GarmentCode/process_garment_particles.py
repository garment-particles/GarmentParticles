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
from pygarment.meshgen.pattern_packing import (
    FINE_TO_COARSE,
    HIERARCHICAL,
    INDIVIDUAL,
    JOINT_OPTIMIZATION,
    pack_pattern_panels,
)
import json
import logging
import traceback
import sys
import os
import igl


DEFAULT_PACKING_STRATEGY = FINE_TO_COARSE
DEFAULT_PACKING_PADDING = 3.0
DEFAULT_PACKING_MAX_ITERATIONS = 500
PACKING_STRATEGIES = (
    FINE_TO_COARSE,
    HIERARCHICAL,
    INDIVIDUAL,
    JOINT_OPTIMIZATION,
)

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

def process_garment_single(
    pattern_folder,
    out_folder,
    packing_strategy=DEFAULT_PACKING_STRATEGY,
    packing_padding=DEFAULT_PACKING_PADDING,
    packing_max_iterations=DEFAULT_PACKING_MAX_ITERATIONS,
):
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
            garment_box_mesh = BoxMesh(pattern_spec, 2.0)
            garment_box_mesh.load()
            garment_box_mesh_fine = BoxMesh(pattern_spec, 1.0)
            garment_box_mesh_fine.load()
        except DegenerateTrianglesError as e:
            logging.error(f"DegenerateTrianglesError for {garment_name}: {e}")
            return False
        except Exception as e:
            logging.error(f"BoxMesh loading error for {garment_name}: {e}")
            return False
                
        front_islands = [
            island
            for island in garment_box_mesh.vertex_texture
            if garment_box_mesh.is_front(island["panel_name"])
        ]
        back_islands = [
            island
            for island in garment_box_mesh.vertex_texture
            if garment_box_mesh.is_back(island["panel_name"])
        ]
        
        uv_dict = {}
        has_overlap = False
        eps = 0.5
        panel_offsets = {}
        
        packing_sides = {}
        for side, islands in (
            ("front", front_islands),
            ("back", back_islands),
        ):
            panel_names = [island['panel_name'] for island in islands]
            packing_result = pack_pattern_panels(
                garment_box_mesh.get_packing_panels(panel_names),
                garment_box_mesh.panel_tree,
                padding=packing_padding,
                strategy=packing_strategy,
                max_iterations=packing_max_iterations,
            )
            panel_offsets.update(packing_result.offsets)
            has_overlap = has_overlap or packing_result.has_overlap
            packing_sides[side] = {
                "panel_names": panel_names,
                "iterations": packing_result.iterations,
                "has_overlap": packing_result.has_overlap,
            }
            for island in islands:
                panel_name = island['panel_name']
                uv_dict[panel_name] = packing_result.panel_vertices[panel_name]
                    
        for vertex_texture in garment_box_mesh.vertex_texture:
            panel_name = vertex_texture['panel_name']
            vertex_texture['uv'] = uv_dict[panel_name]
                    
        if has_overlap:
            logging.warning(f"Has overlap for {garment_name}")
            return False
                
        panel_offsets = {k: v.tolist() for k, v in panel_offsets.items()}
        
        with open(out_folder / f'panel_offsets_{garment_name}.json', 'w') as f:
            json.dump(panel_offsets, f, indent=2)

        packing_metadata = {
            "strategy": packing_strategy,
            "padding_cm": float(packing_padding),
            "max_iterations": int(packing_max_iterations),
            "sides": packing_sides,
        }
        with open(
            out_folder / f'packing_metadata_{garment_name}.json',
            'w',
        ) as f:
            json.dump(packing_metadata, f, indent=2)

        final_inside_pts = []
        final_outside_pts = []
        final_interior_verts = []
        
        for side, islands in [('front', front_islands), ('back', back_islands)]:
            polys = []
            all_verts = []
            all_sim_verts = []
            all_fine_verts = []
            all_faces = []
            all_fine_faces = []
            all_interior_verts_list = []
            all_inside_pts = []
            all_outside_pts = []
            face_offset = 0
            fine_face_offset = 0
            
            for i, island in enumerate(islands):
                panel_name = island['panel_name']
                panel = garment_box_mesh.panels[panel_name]
                fine_panel = garment_box_mesh_fine.panels[panel_name]
                is_front = side == 'front'
                
                fine_verts = garment_box_mesh_fine.packing_panels[
                    panel_name
                ].placed_vertices()
                fine_verts += panel_offsets[panel_name]
                fine_faces = fine_panel.panel_faces
                fine_faces = np.array(fine_faces)
                all_fine_verts.append(fine_verts)
                all_fine_faces.append(fine_faces + fine_face_offset)
                fine_face_offset += len(fine_verts)
                
                verts = island['uv']
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
                
                
                
                boundary_verts = []
                boundary_ids = []
                boundary_verts = []
                inside_pts = []
                outside_pts = []
                for edge_id, edge in enumerate(panel.edges):
                    boundary_ids.append(edge.vertex_range)
                    edge_verts = verts[edge.vertex_range]
                    sim_edge_verts = coarse_sim_verts[edge.vertex_range]
                    boundary_verts.append(edge_verts[:-1])
                    lines = []
                    is_dart_and_skip = False
                    target_edge = None
                    for seam in garment_box_mesh.stitches:
                        if seam.panel_1 == seam.panel_2 == panel_name and (seam.edge_1 == edge_id or seam.edge_2 == edge_id):
                            target_edge = seam.edge_2
                            if seam.edge_2 == edge_id:
                                is_dart_and_skip = True
                            break
                    for j in range(len(edge_verts)-1):
                        p1 = edge_verts[j]
                        sim_p1 = sim_edge_verts[j]
                        p2 = edge_verts[j+1]
                        sim_p2 = sim_edge_verts[j+1]
                        avg_sim_p = (sim_p1 + sim_p2) / 2
                        edge_perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
                        line = get_line_from_dir(edge_perp, (p1 + p2) / 2)
                        lines.append(line)
                        inside_pt = line(eps * judge_orientation(panel_name, is_front))
                        inside_pt = np.concatenate([inside_pt, avg_sim_p], 0)
                        inside_pts.append(inside_pt)
                        if target_edge is None:
                            outside_pt = line(-eps * judge_orientation(panel_name, is_front))
                        else:
                            # dart 
                            if is_dart_and_skip:
                                continue
                            n_verts = len(edge_verts)
                            next_edge = panel.edges[target_edge]
                            next_edge_verts = verts[next_edge.vertex_range]
                            next_p1 = next_edge_verts[n_verts - 1 - j]
                            next_p2 = next_edge_verts[n_verts - j - 2]
                            mid_p1 = (p1 + next_p1) / 2
                            mid_p2 = (p2 + next_p2) / 2
                            outside_pt = (mid_p1 + mid_p2) / 2
                        
                        
                        outside_pt = np.concatenate([outside_pt, avg_sim_p], 0)
                        outside_pts.append(outside_pt)
                        
                        
                
                boundary_verts = np.concatenate(boundary_verts, axis=0)
                boundary_verts = np.concatenate([boundary_verts, boundary_verts[0:1]], axis=0)
                boundary_ids = np.concatenate(boundary_ids, axis=0)
                interior_ids = np.setdiff1d(np.arange(len(verts)), boundary_ids)
                interior_verts = verts[interior_ids]
                interior_sim_verts = coarse_sim_verts[interior_ids]
                all_interior_verts_list.append(np.concatenate([interior_verts, interior_sim_verts], -1))
                poly = Polygon(boundary_verts)
                
                
                _inside_pts = np.array(inside_pts)
                _outside_pts = np.array(outside_pts)
                in_contain_flags = [poly.contains(Point(_inside_pts[i, :2])) for i in range(len(_inside_pts))]
                out_contain_flags = [poly.contains(Point(_outside_pts[i, :2])) for i in range(len(_outside_pts))]
                inside_pts = np.stack([p for i, p in enumerate(_inside_pts) if in_contain_flags[i]])
                outside_pts = np.stack([p for i, p in enumerate(_outside_pts) if not out_contain_flags[i]])
                
                all_inside_pts.append(inside_pts)
                all_outside_pts.append(outside_pts)

            # Process concatenated data for this side
            all_inside_pts = np.concatenate(all_inside_pts, axis=0)
            all_outside_pts = np.concatenate(all_outside_pts, axis=0)
            all_interior_verts = np.concatenate(all_interior_verts_list, axis=0) if all_interior_verts_list else np.empty((0, 2))
                
            # plt.figure(figsize=(10, 10))
            # if len(all_inside_pts) > 0:
            #     plt.scatter(all_inside_pts[:, 0], all_inside_pts[:, 1], color='red', label='inside', s=0.1, alpha=0.6)
            # if len(all_interior_verts) > 0:
            #     plt.scatter(all_interior_verts[:, 0], all_interior_verts[:, 1], color='green', label='interior', s=0.1, alpha=0.6)
            # if len(all_outside_pts) > 0:
            #     plt.scatter(all_outside_pts[:, 0], all_outside_pts[:, 1], color='blue', label='outside', s=0.1, alpha=0.6)
            # plt.axis('equal')
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig(out_folder / f'garment_particles_init_{side}.png', dpi=150, bbox_inches='tight')
            # plt.close()

            all_verts = np.concatenate(all_verts, axis=0)
            inside_uvs, outside_uvs = optimize_particles(all_verts, all_inside_pts[:, :2], all_outside_pts[:, :2], thresh=2)
            inside_pts = np.concatenate([inside_uvs, all_inside_pts[:, 2:]], axis=-1)
            outside_pts = np.concatenate([outside_uvs, all_outside_pts[:, 2:]], axis=-1)
            
            plt.figure(figsize=(10, 10))
            plt.scatter(inside_pts[:, 0], inside_pts[:, 1], s=0.5, alpha=0.6)
            plt.scatter(outside_pts[:, 0], outside_pts[:, 1], s=0.5, alpha=0.6)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(out_folder / f"garment_particles_opt_{side}.png", dpi=150, bbox_inches='tight')
            plt.close()
                    
            final_inside_pts.append(inside_pts)
            final_outside_pts.append(outside_pts)
            final_interior_verts.append(all_interior_verts)
                
            optimized_inside_pts = np.concatenate([inside_pts, all_interior_verts], axis=0)
            regionA, regionB = bichromatic_voronoi_separator(optimized_inside_pts[:, :2], outside_pts[:, :2], bbox=None, pad=0.2)
            n_regions = len(regionA.geoms) if regionA.geom_type == "MultiPolygon" else 1
            if n_regions != len(islands):
                logging.error(f"Bichromatic Voronoi separator returned {n_regions} panels but the garment has {len(islands)}.")
                return False 
            fig, ax = plt.subplots(figsize=(8, 8))

            # Plot regions with error handling
            for geom, alpha, c in [(regionA, 0.1, "red"), (regionB, 0.1, "blue")]:
                if geom.is_empty: 
                    continue
                if geom.geom_type == "Polygon":
                    xs, ys = geom.exterior.xy
                    if c == "red":
                        ax.plot(xs, ys, linewidth=2, color=c)
                else:
                    for g in geom.geoms:
                        xs, ys = g.exterior.xy
                        if c == "red":
                            ax.plot(xs, ys, linewidth=2, color=c)

            ax.set_aspect('equal', 'box')
            ax.set_title("Bichromatic Voronoi separator (bisector) and regions")
            plt.tight_layout()
            plt.savefig(out_folder / f'garment_particles_opt_bis_{side}.png', dpi=150, bbox_inches='tight')
            plt.close()
                
        np.savez_compressed(out_folder / f'garment_particles_{garment_name}.npz', 
                            inside_pts_front=final_inside_pts[0], 
                            outside_pts_front=final_outside_pts[0], 
                            interior_verts_front=final_interior_verts[0],
                            inside_pts_back=final_inside_pts[1], 
                            outside_pts_back=final_outside_pts[1], 
                            interior_verts_back=final_interior_verts[1])
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
    parser.add_argument("--output_dir", type=str, default="/orion/u/w4756677/garment/gcdv2/garment_particles_filtered")
    parser.add_argument("--resume", action="store_true", help="Skip already processed garments")
    parser.add_argument(
        "--packing_strategy",
        choices=PACKING_STRATEGIES,
        default=DEFAULT_PACKING_STRATEGY,
        help="Semantic panel-packing strategy used before particle extraction",
    )
    parser.add_argument(
        "--packing_padding",
        type=float,
        default=DEFAULT_PACKING_PADDING,
        help="Required panel clearance in centimeters",
    )
    parser.add_argument(
        "--packing_max_iterations",
        type=int,
        default=DEFAULT_PACKING_MAX_ITERATIONS,
        help="Maximum overlap-resolution iterations per packing stage",
    )
    args = parser.parse_args()
    
    with open(args.pattern_list, 'r') as f:
        pattern_folders = f.readlines()
        
    logging.info(f"Processing {len(pattern_folders)} garments")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(
        "Pattern packing: strategy=%s, padding=%s cm, max_iterations=%s",
        args.packing_strategy,
        args.packing_padding,
        args.packing_max_iterations,
    )
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
            success = process_garment_single(
                pattern_folder,
                args.output_dir,
                packing_strategy=args.packing_strategy,
                packing_padding=args.packing_padding,
                packing_max_iterations=args.packing_max_iterations,
            )
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
