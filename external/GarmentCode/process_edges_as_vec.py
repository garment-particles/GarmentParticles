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
import pygarment.pattern.utils as pat_utils
import svgpathtools as svgpath

def get_panel_from_vecs(edge_vecs, scale, shift):
    start = np.array([0, 0])
    unormalized_start = start * scale + shift
    curves = []
    verts = [unormalized_start]
    for edge_vec in edge_vecs:
        end = start + edge_vec[:2]
        unormalized_end = end * scale + shift
        verts.append(unormalized_end)
        flag = edge_vec[6]
        if flag == 1:
            third_point = pat_utils.rel_to_abs_2d(start, end, edge_vec[2:4])
            unormalized_third_point = third_point * scale + shift
            _, _, radius, large_arc, right = pat_utils.arc_from_three_points(start, end, third_point)
            curve = svgpath.Arc(
                    pat_utils.list_to_c(unormalized_start), radius + 1j * radius,
                    rotation=0,
                    large_arc=large_arc,
                    sweep=right, #maya: not right
                    end=pat_utils.list_to_c(unormalized_end)
                )
            
        else:
            control_points = edge_vec[2:6]
            if np.allclose(control_points, 0, atol=1e-3):
                curve = svgpath.Line(*pat_utils.list_to_c([unormalized_start, unormalized_end]))
            else:
                abs_control_points1 = pat_utils.rel_to_abs_2d(unormalized_start, unormalized_end, control_points[:2])
                abs_control_points2 = pat_utils.rel_to_abs_2d(unormalized_start, unormalized_end, control_points[2:])
                curve = svgpath.CubicBezier(*pat_utils.list_to_c([unormalized_start, abs_control_points1, abs_control_points2, unormalized_end]))
        curves.append(curve)
        start = end
        unormalized_start = unormalized_end
    path = svgpath.Path(*curves)
    arrdims = np.array(path.bbox())
    dims = (arrdims[1] - arrdims[0], arrdims[3] - arrdims[2])
    viewbox = (
        arrdims[0] - 2, 
        arrdims[2] - 2, 
        dims[0] + 2 * 2, 
        dims[1] + 2 * 2
    )
    attributes = {
        'fill':  'rgb(227,175,186)',  
        'stroke': 'rgb(51,51,51)', 
        'stroke-width': '0.75'
    }
    dwg = svgpath.wsvg(
        [path], attributes=[attributes], margin_size=0,
        filename="vis.svg", viewbox=viewbox, paths2Drawing=True)
    dwg.save(pretty=True)
    return curves

def process_garment_single(pattern_folder, out_folder, visualize=False):
    try:
        garment_name = pattern_folder.split('/')[-1]
        
        out_folder = Path(out_folder) / garment_name
        out_folder.mkdir(parents=True, exist_ok=True)
        
        pattern_spec = Path(pattern_folder) / f'{garment_name}_specification.json'
        
        try:
            garment_box_mesh = BoxMesh(pattern_spec, 2.5)
            garment_box_mesh.load()
        except DegenerateTrianglesError as e:
            logging.error(f"DegenerateTrianglesError for {garment_name}: {e}")
            return False
        except Exception as e:
            logging.error(f"BoxMesh loading error for {garment_name}: {e}")
            return False

        panel_edge_vecs = {}
        
        for i, (panel_name, panel) in enumerate(garment_box_mesh.panels.items()):
            
            verts = panel.panel_vertices
            verts = np.array(verts)
            
            # scale = (verts.max(0) - verts.min(0))
            # scale = 1
            # shift = verts.min(0)
            # shift = 0
            # panel.normalize(scale, shift)
            edge_vecs = panel.as_vector()
            panel_edge_vecs[panel_name] = edge_vecs


        if visualize:
            curves = get_panel_from_vecs(edge_vecs, scale, shift)
            
        np.savez_compressed(out_folder / f'panel_edge_vecs_{garment_name}.npz', **panel_edge_vecs)
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
    parser.add_argument("--output_dir", type=str, default="/orion/u/w4756677/garment/gcdv2/panel_edge_vecs")
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
