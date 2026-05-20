"""Routines for processing UV coordinated for garments and generating texture maps"""
import numpy as np
import igl
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from anytree import AnyNode, LevelOrderIter, LevelOrderGroupIter
from shapely.geometry import Polygon
from shapely.affinity import translate
from shapely.ops import unary_union

# SECTION UV islands texture creation 
def texture_mesh_islands(
    vertex_texture: List[Dict],
    panel_tree: AnyNode,
    out_texture_image_path: Path, 
    out_fabric_tex_image_path: Optional[Path] = None, 
    out_mtl_file_path: Optional[Path] = None, 
    boundary_width: float = 0.3, 
    dpi: int = 1200, 
    background_img_path: Optional[Path] = None,
    background_resolution: int = 1,
    uv_padding: int = 1, 
    mat_name_prefix: str = 'panels_texture'
):
    """
        Returns updated uv coordinates (properly normalized and aligned with the created texture)
        and material information for front/back panels
    """
    front_islands = [island for island in vertex_texture if any(node.name == island['panel_name'] for node in LevelOrderIter(panel_tree.children[0]))]
    back_islands = [island for island in vertex_texture if any(node.name == island['panel_name'] for node in LevelOrderIter(panel_tree.children[1]))]
    uv_dict = {}
    
    materials = {}
    max_uv, min_uv = -np.inf, np.inf
    for group_name, islands in [('front', front_islands), ('back', back_islands)]:
        if not islands:
            continue

        group_uvs, boundary_uv_to_draw = unwarp_UV(islands, panel_tree, padding=uv_padding)       
        max_uv = max(max_uv, np.array(group_uvs).max())
        min_uv = min(min_uv, np.array(group_uvs).min())
        
        norm_uvs, width = normalize_UVs(group_uvs, axis_padding=uv_padding)
        

        # Create texture image
        tex_path = out_texture_image_path.with_name(f'{out_texture_image_path.stem}_{group_name}{out_texture_image_path.suffix}')
        create_UV_island_texture(
            boundary_uv_to_draw, width, width,
            texture_image_path=tex_path,
            boundary_width=boundary_width,
            dpi=dpi,
            preserve_alpha=True
        )

        fabric_tex_path = None
        if out_fabric_tex_image_path is not None and background_img_path is not None:
            fabric_tex_path = out_fabric_tex_image_path.with_name(f'{out_fabric_tex_image_path.stem}_{group_name}{out_fabric_tex_image_path.suffix}')
            create_UV_island_texture(
                boundary_uv_to_draw, width, width,
                texture_image_path=fabric_tex_path,
                boundary_width=boundary_width,
                dpi=dpi,
                background_img_path=background_img_path, 
                background_resolution=background_resolution,
                preserve_alpha=False  
            )

        texture_filename = fabric_tex_path.name if fabric_tex_path is not None else tex_path.name
        mat_name = f'{mat_name_prefix}_{group_name}'
        materials[group_name] = {'name': mat_name, 'map_Kd': texture_filename}

        # Place UVs back into the global UV array
        start = 0
        uv_groups = []
        for i, island in enumerate(islands):
            uv_dict[island["panel_name"]] = norm_uvs[start:start + len(island['uv'])]
            uv_groups.append(norm_uvs[start:start + len(island['uv'])])
            start += len(island['uv'])
        
    
    all_uvs = np.concatenate([uv_dict[island["panel_name"]] for island in vertex_texture])
    
    if out_mtl_file_path:
        save_texture_mtl_multi(out_mtl_file_path, materials)

    return all_uvs, materials, uv_dict, max_uv, min_uv

def _uv_connected_components(face_texture_coords):
    # Find connected components of face
    face_components = igl.facet_components(face_texture_coords)[1]
    num_ccs = max(face_components) + 1

    # Derive vertex component indices from faces
    vert_components = np.zeros(face_texture_coords.max() + 1, dtype=int)
    for i in range(num_ccs):
        verts_in_cc = np.unique(face_texture_coords[face_components == i])
        vert_components[verts_in_cc] = i

    return vert_components, face_components, num_ccs

def _bbox(uv: np.ndarray) -> Tuple[float, float, float, float]:
    """Return width, height, min_u, min_v for an island."""
    mn = uv.min(axis=0)
    mx = uv.max(axis=0)
    return (mx[0] - mn[0], mx[1] - mn[1], *mn)

def unwarp_UV(panels, panel_tree, padding=1):
    # Unwrap uvs for each connected component------------------------
    # 1. Initial placement based on 3D panel centers
    polygons = []
    for i in range(len(panels)):
        uvs = np.array(panels[i]["uv"])
        # uvs = np.array(panels[i]["uv"]) + panels[i]["translation"]
        boundary_ids = igl.boundary_loop(panels[i]["face_texture_coords"])
        boundary_uvs = uvs[boundary_ids]
        flat_rot_angle = panels[i]['rotation']
        rotation_center = panels[i]['rotation_center']
        rotation_matrix = np.array([[np.cos(flat_rot_angle), -np.sin(flat_rot_angle)], [np.sin(flat_rot_angle), np.cos(flat_rot_angle)]])
        rotated_boundary_uvs = (rotation_matrix @ (boundary_uvs - rotation_center)[..., None])[..., 0] + rotation_center
        panels[i]["uv"] = (rotation_matrix @ (uvs - rotation_center)[..., None])[..., 0] + rotation_center + panels[i]["translation"]
        polygons.append(Polygon(rotated_boundary_uvs + panels[i]["translation"]))
        # print(boundary_uvs)
        # print(np.all(boundary_uvs == np.array(polygons[-1].boundary.coords)[:-1]))

    # 2. Hierarchically resolve overlaps using polygons
    bufferred_polygons = [polygon.buffer(padding/2) for polygon in polygons]
    name_to_idx = {island['panel_name']: i for i, island in enumerate(panels)}

    def move_node_and_descendants(node_to_move, move_x, move_y, name_to_idx, polygons):
        nodes_to_move = [node_to_move] + list(node_to_move.descendants)
        for node in nodes_to_move:
            if node.name in name_to_idx:
                idx = name_to_idx[node.name]
                polygons[idx] = translate(polygons[idx], xoff=move_x, yoff=move_y)

    iterations = 500
    for _ in tqdm(range(iterations)):
        had_overlap = False
        # Resolve sibling overlaps
        for level_nodes in LevelOrderGroupIter(panel_tree):
            for i in range(len(level_nodes)):
                for j in range(i + 1, len(level_nodes)):
                    node_i = level_nodes[i]
                    node_j = level_nodes[j]

                    if node_i.name not in name_to_idx or node_j.name not in name_to_idx:
                        continue

                    idx_i = name_to_idx[node_i.name]
                    idx_j = name_to_idx[node_j.name]

                    poly_i = bufferred_polygons[idx_i]
                    poly_j = bufferred_polygons[idx_j]

                    if poly_i.intersects(poly_j):
                        had_overlap = True
                        # Compute centroids
                        centroid_i = np.array([poly_i.centroid.x, poly_i.centroid.y])
                        centroid_j = np.array([poly_j.centroid.x, poly_j.centroid.y])
                        direction = centroid_j - centroid_i
                        direction_norm = np.linalg.norm(direction)
                        if direction_norm == 0:
                            direction = np.array([1.0, 0.0])
                        direction = direction / direction_norm
                        # Move each polygon away from each other by padding/2
                        move_vec = direction * (padding) / 2
                        move_node_and_descendants(node_i, -move_vec[0]/2, -move_vec[1]/2, name_to_idx, bufferred_polygons)
                        move_node_and_descendants(node_i, -move_vec[0]/2, -move_vec[1]/2, name_to_idx, polygons)
                        move_node_and_descendants(node_j, move_vec[0]/2, move_vec[1]/2, name_to_idx, bufferred_polygons)
                        move_node_and_descendants(node_j, move_vec[0]/2, move_vec[1]/2, name_to_idx, polygons)
        # Resolve parent-child overlaps
        for node in LevelOrderIter(panel_tree):
            if node.is_root or node.parent is None:
                continue
            parent = node.parent
            if node.name not in name_to_idx or parent.name not in name_to_idx:
                continue
            idx_child = name_to_idx[node.name]
            idx_parent = name_to_idx[parent.name]
            poly_child = bufferred_polygons[idx_child]
            poly_parent = bufferred_polygons[idx_parent]
            if poly_child.intersects(poly_parent):
                had_overlap = True
                centroid_child = np.array([poly_child.centroid.x, poly_child.centroid.y])
                centroid_parent = np.array([poly_parent.centroid.x, poly_parent.centroid.y])
                direction = centroid_child - centroid_parent
                direction_norm = np.linalg.norm(direction)
                if direction_norm == 0:
                    direction = np.array([1.0, 0.0])
                else:
                    direction = direction / direction_norm
                move_vec = direction * padding
                move_node_and_descendants(node, move_vec[0], move_vec[1], name_to_idx, polygons)
                move_node_and_descendants(node, move_vec[0], move_vec[1], name_to_idx, bufferred_polygons)
                
        if not had_overlap:
            break
    
    if had_overlap:
        print(f'Had overlap after {iterations} iterations')

    # 3) translate local UVs into packed space
    # First find global min u,v coordinates
    # Shift all UVs to be non-negative and apply packing offsets
    offsets = {}
    for i in range(len(panels)):
        # Get the translation vector from the original polygon to the moved polygon
        uvs = np.array(panels[i]["uv"])
        boundary_ids = igl.boundary_loop(panels[i]["face_texture_coords"])
        boundary_uvs = uvs[boundary_ids]
        orig_poly = Polygon(boundary_uvs)
        moved_poly = polygons[i]
        # Calculate translation offset
        offset = np.array([moved_poly.centroid.x, moved_poly.centroid.y]) - np.array([orig_poly.centroid.x, orig_poly.centroid.y])
        offsets[panels[i]["panel_name"]] = offset
    
    all_uvs = []
    boundary_uv_to_draw = []
    for i in range(len(panels)):
        uvs = np.array(panels[i]["uv"]) + offsets[panels[i]["panel_name"]]
        boundary_ids = igl.boundary_loop(panels[i]["face_texture_coords"])
        boundary_uvs = uvs[boundary_ids]
        boundary_uv_to_draw.append(boundary_uvs)
        all_uvs.append(uvs.tolist())

    return all_uvs, boundary_uv_to_draw, had_overlap, offsets

def unwarp_UV_hierarchical(panels, panel_tree, padding=1):
    """
    Similar to unwarp_UV but performs intersection checks on the union of all children
    for each node, not just the individual island.
    """
    
    # Initial setup - same as unwarp_UV
    polygons = []
    for i in range(len(panels)):
        uvs = np.array(panels[i]["uv"])
        boundary_ids = igl.boundary_loop(panels[i]["face_texture_coords"])
        boundary_uvs = uvs[boundary_ids]
        flat_rot_angle = panels[i]['rotation']
        rotation_center = panels[i]['rotation_center']
        rotation_matrix = np.array([[np.cos(flat_rot_angle), -np.sin(flat_rot_angle)], [np.sin(flat_rot_angle), np.cos(flat_rot_angle)]])
        rotated_boundary_uvs = (rotation_matrix @ (boundary_uvs - rotation_center)[..., None])[..., 0] + rotation_center
        panels[i]["uv"] = (rotation_matrix @ (uvs - rotation_center)[..., None])[..., 0] + rotation_center + panels[i]["translation"]
        polygons.append(Polygon(rotated_boundary_uvs + panels[i]["translation"]))

    bufferred_polygons = [polygon.buffer(padding/2) for polygon in polygons]
    name_to_idx = {island['panel_name']: i for i, island in enumerate(panels)}

    def get_node_union_polygon(node, name_to_idx, polygons):
        """Get the union of polygons for a node and all its descendants"""
        node_polygons = []
        
        # Add current node's polygon if it exists
        if node.name in name_to_idx:
            idx = name_to_idx[node.name]
            node_polygons.append(polygons[idx])
        
        # Add all descendant polygons
        for descendant in node.descendants:
            if descendant.name in name_to_idx:
                idx = name_to_idx[descendant.name]
                node_polygons.append(polygons[idx])
        
        if node_polygons:
            return unary_union(node_polygons)
        return None

    def move_node_and_descendants(node_to_move, move_x, move_y, name_to_idx, polygons):
        nodes_to_move = [node_to_move] + list(node_to_move.descendants)
        for node in nodes_to_move:
            if node.name in name_to_idx:
                idx = name_to_idx[node.name]
                polygons[idx] = translate(polygons[idx], xoff=move_x, yoff=move_y)

    iterations = 500
    for _ in tqdm(range(iterations)):
        had_overlap = False
        
        # Resolve sibling overlaps using hierarchical union polygons
        for node in LevelOrderIter(panel_tree):
            if node.is_leaf:
                continue 
            level_nodes = list(node.children)
            for i in range(len(level_nodes)):
                for j in range(i + 1, len(level_nodes)):
                    node_i = level_nodes[i]
                    node_j = level_nodes[j]

                    
                    # Get union polygons for each node and its descendants
                    union_i = get_node_union_polygon(node_i, name_to_idx, bufferred_polygons)
                    union_j = get_node_union_polygon(node_j, name_to_idx, bufferred_polygons)

                    if union_i is None or union_j is None:
                        continue

                    if union_i.intersects(union_j):
                        had_overlap = True
                        # Compute centroids of union polygons
                        centroid_i = np.array([union_i.centroid.x, union_i.centroid.y])
                        centroid_j = np.array([union_j.centroid.x, union_j.centroid.y])
                        direction = centroid_j - centroid_i
                        if node.name in ["front", "back"]:
                            # the second level only move vertically. 
                            direction = direction - direction[0] * np.array([1, 0])
                        if node.name in ["top"]:
                            # the second level only move horizontally. 
                            direction = direction - direction[1] * np.array([0, 1])
                        if "pant" in node_i.name or "pant" in node_j.name:
                            # the second level only move horizontally. 
                            direction = direction - direction[1] * np.array([0, 1])
                            
                        direction_norm = np.linalg.norm(direction)
                        if direction_norm == 0:
                            direction = np.array([1.0, 0.0])
                        direction = direction / direction_norm
                        
                        # Move each hierarchy away from each other by padding/2
                        move_vec = direction * (padding) / 2
                        move_node_and_descendants(node_i, -move_vec[0]/2, -move_vec[1]/2, name_to_idx, bufferred_polygons)
                        move_node_and_descendants(node_i, -move_vec[0]/2, -move_vec[1]/2, name_to_idx, polygons)
                        move_node_and_descendants(node_j, move_vec[0]/2, move_vec[1]/2, name_to_idx, bufferred_polygons)
                        move_node_and_descendants(node_j, move_vec[0]/2, move_vec[1]/2, name_to_idx, polygons)
        
        # Resolve parent-child overlaps using hierarchical union polygons
        # import ipdb; ipdb.set_trace()
        for node in LevelOrderIter(panel_tree):
            if node.is_root or node.parent is None:
                continue
            parent = node.parent
            
            if node.name not in name_to_idx or parent.name not in name_to_idx:
                continue
            # Get union polygons
            idx_child = name_to_idx[node.name]
            idx_parent = name_to_idx[parent.name]
            poly_child = get_node_union_polygon(node, name_to_idx, bufferred_polygons)
            poly_parent = bufferred_polygons[idx_parent]
            
            if poly_child is None or poly_parent is None:
                continue
                
            if poly_child.intersects(poly_parent):
                had_overlap = True
                centroid_child = np.array([poly_child.centroid.x, poly_child.centroid.y])
                centroid_parent = np.array([poly_parent.centroid.x, poly_parent.centroid.y])
                direction = centroid_child - centroid_parent
                if "torso" in parent.name and "sleeve" in node.name:
                    direction = direction - direction[1] * np.array([0, 1])
                direction_norm = np.linalg.norm(direction)
                if direction_norm == 0:
                    direction = np.array([1.0, 0.0])
                else:
                    direction = direction / direction_norm
                move_vec = direction * padding
                move_node_and_descendants(node, move_vec[0], move_vec[1], name_to_idx, polygons)
                move_node_and_descendants(node, move_vec[0], move_vec[1], name_to_idx, bufferred_polygons)
                
        if not had_overlap:
            break
    
    if had_overlap:
        print(f'Had overlap after {iterations} iterations')

    # Calculate final offsets and apply to UVs - same as unwarp_UV
    offsets = {}
    for i in range(len(panels)):
        uvs = np.array(panels[i]["uv"])
        boundary_ids = igl.boundary_loop(panels[i]["face_texture_coords"])
        boundary_uvs = uvs[boundary_ids]
        orig_poly = Polygon(boundary_uvs)
        moved_poly = polygons[i]
        offset = np.array([moved_poly.centroid.x, moved_poly.centroid.y]) - np.array([orig_poly.centroid.x, orig_poly.centroid.y])
        offsets[panels[i]["panel_name"]] = offset
    
    all_uvs = []
    boundary_uv_to_draw = []
    for i in range(len(panels)):
        uvs = np.array(panels[i]["uv"]) + offsets[panels[i]["panel_name"]]
        boundary_ids = igl.boundary_loop(panels[i]["face_texture_coords"])
        boundary_uvs = uvs[boundary_ids]
        boundary_uv_to_draw.append(boundary_uvs)
        all_uvs.append(uvs.tolist())

    return all_uvs, boundary_uv_to_draw, had_overlap, offsets

def normalize_UVs(all_uvs, axis_padding=3):
    # normalize all_uvs
    uv_list_raw = np.concatenate(all_uvs)
    uv_list = uv_list_raw

    norm_x = max(uv_list_raw[:,0]) + axis_padding
    norm_y = max(uv_list_raw[:,1]) + axis_padding
    norm_max = max(norm_x, norm_y)
    uv_list[:,0] = uv_list_raw[:,0] / norm_max
    uv_list[:,1] = uv_list_raw[:,1] / norm_max

    return uv_list, norm_max

def create_UV_island_texture(
        boundary_uv_to_draw, 
        width, height, 
        texture_image_path, 
        boundary_width=0.3, 
        boundary_color='black',
        dpi=1200,
        color_alpha=0.65,
        background_alpha=0.8,
        background_img_path=None,
        background_resolution=5,
        preserve_alpha=True
    ):
    """Create texture image from the set of UV boundary loops (e.g. sewing pattern panels). 
        It renders the border of the loops and fills them in with color 
        Params: 
            * boundary_uv_to_draw -- 2D list -- sequence of 2D vertices on each of the boundaries. The order is IMPORTANT. The vertices will be connected 
                by boundary edges sequentially
            * width, height -- the dimentions of the UV map  
            * texture_image_path -- filepath to same a texture image to
            * boundary_width -- width of the boundary outline 
            * dpi -- resolution of the output image
    """
    n_components = len(boundary_uv_to_draw)

    # Figure size
    fig, ax = plt.subplots()
    fig.set_size_inches(width / 100, height / 100)  # width & height are usually given in cm

    # Colors
    shift = 0.17
    divisor = max(5, n_components)
    cmap = matplotlib.colormaps['twilight']   # copper cool  spring winter twilight  # Using smooth Matplotlib colormaps
    color_sample = [cmap((1 - shift) * id / divisor) for id in range(divisor)]

    # Background -- garment style
    if background_img_path is not None:
        back_crop_scale = background_resolution
        back_img = plt.imread(background_img_path)
        ax.imshow(
            back_img[:int(width * back_crop_scale), :int(height * back_crop_scale), :], 
            extent=[0, width, 0, height], 
            alpha=background_alpha,
            aspect='equal'
        )

    # Draw the UV island boundaries and fill them up
    for i in range(n_components):
        polygon_x = [vert[0] for vert in boundary_uv_to_draw[i]]
        polygon_x.append(polygon_x[0])  # Loop
        polygon_y = [vert[1] for vert in boundary_uv_to_draw[i]]
        polygon_y.append(polygon_y[0])  # Loop

        color = list(color_sample[i])
        color[-1] = color_alpha   # Alpha - transparency for blending with backround

        plt.fill(polygon_x, polygon_y, 
                 color=color, 
                 edgecolor=boundary_color, linestyle='-', linewidth=boundary_width / 2  # Boundary stylings
        )
        
    ax.set_aspect('equal')

    # Set the axis to be tight
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])

    # Hide the axis
    plt.axis('off')

    # Save image
    plt.savefig(texture_image_path, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=preserve_alpha)

    # Cleanup
    plt.close()

# !SECTION

# SECTION Saving textures information to files
def save_texture_mtl_multi(mtl_file_path, materials):
    """Save MTL file with multiple materials"""
    with open(mtl_file_path, 'w') as file:
        for group_name, material in materials.items():
            new_material_lines = [
                f'newmtl {material["name"]}\n',
                'Ns 0.000000\n',
                'Ka 1.000000 1.000000 1.000000\n',
                'Ks 0.000000 0.000000 0.000000\n',
                'Ke 0.000000 0.000000 0.000000\n',
                'Ni 1.000000\n',
                'd 1.000000\n',
                'illum 1\n',
                f'map_Kd {material["map_Kd"]}\n\n'
            ]
            file.writelines(new_material_lines)

def save_obj_with_materials(
        output_file_path, 
        vertices, faces_with_texture, uv_list, 
        vertex_texture_info,
        panel_names,  # List of panel names for each face
        panel_to_material,  # Dict: panel_name -> material_name
        vert_normals=None, 
        mtl_file_name=None
):
    """Save an obj file with multiple materials for front/back panels"""
    with open(output_file_path, 'w') as f:
        if mtl_file_name is not None:
            f.write(f'mtllib {mtl_file_name}\n')

        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        for vt in uv_list:
            f.write(f"vt {vt[0]} {vt[1]}\n")

        if vert_normals is not None:
            for vn in vert_normals:
                f.write(f"vn {vn[0]} {vn[1]} {vn[2]}\n")
            
        f.write('s 1\n')
        
        current_material = None
        for i, (face, panel_name) in enumerate(zip(faces_with_texture, panel_names)):
            mat = panel_to_material[panel_name]
            if mat != current_material:
                f.write(f'usemtl {mat}\n')
                current_material = mat
            
            v_id0, tex_id0, v_id1, tex_id1, v_id2, tex_id2 = face
            if vert_normals is not None:
                f.write(f"f {v_id0 + 1}/{tex_id0 + 1}/{v_id0 + 1} "
                        f"{v_id1 + 1}/{tex_id1 + 1}/{v_id1 + 1} "
                        f"{v_id2 + 1}/{tex_id2 + 1}/{v_id2 + 1}\n")
            else:
                f.write(f"f {v_id0 + 1}/{tex_id0 + 1} "
                        f"{v_id1 + 1}/{tex_id1 + 1} "
                        f"{v_id2 + 1}/{tex_id2 + 1}\n")

def add_texture_to_obj(obj_file_path, output_file_path, uv_list, mtl_file_name, mat_name):
    # Update OBJ-----------------------------------------------------

    with open(obj_file_path, 'r') as file:
        lines = file.readlines()

    uv_index = 0
    updated_lines = []
    mtllib_exists = False
    inserted = False

    s_and_usemtl_lines = ['s 1\n', f'usemtl {mat_name}\n']

    for line in lines:
        if line.startswith('vt '):
            # Format the new UV coordinates
            uv = uv_list[uv_index]
            new_uv_line = f'vt {uv[0]:.6f} {uv[1]:.6f}\n'
            updated_lines.append(new_uv_line)
            uv_index += 1
        elif line.startswith('mtllib '):
            # Ensure the mtllib line points to the correct MTL file
            new_mtl_line = f'mtllib {mtl_file_name}\n'
            updated_lines.append(new_mtl_line)
            mtllib_exists = True
        elif line.startswith('f') and not inserted:
            # Insert the s and usemtl lines before the first face line
            updated_lines.extend(s_and_usemtl_lines)
            inserted = True
            updated_lines.append(line)
        else:
            updated_lines.append(line)
            
    # If mtllib line does not exist, add it at the beginning
    if not mtllib_exists:
        updated_lines.insert(0, f'mtllib {mtl_file_name}\n')

    with open(output_file_path, 'w') as file:
        file.writelines(updated_lines)

# !SECTION