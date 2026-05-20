import matplotlib.pyplot as plt 
import numpy as np
import os 
from glob import glob
import cv2
import igl
import h5py
from skimage.draw import polygon
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
overlap_list = open("assets/gcd_image_stats/has_overlap_list.txt", "r").readlines()
overlap_list = [os.path.basename(line.strip()).replace("_texture.hdf5", "") for line in overlap_list]
oversize_list = open("assets/gcd_image_stats/oversize_list.txt", "r").readlines()
oversize_list = [os.path.basename(line.strip()).replace("_texture.hdf5", "") for line in oversize_list]
def get_command_args():
    """command line arguments to control the run"""
    # https://stackoverflow.com/questions/40001892/reading-named-command-arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_path', '-o', 
        help='output path', 
        type=str, 
        default='/miele/data/gcd/gcdv2_images/')
    parser.add_argument(
        '--data_path', '-d', 
        help='data path', 
        type=str, 
        default='/miele/data/gcd/gcdv2_images_rerun_again/')
    parser.add_argument(
        '--chunk_id', '-c', 
        help='chunk id', 
        type=int, 
        default=0)


    args = parser.parse_args()
    print('Commandline arguments: ', args)

    return args

min_vert = np.array([-135,   -40])
max_vert = np.array([135, 200])
min_color = np.array([-63.32013931, -12, -40])
max_color = np.array([63.51664658, 172.80234273,  40])
def normalize_color(color, m=None, M=None):
    if m is None:
        m = np.min(color)
    if M is None:
        M = np.max(color)
    return (color - m) / (M - m)
def rasterize_polygon(vertices, faces, face_colors, image_size=(64,64)):
    # Initialize blank image with alpha channel
    img = np.zeros((image_size[1], image_size[0], 4), dtype=np.uint8)

    # Normalize vertices to image size if needed
    scale = max(max_vert - min_vert)
    norm_vertices = (vertices - min_vert) / scale
    norm_vertices[:,0] *= image_size[0]-1
    norm_vertices[:,1] *= image_size[1]-1
    norm_vertices[:,1] = image_size[1] - norm_vertices[:,1]
    face_colors = normalize_color(face_colors, min_color, max_color)

    # Rasterize each face
    for face, color in zip(faces, face_colors):
        pts = norm_vertices[face]
        rr, cc = polygon(pts[:,1], pts[:,0], img.shape)
        if np.max(color) <= 1.0:
            color = (np.array(color) * 255).astype(np.uint8)
        # Set RGB and alpha values
        img[rr, cc, :3] = color
        img[rr, cc, 3] = 255  # Set alpha to fully opaque for colored pixels
    
    return img



def process_texture_file(data_name, out_path, chunk_id, data_path):
    if os.path.exists(os.path.join(out_path, data_name)) and \
        os.path.exists(os.path.join(out_path, data_name, "front_1024.png")) and \
        os.path.exists(os.path.join(out_path, data_name, "back_1024.png")):
        print(f"Skipping {data_name} because it already exists")
        return
    
    texture_file = os.path.join(data_path, f"chunk_{chunk_id}", f'{data_name}_texture.hdf5')
    if (not os.path.exists(texture_file)) or \
        data_name in overlap_list or \
        data_name in oversize_list:
        print(f"Skipping {data_name} because it is in overlap or oversize list or does not exist")
        return
    with h5py.File(texture_file, 'r') as f:
        # Initialize lists for front and back panels
        front_verts, front_faces, front_colors = [], [], []
        back_verts, back_faces, back_colors = [], [], []
        
        # Collect panel data
        front_face_offset = 0
        back_face_offset = 0
        for panel_name in f.keys():
            if panel_name == 'has_overlap':
                continue
                
            panel_data = f[panel_name]
            verts = panel_data['vertices'][()]
            faces = panel_data['faces'][()]
            colors = panel_data['colors'][()]
            
            if panel_name.startswith('front_'):
                front_verts.append(verts)
                front_faces.append(faces + front_face_offset)
                front_colors.append(colors)
                front_face_offset += len(verts)
            elif panel_name.startswith('back_'):
                back_verts.append(verts)
                back_faces.append(faces + back_face_offset)
                back_colors.append(colors)
                back_face_offset += len(verts)
                
    # Combine panels and render
    oversize = False
    all_verts = []
    all_faces = []
    all_colors = []
    for panel_type, verts, faces, colors in [
        ('front', front_verts, front_faces, front_colors),
        ('back', back_verts, back_faces, back_colors)
    ]:
        if len(verts) == 0:
            continue
            
        # Combine all panels
        verts = np.concatenate(verts, axis=0)
        if np.any(verts > max_vert) or np.any(verts < min_vert):
            oversize = True
            break
        faces = np.concatenate(faces, axis=0)
        colors = np.concatenate(colors, axis=0)
        if np.any(colors > max_color) or np.any(colors < min_color):
            oversize = True
            break
        all_verts.append(verts)
        all_faces.append(faces)
        all_colors.append(colors)
    if oversize:
        print(f"Skipping {data_name} because it is oversize")
        return

    os.makedirs(os.path.join(out_path, data_name), exist_ok=True)
    for panel_type, verts, faces, colors in [
        ("front", all_verts[0], all_faces[0], all_colors[0]),
        ("back", all_verts[1], all_faces[1], all_colors[1])
    ]:
        image = rasterize_polygon(verts, faces, colors, image_size=(1024, 1024))
        cv2.imwrite(os.path.join(out_path, data_name, f'{panel_type}_1024.png'), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
    print(f"Processed {data_name} to {os.path.join(out_path, data_name)}")

if __name__ == "__main__":
    args = get_command_args()
    out_path = args.out_path
    chunk_id = args.chunk_id
    data_path = args.data_path
    file_list = f"assets/pattern_lists/all_pattern_list_{chunk_id}.txt"
    with open(file_list, 'r') as f:
        data_names = f.read().splitlines()
        data_names = [os.path.basename(folder) for folder in data_names]
    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor(max_workers=16) as executor:
        process_fn = partial(process_texture_file, out_path=out_path, chunk_id=chunk_id, data_path=data_path)
        list(tqdm(executor.map(process_fn, data_names), total=len(data_names)))