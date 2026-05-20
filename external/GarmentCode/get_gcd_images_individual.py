import matplotlib.pyplot as plt 
import numpy as np
import os 
import cv2
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
    min_vert = np.min(vertices, axis=0)
    max_vert = np.max(vertices, axis=0)
    scale = max_vert - min_vert
    norm_vertices = (vertices - min_vert) / scale
    norm_vertices[:,0] *= image_size[0]-1
    norm_vertices[:,1] *= image_size[1]-1
    norm_vertices[:,1] = image_size[1] - norm_vertices[:,1]
    
    m, M = np.min(face_colors, axis=0), np.max(face_colors, axis=0)
    face_colors = normalize_color(face_colors, m, M)

    # Rasterize each face
    for face, color in zip(faces, face_colors):
        pts = norm_vertices[face]
        rr, cc = polygon(pts[:,1], pts[:,0], img.shape)
        if np.max(color) <= 1.0:
            color = (np.array(color) * 255).astype(np.uint8)
        # Set RGB and alpha values
        img[rr, cc, :3] = color
        img[rr, cc, 3] = 255  # Set alpha to fully opaque for colored pixels
    
    return img, scale.tolist(), np.array([m, M]).tolist()



def process_texture_file(data_name, out_path, chunk_id, data_path):
    
    
    texture_file = os.path.join(data_path, f"chunk_{chunk_id}", f'{data_name}_texture.hdf5')
    if (not os.path.exists(texture_file)) or \
        data_name in overlap_list or \
        data_name in oversize_list:
        print(f"Skipping {data_name} because it is in overlap or oversize list or does not exist")
        return
    with h5py.File(texture_file, 'r') as f:
        # Initialize lists for front and back panels
        vert_list, face_list, color_list, name_list = [], [], [], []
        
        # Collect panel data
        for panel_name in f.keys():
            panel_data = f[panel_name]
            verts = panel_data['vertices'][()]
            faces = panel_data['faces'][()]
            colors = panel_data['colors'][()]
            
            vert_list.append(verts)
            face_list.append(faces)
            color_list.append(colors)
            name_list.append(panel_name)
    if os.path.exists(os.path.join(out_path, data_name)):
        continue_processing = True 
        for name in name_list:
            continue_processing = continue_processing and (not os.path.exists(os.path.join(out_path, data_name, f'{name}_1024.png')))
        if not continue_processing:
            print(f"Skipping {data_name} because it already exists")
            return


    os.makedirs(os.path.join(out_path, data_name), exist_ok=True)
    min_max_vert_list = []
    scale_list = []
    for verts, faces, colors, name in zip(vert_list, face_list, color_list, name_list):
        # First rasterize the texture normally
        image, scale, min_max_vert = rasterize_polygon(verts, faces, colors, image_size=(1024, 1024))
        
        cv2.imwrite(os.path.join(out_path, data_name, f'{name}_1024.png'), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
        min_max_vert_list.append(min_max_vert)
        scale_list.append(scale)
    
    with open(os.path.join(out_path, data_name, "meta_stats.txt"), "w") as f:
        for name, min_max_vert, scale in zip(name_list, min_max_vert_list, scale_list):
            f.write(f"{name} {min_max_vert[0][0]} {min_max_vert[0][1]} {min_max_vert[0][2]} {min_max_vert[1][0]} {min_max_vert[1][1]} {min_max_vert[1][2]} {scale[0]} {scale[1]}\n")
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
    with ThreadPoolExecutor(max_workers=16) as executor:
        process_fn = partial(process_texture_file, out_path=out_path, chunk_id=chunk_id, data_path=data_path)
        list(tqdm(executor.map(process_fn, data_names), total=len(data_names)))