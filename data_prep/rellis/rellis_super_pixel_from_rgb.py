import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

from tqdm import tqdm

import argparse

if __name__ == "__main__":

    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Generate super pixel from rgb image")
    parser.add_argument("--data_root", type=str, help="Data root", required=True)
    parser.add_argument("--seqs", nargs='+', help="Rellis sequence number", required=True)

    # Parse the command line arguments to an object
    args = parser.parse_args()

    print("*********************************************")
    print("[i] start rellis_super_pixel_from_rgb.py")
    print("[i] data root: ", args.data_root)
    print("[i] sequences: ", args.seqs)

    for seq in args.seqs:

        seq_str = str(seq).zfill(5)   
        f = open(os.path.join(args.data_root, "Rellis-3D", seq_str, "poses.txt"), 'r')
        poses = f.readlines()
        f.close()
        
        save_root = os.path.join(args.data_root, "Rellis-3D-custom", str(seq).zfill(5), "super_pixel")
        os.makedirs(save_root, exist_ok=True)

        file_list = os.listdir(os.path.join(args.data_root, "Rellis-3D", seq_str, "pylon_camera_node"))
        for fn in tqdm(file_list):
            rgb_path = os.path.join(args.data_root, "Rellis-3D", seq_str, "pylon_camera_node", fn)
            image = img_as_float(io.imread(rgb_path))

            segments = slic(image, n_segments=100, sigma=5)

            img_mask = mark_boundaries(np.zeros_like(image), np.array(segments), outline_color=(0,0,0), color=(1,1,1))
            plt.imsave(os.path.join(save_root, fn[5:11] + ".png"), img_mask)

    print("[i] end rellis_super_pixel_from_rgb.py")
    print("*********************************************")