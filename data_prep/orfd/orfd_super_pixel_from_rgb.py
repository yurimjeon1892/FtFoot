import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

import argparse
from tqdm import tqdm

if __name__ == "__main__":

    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Generate super pixel from rgb image")
    parser.add_argument("--data_root", type=str, help="Dataset directory", required=True)
    parser.add_argument("--mode", type=str, help="training/validation/testing", required=True)

    # Parse the command line arguments to an object
    args = parser.parse_args()

    print("*********************************************")
    print("[i] start orfd_super_pixel_from_rgb.py")
    print("[i] dataset directory: ", args.data_root)
    print("[i] mode: ", args.mode)
    
    save_root = os.path.join(args.data_root, "ORFD-custom", args.mode, "super_pixel")
    os.makedirs(save_root, exist_ok=True)

    file_list = os.listdir(os.path.join(args.data_root, "Final_Dataset", args.mode, "image_data"))
    for fn in tqdm(file_list):
        rgb_path = os.path.join(args.data_root, "Final_Dataset", args.mode, "image_data", fn)
        image = img_as_float(io.imread(rgb_path))

        segments = slic(image, n_segments=100, sigma=5)

        img_mask = mark_boundaries(np.zeros_like(image), np.array(segments), outline_color=(0,0,0), color=(1,1,1))
        plt.imsave(os.path.join(save_root, fn), img_mask)
        # print(os.path.join(save_root, fn))

    print("[i] end orfd_super_pixel_from_rgb.py")
    print("*********************************************")