import os
import cv2
import natsort 

import numpy as np
import open3d as o3d
import argparse
from tqdm import tqdm

import sys; sys.path.append('./')

from common.utils_loader import sn_image_from_npy
from common.utils import save_image

if __name__ == "__main__":

    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Generate surface normal from depth image")
    parser.add_argument("--data_root", type=str, help="Data root", required=True)
    parser.add_argument("--mode", type=str, help="ORFD mode: training/validation/testing", required=True)

    # Parse the command line arguments to an object
    args = parser.parse_args()

    print("*********************************************")
    print("[i] start orfd_surface_normal_from_depth.py")
    print("[i] data root: ", args.data_root)
    print("[i] mode: ", args.mode)
    
    raw_cam_img_size = (720, 1280)
           
    save_root = os.path.join(args.data_root, "ORFD-custom", args.mode, "surface_normal")
    os.makedirs(save_root, exist_ok=True)
    
    save_root_seq = os.path.join(args.data_root, "ORFD-custom", args.mode, "surface_normal_2")        
    os.makedirs(save_root_seq, exist_ok=True)
    
    depth_path = os.path.join(args.data_root, "Final_Dataset", args.mode, "sparse_depth")

    file_list = natsort.natsorted(os.listdir(depth_path))
    for fn in tqdm(file_list):
        depth_image = cv2.imread(os.path.join(depth_path, fn), cv2.IMREAD_ANYDEPTH)
        depth_image = np.array(depth_image)

        d = []
        for h in range(depth_image.shape[0]):
            for w in range(depth_image.shape[1]):
                depth = depth_image[h, w] / (2 ** 12)
                if depth == 0 or depth > 2 ** 3: continue
                elif h < depth_image.shape[0] * 2. / 5.: continue
                d.append([h / 100., w / 100., depth])        
        d = np.array(d)

        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(d)
        source_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=50))
        # o3d.visualization.draw_geometries([source_pcd])
        # exit()

        normals_ = np.array(source_pcd.normals)
        axis_ = np.array([0, 0, -1])
        dot_ = axis_.T @ normals_.T

        for i in range(dot_.shape[0]):
            if dot_[i] <= 0.0: source_pcd.normals[i] *= -1
        # o3d.visualization.draw_geometries([source_pcd])
        # exit()

        points_ = np.array(source_pcd.points)
        normals_ = np.array(source_pcd.normals)

        sn_npy = []
        for i in range(points_.shape[0]):
            x, y = int(points_[i, 0] * 100), int(points_[i, 1] * 100)
            sn_npy.append([x, y, normals_[i, 0], normals_[i, 1], normals_[i, 2]])
        sn_npy = np.array(sn_npy)

        np.save(os.path.join(save_root, fn[:-4] + ".npy"), sn_npy) 
        
        # # this is just for visualization. if you dont't want it, comment out the line below        
        # sn_img = sn_image_from_npy(sn_npy, raw_cam_img_size, px=3)      
        # sn_img = (sn_img + 1) * 127
        # save_image(sn_img, os.path.join(save_root_seq, fn[:-4] + ".png"))
        

    print("[i] end orfd_surface_normal_from_depth.py")
    print("*********************************************")