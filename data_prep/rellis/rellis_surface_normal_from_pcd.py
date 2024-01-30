import os
import argparse
import natsort 
import numpy as np
import open3d as o3d

from tqdm import tqdm

import sys; sys.path.append('./')

from common.utils_loader import pcd_read, get_cam_mtx, get_lidar2cam_mtx, sn_image_from_npy
from common.utils import save_image

if __name__ == "__main__":

    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Generate surface normal from point cloud")
    parser.add_argument("--data_root", type=str, help="Data root", required=True)
    parser.add_argument("--seqs", nargs='+', help="Rellis sequence number", required=True)

    # Parse the command line arguments to an object
    args = parser.parse_args()

    print("*********************************************")
    print("[i] start rellis_surface_normal_from_pcd.py")
    print("[i] data root: ", args.data_root)
    print("[i] sequences: ", args.seqs)

    raw_cam_img_size = [1200, 1920]

    for seq in args.seqs:
        seq = int(seq)

        calib = {}  
        Tr_fn = os.path.join(args.data_root, "Rellis_3D", str(seq).zfill(5), "transforms.yaml")
        RT = get_lidar2cam_mtx(Tr_fn)
        calib["Tr"] = RT
        calib["Tr_inv"] = np.linalg.inv(RT)

        P_fn = os.path.join(args.data_root, "Rellis-3D", str(seq).zfill(5), "camera_info.txt")
        P = get_cam_mtx(P_fn)
        P_eye = np.eye(4)
        P_eye[:3,:3] = P
        P_eye = P_eye
        calib["P"] = P_eye
        calib["P_inv"] = np.linalg.inv(P_eye)  
  
        f = open(os.path.join(args.data_root, "Rellis-3D", str(seq).zfill(5), "poses.txt"), 'r')
        poses = f.readlines()
        f.close()
        
        save_root = os.path.join(args.data_root, "Rellis-3D-custom", str(seq).zfill(5), "surface_normal")
        os.makedirs(save_root, exist_ok=True)
        
        save_root_clr = os.path.join(args.data_root, "Rellis-3D-custom", str(seq).zfill(5), "surface_normal_2")
        os.makedirs(save_root_clr, exist_ok=True)

        calib_mtx = calib["P"] @ calib["Tr"] 

        file_list = os.listdir(os.path.join(args.data_root, "Rellis-3D", str(seq).zfill(5), "os1_cloud_node_kitti_bin"))
        file_list = natsort.natsorted(file_list)
        for fn in tqdm(file_list):
            str_seq_i = fn[:-4]

            pc = pcd_read(os.path.join(args.data_root, "Rellis-3D", str(seq).zfill(5), "os1_cloud_node_kitti_bin", str_seq_i + '.bin'))
            pc = pc.T
            pc = np.concatenate([pc[:3, :], np.ones((1, pc.shape[1]))], 0)
            pc = calib_mtx @ pc   
            pc = pc[:3, :]

            d = []
            for idx, xyw in enumerate(pc.T):        
                x = xyw[0]
                y = xyw[1] 
                w = xyw[2] 
                is_in_img = (
                    w > 0 and 0 <= x < w * raw_cam_img_size[1] and 0 <= y < w * raw_cam_img_size[0]
                )        
                if is_in_img:
                    d.append([y / (w * 100), x / (w * 100), w ])
            d = np.array(d)

            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(d)

            source_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=50))
            # o3d.visualization.draw_geometries([source_pcd])

            normals_ = np.array(source_pcd.normals)
            axis_ = np.array([0, 0, -1])
            dot_ = axis_.T @ normals_.T

            for i in range(dot_.shape[0]):
                if dot_[i] <= 0.0: source_pcd.normals[i] *= -1
            # o3d.visualization.draw_geometries([source_pcd])

            points_ = np.array(source_pcd.points)
            normals_ = np.array(source_pcd.normals)

            sn_npy = []
            for i in range(points_.shape[0]):
                x, y = int(points_[i, 0] * 100), int(points_[i, 1] * 100)
                sn_npy.append([x, y, normals_[i, 0], normals_[i, 1], normals_[i, 2]])
            sn_npy = np.array(sn_npy)

            np.save(os.path.join(save_root, str_seq_i + ".npy"), sn_npy) 
            # print(os.path.join(save_root, str_seq_i + ".npy"))
            
            # # this is just for visualization. if you dont't want it, comment out the line below            
            # sn_img = sn_image_from_npy(sn_npy, raw_cam_img_size, px=9)      
            # sn_img = (sn_img + 1) * 127
            # save_image(sn_img, os.path.join(save_root_clr, str_seq_i + ".png"))

    print("[i] end rellis_surface_normal_from_pcd.py")
    print("*********************************************")