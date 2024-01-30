import os
import numpy as np
import argparse
import natsort 
import matplotlib.pyplot as plt
from skimage import io

import sys; sys.path.append('./')

from common.utils_loader import read_calib_file_orfd, pose_read_orfd, bin_read
from common.utils import save_image, minmax_color_img_from_img_numpy

import open3d as o3d

def search_for_accumulation(data_path, mode, seq_list, poses, seq_i, seq_sample_num, P_oi, stride):

    P_io = np.linalg.inv(P_oi)

    pc_np_list = []

    counter = 0
    while len(pc_np_list) < 10:
        counter += 1
        seq_j = seq_i + stride * counter
        if seq_j < 0 or seq_j >= seq_sample_num:
            break

        str_seq_j = seq_list[seq_j]
        # print('str_seq_ij: ', seq_i, str_seq_j)
        
        pc_j = bin_read(os.path.join(data_path, mode, 'lidar_data', str_seq_j + '.bin'))
        pc_j = pc_j.T
        P_oj = pose_read_orfd(poses[seq_j])
        P_ij = P_io @ P_oj

        pc_j = np.concatenate((pc_j[:3, :], np.ones((1, pc_j.shape[1]), dtype=pc_j.dtype)), axis=0)
        pc_j = P_ij @ pc_j
        pc_j = pc_j[:3, :]

        pc_np_list.append(pc_j)

    return pc_np_list

def get_accumulated_pc(data_path, mode, seq_list, poses, seq_i):
    
    pc_np = bin_read(os.path.join(data_path, mode, 'lidar_data', str_frame +'.bin')) 
    pc_np = pc_np.T
    # shuffle the point cloud data, this is necessary!
    pc_np = pc_np[:, np.random.permutation(pc_np.shape[1])]
    pc_np = pc_np[:3, :]  # 3xN

    pc_np_list = [pc_np]

    # pose of i
    seq_sample_num = len(poses)
    P_oi = pose_read_orfd(poses[seq_i])   

    # search for previous
    prev_pc_np_list = search_for_accumulation(data_path, mode, seq_list, poses, seq_i, seq_sample_num,
                                              P_oi, -1)
    # search for next
    next_pc_np_list = search_for_accumulation(data_path, mode, seq_list, poses, seq_i, seq_sample_num,
                                              P_oi, 1)

    pc_np_list = pc_np_list + prev_pc_np_list + next_pc_np_list

    pc_np = np.concatenate(pc_np_list, axis=1)

    return pc_np.T

def plot_depth(calib_mtx, pcd, raw_cam_img_size):
    depth = np.zeros(raw_cam_img_size, dtype=float)
    pcd_ones = np.concatenate([pcd, np.ones((pcd.shape[0], 1))], -1)     
        
    xyw_ = calib_mtx @ pcd_ones.T
    for idx, xyw in enumerate(xyw_.T):        
        x = xyw[0]
        y = xyw[1] 
        w = xyw[2] 
        is_in_img = (
            w > 0 and 0 <= x < w * raw_cam_img_size[1] and 0 <= y < w * raw_cam_img_size[0]
        )        
        if is_in_img:
            xx, yy = int(x / (w)), int(y / (w))
            depth[yy, xx] = w
            
    return depth


if __name__ == "__main__":

    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Generate depth from point cloud")
    parser.add_argument("--data_root", type=str, help="Data root", required=True)
    parser.add_argument("--mode", type=str, help="ORFD mode: training/validation/testing", required=True)

    # Parse the command line arguments to an object
    args = parser.parse_args()

    print("*********************************************")
    print("[i] start orfd_depth_from_pcd.py")
    print("[i] data root: ", args.data_root)
    print("[i] mode: ", args.mode)
    
    raw_cam_img_size = (720, 1280)
    
    save_root = os.path.join(args.data_root, "ORFD-custom", args.mode, "depth")
    os.makedirs(save_root, exist_ok=True)

    seq_dict = {}
    data_path = os.path.join(args.data_root, "Final_Dataset", args.mode, "lidar_data")
    file_list = natsort.natsorted(os.listdir(data_path))
    for fname in file_list:
        k = fname[:8]
        if k not in seq_dict.keys(): seq_dict[k] = []            
        seq_dict[k].append(fname[:-4])
        
    sample_list = []        
    for seq_name in seq_dict.keys():    
        if  not os.path.isfile(os.path.join(args.data_root, "ORFD-custom", args.mode, "pose", "pose_" + seq_name + ".csv")): continue
        
        f = open(os.path.join(args.data_root, "ORFD-custom", args.mode, "pose", "pose_" + seq_name + ".csv"), 'r')
        poses_all = f.readlines()
        f.close()
                
        for i, str_frame in enumerate(seq_dict[seq_name]):
            
            if i != 400 : continue
            
            # pc = bin_read(os.path.join(args.data_root, "Final_Dataset", args.mode, 'lidar_data', str_frame +'.bin')) 
            pc = get_accumulated_pc(args.data_root, "Final_Dataset", args.mode, seq_dict[seq_name], poses_all, i)            
            
            calib_mtx = read_calib_file_orfd(os.path.join(args.data_root, "Final_Dataset", args.mode, 'calib', str_frame +'.txt'))            
            image = io.imread(os.path.join(args.data_root, "Final_Dataset", args.mode, "image_data", str_frame +'.png'))
            
            d_np = plot_depth(calib_mtx, pc, raw_cam_img_size)            
            d_img, mask = minmax_color_img_from_img_numpy(d_np, plt.cm.plasma, px=2, valid_mask=True)   
            image[mask] = d_img[mask]
            
            save_image(image, os.path.join(args.data_root, "ORFD-custom", args.mode, "depth", str_frame + ".png"))
            print(os.path.join(args.data_root, "ORFD-custom", args.mode, "depth", str_frame + ".png"))
            
            pcd = o3d.geometry.PointCloud()  
            pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
            o3d.visualization.draw_geometries([pcd])
            exit()
            

    print("[i] end orfd_depth_from_pcd.py")
    print("*********************************************")