import os
import numpy as np
import argparse
import natsort 

import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm

import sys; sys.path.append('./')

from common.utils import minmax_color_img_from_img_numpy, save_image
from common.utils_loader import read_calib_file_orfd, pose_read_orfd, bin_read

import open3d as o3d   

def make_foot_prints_pc(Pijs, num_cline=10, rad_circle=1):
    
    origin_os1 = []
    for i in range(360):
        rad_ = 2 * np.pi * i / 360 
        for r in range(num_cline + 1):
            rr = (r / num_cline) * 0.5 * rad_circle
            origin_os1.append([rr * np.cos(rad_), rr * np.sin(rad_) + rad_circle, 0, 1])
    origin_os1 = np.array(origin_os1)  
    
    foot_prints_ = []
    for Pij in Pijs:        
        pnts = Pij @ origin_os1.T
        pnts[2, :] = -1.25
        foot_prints_.append(pnts.T[:, :3])
    foot_prints_pc = np.concatenate(foot_prints_, 0)
    
    return foot_prints_pc
    

def plot_foot_print(foot_prints_pc, calib_mtx, raw_cam_img_size, binary=True):
    
    foot_print_map = np.zeros(raw_cam_img_size, dtype=float)
    
    foot_prints_one = np.concatenate([foot_prints_pc, np.ones((foot_prints_pc.shape[0], 1))], -1)
    cj_on_Pi = calib_mtx @ foot_prints_one.T
    for _, xyw in enumerate(cj_on_Pi.T):        
        x = xyw[0]
        y = xyw[1] 
        w = xyw[2] 
        is_in_img = (
            w > 0 and 0 <= x < w * raw_cam_img_size[1] and 0 <= y < w * raw_cam_img_size[0]
        )        
        if is_in_img:
            xx, yy = int(x / (w)), int(y / (w))
            if binary: foot_print_map[yy, xx] = 1
            else: foot_print_map[yy, xx] = w

    if binary: foot_print_map = np.tile(np.expand_dims(foot_print_map, -1), (1, 1, 3))

    return foot_print_map


if __name__ == "__main__":

    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Plot footprint from pose")
    parser.add_argument("--data_root", type=str, help="Data root", required=True)
    parser.add_argument("--mode", type=str, help="ORFD mode: training/validation", required=True)
    parser.add_argument("--seq", type=str, help="", required=True)
    parser.add_argument("--num_pose", type=int, help="", default=75)

    # Parse the command line arguments to an object
    args = parser.parse_args()

    print("*********************************************")
    print("[i] start orfd_plot_footprint.py")
    print("[i] data root: ", args.data_root)
    print("[i] mode: ", args.mode)
    print("[i] seq: ", args.seq)
    print("[i] num_pose: ", args.num_pose)
    
    raw_cam_img_size = (720, 1280)
    
    save_root = os.path.join(args.data_root, "ORFD-custom", args.mode, "foot_print")
    os.makedirs(save_root, exist_ok=True)
    save_root_2 = os.path.join(args.data_root, "ORFD-custom", args.mode, "foot_print_2")
    os.makedirs(save_root_2, exist_ok=True)

    seq_dict = {}
    data_path = os.path.join(args.data_root, "Final_Dataset", args.mode, "image_data")
    file_list = natsort.natsorted(os.listdir(data_path))
    for fname in file_list:
        k = fname[:8]
        if k not in seq_dict.keys(): seq_dict[k] = []            
        seq_dict[k].append(fname[:-4])
        
    sample_list = []        
    for seq_name in seq_dict.keys():    
        
        if seq_name != args.seq: continue
        
        if not os.path.isfile(os.path.join(args.data_root, "ORFD-custom", args.mode, "pose", "pose_" + seq_name + ".csv")): continue
        
        f = open(os.path.join(args.data_root, "ORFD-custom", args.mode, "pose", "pose_" + seq_name + ".csv"), 'r')
        poses_all = f.readlines()
        f.close()
        
        save_root_seq = os.path.join(args.data_root, "ORFD-custom", args.mode, "foot_print_2", seq_name)        
        os.makedirs(save_root_seq, exist_ok=True)
        
        description = "[i] seq: " + seq_name
        for i, str_frame in enumerate(tqdm(seq_dict[seq_name]), desc=description):
            
            # if i < 200 : continue
            # if i > len(poses_all) - 50 : continue
            
            calib_mtx = read_calib_file_orfd(os.path.join(args.data_root, "Final_Dataset", args.mode, 'calib', str_frame +'.txt'))
            image = io.imread(os.path.join(args.data_root, "Final_Dataset", args.mode, "image_data", str_frame +'.png'))
            # pc = bin_read(os.path.join(args.data_root, "Final_Dataset", args.mode, 'lidar_data', str_frame +'.bin')) 
            
            # pose of i
            Pijs = []
            P_oi = pose_read_orfd(poses_all[i])            
            for seq_j in range(i, min(i + args.num_pose, len(poses_all))):
                P_oj = pose_read_orfd(poses_all[seq_j])
                P_ij = np.linalg.inv(P_oi) @ P_oj # 4x4
                Pijs.append(P_ij)
                
            foot_prints_pc = make_foot_prints_pc(Pijs, num_cline=10, rad_circle=0.5)    
            
            # pc = np.concatenate([foot_prints_pc[:, :3], pc], 0)    
            # pcd = o3d.geometry.PointCloud()  
            # pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
            # o3d.visualization.draw_geometries([pcd])
            # exit()
            
            fp_np = plot_foot_print(foot_prints_pc, calib_mtx, raw_cam_img_size, binary=True)     
            if np.sum(fp_np[:, :, 0]) == 0: continue 
            
            plt.imsave(os.path.join(save_root, str_frame + ".png"), fp_np)
            
            # # this is just for visualization. if you dont't want it, comment out the line below                
            # fp_np = plot_foot_print(foot_prints_pc, calib_mtx, raw_cam_img_size, binary=False)
            # d_img, mask = minmax_color_img_from_img_numpy(fp_np, plt.cm.plasma, px=1, valid_mask=True)   
            # image[mask] = d_img[mask]    
            # save_image(image, os.path.join(save_root_seq, str_frame + ".png"))

    print("[i] end orfd_plot_footprint.py")
    print("*********************************************")