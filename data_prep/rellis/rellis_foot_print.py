import os
import numpy as np
import argparse
import natsort

from tqdm import tqdm

import sys; sys.path.append('./')

import matplotlib.pyplot as plt

from common.utils import minmax_color_img_from_img_numpy, save_image
from common.utils_loader import pose_read, rgb_read, get_cam_mtx, get_lidar2cam_mtx

def make_sample_dataset(args):  

    calib_dict, cam_name_dict = {}, {}
    for seq in args.seqs:

        calib_dict[seq] = {}  
        Tr_fn = os.path.join(args.data_root, "Rellis_3D", str(seq).zfill(5), "transforms.yaml")
        RT = get_lidar2cam_mtx(Tr_fn)
        calib_dict[seq]["Tr"] = RT
        calib_dict[seq]["Tr_inv"] = np.linalg.inv(RT)

        P_fn = os.path.join(args.data_root, "Rellis-3D", str(seq).zfill(5), "camera_info.txt")
        P = get_cam_mtx(P_fn)
        P_eye = np.eye(4)
        P_eye[:3,:3] = P
        P_eye = P_eye
        calib_dict[seq]["P"] = P_eye
        calib_dict[seq]["P_inv"] = np.linalg.inv(P_eye)  

        cam_name_dict[seq] = {}
        file_list = os.listdir(os.path.join(args.data_root, "Rellis-3D", str(seq).zfill(5), "pylon_camera_node"))
        for file_one in file_list:
            fn = file_one.split("/")[-1]
            cam_name_dict[seq][fn[5:11]] = fn[:-4]

    sample_list = []
    for seq in args.seqs:         

        seq_str = str(seq).zfill(5)   
        f = open(os.path.join(args.data_root, "Rellis-3D", seq_str, "poses.txt"), 'r')
        poses_all = f.readlines()
        f.close()

        file_list = os.listdir(os.path.join(args.data_root, "Rellis-3D", seq_str, "os1_cloud_node_kitti_bin"))
        file_list = natsort.natsorted(file_list)
        for fn in file_list:
            int_frame = int(fn[:-4])

            # pose of i
            P_oi = pose_read(poses_all[int_frame])
            Pijs = []
            for fr_n in range(int_frame, min(int_frame + 150, len(poses_all))):
                P_oj = pose_read(poses_all[fr_n])
                P_ij = np.linalg.inv(P_oi) @ P_oj # 4x4
                Pijs.append(P_ij)

            str_frame = str(int_frame).zfill(6)
            indiv_sample = {"image": os.path.join(args.data_root, "Rellis-3D", seq_str, "pylon_camera_node", cam_name_dict[seq][str_frame] + ".jpg"),
                            "point": os.path.join(args.data_root, "Rellis-3D", seq_str, "os1_cloud_node_kitti_bin", str_frame + '.bin'),
                            "calib": calib_dict[seq],
                            "Pijs": Pijs,
                            "save_path": os.path.join(args.data_root, "Rellis-3D-custom", seq_str), 
                            "fname": str_frame}
            sample_list.append(indiv_sample)
    return sample_list, calib_dict

def make_foot_prints_pc(Pijs, num_cline=10, rad_circle=1):
    
    origin_os1 = []
    for i in range(360):
        rad_ = 2 * np.pi * i / 360 
        for r in range(num_cline + 1):
            rr = (r / num_cline) * 0.5 * rad_circle
            # origin_os1.append([rr * np.cos(rad_), rr * np.sin(rad_) + rad_circle, 0, 1])
            origin_os1.append([rr * np.cos(rad_) + 1, rr * np.sin(rad_), -1, 1])
    origin_os1 = np.array(origin_os1)  
    
    foot_prints_ = []
    for Pij in Pijs:        
        pnts = Pij @ origin_os1.T
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
            xx, yy = int(x / w), int(y / w)
            if binary: foot_print_map[yy, xx] = 1
            else: foot_print_map[yy, xx] = w

    if binary: foot_print_map = np.tile(np.expand_dims(foot_print_map, -1), (1, 1, 3))

    return foot_print_map


if __name__ == "__main__":

    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Plot footprint from pose")
    parser.add_argument("--data_root", type=str, help="Data root", required=True)
    parser.add_argument("--seqs", nargs='+', help="Rellis sequence number", required=True)

    # Parse the command line arguments to an object
    args = parser.parse_args()

    print("*********************************************")
    print("[i] start rellis_foot_print.py")
    print("[i] data root: ", args.data_root)
    print("[i] sequences: ", args.seqs)
    
    raw_cam_img_size = (1200, 1920)

    sample_list, calib_dict = make_sample_dataset(args)
    
    for seq in args.seqs:   
        seq_str = str(seq).zfill(5)  
        save_root = os.path.join(args.data_root, "Rellis-3D-custom", seq_str, "foot_print")
        os.makedirs(save_root, exist_ok=True)
        save_root_2 = os.path.join(args.data_root, "Rellis-3D-custom", seq_str, "foot_print_2")
        os.makedirs(save_root_2, exist_ok=True)

    sample_item_list = []
    for sample_one in tqdm(sample_list):
        
        image = rgb_read(sample_one["image"])        
        calib_mtx = sample_one["calib"]["P"] @ sample_one["calib"]["Tr"] 
        
        foot_prints_pc = make_foot_prints_pc(sample_one["Pijs"], num_cline=10, rad_circle=0.5)    
        
        fp_np = plot_foot_print(foot_prints_pc, calib_mtx, raw_cam_img_size, binary=True)            
        save_image(fp_np * 255, os.path.join(sample_one["save_path"], "foot_print", sample_one["fname"] + ".png"))
        
        # # this is just for visualization. if you dont't want it, comment out the line below        
        # fp_np = plot_foot_print(foot_prints_pc, calib_mtx, raw_cam_img_size, binary=False)
        # d_img, mask = minmax_color_img_from_img_numpy(fp_np, plt.cm.plasma, px=1, valid_mask=True)   
        # image[mask] = d_img[mask]    
        # save_image(image, os.path.join(sample_one["save_path"], "foot_print_2", sample_one["fname"] + ".png"))
        

    print("[i] end rellis_foot_print.py")
    print("*********************************************")