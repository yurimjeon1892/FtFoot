import os
import argparse
import numpy as np
import open3d as o3d
from PIL import Image

from tqdm import tqdm
import sys; sys.path.append('./')

from common.utils import rgb_read, pcd_read, save_image
from common.utils_plot import read_raw_sample_list, semkitti_label_to_color

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, help="Path to the rellis dataset", default="/storage2/public/rellis-3d")
    parser.add_argument("--seq", type=int, help="Rellis sequence number", default=4)
    
    parser.add_argument("--start_num", type=int, help="start frame num", default=350)
    parser.add_argument("--end_num", type=int, help="end frame num", default=400)
    
    parser.add_argument("--pcd_range", type=float, help="pcd range", default=30.)
    
    parser.add_argument("--meter_to_grid", type=int, help="meter-to-pixel", default=10) 
    parser.add_argument("--pixel_size", type=int, help="pixel-size", default=1) 
    
    parser.add_argument("--show_open3d", action='store_true')
    
    parser.add_argument("--save_traj_img", action='store_true')    
    parser.add_argument("--save_rgb_img", action='store_true')        
    parser.add_argument("--save_semseg_img", action='store_true')
    
    parser.add_argument("--save_valid_map", action='store_true')        
    parser.add_argument("--save_pcd_file", action='store_true')        
    
    parser.add_argument("--cost_path", type=str, help="Path to the cost map", default="")   
    
    args = parser.parse_args()
    
    raw_cam_img_size = (1200, 1920)
    
    sample_list, calib_dict = read_raw_sample_list(args)
    
    seq_str = str(args.seq).zfill(5)
    
    xyz_all, rgb_all, pose_all, cost_all = [], [], [], []
    for sample_one in tqdm(sample_list):
        pose_all.append(sample_one["P_oi"] ) 
        
        im_np = rgb_read(sample_one["image"])  
        
        pc_np = pcd_read(sample_one["point"])  
        pc_np_norm = np.linalg.norm(pc_np, axis=1)
        mask = pc_np_norm < args.pcd_range
        pc_np = pc_np[mask] 
        # print("pc_np", pc_np.shape)
        
        if args.save_semseg_img :
            label = np.fromfile(sample_one["label"], dtype=np.uint32)
            label = label.reshape((-1))
            label = label[mask]
            # print("label", label.shape)
        
        # in case you want to load some file..
        if args.cost_path != "":
            file_path_tmp = os.path.join(args.cost_path, seq_str + "_" + str(sample_one["fnum"]).zfill(6) + ".png")
            cost_np = rgb_read(file_path_tmp)
            cost_img = Image.fromarray(cost_np)
            cost_img = cost_img.resize((raw_cam_img_size[1], raw_cam_img_size[0]))
            cost_np = np.array(cost_img) / 255
        
        # depth_from_cartesian_pc_numpy
        pc_one = np.concatenate([pc_np.T[:3, :], np.ones((1, pc_np.shape[0]))], 0)
        pc_xyw = calib_dict["P"] @ calib_dict["Tr"] @ pc_one.copy()  
                
        pc_in_fov, pc_in_fov_clr, pc_in_fov_cost = [], [], []
        for i, xyw in enumerate(pc_xyw.T):     
            x = xyw[0]
            y = xyw[1] 
            w = xyw[2] 
            is_in_img = (
                w > 0 and 0 <= x < w * raw_cam_img_size[1] and 0 <= y < w * raw_cam_img_size[0]
            )        
            if is_in_img:
                pc_in_fov.append([pc_np[i, :3]])
                pc_in_fov_clr.append([im_np[int(y / w), int(x / w), :]])
                if args.save_semseg_img:
                    pc_in_fov_cost.append([label[i]])
                if args.cost_path != "":
                    pc_in_fov_cost.append([cost_np[int(y / w), int(x / w), :]])           
                    
        pc_in_fov = np.concatenate(pc_in_fov, 0)

        pc_in_fov_clr = np.concatenate(pc_in_fov_clr, 0)    
        if len(pc_in_fov_cost) > 0:
            pc_in_fov_cost = np.concatenate(pc_in_fov_cost, 0)  
        
        pc_in_fov = np.concatenate([pc_in_fov.T[:3, :], np.ones((1, pc_in_fov.shape[0]))], 0)
        pc_in_fov = sample_one["P_oi"] @ pc_in_fov
        xyz_all.append(pc_in_fov.T[:, :3])        
        
        rgb_all.append(pc_in_fov_clr)
        cost_all.append(pc_in_fov_cost)
        
    xyz_all = np.concatenate(xyz_all, 0)
    rgb_all = np.concatenate(rgb_all, 0) / 255
    cost_all = np.concatenate(cost_all, 0)
    
    pose_xyz_all = []
    for P_oi in pose_all:  
        pose_xyz_all.append(np.expand_dims(P_oi[:3, -1], 0))            
    pose_xyz_all = np.concatenate(pose_xyz_all, 0)

    max_x, min_x = np.max(pose_xyz_all[:, 0]) + args.pcd_range + 2.5, np.min(pose_xyz_all[:, 0]) - args.pcd_range / 2 - 2.5
    max_y, min_y = np.max(pose_xyz_all[:, 1]) + args.pcd_range / 2 + 2.5, np.min(pose_xyz_all[:, 1]) - args.pcd_range - 2.5
    H, W = int((max_x - min_x) * args.meter_to_grid), int((max_y - min_y) * args.meter_to_grid)  
    print(min_x, min_y)
    # exit()
    
    if args.save_semseg_img:
        pc_new_clr = []
        for i in range(cost_all.shape[0]):
            pc_new_clr.append([semkitti_label_to_color[cost_all[i]]])
        semseg_rgb_all = np.concatenate(pc_new_clr, 0) / 255 
        
    if args.cost_path != "":
        pc_new_clr = []
        for i in range(cost_all.shape[0]):
            pc_new_clr.append([cost_all[i]])
        cost_rgb_all = np.concatenate(pc_new_clr, 0)
           
    if not args.show_open3d:
        prefix = str(args.start_num).zfill(3) + "_" + str(args.end_num).zfill(3)
        save_path = os.path.join("../outputs", "map", prefix)
        os.makedirs(save_path, exist_ok=True)         
    else:                
        show_pcd = xyz_all
        show_clr = rgb_all
        if args.cost_path != "": show_clr = cost_rgb_all
    
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(show_pcd)
        pcd.colors = o3d.utility.Vector3dVector(show_clr)
        o3d.visualization.draw_geometries([pcd])         
        exit()
        
    ######################## pose ########################    
    if args.save_traj_img:        
                    
        footprint_map = draw_map(pose_xyz_all, np.zeros_like(pose_xyz_all), H, W, min_x, min_y, args.meter_to_grid, args.pixel_size)            
        save_name = os.path.join(save_path, prefix + "_footprint.png")            
        save_image(footprint_map, save_name)
        print("[i]", save_name)
        
        pos_npy = []        
        for i in range(pose_xyz_all.shape[0]):
            px, py = int((pose_xyz_all[i, 0] - min_x) * args.meter_to_grid) , int((pose_xyz_all[i, 1] - min_y) * args.meter_to_grid)
            pos_npy.append([[px, py]])
        pos_npy = np.concatenate(pos_npy, 0)
        save_name = os.path.join(save_path, prefix + "_footprint_path.npy")    
        np.save(save_name, pos_npy)        
        print("[i]", save_name)
        
    ######################## rgb ########################    
    if args.save_rgb_img:        
        rgb_map = draw_map(xyz_all, rgb_all, H, W, min_x, min_y, args.meter_to_grid, args.pixel_size)
        save_name = os.path.join(save_path, prefix + "_rgb.png") 
        save_image(rgb_map, save_name)
        print("[i]", save_name)
        
    ######################## point ########################
    if args.save_valid_map:            
        valid_ground = draw_map(xyz_all, np.ones_like(xyz_all), H, W, min_x, min_y, args.meter_to_grid, args.pixel_size, 0)            
        save_name = os.path.join(save_path, prefix + "_valid.png")         
        save_image(valid_ground, save_name)        
        print("[i]", save_name)
        
    ######################## semseg ########################    
    if args.save_semseg_img:        
        mask = cost_all != 0
        xyz_all = xyz_all[mask]
        semseg_rgb_all = semseg_rgb_all[mask]
        semseg_map = draw_map(xyz_all, semseg_rgb_all, H, W, min_x, min_y, args.meter_to_grid, args.pixel_size)
        save_name = os.path.join(save_path, prefix + "_semseg.png")  
        save_image(semseg_map, save_name)
        print("[i]", save_name)
    
    ######################## cost ######################## 
    if args.cost_path != "":
        cost_name = args.cost_path.split("/")[-1]
        cost_map = draw_map(xyz_all, cost_rgb_all, H, W, min_x, min_y, args.meter_to_grid, args.pixel_size)
        save_name = os.path.join(save_path, prefix + "_" + cost_name + ".png")  
        save_image(cost_map, save_name)
        print("[i]", save_name)
        
    ######################## pcd ########################
    if args.save_pcd_file:   
        
        xyz = xyz_all[:,0:3]     
        pcd = o3d.geometry.PointCloud()        
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals()
        save_name = os.path.join(save_path, prefix + ".pcd")
        o3d.io.write_point_cloud(save_name, pcd)
        print("[i]", save_name)
        
    return

def draw_map(xyz_all, rgb_all, H, W, min_x, min_y, meter_to_grid, pixel_size, canvas=1):
    out_map = np.zeros((H, W, 3)) + canvas
    for i in range(xyz_all.shape[0]):
        px, py = int((xyz_all[i, 0] - min_x) * meter_to_grid) , int((xyz_all[i, 1] - min_y) * meter_to_grid)
        if px < 0 or px > H or py < 0 or py > W: continue
        x_pixel_start, x_pixel_end = max(0, px - pixel_size), min(H, px + pixel_size)
        y_pixel_start, y_pixel_end = max(0, py - pixel_size), min(W, py + pixel_size)
        out_map[x_pixel_start:x_pixel_end, y_pixel_start : y_pixel_end, :] = rgb_all[i, :] 
    return out_map * 255

if __name__ == '__main__':
    main()