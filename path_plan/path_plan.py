import os
import sys; sys.path.append('./')

import argparse

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import datetime

from loader import get_lp_planner
from misc import *

from common.utils import pose_read, rgb_read

def main():

    parser = argparse.ArgumentParser()
        
    parser.add_argument('--local_planner_type', type=str, default='TRRTSTAR', help='RRT | RRTSTAR | TRRTSTAR')
    parser.add_argument('--max_path_iter', type=int, default=1000)
    parser.add_argument('--path_tick', type=float, default=0.1)
    parser.add_argument('--max_extend_length', type=float, default=10.0)
    parser.add_argument('--goal_p_th', type=float, default=1)
    parser.add_argument('--t_weight', type=float, default=20.)
    parser.add_argument('--cost_th', type=float, default=0.5)
    parser.add_argument('--animation', action='store_true', default=False)
    
    parser.add_argument('--goal_sample_rate', type=float, default=0.1)
    
    parser.add_argument('--bias_sampling', action='store_true', default=False)
    parser.add_argument('--use_constraint', action='store_true', default=False)
    parser.add_argument('--try_goal', action='store_true', default=False)
    
    parser.add_argument("--data_root", type=str, help="Path to the rellis dataset", default="/storage2/public/rellis-3d")
    parser.add_argument("--seq", type=int, help="Rellis sequence number", default=4)
    
    parser.add_argument("--start_num", type=int, help="start frame num", default=400)
    parser.add_argument("--end_num", type=int, help="end frame num", default=900)
    
    parser.add_argument("--pcd_range", type=float, help="pcd range", default=30.)
    
    parser.add_argument("--meter_to_pixel", type=int, help="meter-to-pixel", default=10) 

    parser.add_argument('--cost_map_root', type=str, default='../outputs/map')
    parser.add_argument('--path_root', type=str, default='../outputs/path')
    
    parser.add_argument('--cost_map_name', type=str, required=True, help='*.png')
        
    args = parser.parse_args()
    
    cost_np, bound = read_cost_map(args)    
    start, goal = read_start_goal_pos(args)   
    
    make_path(args, start, goal, cost_np, bound)       
    return    

def read_start_goal_pos(args, offset_num=50):
            
    seq_str = str(args.seq).zfill(5)   
    f = open(os.path.join(args.data_root, "Rellis-3D", seq_str, "poses.txt"), 'r')
    poses_all = f.readlines()
    f.close()
    
    poses_all = poses_all[args.start_num:args.end_num]    
    
    pose_xyz_all = []
    for pose_ in poses_all:
        P_oi = pose_read(pose_)
        pose_xyz_all.append(np.expand_dims(P_oi[:3, -1], 0))            
    pose_xyz_all = np.concatenate(pose_xyz_all, 0)

    min_x = np.min(pose_xyz_all[:, 0]) - args.pcd_range / 2 - 2.5
    min_y = np.min(pose_xyz_all[:, 1]) - args.pcd_range - 2.5
        
    x0, y0 = int((pose_xyz_all[offset_num, 0] - min_x) * args.meter_to_pixel) , int((pose_xyz_all[offset_num, 1] - min_y) * args.meter_to_pixel)
    x1, y1 = int((pose_xyz_all[-1, 0] - min_x) * args.meter_to_pixel) , int((pose_xyz_all[-1, 1] - min_y) * args.meter_to_pixel)
    
    print("[i] start: ", x0, y0)
    print("[i] goal: ", x1, y1)
    
    return (x0, y0, 0), (x1, y1, 0)

def read_cost_map(args):    
    
    prefix = str(args.start_num).zfill(3) + "_" + str(args.end_num).zfill(3) 
    valid_ground_path = os.path.join(args.cost_map_root, prefix, prefix + "_valid.png")
    cost_map_path = os.path.join(args.cost_map_root, args.cost_map_name)
    
    cost_np = rgb_read(cost_map_path) / 255    
    ground_np = rgb_read(valid_ground_path) / 255      
    valid_cost_np = np.where(ground_np == 1, cost_np, np.nan)
        
    bound = (0, valid_cost_np.shape[0], 0, valid_cost_np.shape[1])  
    
    return valid_cost_np[:, :, 0], bound
    
def get_height(point, height_map):
    cx, cy = int(point[0]), int(point[1])    
    pz = height_map[cx, cy]
    if np.isnan(pz): pz = -np.inf
    return pz

def get_height_fn(height_map):   
    return lambda x: np.array([*x, get_height(x, height_map)])

def get_cost(point, cost_map, precision=8):
    cx, cy = int(point[0]), int(point[1])    
    cost = round(cost_map[cx,cy], precision)
    if np.isnan(cost): cost = 1.0
    return cost

def get_cost_fn(local_cost_map):
    return lambda x: get_cost(x, local_cost_map)

def get_value_fn(self, args, local_value_map):
    return lambda x: self.get_value_from_map(x, args, local_value_map)

def is_point_valid(point, lp_planner):

    height = lp_planner.height_fn(point)[2]
    tau = lp_planner.cost_fn(point)

    if np.isnan(height) or tau > 0.9 :
        print("point is not valid")
        print(f'point {point}, height : {np.nan}, trav : {tau}')     
    return

def set_local_planner(args, start, goal, bound, cost_map, planner):
    
    cost_map[start[0], start[1]] = 0
    cost_map[goal[0], goal[1]] = 0

    planner.bounds = bound
    planner.goal_p_th = args.goal_p_th
    planner.t_weight = args.t_weight
    planner.set_start_goal(start, goal)
    planner.height_fn = get_height_fn(cost_map)
    planner.cost_fn = get_cost_fn(cost_map)
    return

def make_path(args, start, goal, cost_map, bound):
    
    # polygons = [np.array([
    #     [0.5, 0.5],
    #     [-0.5, 0.5],
    #     [-0.5, -0.5],
    #     [0.5, -0.5]
    # ]) * args.meter_to_pixel] # 1m x 1m margin for robot
    
    lp_planner = get_lp_planner(type=args.local_planner_type,
                                max_extend_length=args.max_extend_length,
                                goal_sample_rate=args.goal_sample_rate,     
                                max_iter=args.max_path_iter,                           
                                path_tick=args.path_tick,
                                animation=args.animation,
                                try_goal=args.try_goal,
                                cost_th=args.cost_th,
                                bias_sampling=args.bias_sampling,
                                # polygons=polygons
                                )

    set_local_planner(args, start, goal, bound, cost_map, lp_planner)   

    is_point_valid(start, lp_planner)
    is_point_valid(goal, lp_planner)

    optimal_path, min_cost = lp_planner.plan()

    if optimal_path is not None:
        goal_iter = lp_planner.goal_iter

    if optimal_path is not None:
        prYellow(f'\tMinimum cost: {min_cost:.2f}, goal_iter : {goal_iter:.0f}')
        prYellow(f"\tpath: {optimal_path[:5]}, shape = {optimal_path.shape}")  

        # plt.imshow(trav_map.T)
        # plt.show()
        
        # if args.save:
        plt.figure(figsize=(6,6))
        lp_planner.plot_scene(start, goal, bound)
        lp_planner.draw_graph()
        if optimal_path is not None:
            lp_planner.plot_path(optimal_path)

        ax = plt.gca()
        ax.grid(False)
        w,h = cost_map.shape
        x = np.arange(0, w, 1)
        y = np.arange(0, h, 1)
        XX, YY = map(np.transpose, np.meshgrid(x,y))
        im = ax.pcolormesh(XX, YY, cost_map, alpha=0.5)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.xlim((bound[0], bound[1]))
        plt.ylim((bound[2], bound[3]))
        ax.set_aspect('equal')
        
        prefix = str(args.start_num).zfill(3) + "_" + str(args.end_num).zfill(3)     
        fname = args.cost_map_name.split("/")[-1][:-4] + "_" + datetime.datetime.today().strftime('%H:%M:%S')    
        
        os.makedirs(os.path.join(args.path_root, prefix), exist_ok=True)        
        
        plt.savefig(os.path.join(args.path_root, prefix, fname + '_path.png'))
        np.save(os.path.join(args.path_root, prefix, fname + '_path.npy'), optimal_path)
        print(fname + '_path.png')
        
    else:
        prRed('Local Planner could not find optimial path')
        
    return optimal_path

if __name__=='__main__':    
    main()