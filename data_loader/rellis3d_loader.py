import os
import numpy as np
import torch.utils.data as data
import random
import natsort

from common.utils_loader import *

__all__ = ['RELLIS_3D']

class RELLIS_3D(data.Dataset):

    def __init__(self, cfg, mode="test"):

        self.mode = mode
        self.data_path = cfg.data_root
        self.load_interval = cfg.load_interval
        
        self.raw_cam_img_size = cfg.raw_cam_img_size        
        self.ratio = cfg.ratio
        
        self.input_img_size = (360, 640)
        
        if mode == "train":
            self.num_samples = cfg.num_train_samples
            self.seqs = [0, 1, 2, 3]
        elif mode == 'valid':
            self.num_samples = cfg.num_val_samples
            self.seqs = [4]
        elif mode == 'test':
            self.num_samples = cfg.num_test_samples       
            self.seqs = [4]
        else:
            print('wrong mode: ', mode)
            exit()
        
        self.frame_crop_idx = { # valid frames which contains footprints, start frame / end frame
            0: [100, 2535],
            1: [50, 2045],
            2: [130, 4035],
            3: [20, 1855],
            4: [100, 1860]
        }
        
        self.samples = self.make_sample_dataset()
        
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.data_path + "\n"))

    def __len__(self):
        return len(self.samples)

    def make_sample_dataset(self):  
        
        calib_dict, cam_name_dict = {}, {}
        for seq in self.seqs:

            calib_dict[seq] = {}  
            Tr_fn = os.path.join(self.data_path, "Rellis_3D", str(seq).zfill(5), "transforms.yaml")
            RT = get_lidar2cam_mtx(Tr_fn)
            calib_dict[seq]["Tr"] = RT
            calib_dict[seq]["Tr_inv"] = np.linalg.inv(RT)

            P_fn = os.path.join(self.data_path, "Rellis-3D", str(seq).zfill(5), "camera_info.txt")
            P = get_cam_mtx(P_fn)
            P_eye = np.eye(4)
            P_eye[:3,:3] = P
            P_eye = P_eye
            calib_dict[seq]["P"] = P_eye
            calib_dict[seq]["P_inv"] = np.linalg.inv(P_eye)  

            cam_name_dict[seq] = {}
            file_list = os.listdir(os.path.join(self.data_path, "Rellis-3D", str(seq).zfill(5), "pylon_camera_node"))
            for file_one in file_list:
                fn = file_one.split("/")[-1]
                cam_name_dict[seq][fn[5:11]] = fn[:-4] 

        sample_list = []
        for seq in self.seqs:   
            seq_str = str(seq).zfill(5)  
            file_list = os.listdir(os.path.join(self.data_path, "Rellis-3D", seq_str, "os1_cloud_node_kitti_bin"))
            file_list = natsort.natsorted(file_list)
            
            file_list = file_list[self.frame_crop_idx[seq][0]:self.frame_crop_idx[seq][1]] # first / last frames don't have footprints..

            for fn in file_list:
                str_frame = fn[:-4]
                if self.mode != "test":
                    indiv_sample = {"image": os.path.join(self.data_path, "Rellis-3D", seq_str, "pylon_camera_node", cam_name_dict[seq][str_frame] + ".jpg"),
                                    "point": os.path.join(self.data_path, "Rellis-3D", seq_str, "os1_cloud_node_kitti_bin", str_frame + '.bin'),
                                    "sn": os.path.join(self.data_path, "Rellis-3D-custom", seq_str, "surface_normal", str_frame + '.npy'),     
                                    "sp": os.path.join(self.data_path, "Rellis-3D-custom", seq_str, "super_pixel", str_frame + ".png"),  
                                    "fp": os.path.join(self.data_path, "Rellis-3D-custom", seq_str, "foot_print", str_frame + ".png"),                                      
                                    "calib": calib_dict[seq],
                                    "fname": seq_str + '_' + str_frame}
                else:
                    indiv_sample = {"image": os.path.join(self.data_path, "Rellis-3D", seq_str, "pylon_camera_node", cam_name_dict[seq][str_frame] + ".jpg"),
                                    "point": os.path.join(self.data_path, "Rellis-3D", seq_str, "os1_cloud_node_kitti_bin", str_frame + '.bin'),
                                    "calib": calib_dict[seq],
                                    "fname": seq_str + '_' + str_frame}

                sample_list.append(indiv_sample)
        sample_list = sample_list[::self.load_interval]

        if self.num_samples > 0: sample_list = sample_list[:self.num_samples]
        else: self.num_samples = len(sample_list)
        
        if self.mode != "test": random.shuffle(sample_list)    
        
        return sample_list

    def __getitem__(self, index):
        
        # preprocess image
        im_np = rgb_read(self.samples[index]["image"])
        im_np = crop_image(im_np, self.raw_cam_img_size)
        im_np = resize_rgb_image(im_np, (int(self.raw_cam_img_size[0] / self.ratio), int(self.raw_cam_img_size[1] / self.ratio)))   
        im_np = np.transpose(im_np, (2, 0, 1))  
        im_np = im_np / (2 ** 8)
        
        # preprocess pcd
        pc_np = pcd_read(self.samples[index]["point"])    
        pc_np = pc_np.T
        # shuffle the point cloud data, this is necessary!
        pc_np = pc_np[:, np.random.permutation(pc_np.shape[1])]
        pc_np = pc_np[:3, :]  # 3xN 

        # preprocess calib
        calib_mtx = self.samples[index]["calib"]["P"] @ self.samples[index]["calib"]["Tr"] 

        # preprocess depth
        depth_np = depth_img_from_cartesian_pc_numpy(pc_np, calib_mtx, self.raw_cam_img_size)
        depth_np = resize_depth_image(depth_np, (int(self.raw_cam_img_size[0] / self.ratio), int(self.raw_cam_img_size[1] / self.ratio)))   
        depth_np = np.clip(np.expand_dims(depth_np, 0) / 48, 0, 1.0)

        # preprocess input
        rgbd = np.concatenate([im_np, depth_np], 0)
        rgbd = rgbd[:, -self.input_img_size[0]:, -self.input_img_size[1]:]

        if self.mode != "test":
            
            sn_np = np.load(self.samples[index]["sn"])            
            sn_np = sn_image_from_npy(sn_np, self.raw_cam_img_size, px=7)
            sn_np = resize_sn_image(sn_np, (int(self.raw_cam_img_size[0] / self.ratio), int(self.raw_cam_img_size[1] / self.ratio)))          
            sn_np = np.transpose(sn_np, (2, 0, 1)) 
            
            sp_np = rgb_read(self.samples[index]["sp"])
            sp_np = resize_rgb_image(sp_np, (int(self.raw_cam_img_size[0] / self.ratio), int(self.raw_cam_img_size[1] / self.ratio)))   
            sp_np = np.transpose(sp_np, (2, 0, 1))[:1, :, :]  
            sp_np[sp_np > 0] = 1
            
            fp_np = rgb_read(self.samples[index]["fp"])
            fp_np = resize_rgb_image(fp_np, (int(self.raw_cam_img_size[0] / self.ratio), int(self.raw_cam_img_size[1] / self.ratio)))   
            fp_np = np.transpose(fp_np, (2, 0, 1))[:1, :, :]  
            fp_np[fp_np > 0] = 1
            
            sn_np = sn_np[:, -self.input_img_size[0]:, -self.input_img_size[1]:]
            sp_np = sp_np[:, -self.input_img_size[0]:, -self.input_img_size[1]:]
            fp_np = fp_np[:, -self.input_img_size[0]:, -self.input_img_size[1]:]
                        
            if random.random() < 0.5:
                rgbd = image_flip(rgbd)
                sn_np = image_flip(sn_np)
                sp_np = image_flip(sp_np)
                fp_np = image_flip(fp_np)
                
            gts = {            
                "sn": sn_np,
                "sp": sp_np,
                "fp": fp_np,
                }
        else:
            # calib_mtx = np.load(self.samples[index]["calib_mtx"])
            gts = {
                # "calib_mtx" : calib_mtx
            }

        return rgbd, gts, self.samples[index]["fname"]



