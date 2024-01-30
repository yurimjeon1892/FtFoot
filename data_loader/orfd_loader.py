import os
import numpy as np
import torch.utils.data as data
import random
import natsort

from common.utils_loader import *

__all__ = ["ORFD"]

class ORFD(data.Dataset):

    def __init__(self, cfg, mode):
        
        self.data_path = cfg.data_root
        self.load_interval = cfg.load_interval
        
        self.raw_cam_img_size = cfg.raw_cam_img_size
        self.ratio = cfg.ratio
        
        if mode == "train":
            self.mode = "training"
            self.num_samples = cfg.num_train_samples
        elif mode == "valid":
            self.mode = "validation"
            self.num_samples = cfg.num_val_samples
        elif mode == "test":
            self.mode = "testing"
            self.num_samples = cfg.num_test_samples
        else:
            print("wrong mode: ", mode)
            exit()

        self.samples = self.make_sample_dataset()
        
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.data_path + "\n"))

    def __len__(self):
        return len(self.samples)

    def make_sample_dataset(self):  
        
        sample_list = []
        
        if self.mode != "testing":        
            foot_print_path = os.path.join(self.data_path, "ORFD-custom", self.mode, "foot_print")
            file_list = natsort.natsorted(os.listdir(foot_print_path))
            for fn in file_list:
                str_frame = fn[:-4]            
                indiv_sample = {
                    "rgb": os.path.join(self.data_path, "Final_Dataset", self.mode, "image_data", str_frame + ".png"),
                    "depth": os.path.join(self.data_path, "Final_Dataset", self.mode, "sparse_depth", str_frame + ".png"),
                    "sn": os.path.join(self.data_path, "ORFD-custom", self.mode, "surface_normal", str_frame + ".npy"),
                    "sp": os.path.join(self.data_path, "ORFD-custom", self.mode, "super_pixel", str_frame + ".png"),
                    "fp": os.path.join(self.data_path, "ORFD-custom", self.mode, "foot_print", str_frame + ".png"),
                    "gt": os.path.join(self.data_path, "Final_Dataset", self.mode, "gt_image", str_frame + "_fillcolor.png"),
                    "fname": str_frame                    
                }
                sample_list.append(indiv_sample)
        else:
            rgb_path = os.path.join(self.data_path, "Final_Dataset", self.mode, "image_data")
            file_list = natsort.natsorted(os.listdir(rgb_path))
            for fn in file_list:
                str_frame = fn[:-4]
                indiv_sample = {
                    "rgb": os.path.join(self.data_path, "Final_Dataset", self.mode, "image_data", str_frame + ".png"),
                    "depth": os.path.join(self.data_path, "Final_Dataset", self.mode, "sparse_depth", str_frame + ".png"),
                    "fname": str_frame      
                    }
                sample_list.append(indiv_sample)
            ########################################################################################
            # rgb_path = os.path.join(self.data_path, "Final_Dataset", "validation", "image_data")
            # file_list = natsort.natsorted(os.listdir(rgb_path))
            # for fn in file_list:
            #     str_frame = fn[:-4]
            #     indiv_sample = {
            #         "rgb": os.path.join(self.data_path, "Final_Dataset", "validation", "image_data", str_frame + ".png"),
            #         "depth": os.path.join(self.data_path, "Final_Dataset", "validation", "sparse_depth", str_frame + ".png"),
            #         "fname": str_frame      
            #         }
            #     sample_list.append(indiv_sample)
            ########################################################################################
            
        sample_list = sample_list[::self.load_interval]
        if self.num_samples > 0: sample_list = sample_list[:self.num_samples]
        else: self.num_samples = len(sample_list)
        if self.mode != "test": random.shuffle(sample_list)    
        
        return sample_list

    def __getitem__(self, index):
        
        im_np = rgb_read(self.samples[index]["rgb"])
        im_np = crop_image(im_np, self.raw_cam_img_size)
        im_np = resize_rgb_image(im_np, (int(self.raw_cam_img_size[0] / self.ratio), int(self.raw_cam_img_size[1] / self.ratio)))   
        im_np = np.transpose(im_np, (2, 0, 1))  
        im_np = im_np / (2 ** 8)      
        # print("im_np", im_np.shape, np.min(im_np), np.max(im_np))  
        
        depth_np = depth_read(self.samples[index]["depth"])
        depth_np = resize_depth_image(depth_np, (int(self.raw_cam_img_size[0] / self.ratio), int(self.raw_cam_img_size[1] / self.ratio)))   
        depth_np = np.expand_dims(depth_np, 0)
        depth_np = depth_np / (2 ** 16)
        # print("depth_np", depth_np.shape, np.min(depth_np), np.max(depth_np)) 
        
        rgbd = np.concatenate([im_np, depth_np], 0)
        
        fname = self.samples[index]["fname"]

        if self.mode != "testing":
            
            sn_np = np.load(self.samples[index]["sn"])
            sn_np = sn_image_from_npy(sn_np, self.raw_cam_img_size, px=3)
            sn_np = resize_sn_image(sn_np, (int(self.raw_cam_img_size[0] / self.ratio), int(self.raw_cam_img_size[1] / self.ratio)))          
            sn_np = np.transpose(sn_np, (2, 0, 1)) 
            # print("sn_np", sn_np.shape, np.min(sn_np), np.max(sn_np)) 
        
            sp_np = rgb_read(self.samples[index]["sp"])
            sp_np = resize_rgb_image(sp_np, (int(self.raw_cam_img_size[0] / self.ratio), int(self.raw_cam_img_size[1] / self.ratio)))   
            sp_np = np.transpose(sp_np, (2, 0, 1))[:1, :, :]  
            sp_np[sp_np > 0] = 1
            
            fp_np = rgb_read(self.samples[index]["fp"])           
            fp_np = resize_rgb_image(fp_np, (int(self.raw_cam_img_size[0] / self.ratio), int(self.raw_cam_img_size[1] / self.ratio)))   
            fp_np = np.transpose(fp_np, (2, 0, 1))[:1, :, :]  
            fp_np[fp_np > 0] = 1
            # print("fp_np", fp_np.shape, np.min(fp_np), np.max(fp_np)) 
            
            gt_np = bin_img_read(self.samples[index]["gt"])
            gt_np = np.expand_dims(gt_np, 0)
            
            if random.random() < 0.5:
                rgbd = image_flip(rgbd)
                sn_np = image_flip(sn_np)
                sp_np = image_flip(sp_np)
                fp_np = image_flip(fp_np)
                gt_np = image_flip(gt_np)
                
            gts = {            
                "sn": sn_np,
                "sp": sp_np,
                "fp": fp_np,
                "gt": gt_np,
                }
        else:
            gts = {
            }

        return rgbd, gts, fname



