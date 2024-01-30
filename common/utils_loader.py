import math
import numpy as np
import os
import yaml

from PIL import Image
from scipy.spatial.transform import Rotation

def image_flip(img):
    img_flip = np.zeros_like(img)
    for c in range(img.shape[0]):
        img_flip[c,...] = np.flip(img[c, ...], axis=1)
    img = img_flip
    return img

def crop_image(img, target_size, init=False): 
    """ 
    :param img: image (numpy array, H x W x 3)
    :param target_size: crop size (tuple, 2)
    :return cropped_img: cropped image (numpy array, H' x W' x 3)
    """
    img = np.array(img)    
    if img.ndim == 3 and img.shape[2] != 3:
        img = np.transpose(img, (1, 2, 0))    
    h = img.shape[0]
    w = img.shape[1]
    i = int(math.floor((h - target_size[0]) / 2.))
    j = int(math.floor((w - target_size[1]) / 2.))
    if init : i, j = 0, 0

    if img.ndim == 3:
        img = img[i:i + target_size[0], j:j + target_size[1], :]
    else:
        img = img[i:i + target_size[0], j:j + target_size[1]]   
    # cropped_img = img.astype('uint8') 
    return img

def resize_rgb_image(img, target_size):   
    """ 
    :param img: image (numpy array, H x W x 3)
    :param target_size: target size (tuple, 2)
    :return resized_img: resized image (numpy array, H' x W' x 3)
    """
    img = np.array(img)    
    if img.ndim == 3 and img.shape[2] != 3:
        img = np.transpose(img, (1, 2, 0))
    img_resize = Image.fromarray(img.astype(np.uint8))    
    resized_img = img_resize.resize((target_size[1], target_size[0])) 
    resized_img = np.array(resized_img)
    return resized_img

def resize_depth_image(img, target_size):   
    """ 
    :param img: image (numpy array, H x W)
    :param target_size: target size (tuple, 2)
    :return resized_img: resized image (numpy array, H' x W')
    """    
    img = np.array(img)   
    h_ratio, w_ratio = target_size[0] / img.shape[0], target_size[1] / img.shape[1]
    out_img = np.zeros((target_size[0], target_size[1]))
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            if img[h, w] != 0: out_img[int(h * h_ratio), int(w * w_ratio)] = img[h, w]
    return out_img

def resize_sn_image(img, target_size):   
    """ 
    :param img: image (numpy array, H x W x 3)
    :param target_size: target size (tuple, 2)
    :return resized_img: resized image (numpy array, H' x W' x 3)
    """    
    img = np.array(img)   
    if img.shape[2] != 3: img = np.transpose(img, (1, 2, 0))
    h_ratio, w_ratio = target_size[0] / img.shape[0], target_size[1] / img.shape[1]
    out_img = np.zeros((target_size[0], target_size[1], img.shape[2])) - 1
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            for c in range(img.shape[2]):
                if img[h, w, c] != -1: out_img[int(h * h_ratio):int((h + 1) * h_ratio), int(w * w_ratio):int((w+1) * w_ratio), c] = img[h, w, c]
    return out_img

def depth_img_from_cartesian_pc_numpy(pc, cam_T_velo, raw_cam_img_size):  
    """
    :param pc: point cloud (numpy array, 3 x N)
    :param cam_T_velo: extrinsic calibration matrix (numpy array, 3 x 4)
    :param raw_cam_img_size: camera image size (tuple, 2)
    :return depth_img: depth image (numpy array, H x W)
    """
    pc = np.concatenate([pc[:3, :], np.ones((1, pc.shape[1]))], 0)
    pc = cam_T_velo @ pc    
    
    depth_img = np.zeros(shape=raw_cam_img_size)
    for idx, xyw in enumerate(pc.T):        
        x = xyw[0]
        y = xyw[1] 
        w = xyw[2] 
        is_in_img = (
            w > 0 and 0 <= x < w * raw_cam_img_size[1] and 0 <= y < w * raw_cam_img_size[0]
        )        
        if is_in_img:
            depth_img[int(y / w), int(x / w)] = w
    return depth_img

def sn_image_from_npy(sn_np, raw_cam_img_size, px):
    sn_img = np.zeros((raw_cam_img_size[0], raw_cam_img_size[1], 3)) - 1
    for i in range(sn_np.shape[0]):
        y_min, y_max = np.maximum(0, int(sn_np[i, 0]) - px), np.minimum(raw_cam_img_size[0] - 1, int(sn_np[i, 0]) + px + 1)
        x_min, x_max = np.maximum(0, int(sn_np[i, 1]) - px), np.minimum(raw_cam_img_size[1] - 1, int(sn_np[i, 1]) + px + 1)
        sn_img[y_min:y_max, x_min:x_max, :] = sn_np[i, 2:]
    return sn_img

def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    return rgb_png[:, :, :3]

def depth_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    d = np.array(img_file, dtype='uint16')  # in the range [0,255]
    return d

def pcd_read(filename):
    scan = np.fromfile(filename, dtype=np.float32)
    return scan.reshape((-1, 4))

def pose_read(line1):
    pose1 = line1.split(' ')
    pose1 = [float(p) for p in pose1]
    pose1 = np.array(pose1, dtype=float)
    pose1 = pose1.reshape((3, 4))
    pose1_eye = np.eye(4)
    pose1_eye[:3, :] = pose1
    return pose1_eye 

def bin_img_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    rgb_png = np.array(img_file) / 255 # in the range [0,255]
    return np.array(rgb_png, dtype='uint8')

def read_calib_file_orfd(filepath):
    rawdata = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                rawdata[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
            
    cam_K = np.reshape(rawdata['cam_K'], (3,3))
    cam_RT = np.reshape(rawdata['cam_RT'], (4,4))
    lidar_R = np.reshape(rawdata['lidar_R'], (3,3))
    lidar_T = rawdata['lidar_T']
    
    velo2cam = np.hstack((lidar_R, lidar_T[..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
    
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam_K
    P_rect = cam_RT
    # P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
    P_velo2im = R_cam2rect @ P_rect @ velo2cam
    
    return P_velo2im

def pose_read_orfd(line1):
    pose1 = line1.split(',')
    pose1 = [float(p) for p in pose1]
    pose1 = np.array(pose1, dtype=float)    
    pose1 = pose1.reshape((4, 4))
    return pose1  

def bin_read(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32)
    scan = scan.reshape((-1, 5))
    ptcloud_xyz = scan[:, :3]  
    return ptcloud_xyz

def get_lidar2cam_mtx(filepath):
    with open(filepath,'r') as f:
        data = yaml.load(f,Loader= yaml.Loader)
    q = data['os1_cloud_node-pylon_camera_node']['q']
    q = np.array([q['x'],q['y'],q['z'],q['w']])
    t = data['os1_cloud_node-pylon_camera_node']['t']
    t = np.array([t['x'],t['y'],t['z']])
    R_vc = Rotation.from_quat(q)
    R_vc = R_vc.as_matrix()

    RT = np.eye(4,4)
    RT[:3,:3] = R_vc
    RT[:3,-1] = t
    RT = np.linalg.inv(RT)
    return RT

def get_cam_mtx(filepath):
    data = np.loadtxt(filepath)
    P = np.zeros((3,3))
    P[0,0] = data[0]
    P[1,1] = data[1]
    P[2,2] = 1
    P[0,2] = data[2]
    P[1,2] = data[3]
    return P