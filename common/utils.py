import torch
import numpy as np
from PIL import Image
from .utils_loader import pose_read, rgb_read, pcd_read, resize_rgb_image
from .utils_summary import minmax_color_img_from_img_numpy

__all__ = [
    'AverageMeter',
    'rgbd_random_aug',
    'feat_random_aug',
    'save_image'
]

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def rgbd_random_aug(rgbd, random_type, crop_type):

    if random_type == 1:
        rgbd_ss = torch.flip(rgbd,[3])

    elif random_type == 2:
        input_size = rgbd.size()[2:4]
        if crop_type < 0.25:
            rgbd_ss = rgbd[:, :, 0:input_size[0] - 20, 0:input_size[1] - 40].clone()
        elif crop_type < 0.5:
            rgbd_ss = rgbd[:, :, 0:input_size[0] - 20, 40:input_size[1]].clone()
        elif crop_type < 0.75:
            rgbd_ss = rgbd[:, :, 20:input_size[0], 0:input_size[1] - 40].clone()
        else:
            rgbd_ss = rgbd[:, :, 20:input_size[0], 40:input_size[1]].clone()

    return rgbd_ss

def feat_random_aug(feat, feat_ss, random_type, crop_type):

    if random_type == 1:
        feature = feat
        feature_ss = torch.flip(feat_ss,[3])
            
    elif random_type == 2:
        feature_ss = feat_ss   

        _, _, h, w = feat.size()
        _, _, h_ss, w_ss = feat_ss.size()    

        if crop_type < 0.25:
            feature = feat[:,:,0:h_ss,0:w_ss]
        elif crop_type < 0.5:
            feature = feat[:,:,0:h_ss,w-w_ss:w]
        elif crop_type < 0.75:
            feature = feat[:,:,h-h_ss:h,0:w_ss]
        else:
            feature = feat[:,:,h-h_ss:h,w-w_ss:w]

    return feature, feature_ss

def save_image(img, fname):
    """
    :param img: image (numpy array, H x W x 3)
    :param fname: file name (string)
    """
    img = np.array(img).astype('uint8')
    
    if img.ndim == 3 and img.shape[2] != 3:
        img = np.transpose(img, (1, 2, 0))        
    elif img.ndim == 2:
        img = np.expand_dims(img, -1)
        img = np.tile(img, (1, 1, 3))
        
    im = Image.fromarray(img.astype(np.uint8))
    im.save(fname)
    return

 

