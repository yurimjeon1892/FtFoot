import os
import sys

import torch
import yaml
import numpy as np
import time

from tqdm import tqdm
from easydict import EasyDict as edict

import data_loader
from common.utils import save_image, resize_rgb_image
from common.utils_init import init_model, resume_state

def main():
        
    with open(os.path.join(sys.argv[1]), 'r') as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)
    config = edict(config_data)    

    test_dataset = data_loader.__dict__[config.data_config.name](
            cfg=config.data_config,
            mode="test"
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    model = init_model(config.model)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    model.cuda()
    
    resume_path = os.path.join(config.ckpt_root, config.resume_path)
    model, _, _ = resume_state(resume_path, model)
    
    rn = config.resume_path.split("/")[-2]    
    cn = config.resume_path.split("/")[-1][:-4] 
    out_dir = os.path.join("../outputs/prediction", rn + "-" + cn + "-" + config.data_config.name)
    os.makedirs(out_dir, exist_ok=True)
    
    test(model, test_loader, out_dir, config.data_config.name)
    
    return
    
def test(model, loader, out_dir, dataset):

    model.eval()
    torch.cuda.synchronize() 
    times = []
    for _, (rgbd, _, fname) in enumerate(tqdm(loader)):
        with torch.no_grad():
            rgbd = rgbd.cuda().float()
            torch.cuda.synchronize()
            start_epoch = time.time()
            preds = model(rgbd)
            torch.cuda.synchronize()
            end_epoch = time.time()
            elapsed = end_epoch - start_epoch
            times.append(elapsed)
        save_trav_result(out_dir, dataset, fname, preds)
        
    times = np.mean(np.array(times))
    print("[i] mean proc. time : ", times)
    
    return

def save_trav_result(save_path, dataset, fname, preds):
    
    pred_trav_np = preds["trav"].softmax(dim=1)
    pred_trav_np = pred_trav_np.cpu().detach().numpy()
    if dataset == "RELLIS_3D":
        for b in range(len(fname)):
            trav_b, name = pred_trav_np[b, -1, :, :], fname[b]
            # trav_b = 1 - trav_b          
            save_image(trav_b * 255, os.path.join(save_path, name + ".png"))    
    elif dataset == "ORFD": 
        raw_cam_img_size = [720, 1280]
        for b in range(len(fname)):
            trav_b, name = pred_trav_np[b, ...], fname[b]
            
            trav_b = trav_b[1, :, :] 
            trav_b = trav_b * 255
            
            # trav_b = np.argmax(trav_b, axis=0)
            # trav_b = resize_rgb_image(trav_b  * 255, raw_cam_img_size)
            
            save_image(trav_b, os.path.join(save_path, name + ".png"))   
    return

if __name__ == '__main__':
    main()
