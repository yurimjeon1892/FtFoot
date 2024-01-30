import os
import torch
import yaml
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch.nn.functional as F
import random
import sys
import shutil

import datetime

from common.utils import AverageMeter, rgbd_random_aug, feat_random_aug
from common.utils_summary import update_summary
from common.utils_init import *

import numpy as np
from eval_orfd import confusion_matrix_orfd, getScores

def main():
    
    with open(os.path.join(sys.argv[1]), 'r') as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)
    config = edict(config_data)

    os.makedirs(config.ckpt_root, mode=0o777, exist_ok=True)             

    config.manual_seed = init_seed(config.manual_seed)    
    train_loader, val_loader = init_dataset(config.data_config, config.batch_size, config.num_workers)
    model = init_model(config.model)

    torch.cuda.empty_cache()
    model.cuda()
    if config.resume_path :
        resume_path = os.path.join(config.ckpt_root, config.resume_path)
        model, epoch, best_metric = resume_state(resume_path, model)
        start_epoch = epoch + 1 
    else:
        start_epoch = 1
        
    if config.data_config.name == "ORFD": best_metric = 0
    else: best_metric = 1000
    torch.backends.cudnn.benchmark = True

    criterions = init_criterion()

    optimizer = init_optim(config.optim_config, model)
    lr_scheduler = init_lr_scheduler(config.lr_config, optimizer)
    
    out_dir = os.path.join(config.ckpt_root, config.data_config.name + "-" + datetime.datetime.today().strftime('%d-%m-%y-%H:%M:%S'))
    summary = SummaryWriter(os.path.join(out_dir, 'tb'))  
    shutil.copyfile(sys.argv[1], os.path.join(out_dir, 'config.yaml'))  

    print("[i] checkpoint dir: ", os.path.join(out_dir))
    for epoch in range(start_epoch, config.end_epoch + 1):        
        train(model, train_loader, criterions, optimizer, summary, epoch, config.loss_config)
        is_best = False        
        if config.data_config.name == "ORFD":
            val_metric = valid_orfd(model, val_loader, criterions, summary, epoch, config.loss_config)        
            if val_metric["total"] > best_metric:
                best_metric = val_metric["total"]
                is_best = True
        else:
            val_metric = valid(model, val_loader, criterions, summary, epoch, config.loss_config)        
            if val_metric["total"] < best_metric:
                best_metric = val_metric["total"]
                is_best = True
        save_state(model, epoch, best_metric, is_best, out_dir)    
        lr_scheduler.step()
        
    return

def train(model, loader, criterions, optimizer, summary, epoch, loss_config):

    lss1, lss2, lss3, lss4 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    description = '[i] Train {:>2}'.format(epoch)
    for batch_idx, (rgbd, gts, fname) in enumerate(tqdm(loader, desc=description, unit="batches")):

        input_size = rgbd.size()[2:4]

        model.train()
        rgbd = rgbd.cuda().float()        

        random_type = random.randint(1, 2)
        crop_type = random.random()

        rgbd_ss = rgbd_random_aug(rgbd, random_type, crop_type)
        
        preds = model(rgbd, batch_idx == 0 and epoch == 1)
        with torch.no_grad():
            preds_ss = model(rgbd_ss)

        feat = preds["feat"]
        feat_ss = preds_ss["feat"].detach_()
        feature, feature_ss = feat_random_aug(feat, feat_ss, random_type, crop_type)

        loss_ss = criterions["ss"](feature, feature_ss, None, 0)
            
        pred_up = F.interpolate(preds["trav"], size=input_size, mode='bilinear', align_corners=True)           

        loss_sn = criterions["sn"](preds["sn"], gts["sn"].cuda()) 
        loss_ce = criterions["ce"](pred_up, gts["fp"].squeeze(1).cuda().long())
        loss_se = criterions["se"](pred_up, gts["sp"].cuda().long())

        loss = loss_ss * loss_config.lamb_ss + loss_se * loss_config.lamb_se + loss_ce * loss_config.lamb_ce + loss_sn * loss_config.lamb_sn

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lss1.update(loss_ss.item() * loss_config.lamb_ss)
        lss2.update(loss_se.item() * loss_config.lamb_se)
        lss3.update(loss_ce.item() * loss_config.lamb_ce)
        lss4.update(loss_sn.item() * loss_config.lamb_sn)

    metric_dict = {
        "ss": lss1.avg,
        "se": lss2.avg,
        "ce": lss3.avg,
        "sn": lss4.avg,
        "total": lss1.avg + lss2.avg + lss3.avg + lss4.avg
    }
    update_summary(summary, rgbd, gts, preds, metric_dict, rgbd_ss, preds_ss, epoch, "train")
    # print('[i] epoch {}'.format(epoch), end = ' ')
    for k in metric_dict.keys():
        print(k + ": {:.4f},".format(metric_dict[k]), end = ' ')
    print()
    return

def valid(model, loader, criterions, summary, epoch, loss_config):
        
    lss1, lss2, lss3, lss4 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    model.eval()
    description = '[i] Valid {:>2}'.format(epoch)
    with torch.no_grad(): 
        for _, (rgbd, gts, fname) in enumerate(tqdm(loader, desc=description, unit="batches")):
            
            input_size = rgbd.size()[2:4]
            rgbd = rgbd.cuda().float()

            random_type = random.randint(1, 2)
            crop_type = random.random()
            rgbd_ss = rgbd_random_aug(rgbd, random_type, crop_type)

            preds = model(rgbd)
            preds_ss = model(rgbd_ss)

            feat = preds["feat"]
            feat_ss = preds_ss["feat"].detach_()

            feature, feature_ss = feat_random_aug(feat, feat_ss, random_type, crop_type)

            loss_ss = criterions["ss"](feature, feature_ss, None, 0)
                     
            pred_up = F.interpolate(preds["trav"], size=input_size, mode='bilinear', align_corners=True)           

            loss_sn = criterions["sn"](preds["sn"], gts["sn"].cuda()) 
            loss_ce = criterions["ce"](pred_up, gts["fp"].squeeze(1).cuda().long())
            loss_se = criterions["se"](pred_up, gts["sp"].cuda().long())

            lss1.update(loss_ss.item() * loss_config.lamb_ss)
            lss2.update(loss_se.item() * loss_config.lamb_se)
            lss3.update(loss_ce.item() * loss_config.lamb_ce)
            lss4.update(loss_sn.item() * loss_config.lamb_sn)

    metric_dict = {
        "ss": lss1.avg,
        "se": lss2.avg,
        "ce": lss3.avg,
        "sn": lss4.avg,
        "total": lss1.avg + lss2.avg + lss3.avg + lss4.avg
    }
    update_summary(summary, rgbd, gts, preds, metric_dict, rgbd_ss, preds_ss, epoch, "valid")
    for k in metric_dict.keys():
        print(k + ": {:.4f},".format(metric_dict[k]), end = ' ')
    print()
    return metric_dict

def valid_orfd(model, loader, criterions, summary, epoch, loss_config):
    
    raw_cam_img_size = (720, 1280)
    num_labels = 2    
    conf_mat = np.zeros((num_labels, num_labels), dtype=np.float16)    
        
    lss1, lss2, lss3, lss4 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    model.eval()
    description = '[i] Valid {:>2}'.format(epoch)
    with torch.no_grad(): 
        for _, (rgbd, gts, fname) in enumerate(tqdm(loader, desc=description, unit="batches")):
            
            input_size = rgbd.size()[2:4]
            rgbd = rgbd.cuda().float()

            random_type = random.randint(1, 2)
            crop_type = random.random()
            rgbd_ss = rgbd_random_aug(rgbd, random_type, crop_type)

            preds = model(rgbd)
            preds_ss = model(rgbd_ss)

            feat = preds["feat"]
            feat_ss = preds_ss["feat"].detach_()

            feature, feature_ss = feat_random_aug(feat, feat_ss, random_type, crop_type)

            loss_ss = criterions["ss"](feature, feature_ss, None, 0)
                     
            pred_up = F.interpolate(preds["trav"], size=input_size, mode='bilinear', align_corners=True)           

            loss_sn = criterions["sn"](preds["sn"], gts["sn"].cuda()) 
            loss_ce = criterions["ce"](pred_up, gts["fp"].squeeze(1).cuda().long())
            loss_se = criterions["se"](pred_up, gts["sp"].cuda().long())

            lss1.update(loss_ss.item() * loss_config.lamb_ss)
            lss2.update(loss_se.item() * loss_config.lamb_se)
            lss3.update(loss_ce.item() * loss_config.lamb_ce)
            lss4.update(loss_sn.item() * loss_config.lamb_sn)
            
            # evaluation for orfd 
            pred_trav_np = preds["trav"].softmax(dim=1)
            pred_trav_np = F.interpolate(pred_trav_np, size=raw_cam_img_size, mode='bilinear', align_corners=True) 
            pred_trav_np = pred_trav_np.cpu().detach().numpy()
            for b in range(len(fname)):
                pred_ = np.expand_dims(np.argmax(pred_trav_np[b, ...], 0), 0)
                gt_ = gts["gt"][b, ...].cpu().detach().numpy()
                conf_mat_ = confusion_matrix_orfd(gt_, pred_, num_labels)   
                conf_mat = conf_mat + conf_mat_

    globalacc, pre, recall, F_score, iou = getScores(conf_mat)
    
    metric_dict = {
        "ss": lss1.avg,
        "se": lss2.avg,
        "ce": lss3.avg,
        "sn": lss4.avg,
        
        "acc": globalacc,
        "pre": pre,
        "recall": recall, 
        "F_score": F_score, 
        "iou": iou,
        
        "total": iou
    }
    
    update_summary(summary, rgbd, gts, preds, metric_dict, rgbd_ss, preds_ss, epoch, "valid")
    for k in metric_dict.keys():
        print(k + ": {:.4f},".format(metric_dict[k]), end = ' ')
    print()
    return metric_dict

if __name__ == '__main__':
    main()

