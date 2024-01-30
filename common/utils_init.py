import os
import torch
import random
import shutil
import numpy as np

import data_loader
import model.models as models
import model.optimizers as optimizers
import model.criteria as criteria

__all__ = [
    'init_seed',
    'init_dataset',
    'init_model',
    'init_optim',
    'init_lr_scheduler',
    'init_criterion',
    'save_state',
    'resume_state',
]

def config_param(model):
    param_groups = []
    other_params = []
    for name, param in model.named_parameters():
        if len(param.shape) == 1:
            g = {'params': [param], 'weight_decay': 0.0}
            param_groups.append(g)
        else:
            other_params.append(param)
    param_groups.append({'params': other_params})
    return param_groups

def save_state(model, epoch, best_metric, is_best, out_dir, filename="checkpoint.pth"):
    os.makedirs(out_dir, exist_ok=True)
    model_state_dict = model.state_dict()
    state_dict = {
        'model': model_state_dict,
        'epoch': epoch,
        'best_metric': best_metric
    }
    torch.save(state_dict, os.path.join(out_dir, filename))
    
    shutil.copyfile(
        os.path.join(out_dir, filename),
        os.path.join(out_dir, 'epoch_' + str(epoch) +'.pth'))
    
    print(os.path.join(out_dir, 'epoch_' + str(epoch) +'.pth'))
    
    if is_best:
        shutil.copyfile(
            os.path.join(out_dir, filename),
            os.path.join(out_dir, 'model_best.pth'))

    # if epoch > 3 :
    #     prev_checkpoint_filename = os.path.join(
    #         out_dir, 'checkpoint_' + str(epoch - 3) + '.pth')
    #     if os.path.exists(prev_checkpoint_filename):
    #         os.remove(prev_checkpoint_filename)
            
    if os.path.exists(os.path.join(out_dir, filename)):
            os.remove(os.path.join(out_dir, filename))
    return

def resume_state(resume_path, model, convert_dict={}):
    checkpoint = torch.load(resume_path)
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['model']
    update_dict = update_dict_filter(pretrained_dict, convert_dict, model_dict)
    model_dict.update(update_dict)
    model.load_state_dict(update_dict, strict=True)
    print("[i] load ", resume_path)
    print("[i] epoch", checkpoint['epoch'], ", best_metric", checkpoint['best_metric'])
    return model, checkpoint['epoch'], checkpoint['best_metric']

def update_dict_filter(pretrained_dict, convert_dict, model_dict):
    update_dict = {}
    for pretrainedk in pretrained_dict.keys():
        converted = False
        for cvtk in convert_dict.keys():   
            if cvtk in pretrainedk: 
                newk = pretrainedk.replace(cvtk, convert_dict[cvtk])
                update_dict[newk] = pretrained_dict[pretrainedk]
                converted = True
                print(pretrainedk , '-->', newk)
        if converted == False:
            update_dict[pretrainedk] = pretrained_dict[pretrainedk]
    update_dict = {k: v for k, v in update_dict.items() if k in model_dict}
    return update_dict

def grad_false_keys_filter(model, grad_false_keys):
    for k, p in model.named_parameters():
        k_requires_grad = True
        for grad_false_key in grad_false_keys:
            if grad_false_key in k:
                p.requires_grad = False
                k_requires_grad = False
        if k_requires_grad: print(k)
    return model

def init_seed(cfg_manual_seed):
    if cfg_manual_seed == 0:
        cfg_manual_seed = random.randint(1, 10000)
    torch.initial_seed()
    random.seed(cfg_manual_seed)
    np.random.seed(cfg_manual_seed)
    torch.manual_seed(cfg_manual_seed)
    torch.cuda.manual_seed_all(cfg_manual_seed)
    print("[i] complete init_seed :", cfg_manual_seed)
    return cfg_manual_seed

def init_model(cfg_model):
    net = getattr(models, cfg_model)()
    print("[i] complete init_model: {:.3f} M".format(
        sum(p.numel() for p in net.parameters())/1000000.0))
    return net

def init_dataset(cfg_data, batch_size, num_workers):
    train_dataset = data_loader.__dict__[cfg_data.name](            
            cfg=cfg_data,
            mode="train"
        )
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
        )
    val_dataset = data_loader.__dict__[cfg_data.name](            
            cfg=cfg_data,
            mode="valid"
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )
    print("[i] complete init_data loader")
    return train_loader, val_loader

def init_optim(cfg_optim, model):
    key, params = cfg_optim.popitem()
    print("[i] complete init_optim")
    return getattr(optimizers, key)(config_param(model), **params)

def init_lr_scheduler(cfg_lr, optimizer):
    key, params = cfg_lr.popitem()
    print("[i] complete init_lr_scheduler")
    return getattr(torch.optim.lr_scheduler, key)(optimizer, **params)

def init_criterion():
    criterion_ss = criteria.MSELoss_mask()
    criterion_se = criteria.EntropyLoss()
    criterion_ce = criteria.SegLoss(ignore_label=255, ratio=0.1)
    criterion_rmse = criteria.RMSE()  
    
    criterions = {
        "ss": criterion_ss,
        "se": criterion_se,
        "ce": criterion_ce,
        "sn": criterion_rmse
    }
    print("[i] complete init_criterion")
    return criterions