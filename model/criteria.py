#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    criteria.py
# @Project:     GuideNet
# @Author:      jie
# @Time:        2021/3/14 7:51 PM

# from multiprocessing import reduction
import torch
import torch.nn as nn

import torch.nn.functional as F

class RMSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target):
        # val_pixels = (target > -1 -1e-3).float().cuda()
        val_pixels = (target > -1).float().cuda()
        err = (target.cuda() * val_pixels - outputs * val_pixels) ** 2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        loss = torch.sqrt(loss / cnt)
        loss = torch.mean(loss)
        return loss


class MSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target):
        # val_pixels = (target > -1 -1e-3).float().cuda()
        val_pixels = (target > -1).float().cuda()
        loss = target * val_pixels - outputs * val_pixels
        loss = loss ** 2
        loss = torch.mean(loss)
        return loss

# label = 255 is ambiguious label, and only some gts have this label.
class SegLoss ( nn.Module ) :
    def __init__(self, ratio=0.1, ignore_label=255, mode=1) :
        super ( SegLoss, self ).__init__ ()
        self.ratio = ratio
        if mode == 1 :
            self.obj = torch.nn.CrossEntropyLoss ( ignore_index=ignore_label , reduction='none')
        else :
            self.obj = torch.nn.NLLLoss2d ( ignore_index=ignore_label )

    def __call__(self, pred, label) :
        loss = self.obj ( pred, label )   
        pos_mask = (label != 0)
        neg_mask = (label == 0)        
        loss_p = torch.mean(loss[pos_mask])        
        loss_n = torch.mean(loss[neg_mask])
        if torch.sum(pos_mask) == 0: loss_p = 0
        loss = loss_p + loss_n * self.ratio
        return loss


class EntropyLoss ( nn.Module ) :
    def __init__(self) :
        super ( EntropyLoss, self ).__init__ ()

    def forward(self, x, mask, mode=1) :
        # mask_size = mask.size()[1:3]
        # x_softmax = F.softmax(x, dim = 1)
        # x_logsoftmax = F.log_softmax(x, dim = 1)
        # x_softmax_up = F.interpolate(x_softmax, size=mask_size, mode='bilinear', align_corners=True)
        # x_logsoftmax_up = F.interpolate(x_logsoftmax, size=mask_size, mode='bilinear', align_corners=True)
        # b = x_softmax_up * x_logsoftmax_up

        if mode == 1 :
            mask = 1.0 - mask / 255
            b = F.softmax ( x, dim=1 ) * F.log_softmax ( x, dim=1 )
            b = torch.sum ( b, dim=1 )
            entropy = b.mul ( mask )
            loss = -1.0 * torch.sum ( entropy ) / torch.sum ( mask )
        else :
            b = F.softmax ( x, dim=1 ) * F.log_softmax ( x, dim=1 )
            b = torch.sum ( b, dim=1 )
            loss = -1.0 * torch.mean ( b )
        return loss

class MSELoss_mask ( nn.Module ) :
    def __init__(self) :
        super ( MSELoss_mask, self ).__init__ ()
        self.criterion_mse = nn.MSELoss ( reduction='none' )
        self.criterion_mse_mean = nn.MSELoss ( reduction='mean' )

    def forward(self, x1, x2, mask=None, mask_type=0) :
        if mask_type == 0 :
            loss = self.criterion_mse_mean ( x1, x2 )
        elif mask_type == 1 :
            mse_loss = self.criterion_mse ( x1, x2 )
            input_size = x1.size ()[2 :4]
            batch_size = x1.size ()[1]
            mask = F.interpolate ( torch.unsqueeze ( mask, 1 ).float (), size=input_size, mode='nearest' )
            mask_ignore = (mask != 255) & (mask != 0)
            mse_mask_loss = mse_loss.mul ( mask_ignore )
            loss = torch.sum ( mse_mask_loss ) / (torch.sum ( mask_ignore ) * batch_size)
        else :
            mse_loss = self.criterion_mse ( x1, x2 )
            input_size = x1.size ()[2 :4]
            batch_size = x1.size ()[1]
            mask = F.interpolate ( torch.unsqueeze ( mask, 1 ), size=input_size, mode='bilinear' )
            mse_mask_loss = mse_loss.mul ( mask )
            loss = torch.sum ( mse_mask_loss ) / (torch.sum ( mask ) * batch_size)
        return loss