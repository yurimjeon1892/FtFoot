import torch
import torch.nn as nn
import encoding
from scipy.stats import truncnorm
import math

from model.utils_nn import *

__all__ = [
    'STANDARD4'
    'GUIDE4'
    'GFL4'
]

class Baseline(nn.Module):

    def __init__(self, guide, block=BasicBlock, bc=16, \
        img_layers=[2, 2, 2, 2, 2], depth_layers=[2, 2, 2, 2, 2],
        norm_layer=nn.BatchNorm2d, weight_ks=3):
        super().__init__()

        self._norm_layer = norm_layer
        in_channels = bc * 2

        self.layer1_0 = Basic2d(4, bc * 2, norm_layer=norm_layer, kernel_size=5, padding=2)
        self.inplanes = in_channels
        self.layer1_1 = self._make_layer(block, in_channels * 2, img_layers[0], stride=2)
        self.inplanes = in_channels * 2 * block.expansion
        self.layer1_2 = self._make_layer(block, in_channels * 4, img_layers[1], stride=2)        
        self.inplanes = in_channels * 4 * block.expansion
        self.layer1_3 = self._make_layer(block, in_channels * 8, img_layers[2], stride=2)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer1_4 = self._make_layer(block, in_channels * 8, img_layers[3], stride=2)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer1_5 = self._make_layer(block, in_channels * 8, img_layers[4], stride=2)

        self.dlayer1_4 = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)
        self.dlayer1_3 = Basic2dTrans(in_channels * 16, in_channels * 8, norm_layer)
        self.dlayer1_2 = Basic2dTrans(in_channels * 16, in_channels * 4, norm_layer)
        self.dlayer1_1 = Basic2dTrans(in_channels * 8, in_channels * 2, norm_layer)
        self.dlayer1_0 = Basic2dTrans(in_channels * 4, in_channels * 2, norm_layer)

        self.layer2_0 = nn.Conv2d(in_channels * 2, 3, kernel_size=3, stride=1, padding=1)        

        # guided filter layer path   

        self.guide1 = guide(in_channels * 2, in_channels * 2, norm_layer, weight_ks)
        self.guide2 = guide(in_channels * 4, in_channels * 4, norm_layer, weight_ks)
        self.guide3 = guide(in_channels * 8, in_channels * 8, norm_layer, weight_ks)
        self.guide4 = guide(in_channels * 8, in_channels * 8, norm_layer, weight_ks)

        self.inplanes = 3
        self.layer3_1 = self._make_layer(block, in_channels * 2, depth_layers[0], stride=2)
        self.inplanes = in_channels * 2 * block.expansion
        self.layer3_2 = self._make_layer(block, in_channels * 4, depth_layers[1], stride=2)
        self.inplanes = in_channels * 4 * block.expansion
        self.layer3_3 = self._make_layer(block, in_channels * 8, depth_layers[2], stride=2)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer3_4 = self._make_layer(block, in_channels * 8, depth_layers[3], stride=2)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer3_5 = self._make_layer(block, in_channels * 8, depth_layers[4], stride=2)

        self.dlayer4_4 = Basic2dTrans(in_channels * 16, in_channels * 8, norm_layer)
        self.dlayer4_3 = Basic2dTrans(in_channels * 16, in_channels * 8, norm_layer)

        # footprint

        self.random_walk = RW_Module(in_channels * 8, shrink_factor=1)

        self.layer5_1 = block(in_channels * 8, in_channels * 8, norm_layer=norm_layer, act=False)
        self.layer5_2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels * 8, 2, 1))

        self._initialize_weights()

    def forward(self, rgbd, check=False):

        if check:
            print("rgbd        :", rgbd.size())

        x1_0 = self.layer1_0(rgbd)
        x1_1 = self.layer1_1(x1_0)
        x1_2 = self.layer1_2(x1_1)
        x1_3 = self.layer1_3(x1_2)
        x1_4 = self.layer1_4(x1_3)
        x1_5 = self.layer1_5(x1_4)

        if check:
            print("x1_0        :", x1_0.size())
            print("x1_1        :", x1_1.size())
            print("x1_2        :", x1_2.size())
            print("x1_3        :", x1_3.size())
            print("x1_4        :", x1_4.size())
            print("x1_5        :", x1_5.size())

        dx1_4 = self.dlayer1_4(x1_5)
        dx1_4 = dx1_4[:, :, :x1_4.size(2), :x1_4.size(3)]
        dx1_4_cat = torch.cat([dx1_4, x1_4], 1)
        dx1_3 = self.dlayer1_3(dx1_4_cat)
        dx1_3 = dx1_3[:, :, :x1_3.size(2), :x1_3.size(3)]
        dx1_3_cat = torch.cat([dx1_3, x1_3], 1)
        dx1_2 = self.dlayer1_2(dx1_3_cat)
        dx1_2 = dx1_2[:, :, :x1_2.size(2), :x1_2.size(3)]
        dx1_2_cat = torch.cat([dx1_2, x1_2], 1)
        dx1_1 = self.dlayer1_1(dx1_2_cat)
        dx1_1 = dx1_1[:, :, :x1_1.size(2), :x1_1.size(3)]
        dx1_1_cat = torch.cat([dx1_1, x1_1], 1)
        dx1_0 = self.dlayer1_0(dx1_1_cat)

        if check:
            print("dx1_4       :", dx1_4.size())
            print("dx1_4_cat   :", dx1_4_cat.size())
            print("dx1_3       :", dx1_3.size())
            print("dx1_3_cat   :", dx1_3_cat.size())
            print("dx1_2       :", dx1_2.size())
            print("dx1_2_cat   :", dx1_2_cat.size())
            print("dx1_1       :", dx1_1.size())
            print("dx1_1_cat   :", dx1_1_cat.size())
            print("dx1_0       :", dx1_0.size())

        # surface normal path
        sn_raw = self.layer2_0(dx1_0)
        sn_dn = torch.unsqueeze(torch.sum(torch.pow(sn_raw, 2), 1), 1)
        sn_mask = sn_dn < 1e-2
        sn_dn[sn_mask] = 1.
        sn_dn = torch.sqrt(sn_dn)
        sn = sn_raw / sn_dn
        sn[torch.isnan(sn)] = 0
        sn[torch.isinf(sn)] = 0    

        if check:        
            print("sn          :", sn.size())  

        x3_1 = self.layer3_1(sn)              
        g1 = self.guide1(dx1_1, x3_1)
        x3_2 = self.layer3_2(g1)
        g2 = self.guide2(dx1_2, x3_2)
        x3_3 = self.layer3_3(g2)
        g3 = self.guide3(dx1_3, x3_3)
        x3_4 = self.layer3_4(g3)
        g4 = self.guide4(dx1_4, x3_4)
        x3_5 = self.layer3_5(g4)

        if check:
            print("x3_1        :", x3_1.size())            
            print("g1          :", g1 .size())
            print("x3_2        :", x3_2.size())
            print("g2          :", g2.size())
            print("x3_3        :", x3_3.size())
            print("g3          :", g3.size())
            print("x3_4        :", x3_4.size())
            print("g4          :", g4.size())
            print("x3_5        :", x3_5.size())

        x4_5 = torch.cat([x1_5, x3_5], 1)
        dx4_4 = self.dlayer4_4(x4_5)
        dx4_4_cat = torch.cat([dx4_4[:, :, :g4.size(2), :g4.size(3)], g4], 1)
        dc4_3 = self.dlayer4_3(dx4_4_cat)

        if check:   
            print("x4_5        :", x4_5.size())
            print("dx4_4       :", dx4_4.size())
            print("dx4_4_cat   :", dx4_4_cat.size())
            print("dc4_3       :", dc4_3.size())

        # footprint path
        rw1, _ = self.random_walk(dc4_3)
        trav = self.layer5_1(rw1)
        trav = self.layer5_2(trav)

        ret = {
            "feat": rw1,
            "sn": sn,
            "trav": trav,
        }

        return ret

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        def truncated_normal_(num, mean=0., std=1.):
            lower = -2 * std
            upper = 2 * std
            X = truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std)
            samples = X.rvs(num)
            output = torch.from_numpy(samples)
            return output

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                data = truncated_normal_(m.weight.nelement(), mean=0, std=math.sqrt(1.3 * 2. / n))
                data = data.type_as(m.weight.data)
                m.weight.data = data.view_as(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

def STANDARD4():
    return Baseline(norm_layer=encoding.nn.SyncBatchNorm, guide=STANDARD, weight_ks=3, bc=4)

def GUIDE4():
    return Baseline(norm_layer=encoding.nn.SyncBatchNorm, guide=GUIDE, weight_ks=1, bc=4)

def GFL4():
    return Baseline(norm_layer=encoding.nn.SyncBatchNorm, guide=GFL, weight_ks=1, bc=4)