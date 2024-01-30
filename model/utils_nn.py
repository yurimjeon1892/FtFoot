import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.nn import functional as Func
import GuideConv

def Conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Conv2dLocal_F(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = GuideConv.Conv2dLocal_F(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input, grad_weight = GuideConv.Conv2dLocal_B(input, weight, grad_output)
        return grad_input, grad_weight


class Conv2dLocal(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, input, weight):
        output = Conv2dLocal_F.apply(input, weight)
        return output


class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=False)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=True)
        self.conv = nn.Sequential(conv, )
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=2, padding=1, output_padding=1, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class Basic2dLocal(nn.Module):
    def __init__(self, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = Conv2dLocal()
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, weight):
        out = self.conv(input, weight)
        out = self.bn(out)
        out = self.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, act=True):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.act = act

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if self.act:
            out = self.relu(out)
        return out

class RW_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, shrink_factor):
        super(RW_Module, self).__init__()
        self.chanel_in = in_dim
        self.shrink_factor = shrink_factor

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        
    def own_softmax1(self, x):
    
        maxes1 = torch.max(x, 1, keepdim=True)[0]
        maxes2 = torch.max(x, 2, keepdim=True)[0]
        x_exp = torch.exp(x-0.5*maxes1-0.5*maxes2)
        x_exp_sum_sqrt = torch.sqrt(torch.sum(x_exp, 2, keepdim=True))

        return (x_exp/x_exp_sum_sqrt)/torch.transpose(x_exp_sum_sqrt, 1, 2)
    
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        x_shrink = x
        m_batchsize, C, height, width = x.size()
        if self.shrink_factor != 1:
            height = (height - 1) // self.shrink_factor + 1
            width = (width - 1) // self.shrink_factor + 1
            x_shrink = Func.interpolate(x_shrink, size=(height, width), mode='bilinear', align_corners=True)            
        
        proj_query = self.query_conv(x_shrink).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x_shrink).view(m_batchsize, -1, width*height)
        
        energy = torch.bmm(proj_query, proj_key)

        attention = self.softmax(energy)

        proj_value = self.value_conv(x_shrink).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        
        if self.shrink_factor != 1:
            height = (height - 1) * self.shrink_factor + 1
            width = (width - 1) * self.shrink_factor + 1
            out = Func.interpolate(out, size=(height, width), mode='bilinear', align_corners=True)

        out = self.gamma*out + x
        return out,energy

class STANDARD(nn.Module):

    def __init__(self, input_planes, weight_planes, norm_layer=None, weight_ks=3):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = Basic2d(input_planes + weight_planes, input_planes + weight_planes, norm_layer, kernel_size=weight_ks)
        self.conv2 = Basic2d(input_planes + weight_planes, input_planes + weight_planes, norm_layer, kernel_size=weight_ks)
        self.conv3 = Basic2d(input_planes + weight_planes, input_planes + weight_planes, norm_layer, kernel_size=weight_ks)
        self.conv4 = Basic2d(input_planes + weight_planes, input_planes + weight_planes, norm_layer, kernel_size=1, padding=0)
        self.conv5 = Basic2d(input_planes + weight_planes, input_planes, norm_layer)

    def forward(self, input, weight):
        
        x = torch.cat([input, weight], 1)        
        x1 = self.conv1(x)        
        x2 = self.conv2(x1)        
        x3 = self.conv3(x2)         
        x4 = self.conv4(x3)       
        x5 = self.conv5(x4)

        return x5

class GUIDE(nn.Module):

    def __init__(self, input_planes, weight_planes, norm_layer=None, weight_ks=3):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.local = Basic2dLocal(input_planes, norm_layer)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv11 = Basic2d(input_planes + weight_planes, input_planes, None)
        self.conv12 = nn.Conv2d(input_planes, input_planes * 9, kernel_size=weight_ks, padding=weight_ks // 2)
        self.conv21 = Basic2d(input_planes + weight_planes, input_planes, None)
        self.conv22 = nn.Conv2d(input_planes, input_planes * input_planes, kernel_size=1, padding=0)
        self.br = nn.Sequential(
            norm_layer(num_features=input_planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic2d(input_planes, input_planes, norm_layer)

    def forward(self, input, weight):
        
        B, Ci, H, W = input.shape
        weight = torch.cat([input, weight], 1)        
        weight11 = self.conv11(weight)        
        weight12 = self.conv12(weight11)        
        weight21 = self.conv21(weight)        
        weight21_ = self.pool(weight21)        
        weight22_ = self.conv22(weight21_)
        weight22 = weight22_.view(B, -1, Ci)

        out1_ = self.local(input, weight12)
        out1 = out1_.view(B, Ci, -1)             
        out2_ = torch.bmm(weight22, out1)
        out2 = out2_.view(B, Ci, H, W)        
        out3 = self.br(out2)        
        out4 = self.conv3(out3)

        return out4

class GFL(nn.Module):

    def __init__(self, input_planes, weight_planes, norm_layer=None, weight_ks=3, return_ks=False):
        super().__init__()
        self.return_ks = return_ks
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.local = Basic2dLocal(input_planes, norm_layer)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv11 = Basic2d(input_planes + weight_planes, input_planes, None)
        self.conv12 = nn.Conv2d(input_planes, input_planes * 9, kernel_size=weight_ks, padding=weight_ks // 2)
        self.conv21 = Basic2d(input_planes + weight_planes, input_planes, None)
        self.conv22 = nn.Conv2d(input_planes, input_planes * input_planes, kernel_size=1, padding=0)
        self.br = nn.Sequential(
            norm_layer(num_features=input_planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic2d(input_planes, input_planes, norm_layer)
        self.conv00 = Basic2d(input_planes + weight_planes, 2, None)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, weight):
        
        B, Ci, H, W = input.shape
        input_weight = torch.cat([input, weight], 1) 
        mask = self.conv00(input_weight)  
        mask = self.softmax(mask)

        input = input * torch.unsqueeze(mask[:, 0, :, :], 1)
        weight = weight * torch.unsqueeze(mask[:, 1, :, :], 1)
        input_weight2 = torch.cat([input, weight], 1) 

        weight11 = self.conv11(input_weight2)        
        weight12 = self.conv12(weight11)        
        weight21 = self.conv21(input_weight2)        
        weight21_ = self.pool(weight21)        
        weight22_ = self.conv22(weight21_)
        weight22 = weight22_.view(B, -1, Ci)

        out1_ = self.local(input, weight12)
        out1 = out1_.view(B, Ci, -1)             
        out2_ = torch.bmm(weight22, out1)
        out2 = out2_.view(B, Ci, H, W)        
        out3 = self.br(out2)        
        out4 = self.conv3(out3)

        if self.return_ks:
            return out4, weight12
        
        return out4