# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-05-04 15:42 
Written by Yuwei Jin (642281525@qq.com)
"""

import torch
import torch.nn as nn
from libs.functional import init_weights
from torch.nn import functional as F


# ------ 3x3 Conv -----
class Conv3x3(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size=3, padding=dilation, stride=stride, dilation=dilation, groups=groups, bias=bias)
        init_weights(self.conv)

    def forward(self, x):
        out = self.conv(x)
        return out


# ------ 3x3 Conv + BN + ReLU ------
class Conv3x3_BN_ReLU(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, padding=1, dilation=1, groups=1, bn_momentum=0.1):
        super(Conv3x3_BN_ReLU, self).__init__()

        if dilation > 1:
            padding = dilation

        self.conv = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_chs, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

        init_weights(self.conv)

    def forward(self, x):
        return self.conv(x)


class Conv3x3_BN_PReLU(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, padding=1, dilation=1, groups=1, bn_momentum=0.1):
        super().__init__()

        if dilation > 1:
            padding = dilation

        self.conv = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_chs, momentum=bn_momentum),
            nn.PReLU()
        )

        init_weights(self.conv)

    def forward(self, x):
        return self.conv(x)


class Conv3x3_BN_LeakyReLU(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, padding=1, dilation=1, groups=1, bn_momentum=0.1, negative_slope=0.1):
        super().__init__()

        if dilation > 1:
            padding = dilation

        self.conv = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_chs, momentum=bn_momentum),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )

        init_weights(self.conv, nonlinearity='leakyrelu')

    def forward(self, x):
        return self.conv(x)


class BN_ReLU_Conv3x3(nn.Module):
    """ pre-activate """

    def __init__(self, in_chs, out_chs, stride=1, padding=1, dilation=1, groups=1, momentum=0.1):
        super(BN_ReLU_Conv3x3, self).__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_chs, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        )

        init_weights(self.conv)

    def forward(self, x):
        return self.conv(x)


# ----- Conv 1x1 -----
class Conv1x1(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, groups=1, bias=True):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=stride, groups=groups, bias=bias)

        init_weights(self.conv)

    def forward(self, x):
        out = self.conv(x)
        return out


class Conv1x1_BN_ReLU(nn.Module):
    def __init__(self, in_chs, out_chs, groups=1, bn_momentum=0.1):
        super(Conv1x1_BN_ReLU, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_chs, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        init_weights(self.conv)

    def forward(self, x):
        return self.conv(x)


class Conv1x1_BN_PReLU(nn.Module):
    def __init__(self, in_chs, out_chs, groups=1, bn_momentum=0.1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_chs, momentum=bn_momentum),
            nn.PReLU()
        )
        init_weights(self.conv)

    def forward(self, x):
        return self.conv(x)


class Conv1x1_BN_LeakyReLU(nn.Module):
    def __init__(self, in_chs, out_chs, groups=1, bn_momentum=0.1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_chs, momentum=bn_momentum),
            nn.PReLU()
        )
        init_weights(self.conv, nonlinearity='leakyrelu')

    def forward(self, x):
        return self.conv(x)


class BN_ReLU_Conv1x1(nn.Module):
    """ pre-activate """

    def __init__(self, in_chs, out_chs, stride=1, dilation=1, groups=1, momentum=0.1):
        super(BN_ReLU_Conv1x1, self).__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_chs, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=stride, dilation=dilation, groups=groups, bias=False)
        )

        init_weights(self.conv)

    def forward(self, x):
        return self.conv(x)


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        # self.bn = BatchNorm2d(out_chan)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()

        init_weights(self.conv, self.bn)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class DWConv(nn.Module):
    """Depthwise-separable convolution"""

    def __init__(self, in_chs, out_chs, dilation=1, relu=True):
        super(DWConv, self).__init__()

        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_chs, in_chs, kernel_size=3, groups=in_chs, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(in_chs)
        )

        self.point_conv = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_chs)
        )

        self.relu = relu

        init_weights(self.point_conv, self.depth_conv)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)

        if self.relu:
            out = torch.relu(out)

        return out


# ----- for classification -----
class Classifier(nn.Module):
    """
    Convolution classify module.
    """

    def __init__(self, in_channels, num_classes):
        super(Classifier, self).__init__()

        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        init_weights(self.conv)

    def forward(self, x):
        return self.conv(x)


class Aux_Module(nn.Module):
    """
    auxiliary criterion module for resnet deep supervision.
    """

    def __init__(self, in_channels, mid_channels, num_classes=1, drop_out=0.1):
        super(Aux_Module, self).__init__()

        self.aux = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_out),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1)
        )
        init_weights(self.aux)

    def forward(self, x):
        res = self.aux(x)
        return res


# --------- basic residual module
class BasicResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=False),
                nn.BatchNorm2d(out_channel)
            )
            init_weights(self.downsample)

        init_weights(self.conv1, self.bn1, self.conv2, self.bn2)

    def forward(self, x):
        identity = x
        if hasattr(self, 'downsample'):
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4)
            )
            init_weights(self.downsample)

        init_weights(self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if hasattr(self, 'downsample'):
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AlignModule(nn.Module):
    """
    Alignmodule for feature align.
    """

    def __init__(self, inplane, outplane):
        super(AlignModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=3, padding=1, bias=False)

        init_weights(self.down_h, self.down_l, self.flow_make)

    def forward(self, low_feature, h_feature):
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=False)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    @staticmethod
    def flow_warp(input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)

        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)

        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)

        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class CrossLevelGate(nn.Module):
    def __init__(self, in_chs):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_chs, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.ones(1))

        init_weights(self.gate)

    def forward(self, low_f, high_f):
        x = torch.cat([low_f, high_f], dim=1)
        out = self.gate(x) * self.gamma * low_f
        return out


class RefinementResidualBlock(nn.Module):
    """ ref: Discriminative Feature Network """

    def __init__(self, in_chs, out_chs):
        super(RefinementResidualBlock, self).__init__()
        self.conv1 = Conv1x1_BN_ReLU(in_chs, out_chs)

        if in_chs > 256:
            self.dropout = nn.Dropout2d(0.1)

        self.conv2 = nn.Sequential(
            Conv3x3_BN_ReLU(out_chs, out_chs),
            Conv3x3(out_chs, out_chs),
            nn.BatchNorm2d(out_chs)
        )

        self.identity = nn.Identity()

        init_weights(self.conv1, self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        identity = self.identity(x)
        out = self.conv2(x)
        out += identity
        out = torch.relu(out)
        return out


class CAB(nn.Module):
    ''' Channel Attention Block
    Learning a Discriminative Feature Network for Semantic Segmentation (CVPR2018)(face++)
    '''

    def __init__(self, in_channels, out_channels):
        super(CAB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x  # high, low
        x = torch.cat([x1, x2], dim=1)
        x = self.global_pooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        x2 = x * x2
        res = x2 + x1
        return res


class AttentionRefinementModule(nn.Module):
    """ref bisenet_v1"""

    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = Conv3x3_BN_ReLU(in_chan, out_chan)
        self.conv_atten = nn.Conv2d(out_chan, 1, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

        init_weights(self.conv_atten, self.bn_atten)

    def forward(self, x):
        feat = self.conv(x)
        atten = self.conv_atten(feat)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out


class CEBlock(nn.Module):
    """
    Context embedding block
    Ref: BiSeNet_v2
    """

    def __init__(self, in_chs):
        super(CEBlock, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_gap = ConvBNReLU(in_chs, in_chs, 1, stride=1, padding=0)
        self.arm = AttentionRefinementModule(in_chs, in_chs)

        # TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = DWConv(in_chs, in_chs, 3)

    def forward(self, x):
        feat = self.gap(x)
        feat = self.conv_gap(feat)
        feat = F.interpolate(feat, x.size()[2:], mode='bilinear', align_corners=True)
        feat = feat + self.arm(feat)
        feat = self.conv_last(feat)
        return feat


class SelectiveRefineModule(nn.Module):
    def __init__(self, in_chs_l, in_chs_h, out_chs):
        super(SelectiveRefineModule, self).__init__()

        self.fusion_conv = Conv1x1_BN_ReLU(in_chs_l + in_chs_h, out_chs)

        self.conv1 = Conv3x3(out_chs, out_chs, dilation=1)
        self.conv2 = Conv3x3(out_chs, out_chs, dilation=2)

        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv1x1(out_chs, out_chs, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, low_f, high_f):
        x = torch.cat([low_f, high_f], dim=1)
        x = self.fusion_conv(x)

        x1 = self.conv1(x)
        x2 = self.conv2(x)

        y = x1 + x2

        att = self.att(y)

        out = att * x1 + att * x2

        return out

class FSModule(nn.Module):
    """
    Ref:  Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery
    """

    def __init__(self, in_chs, out_chs):
        super(FSModule, self).__init__()
        self.conv1 = Conv1x1_BN_ReLU(in_chs, out_chs)
        self.conv2 = Conv1x1_BN_ReLU(in_chs, out_chs)

    def forward(self, v, u):
        x = self.conv1(v)
        r = torch.mul(x, u)
        k = self.conv2(v)
        z = k / (1 + torch.exp(-r))
        return z


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_chs_l, in_chs_h, out_chs):
        super().__init__()

        if in_chs_l != in_chs_h:
            self.reduction_conv = Conv1x1_BN_ReLU(in_chs_h, in_chs_l)

        self.conv = nn.Sequential(
            Conv3x3_BN_ReLU(in_chs_l * 2, in_chs_l),
            Conv3x3_BN_ReLU(in_chs_l, out_chs)
        )

    def forward(self, low_level_f, high_level_f):
        if hasattr(self, 'reduction_conv'):
            high_level_f = self.reduction_conv(high_level_f)

        if high_level_f.size()[2:] != low_level_f.size()[2:]:
            high_level_f = F.interpolate(high_level_f, size=low_level_f.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat([low_level_f, high_level_f], dim=1)

        out = self.conv(x)

        return out


class UNetEncoderBlock(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(UNetEncoderBlock, self).__init__()

        self.conv = nn.Sequential(
            Conv3x3_BN_ReLU(in_chs, out_chs),
            Conv3x3_BN_ReLU(out_chs, out_chs)
        )

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.maxpool(x1)
        return x1, x2


class AtrousPyramidPooling(nn.Module):
    def __init__(self, in_chs, reduction_dim=256, rates=(1, 6, 12, 18)):
        super(AtrousPyramidPooling, self).__init__()

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv1x1_BN_ReLU(in_chs, reduction_dim)
        )

        self.conv1 = Conv3x3_BN_ReLU(in_chs, reduction_dim, dilation=rates[0])
        self.conv2 = Conv3x3_BN_ReLU(in_chs, reduction_dim, dilation=rates[1])
        self.conv3 = Conv3x3_BN_ReLU(in_chs, reduction_dim, dilation=rates[2])
        self.conv4 = Conv3x3_BN_ReLU(in_chs, reduction_dim, dilation=rates[3])

        self.bottle_conv = nn.Sequential(
            Conv1x1_BN_ReLU(5 * reduction_dim, reduction_dim),
            nn.Dropout2d(p=0.2)
        )

    def forward(self, x):
        pool = self.pool(x)
        pool = F.upsample(input=pool, size=x.size()[2:], mode='bilinear', align_corners=True)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        x = torch.cat([pool, x1, x2, x3, x4], dim=1)

        return self.bottle_conv(x)


class PSPModule(nn.Module):
    def __init__(self, in_chs, reduction_dim, out_chs, bins=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([self._make_stage(in_chs, reduction_dim, b) for b in bins])

        self.bottle_conv = nn.Sequential(
            nn.Conv2d(in_chs * 2, out_chs, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )

        init_weights(self.stages, self.bottle_conv)

    @staticmethod
    def _make_stage(in_chs, reduction_dim, bin):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(bin),
            nn.Conv2d(in_chs, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        H, W = x.size()[2:]

        out = x
        for f in self.stages:
            y = f(x)
            y = F.upsample(input=y, size=(H, W), mode='bilinear', align_corners=True)
            out = torch.cat([out, y], dim=1)

        return self.bottle_conv(out)
