# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-07-28 14:57 
Written by Yuwei Jin (642281525@qq.com)
"""

import torch
import torch.nn as nn

from libs.attention.se import SEBlock
from libs.modules import AlignModule, Conv3x3_BN_ReLU, Conv1x1_BN_ReLU
from libs.functional import upsample, init_weights


class AlignDecoder(nn.Module):
    def __init__(self, in_chs):
        super(AlignDecoder, self).__init__()

        self.align = AlignModule(in_chs, in_chs)

        self.fusion_conv = Conv3x3_BN_ReLU(in_chs, in_chs)

    def forward(self, low_f, high_f):
        high_f = self.align(low_f, high_f)

        x = low_f + high_f
        out = self.fusion_conv(x)

        return out


class DEM(nn.Module):
    def __init__(self, in_chs):
        super(DEM, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1))

        self.fusion_conv = Conv3x3_BN_ReLU(in_chs, in_chs)

    def forward(self, low_f, high_f):
        if high_f.size()[2:] != low_f.size()[2:]:
            high_f = upsample(high_f, size=low_f.size()[2:])

        diff = low_f - high_f
        g = self.alpha * torch.log(1 + torch.pow(diff, 2))

        out = g * low_f

        out = self.fusion_conv(out + high_f)

        return out


class DEGate(nn.Module):
    def __init__(self, in_chs_h, in_chs_l):
        super(DEGate, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_chs_l + in_chs_h, in_chs_l, 1),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, low_f, high_f):
        high_f = upsample(high_f, size=low_f.size()[2:])
        x = torch.cat([high_f, low_f], dim=1)

        gate = self.alpha * self.conv(x)

        return gate * low_f


class CEBlock(nn.Module):

    def __init__(self, in_chs=320):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs)
        self.conv_gap = Conv1x1_BN_ReLU(in_chs, in_chs)
        # TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = Conv3x3_BN_ReLU(in_chs, in_chs)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat


class GateFusion(nn.Module):
    def __init__(self, in_chs):
        super(GateFusion, self).__init__()

        self.fusion_conv = Conv1x1_BN_ReLU(in_chs * 2, in_chs)

        self.se = SEBlock(in_chs)

        self.gate = nn.Sequential(
            nn.Conv2d(in_chs * 2, 1, 1, bias=False),
            nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.zeros(1))

        self.decoder = nn.Sequential(
            Conv3x3_BN_ReLU(2 * in_chs, in_chs),
            Conv3x3_BN_ReLU(in_chs, in_chs)
        )

        init_weights(self.gate)

    def forward(self, low_f, high_f):
        if high_f.size()[2:] != low_f.size()[2:]:
            high_f = nn.UpsamplingBilinear2d(scale_factor=2)(high_f)

        x = torch.cat([high_f, low_f], dim=1)

        g = self.gate(x)

        low_f = (1 - g) * low_f
        high_f = g * high_f

        x = torch.cat([low_f, high_f], dim=1)

        return self.decoder(x)

