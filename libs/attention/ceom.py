# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-07-06 13:29 
Written by Yuwei Jin (642281525@qq.com)
"""
from libs.functional import init_weights

"""
class enhanced output module
"""

import torch
import torch.nn as nn


class _OAM(nn.Module):
    def __init__(self, in_chs):
        super(_OAM, self).__init__()

        self.project_conv = nn.Conv2d(in_chs, in_chs, 1, bias=False)
        self.prediction = nn.Sequential(
            nn.Conv2d(in_chs, 2, 1, bias=False),
            nn.Softmax(dim=1)
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2*in_chs, in_chs, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_chs),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_chs),
            nn.ReLU(inplace=True)
        )

        init_weights(self.prediction, self.prediction, self.fusion_conv)

        # self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj = self.project_conv(x)
        # proj reshape
        proj = proj.view(B, C, -1)  # B×C×(HW)

        prediction = self.prediction(x)  # B×2×(HW)
        # p reshape
        p = prediction.view(B, 2, -1).permute(0, 2, 1)

        energy = torch.bmm(proj, p)
        affinity = torch.softmax(energy, dim=-1)
        # affinity = self.gamma * affinity

        p = prediction.view(B, 2, -1)
        y = torch.bmm(affinity, p)
        y = y.view(B, C, H, W)

        out = torch.cat([x, y], dim=1)
        out = self.fusion_conv(out)

        if self.training:
            return out, prediction
        else:
            return out


class _CAM(nn.Module):
    def __init__(self, in_chs, r=16):
        super(_CAM, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)

        mid_chs = in_chs // r

        self.fc = nn.Sequential(
            nn.Conv2d(in_chs, mid_chs, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(mid_chs, in_chs, 1, bias=True),
            nn.Sigmoid()
        )

        init_weights(self.fc)

    def forward(self, x):
        gap = self.gap(x)
        a = self.fc(gap)
        out = a * x + x
        return out


class OEM(nn.Module):
    def __init__(self, in_chs):
        super(OEM, self).__init__()
        self.oam = _OAM(in_chs)
        self.cam = _CAM(in_chs)

    def forward(self, x):
        if self.training:
            out, aux_p = self.oam(x)
            out = self.cam(out)
            return out, aux_p
        else:
            out = self.oam(x)
            out = self.cam(out)
            return out


if __name__ == '__main__':
    x = torch.randn(1, 96, 128, 128)
    model = OEM(128).eval()
    # y = model(x)

    from torchstat import stat
    stat(model, (128, 64, 64))
