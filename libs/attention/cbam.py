# -*- coding: utf-8 -*-
"""
Copyright (c) Yuwei Jin
Created on 2021-03-10 20:01
Written by Yuwei Jin (642281525@qq.com)
"""
import torch
import torch.nn as nn
from libs.functional import init_weights


class CAM(nn.Module):
    def __init__(self, in_chs, ratio=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_chs, in_chs // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_chs // 16, in_chs, 1, bias=False)

        init_weights(self.fc1, self.fc2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        y = avg_out + max_out
        out = self.sigmoid(y) * x

        return out


class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

        init_weights(self.conv1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        out = y * x

        return out


class CBAM(nn.Module):
    def __init__(self, in_chs):
        super(CBAM, self).__init__()

        self.cam = CAM(in_chs)
        self.sam = SAM()

    def forward(self, x):
        x = self.cam(x)
        y = self.sam(x)

        return y
