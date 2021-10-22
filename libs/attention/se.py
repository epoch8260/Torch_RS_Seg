# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-07-19 21:44 
Written by Yuwei Jin (642281525@qq.com)
"""
import torch
import torch.nn as nn
from libs.functional import init_weights


class SEBlock(nn.Module):
    def __init__(self, in_chs, r=16):
        super(SEBlock, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_chs, in_chs // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chs // r, in_chs, 1),
            nn.Sigmoid()
        )

        init_weights(self.fc)

    def forward(self, x):
        gap = self.gap(x)
        weight = self.fc(gap)
        return weight * x
