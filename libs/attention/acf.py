# -*- encoding: utf-8 -*-
"""
Copyright (c) Yuwei Jin
Created on 2021/4/26 下午9:50
Written by Yuwei Jin (yuwei_jin@163.com)
"""

import torch
import torch.nn as nn

BatchNorm2d = nn.BatchNorm2d


class acf_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_channels, out_channels):
        super(acf_Module, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.init_weight()

    def forward(self, feat_ffm, coarse_x):
        """
            inputs :
                feat_ffm : input feature maps( B X C X H X W), C is channel
                coarse_x : input feature maps( B X N X H X W), N is class
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, N, height, width = coarse_x.size()

        # CCB: Class Center Block start...
        # 1x1conv -> F'
        feat_ffm = self.conv1(feat_ffm)
        b, C, h, w = feat_ffm.size()

        # P_coarse reshape ->(B, N, W*H)
        proj_query = coarse_x.view(m_batchsize, N, -1)

        # F' reshape and transpose -> (B, W*H, C')
        proj_key = feat_ffm.view(b, C, -1).permute(0, 2, 1)

        # multiply & normalize ->(B, N, C')
        energy = torch.bmm(proj_query, proj_key)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy)
        # CCB: Class Center Block end...

        # CAB: Class Attention Block start...
        # transpose ->(B, C', N)
        attention = attention.permute(0, 2, 1)

        # (B, N, W*H)
        proj_value = coarse_x.view(m_batchsize, N, -1)

        # # multiply (B, C', N)(B, N, W*H)-->(B, C, W*H)
        out = torch.bmm(attention, proj_value)

        out = out.view(m_batchsize, C, height, width)

        # 1x1conv
        out = self.conv2(out)
        # CAB: Class Attention Block end...

        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class ACFModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ACFModule, self).__init__()

        # self.conva = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)

        self.acf = acf_Module(in_channels, out_channels)

        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(1024, 256, kernel_size=3, padding=1, dilation=1, bias=False),
        #     InPlaceABNSync(256),
        #     nn.Dropout2d(0.1,False),
        #     nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # )

    def forward(self, x, coarse_x):
        class_output = self.acf(x, coarse_x)
        # feat_cat = torch.cat([class_output, output],dim=1)
        return class_output
