# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-07-10 20:10 
Written by Yuwei Jin (642281525@qq.com)
"""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, in_chs, r=8):
        super(SelfAttention, self).__init__()

        query_chs = in_chs // r
        key_chs = in_chs // r

        self.query_conv = nn.Sequential(
            nn.Conv2d(in_chs, query_chs, kernel_size=1, bias=False),
            nn.BatchNorm2d(query_chs)
        )

        self.key_conv = nn.Sequential(
            nn.Conv2d(in_chs, key_chs, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_chs)
        )

        self.value_conv = nn.Sequential(
            nn.Conv2d(in_chs, in_chs, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_chs)
        )

        self.query_chs = query_chs
        self.key_chs = key_chs

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()

        q = self.query_conv(x).view(B, -1, H*W).permute(0, 2, 1)
        k = self.key_conv(x).view(B, -1, H*W)
        a = torch.bmm(q, k)
        a = torch.softmax(a, dim=-1)

        v = self.value_conv(x).view(B, -1, H*W)

        out = torch.bmm(v, a.permute(0, 2, 1))

        out = out.view(B, C, H, W)

        out = self.gamma * out + x

        return out


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # 先经过3个卷积层生成3个新特征图B C D （尺寸不变）
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # α尺度系数初始化为0，并逐渐地学习分配到更大的权重

        self.softmax = nn.Softmax(dim=-1)  # 对每一行进行softmax

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × (H×W) × (H×W)
        """
        m_batchsize, C, height, width = x.size()
        # B -> (N,C,HW) -> (N,HW,C)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # C -> (N,C,HW)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        # BC，空间注意图 -> (N,HW,HW)
        energy = torch.bmm(proj_query, proj_key)
        # S = softmax(BC) -> (N,HW,HW)
        attention = self.softmax(energy)
        # D -> (N,C,HW)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        # DS -> (N,C,HW)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # torch.bmm表示批次矩阵乘法
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


if __name__ == '__main__':
    # SA = SelfAttention(512, r=8)
    SA = PAM_Module(512)
    from torchstat import stat

    input_sizes = [16, 32, 48, 64, 80, 96, 112]
    for s in input_sizes:
        print("size = ", s)
        stat(SA, input_size=(512, s, 2*s))
