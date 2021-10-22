# -*- encoding: utf-8 -*-
"""
Copyright (c) Yuwei Jin
Created on 2021/4/27 下午7:14
Written by Yuwei Jin (yuwei_jin@163.com)
"""
from functools import reduce

import torch
import torch.nn as nn

from libs.modules import Conv3x3_BN_ReLU, Conv1x1_BN_ReLU, Conv1x1


class SKModule(nn.Module):
    def __init__(self, in_chs, out_chs, reduction_dim=16, num_branch=2, max_l=32):
        super(SKModule, self).__init__()

        d = max(in_chs // reduction_dim, max_l)  # 计算向量Z 的长度d
        self.num_branch = num_branch
        self.out_chs = out_chs

        self.conv = nn.ModuleList()
        for i in range(num_branch):
            self.conv.append(Conv3x3_BN_ReLU(2*in_chs, out_chs, dilation=i + 1, padding=i + 1))

        self.fc = nn.Sequential(
            Conv1x1_BN_ReLU(out_chs, d),
            Conv1x1(d, out_chs * num_branch, bias=False)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, h, w = x.size()

        # different scale conv
        conv_out = []
        for i, conv in enumerate(self.conv):
            conv_out.append(conv(x))

        U = reduce(lambda v0,v1:v0+v1,conv_out)
        p = self.global_avg_pool(U)
        p = self.fc(p)
        p = p.view(b, self.num_branch, self.out_chs, -1)
        a = torch.softmax(p, dim=1)

        a = list(a.chunk(self.num_branch, dim=1))  # split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a = list(map(lambda x: x.reshape(b, self.out_chs, 1, 1), a))  # 将所有分块  调整形状，即扩展两维
        V = list(map(lambda x, y: x * y, conv_out, a))  # 权重与对应  不同卷积核输出的U 逐元素相乘
        # V = reduce(lambda x, y: x + y, V)  # 两个加权后的特征 逐元素相加
        out = torch.cat(V, dim=1)

        return out


if __name__ == '__main__':
    x = torch.rand(1, 256, 256, 256)
    y = torch.rand(1, 256, 256, 256)

    model = SKModule(256, 256).eval()

    out = model(x, y)

    print(out.shape)
