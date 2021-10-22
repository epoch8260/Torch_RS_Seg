# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-07-30 18:56 
Written by Yuwei Jin (642281525@qq.com)
"""

import torch
import torch.nn as nn

from torchvision.models.mobilenetv3 import mobilenet_v3_small


class MobileNetv3(nn.Module):
    def __init__(self):
        super(MobileNetv3, self).__init__()

        net = mobilenet_v3_small(pretrained=True)

        self.layer1 = net.features[:2]  # 4x
        self.layer2 = net.features[2:4]  # 8x
        self.layer3 = net.features[4:9]  # 16x
        self.layer4 = net.features[9:11]  # 32x

    def forward(self, x):
        y1 = self.layer1(x)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer4(y3)
        return y1, y2, y3, y4


if __name__ == '__main__':
    model = MobileNetv3()

    from torchstat import stat

    stat(model, (3, 512, 512))
