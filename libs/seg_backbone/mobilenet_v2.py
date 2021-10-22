# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-07-05 14:16 
Written by Yuwei Jin (642281525@qq.com)
"""
import torch
import torch.nn as nn

from torchvision.models.mobilenetv2 import mobilenet_v2


class MobileNetv2(nn.Module):
    def __init__(self):
        super(MobileNetv2, self).__init__()
        net = mobilenet_v2(pretrained=True)

        self.layer1 = net.features[:4]  # 4x
        self.layer2 = net.features[4:7]  # 8x
        self.layer3 = net.features[7:14]  # 16x
        self.layer4 = net.features[14:18]  # 32x

    def forward(self, x):
        y1 = self.layer1(x)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer4(y3)
        return y1, y2, y3, y4


if __name__ == '__main__':
    print(mobilenet_v2(pretrained=True))
    x = torch.randn(1, 3, 224, 224)
    model = MobileNetv2()

    from torchstat import stat
    stat(model, (3, 512, 512))
    y = model(x)
    print(y.size())
