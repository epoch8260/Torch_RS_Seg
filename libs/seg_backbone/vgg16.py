# -*- encoding: utf-8 -*-
"""
Copyright (c) Yuwei Jin
Created on 2021/4/22 下午5:01
Written by Yuwei Jin(yuwei_jin@163.com)
"""

import torch.nn as nn
import torch

from torchvision import models


class VGGNet(nn.Module):
    def __init__(self, pretrained=True):
        super(VGGNet, self).__init__()

        self.vgg = models.vgg16_bn(pretrained=pretrained)

        del self.vgg.classifier

    def forward(self, x):
        layer1 = self.vgg.features[0:6](x)  # 64*512*512
        layer2 = self.vgg.features[6:13](layer1)  # 128*256*256
        layer3 = self.vgg.features[13:23](layer2)  # 256*128*128
        layer4 = self.vgg.features[23:33](layer3)  # 512*64*64
        layer5 = self.vgg.features[33:43](layer4)  # 512*32*32
        return layer1, layer2, layer3, layer4, layer5


if __name__ == '__main__':
    x = torch.randn((1, 3, 512, 512))
    model = VGGNet().eval()
    layer1, layer2, layer3, layer4, layer5 = model(x)
    print(layer5.size())

    print(models.vgg16_bn(pretrained=True))
