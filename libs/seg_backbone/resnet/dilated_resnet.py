# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-05-04 16:47 
Written by Yuwei Jin (642281525@qq.com)
"""
import os

"""
resnet model, borrowed from EMANet. Thanks the authors for providing their codes.
"""

import math
import torch
import torch.nn as nn

from libs.functional import init_weights

norm_layer = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, dilation, dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, stride=8):
        self.inplanes = 128
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        if stride == 16:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, grids=[1, 2, 4])
        elif stride == 8:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, grids=[1, 2, 4])
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilation=1)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, grids=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion)
            )

        layers = []
        if grids is None:
            grids = [1] * blocks

        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation * grids[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4


def resnet(n_layers, stride, multi_grid=False):
    ckpt_folder = os.path.split(os.path.realpath(__file__))[0]
    supported_model = [50, 101]
    if n_layers in supported_model:
        layers = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3]
        }[n_layers]
        pretrained_path = {
            50: os.path.join(ckpt_folder, 'resnet50-ebb6acbb.pth'),
            101: os.path.join(ckpt_folder, 'resnet101-2a57e44d.pth')
        }[n_layers]

        net = ResNet(Bottleneck, layers=layers, stride=stride)
        print("loading resnet pretrained weight....")
        state_dict = torch.load(pretrained_path)
        net.load_state_dict(state_dict, strict=False)
        print("loaded pretrained imagenet weight successfully.")

        if multi_grid:
            multi_grid_dilations = [1, 2, 4]
            net.layer4.add_module('layer5', nn.Sequential(
                Bottleneck(inplanes=2048, planes=512, dilation=2 * multi_grid_dilations[0]),
                Bottleneck(inplanes=2048, planes=512, dilation=2 * multi_grid_dilations[1]),
                Bottleneck(inplanes=2048, planes=512, dilation=2 * multi_grid_dilations[2])
            ))
            net.layer4.add_module('layer6', nn.Sequential(
                Bottleneck(inplanes=2048, planes=512, dilation=4 * multi_grid_dilations[0]),
                Bottleneck(inplanes=2048, planes=512, dilation=4 * multi_grid_dilations[1]),
                Bottleneck(inplanes=2048, planes=512, dilation=4 * multi_grid_dilations[2])
            ))
            net.layer4.add_module('layer7', nn.Sequential(
                Bottleneck(inplanes=2048, planes=512, dilation=8 * multi_grid_dilations[0]),
                Bottleneck(inplanes=2048, planes=512, dilation=8 * multi_grid_dilations[1]),
                Bottleneck(inplanes=2048, planes=512, dilation=8 * multi_grid_dilations[2])
            ))

            init_weights(net.layer4.layer5, net.layer4.layer6, net.layer4.layer7)

        return net
    else:
        raise NotImplementedError


if __name__ == '__main__':
    model = resnet(50, 32, multi_grid=False)

    from torchstat import stat
    stat(model, (3, 512, 512))
    # x = torch.randn(3, 3, 256, 256)
    # y = model(x)
    # z = 0

    print()
