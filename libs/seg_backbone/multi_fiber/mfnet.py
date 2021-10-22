# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-05-19 14:21 
Written by Yuwei Jin (642281525@qq.com)
"""
import logging
import os
from collections import OrderedDict

import torch.nn as nn


class BN_AC_Conv2d(nn.Module):

    def __init__(self, num_in, num_filter, kernel=1, pad=0, stride=1, g=1, bias=False):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_in, num_filter, kernel_size=kernel, padding=pad, stride=stride, groups=g, bias=bias)

    def forward(self, x):
        h = self.relu(self.bn(x))
        h = self.conv(h)
        return h


class MF_UNIT(nn.Module):

    def __init__(self, num_in, num_mid, num_out, g=1, stride=1, first_block=False):
        super(MF_UNIT, self).__init__()
        num_ix = int(num_mid / 4)

        # prepare input
        self.conv_i1 = BN_AC_Conv2d(num_in=num_in, num_filter=num_ix, kernel=1, pad=0)
        self.conv_i2 = BN_AC_Conv2d(num_in=num_ix, num_filter=num_in, kernel=1, pad=0)

        # main part
        self.conv_m1 = BN_AC_Conv2d(num_in=num_in, num_filter=num_mid, kernel=3, pad=1, stride=stride, g=g)

        if first_block:
            self.conv_m2 = BN_AC_Conv2d(num_in=num_mid, num_filter=num_out, kernel=1, pad=0)
        else:
            self.conv_m2 = BN_AC_Conv2d(num_in=num_mid, num_filter=num_out, kernel=3, pad=1, g=g)

        # adapter
        if first_block:
            self.conv_w1 = BN_AC_Conv2d(num_in=num_in, num_filter=num_out, kernel=1, pad=0, stride=stride)

    def forward(self, x):

        h = self.conv_i1(x)
        x_in = x + self.conv_i2(h)

        h = self.conv_m1(x_in)
        h = self.conv_m2(h)

        if hasattr(self, 'conv_w1'):
            x = self.conv_w1(x)

        return h + x


class MFNET_2D(nn.Module):

    def __init__(self, num_classes=1000, pretrained=False, groups=16):
        super(MFNET_2D, self).__init__()

        k_sec = {2: 3,
                 3: 4,
                 4: 6,
                 5: 3}

        # conv1 - x224 (x16)
        conv1_num_out = 16
        self.conv1 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(3, conv1_num_out, kernel_size=5, padding=2, stride=2, bias=False)),
                ('bn', nn.BatchNorm2d(conv1_num_out)),
                ('relu', nn.ReLU(inplace=True))
            ])
        )
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        # conv2 - x56 (x8)
        num_mid = 96
        conv2_num_out = 96
        self.conv2 = nn.Sequential(
            OrderedDict([
                ("B%02d" % i,
                 MF_UNIT(num_in=32 if i == 1 else conv2_num_out,
                         num_mid=num_mid,
                         num_out=conv2_num_out,
                         stride=1,
                         g=groups,
                         first_block=(i==1))
                 ) for i in range(1, k_sec[2] + 1)
            ])
        )

        # conv3 - x28 (x8)
        num_mid *= 2
        conv3_num_out = 2 * conv2_num_out
        self.conv3 = nn.Sequential(OrderedDict([
            ("B%02d" % i, MF_UNIT(num_in=conv2_num_out if i == 1 else conv3_num_out,
                                  num_mid=num_mid,
                                  num_out=conv3_num_out,
                                  stride=2 if i == 1 else 1,
                                  g=groups,
                                  first_block=(i==1))
             ) for i in range(1, k_sec[3] + 1)
        ]))


        # conv4 - x14 (x8)
        num_mid *= 2
        conv4_num_out = 2 * conv3_num_out
        self.conv4 = nn.Sequential(OrderedDict([
            ("B%02d" % i, MF_UNIT(num_in=conv3_num_out if i == 1 else conv4_num_out,
                                  num_mid=num_mid,
                                  num_out=conv4_num_out,
                                  stride=2 if i == 1 else 1,
                                  g=groups,
                                  first_block=(i == 1))
             ) for i in range(1, k_sec[4] + 1)
        ]))

        # conv5 - x7 (x8)
        num_mid *= 2
        conv5_num_out = 2 * conv4_num_out
        self.conv5 = nn.Sequential(OrderedDict([
            ("B%02d" % i, MF_UNIT(num_in=conv4_num_out if i == 1 else conv5_num_out,
                                  num_mid=num_mid,
                                  num_out=conv5_num_out,
                                  stride=2 if i == 1 else 1,
                                  g=groups,
                                  first_block=(i == 1))
             ) for i in range(1, k_sec[5] + 1)
        ]))

        # final
        self.tail = nn.Sequential(OrderedDict([
            ('bn', nn.BatchNorm2d(conv5_num_out)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.globalpool = nn.Sequential(OrderedDict([
            ('avg', nn.AvgPool3d(kernel_size=7, stride=1)),
            # ('dropout', nn.Dropout(p=0.5)), only for fine-tuning
        ]))
        self.classifier = nn.Linear(conv5_num_out, num_classes)

        #############
        # Initialization

        if pretrained:
            import torch
            load_method = 'inflation'  # 'random', 'inflation'
            pretrained_model = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'MFNet2D_ImageNet1k-0000.pth')
            logging.info("Network:: graph initialized, loading pretrained model: `{}'".format(pretrained_model))
            assert os.path.exists(pretrained_model), "cannot locate: `{}'".format(pretrained_model)
            state_dict_2d = torch.load(pretrained_model)
            self.load_state_dict(state_dict_2d, strict=True)
            logging.info("loading pretrained model: `{}' successfully".format(pretrained_model))
        else:
            logging.info("Network:: graph initialized, use random inilization!")

    def forward(self, x):
        # assert x.shape[2] == 16

        h = self.conv1(x)  # x224 -> x112
        h = self.maxpool(h)  # x112 ->  x56

        h1 = self.conv2(h)  # x56 ->  x56
        h2 = self.conv3(h1)  # x56 ->  x28
        h3 = self.conv4(h2)  # x28 ->  x14
        h4 = self.conv5(h3)  # x14 ->   x7

        # h = self.tail(h)
        # h = self.globalpool(h)
        #
        # h = h.view(h.shape[0], -1)
        # h = self.classifier(h)

        return h1, h2, h3, h4


if __name__ == "__main__":
    import torch

    logging.getLogger().setLevel(logging.DEBUG)
    # ---------
    net = MFNET_2D(num_classes=1000, pretrained=True)
    data = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
    output = net(data)
    torch.save({'state_dict': net.state_dict()}, './tmp.pth')

    del net.classifier, net.tail, net.globalpool

    from torchstat import stat
    stat(net, (3, 512, 512))
