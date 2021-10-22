# -*- encoding: utf-8 -*-
"""
Copyright (c) Yuwei Jin
Created on 2021/4/4 下午10:30
Written by Yuwei Jin(yuwei_jin@163.com)
"""
import torch
import torch.nn as nn
from libs.modules import Conv1x1_BN_ReLU


class CPAMEnc(nn.Module):
    """
    CPAM encoding module
    """

    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(CPAMEnc, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(4)
        self.pool4 = nn.AdaptiveAvgPool2d(8)

        self.conv1 = Conv1x1_BN_ReLU(in_channels, in_channels)
        self.conv2 = Conv1x1_BN_ReLU(in_channels, in_channels)
        self.conv3 = Conv1x1_BN_ReLU(in_channels, in_channels)
        self.conv4 = Conv1x1_BN_ReLU(in_channels, in_channels)

    def forward(self, x):
        b, c, h, w = x.size()

        feat1 = self.conv1(self.pool1(x)).view(b, c, -1)
        feat2 = self.conv2(self.pool2(x)).view(b, c, -1)
        feat3 = self.conv3(self.pool3(x)).view(b, c, -1)
        feat4 = self.conv4(self.pool4(x)).view(b, c, -1)

        return torch.cat((feat1, feat2, feat3, feat4), 2)


class CPAMDec(nn.Module):
    """
    CPAM decoding module
    """

    def __init__(self, in_channels):
        super(CPAMDec, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

        self.conv_query = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)  # query_conv2
        self.conv_key = nn.Linear(in_channels, in_channels // 4)  # key_conv2
        self.conv_value = nn.Linear(in_channels, in_channels)  # value2

    def forward(self, x, y):
        """
            inputs :
                x : input feature(N,C,H,W) y:gathering centers(N,K,M)
            returns :
                out : compact position attention feature
                attention map: (H*W)*M
        """
        m_batchsize, C, width, height = x.size()
        m_batchsize, K, M = y.size()

        proj_query = self.conv_query(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # BxNxd
        proj_key = self.conv_key(y).view(m_batchsize, K, -1).permute(0, 2, 1)  # BxdxK
        energy = torch.bmm(proj_query, proj_key)  # BxNxK
        attention = self.softmax(energy)  # BxNxk

        proj_value = self.conv_value(y).permute(0, 2, 1)  # BxCxK
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # BxCxN
        out = out.view(m_batchsize, C, width, height)
        out = self.scale * out + x
        return out


class MSAM(nn.Module):
    def __init__(self, in_chs):
        super(MSAM, self).__init__()
        self.pam_enc = CPAMEnc(in_chs)
        self.pam_dec = CPAMDec(in_chs)

    def forward(self, x):
        y = self.pam_enc(x).permute(0, 2, 1)
        y = self.pam_dec(x, y)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 256, 32, 32)
    model = MSAM(256).eval()
    # y = model(x)

    from torchstat import stat
    stat(MSAM(256), input_size=(256, 32, 32))
    # input_sizes = [16, 32, 48, 64, 80, 96]
    # for s in input_sizes:
    #     pam = MSAM(in_chs=512).eval()
    #     print("size = ", s)
    #     stat(pam, input_size=(512, s, 2*s))

