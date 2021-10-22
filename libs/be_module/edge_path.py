# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-07-27 19:29 
Written by Yuwei Jin (642281525@qq.com)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from libs.criterion import BinaryDiceLoss
from libs.modules import Conv3x3_BN_ReLU, Classifier


class DetailHead(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(DetailHead, self).__init__()

        self.detail_head = nn.Sequential(
            Conv3x3_BN_ReLU(in_chs, out_chs),
            Classifier(out_chs, 1)
        )

    def forward(self, x):
        out = self.detail_head(x)
        return out


class EdgePath(nn.Module):
    def __init__(self, reduction_dim=96):
        super(EdgePath, self).__init__()

        self.detail_head = nn.ModuleList()
        chs = [24, 32, 96]
        for i in range(3):
            self.detail_head.append(DetailHead(chs[i], reduction_dim))

        self.cls = Classifier(3, 1)

    def forward(self, feats):
        out = []
        for i, f in enumerate(feats):
            x = self.detail_head[i](f)
            x = nn.UpsamplingBilinear2d(scale_factor=2**(i+1))(x)
            out.append(x)

        x = torch.cat([out[0], out[1], out[2]], dim=1)
        x = self.cls(x)

        return out[0], out[1], out[2], x


class EdgePathLoss(nn.Module):
    def __init__(self):
        super(EdgePathLoss, self).__init__()

    @staticmethod
    def _laplace_kernel(x: torch.Tensor) -> torch.Tensor:
        """
        Return:
            soft edge
        """
        kernel = -1 * torch.ones((1, 1, 3, 3), dtype=torch.half, requires_grad=False).cuda()
        kernel[:, :, 1, 1] = 8.
        with torch.no_grad():
            out = F.conv2d(x, weight=kernel, padding=1, stride=1, bias=None)
        return out

    def _get_pos_weight(self, gt):
        eposion = 1e-10
        count_pos = torch.sum(gt) * 1.0 + eposion
        count_neg = torch.sum(1. - gt) * 1.0
        beta = count_neg / count_pos
        return beta

    def forward(self, x, gt):
        # boundary target
        edge_gt = self._laplace_kernel(gt)
        edge_gt = edge_gt > self.edge_thr

        # class-balanced_sigmoid_cross_entropy
        pos_weight = self._get_pos_weight(edge_gt)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        loss = 0
        for f in x:
            loss = bce(f, edge_gt) + loss
        return loss

