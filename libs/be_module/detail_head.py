# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-07-27 10:38 
Written by Yuwei Jin (642281525@qq.com)
"""
import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F

from libs.modules import Conv3x3_BN_ReLU, Classifier

from libs.criterion import BinaryDiceLoss


class DetailHead(nn.Module):
    def __init__(self, in_chs):
        super(DetailHead, self).__init__()

        self.detail_head = nn.Sequential(
            Conv3x3_BN_ReLU(in_chs, in_chs),
            Classifier(in_chs, 1)
        )

    def forward(self, x):
        out = self.detail_head(x)
        return out




class DetailLoss(nn.Module):
    def __init__(self, edge_thr=0.1):
        super(DetailLoss, self).__init__()

        self.edge_thr = edge_thr

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = BinaryDiceLoss()

    @staticmethod
    def _laplace_kernel(x: torch.Tensor) -> torch.Tensor:
        """
        Return:
            soft edge
        """
        kernel = -1 * torch.ones((1, 1, 3, 3), dtype=torch.half, requires_grad=False)
        kernel[:, :, 1, 1] = 8.
        with torch.no_grad():
            out = F.conv2d(x, weight=kernel, padding=1, stride=1, bias=None)
        return out

    def _get_boundary(self, input, target):
        # boundary logits
        edge_prob = self._laplace_kernel(input)

        # boundary target
        edge_gt = self._laplace_kernel(target)
        edge_gt = edge_gt > self.edge_thr

        loss = self.bce(edge_prob, edge_gt) + self.dice(edge_prob, edge_gt)

        return loss




