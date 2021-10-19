# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-07-15 09:09 
Written by Yuwei Jin (642281525@qq.com)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.functional import edge_extraction


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class """

    def __init__(self, smooth=1., reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target):
        assert input.shape == target.shape, "predict & target batch size don't match"

        input = torch.sigmoid(input)

        n = input.size()[0]

        prob = input.view(n, -1)
        mask = target.view(n, -1)
        intersection = (prob * mask).sum(1)
        loss = 1 - ((2. * intersection + self.smooth) / (prob.sum(1) + mask.sum(1) + self.smooth))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = BinaryDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, x, y) -> torch.Tensor:
        """
        :param x: model predictions
        :param y: label
        :return: loss
        """
        t = torch.sigmoid(x)

        loss = self.bce(x, y) + self.dice_weight * self.dice(t, y)

        return loss


class EdgeLoss(nn.Module):
    def __init__(self, alpha=1, edge_extraction_operator='laplacian', thr=0.2):
        super(EdgeLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = BinaryDiceLoss()
        self.operator = edge_extraction_operator
        self.thr = thr

    def forward(self, prob, label):
        """
            input: B*C*H*W
            label: B*H*W or B*1*H*W
        """

        # obtain reference binary edge map from label
        b_gt = edge_extraction(label, self.operator) > self.thr

        # obtain predicted soft edge map from model prediction
        b_prob = edge_extraction(prob, self.operator)
        b_prob = torch.sigmoid(b_prob)

        loss = self.dice_loss(b_prob, b_gt.float()) + F.binary_cross_entropy(input=b_prob, target=b_gt.float())

        return loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7, min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.criterion(score, target)
        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):
        return self._ohem_forward(score, target)

