# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-05-04 15:28 
Written by Yuwei Jin (642281525@qq.com)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


def resize_tensor(input, size, resize_mode='bilinear'):
    if not isinstance(input, torch.Tensor):
        raise ValueError('Input should be torch.Tensor type. Got unexpected type:{}'.format(type(input)))

    if resize_mode == 'bilinear':
        out = F.interpolate(input, size=size, mode=resize_mode, align_corners=True)
    elif resize_mode == 'nearset':
        out = F.interpolate(input, size=size, mode=resize_mode)
    else:
        raise NotImplementedError()

    return out


# ----- upsample ------
def upsample(x, size, mode='bilinear'):
    if mode == 'bilinear':
        out = F.interpolate(x, size=size, mode=mode, align_corners=True)
    elif mode == 'nearest':
        out = F.interpolate(x, size=size, mode=mode)
    else:
        raise NotImplementedError('Upsample mode is not supported! Got mode: {}'.format(mode))

    return out


def resize_im(im, size, mode='bilinear'):
    """Resize image with given size."""
    if mode == 'bilinear':
        out = F.interpolate(im, size=size, mode=mode, align_corners=True)
    elif mode == 'nearest':
        out = F.interpolate(im, size=size, mode=mode)
    else:
        raise NotImplementedError('Interpolation manner is not supportted.')

    return out


# ------ initialize model weights ------
def _real_init_weights(m, nonlinearity='relu'):
    if isinstance(m, list):
        for mini_m in m:
            _real_init_weights(mini_m)
    else:
        if isinstance(m, nn.Conv2d):
            print('initializing Convolution layer...')
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            print('initializing BN layer...')
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            print('initializing Linear layer...')
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Module):
            for mini_m in m.children():
                _real_init_weights(mini_m)
        else:
            raise NotImplementedError('Unknown init module.')


def init_weights(*models, nonlinearity='relu'):
    """
    Initialize model's modules.
    """
    print('initializing modules...')
    for model in models:
        _real_init_weights(model, nonlinearity=nonlinearity)


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


def global_weighted_average_pool(x):
    eps = 1e-10
    b, c, _, _ = x.size()
    x1 = x.view(b, c, -1)
    w = x / (torch.sum(x1, dim=2).view(b, c, 1, 1) + eps)
    y = x * w
    y = y.view(b, c, -1)
    y = torch.sum(y, dim=2).view(b, c, 1, 1)


# laplacian kernel for edge detection
def edge_extraction(x: torch.Tensor, operator='laplacian'):
    """extract the edge of feature map by utilizing specific kernel operator
    Args:
        x: input feature map (type: torch.Tensor)
        operator: 'laplacian', 'sobel', 'prewitt', default is laplacian
    Return:
        soft edge
    """

    def _conv2d(input, kernel):
        return F.conv2d(input, weight=kernel, padding=1, stride=1, bias=None)

    if operator == 'laplacian':
        kernel = -1 * torch.ones(1, 1, 3, 3, dtype='float32', requires_grad=False)
        kernel[:, :, 1, 1] = 8.
        return _conv2d(x, kernel)
    elif operator == 'sobel':
        kernel_x = torch.Tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]])
        kernel_x.requires_grad = False
        gx = _conv2d(input, kernel_x)
        kernel_y = torch.Tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
        kernel_y.requires_grad = False
        gy = _conv2d(input, kernel_y)
        return torch.sqrt(torch.pow(gx, 2) + torch.pow(gy, 2))
    elif operator == 'prewitt':
        kernel_x = torch.Tensor([[[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]]])
        kernel_x.requires_grad = False
        gx = _conv2d(input, kernel_x)
        kernel_y = torch.Tensor([[[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]]])
        kernel_y.requires_grad = False
        gy = _conv2d(input, kernel_y)
        return torch.sqrt(torch.pow(gx, 2) + torch.pow(gy, 2))
    else:
        raise NotImplementedError


if __name__ == '__main__':
    from libs.modules import DWConv
    X = DWConv(256, 125)