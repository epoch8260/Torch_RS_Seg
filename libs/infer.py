# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-07-19 14:38 
Written by Yuwei Jin (642281525@qq.com)
"""
import os
import numpy as np
import torch
from abc import ABC, abstractmethod
from torch.cuda.amp import autocast
from torchvision.transforms.functional import vflip, hflip

from configs import cfg
from libs.functional import resize_tensor


class Infer(ABC):
    def __init__(self, model, num_class):
        model = model().cuda()
        try:
            ckpt_path = os.path.join(cfg.TRAIN.CKPT.SAVE_DIR, 'ckpt.pth')
            weight = torch.load(ckpt_path)['net']
            model.load_state_dict(weight)
        except FileNotFoundError:
            raise FileNotFoundError('saved model weight is not found in dir: {}'.format(cfg.TRAIN.CKPT.SAVE_DIR))
        self.model = model.eval()
        self.conf_mat = np.zeros((num_class, num_class))

    def _infer(self, im: torch.Tensor) -> torch.Tensor:
        im = im.cuda()
        if cfg.TEST.H_FLIP and cfg.TEST.V_FLIP:
            x = torch.cat([im, hflip(im)], dim=0)
            x = torch.cat([x, vflip(im)], dim=0)
            y = self.model(x)
            out = y[0, :, :, :] + hflip(y[1, :, :, :]) + vflip(y[2, :, :, :])
            return out.unsqueeze(dim=0) / 3.0
        elif cfg.TEST.H_FLIP and not cfg.TEST.V_FLIP:
            x = torch.cat([im, hflip(im)], dim=0)
            y = self.model(x)
            out = y[0, :, :, :] + hflip(y[1, :, :, :])
            return out.unsqueeze(dim=0) / 2.0
        elif not cfg.TEST.H_FLIP and cfg.TEST.V_FLIP:
            x = torch.cat([im, vflip(im)], dim=0)
            y = self.model(x)
            out = y[0, :, :, :] + vflip(y[1, :, :, :])
            return out.unsqueeze(dim=0) / 2.0
        else:
            return self.model(im)

    def _multi_scale_infer(self, im: torch.Tensor) -> torch.Tensor:
        """ multi scale inference """
        _, _, ori_h, ori_w = im.size()
        final_prob = torch.zeros([1, cfg.MODEL.OUTPUT_CHANNELS, ori_h, ori_w]).cuda()

        for scale in cfg.TEST.MS_SCALE_LIST:
            new_h = int(ori_h * scale)
            new_w = int(ori_w * scale)
            new_im = resize_tensor(input=im, size=[new_h, new_w])

            prob = self._infer(new_im)
            prob = resize_tensor(prob, (ori_h, ori_w))
            final_prob += prob

        final_prob /= len(cfg.TEST.MS_SCALE_LIST)

        return final_prob

    def forward(self, x):
        if cfg.TEST.MS:
            out = self._multi_scale_infer(x)
        else:
            out = self._infer(x)
        return out
