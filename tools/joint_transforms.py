# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-06-22 09:17 
Written by Yuwei Jin (642281525@qq.com)
"""

import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import functional as F

from configs import cfg
from tools import utils

_AUG = cfg.TRAIN.AUG


def _func_wrap(images, func, **kwargs):
    out = [func(im, **kwargs) for im in images]
    return out


def _to_tensor(im, gt):
    if not isinstance(im, torch.Tensor) or not isinstance(gt, torch.Tensor):
        if isinstance(im, np.ndarray):
            im = torch.from_numpy(im).permute(2, 0, 1).float()
            gt = torch.from_numpy(gt).unsqueeze(dim=0)

        if isinstance(im, Image.Image):
            im, gt = _func_wrap([im, gt], F.to_tensor)

    return im, gt


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im, gt=None):
        if gt is not None:
            for t in self.transforms:
                im, gt = t(im, gt)
            return im, gt
        else:
            for t in self.transforms:
                im = t(im)
            return im


class RandomScale(object):
    def __init__(self, p=_AUG.MS_P, scale_list=_AUG.SCALE_LIST):
        self.p = p
        self.scale_list = scale_list

    def __call__(self, im: np.ndarray, gt: np.ndarray):
        if random.random() < self.p:
            scale = float(random.choice(self.scale_list))
            if scale != 1.0:
                h, w = im.shape[0], im.shape[1]
                new_h = int(scale * h + 0.5)
                new_w = int(scale * w + 0.5)

                im = cv2.resize(im, dsize=(new_h, new_w), interpolation=cv2.INTER_LINEAR)
                gt = cv2.resize(gt, dsize=(new_h, new_w), interpolation=cv2.INTER_NEAREST)

        return im, gt


class RandomCrop(object):
    def __init__(self,
                 crop_size=_AUG.CROP_SIZE,
                 mean=None,
                 center_crop=_AUG.CENTER_CROP,
                 ignore_label=cfg.DATASET.IGNORE_LABEL):

        if mean is None:
            mean = [0.485, 0.456, 0.406]
        self.crop_size = crop_size
        self.center_crop = center_crop
        self.mean = mean
        self.ignore_label = ignore_label

    def __call__(self, im: np.ndarray, gt: np.ndarray):
        h, w = im.shape[0], im.shape[1]

        pad_h = max(self.crop_size[0] - h, 0)
        pad_w = max(self.crop_size[1] - w, 0)

        if pad_h > 0 or pad_w > 0:
            pad_val = np.asarray(self.mean)
            im = cv2.copyMakeBorder(im, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_val)
            gt = cv2.copyMakeBorder(gt, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_label)
            x1 = 0
            y1 = 0
        elif pad_w == 0 or pad_w == 0:
            x1, y1 = 0, 0
        else:
            x1 = np.random.randint(0, abs(h - self.crop_size[0]))
            y1 = np.random.randint(0, abs(w - self.crop_size[1]))

        x2 = x1 + self.crop_size[0]
        y2 = y1 + self.crop_size[1]

        im = im[x1:x2, y1:y2, :]
        gt = gt[x1:x2, y1:y2]

        return im, gt


class RandomFlip(object):
    def __init__(self, p=_AUG.FLIP_P):
        self.p = p

    def __call__(self, im, gt):
        if random.random() < self.p:
            if np.random.randint(2) == 1:  # horizontally flip
                flipcode = 1
            else:
                flipcode = 0
            im, gt = cv2.flip(im, flipCode=flipcode), cv2.flip(gt, flipCode=flipcode)
        return im, gt


class RandomRotation(object):
    def __init__(self, angle=_AUG.ROTATION_ANGLE, p=_AUG.ROTATION_P):
        self.angle = [90, 180, 270] if angle is None else angle
        self.p = p

    def __call__(self, im, gt):
        im, gt = _to_tensor(im, gt)

        if random.random() < self.p:
            if isinstance(self.angle, int):
                angle = random.uniform(-1 * self.angle, self.angle)
            else:
                angle = np.random.choice(self.angle)

            angle = float(angle)

            im = F.rotate(im, angle=angle, interpolation=F.InterpolationMode.BILINEAR)
            gt = F.rotate(gt, angle=angle, interpolation=F.InterpolationMode.NEAREST)

        return im, gt


class RandomColorJitter(object):
    def __init__(self,
                 brightness_factor=_AUG.BRIGHTNESS_FACTOR,
                 contrast_factor=_AUG.CONTRAST_FACTOR,
                 hue_factor=_AUG.HUE_FACTOR,
                 saturation_factor=_AUG.SATURATION_FACTOR,
                 p=_AUG.C0LOR_JITTER_P):

        if not (0 <= hue_factor <= 0.5):
            raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))
        self.p = p
        self.brightness_factor = random.uniform(max(0, 1 - brightness_factor), 1 + brightness_factor)
        self.contrast_factor = random.uniform(max(0, 1 - contrast_factor), 1 + contrast_factor)
        self.hue_factor = random.uniform(-1 * hue_factor, hue_factor)
        self.saturation_factor = random.uniform(max(0, 1 - saturation_factor), 1 + saturation_factor)

    def __call__(self, im, gt):
        im, gt = _to_tensor(im, gt)

        if random.random() < self.p:
            if np.random.randint(4) == 0:
                im = F.adjust_hue(im, hue_factor=self.hue_factor)
            elif np.random.randint(4) == 1:
                im = F.adjust_saturation(im, saturation_factor=self.saturation_factor)
            elif np.random.randint(4) == 2:
                im = F.adjust_brightness(im, brightness_factor=self.brightness_factor)
            else:
                im = F.adjust_contrast(im, contrast_factor=self.contrast_factor)

        return im, gt


class RandomGaussianBlur(object):
    def __init__(self, radius=_AUG.GAUSSIAN_BLUR_RADIUS, p=_AUG.GAUSSIAN_BLUR_P):
        self.radius = radius
        self.p = p

    def _blur(self, img):
        if not isinstance(img, Image.Image):
            if isinstance(img, torch.Tensor):
                img = F.to_pil_image(img)
            else:
                img = Image.fromarray(img)

        x = img.filter(ImageFilter.GaussianBlur(radius=self.radius))
        x = F.to_tensor(x)
        return x

    def __call__(self, im, gt):
        if random.random() < self.p:
            im = self._blur(im)
        return im, gt


class RandomGaussianNoise(object):
    def __init__(self, mean=_AUG.GAUSSIAN_NOISE_MEAN, var=_AUG.GAUSSIAN_NOISE_VAR, p=_AUG.GAUSSIAN_NOISE_P):
        self.mean = mean
        self.var = var
        self.p = p

    def _add_noise(self, img):
        img = np.asarray(img)
        img = np.array(img / 255, dtype=float)
        noise = np.random.normal(self.mean, self.var ** 0.5, img.shape)
        out = img + noise

        if out.min() < 0:
            low_clip = -1
        else:
            low_clip = 0

        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out * 255)
        out = Image.fromarray(out).convert('RGB')
        return out

    def __call__(self, im, gt):
        im, gt = _to_tensor(im, gt)
        if random.random() < self.p:
            im = self._add_noise(im)
        return im, gt


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, im, gt=None):
        if not isinstance(im, torch.Tensor):
            im = torch.from_numpy(im).permute(2, 0, 1).float()
            gt = torch.from_numpy(gt).unsqueeze(dim=0)

        im = F.normalize(im, mean=self.mean, std=self.std)

        if gt is not None:
            return im, gt
        else:
            return im


if __name__ == '__main__':

    def imshow(t0_im, t1_im, label):
        # t0_im = np.uint8(t0_im)
        # t1_im = np.uint8(t1_im)

        from matplotlib import pyplot as plt
        plt.subplot(131)
        plt.imshow(t0_im)
        plt.subplot(132)
        plt.imshow(t1_im)
        plt.subplot(133)
        plt.imshow(label)
        plt.show()


    def tensor2np(image):
        if len(image.shape) > 2:
            return image.permute(1, 2, 0).cpu().detach().numpy()
        else:
            return image.cpu().detach().numpy()


    im = cv2.imread('69_im.tif')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255.0
    gt = cv2.imread('69.tif', cv2.IMREAD_GRAYSCALE)

    tf = Compose([
        RandomScale(scale_list=[0.5], p=1),
        RandomCrop(crop_size=[352, 352], mean=[0.485, 0.456, 0.406], center_crop=False, ignore_label=0),
        RandomFlip(p=1),
        RandomRotation(angle=[90, 180, 270], p=1),
        RandomColorJitter(p=1),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # RandomGaussianBlur(p=1),
        # RandomGaussianNoise(p=1)
    ])

    im, gt = tf(im, gt)

    im = im.permute(1, 2, 0)
    im = utils.tensor_to_numpy(im)
    gt = utils.tensor_to_numpy(gt)

    from matplotlib import pyplot as plt

    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.subplot(1, 2, 2)
    plt.imshow(gt)
    plt.show()

    # tf = Compose([
    #     RandomFlip(p=1),
    #     RandomRotation(p=1),
    #     # RandomGaussianNoise(p=1),
    #     # RandomGaussianBlur(p=1),
    #     RandomColorJitter(p=1),
    #     ToTensor(),
    #     Normalize(means={
    #         't0': [0.485, 0.456, 0.406],
    #         't1': [0.485, 0.456, 0.406]
    #     },
    #         stds={
    #             't0': [0.229, 0.224, 0.225],
    #             't1': [0.229, 0.224, 0.225]
    #         }
    #     ),
    #     RandomScale(p=1, scale_list=[0.5])
    # ])
    #
    # t0_im, t1_im, label = tf(ori_t0_im, ori_t1_im, ori_label)
    # t0_im, t1_im, label = _func_wrap([t0_im, t1_im, label], tensor2np)
    # imshow(t0_im, t1_im, label)
