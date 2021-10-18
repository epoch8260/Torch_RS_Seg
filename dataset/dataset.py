# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-06-11 15:36 
Written by Yuwei Jin (642281525@qq.com)
"""
import os

import cv2
import torch
import random

import numpy as np

from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from datasets import joint_transforms
from tools import utils

from configs import cfg


def _make_samples(dataset_dir):
    im_dir = os.path.join(dataset_dir, 'image')
    gt_dir = os.path.join(dataset_dir, 'label')

    samples = []
    for f in os.listdir(im_dir):
        if f.endswith(cfg.DATASET.IMG_EXTENSION):
            im = os.path.join(im_dir, f)
            gt = os.path.join(gt_dir, f)
            samples.append((im, gt))

    if len(samples) == 0:
        raise FileNotFoundError('None of image sample is found.')

    return samples


def _parse_train_aug_method(mean):
    aug = cfg.TRAIN.AUG
    out = []
    if aug.MULTI_SCALE:
        out.append(joint_transforms.RandomScale())

    out.append(joint_transforms.RandomCrop(mean=mean))

    if aug.FLIP:
        out.append(joint_transforms.RandomFlip())
    if aug.ROTATION:
        out.append(joint_transforms.RandomRotation())
    if aug.COLOR_JITTER:
        out.append(joint_transforms.RandomColorJitter())
    if aug.GAUSSIAN_BLUR:
        out.append(joint_transforms.RandomGaussianBlur())
    if aug.GAUSSIAN_NOISE:
        out.append(joint_transforms.RandomGaussianNoise())
    return out


def _read_image(im_src, gt_src):
    im = cv2.imread(im_src)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255.0

    gt = cv2.imread(gt_src, cv2.IMREAD_GRAYSCALE)
    gt = gt / 255

    return im, gt


class _TrainValTestDataset(Dataset):
    def __init__(self, dataset_dir, training=True, test=False):
        mean, std = utils.load_mean_std()

        # define data augmentations
        temp = joint_transforms.Normalize(mean, std)

        if training:
            transforms = _parse_train_aug_method(mean)
            transforms.append(temp)
        else:
            transforms = [temp]

        self.transforms = joint_transforms.Compose(transforms)

        if training:
            self.samples = _make_samples(dataset_dir=dataset_dir)
        else:
            self.samples = _make_samples(dataset_dir=dataset_dir)

        self.test = test

    def __getitem__(self, index):
        im_src, gt_src = self.samples[index]

        im, gt = _read_image(im_src, gt_src)

        if self.test:
            f = im_src.split('/')
            im_name = f[-1]
            im, gt = self.transforms(im, gt)
            return im_name, im, gt
        else:
            im, gt = self.transforms(im, gt)
            return im, gt

    def __len__(self):
        return len(self.samples)


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_train_data():
    dataset = _TrainValTestDataset(dataset_dir=cfg.DATASET.TRAIN_SET, training=True)
    data = DataLoader(
        dataset=dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        drop_last=cfg.DATALOADER.DROP_LAST,
        num_workers=cfg.DATALOADER.WORKERS,
        pin_memory=cfg.DATALOADER.PIP_MEMORY,
        worker_init_fn=_seed_worker
    )
    print('{} training images were loaded from dir:{}'.format(dataset.__len__(), cfg.DATASET.TRAIN_SET))
    return data


def load_val_data():
    dataset = _TrainValTestDataset(dataset_dir=cfg.DATASET.VALID_SET, training=False)
    data = DataLoader(
        dataset=dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATALOADER.WORKERS,
        drop_last=True,
        pin_memory=True
    )
    print('{} validate images were loaded from dir:{}'.format(dataset.__len__(), cfg.DATASET.VALID_SET))
    return data


def load_test_data():
    test_dataset = _TrainValTestDataset(dataset_dir=cfg.DATASET.TEST_SET, training=False, test=True)
    data = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=True
    )
    return data
