# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-10-19 13:41 
Written by Yuwei Jin (642281525@qq.com)
"""

import os
import random
import re
import shutil
import time
import cv2
import pandas as pd
import torch
import logging
import numpy as np
from matplotlib import pyplot as plt
from nb_log import get_logger
from PIL import Image

from torch import Tensor, nn
from torch.cuda.amp import autocast as autocast
from torchvision.transforms.functional import hflip, vflip
from decimal import Decimal, ROUND_HALF_UP
from typing import Union

from configs import cfg
from libs.functional import resize_tensor
from libs.lr_scheduler import *


# ------ IO related utils ------
def mk_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mk_dirs_r(file_path):
    if os.path.exists(file_path):
        shutil.rmtree(file_path, ignore_errors=True)
    os.makedirs(file_path, exist_ok=True)


def rm_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


# ------------------------ file related utils -------------------------------
def get_whole_file_list(path, img_suffix='.tif') -> list:
    """return example: /home/jyw/train/img/1.tif"""
    out = []
    for f in os.listdir(path):
        if f.endswith(img_suffix):
            out.append(os.path.join(path, f))
    return out


def list_to_txt(str_list, output_name):
    """write list to local txt file"""
    with open(output_name, 'w') as f:
        for s in str_list:
            f.write(str(s) + '\n')


def txt_to_list(txt_file) -> list:
    """read local txt to list"""
    out = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for s in lines:
            out.append(s.rstrip("\n"))
    return out


def get_image_name_list(path, im_extensions='.tif'):
    """return example:
    ['1.tif', '2.tif']
    """
    out = []
    for f in os.listdir(path):
        if f.endswith(im_extensions):
            out.append(f)
    return out


# ------ pytorch training related utils ------
def save_train_cfg():
    import json
    save_dir = cfg.MODEL.OUTPUT_DIR
    mk_dirs_r(save_dir)
    file_name = "training_cfg.json"
    save_dir = os.path.join(save_dir, file_name)
    with open(save_dir, 'w+', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False)


def fixed_np_random_seed(seed=2048):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def fixed_torch_seed(seed=2048):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    if cfg.TRAIN.CUDNN_DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_mean_std():
    data = cfg.DATASET
    # load mean std
    if cfg.DATASET.MEAN_STD == 'imagenet':
        mean = [0.485, 0.456, 0.406]  # imagenet mean and std value
        std = [0.229, 0.224, 0.225]
        return mean, std
    else:
        mean_std_file = os.path.join(cfg.DATASET.TRAIN_SET, os.path.join(data.TRAIN_SET, data.NAME + '_mean_std.csv'))
        try:
            data = pd.read_csv(mean_std_file)
            mean = data['mean'].values.tolist()
            std = data['std'].values.tolist()
            return mean, std
        except FileNotFoundError:
            print('{} mean-std.csv file is not found.'.format(mean_std_file))


def parse_optimizer(params):
    optim = cfg.TRAIN.OPTIMIZER
    if re.search(optim, 'sgd', re.IGNORECASE):
        return torch.optim.SGD(
            params=params,
            lr=cfg.TRAIN.INIT_LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD
        )
    elif re.search(optim, 'adam', re.IGNORECASE):
        return torch.optim.Adam(
            params=params,
            lr=cfg.TRAIN.INIT_LR,
            weight_decay=cfg.TRAIN.WD
        )
    else:
        raise NotImplementedError('optimizer {} is not supported.'.format(optim))


def lr_scheduler(optimizer, train_data):
    train_batch_steps = math.floor(len(train_data.dataset) / cfg.TRAIN.BATCH_SIZE)

    lr = cfg.TRAIN.LR

    if re.search(lr.SCHEDULER, 'ExponentialLR', re.IGNORECASE):
        if lr.USE_WARMUP:
            scheduler = WarmExponentialLR(optimizer=optimizer,
                                          gamma=lr.POWER,
                                          eta_min_lr=lr.ETA_MIN_LR,
                                          warm_start_lr=lr.WARMUP_START_LR,
                                          warm_period=lr.WARMUP_STEPS)
        else:
            scheduler = ExponentialLR(optimizer=optimizer, gamma=lr.POWER, eta_min_lr=lr.ETA_MIN_LR)

    elif re.search(lr.SCHEDULER, 'Poly', re.IGNORECASE):
        max_iters = cfg.TRAIN.EPOCHS * train_batch_steps
        if lr.USE_WARMUP:
            warmup_period = lr.WARMUP_STEPS * train_batch_steps
            scheduler = WarmPolyLR(optimizer=optimizer,
                                   power=lr.POWER,
                                   max_iterations=max_iters,
                                   warm_start_lr=lr.WARMUP_START_LR,
                                   warm_period=warmup_period)
        else:
            scheduler = PolyLR(optimizer=optimizer, power=lr.POWER, max_iterations=max_iters)
    elif re.search(lr.SCHEDULER, 'Cosine', re.IGNORECASE):
        max_iters = cfg.TRAIN.EPOCHS * train_batch_steps
        if lr.USE_WARMUP:
            warmup_period = lr.WARMUP_STEPS * train_batch_steps
            scheduler = WarmCosineLR(optimizer=optimizer,
                                     eta_min_lr=lr.ETA_MIN_LR,
                                     T_max=max_iters,
                                     warm_start_lr=lr.WARMUP_START_LR,
                                     warm_period=warmup_period)
        else:
            scheduler = CosineLR(optimizer=optimizer, eta_min_lr=lr.ETA_MIN_LR, T_max=max_iters)
    else:
        raise NotImplementedError("{} is not implemented.".format(lr.SCHEDULER))

    return scheduler


def parse_optimizer(params):
    optim = cfg.TRAIN.OPTIMIZER
    if re.search(optim, 'sgd', re.IGNORECASE):
        return torch.optim.SGD(
            params=params,
            lr=cfg.TRAIN.INIT_LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD
        )
    elif re.search(optim, 'adam', re.IGNORECASE):
        return torch.optim.Adam(
            params=params,
            lr=cfg.TRAIN.INIT_LR,
            weight_decay=cfg.TRAIN.WD
        )
    else:
        raise NotImplementedError('optimizer {} is not supported.'.format(optim))


# ------------------------ tensor related utils -------------------------------
def is_tensor(img):
    return isinstance(img, torch.Tensor)


def is_numpy_array(img):
    return isinstance(img, np.ndarray)


def tensor_to_numpy(data: torch.Tensor) -> np.ndarray:
    """Convert tensor data to numpy
        Args:
            data: input data (Tensor type)
    """
    if is_tensor(data):
        data = data.squeeze().detach().cpu().numpy()
        return data
    else:
        raise TypeError('input data should be Tensor type. Got {}'.format(type(data)))


