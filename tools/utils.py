# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-05-04 19:50
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


def fast_hist(input: np.ndarray, label: np.ndarray, num_class: int) -> np.ndarray:
    """calculation confusion matrix"""
    assert input.shape == label.shape

    input = input.flatten().astype('int')
    label = label.flatten().astype('int')

    mask = (label >= 0) & (label < num_class)  # 去除边界为0的情况
    label = num_class * label[mask] + input[mask]
    count = np.bincount(label, minlength=num_class ** 2)
    conf_mat = count.reshape(num_class, num_class)

    return conf_mat


def is_tensor(img):
    return isinstance(img, torch.Tensor)


def is_numpy_array(img):
    return isinstance(img, np.ndarray)


def tensor_to_numpy(data: torch.Tensor) -> np.ndarray:
    """Convert tensor data to numpy
        Args:
            data: input data (Tensor type)
            normalize: applying softmax or sigmoid to normalize the input data
    """
    if is_tensor(data):
        data = data.squeeze().detach().cpu().numpy()
        return data
    else:
        raise TypeError('input data should be Tensor type. Got {}'.format(type(data)))


def add_list_to_dataframe(data, index, df):
    """Convert list to DataFrame"""
    columns = df.columns
    dic = dict(map(lambda x, y: [x, y], columns, data))
    df.loc[index] = dic

    return df


class Counter(object):
    def __init__(self, val=-1):
        self.init_val = val
        self.val = self.init_val

    def update(self):
        self.val += 1

    def reset(self):
        self.val = self.init_val

    def get_val(self):
        return self.val


def find_best_value_index(value_list: list, higher_better=True):
    def _sub_func(value: list):
        temp_list = []
        for i, v in enumerate(value_list):
            if v == value:
                temp_list.append(i)
        return temp_list[-1]

    if higher_better:
        max_val = max(value_list)
        return _sub_func(value=max_val)
    else:
        min_val = min(value_list)
        return _sub_func(value=min_val)


def get_placeholder(x):
    if x / 100 < 1:
        r = 2
    elif 1 <= x / 100 < 10:
        r = 3
    else:
        r = 4
    return r


def time_counter(func):
    def inner(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        return (end - start) / 60.

    return inner


def tensor_binarize(prob: torch.Tensor) -> np.ndarray:
    if cfg.MODEL.OUTPUT_CHANNELS > 1:
        prob = torch.softmax(prob, dim=1)
        pred = torch.argmax(prob, dim=1)
    else:
        prob = torch.sigmoid(prob)
        pred = prob > 0.5

    pred = tensor_to_numpy(pred)

    return pred


def save_binary_map(prob: torch.Tensor, im_name: str, save_dir: str):
    res = tensor_binarize(prob) * 255  # res
    filename = os.path.join(save_dir, cfg.MODEL.NAME + '_' + im_name + '_binary.png')
    cv2.imwrite(filename, res)


def save_prob_map(im_name, prob, save_dir):
    if is_tensor(prob):
        if cfg.MODEL.OUTPUT_CHANNELS > 1:
            prob = torch.softmax(prob, dim=1)
            prob = tensor_to_numpy(prob)[1, :, :]
        else:
            prob = torch.sigmoid(prob)
            # prob = torch.relu(prob)
            prob = tensor_to_numpy(data=prob)
            # eps = 1e-10
            # prob = prob / (prob.max() + eps)
            # prob = np.where(prob<0, 0, prob)

    prob = np.round(prob * 255)

    filename = os.path.join(save_dir, cfg.MODEL.NAME + '_' + im_name + '_prob.png')
    cv2.imwrite(filename, prob.astype(np.uint8))


def save_palette_map(im_name, prob, gt, save_dir):
    # obtain binary prediction
    if isinstance(prob, np.ndarray):
        p = prob > 0.5
    else:
        p = tensor_binarize(prob)

    def inner_fun(p_flg, gt_flag, color):
        r = np.where((p == p_flg) & (gt == gt_flag), color[0], 0)
        g = np.where((p == p_flg) & (gt == gt_flag), color[1], 0)
        b = np.where((p == p_flg) & (gt == gt_flag), color[2], 0)
        return cv2.merge([b, g, r]).astype(np.uint8)

    bg = inner_fun(0, 0, cfg.TEST.BACKGROUND_COLOR)
    tp = inner_fun(1, 1, cfg.TEST.TP_COLOR)
    fp = inner_fun(1, 0, cfg.TEST.FP_COLOR)
    fn = inner_fun(0, 1, cfg.TEST.FN_COLOR)

    out = (tp + bg + fp + fn)
    output_path = os.path.join(save_dir, cfg.MODEL.NAME + '_' + im_name + '_palette.png')
    cv2.imwrite(output_path, out)


def save_alpha_map(im_name, prob, gt, save_dir):
    data = cfg.DATASET
    if cfg.TEST.STITCH_RES:
        ori_image = Image.open(os.path.join(data.TEST_WHOLE_IMAGE, im_name + data.IMG_EXTENSION)).convert('RGBA')
    else:
        ori_image = Image.open(os.path.join(data.TEST_SET, 'image', im_name + data.IMG_EXTENSION)).convert('RGBA')

    if not is_tensor(prob):
        p = prob > 0.5
    else:
        p = tensor_binarize(prob)

    def inner_fun(p_flg, gt_flag, color):
        r = np.where((p == p_flg) & (gt == gt_flag), color[0], 0)
        g = np.where((p == p_flg) & (gt == gt_flag), color[1], 0)
        b = np.where((p == p_flg) & (gt == gt_flag), color[2], 0)
        return cv2.merge([r, g, b]).astype(np.uint8)

    tp = inner_fun(1, 1, cfg.TEST.TP_COLOR)
    fp = inner_fun(1, 0, cfg.TEST.FP_COLOR)
    fn = inner_fun(0, 1, cfg.TEST.FN_COLOR)

    mask = (tp + fp + fn)
    mask = Image.fromarray(mask).convert('RGBA')
    image = Image.blend(ori_image, mask, cfg.TEST.ALPHA)

    # image_rgba = np.array(image)
    # ori_image = np.array(ori_image)
    # #
    # ori_image[:, :, 0] = np.where(p[:, :] == 1, image_rgba[:, :, 0], ori_image[:, :, 0])
    # ori_image[:, :, 1] = np.where(p[:, :] == 1, image_rgba[:, :, 1], ori_image[:, :, 1])
    # ori_image[:, :, 2] = np.where(p[:, :] == 1, image_rgba[:, :, 2], ori_image[:, :, 2])
    # #
    # image = Image.fromarray(ori_image.astype(np.uint8))
    image.save(os.path.join(save_dir, cfg.MODEL.NAME + '_' + im_name + '_alpha.png'))


def get_whole_file_list(path, img_suffix='.tif'):
    out = []
    for f in os.listdir(path):
        if f.endswith(img_suffix):
            out.append(os.path.join(path, f))

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


def get_distance_transform(label: Tensor) -> Tensor:
    """binary label distance transform"""

    B = label.size()[0]

    label = label.squeeze()

    out = torch.zeros_like(label)
    label = tensor_to_numpy(label)

    for i in range(B):
        m = np.uint8(label[i, :, :])
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 0)
        eps = 1e-10
        dist = dist / (dist.max() + eps)
        dist = torch.from_numpy(dist).unsqueeze(dim=0).cuda()
        out[i, :, :] = dist

    out = torch.unsqueeze(out, dim=1)

    return out


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


def cal_fps(model, input_size=(3, 512, 512), epochs=200):
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)

    times = np.zeros((epochs, 1))

    model = model.cuda().eval()
    x = torch.randn(input_size).unsqueeze(dim=0).cuda()

    # GPU-WARM-UP
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)

    # MEASURE PERFORMANCE
    mean_time, fps = [], []
    for _ in range(3):
        with torch.no_grad():
            for epoch in range(epochs):
                tic.record()
                _ = model(x)
                toc.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                times[epoch] = tic.elapsed_time(toc)

        temp = np.sum(times) / epochs
        mean_time.append(temp)
        fps.append(1000.0 / temp)
    print('mean: {:.3f}ms, fps: {:.2f}'.format(np.mean(mean_time), np.mean(fps)))


def save_feat(f, name, scale_factor, way='mean', ori_im=None):
    import numpy as np
    import cv2

    f = torch.relu(f)
    f = nn.UpsamplingBilinear2d(scale_factor=scale_factor)(f)
    f = f.squeeze().detach().cpu().numpy()
    if len(f.shape) > 2:
        if way == 'mean':
            f = np.mean(f, axis=0)
        else:
            f = np.max(f, axis=0)

        f -= np.min(f)
        f /= np.max(f)

    heatmap = cv2.applyColorMap(np.uint8(255 * f), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    # heatmap = heatmap[..., ::-1]  # gbr to rgb

    if ori_im is None:
        f = np.uint8(f * 255)
        fname, _ = name.split('.')
        fname = fname + '_ori.png'
        cv2.imwrite(fname, f)
        heatmap = np.uint8(heatmap * 255)
        cv2.imwrite(name, heatmap)
    else:
        # 合并heatmap到原始图像
        ori_im = np.array(ori_im) / 255
        heatmap = heatmap + np.float32(ori_im) * 0.7
        heatmap = np.uint8(heatmap * 255)
        cv2.imwrite(name, heatmap)


def list_to_txt(str_list, output_name):
    with open(output_name, 'w') as f:
        for s in str_list:
            f.write(str(s) + '\n')


def txt_to_list(txt_file) -> list:
    out = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for s in lines:
            out.append(s.rstrip("\n"))
    return out
