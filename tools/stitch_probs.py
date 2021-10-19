# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-06-13 20:14 
Written by Yuwei Jin (642281525@qq.com)
"""

import json
import cv2
import os
import numpy as np
from time import time
from multiprocessing import Pool

from configs import cfg
from tools.metrics import BETestMetrics
from tools import utils


def _get_big_im_names():
    im_names = []
    for f in os.listdir(cfg.DATASET.TEST_WHOLE_LABEL):
        if f.endswith(cfg.DATASET.IMG_EXTENSION):
            temp, _ = f.split('.')
            im_names.append(temp)
    return im_names


def _load_crop_info(file):
    src = os.path.join(cfg.DATASET.TEST_SET, 'crop_info', file + '_crop_info' + '.json')
    with open(src) as json_file:
        info = json.load(json_file)

    rows = info['rows']
    cols = info['cols']
    patch_size = info['patch_size']
    m = info['m']
    n = info['n']

    if 'overlap' in info.keys():
        overlap = info['overlap']
        return rows, cols, patch_size, overlap, m, n
    else:
        central_win_size = info['central_win_size']
        return rows, cols, patch_size, central_win_size, m, n


def _read_prob_img(img_name: str, index: int):
    # name format: model name + '_' + image name + '_prob.png'
    temp = cfg.MODEL.NAME + '_' + img_name + '-' + str(index) + '_prob' + '.png'

    src = os.path.join(cfg.TEST.PROB_SAVE_DIR, temp)
    prob = cv2.imread(src, cv2.IMREAD_GRAYSCALE) / 255  # value should be [0,1]

    return prob


class Stitch:
    def __init__(self):
        self.evaluator = BETestMetrics()
        utils.mk_dirs_r(cfg.TEST.STITCH_SAVE_DIR)

    def _save_res(self, im_name, whole_prob, gt):
        self.evaluator.cal_batch_metric(im_name, whole_prob, gt)
        utils.save_prob_map(im_name, whole_prob, cfg.TEST.STITCH_SAVE_DIR)
        utils.save_alpha_map(im_name, whole_prob, gt, cfg.TEST.STITCH_SAVE_DIR)

    def _sub_func_dilated(self, im_name):
        print("stitching: {}{}".format(im_name, cfg.DATASET.IMG_EXTENSION))

        rows, cols, patch_size, win, m, n = _load_crop_info(im_name)
        pad = int((patch_size - win) // 2)

        whole_prob = np.zeros((rows, cols))

        k = 0
        for i in range(m):
            x1 = i * win
            x2 = x1 + win
            if i == m - 1:  # 超出图像范围
                x1, x2 = (m - 1) * win, rows

            for j in range(n):
                y1 = j * win
                y2 = y1 + win
                if j == n - 1:  # 超出图像范围
                    y1, y2 = (n - 1) * win, cols

                k += 1

                patch = _read_prob_img(im_name, index=k)

                a1, a2 = pad, pad + win
                b1, b2 = pad, pad + win
                if i == m - 1:
                    a1, a2 = pad, pad + rows - (m - 1) * win
                if j == n - 1:
                    b1, b2 = pad, pad + cols - (n - 1) * win

                # weight[x1:x2, y1:y2] = weight[x1:x2, y1:y2] + 1
                whole_prob[x1:x2, y1:y2] = patch[a1:a2, b1:b2]

        gt_src = os.path.join(cfg.DATASET.TEST_WHOLE_LABEL, im_name + cfg.DATASET.IMG_EXTENSION)
        gt = cv2.imread(gt_src, cv2.IMREAD_GRAYSCALE) / 255

        self.evaluator.cal_batch_metric(im_name, whole_prob, gt)

        if cfg.TEST.STITCH_SAVE_RES:
            self._save_res(whole_prob=whole_prob, im_name=im_name, gt=gt)

    def _sub_func_overlap(self, im_name):
        print("stitching: {}{}".format(im_name, cfg.DATASET.IMG_EXTENSION))

        rows, cols, patch_size, overlap, m, n = _load_crop_info(im_name)
        weight = np.zeros((rows, cols))
        whole_prob = np.zeros((rows, cols))

        k = 0
        for i in range(m):
            x1 = i * (patch_size - overlap)
            x2 = x1 + patch_size
            if x2 >= rows:  # 超出图像范围
                x1, x2 = rows - patch_size, rows
            for j in range(n):
                y1 = j * (patch_size - overlap)
                y2 = y1 + patch_size
                if y2 >= cols:  # 超出图像范围
                    y1, y2 = cols - patch_size, cols

                k += 1

                patch = _read_prob_img(img_name=im_name, index=k)

                weight[x1:x2, y1:y2] = weight[x1:x2, y1:y2] + 1
                whole_prob[x1:x2, y1:y2] = whole_prob[x1:x2, y1:y2] + patch

        whole_prob = whole_prob / weight

        gt_src = os.path.join(cfg.DATASET.TEST_WHOLE_LABEL, im_name + cfg.DATASET.IMG_EXTENSION)
        gt = cv2.imread(gt_src, cv2.IMREAD_GRAYSCALE) / 255

        self.evaluator.cal_batch_metric(im_name, whole_prob, gt)

        if cfg.TEST.STITCH_SAVE_RES:
            self._save_res(whole_prob=whole_prob, im_name=im_name, gt=gt)

    def _sub_func(self, func):
        """func: specific stitch function"""
        tic = time()
        files = _get_big_im_names()
        # if len(files) >= 8:  # applying multi processing
        #     pool = Pool(processes=8)
        #     pool.map(func, files)
        #     pool.close()
        #     pool.join()
        # else:
        for f in files:
            func(f)
        toc = time()
        print("Stitch processing elapsed time: {:.2f}".format(toc - tic) + "s")
        self.evaluator.average()

    def __call__(self):
        if cfg.TEST.USE_DILATED_PREDICT:
            self._sub_func(self._sub_func_dilated)
        else:
            self._sub_func(self._sub_func_overlap)
