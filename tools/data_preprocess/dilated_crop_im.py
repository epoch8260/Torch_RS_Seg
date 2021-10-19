# -*- encoding: utf-8 -*-
"""
Copyright (c) Yuwei Jin
Created on 2021/5/14 上午8:53
Written by Yuwei Jin (yuwei_jin@163.com)
"""

import os
import cv2
import math
import json
from multiprocessing import Pool

from tools import utils

patch_size = 512
central_win_size = 256
im_suffix = '.tif'
gt_suffix = '.tif'
phase = 'test'
dataset_name = 'Potsdam'


root_dir = '/home/jyw/BaseDataset/建筑提取/Potadam/split_images/test'
im_src = os.path.join(root_dir, 'image')
gt_src = os.path.join(root_dir, 'label')

output_root_dir = '/home/jyw/Dataset/BE'
output_root_dir = os.path.join(output_root_dir, dataset_name, phase)

im_output_src = os.path.join(output_root_dir, 'image')
gt_output_src = os.path.join(output_root_dir, 'label')
crop_info_dir = os.path.join(output_root_dir, 'crop_info')

del root_dir, output_root_dir


def save_crop_info(im, im_name):
    rows, cols = im.shape[0], im.shape[1]
    m = math.ceil(rows/central_win_size)
    n = math.ceil(cols/central_win_size)

    info = {'rows': rows,
            'cols': cols,
            'patch_size': patch_size,
            'central_win_size': central_win_size,
            'm': m,
            'n': n
            }

    with open(os.path.join(crop_info_dir, im_name + '_crop_info.json'), 'w') as f:
        f.write(json.dumps(info))


def _crop(im_name):
    im = cv2.imread(os.path.join(im_src, im_name))
    gt = cv2.imread(os.path.join(gt_src, im_name), cv2.IMREAD_GRAYSCALE)

    print('process: ' + im_name)
    im_name, _ = im_name.split('.')
    save_crop_info(im, im_name)

    # pad image and label
    win = central_win_size
    rows, cols = im.shape[0], im.shape[1]
    pad = (patch_size - central_win_size) // 2
    m = math.ceil(rows / central_win_size)
    n = math.ceil(cols / central_win_size)
    top = pad
    bottom = win * m - rows + pad
    left = pad
    right = win * n - cols + pad

    mean, std = utils.load_mean_std()
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=mean)
    gt = cv2.copyMakeBorder(gt, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    k = 0
    for i in range(m):
        x1 = i * win + top - pad
        x2 = x1 + win + 2 * pad
        if i == 0:
            x1, x2 = 0, patch_size

        for j in range(n):
            y1 = j * win - pad + left
            y2 = y1 + win + 2 * pad
            if j == 0:
                y1, y2 = 0, patch_size

            im_patch = im[x1:x2, y1:y2, :]
            gt_patch = gt[x1:x2, y1:y2]

            k += 1
            cv2.imwrite(os.path.join(im_output_src, im_name + '-' + str(k) + im_suffix), im_patch)
            cv2.imwrite(os.path.join(gt_output_src, gt_output_src, im_name + '-' + str(k) + im_suffix), gt_patch)


@utils.time_counter
def crop():
    utils.mk_dirs_r(im_output_src)
    utils.mk_dirs_r(gt_output_src)
    utils.mk_dirs_r(crop_info_dir)

    im_lists = utils.get_image_name_list(im_src, im_suffix)
    if len(im_lists) >= 8:  # applying multi processing
        pool = Pool(processes=16)
        pool.map(_crop, im_lists)
        pool.close()
        pool.join()
    else:
        for f in im_lists:
            _crop(f)


if __name__ == '__main__':
    time = crop()
    print('elapsed time: {:.4f}'.format(time))



