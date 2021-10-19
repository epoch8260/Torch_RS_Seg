# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-05-13 16:16 
Written by Yuwei Jin (642281525@qq.com)
"""

import os
import cv2
import math
import json
from multiprocessing import Pool

from tools import utils

patch_size = 512
overlap = 256
im_suffix = '.tif'
gt_suffix = '.tif'
phase = 'test'
dataset_name = 'Chicago'

'''
original image organization example:
root_dir:
----dataset_name
-------train
----------gt
'''
root_dir = '/home/jyw/BaseDataset/建筑提取/Chicago'
root_dir = os.path.join(root_dir, phase)
im_src = os.path.join(root_dir, 'image')
gt_src = os.path.join(root_dir, 'label')

output_root_dir = '/home/jyw/Dataset/BE'
output_root_dir = os.path.join(output_root_dir, dataset_name, phase)
im_output_src = os.path.join(output_root_dir, 'image')
gt_output_src = os.path.join(output_root_dir, 'label')
crop_info_dir = os.path.join(output_root_dir, 'crop_info')

del root_dir, output_root_dir


def record_crop_info(im, im_name):
    rows, cols = im.shape[0], im.shape[1]
    m = math.ceil((rows - patch_size) / (patch_size - overlap) + 1)
    n = math.ceil((cols - patch_size) / (patch_size - overlap) + 1)

    info = {'rows': rows,
            'cols': cols,
            'patch_size': patch_size,
            'overlap': overlap,
            'm': m,
            'n': n
            }

    im_name, _ = im_name.split('.')

    with open(os.path.join(crop_info_dir, im_name + '_crop_info.json'), 'w') as f:
        f.write(json.dumps(info))


def _crop(im_name):
    im = cv2.imread(os.path.join(im_src, im_name))
    # ttt = im_name.split('_RGB.tif')[0]
    gt = cv2.imread(os.path.join(gt_src, im_name), cv2.IMREAD_GRAYSCALE)

    record_crop_info(im, im_name)
    im_name, _ = im_name.split('.')

    rows, cols = im.shape[0], im.shape[1]
    m = math.ceil((rows - patch_size) / (patch_size - overlap) + 1)
    n = math.ceil((cols - patch_size) / (patch_size - overlap) + 1)

    k = 0
    for i in range(m):
        x1 = i * (patch_size - overlap)
        x2 = x1 + patch_size
        if x2 >= rows:  # 超出图像范围
            x1, x2 = rows - patch_size, rows
        if i == 0:
            x1, x2 = 0, patch_size

        for j in range(n):
            y1 = j * (patch_size - overlap)
            y2 = y1 + patch_size
            if y2 >= cols:  # 超出图像范围
                y1, y2 = cols - patch_size, cols
            if j == 0:
                y1, y2 = 0, patch_size

            im_patch = im[x1:x2, y1:y2, :]
            gt_patch = gt[x1:x2, y1:y2]

            k += 1

            cv2.imwrite(os.path.join(im_output_src, im_name + '-' + str(k) + im_suffix), im_patch)
            cv2.imwrite(os.path.join(gt_output_src, im_name + '-' + str(k) + im_suffix), gt_patch)


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
