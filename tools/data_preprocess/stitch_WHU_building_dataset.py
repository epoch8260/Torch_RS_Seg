# -*- encoding: utf-8 -*-
"""
Copyright (c) Yuwei Jin
Created on 2021/10/22 下午2:51
Written by Yuwei Jin (yuwei_jin@163.com)
"""

import os
import cv2
import numpy as np

from tools import utils

phase = 'test'
root_dir = '/home/jyw/My_DatasetPatches/BE/WHU'

out_root_dir = '/home/jyw/My_BaseDatsetRepo/建筑提取/WHU/stitched_images'
im_path = os.path.join(root_dir, phase, 'image')
gt_path = os.path.join(root_dir, phase, 'label')
out_im_path = os.path.join(out_root_dir, phase, 'image')
out_gt_path = os.path.join(out_root_dir, phase, 'label')

patch = 512


def _sub_func(m, n, save_name, flag=None):
    rows = m * patch
    cols = n * patch

    im = np.zeros((rows, cols, 3), dtype=np.uint8)
    gt = np.zeros((rows, cols), dtype=np.uint8)

    k = 0
    for i in range(m):
        x1 = i * patch
        x2 = (i + 1) * patch
        for j in range(n):
            y1 = j * patch
            y2 = (j + 1) * patch

            if flag is not None:
                s = flag + str(k) + '.tif'
            else:
                s = str(k) + '.tif'

            k += 1

            print(s)

            im_patch = cv2.imread(os.path.join(im_path, s))
            gt_patch = cv2.imread(os.path.join(gt_path, s), cv2.IMREAD_GRAYSCALE)

            im[x1:x2, y1:y2, :] = im_patch
            gt[x1:x2, y1:y2] = gt_patch

    cv2.imwrite(os.path.join(out_im_path, save_name), im)
    cv2.imwrite(os.path.join(out_gt_path, save_name), gt)


def stitch_train():
    utils.mk_dirs_r(out_im_path)
    utils.mk_dirs_r(out_gt_path)

    # im_lists = utils.get_image_name_list(im_path)

    _sub_func(42, 17, 'test1.tif')
    _sub_func(74, 23, 'test2.tif', flag='2_')


if __name__ == '__main__':
    stitch_train()
