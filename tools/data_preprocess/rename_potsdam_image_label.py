# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-07-22 13:15 
Written by Yuwei Jin (642281525@qq.com)
"""

import os

from tools import utils


def rename_image_label(im_src, gt_src):
    im_name = utils.get_image_name_list(im_src)

    for f in im_name:
        if f.endswith('_RGB.tif'):
            new_name, _ = f.split('_RGB.tif')
            src1 = os.path.join(im_src, f)
            src2 = os.path.join(im_src, new_name + '.tif')
            os.rename(src1, src2)

            src1 = os.path.join(gt_src, new_name + '_label.tif')
            src2 = os.path.join(gt_src, new_name + '.tif')
            os.rename(src1, src2)

if __name__ == '__main__':
    im_src = r"/home/jyw/BaseDataset/建筑提取/Potadam/Original_images/image"
    gt_src = r"/home/jyw/BaseDataset/建筑提取/Potadam/Original_images/label"
    rename_image_label(im_src, gt_src)

