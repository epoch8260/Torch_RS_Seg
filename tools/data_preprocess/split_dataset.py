# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-06-20 19:41 
Written by Yuwei Jin (642281525@qq.com)
"""

from sklearn.model_selection import train_test_split
import os
import cv2


def get_img_list(src):
    res = []
    for f in os.listdir(src):
        if f.endswith('.tif'):
            res.append(f)
    return res


def read_im(src, f):
    t0_im = cv2.imread(os.path.join(src, 't0', f))
    t1_im = cv2.imread(os.path.join(src, 't1', f))
    label = cv2.imread(os.path.join(src, 'label', f), cv2.IMREAD_GRAYSCALE)
    return t0_im, t1_im, label


if __name__ == '__main__':
    src = r'F:\UnifiedDeepLearningDemo\ChangeDetection\CD_Data_GZ_crop'
    img_list = get_img_list(r'F:\UnifiedDeepLearningDemo\ChangeDetection\CD_Data_GZ_crop\t0')
    train_set, val_set = train_test_split(img_list, test_size=0.3)

    out_dir_root = 'CD_DATA_GZ'
    from tools import utils

    train_t0_im_output_dir = os.path.join(out_dir_root, 'train', 't0')
    train_t1_im_output_dir = os.path.join(out_dir_root, 'train', 't1')
    train_label_output_dir = os.path.join(out_dir_root, 'train', 'label')

    val_t0_im_output_dir = os.path.join(out_dir_root, 'val', 't0')
    val_t1_im_output_dir = os.path.join(out_dir_root, 'val', 't1')
    val_label_output_dir = os.path.join(out_dir_root, 'val', 'label')

    utils.mk_dirs_r(train_label_output_dir)
    utils.mk_dirs_r(train_t0_im_output_dir)
    utils.mk_dirs_r(train_t1_im_output_dir)

    utils.mk_dirs_r(val_label_output_dir)
    utils.mk_dirs_r(val_t0_im_output_dir)
    utils.mk_dirs_r(val_t1_im_output_dir)

    for f in img_list:
        t0_im, t1_im, label = read_im(src, f)
        if f in train_set:
            cv2.imwrite(os.path.join(train_t0_im_output_dir, f), t0_im)
            cv2.imwrite(os.path.join(train_t1_im_output_dir, f), t1_im)
            cv2.imwrite(os.path.join(train_label_output_dir, f), label)
        else:
            cv2.imwrite(os.path.join(val_t0_im_output_dir, f), t0_im)
            cv2.imwrite(os.path.join(val_t1_im_output_dir, f), t1_im)
            cv2.imwrite(os.path.join(val_label_output_dir, f), label)


