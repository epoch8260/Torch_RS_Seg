# -*- encoding: utf-8 -*-
"""
Copyright (c) Yuwei Jin
Created on 2021/10/18 下午9:38
Written by Yuwei Jin (yuwei_jin@163.com)
"""

import os
import cv2

from configs import cfg
from tools import utils
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    train_lists = utils.get_whole_file_list(os.path.join(cfg.DATASET.TRAIN_SET, 'label'))
    print(len(train_lists))

    filterd_lists = []
    for f in train_lists:
        gt = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if gt.sum() > 500:
            filterd_lists.append(f)

    print(len(filterd_lists))

    unlabeled_data, labeled_data = train_test_split(filterd_lists, test_size=0.50)

    utils.list_to_txt(unlabeled_data, 'unlabeled_data.txt')
    utils.list_to_txt(labeled_data, 'labeled_data.txt')