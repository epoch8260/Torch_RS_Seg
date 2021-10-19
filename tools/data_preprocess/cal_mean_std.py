# -*- coding: utf-8 -*-
"""
Created on 2020/7/20 20:14
@author: jyw
"""

# TODO: This script aims to compute the mean value and standard deviation of each temporal images.

import os
import cv2
from multiprocessing import Pool
import datetime
import numpy as np
import scipy.io as io

import pandas as pd


def sub_procedure(file):
    im = cv2.imread(file).astype(np.float32)

    mean_b, std_b = cv2.meanStdDev(im[:, :, 0] / 255.0)
    mean_g, std_g = cv2.meanStdDev(im[:, :, 1] / 255.0)
    mean_r, std_r = cv2.meanStdDev(im[:, :, 2] / 255.0)

    return [mean_r, mean_g, mean_b], [std_r, std_g, std_b]


def _cal_mean_std(save_name, files):
    start_t = datetime.datetime.now()
    pool = Pool(processes=16)
    result = pool.map(sub_procedure, files)
    pool.close()
    pool.join()
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()

    print("计算均值和标准差共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")

    num_ims = len(files)
    means = np.zeros(3)
    stdevs = np.zeros(3)
    for i in range(num_ims):
        for j in range(3):
            means[j] += result[i][0][j]
            stdevs[j] += result[i][1][j]

    means = means / num_ims
    stdevs = stdevs / num_ims

    pd_mean_std = pd.DataFrame(columns=["mean", "std"])
    pd_mean_std['mean'] = means
    pd_mean_std['std'] = stdevs
    pd_mean_std.to_csv(save_name)

    print("mean:")
    print(means)
    print('stds:')
    print(stdevs)

    return means, stdevs


class CalMeanStd:
    def __init__(self, train_im_dir, dataset_name, img_extension='.tif', num_bands=3):
        self.train_im_dir = train_im_dir
        self.dataset_name = dataset_name
        self.img_extension = img_extension
        self.num_bands = num_bands

    def _get_image_names(self, im_dir):
        files = []
        for file in os.listdir(im_dir):
            if file.endswith(self.img_extension):
                files.append(os.path.join(im_dir, file))
        return files

    def run(self):
        im_src = os.path.join(self.train_im_dir, 'image')
        fname = os.path.join(self.train_im_dir, self.dataset_name + '_mean_std.csv')
        # calculation t0 img
        files = self._get_image_names(im_src)
        _cal_mean_std(fname, files)


if __name__ == '__main__':
    train_im_dir = "/home/jyw/Dataset/BE/Vaihingen/train"
    dataset_name = 'Vaihingen'

    CalMeanStd(train_im_dir=train_im_dir, dataset_name=dataset_name).run()


