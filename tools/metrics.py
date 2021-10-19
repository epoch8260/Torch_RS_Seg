# -*- encoding: utf-8 -*-
"""
Copyright (c) Yuwei Jin
Created on 2021/3/27 下午10:15
Written by Yuwei Jin(yuwei_jin@163.com)
"""

import os
import cv2
import numpy as np
import pandas as pd

from tabulate import tabulate

from libs.base_metrics import Metrics
from tools import utils
from configs import cfg


class BEMetrics(Metrics):
    """ Base metrics for building extraction task"""

    def __init__(self):
        super().__init__(num_class=cfg.DATASET.NUM_CLASS)

        # supported metrics
        self.supported_metrics = {
            'pre': super().cal_precision,
            'rec': super().cal_recall,
            'f1': super().cal_fbeta_score,
            'oa': super().cal_overall_accuracy,
            'iou': super().cal_IoU,
            'miou': super().cal_mIoU,
            'kappa': super().cal_Kappa
        }

    def cal_acc(self, employed_metrics=cfg.TRAIN.EMPLOYED_METRICS, return_acc_dict=False) -> list or tuple:
        """Calculate the accuracy according to the specific metric, e.g., precision, recall, F1, IoU, mIoU
        """
        acc = []
        for m in employed_metrics:
            res = self.supported_metrics[m]()
            res = res * 100 if m not in ['kappa', 'mcc'] else res  # format
            if isinstance(res, np.ndarray):
                res = res[-1]
            acc.append(res)

        self.reset_confusion_matrix()  # reset confusion matrix

        if return_acc_dict:
            acc_dict = {}
            for i, m in enumerate(employed_metrics):
                if m in ['kappa', 'mcc']:
                    acc_dict[m] = "{:.4f}".format(acc[i])
                else:
                    acc_dict[m] = "{:.2f}".format(acc[i])
            return acc, acc_dict
        else:
            return acc


class BETestMetrics(BEMetrics):
    """
    Class for computing the accuracy of change detection task.
    """
    def __init__(self):
        super().__init__()

        self.confusion_matrix = None

        # for test
        columns = ['image_ID']
        columns.extend(cfg.TEST.METRICS)
        self.test_metric = pd.DataFrame(columns=columns)  # record the accuracy for per test image
        self.test_step = 0

        self.conf_mat = np.zeros((self.num_class, self.num_class))  # confusion matrix for test

    def cal_batch_metric(self, img_name, prob, label):
        """
        prob: model prediction
        img_name: test image name
        label src: corresponding label dir. if label_src=None, label_src=cfg.DATASET.TEST else label_src=cfg.DATASET.TEST_WHOLE_LABEL
        """
        self.test_step += 1

        if isinstance(prob, np.ndarray):
            prob = prob > 0.5
        else:
            prob = utils.tensor_binarize(prob)

        hist = utils.fast_hist(prob, label, num_class=cfg.DATASET.NUM_CLASS)
        self.conf_mat += hist

        self.confusion_matrix = hist

        if label.sum() > 0:
            acc = self.cal_acc(employed_metrics=cfg.TEST.METRICS)
        else:
            acc = [np.nan for _ in cfg.TEST.METRICS]

        data = [img_name]
        data.extend(acc)

        self.test_metric = utils.add_list_to_dataframe(data=data, index=self.test_step, df=self.test_metric)

    def average(self):
        """ Main function:
            1. Save accuracy result to disk for each test image.
            2. Calculate average accuracy for all test images.
        """
        output_dir = os.path.join(cfg.TEST.OUTPUT_DIR, 'acc')
        utils.mk_dirs_r(output_dir)

        csv_file_name = cfg.MODEL.NAME + '_' + cfg.DATASET.NAME + '_acc.csv'
        self.test_metric.to_csv(os.path.join(output_dir, csv_file_name))

        # compute average accuracy
        columns = ['dataset', 'method']
        columns.extend(cfg.TEST.METRICS)
        df = pd.DataFrame(columns=columns)

        metric = self.test_metric.values[:, 1::]
        avg = list(np.nanmean(metric, axis=0))
        data = [cfg.DATASET.NAME, cfg.MODEL.NAME]
        data.extend(avg)
        df = utils.add_list_to_dataframe(data=data, index=0, df=df)

        # --- test different accuracy calculation mode
        self.confusion_matrix = self.conf_mat
        acc = self.cal_acc(cfg.TEST.METRICS)
        data = [cfg.DATASET.NAME, cfg.MODEL.NAME]
        data.extend(acc)
        df = utils.add_list_to_dataframe(data=data, index=1, df=df)

        df.index = ['acc_avg', 'hist_sum']
        csv_name = cfg.MODEL.NAME + '_' + cfg.DATASET.NAME + '_avg_acc.csv'
        df.to_csv(os.path.join(output_dir, csv_name))

        print(tabulate(df, headers=df.columns, tablefmt='psql'))
