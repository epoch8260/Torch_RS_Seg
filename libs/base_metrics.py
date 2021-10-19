# -*- encoding: utf-8 -*-
"""
Copyright (c) Yuwei Jin
Created on 2021/4/11 下午3:48
Written by Yuwei Jin(yuwei_jin@163.com)
"""

import math
import numpy as np


class Metrics(object):
    """Base metrics for pixel-wise segmentation task.
    implemented metrics including:
        1. class-wise pixel accuracy (precision)
        2. class-wise recall
        3. class-wise F1-score
        4. overall accuracy
        5. class-wise IoU
        6. mean IoU
        7. class-wise frequency weighted IoU
        8. Kappa coefficient
        9. Matthews Correlation Coefficient
        # ----- for change detection task -----
        10. missed detection rate
        11. false alarm rate
        12. overall error
    """

    def __init__(self, num_class):

        self.num_class = num_class
        # confusion matrix: num_class*num_class
        """
        L\P      prediction
                 0      1
        label 0  TN    FP
              1  FN    TP
        """
        self.confusion_matrix = np.zeros((num_class,) * 2).astype('int64')
        self.label_range = range(self.num_class)

        self.eps = 1e-10

    def reset_confusion_matrix(self):
        # after computing the confusion matrix for each iter, then reset the confusion matrix
        self.confusion_matrix = np.zeros((self.num_class, )*2)

    def cal_confusion_matrix(self, input, label):
        """
        Args:
            input: model prediction (Type: numpy array)
            label: reference ground truth (Type: numpy array)
        Return:
            confusion matrix (Type: numpy array)

        """
        if label.max() != self.num_class - 1:
            raise ValueError('The value range in input should be consistent with the number of classes. Got {}'.format(label.max()))
        assert len(input.shape) == len(label.shape)
        input = input.astype('int')
        label = label.astype('int')

        mask = (label >= 0) & (label < self.num_class)
        label = self.num_class * label[mask] + input[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        conf_mat = count.reshape(self.num_class, self.num_class)

        return conf_mat

    def cal_precision(self):
        # precision = TP / (TP + FP)
        precision = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=0) + self.eps)
        return precision

    def cal_recall(self):
        # recall = TP / (TP + FN)
        recall = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + self.eps)
        return recall

    def cal_fbeta_score(self, beta=1):
        """
        Fbeta = (1+beta^2)*precision*recall/(beta^2*precision + recall)
        """
        pre = self.cal_precision()
        rec = self.cal_recall()
        f1 = 2 * beta * pre * rec / (beta * pre + rec + self.eps)
        return f1

    def cal_overall_accuracy(self):
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def cal_IoU(self):
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix)
        IoU = intersection / (union + self.eps)
        return IoU

    def cal_mIoU(self):
        # Calculate mean IoU
        return np.nanmean(self.cal_IoU())

    def cal_fwIoU(self):
        # fwIoU = [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                                               np.diag(self.confusion_matrix))
        fwIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwIoU

    def cal_Kappa(self):
        p0 = self.cal_overall_accuracy()
        cols = np.sum(self.confusion_matrix, axis=0)
        rows = np.sum(self.confusion_matrix, axis=1)
        sum_total = self.confusion_matrix.sum()
        Pe = np.dot(cols, rows) / float(sum_total ** 2)
        kappa = (p0 - Pe) / (1 - Pe + self.eps)

        return kappa

    def cal_MCC(self):
        tp = np.diag(self.confusion_matrix)

        # calculate tn
        tn = np.zeros(self.num_class)
        for i in range(self.num_class):
            temp = np.ma.masked_array(self.confusion_matrix, mask=False)
            temp.mask[i, :] = True
            temp.mask[:, i] = True
            tn[i] = temp.sum()

        fp = np.sum(self.confusion_matrix, axis=1) - tp
        fn = np.sum(self.confusion_matrix, axis=0) - tp

        # normalization
        n = self.confusion_matrix.sum()
        tn = tn / n
        fn = fn / n
        tp = tp / n
        fp = fp / n

        num = np.multiply(tp, tn) - np.multiply(fp, fn)
        den = np.multiply(tp + fp, tp + fn)
        den = np.multiply(den, tn + fp)
        den = np.multiply(den, tn + fn)
        den = np.sqrt(den)

        mcc = num / (den + self.eps)

        return mcc

    def cal_missed_detection_rate(self):
        """MD = Nm / Nc
           Nm = FN
        """
        fn = np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix)
        fn = fn.tolist()
        fn.reverse()
        fn = np.asarray(fn)
        p = np.sum(self.confusion_matrix, axis=1)
        md = fn / (p + self.eps)
        return md

    def cal_false_alarm_rate(self):
        """FA = FP / (TN + FP)"""
        fp = np.sum(self.confusion_matrix, axis=1) - np.diag(self.confusion_matrix)
        n = np.sum(self.confusion_matrix, axis=1)
        fa = fp / n
        fa = fa.tolist()
        fa.reverse()
        fa = np.asarray(fa)
        return fa

    def cal_overall_error(self):
        oa = self.cal_overall_accuracy()
        oe = 1 - oa
        return oe
