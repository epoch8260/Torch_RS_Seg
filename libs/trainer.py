# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-07-18 20:53 
Written by Yuwei Jin (642281525@qq.com)
"""
import os
import re

import torch
import numpy as np
from tensorboardX import SummaryWriter
from abc import ABC, abstractmethod
from nb_log import get_logger
from torch.cuda.amp import autocast, GradScaler

from configs import cfg
from tools import utils

scaler = GradScaler()  # fp-16 training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_class = cfg.DATASET.NUM_CLASS


class BaseTrainer(ABC):
    def __init__(self, model, optimizer, scheduler, criterion):
        self.curr_epoch = 1
        self.curr_step = 1
        self.curr_saved_epoch = None

        self.train_batch_loss = 0
        self.valid_batch_loss = 0

        self.key_metric_record = []
        self.last_best_epoch = 0

        self.train_batch_hist = np.zeros((num_class,) * 2)
        self.valid_batch_hist = np.zeros((num_class,) * 2)

        self.employed_metrics = ['loss']
        self.employed_metrics.extend(cfg.TRAIN.EMPLOYED_METRICS)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.writer = SummaryWriter(cfg.TRAIN.LOGS_SAVE_DIR)

    def epoch_reset(self):
        self.train_batch_hist = np.zeros((num_class,) * 2)
        self.valid_batch_hist = np.zeros((num_class,) * 2)
        self.train_batch_loss = 0
        self.valid_batch_loss = 0

    def step_update_lr(self):
        c1 = re.search(cfg.TRAIN.LR.SCHEDULER, 'Poly', re.IGNORECASE)
        c2 = re.search(cfg.TRAIN.LR.SCHEDULER, 'Cosine', re.IGNORECASE)
        if c1 or c2:
            self.scheduler.step()

    def epoch_update_lr(self):
        c1 = re.search(cfg.TRAIN.LR.SCHEDULER, 'Poly', re.IGNORECASE)
        c2 = re.search(cfg.TRAIN.LR.SCHEDULER, 'Cosine', re.IGNORECASE)
        if not c1 and not c2:
            self.scheduler.step()

    def get_optimizer_lr(self) -> float:
        """ Get optimizer learning rate """
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    @staticmethod
    def tensor_to_gpu(im, gt):
        im = im.cuda()
        if cfg.MODEL.OUTPUT_CHANNELS > 1:  # using softmax, label dim is: B*H*W, sigmoid is: B*1*H*W
            gt = gt.squeeze().long()
        gt = gt.cuda()
        return im, gt

    def forward_backward(self, im, gt):
        """"""
        im, gt = self.tensor_to_gpu(im, gt)
        self.optimizer.zero_grad()  # zero model gradient
        if cfg.TRAIN.FP16:
            with autocast():  # Runs the forward pass with autocasting.
                prob = self.model(im)
                loss = self.criterion(prob, gt)

                scaler.scale(loss).backward()  # gradient backward
                scaler.step(self.optimizer)  # optimizer parameters
                scaler.update()  # Updates the scale for next iteration.
        else:
            prob = self.model(im)
            loss = self.criterion(prob, gt)
            loss.backward()  # gradient backward
            self.optimizer.step()  # optimizer parameters

        self.step_update_lr()  # update learning rate if scheduler is poly, cosine
        self.curr_step += 1  # update current iteration counter

        return prob, loss.item()

    def forward(self, im, gt):
        im, gt = self.tensor_to_gpu(im, gt)
        with torch.no_grad():
            prob = self.model(im)
            loss = self.criterion(prob, gt)
        return prob, loss.item()

    @staticmethod
    def print_train_logs(log_head: str, loss: float, acc_dict: dict, phase='train'):
        logs = log_head + "loss: {:.6f}".format(loss)
        for key, value in acc_dict.items():
            logs = logs + " - " + key + ": " + str(value).rjust(5, ' ')
        logger = get_logger(phase)
        logger.info(logs)

    def report_epoch_logs(self, train_logs, valid_logs):
        key = cfg.TRAIN.KEY_METRIC
        logger = get_logger('epoch: {}-{} training summary'.format(self.curr_epoch, cfg.TRAIN.EPOCHS))
        print('--' * 74)
        logger.critical("train - loss: {:.6f} - {}: {} - time: {} min".format(train_logs['loss'], key, train_logs[key], train_logs['time']))
        logger.critical(" eval - loss: {:.6f} - {}: {} - time: {} min".format(valid_logs['loss'], key, valid_logs[key], valid_logs['time']))

        self.key_metric_record.append(float(valid_logs[cfg.TRAIN.KEY_METRIC]))
        j = utils.find_best_value_index(self.key_metric_record, cfg.TRAIN.HIGHER_BETTER)

        logger = get_logger('epoch: {}-{} training summary'.format(self.curr_epoch, cfg.TRAIN.EPOCHS))
        logger.debug('best {} record is at epoch - {}, best record - {:.2f}'.format(key, j + 1, self.key_metric_record[j]))
        print('--' * 74)

    def save_checkpoint(self, model_states):
        ckpt = cfg.TRAIN.CKPT
        start_save_epoch = ckpt.START_SAVE_EPOCH
        save_path = os.path.join(ckpt.SAVE_DIR, 'ckpt.pth')

        if self.curr_epoch >= start_save_epoch:
            if not ckpt.DELETE_OLD:
                save_path = os.path.join(ckpt.SAVE_DIR, str(self.curr_epoch) + '_' + 'ckpt.pth')
                torch.save(model_states, save_path)
                self.curr_saved_epoch = self.curr_epoch
            else:
                if self.curr_epoch == start_save_epoch:
                    torch.save(model_states, save_path)
                    self.last_best_epoch = self.curr_epoch
                    self.curr_saved_epoch = self.curr_epoch
                else:
                    value = self.key_metric_record[start_save_epoch - 1:]
                    index = utils.find_best_value_index(value, cfg.TRAIN.HIGHER_BETTER)
                    curr_best_epoch = index + start_save_epoch

                    if curr_best_epoch >= self.curr_epoch:
                        os.remove(save_path)
                        torch.save(model_states, save_path)
                        self.last_best_epoch = curr_best_epoch
                        self.curr_saved_epoch = curr_best_epoch

        logger = get_logger('checkpoint saving info')
        if self.curr_saved_epoch is not None:
            logger.debug('currently saved model is at epoch - {}'.format(self.curr_saved_epoch))
        else:
            logger.debug('the process of saving model is not started')

        print('--' * 66)

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def eval(self):
        raise NotImplementedError

    @abstractmethod
    def cal_batch_acc(self, prob: torch.Tensor, gt: torch.Tensor) -> tuple:
        """Compute batch accuracy during training and validation"""
        pass
