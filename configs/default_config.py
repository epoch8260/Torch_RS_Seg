# -*- coding: utf-8 -*-
"""
Copyright (c) Yuwei Jin
Created on 2020-11-10 14:39
Written by Yuwei Jin (642281525@qq.com)
"""

import os
import multiprocessing
from yacs.config import CfgNode as CN

# TODO: It is necessary to change the following three items according to the specific model and data

model_name = 'GRRNet'
dataset_name = 'Potsdam'  # Vaihingen
dilated_predict = True
seed = 1024  #

# -----------------------------------------------------------------------------------------------------
_C = CN()

# Dataset related settings
_C.DATASET = CN()
_C.DATASET.NAME = dataset_name  # Massachusetts WHU
_C.DATASET.ROOT = '/home/jyw/Dataset/BE'  # dataset root dir
_C.DATASET.MEAN_STD = 'imagenet'
_C.DATASET.TRAIN_SET = os.path.join(_C.DATASET.ROOT, _C.DATASET.NAME, 'train')
_C.DATASET.VALID_SET = os.path.join(_C.DATASET.ROOT, _C.DATASET.NAME, 'val')
_C.DATASET.TEST_SET = os.path.join(_C.DATASET.ROOT, _C.DATASET.NAME, 'test')
_C.DATASET.TEST_WHOLE_LABEL = os.path.join(_C.DATASET.ROOT, _C.DATASET.NAME, 'test_whole_label')
_C.DATASET.TEST_WHOLE_IMAGE = os.path.join(_C.DATASET.ROOT, _C.DATASET.NAME, 'test_whole_image')
_C.DATASET.IMG_EXTENSION = '.tif'
_C.DATASET.NUM_CLASS = 2

# Common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = model_name
_C.MODEL.OUTPUT_STRIDE = 8  # for resnet series to control the output stride, optional[8,16,32]
_C.MODEL.BACKBONE = 'resnet50'
_C.MODEL.USE_STEM = True
_C.MODEL.INTERPOLATE_ALIGN = True
_C.MODEL.OUTPUT_DIR = os.path.join('results', _C.MODEL.NAME + '_' + _C.DATASET.NAME)
_C.MODEL.OUTPUT_CHANNELS = 1

# Training related params
_C.TRAIN = CN()
_C.TRAIN.SEED = seed
_C.TRAIN.CUDNN_DETERMINISTIC = False
_C.TRAIN.EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 6
_C.TRAIN.INIT_LR = 0.001
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 5e-4  # Weight decay
_C.TRAIN.FP16 = False
_C.TRAIN.MSG_ITER = 10
_C.TRAIN.RESUME = False  # 加载模型基础上继续训练
_C.TRAIN.LOGS_SAVE_DIR = os.path.join(_C.MODEL.OUTPUT_DIR, 'train_logs')
# support
_C.TRAIN.KEY_METRIC = 'iou'
_C.TRAIN.HIGHER_BETTER = True
_C.TRAIN.EMPLOYED_METRICS = ['pre', 'rec', 'f1', 'iou', 'miou']

_C.TRAIN.LR = CN()
_C.TRAIN.LR.SCHEDULER = 'Poly'
_C.TRAIN.LR.POWER = 0.9
_C.TRAIN.LR.USE_WARMUP = False
_C.TRAIN.LR.WARMUP_START_LR = 5e-1
_C.TRAIN.LR.WARMUP_STEPS = 5  # period of warmup
_C.TRAIN.LR.MILESTONES = None  # milestones for multi-step scheduler
_C.TRAIN.LR.ADJUST_PERIOD = None  # period of adjust learning rate
_C.TRAIN.LR.ETA_MIN_LR = 0.0  # minimum learning rate
_C.TRAIN.LR.T_MAX = _C.TRAIN.EPOCHS  # for CosineAnnealing decay

# checkpoint saving settings
_C.TRAIN.CKPT = CN()
_C.TRAIN.CKPT.SAVE_DIR = os.path.join(_C.MODEL.OUTPUT_DIR, 'checkpoint')
_C.TRAIN.CKPT.START_SAVE_EPOCH = 1
_C.TRAIN.CKPT.DELETE_OLD = False
_C.TRAIN.CKPT.SAVE_DIR = os.path.join(_C.MODEL.OUTPUT_DIR, 'checkpoint')

# data augmentation settings
_C.TRAIN.AUG = CN()
_C.TRAIN.AUG.MULTI_SCALE = True
_C.TRAIN.AUG.CENTER_CROP = False
# TODO: if the size of input data has changed, it must be changed.
_C.TRAIN.AUG.CROP_SIZE = [512, 512]
_C.TRAIN.AUG.SCALE_LIST = [0.75, 1.0, 1.25, 1.5, 1.75]
_C.TRAIN.AUG.MS_P = 0.5
# rotation
_C.TRAIN.AUG.ROTATION = False
_C.TRAIN.AUG.ROTATION_ANGLE = [90, 180, 270]
_C.TRAIN.AUG.ROTATION_P = 0.5
# flip
_C.TRAIN.AUG.FLIP = False
_C.TRAIN.AUG.FLIP_P = 0.5
# color jitter
_C.TRAIN.AUG.COLOR_JITTER = False
_C.TRAIN.AUG.C0LOR_JITTER_P = 0.5
# How much to adjust the brightness. Can be any non negative number.
# 0 gives a black image, 1 gives the original image while 2 increases the brightness by a factor of 2.
_C.TRAIN.AUG.BRIGHTNESS_FACTOR = 0.5
# How much to adjust the saturation.
# 0 will give a black and white image, 1 will give the original image while 2 will enhance the saturation by a factor of 2.
_C.TRAIN.AUG.SATURATION_FACTOR = 0.5
# How much to adjust the contrast. Can be any non negative number.
# 0 gives a solid gray image, 1 gives the original image while 2 increases the contrast by a factor of 2.
_C.TRAIN.AUG.CONTRAST_FACTOR = 0.5
# is the amount of shift in H channel and must be in the
#     interval `[-0.5, 0.5]`.
_C.TRAIN.AUG.HUE_FACTOR = 0.5

_C.TRAIN.AUG.GAUSSIAN_BLUR = False
_C.TRAIN.AUG.GAUSSIAN_BLUR_RADIUS = 2
_C.TRAIN.AUG.GAUSSIAN_BLUR_P = 0.5

_C.TRAIN.AUG.GAUSSIAN_NOISE = False
_C.TRAIN.AUG.GAUSSIAN_NOISE_MEAN = 0
_C.TRAIN.AUG.GAUSSIAN_NOISE_VAR = 0.001
_C.TRAIN.AUG.GAUSSIAN_NOISE_P = 0.5

# dataloader related params
_C.DATALOADER = CN()
_C.DATALOADER.WORKERS = multiprocessing.cpu_count()
_C.DATALOADER.PIP_MEMORY = True
_C.DATALOADER.DROP_LAST = True

# test related params
_C.TEST = CN()
_C.TEST.NORMAL = False
_C.TEST.V_FLIP = False
_C.TEST.H_FLIP = False
_C.TEST.MS = False
_C.TEST.MS_SCALE_LIST = [0.75, 1, 1.25, 1.5, 1.75]
_C.TEST.FP16_VAL = False
_C.TEST.METRICS = ['pre', 'rec', 'f1', 'iou', 'miou']

_C.TEST.USE_DILATED_PREDICT = dilated_predict

# saving settings for test
_C.TEST.OUTPUT_DIR = os.path.join(_C.MODEL.OUTPUT_DIR, 'test_results')
_C.TEST.SAVE_RES = True
_C.TEST.PROB_SAVE_DIR = os.path.join(_C.TEST.OUTPUT_DIR, 'cropped_im_res')
_C.TEST.STITCH_RES = True  # some predicted results of the cropped images may be needed to stitch.
_C.TEST.STITCH_SAVE_RES = True
_C.TEST.STITCH_SAVE_DIR = os.path.join(_C.TEST.OUTPUT_DIR, 'whole_im_res')

# result palette
_C.TEST.BACKGROUND_COLOR = [0, 0, 0]
_C.TEST.TP_COLOR = [0, 0, 255]
_C.TEST.FP_COLOR = [0, 255, 255]
_C.TEST.FN_COLOR = [0, 255, 30]
_C.TEST.ALPHA = 0.5


def update_config():
    cfg = _C.clone()
    if dataset_name is not None:
        this_dir = os.path.dirname(__file__)
        cfg_file = os.path.join(this_dir, dataset_name + '.yaml')
        cfg.merge_from_file(cfg_file)
        cfg.freeze()

    return cfg


cfg = update_config()

if __name__ == '__main__':
    print(cfg.TEST.USE_DILATED_PREDICT)

    import json

    data = json.dumps(cfg.TRAIN)
    print(json.dumps(cfg.TRAIN, sort_keys=False, indent=4))

    print(json.dumps(cfg.TRAIN))
