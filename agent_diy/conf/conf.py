#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class Config:
    # Basic hyperparameters
    # 基础超参数
    LR = 3e-4

    # training params
    # 训练参数
    NUM_LEARNING_EPOCHS = 5

    # mini batch size =  nsteps / nminibatches
    # 小批量大小 = 步数总数 / 小批量数量
    NUM_MINI_BATCHES = 4
    MIN_NORMALIZED_STD = [0.05, 0.02, 0.05] * 4

    # Model saving settings
    # 模型保存设置
    MODEL_SAVE_INTERVAL = 500

    # Environment interaction settings
    # 环境交互设置
    NUM_STEPS_PER_ENV = 24

    # check for potential saves every this many iterations
    # 每隔指定迭代次数检查可能的保存点
    MODEL_SAVE_INTERVAL = 500
