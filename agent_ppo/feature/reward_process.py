#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import torch


def _reward_progress(self):
    """
    直接奖励机器人在X轴（前进方向）上的速度。
    """
    # self.root_states[:, 7:10] 是世界坐标系下的线速度
    # 我们只关心X方向的速度
    forward_vel = self.root_states[:, 7]

    # 只有当指令是要求前进时，才给予这个奖励，避免在其他指令下产生冲突
    # self.commands[:, 0] 是X方向的速度指令
    progress_reward = forward_vel * (self.commands[:, 0] > 0)

    # 将负的前进速度裁剪掉，我们不希望因为后退而惩罚它（其他惩罚项会处理摔倒等情况）
    return torch.clip(progress_reward, min=0.0)
