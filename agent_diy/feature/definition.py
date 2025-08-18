#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import create_cls, Frame, attached
import torch
import numpy as np
import collections
from agent_diy.conf.conf import Config


SampleData = create_cls("SampleData", obs=None, actions=None, done=None, rewards=None)

ObsData = create_cls("ObsData", feature=None, legal_action=None)

ActData = create_cls(
    "ActData",
    action=None,
)


@attached
def sample_process(collector):
    return collector.sample_process()


# Create the sample for the current frame
# 创建当前帧的样本
def build_frame(frame_no, obs, actions, dones, rewards):

    frame = Frame(
        frame_no=frame_no,
        obs=obs,
        actions=actions,
        done=dones,
        rewards=rewards,
    )
    return frame


def obs_normalizer(obs):
    pass


@attached
def SampleData2NumpyData(g_data):
    pass


@attached
def NumpyData2SampleData(s_data):
    pass
