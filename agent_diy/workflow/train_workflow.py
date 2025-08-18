#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from tools.base_env_process import IsaacProcessManager
from kaiwu_agent.utils.common_func import Frame, attached
import random
import os
from agent_diy.feature.definition import (
    sample_process,
)
from agent_diy.conf.conf import Config
from tools.model_pool_utils import get_valid_model_pool
from tools.train_env_conf_validate import read_usr_conf
from tools.metrics_utils import get_training_metrics


@attached
def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):

    # Number of agents, in legged_robot_locomotion_control the value is 2
    # 智能体数量，在legged_robot_locomotion_control中值为2
    agent = agents[0]

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error(f"usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        raise Exception("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")

    game_id = f'{os.getenv("KAIWU_TASK_ID", "1")}_{os.getenv("KAIWU_ROUND_INDEX", "1")}_1'

    # legged_robot_locomotion_control environment
    # legged_robot_locomotion_control环境
    env = IsaacProcessManager(game_id)

    # Please implement your DIY algorithm flow
    # 请实现你DIY的算法流程
    # ......

    # model saving
    # 保存模型
    agent.save_model()

    env.close()

    return
