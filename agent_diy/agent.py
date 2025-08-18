#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    BaseAgent,
)
from agent_diy.model.model import Model
from kaiwu_agent.utils.common_func import attached
from agent_diy.feature.definition import *
from agent_diy.conf.conf import Config
from agent_diy.algorithm.algorithm import Algorithm


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        super().__init__(agent_type, device, logger, monitor)
        self.cur_model_name = ""
        self.device = device
        # Create Model and convert the model to a channel-last memory format to achieve better performance.
        # 创建模型, 将模型转换为通道后内存格式，以获得更好的性能。
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        # env info
        # 环境信息
        self.hero_camp = 0
        self.player_id = 0
        self.game_id = None

        # tools
        # 工具
        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor
        self.algorithm = Algorithm(self.model, self.device, self.logger, self.monitor)

    @predict_wrapper
    def predict(self, list_obs_data):
        (obs, critic_obs) = list_obs_data
        with torch.no_grad():
            (
                actions,
                values,
                actions_log_prob,
                action_mean,
                action_sigma,
                observations,
                critic_observations,
            ) = self.algorithm.act(obs, critic_obs)

        return [ActData(action=actions)]

    @exploit_wrapper
    def exploit(self, list_obs_data):
        (obs, critic_obs) = list_obs_data
        with torch.no_grad():
            (
                actions,
                values,
                actions_log_prob,
                action_mean,
                action_sigma,
                observations,
                critic_observations,
            ) = self.algorithm.act(obs, critic_obs)

        return [ActData(action=actions)]

    @learn_wrapper
    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def predict_local(self, obs, critic_obs):
        """
        local predict
        本地预测
        """
        return self.algorithm.act(obs, critic_obs)

    def action_process(self, act_data):
        pass

    def observation_process(self, obs_q):
        pass

    def reset(self):
        pass

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        """
        To save the model, it can consist of multiple files, and it is important to ensure that
        each filename includes the "model.ckpt-id" field.
        保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        """
        When loading the model, you can load multiple files, and it is important to ensure that
        each filename matches the one used during the save_model process.
        加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
        else:
            self.model.load_state_dict(
                torch.load(
                    model_file_path,
                    map_location=self.device,
                )
            )
            self.cur_model_name = model_file_path
            self.logger.info(f"load model {model_file_path} successfully")
