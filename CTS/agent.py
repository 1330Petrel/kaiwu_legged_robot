#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import os
import numpy as np

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    BaseAgent,
)
from kaiwu_agent.utils.common_func import attached
from tools.train_env_conf_validate import read_usr_conf, check_usr_conf

from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import *
from agent_ppo.model.model import CTS_ActorCritic
from agent_ppo.algorithm.algorithm import Algorithm


@attached
class Agent(BaseAgent):
    def __init__(
        self,
        agent_type: str = "player",
        device: str = "cuda",
        logger=None,
        monitor=None,
    ) -> None:
        self.cur_model_name = "CTSActorCritic"
        self.device = device
        self.logger = logger
        self.monitor = monitor

        usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", self.logger)
        if usr_conf is None:
            self.logger.error(
                f"usr_conf is None, please check agent_ppo/conf/train_env_conf.toml"
            )
            raise Exception(
                f"usr_conf is None, please check agent_ppo/conf/train_env_conf.toml"
            )

        valid, message = check_usr_conf(usr_conf, False, self.logger)
        if not valid:
            self.logger.error(
                f"check_usr_conf is {valid}, message is {message}, please check agent_ppo/conf/train_env_conf.toml"
            )
            raise Exception(
                f"check_usr_conf is {valid}, message is {message}, please check agent_ppo/conf/train_env_conf.toml"
            )

        self.num_envs = usr_conf["env"]["num_envs"]
        self.num_proprioceptive_obs = usr_conf["env"]["num_observations"] - 3 # no lin_vel
        self.num_privileged_obs = usr_conf["env"]["num_privileged_obs"]
        self.num_actions = usr_conf["env"]["num_actions"]
        self.history_length = (
            usr_conf["env"]["history_length"] + 1
        )  # include current obs

        # Create Model and convert the model to a channel-last memory format to achieve better performance.
        # 创建模型, 将模型转换为通道后内存格式，以获得更好的性能。
        self.model = CTS_ActorCritic(
            self.num_proprioceptive_obs,
            self.num_privileged_obs,
            self.num_actions,
            self.history_length,
        ).to(self.device)
        self.logger.info(f"Privileged Encoder: {self.model.privileged_encoder}")
        self.logger.info(f"Proprio Encoder: {self.model.proprioceptive_encoder}")
        self.logger.info(f"Actor MLP: {self.model.actor}")
        self.logger.info(f"Critic MLP: {self.model.critic}")

        # Create Algorithm
        # 创建算法
        self.algorithm = Algorithm(
            self.model, device=self.device, logger=self.logger, monitor=self.monitor
        )

        # env info
        # 环境信息
        self.game_id = None
        self.train_count = 0
        self.predict_count = 0

        # tools
        # 工具
        self.reward_manager = None
        self.num_steps_per_env = Config.NUM_STEPS_PER_ENV
        self.save_interval = Config.MODEL_SAVE_INTERVAL

        # init storage and model
        # 初始化样本池
        self.algorithm.init_storage(
            self.num_envs,
            self.num_steps_per_env,
            self.history_length,
            [self.num_proprioceptive_obs],
            [self.num_privileged_obs],
            [self.num_actions],
        )

        # Evaluation History Buffer
        size = os.getenv("ENV_SIZE", 5)
        self.trajectory_history = torch.zeros(
            size=(int(size), self.history_length, self.num_proprioceptive_obs - 3),
            device=self.device,
        )

        super().__init__(device, logger, monitor)

    @predict_wrapper
    def predict(self, list_obs_data) -> list:
        (current_proprio_obs, privileged_obs, proprio_history) = list_obs_data
        with torch.no_grad():
            (
                actions,
                values,
                actions_log_prob,
                action_mean,
                action_sigma,
                proprioceptive_obs_stg,
                privileged_obs_stg,
            ) = self.algorithm.act_(
                current_proprio_obs, privileged_obs, proprio_history
            )
        return [ActData(action=actions)]

    @exploit_wrapper
    def exploit(self, list_obs_data) -> list:
        (obs) = list_obs_data
        proprioceptive_obs = obs[:, 3:]

        # Reset
        action_slice = obs[:, 36:48]
        command_norm = torch.norm(action_slice, p=1, dim=1)
        is_reset_signal = command_norm < 1e-6
        if is_reset_signal.any():
            reset_indices = is_reset_signal.nonzero(as_tuple=False).squeeze(-1)
            self.trajectory_history[reset_indices] = 0.0

        # Update
        obs_without_command = torch.concat(
            (
                proprioceptive_obs[:, :6],
                proprioceptive_obs[:, 9:],
            ),
            dim=1,
        )
        self.trajectory_history[:, :-1] = self.trajectory_history[:, 1:].clone()
        self.trajectory_history[:, -1] = obs_without_command

        with torch.no_grad():
            proprioceptive_obs[:, 6] = 1.0 * 3
            actions = self.algorithm.model.act_inference(
                proprioceptive_obs, self.trajectory_history.flatten(1)
            )

        return [ActData(action=actions)]

    @learn_wrapper
    def learn(self, list_sample_data=None) -> None:
        self.train_count += 1
        return self.algorithm.learn(list_sample_data)

    def predict_local(
        self,
        current_proprio_obs: torch.Tensor,
        privileged_obs: torch.Tensor,
        proprio_history: torch.Tensor,
    ) -> tuple:
        self.predict_count += 1
        return self.algorithm.act_(current_proprio_obs, privileged_obs, proprio_history)

    def action_process(self, act_data):
        pass

    def observation_process(self, obs_q):
        pass

    def reset(self):
        pass

    @save_model_wrapper
    def save_model(self, path: str = "", id: str = "1") -> None:
        """
        To save the model, it can consist of multiple files, and it is important to ensure that
        each filename includes the "model.ckpt-id" field.
        保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path: str = "", id: str"1") -> None:
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
