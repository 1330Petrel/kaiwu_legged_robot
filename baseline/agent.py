#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import torch.optim as optim

from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    BaseAgent,
)
from Baseline.model.model import TeacherActorCritic
from kaiwu_agent.utils.common_func import attached
from Baseline.feature.definition import *
from Baseline.conf.conf import Config
from Baseline.algorithm.algorithm import Algorithm
from tools.train_env_conf_validate import read_usr_conf, check_usr_conf


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device="cuda", logger=None, monitor=None):
        self.cur_model_name = "ActorCritic"
        self.device = device
        self.logger = logger

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
        self.num_obs = usr_conf["env"]["num_privileged_obs"]
        self.num_actor_obs = usr_conf["env"]["num_privileged_obs"]
        self.num_privileged_obs = usr_conf["env"]["num_privileged_obs"]
        self.num_critic_obs = usr_conf["env"]["num_privileged_obs"]
        self.num_actions = usr_conf["env"]["num_actions"]
        self.height_dim = usr_conf["env"]["height_dim"]  # 187
        self.privileged_dim = usr_conf["env"]["privileged_dim"]  # 27
        # self.history_length = usr_conf["env"]["history_length"]
        self.history_length = 10

        # Create Model and convert the model to a channel-last memory format to achieve better performance.
        # 创建模型, 将模型转换为通道后内存格式，以获得更好的性能。
        self.model = TeacherActorCritic(
            num_actor_obs=self.num_actor_obs,
            num_critic_obs=self.num_critic_obs,
            num_actions=self.num_actions,
            height_dim=self.height_dim,
            privileged_dim=self.privileged_dim,
            history_dim=self.history_length
            * (self.num_actor_obs - self.privileged_dim - self.height_dim - 3),
            history_length=self.history_length,
        ).to(self.device)

        self.logger.info(f"Actor MLP: {self.model.actor}")
        self.logger.info(f"Critic MLP: {self.model.critic}")
        params = [{"params": self.model.parameters(), "name": "actor_critic"}]
        self.optimizer = optim.Adam(params, lr=Config.LR)

        # env info
        # 环境信息
        self.game_id = None

        self.train_count = 0
        self.predict_count = 0

        # tools
        # 工具
        self.reward_manager = None
        self.monitor = monitor
        self.algorithm = Algorithm(
            self.model, self.optimizer, self.device, self.logger, self.monitor
        )
        self.num_steps_per_env = Config.NUM_STEPS_PER_ENV
        self.save_interval = Config.MODEL_SAVE_INTERVAL

        # init storage and model
        # 初始化样本池
        self.algorithm.init_storage(
            self.num_envs,
            self.num_steps_per_env,
            [self.num_actor_obs],
            [self.num_privileged_obs],
            [self.num_actions],
        )

        # Evaluation History Buffer
        self.eval_history_buffer = None

        super().__init__(device, logger, monitor)

    @predict_wrapper
    def predict(self, list_obs_data):
        (obs, critic_obs, history) = list_obs_data
        with torch.no_grad():
            (
                actions,
                values,
                actions_log_prob,
                action_mean,
                action_sigma,
                observations,
                critic_observations,
            ) = self.algorithm.act_(obs, critic_obs, history)

        return [ActData(action=actions)]

    @exploit_wrapper
    def exploit(self, list_obs_data):

        (obs) = list_obs_data

        # 初始化
        if self.eval_history_buffer is None:
            if obs is not None:
                num_envs = obs.shape[0]
                self.eval_history_buffer = torch.zeros(
                    size=(
                        num_envs,
                        self.history_length,
                        self.num_obs - self.privileged_dim - self.height_dim - 3,
                    ),
                    device=self.device,
                )

        # Reset
        action_slice = obs[:, 36:48]
        command_norm = torch.norm(action_slice, p=1, dim=1)
        is_reset_signal = command_norm < 1e-6
        if is_reset_signal.any():
            reset_indices = is_reset_signal.nonzero(as_tuple=False).squeeze(-1)
            self.eval_history_buffer[reset_indices] = 0.0

        # Update
        obs_without_command = torch.concat(
            (
                obs[:, 3:9],
                obs[:, 12:],
            ),
            dim=1,
        )
        self.eval_history_buffer[:, :-1] = self.eval_history_buffer[:, 1:].clone()
        self.eval_history_buffer[:, -1] = obs_without_command
        history_for_model = self.eval_history_buffer.flatten(1)

        with torch.no_grad():
            obs[:, 9] = 1.0 * 2
            actions = self.algorithm.actor_critic.act_deterministic(
                obs, history_for_model
            )

        return [ActData(action=actions)]

    @learn_wrapper
    def learn(self, list_sample_data=None):
        self.train_count += 1
        return self.algorithm.learn(list_sample_data)

    def predict_local(self, obs, critic_obs, history):
        """
        local predict
        本地预测
        """
        self.predict_count += 1
        return self.algorithm.act_(obs, critic_obs, history)

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
