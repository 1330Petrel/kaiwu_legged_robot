#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.modules import rnn
from Baseline.conf.conf import Config


class TeacherActorCritic(nn.Module):
    """
    Teacher model's Actor-Critic network with history encoder, policy network and value network
    教师模型的Actor-Critic网络, 包含历史编码器、策略网络和价值网络
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        encoder_hidden_dims=[256, 128],
        predictor_hidden_dims=[64, 32],
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        fixed_std=False,
        latent_dim=32 + 3,
        height_dim=187,
        privileged_dim=3 + 24,
        history_dim=42 * 1,
        history_length=1,
        single_history_dim=42,
        **kwargs
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(TeacherActorCritic, self).__init__()

        activation = get_activation(activation)

        size = os.getenv("ENV_SIZE", 8)
        self.trajectory_history = torch.zeros(
            size=(int(size), history_length, single_history_dim), device="cuda"
        )

        self.latent_dim = latent_dim
        self.height_dim = height_dim
        self.privileged_dim = privileged_dim

        # Input dimension for the actor network, composed of latent vector dimension and linear velocity/command dimensions
        # 策略网络的输入维度，由潜在向量维度和线速度、命令维度组成
        mlp_input_dim_a = latent_dim + 3
        # Input dimension for the critic network, directly using privileged observation dimensions
        # 价值网络的输入维度，直接使用特权观测的维度
        mlp_input_dim_c = num_critic_obs

        # History Encoder
        # 历史编码器构建
        encoder_layers = []
        encoder_layers.append(nn.Linear(history_dim, encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], latent_dim))
            else:
                encoder_layers.append(
                    nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1])
                )
                encoder_layers.append(activation)
        self.history_encoder = nn.Sequential(*encoder_layers)

        # Build policy network
        # 策略网络构建
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1])
                )
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # # Build value network with layer norm
        # # 价值网络构建（含层标准化）
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1])
                )
                critic_layers.append(nn.LayerNorm([critic_hidden_dims[l + 1]]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise initialization
        # 动作噪声初始化
        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None
        # disable args validation for speedup
        # 禁用分布验证加速
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        if not self.fixed_std:
            std = torch.clamp(self.std, min=0.01, max=2.0)
        else:
            std = self.std.to(mean.device)
        self.distribution = Normal(mean, std)

    def act(self, observations, history, **kwargs):
        latent_vector = self.history_encoder(history)
        command = observations[:, self.privileged_dim + 6 : self.privileged_dim + 9]
        concat_observations = torch.concat((latent_vector, command), dim=-1)
        self.update_distribution(concat_observations)
        return self.distribution.sample()

    def get_latent_vector(self, observations, history, **kwargs):
        latent_vector = self.history_encoder(history)
        return latent_vector

    def get_linear_vel(self, observations, history, **kwargs):
        latent_vector = self.history_encoder(history)
        linear_vel = latent_vector[:, -3:]
        return linear_vel

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history=None):
        obs_without_command = torch.concat(
            (
                observations[:, 3:9],
                observations[:, 12:],
            ),
            dim=1,
        )
        self.trajectory_history[:, :-1] = self.trajectory_history[:, 1:].clone()
        self.trajectory_history[:, -1] = obs_without_command
        with torch.no_grad():
            latent_vector = self.history_encoder(self.trajectory_history.flatten(1))
            command = observations[:, 9:12]
            concat_observations = torch.concat((latent_vector, command), dim=-1)
            actions_mean = self.actor(concat_observations)
        return actions_mean

    def act_deterministic(self, observations, history):
        """
        根据当前观测和【外部传入的】历史数据，为评估时的 exploit 函数设计
        """
        # history 参数期望是已经展平的 [batch, history_length * obs_dim] 格式
        with torch.no_grad():
            latent_vector = self.history_encoder(history)
            actions_mean = self.actor(
                torch.cat((latent_vector, observations[:, 9:12]), dim=-1)
            )
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
