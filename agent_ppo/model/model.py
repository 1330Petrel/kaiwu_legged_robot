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
from agent_ppo.conf.conf import Config


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
        self.critic = CriticWithHeightConv(
            mlp_input_dim_c,
            critic_hidden_dims,
            activation
        )
        # critic_layers = []
        # critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        # critic_layers.append(activation)
        # for l in range(len(critic_hidden_dims)):
        #     if l == len(critic_hidden_dims) - 1:
        #         critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
        #     else:
        #         critic_layers.append(
        #             nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1])
        #         )
        #         critic_layers.append(nn.LayerNorm([critic_hidden_dims[l + 1]]))
        #         critic_layers.append(activation)
        # self.critic = nn.Sequential(*critic_layers)

        # Action noise initialization
        # 动作噪声初始化
        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None
        # disable args validation for speedup
        # 禁用分布验证加速
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

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
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean * 0.0 + std)

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
        self.trajectory_history = torch.concat(
            (self.trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1
        )
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
        latent_vector = self.history_encoder(history)
        actions_mean = self.actor(
            torch.cat((latent_vector, observations[:, 9:12]), dim=-1)
        )
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


class CriticWithHeightConv(nn.Module):
    def __init__(self, mlp_input_dim_c: int, critic_hidden_dims, activation: nn.Module):
        super().__init__()
        # ---- 高度图配置 ----
        self.height_h, self.height_w = 17, 11
        self.height_dim = self.height_h * self.height_w  # 187
        assert mlp_input_dim_c >= self.height_dim, "mlp_input_dim_c 小于 height_dim!"
        self.other_dim = mlp_input_dim_c - self.height_dim

        # （可选）把测量范围编码为坐标通道，默认关闭，避免改变现有输入分布
        self.use_coord_channels = True
        self.register_buffer(
            "coord_x",
            torch.linspace(-0.8, 0.8, steps=self.height_h).view(1, 1, self.height_h, 1).expand(1, 1, self.height_h, self.height_w)
        )
        self.register_buffer(
            "coord_y",
            torch.linspace(-0.5, 0.5, steps=self.height_w).view(1, 1, 1, self.height_w).expand(1, 1, self.height_h, self.height_w)
        )
        in_channels = 3 if self.use_coord_channels else 1

        # ---- 高度图卷积编码器（保持轻量，输出为全局池化后的通道向量）----
        # 你可以按需调整通道数与层数；AdaptiveAvgPool2d(1) 保证展平后维度固定为 C_out
        conv_out_channels = 32
        self.height_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=True),
            activation,
            nn.Conv2d(16, conv_out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            activation,
            nn.AdaptiveAvgPool2d(1),     # (N, C, 1, 1)
            nn.Flatten(),                # (N, C)
            nn.LayerNorm(conv_out_channels),
        )

        # ---- 价值网络 MLP：输入 = 其余特征 + 卷积特征 ----
        critic_input_dim = self.other_dim + conv_out_channels
        critic_layers = []
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(nn.LayerNorm([critic_hidden_dims[l + 1]]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, critic_input: torch.Tensor) -> torch.Tensor:
        """
        critic_input: (N, mlp_input_dim_c)
        假设高度向量位于输入的“尾部”187维（你说的“倒数 height_dim=187”）。
        如你的拼接顺序不同，请在这里调整切片。
        """
        assert critic_input.dim() == 2 and critic_input.size(-1) >= self.height_dim

        # 其余特征（前部）
        other = critic_input[:, :-self.height_dim]

        # 高度向量（尾部）→ (N, 1, 17, 11)
        height_flat = critic_input[:, -self.height_dim:]
        height_map = height_flat.view(-1, 1, self.height_h, self.height_w).contiguous()

        # （可选）加入坐标通道（CoordConv 风格），用测量范围 [-0.8,0.8] × [-0.4,0.4]
        if self.use_coord_channels:
            # 注意：coord_x/coord_y 是注册的 buffer，会随模型搬运设备
            coord_x = self.coord_x.expand(height_map.size(0), -1, -1, -1)
            coord_y = self.coord_y.expand(height_map.size(0), -1, -1, -1)
            height_in = torch.cat([height_map, coord_x, coord_y], dim=1)
        else:
            height_in = height_map

        # 卷积特征
        h_feat = self.height_encoder(height_in)  # (N, conv_out_channels)

        # 拼接并过 MLP critic
        critic_feat = torch.cat([other, h_feat], dim=-1)
        value = self.critic(critic_feat)  # (N, 1)
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
