#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import torch
from torch import nn
from torch.distributions import Normal


class TeacherActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_obs,
        history_length,
        num_actions,
        latent_dim=32,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[256, 128, 64],
        activation="elu",
        init_noise_std=1.0,
        **kwargs
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(TeacherActorCritic, self).__init__()

        activation = get_activation(activation)
        num_obs = num_obs - 3  # no lin_vel
        self.history_length = history_length
        self.num_obs = num_obs

        # 1. 历史编码器 (MLP Encoder)
        # 输入是展平后的历史本体感受信息，输出是低维的潜向量 z
        self.mlp_encoder = nn.Sequential(
            *mlp_factory(
                activation=activation,
                input_dims=history_length * num_obs,
                out_dims=latent_dim,
                hidden_dims=encoder_hidden_dims,
            )
        )

        # 2. 潜空间转移模型 (Transition Model)
        # 输入是当前潜向量z和当前动作a的拼接，输出是预测的下一个潜向量z'
        self.trans = nn.Sequential(
            nn.Linear(latent_dim + num_actions, latent_dim * 2),
            nn.ELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # 3. 策略网络构建
        # 输入是当前潜向量z和当前本体感受信息prop的拼接，输出是动作的均值
        self.actor = nn.Sequential(
            *mlp_factory(
                activation=activation,
                input_dims=latent_dim + num_obs,
                out_dims=num_actions,
                hidden_dims=actor_hidden_dims,
            )
        )

        # 4. 价值网络构建（含层标准化）
        # 输入与价值网络相同，输出是当前状态的价值评估 (一个标量)
        self.critic = nn.Sequential(
            *mlp_factory(
                activation=activation,
                input_dims=latent_dim + num_obs,
                out_dims=1,
                hidden_dims=critic_hidden_dims,
                layer_norm=True,
            )
        )

        # 动作分布的标准差，作为一个可学习的参数
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

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
        """根据输入观测（拼接后的z和prop）更新动作分布"""
        mean = self.actor(observations)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, std)

    def encode(self, observations):
        """将历史信息编码为潜向量 z"""
        history = observations[:, -self.history_length * self.num_obs :]
        self.z = self.mlp_encoder(history)
        return observations[:, 3 : self.num_obs + 3], self.z

    def act(self, observations, **kwargs):
        obs, self.z = self.encode(observations)
        actor_obs = torch.concat([self.z.detach(), obs], dim=-1)
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        obs, self.z = self.encode(observations)
        actor_obs = torch.concat([self.z.detach(), obs], dim=-1)
        actions_mean = self.actor(actor_obs)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        obs, self.z = self.encode(critic_observations)
        critic_obs = torch.concat([self.z, obs], dim=-1)
        value = self.critic(critic_obs)
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


def mlp_factory(
    activation, input_dims, out_dims, hidden_dims, layer_norm=False, last_act=False
):
    """
    一个多层感知机（MLP）的工厂函数，根据输入、输出和隐藏层维度构建一个MLP

    Args:
        activation (nn.Module): 激活函数实例
        input_dims (int): 输入维度
        out_dims (int): 输出维度。如果为None，则没有输出层
        hidden_dims (list of int): 隐藏层维度列表
        layer_norm (bool): 是否在每个隐藏层之后添加层归一化
        last_act (bool): 是否在最后一层（输出层）之后也添加激活函数

    Returns:
        list: 一个包含所有网络层的列表
    """
    layers = []
    # 输入层
    layers.append(nn.Linear(input_dims, hidden_dims[0]))
    layers.append(activation)
    # 隐藏层
    for l in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dims[l + 1]))
        layers.append(activation)

    # 输出层
    if out_dims:
        layers.append(nn.Linear(hidden_dims[-1], out_dims))
    # 是否添加最后的激活函数
    if last_act:
        layers.append(activation)

    return layers
