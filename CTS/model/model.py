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
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
from typing import NoReturn


class CTS_ActorCritic(nn.Module):
    """
    CTS Actor-Critic网络模型，包含：
    1. 教师使用的特权编码器
    2. 学生使用的本体感知编码器
    3. 教师和学生共享的策略网络（Actor）和价值网络（Critic）
    """

    is_recurrent = False

    def __init__(
        self,
        num_proprioceptive_obs: int,  # 本体感知观测维度 (for student)
        num_privileged_obs: int,  # 特权观测维度 (for teacher)
        num_actions: int,
        history_length: int,
        encoder_hidden_dims: list = [512, 256],
        actor_hidden_dims: list = [512, 256, 128],
        critic_hidden_dims: list = [512, 256, 128],
        latent_dim: int = 40,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **kwargs: dict,
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(CTS_ActorCritic, self).__init__()

        activation_mod = get_activation(activation)

        # 1. Privileged Encoder (for Teacher)
        # 1. 特权编码器（教师使用）
        self.privileged_encoder = nn.Sequential(
            *mlp_factory(
                activation_mod,
                input_dims=num_privileged_obs,
                out_dims=latent_dim,
                hidden_dims=encoder_hidden_dims,
            )
        )

        # 2. Proprioceptive Encoder (for Student)
        # 2. 本体感知编码器（学生使用）
        self.proprioceptive_encoder = nn.Sequential(
            *mlp_factory(
                activation_mod,
                input_dims=(num_proprioceptive_obs - 3) * history_length + 3,  # cmd
                out_dims=latent_dim,
                hidden_dims=encoder_hidden_dims,
            )
        )

        # 3. Shared Policy Network (Actor)
        # 3. 共享的策略网络 (Actor)
        self.actor = nn.Sequential(
            *mlp_factory(
                activation_mod,
                input_dims=latent_dim + num_proprioceptive_obs,
                out_dims=num_actions,
                hidden_dims=actor_hidden_dims,
            )
        )

        # 4. Shared Value Network (Critic)
        # 4. 共享的价值网络 (Critic)
        self.critic = nn.Sequential(
            *mlp_factory(
                activation_mod,
                input_dims=latent_dim + num_privileged_obs,
                out_dims=1,
                hidden_dims=critic_hidden_dims,
                # layer_norm=True,  # 在Critic中使用层归一化
            )
        )

        # Action noise initialization
        # 动作噪声初始化
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        # Action distribution
        # 动作分布
        self.distribution = None

        # disable args validation for speedup
        # 禁用分布验证加速
        Normal.set_default_validate_args(False)

        # Initialize network weights
        # 初始化网络权重
        # self._init_networks()

    def _init_networks(self) -> None:
        self.init_weights(self.privileged_encoder, np.sqrt(2))
        self.init_weights(self.proprioceptive_encoder, np.sqrt(2))
        self.init_weights(self.actor, [np.sqrt(2), np.sqrt(2), np.sqrt(2), 0.01])
        self.init_weights(self.critic, 1.0)

    @staticmethod
    def init_weights(sequential: nn.Sequential, scales) -> None:

        def get_scale(idx: int) -> float:
            return scales[idx] if isinstance(scales, (list, tuple)) else scales

        idx = 0
        for module in sequential:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=get_scale(idx))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                idx += 1
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def reset(self, dones) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_input: torch.Tensor) -> None:
        mean = self.actor(actor_input)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(
        self,
        current_proprio_obs: torch.Tensor,
        privileged_obs: torch.Tensor,
        proprio_history: torch.Tensor,
        teacher_mask: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        latent_vector = self.get_latent_vector(
            privileged_obs, proprio_history, teacher_mask
        )
        actor_input = torch.cat((latent_vector, current_proprio_obs), dim=-1)
        self.update_distribution(actor_input)
        return self.distribution.sample()

    def act_inference(
        self, current_proprio_obs: torch.Tensor, proprio_history: torch.Tensor
    ) -> torch.Tensor:
        with torch.inference_mode():
            z_student = self.proprioceptive_encoder(proprio_history)
            z_student_norm = F.normalize(z_student, p=2, dim=-1)

            actor_input = torch.cat((z_student_norm, current_proprio_obs), dim=-1)

            return self.actor(actor_input)

    def evaluate(
        self,
        privileged_obs: torch.Tensor,
        proprio_history: torch.Tensor,
        teacher_mask: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        # The value loss gradient NOT flow back through z_t to the encoders
        with torch.no_grad():
            latent_vector = self.get_latent_vector(
                privileged_obs, proprio_history, teacher_mask
            )
        critic_input = torch.cat((latent_vector, privileged_obs), dim=-1)
        value = self.critic(critic_input)
        return value

    def get_latent_vector(
        self,
        privileged_obs: torch.Tensor,
        proprio_history: torch.Tensor,
        teacher_mask: torch.Tensor,
    ) -> torch.Tensor:
        # --- 1. Compute both latent vectors ---
        z_privileged = self.privileged_encoder(privileged_obs)
        z_privileged_norm = F.normalize(z_privileged, p=2, dim=-1)

        with torch.no_grad():
            # Student's latent vector is detached to prevent gradient flow
            z_proprio = self.proprioceptive_encoder(proprio_history)
            z_proprio_norm = F.normalize(z_proprio, p=2, dim=-1)

        # --- 2. Select latent vector based on env type ---
        latent_vector = torch.where(teacher_mask, z_privileged_norm, z_proprio_norm)

        return latent_vector

    def get_latent_vectors_for_reconstruction(
        self, privileged_obs: torch.Tensor, proprio_history: torch.Tensor
    ) -> tuple:
        # Computes latent vectors specifically for the reconstruction loss
        # The teacher's vector is detached here to prevent incorrect gradient flow
        with torch.no_grad():
            z_privileged = self.privileged_encoder(privileged_obs)
            z_privileged_norm = F.normalize(z_privileged, p=2, dim=-1)

        z_proprio = self.proprioceptive_encoder(proprio_history)
        z_proprio_norm = F.normalize(z_proprio, p=2, dim=-1)

        return z_privileged_norm, z_proprio_norm

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)


def get_activation(act_name: str) -> nn.Module:
    act_dict = {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "crelu": nn.CELU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "softplus": nn.Softplus(),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),
        "mish": nn.Mish(),
        "identity": nn.Identity(),
    }

    act_name = act_name.lower()
    if act_name in act_dict:
        return act_dict[act_name]
    else:
        raise ValueError(
            f"Invalid activation function '{act_name}'. Valid activations are: {list(act_dict.keys())}"
        )


def mlp_factory(
    activation: nn.Module,
    input_dims: int,
    out_dims: int,
    hidden_dims: list,
    layer_norm: bool = False,
    last_act: bool = False,
) -> list:
    """
    MLP工厂函数，根据输入、输出和隐藏层维度构建一个MLP

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
