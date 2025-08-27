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
from agent_ppo.conf.conf import Config

import pdb


SampleData = create_cls(
    "SampleData",
    obs=None,
    critic_obs=None,
    actions=None,
    history=None,
    target_values=None,
    advantages=None,
    returns=None,
    old_action_log_prob=None,
    old_mu=None,
    old_sigma=None,
    hid_states=None,
    masks=None,
)

ObsData = create_cls("ObsData", feature=None, legal_action=None)

ActData = create_cls(
    "ActData",
    action=None,
)


@attached
def sample_process(collector):
    return collector.sample_process()


def obs_normalizer(obs):
    pass


@attached
def SampleData2NumpyData(g_data):
    for id, data in enumerate(g_data):
        if data is None or isinstance(data, tuple):
            g_data[id] = np.zeros_like(g_data[id - 1])
        g_data[id] = data.detach().cpu().numpy()
    return g_data


@attached
def NumpyData2SampleData(s_data):
    return SampleData(
        obs=s_data[0],
        critic_obs=s_data[1],
        actions=s_data[2],
        history=s_data[3],
        target_values=s_data[4],
        advantages=s_data[5],
        returns=s_data[6],
        old_action_log_prob=s_data[7],
        old_mu=s_data[8],
        old_sigma=s_data[9],
        hid_states=s_data[10],
        masks=s_data[11],
    )


class RolloutStorage:
    """
    Experience replay buffer for PPO algorithm

    PPO算法经验回放缓冲区
    """

    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

            self.history = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        history_obs_dim=42,
        history_length=10,
        device="cpu",
    ):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=self.device
        )
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env,
                num_envs,
                *privileged_obs_shape,
                device=self.device
            )
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        ).byte()

        self.history = torch.zeros(
            num_transitions_per_env,
            num_envs,
            history_length * history_obs_dim,
            device=self.device,
        )

        self.actions_log_prob = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.mu = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.sigma = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(
                transition.critic_observations
            )
        self.actions[self.step].copy_(transition.actions)

        self.history[self.step].copy_(transition.history)

        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        """
        Save RNN hidden states

        保存RNN隐藏状态
        """
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        # 处理GRU/LSTM隐藏状态格式
        hid_a = (
            hidden_states[0]
            if isinstance(hidden_states[0], tuple)
            else (hidden_states[0],)
        )
        hid_c = (
            hidden_states[1]
            if isinstance(hidden_states[1], tuple)
            else (hidden_states[1],)
        )

        # initialize if needed
        # 初始化存储空间
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(
                    self.observations.shape[0], *hid_a[i].shape, device=self.device
                )
                for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(
                    self.observations.shape[0], *hid_c[i].shape, device=self.device
                )
                for i in range(len(hid_c))
            ]
        # copy the states
        # 拷贝隐藏状态数据
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        """
        Reset buffer pointer

        重置缓冲区指针
        """
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        """
        Calculate returns and advantages using GAE

        使用GAE方法计算回报和优势函数
        """
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        # 标准化优势函数
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def get_statistics(self):
        """
        Get trajectory statistics

        获取轨迹统计信息
        """
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (
                flat_dones.new_tensor([-1], dtype=torch.int64),
                flat_dones.nonzero(as_tuple=False)[:, 0],
            )
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """
        Generate mini-batches for training

        生成训练用的小批量数据
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        history = self.history.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                history_batch = history[batch_idx]

                yield obs_batch, critic_observations_batch, actions_batch, history_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None
