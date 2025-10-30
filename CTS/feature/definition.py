#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import create_cls, attached

import torch
import numpy as np
from collections.abc import Generator

SampleData = create_cls(
    "SampleData",
    proprioceptive_obs=None,
    privileged_obs=None,
    proprio_history=None,
    actions=None,
    target_values=None,
    advantages=None,
    returns=None,
    old_action_log_prob=None,
    old_mu=None,
    old_sigma=None,
    teacher_mask=None,
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
        proprioceptive_obs=s_data[0],
        privileged_obs=s_data[1],
        proprio_history=s_data[2],
        actions=s_data[3],
        target_values=s_data[4],
        advantages=s_data[5],
        returns=s_data[6],
        old_action_log_prob=s_data[7],
        old_mu=s_data[8],
        old_sigma=s_data[9],
        teacher_mask=s_data[10],
        hid_states=s_data[11],
        masks=s_data[12],
    )


class RolloutStorage:
    """
    Experience replay buffer for PPO algorithm

    PPO算法经验回放缓冲区
    """

    class Transition:
        def __init__(self) -> None:
            self.proprioceptive_obs: torch.Tensor | None = None
            self.privileged_obs: torch.Tensor | None = None
            self.proprio_history: torch.Tensor | None = None
            self.actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.dones: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.actions_log_prob: torch.Tensor | None = None
            self.action_mean: torch.Tensor | None = None
            self.action_sigma: torch.Tensor | None = None
            self.teacher_mask: torch.Tensor | None = None
            self.hidden_states: tuple = (
                None,
                None,
            )

        def clear(self) -> None:
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        history_length: int,
        proprioceptive_obs_shape: list,
        privileged_obs_shape: list,
        actions_shape: list,
        history_obs_dim: int = 42,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # Core
        self.proprioceptive_obs = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *proprioceptive_obs_shape,
            device=self.device
        )
        self.privileged_obs = torch.zeros(
            num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
        )
        self.proprio_history = torch.zeros(
            num_transitions_per_env,
            num_envs,
            history_length * history_obs_dim,
            device=self.device,
        )

        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        ).byte()

        # For reinforcement learning
        self.values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.mu = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.sigma = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )

        # Masks to separate teacher and student agents
        self.teacher_mask = torch.zeros(
            num_transitions_per_env, num_envs, 1, dtype=torch.bool, device=self.device
        )

        # For RNN networks
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        # Counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition) -> None:
        # Check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow")

        # Core
        self.proprioceptive_obs[self.step].copy_(transition.proprioceptive_obs)
        self.privileged_obs[self.step].copy_(transition.privileged_obs)
        self.proprio_history[self.step].copy_(transition.proprio_history)

        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # For reinforcement learning
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        # Masks to separate teacher and student agents
        self.teacher_mask[self.step].copy_(transition.teacher_mask)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # Increment the counter
        self.step += 1

    def _save_hidden_states(self, hidden_states: tuple) -> None:
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
                    self.proprioceptive_obs.shape[0],
                    *hid_a[i].shape,
                    device=self.device
                )
                for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(
                    self.proprioceptive_obs.shape[0],
                    *hid_c[i].shape,
                    device=self.device
                )
                for i in range(len(hid_c))
            ]
        # copy the states
        # 拷贝隐藏状态数据
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self) -> None:
        self.step = 0

    def compute_returns(
        self, last_values: torch.Tensor, gamma: float, lam: float
    ) -> None:
        """
        Calculate returns and advantages using GAE

        使用GAE方法计算回报和优势函数
        """
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = (
                last_values
                if step == self.num_transitions_per_env - 1
                else self.values[step + 1]
            )
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        # 标准化优势函数
        self.advantages = self.returns - self.values
        # Normalize the advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def get_statistics(self) -> tuple:
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

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int) -> Generator:
        """
        Generate mini-batches for training

        生成训练用的小批量数据
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        # Core
        proprioceptive_obs = self.proprioceptive_obs.flatten(0, 1)
        privileged_obs = self.privileged_obs.flatten(0, 1)
        proprio_history = self.proprio_history.flatten(0, 1)

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # For mask
        teacher_mask = self.teacher_mask.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                proprioceptive_obs_batch = proprioceptive_obs[batch_idx]
                privileged_obs_batch = privileged_obs[batch_idx]
                proprio_history_batch = proprio_history[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                teacher_mask_batch = teacher_mask[batch_idx]

                # Yield the mini-batch
                yield (
                    proprioceptive_obs_batch,
                    privileged_obs_batch,
                    proprio_history_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    teacher_mask_batch,
                    (
                        None,
                        None,
                    ),
                    None,
                )
