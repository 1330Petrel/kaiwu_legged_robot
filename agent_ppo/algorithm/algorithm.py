#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import torch.nn as nn
import numpy as np
import os
import time
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import RolloutStorage


class Algorithm:
    def __init__(
        self,
        model,
        optimizer,
        device=None,
        logger=None,
        monitor=None,
    ):
        self.device = device
        self.actor_critic = model
        self.optimizer = optimizer
        self.parameters = [
            p
            for param_group in self.optimizer.param_groups
            for p in param_group["params"]
        ]
        self.train_step = 0

        self.desired_kl = 0.01
        self.schedule = "adaptive"

        self.clip_param = 0.2
        self.gamma = 0.99
        self.lam = 0.95
        self.value_loss_coef = 1.0
        self.entropy_coef = 0.01
        self.vel_predict_coef = 1.0
        self.learning_rate = Config.LR
        self.max_grad_norm = 1.0
        self.use_clipped_value_loss = True
        self.num_mini_batches = Config.NUM_MINI_BATCHES
        self.num_learning_epochs = Config.NUM_LEARNING_EPOCHS
        self.min_std = torch.tensor(Config.MIN_NORMALIZED_STD, device=self.device)

        self.logger = logger
        self.monitor = monitor
        self.storage = None
        self.transition = RolloutStorage.Transition()

        self.last_report_monitor_time = 0

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            device=self.device,
        )

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, history):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        # 计算动作和值函数
        self.transition.history = history
        aug_obs, aug_critic_obs = obs.detach(), critic_obs.detach()
        self.transition.actions = self.actor_critic.act(aug_obs, history).detach()
        self.transition.values = self.actor_critic.evaluate(aug_critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        # 需要在环境执行step()前记录观测值
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def act_(self, origin_obs, origin_critic_obs, history):
        # Compute the actions and values
        # 计算动作和值函数
        obs, critic_obs = origin_obs.detach(), origin_critic_obs.detach()
        actions = self.actor_critic.act(obs, history).detach()
        values = self.actor_critic.evaluate(critic_obs).detach()
        actions_log_prob = self.actor_critic.get_actions_log_prob(actions).detach()
        action_mean = self.actor_critic.action_mean.detach()
        action_sigma = self.actor_critic.action_std.detach()
        return (
            actions,
            values,
            actions_log_prob,
            action_mean,
            action_sigma,
            obs,
            critic_obs,
        )

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        # 超时时的引导更新
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        # 记录状态转移
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        aug_last_critic_obs = last_critic_obs.detach()
        last_values = self.actor_critic.evaluate(aug_last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def learn(self, batch_data=None):
        # Code to implement model training
        # 实现模型训练的代码
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_vel_predict_loss = 0
        mean_entropy_loss = 0

        if batch_data is None:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = batch_data

        for sample in generator:
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                history_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
            ) = sample
            # self.logger.info(f"actions:{actions_batch[0]}")
            aug_obs_batch = obs_batch.detach()
            self.actor_critic.act(
                aug_obs_batch,
                history_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[0],
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            aug_critic_obs_batch = critic_obs_batch.detach()
            value_batch = self.actor_critic.evaluate(
                aug_critic_obs_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[1],
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl != None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
                    # self.logger.info(f"learning_rate:{self.learning_rate}")

            # Surrogate loss
            # 替代损失
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            # 值函数损失
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Linear vel predict loss
            # 线速度预测损失
            predicted_linear_vel = self.actor_critic.get_linear_vel(
                aug_obs_batch, history_batch
            )
            target_linear_vel = obs_batch[
                :,
                self.actor_critic.privileged_dim - 3 : self.actor_critic.privileged_dim,
            ]
            vel_predict_loss = (predicted_linear_vel - target_linear_vel).pow(2).mean()

            loss = (
                surrogate_loss
                + self.vel_predict_coef * vel_predict_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Gradient step
            # 梯度更新步骤
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if not self.actor_critic.fixed_std and self.min_std is not None:
                self.actor_critic.std.data = self.actor_critic.std.data.clamp(
                    min=self.min_std
                )

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_vel_predict_loss += self.vel_predict_coef * vel_predict_loss.item()
            mean_entropy_loss += self.entropy_coef * entropy_batch.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_vel_predict_loss /= num_updates
        self.storage.clear()

        # Periodically report monitoring
        # 按照间隔上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "policy_loss": mean_surrogate_loss,
                "value_loss": mean_value_loss,
                "entropy_loss": mean_entropy_loss,
                "total_loss": mean_surrogate_loss + mean_value_loss + mean_entropy_loss,
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})

            self.last_report_monitor_time = now

        return (
            mean_surrogate_loss,
            mean_value_loss,
            mean_vel_predict_loss,
            mean_entropy_loss,
        )
