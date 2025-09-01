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
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import RolloutStorage
from agent_diy.model.model import TeacherActorCritic


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
        self.actor_critic: TeacherActorCritic = model
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
        self.triplet_loss_coef = 1e-3
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
        obs_shape,
        privileged_obs_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            obs_shape,
            privileged_obs_shape,
            action_shape,
            device=self.device,
        )

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, history):
        # Compute the actions and values
        # 计算动作和值函数
        self.transition.history = history
        self.transition.actions = self.actor_critic.act(obs, history).detach()
        self.transition.values = self.actor_critic.evaluate(
            critic_obs, history
        ).detach()
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

    def act_(self, obs, critic_obs, history):
        # Compute the actions and values
        # 计算动作和值函数
        actions = self.actor_critic.act(obs, history).detach()
        values = self.actor_critic.evaluate(critic_obs, history).detach()
        actions_log_prob = self.actor_critic.get_actions_log_prob(actions).detach()
        action_mean = self.actor_critic.action_mean.detach()
        action_sigma = self.actor_critic.action_std.detach()
        return (
            actions,
            values,
            actions_log_prob,
            action_mean,
            action_sigma,
        )

    def process_env_step(self, next_obs, next_history, rewards, dones, infos):
        self.transition.next_observations = next_obs
        self.transition.next_history = next_history
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

    def compute_returns(self, last_critic_obs, history):
        last_values = self.actor_critic.evaluate(last_critic_obs, history).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def learn(self, batch_data=None):
        # Code to implement model training
        # 实现模型训练的代码
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_triplet_loss = 0
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
                next_obs_batch,
                critic_obs_batch,
                actions_batch,
                history_batch,
                next_history_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
            ) = sample

            # --- 重新计算当前策略下的动作概率和价值 ---
            # 传入旧的观测，用当前的(已更新的)模型进行前向传播
            self.actor_critic.act(
                obs_batch,
                history_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[0],
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch,
                history_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[1],
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # --- KL散度与自适应学习率 ---
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

            # Entropy loss
            # 熵损失
            entropy_loss = -self.entropy_coef * entropy_batch.mean()

            # --------- 三元组损失 (Triplet Loss) ---------
            # 1. 编码当前状态的隐向量 z_t
            _, z = self.actor_critic.encode(obs_batch, history_batch)
            # 2. 使用转移模型 (trans) 预测下一个隐向量 z_t+1_pred (锚点)
            pred_next_z = self.actor_critic.trans(torch.cat([z, actions_batch], dim=1))
            # 3. 编码真实的下一个状态，得到目标隐向量 z_t+1_target (正样本)
            _, targ_next_z = self.actor_critic.encode(
                next_obs_batch, next_history_batch
            )
            # 4. 在批次内随机打乱，构造负样本 z_t+1_neg
            batch_size = z.size(0)
            perm = np.random.permutation(batch_size)
            # 负样本是来自同一批次中其他轨迹的下一个状态的隐向量
            next_neg_z = targ_next_z[perm].detach()
            # 5. 正样本损失：预测值与真实目标值的L2距离，希望这个距离越小越好
            pos_diff = targ_next_z - pred_next_z
            pos_loss = (pos_diff.pow(2)).sum(1).mean()
            # 6. 负样本损失：使用Hinge Loss 希望真实目标值与负样本的距离越大越好
            neg_diff = targ_next_z - next_neg_z
            neg_loss_raw = (neg_diff.pow(2)).sum(1)

            zeros = torch.zeros_like(pos_loss)
            # 目标是让负样本距离 > 1.0，如果小于1.0，则产生损失
            neg_loss = torch.max(zeros, 1.0 - neg_loss_raw).mean()
            # 7. 最终三元组损失 = 拉近正样本 + 推远负样本
            triplet_loss = pos_loss + neg_loss

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                + entropy_loss
                + self.triplet_loss_coef * triplet_loss
            )

            # Gradient step
            # 梯度更新步骤
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_loss.item()
            mean_triplet_loss += triplet_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_triplet_loss /= num_updates
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
            mean_triplet_loss,
            mean_entropy_loss,
        )
