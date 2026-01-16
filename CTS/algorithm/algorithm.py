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
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import itertools

from CTS.conf.conf import Config
from CTS.feature.definition import RolloutStorage
from CTS.model.model import CTS_ActorCritic


class Algorithm:

    def __init__(
        self,
        model: CTS_ActorCritic,
        num_learning_epochs: int = Config.NUM_LEARNING_EPOCHS,
        num_mini_batches: int = Config.NUM_MINI_BATCHES,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = Config.LR,
        recon_learning_rate: float = Config.RECON_LR,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cpu",
        logger=None,
        monitor=None,
    ) -> None:
        # Device-related parameters
        self.device = device

        # PPO components
        self.model = model
        self.model.to(self.device)

        # Create optimizer
        # Optimizer for PPO (updates privileged encoder, actor, critic)
        self.ppo_components = itertools.chain(
            model.privileged_encoder.parameters(),
            model.actor.parameters(),
            model.critic.parameters(),
            [model.std],
        )
        self.ppo_optimizer = optim.Adam(
            self.ppo_components,
            lr=learning_rate,
        )
        # Optimizer for Reconstruction (updates ONLY proprioceptive encoder)
        # Using a fixed learning rate
        self.reconstruction_optimizer = optim.Adam(
            model.proprioceptive_encoder.parameters(),
            lr=recon_learning_rate,
        )

        # Create rollout storage
        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()

        # Create masks to separate teacher and student agents
        indices = torch.arange(Config.NUM_ENVS, device=self.device)
        self.teacher_mask = (indices % 4 != 3).unsqueeze(-1)

        # PPO parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.recon_learning_rate = recon_learning_rate
        self.min_std = torch.tensor(Config.MIN_NORMALIZED_STD, device=self.device)

        self.logger = logger
        self.monitor = monitor

        self.train_step = 0
        self.last_report_monitor_time = 0

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        history_length: int,
        proprioceptive_obs_shape: list,
        privileged_obs_shape: list,
        action_shape: list,
        history_obs_dim: int,
    ) -> None:
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            history_length,
            proprioceptive_obs_shape,
            privileged_obs_shape,
            action_shape,
            history_obs_dim,
            device=self.device,
        )

    def test_mode(self) -> None:
        self.model.eval()

    def train_mode(self) -> None:
        self.model.train()

    def act(
        self,
        current_proprio_obs: torch.Tensor,
        privileged_obs: torch.Tensor,
        proprio_history: torch.Tensor,
    ) -> torch.Tensor:
        if self.model.is_recurrent:
            self.transition.hidden_states = self.model.get_hidden_states()
        # Compute the actions and values
        # 计算动作和值函数
        self.transition.actions = self.model.act(
            current_proprio_obs, privileged_obs, proprio_history, self.teacher_mask
        ).detach()
        self.transition.values = self.model.evaluate(
            privileged_obs, proprio_history, self.teacher_mask
        ).detach()
        self.transition.actions_log_prob = self.model.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.model.action_mean.detach()
        self.transition.action_sigma = self.model.action_std.detach()
        # Record observations before env.step()
        # 在环境执行step()前记录观测值
        self.transition.proprioceptive_obs = current_proprio_obs
        self.transition.privileged_obs = privileged_obs
        self.transition.proprio_history = proprio_history
        return self.transition.actions

    def act_(
        self,
        current_proprio_obs: torch.Tensor,
        privileged_obs: torch.Tensor,
        proprio_history: torch.Tensor,
    ) -> tuple:
        # Compute the actions and values
        # 计算动作和值函数
        actions = self.model.act(
            current_proprio_obs, privileged_obs, proprio_history, self.teacher_mask
        ).detach()
        values = self.model.evaluate(
            privileged_obs, proprio_history, self.teacher_mask
        ).detach()
        actions_log_prob = self.model.get_actions_log_prob(actions).detach()
        action_mean = self.model.action_mean.detach()
        action_sigma = self.model.action_std.detach()
        return (
            actions,
            values,
            actions_log_prob,
            action_mean,
            action_sigma,
            current_proprio_obs,
            privileged_obs,
        )

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
    ) -> None:
        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.teacher_mask = self.teacher_mask

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
        self.model.reset(dones)

    def compute_returns(
        self,
        privileged_obs: torch.Tensor,
        proprio_history: torch.Tensor,
    ) -> None:
        last_values = self.model.evaluate(
            privileged_obs, proprio_history, self.teacher_mask
        ).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def learn(self, batch_data=None) -> None:
        # Code to implement model training
        # 实现模型训练的代码
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_recon_loss = 0
        mean_learning_rate = 0.0

        if batch_data is None:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = batch_data

        for (
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
            hid_states_batch,
            masks_batch,
        ) in generator:
            # Recompute actions log prob and entropy for current batch of transitions
            self.model.act(
                current_proprio_obs=proprioceptive_obs_batch,
                privileged_obs=privileged_obs_batch,
                proprio_history=proprio_history_batch,
                teacher_mask=teacher_mask_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[0],
            )
            actions_log_prob_batch = self.model.get_actions_log_prob(actions_batch)
            value_batch = self.model.evaluate(
                privileged_obs=privileged_obs_batch,
                proprio_history=proprio_history_batch,
                teacher_mask=teacher_mask_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[1],
            )
            mu_batch = self.model.action_mean
            sigma_batch = self.model.action_std
            entropy_batch = self.model.entropy

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
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

                    # Update the learning rate for all parameter groups
                    for param_group in self.ppo_optimizer.param_groups:
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

            ppo_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Reconstruction Loss Calculation
            student_mask_batch = torch.logical_not(teacher_mask_batch).view(-1)
            z_teacher_recon, z_student_recon = (
                self.model.get_latent_vectors_for_reconstruction(
                    privileged_obs_batch, proprio_history_batch
                )
            )
            # only care about the student agents' reconstruction quality
            recon_loss = F.mse_loss(
                z_student_recon[student_mask_batch],
                z_teacher_recon[student_mask_batch],
            )

            # Compute the gradients for PPO
            # 梯度更新步骤
            # 1. Update PPO components
            self.ppo_optimizer.zero_grad()
            ppo_loss.backward(retain_graph=True)

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(
                self.ppo_components,
                self.max_grad_norm,
            )
            self.ppo_optimizer.step()

            if self.min_std is not None:
                self.model.std.data = self.model.std.data.clamp(min=self.min_std)

            # 2. Update Reconstruction component
            self.reconstruction_optimizer.zero_grad()
            recon_loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.proprioceptive_encoder.parameters(), self.max_grad_norm
            )
            self.reconstruction_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_batch.mean().item()
            mean_recon_loss += recon_loss.item()
            mean_learning_rate += self.learning_rate

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_recon_loss /= num_updates
        mean_learning_rate /= num_updates

        # Clear the storage
        self.storage.clear()

        # Periodically report monitoring
        # 按照间隔上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "policy_loss": mean_surrogate_loss,
                "value_loss": mean_value_loss,
                "entropy_loss": mean_recon_loss,
                "total_loss": mean_learning_rate * 1000,
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})

            self.last_report_monitor_time = now

        return None
