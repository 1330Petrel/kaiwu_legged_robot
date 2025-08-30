#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from tools.base_env_process import IsaacProcessManager
from kaiwu_agent.utils.common_func import Frame, attached
import random
import os
import time
from agent_ppo.feature.definition import (
    sample_process,
)
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import RolloutStorage
from tools.train_env_conf_validate import read_usr_conf
from tools.metrics_utils import get_training_metrics
from tools.utils import calculate_terrain_stats
import torch
import statistics
from collections import deque, defaultdict
from agent_ppo.agent import Agent
import pdb
import math


@attached
def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    # Number of agents, in legged_robot_locomotion_control the value is 1
    # 智能体数量，在legged_robot_locomotion_control中值为1
    agent = agents[0]

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error(
            f"usr_conf is None, please check agent_ppo/conf/train_env_conf.toml"
        )
        raise Exception(
            "usr_conf is None, please check agent_ppo/conf/train_env_conf.toml"
        )

    game_id = (
        f'{os.getenv("KAIWU_TASK_ID", "1")}_{os.getenv("KAIWU_ROUND_INDEX", "1")}_1'
    )

    # legged_robot_locomotion_control environment
    # legged_robot_locomotion_control环境
    env = IsaacProcessManager(game_id)

    agent.algorithm.actor_critic.train()
    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(agent.num_envs, dtype=torch.float, device=agent.device)
    cur_episode_length = torch.zeros(
        agent.num_envs, dtype=torch.float, device=agent.device
    )

    # Initialize Experience Replay Buffer and Store Trajectory Data
    # 初始化经验回放池，存储轨迹数据
    storage = RolloutStorage(
        agent.num_envs,
        agent.num_steps_per_env,
        [agent.num_actor_obs],
        [agent.num_privileged_obs],
        [agent.num_actions],
        device=agent.device,
    )

    # Environment Reset and Initial Observation Acquisition
    # 环境重置与初始观测获取
    data, extra_info = env.reset(usr_conf=usr_conf)
    if extra_info.result_code != 0:
        error_message = f"reset failed, result is {extra_info.result_message}"
        logger.error(error_message)
        raise Exception(error_message)

    (obs, critic_obs) = data
    obs = torch.clone(critic_obs)
    logger.info(f"critic_obs.shape:{critic_obs.shape}")

    # Trajectory History Initialization
    # 轨迹历史初始化
    trajectory_history = torch.zeros(
        size=(
            agent.num_envs,
            agent.history_length,
            agent.num_obs - agent.privileged_dim - agent.height_dim - 3,
        ),
        device=agent.device,
    )
    obs_without_command = torch.concat(
        (
            obs[:, agent.privileged_dim : agent.privileged_dim + 6],
            obs[:, agent.privileged_dim + 9 : -agent.height_dim],
        ),
        dim=1,
    )
    trajectory_history = torch.concat(
        (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1
    )

    last_obs, last_critic_obs = torch.clone(obs), torch.clone(critic_obs)
    last_report_monitor_time = 0

    episode = 0

    # Main Training Loop
    # 主训练循环
    while True:
        # logger.info(f"Episode {episode} start, usr_conf is {usr_conf}")
        start_time = time.time()
        # Phase 1: Data Collection
        # 阶段1：数据收集
        batch_data, trajectory_history, last_obs, last_critic_obs = run_episodes_(
            env,
            agent,
            storage,
            logger,
            trajectory_history,
            last_obs,
            last_critic_obs,
            episode,
            ep_infos,
            cur_reward_sum,
            cur_episode_length,
            rewbuffer,
            lenbuffer,
        )

        episode += 1

        storage.clear()

        # Phase 2: Policy Update
        # 阶段2：策略更新
        agent.learn(batch_data)
        end_time = time.time()
        total_cost_time = round(end_time - start_time, 2)
        # logger.info(f"Episode {episode} end, cost_time is {total_cost_time} s")

        # Phase 3: Monitoring Metrics Processing
        # 阶段3：监控指标处理
        now = time.time()
        if now - last_report_monitor_time >= 60:
            monitor_data = {
                "actor_predict_succ_cnt": agent.predict_count,
                "episode_cnt": episode,
                "total_cost_time": total_cost_time,
                "sample_production_and_consumption_ratio": 1,
                "terrain_curriculum": usr_conf["terrain"]["curriculum"],
                "actor_load_last_model_succ_cnt": agent.predict_count,
                "sample_receive_cnt": episode * agent.num_steps_per_env,
            }

            if len(rewbuffer) > 0:
                monitor_data["reward"] = statistics.mean(rewbuffer)

            monitor_data["diy_1"] = torch.mean(agent.algorithm.actor_critic.std).item()

            reward_keys = [
                "rew_tracking_ang_vel",
                "rew_tracking_lin_vel",
                "rew_torques",
                "max_command_x",
                "rew_feet_air_time",
                "rew_action_rate",
                "rew_ang_vel_xy",
                "rew_collision",
                "rew_dof_acc",
                "rew_dof_pos_limits",
                "rew_lin_vel_z",
            ]

            diy_keys = [
                "rew_stumble",
                "rew_symmetric_contact",
                "rew_forward_lean",
                "rew_swing_trajectory",
            ]

            if len(ep_infos) > 0:
                terrain_data = {
                    "stability": [],
                    "success": [],
                    "velocity": [],
                    "level": [],
                    "types": [],
                }

                generic_metrics = defaultdict(list)
                diy_metrics = defaultdict(list)

                terrain_level_values = []

                for ep_info in ep_infos:
                    for key in [
                        "episode_stability",
                        "episode_success",
                        "episode_velocity",
                    ]:
                        if key in ep_info:
                            tensor = (
                                ep_info[key]
                                if isinstance(ep_info[key], torch.Tensor)
                                else torch.tensor(ep_info[key], device=agent.device)
                            )
                            terrain_data[key.split("_")[-1]].append(tensor)

                    if "terrain_level" in ep_info:
                        terrain_data["level"].append(ep_info["terrain_level"])
                        terrain_level_values.append(
                            torch.mean(ep_info["terrain_level"].float())
                        )

                    if "terrain_types" in ep_info:
                        terrain_data["types"].append(ep_info["terrain_types"])

                    for key in reward_keys:
                        if key in ep_info:
                            metric = ep_info[key]
                            if not isinstance(metric, torch.Tensor):
                                metric = torch.tensor(metric, device=agent.device)
                            processed_metric = metric.float().mean()
                        elif key == "rew_dof_pos_limits":
                            if "rew_progress" in ep_info:
                                metric = ep_info["rew_progress"]
                                if not isinstance(metric, torch.Tensor):
                                    metric = torch.tensor(metric, device=agent.device)
                                processed_metric = metric.float().mean()
                        else:
                            processed_metric = torch.tensor(
                                0.0, device=agent.device, dtype=torch.float32
                            )
                        generic_metrics[key].append(processed_metric)

                    for diy_key in diy_keys:
                        if diy_key in ep_info:
                            metric = ep_info[diy_key]
                            if not isinstance(metric, torch.Tensor):
                                metric = torch.tensor(metric, device=agent.device)
                            processed_metric = metric.float().mean()
                        else:
                            processed_metric = torch.tensor(
                                0.0, device=agent.device, dtype=torch.float32
                            )
                        diy_metrics[diy_key].append(processed_metric)

                if terrain_level_values:
                    monitor_data["terrain_level"] = torch.mean(
                        torch.stack(terrain_level_values)
                    ).item()
                else:
                    monitor_data["terrain_level"] = 0

                if all(len(v) > 0 for v in terrain_data.values()):
                    combined = {
                        k: v[-1]
                        for k, v in terrain_data.items()
                        if k in ["stability", "success", "velocity", "level", "types"]
                    }

                    terrain_stats = calculate_terrain_stats(
                        episode_stability=combined["stability"],
                        episode_success=combined["success"],
                        episode_velocity=combined["velocity"],
                        terrain_level=combined["level"],
                        terrain_types=combined["types"],
                        usr_conf=usr_conf,
                    )

                    for stat_key, stat_value in terrain_stats.items():
                        monitor_data[stat_key] = stat_value

                for metric_key, values in generic_metrics.items():
                    if values:
                        monitor_data[metric_key] = torch.stack(values).mean().item()
                    else:
                        monitor_data[metric_key] = 0.0

                for i, diy_key in enumerate(diy_keys):
                    if diy_metrics[diy_key]:
                        monitor_data[f"diy_{i+2}"] = (
                            torch.stack(diy_metrics[diy_key]).mean().item()
                        )
                    else:
                        monitor_data[f"diy_{i+2}"] = 0.0

                monitor_data["episode_reward"] = 0.0
                for key in reward_keys:
                    monitor_data["episode_reward"] += monitor_data.get(key, 0)

            monitor.put_data({os.getpid(): monitor_data})
            last_report_monitor_time = now

        # training_metrics = get_training_metrics()
        # if training_metrics:
        #     for key, value in training_metrics.items():
        #         if key == "env":
        #             for env_key, env_value in value.items():
        #                 logger.info(f"training_metrics {key} {env_key} is {env_value}")
        #         else:
        #             logger.info(f"training_metrics {key} is {value}")

        ep_infos.clear()

        # Phase 4: Model Saving
        # 阶段4：模型保存
        if episode % Config.MODEL_SAVE_INTERVAL == 0:
            agent.save_model(id=episode)

    # Close environment
    # 关闭环境
    env.close()


def run_episodes_(
    env,
    agent,
    storage,
    logger,
    trajectory_history,
    last_obs,
    last_critic_obs,
    episode,
    ep_infos,
    cur_reward_sum,
    cur_episode_length,
    rewbuffer,
    lenbuffer,
):
    def update_transition(
        history,
        actions,
        values,
        actions_log_prob,
        action_mean,
        action_sigma,
        obs,
        critic_obs,
        rewards,
        dones,
        infos,
    ):
        transition.history = history
        transition.actions = actions
        transition.values = values
        transition.actions_log_prob = actions_log_prob
        transition.action_mean = action_mean
        transition.action_sigma = action_sigma
        # need to record obs and critic_obs before env.step()
        # 在执行env.step()前记录obs和critic_obs前需要
        transition.observations = obs
        transition.critic_observations = critic_obs
        transition.rewards = rewards.clone()
        transition.dones = dones
        # Bootstrapping on time outs
        # 处理timeouts
        if "time_outs" in infos:
            transition.rewards += agent.algorithm.gamma * torch.squeeze(
                transition.values * infos["time_outs"].unsqueeze(1).to(agent.device), 1
            )

    # Trajectory data storage structure
    # 轨迹数据存储结构
    transition = RolloutStorage.Transition()

    obs, critic_obs = last_obs, last_critic_obs

    # Policy execution loop
    # 策略执行循环
    with torch.inference_mode():
        for i in range(agent.num_steps_per_env):
            history = trajectory_history.flatten(1).to(agent.device)

            # Action generation
            # 动作生成
            (
                actions,
                values,
                actions_log_prob,
                action_mean,
                action_sigma,
                detach_obs,
                detach_critic_obs,
            ) = agent.predict_local(obs, critic_obs, history)

            command_actions = torch.clip(actions, -6.0, 6.0).to(agent.device)
            # if i == 0:
            #     logger.info(f"clipped_action:{command_actions}")

            # Environment interaction
            # 环境交互
            data, extra_info = env.step(command=command_actions)
            if extra_info.result_code != 0:
                error_message = f"step failed, result is {extra_info.result_message}"
                logger.error(error_message)
                raise Exception(error_message)
            frame_no, obs, rewards, terminated, truncated, (infos, privileged_obs) = (
                data
            )
            obs = torch.clone(privileged_obs)
            if obs is None:
                logger.info(f"episode {episode}, is None happened!")
                break

            dones = torch.logical_or(terminated, truncated)

            critic_obs = privileged_obs if privileged_obs is not None else obs
            obs, critic_obs, rewards, dones = (
                obs.to(agent.device),
                critic_obs.to(agent.device),
                rewards.to(agent.device),
                dones.to(agent.device),
            )

            # process trajectory history
            # 处理历史轨迹
            update_transition(
                history,
                actions,
                values,
                actions_log_prob,
                action_mean,
                action_sigma,
                detach_obs,
                detach_critic_obs,
                rewards,
                dones,
                infos,
            )
            storage.add_transitions(transition)
            transition.clear()

            env_ids = dones.nonzero(as_tuple=False).flatten()
            trajectory_history[env_ids] = 0
            obs_without_command = torch.concat(
                (
                    obs[:, agent.privileged_dim : agent.privileged_dim + 6],
                    obs[:, agent.privileged_dim + 9 : -agent.height_dim],
                ),
                dim=1,
            )
            trajectory_history = torch.concat(
                (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1
            )

            if "episode" in infos:
                ep_infos.append(infos["episode"])
            cur_reward_sum += rewards
            cur_episode_length += 1
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

        # Advantage function computation
        # 优势函数计算
        last_critic_obs = torch.clone(critic_obs)
        aug_last_critic_obs = last_critic_obs.detach()
        last_values = agent.algorithm.actor_critic.evaluate(
            aug_last_critic_obs
        ).detach()
        storage.compute_returns(last_values, agent.algorithm.gamma, agent.algorithm.lam)
        last_obs = torch.clone(obs)

    # logger.info(
    #     f"non_zero:{torch.nonzero(storage.observations[:,0,:], as_tuple=False)}"
    # )
    # logger.info(
    #     f"obs:{storage.observations[:, 0, agent.privileged_dim+6:agent.privileged_dim+9]}"
    # )
    # logger.info(f"adv:{storage.advantages[:,0]}")
    # logger.info(f"values:{storage.values[:,0]}")

    # Generate training batches
    # 生成训练批次
    generator = storage.mini_batch_generator(
        agent.algorithm.num_mini_batches, agent.algorithm.num_learning_epochs
    )
    batch_data = []
    for mini_batch in generator:
        batch_data.append(mini_batch)
    storage.clear()
    return batch_data, trajectory_history, last_obs, last_critic_obs
