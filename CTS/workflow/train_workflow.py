#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from tools.base_env_process import IsaacProcessManager
from kaiwu_agent.utils.common_func import attached
from tools.train_env_conf_validate import read_usr_conf
from tools.utils import calculate_terrain_stats

import torch
import os
import time
import statistics
from collections import deque, defaultdict

from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import RolloutStorage
from agent_ppo.agent import Agent


@attached
def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs) -> None:
    # Number of agents, in legged_robot_locomotion_control the value is 1
    # 智能体数量，在legged_robot_locomotion_control中值为1
    agent: Agent = agents[0]

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
    logger.info(f"usr_conf: {usr_conf}")
    game_id = (
        f'{os.getenv("KAIWU_TASK_ID", "1")}_{os.getenv("KAIWU_ROUND_INDEX", "1")}_1'
    )

    # legged_robot_locomotion_control environment
    # legged_robot_locomotion_control环境
    env = IsaacProcessManager(game_id)
    agent.algorithm.model.train()

    # Book keeping
    ep_infos = []
    base_height = []
    reset_duration = []
    rewbuffer = deque([0.0], maxlen=100)
    cur_reward_sum = torch.zeros(agent.num_envs, dtype=torch.float, device=agent.device)

    # Initialize Experience Replay Buffer and Store Trajectory Data
    # 初始化经验回放池，存储轨迹数据
    storage = RolloutStorage(
        agent.num_envs,
        agent.num_steps_per_env,
        agent.history_length,
        [agent.num_proprioceptive_obs],
        [agent.num_privileged_obs],
        [agent.num_actions],
        agent.num_proprioceptive_obs - 3,
        device=agent.device,
    )

    # Initialize Trajectory History
    # 初始化轨迹历史
    trajectory_history = torch.zeros(
        size=(
            agent.num_envs,
            agent.history_length,
            agent.num_proprioceptive_obs - 3,
        ),
        device=agent.device,
    )

    # Create masks to separate teacher and student agents
    indices = torch.arange(agent.num_envs, device=agent.device)
    teacher_mask0 = indices % 4 != 3
    student_mask0 = ~teacher_mask0
    teacher_mask = teacher_mask0.unsqueeze(-1)

    # Environment Reset and Initial Observation Acquisition
    # 环境重置与初始观测获取
    data, extra_info = env.reset(usr_conf=usr_conf)
    if extra_info.result_code != 0:
        error_message = f"reset failed, result is {extra_info.result_message}"
        logger.error(error_message)
        raise Exception(error_message)

    (obs, privileged_obs) = data
    proprioceptive_obs = obs[:, 3:]
    obs_without_command = torch.concat(
        (
            proprioceptive_obs[:, :6],
            proprioceptive_obs[:, 9:],
        ),
        dim=1,
    )
    trajectory_history = torch.concat(
        (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1
    )
    last_proprio_obs = torch.clone(proprioceptive_obs)
    last_privileged_obs = torch.clone(privileged_obs)

    # Logging and Monitoring
    # 日志与监控
    last_report_monitor_time = 0
    episode = 0
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
        "rew_a",
        # "rew_action_smoothness",
        # "rew_stand_still",
        "rew_symmetric_contact",
        "rew_feet_regulation",
    ]

    # Main Training Loop
    # 主训练循环
    while True:
        # Phase 1: Data Collection
        # 阶段1：数据收集
        batch_data, trajectory_history, last_proprio_obs, last_privileged_obs = (
            run_episodes_(
                env,
                agent,
                storage,
                logger,
                trajectory_history,
                last_proprio_obs,
                last_privileged_obs,
                teacher_mask,
                episode,
                ep_infos,
                base_height,
                reset_duration,
                cur_reward_sum,
                rewbuffer,
            )
        )
        episode += 1

        # Phase 2: Policy Update
        # 阶段2：策略更新
        agent.learn(batch_data)

        # Phase 3: Monitoring Metrics Processing
        # 阶段3：监控指标处理
        now = time.time()
        if now - last_report_monitor_time >= 60:
            monitor_data = {
                "diy_1": torch.mean(agent.algorithm.model.std).item(),
                "reward": statistics.mean(rewbuffer),
            }

            if len(reset_duration) > 0:
                monitor_data["total_cost_time"] = int(statistics.mean(reset_duration))
                reset_duration.clear()
            if len(base_height) > 0:
                monitor_data["diy_5"] = statistics.mean(base_height)
                base_height.clear()

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

                reset_distance = []
                success_values_teachers = []
                success_values_students = []
                terrain_level_values_teachers = []
                terrain_level_values_students = []

                for ep_info in ep_infos:
                    for key in [
                        "episode_stability",
                        "episode_success",
                        "episode_velocity",
                    ]:
                        if key in ep_info:
                            tensor = ep_info[key]
                            terrain_data[key.split("_")[-1]].append(tensor)
                    success_values_teachers.append(
                        torch.mean(
                            ep_info["episode_success"][teacher_mask0].float()
                        ).item()
                    )
                    success_values_students.append(
                        torch.mean(
                            ep_info["episode_success"][student_mask0].float()
                        ).item()
                    )

                    if "terrain_level" in ep_info:
                        terrain_data["level"].append(ep_info["terrain_level"])
                        terrain_level_values_teachers.append(
                            torch.mean(
                                ep_info["terrain_level"][teacher_mask0].float()
                            ).item()
                        )
                        terrain_level_values_students.append(
                            torch.mean(
                                ep_info["terrain_level"][student_mask0].float()
                            ).item()
                        )

                    if "terrain_types" in ep_info:
                        terrain_data["types"].append(ep_info["terrain_types"])

                    if "reset_distance" in ep_info:
                        reset_distance.append(
                            torch.mean(ep_info["reset_distance"].float()).item()
                        )

                    for key in reward_keys:
                        if key in ep_info:
                            metric = ep_info[key]
                            if not isinstance(metric, torch.Tensor):
                                metric = torch.tensor(metric, device=agent.device)
                            processed_metric = metric.float().mean()
                        elif key == "rew_feet_air_time":
                            if "rew_dof_error_named" in ep_info:
                                metric = ep_info["rew_dof_error_named"]
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

                if reset_distance:
                    monitor_data["sample_production_and_consumption_ratio"] = (
                        statistics.mean(reset_distance)
                    )
                    monitor_data["episode_cnt"] = int(
                        statistics.mean(success_values_teachers) * 500
                    )
                    monitor_data["sample_receive_cnt"] = int(
                        statistics.mean(success_values_students) * 500
                    )
                    monitor_data["terrain_curriculum"] = statistics.mean(
                        terrain_level_values_teachers
                    )
                    monitor_data["terrain_level"] = statistics.mean(
                        terrain_level_values_students
                    )
                else:
                    monitor_data["sample_production_and_consumption_ratio"] = 0.0
                    monitor_data["episode_cnt"] = 0.0
                    monitor_data["sample_receive_cnt"] = 0.0
                    monitor_data["terrain_curriculum"] = 0.0
                    monitor_data["terrain_level"] = 0.0

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

                monitor_data["episode_reward"] = 0.0
                for metric_key, values in generic_metrics.items():
                    if values:
                        monitor_data[metric_key] = torch.stack(values).mean().item()
                    else:
                        monitor_data[metric_key] = 0.0
                    monitor_data["episode_reward"] += monitor_data[metric_key]

                for i, diy_key in enumerate(diy_keys):
                    if diy_metrics[diy_key]:
                        monitor_data[f"diy_{i+2}"] = (
                            torch.stack(diy_metrics[diy_key]).mean().item()
                        )
                    else:
                        monitor_data[f"diy_{i+2}"] = 0.0
                    monitor_data["episode_reward"] += monitor_data[f"diy_{i+2}"]

            monitor.put_data({os.getpid(): monitor_data})
            last_report_monitor_time = now

        ep_infos.clear()

        # Phase 4: Model Saving
        # 阶段4：模型保存
        if episode % Config.MODEL_SAVE_INTERVAL == 0:
            agent.save_model(id=str(episode))

    # Close environment
    # 关闭环境
    env.close()


def run_episodes_(
    env,
    agent: Agent,
    storage: RolloutStorage,
    logger,
    trajectory_history: torch.Tensor,
    last_proprio_obs: torch.Tensor,
    last_privileged_obs: torch.Tensor,
    teacher_mask: torch.Tensor,
    episode: int,
    ep_infos: list,
    base_height: list,
    reset_duration: list,
    cur_reward_sum: torch.Tensor,
    rewbuffer: deque,
) -> tuple:

    def update_transition(
        actions: torch.Tensor,
        values: torch.Tensor,
        actions_log_prob: torch.Tensor,
        action_mean: torch.Tensor,
        action_sigma: torch.Tensor,
        proprioceptive_obs: torch.Tensor,
        privileged_obs: torch.Tensor,
        proprio_history: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        teacher_mask: torch.Tensor,
        infos: dict,
    ) -> None:
        transition.actions = actions
        transition.values = values
        transition.actions_log_prob = actions_log_prob
        transition.action_mean = action_mean
        transition.action_sigma = action_sigma
        transition.proprioceptive_obs = proprioceptive_obs
        transition.privileged_obs = privileged_obs
        transition.proprio_history = proprio_history
        transition.rewards = rewards.clone()
        transition.dones = dones
        transition.teacher_mask = teacher_mask

        # Bootstrapping on time outs
        # 处理timeouts
        if "time_outs" in infos:
            transition.rewards += agent.algorithm.gamma * torch.squeeze(
                transition.values * infos["time_outs"].unsqueeze(1).to(agent.device), 1
            )

    # Rollout storage for trajectory data
    transition = RolloutStorage.Transition()
    # Initial observations
    current_proprio_obs, privileged_obs = last_proprio_obs, last_privileged_obs

    # Policy execution loop
    # 策略执行循环
    with torch.inference_mode():
        for _ in range(agent.num_steps_per_env):
            # Generate proprioceptive history
            proprio_history = torch.concat(
                (trajectory_history.flatten(1), current_proprio_obs[:, 6:9]), dim=1
            )

            # Action generation
            # 动作生成
            (
                actions,
                values,
                actions_log_prob,
                action_mean,
                action_sigma,
                proprioceptive_obs_stg,
                privileged_obs_stg,
            ) = agent.predict_local(
                current_proprio_obs, privileged_obs, proprio_history
            )
            actions.clamp_(agent.clip_actions[0], agent.clip_actions[1])

            # Environment interaction
            # 环境交互
            data, extra_info = env.step(command=actions.to(agent.device))
            if extra_info.result_code != 0:
                error_message = f"step failed, result is {extra_info.result_message}"
                logger.error(error_message)
                raise Exception(error_message)
            (
                _,
                obs,
                rewards,
                terminated,
                truncated,
                (infos, privileged_obs),
            ) = data
            if privileged_obs is None:
                logger.warning(f"episode {episode} is Not happened!")
                break

            # Move to device
            dones = torch.logical_or(terminated, truncated)
            current_proprio_obs, privileged_obs, rewards, dones = (
                obs[:, 3:].to(agent.device),
                privileged_obs.to(agent.device),
                rewards.to(agent.device),
                dones.to(agent.device),
            )

            # Process the step data
            update_transition(
                actions,
                values,
                actions_log_prob,
                action_mean,
                action_sigma,
                proprioceptive_obs_stg,
                privileged_obs_stg,
                proprio_history,
                rewards,
                dones,
                teacher_mask,
                infos,
            )
            storage.add_transitions(transition)
            transition.clear()

            # Update trajectory history
            # 更新轨迹历史
            env_ids = dones.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                trajectory_history[env_ids] = 0.0
            obs_without_command = torch.concat(
                (
                    current_proprio_obs[:, :6],
                    current_proprio_obs[:, 9:],
                ),
                dim=1,
            )
            trajectory_history[:, :-1] = trajectory_history[:, 1:].clone()
            trajectory_history[:, -1] = obs_without_command

            # Book keeping
            if "episode" in infos:
                ep_infos.append(infos["episode"])
            if "base_height" in infos:
                base_height.append(infos["base_height"])
            if "reset_duration" in infos:
                reset_duration.append(infos["reset_duration"])
            cur_reward_sum += rewards
            # Clear data for completed episodes
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0

        # Advantage function computation
        # 优势函数计算
        last_values = agent.algorithm.model.evaluate(
            privileged_obs,
            torch.concat(
                (trajectory_history.flatten(1), current_proprio_obs[:, 6:9]), dim=1
            ),
            teacher_mask,
        ).detach()
        storage.compute_returns(last_values, agent.algorithm.gamma, agent.algorithm.lam)
        last_privileged_obs.copy_(privileged_obs)
        last_proprio_obs.copy_(current_proprio_obs)

    # Generate training batches
    # 生成训练批次
    generator = storage.mini_batch_generator(
        agent.algorithm.num_mini_batches, agent.algorithm.num_learning_epochs
    )
    batch_data = []
    for mini_batch in generator:
        batch_data.append(mini_batch)

    # Clear storage
    storage.clear()

    return batch_data, trajectory_history, last_proprio_obs, last_privileged_obs
