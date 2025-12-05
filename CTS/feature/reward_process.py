#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import torch
import numpy as np
from isaacgym import gymapi, gymtorch
from legged_gym.utils.math import quat_apply_yaw
from isaacgym.torch_utils import torch_rand_float

init = False
num_height_points = 0
height_points = None
height = None
slope_mask = None
slope_yaw_cmd = None
disturbance = None
disturbance_interval = 0
action_history_buf = None
last_contacts_1 = None
last_contacts_2 = None
left_indices = None
right_indices = None


def _reward_a(self) -> torch.Tensor:
    global init, disturbance_interval, slope_mask, slope_yaw_cmd
    if not init:
        init = True

        # 随机质心位置
        com_pos = torch.zeros(self.num_envs, 2, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_com_pos:
            rng = self.cfg.domain_rand.com_pos_range_yz
            for i in range(self.num_envs):
                env_handle = self.envs[i]
                actor_handle = self.actor_handles[i]
                body_props = self.gym.get_actor_rigid_body_properties(
                    env_handle, actor_handle
                )
                com_pos_y = np.random.uniform(rng[0], rng[1])
                com_pos_z = np.random.uniform(rng[0], rng[1])
                com_pos[i, 0] = com_pos_y
                com_pos[i, 1] = com_pos_z
                body_props[0].com += gymapi.Vec3(0.0, com_pos_y, com_pos_z)
                self.gym.set_actor_rigid_body_properties(
                    env_handle, actor_handle, body_props, recomputeInertia=True
                )

        # 归一化参数
        self.randomized_com_pos *= 2.0 / 3.0
        self.randomized_com_pos = torch.cat((self.randomized_com_pos, com_pos), dim=-1)
        scale, shift = _get_scale_shift(self.cfg.domain_rand.friction_range)
        self.randomized_frictions = (
            self.randomized_frictions[:, 0].unsqueeze(1) - shift
        ) * scale
        scale, shift = _get_scale_shift(self.cfg.domain_rand.restitution_range)
        self.randomized_restitutions = (
            self.randomized_restitutions[:, 0].unsqueeze(1) - shift
        ) * scale
        scale, shift = _get_scale_shift(self.cfg.domain_rand.added_mass_range)
        self.randomized_added_masses = (
            self.randomized_added_masses[:, 0].unsqueeze(1) - shift
        ) * scale

        # 延迟范围
        self.latency_range = [0, 4]

        # 斜坡转向指令
        slope_mask = (self.terrain_types == 4) | (self.terrain_types == 5)
        slope_yaw_cmd = torch.full(
            (self.num_envs,),
            4.0,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # 扰动间隔
        global disturbance
        disturbance = torch.zeros(
            self.num_envs,
            self.num_bodies,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        disturbance_interval = np.ceil(
            self.cfg.domain_rand.disturbance_interval_s / self.dt
        )

        # 基座高度测量点
        _init_height_points2(self)

        # 扭矩和加速度缓存
        holder = torch.zeros(
            self.num_envs,
            self.num_dof * 2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.randomized_com_pos = torch.cat((holder, self.randomized_com_pos), dim=-1)

        # 动作历史
        global action_history_buf
        action_history_buf = torch.zeros(
            self.num_envs,
            3,
            self.num_dofs,
            device=self.device,
            dtype=torch.float,
        )

        # 足端接触状态记录
        global last_contacts_1, last_contacts_2
        last_contacts_1 = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        last_contacts_2 = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )

        # 参与默认关节默认位置惩罚的关节索引
        global left_indices, right_indices
        dof_error_named_indices = torch.tensor(
            [self.dof_names.index(name) for name in self.cfg.rewards.dof_error_names],
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        left_indices = dof_error_named_indices[[0, 2]]
        right_indices = dof_error_named_indices[[1, 3]]

    # 斜坡地形转向指令覆盖
    zero_cmd = torch.norm(self.commands[:, :2], p=1, dim=1) < 0.1
    override_mask = slope_mask & zero_cmd
    if override_mask.any():
        need_sample = override_mask & (slope_yaw_cmd > 2.0)
        sample_ids = need_sample.nonzero(as_tuple=False).flatten()

        if len(sample_ids) > 0:
            slope_yaw_cmd[sample_ids] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(sample_ids), 1),
                device=self.device,
            ).squeeze(1)
            slope_yaw_cmd[sample_ids] *= torch.abs(slope_yaw_cmd[sample_ids]) > 0.2
        self.commands[override_mask, 2] = slope_yaw_cmd[override_mask]

    exit_mask = slope_mask & (~zero_cmd)
    slope_yaw_cmd[exit_mask] = 4.0

    # 外部扰动
    if self.cfg.domain_rand.disturbance and (
        self.common_step_counter % disturbance_interval == 0
    ):
        _disturbance_robots(self)

    # 终止条件
    self.reset_buf = torch.any(
        torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1)
        > 1.0,
        dim=1,
    )
    self.reset_buf |= torch.logical_or(
        torch.abs(self.rpy[:, 1]) > 1.0, torch.abs(self.rpy[:, 0]) > 0.8
    )
    # 失败条件
    fail_buf = self.reset_buf.clone()
    self.reset_buf |= self.time_out_buf

    # 扭矩和加速度
    self.randomized_com_pos[:, : self.num_dof] = (
        0.05 * self.torques / self.torque_limits
    )
    self.randomized_com_pos[:, self.num_dof : self.num_dof * 2] = (
        0.05 * self.obs_scales.dof_vel * self.last_dof_vel
    )

    # 指令
    self.command_ranges["lin_vel_x"][0] = -1.0

    # 重置
    env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
    if len(env_ids) > 0:
        slope_reset_ids = env_ids[slope_mask[env_ids]]
        if len(slope_reset_ids) > 0:
            slope_yaw_cmd[slope_reset_ids] = 4.0
            self.commands[slope_reset_ids, 2] = 0.0

        self.extras["reset_duration"] = torch.mean(
            self.episode_length_buf[env_ids].float()
        ).item()

    return fail_buf.float()


def _reward_powers(self) -> torch.Tensor:
    return torch.sum(torch.abs(self.torques) * torch.abs(self.dof_vel), dim=1)


def _reward_base_height(self) -> torch.Tensor:
    global height
    height = _get_heights2(self)
    base_height = self.root_states[:, 2] - height
    self.extras["base_height"] = torch.mean(base_height).item()
    return torch.square(base_height - self.cfg.rewards.base_height_target)


def _reward_action_smoothness(self) -> torch.Tensor:
    global action_history_buf
    # 更新
    action_history_buf[:, 1:, :] = action_history_buf[:, :-1, :].clone()
    action_history_buf[:, 0, :] = self.actions

    reward = torch.sum(
        torch.square(
            action_history_buf[:, 0, :]
            - 2 * action_history_buf[:, 1, :]
            + action_history_buf[:, 2, :]
        ),
        dim=1,
    )

    # 重置
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
    if len(reset_env_ids) > 0:
        action_history_buf[reset_env_ids, :, :] = 0.0

    return reward


def _reward_stand_still(self) -> torch.Tensor:
    return (torch.norm(self.commands[:, :2], p=1, dim=1) < 0.1) * (
        torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
    )


def _reward_feet_regulation(self) -> torch.Tensor:
    global last_contacts_1

    # 获取足端接触状态
    contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
    contact_filt = torch.logical_or(contact, last_contacts_1)
    last_contacts_1.copy_(contact)
    # 获取足端速度
    feet_vel_xy = self.rigid_body_lin_vel[:, self.feet_indices, :2]
    v_feet_xy_sq = torch.sum(torch.square(feet_vel_xy).clamp_max_(2.0), dim=-1)
    # 仅对接触地面的足端计算惩罚
    feet_penalty = v_feet_xy_sq * contact_filt.float()

    # 重置
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
    if len(reset_env_ids) > 0:
        last_contacts_1[reset_env_ids] = False

    return torch.sum(feet_penalty, dim=1)


def _reward_symmetric_contact(self) -> torch.Tensor:
    global last_contacts_2

    # 获取足端接触状态
    contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
    contact_filt = torch.logical_or(contact, last_contacts_2)
    last_contacts_2.copy_(contact)
    # 计算对角腿的同步性
    sync_diag1 = torch.where(contact_filt[:, 0] == contact_filt[:, 3], 1.0, 0.0)
    sync_diag2 = torch.where(contact_filt[:, 1] == contact_filt[:, 2], 1.0, 0.0)
    # 只在机器人x方向运动时关心
    is_moving = (self.base_lin_vel[:, 0] * torch.sign(self.commands[:, 0]) > 0.05) | (
        self.base_lin_vel[:, 1] * torch.sign(self.commands[:, 1]) > 0.05
    )
    reward = is_moving * (sync_diag1 + sync_diag2)

    # 重置
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
    if len(reset_env_ids) > 0:
        last_contacts_2[reset_env_ids] = False

    return reward


def _reward_dof_error_named(self) -> torch.Tensor:
    global left_indices, right_indices
    # 左腿内收为负, 右腿内收为正
    left_diff = torch.clamp(self.dof_pos[:, left_indices] + 0.05, min=-1.0, max=0.0)
    right_diff = torch.clamp(self.dof_pos[:, right_indices] - 0.05, min=0.0, max=1.0)
    dof_error = torch.sum(torch.square(left_diff), dim=1) + torch.sum(
        torch.square(right_diff), dim=1
    )
    dof_error.clamp_max_(1.0)
    # 仅在机器人x方向运动时关心
    is_moving = (
        (self.base_lin_vel[:, 0] * torch.sign(self.commands[:, 0]) > 0.05)
        & (torch.abs(self.base_lin_vel[:, 1]) < 0.1)
        & (torch.abs(self.base_ang_vel[:, 2]) < 0.1)
    )
    return is_moving * dof_error


# ---------------辅助函数---------------------


def _get_scale_shift(range: list | tuple) -> tuple:
    scale = 2.0 / (range[1] - range[0])
    shift = (range[1] + range[0]) / 2.0
    return scale, shift


def _init_height_points2(self) -> None:
    """Get points at which the height measurments are sampled (in base frame)

    [torch.Tensor]: Tensor of shape (num_envs, num_base_height_points, 3)
    """
    global num_height_points, height_points

    x = torch.tensor(
        [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2],
        device=self.device,
        requires_grad=False,
    )
    y = torch.tensor(
        [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15],
        device=self.device,
        requires_grad=False,
    )
    grid_x, grid_y = torch.meshgrid(x, y)

    num_height_points = grid_x.numel()
    height_points = torch.zeros(
        self.num_envs,
        num_height_points,
        3,
        device=self.device,
        requires_grad=False,
    )
    height_points[:, :, 0] = grid_x.flatten()
    height_points[:, :, 1] = grid_y.flatten()


def _get_heights2(self) -> torch.Tensor:
    """Samples heights of the terrain at required points around each robot.
    The points are offset by the base's position and rotated by the base's yaw
    """
    global num_height_points, height_points

    if self.cfg.terrain.mesh_type == "plane":
        return self.root_states[:, 2].clone()
    elif self.cfg.terrain.mesh_type == "none":
        raise NameError("Can't measure height with terrain mesh type 'none'")

    points = quat_apply_yaw(
        self.base_quat.repeat(1, num_height_points),
        height_points,
    ) + (self.root_states[:, :3]).unsqueeze(1)

    points += self.terrain.cfg.border_size
    points = (points / self.terrain.cfg.horizontal_scale).long()
    px = points[:, :, 0].view(-1)
    py = points[:, :, 1].view(-1)
    px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
    py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

    heights1 = self.height_samples[px, py]
    heights2 = self.height_samples[px + 1, py]
    heights3 = self.height_samples[px, py + 1]
    heights = torch.min(heights1, heights2)
    heights = torch.min(heights, heights3)

    base_height = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    return torch.mean(base_height, dim=1)


def _disturbance_robots(self):
    """Random add disturbance force to the robots"""
    global disturbance
    max_force = self.cfg.domain_rand.max_disturbance_force
    disturbance[:, 0, :] = torch_rand_float(
        -max_force, max_force, (self.num_envs, 3), device=self.device
    )
    self.gym.apply_rigid_body_force_tensors(
        self.sim,
        forceTensor=gymtorch.unwrap_tensor(disturbance),
        space=gymapi.CoordinateSpace.LOCAL_SPACE,
    )
