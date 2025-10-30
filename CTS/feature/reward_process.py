#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import torch
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.utils.math import quat_apply_yaw

init = False
num_height_points = 0
height_points = None
height = None
action_history_buf = None
env_origins = None

last_contacts_1 = None
last_contacts_2 = None
feet_air_time = None
peak_swing_height = None


def _reward_a(self) -> torch.Tensor:
    global init
    if not init:
        init = True

        # 修改默认关节角度
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.7,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.7,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
        }
        self.default_dof_pos = self.default_dof_pos.squeeze(0)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = default_joint_angles[name]
            self.default_dof_pos[i] = angle
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # 初始化基座高度测量点
        _init_height_points2(self)

        # 初始化动作历史
        global action_history_buf
        action_history_buf = torch.zeros(
            self.num_envs,
            3,
            self.num_dofs,
            device=self.device,
            dtype=torch.float,
        )

    # 终止条件
    self.reset_buf = torch.any(
        torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1)
        > 1.0,
        dim=1,
    )
    self.reset_buf |= torch.logical_or(
        torch.abs(self.rpy[:, 1]) > 1.0, torch.abs(self.rpy[:, 0]) > 0.8
    )
    # 失败条件1
    fail_buf = self.reset_buf.clone()
    self.reset_buf |= self.time_out_buf

    # 失败条件2
    env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
    if len(env_ids) > 0:
        scale = self.max_episode_length_s * 0.5
        dis_threshold = self.terrain.env_length / 2
        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1
        )
        cmd_dis = torch.norm(self.commands[env_ids, :2], dim=1) * scale
        time_out = (distance <= dis_threshold) & (distance < cmd_dis)
        fail_buf[env_ids] |= time_out

    # 记录重置时时长
    env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
    if len(env_ids) > 0:
        self.extras["reset_duration"] = (
            torch.mean(self.episode_length_buf[env_ids].float()).item() * self.dt
        )

    # 指令
    self.command_ranges["lin_vel_x"][0] = max(self.command_ranges["lin_vel_x"][0], -1.0)

    return fail_buf.float()


def _reward_powers(self):
    return torch.sum(torch.abs(self.torques) * torch.abs(self.dof_vel), dim=1)


def _reward_base_height(self):
    global height
    height = _get_heights2(self)
    base_height = self.root_states[:, 2] - height
    return (torch.norm(self.commands[:, :2], p=1, dim=1) >= 0.1) * torch.square(
        base_height - self.cfg.rewards.base_height_target
    )


def _reward_action_smoothness(self):
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


def _reward_feet_regulation(self):
    global height

    feet_height = self.rigid_body_states.view(self.num_envs, -1, 13)[
        :, self.feet_indices, 2
    ]
    feet_vel_xy = self.rigid_body_states.view(self.num_envs, -1, 13)[
        :, self.feet_indices, 7:9
    ]

    p_feet_z = feet_height - height.view(-1, 1)
    p_feet_z = torch.clip(p_feet_z, min=0.0)

    v_feet_xy_sq = torch.sum(torch.square(feet_vel_xy[..., :2]), dim=-1)

    decay_factor = -0.025 * self.cfg.rewards.base_height_target
    feet_penalty = v_feet_xy_sq * torch.exp(p_feet_z / decay_factor)

    return torch.sum(feet_penalty, dim=1)


def _reward_symmetric_contact(self):
    """
    奖励对角足同步接触地面，鼓励稳定且有力的Trot步态
    """
    global last_contacts_1
    if last_contacts_1 is None:
        last_contacts_1 = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )

    # 获取足端接触状态
    contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
    contact_filt = torch.logical_or(contact, last_contacts_1)
    last_contacts_1.copy_(contact)
    # 计算每对对角腿的同步性：
    sync_diag1 = torch.where(contact_filt[:, 0] == contact_filt[:, 3], 1.0, 0.0)
    sync_diag2 = torch.where(contact_filt[:, 1] == contact_filt[:, 2], 1.0, 0.0)
    # 只在机器人运动时关心这个
    is_moving = self.base_lin_vel[:, 0] * torch.sign(self.commands[:, 0]) > 0.05
    reward = is_moving * (sync_diag1 + sync_diag2)

    # 重置
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
    if len(reset_env_ids) > 0:
        last_contacts_1[reset_env_ids] = False

    return reward


def _reward_swing_trajectory(self):
    """
    在足端落地的一瞬间，根据其在空中达到的峰值高度给予一次性奖励
    """
    global last_contacts_2, feet_air_time, peak_swing_height
    if last_contacts_2 is None:
        last_contacts_2 = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        peak_swing_height = torch.full(
            (self.num_envs, self.feet_indices.shape[0]),
            -torch.inf,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

    # 检测接触状态
    contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
    contact_filt = torch.logical_or(contact, last_contacts_2)
    last_contacts_2.copy_(contact)

    # 将足端位置转换到机体坐标系
    feet_pos = self.rigid_body_states.view(self.num_envs, -1, 13)[
        :, self.feet_indices, 0:3
    ]
    cur_footpos_translated = feet_pos - self.root_states[:, 0:3].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(
        self.num_envs, len(self.feet_indices), 3, device=self.device
    )
    for i in range(len(self.feet_indices)):
        footpos_in_body_frame[:, i, :] = quat_rotate_inverse(
            self.base_quat, cur_footpos_translated[:, i, :]
        )

    # 对于在空中的脚，持续更新其达到的最大高度
    torch.maximum(
        peak_swing_height, footpos_in_body_frame[:, :, 2], out=peak_swing_height
    )

    # 在落地瞬间，根据记录的峰值高度计算奖励
    height_error = torch.square(peak_swing_height - self.cfg.rewards.feet_height_target)
    reward_at_peak = torch.exp(-height_error / self.cfg.rewards.feet_tracking_sigma)
    first_contact = (feet_air_time > 0.0) & contact_filt
    valid_mask = (
        first_contact
        & (peak_swing_height >= self.cfg.rewards.meaningful_height_target[0])
        & (peak_swing_height < self.cfg.rewards.meaningful_height_target[1])
    )
    reward = torch.sum(valid_mask * reward_at_peak, dim=1)
    is_moving = self.base_lin_vel[:, 0] * torch.sign(self.commands[:, 0]) > 0.05
    reward *= is_moving

    # 更新每条腿的腾空时间和峰值高度
    feet_air_time += self.dt
    feet_air_time *= ~contact_filt
    peak_swing_height *= ~contact_filt

    # 重置
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
    if len(reset_env_ids) > 0:
        last_contacts_2[reset_env_ids] = False
        feet_air_time[reset_env_ids] = 0.0
        peak_swing_height[reset_env_ids] = -torch.inf

    return reward


def _reward_score(self):
    env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
    out = torch.zeros(
        self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
    )
    if len(env_ids) == 0:
        return out

    # 只对活跃 env 进行切片计算，减少无用开销
    rs = self.root_states[env_ids, :2]
    origins = self.env_origins[env_ids, :2]
    dist = torch.norm(rs - origins, dim=1)

    # 防止 episode_length 为 0
    ep_len = torch.clamp(self.episode_length_buf[env_ids], min=1.0)

    success = (dist > (self.terrain.env_length / 2)).to(torch.float)

    velocity = dist / (ep_len * self.dt)
    vel_term = ((2.0 / (1.0 + torch.exp(-velocity))) - 1.0) * 0.3

    stability = self.episode_sums["torques"][env_ids] / self.episode_length_buf[env_ids]
    stab_term = torch.clamp(torch.exp(stability * 1000.0), max=1.0) * 0.2

    out[env_ids] = success * 0.5 + vel_term + stab_term
    return out


# ---------------辅助函数---------------------


def _init_height_points2(self):
    """Get points at which the height measurments are sampled (in base frame)

    [torch.Tensor]: Tensor of shape (num_envs, num_base_height_points, 3)
    """
    global num_height_points, height_points

    y = torch.tensor(
        [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2],
        device=self.device,
        requires_grad=False,
    )
    x = torch.tensor(
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


def _get_heights2(self, env_ids=None):
    """Samples heights of the terrain at required points around each robot.
    The points are offset by the base's position and rotated by the base's yaw
    """
    global num_height_points, height_points

    if self.cfg.terrain.mesh_type == "plane":
        return self.root_states[:, 2].clone()
    elif self.cfg.terrain.mesh_type == "none":
        raise NameError("Can't measure height with terrain mesh type 'none'")

    if env_ids:
        points = quat_apply_yaw(
            self.base_quat[env_ids].repeat(1, num_height_points),
            height_points[env_ids],
        ) + (self.root_states[env_ids, :3]).unsqueeze(1)
    else:
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
