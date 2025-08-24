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

num_base_height_points = 0
base_height_points = None


def _reward_base_height(self):
    """
    覆盖legged_robot中的基座高度惩罚项的错误实现
    """
    # Penalize base height away from target
    base_height = _get_base_heights(self)
    return torch.square(base_height - self.cfg.rewards.base_height_target)


def _reward_powers(self):
    # Penalize torques
    return torch.sum(torch.abs(self.torques) * torch.abs(self.dof_vel), dim=1)


def _reward_foot_clearance(self):
    """
    摆动腿惩罚
    """
    # 将足端位置和速度转换到机体坐标系
    rigid_body_states = self.rigid_body_states.view(self.num_envs, -1, 13)
    feet_pos = rigid_body_states[:, self.feet_indices, 0:3]
    feet_vel = rigid_body_states[:, self.feet_indices, 7:10]
    cur_footpos_translated = feet_pos - self.root_states[:, 0:3].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(
        self.num_envs, len(self.feet_indices), 3, device=self.device
    )
    cur_footvel_translated = feet_vel - self.root_states[:, 7:10].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(
        self.num_envs, len(self.feet_indices), 3, device=self.device
    )
    for i in range(len(self.feet_indices)):
        footpos_in_body_frame[:, i, :] = quat_rotate_inverse(
            self.base_quat, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = quat_rotate_inverse(
            self.base_quat, cur_footvel_translated[:, i, :]
        )

    # 计算足端高度误差和侧向速度
    height_error = torch.square(
        footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target
    ).view(self.num_envs, -1)
    foot_leteral_vel = torch.sqrt(
        torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)
    ).view(self.num_envs, -1)

    # 只有当足端侧向速度较大时 (即在摆动时)，才计算这个奖励
    clearance_reward = height_error * foot_leteral_vel

    return torch.sum(clearance_reward, dim=1)


def _reward_progress(self):
    """
    直接奖励机器人在X轴（前进方向）上的速度
    """
    # self.root_states[:, 7:10] 是世界坐标系下的线速度
    # 我们只关心X方向的速度
    forward_vel = self.root_states[:, 7]

    # 只有当指令是要求前进时，才给予这个奖励，避免在其他指令下产生冲突
    # self.commands[:, 0] 是X方向的速度指令
    progress_reward = forward_vel * (self.commands[:, 0] > 0)

    # 将负的前进速度裁剪掉，我们不希望因为后退而惩罚它
    return torch.clip(progress_reward, min=0.0)


# ---------------辅助函数---------------------


def _init_base_height_points(self):
    """Returns points at which the height measurments are sampled (in base frame)

    Returns:
        [torch.Tensor]: Tensor of shape (num_envs, num_base_height_points, 3)
    """
    global num_base_height_points, base_height_points

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

    num_base_height_points = grid_x.numel()
    base_height_points = torch.zeros(
        self.num_envs,
        num_base_height_points,
        3,
        device=self.device,
        requires_grad=False,
    )
    base_height_points[:, :, 0] = grid_x.flatten()
    base_height_points[:, :, 1] = grid_y.flatten()


def _get_base_heights(self, env_ids=None):
    """Samples heights of the terrain at required points around each robot.
        The points are offset by the base's position and rotated by the base's yaw

    Args:
        env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

    Raises:
        NameError: [description]

    Returns:
        [type]: [description]
    """
    global num_base_height_points, base_height_points
    if num_base_height_points == 0 or base_height_points is None:
        _init_base_height_points(self)

    if self.cfg.terrain.mesh_type == "plane":
        return self.root_states[:, 2].clone()
    elif self.cfg.terrain.mesh_type == "none":
        raise NameError("Can't measure height with terrain mesh type 'none'")

    if env_ids:
        points = quat_apply_yaw(
            self.base_quat[env_ids].repeat(1, num_base_height_points),
            base_height_points[env_ids],
        ) + (self.root_states[env_ids, :3]).unsqueeze(1)
    else:
        points = quat_apply_yaw(
            self.base_quat.repeat(1, num_base_height_points),
            base_height_points,
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
    # heights = (heights1 + heights2 + heights3) / 3

    base_height = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - base_height, dim=1)

    return base_height
