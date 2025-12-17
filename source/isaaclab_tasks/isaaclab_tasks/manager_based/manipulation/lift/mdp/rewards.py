# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, axis_angle_from_quat, matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("objs")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    objects: RigidObject = env.scene[object_cfg.name]
    height = objects.data.object_state_w[torch.arange(env.num_envs).to(device=env.device), env.object_tracking_inds][:, 2]
    return torch.where(height > minimal_height, height, 0)

def object_penalty_xy(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("objs")
) -> torch.Tensor:
    objects: RigidObject = env.scene[object_cfg.name]
    return -torch.norm(objects.data.object_state_w[torch.arange(env.num_envs).to(device=env.device), env.object_tracking_inds][:, :2] - env.scene.env_origins[:, :2] - objects.data.default_object_state[torch.arange(env.num_envs).to(device=env.device), env.object_tracking_inds][:, :2], dim=1)**2


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("objs"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    objects: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    object_pos_w = objects.data.object_state_w[torch.arange(env.num_envs).to(device=env.device), env.object_tracking_inds][:, :3]
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(object_pos_w - ee_w, dim=1)

    # return 1 - torch.tanh(object_ee_distance / std)  
    return -object_ee_distance**2


# def in_air(
#     env: ManagerBasedRLEnv,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
# ) -> torch.Tensor:
    
#     object: RigidObject = env.scene[object_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     # Target object position: (num_envs, 3)
#     cube_pos_w = object.data.root_pos_w
#     # End-effector position: (num_envs, 3)
#     ee_w = ee_frame.data.target_pos_w[..., 0, :]
#     # Distance of the end-effector to the object: (num_envs,) 


#     cube_pos_z = cube_pos_w[:, 2] 
#     ee_z = ee_w[:, 2] 

#     return torch.clip(ee_z - (cube_pos_z + 0.1), min=0.0)

def ori_ee(
    env: ManagerBasedRLEnv,
    # object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name] 
    ee_frame_matrix = matrix_from_quat(ee_frame.data.target_quat_source)[:, 0].permute(0, 2, 1)
    z_axis = ee_frame_matrix @ torch.tensor([[0], [0], [1]], dtype=torch.float32, device=ee_frame.device)
    z_axis = z_axis[:, :, 0]
    scalari = (z_axis * torch.tensor([[0, 0, -1]], dtype=torch.float32, device=ee_frame.device)).sum(dim=-1)
    return scalari

def gripper_dist_reg(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name] 
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    return torch.norm(ee_w, dim=1)**2


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
