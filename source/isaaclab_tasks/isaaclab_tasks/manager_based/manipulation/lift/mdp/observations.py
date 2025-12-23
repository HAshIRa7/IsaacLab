# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms 
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_position_in_robot_root_frame(
     env: ManagerBasedRLEnv,
     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
     objects_cfg: SceneEntityCfg = SceneEntityCfg("objs")
) -> torch.Tensor: 
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[objects_cfg.name] 
    object_pos_w = object.data.object_state_w[torch.arange(env.num_envs).to(device=env.device), env.object_tracking_inds][:, :3]  
    gripper_link_index = robot.data.body_names.index('panda_hand')
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.body_state_w[:, gripper_link_index, :3], robot.data.body_state_w[:, gripper_link_index, 3:7], object_pos_w,
    )
    return object_pos_b 

def object_yaw_in_world_frame(
    env: ManagerBasedRLEnv,
    # robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # object_cfg: SceneEntityCfg = SceneEntityCfg("object")
    objects_cfg: SceneEntityCfg = SceneEntityCfg("objs")
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    object: RigidObject = env.scene[objects_cfg.name]
    object_quat_w = object.data.object_state_w[torch.arange(env.num_envs).to(device=env.device), env.object_tracking_inds][:, 3:7] 
    yaw_in_world_frame = euler_xyz_from_quat(object_quat_w)[2].unsqueeze(dim=-1) 

    return torch.where(yaw_in_world_frame < torch.pi, yaw_in_world_frame, yaw_in_world_frame - 2 * torch.pi) 

def object_roll_in_world_frame(
    env: ManagerBasedRLEnv,
    objects_cfg: SceneEntityCfg = SceneEntityCfg("objs")
) -> torch.Tensor:
    
    object: RigidObject = env.scene[objects_cfg.name] 
    object_quat_w = object.data.object_state_w[torch.arange(env.num_envs).to(device=env.device), env.object_tracking_inds][:, 3:7] 
    roll_in_world_frame = euler_xyz_from_quat(object_quat_w)[0].unsqueeze(dim=-1) 

    return torch.where(roll_in_world_frame < torch.pi, roll_in_world_frame, roll_in_world_frame - 2 * torch.pi)

def class_type(
    env: ManagerBasedRLEnv,
): 
    one_hot = torch.zeros(size=(env.num_envs, 5)).to(device=env.device) 
    one_hot[torch.arange(env.num_envs), env.object_tracking_inds] = 1
    return one_hot


def depth_table_image(
    env: ManagerBasedRLEnv,
    # robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
) -> torch.Tensor:
    """Depth Image of table""" 
    camera = env.scene[camera_cfg.name] 
    DEPTH_MAX = 1.4
    depth_data = camera.data.output['distance_to_image_plane']
    # rgb_data = camera.data.output['rgb'][..., :3] 
    depth_data = torch.clip(depth_data, 0., DEPTH_MAX) / DEPTH_MAX
    depth_data = torch.abs(depth_data - DEPTH_MAX) 
    # rgb_data = rgb_data / 255
    # camera_data = torch.concat((rgb_data, depth_data), dim=-1)
    # camera_data = camera_data.permute(0, 3, 1, 2)
    # return camera_data
    return depth_data
