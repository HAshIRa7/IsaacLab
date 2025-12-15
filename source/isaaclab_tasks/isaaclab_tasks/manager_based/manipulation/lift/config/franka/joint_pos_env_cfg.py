# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg 
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR 
import isaaclab.sim as sim_utils

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg 
from dataclasses import MISSING

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

CAM_HEIGHT, CAM_WIDTH = 224, 224
MODALITIES = ["rgb", "distance_to_image_plane", "instance_segmentation_fast"]

focus_distance = 44. # 400.0
focal_length_cm = 1.69 # 18.14756
horizontal_aperture_mm = 2.59 # 20.955
sensor_width_m = horizontal_aperture_mm / 1000
focal_length_m = focal_length_cm / 1000  # error in orbit, it ignore the unit of focal length

CAMERA_CFG = CameraCfg(
    update_period=0.001,
    height=CAM_HEIGHT,
    width=CAM_WIDTH,
    data_types=[*MODALITIES],
    colorize_instance_id_segmentation=False,
    colorize_semantic_segmentation=False,
    colorize_instance_segmentation=False,
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=focal_length_cm,
        focus_distance=focus_distance,
        horizontal_aperture=horizontal_aperture_mm,
        clipping_range=(0.1, 1.0e5),
    ),
    offset=MISSING,
)

@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") 
        self.scene.camera = CAMERA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot/panda_hand/camera", 
                                offset=CameraCfg.OffsetCfg(
                                    pos = (0.15, 0.0, -0.1),
                                    rot = (0.7071067811865476, 0.0, 0, 0.7071067811865475),
                                    # pos=(0.1, -0.1, 0.3), 
                                    # rot=(-0.36946689350154266, 0.6214172282774272, -0.6029111395713285, 0.3374051345200147), 
                                    convention="ros")
                            )

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # self.scene.chipasiki = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object/Chipasiki",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, 0.055], rot=[1, 0, 0, 0]),
        #     spawn=UsdFileCfg(
        #         usd_path=f"/home/maslennikov-egor/chipasi/orbit_obj.usd",
        #         scale=(0.8, 0.8, 0.8),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # ) 

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
