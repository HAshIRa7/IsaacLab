# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Scene definition
##


# @configclass
# class ObjectTableSceneCfg(InteractiveSceneCfg):
#     """Configuration for the lift scene with a robot and a object.
#     This is the abstract base implementation, the exact scene is defined in the derived classes
#     which need to set the target object, robot and end-effector frames
#     """

#     # robots: will be populated by agent env cfg
#     robot: ArticulationCfg = MISSING
#     # end-effector sensor: will be populated by agent env cfg
#     ee_frame: FrameTransformerCfg = MISSING
#     # target object: will be populated by agent env cfg
#     object: RigidObjectCfg | DeformableObjectCfg = MISSING
#     # chipasiki: RigidObjectCfg | DeformableObjectCfg = MISSING

#     # Table
#     table = AssetBaseCfg(
#         prim_path="{ENV_REGEX_NS}/Table",
#         init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
#         spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
#     ) 

#     # chipasiki = AssetBaseCfg(
#     #     prim_path="{ENV_REGEX_NS}/Chipasi",
#     #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1.0, 0.0, 0.0, 0.0]),
#     #     spawn=UsdFileCfg(usd_path=f"/home/maslennikov-egor/chipasi/chipasi.usd"),
#     # )

#     # plane
#     plane = AssetBaseCfg(
#         prim_path="/World/GroundPlane",
#         init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
#         spawn=GroundPlaneCfg(),
#     )

#     # lights
#     light = AssetBaseCfg(
#         prim_path="/World/light",
#         spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
#     )
 
#     obj = RigidObjectCfg(
#         spawn=sim_utils.UsdFileCfg(
#             usd_path='/home/maslennikov-egor/chipasi/orbit_obj.usd',
#             rigid_props=sim_utils.RigidBodyPropertiesCfg(
#                     rigid_body_enabled=True,
#                     disable_gravity=False,
#                     max_depenetration_velocity=50.0,
#                     linear_damping = 1,
#                     angular_damping = 2,
#                     max_contact_impulse = float("inf"),
#                     max_linear_velocity=1,
#                     solver_position_iteration_count=32,
#                     solver_velocity_iteration_count=16,
#                     stabilization_threshold=0.1,
#                     ),
#             mass_props = sim_utils.MassPropertiesCfg(density=5.0),
#             articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#                 articulation_enabled=False,
#             ),
#             # semantic_tags=[("class", f"{MGN_PATH.split('/')[-2]}"), ("color", "red")],
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(
#             pos=(0., .5, 0.2),
#             rot=(1.0, 0.0, 0.0, 0.0),
#         ),
#         collision_group = 0,
#         prim_path = "{ENV_REGEX_NS}/obj"
#     ) 


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """ 

    def  __init__(self, **kwargs):

        # robots: will be populated by agent env cfg
        # self.robot: ArticulationCfg = MISSING
        # # end-effector sensor: will be populated by agent env cfg
        # ee_frame: FrameTransformerCfg = MISSING
        # # target object: will be populated by agent env cfg
        # object: RigidObjectCfg | DeformableObjectCfg = MISSING
        # # chipasiki: RigidObjectCfg | DeformableObjectCfg = MISSING

        # Table
        self.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
            spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        ) 

        # chipasiki = AssetBaseCfg(
        #     prim_path="{ENV_REGEX_NS}/Chipasi",
        #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1.0, 0.0, 0.0, 0.0]),
        #     spawn=UsdFileCfg(usd_path=f"/home/maslennikov-egor/chipasi/chipasi.usd"),
        # )

        # plane
        self.plane = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
            spawn=GroundPlaneCfg(),
        )

        # lights
        self.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
    
        obj_cfg = RigidObjectCfg(
            spawn=sim_utils.UsdFileCfg(
                # /home/maslennikov-egor/MetaIsaacGrasp/models/models_ifl/010/orbit_obj.usd
                usd_path='/home/maslennikov-egor/MetaIsaacGrasp/models/models_ifl/my_model/chips_bag.usd',
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        disable_gravity=False,
                        max_depenetration_velocity=50.0,
                        linear_damping = 1,
                        angular_damping = 2,
                        max_contact_impulse = float("inf"),
                        max_linear_velocity=1,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=16,
                        stabilization_threshold=0.1,
                        ),
                mass_props = sim_utils.MassPropertiesCfg(density=5.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,
                ),
                # semantic_tags=[("class", f"{MGN_PATH.split('/')[-2]}"), ("color", "red")],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.4, 0.0, 0.055),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            collision_group = 0,
            prim_path = "{ENV_REGEX_NS}/obj"
        ) 

        cube_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path="/home/maslennikov-egor/MetaIsaacGrasp/models/models_ifl/can/chips_bag.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    disable_gravity=False,
                    max_depenetration_velocity=4.0,
                    linear_damping = .1,
                    angular_damping = .1,
                    max_contact_impulse = float("inf"),
                    max_angular_velocity=2,
                    max_linear_velocity=2,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=10,
                    stabilization_threshold=0.1,
                ),
                mass_props = sim_utils.MassPropertiesCfg(density=5.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,
                ),
            ),
            collision_group=0
        )

        obj_1_cfg = RigidObjectCfg(
            spawn=sim_utils.UsdFileCfg(
                # /home/maslennikov-egor/MetaIsaacGrasp/models/models_ifl/010/orbit_obj.usd
                usd_path='/home/maslennikov-egor/MetaIsaacGrasp/models/models_ifl/can/chips_bag.usd',
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        disable_gravity=False,
                        max_depenetration_velocity=50.0,
                        linear_damping = 1,
                        angular_damping = 2,
                        max_contact_impulse = float("inf"),
                        max_linear_velocity=1,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=16,
                        stabilization_threshold=0.1,
                        ),
                mass_props = sim_utils.MassPropertiesCfg(density=5.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,
                ),
                # semantic_tags=[("class", f"{MGN_PATH.split('/')[-2]}"), ("color", "red")],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.4, 0.0, 0.055),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            collision_group = 0,
            prim_path = "{ENV_REGEX_NS}/obj_1"
        )
        obj_2_cfg = RigidObjectCfg(
            spawn=sim_utils.UsdFileCfg(
                # /home/maslennikov-egor/MetaIsaacGrasp/models/models_ifl/010/orbit_obj.usd
                usd_path='/home/maslennikov-egor/MetaIsaacGrasp/models/models_ifl/tetra/chips_bag.usd',
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        disable_gravity=False,
                        max_depenetration_velocity=50.0,
                        linear_damping = 1,
                        angular_damping = 2,
                        max_contact_impulse = float("inf"),
                        max_linear_velocity=1,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=16,
                        stabilization_threshold=0.1,
                        ),
                mass_props = sim_utils.MassPropertiesCfg(density=5.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,
                ),
                # semantic_tags=[("class", f"{MGN_PATH.split('/')[-2]}"), ("color", "red")],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.4, -0.3, 0.055),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            collision_group = 0,
            prim_path = "{ENV_REGEX_NS}/obj_2"
        ) 


        OBJ_CFGs = [obj_2_cfg, obj_1_cfg]# obj_2_cfg]

        super().__init__(**kwargs) 

        # self.obj = obj_cfg.replace(prim_path="{ENV_REGEX_NS}/obj")
        # self.obj_1 = obj_1_cfg.replace(prim_path="{ENV_REGEX_NS}/obj_1") 
        # self.obj_2 = obj_2_cfg.replace(prim_path="{ENV_REGEX_NS}/obj_2") 


        self.objs: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
            rigid_objects={
                f"obj_{i}": OBJ_CFGs[i].replace(
                    prim_path="{ENV_REGEX_NS}/obj_"+str(i)
                )
                for i in range(len(OBJ_CFGs))
            },
        )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame) 
        # depth_image = ObsTerm(func=mdp.depht_table_image)
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)  
        class_type = ObsTerm(func=mdp.class_type)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # self.history_length = 4
    
    @configclass
    class CriticCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame) 
        # depth_image = ObsTerm(func=mdp.depht_table_image)
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action) 
        class_type = ObsTerm(func=mdp.class_type)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # self.history_length = 4
    
    # @configclass
    # class DepthImage(ObsGroup):
    #     depth_image = ObsTerm(func=mdp.depht_table_image)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    # depth_image: DepthImage = DepthImage()



@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.15, 0.15), "z": (0.0, 0.0), "yaw": (-0.5, 0.5)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("objs"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=20.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.05}, weight=30.0)  

    object_panelty_xy = RewTerm(func=mdp.object_penalty_xy, weight=0.02)

    # stay_in_air = RewTerm(func=mdp.in_air, weight=0.5) 

    ori_gripper = RewTerm(func=mdp.ori_ee, weight=0.3) 

    regi = RewTerm(func=mdp.gripper_dist_reg, weight=0.00005)

    # object_goal_tracking = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=16.0,
    # )

    # object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=5.0,
    # )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-8e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-3e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("objs")}
    ) 

    object_xy_drift = DoneTerm(
        func=mdp.object_termination_xy, params={"object_cfg": SceneEntityCfg("objs")}
    )



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    # )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01 # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.enable_ccd = True
