

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##

AT_DOG2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/atdog/dog2/urdf/dog2.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*L_thigh_joint": 0.8,
            ".*R_thigh_joint": -0.8,
            ".*_calf_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
            min_delay=1,  # physics steps (sim.dt=0.005s): 1 * 5ms = 5ms
            max_delay=2,  # physics steps (fixed delay): 1 * 5ms = 5ms
        ),
    },
)
"""Configuration of atdog dog2 using delayed PD actuators."""

AT_DOG_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/atdog/dog/urdf/dog.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*L_thigh_joint": 0.8,
            ".*R_thigh_joint": -0.8,
            ".*_calf_joint": 0.0,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_(hip|thigh|calf)_joint"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=35.0,
            damping=1.0,
            friction=0.0,
            min_delay=1,  # physics steps (sim.dt=0.005s): 1 * 5ms = 5ms
            max_delay=2,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=23.5,
            velocity_limit_sim=30.0,
            stiffness=0.0,
            damping=0.8,
            friction=0.0,
        ),
    },
)
"""Configuration of atdog dog using wheel-aware leg and wheel actuators."""
