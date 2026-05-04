# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025 Linden
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from robot_lab.assets.atdog import AT_DOG2_CFG

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class ATDog2AmpWalkEnvCfg(DirectRLEnvCfg):
    """ATDog2 AMP environment config."""

    # basic reward
    rew_termination = -0
    rew_action_l2 = -0.1
    rew_joint_pos_limits = -10
    rew_joint_acc_l2 = -1.0e-06
    rew_joint_vel_l2 = -0.001
    # velocity tracking reward parameters
    rew_track_lin_vel_xy_exp = 20.0
    rew_track_ang_vel_z_exp = 15.0
    track_lin_vel_xy_std = 0.5
    track_ang_vel_z_std = 0.5
    # imitation reward parameters
    rew_imitation_pos = 1.0
    rew_imitation_rot = 0.5
    rew_imitation_joint_pos = 2.5
    rew_imitation_joint_vel = 1.0
    imitation_sigma_pos = 1.2
    imitation_sigma_rot = 0.5
    imitation_sigma_joint_pos = 1.5
    imitation_sigma_joint_vel = 8.0

    # env
    episode_length_s = 10.0
    decimation = 1
    dt = 1 / 60

    # spaces
    # policy obs aligned with RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-v0 habits:
    # base_ang_vel(3) + projected_gravity(3) + velocity_command(3) + joint_pos(12) + joint_vel(12) + last_action(12)
    # (no base linear velocity in policy input)
    observation_space = 45
    action_space = 12
    state_space = 0
    num_amp_observations = 3
    amp_observation_space = 44

    early_termination = True
    termination_height = 0.10

    # Replace this with your own VMC-recorded motion file
    motion_file = os.path.join(MOTIONS_DIR, "atdog2_vmc_walk.npz")
    reference_body = "base"
    key_body_names = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
    command_velocity_range = {
        "lin_vel_x": (-0.6, 1.0),
        "lin_vel_y": (-0.4, 0.4),
        "ang_vel_z": (-1.0, 1.0),
    }
    command_resample_time_s = 3.0
    reset_strategy = "random-start"  # default, random, random-start
    """Strategy to be followed when resetting each environment.

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=dt,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = AT_DOG2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
