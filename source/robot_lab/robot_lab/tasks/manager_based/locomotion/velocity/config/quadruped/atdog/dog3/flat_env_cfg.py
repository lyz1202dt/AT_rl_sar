# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import ATDogDog3RoughEnvCfg


@configclass
class ATDogDog3FlatEnvCfg(ATDogDog3RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Keep flat-task resets very close to the nominal standing state.
        # Small pose perturbations help robustness without spawning the robot
        # in obviously unstable configurations.
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (0.0, 0.02),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.1, 0.1),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "ATDogDog3FlatEnvCfg":
            self.disable_zero_weight_rewards()
