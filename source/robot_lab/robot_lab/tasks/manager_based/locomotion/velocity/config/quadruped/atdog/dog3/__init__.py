# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-ATDog-Dog3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:ATDogDog3FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDog3FlatPPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:ATDogDog3FlatHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDog3FlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-ATDog-Dog3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:ATDogDog3RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDog3RoughPPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:ATDogDog3RoughHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDog3RoughTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Stairs-ATDog-Dog3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:ATDogDog3StairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDog3StairsPPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:ATDogDog3StairsHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDog3StairsTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Sand-ATDog-Dog3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sand_env_cfg:ATDogDog3SandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDog3SandPPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:ATDogDog3SandHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDog3SandTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Slope-ATDog-Dog3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.slope_env_cfg:ATDogDog3SlopeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDog3SlopePPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:ATDogDog3SlopeHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDog3SlopeTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Bar-ATDog-Dog3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bar_env_cfg:ATDogDog3BarEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDog3BarPPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:ATDogDog3BarHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDog3BarTrainerCfg",
    },
)
