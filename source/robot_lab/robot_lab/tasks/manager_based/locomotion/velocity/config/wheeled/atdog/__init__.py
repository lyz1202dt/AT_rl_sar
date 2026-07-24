# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-ATDog-Dog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:ATDogDogFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDogFlatPPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:ATDogDogFlatHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDogFlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-ATDog-Dog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:ATDogDogRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDogRoughPPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:ATDogDogRoughHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDogRoughTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Slope-ATDog-Dog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.slope_env_cfg:ATDogDogSlopeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDogSlopePPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:ATDogDogSlopeHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDogSlopeTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Sand-ATDog-Dog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sand_env_cfg:ATDogDogSandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDogSandPPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:ATDogDogSandHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDogSandTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Stairs-ATDog-Dog-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:ATDogDogStairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDogStairsPPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:ATDogDogStairsHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDogStairsTrainerCfg",
    },
)
