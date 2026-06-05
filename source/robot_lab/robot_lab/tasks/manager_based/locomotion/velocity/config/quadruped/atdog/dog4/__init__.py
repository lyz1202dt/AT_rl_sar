# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-ATDog-Dog4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:ATDogDog4FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDog4FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDog4FlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-ATDog-Dog4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:ATDogDog4RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDog4RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDog4RoughTrainerCfg",
    },
)

# gym.register(
#     id="RobotLab-Isaac-Velocity-Stairs-ATDog-Dog4-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:ATDogDog4StairsEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDog4StairsPPORunnerCfg",
#         "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDog4StairsTrainerCfg",
#     },
# )

# gym.register(
#     id="RobotLab-Isaac-Velocity-Sand-ATDog-Dog4-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.sand_env_cfg:ATDogDog4SandEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDog4SandPPORunnerCfg",
#         "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDog4SandTrainerCfg",
#     },
# )

# gym.register(
#     id="RobotLab-Isaac-Velocity-Slope-ATDog-Dog4-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.slope_env_cfg:ATDogDog4SlopeEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDog4SlopePPORunnerCfg",
#         "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDog4SlopeTrainerCfg",
#     },
# )
