# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ATDogDog4RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "atdog_dog4_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

# @configclass
# class ATDogDog4SandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 24
#     max_iterations = 20000
#     save_interval = 100
#     experiment_name = "atdog_dog4_sand"
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_obs_normalization=False,
#         critic_obs_normalization=False,
#         actor_hidden_dims=[512, 256, 128],
#         critic_hidden_dims=[512, 256, 128],
#         activation="elu",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.01,
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         learning_rate=1.0e-3,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )
    
    
    
# @configclass
# class ATDogDog4SlopePPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 24
#     max_iterations = 20000
#     save_interval = 100
#     experiment_name = "atdog_dog4_slope"
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_obs_normalization=False,
#         critic_obs_normalization=False,
#         actor_hidden_dims=[512, 256, 128],
#         critic_hidden_dims=[512, 256, 128],
#         activation="elu",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.01,
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         learning_rate=1.0e-3,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )
    
    
    
# @configclass
# class ATDogDog4StairsPPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 24
#     max_iterations = 20000
#     save_interval = 100
#     experiment_name = "atdog_dog4_stairs"
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_obs_normalization=False,
#         critic_obs_normalization=False,
#         actor_hidden_dims=[512, 256, 128],
#         critic_hidden_dims=[512, 256, 128],
#         activation="elu",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.01,
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         learning_rate=1.0e-3,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )


@configclass
class ATDogDog4FlatPPORunnerCfg(ATDogDog4RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "atdog_dog4_flat"
