# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlHimActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "HIMActorCritic"
    num_one_step_obs = None
    history_length = 6
    estimator_encoder_hidden_dims = [128, 64, 16]
    estimator_target_hidden_dims = [128, 64]
    estimator_learning_rate = 1.0e-3
    estimator_max_grad_norm = 10.0
    estimator_num_prototypes = 32
    estimator_temperature = 3.0
    estimator_vel_slice = None
    estimator_target_slice = None


@configclass
class ATDogDog2RoughHIMRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "HIMOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "atdog_dog2_rough_him"
    policy = RslRlHimActorCriticCfg(
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


@configclass
class ATDogDog2SandHIMRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "HIMOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "atdog_dog2_sand_him"
    policy = RslRlHimActorCriticCfg(
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


@configclass
class ATDogDog2SlopeHIMRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "HIMOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "atdog_dog2_slope_him"
    policy = RslRlHimActorCriticCfg(
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


@configclass
class ATDogDog2StairsHIMRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "HIMOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "atdog_dog2_stairs_him"
    policy = RslRlHimActorCriticCfg(
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


@configclass
class ATDogDog2FlatHIMRunnerCfg(ATDogDog2RoughHIMRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "atdog_dog2_flat_him"


@configclass
class ATDogDog2BarHIMRunnerCfg(ATDogDog2RoughHIMRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "atdog_dog2_bar_him"
