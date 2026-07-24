# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from .him_estimator import HIMEstimator, get_activation


class HIMActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_one_step_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        critic_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        estimator_encoder_hidden_dims: list[int] | tuple[int, ...] = (128, 64, 16),
        estimator_target_hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        estimator_learning_rate: float = 1e-3,
        estimator_max_grad_norm: float = 10.0,
        estimator_num_prototypes: int = 32,
        estimator_temperature: float = 3.0,
        estimator_vel_slice: tuple[int, int] = (45, 48),
        estimator_target_slice: tuple[int, int] = (3, 48),
        **_: object,
    ):
        super().__init__()

        if num_actor_obs % num_one_step_obs != 0:
            raise ValueError(
                f"Actor observation dim {num_actor_obs} is not divisible by one-step dim {num_one_step_obs}."
            )

        self.history_size = num_actor_obs // num_one_step_obs
        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions
        self.num_one_step_obs = num_one_step_obs
        self.vel_slice = tuple(estimator_vel_slice)
        self.target_slice = tuple(estimator_target_slice)
        self.num_velocity_features = self.vel_slice[1] - self.vel_slice[0]
        self.num_latent = int(estimator_encoder_hidden_dims[-1])

        activation_layer = get_activation(activation)

        self.estimator = HIMEstimator(
            temporal_steps=self.history_size,
            num_one_step_obs=num_one_step_obs,
            vel_slice=self.vel_slice,
            target_slice=self.target_slice,
            enc_hidden_dims=estimator_encoder_hidden_dims,
            tar_hidden_dims=estimator_target_hidden_dims,
            activation=activation,
            learning_rate=estimator_learning_rate,
            max_grad_norm=estimator_max_grad_norm,
            num_prototypes=estimator_num_prototypes,
            temperature=estimator_temperature,
        )

        actor_layers: list[nn.Module] = []
        actor_input_dim = num_one_step_obs + self.num_velocity_features + self.num_latent
        actor_layers.extend([nn.Linear(actor_input_dim, actor_hidden_dims[0]), activation_layer])
        for idx, hidden_dim in enumerate(actor_hidden_dims):
            if idx == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(hidden_dim, num_actions))
            else:
                actor_layers.extend([nn.Linear(hidden_dim, actor_hidden_dims[idx + 1]), activation_layer])
        self.actor = nn.Sequential(*actor_layers)

        critic_layers: list[nn.Module] = []
        critic_layers.extend([nn.Linear(num_critic_obs, critic_hidden_dims[0]), activation_layer])
        for idx, hidden_dim in enumerate(critic_hidden_dims):
            if idx == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(hidden_dim, 1))
            else:
                critic_layers.extend([nn.Linear(hidden_dim, critic_hidden_dims[idx + 1]), activation_layer])
        self.critic = nn.Sequential(*critic_layers)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args = False

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution is not initialized.")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution is not initialized.")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution is not initialized.")
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None):
        del dones

    def update_distribution(self, obs_history: torch.Tensor):
        with torch.no_grad():
            pred_vel, latent = self.estimator(obs_history)
        actor_input = torch.cat((obs_history[:, : self.num_one_step_obs], pred_vel, latent), dim=-1)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs_history: torch.Tensor | None = None, **_: object) -> torch.Tensor:
        if obs_history is None:
            raise ValueError("HIM actor requires observation history.")
        self.update_distribution(obs_history)
        return self.distribution.sample()

    def act_inference(self, obs_history: torch.Tensor, observations: torch.Tensor | None = None) -> torch.Tensor:
        del observations
        pred_vel, latent = self.estimator(obs_history)
        return self.actor(torch.cat((obs_history[:, : self.num_one_step_obs], pred_vel, latent), dim=-1))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution is not initialized.")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **_: object) -> torch.Tensor:
        return self.critic(critic_observations)
