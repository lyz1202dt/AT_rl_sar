# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class HIMVecEnvWrapper:
    def __init__(self, env, clip_actions: float, num_one_step_obs: int | None = None):
        self._raw_env = env
        self._wrapped_env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

        self.num_envs = int(self._wrapped_env.num_envs)
        self.num_actions = int(self._wrapped_env.num_actions)
        self.device = self._wrapped_env.device
        self.max_episode_length = self._wrapped_env.max_episode_length

        self.policy_term_slices, self.policy_one_step_term_slices = self._build_group_layout("policy")
        self.critic_term_slices, _ = self._build_group_layout("critic")

        inferred_one_step_obs = self._infer_num_one_step_obs()
        requested_one_step_obs = None if num_one_step_obs is None else int(num_one_step_obs)
        if inferred_one_step_obs is None and requested_one_step_obs is None:
            raise ValueError("Unable to infer HIM one-step actor observation dimension from the policy group.")
        if (
            inferred_one_step_obs is not None
            and requested_one_step_obs is not None
            and inferred_one_step_obs != requested_one_step_obs
        ):
            print(
                "[WARN] HIM policy one-step observation dim mismatch: "
                f"configured={requested_one_step_obs}, inferred={inferred_one_step_obs}. "
                "Using inferred value."
            )
        self.num_one_step_obs = (
            inferred_one_step_obs if inferred_one_step_obs is not None else requested_one_step_obs
        )

        self.default_estimator_vel_slice, self.default_estimator_target_slice = self._infer_estimator_slices()

        actor_obs, critic_obs = self._split_observations(self._wrapped_env.get_observations())
        self._cache_observations(actor_obs, critic_obs)
        self.num_obs = int(actor_obs.shape[-1])
        self.num_privileged_obs = None if critic_obs is None else int(critic_obs.shape[-1])

        if self.num_obs % self.num_one_step_obs != 0:
            raise ValueError(
                "Policy observation history is incompatible with HIM. "
                f"actor_obs_dim={self.num_obs}, one_step_dim={self.num_one_step_obs}."
            )

    @property
    def unwrapped(self):
        return self._wrapped_env.unwrapped

    def reset(self):
        obs_dict, extras = self._wrapped_env.reset()
        actor_obs, critic_obs = self._split_observations(obs_dict)
        self._cache_observations(actor_obs, critic_obs)
        extras = dict(extras)
        if critic_obs is not None:
            extras.setdefault("critic_obs", critic_obs)
        return actor_obs, extras

    def step(self, actions: torch.Tensor):
        obs_dict, rewards, dones, extras = self._wrapped_env.step(actions)
        actor_obs, critic_obs = self._split_observations(obs_dict)
        self._cache_observations(actor_obs, critic_obs)
        extras = dict(extras)
        if critic_obs is not None:
            extras.setdefault("critic_obs", critic_obs)
        return actor_obs, rewards, dones, extras

    def get_observations(self) -> torch.Tensor:
        actor_obs, critic_obs = self._split_observations(self._wrapped_env.get_observations())
        self._cache_observations(actor_obs, critic_obs)
        return actor_obs

    def get_privileged_observations(self) -> torch.Tensor | None:
        if self._latest_critic_obs is not None:
            return self._latest_critic_obs
        _, critic_obs = self._split_observations(self._wrapped_env.get_observations())
        self._latest_critic_obs = critic_obs
        return critic_obs

    def get_critic_observations(self) -> torch.Tensor:
        critic_obs = self.get_privileged_observations()
        if critic_obs is None:
            return self.get_observations()
        return critic_obs

    def _cache_observations(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor | None):
        self._latest_actor_obs = actor_obs
        self._latest_critic_obs = critic_obs

    def _split_observations(self, observations) -> tuple[torch.Tensor, torch.Tensor | None]:
        actor_obs = self._extract_group_observations(observations, "policy")
        if actor_obs is None:
            raise KeyError("HIM requires a 'policy' observation group.")

        critic_obs = self._extract_group_observations(observations, "critic")
        if critic_obs is None:
            critic_obs = self._compute_group("critic")
        return actor_obs, critic_obs

    def _extract_group_observations(self, observations, group_name: str) -> torch.Tensor | None:
        if observations is None:
            return None

        group_obs = None
        try:
            group_obs = observations[group_name]
        except Exception:
            if isinstance(observations, Mapping):
                group_obs = observations.get(group_name)

        if group_obs is None:
            return None
        return self._flatten_group_tensor(group_obs)

    def _compute_group(self, group_name: str) -> torch.Tensor | None:
        observation_manager = getattr(self.unwrapped, "observation_manager", None)
        if observation_manager is None:
            return None
        try:
            group_obs = observation_manager.compute_group(group_name=group_name)
        except Exception:
            return None
        return self._flatten_group_tensor(group_obs)

    @staticmethod
    def _flatten_group_tensor(group_obs) -> torch.Tensor:
        if not isinstance(group_obs, torch.Tensor):
            group_obs = torch.as_tensor(group_obs)
        if group_obs.ndim == 1:
            group_obs = group_obs.unsqueeze(0)
        elif group_obs.ndim > 2:
            group_obs = group_obs.flatten(start_dim=1)
        return group_obs

    def _infer_num_one_step_obs(self) -> int | None:
        if not self.policy_one_step_term_slices:
            return None
        return max(term_slice.stop for term_slice in self.policy_one_step_term_slices.values())

    def _build_group_layout(self, group_name: str) -> tuple[dict[str, slice], dict[str, slice]]:
        observation_manager = getattr(self.unwrapped, "observation_manager", None)
        if observation_manager is None:
            return {}, {}

        term_names = getattr(observation_manager, "_group_obs_term_names", {}).get(group_name, [])
        term_cfgs = getattr(observation_manager, "_group_obs_term_cfgs", {}).get(group_name, [])
        term_dims = observation_manager.group_obs_term_dim.get(group_name, [])
        if not term_names or not term_cfgs or not term_dims:
            return {}, {}

        group_layout: dict[str, slice] = {}
        one_step_layout: dict[str, slice] = {}
        group_cursor = 0
        one_step_cursor = 0

        for term_name, term_cfg, term_dim in zip(term_names, term_cfgs, term_dims, strict=False):
            total_dim = self._shape_numel(term_dim)
            base_dim = self._infer_base_term_dim(term_cfg, total_dim)
            group_layout[term_name] = slice(group_cursor, group_cursor + total_dim)
            one_step_layout[term_name] = slice(one_step_cursor, one_step_cursor + base_dim)
            group_cursor += total_dim
            one_step_cursor += base_dim

        return group_layout, one_step_layout

    def _infer_base_term_dim(self, term_cfg, total_dim: int) -> int:
        try:
            sample = term_cfg.func(self.unwrapped, **term_cfg.params)
            sample = self._flatten_group_tensor(sample)
            return int(sample.shape[-1])
        except Exception:
            history_length = int(getattr(term_cfg, "history_length", 0) or 0)
            if history_length <= 1 or total_dim % history_length != 0:
                return total_dim
            return total_dim // history_length

    def _infer_estimator_slices(self) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        vel_slice = self._term_slice(self.critic_term_slices, "base_lin_vel")
        target_slice = self._shared_policy_target_slice()
        return vel_slice, target_slice

    @staticmethod
    def _term_slice(layout: dict[str, slice], term_name: str) -> tuple[int, int] | None:
        term_slice = layout.get(term_name)
        if term_slice is None:
            return None
        return (term_slice.start, term_slice.stop)

    def _shared_policy_target_slice(self) -> tuple[int, int] | None:
        if not self.policy_term_slices or not self.critic_term_slices:
            return None

        shared_names = [name for name in self.policy_term_slices if name in self.critic_term_slices]
        if not shared_names:
            return None

        slices = [self.critic_term_slices[name] for name in shared_names]
        start = slices[0].start
        stop = slices[0].stop
        for term_slice in slices[1:]:
            if term_slice.start != stop:
                return None
            stop = term_slice.stop
        return (start, stop)

    @staticmethod
    def _shape_numel(shape) -> int:
        if isinstance(shape, int):
            return int(shape)
        if len(shape) == 0:
            return 1
        numel = 1
        for dim in shape:
            numel *= int(dim)
        return numel

    def __getattr__(self, name: str):
        return getattr(self._wrapped_env, name)
