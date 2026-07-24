# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import os

import torch
import torch.nn.functional as F


class _HIMPolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic, normalizer=None):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
        self.num_one_step_obs = int(actor_critic.num_one_step_obs)
        self.num_velocity_features = int(actor_critic.num_velocity_features)
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else None

    def forward(self, obs_history: torch.Tensor):
        if self.normalizer is not None:
            obs_history = self.normalizer(obs_history)
        parts = self.estimator(obs_history)
        pred_vel = parts[..., : self.num_velocity_features]
        latent = F.normalize(parts[..., self.num_velocity_features :], dim=-1, p=2.0)
        actor_input = torch.cat((obs_history[:, : self.num_one_step_obs], pred_vel, latent), dim=-1)
        return self.actor(actor_input)


def export_him_policy_as_jit(actor_critic, path: str, filename: str = "policy.pt", normalizer=None):
    os.makedirs(path, exist_ok=True)
    exporter = _HIMPolicyExporter(actor_critic, normalizer=normalizer).to("cpu")
    scripted = torch.jit.script(exporter)
    scripted.save(os.path.join(path, filename))


def export_him_policy_as_onnx(actor_critic, path: str, filename: str = "policy.onnx", normalizer=None):
    os.makedirs(path, exist_ok=True)
    exporter = _HIMPolicyExporter(actor_critic, normalizer=normalizer).to("cpu")
    exporter.eval()
    dummy = torch.zeros(1, actor_critic.num_actor_obs, dtype=torch.float32)
    torch.onnx.export(
        exporter,
        dummy,
        os.path.join(path, filename),
        input_names=["obs_history"],
        output_names=["actions"],
        opset_version=17,
    )
