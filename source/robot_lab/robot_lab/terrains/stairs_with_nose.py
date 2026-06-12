# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh
from isaaclab.terrains.trimesh import mesh_terrains
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPyramidStairsTerrainCfg
from isaaclab.utils import configclass


def pyramid_stairs_with_nose_terrain(
    difficulty: float, cfg: MeshPyramidStairsWithNoseTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate pyramid stairs and add a tread nose on every step edge."""
    meshes_list, origin = mesh_terrains.pyramid_stairs_terrain(difficulty, cfg)
    step_height = _resolve_step_height(difficulty, cfg)

    if not _validate_nose_cfg(cfg, step_height):
        return meshes_list, origin

    if cfg.include_outer_nose and cfg.border_width < cfg.nose_depth:
        raise ValueError(
            "include_outer_nose=True requires border_width >= nose_depth so the outer nose stays inside the terrain."
        )

    num_steps = _resolve_num_steps(cfg)
    meshes_list += _make_pyramid_nose_meshes(cfg, step_height, num_steps)
    return meshes_list, origin


def inverted_pyramid_stairs_with_nose_terrain(
    difficulty: float, cfg: MeshInvertedPyramidStairsWithNoseTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate inverted pyramid stairs and add a tread nose on every step edge."""
    meshes_list, origin = mesh_terrains.inverted_pyramid_stairs_terrain(difficulty, cfg)
    step_height = _resolve_step_height(difficulty, cfg)

    if not _validate_nose_cfg(cfg, step_height):
        return meshes_list, origin

    num_steps = _resolve_num_steps(cfg)
    meshes_list += _make_inverted_pyramid_nose_meshes(cfg, step_height, num_steps)
    return meshes_list, origin


def _resolve_step_height(difficulty: float, cfg: MeshPyramidStairsWithNoseTerrainCfg) -> float:
    return cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])


def _resolve_num_steps(cfg: MeshPyramidStairsWithNoseTerrainCfg) -> int:
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    return int(min(num_steps_x, num_steps_y))


def _validate_nose_cfg(cfg: MeshPyramidStairsWithNoseTerrainCfg, step_height: float) -> bool:
    if cfg.holes:
        raise ValueError("Stairs with nose do not support holes=True.")
    if cfg.nose_depth < 0.0:
        raise ValueError(f"nose_depth must be non-negative. Received {cfg.nose_depth}.")
    if cfg.nose_height < 0.0:
        raise ValueError(f"nose_height must be non-negative. Received {cfg.nose_height}.")
    if cfg.nose_depth == 0.0 or cfg.nose_height == 0.0:
        return False
    if cfg.nose_depth >= cfg.step_width:
        raise ValueError(
            f"nose_depth must be smaller than step_width to keep a usable tread. "
            f"Received nose_depth={cfg.nose_depth}, step_width={cfg.step_width}."
        )
    if cfg.nose_height >= step_height:
        raise ValueError(
            f"nose_height must be smaller than the generated step height. "
            f"Received nose_height={cfg.nose_height}, step_height={step_height}."
        )
    return True


def _make_pyramid_nose_meshes(
    cfg: MeshPyramidStairsWithNoseTerrainCfg, step_height: float, num_steps: int
) -> list[trimesh.Trimesh]:
    meshes_list: list[trimesh.Trimesh] = []
    terrain_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1])
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    start_level = 1 if cfg.include_outer_nose else 2

    for level in range(start_level, num_steps + 2):
        boundary_offset = (level - 1) * cfg.step_width
        boundary_size = (
            terrain_size[0] - 2 * boundary_offset,
            terrain_size[1] - 2 * boundary_offset,
        )
        if boundary_size[0] <= 0.0 or boundary_size[1] <= 0.0:
            continue
        top_z = level * step_height
        meshes_list += _make_nose_ring_meshes(terrain_center, boundary_size, top_z, cfg, inward=False)

    return meshes_list


def _make_inverted_pyramid_nose_meshes(
    cfg: MeshInvertedPyramidStairsWithNoseTerrainCfg, step_height: float, num_steps: int
) -> list[trimesh.Trimesh]:
    meshes_list: list[trimesh.Trimesh] = []
    terrain_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1])
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    start_level = 0 if cfg.include_outer_nose else 1

    for level in range(start_level, num_steps + 1):
        boundary_offset = level * cfg.step_width
        boundary_size = (
            terrain_size[0] - 2 * boundary_offset,
            terrain_size[1] - 2 * boundary_offset,
        )
        if boundary_size[0] <= 2 * cfg.nose_depth or boundary_size[1] <= 2 * cfg.nose_depth:
            continue
        top_z = -level * step_height
        meshes_list += _make_nose_ring_meshes(terrain_center, boundary_size, top_z, cfg, inward=True)

    return meshes_list


def _make_nose_ring_meshes(
    terrain_center: tuple[float, float],
    boundary_size: tuple[float, float],
    top_z: float,
    cfg: MeshPyramidStairsWithNoseTerrainCfg,
    inward: bool,
) -> list[trimesh.Trimesh]:
    cx, cy = terrain_center
    half_x = 0.5 * boundary_size[0]
    half_y = 0.5 * boundary_size[1]
    z = top_z - 0.5 * cfg.nose_height

    if inward:
        top_dims = (boundary_size[0], cfg.nose_depth, cfg.nose_height)
        side_dims = (cfg.nose_depth, boundary_size[1] - 2 * cfg.nose_depth, cfg.nose_height)
        boxes = [
            (top_dims, (cx, cy + half_y - 0.5 * cfg.nose_depth, z)),
            (top_dims, (cx, cy - half_y + 0.5 * cfg.nose_depth, z)),
            (side_dims, (cx + half_x - 0.5 * cfg.nose_depth, cy, z)),
            (side_dims, (cx - half_x + 0.5 * cfg.nose_depth, cy, z)),
        ]
    else:
        top_dims = (boundary_size[0] + 2 * cfg.nose_depth, cfg.nose_depth, cfg.nose_height)
        side_dims = (cfg.nose_depth, boundary_size[1], cfg.nose_height)
        boxes = [
            (top_dims, (cx, cy + half_y + 0.5 * cfg.nose_depth, z)),
            (top_dims, (cx, cy - half_y - 0.5 * cfg.nose_depth, z)),
            (side_dims, (cx + half_x + 0.5 * cfg.nose_depth, cy, z)),
            (side_dims, (cx - half_x - 0.5 * cfg.nose_depth, cy, z)),
        ]

    return [
        trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos)) for dims, pos in boxes
    ]


@configclass
class MeshPyramidStairsWithNoseTerrainCfg(MeshPyramidStairsTerrainCfg):
    """Configuration for a pyramid stair mesh terrain with tread noses.

    The nose is a thin overhang lip placed at the front edge of each tread.
    """

    function = pyramid_stairs_with_nose_terrain

    nose_depth: float = MISSING
    """Horizontal overhang depth of each tread nose (in m)."""

    nose_height: float = 0.025
    """Vertical thickness of each tread nose (in m)."""

    include_outer_nose: bool = True
    """Whether to add a nose to the first step edge."""


@configclass
class MeshInvertedPyramidStairsWithNoseTerrainCfg(MeshPyramidStairsWithNoseTerrainCfg):
    """Configuration for an inverted pyramid stair mesh terrain with tread noses."""

    function = inverted_pyramid_stairs_with_nose_terrain
