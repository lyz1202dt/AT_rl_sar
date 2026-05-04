# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Humanoid Motion Data Simple Converter

This script only converts CSV motion data directly to NPZ format.

USAGE:
    python data_convert_simple.py

DESCRIPTION:
    1. Read raw CSV motion data
    2. Directly save as NPZ format, keeping the original frame rate

INPUT:
    - CSV file (joints + root pose)
    - URDF file
    - mesh directory

OUTPUT:
    - NPZ file, same content as original data_convert.py

REQUIREMENTS:
    - numpy
    - pandas
    - pinocchio
"""

CSV_FILE = "/workspace/data/atdog2_vmc_walk.csv"
URDF_FILE = "/workspace/isaaclab_assets/Robots/atdog/dog2/urdf/dog2.urdf"
MESH_DIR = "/workspace/isaaclab_assets/Robots/atdog/dog2"
NPZ_FILE = "/workspace/robot_lab/source/robot_lab/robot_lab/tasks/direct/atdog2_amp/motions/atdog2_vmc_walk.npz"

import numpy as np

import pandas as pd
import pinocchio as pin

CSV_HAS_HEADER = True
USE_CSV_VELOCITY_IF_AVAILABLE = True


def quaternion_inverse(q):
    # Input q: (w, x, y, z), returns its inverse.
    w, x, y, z = q
    norm_sq = w * w + x * x + y * y + z * z
    if norm_sq < 1e-8:
        norm_sq = 1e-8
    return np.array([w, -x, -y, -z], dtype=q.dtype) / norm_sq


def quaternion_multiply(q1, q2):
    # Input/output: (w, x, y, z)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=q1.dtype)


def compute_angular_velocity(q_prev, q_next, dt, eps=1e-8):
    """
    Compute angular velocity from adjacent quaternions (w, x, y, z):
      - Relative rotation q_rel = inv(q_prev) * q_next
      - Extract rotation angle and axis from q_rel
      - Return (angle / dt) * axis
    """
    q_inv = quaternion_inverse(q_prev)
    q_rel = quaternion_multiply(q_inv, q_next)
    norm_q_rel = np.linalg.norm(q_rel)
    if norm_q_rel < eps:
        return np.zeros(3, dtype=np.float32)
    q_rel /= norm_q_rel
    w = np.clip(q_rel[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(1.0 - w * w)
    if sin_half < eps:
        return np.zeros(3, dtype=np.float32)
    axis = q_rel[1:] / sin_half
    return (angle / dt) * axis


def build_pin_robot(urdf_path, mesh_dir):
    """
    Load URDF file and construct a pin.RobotWrapper with free-flyer.
    Args:
        urdf_path: Path to the URDF file
        mesh_dir: Directory containing associated mesh files
    Returns:
        robot (pin.RobotWrapper)
    """
    robot = pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_dir, pin.JointModelFreeFlyer())
    return robot


def main():
    # Joint names
    joint_names = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]
    dof_names = np.array(joint_names, dtype=np.str_)

    # 1. Read CSV data
    csv_file = CSV_FILE
    if CSV_HAS_HEADER:
        df = pd.read_csv(csv_file)
    else:
        df = pd.read_csv(csv_file, header=None)

    start_idx = 250
    end_idx = 250 + 10 * 30  # 10s
    end_idx += 1
    df_slice = df.iloc[start_idx:end_idx].copy()
    N = len(df_slice)
    print(f"Loading CSV: {csv_file}, frame range [{start_idx}:{end_idx}], total {N} frames.")

    # Original sampling rate
    fps = 30
    dt = 1.0 / fps

    # Root and joint data
    if CSV_HAS_HEADER:
        root_cols = ["root_pos_x", "root_pos_y", "root_pos_z", "root_quat_w", "root_quat_x", "root_quat_y", "root_quat_z"]
        missing_root = [c for c in root_cols if c not in df_slice.columns]
        missing_joint = [j for j in joint_names if j not in df_slice.columns]
        if missing_root:
            raise ValueError(f"Missing required root columns in CSV: {missing_root}")
        if missing_joint:
            raise ValueError(f"Missing required joint position columns in CSV: {missing_joint}")

        root_data = df_slice[root_cols].to_numpy(dtype=np.float32)
        joint_data = df_slice[joint_names].to_numpy(dtype=np.float32)
    else:
        data_orig = df_slice.to_numpy(dtype=np.float32)
        root_data = data_orig[:, :7]  # (N, 7)
        joint_data = data_orig[:, 7 : 7 + len(joint_names)]  # (N, D)

    # Joint positions
    dof_positions = joint_data.copy()  # (N, D)

    # Joint velocities (prefer CSV velocities if available)
    dof_velocities = np.zeros_like(dof_positions)
    used_csv_joint_vel = False
    if CSV_HAS_HEADER and USE_CSV_VELOCITY_IF_AVAILABLE:
        vel_cols = [f"{j}_vel" for j in joint_names]
        if all(c in df_slice.columns for c in vel_cols):
            dof_velocities = df_slice[vel_cols].to_numpy(dtype=np.float32)
            used_csv_joint_vel = True

    if not used_csv_joint_vel:
        dof_velocities[1:-1] = (dof_positions[2:] - dof_positions[:-2]) / (2 * dt)
        dof_velocities[0] = (dof_positions[1] - dof_positions[0]) / dt
        dof_velocities[-1] = (dof_positions[-1] - dof_positions[-2]) / dt

    # Body link names
    body_names = [
        "base",
        "FR_hip",
        "FR_thigh",
        "FR_calf",
        "FL_hip",
        "FL_thigh",
        "FL_calf",
        "RR_hip",
        "RR_thigh",
        "RR_calf",
        "RL_hip",
        "RL_thigh",
        "RL_calf",
    ]
    body_names = np.array(body_names, dtype=np.str_)
    B = len(body_names)

    body_positions = np.zeros((N, B, 3), dtype=np.float32)
    body_rotations = np.zeros((N, B, 4), dtype=np.float32)

    # Pinocchio forward kinematics
    urdf_path = URDF_FILE
    mesh_dir = MESH_DIR
    robot = build_pin_robot(urdf_path, mesh_dir)
    model = robot.model
    data_pk = robot.data
    nq = model.nq
    if (7 + joint_data.shape[1]) != nq:
        print(
            f"Warning: CSV columns={7 + joint_data.shape[1]}, but pinocchio nq={nq}, may need to check or adjust script"
            " parsing."
        )
    q_pin = pin.neutral(model)
    for i in range(N):
        q_pin[0:3] = root_data[i, 0:3]
        q_pin[3:7] = root_data[i, 3:7]
        dofD = joint_data.shape[1]
        q_pin[7 : 7 + dofD] = joint_data[i, :]
        pin.forwardKinematics(model, data_pk, q_pin)
        pin.updateFramePlacements(model, data_pk)
        for j, link_name in enumerate(body_names):
            fid = model.getFrameId(link_name)
            link_tf = data_pk.oMf[fid]
            body_positions[i, j, :] = link_tf.translation
            quat_xyzw = pin.Quaternion(link_tf.rotation)
            body_rotations[i, j, :] = np.array([quat_xyzw.w, quat_xyzw.x, quat_xyzw.y, quat_xyzw.z], dtype=np.float32)

    # Linear velocities
    body_linear_velocities = np.zeros_like(body_positions)
    body_linear_velocities[1:-1] = (body_positions[2:] - body_positions[:-2]) / (2 * dt)
    body_linear_velocities[0] = (body_positions[1] - body_positions[0]) / dt
    body_linear_velocities[-1] = (body_positions[-1] - body_positions[-2]) / dt

    # Angular velocities
    body_angular_velocities = np.zeros((N, B, 3), dtype=np.float32)
    for j in range(B):
        quats = body_rotations[:, j, :]
        angular_vels = np.zeros((N, 3), dtype=np.float32)
        if N > 1:
            angular_vels[0] = compute_angular_velocity(quats[0], quats[1], dt)
            angular_vels[-1] = compute_angular_velocity(quats[-2], quats[-1], dt)
        for k in range(1, N - 1):
            av1 = compute_angular_velocity(quats[k - 1], quats[k], dt)
            av2 = compute_angular_velocity(quats[k], quats[k + 1], dt)
            angular_vels[k] = 0.5 * (av1 + av2)
        body_angular_velocities[:, j, :] = angular_vels

    # Prefer base linear/angular velocities from CSV if available
    used_csv_base_vel = False
    if CSV_HAS_HEADER and USE_CSV_VELOCITY_IF_AVAILABLE:
        base_lin_cols = ["base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z"]
        base_ang_cols = ["base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"]
        if all(c in df_slice.columns for c in base_lin_cols + base_ang_cols):
            base_idx = body_names.tolist().index("base")
            body_linear_velocities[:, base_idx, :] = df_slice[base_lin_cols].to_numpy(dtype=np.float32)
            body_angular_velocities[:, base_idx, :] = df_slice[base_ang_cols].to_numpy(dtype=np.float32)
            used_csv_base_vel = True

    # Save
    data_dict = {
        "fps": fps,
        "dof_names": dof_names,
        "body_names": body_names,
        "dof_positions": dof_positions,
        "dof_velocities": dof_velocities,
        "body_positions": body_positions,
        "body_rotations": body_rotations,
        "body_linear_velocities": body_linear_velocities,
        "body_angular_velocities": body_angular_velocities,
    }
    out_filename = NPZ_FILE
    np.savez(out_filename, **data_dict)
    print(f"Conversion completed, data saved to {out_filename}")
    print("fps:", fps)
    print("dof_names:", dof_names.shape)
    print("body_names:", body_names.shape)
    print("dof_positions:", dof_positions.shape)
    print("dof_velocities:", dof_velocities.shape)
    print(f"joint_vel_source: {'csv' if used_csv_joint_vel else 'finite-difference'}")
    print("body_positions:", body_positions.shape)
    print("body_rotations:", body_rotations.shape)
    print("body_linear_velocities:", body_linear_velocities.shape)
    print("body_angular_velocities:", body_angular_velocities.shape)
    print(f"base_vel_source: {'csv' if used_csv_base_vel else 'finite-difference'}")


if __name__ == "__main__":
    main()
