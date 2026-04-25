"""Body-frame 6-DoF velocity labels for the Go2 v2 dataset.

`sensors.npz['labels']` stores:
  [:, :3]  gt_vel_x/y/z          — WORLD-FRAME linear velocity (from (p_t+1 - p_t)/dt on world pose)
  [:, 3:]  omega_body_x/y/z      — BODY-FRAME angular velocity (fixed in Session 18)

A pure body-frame state estimator wants both in body frame. This module
rotates the linear-velocity part using the stored GT quaternion from data.csv.
Also clips angular velocity to remove sensor-glitch outliers (7 trajectories
hit 150-188 rad/s per Session 17 — clip at ±10 rad/s which is well above
aggressive Go2 yaw rate).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz body-to-world → (N, 3, 3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.empty((len(q), 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def compute_body_labels(traj_dir: str, max_angular_vel_rad: float = 10.0) -> dict:
    """Produce body-frame (lin_vel, ang_vel) aligned to sensors.npz['frame_idx']."""
    s = np.load(os.path.join(traj_dir, "sensors.npz"))
    frame_idx = s["frame_idx"].astype(np.int32)
    labels = s["labels"].astype(np.float32)   # world-frame lin, body-frame ang

    df = pd.read_csv(os.path.join(traj_dir, "data.csv")).set_index("frame_idx").loc[frame_idx]
    q_wxyz = df[["gt_ori_w", "gt_ori_x", "gt_ori_y", "gt_ori_z"]].to_numpy(np.float32)
    R_wb = quat_wxyz_to_R(q_wxyz)

    lin_world = labels[:, :3]                                     # (N, 3)
    lin_body = np.einsum("nij,nj->ni", R_wb.transpose(0, 2, 1),   # R_wb^T @ v_world
                         lin_world).astype(np.float32)
    ang_body = np.clip(labels[:, 3:6], -max_angular_vel_rad, max_angular_vel_rad)
    return {
        "frame_idx": frame_idx,
        "lin_vel_body": lin_body,
        "ang_vel_body": ang_body.astype(np.float32),
        "gt_quat_wxyz": q_wxyz,
    }
