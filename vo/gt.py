"""Ground-truth pose loader for the Go2 v2 dataset.

data.csv gives per-frame body pose in world frame; sensors.npz['frame_idx'] is
the authoritative list of valid frames the DataLoader uses. We align GT to
that list so both VO output and GT share identical indexing.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd


def load_gt(traj_dir: str) -> dict:
    """Return GT body pose (world frame) aligned to sensors.npz['frame_idx'].

    Keys in returned dict:
      frame_idx : (N,) int32
      t_sec     : (N,) float64  frame timestamp (seconds)
      pos       : (N, 3) float32  world-frame body position
      quat_wxyz : (N, 4) float32  world-frame body orientation
      vel       : (N, 3) float32  world-frame linear velocity (from data.csv)
    """
    sens = np.load(os.path.join(traj_dir, "sensors.npz"))
    frame_idx = sens["frame_idx"].astype(np.int64)
    df = pd.read_csv(os.path.join(traj_dir, "data.csv"))
    df = df.set_index("frame_idx").loc[frame_idx].reset_index()
    return {
        "frame_idx": frame_idx.astype(np.int32),
        "t_sec": df["timestamp"].to_numpy(dtype=np.float64),
        "pos": df[["gt_pos_x", "gt_pos_y", "gt_pos_z"]].to_numpy(dtype=np.float32),
        "quat_wxyz": df[["gt_ori_w", "gt_ori_x", "gt_ori_y", "gt_ori_z"]].to_numpy(dtype=np.float32),
        "vel": df[["gt_vel_x", "gt_vel_y", "gt_vel_z"]].to_numpy(dtype=np.float32),
    }
