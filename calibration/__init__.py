"""FR_calf calibration for the Go2 v2 dataset.

The Isaac Sim Go2 USD asset produces a systematic asymmetry on the FR calf
joint (~0.7 rad more bent than commanded, vs 0.05–0.17 rad tracking error on
the other three calves). Verified by correlated body roll/pitch (see
CHANGES.md Session 20). This module provides a non-destructive correction
— `sensors.npz` is left untouched; `calibration.npz` sits alongside it and
is applied at load time.

Channel layout (confirmed in CHANGES.md Session 20):
    joints[:, 0..3]   = hip positions     [FL, FR, RL, RR]
    joints[:, 4..7]   = thigh positions   [FL, FR, RL, RR]
    joints[:, 8..11]  = calf positions    [FL, FR, RL, RR]
    joints[:, 12..15] = hip velocities    [FL, FR, RL, RR]
    joints[:, 16..19] = thigh velocities  [FL, FR, RL, RR]
    joints[:, 20..23] = calf velocities   [FL, FR, RL, RR]

Only channel 9 (FR_calf_pos) is adjusted; velocities are the time-derivative
of the actual physical angle and therefore do not need a static offset.
"""
from __future__ import annotations
import os
import numpy as np


FR_CALF_POS_IDX = 9


def load_calibrated_joints(traj_dir: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Load joints from sensors.npz with FR_calf offset applied if available.

    Returns (joints, frame_idx, applied_offset). applied_offset is 0.0 if no
    calibration.npz exists for this trajectory.
    """
    s = np.load(os.path.join(traj_dir, "sensors.npz"))
    joints = s["joints"].astype(np.float32).copy()
    frame_idx = s["frame_idx"].astype(np.int32)

    cal_path = os.path.join(traj_dir, "calibration.npz")
    if os.path.exists(cal_path):
        cal = np.load(cal_path)
        offset = float(cal["fr_calf_offset"])
        joints[:, FR_CALF_POS_IDX] += offset
        return joints, frame_idx, offset
    return joints, frame_idx, 0.0
