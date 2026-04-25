"""Fusion dataset: sliding-window clips over (kinematics, INS) with body-frame labels.

Each item is a clip of T consecutive timesteps. Label is the 6-DoF body-frame
velocity at the last timestep of the clip (standard auto-regressive setup).
"""
from __future__ import annotations
import glob
import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


KINEMATICS_DIM = 31        # foot_pos 12 + foot_vel 12 + contact 4 + v_body_legs 3
INS_DIM = 10               # quat 4 + accel_world 3 + gyro_body 3
LABEL_DIM = 6


def _build_kin_features(kin: dict, start: int, end: int) -> np.ndarray:
    fp = kin["foot_pos_body"][start:end].reshape(end - start, 12)
    fv = kin["foot_vel_body"][start:end].reshape(end - start, 12)
    cp = kin["contact_prob"][start:end]                       # (T, 4)
    vb = kin["v_body_legs"][start:end]                        # (T, 3)
    return np.concatenate([fp, fv, cp, vb], axis=1).astype(np.float32)


def _build_ins_features(ins: dict, start: int, end: int) -> np.ndarray:
    q  = ins["quat_wxyz"][start:end]                          # (T, 4)
    aw = ins["accel_world"][start:end]                        # (T, 3)
    gb = ins["gyro_body"][start:end]                          # (T, 3)
    return np.concatenate([q, aw, gb], axis=1).astype(np.float32)


def split_trajectories(
    root: str,
    train_count: int = 650,
    val_count: int = 150,
    test_count: int = 208,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Stratified-by-terrain split, matching go2_research Session 17 convention.

    Excludes FALL trajectories. Groups remaining by `metadata.terrain_type` and
    splits each group proportionally, with a rounding correction to hit the
    exact requested counts.
    """
    trajs = sorted(glob.glob(os.path.join(root, "trajectory_*")))
    trajs = [t for t in trajs
             if os.path.isdir(t)
             and os.path.exists(os.path.join(t, "sensors.npz"))
             and not os.path.exists(os.path.join(t, "FALL"))]

    by_terrain: dict[str, list[str]] = {}
    for t in trajs:
        mj = os.path.join(t, "metadata.json")
        if not os.path.exists(mj):
            continue
        try:
            terrain = json.load(open(mj)).get("terrain_type", "unknown")
        except Exception:
            terrain = "unknown"
        by_terrain.setdefault(terrain, []).append(t)

    total = sum(len(v) for v in by_terrain.values())
    need = train_count + val_count + test_count
    assert need <= total, f"Requested {need} trajectories, have {total} clean"

    rng = np.random.RandomState(seed)
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for terrain, items in sorted(by_terrain.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_tr = int(round(n * train_count / total))
        n_va = int(round(n * val_count / total))
        n_te = n - n_tr - n_va
        train.extend(shuffled[:n_tr])
        val.extend(shuffled[n_tr:n_tr + n_va])
        test.extend(shuffled[n_tr + n_va:])

    # Rounding correction — trim/grow to requested counts from the largest group.
    def _fix(split: list[str], target: int):
        while len(split) > target:
            split.pop()
        while len(split) < target:
            # pull a trajectory that no split has yet
            assigned = set(train + val + test)
            for t in trajs:
                if t not in assigned:
                    split.append(t)
                    break
            else:
                break
    _fix(train, train_count)
    _fix(val, val_count)
    _fix(test, test_count)
    return train, val, test


class GoFusionDataset(Dataset):
    def __init__(
        self,
        trajectories: list[str],
        clip_len: int = 40,
        stride: int = 8,
        kin_norm: Optional[tuple[np.ndarray, np.ndarray]] = None,
        ins_norm: Optional[tuple[np.ndarray, np.ndarray]] = None,
        ins_file: str = "ins.npz",       # "ins.npz" (IMU-only) or "ins_marg.npz" (MARG stitched)
    ):
        self.clip_len = clip_len
        self.stride = stride
        self.trajs = list(trajectories)
        self.kin_norm = kin_norm
        self.ins_norm = ins_norm
        self.ins_file = ins_file

        # Precompute clip index: each entry is (traj_dir, start_frame_in_clip_array).
        self.clips: list[tuple[str, int]] = []
        self.traj_n: dict[str, int] = {}
        for t in self.trajs:
            n = len(np.load(os.path.join(t, "sensors.npz"))["frame_idx"])
            self.traj_n[t] = n
            last_start = n - clip_len
            if last_start < 0:
                continue
            for s in range(0, last_start + 1, stride):
                self.clips.append((t, s))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        traj, start = self.clips[idx]
        end = start + self.clip_len

        kin = np.load(os.path.join(traj, "kin.npz"))
        ins = np.load(os.path.join(traj, self.ins_file))
        lab = np.load(os.path.join(traj, "labels_body.npz"))

        kin_feat = _build_kin_features(kin, start, end)                  # (T, 31)
        ins_feat = _build_ins_features(ins, start, end)                  # (T, 10)
        # Label at last timestep.
        lin = lab["lin_vel_body"][end - 1]                               # (3,)
        ang = lab["ang_vel_body"][end - 1]                               # (3,)
        label = np.concatenate([lin, ang], axis=0).astype(np.float32)    # (6,)

        if self.kin_norm is not None:
            kin_feat = (kin_feat - self.kin_norm[0]) / (self.kin_norm[1] + 1e-6)
        if self.ins_norm is not None:
            ins_feat = (ins_feat - self.ins_norm[0]) / (self.ins_norm[1] + 1e-6)

        return {
            "kin": torch.from_numpy(kin_feat),    # (T, 31)
            "ins": torch.from_numpy(ins_feat),    # (T, 10)
            "label": torch.from_numpy(label),     # (6,)
        }


def compute_norm_stats(dataset_trajs: list[str], sample_frac: float = 0.2,
                        seed: int = 42, ins_file: str = "ins.npz") -> dict:
    """Compute per-channel mean/std over a random subset of training trajectories."""
    rng = np.random.RandomState(seed)
    subset = list(dataset_trajs)
    rng.shuffle(subset)
    subset = subset[:max(1, int(len(subset) * sample_frac))]

    kin_rows: list[np.ndarray] = []
    ins_rows: list[np.ndarray] = []
    for t in subset:
        kin = np.load(os.path.join(t, "kin.npz"))
        ins = np.load(os.path.join(t, ins_file))
        N = len(kin["foot_pos_body"])
        kin_rows.append(_build_kin_features(kin, 0, N))
        ins_rows.append(_build_ins_features(ins, 0, N))
    K = np.concatenate(kin_rows, axis=0)
    I = np.concatenate(ins_rows, axis=0)
    return {
        "kin_mean": K.mean(0).astype(np.float32),
        "kin_std":  K.std(0).astype(np.float32),
        "ins_mean": I.mean(0).astype(np.float32),
        "ins_std":  I.std(0).astype(np.float32),
    }
