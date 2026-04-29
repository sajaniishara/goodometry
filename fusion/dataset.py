"""Fusion dataset: sliding-window clips over (kinematics, INS[, VO]) with body-frame labels.

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


KINEMATICS_DIM    = 31     # foot_pos 12 + foot_vel 12 + contact 4 + v_body_legs 3
KINEMATICS_V3_DIM = 35     # foot_pos 12 + foot_vel 12 + contact 4 + leg_odom_delta 7
INS_DIM = 10               # quat 4 + accel_world 3 + gyro_body 3
VO_DIM = 7                 # per-frame SE(3) delta in prev-camera frame: [dt_x, dt_y, dt_z, dq_x, dq_y, dq_z, dq_w]
LABEL_DIM = 6


def _build_kin_features(kin: dict, start: int, end: int) -> np.ndarray:
    """31-D kinematics feature (fusion_v1 / fusion_v2 — unchanged)."""
    fp = kin["foot_pos_body"][start:end].reshape(end - start, 12)
    fv = kin["foot_vel_body"][start:end].reshape(end - start, 12)
    cp = kin["contact_prob"][start:end]                       # (T, 4)
    vb = kin["v_body_legs"][start:end]                        # (T, 3)
    return np.concatenate([fp, fv, cp, vb], axis=1).astype(np.float32)


def _build_kin_v3_features(kin: dict, start: int, end: int) -> np.ndarray:
    """35-D kinematics feature (fusion_v3): replaces v_body_legs(3) with
    leg_odom_delta(7) — SE(3) per-frame delta [Δtx,Δty,Δtz,Δqx,Δqy,Δqz,Δqw].
    foot_pos/vel and contact_prob are identical to the v2 branch."""
    fp  = kin["foot_pos_body"][start:end].reshape(end - start, 12)
    fv  = kin["foot_vel_body"][start:end].reshape(end - start, 12)
    cp  = kin["contact_prob"][start:end]                      # (T, 4)
    lod = kin["leg_odom_delta"][start:end]                    # (T, 7)
    return np.concatenate([fp, fv, cp, lod], axis=1).astype(np.float32)


def _build_ins_features(ins: dict, start: int, end: int) -> np.ndarray:
    q  = ins["quat_wxyz"][start:end]                          # (T, 4)
    aw = ins["accel_world"][start:end]                        # (T, 3)
    gb = ins["gyro_body"][start:end]                          # (T, 3)
    return np.concatenate([q, aw, gb], axis=1).astype(np.float32)


def _build_vo_features(vo: dict, start: int, end: int) -> np.ndarray:
    """(T, 7) per-frame SE(3) delta expressed in the previous camera frame.

    pose_7d is [tx, ty, tz, qx, qy, qz, qw] camera-to-world (DROID convention).
    Delta at timestep i:
      dt    = R_prev^T @ (t_curr - t_prev)   — translation in prev-cam coords
      dq    = q_prev^{-1} ⊗ q_curr           — rotation delta (xyzw)
    For i=0 (no previous frame) the feature is the identity delta [0,0,0, 0,0,0,1].
    Invalid frames (valid=False) also get the identity delta.
    """
    T = end - start
    pose  = vo["pose_7d"].astype(np.float32)   # (N, 7)
    valid = vo["valid"]                         # (N,) bool

    feats = np.zeros((T, 7), dtype=np.float32)
    feats[:, 6] = 1.0  # identity quaternion w component

    for i in range(T):
        ci = start + i
        pi = ci - 1
        if pi < 0 or not valid[ci] or not valid[pi]:
            continue

        t_c, t_p = pose[ci, :3], pose[pi, :3]
        qx, qy, qz, qw = pose[pi, 3], pose[pi, 4], pose[pi, 5], pose[pi, 6]

        # R_prev (camera-to-world rotation at prev frame)
        R = np.array([
            [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ], dtype=np.float32)
        dt = R.T @ (t_c - t_p)

        # q_prev^{-1} ⊗ q_curr  (conjugate of unit quat = inverse)
        px, py, pz, pw = -qx, -qy, -qz, qw   # conjugate
        cx, cy, cz, cw = pose[ci, 3], pose[ci, 4], pose[ci, 5], pose[ci, 6]
        dq = np.array([
            pw*cx + px*cw + py*cz - pz*cy,
            pw*cy - px*cz + py*cw + pz*cx,
            pw*cz + px*cy - py*cx + pz*cw,
            pw*cw - px*cx - py*cy - pz*cz,
        ], dtype=np.float32)

        feats[i] = np.concatenate([dt, dq])

    return feats


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
        vo_norm:  Optional[tuple[np.ndarray, np.ndarray]] = None,
        ins_file: str = "ins.npz",    # "ins.npz" (IMU-only) or "ins_marg.npz" (MARG)
        vo_file:  str = "vo.npz",
        use_vo:   bool = False,
        kin_v3:   bool = False,       # use 35-D kin features (leg_odom_delta) for fusion_v3
    ):
        self.clip_len = clip_len
        self.stride = stride
        self.trajs = list(trajectories)
        self.kin_norm = kin_norm
        self.ins_norm = ins_norm
        self.vo_norm  = vo_norm
        self.ins_file = ins_file
        self.vo_file  = vo_file
        self.use_vo   = use_vo

        # Precompute clip index: each entry is (traj_dir, start_frame_in_clip_array).
        self.kin_v3   = kin_v3
        self.clips: list[tuple[str, int]] = []
        self.traj_n: dict[str, int] = {}
        for t in self.trajs:
            if use_vo and not os.path.exists(os.path.join(t, vo_file)):
                continue
            if kin_v3 and not np.load(os.path.join(t, "kin.npz"), mmap_mode="r").__contains__("leg_odom_delta"):
                continue
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

        kin_feat = (_build_kin_v3_features(kin, start, end) if self.kin_v3
                    else _build_kin_features(kin, start, end))            # (T, 35) or (T, 31)
        ins_feat = _build_ins_features(ins, start, end)                  # (T, 10)
        # Label at last timestep.
        lin = lab["lin_vel_body"][end - 1]                               # (3,)
        ang = lab["ang_vel_body"][end - 1]                               # (3,)
        label = np.concatenate([lin, ang], axis=0).astype(np.float32)    # (6,)

        if self.kin_norm is not None:
            kin_feat = (kin_feat - self.kin_norm[0]) / (self.kin_norm[1] + 1e-6)
        if self.ins_norm is not None:
            ins_feat = (ins_feat - self.ins_norm[0]) / (self.ins_norm[1] + 1e-6)

        item = {
            "kin":   torch.from_numpy(kin_feat),   # (T, 31)
            "ins":   torch.from_numpy(ins_feat),   # (T, 10)
            "label": torch.from_numpy(label),      # (6,)
        }

        if self.use_vo:
            vo = np.load(os.path.join(traj, self.vo_file))
            vo_feat = _build_vo_features(vo, start, end)                 # (T, 7)
            if self.vo_norm is not None:
                vo_feat = (vo_feat - self.vo_norm[0]) / (self.vo_norm[1] + 1e-6)
            item["vo"] = torch.from_numpy(vo_feat)                       # (T, 7)

        return item


def compute_norm_stats(dataset_trajs: list[str], sample_frac: float = 0.2,
                        seed: int = 42, ins_file: str = "ins.npz",
                        use_vo: bool = False, vo_file: str = "vo.npz",
                        kin_v3: bool = False) -> dict:
    """Compute per-channel mean/std over a random subset of training trajectories."""
    rng = np.random.RandomState(seed)
    subset = list(dataset_trajs)
    rng.shuffle(subset)
    subset = subset[:max(1, int(len(subset) * sample_frac))]

    kin_rows: list[np.ndarray] = []
    ins_rows: list[np.ndarray] = []
    vo_rows:  list[np.ndarray] = []
    for t in subset:
        if use_vo and not os.path.exists(os.path.join(t, vo_file)):
            continue
        kin = np.load(os.path.join(t, "kin.npz"))
        ins = np.load(os.path.join(t, ins_file))
        N = len(kin["foot_pos_body"])
        kin_rows.append(_build_kin_v3_features(kin, 0, N) if kin_v3
                        else _build_kin_features(kin, 0, N))
        ins_rows.append(_build_ins_features(ins, 0, N))
        if use_vo:
            vo = np.load(os.path.join(t, vo_file))
            vo_rows.append(_build_vo_features(vo, 0, N))

    K = np.concatenate(kin_rows, axis=0)
    I = np.concatenate(ins_rows, axis=0)
    stats = {
        "kin_mean": K.mean(0).astype(np.float32),
        "kin_std":  K.std(0).astype(np.float32),
        "ins_mean": I.mean(0).astype(np.float32),
        "ins_std":  I.std(0).astype(np.float32),
    }
    if use_vo and vo_rows:
        V = np.concatenate(vo_rows, axis=0)
        stats["vo_mean"] = V.mean(0).astype(np.float32)
        stats["vo_std"]  = V.std(0).astype(np.float32)
    return stats
