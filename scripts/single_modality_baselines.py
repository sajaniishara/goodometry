"""Single-modality baselines: how well does each preprocessing arm predict
body-frame velocity *on its own*, with no learned fusion?

Computed on the exact same 208-trajectory test set as fusion_v2 (650/150/208,
seed=42), and on the same clip indexing (clip_len=40, stride=8). Each clip's
prediction is taken at the last timestep, matching the fusion model's
auto-regressive setup.

Baselines:
  kin-only :  lin_pred = v_body_legs[end-1]  (body-frame m/s, native)
              ang_pred = N/A
  ins-only :  ang_pred = gyro_body[end-1]    (body-frame rad/s, native)
              lin_pred = N/A (integrating accel diverges)
  vo-only  :  per-frame Δt × 30 Hz from pose_7d in previous-camera frame.
              Per-axis numbers treat camera ≈ body (Go2 head cam, small
              misalignment). Speed magnitude is frame-agnostic.

Output: prints a table and writes runs/single_modality_baselines/results.json.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/anyone/projects/goodometry")
from fusion.dataset import split_trajectories


ROOT = "/mnt/data/go2_research_dataset_v2"
CLIP_LEN = 40
STRIDE = 8
DT = 1.0 / 30.0   # 30 Hz


def quat_to_rotvec(q):
    """xyzw → rotation vector (axis * angle). q: (..., 4)."""
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    qw = np.clip(qw, -1.0, 1.0)
    angle = 2.0 * np.arccos(qw)
    s = np.sqrt(np.maximum(1.0 - qw * qw, 0.0))
    axis = np.zeros_like(q[..., :3])
    big = s > 1e-8
    axis[big, 0] = qx[big] / s[big]
    axis[big, 1] = qy[big] / s[big]
    axis[big, 2] = qz[big] / s[big]
    return axis * angle[..., None]


def build_vo_velocity(pose_7d, valid):
    """Per-frame body-velocity estimate from absolute camera-to-world poses.

    Returns (lin_vel, ang_vel) in approximate body frame, scaled to m/s and
    rad/s. Caveat: assumes camera frame ≈ body frame (Go2 head-cam is
    forward-facing; misalignment is small but not zero).
    """
    N = len(pose_7d)
    lin = np.zeros((N, 3), dtype=np.float32)
    ang = np.zeros((N, 3), dtype=np.float32)
    pose = pose_7d.astype(np.float32)
    for i in range(1, N):
        if not (valid[i] and valid[i - 1]):
            continue
        t_c, t_p = pose[i, :3], pose[i - 1, :3]
        qx, qy, qz, qw = pose[i - 1, 3], pose[i - 1, 4], pose[i - 1, 5], pose[i - 1, 6]
        # R_prev (cam→world)
        R = np.array([
            [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ], dtype=np.float32)
        dt_prev = R.T @ (t_c - t_p)
        lin[i] = dt_prev / DT

        # q_prev^{-1} ⊗ q_curr → rotation vector / dt = angular velocity
        px, py, pz, pw = -qx, -qy, -qz, qw
        cx, cy, cz, cw = pose[i, 3], pose[i, 4], pose[i, 5], pose[i, 6]
        dq = np.array([
            pw*cx + px*cw + py*cz - pz*cy,
            pw*cy - px*cz + py*cw + pz*cx,
            pw*cz + px*cy - py*cx + pz*cw,
            pw*cw - px*cx - py*cy - pz*cz,
        ], dtype=np.float32)
        rv = quat_to_rotvec(dq[None, :])[0]
        ang[i] = rv / DT
    return lin, ang


def compute_clip_predictions(test_trajs):
    """Iterate the same 92,935 clips fusion_v2 sees; collect predictions and
    GT at the *last* timestep of each clip.

    Returns dict of arrays, each (N_clips, 3).
    """
    kin_lin, ins_ang = [], []
    vo_lin, vo_ang = [], []
    gt_lin, gt_ang = [], []
    vo_valid = []   # mask for clips where VO is valid at the end frame

    for traj in sorted(test_trajs):
        kin = np.load(os.path.join(traj, "kin.npz"))
        ins = np.load(os.path.join(traj, "ins.npz"))
        lab = np.load(os.path.join(traj, "labels_body.npz"))
        vo  = np.load(os.path.join(traj, "vo.npz")) if os.path.exists(os.path.join(traj, "vo.npz")) else None

        n = len(kin["v_body_legs"])
        last_start = n - CLIP_LEN
        if last_start < 0:
            continue

        if vo is not None:
            vo_lin_full, vo_ang_full = build_vo_velocity(vo["pose_7d"], vo["valid"])

        for s in range(0, last_start + 1, STRIDE):
            end = s + CLIP_LEN - 1   # last-timestep index, matches fusion's read-out

            kin_lin.append(kin["v_body_legs"][end])
            ins_ang.append(ins["gyro_body"][end])
            gt_lin.append(lab["lin_vel_body"][end])
            gt_ang.append(lab["ang_vel_body"][end])

            if vo is not None and vo["valid"][end] and vo["valid"][end - 1]:
                vo_lin.append(vo_lin_full[end])
                vo_ang.append(vo_ang_full[end])
                vo_valid.append(True)
            else:
                vo_lin.append(np.zeros(3, dtype=np.float32))
                vo_ang.append(np.zeros(3, dtype=np.float32))
                vo_valid.append(False)

    return {
        "kin_lin": np.stack(kin_lin),
        "ins_ang": np.stack(ins_ang),
        "vo_lin":  np.stack(vo_lin),
        "vo_ang":  np.stack(vo_ang),
        "gt_lin":  np.stack(gt_lin),
        "gt_ang":  np.stack(gt_ang),
        "vo_valid": np.array(vo_valid, dtype=bool),
    }


def rmse(err):
    return float(np.sqrt((err ** 2).mean()))


def per_axis_rmse(err):
    return np.sqrt((err ** 2).mean(axis=0)).tolist()


def main():
    train_trajs, val_trajs, test_trajs = split_trajectories(ROOT, 650, 150, 208, seed=42)
    print(f"test trajectories: {len(test_trajs)}")

    P = compute_clip_predictions(test_trajs)
    n_clips = len(P["gt_lin"])
    n_vo    = int(P["vo_valid"].sum())
    print(f"clips: {n_clips:,}  (VO valid: {n_vo:,})")

    out = {"n_clips": n_clips, "n_vo_valid": n_vo, "baselines": {}}

    # ---------------------------------------------------------------------
    # kin-only — lin from v_body_legs
    # ---------------------------------------------------------------------
    err_lin = P["kin_lin"] - P["gt_lin"]
    out["baselines"]["kin_only"] = {
        "lin_rmse":     rmse(err_lin),
        "lin_per_axis": per_axis_rmse(err_lin),
        "ang_rmse":     None,
        "ang_per_axis": None,
        "speed_rmse":   rmse(np.linalg.norm(P["kin_lin"], axis=1) - np.linalg.norm(P["gt_lin"], axis=1)),
        "notes": "v_body_legs from stance-foot constraint. Body frame, native.",
    }

    # ---------------------------------------------------------------------
    # ins-only — ang from gyro_body
    # ---------------------------------------------------------------------
    err_ang = P["ins_ang"] - P["gt_ang"]
    out["baselines"]["ins_only"] = {
        "lin_rmse":     None,
        "lin_per_axis": None,
        "ang_rmse":     rmse(err_ang),
        "ang_per_axis": per_axis_rmse(err_ang),
        "speed_rmse":   rmse(np.linalg.norm(P["ins_ang"], axis=1) - np.linalg.norm(P["gt_ang"], axis=1)),
        "notes": "gyro_body raw. Body frame, native. Linear from accel diverges.",
    }

    # ---------------------------------------------------------------------
    # vo-only — Δt × 30 Hz from camera poses; per-axis treats cam ≈ body
    # ---------------------------------------------------------------------
    valid = P["vo_valid"]
    vo_lin_v = P["vo_lin"][valid]
    vo_ang_v = P["vo_ang"][valid]
    gt_lin_v = P["gt_lin"][valid]
    gt_ang_v = P["gt_ang"][valid]
    err_lin_v = vo_lin_v - gt_lin_v
    err_ang_v = vo_ang_v - gt_ang_v
    spd_pred = np.linalg.norm(vo_lin_v, axis=1)
    spd_gt   = np.linalg.norm(gt_lin_v, axis=1)
    out["baselines"]["vo_only"] = {
        "lin_rmse":           rmse(err_lin_v),
        "lin_per_axis":       per_axis_rmse(err_lin_v),
        "ang_rmse":           rmse(err_ang_v),
        "ang_per_axis":       per_axis_rmse(err_ang_v),
        "speed_rmse":         rmse(spd_pred - spd_gt),
        "n_clips_evaluated":  int(valid.sum()),
        "notes": ("Δt × 30 Hz from DROID-SLAM camera-to-world poses. "
                  "Per-axis numbers assume camera frame ≈ body frame "
                  "(Go2 head-cam, small misalignment). Speed magnitude is "
                  "frame-agnostic and the most directly interpretable."),
    }

    # ---------------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------------
    print()
    print("=" * 72)
    print(f"Single-modality baselines on the {n_clips:,}-clip test set")
    print("=" * 72)

    print(f"\n--- kin-only ({P['kin_lin'].shape[0]:,} clips) ---")
    b = out["baselines"]["kin_only"]
    print(f"  Linear RMSE:  {b['lin_rmse']:.4f} m/s")
    print(f"    per axis (vx, vy, vz): "
          f"{b['lin_per_axis'][0]:.4f}, {b['lin_per_axis'][1]:.4f}, {b['lin_per_axis'][2]:.4f}")
    print(f"  Speed-magnitude RMSE: {b['speed_rmse']:.4f} m/s")
    print(f"  Angular: not estimated by kin")

    print(f"\n--- ins-only (gyro, {P['ins_ang'].shape[0]:,} clips) ---")
    b = out["baselines"]["ins_only"]
    print(f"  Angular RMSE: {b['ang_rmse']:.4f} rad/s")
    print(f"    per axis (ωx, ωy, ωz): "
          f"{b['ang_per_axis'][0]:.4f}, {b['ang_per_axis'][1]:.4f}, {b['ang_per_axis'][2]:.4f}")
    print(f"  Linear: not estimated by IMU alone (accel integration diverges)")

    print(f"\n--- vo-only ({n_vo:,} clips with valid VO) ---")
    b = out["baselines"]["vo_only"]
    print(f"  Linear RMSE:           {b['lin_rmse']:.4f} m/s   (cam ≈ body assumption)")
    print(f"    per axis (vx, vy, vz): "
          f"{b['lin_per_axis'][0]:.4f}, {b['lin_per_axis'][1]:.4f}, {b['lin_per_axis'][2]:.4f}")
    print(f"  Angular RMSE:          {b['ang_rmse']:.4f} rad/s (cam ≈ body assumption)")
    print(f"    per axis (ωx, ωy, ωz): "
          f"{b['ang_per_axis'][0]:.4f}, {b['ang_per_axis'][1]:.4f}, {b['ang_per_axis'][2]:.4f}")
    print(f"  Speed-magnitude RMSE:  {b['speed_rmse']:.4f} m/s   (frame-agnostic)")

    # Compare to fusion_v2
    print()
    print("=" * 72)
    print("Comparison vs fusion_v2 (the learned model that combines all three)")
    print("=" * 72)
    fv2 = json.load(open("runs/fusion_v2/test_results.json"))
    print(f"  fusion_v2 overall RMSE: {fv2['rmse_overall']:.4f}")
    print(f"  fusion_v2 linear RMSE:  {fv2['rmse_lin']:.4f} m/s")
    print(f"  fusion_v2 angular RMSE: {fv2['rmse_ang']:.4f} rad/s")
    print()
    print("How much does fusion improve over each best single-modality baseline?")
    kin_lin = out["baselines"]["kin_only"]["lin_rmse"]
    ins_ang = out["baselines"]["ins_only"]["ang_rmse"]
    vo_spd  = out["baselines"]["vo_only"]["speed_rmse"]
    print(f"  Linear:  fusion {fv2['rmse_lin']:.4f} vs kin-only {kin_lin:.4f} m/s "
          f"({100*(kin_lin - fv2['rmse_lin'])/kin_lin:+.1f}% improvement)")
    print(f"  Angular: fusion {fv2['rmse_ang']:.4f} vs ins-only {ins_ang:.4f} rad/s "
          f"({100*(ins_ang - fv2['rmse_ang'])/ins_ang:+.1f}% improvement)")
    print(f"  Speed:   fusion (computed below) vs vo-only {vo_spd:.4f} m/s")

    out_dir = Path("runs/single_modality_baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_dir/'results.json'}")


if __name__ == "__main__":
    main()
