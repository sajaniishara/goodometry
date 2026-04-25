"""Run kinematic preprocessing on every clean Go2 v2 trajectory (CPU).

Per trajectory writes `/mnt/data/go2_research_dataset_v2/<traj>/kin.npz`:

    frame_idx      (N,)         int32    # matches sensors.npz['frame_idx']
    foot_pos_body  (N, 4, 3)    float32  # FK foot pos in body frame, [FL, FR, RL, RR]
    foot_vel_body  (N, 4, 3)    float32  # finite-diff on foot_pos_body
    contact_prob   (N, 4)       float32  # velocity-gate contact probability
    v_body_legs    (N, 3)       float32  # body-frame linear velocity estimate

Reads the FR_calf calibration offset from `calibration.npz` via
`goodometry.calibration.load_calibrated_joints`. CPU-only: single-process per
trajectory (the FK is already internally batched across timesteps).

Resume-safe: existing kin.npz files are skipped unless `--force`.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import time
import traceback
import numpy as np
import torch

sys.path.insert(0, "/home/anyone/projects/goodometry")
from calibration import load_calibrated_joints
from kinematics.fk import Go2FK, foot_vel_body_from_finite_diff
from kinematics.contact import velocity_gate_contact
from kinematics.leg_odom import body_velocity_from_legs

ROOT = "/mnt/data/go2_research_dataset_v2"
LOG_DIR = "/home/anyone/projects/goodometry/logs"
DT = 1.0 / 30.0


def process(fk: Go2FK, traj_dir: str) -> dict:
    joints, frame_idx, offset = load_calibrated_joints(traj_dir)
    imu = np.load(os.path.join(traj_dir, "sensors.npz"))["imu"]
    with torch.no_grad():
        q = torch.from_numpy(joints[:, :12]).float()
        foot_pos = fk.foot_positions_body(q)                      # (N, 4, 3)
    foot_vel = foot_vel_body_from_finite_diff(foot_pos, DT)       # (N, 4, 3)
    contact = velocity_gate_contact(foot_vel)                     # (N, 4)
    omega = torch.from_numpy(imu[:, 3:6]).float()                 # gyro (body frame)
    v_legs = body_velocity_from_legs(foot_pos, foot_vel, omega, contact)

    return {
        "frame_idx":     frame_idx.astype(np.int32),
        "foot_pos_body": foot_pos.cpu().numpy().astype(np.float32),
        "foot_vel_body": foot_vel.cpu().numpy().astype(np.float32),
        "contact_prob":  contact.cpu().numpy().astype(np.float32),
        "v_body_legs":   v_legs.cpu().numpy().astype(np.float32),
        "applied_fr_calf_offset": np.float32(offset),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "kin_scale.jsonl")

    trajs = sorted(glob.glob(os.path.join(args.root, "trajectory_*")))
    trajs = [t for t in trajs
             if os.path.isdir(t)
             and os.path.exists(os.path.join(t, "sensors.npz"))
             and not os.path.exists(os.path.join(t, "FALL"))]
    trajs = trajs[args.start: args.end]
    print(f"{len(trajs)} trajectories in scope  log={log_path}", flush=True)

    # Load FK once; chain is shared across all trajectories.
    fk = Go2FK(device="cpu")

    ok = fail = skip = 0
    total_t0 = time.time()
    for i, traj_dir in enumerate(trajs):
        name = os.path.basename(traj_dir)
        out_path = os.path.join(traj_dir, "kin.npz")
        if os.path.exists(out_path) and not args.force:
            skip += 1
            continue
        t0 = time.time()
        try:
            payload = process(fk, traj_dir)
            np.savez(out_path, **payload)
            dt = time.time() - t0
            rec = {"ts": time.time(), "traj": name, "status": "ok",
                   "n": int(len(payload["frame_idx"])), "elapsed_s": dt,
                   "applied_offset": float(payload["applied_fr_calf_offset"])}
            ok += 1
            if ok % 50 == 0 or ok == 1:
                print(f"[{i+1}/{len(trajs)}] {name} ok  n={rec['n']}  {dt:.2f}s  offset={rec['applied_offset']:+.3f}",
                      flush=True)
        except Exception as e:
            rec = {"ts": time.time(), "traj": name, "status": "error",
                   "error": repr(e), "tb": traceback.format_exc()}
            fail += 1
            print(f"[{i+1}/{len(trajs)}] {name} FAIL  {e!r}", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    print(f"\nDone. ok={ok} fail={fail} skip={skip} total_s={time.time()-total_t0:.1f}",
          flush=True)


if __name__ == "__main__":
    main()
