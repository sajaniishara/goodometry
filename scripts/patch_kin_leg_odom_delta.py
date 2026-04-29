"""Patch existing kin.npz files to add leg_odom_delta (N, 7).

Reads kin.npz (v_body_legs) + sensors.npz (imu gyro), computes per-frame
SE(3) delta via leg_odom_se3_deltas(), and re-saves kin.npz with the new key.
All existing keys are preserved unchanged.

Resume-safe: skips trajectories where kin.npz already has 'leg_odom_delta'
unless --force is passed.

Usage:
    python scripts/patch_kin_leg_odom_delta.py
    python scripts/patch_kin_leg_odom_delta.py --force
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

sys.path.insert(0, "/home/anyone/projects/goodometry")
from kinematics.leg_odom import leg_odom_se3_deltas

ROOT    = "/mnt/data/go2_research_dataset_v2"
LOG_DIR = "/home/anyone/projects/goodometry/logs"
DT      = 1.0 / 30.0


def patch(traj_dir: str) -> int:
    kin_path = os.path.join(traj_dir, "kin.npz")
    sen_path = os.path.join(traj_dir, "sensors.npz")

    kin = np.load(kin_path)
    sen = np.load(sen_path)

    v_body_legs = kin["v_body_legs"]          # (N, 3)
    gyro_body   = sen["imu"][:, 3:6]          # (N, 3)  gyro channels 3-5

    # align lengths (sensors may have more frames than kin due to filtering)
    N = len(v_body_legs)
    gyro_body = gyro_body[:N]

    delta = leg_odom_se3_deltas(v_body_legs, gyro_body, dt=DT)  # (N, 7)

    # rebuild dict with all existing keys + new one
    payload = {k: kin[k] for k in kin.files}
    payload["leg_odom_delta"] = delta
    np.savez(kin_path, **payload)
    return N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",  default=ROOT)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end",   type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "patch_leg_odom_delta.jsonl")

    trajs = sorted(glob.glob(os.path.join(args.root, "trajectory_*")))
    trajs = [t for t in trajs
             if os.path.isdir(t)
             and os.path.exists(os.path.join(t, "kin.npz"))
             and os.path.exists(os.path.join(t, "sensors.npz"))
             and not os.path.exists(os.path.join(t, "FALL"))]
    trajs = trajs[args.start: args.end]
    print(f"{len(trajs)} trajectories in scope", flush=True)

    ok = fail = skip = 0
    t_total = time.time()

    for i, traj_dir in enumerate(trajs):
        name = os.path.basename(traj_dir)

        if not args.force:
            kin = np.load(os.path.join(traj_dir, "kin.npz"))
            if "leg_odom_delta" in kin.files:
                skip += 1
                continue

        t0 = time.time()
        try:
            N = patch(traj_dir)
            elapsed = time.time() - t0
            rec = {"ts": time.time(), "traj": name, "status": "ok",
                   "n": N, "elapsed_s": round(elapsed, 3)}
            ok += 1
            if ok % 100 == 0 or ok == 1:
                print(f"[{i+1}/{len(trajs)}] {name}  n={N}  {elapsed:.2f}s", flush=True)
        except Exception as e:
            rec = {"ts": time.time(), "traj": name, "status": "error",
                   "error": repr(e), "tb": traceback.format_exc()}
            fail += 1
            print(f"[{i+1}/{len(trajs)}] {name} FAIL  {e!r}", flush=True)

        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    elapsed_total = time.time() - t_total
    print(f"\nDone. ok={ok}  fail={fail}  skip={skip}  total={elapsed_total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
