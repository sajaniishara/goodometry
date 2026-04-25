"""Run the winning VO method on every clean Go2 v2 trajectory.

Writes `/mnt/data/go2_research_dataset_v2/<traj>/vo.npz` with:

    frame_idx  (N,)        int32   # matches sensors.npz exactly
    pose_7d    (N, 7)      float32 # [tx,ty,tz,qx,qy,qz,qw] camera-to-world
    pose_mat   (N, 4, 4)   float32 # SE(3) matrix form
    valid      (N,)        bool    # False for frames VO dropped / failed

Skips trajectories that already have a vo.npz (resume-safe). Skips FALL
trajectories. Logs per-trajectory timings and failures to
/home/anyone/projects/goodometry/logs/vo_scale_<method>.jsonl.
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

INTRIN = (914.0, 914.0, 640.0, 360.0)
ROOT = "/mnt/data/go2_research_dataset_v2"
LOG_DIR = "/home/anyone/projects/goodometry/logs"
os.makedirs(LOG_DIR, exist_ok=True)


def run_one(method: str, traj_dir: str) -> dict:
    if method == "dpvo":
        from vo.dpvo_runner import run_dpvo
        return run_dpvo(traj_dir, INTRIN)
    elif method == "droid":
        from vo.droid_runner import run_droid
        return run_droid(traj_dir, INTRIN)
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpvo", "droid"], required=True)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--start", type=int, default=0, help="start index into sorted trajectory list")
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="overwrite existing vo.npz")
    args = ap.parse_args()

    trajs = sorted(glob.glob(os.path.join(args.root, "trajectory_*")))
    trajs = [t for t in trajs
             if os.path.isdir(t)
             and os.path.exists(os.path.join(t, "sensors.npz"))
             and not os.path.exists(os.path.join(t, "FALL"))]
    trajs = trajs[args.start: args.end]
    log_path = os.path.join(LOG_DIR, f"vo_scale_{args.method}.jsonl")
    print(f"{len(trajs)} trajectories in scope  method={args.method}  log={log_path}", flush=True)

    total_start = time.time()
    ok, fail, skip = 0, 0, 0
    for i, traj_dir in enumerate(trajs):
        out_path = os.path.join(traj_dir, "vo.npz")
        name = os.path.basename(traj_dir)
        if os.path.exists(out_path) and not args.force:
            skip += 1
            continue
        t0 = time.time()
        try:
            out = run_one(args.method, traj_dir)
            np.savez(
                out_path,
                frame_idx=out["frame_idx"],
                pose_7d=out["pose_7d"],
                pose_mat=out["pose_mat"],
                valid=out["valid"],
            )
            dt = time.time() - t0
            rec = {
                "ts": time.time(), "traj": name, "status": "ok",
                "n": int(len(out["frame_idx"])),
                "elapsed_s": dt, "fps": out.get("fps", 0.0),
            }
            ok += 1
            print(f"[{i+1}/{len(trajs)}] {name} ok  n={rec['n']}  {dt/60:.1f} min  fps={rec['fps']:.1f}", flush=True)
        except Exception as e:
            rec = {"ts": time.time(), "traj": name, "status": "error",
                   "error": repr(e), "tb": traceback.format_exc()}
            fail += 1
            print(f"[{i+1}/{len(trajs)}] {name} FAIL  {e!r}", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    print(f"\nDone. ok={ok} fail={fail} skip={skip} total_min={(time.time()-total_start)/60:.1f}",
          flush=True)


if __name__ == "__main__":
    main()
