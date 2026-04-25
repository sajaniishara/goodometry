"""Write per-trajectory `labels_body.npz` next to sensors.npz.

One-time CPU preprocessing. Keeps DataLoader fast (no CSV parsing per clip).
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import time
import numpy as np

sys.path.insert(0, "/home/anyone/projects/goodometry")
from labels import compute_body_labels


ROOT = "/mnt/data/go2_research_dataset_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-angular-vel", type=float, default=10.0)
    args = ap.parse_args()

    trajs = sorted(glob.glob(os.path.join(args.root, "trajectory_*")))
    trajs = [t for t in trajs
             if os.path.isdir(t)
             and os.path.exists(os.path.join(t, "sensors.npz"))
             and not os.path.exists(os.path.join(t, "FALL"))]

    ok = skip = 0
    t0 = time.time()
    for i, traj_dir in enumerate(trajs):
        out = os.path.join(traj_dir, "labels_body.npz")
        if os.path.exists(out) and not args.force:
            skip += 1
            continue
        rec = compute_body_labels(traj_dir, max_angular_vel_rad=args.max_angular_vel)
        np.savez(out, **rec)
        ok += 1
        if ok == 1 or ok % 100 == 0:
            print(f"[{i+1}/{len(trajs)}] wrote {os.path.basename(traj_dir)}/labels_body.npz", flush=True)
    print(f"\nDone. wrote={ok} skipped={skip} total_s={time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
