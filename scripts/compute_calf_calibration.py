"""Compute and save a per-trajectory FR_calf calibration offset.

Writes `/mnt/data/go2_research_dataset_v2/<traj>/calibration.npz` containing:

    fr_calf_offset    : float32   offset to ADD to joints[:, 9]
    fr_calf_raw_mean  : float32   observed FR_calf mean (for diagnostics)
    other_calf_mean   : float32   target = mean of FL/RL/RR calf means
    other_calf_fl     : float32   FL calf mean
    other_calf_rl     : float32   RL calf mean
    other_calf_rr     : float32   RR calf mean
    n_frames          : int32     trajectory length

Non-destructive: `sensors.npz` is not modified. Use
`goodometry.calibration.load_calibrated_joints` to consume corrected values.

Skips trajectories that already have calibration.npz unless `--force`.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import numpy as np

ROOT = "/mnt/data/go2_research_dataset_v2"
CONFIGS_DIR = "/home/anyone/projects/goodometry/configs"


def compute_offset(joints: np.ndarray) -> dict:
    fl = float(joints[:, 8].mean())
    fr = float(joints[:, 9].mean())
    rl = float(joints[:, 10].mean())
    rr = float(joints[:, 11].mean())
    target = (fl + rl + rr) / 3.0
    return {
        "fr_calf_offset": np.float32(target - fr),
        "fr_calf_raw_mean": np.float32(fr),
        "other_calf_mean": np.float32(target),
        "other_calf_fl": np.float32(fl),
        "other_calf_rl": np.float32(rl),
        "other_calf_rr": np.float32(rr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing calibration.npz files")
    ap.add_argument("--dry-run", action="store_true", help="compute but don't write")
    args = ap.parse_args()

    trajs = sorted(glob.glob(os.path.join(args.root, "trajectory_*")))
    trajs = [t for t in trajs
             if os.path.isdir(t)
             and os.path.exists(os.path.join(t, "sensors.npz"))
             and not os.path.exists(os.path.join(t, "FALL"))]
    print(f"Found {len(trajs)} clean trajectories")

    offsets: dict[str, float] = {}
    skipped = 0
    written = 0
    for i, traj_dir in enumerate(trajs):
        name = os.path.basename(traj_dir)
        cal_path = os.path.join(traj_dir, "calibration.npz")
        if os.path.exists(cal_path) and not args.force:
            offsets[name] = float(np.load(cal_path)["fr_calf_offset"])
            skipped += 1
            continue
        joints = np.load(os.path.join(traj_dir, "sensors.npz"))["joints"]
        rec = compute_offset(joints)
        rec["n_frames"] = np.int32(joints.shape[0])
        if not args.dry_run:
            np.savez(cal_path, **rec)
        offsets[name] = float(rec["fr_calf_offset"])
        written += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(trajs)}] last offset={offsets[name]:+.4f}")

    vals = np.array(list(offsets.values()))
    print(f"\nProcessed {len(vals)} trajectories  (wrote {written}, skipped {skipped})")
    print(f"FR_calf offset distribution (rad):")
    print(f"  mean    {vals.mean():+.4f}")
    print(f"  median  {np.median(vals):+.4f}")
    print(f"  std     {vals.std():.4f}")
    print(f"  min     {vals.min():+.4f}")
    print(f"  max     {vals.max():+.4f}")
    print(f"  p5/p95  {np.percentile(vals, 5):+.4f} / {np.percentile(vals, 95):+.4f}")

    summary = {
        "method": "per_trajectory_static_mean",
        "rationale": "offset = mean(FL, RL, RR calf means) - mean(FR calf). Added to joints[:,9] at load time.",
        "channel_corrected": "joints[:, 9]  (FR_calf_pos)",
        "channels_untouched": "velocities (ch20..23), other legs",
        "n_trajectories": int(len(vals)),
        "stats": {
            "mean_offset_rad": float(vals.mean()),
            "median_offset_rad": float(np.median(vals)),
            "std_offset_rad": float(vals.std()),
            "min_offset_rad": float(vals.min()),
            "max_offset_rad": float(vals.max()),
            "p5_offset_rad": float(np.percentile(vals, 5)),
            "p95_offset_rad": float(np.percentile(vals, 95)),
        },
        "per_trajectory_offsets": offsets,
    }
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    out = os.path.join(CONFIGS_DIR, "fr_calf_offsets.json")
    if not args.dry_run:
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to {out}")
    else:
        print(f"\n(dry-run; would have written {out})")


if __name__ == "__main__":
    main()
