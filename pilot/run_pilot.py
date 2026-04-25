"""Pilot: run DPVO + DROID-SLAM on 5 diverse trajectories, score vs GT."""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, "/home/anyone/projects/goodometry")
from vo.dpvo_runner import run_dpvo
from vo.droid_runner import run_droid
from vo.gt import load_gt
from vo.eval import ate, rpe, quat_wxyz_to_R

ROOT = "/mnt/data/go2_research_dataset_v2"
OUT = "/home/anyone/projects/goodometry/pilot"
INTRIN = (914.0, 914.0, 640.0, 360.0)

PILOT = [
    "trajectory_000717_20260325_083252",  # flat / forward
    "trajectory_000697_20260325_050838",  # forest / forward
    "trajectory_000206_20260321_202202",  # forest / circular
    "trajectory_000110_20260321_042724",  # forest / zigzag
    "trajectory_000871_20260326_073105",  # uneven / forward
]


def score(name: str, out: dict, gt: dict) -> dict:
    N = min(len(out["frame_idx"]), len(gt["frame_idx"]))
    pred_pos = out["pose_mat"][:N, :3, 3]
    pred_R = out["pose_mat"][:N, :3, :3]
    gt_pos = gt["pos"][:N]
    gt_R = quat_wxyz_to_R(gt["quat_wxyz"][:N])
    result = {
        "name": name,
        "n_frames": N,
        "elapsed_s": out["elapsed_s"],
        "fps": out["fps"],
    }
    result.update({f"sim3_{k}": v for k, v in ate(pred_pos, gt_pos, with_scale=True).items()})
    result.update({f"se3_{k}": v for k, v in ate(pred_pos, gt_pos, with_scale=False).items()})
    result.update(rpe(pred_pos, pred_R, gt_pos, gt_R, delta_frames=30))   # 1 s at 30 FPS
    result.update(rpe(pred_pos, pred_R, gt_pos, gt_R, delta_frames=3))    # 0.1 s (per-step)
    return result


def main():
    all_rows = []
    for traj in PILOT:
        traj_dir = os.path.join(ROOT, traj)
        gt = load_gt(traj_dir)

        # DPVO (mono, left image, full 1280x720)
        t0 = time.time()
        try:
            out_dpvo = run_dpvo(traj_dir, INTRIN)
            r = score("dpvo", out_dpvo, gt)
        except Exception as e:
            r = {"name": "dpvo", "error": repr(e), "n_frames": 0}
        r["trajectory"] = traj
        r["wall_s"] = time.time() - t0
        all_rows.append(r)
        print(json.dumps(r, indent=0))
        # save per-trajectory VO output
        np.savez(
            os.path.join(OUT, f"{traj}_dpvo.npz"),
            frame_idx=out_dpvo["frame_idx"],
            pose_7d=out_dpvo["pose_7d"],
            pose_mat=out_dpvo["pose_mat"],
            valid=out_dpvo["valid"],
        )

        # DROID (stereo)
        t0 = time.time()
        try:
            out_droid = run_droid(traj_dir, INTRIN)
            r = score("droid", out_droid, gt)
        except Exception as e:
            r = {"name": "droid", "error": repr(e), "n_frames": 0}
        r["trajectory"] = traj
        r["wall_s"] = time.time() - t0
        all_rows.append(r)
        print(json.dumps(r, indent=0))
        np.savez(
            os.path.join(OUT, f"{traj}_droid.npz"),
            frame_idx=out_droid["frame_idx"],
            pose_7d=out_droid["pose_7d"],
            pose_mat=out_droid["pose_mat"],
            valid=out_droid["valid"],
        )

    with open(os.path.join(OUT, "pilot_results.json"), "w") as f:
        json.dump(all_rows, f, indent=2)
    print("\n=== SUMMARY ===")
    for r in all_rows:
        traj = r.get("trajectory", "?").split("_")[1]
        nm = r["name"]
        if "error" in r:
            print(f"{traj} {nm}: ERROR {r['error']}")
            continue
        print(f"{traj} {nm:5s} n={r['n_frames']:4d} "
              f"sim3_ate={r.get('sim3_ate_rmse_m',0):.3f}m "
              f"se3_ate={r.get('se3_ate_rmse_m',0):.3f}m "
              f"scale={r.get('sim3_scale',0):.3f} "
              f"rpe1s_t={r.get('rpe_trans_rmse_30f_m',0):.3f}m "
              f"rpe1s_r={r.get('rpe_rot_rmse_30f_deg',0):.1f}deg "
              f"fps={r.get('fps',0):.1f}")


if __name__ == "__main__":
    main()
