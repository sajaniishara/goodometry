"""Quick DROID-SLAM (stereo) smoke test on the first ~300 frames of a pilot trajectory."""
import os, sys
import numpy as np
sys.path.insert(0, "/home/anyone/projects/goodometry")
from vo.droid_runner import run_droid
from vo.gt import load_gt
from vo.eval import ate, rpe, quat_wxyz_to_R

TRAJ = "/tmp/smoke_traj"  # prepared by smoke_dpvo.py
INTRIN = (914.0, 914.0, 640.0, 360.0)
out = run_droid(TRAJ, INTRIN)
gt = load_gt(TRAJ)
N = min(len(out["frame_idx"]), len(gt["frame_idx"]))
print(f"DROID: {len(out['frame_idx'])} frames in {out['elapsed_s']:.1f}s ({out['fps']:.1f} FPS)")
pred_pos = out["pose_mat"][:N, :3, 3]
pred_R = out["pose_mat"][:N, :3, :3]
gt_pos = gt["pos"][:N]
gt_R = quat_wxyz_to_R(gt["quat_wxyz"][:N])
print("ATE (sim3):", ate(pred_pos, gt_pos, with_scale=True))
print("ATE (SE3) :", ate(pred_pos, gt_pos, with_scale=False))
print("RPE 30f:  ", rpe(pred_pos, pred_R, gt_pos, gt_R, delta_frames=30))
