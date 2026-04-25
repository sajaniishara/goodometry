"""Quick DPVO smoke test on the first ~300 frames of a pilot trajectory."""
import os, sys
import numpy as np
sys.path.insert(0, "/home/anyone/projects/goodometry")
from vo.dpvo_runner import run_dpvo
from vo.gt import load_gt
from vo.eval import ate, rpe, quat_wxyz_to_R

TRAJ = "/mnt/data/go2_research_dataset_v2/trajectory_000717_20260325_083252"
# Temporarily slice sensors.npz so we only process ~300 frames
import numpy as np
orig = dict(np.load(os.path.join(TRAJ, "sensors.npz")))
subset = "/tmp/smoke_traj"
os.makedirs(subset, exist_ok=True)
# Create a symlink to trajectory frames, but write a subset sensors.npz
import shutil
for name in os.listdir(TRAJ):
    src = os.path.join(TRAJ, name)
    dst = os.path.join(subset, name)
    if os.path.islink(dst) or os.path.exists(dst):
        continue
    if name == "sensors.npz":
        continue
    os.symlink(src, dst)
N = 300
np.savez(os.path.join(subset, "sensors.npz"),
         frame_idx=orig["frame_idx"][:N],
         imu=orig["imu"][:N], mag=orig["mag"][:N],
         joints=orig["joints"][:N], labels=orig["labels"][:N])
# data.csv is a symlink already; reading full CSV and indexing by frame_idx still works

INTRIN = (914.0, 914.0, 640.0, 360.0)
out = run_dpvo(subset, INTRIN)
gt = load_gt(subset)
print(f"DPVO: {len(out['frame_idx'])} frames in {out['elapsed_s']:.1f}s ({out['fps']:.1f} FPS)")
print(f"GT:   {len(gt['frame_idx'])} frames")
assert np.array_equal(out["frame_idx"], gt["frame_idx"][:len(out["frame_idx"])])
pred_pos = out["pose_mat"][:, :3, 3]
pred_R = out["pose_mat"][:, :3, :3]
gt_pos = gt["pos"][:len(pred_pos)]
gt_R = quat_wxyz_to_R(gt["quat_wxyz"][:len(pred_pos)])
print("ATE (sim3):", ate(pred_pos, gt_pos, with_scale=True))
print("RPE 30f:  ", rpe(pred_pos, pred_R, gt_pos, gt_R, delta_frames=30))
