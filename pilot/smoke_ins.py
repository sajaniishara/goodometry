"""INS smoke: run Madgwick on one trajectory, compare quat vs GT, verify gravity removal."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/anyone/projects/goodometry")
from ins.madgwick import run_madgwick, _quat_to_euler_rpy

TRAJ = "/mnt/data/go2_research_dataset_v2/trajectory_000717_20260325_083252"

sens = np.load(os.path.join(TRAJ, "sensors.npz"))
df = pd.read_csv(os.path.join(TRAJ, "data.csv")).set_index("frame_idx").loc[sens["frame_idx"]]
gt_quat_wxyz = df[["gt_ori_w", "gt_ori_x", "gt_ori_y", "gt_ori_z"]].to_numpy(dtype=np.float32)

# IMU-only (6-axis) — default for the Go2 pipeline (sim mag is hardcoded; won't transfer).
out = run_madgwick(sens["imu"], sens["mag"], sample_rate_hz=30.0, use_mag=False)

N = len(out["quat_wxyz"])
print(f"trajectory {os.path.basename(TRAJ)}  N={N}")

print()
print("--- quaternion unit-norm check ---")
norms = np.linalg.norm(out["quat_wxyz"], axis=1)
print(f"  quat |q|  mean={norms.mean():.6f}  min={norms.min():.6f}  max={norms.max():.6f}")

print()
print("--- gravity compensation check (accel_body should have mean ~ 0) ---")
accel_raw_magn = np.linalg.norm(sens["imu"][:, :3], axis=1)
accel_comp_magn = np.linalg.norm(out["accel_body"], axis=1)
print(f"  raw   |accel|  mean={accel_raw_magn.mean():.3f}  std={accel_raw_magn.std():.3f}  m/s²")
print(f"  comp  |accel|  mean={accel_comp_magn.mean():.3f}  std={accel_comp_magn.std():.3f}  m/s²   (should be << g)")

print()
print("--- quaternion vs GT orientation (skip first 5s of filter settling) ---")
skip = int(30 * 5)  # drop 5s of filter warmup
# angle between quats: θ = 2*acos(|<q_est, q_gt>|)
dot = np.abs((out["quat_wxyz"][skip:] * gt_quat_wxyz[skip:]).sum(axis=1))
dot = np.clip(dot, -1.0, 1.0)
ang_err_rad = 2.0 * np.arccos(dot)
ang_err_deg = np.degrees(ang_err_rad)
print(f"  quat angle error  mean={ang_err_deg.mean():.2f}°  median={np.median(ang_err_deg):.2f}°  p95={np.percentile(ang_err_deg,95):.2f}°  max={ang_err_deg.max():.2f}°")

print()
print("--- Euler vs GT (skip warmup) ---")
est_rpy = out["euler_rpy"][skip:]
gt_rpy = df[["gt_ori_roll", "gt_ori_pitch", "gt_ori_yaw"]].to_numpy(dtype=np.float32)[skip:]
def wrap_pi(a): return (a + np.pi) % (2 * np.pi) - np.pi
for i, axis in enumerate(["roll", "pitch", "yaw"]):
    err = wrap_pi(est_rpy[:, i] - gt_rpy[:, i])
    print(f"  {axis}  est_mean={est_rpy[:,i].mean():+.3f}  gt_mean={gt_rpy[:,i].mean():+.3f}  RMSE={np.sqrt((err**2).mean()):.3f} rad ({np.degrees(np.sqrt((err**2).mean())):.1f}°)")

print()
print("--- flags (fraction of frames in each abnormal state) ---")
labels = ["initialising", "angular_recovery", "accel_recovery", "magnetic_recovery"]
for i, name in enumerate(labels):
    print(f"  {name:22s}  {out['flags'][:, i].mean():.3f}")
