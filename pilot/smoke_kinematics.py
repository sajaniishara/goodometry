"""Smoke-test the kinematic preprocessing on one trajectory.

What we check:
  1. Foot positions in body frame are geometrically sensible
     (front feet ahead, rear feet behind, legs splayed left/right,
     feet ~0.30 m below body origin).
  2. Contact probabilities show the expected trot-gait diagonal pattern
     (FL+RR alternate stance with FR+RL).
  3. Leg-odometry body velocity approximates the GT label (sensors.npz['labels'][:3]).
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/anyone/projects/goodometry")
from calibration import load_calibrated_joints
from kinematics.fk import Go2FK, foot_vel_body_from_finite_diff
from kinematics.contact import velocity_gate_contact
from kinematics.leg_odom import body_velocity_from_legs

TRAJ = "/mnt/data/go2_research_dataset_v2/trajectory_000717_20260325_083252"
DT = 1.0 / 30.0

joints, frame_idx, offset = load_calibrated_joints(TRAJ)
sens = np.load(os.path.join(TRAJ, "sensors.npz"))
imu = sens["imu"]             # (N, 6) [accel_xyz, gyro_xyz]
labels = sens["labels"]       # (N, 6) [vx, vy, vz, ωx, ωy, ωz]
print(f"trajectory: {os.path.basename(TRAJ)}   N = {len(frame_idx)}   offset = {offset:+.3f} rad")

fk = Go2FK(device="cpu")
with torch.no_grad():
    q = torch.from_numpy(joints[:, :12]).float()           # 12 positions
    foot_pos = fk.foot_positions_body(q)                   # (N, 4, 3)
foot_vel = foot_vel_body_from_finite_diff(foot_pos, DT)    # (N, 4, 3)
contact = velocity_gate_contact(foot_vel)                  # (N, 4)
omega = torch.from_numpy(imu[:, 3:6]).float()              # (N, 3)
v_legs = body_velocity_from_legs(foot_pos, foot_vel, omega, contact)

# to numpy
foot_pos_np = foot_pos.numpy(); foot_vel_np = foot_vel.numpy()
contact_np = contact.numpy();   v_legs_np = v_legs.numpy()
v_gt = labels[:, :3]

print()
print("--- FK sanity: mean foot position in body frame (x=forward, y=left, z=up) ---")
leg_names = ["FL", "FR", "RL", "RR"]
for i, n in enumerate(leg_names):
    m = foot_pos_np[:, i].mean(axis=0)
    print(f"  {n}  x={m[0]:+.3f}  y={m[1]:+.3f}  z={m[2]:+.3f}")

print()
print("--- Contact stats (fraction of frames in stance) ---")
for i, n in enumerate(leg_names):
    frac = (contact_np[:, i] > 0.5).mean()
    mean_p = contact_np[:, i].mean()
    print(f"  {n}  mean_contact_prob={mean_p:.3f}   frac>0.5={frac:.2f}")

print()
print("--- Diagonal correlation (trot: FL⇄RR in phase, FR⇄RL in phase, FL⇄FR anti-phase) ---")
print(f"  corr(FL, RR) = {np.corrcoef(contact_np[:, 0], contact_np[:, 3])[0,1]:+.3f}  (want + for trot)")
print(f"  corr(FR, RL) = {np.corrcoef(contact_np[:, 1], contact_np[:, 2])[0,1]:+.3f}  (want + for trot)")
print(f"  corr(FL, FR) = {np.corrcoef(contact_np[:, 0], contact_np[:, 1])[0,1]:+.3f}  (want -)")

print()
print("--- Leg odometry vs GT body velocity ---")
print("  (`labels[:,:3]` is WORLD-frame from get_world_pose(); rotate → body for a fair compare)")

# rotate world-frame GT linear velocity into body frame using gt_ori quaternion
import pandas as pd
df = pd.read_csv(os.path.join(TRAJ, "data.csv")).set_index("frame_idx").loc[frame_idx]
q_wxyz = df[["gt_ori_w", "gt_ori_x", "gt_ori_y", "gt_ori_z"]].to_numpy(dtype=np.float64)
# R_wb: world ← body; we want v_body = R_wb^T @ v_world
def quat_to_R(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w*w + x*x + y*y + z*z); w, x, y, z = w/n, x/n, y/n, z/n
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R

R_wb = quat_to_R(q_wxyz)                     # (N, 3, 3)
v_gt_world = labels[:, :3]                   # (N, 3)
v_gt_body = np.einsum("nij,nj->ni", R_wb.transpose(0, 2, 1), v_gt_world)

print()
print("  axis   GT_body_mean   LO_mean    RMSE(body)    corr(body)")
for i, a in enumerate(["vx", "vy", "vz"]):
    err = v_legs_np[:, i] - v_gt_body[:, i]
    rmse = float(np.sqrt((err ** 2).mean()))
    cor = float(np.corrcoef(v_legs_np[:, i], v_gt_body[:, i])[0, 1])
    print(f"  {a}      {v_gt_body[:, i].mean():+.3f}        {v_legs_np[:, i].mean():+.3f}     {rmse:.3f}         {cor:+.3f}")

print()
print("  sanity: GT world vs GT body (should disagree when robot is yawed)")
for i, a in enumerate(["vx", "vy", "vz"]):
    print(f"    {a}  world_mean={v_gt_world[:, i].mean():+.3f}  body_mean={v_gt_body[:, i].mean():+.3f}")
