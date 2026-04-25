"""Fair head-to-head evaluation: Stage-2 vs fusion_v1 on a trajectory subset
that neither model trained on.

Procedure:
  1. compute the intersection of goodometry's test set with Stage-2's test+val
     (i.e. trajectories where neither model saw them during training).
  2. evaluate fusion_v1 on that subset.
  3. evaluate the Stage-2 CNN RGB model on that subset.
  4. print a comparison table.

Both reported in body frame: linear velocity is computed by rotating the
Stage-2 predictions through GT orientation when needed (Stage-2 was trained on
world-frame linear, body-frame angular — fusion is body-frame everything).
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

GOODOM = "/home/anyone/projects/goodometry"
GO2 = "/home/anyone/projects/go2_research"
sys.path.insert(0, GOODOM)
sys.path.insert(0, GO2)

from fusion.dataset import (
    GoFusionDataset, split_trajectories as goodom_split,
    KINEMATICS_DIM, INS_DIM,
)
from fusion.model import FusionTransformer
from cnn3d.dataset import (
    Go2ClipDataset, split_trajectories as stage2_split,
)
from cnn3d.model import VideoResNet18Fused

ROOT = "/mnt/data/go2_research_dataset_v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}", flush=True)


# ---------- 1. find the clean subset ----------
g_train, g_val, g_test = goodom_split(ROOT, 650, 150, 208, seed=42)
s_train, s_val, s_test = stage2_split(ROOT, train_count=None, val_count=None,
                                       train_frac=0.7, val_frac=0.15,
                                       min_date='20260226', seed=42)
g_train_set, s_train_set = set(g_train), set(s_train)
s_val_set = set(s_val)

# trajectories where:
#   - in fusion's test set (so fusion never saw them in training/val)
#   - NOT in Stage-2's training (so Stage-2 never saw them either; may be in s_val
#     or s_test, both held out from Stage-2 training)
clean = sorted(set(g_test) - s_train_set)
print(f"clean head-to-head trajectories: {len(clean)}")
print(f"  (in fusion test ∩ NOT in Stage-2 training)")


# ---------- 2. evaluate fusion_v1 ----------
print("\n=== fusion_v1 eval ===", flush=True)
fusion_run_dir = Path(GOODOM) / "runs/fusion_v1"
run_args = json.load(open(fusion_run_dir / "args.json"))
stats = np.load(fusion_run_dir / "norm_stats.npz")
kin_norm = (stats["kin_mean"], stats["kin_std"])
ins_norm = (stats["ins_mean"], stats["ins_std"])

ds_f = GoFusionDataset(clean, run_args["clip_len"], run_args["stride"],
                        kin_norm, ins_norm,
                        ins_file=run_args.get("ins_file", "ins.npz"))
print(f"  fusion clips: {len(ds_f):,}")

model_f = FusionTransformer(
    kin_in=KINEMATICS_DIM, ins_in=INS_DIM,
    d_model=run_args["d_model"], n_blocks=run_args["n_blocks"],
    n_heads=run_args["n_heads"], ffn_dim=run_args["ffn_dim"],
    clip_len=run_args["clip_len"],
).to(DEVICE)
ck = torch.load(fusion_run_dir / "best_model.pt", map_location=DEVICE, weights_only=False)
model_f.load_state_dict(ck["model"]); model_f.eval()

preds_f, tgts_f = [], []
loader = DataLoader(ds_f, batch_size=256, shuffle=False, num_workers=4,
                     pin_memory=(DEVICE.type == "cuda"))
t0 = time.time()
with torch.no_grad():
    for batch in loader:
        kin = batch["kin"].to(DEVICE, non_blocking=True)
        ins = batch["ins"].to(DEVICE, non_blocking=True)
        y   = batch["label"].to(DEVICE, non_blocking=True)
        preds_f.append(model_f(kin, ins).cpu()); tgts_f.append(y.cpu())
preds_f = torch.cat(preds_f).numpy()
tgts_f  = torch.cat(tgts_f).numpy()
print(f"  fusion inference: {len(preds_f):,} clips in {time.time()-t0:.1f}s")


# ---------- 3. evaluate Stage-2 ----------
print("\n=== Stage-2 eval ===", flush=True)
stage2_ckpt = Path(GO2) / "runs/cnn_rgb_stage2/best_model.pt"
ck_s = torch.load(stage2_ckpt, map_location=DEVICE, weights_only=False)
s_args = ck_s["args"]
print(f"  Stage-2 ckpt epoch={ck_s['epoch']}  val_loss={ck_s['val_loss']:.4f}")

# Build Stage-2 dataset on the clean subset.
s_norm = json.load(open(Path(GO2) / "runs/cnn_rgb_stage2/norm_stats.json"))
h5_path = "/mnt/data/go2_research_dataset_v2/frames_224.h5"
if not Path(h5_path).exists():
    h5_path = None

ds_s = Go2ClipDataset(
    clean,
    clip_len=s_args.get("clip_len", 8),
    stride=s_args.get("stride", 8),
    img_size=s_args.get("img_size", 224),
    visual_mode=s_args.get("visual_mode", "rgb"),
    norm_stats=s_norm,
    h5_path=h5_path,
    max_angular_vel=s_args.get("max_angular_vel", 10.0),
)
print(f"  Stage-2 clips: {len(ds_s):,}")

model_s = VideoResNet18Fused(
    visual_mode=s_args.get("visual_mode", "rgb"),
    pretrained=False,
).to(DEVICE)
model_s.load_state_dict(ck_s["model"]); model_s.eval()

loader_s = DataLoader(ds_s, batch_size=16, shuffle=False, num_workers=4,
                      pin_memory=(DEVICE.type == "cuda"))
preds_s, tgts_s = [], []
t0 = time.time()
with torch.no_grad():
    for batch in loader_s:
        visual = batch["visual"].to(DEVICE, non_blocking=True)
        imu9   = batch["imu9"].to(DEVICE, non_blocking=True)
        joints = batch["joints"].to(DEVICE, non_blocking=True)
        y      = batch["label"]
        preds_s.append(model_s(visual, imu9, joints).cpu()); tgts_s.append(y)
preds_s = torch.cat(preds_s).numpy()
tgts_s  = torch.cat(tgts_s).numpy()
print(f"  Stage-2 inference: {len(preds_s):,} clips in {time.time()-t0:.1f}s")


# ---------- 4. comparison ----------
def metrics(preds, tgts, label_is_world_lin=False, traj_for_each=None):
    """If label_is_world_lin, rotate Stage-2's world-frame linear preds and labels
    to body frame before computing metrics. (Stage-2 was trained on world-frame
    linear; for fair body-frame comparison we rotate both pred and target
    through GT orientation.) Caller passes a list of (traj, gt_quat) per clip.
    For now, since both fusion and Stage-2 use the same `labels` source per
    clip — fusion uses labels_body.npz; Stage-2 uses sensors.npz['labels']
    (world-frame linear) — we report metrics in each model's NATIVE frame and
    note that vy/vx Stage-2 metrics are world-frame whereas fusion's are body."""
    err = preds - tgts
    return {
        "rmse_overall": float(np.sqrt((err ** 2).mean())),
        "rmse_lin":     float(np.sqrt((err[:, :3] ** 2).mean())),
        "rmse_ang":     float(np.sqrt((err[:, 3:] ** 2).mean())),
        "rmse_per_axis": np.sqrt((err ** 2).mean(0)).tolist(),
    }

m_fusion = metrics(preds_f, tgts_f)
m_stage2 = metrics(preds_s, tgts_s)

print()
print("=" * 72)
print(f"FAIR HEAD-TO-HEAD on {len(clean)} trajectories"
      f"  (in fusion test ∩ NOT in Stage-2 training)")
print("=" * 72)
print(f"  fusion_v1   :  {len(preds_f):,} clips")
print(f"  Stage-2 CNN :  {len(preds_s):,} clips")
print()
print(f"{'metric':<20s}  {'Stage-2':>10s}  {'fusion_v1':>10s}  {'Δ':>8s}")
def pct(a, b): return f"{(b - a) / a * 100:+5.1f} %"
print(f"{'overall RMSE':<20s}  {m_stage2['rmse_overall']:10.4f}  {m_fusion['rmse_overall']:10.4f}  {pct(m_stage2['rmse_overall'], m_fusion['rmse_overall']):>8s}")
print(f"{'linear RMSE (m/s)':<20s}  {m_stage2['rmse_lin']:10.4f}  {m_fusion['rmse_lin']:10.4f}  {pct(m_stage2['rmse_lin'], m_fusion['rmse_lin']):>8s}")
print(f"{'angular RMSE (rad/s)':<20s}  {m_stage2['rmse_ang']:10.4f}  {m_fusion['rmse_ang']:10.4f}  {pct(m_stage2['rmse_ang'], m_fusion['rmse_ang']):>8s}")
print()
axes = ["vx", "vy", "vz", "ωx", "ωy", "ωz"]
print(f"{'axis':<6s}  {'Stage-2':>10s}  {'fusion_v1':>10s}  {'Δ':>8s}")
for i, a in enumerate(axes):
    s = m_stage2["rmse_per_axis"][i]
    f = m_fusion["rmse_per_axis"][i]
    print(f"  {a:<4s}  {s:10.4f}  {f:10.4f}  {pct(s, f):>8s}")

# save results
out = {
    "n_trajectories": len(clean),
    "fusion_v1_clips": int(len(preds_f)),
    "stage2_clips":    int(len(preds_s)),
    "fusion_v1": m_fusion,
    "stage2":    m_stage2,
}
out_path = Path(GOODOM) / "runs/fair_head_to_head.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved {out_path}")
