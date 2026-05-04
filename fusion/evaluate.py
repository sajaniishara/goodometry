"""Evaluate a trained fusion model on the test split.

Per-axis RMSE and percentiles over the 208-trajectory test set, evaluated
in body frame. Reads `args.json`, `norm_stats.npz`, `best_model.pt` from
the run directory.

Usage:
    python fusion/evaluate.py --run-dir runs/fusion_v1
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/anyone/projects/goodometry")
from fusion.dataset       import GoFusionDataset, split_trajectories, KINEMATICS_DIM, KINEMATICS_V3_DIM, INS_DIM, VO_DIM
from fusion.model         import FusionTransformer
from fusion.temporal_cnn  import FusionTCN
from fusion.mvit          import FusionMViT


ROOT = "/mnt/data/go2_research_dataset_v2"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_args = json.load(open(run_dir / "args.json"))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    print(f"run_dir: {run_dir}", flush=True)

    # Reuse the same split as training.
    _, _, test_trajs = split_trajectories(
        args.root,
        run_args["train_count"], run_args["val_count"], run_args["test_count"],
        run_args["seed"])
    print(f"test trajectories: {len(test_trajs)}", flush=True)

    stats = np.load(run_dir / "norm_stats.npz")
    kin_norm = (stats["kin_mean"], stats["kin_std"])
    ins_norm = (stats["ins_mean"], stats["ins_std"])
    ins_file = run_args.get("ins_file", "ins.npz")
    use_vo   = run_args.get("use_vo", False)
    vo_file  = run_args.get("vo_file", "vo.npz")
    vo_norm  = (stats["vo_mean"], stats["vo_std"]) if use_vo else None
    use_kin_v3 = run_args.get("use_kin_v3", False)
    kin_dim    = KINEMATICS_V3_DIM if use_kin_v3 else KINEMATICS_DIM

    test_ds = GoFusionDataset(
        test_trajs,
        clip_len=run_args["clip_len"],
        stride=run_args["stride"],
        kin_norm=kin_norm,
        ins_norm=ins_norm,
        vo_norm=vo_norm,
        ins_file=ins_file,
        vo_file=vo_file,
        use_vo=use_vo,
        kin_v3=use_kin_v3,
    )
    print(f"test clips: {len(test_ds):,}", flush=True)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))

    arch = run_args.get("arch", "transformer")
    if arch == "transformer":
        model = FusionTransformer(
            kin_in=kin_dim, ins_in=INS_DIM,
            vo_in=VO_DIM if use_vo else None,
            d_model=run_args["d_model"], n_blocks=run_args["n_blocks"],
            n_heads=run_args["n_heads"], ffn_dim=run_args["ffn_dim"],
            clip_len=run_args["clip_len"],
        ).to(device)
    elif arch == "tcn":
        model = FusionTCN(
            kin_in=kin_dim, ins_in=INS_DIM,
            vo_in=VO_DIM if use_vo else None,
            clip_len=run_args["clip_len"],
        ).to(device)
    elif arch == "mvit":
        model = FusionMViT(
            kin_in=kin_dim, ins_in=INS_DIM,
            vo_in=VO_DIM if use_vo else None,
            n_heads=run_args["n_heads"],
            clip_len=run_args["clip_len"],
        ).to(device)
    else:
        raise ValueError(f"unknown arch in run_args: {arch}")
    print(f"arch: {arch}", flush=True)
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded best_model.pt from epoch {ckpt['epoch']}", flush=True)

    preds = []; tgts = []
    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            kin = batch["kin"].to(device, non_blocking=True)
            ins = batch["ins"].to(device, non_blocking=True)
            y   = batch["label"].to(device, non_blocking=True)
            vo  = batch["vo"].to(device, non_blocking=True) if use_vo else None
            p = model(kin, ins, vo=vo)
            preds.append(p.cpu()); tgts.append(y.cpu())
    preds = torch.cat(preds, dim=0).numpy()
    tgts  = torch.cat(tgts,  dim=0).numpy()
    elapsed = time.time() - t0
    print(f"inference: {len(preds):,} clips in {elapsed:.1f}s", flush=True)

    err = preds - tgts
    rmse_overall = float(np.sqrt((err ** 2).mean()))
    rmse_lin = float(np.sqrt((err[:, :3] ** 2).mean()))
    rmse_ang = float(np.sqrt((err[:, 3:] ** 2).mean()))
    rmse_per_axis = np.sqrt((err ** 2).mean(axis=0))
    abs_err = np.abs(err)
    p50 = np.median(abs_err, axis=0)
    p95 = np.percentile(abs_err, 95, axis=0)

    axes = ["vx", "vy", "vz", "ωx", "ωy", "ωz"]
    print()
    print(f"{'axis':>4s}  {'rmse':>8s}  {'p50':>8s}  {'p95':>8s}  {'units':>8s}")
    for i, a in enumerate(axes):
        units = "m/s" if i < 3 else "rad/s"
        print(f"  {a:2s}  {rmse_per_axis[i]:8.4f}  {p50[i]:8.4f}  {p95[i]:8.4f}  {units:>8s}")
    print()
    print(f"overall   RMSE = {rmse_overall:.4f}")
    print(f"linear    RMSE = {rmse_lin:.4f} m/s")
    print(f"angular   RMSE = {rmse_ang:.4f} rad/s")

    result = {
        "run_dir": str(run_dir),
        "epoch": int(ckpt["epoch"]),
        "n_test_clips": int(len(preds)),
        "rmse_overall": rmse_overall,
        "rmse_lin": rmse_lin,
        "rmse_ang": rmse_ang,
        "rmse_per_axis": rmse_per_axis.tolist(),
        "p50_per_axis": p50.tolist(),
        "p95_per_axis": p95.tolist(),
        "ins_file": ins_file,
    }
    with open(run_dir / "test_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {run_dir / 'test_results.json'}")

    # Per-trajectory breakdown (clips are in order — shuffle=False).
    traj_preds: dict[str, list] = {}
    traj_tgts:  dict[str, list] = {}
    for i, (traj_dir, _) in enumerate(test_ds.clips):
        traj_preds.setdefault(traj_dir, []).append(preds[i])
        traj_tgts.setdefault(traj_dir, []).append(tgts[i])

    per_traj = []
    for traj_dir in sorted(traj_preds.keys()):
        p = np.stack(traj_preds[traj_dir])
        t = np.stack(traj_tgts[traj_dir])
        e = p - t
        per_traj.append({
            "trajectory": os.path.basename(traj_dir),
            "n_clips": len(p),
            "rmse_overall": float(np.sqrt((e ** 2).mean())),
            "rmse_lin":     float(np.sqrt((e[:, :3] ** 2).mean())),
            "rmse_ang":     float(np.sqrt((e[:, 3:] ** 2).mean())),
            "rmse_per_axis": np.sqrt((e ** 2).mean(axis=0)).tolist(),
            "p50_per_axis":  np.median(np.abs(e), axis=0).tolist(),
            "p95_per_axis":  np.percentile(np.abs(e), 95, axis=0).tolist(),
        })

    with open(run_dir / "per_traj_results.json", "w") as f:
        json.dump(per_traj, f, indent=2)
    print(f"Saved {run_dir / 'per_traj_results.json'} ({len(per_traj)} trajectories)")


if __name__ == "__main__":
    main()
