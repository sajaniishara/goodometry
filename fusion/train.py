"""Train the factorized-attention fusion model on kinematics + INS.

Run:
    python fusion/train.py --output-dir runs/fusion_v1 --epochs 50 --batch-size 64
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/anyone/projects/goodometry")
from fusion.dataset import (
    GoFusionDataset, split_trajectories, compute_norm_stats,
    KINEMATICS_DIM, KINEMATICS_V3_DIM, INS_DIM, VO_DIM, LABEL_DIM,
)
from fusion.model        import FusionTransformer, param_count
from fusion.temporal_cnn import FusionTCN
from fusion.mvit         import FusionMViT


ROOT = "/mnt/data/go2_research_dataset_v2"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--clip-len", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--ffn-dim", type=int, default=256)
    ap.add_argument("--modality-dropout", type=float, default=0.1,
                    help="per-sample probability of dropping each modality at training time")
    ap.add_argument("--train-count", type=int, default=650)
    ap.add_argument("--val-count", type=int, default=150)
    ap.add_argument("--test-count", type=int, default=208)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-trajs", type=int, default=None,
                    help="optional override (useful for CPU smoke run)")
    ap.add_argument("--max-val-trajs", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--angular-loss-weight", type=float, default=5.0,
                    help="multiply angular-velocity loss component (§6.1 recipe)")
    ap.add_argument("--loss", choices=["mse", "huber"], default="mse")
    ap.add_argument("--ins-file", default="ins.npz",
                    help="ins.npz (IMU-only) or ins_marg.npz (stitched MARG with drift-free yaw)")
    ap.add_argument("--use-vo", action="store_true", default=False,
                    help="add VO modality (fusion_v2); requires vo.npz in each trajectory dir")
    ap.add_argument("--vo-file", default="vo.npz")
    ap.add_argument("--use-kin-v3", action="store_true", default=False,
                    help="use 35-D kin features with leg_odom_delta SE(3) (fusion_v3); "
                         "requires patch_kin_leg_odom_delta.py to have been run")
    ap.add_argument("--arch", choices=["transformer", "tcn", "mvit"], default="transformer",
                    help="transformer = factorized FusionTransformer (fusion_v1/v2/v3); "
                         "tcn = R3D-18-style 1D temporal CNN; "
                         "mvit = multiscale factorized transformer (MViT-inspired)")
    return ap.parse_args()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    err = (pred - target).detach().cpu()
    rmse = torch.sqrt((err ** 2).mean(dim=0))
    return {
        "rmse_all":   float(torch.sqrt((err ** 2).mean())),
        "rmse_lin":   float(torch.sqrt((err[:, :3] ** 2).mean())),
        "rmse_ang":   float(torch.sqrt((err[:, 3:] ** 2).mean())),
        "rmse_per_axis": rmse.numpy().tolist(),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  cuda_avail={torch.cuda.is_available()}  ngpu={torch.cuda.device_count()}", flush=True)

    # --- split and norm stats ---
    train_trajs, val_trajs, test_trajs = split_trajectories(
        args.root, args.train_count, args.val_count, args.test_count, args.seed)
    if args.max_train_trajs is not None:
        train_trajs = train_trajs[:args.max_train_trajs]
    if args.max_val_trajs is not None:
        val_trajs = val_trajs[:args.max_val_trajs]
    print(f"split: train={len(train_trajs)} val={len(val_trajs)} test={len(test_trajs)}", flush=True)

    kin_dim = KINEMATICS_V3_DIM if args.use_kin_v3 else KINEMATICS_DIM
    stats = compute_norm_stats(train_trajs, sample_frac=0.2, seed=args.seed,
                                ins_file=args.ins_file,
                                use_vo=args.use_vo, vo_file=args.vo_file,
                                kin_v3=args.use_kin_v3)
    np.savez(out / "norm_stats.npz", **stats)
    msg = (f"norm stats:  kin|μ|={np.abs(stats['kin_mean']).mean():.3f}"
           f"  kin|σ|={stats['kin_std'].mean():.3f}"
           f"  ins|μ|={np.abs(stats['ins_mean']).mean():.3f}"
           f"  ins|σ|={stats['ins_std'].mean():.3f}")
    if args.use_vo:
        msg += (f"  vo|μ|={np.abs(stats['vo_mean']).mean():.3f}"
                f"  vo|σ|={stats['vo_std'].mean():.3f}")
    print(msg, flush=True)
    kin_norm = (stats['kin_mean'], stats['kin_std'])
    ins_norm = (stats['ins_mean'], stats['ins_std'])
    vo_norm  = (stats['vo_mean'], stats['vo_std']) if args.use_vo else None

    train_ds = GoFusionDataset(train_trajs, args.clip_len, args.stride, kin_norm, ins_norm,
                                vo_norm=vo_norm, ins_file=args.ins_file,
                                vo_file=args.vo_file, use_vo=args.use_vo,
                                kin_v3=args.use_kin_v3)
    val_ds   = GoFusionDataset(val_trajs,   args.clip_len, args.stride, kin_norm, ins_norm,
                                vo_norm=vo_norm, ins_file=args.ins_file,
                                vo_file=args.vo_file, use_vo=args.use_vo,
                                kin_v3=args.use_kin_v3)
    print(f"clips: train={len(train_ds):,}  val={len(val_ds):,}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # --- model + optimizer ---
    if args.arch == "transformer":
        model = FusionTransformer(
            kin_in=kin_dim, ins_in=INS_DIM,
            vo_in=VO_DIM if args.use_vo else None,
            d_model=args.d_model, n_blocks=args.n_blocks, n_heads=args.n_heads,
            ffn_dim=args.ffn_dim, clip_len=args.clip_len,
        ).to(device)
    elif args.arch == "tcn":
        model = FusionTCN(
            kin_in=kin_dim, ins_in=INS_DIM,
            vo_in=VO_DIM if args.use_vo else None,
            clip_len=args.clip_len,
        ).to(device)
    elif args.arch == "mvit":
        model = FusionMViT(
            kin_in=kin_dim, ins_in=INS_DIM,
            vo_in=VO_DIM if args.use_vo else None,
            n_heads=args.n_heads,
            clip_len=args.clip_len,
        ).to(device)
    else:
        raise ValueError(f"unknown --arch {args.arch}")
    n_params = param_count(model)
    print(f"{args.arch} params: {n_params:,}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=3)
    loss_fn = (nn.MSELoss(reduction="none") if args.loss == "mse"
               else nn.HuberLoss(reduction="none", delta=1.0))

    # weight vector for lin/ang balance (§6.1)
    w = torch.tensor([1., 1., 1.,
                      args.angular_loss_weight, args.angular_loss_weight, args.angular_loss_weight],
                     device=device)

    log_rows = []
    csv_path = out / "train_log.csv"
    best_val = float("inf")
    stale = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss_sum = 0.0; train_n = 0
        for batch in train_loader:
            kin = batch["kin"].to(device, non_blocking=True)
            ins = batch["ins"].to(device, non_blocking=True)
            y   = batch["label"].to(device, non_blocking=True)
            vo  = batch["vo"].to(device, non_blocking=True) if args.use_vo else None

            # modality dropout (§6.5) — independent per sample, at least one always on
            B = y.size(0)
            if args.modality_dropout > 0:
                kin_keep = (torch.rand(B, device=device) > args.modality_dropout).float()
                ins_keep = (torch.rand(B, device=device) > args.modality_dropout).float()
                vo_keep  = ((torch.rand(B, device=device) > args.modality_dropout).float()
                            if args.use_vo else None)
                all_off  = (kin_keep == 0) & (ins_keep == 0)
                if vo_keep is not None:
                    all_off = all_off & (vo_keep == 0)
                kin_keep[all_off] = 1.0   # ensure at least one is on
            else:
                kin_keep = ins_keep = vo_keep = None

            pred = model(kin, ins, vo=vo,
                         kin_mask=kin_keep, ins_mask=ins_keep, vo_mask=vo_keep)
            loss = (loss_fn(pred, y) * w).mean()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            train_loss_sum += float(loss.item()) * B
            train_n += B

        # --- val ---
        model.eval()
        preds = []; targets = []
        with torch.no_grad():
            for batch in val_loader:
                kin = batch["kin"].to(device, non_blocking=True)
                ins = batch["ins"].to(device, non_blocking=True)
                y   = batch["label"].to(device, non_blocking=True)
                vo  = batch["vo"].to(device, non_blocking=True) if args.use_vo else None
                p = model(kin, ins, vo=vo)
                preds.append(p); targets.append(y)
        preds = torch.cat(preds, dim=0); targets = torch.cat(targets, dim=0)
        val_loss = float(((preds - targets) ** 2).mean().item())   # plain MSE for comparability
        metrics = compute_metrics(preds, targets)

        sched.step(val_loss)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        train_loss = train_loss_sum / max(train_n, 1)
        msg = (f"{now}  ep {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
               f"val_loss={val_loss:.4f}  rmse={metrics['rmse_all']:.4f}  "
               f"lin={metrics['rmse_lin']:.4f}  ang={metrics['rmse_ang']:.4f}  "
               f"elapsed={time.time()-t0:.0f}s")
        print(msg, flush=True)

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics,
               "elapsed_s": time.time() - t0}
        log_rows.append(row)
        with open(csv_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=[k for k in row.keys() if k != "rmse_per_axis"])
            wr.writeheader()
            for r in log_rows:
                wr.writerow({k: v for k, v in r.items() if k != "rmse_per_axis"})
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_loss": val_loss,
                "metrics": metrics,
                "args": vars(args),
            }, out / "best_model.pt")
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch} (patience {args.patience})", flush=True)
                break

        torch.save({"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss},
                   out / "last_model.pt")

    print(f"done. best_val={best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
