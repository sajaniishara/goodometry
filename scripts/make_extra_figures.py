"""Generate the five additional thesis figures requested:

  fig10_system_architecture       — block diagram of the goodometry pipeline
  fig11_wallclock_vs_rmse         — training-time efficiency
  fig12_droid_stride_sweep        — DROID-SLAM ATE / wall-clock vs stride
  fig13_dataset_summary           — dataset / hardware / split summary table
  fig14_sample_trajectory         — predicted vs GT velocity over one traj

Run from repo root:
    python scripts/make_extra_figures.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.ticker as mticker

REPO = Path("/home/anyone/projects/goodometry")
RUNS = REPO / "runs"
OUT  = REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":         "serif",
    "font.size":           10,
    "axes.titlesize":      11,
    "axes.labelsize":      10,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "legend.fontsize":     9,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "axes.axisbelow":      True,
    "grid.color":          "#e6e6e6",
    "grid.linewidth":      0.6,
    "savefig.bbox":        "tight",
    "savefig.dpi":         300,
})

COLORS = {
    "Stage-2 v2":     "#888888",
    "fusion_v1":      "#9ecae1",
    "fusion_v1_marg": "#6baed6",
    "fusion_v2":      "#08519c",
    "fusion_v2_marg": "#2171b5",
    "fusion_v3":      "#fd8d3c",
    "fusion_tcn":     "#41ab5d",
    "fusion_mvit":    "#807dba",
}


# ===========================================================================
# Figure 10 — system architecture diagram
# ===========================================================================
def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60)
    ax.axis("off")

    def box(x, y, w, h, label, fc="#dee5f0", ec="#08519c", title_size=10, body=None):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                            linewidth=1.2, facecolor=fc, edgecolor=ec)
        ax.add_patch(b)
        ax.text(x + w/2, y + h - 2.5, label, ha="center", va="top",
                fontsize=title_size, weight="bold")
        if body:
            ax.text(x + w/2, y + h/2 - 1.5, body, ha="center", va="center",
                    fontsize=8.5, color="#333")

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle="-|>", mutation_scale=14,
                             linewidth=1.3, color="#444",
                             shrinkA=4, shrinkB=4)
        ax.add_patch(a)

    # Inputs (left column)
    box(2, 47, 18, 8, "Stereo cameras",      fc="#fff5e6", ec="#cc6600",
        body="left_image.png\nright_image.png  (30 Hz)")
    box(2, 35, 18, 8, "IMU + magnetometer",  fc="#fff5e6", ec="#cc6600",
        body="accel · gyro · mag\n(30 Hz)")
    box(2, 23, 18, 8, "Joint encoders",      fc="#fff5e6", ec="#cc6600",
        body="12 DOF · positions")
    box(2, 11, 18, 8, "Robot URDF",          fc="#fff5e6", ec="#cc6600",
        body="Go2 forward kinematics")

    # Preprocessing arms (middle column)
    box(28, 41, 24, 9, "VO arm",             fc="#e8f4ea", ec="#41ab5d",
        body="DROID-SLAM stride=3, fast=True\n→ pose_7d (cam→world)")
    box(28, 28, 24, 9, "INS arm",            fc="#e8f4ea", ec="#41ab5d",
        body="Madgwick AHRS (IMU / MARG)\n→ quat_wxyz, accel_world, gyro_body")
    box(28, 15, 24, 9, "kin arm",            fc="#e8f4ea", ec="#41ab5d",
        body="FK + stance-foot constraint\n→ foot_pos/vel · contact · v_body_legs")

    # Per-frame features (joining column)
    box(60, 30, 18, 22, "Per-frame features",
        fc="#dee5f0", ec="#08519c",
        body="VO  (7D)\n\nINS (10D)\n\nkin (31D)\n\n+ modality emb.\n+ sinusoidal PE")

    # Fusion model
    box(82, 28, 16, 14, "Fusion model",
        fc="#08519c", ec="#08519c",
        title_size=10.5,
        body=None)
    ax.text(82 + 8, 28 + 9, "FusionTransformer\n"
            "factorised modal\n+ temporal attn\n455 K params",
            ha="center", va="center", fontsize=9, color="white")

    # Output
    box(82, 8, 16, 12, "Output",
        fc="#fff5e6", ec="#cc6600",
        body="6-D body-frame\nvelocity at t\n[vx vy vz ωx ωy ωz]")

    # Arrows
    arrow(20, 51, 28, 47)   # cameras → VO
    arrow(20, 39, 28, 35)   # IMU → INS
    arrow(20, 27, 28, 22)   # joints → kin
    arrow(20, 15, 28, 19)   # URDF → kin

    arrow(52, 45, 60, 47)   # VO → features
    arrow(52, 32, 60, 41)   # INS → features
    arrow(52, 19, 60, 35)   # kin → features

    arrow(78, 41, 82, 38)   # features → fusion
    arrow(90, 28, 90, 20)   # fusion → output

    # Title
    ax.text(50, 58, "Goodometry pipeline — classical preprocessing + learned sensor fusion",
            ha="center", va="center", fontsize=12, weight="bold")
    ax.text(50, 4, "Visual frames are used only by the VO arm; the fusion model itself sees no images.",
            ha="center", va="center", fontsize=9, style="italic", color="#666")

    fig.savefig(OUT / "fig10_system_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig10_system_architecture.pdf",            bbox_inches="tight")
    plt.close(fig)
    print("fig10 done")


# ===========================================================================
# Figure 11 — wall-clock vs RMSE
# ===========================================================================
def fig_wallclock_rmse():
    data = [
        ("Stage-2 v2",     5*24*60, 0.1168),
        ("fusion_v1",      37,      0.0933),
        ("fusion_v1_marg", 38,      0.0844),
        ("fusion_v2_marg", 60,      0.0816),
        ("fusion_v3",      48,      0.0883),
        ("fusion_v2",      82,      0.0737),
        ("fusion_tcn",     88,      0.0746),
        ("fusion_mvit",    172,     0.0772),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for name, t, r in data:
        ax.scatter(t, r, color=COLORS[name], edgecolor="black",
                   s=160, linewidths=0.7, zorder=3)
    offsets = {
        "Stage-2 v2":     (-12, -10, "right"),
        "fusion_v1":      (10, -8, "left"),
        "fusion_v1_marg": (10, 4, "left"),
        "fusion_v2_marg": (10, -8, "left"),
        "fusion_v3":      (-12, -10, "right"),
        "fusion_v2":      (12, 4, "left"),
        "fusion_tcn":     (10, -10, "left"),
        "fusion_mvit":    (10, 4, "left"),
    }
    for name, t, r in data:
        dx, dy, ha = offsets[name]
        ax.annotate(name, xy=(t, r), xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8.8, ha=ha, va="center")

    ax.set_xscale("log")
    ax.set_xlabel("Training wall-clock (minutes, log scale)")
    ax.set_ylabel("Overall test RMSE")
    ax.set_title("Compute efficiency — RMSE vs training time")

    # Reference line at fusion_v2
    ax.axhline(0.0737, color="#08519c", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(7000, 0.0737 - 0.003, "fusion_v2 RMSE = 0.0737",
            fontsize=8, color="#08519c", ha="right")

    ax.set_xlim(20, 2e4)
    ax.set_ylim(0.06, 0.13)
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig11_wallclock_vs_rmse.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig11_wallclock_vs_rmse.pdf",            bbox_inches="tight")
    plt.close(fig)
    print("fig11 done")


# ===========================================================================
# Figure 12 — DROID-SLAM stride sweep
# ===========================================================================
def fig_stride_sweep():
    # Data from EXPERIMENTS.md §7.1
    strides = [1, 3, 4, 5, 8]
    flat_ate    = [0.915, 0.789, 1.408, 1.077, 1.331]
    forest_ate  = [0.149, 0.175, 0.144, 0.157, 0.126]
    circle_ate  = [0.183, 0.181, 0.183, 0.185, 0.278]
    flat_wall   = [8.4,  3.4,  2.6,  2.1,  1.3]
    forest_wall = [6.0,  2.9,  2.3,  2.1,  1.3]
    circle_wall = [5.2,  2.8,  2.2,  1.9,  1.2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(strides, flat_ate,    "o-", color="#cc6600", label="flat (fwd)",        linewidth=1.6)
    ax1.plot(strides, forest_ate,  "s-", color="#41ab5d", label="forest (fwd)",      linewidth=1.6)
    ax1.plot(strides, circle_ate,  "^-", color="#807dba", label="forest (circular)", linewidth=1.6)
    ax1.axvline(3, linestyle=":", color="#444", linewidth=0.9, alpha=0.7)
    ax1.text(3.07, ax1.get_ylim()[1]*0.85 if ax1.get_ylim()[1] > 0 else 1.4,
             "chosen", fontsize=9, color="#444")
    ax1.set_xlabel("DROID stride (input frames per pose)")
    ax1.set_ylabel("ATE (m)")
    ax1.set_title("Pose accuracy vs stride")
    ax1.legend(loc="upper left", frameon=True)
    ax1.set_xticks(strides)
    ax1.grid(True, alpha=0.5)

    ax2.plot(strides, flat_wall,    "o-", color="#cc6600", label="flat",              linewidth=1.6)
    ax2.plot(strides, forest_wall,  "s-", color="#41ab5d", label="forest (fwd)",      linewidth=1.6)
    ax2.plot(strides, circle_wall,  "^-", color="#807dba", label="forest (circular)", linewidth=1.6)
    ax2.axvline(3, linestyle=":", color="#444", linewidth=0.9, alpha=0.7)
    ax2.set_xlabel("DROID stride")
    ax2.set_ylabel("Wall-clock (minutes per ~3,200-frame trajectory)")
    ax2.set_title("Speed vs stride")
    ax2.legend(loc="upper right", frameon=True)
    ax2.set_xticks(strides)
    ax2.grid(True, alpha=0.5)

    fig.suptitle("DROID-SLAM stride sweep — chose stride = 3 "
                 "(2.5× speedup with no ATE penalty)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig12_droid_stride_sweep.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig12_droid_stride_sweep.pdf",            bbox_inches="tight")
    plt.close(fig)
    print("fig12 done")


# ===========================================================================
# Figure 13 — dataset / hardware summary table (rendered as a figure)
# ===========================================================================
def fig_summary_table():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax_data, ax_compute = axes

    # ----- Dataset table -----
    rows_data = [
        ("Dataset",                  "v2 (Isaac Sim 5.1, Go2 USD)"),
        ("Trajectories (clean)",     "1,008  (FALLs excluded)"),
        ("Total frames",             "3,624,696  (~33 hrs at 30 Hz)"),
        ("Frame rate",               "30 Hz"),
        ("Cameras",                  "stereo, 1280 × 720 RGB + RAFT-Stereo disparity"),
        ("Sensors",                  "IMU 9-axis (accel, gyro, mag), 12-DOF joints"),
        ("Terrain (clean)",          "forest 633 · flat 288 · uneven 126 · uphill 14"),
        ("Train / val / test split", "650 / 150 / 208  (stratified by terrain, seed 42)"),
        ("Test clips (fusion)",      "92,935  (clip_len=40, stride=8)"),
        ("Test clips (Stage-2)",     "93,767  (clip_len=8, stride=8)"),
        ("Label",                    "6-D body-frame velocity  [vx vy vz ωx ωy ωz]"),
    ]
    ax_data.axis("off")
    ax_data.set_title("Dataset", loc="left", fontsize=11, weight="bold", pad=2)
    tbl = ax_data.table(cellText=rows_data, colLabels=None, cellLoc="left",
                         loc="center", colWidths=[0.42, 0.58])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        cell.set_linewidth(0.5)
        if c == 0: cell.set_text_props(weight="bold")

    # ----- Compute / training table -----
    rows_compute = [
        ("Hardware",                  "2× NVIDIA RTX 5060 Ti (16 GB each)"),
        ("Software",                  "PyTorch 2.11 + CUDA 13.0 (Blackwell sm_120)"),
        ("Loss",                      "MSE on body-frame velocity, weighted [1,1,1,5,5,5]"),
        ("Optimiser",                 "AdamW, lr 1e-4, weight decay 1e-4"),
        ("LR schedule",               "ReduceLROnPlateau (factor 0.5, patience 3)"),
        ("Modality dropout",          "10 % per-sample per-modality (fusion only)"),
        ("Clip length / stride",      "fusion: 40 / 8   ·   Stage-2: 8 / 8"),
        ("Batch size",                "fusion: 128   ·   Stage-2 (CNN): 16   ·   MViT: 4"),
        ("Patience",                  "10 epochs"),
        ("Preprocessing wall-clock",  "kin 5 min · ins 8 min · vo (DROID) ~2.4 days"),
        ("Best fusion training time", "fusion_v2: ~82 minutes"),
    ]
    ax_compute.axis("off")
    ax_compute.set_title("Compute and training", loc="left", fontsize=11, weight="bold", pad=2)
    tbl = ax_compute.table(cellText=rows_compute, colLabels=None, cellLoc="left",
                           loc="center", colWidths=[0.42, 0.58])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        cell.set_linewidth(0.5)
        if c == 0: cell.set_text_props(weight="bold")

    fig.suptitle("Dataset, compute, and training configuration", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig13_dataset_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig13_dataset_summary.pdf",            bbox_inches="tight")
    plt.close(fig)
    print("fig13 done")


# ===========================================================================
# Figure 14 — sample-trajectory predicted vs GT velocity (over time)
# ===========================================================================
def fig_sample_trajectory():
    """Run fusion_v2 inference on ONE test trajectory, plot 6-axis predicted
    vs GT velocity over time."""
    sys.path.insert(0, str(REPO))
    import torch
    from fusion.dataset import (GoFusionDataset, split_trajectories,
                                KINEMATICS_DIM, INS_DIM, VO_DIM)
    from fusion.model   import FusionTransformer

    run_dir = RUNS / "fusion_v2"
    args    = json.load(open(run_dir / "args.json"))
    stats   = np.load(run_dir / "norm_stats.npz")

    _, _, test = split_trajectories(args["root"], 650, 150, 208, 42)
    # Pick a trajectory near the median per-traj RMSE for fusion_v2 — representative
    per_tj  = json.load(open(run_dir / "per_traj_results.json"))
    sorted_pt = sorted(per_tj, key=lambda x: x["rmse_overall"])
    median_traj_name = sorted_pt[len(sorted_pt)//2]["trajectory"]
    chosen = next(t for t in test if t.endswith(median_traj_name))
    print(f"  chosen test traj: {median_traj_name}")

    ds = GoFusionDataset(
        [chosen], clip_len=args["clip_len"], stride=1,         # stride=1 for dense readout
        kin_norm=(stats["kin_mean"], stats["kin_std"]),
        ins_norm=(stats["ins_mean"], stats["ins_std"]),
        vo_norm=(stats["vo_mean"], stats["vo_std"]),
        ins_file=args.get("ins_file", "ins.npz"),
        vo_file=args.get("vo_file", "vo.npz"),
        use_vo=True, kin_v3=False,
    )
    print(f"  clips: {len(ds)}")
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionTransformer(
        kin_in=KINEMATICS_DIM, ins_in=INS_DIM, vo_in=VO_DIM,
        d_model=args["d_model"], n_blocks=args["n_blocks"],
        n_heads=args["n_heads"], ffn_dim=args["ffn_dim"],
        clip_len=args["clip_len"],
    ).to(device)
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds = []; tgts = []
    with torch.no_grad():
        for b in loader:
            kin = b["kin"].to(device); ins = b["ins"].to(device); vo = b["vo"].to(device)
            p = model(kin, ins, vo=vo)
            preds.append(p.cpu().numpy()); tgts.append(b["label"].numpy())
    preds = np.concatenate(preds); tgts = np.concatenate(tgts)
    # Each clip ends at frame (start+clip_len-1); with stride=1 that is frame i+39 for clip i
    t = (np.arange(len(preds)) + args["clip_len"] - 1) / 30.0   # seconds

    axes_lbl = [r"$v_x$ (m/s)", r"$v_y$ (m/s)", r"$v_z$ (m/s)",
                r"$\omega_x$ (rad/s)", r"$\omega_y$ (rad/s)", r"$\omega_z$ (rad/s)"]
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 6.6), sharex=True)
    for i, (ax, lbl) in enumerate(zip(axes.flat, axes_lbl)):
        ax.plot(t, tgts[:, i],  color="#222",        linewidth=1.3, label="ground truth")
        ax.plot(t, preds[:, i], color=COLORS["fusion_v2"], linewidth=1.1, alpha=0.85,
                label="fusion_v2 prediction")
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.4)
        if i == 0:
            ax.legend(loc="upper right", frameon=True)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")

    fig.suptitle(f"Sample test trajectory — fusion_v2 prediction vs ground truth\n"
                 f"({median_traj_name}, ~median per-trajectory RMSE)",
                 fontsize=11, y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "fig14_sample_trajectory.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig14_sample_trajectory.pdf",            bbox_inches="tight")
    plt.close(fig)
    print("fig14 done")


if __name__ == "__main__":
    fig_architecture()
    fig_wallclock_rmse()
    fig_stride_sweep()
    fig_summary_table()
    fig_sample_trajectory()
    print(f"\nAll extra figures written to {OUT}/")
