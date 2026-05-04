"""Generate publication-quality thesis figures.

Reads per-model test results, per-trajectory JSON, and training CSVs from
runs/, plus the single-modality baseline JSON, and writes 8 figures (each
as both PNG @ 300 DPI and PDF) to figures/.

Run from repo root:
    python scripts/make_thesis_figures.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


REPO   = Path("/home/anyone/projects/goodometry")
RUNS   = REPO / "runs"
OUT    = REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style — publication / thesis
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":         "serif",
    "font.size":           10,
    "axes.titlesize":      11,
    "axes.labelsize":      10,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "legend.fontsize":     9,
    "figure.titlesize":    12,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "axes.axisbelow":      True,
    "grid.color":          "#e6e6e6",
    "grid.linewidth":      0.6,
    "lines.linewidth":     1.6,
    "savefig.bbox":        "tight",
    "savefig.dpi":         300,
})

# Consistent colors per model — used across every figure
COLORS = {
    "Stage-2 v2":     "#888888",
    "fusion_v1":      "#9ecae1",
    "fusion_v1_marg": "#6baed6",
    "fusion_v2":      "#08519c",   # best — dark blue
    "fusion_v2_marg": "#2171b5",
    "fusion_v3":      "#fd8d3c",   # warning — orange
    "fusion_tcn":     "#41ab5d",   # green
    "fusion_mvit":    "#807dba",   # purple
    "kin only":       "#bcbddc",
    "IMU only":       "#fdae6b",
    "VO only":        "#fdd0a2",
}

AXIS_LABELS = [r"$v_x$", r"$v_y$", r"$v_z$",
               r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_test(run):
    return json.load(open(RUNS / run / "test_results.json"))


def load_per_traj(run):
    return json.load(open(RUNS / run / "per_traj_results.json"))


def load_train_log(run):
    return pd.read_csv(RUNS / run / "train_log.csv")


fusion_runs = ["fusion_v1", "fusion_v1_marg", "fusion_v2",
               "fusion_v2_marg", "fusion_v3", "fusion_tcn", "fusion_mvit"]

tests   = {r: load_test(r)     for r in fusion_runs}
# per_traj_results.json was added later — fusion_v1/fusion_v1_marg don't have it
per_tj  = {r: load_per_traj(r) for r in fusion_runs
           if (RUNS / r / "per_traj_results.json").exists()}
logs    = {r: load_train_log(r) for r in fusion_runs}
basel   = json.load(open(RUNS / "single_modality_baselines" / "results.json"))

# Stage-2 v2 — manually keyed (no JSON)
STAGE2_V2 = {
    "rmse_overall": 0.1168,
    "rmse_lin":     0.0893,
    "rmse_ang":     0.1389,
    "rmse_per_axis":[0.1020, 0.1031, 0.0537, 0.1887, 0.1171, 0.0928],
    "params":       33485894,
    "train_time_min": 5 * 24 * 60,   # ~5 days in minutes
    "frame_lin":    "world",
}

PARAMS = {
    "Stage-2 v2":     33485894,
    "fusion_v1":      437382,
    "fusion_v1_marg": 437382,
    "fusion_v2":      455046,
    "fusion_v2_marg": 455046,
    "fusion_v3":      455558,
    "fusion_tcn":     1023798,
    "fusion_mvit":    758566,
}

TRAIN_MIN = {
    "Stage-2 v2":     5 * 24 * 60,
    "fusion_v1":      37,
    "fusion_v1_marg": 38,
    "fusion_v2":      82,
    "fusion_v2_marg": 60,
    "fusion_v3":      48,
    "fusion_tcn":     88,
    "fusion_mvit":    172,
}


def overall(model):
    if model == "Stage-2 v2":
        return STAGE2_V2["rmse_overall"]
    return tests[model]["rmse_overall"]


def lin(model):
    if model == "Stage-2 v2":
        return STAGE2_V2["rmse_lin"]
    return tests[model]["rmse_lin"]


def ang(model):
    if model == "Stage-2 v2":
        return STAGE2_V2["rmse_ang"]
    return tests[model]["rmse_ang"]


def per_axis(model):
    if model == "Stage-2 v2":
        return STAGE2_V2["rmse_per_axis"]
    return tests[model]["rmse_per_axis"]


# ===========================================================================
# Figure 1 — overall RMSE bar chart, all models
# ===========================================================================
def fig1_overall_rmse():
    models = ["Stage-2 v2", "fusion_v1", "fusion_v1_marg",
              "fusion_v3", "fusion_mvit", "fusion_tcn",
              "fusion_v2_marg", "fusion_v2"]
    # Sort by overall RMSE ascending so best is at the top
    models = sorted(models, key=overall)
    rmse   = [overall(m) for m in models]
    colors = [COLORS[m]  for m in models]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    bars = ax.barh(models, rmse, color=colors, edgecolor="white")
    for b, v in zip(bars, rmse):
        ax.text(v + 0.0015, b.get_y() + b.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8.5)
    # Highlight fusion_v2
    for b, m in zip(bars, models):
        if m == "fusion_v2":
            b.set_edgecolor("black")
            b.set_linewidth(1.2)

    ax.set_xlabel("Overall test RMSE (208 trajectories)")
    ax.set_title("Overall RMSE — all models on the matched 208-trajectory test set")
    ax.set_xlim(0, max(rmse) * 1.15)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    fig.savefig(OUT / "fig1_overall_rmse.png")
    fig.savefig(OUT / "fig1_overall_rmse.pdf")
    plt.close(fig)


# ===========================================================================
# Figure 2 — per-axis RMSE grouped bars (key models)
# ===========================================================================
def fig2_per_axis_rmse():
    models = ["Stage-2 v2", "fusion_v1", "fusion_v1_marg", "fusion_v2",
              "fusion_tcn", "fusion_mvit"]
    n_models = len(models)
    n_axes   = 6
    width = 0.8 / n_models
    x = np.arange(n_axes)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, m in enumerate(models):
        vals = per_axis(m)
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=m, color=COLORS[m], edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(AXIS_LABELS)
    ax.set_ylabel("Per-axis RMSE")
    ax.set_title("Per-axis test RMSE — six body-frame velocity components")
    ax.legend(loc="upper right", ncol=2, frameon=True)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    fig.savefig(OUT / "fig2_per_axis_rmse.png")
    fig.savefig(OUT / "fig2_per_axis_rmse.pdf")
    plt.close(fig)


# ===========================================================================
# Figure 3 — single-modality baselines vs fusion_v2
# ===========================================================================
def fig3_single_modality_vs_fusion():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.6))

    # --- Linear panel ---
    kin = basel["baselines"]["kin_only"]
    lin_fusion = lin("fusion_v2")
    bars = ax1.bar(["kin only", "fusion_v2"],
                    [kin["lin_rmse"], lin_fusion],
                    color=[COLORS["kin only"], COLORS["fusion_v2"]],
                    edgecolor="white", width=0.55)
    for b, v in zip(bars, [kin["lin_rmse"], lin_fusion]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.005,
                 f"{v:.4f}", ha="center", fontsize=9)
    ax1.set_ylabel("Linear RMSE (m/s)")
    ax1.set_title("Linear — kin alone vs fusion")
    ax1.set_ylim(0, kin["lin_rmse"] * 1.18)
    pct = 100 * (kin["lin_rmse"] - lin_fusion) / kin["lin_rmse"]
    ax1.annotate(f"−{pct:.1f}%", xy=(0.5, kin["lin_rmse"] * 0.55),
                 ha="center", fontsize=12, color="#08519c", weight="bold")
    ax1.grid(True, axis="y"); ax1.grid(False, axis="x")

    # --- Angular panel ---
    imu = basel["baselines"]["ins_only"]
    ang_fusion = ang("fusion_v2")
    bars = ax2.bar(["IMU only", "fusion_v2"],
                    [imu["ang_rmse"], ang_fusion],
                    color=[COLORS["IMU only"], COLORS["fusion_v2"]],
                    edgecolor="white", width=0.55)
    for b, v in zip(bars, [imu["ang_rmse"], ang_fusion]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.006,
                 f"{v:.4f}", ha="center", fontsize=9)
    ax2.set_ylabel("Angular RMSE (rad/s)")
    ax2.set_title("Angular — IMU alone vs fusion")
    ax2.set_ylim(0, imu["ang_rmse"] * 1.18)
    pct = 100 * (imu["ang_rmse"] - ang_fusion) / imu["ang_rmse"]
    ax2.annotate(f"−{pct:.1f}%", xy=(0.5, imu["ang_rmse"] * 0.55),
                 ha="center", fontsize=12, color="#08519c", weight="bold")
    ax2.grid(True, axis="y"); ax2.grid(False, axis="x")

    fig.suptitle("Single-modality baselines vs learned fusion (fusion_v2)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig3_single_modality_vs_fusion.png")
    fig.savefig(OUT / "fig3_single_modality_vs_fusion.pdf")
    plt.close(fig)


# ===========================================================================
# Figure 4 — modality ablation
# ===========================================================================
def fig4_modality_ablation():
    """fusion_v1 (kin+IMU) → fusion_v1_marg (+drift-free yaw) →
       fusion_v2 (+VO) → fusion_v2_marg (MARG+VO redundancy)"""
    models = ["fusion_v1", "fusion_v1_marg", "fusion_v2", "fusion_v2_marg"]
    short  = ["kin + IMU", "kin + MARG", "kin + IMU + VO", "kin + MARG + VO"]

    metrics = {
        "Overall RMSE":         [overall(m) for m in models],
        "Linear RMSE (m/s)":    [lin(m)     for m in models],
        "Angular RMSE (rad/s)": [ang(m)     for m in models],
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    x = np.arange(len(models))
    for ax, (k, vals) in zip(axes, metrics.items()):
        bars = ax.bar(x, vals, color=[COLORS[m] for m in models], edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
                    f"{v:.4f}", ha="center", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=8.5, rotation=20, ha="right")
        ax.set_title(k)
        ax.grid(True, axis="y"); ax.grid(False, axis="x")
        ax.set_ylim(0, max(vals) * 1.20)

    fig.suptitle("Modality ablation — what each modality contributes",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_modality_ablation.png")
    fig.savefig(OUT / "fig4_modality_ablation.pdf")
    plt.close(fig)


# ===========================================================================
# Figure 5 — parameter efficiency
# ===========================================================================
def fig5_param_efficiency():
    models = list(PARAMS.keys())
    rmse   = [overall(m) for m in models]
    params = [PARAMS[m]  for m in models]
    cols   = [COLORS[m]  for m in models]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for m, r, p, c in zip(models, rmse, params, cols):
        ax.scatter(p, r, color=c, edgecolor="black", s=160, linewidths=0.7, zorder=3)

    # Pixel-offset annotations (works correctly under log scale)
    offsets = {
        "Stage-2 v2":     (-12, -10, "right"),
        "fusion_v1":      (10, 4, "left"),
        "fusion_v1_marg": (10, 4, "left"),
        "fusion_v2":      (10, -10, "left"),
        "fusion_v2_marg": (10, 4, "left"),
        "fusion_v3":      (10, 4, "left"),
        "fusion_tcn":     (10, -10, "left"),
        "fusion_mvit":    (10, 4, "left"),
    }
    for m, p, r in zip(models, params, rmse):
        dx, dy, ha = offsets.get(m, (10, 4, "left"))
        ax.annotate(m, xy=(p, r), xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8.8, ha=ha, va="center")

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Overall test RMSE")
    ax.set_title("Parameter efficiency — RMSE vs model size")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}K"))

    # Pareto reference line at fusion_v2's RMSE
    ax.axhline(overall("fusion_v2"), color="#08519c", linestyle="--",
               linewidth=0.8, alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.5, overall("fusion_v2") - 0.003,
            "fusion_v2 RMSE = 0.0737", fontsize=8, color="#08519c", ha="right")

    # Add a bit of padding so labels don't touch edges
    ax.set_xlim(2e5, 1.5e8)
    ax.set_ylim(0.06, 0.13)
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_param_efficiency.png")
    fig.savefig(OUT / "fig5_param_efficiency.pdf")
    plt.close(fig)


# ===========================================================================
# Figure 6 — validation loss curves
# ===========================================================================
def fig6_val_loss_curves():
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    # More distinguishable order/styles
    styles = {
        "fusion_v1":      ("--", 1.4),
        "fusion_v1_marg": ("--", 1.4),
        "fusion_v2":      ("-",  2.0),
        "fusion_v2_marg": ("-",  1.4),
        "fusion_v3":      ("-",  1.4),
        "fusion_tcn":     ("-",  1.8),
        "fusion_mvit":    ("-",  1.6),
    }
    for m in fusion_runs:
        df = logs[m]
        ls, lw = styles[m]
        ax.plot(df["epoch"], df["val_loss"], label=m,
                color=COLORS[m], linewidth=lw, linestyle=ls, alpha=0.95)
        # Mark best epoch
        best_idx = df["val_loss"].idxmin()
        ax.scatter(df["epoch"].iloc[best_idx], df["val_loss"].iloc[best_idx],
                   color=COLORS[m], s=44, zorder=5,
                   edgecolor="black", linewidths=0.6)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss (MSE, log scale)")
    ax.set_title("Validation loss during training — all fusion variants  "
                 "(• = best epoch, dashed = no VO)")
    ax.set_yscale("log")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_val_loss_curves.png")
    fig.savefig(OUT / "fig6_val_loss_curves.pdf")
    plt.close(fig)


# ===========================================================================
# Figure 7 — per-trajectory CDF
# ===========================================================================
def fig7_per_traj_cdf():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in ["fusion_v2", "fusion_v3", "fusion_tcn", "fusion_mvit"]:
        if m not in per_tj:
            continue
        rs = sorted(t["rmse_overall"] for t in per_tj[m])
        cdf = np.arange(1, len(rs) + 1) / len(rs)
        ax.plot(rs, cdf, label=m, color=COLORS[m], linewidth=1.6)

    # Reference lines
    ax.axhline(0.5, color="#bbbbbb", linestyle=":", linewidth=0.8)
    ax.axhline(0.95, color="#bbbbbb", linestyle=":", linewidth=0.8)
    ax.text(ax.get_xlim()[1] * 0.7, 0.51, "median", fontsize=8, color="#666666")
    ax.text(ax.get_xlim()[1] * 0.7, 0.96, "P95", fontsize=8, color="#666666")

    ax.set_xlabel("Per-trajectory overall RMSE")
    ax.set_ylabel("Cumulative fraction of trajectories")
    ax.set_title("Per-trajectory error distribution (208 test trajectories)")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.5)
    fig.savefig(OUT / "fig7_per_traj_cdf.png")
    fig.savefig(OUT / "fig7_per_traj_cdf.pdf")
    plt.close(fig)


# ===========================================================================
# Figure 8 — architecture comparison (fusion_v2 vs tcn vs mvit)
# ===========================================================================
def fig8_architecture_comparison():
    models = ["fusion_v2", "fusion_tcn", "fusion_mvit"]
    n_models = len(models)
    width = 0.25
    x = np.arange(6)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4),
                              gridspec_kw={"width_ratios": [3, 1.0]})
    ax1, ax2 = axes

    # Per-axis bars
    for i, m in enumerate(models):
        vals = per_axis(m)
        offset = (i - 1) * width
        ax1.bar(x + offset, vals, width, label=m, color=COLORS[m],
                edgecolor="white", linewidth=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(AXIS_LABELS)
    ax1.set_ylabel("Per-axis RMSE")
    ax1.set_title("Architecture comparison — per-axis")
    ax1.legend(loc="upper right", frameon=True, ncol=1)
    ax1.grid(True, axis="y"); ax1.grid(False, axis="x")

    # Aggregate bars (overall, lin, ang)
    metrics = ["Overall", "Linear", "Angular"]
    vals = {m: [overall(m), lin(m), ang(m)] for m in models}
    n_models = len(models); width2 = 0.25
    x2 = np.arange(len(metrics))
    for i, m in enumerate(models):
        ax2.bar(x2 + (i - 1) * width2, vals[m], width2, color=COLORS[m],
                edgecolor="white", linewidth=0.4)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics, fontsize=8.5)
    ax2.set_ylabel("RMSE")
    ax2.set_title("Aggregate")
    ax2.grid(True, axis="y"); ax2.grid(False, axis="x")

    fig.tight_layout()
    fig.savefig(OUT / "fig8_architecture_comparison.png")
    fig.savefig(OUT / "fig8_architecture_comparison.pdf")
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    fig1_overall_rmse();             print("fig1 done")
    fig2_per_axis_rmse();            print("fig2 done")
    fig3_single_modality_vs_fusion();print("fig3 done")
    fig4_modality_ablation();        print("fig4 done")
    fig5_param_efficiency();         print("fig5 done")
    fig6_val_loss_curves();          print("fig6 done")
    fig7_per_traj_cdf();             print("fig7 done")
    fig8_architecture_comparison();  print("fig8 done")
    print(f"\nAll figures written to {OUT}/")
