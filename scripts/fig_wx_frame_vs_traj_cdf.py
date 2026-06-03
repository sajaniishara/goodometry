#!/usr/bin/env python3
"""Figure: fusion_v2 ω_x (roll) error — per-frame vs. per-trajectory CDFs.

Overlays two cumulative distributions of fusion_v2's roll-rate error to show
that the elevated ω_x error is shaped by *frame-level* variance — a heavy tail
across individual clips — rather than by a handful of bad trajectories:

  1. Per-frame |ω_x error|  over all ~92,935 test clips     -> heavy-tailed.
  2. Per-trajectory ω_x RMSE over the 208 test trajectories -> steep & tight.

The visual contrast (long tail on the frame curve, tight rise on the trajectory
curve) is the point: the error lives in scattered hard frames, not in a few
bad trajectories.

fusion_v2 only (fusion_v3 intentionally excluded).

Inputs — produced by `python fusion/evaluate.py --run-dir runs/fusion_v2`:
  runs/fusion_v2/per_traj_results.json  per-trajectory metrics; this script
                                        reads rmse_per_axis[3] (index 3 = ω_x).
  runs/fusion_v2/per_clip_wx_err.npy    signed per-clip ω_x error
                                        (preds[:, 3] - tgts[:, 3]).

Run from anywhere:
    python scripts/fig_wx_frame_vs_traj_cdf.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RUN  = REPO / "runs" / "fusion_v2"
OUT  = REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)

WX = 3  # index of ω_x in the 6-DoF [vx, vy, vz, ωx, ωy, ωz] vector

# Style — matches scripts/make_thesis_figures.py
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.color":        "#e6e6e6",
    "grid.linewidth":    0.6,
    "lines.linewidth":   1.6,
    "savefig.bbox":      "tight",
    "savefig.dpi":       300,
})

FRAME_COLOR = "#08519c"   # fusion_v2 dark blue
TRAJ_COLOR  = "#a50f15"   # dark red — distinct, not tied to any other model


def cdf(values):
    v = np.sort(np.asarray(values, dtype=float))
    return v, np.arange(1, len(v) + 1) / len(v)


def main():
    clip_path = RUN / "per_clip_wx_err.npy"
    traj_path = RUN / "per_traj_results.json"
    if not clip_path.exists():
        sys.exit(f"missing {clip_path}\n"
                 "Run `python fusion/evaluate.py --run-dir runs/fusion_v2` "
                 "(patched to dump per_clip_wx_err.npy) first.")
    if not traj_path.exists():
        sys.exit(f"missing {traj_path}\n"
                 "Run `python fusion/evaluate.py --run-dir runs/fusion_v2` first.")

    frame = np.abs(np.load(clip_path))
    per_tj = json.load(open(traj_path))
    traj   = np.array([t["rmse_per_axis"][WX] for t in per_tj])

    fx, fy = cdf(frame)
    tx, ty = cdf(traj)

    med = float(np.median(frame))
    p95 = float(np.percentile(frame, 95))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(fx, fy, color=FRAME_COLOR, linewidth=1.8,
            label=rf"Per-frame $|\omega_x$ error$|$  (N={len(frame):,} clips)")
    ax.plot(tx, ty, color=TRAJ_COLOR, linewidth=1.8,
            label=rf"Per-trajectory $\omega_x$ RMSE  (N={len(traj)} trajs)")

    # Reference quantile lines + annotated frame-curve median / P95.
    ax.axhline(0.50, color="#cccccc", linestyle=":", linewidth=0.8)
    ax.axhline(0.95, color="#cccccc", linestyle=":", linewidth=0.8)
    for val, q, txt in [(med, 0.50, f"median  {med:.4f}"),
                        (p95, 0.95, f"P95  {p95:.4f}")]:
        ax.plot([val], [q], "o", color=FRAME_COLOR, markersize=5, zorder=5)
        ax.annotate(txt, (val, q), textcoords="offset points", xytext=(9, -11),
                    fontsize=8, color=FRAME_COLOR)
    ax.annotate(rf"P95 / median $\approx$ {p95 / med:.1f}$\times$",
                (p95, 0.95), textcoords="offset points", xytext=(9, 8),
                fontsize=8, color=FRAME_COLOR)

    ax.set_xlabel(r"$\omega_x$ (roll) error  (rad/s)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(r"fusion_v2 $\omega_x$ error: frame-level variance vs. "
                 "per-trajectory spread")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)

    fig.savefig(OUT / "fig_wx_frame_vs_traj_cdf.png")
    fig.savefig(OUT / "fig_wx_frame_vs_traj_cdf.pdf")
    plt.close(fig)
    print(f"frame: median={med:.4f}  P95={p95:.4f}  ratio={p95 / med:.2f}x")
    print(f"traj : median={np.median(traj):.4f}  P95={np.percentile(traj, 95):.4f}")
    print(f"Saved {OUT / 'fig_wx_frame_vs_traj_cdf.png'}")


if __name__ == "__main__":
    main()
