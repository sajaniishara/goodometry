"""Sweep stride ∈ {1, 3, 4, 5, 8} on three pilot trajectories, report ATE/RPE vs GT."""
import os, sys, time, json
import numpy as np

sys.path.insert(0, "/home/anyone/projects/goodometry")
from vo.droid_runner import run_droid
from vo.gt import load_gt
from vo.eval import ate, rpe, quat_wxyz_to_R

TRAJS = [
    "/mnt/data/go2_research_dataset_v2/trajectory_000717_20260325_083252",  # flat / forward
    "/mnt/data/go2_research_dataset_v2/trajectory_000697_20260325_050838",  # forest / forward
    "/mnt/data/go2_research_dataset_v2/trajectory_000206_20260321_202202",  # forest / circular
]
INTRIN = (914.0, 914.0, 640.0, 360.0)
STRIDES = [1, 3, 4, 5, 8]

results = []
for traj in TRAJS:
    name = os.path.basename(traj)
    gt = load_gt(traj)
    pred_pos_baseline = None  # for stride=1 reference
    for stride in STRIDES:
        print(f"\n=== {name}  stride={stride} ===", flush=True)
        t0 = time.time()
        try:
            out = run_droid(traj, INTRIN, stride=stride, fast=True)
            wall = time.time() - t0
            n = len(out["frame_idx"])
            pred_pos = out["pose_mat"][:, :3, 3]
            pred_R = out["pose_mat"][:, :3, :3]
            gt_pos = gt["pos"][:n]
            gt_R = quat_wxyz_to_R(gt["quat_wxyz"][:n])
            metrics = {
                **ate(pred_pos, gt_pos, with_scale=True),
                **rpe(pred_pos, pred_R, gt_pos, gt_R, delta_frames=30),  # 1 s
                "n_droid_frames": out.get("n_droid_frames", n),
                "wall_s": wall,
                "fps_effective": n / max(wall, 1e-6),
            }
            print(f"  n={n}  droid_n={metrics['n_droid_frames']}  wall={wall/60:.1f}min  fps={metrics['fps_effective']:.1f}")
            print(f"  ATE={metrics['ate_rmse_m']:.3f}m  scale={metrics['scale']:.3f}  "
                  f"RPE_t_1s={metrics['rpe_trans_rmse_30f_m']:.3f}m  "
                  f"RPE_r_1s={metrics['rpe_rot_rmse_30f_deg']:.1f}°")
            results.append({
                "trajectory": name,
                "stride": stride,
                **{k: float(v) if isinstance(v, (int, float, np.floating)) else v
                   for k, v in metrics.items() if not isinstance(v, list)},
            })
        except Exception as e:
            print(f"  FAIL: {e!r}")
            results.append({"trajectory": name, "stride": stride, "error": repr(e)})

with open("/home/anyone/projects/goodometry/pilot/stride_sweep_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\n\n=== SUMMARY ===")
print(f"{'trajectory':>15s}  {'str':>3s}  {'wall':>6s}  {'fps':>5s}  {'ATE_m':>7s}  {'RPE_t':>6s}  {'RPE_r':>6s}")
for r in results:
    if "error" in r:
        print(f"{r['trajectory'][-15:]:>15s}  {r['stride']:>3d}  ERROR  {r['error'][:50]}")
        continue
    print(f"{r['trajectory'][-15:]:>15s}  {r['stride']:>3d}  "
          f"{r['wall_s']/60:>5.1f}m  {r['fps_effective']:>5.1f}  "
          f"{r['ate_rmse_m']:>7.4f}  {r['rpe_trans_rmse_30f_m']:>6.3f}  {r['rpe_rot_rmse_30f_deg']:>6.1f}")

print(f"\nResults saved to pilot/stride_sweep_results.json")
