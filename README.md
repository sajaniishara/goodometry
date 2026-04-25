# goodometry

Multimodal neural fusion pipeline for **Unitree Go2 6-DoF body-state estimation**, sibling project to [`go2-research`](https://github.com/sajaniishara/go2-research).

The approach: **classical per-sensor preprocessing → small fusion transformer**, instead of end-to-end CNN/MViT on raw stereo + raw sensors. Beats the v2 Stage-2 CNN baseline by 20–35 % on test RMSE while being **77× smaller** and trained **~200× faster**.

```
sensors/                 preprocessing                     fusion model
─────────────────────────────────────────────────────────────────────
stereo cameras    →   DROID-SLAM         →  vo.npz         ┐
joint encoders    →   pytorch_kinematics →  kin.npz        ├→ FusionTransformer → 6-DoF body velocity
IMU (+ mag)       →   imufusion Madgwick →  ins.npz        ┘  (factorized modal-then-temporal,
GT poses          →   R_wb^T rotation    →  labels_body.npz    437 K params, ~1.3 s context)
FR_calf bias      →   per-traj offset    →  calibration.npz
```

## Documentation

- **[`PIPELINE.md`](./PIPELINE.md)** — comprehensive project reference (motivation, architecture, every preprocessing arm with frame conventions and gotchas, fusion model spec, results, code organisation, reproducibility commands).
- **[`EXPERIMENTS.md`](./EXPERIMENTS.md)** — focused configurations + comparison results doc (training hyperparameters, test split details, all evaluations, summary tables).
- **`godometry/`** — the original two design docs that motivated the pivot (`go2-preprocessing-options.md`, `go2-concrete-algorithms.md`).
- The corresponding session log lives in [`go2-research/CHANGES.md`](https://github.com/sajaniishara/go2-research/blob/master-goodometry/CHANGES.md), Sessions 20–22.

## Headline result (held-out test, 71 trajectories neither model trained on)

| metric | Stage-2 CNN RGB | **fusion_v1** | Δ |
|---|---:|---:|---:|
| Parameters | 33,485,894 | **437,382** | 77× smaller |
| Train wall-clock | ~5 days | **~37 min** | ~200× faster |
| Visual input | RGB stereo | none yet | — |
| Overall RMSE | 0.1015 | **0.0809** | −20.3 % |
| Linear RMSE (m/s) | 0.0907 | **0.0589** | −35.0 % |
| Angular RMSE (rad/s) | 0.1112 | **0.0981** | −11.8 % |

`fusion_v2` (with VO modality) is queued — DROID-SLAM at-scale precompute is in progress.

## Repository layout

```
goodometry/
├── README.md / PIPELINE.md / EXPERIMENTS.md
├── godometry/        original design docs
├── configs/          camera intrinsics, FR_calf calibration summary
├── calibration/      FR_calf per-trajectory offset loader
├── kinematics/       FK + finite-diff foot velocity + contact + leg odometry
├── ins/              imufusion Madgwick (IMU-only and MARG stitched)
├── labels.py         body-frame label rotation
├── vo/               DPVO + DROID-SLAM runners + GT loader + eval (ATE/RPE)
├── fusion/           dataset, FusionTransformer, train, evaluate
├── scripts/          env setup + at-scale runners + precompute + launchers
└── pilot/            smoke tests for each arm
```

`third_party/`, `runs/`, `logs/` are gitignored. See `scripts/env.sh` for the activation environment (reuses `~/projects/isaac/env_isaaclab` with torch 2.11+cu130).

## Reproducing

```bash
source scripts/env.sh

# one-time preprocessing (writes <traj>/{calibration,kin,ins,ins_marg,labels_body}.npz):
python scripts/compute_calf_calibration.py
python scripts/run_kin_at_scale.py
python scripts/run_ins_at_scale.py             # IMU-only ins.npz
python scripts/run_ins_at_scale.py --with-mag  # MARG ins_marg.npz
python scripts/precompute_labels_body.py

# train fusion_v1:
python -u fusion/train.py --output-dir runs/fusion_v1 --device cuda \
    --epochs 50 --batch-size 128 --num-workers 4 --patience 10

# test set:
python fusion/evaluate.py --run-dir runs/fusion_v1 --device cuda

# fair head-to-head vs Stage-2:
python scripts/fair_test_eval.py
```
