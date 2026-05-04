# Goodometry — Change Log

All sessions for the `~/projects/goodometry/` pipeline.  
Companion project log lives in `~/projects/go2_research/CHANGES.md` (Sessions 1–22).

---

## Session 22 — 2026-04-24 · Initial goodometry setup

**Goal:** Stand up the goodometry pipeline from scratch: preprocessing arms, fusion model, training.

### What was done

1. **FR_calf calibration** (`calibration/`, `scripts/compute_calf_calibration.py`)
   - Discovered that `sensors.npz['joints']` channel 9 (FR_calf) is systematically biased across all 1,008 trajectories by a per-trajectory mean of ~0.6 rad vs the other three calves.
   - Root cause: Go2 USD asset in Isaac Sim 5.1 — bad FR_calf collision mesh lets the foot penetrate the ground under load, causing Kp-driven calf compression.
   - Fix: per-trajectory static offset `offset = mean(FL, RL, RR calf) − mean(FR calf)`, applied at load time via `calibration.load_calibrated_joints()`. `sensors.npz` untouched.
   - Verified: foot-Z drop to −0.255 m ± 0.027 (was −0.225 raw, expected −0.260 standing).

2. **Body-frame labels** (`labels.py`, `scripts/precompute_labels_body.py`)
   - Found that `sensors.npz['labels'][:, :3]` is world-frame linear velocity — never rotated to body frame. Fixed by rotating via GT quaternion. Angular velocity already body-frame; clipped to ±10 rad/s.
   - Output: `labels_body.npz` per trajectory.

3. **Kinematics arm** (`kinematics/`, `scripts/run_kin_at_scale.py`)
   - Forward kinematics with `pytorch_kinematics` on Go2 URDF.
   - Two index-mapping fixes: Isaac Sim grouped DOF order vs per-leg URDF order; collapsed `*_foot_joint` in URDF chain (manual −0.213 m foot offset).
   - Foot velocities by centered finite difference at 30 Hz.
   - Soft contact probability by velocity-gate: ≤0.10 m/s → 1.0, ≥0.50 m/s → 0.0.
   - Body velocity from stance-foot constraint (§1.2 of `go2-concrete-algorithms.md`).
   - **Output `kin.npz`** (31D features): `foot_pos_body (N,4,3)`, `foot_vel_body (N,4,3)`, `contact_prob (N,4)`, `v_body_legs (N,3)`.
   - Ran at scale: 1,008 trajectories, 288 s CPU, 0 failures.

4. **INS arm** (`ins/madgwick.py`, `scripts/run_ins_at_scale.py`)
   - xIOTechnologies `imufusion` Madgwick AHRS.
   - Two variants: `ins.npz` (IMU-only 6-axis) and `ins_marg.npz` (MARG 9-axis, drift-free yaw via two-pass stitch).
   - MARG fixes: mag axis remap `[N,E,D] → [E,N,U]`; imufusion MARG mode failed to converge, so external tilt-compensated yaw from IMU roll/pitch instead.
   - Tuning: `gain=1.0`, `acceleration_rejection=0.0` (default 10° triggers on gait foot-impact spikes).
   - Ran at scale: 1,008 trajectories each.

5. **Fusion model** (`fusion/model.py`)
   - `FusionTransformer`: factorized modal-then-temporal causal transformer.
   - M=2 (kin + INS): 437,382 params. d_model=128, n_blocks=2, n_heads=4, ffn_dim=256.
   - Sinusoidal temporal positional encoding + learned per-modality embedding.
   - Causal mask on temporal attention.

6. **fusion_v1 trained** (kinematics + INS, IMU-only)
   - 650/150/208 stratified split by terrain, seed=42.
   - 50 epochs max, patience=10. Early-stopped at epoch 17, best at epoch 7.
   - **Test RMSE: 0.0933 overall, 0.0637 m/s linear, 0.1156 rad/s angular.**
   - Beats Stage-2 CNN (0.1270) with 77× fewer params, 200× faster training.

7. **fusion_v1_marg trained** (kinematics + INS-MARG)
   - Best at epoch 11. **Test RMSE: 0.0844** (−9.5% vs fusion_v1).
   - Drift-free yaw helps ~8–10% across metrics, biggest gain on `ωx` (−13.9%).

8. **Fair head-to-head** (`scripts/fair_test_eval.py`)
   - Evaluated both fusion_v1 and Stage-2 on 71 trajectories held out from both training sets.
   - fusion_v1 wins by −20.3% overall, −35.0% linear, −11.8% angular.
   - Biggest single-axis win: `vy` −50.2% (lateral velocity, directly observable from FK foot positions).

---

## Session 23 — 2026-04-25 · VO arm + fusion_v2 + cnn3d split alignment

**Goal:** Finish the VO preprocessing arm at scale; train fusion_v2 (kin+INS+VO); align cnn3d split for apples-to-apples comparison.

### DROID-SLAM at scale

- Stride sweep on 3 pilot trajectories × 5 stride values: stride=3 gives best weighted-average ATE across the dataset terrain mix (62% forest, 28% flat, 10% uneven) — 8% better ATE than stride=1 at 2.7× faster wall-clock.
- Flat terrain has a cliff at stride=4 (ATE +78%); forest is forgiving up to stride=8.
- Why stride>1 can win: at 30 Hz, inter-frame motion ≈1.3 cm, smaller than feature-matching noise. Stride=3 gives ~4 cm per pair — within DROID's TartanAir training distribution.
- At-scale config: stride=3, `fast=True` (skip backend BA), parallel I/O with `IMREAD_REDUCED_COLOR_2`. Wall-clock: ~3.4 min/traj, 1,008 trajectories projected ~2.4 days.
- **Run completed: 1,008 vo.npz files.**
- `vo.npz` schema: `pose_7d (N,7)` [tx,ty,tz,qx,qy,qz,qw] cam→world, `valid (N,)`, `stride_used`, `n_droid_frames`.

### fusion_v2 (kin + INS + VO)

- Added VO modality token (7D MLP→128) to `FusionTransformer` — M=3, 455,046 params.
- VO feature: per-frame SE(3) delta in prev-camera frame computed from absolute DROID poses.
- Trained: best val_loss 0.0126 at epoch 31, ~82 min wall-clock.
- **Test RMSE: 0.0737 overall, 0.0483 m/s linear, 0.0924 rad/s angular. Current best model.**
- VO improves over fusion_v1_marg by −12.7% overall; closes the `vx`/`vz` gap (visual ego-motion axes).

### fusion_v2_marg (kin + INS-MARG + VO)

- Same as fusion_v2, `--ins-file ins_marg.npz`. Best at epoch 18, ~60 min.
- **Test RMSE: 0.0816** — *worse* than fusion_v2 on every axis.
- Finding: MARG's drift-free yaw is redundant when VO is present (VO's SE(3) deltas already carry relative rotation). IMU-only + VO = better combination.

### `evaluate.py` — per-trajectory output

- Added `per_traj_results.json` output: 208 trajectories with per-axis RMSE/P50/P95.
- Added `use_kin_v3` field read from `args.json` for forward compatibility.

### cnn3d split alignment (`go2_research/`)

- Found that `cnn3d/dataset.py` used `np.random.default_rng(42)` vs goodometry's `np.random.RandomState(42)`, and different counts (705/151/152 vs 650/150/208). Made fair comparison impossible.
- Rewrote `cnn3d/dataset.py:split_trajectories()` to match goodometry exactly: `RandomState(42)`, counts 650/149/208, same proportional formula and rounding correction.
- Updated `cnn3d/train.py`: removed `--train-frac`, `--val-frac`; added `--test-count`; defaults 650/150/208.
- Updated `cnn3d/evaluate.py`: new `split_trajectories` call signature.
- Verified: 208/208 test trajectories identical between the two pipelines.
- Re-launched Stage-2 CNN RGB training on GPU 0 with corrected split (still running).

### EXPERIMENTS.md and Excel

- Added fusion_v2 and fusion_v2_marg results to EXPERIMENTS.md §1.3/1.4/3.4/3.5.
- Updated §4.1 summary table to 6-model comparison.
- Updated Excel: `go2_model_results.xlsx` (v2 Dataset sheet) and `go2_experiment_summary.xlsx` (Model Comparison, Goodometry Results, new "fusion_v2 Per-Trajectory" sheet with 208-traj side-by-side RMSE/P50/P95).

### `launch_fusion_v2.sh` bug fix

- `grep -c` returns exit code 1 on zero matches, causing the `ERROR_COUNT` substitution to fail and produce `"0\n0"` — a `[[ ... -gt 0 ]]` syntax error. Fixed: `ERROR_COUNT=$(grep -c ...) || ERROR_COUNT=0`.

---

## Session 24 — 2026-04-30 · fusion_v3: kin_v3 SE(3) delta kinematics

**Goal:** Upgrade the kinematics branch to provide richer per-frame pose-change information (SE(3) delta instead of raw linear velocity estimate), creating fusion_v3 while preserving fusion_v2 unchanged.

### Motivation

- `v_body_legs (N,3)` in kin.npz is a per-timestep linear velocity from the stance-foot constraint — noisy at 30 Hz due to actuator jitter and finite-difference amplification.
- An SE(3) per-frame delta `[Δtx, Δty, Δtz, Δqx, Δqy, Δqz, Δqw]` encodes *both translation and rotation change* in a single compact token the transformer can attend to directly, instead of requiring the model to infer rotation change from the IMU channel.
- Replacing the 3D linear velocity with a 7D SE(3) delta changes the kin feature from 31D → 35D. All other features (foot_pos, foot_vel, contact_prob) are unchanged.

### kinematics/leg_odom.py — `leg_odom_se3_deltas()`

New function added (existing `body_velocity_from_legs` unchanged):

```python
def leg_odom_se3_deltas(v_body_legs, gyro_body, dt=1/30):
    """Per-frame SE(3) delta: Δt = v_body_legs * dt, Δq = quat_from_rotvec(gyro * dt)."""
    N = len(v_body_legs)
    out = np.zeros((N, 7), dtype=np.float32)
    out[:, 6] = 1.0                             # identity qw
    out[:, :3] = (v_body_legs * dt)             # Δt in body frame
    rvecs = (gyro_body * dt)                    # rotation vector
    angles = np.linalg.norm(rvecs, axis=1)
    valid = angles > 1e-8
    ax = rvecs[valid] / angles[valid, None]
    ha = angles[valid] * 0.5
    out[valid, 3:6] = ax * np.sin(ha)[:, None]  # qx, qy, qz
    out[valid, 6]   = np.cos(ha)                # qw
    return out
```

Rodrigues formula implemented directly (no scipy) — unit axis × sin(θ/2) for xyz, cos(θ/2) for w. Identity delta `[0,0,0, 0,0,0,1]` for near-zero rotation (|rvec| < 1e-8).

### scripts/patch_kin_leg_odom_delta.py — new

- Reads existing `kin.npz` (`v_body_legs`) + `sensors.npz` (gyro channels 3:6).
- Calls `leg_odom_se3_deltas()`, re-saves `kin.npz` with all existing keys plus `leg_odom_delta (N,7)`.
- Resume-safe: skips trajectories where `leg_odom_delta` is already in `kin.files`. `--force` overrides.
- Ran: 1,008 trajectories in 81 seconds, 0 failures.
- All 1,008 `kin.npz` files now contain both `v_body_legs` (legacy) and `leg_odom_delta` (v3).

### fusion/dataset.py — kin_v3 support

```python
KINEMATICS_DIM    = 31     # foot_pos 12 + foot_vel 12 + contact 4 + v_body_legs 3   (v1/v2)
KINEMATICS_V3_DIM = 35     # foot_pos 12 + foot_vel 12 + contact 4 + leg_odom_delta 7 (v3)
```

- Added `_build_kin_v3_features(kin, start, end)` — identical to `_build_kin_features` except the last column block is `leg_odom_delta[start:end]` (7D) instead of `v_body_legs[start:end]` (3D).
- `GoFusionDataset`: added `kin_v3: bool = False` parameter. When `True`, clip-index building skips trajectories whose `kin.npz` doesn't contain `leg_odom_delta`.
- `compute_norm_stats`: added `kin_v3` parameter, dispatches to correct builder.

### fusion/train.py — `--use-kin-v3` flag

- Added `--use-kin-v3` (action="store_true", default=False).
- `kin_dim = KINEMATICS_V3_DIM if args.use_kin_v3 else KINEMATICS_DIM`.
- Model constructed with `kin_in=kin_dim`. All else unchanged.

### fusion/evaluate.py — kin_v3 support

- Reads `use_kin_v3` from saved `args.json`, selects correct `kin_dim`.
- **Also added in this session**: `per_traj_results.json` output (208-traj per-axis RMSE/P50/P95).

### scripts/launch_fusion_v3.sh — new

- Pre-flight checks: `>= 800` vo.npz files AND `>= 800` kin.npz with `leg_odom_delta`.
- Launches with `--use-vo --use-kin-v3 --ins-file ins.npz` (IMU-only default).
- `--with-marg` flag for MARG ablation: produces `runs/fusion_v3_marg`.
- `GPU` env variable override (default 1): `GPU=1 bash scripts/launch_fusion_v3.sh`.

### RAFT-Stereo paused

- RAFT-Stereo disparity precompute (PID 157759, GPU 1, ~87% done — 879/1,008 trajectories) killed to free GPU 1 for fusion_v3.
- Resumable: `precompute_disparity_h5.py` has HDF5 skip-existing logic — re-running restarts from where it left off without re-processing completed trajectories.

### fusion_v3 training launched

- GPU 1, PID 588249, log: `logs/fusion_v3.log`.
- Split: 650 train / 149 val / 208 test, seed=42. Clips: 289,164 train / 66,037 val.
- Params: **455,558** (512 more than fusion_v2 — the extra 4 kin input channels to the projection MLP).
- Status: training in progress as of 2026-04-30.

### Design decision: edit kin branch vs separate arm

Decided to *edit* the kin branch input features (v3 flag on the same M=3 transformer) rather than add a 4th modality arm. Rationale: leg_odom_delta is the same physical signal as v_body_legs, just a richer encoding of it — not a new sensing modality. The extra arm would add parameters without modality-dropout benefit. Both v2 (31D kin) and v3 (35D kin) are preserved: `--use-kin-v3` selects the branch at training time.

---

## Session 25 — 2026-05-03 · fusion_tcn + fusion_mvit (visual-architecture-style sensor fusion)

**Goal:** Compare the existing factorized-attention `FusionTransformer` against two alternative sensor-fusion architectures inspired by visual encoders — a 1D temporal CNN (R3D-18-style) and a multiscale transformer (MViT-style). No image inputs; same kin + INS + VO data and the exact same 650/150/208 split as fusion_v2.

### `fusion/temporal_cnn.py` — `FusionTCN` (new)

- R3D-18-style structure adapted to 1D sensor sequences: per-modality per-timestep linear embedders (kin→32, ins→16, vo→16), channel-concat to 64-D, then a stem Conv1d + 4 stages of `BasicBlock1D` residual blocks (Conv1d-BN-ReLU-Conv1d-BN + identity/projected residual).
- Stage layout: `(64, 64, 128, 256)` channels, 2 blocks per stage, stride-2 temporal downsample between stages → T=40 → 40 → 20 → 10 → 5; global avg pool over T → linear head → 6.
- **Params: 1,023,798.**

### `fusion/mvit.py` — `FusionMViT` (new)

- Multiscale factorized transformer over `(B, T, M, D)` tokens — M = number of modalities (3 with VO).
- Per-modality embedders project to D=64 per timestep; modality embedding + sinusoidal temporal positional encoding.
- 3 stages with progressive temporal pooling (stride-2 mean over T pairs) and channel expansion: dims `(64, 96, 128)`, 2 blocks per stage. Each block reuses the modal-then-temporal attention pattern from `fusion/model.py::FactorizedBlock`, but **without the causal mask** (single regression at clip end, not per-timestep streaming).
- Readout: mean over (T, M) → LayerNorm → Linear → 6.
- **Params: 758,566.**

### `fusion/train.py` — `--arch` flag

- New `--arch {transformer, tcn, mvit}` argument (default `transformer` keeps existing behavior).
- `transformer` branch unchanged (forwards to `FusionTransformer` with d_model/n_blocks/n_heads/ffn_dim args).
- `tcn` branch builds `FusionTCN(kin_in, ins_in, vo_in, clip_len)`.
- `mvit` branch builds `FusionMViT(kin_in, ins_in, vo_in, n_heads, clip_len)`.
- Param count log line updated to `f"{args.arch} params: {n_params:,}"`.

### Launch scripts

- `scripts/launch_fusion_tcn.sh` — runs `fusion/train.py --arch tcn --use-vo` with the same `--epochs 50 --batch-size 128 --patience 10` as `launch_fusion_v2.sh`. Output: `runs/fusion_tcn`.
- `scripts/launch_fusion_mvit.sh` — same but `--arch mvit`. Output: `runs/fusion_mvit`.

### `scripts/watch_cnn_then_fusion.sh` — chained watcher

Polls every 60 s. When `pgrep -f "cnn_rgb_stage2_v2"` returns no PID, fires `launch_fusion_tcn.sh`, waits for its pidfile to exit, then fires `launch_fusion_mvit.sh`. All three on GPU 0. Currently running, PID 952028.

### Earlier in the session — MViT runs killed

- MViT RGB Stage-2 was launched on GPU 1 on 2026-05-01; the previous `watch_raft_then_mvit.sh` had auto-launched it after RAFT-Stereo finished.
- Both MViT RGB and (auto-fired) MViT disparity Stage-2 were killed before completing epoch 1 — GPU 1 was needed for unitree RL training (`unitree_rl_lab/scripts/rsl_rl/train.py`, currently running there).
- Both `runs/mvit_rgb_stage2/` and `runs/mvit_disparity_stage2/` only contain `norm_stats.json`. Both watchers (`watch_cnn_rgb_then_disparity.sh` PID 747512, `watch_mvit_rgb_then_disparity.sh` PID 747537) have exited.
- CNN RGB Stage-2 v2 (GPU 0) still healthy: epoch 27/50, best val_loss 0.0221 at epoch 18, no_improve=9 — likely early-stops at epoch 28 (next ~4 hours).

### Design notes

- "3D CNN for sensor fusion" reinterpreted as 1D temporal CNN: there is no spatial dim in the sensor data, so the R3D-18 idea reduces to BasicBlock1D over the time axis with channel-concat across modalities at the input.
- "MViT for sensor fusion" reinterpreted as multiscale factorized transformer over `(T, M)` tokens — the MViT pyramid (progressive token reduction + channel expansion) is the directly portable idea; pooling attention itself simplifies to mean-pool between stages here because keys/values are not the bottleneck.
- Both architectures are roughly param-fair to fusion_v2 (455K) within a 2× factor; if strict size-matching is needed for the thesis, `stage_dims` / `blocks_per_stage` can be reduced.

---

### Test results (added 2026-05-04)

After CNN RGB Stage-2 v2 early-stopped at epoch 28 (best val_loss 0.0221 at epoch 18), the chained watcher fired both new architectures sequentially on GPU 0. Both completed and were evaluated on the 208-trajectory test set (92,935 clips):

- **fusion_tcn:** test RMSE 0.0746 overall, **0.0429 m/s linear (best of all models)**, 0.0964 rad/s angular. Best at epoch 29, ~88 min wall-clock. Wins every linear axis (vx −17.8 %, vy −12.5 %, vz −1.3 % vs fusion_v2). All angular axes 1–12 % worse.
- **fusion_mvit:** test RMSE 0.0772 overall, 0.0501 m/s linear, 0.0970 rad/s angular. Best at epoch 34, ~172 min wall-clock. Worse than fusion_v2 on every axis except `ωz` (where it still loses to fusion_v2).

**Architectural conclusion.** fusion_v2 (factorised causal transformer, 455K params) is **not capacity-limited** — extra parameters from a 2.25× larger TCN buy a clear win on translation but don't unlock new accuracy on rotation. fusion_v2 still wins on overall RMSE and on every angular axis. For tasks that care about *linear* velocity specifically (e.g. position integration), fusion_tcn is the better choice. The MViT-style multiscale pyramid hurts at this scale: with T=40 / M=3 there is no token-count problem to solve, so mean-pool stride-2 between stages just discards information.

`fusion/evaluate.py` was extended to dispatch on `args["arch"]`; previously it hard-instantiated `FusionTransformer`.

EXPERIMENTS.md updated: §1.6 / §1.7 final stats, §3.7 / §3.8 detailed test breakdown, §4.1.1 head-to-head table with every per-axis number, §6 file pointers extended for new run dirs.

### CNN disparity Stage-2 — started then stopped

A separate CNN disparity Stage-2 run was started on GPU 0 around 2026-05-03 19:00 (`launch_cnn_disparity_stage2.sh`, PID 965107) and ran for ~17 hours (epoch 3/50, best val_loss 0.0476). User asked to stop it on 2026-05-04 — killed cleanly via SIGTERM on the process group. Checkpoints preserved at `runs/cnn_disparity_stage2/{best,last}_model.pt` for possible later resume.

### CNN RGB Stage-2 v2 evaluation

Launched `cnn3d/evaluate.py --checkpoint runs/cnn_rgb_stage2_v2/best_model.pt`. Will be reported here when complete.

---

## Pending

- **CNN RGB Stage-2 v2 evaluation** — running now; update EXPERIMENTS.md/Excel when it finishes.
- **MViT visual Stage-2 (RGB and disparity)** — both killed on 2026-05-01; need to be relaunched once GPU 1 is free again (currently on unitree RL training).
- **CNN disparity Stage-2** — partial run preserved, user-stopped at epoch 3. Resume or rerun if needed.
- **Excel updates** — `go2_model_results.xlsx` and `go2_experiment_summary.xlsx` need the fusion_tcn / fusion_mvit rows and Per-Trajectory sheets.
- **Thesis graphs** — deferred; user to decide which figures to generate.
