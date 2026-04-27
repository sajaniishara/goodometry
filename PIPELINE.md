# Goodometry — Multimodal Fusion Pipeline for Unitree Go2 State Estimation

**Project**: `~/projects/goodometry/`
**Data**: `/mnt/data/go2_research_dataset_v2/` (1,008 clean trajectories, 1,081 collected, 73 falls excluded)
**Status as of 2026-04-26**: Three of four preprocessing arms complete; both fusion models (`fusion_v1` and `fusion_v1_marg`) trained, evaluated, and head-to-head'd against the Stage-2 RGB CNN baseline. VO arm restarted at stride=3 (Session 23) after empirical validation — ~2.4 days remaining for the full 1,008 trajectories. RAFT-Stereo disparity precompute also resumed at iters=32 on GPU 1 (~5 days remaining) for the existing v2 disparity-input experiments.

---

## 1. Goal and motivation

Predict the Unitree Go2's **6-DoF body-frame velocity** `(vx, vy, vz, ωx, ωy, ωz)` at every timestep from its onboard sensors:

- Stereo cameras (1280×720 RGB, 30 FPS, 0.120 m baseline)
- IMU (6-axis, 200 Hz physics resampled to 30 Hz)
- Magnetometer (3-axis, simulated)
- Joint encoders (12 joints × position+velocity, 30 Hz)

**Pivot from earlier pipeline.** The v1/v2 work in `go2_research/` (Sessions 1–21 of `CHANGES.md`) used end-to-end 3D-CNN / MViT models on raw stereo + raw sensors. Best v1 test RMSE was 0.209; best Stage-2 test RMSE is 0.127 (this session). Goodometry takes the alternative path described in `godometry/go2-concrete-algorithms.md`: **classical per-sensor preprocessing → small neural fusion transformer**. The goal is faster training, better sample efficiency, principled handling of sensor dropouts, and cleaner sim-to-real transfer.

---

## 2. Architecture overview

```
┌──────────────────────────────────────────────────────────────────────┐
│ Per-trajectory preprocessing (per `frame_idx` aligned)               │
├──────────────────────────────────────────────────────────────────────┤
│ Stereo camera   →  DROID-SLAM     →  vo.npz       (SE(3) per frame)  │
│ Joint encoders  →  pytorch_kin    →  kin.npz      (31D features)     │
│ IMU + (mag)     →  Madgwick       →  ins.npz      (10D features)     │
│ World-frame GT  →  R_wb^T rotate  →  labels_body  (6D supervision)   │
│ FR_calf         →  per-traj offset →  calibration.npz                │
└──────────────────────────────────────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Fusion model (FusionTransformer, ~440 K params)                      │
│                                                                      │
│   [kin 31D] → MLP → 128D ─┐                                          │
│   [ins 10D] → MLP → 128D ─┼→ (B, T=40, M=2, D=128)                   │
│   [vo  7D]  → MLP → 128D ─┘   (M=3 once VO arm is done)              │
│                                                                      │
│   Block × 2:                                                         │
│      modal self-attn over M tokens (cheap)                           │
│      causal temporal self-attn over T=40 (1.3 s context)             │
│      FFN                                                             │
│                                                                      │
│   Mean over M → last timestep → Linear → (vx,vy,vz,ωx,ωy,ωz) body    │
└──────────────────────────────────────────────────────────────────────┘
```

Per-timestep input feature dimension once the VO arm is done: ~48 channels (kin 31 + ins 10 + vo 7). No LiDAR — Go2 doesn't have one in this dataset.

---

## 3. Preprocessing arms

All outputs are saved as `<traj>/<arm>.npz` next to the existing `sensors.npz`. Every file shares the same `frame_idx` key from `sensors.npz` so any downstream code can join modalities by inner-join on `frame_idx`.

### 3.1 Joint kinematics → `kin.npz` ✅ 1,008 / 1,008 (446 MB total, 288 s CPU at scale)

**Pipeline** (`goodometry/kinematics/`):

1. **Forward kinematics** with `pytorch_kinematics` on the Unitree Go2 URDF (`unitree_ros/robots/go2_description/urdf/go2_description.urdf`). Two index-mapping gotchas had to be fixed:
   - `sensors.npz['joints']` uses Isaac Sim's *grouped* DOF order `[FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, …, FL_calf, …]`, while the URDF chain expects *per-leg* `[FL_hip, FL_thigh, FL_calf, FR_hip, …]`. Remap vector: `[0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]`.
   - `pytorch_kinematics.build_chain_from_urdf` collapses the URDF's fixed `*_foot_joint`. Apply the URDF-verified foot offset `(0, 0, -0.213) m` manually in each calf frame.
2. **Foot velocities** by centered finite difference at 30 Hz with forward/back differences at endpoints.
3. **Soft contact probability** by velocity-gate with linear ramp:
   ```
   |v_foot^B|  ≤ 0.10 m/s  →  contact_prob = 1.0
   |v_foot^B|  ≥ 0.50 m/s  →  contact_prob = 0.0
                      (linear in between)
   ```
4. **Body-velocity-from-legs** via `go2-concrete-algorithms.md` §1.2 stance-foot constraint:
   ```
   v_body^B = -ω_body^B × p_foot^B - dp_foot^B/dt    (per stance foot)
            (combine across feet weighted by contact_prob)
   ```
5. Reads joints via `calibration.load_calibrated_joints()` so the FR_calf offset is applied at the input (see §3.5).

**Output schema** (`<traj>/kin.npz`):

```
frame_idx              (N,)        int32
foot_pos_body          (N, 4, 3)   float32   [FL, FR, RL, RR] in body frame
foot_vel_body          (N, 4, 3)   float32
contact_prob           (N, 4)      float32
v_body_legs            (N, 3)      float32
applied_fr_calf_offset (scalar)    float32   diagnostic
```

**Sanity check** (averaged across 30 random trajectories):

```
Foot Z below body origin (expected ~-0.26 m for standing pose):
  FL  z = -0.259 ± 0.028 m
  FR  z = -0.255 ± 0.027 m   ← in-band post-calibration (was ~-0.225 raw)
  RL  z = -0.265 ± 0.028 m
  RR  z = -0.259 ± 0.029 m

Leg-odometry vs GT body velocity (one trajectory, 3,136 frames):
  vx  RMSE 0.27 m/s   corr +0.18
  vy  RMSE 0.19 m/s   corr +0.56
  vz  RMSE 0.25 m/s   corr +0.27
```

The leg-odometry estimate is noisy at per-timestep scale (actuator jitter + finite-diff amplification) but the **raw FK foot positions and velocities** are the more informative features the fusion model actually uses; the `v_body_legs` channel is a courtesy "first-cut" body velocity that's averaged out by temporal attention.

### 3.2 IMU + magnetometer → `ins.npz` and `ins_marg.npz` ✅ 1,008 each (253 + 258 MB, ~30 s each)

**Pipeline** (`goodometry/ins/`): xIOTechnologies `imufusion` Madgwick AHRS filter (production C library by Madgwick himself, Python bindings).

**Two output variants:**

| File | Mode | Yaw behaviour | Default |
|---|---|---|---|
| `ins.npz` | IMU-only (6-axis) | Drifts (free integration) | ✅ |
| `ins_marg.npz` | MARG (9-axis, stitched) | Drift-free, ~5° declination offset | for ablation |

**Tuned settings** (gait-aware):

```python
imufusion.Settings(
    convention=ENU,                  # body Z-up requires Z-up world convention
    gain=1.0,                        # higher than imufusion's default 0.5
    gyro_range=2000.0,               # dps
    acceleration_rejection_deg=0.0,  # 0 = no rejection.
                                     # Default 10° rejection triggers constantly during
                                     # ±4 m/s² trot foot-impact spikes, leaving the filter
                                     # gyro-only and drifting roll/pitch by 15°.
    magnetic_rejection_deg=0.0,      # only matters when use_mag=True (we don't use MARG mode)
    recovery_trigger_period=int(5 * 30 Hz),
)
```

**The MARG path needed two fixes** that aren't obvious:

1. **Axis order remap.** The Go2 sim writes `mag = [N, E, D]` per the user's confirmation (`23.0, 2.0, -42.0` μT for an S-hemisphere mid-latitude reference). imufusion with `CONVENTION_ENU` expects `[E, N, U]`. Remap before feeding: `mag_enu = [m_y, m_x, -m_z]`. Verified by reconstruction: `R_wb^T @ [23, 2, -42]` matches stored mag with **correlation 0.9996** per axis and residual std 0.499 μT (matches the sim's declared N(0, 0.5) μT noise floor exactly).
2. **imufusion's MARG mode doesn't converge** on this data — at every rejection threshold from 0 to 180°, yaw RMSE was ~95° (basically gyro-drift). Tilt-compensated yaw computed externally with the IMU-only filter's roll/pitch gives **yaw RMSE 17°** including a constant ~5° declination offset. Final approach: two-pass stitch — (i) run IMU-only Madgwick → reliable roll/pitch, (ii) tilt-compensate the body-frame mag using those roll/pitch values to get drift-free yaw, (iii) build the quaternion from `(roll_imu, pitch_imu, yaw_mag)`.

**Output schema** (both files):

```
frame_idx    (N,)      int32
quat_wxyz    (N, 4)    float32   body-to-filter-world quaternion (unit-norm)
gyro_body    (N, 3)    float32   pass-through (already body frame)
accel_body   (N, 3)    float32   gravity-compensated, body frame
accel_world  (N, 3)    float32   gravity-compensated, filter-world frame
euler_rpy    (N, 3)    float32   roll/pitch/yaw from quaternion
flags        (N, 4)    uint8     [init, ang_recov, acc_recov, mag_recov]
use_mag      (scalar)  bool
```

**Sanity check** (30 random trajectories, IMU-only mode):

```
quaternion unit-norm:  every file min=1.000000 max=1.000000
roll  vs GT:  median RMSE  7.9°    p95  36.6°
pitch vs GT:  median RMSE  5.7°    p95  23.0°
yaw   vs GT:  drifts (no ext reference)
|accel_body| residual: mean 5.3 m/s²   ← real gait-induced linear accel,
                                        not filter error
```

### 3.3 Camera → `vo.npz` (DROID-SLAM stereo) 🔄 in progress on GPU 0

**Pipeline.** DROID-SLAM in stereo mode, fed left+right at 360×640 (decoded at half-res from 1280×720 PNG via `cv2.IMREAD_REDUCED_COLOR_2`). Per-frame SE(3) camera-to-world pose + a `valid` flag per timestep. Skipped frames (stride=3) are SLERP-interpolated to fill the full `sensors.npz['frame_idx']` coverage.

**Pilot (2 trajectories, before user-directed scale-up):**

```
Trajectory               Method   ATE_sim3  RPE_t_1s  scale  FPS
flat / forward          DPVO mono  0.88 m   0.110 m   6.54   13.1
flat / forward          DROID stereo  0.91 m   0.510 m   0.74   5.7
forest / forward        DPVO mono  0.36 m   0.102 m   4.00   13.0
forest / forward        DROID stereo  0.14 m   0.131 m   1.13   9.3   ← winner
```

DROID won on forest (target deployment terrain) by 2.5× ATE; DPVO won on low-texture flat ground. Forest = the dominant terrain in the v2 dataset (per Session 11 of the original CHANGES), so DROID was selected.

**Optimisations (Session 23 stride sweep + I/O):**

- **`stride=3`** with SLERP interpolation. Empirically validated against stride 1, 4, 5, 8 on 3 pilot trajectories — stride=3 gives the best weighted-average ATE across the dataset's terrain mix, while being 2.5× faster than stride=1. See `EXPERIMENTS.md` §7 for the full sweep results.
- **`fast=True`** — skip both `backend(7)` and `backend(12)` global BA passes. Trajectories are 1–2 minutes with no revisits, so frontend's local-window BA + `traj_filler` is enough.
- **Parallel I/O + `IMREAD_REDUCED_COLOR_2`** — drops PNG load+decode time from 60 s → 9 s per 3,500-frame trajectory.

**Combined wall-clock**: ~3.4 min/trajectory, ~17 fps effective. Total projected: **~2.4 days** for the full 1,008 trajectories (down from the original ~7 days).

**Failures.** ~1 % of trajectories hit a keyframe-buffer overflow in DROID's frontend (`IndexError: index 512 is out of bounds for dimension 0 with size 512`). They're logged as `status: error` in `logs/vo_scale_droid.jsonl`; will be re-run with `--buffer 2048` at the end.

**Output schema** (`<traj>/vo.npz`):

```
frame_idx        (N,)        int32     # matches sensors.npz exactly (full coverage via interp)
pose_7d          (N, 7)      float32   # [tx, ty, tz, qx, qy, qz, qw] cam→world
pose_mat         (N, 4, 4)   float32   # SE(3) matrix form
valid            (N,)        bool
stride_used      (scalar)    int32     # diagnostic — 3 in current config
n_droid_frames   (scalar)    int32     # how many DROID actually processed (vs interpolated)
```

### 3.4 Body-frame labels → `labels_body.npz` ✅ 1,008 / 1,008 (small, sub-second per file)

**The problem nobody had caught.** Session 18 fixed the *angular* velocity label to body frame, but `sensors.npz['labels'][:, :3]` is still **world-frame** linear velocity — `preprocess_dataset.py` passes `gt_vel_x/y/z` through unchanged from the CSV, and those are computed by `(world_position − prev_world_position) / dt`. Confirmed at `go2_research_system.py:1380`.

**Fix** (`goodometry/labels.py`): rotate `labels[:, :3]` from world to body using the GT quaternion, clip angular velocity to ±10 rad/s (matching Session 17). Precomputed once and saved per trajectory.

**Output schema** (`<traj>/labels_body.npz`):

```
frame_idx      (N,)     int32
lin_vel_body   (N, 3)   float32   ← rotated to body frame
ang_vel_body   (N, 3)   float32   ← clipped at ±10 rad/s
gt_quat_wxyz   (N, 4)   float32   ← cached for any other rotation need
```

### 3.5 FR_calf calibration → `calibration.npz` ✅ 1,008 / 1,008

**The discovery.** `sensors.npz['joints']` channel 9 (FR_calf_pos) is systematically biased in **all 1,008 trajectories** by a per-trajectory mean of ~0.6 rad relative to the other three calves. Independently confirmed by GT body posture: across 30 forward-path trajectories, body roll mean = +0.37 rad (right side down) and pitch mean = +0.21 rad (front side down) — consistently sagging on the front-right corner, exactly what a too-bent FR calf would produce.

**Diagnosis.** Commands and PD gains are fully symmetric (`calf_offset = -1.5`, `calf_amp = 0.35`, `kp = 80, kd = 2.0` for every joint). The asymmetry lives in the Isaac Sim Go2 USD asset (`/Isaac/Robots/Unitree/Go2/go2.usd`, shipped with Isaac Sim 5.1) — likely a bad FR_calf collision/foot mesh that lets the foot penetrate the ground, causing Kp-driven calf compression under load. This fits the observed mean offset *and* the much-higher between-trajectory variance (FR std 0.33 vs others 0.04–0.10 — variation comes from contact-force-dependent compression).

**Remediation.** Per-trajectory static offset `offset = mean(FL, RL, RR calf) − mean(FR calf)`, applied to `joints[:, 9]` at load time via `goodometry.calibration.load_calibrated_joints()`. Non-destructive — `sensors.npz` is left untouched.

```
                       RAW                            CALIBRATED
                median  mean    std       →    median  mean    std
FL_calf        -1.621  -1.656  0.187            -1.621  -1.656  0.187   (untouched)
FR_calf        -2.219  -2.184  0.334            -1.602  -1.618  0.110   ← in-band
RL_calf        -1.567  -1.587  0.167            -1.567  -1.587  0.167   (untouched)
RR_calf        -1.608  -1.611  0.146            -1.608  -1.611  0.146   (untouched)
```

Velocities are not corrected — a static offset has zero time-derivative.

**Per-trajectory offset distribution** (rad): mean +0.566, median +0.605, std 0.371, range [−1.00, +1.81], p5–p95 [−0.04, +1.05].

---

## 4. Fusion model

`goodometry/fusion/model.py :: FusionTransformer`. Factorized modal-then-temporal causal transformer per `go2-concrete-algorithms.md` §5.2.

```
Input shape:  kin (B, T=40, 31)  +  ins (B, T=40, 10)
Embed:        per-modality 2-layer MLP → 128
Stack:        (B, T, M=2, 128)  modal × temporal blocks × 2
Output head:  mean over M → last timestep → LayerNorm → Linear(128, 6)

Total parameters: 437,382
Heads: 4   FFN dim: 256   Dropout: 0.1
Causal mask: triu over the temporal axis
Modality embedding: learned per-modality bias added at input (allows cross-modality positional cues to attention)
Sinusoidal temporal positional encoding
```

The same architecture trivially extends to M=3 (kinematics + INS + VO) once the VO arm finishes — adds one more 2-layer MLP and one more `modality_emb[2]` row, no other changes.

---

## 5. Training

`goodometry/fusion/train.py`. Standard PyTorch with a few details from the godometry doc:

| | Value |
|---|---|
| Optimizer | AdamW, lr 1e-4, weight decay 1e-4 |
| LR schedule | ReduceLROnPlateau, factor 0.5, patience 3 |
| Loss | MSE on 6-D body velocity, with per-component weights `[1, 1, 1, 5, 5, 5]` (rotation 5× translation per §6.1 recipe) |
| Modality dropout | 10 % per-sample chance to zero each modality, with "don't drop both" safeguard (§6.5) |
| Clip length | 40 timesteps (~1.3 s at 30 Hz) |
| Stride | 8 (new clip every ~270 ms) |
| Batch size | 128 |
| Patience | 10 epochs |
| Max epochs | 50 |
| Norm stats | per-channel z-score from a 20 % training-trajectory subsample |
| Split | stratified 650 / 150 / 208 by terrain, seed=42 (matches Session 17) |

**Wall-clock**: ~130 s/epoch on a single RTX 5060 Ti (16 GB), ~2 hours max. Fits comfortably alongside other GPU workloads (used 0.8 GB VRAM during fusion_v1 training, co-resident with the 9.5 GB Stage-2 training).

---

## 6. Results

### 6.1 fusion_v1 (kinematics + INS, IMU-only) ✅ trained 2026-04-24, evaluated 2026-04-25

Stratified 650/150/208 split, seed=42, 50 epochs max, patience 10. Early-stopped at epoch 17.

**Validation curve:**

```
ep   train    val      rmse    lin     ang
 1   0.1846   0.0276   0.1661  0.0984  0.2132
 2   0.0945   0.0222   0.1490  0.0895  0.1908
 5   0.0637   0.0218   0.1477  0.0726  0.1959
 7   0.0530   0.0176   0.1325  0.0685  0.1745   ← best (saved)
11   0.0430   0.0180   0.1340  0.0622  0.1791
17   0.0338   0.0246   0.1569  0.0595  0.2137   train still falling, val plateau → overfit
```

**Test set** (208 held-out trajectories, 92,935 clips):

```
axis    rmse     p50     p95   units
  vx   0.0786  0.0373  0.1544  m/s
  vy   0.0554  0.0287  0.1085  m/s
  vz   0.0540  0.0271  0.1003  m/s
  ωx   0.1616  0.0767  0.2704  rad/s
  ωy   0.0863  0.0365  0.1338  rad/s
  ωz   0.0806  0.0261  0.1030  rad/s

overall   RMSE = 0.0933
linear    RMSE = 0.0637 m/s
angular   RMSE = 0.1156 rad/s
```

Saved to `runs/fusion_v1/test_results.json`.

### 6.2 fusion_v1_marg (kinematics + INS-MARG, drift-free yaw) ✅ trained + evaluated

Same architecture, reads `ins_marg.npz` instead of `ins.npz`. Early-stopped at epoch 21/50, best at epoch 11 — val_loss **0.0167** vs fusion_v1's 0.0176 (~5% val improvement).

**Test set** (same 208 trajectories, 92,935 clips):

```
axis    rmse     p50     p95   units
  vx   0.0724  0.0344  0.1402  m/s
  vy   0.0504  0.0260  0.0982  m/s
  vz   0.0502  0.0260  0.0915  m/s
  ωx   0.1391  0.0519  0.2171  rad/s
  ωy   0.0875  0.0362  0.1341  rad/s
  ωz   0.0734  0.0228  0.0953  rad/s

overall   RMSE = 0.0844    (-9.5% vs fusion_v1)
linear    RMSE = 0.0586 m/s   (-8.0%)
angular   RMSE = 0.1039 rad/s (-10.1%)
```

**Empirical answer to the mag question**: val curves looked nearly identical, but on held-out test the MARG variant beats IMU-only by **8–10% across every metric**. The biggest single-axis win is `ωx` (roll rate, −13.9%) — drift-free yaw indirectly stabilises the model's body-rate estimates by giving it a more consistent quaternion reference across the 1.3 s context window. So the magnetometer is **not useless** in this configuration, just modestly helpful, and the IMU-only-by-default in Session 21 is justified primarily on sim-to-real grounds (§3.2) rather than because mag adds nothing.

### 6.3 Stage-2 CNN RGB (v2 baseline, end-to-end) ✅ converged 2026-04-25, test eval done

The latest `go2_research/runs/cnn_rgb_stage2/` model — 33.5 M parameter R3D-18 + sensor-branch CNN, trained for 30 epochs on RGB only (Sessions 18–21 of the original CHANGES). Best val_loss 0.0210 at epoch 20. Test on 152 trajectories (705/151/152 stratified split, slightly different from goodometry's 650/150/208 — both stratified by terrain with seed=42, but the count split differs; numbers are comparable in magnitude not on identical samples).

**Test set** (68,557 clips):

```
axis           rmse      MAE   median     P90     P95     P99    Max
vx           0.1004   0.0673   0.0445   0.1532   0.2094  0.3614  1.6228
vy           0.1005   0.0671   0.0437   0.1546   0.2136  0.3538  1.9737
vz           0.0650   0.0380   0.0249   0.0799   0.1111  0.2266  1.7806
roll_rate    0.2098   0.1106   0.0746   0.2173   0.2934  0.6777  7.3399
pitch_rate   0.1385   0.0750   0.0538   0.1460   0.1894  0.3749  4.3414
yaw_rate     0.0958   0.0545   0.0396   0.1088   0.1417  0.2711  6.2167

overall  RMSE  0.1270
linear   RMSE  0.0902 m/s
angular  RMSE  0.1553 rad/s
```

### 6.4 Head-to-head comparison — each model on its own test split

| | Pre-Session-18 v1 best (test) | **Stage-2 (test)** | **fusion_v1 (test)** | **fusion_v1_marg (test)** |
|---|---|---|---|---|
| Architecture | MViT_V2_S + RAFT disp | R3D-18 + sensor CNN | factorized transformer | same |
| Parameters | ~35 M | 33,485,894 | **437,382** | 437,382 |
| Train wall-clock | ~hours | ~5 days | **~37 min** | ~38 min |
| Visual input | RAFT-Stereo disparity | RGB | none | none |
| INS | n/a | imu9 (raw) | IMU-only Madgwick | MARG stitched |
| Test trajectories | (different split) | 152 | 208 | 208 |
| Test clips | — | 68,557 | 92,935 | 92,935 |
| Overall RMSE | 0.2090 | 0.1270 | 0.0933 | **0.0844** |
| Linear RMSE (m/s) | 0.1384 | 0.0902 | 0.0637 | **0.0586** |
| Angular RMSE (rad/s) | 0.2612 | 0.1553 | 0.1156 | **0.1039** |

**Caveat**: each model is on a different test split (and Stage-2 uses world-frame linear targets while fusion uses body-frame). Numbers comparable in magnitude, not strictly the same metric. See `EXPERIMENTS.md` §2 for the split-overlap table.

### 6.5 Fair head-to-head — same 71 trajectories, both models held out

To eliminate the split caveat, evaluated both models on the 71 trajectories that are **in fusion's test set ∩ NOT in Stage-2's training set** (so neither model trained on any of them). 31,791 fusion clips and 32,075 Stage-2 clips.

```
metric                   Stage-2     fusion_v1      Δ
overall RMSE              0.1015      0.0809      −20.3 %
linear  RMSE (m/s)        0.0907      0.0589      −35.0 %
angular RMSE (rad/s)      0.1112      0.0981      −11.8 %

axis        Stage-2     fusion_v1      Δ
  vx         0.1021      0.0734      −28.1 %
  vy         0.1046      0.0521      −50.2 %    ← biggest win
  vz         0.0576      0.0482      −16.3 %
  ωx (roll)  0.1488      0.1428       −4.0 %
  ωy (pitch) 0.0998      0.0744      −25.5 %
  ωz (yaw)   0.0709      0.0542      −23.6 %
```

Saved to `runs/fair_head_to_head.json`.

`fusion_v1` wins on every axis while being **77× smaller, ~200× faster to train, and without a visual modality**. The biggest win is `vy` (lateral velocity, −50%) — the signal FK preprocessing gives the network "for free" via foot-position deltas. Adding the VO arm (`fusion_v2`) when DROID-SLAM at-scale finishes should close the remaining `vx/vz` gaps where visual ego-motion contributes most.

---

## 7. Code organisation

```
~/projects/goodometry/
├── PIPELINE.md                       (this document)
├── godometry/                        (the two design docs from your zip)
│   ├── go2-concrete-algorithms.md
│   └── go2-preprocessing-options.md
├── configs/
│   ├── go2_camera.yaml               (intrinsics, baseline, frame conventions)
│   └── fr_calf_offsets.json          (per-traj calibration summary)
├── calibration/                       (FR_calf offset)
│   └── __init__.py                   (load_calibrated_joints)
├── kinematics/
│   ├── fk.py                         (Go2FK + finite-diff foot velocity)
│   ├── contact.py                    (velocity-gate soft contact)
│   └── leg_odom.py                   (§1.2 body-velocity-from-legs)
├── ins/
│   └── madgwick.py                   (imufusion + two-pass MARG stitch)
├── labels.py                         (world-to-body label rotation)
├── vo/                                (DROID-SLAM + DPVO runners)
│   ├── dpvo_runner.py
│   ├── droid_runner.py
│   ├── gt.py
│   └── eval.py                       (Umeyama ATE / RPE)
├── fusion/                           (the model + trainer)
│   ├── dataset.py                    (GoFusionDataset, stratified split, norm stats)
│   ├── model.py                      (FusionTransformer, FactorizedBlock)
│   ├── train.py
│   └── evaluate.py                   (test-set evaluator)
├── scripts/
│   ├── env.sh                        (env_isaaclab + CUDA 13 toolchain)
│   ├── compute_calf_calibration.py
│   ├── precompute_labels_body.py
│   ├── run_kin_at_scale.py
│   ├── run_ins_at_scale.py           (--with-mag for ins_marg.npz)
│   ├── run_vo_at_scale.py
│   └── launch_droid_at_scale.sh      (setsid nohup launcher)
├── third_party/
│   ├── DPVO/                         (cloned + patched + built)
│   ├── DROID-SLAM/                   (cloned + patched + built)
│   └── unitree_ros/                  (Go2 URDF source)
├── runs/
│   └── fusion_v1/
│       ├── args.json
│       ├── best_model.pt
│       ├── last_model.pt
│       ├── norm_stats.npz
│       ├── train_log.csv
│       ├── metrics.jsonl
│       └── test_results.json
└── logs/                             (training/preprocessing logs)

/mnt/data/go2_research_dataset_v2/<traj>/    (per-trajectory data)
├── sensors.npz       (raw, from Session 13/18)
├── calibration.npz   (FR_calf offset, scalar + diagnostics)
├── kin.npz           (foot pos/vel/contact + leg-odom v_body)
├── ins.npz           (Madgwick IMU-only)
├── ins_marg.npz      (Madgwick MARG stitched, drift-free yaw)
├── labels_body.npz   (world→body rotated labels)
└── vo.npz            (DROID-SLAM SE(3) per frame, when available)
```

---

## 8. Reproducing results

```bash
source ~/projects/goodometry/scripts/env.sh

# 0. one-time preprocessing (already done; commands shown for reproducibility)
python scripts/compute_calf_calibration.py
python scripts/run_kin_at_scale.py
python scripts/run_ins_at_scale.py             # IMU-only ins.npz
python scripts/run_ins_at_scale.py --with-mag  # MARG ins_marg.npz
python scripts/precompute_labels_body.py

# 1. fusion_v1 (already trained)
CUDA_VISIBLE_DEVICES=1 python -u fusion/train.py \
    --output-dir runs/fusion_v1 --device cuda \
    --epochs 50 --batch-size 128 --num-workers 4 --patience 10
# Test:
python fusion/evaluate.py --run-dir runs/fusion_v1 --device cuda

# 2. fusion_v1_marg (in progress)
CUDA_VISIBLE_DEVICES=1 python -u fusion/train.py \
    --output-dir runs/fusion_v1_marg --device cuda \
    --epochs 50 --batch-size 128 --num-workers 4 --patience 10 \
    --ins-file ins_marg.npz
python fusion/evaluate.py --run-dir runs/fusion_v1_marg --device cuda
```

---

## 9. Open issues and next steps

1. **VO arm at scale.** DROID-SLAM at stride=3 (PID 169781, GPU 0, detached). ETA ~2.4 days for the remaining ~1,003 trajectories.
2. **RAFT-Stereo disparity at scale.** Precompute resumed at iters=32 (PID 157759, GPU 1, detached). ETA ~5 days for the remaining ~482 trajectories. For the existing v2 CNN/MViT disparity-input experiments — distinct from the goodometry pipeline.
3. **fusion_v2 (kin + ins + vo).** Adds one modality token to `FusionTransformer`, otherwise identical training. Run when DROID's vo.npz coverage is high enough (could start at ~80 % coverage with held-out trajectories sampled from the completed set). Expected to close the `vx`/`vz` gap where visual ego-motion helps most.
4. **DROID buffer-overflow re-runs.** ~1 % of trajectories hit `IndexError: index 512 is out of bounds for dimension 0 with size 512` (long forest trajectories with many keyframes). Re-run those with `buffer=1024` after the main pass.
5. **FR_calf root-cause.** Current per-trajectory offset masks the issue but doesn't fix it. If the project ever recollects the v2 dataset, investigating the Isaac Sim Go2 USD's FR_calf collision/foot mesh is worthwhile.
6. **Sim-to-real for the magnetometer.** Sim mag is hardcoded `[23, 2, -42]` μT (mid-latitude S-hemisphere reference). The MARG ablation showed it buys ~10 % on test in pure-sim, but real Go2 deployment requires WMM/IGRF for the actual lat/lon plus per-unit hard-iron / soft-iron calibration. See §3.2.
7. **Per-axis residuals.** `ωx` (roll rate) is the worst axis at 0.14–0.16 rad/s — mostly the body-rocking gait that VO and IMU smooth through. Leg kinematics could in principle constrain it (foot impact timing). Worth investigating once VO data is in.
8. **Modality-dropout ablation.** Forced each modality off at *training*; never measured what happens when modalities are dropped at *inference*. The fusion-net's whole sales pitch is graceful degradation — should quantify it.
9. **Stratified split alignment.** The two `split_trajectories` implementations (`go2_research/cnn3d/dataset.py` and `goodometry/fusion/dataset.py`) use different RNG types so even with the same seed they produce different orderings. Future cleanup item: unify them so head-to-head comparisons don't require the 71-trajectory clean-subset workaround.

---

## 10. Why this approach worked

The end-to-end CNN/MViT pipeline was asking one network to *simultaneously* learn:

- The geometry of a quadruped (FK is a known closed-form function),
- Visual ego-motion (well-studied SLAM problem),
- Orientation from IMU (textbook complementary filter),
- The actual fusion of all three.

By doing the first three with classical algorithms and only learning the fourth, goodometry hands the network a 50-channel input where every channel already encodes geometric meaning. The transformer can spend its capacity on **when to trust which modality and how to combine them temporally** — which is the part where a learned model genuinely outperforms hand-designed Kalman gains.

That's why a 0.44 M-parameter model beats a 33.5 M one on every metric while training in 37 min vs 5 days.
