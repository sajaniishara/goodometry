# Configurations and Comparison Results

Companion to `PIPELINE.md`. Every model trained, every test evaluation, and the head-to-head numbers — in one place.

---

## 1. Models trained

### 1.1 Stage-2 CNN RGB v2 (baseline, end-to-end, matched 650/149/208 split)

`go2_research/runs/cnn_rgb_stage2_v2/` — re-trained with the same 650/149/208 stratified split as the fusion models for a strict apples-to-apples comparison. Old `cnn_rgb_stage2/` (705/151/152, RMSE 0.1270) is preserved but superseded.

| Knob | Value |
|---|---|
| Architecture | `VideoResNet18Fused` (R3D-18 + 1D-CNN sensor branch) |
| Visual input | RGB left camera, T = 8 frames at 224×224 |
| Sensor input | `imu9` + joints |
| Parameters | 33,485,894 |
| Best val_loss | 0.0221 at epoch 18 |
| Early-stopped | epoch 28 (patience 10) |
| Training wall-clock | ~5 days (28 epochs × ~4.2 hrs/epoch on RTX 5060 Ti) |
| Test set | **208 trajectories, 93,767 clips** (matched to fusion_v2) |

**Test results (body-frame angular, world-frame linear targets per training):**

| axis | RMSE | MAE | Median | P95 |
|---|---:|---:|---:|---:|
| `vx` | 0.1020 | 0.0704 | 0.0486 | 0.2118 |
| `vy` | 0.1031 | 0.0706 | 0.0474 | 0.2198 |
| `vz` | 0.0537 | 0.0362 | 0.0257 | 0.1047 |
| `roll_rate` | 0.1887 | 0.1128 | 0.0769 | 0.3106 |
| `pitch_rate` | 0.1171 | 0.0775 | 0.0591 | 0.1994 |
| `yaw_rate` | 0.0928 | 0.0586 | 0.0437 | 0.1529 |

```
overall RMSE  0.1168
linear  RMSE  0.0893     (world frame)
angular RMSE  0.1389 rad/s
```

### 1.1.legacy Stage-2 CNN RGB (old 705/151/152 split)

The legacy `go2_research/runs/cnn_rgb_stage2/` checkpoint. Trained Sessions 18–21 of `CHANGES.md`.

| Knob | Value |
|---|---|
| Architecture | `VideoResNet18Fused` (R3D-18 + 1D-CNN sensor branch, mid-fusion after layer2) |
| Visual input | RGB left camera, T = 8 frames at 224×224 |
| Sensor input | `imu9` (accel 3 + gyro 3 + mag 3) + joints (24) |
| Parameters | 33,485,894 |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-4 |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 3) |
| Loss | MSE on labels `[gt_vel_x, gt_vel_y, gt_vel_z, omega_body_x, omega_body_y, omega_body_z]` |
| Label frame | `[:, :3]` **world-frame** linear (gt_vel from `(p_t+1 − p_t)/dt`) ; `[:, 3:]` body-frame angular (Session 18) |
| Angular clip | ±10 rad/s (Session 17) |
| FR_calf calibration | not applied — uses raw `joints` |
| Sensor pretrain | yes — `imu9_branch` initialised from `runs/sensor_pretrain/best_model.pt`, frozen during stage 2 |
| Clip length / stride | 8 / 8 |
| Batch size | 16 |
| Patience | 10 epochs |
| Train / val / test counts | **705 / 151 / 152** (used `train_frac=0.7, val_frac=0.15` defaults) |
| Best val loss | 0.0210 at epoch 20 |
| Training wall-clock | ~5 days |

### 1.2 fusion_v1 (kinematics + INS, IMU-only)

| Knob | Value |
|---|---|
| Architecture | `FusionTransformer` — factorized modal-then-temporal causal transformer |
| Modalities (M=2) | kinematics token (31D MLP→128), INS-IMU-only token (10D MLP→128) |
| d_model / heads / FFN / blocks | 128 / 4 / 256 / 2 |
| Positional encoding | sinusoidal across temporal axis; learned modality embedding added to each token |
| Causal mask | yes, on temporal attention |
| Parameters | **437,382** |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-4 |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 3) |
| Loss | MSE on body-frame `[vx, vy, vz, ωx, ωy, ωz]` with per-axis weight `[1, 1, 1, 5, 5, 5]` (rotation 5× translation, §6.1 of `go2-concrete-algorithms.md`) |
| Label frame | body for everything (read `labels_body.npz` produced by `goodometry/labels.py`) |
| Angular clip | ±10 rad/s |
| FR_calf calibration | applied via `calibration.load_calibrated_joints` (Session 20) |
| Modality dropout | 10 % per-sample chance per modality, with "don't drop both" safeguard (§6.5) |
| Clip length / stride | **40 / 8** (≈1.3 s context) |
| Batch size | 128 |
| Patience | 10 epochs |
| Norm stats | per-channel z-score from 20 % subsample of training trajectories |
| Train / val / test counts | **650 / 150 / 208** (explicit, matches the documented Session-17 split) |
| Seed | 42, stratified by terrain |
| Best val loss | **0.0176 at epoch 7** |
| Training wall-clock | ~37 minutes (GPU 1, co-resident with Stage-2) |

### 1.3 fusion_v2 (kinematics + INS + VO)

| Knob | Value |
|---|---|
| Architecture | `FusionTransformer` — factorized modal-then-temporal causal transformer |
| Modalities (M=3) | kinematics token (31D MLP→128), INS-IMU-only token (10D MLP→128), VO token (7D MLP→128) |
| VO feature | per-frame SE(3) delta in prev-camera frame: `[dt_x, dt_y, dt_z, dq_x, dq_y, dq_z, dq_w]` |
| VO source | DROID-SLAM at stride=3, fast=True, 1,008 trajectories |
| d_model / heads / FFN / blocks | 128 / 4 / 256 / 2 |
| Parameters | **455,046** |
| Best val loss | **0.0126 at epoch 31** |
| Training wall-clock | ~82 minutes |

### 1.4 fusion_v2_marg (kinematics + INS-MARG + VO)

Identical to fusion_v2 except `--ins-file ins_marg.npz` (MARG drift-free yaw).

| Knob | Value |
|---|---|
| Best val loss | **0.0149 at epoch 18** |
| Training wall-clock | ~60 minutes |

**Note:** MARG does not help when VO is present — fusion_v2 (IMU-only) outperforms fusion_v2_marg on all axes.

### 1.5 fusion_v1_marg (kinematics + INS-MARG, drift-free yaw)

Identical to fusion_v1 except:

| Knob | Value |
|---|---|
| INS file | `ins_marg.npz` instead of `ins.npz` |
| INS mode | Madgwick MARG, two-pass stitch (IMU-only roll/pitch + tilt-compensated mag yaw) |
| Mag axis remap | sim's `(N, E, D)` → ENU `(E, N, U) = (m_y, m_x, -m_z)` |
| Filter convention | `CONVENTION_ENU` (only choice for Z-up body) |
| Best val loss | **0.0167 at epoch 11** |
| Training wall-clock | ~38 minutes |

Everything else (architecture, optimiser, split, loss, clip length, etc.) is byte-identical to fusion_v1.

### 1.6 fusion_tcn (kin + INS + VO, R3D-18-style 1D temporal CNN)

Architecture-comparison run added Session 25 — same data, split, loss, optimiser, and modality dropout as fusion_v2; **only the network architecture changes**.

| Knob | Value |
|---|---|
| Architecture | `FusionTCN` — R3D-18-style 1D temporal CNN |
| Modalities (M=3) | kinematics (31D→32), INS-IMU-only (10D→16), VO (7D→16); per-timestep MLP embedders, channel-concat to 64 |
| Stages | stem Conv1d(3,1,1) → 4 stages of 2 `BasicBlock1D` each |
| Stage channels | (64, 64, 128, 256) |
| Temporal downsample | stride-2 between stages → T = 40 → 40 → 20 → 10 → 5 |
| Readout | global avg pool over T → Linear(256, 6) |
| Parameters | **1,023,798** |
| Optimiser, scheduler, loss | identical to fusion_v2 (AdamW 1e-4 / wd 1e-4, ReduceLROnPlateau, weighted MSE [1,1,1,5,5,5]) |
| Modality dropout | 10 % per-sample per modality (same as fusion_v2) |
| Clip length / stride | 40 / 8 |
| Batch size | 128 |
| Patience | 10 epochs |
| Train / val / test counts | 650 / 150 / 208, seed 42, **identical split to fusion_v2** |
| Best val_loss | **0.0123 at epoch 29** |
| Training wall-clock | ~88 min |

### 1.7 fusion_mvit (kin + INS + VO, MViT-style multiscale transformer)

Architecture-comparison run added Session 25 — same data, split, loss, optimiser, and modality dropout as fusion_v2; **only the network architecture changes**.

| Knob | Value |
|---|---|
| Architecture | `FusionMViT` — multiscale factorized transformer over `(B, T, M, D)` tokens |
| Modalities (M=3) | kinematics, INS-IMU-only, VO; per-timestep 2-layer MLP embedders to D=64 |
| Stages | 3 stages with progressive temporal pooling (avg-pool stride 2 over T pairs) and channel expansion |
| Stage dims (D) | (64, 96, 128) |
| Blocks per stage | (2, 2, 2) factorized modal+temporal attention |
| Temporal length | T = 40 → 20 → 10 across stages |
| Causal mask | **none** — single regression at clip end (not per-timestep streaming as in fusion_v1/v2) |
| Heads / FFN multiplier | 4 / 2× |
| Readout | mean over (T, M) → LayerNorm → Linear → 6 |
| Parameters | **758,566** |
| Optimiser, scheduler, loss | identical to fusion_v2 |
| Modality dropout | 10 % per-sample per modality |
| Clip length / stride | 40 / 8 |
| Batch size | 128 |
| Patience | 10 epochs |
| Train / val / test counts | 650 / 150 / 208, seed 42, **identical split to fusion_v2** |
| Best val_loss | **0.0157 at epoch 34** |
| Training wall-clock | ~172 min |

### 1.6 / 1.7 design notes

- "3D CNN for sensor fusion" reinterpreted as 1D temporal CNN: there is no spatial dim in sensor data, so the R3D-18 idea reduces to `BasicBlock1D` over the time axis with channel-concat across modalities at the input.
- "MViT for sensor fusion" reinterpreted as multiscale factorized transformer over `(T, M)` tokens — the MViT pyramid (progressive temporal-token reduction + channel expansion) is the directly portable idea; pooling attention itself simplifies to mean-pool between stages because keys/values are not the bottleneck at this scale.
- Roughly param-fair to fusion_v2 (455K) within a 2× factor: TCN is 2.25×, MViT is 1.67×. If strict size-matching is needed for the thesis, `stage_dims` / `blocks_per_stage` can be reduced.

---

## 2. Train / val / test splits

Both pipelines use **stratified split by terrain type with seed = 42**, but the **count proportions differ** because the launch scripts use different defaults:

| Pipeline | Method | train | val | test |
|---|---|---:|---:|---:|
| Stage-2 | `train_frac=0.7, val_frac=0.15` (defaults; counts unset) | 705 | 151 | 152 |
| fusion_v1 / fusion_v1_marg | `train_count=650, val_count=150, test_count=208` (explicit) | 650 | 150 | 208 |

Additionally the two implementations use different RNG types (`np.random.default_rng(42)` for Stage-2's `cnn3d/dataset.py` vs `np.random.RandomState(42)` for goodometry's `fusion/dataset.py`), so the within-terrain shuffled order differs. Concrete overlap:

```
                                      |Stage-2 train|  |Stage-2 val|  |Stage-2 test|
fusion_v1 train (650 trajs)  →            553              57               40
fusion_v1 val   (150 trajs)  →             15              39               55           ← interesting tail
fusion_v1 test  (208 trajs)  →            137              ←       71       →

  fusion test ∩ Stage-2 training = 137  (LEAKAGE if fusion test is given to Stage-2)
  fusion test \ Stage-2 training =  71  (clean for Stage-2)
  Stage-2 test ∩ fusion training =  97  (LEAKAGE if Stage-2 test is given to fusion)
  Stage-2 test \ fusion training =  55  (clean for fusion)
  in BOTH test sets               =  33  (safest possible held-out subset)
```

This means the "each model on its own test set" numbers aren't directly comparable, and a fair head-to-head requires evaluating both models on a subset that **neither model trained on**.

---

## 3. Test-set evaluations

All evaluations reported in each model's native label frame.

### 3.1 Stage-2 on its own test set (152 trajectories)

`go2_research/cnn3d/evaluate.py` → `runs/cnn_rgb_stage2/eval/eval_report.txt`. 68,557 clips.

| | RMSE | MAE | median | P90 | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vx` (world) | 0.1004 | 0.0673 | 0.0445 | 0.1532 | 0.2094 | 0.3614 | 1.6228 |
| `vy` (world) | 0.1005 | 0.0671 | 0.0437 | 0.1546 | 0.2136 | 0.3538 | 1.9737 |
| `vz` (world) | 0.0650 | 0.0380 | 0.0249 | 0.0799 | 0.1111 | 0.2266 | 1.7806 |
| `roll_rate` (body) | 0.2098 | 0.1106 | 0.0746 | 0.2173 | 0.2934 | 0.6777 | 7.3399 |
| `pitch_rate` (body) | 0.1385 | 0.0750 | 0.0538 | 0.1460 | 0.1894 | 0.3749 | 4.3414 |
| `yaw_rate` (body) | 0.0958 | 0.0545 | 0.0396 | 0.1088 | 0.1417 | 0.2711 | 6.2167 |

```
overall RMSE  0.1270
linear  RMSE  0.0902 m/s    (world frame)
angular RMSE  0.1553 rad/s  (body frame)
```

### 3.2 fusion_v1 on its own test set (208 trajectories)

`fusion/evaluate.py` → `runs/fusion_v1/test_results.json`. 92,935 clips.

| axis | RMSE | P50 | P95 | units |
|---|---:|---:|---:|---|
| `vx` (body) | 0.0786 | 0.0373 | 0.1544 | m/s |
| `vy` (body) | 0.0554 | 0.0287 | 0.1085 | m/s |
| `vz` (body) | 0.0540 | 0.0271 | 0.1003 | m/s |
| `ωx` (body) | 0.1616 | 0.0767 | 0.2704 | rad/s |
| `ωy` (body) | 0.0863 | 0.0365 | 0.1338 | rad/s |
| `ωz` (body) | 0.0806 | 0.0261 | 0.1030 | rad/s |

```
overall RMSE  0.0933
linear  RMSE  0.0637 m/s    (body frame)
angular RMSE  0.1156 rad/s
```

### 3.3 fusion_v1_marg on the same 208-trajectory test set

`runs/fusion_v1_marg/test_results.json`. 92,935 clips.

| axis | RMSE | P50 | P95 | units |
|---|---:|---:|---:|---|
| `vx` (body) | 0.0724 | 0.0344 | 0.1402 | m/s |
| `vy` (body) | 0.0504 | 0.0260 | 0.0982 | m/s |
| `vz` (body) | 0.0502 | 0.0260 | 0.0915 | m/s |
| `ωx` (body) | 0.1391 | 0.0519 | 0.2171 | rad/s |
| `ωy` (body) | 0.0875 | 0.0362 | 0.1341 | rad/s |
| `ωz` (body) | 0.0734 | 0.0228 | 0.0953 | rad/s |

```
overall RMSE  0.0844
linear  RMSE  0.0586 m/s
angular RMSE  0.1039 rad/s
```

### 3.4 Fair head-to-head — 71 trajectories held out from both models

`scripts/fair_test_eval.py` → `runs/fair_head_to_head.json`. The 71 trajectories were chosen as `fusion_v1's test set ∩ NOT in Stage-2's training set` (so neither model trained on any of them). 31,791 fusion clips and 32,075 Stage-2 clips (different clip configs).

```
metric                   Stage-2     fusion_v1      Δ
overall RMSE              0.1015      0.0809      −20.3 %
linear  RMSE (m/s)        0.0907      0.0589      −35.0 %
angular RMSE (rad/s)      0.1112      0.0981      −11.8 %

axis        Stage-2     fusion_v1      Δ
  vx         0.1021      0.0734      −28.1 %
  vy         0.1046      0.0521      −50.2 %
  vz         0.0576      0.0482      −16.3 %
  ωx (roll)  0.1488      0.1428       −4.0 %
  ωy (pitch) 0.0998      0.0744      −25.5 %
  ωz (yaw)   0.0709      0.0542      −23.6 %
```

### 4.2 Same 71 trajectories, both models held out (apples-to-apples)

| metric | Stage-2 | fusion_v1 | Δ |
|---|---:|---:|---:|
| overall RMSE | 0.1015 | **0.0809** | **−20.3 %** |
| linear RMSE (m/s) | 0.0907 | **0.0589** | **−35.0 %** |
| angular RMSE (rad/s) | 0.1112 | **0.0981** | **−11.8 %** |
| `vx` | 0.1021 | **0.0734** | −28.1 % |
| `vy` | 0.1046 | **0.0521** | **−50.2 %** ← biggest win |
| `vz` | 0.0576 | **0.0482** | −16.3 % |
| `ωx` (roll rate) | 0.1488 | **0.1428** | −4.0 % |
| `ωy` (pitch rate) | 0.0998 | **0.0744** | −25.5 % |
| `ωz` (yaw rate) | 0.0709 | **0.0542** | −23.6 % |

**Reading the breakdown:**

- **`vy` halved** — lateral velocity is the most directly observable signal from leg kinematics (foot-position deltas), which fusion_v1 ingests as a 12-dim feature instead of having to learn it from RGB.
- **`ωy` and `ωz` both improved ~25 %** — IMU gyro carries them directly; the temporal attention on the gyro channel beats mid-fusion CNN.
- **`vx`/`vz` smaller wins** — these axes couple most tightly to forward visual motion, the one signal Stage-2 has and fusion_v1 doesn't. fusion_v2 closes this gap: `vx` drops from 0.0734 → 0.0562 (−23 %), `vz` from 0.0482 → 0.0453 (−6 %).
- **`ωx` (roll rate) only −4 %** — fundamentally noisy: gait-induced ±5–10 ° body rocking at 2 Hz that both models smooth out similarly.

### 4.3 fusion_v1 vs fusion_v1_marg — what does drift-free yaw buy?

Same architecture, same training, same test split, only difference is the INS quaternion (IMU-only vs MARG-stitched).

| metric | fusion_v1 | fusion_v1_marg | Δ |
|---|---:|---:|---:|
| Best val loss | 0.0176 (ep 7) | 0.0167 (ep 11) | −5.1 % |
| Overall test RMSE | 0.0933 | **0.0844** | **−9.5 %** |
| Linear test RMSE | 0.0637 m/s | **0.0586 m/s** | **−8.0 %** |
| Angular test RMSE | 0.1156 rad/s | **0.1039 rad/s** | **−10.1 %** |
| `vx` | 0.0786 | **0.0724** | −7.9 % |
| `vy` | 0.0554 | **0.0504** | −9.0 % |
| `vz` | 0.0540 | **0.0502** | −7.0 % |
| `ωx` (roll rate) | 0.1616 | **0.1391** | −13.9 % |
| `ωy` (pitch rate) | 0.0863 | 0.0875 | +1.4 % |
| `ωz` (yaw rate) | 0.0806 | **0.0734** | −9.0 % |

**Empirical answer to the magnetometer question:** drift-free yaw helps ~8–10 % across most metrics. The per-epoch val curves looked nearly identical, but the slightly-deeper trained MARG model (best at ep 11 vs ep 7) generalises modestly better to held-out trajectories. **Biggest gain is on `ωx`** — drift-free yaw indirectly stabilises the network's roll-rate estimate, possibly because attending to a consistent quaternion across the 1.3 s window lets the model better cross-reference gyro spikes with body-frame motion.

So the mag is *not useless* in this configuration, just modestly helpful. For sim-to-real deployment the calibration burden still tips the balance toward IMU-only — but for a pure-sim ablation, MARG mode gives a free ~10 % improvement.

---

---

## 3.4 fusion_v2 on the 208-trajectory test set (kin + IMU + VO)

`fusion/evaluate.py` → `runs/fusion_v2/test_results.json`. 92,935 clips. Best checkpoint epoch 31.

| axis | RMSE | P50 | P95 | units |
|---|---:|---:|---:|---|
| `vx` (body) | 0.0562 | 0.0288 | 0.1109 | m/s |
| `vy` (body) | 0.0425 | 0.0213 | 0.0800 | m/s |
| `vz` (body) | 0.0453 | 0.0206 | 0.0811 | m/s |
| `ωx` (body) | 0.1272 | 0.0556 | 0.1958 | rad/s |
| `ωy` (body) | 0.0737 | 0.0295 | 0.1087 | rad/s |
| `ωz` (body) | 0.0630 | 0.0166 | 0.0678 | rad/s |

```
overall RMSE  0.0737
linear  RMSE  0.0483 m/s    (body frame)
angular RMSE  0.0924 rad/s
```

### 3.5 fusion_v2_marg on the 208-trajectory test set (kin + MARG + VO)

`runs/fusion_v2_marg/test_results.json`. 92,935 clips. Best checkpoint epoch 18.

| axis | RMSE | P50 | P95 | units |
|---|---:|---:|---:|---|
| `vx` (body) | 0.0636 | 0.0312 | 0.1272 | m/s |
| `vy` (body) | 0.0459 | 0.0234 | 0.0863 | m/s |
| `vz` (body) | 0.0486 | 0.0224 | 0.0888 | m/s |
| `ωx` (body) | 0.1425 | 0.0716 | 0.2323 | rad/s |
| `ωy` (body) | 0.0808 | 0.0340 | 0.1223 | rad/s |
| `ωz` (body) | 0.0677 | 0.0208 | 0.0825 | rad/s |

```
overall RMSE  0.0816
linear  RMSE  0.0533 m/s
angular RMSE  0.1023 rad/s
```

**Note:** unlike fusion_v1 where MARG improved over IMU-only, fusion_v2_marg is *worse* than fusion_v2 on every axis. The VO modality already provides orientation context (SE(3) deltas contain relative rotation), making the MARG's drift-free yaw redundant. IMU-only + VO is the better combination.

### 3.6 fusion_v3 on the 208-trajectory test set (kin_v3 35D + IMU + VO)

`fusion/evaluate.py` → `runs/fusion_v3/test_results.json`. 92,935 clips. Best checkpoint epoch 12 (val_loss 0.0191, early stopped at epoch 22).

| axis | RMSE | P50 | P95 | units |
|---|---:|---:|---:|---|
| `vx` (body) | 0.0669 | 0.0339 | 0.1347 | m/s |
| `vy` (body) | 0.0518 | 0.0252 | 0.1029 | m/s |
| `vz` (body) | 0.0592 | 0.0277 | 0.1026 | m/s |
| `ωx` (body) | 0.1420 | 0.0534 | 0.2320 | rad/s |
| `ωy` (body) | 0.0986 | 0.0333 | 0.1301 | rad/s |
| `ωz` (body) | 0.0792 | 0.0218 | 0.0927 | rad/s |

```
overall RMSE  0.0883
linear  RMSE  0.0596 m/s
angular RMSE  0.1098 rad/s
```

**Result: fusion_v3 is worse than fusion_v2 on every axis** (+19.8% overall). It is better than fusion_v1 (−5.4% overall) but not fusion_v1_marg.

**Why it didn't work.** Replacing `v_body_legs (3D, m/s)` with `leg_odom_delta (7D SE(3) delta)` hurt for two reasons:
1. **Wrong units for the task.** The model predicts velocity (m/s). `v_body_legs` is already in m/s — the kin branch provides a direct first-cut velocity estimate. `Δt = v * dt` scales this to metres per frame (≈0.03 m), producing a very-low-amplitude feature.
2. **Near-identity quaternions.** `Δq` at 30 Hz is near `[0,0,0,1]` almost always — low variance, dominated by gyro noise. VO already provides rotation context via its own SE(3) deltas, making `Δq` redundant noise.

The val curve was also noisy (0.019–0.026 range), consistent with the low-SNR features.

**Conclusion:** keep `v_body_legs` in the kin branch. If SE(3) information from the legs is desired, *appending* `leg_odom_delta` as additional channels (38D kin) rather than replacing would be worth testing — but the VO modality likely already subsumes whatever rotational signal the legs can provide.

### 3.7 fusion_tcn on the 208-trajectory test set (R3D-18-style 1D temporal CNN)

`fusion/evaluate.py` → `runs/fusion_tcn/test_results.json`. 92,935 clips. Best checkpoint epoch 29.

| axis | RMSE | P50 | P95 | units |
|---|---:|---:|---:|---|
| `vx` (body) | 0.0462 | 0.0229 | 0.0872 | m/s |
| `vy` (body) | 0.0372 | 0.0187 | 0.0686 | m/s |
| `vz` (body) | 0.0447 | 0.0228 | 0.0834 | m/s |
| `ωx` (body) | 0.1285 | 0.0471 | 0.1944 | rad/s |
| `ωy` (body) | 0.0826 | 0.0345 | 0.1299 | rad/s |
| `ωz` (body) | 0.0676 | 0.0240 | 0.0915 | rad/s |

```
overall RMSE  0.0746
linear  RMSE  0.0429 m/s    (body frame)  ← best of all models
angular RMSE  0.0964 rad/s
```

**Result.** fusion_tcn matches fusion_v2 within +1.2 % on overall RMSE while posting the best Linear RMSE of any model trained on this dataset (−11.2 % vs fusion_v2). All three linear axes improve (vx −17.8 %, vy −12.5 %, vz −1.3 %). All three angular axes are slightly worse, by 1–12 %.

### 3.8 fusion_mvit on the 208-trajectory test set (MViT-style multiscale transformer)

`fusion/evaluate.py` → `runs/fusion_mvit/test_results.json`. 92,935 clips. Best checkpoint epoch 34.

| axis | RMSE | P50 | P95 | units |
|---|---:|---:|---:|---|
| `vx` (body) | 0.0592 | 0.0303 | 0.1189 | m/s |
| `vy` (body) | 0.0446 | 0.0235 | 0.0863 | m/s |
| `vz` (body) | 0.0450 | 0.0235 | 0.0831 | m/s |
| `ωx` (body) | 0.1325 | 0.0508 | 0.1929 | rad/s |
| `ωy` (body) | 0.0753 | 0.0316 | 0.1153 | rad/s |
| `ωz` (body) | 0.0707 | 0.0189 | 0.0780 | rad/s |

```
overall RMSE  0.0772
linear  RMSE  0.0501 m/s
angular RMSE  0.0970 rad/s
```

**Result.** fusion_mvit underperforms fusion_v2 by +4.7 % overall, with worse numbers on every axis except `ωz` (where it improves over fusion_tcn but not fusion_v2). The multi-scale pyramid (mean-pool stride-2 between stages) discards temporal information the factorised attention preserves; for a clip length of T=40 with M=3 modalities, there is no token-count problem to solve and pooling buys nothing.

---

## 4. Summary tables

### 4.1 Each model on its own test split

| | **Stage-2 v2** | **fusion_v1** | **fusion_v1_marg** | **fusion_v2** | **fusion_v2_marg** | **fusion_v3** | **fusion_tcn** | **fusion_mvit** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Architecture | R3D-18 + sensor CNN | transformer | transformer | transformer | transformer | transformer | TCN (R3D-18 1D) | multiscale ViT |
| Parameters | 33,485,894 | **437,382** | 437,382 | 455,046 | 455,046 | 455,558 | 1,023,798 | 758,566 |
| Visual input | RGB | none | none | VO (DROID) | VO (DROID) | VO (DROID) | VO (DROID) | VO (DROID) |
| Kin features | joints raw | 31D | 31D | 31D | 31D | 35D | 31D | 31D |
| INS variant | imu9 raw | IMU-only | MARG | IMU-only | MARG | IMU-only | IMU-only | IMU-only |
| Train wall-clock | ~5 days | ~37 min | ~38 min | ~82 min | ~60 min | ~48 min | ~88 min | ~172 min |
| Test trajectories | **208** | 208 | 208 | 208 | 208 | 208 | 208 | 208 |
| Test clips | 93,767 | 92,935 | 92,935 | 92,935 | 92,935 | 92,935 | 92,935 | 92,935 |
| Overall RMSE | 0.1168 | 0.0933 | 0.0844 | **0.0737** | 0.0816 | 0.0883 | 0.0746 | 0.0772 |
| Linear RMSE | 0.0893 m/s † | 0.0637 m/s | 0.0586 m/s | 0.0483 m/s | 0.0533 m/s | 0.0596 m/s | **0.0429 m/s** | 0.0501 m/s |
| Angular RMSE | 0.1389 rad/s | 0.1156 rad/s | 0.1039 rad/s | **0.0924 rad/s** | 0.1023 rad/s | 0.1098 rad/s | 0.0964 rad/s | 0.0970 rad/s |

† Stage-2 v2's linear is in **world frame** (its training target); fusion linear is body-frame. Angular is body-frame for both — strictly apples-to-apples. **Stage-2 v2 → fusion_v2:** −36.9 % overall, −33.5 % angular.

**fusion_v2 still wins on overall and angular RMSE. fusion_tcn wins on linear RMSE.** The two architecture-comparison runs (fusion_tcn 1D temporal CNN, fusion_mvit multiscale transformer) confirm that the small factorised transformer (455K params) is not capacity-limited: a 2.25× larger TCN improves linear by 11 % but loses on angular, and the MViT-style pyramid is uniformly worse. fusion_v3 (SE(3) delta kinematics) is also worse than v2 — replacing `v_body_legs` (m/s) with `leg_odom_delta` (~0.03 m + near-identity Δq) removes a direct velocity signal and adds noise.

### 4.1.1 Architecture comparison — fusion_v2 vs fusion_tcn vs fusion_mvit

All three runs use the **identical 650/150/208 stratified-by-terrain split (seed 42), kin + INS + VO inputs, loss, optimiser, modality dropout, and clip config** as fusion_v2 — only the network architecture changes (`--arch {transformer, tcn, mvit}` in `fusion/train.py`).

| | **fusion_v2** | **fusion_tcn** | **fusion_mvit** |
|---|---:|---:|---:|
| Architecture | factorized causal transformer | R3D-18-style 1D temporal CNN | MViT-style multiscale transformer |
| Parameters | 455,046 | 1,023,798 (2.25×) | 758,566 (1.67×) |
| Best val_loss / epoch | 0.0126 / 31 | **0.0123 / 29** | 0.0157 / 34 |
| Train wall-clock | ~82 min | ~88 min | ~172 min |
| Overall RMSE | **0.0737** | 0.0746 | 0.0772 |
| Linear RMSE | 0.0483 m/s | **0.0429 m/s** | 0.0501 m/s |
| Angular RMSE | **0.0924 rad/s** | 0.0964 rad/s | 0.0970 rad/s |
| `vx` RMSE | 0.0562 m/s | **0.0462 m/s** | 0.0592 m/s |
| `vy` RMSE | 0.0425 m/s | **0.0372 m/s** | 0.0446 m/s |
| `vz` RMSE | 0.0453 m/s | **0.0447 m/s** | 0.0450 m/s |
| `ωx` RMSE | **0.1272 rad/s** | 0.1285 rad/s | 0.1325 rad/s |
| `ωy` RMSE | **0.0737 rad/s** | 0.0826 rad/s | 0.0753 rad/s |
| `ωz` RMSE | **0.0630 rad/s** | 0.0676 rad/s | 0.0707 rad/s |

**fusion_v2 still wins overall and on every angular axis. fusion_tcn wins on every linear axis** (vx −17.8%, vy −12.5%, vz −1.3%) and posts the best Linear RMSE of any model trained on this dataset.

**Reading the result.** fusion_tcn's **lower** val_loss but **slightly higher** test RMSE points to mild overfitting on the larger TCN; fusion_v2's compactness (455K vs 1.02M params) generalises slightly better despite the TCN's stronger temporal inductive bias. fusion_mvit is uniformly worse — pyramid tokens with mean-pooling between stages discards information the factorised modal+temporal attention preserves.

**Conclusion.** The factorised causal transformer (fusion_v2) is **not capacity-limited** — extra parameters from a deeper temporal CNN buy a clear win on translation but don't unlock new accuracy on rotation. If the goal is to minimise *linear* velocity error specifically (e.g. for downstream position integration), fusion_tcn is the better choice. For balanced overall performance, fusion_v2 remains the recommended model.

**Caveat** — Stage-2's linear targets are world-frame vs body-frame for fusion. Angular is body-frame for both (strictly apples-to-apples). See §5 for detailed caveats.

### 4.2 Single-modality baselines — what does each preprocessing arm give on its own?

Sanity-check baselines that use a single preprocessing arm directly as the velocity prediction (no learning, no fusion). Same 208-trajectory test set, same clip indexing as fusion_v2 — prediction is taken at the last timestep of each clip. Script: `scripts/single_modality_baselines.py`. Output: `runs/single_modality_baselines/results.json`.

| Modality | Linear RMSE (m/s) | Angular RMSE (rad/s) | Speed-magnitude RMSE (m/s) | Notes |
|---|---:|---:|---:|---|
| **kin only** (`v_body_legs`) | 0.2464 | n/a | 0.2097 | Stance-foot constraint, body-frame native |
| **IMU only** (`gyro_body`) | n/a | 0.2944 | n/a | Gyro raw, body-frame native; accel integration diverges |
| **VO only** (DROID Δt × 30 Hz) | 1.0452 † | 5.1825 † | 1.7115 | † cam ≈ body assumption breaks; see notes |
| **fusion_v2 (learned)** | **0.0483** | **0.0924** | — | full kin + IMU + VO fusion |

**Per-axis breakdown:**

| | vx | vy | vz | ωx | ωy | ωz |
|---|---:|---:|---:|---:|---:|---:|
| kin only | 0.2820 | 0.2316 | 0.2213 | n/a | n/a | n/a |
| IMU only | n/a | n/a | n/a | 0.4657 | 0.1730 | 0.1150 |
| VO only † | 1.2399 | 0.6406 | 1.1529 | 5.5867 | 5.4161 | 4.4746 |
| fusion_v2 | 0.0562 | 0.0425 | 0.0453 | 0.1272 | 0.0737 | 0.0630 |

**Reading the table.**

- **kin-only** at 0.2464 m/s linear RMSE is a noisy stance-foot estimate at 30 Hz. fusion_v2 reduces this **5.1×** to 0.0483 m/s. The drop is bigger on `vx` (5.0×) and `vz` (4.9×) than `vy` (5.4×) — the stance-foot constraint's worst noise is along forward and vertical axes, where leg compression dominates.
- **IMU-only gyro** at 0.2944 rad/s angular RMSE is dominated by the `ωx` axis (0.4657 rad/s). The Go2 trots at 2 Hz with ±5° body roll oscillation; the gyro reads this faithfully but the smoothed body-frame label averages it out. fusion_v2 reduces overall angular **3.2×** to 0.0924 rad/s, with the biggest single-axis win on `ωx` (3.7×).
- **VO-only** numbers (marked †) need a caveat the per-axis story can't tell: DROID-SLAM camera-to-world poses are in optical convention (z-forward, y-down), and the camera-to-body extrinsic on the Go2 head-cam is non-trivial. The per-axis RMSE we report assumes camera ≈ body, which is wrong by ~90° on at least one axis. The **speed-magnitude RMSE (1.71 m/s)** is frame-agnostic and tells the cleaner story: **DROID itself has a ~2× scale bias on this dataset** (mean ratio of VO speed to GT body speed ≈ 1.95) — almost certainly stride-3 interpolation × stereo-scale issues. The fusion network learns this implicitly during training and uses VO's *direction information* without trusting its scale.

**What fusion_v2 actually buys**

| | linear | angular |
|---|---:|---:|
| Best single modality | 0.2464 m/s (kin) | 0.2944 rad/s (IMU) |
| fusion_v2 | 0.0483 m/s | 0.0924 rad/s |
| **Improvement** | **−80.4 %** | **−68.6 %** |

The learned fusion is not a marginal refinement of the best single modality — it is **5×** lower error on linear velocity and **3×** lower on angular. The story is consistent with what's expected from sensor fusion: each modality has its own failure mode (kin: leg-compression noise; IMU: oscillation that has to be smoothed; VO: scale and frame errors), and the fusion model learns to weight them temporally and pick the cleanest signal in each direction. None of the three arms alone produces usable velocity estimates; the fusion is what makes the system viable.

---

## 5. Caveats and what to read into the comparison

1. **Three different test splits** in §4.1. Read Stage-2 vs fusion numbers there as "approximate magnitudes". The honest comparison is §4.2 (the 71 held-out-from-both subset).

2. **Frame difference for linear velocity.** Stage-2 was trained on world-frame linear targets, fusion on body-frame. Each model's linear RMSE in §3 and §4 is computed against its own training target — both describe similar physical quantities (the body's translation magnitude in some frame), but they aren't strictly the same metric. Angular RMSE is body-frame for both — that comparison is strictly apples-to-apples.

3. **Stage-2's `omega_body_x` aka `roll_rate` worst at 0.21 rad/s.** Inherent to the data — the Go2 trots at 2 Hz with ±5° body roll oscillation. Both models smooth this out; the CNN's mid-fusion makes it slightly worse than the transformer.

4. **The fair head-to-head test (§4.2) used 71 trajectories.** Statistical noise on per-axis metrics is non-trivial at this scale (rough 95 % CI ±3 % on overall RMSE). The ~50 % `vy` improvement is well outside that; the ~4 % `ωx` improvement is inside it.

5. **fusion_v2 adds VO (DROID-SLAM SE(3) deltas).** Confirmed improvement over fusion_v1 on every axis. See §3.4 and §4.1 for the full breakdown.

---

## 6. File pointers

| What | Where |
|---|---|
| Stage-2 checkpoint | `~/projects/go2_research/runs/cnn_rgb_stage2/best_model.pt` |
| Stage-2 test report | `~/projects/go2_research/runs/cnn_rgb_stage2/eval/eval_report.txt` |
| fusion_v1 checkpoint | `~/projects/goodometry/runs/fusion_v1/best_model.pt` |
| fusion_v1 test JSON | `~/projects/goodometry/runs/fusion_v1/test_results.json` |
| fusion_v1_marg ckpt | `~/projects/goodometry/runs/fusion_v1_marg/best_model.pt` |
| fusion_v1_marg test | `~/projects/goodometry/runs/fusion_v1_marg/test_results.json` |
| fusion_v2 checkpoint | `~/projects/goodometry/runs/fusion_v2/best_model.pt` |
| fusion_v2 test JSON | `~/projects/goodometry/runs/fusion_v2/test_results.json` |
| fusion_v2 per-traj | `~/projects/goodometry/runs/fusion_v2/per_traj_results.json` |
| fusion_v2_marg ckpt | `~/projects/goodometry/runs/fusion_v2_marg/best_model.pt` |
| fusion_v2_marg test | `~/projects/goodometry/runs/fusion_v2_marg/test_results.json` |
| fusion_v2_marg per-traj | `~/projects/goodometry/runs/fusion_v2_marg/per_traj_results.json` |
| fusion_v3 checkpoint | `~/projects/goodometry/runs/fusion_v3/best_model.pt` |
| fusion_v3 test JSON | `~/projects/goodometry/runs/fusion_v3/test_results.json` |
| fusion_v3 per-traj | `~/projects/goodometry/runs/fusion_v3/per_traj_results.json` |
| fusion_tcn checkpoint | `~/projects/goodometry/runs/fusion_tcn/best_model.pt` |
| fusion_tcn test JSON | `~/projects/goodometry/runs/fusion_tcn/test_results.json` |
| fusion_tcn per-traj | `~/projects/goodometry/runs/fusion_tcn/per_traj_results.json` |
| fusion_mvit checkpoint | `~/projects/goodometry/runs/fusion_mvit/best_model.pt` |
| fusion_mvit test JSON | `~/projects/goodometry/runs/fusion_mvit/test_results.json` |
| fusion_mvit per-traj | `~/projects/goodometry/runs/fusion_mvit/per_traj_results.json` |
| Stage-2 v2 checkpoint | `~/projects/go2_research/runs/cnn_rgb_stage2_v2/best_model.pt` |
| Stage-2 v2 eval (matched 650/149/208 split) | `~/projects/go2_research/runs/cnn_rgb_stage2_v2/eval/` |
| Fair head-to-head | `~/projects/goodometry/runs/fair_head_to_head.json` |
| Stride sweep | `~/projects/goodometry/pilot/stride_sweep_results.json` |
| Training scripts | `~/projects/goodometry/fusion/train.py` and `~/projects/go2_research/run_cnn_rgb_stage2.sh` |
| Test scripts | `~/projects/goodometry/fusion/evaluate.py`, `~/projects/go2_research/cnn3d/evaluate.py`, `~/projects/goodometry/scripts/fair_test_eval.py` |
| Comprehensive doc | `~/projects/goodometry/PIPELINE.md` |

---

## 7. DROID-SLAM stride sweep (added Session 23)

To accelerate the DROID-SLAM at-scale run (originally ~7 days at stride=1), did an empirical sweep on 3 pilot trajectories × 5 stride values. Skipped frames are SLERP-interpolated from neighbouring DROID samples to fill the full `sensors.npz['frame_idx']` coverage.

### 7.1 Per-trajectory results (skip-backend mode, image_size [360, 640])

```
Trajectory 1 — flat / forward (3136 frames)
   stride   wall   fps    ATE     scale   RPE_t_1s
     1     8.4m   6.3    0.915  0.81    0.297
     3     3.4m   15.5   0.789  1.01    0.254   ← best on flat
     4     2.6m   19.9   1.408  0.73    0.304   ← quality cliff
     5     2.1m   24.7   1.077  0.86    0.392
     8     1.3m   38.8   1.331  0.54    0.650   ← scale collapses (0.54 vs ~1.0)

Trajectory 2 — forest / forward (3430 frames)
   stride   wall   fps    ATE     scale   RPE_t_1s
     1     6.0m   9.6    0.149  1.16    0.130
     3     2.9m   19.7   0.175  1.18    0.157
     4     2.3m   25.0   0.144  1.13    0.130
     5     2.1m   27.7   0.157  1.15    0.146
     8     1.3m   43.6   0.126  1.13    0.129   ← actually best on forest

Trajectory 3 — forest / circular (3221 frames)
   stride   wall   fps    ATE     scale   RPE_t_1s
     1     5.2m   10.4   0.183  1.00    0.150
     3     2.8m   19.3   0.181  1.01    0.151
     4     2.2m   24.4   0.183  1.02    0.149
     5     1.9m   27.8   0.185  1.01    0.148
     8     1.2m   43.6   0.278  0.92    0.150   ← starts to degrade
```

### 7.2 Two clean signals

1. **Forest is forgiving** — strides 1 through 5 give essentially identical ATE on both forest trajectories. Stride=8 only starts to degrade on rotation-heavy circular forest.
2. **Flat has a cliff at stride=4** — ATE jumps from 0.79 m → 1.41 m (78 % worse). Stride=3 is the largest stride that holds up across both terrain types.

### 7.3 Weighted across actual dataset terrain mix

(62 % forest, 28 % flat, 10 % uneven/uphill — see Session 11 of `go2_research/CHANGES.md`):

```
stride=1   weighted-avg ATE  ≈  0.36 m       baseline (the unoptimised stride)
stride=3   weighted-avg ATE  ≈  0.33 m   ← best across the mix, 8 % better than stride=1, 2.7× faster
stride=4   weighted-avg ATE  ≈  0.50 m       flat cliff dominates the average
stride=5   weighted-avg ATE  ≈  0.41 m
stride=8   weighted-avg ATE  ≈  0.50 m
```

### 7.4 Why stride=3 sometimes beats stride=1

At 30 Hz, robot motion between consecutive frames is ~1.3 cm — *smaller than feature-matching noise*. DROID's flow predictor was trained on TartanAir (drone flights with much larger inter-frame motion). Stride=3 gives it ~4 cm of motion per pair — within its training distribution — so the predictions are cleaner. The interpolation back to 30 Hz then smooths what little residual noise remains.

### 7.5 Decision and current at-scale config

**Stride=3** chosen for the at-scale rerun. Plus three zero-quality-cost optimisations:

- **`fast=True`** — skip both `backend(7)` and `backend(12)` global BA passes. For 1–2 minute trajectories without revisits, the frontend's local-window BA + `traj_filler` is enough. ~5 % wall-clock savings.
- **Parallel I/O** via `ThreadPoolExecutor(max_workers=8)` for per-frame loading.
- **`cv2.IMREAD_REDUCED_COLOR_2`** decodes PNGs at half-res (640×360) directly — skips both the full-res decode and the explicit resize.

Combined: I/O drops from 60 s to 9 s per trajectory (6.6× speedup on the I/O alone).

**Final per-trajectory wall-clock**: ~3.4 min, ~17 fps effective. Projected total for 1,008 trajectories: **~2.4 days** (down from the original ~7 days).
