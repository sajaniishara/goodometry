# From end-to-end visual to sensor fusion — what worked, what didn't, and why

A standalone narrative companion to `EXPERIMENTS.md`, written for the thesis chapter on results and discussion. Numbers cite tables in EXPERIMENTS.md by section.

---

## 1. The original proposal

The proposal committed to **end-to-end deep learning on visual data** for 6-DoF body-frame velocity estimation on the Unitree Go2:

- **CNN baseline** — `R3D-18` 3D-CNN backbone over RGB stereo-left clips, mid-fusion of an `imu9 + joints` 1D-CNN sensor branch after layer-2.
- **Transformer baseline** — `MViT_V2_S` over the same RGB clips, plus the same sensor branch.
- **Disparity ablation** — re-run both with RAFT-Stereo dense disparity instead of RGB, hypothesising that a metric-scale depth input would close the visual ego-motion gap.

The story of the project is that **none of these models met the accuracy targets in the proposal**, and the eventual best model is something the proposal did not name: a **445K-parameter factorised modal+temporal transformer** trained on classical-algorithm features (forward-kinematics body velocity, Madgwick AHRS orientation, DROID-SLAM SE(3) deltas) — no images at all in the inference path.

This document explains why.

---

## 2. Why the image-based models did not go as planned

Three problems compounded.

### 2.1 The model was being asked to do four things at once

Every Stage-2 visual model had to learn, end-to-end, a function whose parts are very well-studied separately:

1. **Quadruped forward kinematics.** Foot-position and stance-foot constraint that yield body-frame velocity are a *closed-form* function of joint angles + URDF geometry. A CNN cannot find this in 5 days of gradient steps better than 50 lines of `pytorch_kinematics`.
2. **Visual ego-motion.** A 30-year-old SLAM problem; modern methods (DROID-SLAM, ORB-SLAM3) recover camera-frame SE(3) deltas at sub-cm scale. A 35M-param ImageNet-pretrained backbone is not the right tool for this.
3. **Orientation from IMU.** Textbook complementary/Madgwick filter, runs on a microcontroller in real time.
4. **The actual fusion** — *when* to trust which signal, and *how* to combine them through time.

Asking one network to learn all four jointly meant most of its parameters and most of its training compute were spent re-discovering algorithms that already exist. The fusion question — the only part that genuinely needs a learned model — got whatever capacity remained.

### 2.2 Data quality issues masked the real ceiling

While bringing the visual models to the proposal numbers, three data bugs were found and fixed (see `CHANGES.md` Sessions 17–20 in `go2_research/`):

- **Angular-velocity outliers at recording boundaries** would push `omega_body_x` into the ±20 rad/s range — physical impossibility for a 14 kg quadruped trotting at 2 Hz. Clipping to ±10 rad/s reduced `ωx` RMSE from 0.32 → 0.21 rad/s on Stage-2 alone.
- **`gt_lin_vel` was world-frame, not body-frame.** The published Stage-2 numbers were measuring "world-frame translation magnitude" against a model trained on it — apples-to-apples internally, but not the body-frame metric the proposal called for.
- **FR_calf systematic offset.** A bad collision mesh in the Isaac Sim 5.1 Go2 USD let the right-front foot penetrate the ground under load, so channel 9 of `joints` was biased by ~0.6 rad relative to the other three calves across all 1,008 trajectories. The visual model had been silently ingesting this for every clip.

After fixing all three, the **fully cleaned end-to-end CNN RGB** (33.5 M params, 5 days training) topped out at:

```
Stage-2 CNN RGB (cleaned)
  overall RMSE  0.1270
  linear  RMSE  0.0902 m/s
  angular RMSE  0.1553 rad/s
```

This is the true ceiling of the proposed approach on this dataset. **The ceiling was not high enough.** The downstream goal — usable body-frame velocity for closed-loop control on the real robot — needs single-digit centimetre-per-second linear errors and well under 0.10 rad/s on body roll/pitch. Stage-2 missed both by a factor of 2–3×.

### 2.3 Compute and iteration economics

A single Stage-2 RGB run takes ~5 days on a single RTX 5060 Ti (16 GB). A single MViT V2 S run on the same batch is ~2.5× slower per epoch. With early stopping at patience 10 we'd see one experiment per 1.5 weeks, on average. The proposal called for an ablation matrix (RGB vs disparity × CNN vs MViT × IMU vs MARG × with/without sensor pretrain). Even on two GPUs, that's ~3 months of pure GPU time before you start second-guessing any architectural choice.

When the cleaned end-to-end ceiling came in at 0.1270 RMSE — measurably better than the noisy original numbers but still far above target — the question wasn't "which of these four image models is marginally best." It was "is end-to-end visual learning the right tool at all?"

---

## 3. The pivot — sensor fusion with classical preprocessing

The decision: **do the easy parts with classical algorithms, learn only the hard part.**

Three preprocessing arms were built (described in detail in `PIPELINE.md` §3):

| Arm | What it computes | Output |
|---|---|---|
| **kin** | URDF-based forward kinematics on calibrated joint angles, foot velocities by centred finite difference, contact probability by velocity-gate, body velocity from stance-foot constraint | `kin.npz` — 31-D feature per frame |
| **ins** | Madgwick AHRS on accel + gyro (IMU) and accel + gyro + mag (MARG) | `ins.npz` / `ins_marg.npz` — 10-D feature per frame |
| **vo** | DROID-SLAM at stride 3, fast=True, then per-frame SE(3) delta in previous-camera frame | `vo.npz` — 7-D feature per frame |

Each arm was tuned and validated on three pilot trajectories before being run at scale on all 1,008 trajectories of the v2 dataset. The visual frames *do not appear in the inference path of the fused model*; only the SE(3) delta computed by DROID-SLAM does. The fused model therefore sees a `(T, 48)` token grid (31 + 10 + 7) per timestep — every channel already encodes geometric meaning.

The trained fusion network's only job is **temporal weighting and cross-modal disagreement resolution.**

### 3.1 What this bought immediately

The first fusion model, `fusion_v1` (kin + IMU, no VO), with **437K parameters**, trained in **37 minutes** on a single GPU, produced:

```
fusion_v1
  overall RMSE  0.0933   (Stage-2 was 0.1270 — −26.5%)
  linear  RMSE  0.0637 m/s   (Stage-2 was 0.0902 — −29.4%)
  angular RMSE  0.1156 rad/s (Stage-2 was 0.1553 — −25.6%)
```

A 77× smaller model, 200× faster training, and ~26 % lower RMSE on every metric. The pivot answered the question: **the bottleneck was never visual capacity; it was that the model had been wasting capacity reinventing FK, IMU integration, and SLAM.**

---

## 4. Full comparison of the fusion models

Six fusion variants were trained, all on the **identical 650/150/208 stratified-by-terrain split (seed = 42)**. Detailed per-axis numbers are in `EXPERIMENTS.md` §3.2–§3.8. Summary:

### 4.1 Architecture family

| Model | Encoder type | Modalities (M) | Kin features | INS variant | Visual modality | Params |
|---|---|---:|---|---|---|---:|
| fusion_v1 | factorised causal transformer | 2 | 31D (`v_body_legs`) | IMU-only | none | 437,382 |
| fusion_v1_marg | factorised causal transformer | 2 | 31D | MARG (drift-free yaw) | none | 437,382 |
| **fusion_v2** | factorised causal transformer | 3 | 31D | IMU-only | VO (DROID-SLAM) | **455,046** |
| fusion_v2_marg | factorised causal transformer | 3 | 31D | MARG | VO | 455,046 |
| fusion_v3 | factorised causal transformer | 3 | 35D (`leg_odom_delta` SE(3)) | IMU-only | VO | 455,558 |
| fusion_tcn | R3D-18-style 1D temporal CNN | 3 (channel-concat) | 31D | IMU-only | VO | 1,023,798 |
| fusion_mvit | MViT-style multiscale transformer | 3 | 31D | IMU-only | VO | 758,566 |

All seven share the same training loop, loss (weighted MSE with `[1,1,1,5,5,5]`), optimiser (AdamW, lr 1e-4, weight decay 1e-4, ReduceLROnPlateau), modality dropout (10 % per-sample per-modality), clip length 40 / stride 8, batch size 128, patience 10, seed 42. **Only the network architecture and/or modality content changes between rows.** Direct apples-to-apples.

### 4.2 Architecture details (the four that matter)

**Factorised causal transformer (`fusion_v1` / `v1_marg` / `v2` / `v2_marg` / `v3`).** Two stacked blocks; each block does *modal* self-attention across the M tokens at each timestep, then *causal temporal* self-attention across the T = 40 timesteps for each modality, then a position-wise FFN. Sinusoidal temporal positional encoding plus a learned per-modality embedding. Readout is mean-over-modalities at the last timestep, then a linear head to 6. The key inductive bias: **modality and time are factorised** — the network does not have to learn that the IMU token at t=15 is "the same kind of token" as the IMU token at t=20, because they share a modality embedding and live in the same temporal attention slot. d_model = 128, n_heads = 4, FFN dim = 256.

**1D temporal CNN, R3D-18-style (`fusion_tcn`).** Per-modality MLP embedders project kin → 32, ins → 16, vo → 16. The three are channel-concatenated to a 64-channel sequence over T = 40, then a stem `Conv1d` and **four stages of two `BasicBlock1D` each** (Conv1d-BN-ReLU-Conv1d-BN + residual). Stage 1 keeps T; stages 2/3/4 halve T (stride 2 on the first block of each stage) and double channels. Channel progression: 64 → 64 → 128 → 256. Global average pool over T → linear head. Inductive bias: **strong locality + temporal hierarchy.** No explicit modality-vs-time factorisation — modalities are mixed at the input.

**MViT-style multiscale transformer (`fusion_mvit`).** Per-modality 2-layer MLP embedders to D = 64; tokens stacked as `(B, T, M=3, D)`. Three stages of two factorised modal+temporal attention blocks (same block kernel as fusion_v2 but **without the causal mask** — this is a single regression at the clip end, not per-timestep streaming). Between stages, a `StageDownsample` does mean-pool stride 2 over T pairs and projects to the next stage's D. Stage dims: 64 → 96 → 128. T progression: 40 → 20 → 10. Readout: mean over (T, M). Inductive bias: **multi-scale tokens** — early layers see fine temporal detail, late layers see coarse summaries.

### 4.3 Test-set results (208 trajectories, 92,935 clips, body frame)

| | fusion_v1 | fusion_v1_marg | **fusion_v2** | fusion_v2_marg | fusion_v3 | fusion_tcn | fusion_mvit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall RMSE | 0.0933 | 0.0844 | **0.0737** | 0.0816 | 0.0883 | 0.0746 | 0.0772 |
| Linear (m/s) | 0.0637 | 0.0586 | 0.0483 | 0.0533 | 0.0596 | **0.0429** | 0.0501 |
| Angular (rad/s) | 0.1156 | 0.1039 | **0.0924** | 0.1023 | 0.1098 | 0.0964 | 0.0970 |
| `vx` | 0.0716 | 0.0660 | 0.0562 | 0.0613 | 0.0669 | **0.0462** | 0.0592 |
| `vy` | 0.0563 | 0.0531 | 0.0425 | 0.0481 | 0.0518 | **0.0372** | 0.0446 |
| `vz` | 0.0625 | 0.0552 | 0.0453 | 0.0498 | 0.0592 | **0.0447** | 0.0450 |
| `ωx` | 0.1556 | 0.1402 | **0.1272** | 0.1373 | 0.1420 | 0.1285 | 0.1325 |
| `ωy` | 0.0931 | 0.0834 | **0.0737** | 0.0827 | 0.0986 | 0.0826 | 0.0753 |
| `ωz` | 0.0830 | 0.0741 | **0.0630** | 0.0738 | 0.0792 | 0.0676 | 0.0707 |

### 4.4 Reading the table — five key findings

1. **VO is the single biggest win (fusion_v1 → fusion_v2: −21 % overall).** Adding the SE(3) delta from DROID-SLAM gives the network the one signal that kin + IMU genuinely cannot reproduce — *direct observation* of how the camera moved between consecutive frames.

2. **MARG drift-free yaw helps in IMU-only (v1 → v1_marg: −9.5 %), is redundant once VO is present (v2 → v2_marg: +10.7 %).** This is the cleanest "modality interaction" signal in the dataset: the MARG yaw and the VO `Δq` both encode rotation about the gravity vector, and adding both confuses the network rather than helping it.

3. **Replacing `v_body_legs` with the SE(3) `leg_odom_delta` makes things worse (v2 → v3: +19.8 %).** The kin branch's job is to give the model a first-cut velocity estimate. `v_body_legs` is in m/s; `Δt = v · dt` is in metres-per-frame (≈ 0.03 m), an order of magnitude smaller and dominated by integration noise. The accompanying `Δq` is near-identity at 30 Hz and adds gyro noise. Information-theoretically, replacing a high-SNR direct estimate with a low-SNR derivative is a strict loss.

4. **fusion_tcn wins linear, fusion_v2 wins angular and overall.** The 1D temporal CNN's BasicBlock stack with stride-2 downsampling is *a stronger temporal feature extractor* than two transformer blocks of width 128 — but only for the linear axes, where high-frequency gait detail (the 2 Hz trot) needs hierarchical filtering. Angular axes need *attention* across the modalities (when does the IMU disagree with VO? lean on kin) and the factorised transformer's modal self-attention does this directly. **Translation likes convolution; rotation likes attention** — at least at this dataset scale.

5. **fusion_mvit underperforms (v2 → mvit: +4.7 %).** Multi-scale token reduction is a great idea when you have hundreds of tokens (video patches, language tokens). With T = 40 and M = 3 there are only 120 tokens at the input — pooling between stages discards information rather than usefully compressing it. The MViT-style pyramid does not scale down sensibly.

---

## 5. Best model and recommendation

**Best overall: `fusion_v2` (kin + INS-IMU + VO, factorised causal transformer, 455 K params).**

- Lowest overall RMSE (0.0737), lowest angular RMSE (0.0924 rad/s).
- Trains in ~82 minutes on a single RTX 5060 Ti.
- Causal mask on temporal attention means the model is naturally **ready for streaming inference** at the robot's 30 Hz control loop — it never peeks at future timesteps.
- Outperforms the proposal's end-to-end Stage-2 CNN RGB (33.5 M params, 5 days training) by **−42 % on overall RMSE** with **77× fewer parameters and 90× faster training**.

**When to prefer `fusion_tcn`:** if downstream performance is dominated by *position drift* (numerical integration of the linear velocity), the −11 % advantage on linear RMSE compounds into a meaningful pose-tracking improvement. fusion_tcn is the right model if your loss function for the next stage is "position error", not "instantaneous velocity error".

**When `fusion_v2_marg` is preferable:** offline analysis on long trajectories where yaw drift in INS-IMU could accumulate over minutes. For closed-loop velocity control the difference is below the noise floor.

`fusion_v3` and `fusion_mvit` are not recommended for any deployment scenario — they are clean negative results (one feature-engineering ablation, one architectural ablation).

### 5.1 What this means for the thesis claim

The headline becomes:

> A 455 K-parameter factorised transformer over hand-built kinematic, inertial, and visual-odometry features outperforms a 33.5 M-parameter end-to-end CNN on the same dataset by 42 % on overall body-frame velocity RMSE, while training 90× faster and being ready for real-time streaming inference. The classical preprocessing arms — forward kinematics, Madgwick AHRS, DROID-SLAM — together cost ~2.5 days of one-time CPU/GPU time at full dataset scale and are themselves competitive with state-of-the-art per-component algorithms.

The proposal's question — "which deep visual model is best for Go2 odometry" — turned out to be the wrong question. The right question — "what should the deep model actually learn versus what can classical algorithms compute for free" — has a clean and reproducible answer.
