# Concrete Algorithms for Go2 State Estimation

Companion to `go2-preprocessing-options.md`. That document was a tool menu; this one specifies concrete algorithms with mathematical formulations, canonical references, and implementations for the actual pipeline: **joint kinematics → leg odometry, camera → visual odometry, LiDAR → LiDAR odometry, 9-DOF INS → attitude + gravity-free acceleration, then neural fusion**.

Assumes Isaac Sim data with ground-truth pose. Isaac Sim exposes per-link contact forces and full world-frame pose/velocity, which makes supervised training straightforward and lets you bypass contact-detection learning if you choose.

> **Verification status (April 2026).** Every repository listed here has been audited for existence, maintenance, install difficulty, license, quickstart quality, and fit for this specific Go2 + Isaac Sim setup. **§8 is the authoritative reference** — if a recommendation in §1–§7 conflicts with §8, §8 wins. Key changes vs the original doc: (a) `differentiable-robot-model` is archived — removed; (b) Cerberus 2.0 is AGPL-3.0 and targets Go1/A1 not Go2 — demoted to reference-only; (c) `imufusion` is preferred over `ahrs` for production; (d) **Isaac Sim's LiDAR emits standard `PointCloud2`, not Livox `CustomMsg`**, which means KISS-ICP eats it directly while FAST-LIO2 / Point-LIO need a shim — this changes the LiDAR default; (e) UniDepth is CC-BY-NC (non-commercial) — prefer Metric3D v2 for metric depth; (f) `torchvision.models.optical_flow.raft_*` is sufficient — skip upstream RAFT.

---

## 1. Joint kinematics → body velocity (leg odometry)

The highest-leverage preprocessing step. Raw 12-DoF joint angles carry no structural prior; FK + contact gives body velocity in world frame.

### 1.1 Forward kinematics

For each leg i with joint angles q_i ∈ R^3, compute foot position in body frame:

```
p_foot^B_i = FK_i(q_i)
```

and stacked Jacobian J_i(q_i) ∈ R^(6 × n_v) via any rigid-body library. Three options:

- **Pinocchio** (BSD) — the reference RBDL. `forwardKinematics(model, data, q)` + `getFrameJacobian(…, LOCAL_WORLD_ALIGNED)`. C++ with Python bindings. https://github.com/stack-of-tasks/pinocchio
- **pytorch_kinematics** (MIT) — batched, GPU, differentiable. Best fit when FK sits inside the training graph. https://github.com/UM-ARM-Lab/pytorch_kinematics
- **Go2Py** (machines-in-motion) — Pinocchio wrapper preloaded with the Go2 URDF. `Go2Model().update(q, dq, T, v)` returns foot positions, Jacobians, body kinematics in one call. https://github.com/machines-in-motion/Go2Py

### 1.2 Leg-odometry equation

For a stance foot under the no-slip assumption, world-frame foot velocity is zero:

```
v_foot^W_i = R_WB ( v_B + ω_B × p_foot^B_i + J_i(q_i) · q_dot_i ) = 0
```

Solve per-foot for body linear velocity in body frame:

```
v_B_i = -J_i(q_i) · q_dot_i - ω_B × p_foot^B_i
```

ω_B comes from the gyro. Combine across stance feet by weighting with contact probability c_i ∈ [0, 1]:

```
v_B = Σ_i (c_i · v_B_i) / Σ_i c_i
```

This is the pseudo-measurement the IMU fuser will consume. The foot-position residual under "foot is stationary while in contact" is the kinematic correction term in every EKF/InEKF below.

### 1.3 Contact detection

Three concrete methods, in order of implementation difficulty:

**(a) Simulator ground-truth contact.** Isaac Sim publishes per-link contact forces. For training, just use them. For sim-to-real, distill (b) or (c) from sim data.

**(b) Generalized-momentum observer (Bledt/Kim).** Define generalized momentum p = M(q) · v. The disturbance observer estimates external wrenches without needing joint accelerations:

```
r = K_O ∫ ( τ + C^T v + g − p_dot ) dt
```

where r ≈ J_c^T · F_ext is the per-leg GRF estimate. Threshold → binary contact. ~50 lines with Pinocchio (which gives M, C, g directly). Reference: De Luca & Mattone 2003; legged extension Bledt et al. ICRA 2018 "Contact Model Fusion for Event-Based Locomotion."

**(c) Learned contact** — `UMich-CURLY/deep-contact-estimator` (Lin et al., CoRL 2021). Multimodal CNN over short proprioceptive windows → 16-class per-leg contact probabilities. Trained on Mini Cheetah, ports to Go2. https://github.com/UMich-CURLY/deep-contact-estimator

### 1.4 Canonical leg-odometry fusion pipelines

- **Bloesch et al. RSS 2012 / IJRR 2013** — observability-consistent EKF, augments state with foothold world positions so the measurement model for a stance foot is a simple difference. This is the "right" way to avoid observability collapse. https://www.roboticsproceedings.org/rss08/p03.pdf
- **Hartley et al. IJRR 2020 — Contact-Aided InEKF.** State on SE_{K+2}(3) — rotation R, velocity v, position p, contact foot positions d_k. Right-invariant error e = X̂ · X^{-1} gives log-linear error dynamics, so linearization is exact, yielding strong convergence. https://arxiv.org/abs/1904.09251 · https://github.com/RossHartley/invariant-ekf · https://github.com/UMich-CURLY/drift
- **Cerberus 2.0 (Yang et al. RA-L 2022 / IROS 2023).** Sliding-window factor graph over IMU preintegration + leg-velocity factors + stereo VO, with **online kinematic calibration** of link lengths and IMU-to-body extrinsics (the key trick — unknown mm-level link-length errors dominate long-run drift). <1% drift. https://arxiv.org/abs/2209.07654 · https://github.com/ShuoYangRobotics/Cerberus2.0
- **VILENS (Wisth et al. T-RO 2022).** Factor graph with a **linear-velocity bias state** added to model imperfect no-slip — observable from the other modalities. Fuses vision + IMU + LiDAR + leg. 62% translational / 51% rotational error reduction over loose coupling. https://arxiv.org/abs/2107.07243

### 1.5 Gotchas

- Rubber foot compression (~mm under load) is what Cerberus calibrates online.
- Leg slip on smooth surfaces is common — reject outliers via Mahalanobis gating or use soft contact probabilities.
- Raw contact thresholds chatter at transition — add Schmitt-trigger or dwell time.

---

## 2. Camera → visual odometry

Go2 front camera is a single wide-angle monocular RGB; the EDU adds a RealSense D435i (stereo + depth). Output options for the downstream fuser: a 6-DoF pose delta, a flow field + depth map, or latent features.

### 2.1 Classical VO

- **ORB-SLAM3 (Campos et al. T-RO 2021).** Feature-based with keyframe graph and local bundle adjustment minimizing Σ ρ(‖π(T_cw · X_w) − u‖_Σ). Loop closure via DBoW2. Visual-inertial mode adds preintegrated IMU factors. Accurate but GPL and painful to build. https://github.com/UZ-SLAMLab/ORB_SLAM3
- **DSO (Engel et al. T-PAMI 2018).** Direct photometric error on high-gradient pixels, full photometric calibration, sliding-window BA. https://github.com/JakobEngel/dso
- **SVO (Forster et al. T-RO 2017).** Semi-direct: direct alignment for tracking, feature-based mapping, depth filters. https://github.com/uzh-rpg/rpg_svo_pro_open
- **pySLAM (luigifreda).** Python hybrid, modern features (SuperPoint, XFeat, ALIKED, DISK) as drop-ins. Easier to iterate. https://github.com/luigifreda/pyslam

### 2.2 Learned VO (PyTorch, fits the fusion plan)

**DPVO — Deep Patch VO (Teed, Lipson, Deng, NeurIPS 2023).** Extracts ~96 sparse patches per frame, tracks via a recurrent update operator on local patch correlations (no dense 4D correlation volume → ~3× faster, 1/3 memory vs DROID), and runs a differentiable Gauss-Newton bundle adjustment over a sliding patch graph. Outputs SE(3) pose at camera rate. Pretrained on TartanAir; zero-shot generalization. **Best modern pick for a learned monocular VO module that just works.** https://arxiv.org/abs/2208.04726 · https://github.com/princeton-vl/DPVO

**DROID-SLAM (Teed, Deng, NeurIPS 2021).** Iterative: recurrent GRU predicts dense flow + confidences → Dense Bundle Adjustment layer solves for per-pixel disparity and camera pose. One checkpoint works mono/stereo/RGB-D. Needs ~11 GB VRAM. https://arxiv.org/abs/2108.10869 · https://github.com/princeton-vl/DROID-SLAM

**TartanVO (Wang et al. CoRL 2020).** Simpler: feature matching → PWC-like flow → pose head. Good for studying / fine-tuning on sim data. https://github.com/castacks/tartanvo

### 2.3 Flow + depth as preprocessing features

If you'd rather hand the fuser rich 2D features than a single pose delta:

- **RAFT (Teed & Deng, ECCV 2020).** CNN encoder → 4D all-pairs correlation volume → ConvGRU does ~12 residual flow updates reading from the correlation volume via learned lookup. `torchvision.models.optical_flow.raft_small / raft_large`. Two lines of code, 2-channel flow. https://github.com/princeton-vl/RAFT
- **Depth Anything V2.** DINOv2 + DPT, trained on synthetic labeled + 62M pseudo-labeled real. Relative depth by default; metric variants exist but are shakier OOD. https://github.com/DepthAnything/Depth-Anything-V2
- **Metric3D v2** (Yin et al. T-PAMI 2024). Canonical-focal-length normalization recovers true metric depth zero-shot. https://github.com/YvanYin/Metric3D
- **UniDepth v2** (Piccinelli et al. CVPR 2024). Predicts a dense metric 3D pointcloud directly, disentangles intrinsics from depth. Best when intrinsics vary. https://github.com/lpiccinelli-eth/UniDepth

### 2.4 Output format recommendation

| Goal | Output | Dim/step |
|---|---|---|
| Small fuser vector input | DPVO SE(3) delta + confidence | 7 or 8 |
| 3D-CNN with visual texture | RAFT flow + Depth-Anything-V2 (downsampled) | 3 channels × H/4 × W/4 |
| Research scale, max accuracy | ORB-SLAM3 stereo-inertial on RealSense | 7 |

**Scale gotcha.** Monocular VO is scale-ambiguous. Classical ORB gives trajectory up to an unknown scalar; DPVO/DROID inherit a learned scale from TartanAir that can be 10–30% wrong in new environments. Fix either by fusing with LiDAR or leg-odometry scale (your plan already does this), or by adding a metric-depth prior.

---

## 3. LiDAR odometry (Unitree L1/L2 solid-state, non-repetitive scan)

The L1/L2 use Livox-style non-repetitive scans. Anything that assumes a 360°-per-rotation Velodyne topology (like LOAM-style edge/planar features) misbehaves. You need time-continuous deskewing.

### 3.1 KISS-ICP (Vizzo et al. RA-L 2023)

Five ingredients, no IMU required:

1. **Scan deskew** via constant-velocity prediction: for point p_k sampled at time τ within the scan, warp by T_{t-1}^{-1} · exp((τ/Δt) · log(T_{t-2}^{-1} · T_{t-1})).
2. **Adaptive voxel subsample** of scan and map.
3. **Constant-velocity motion prediction**: T_pred = T_{t-1} · (T_{t-2}^{-1} · T_{t-1}).
4. **Adaptive correspondence threshold** τ_t from the running deviation between predicted and estimated poses: σ_t^2 = Σ_s (‖ΔT_s‖)^2 over recent scans where deviation exceeds a minimum, then τ_t = 3σ_t. Tightens as motion stabilizes.
5. **Robust point-to-point ICP**: min_T Σ ρ_τ(‖T · p_i − q_i‖) with Geman-McClure kernel ρ_τ(r) = r² / (τ² + r²). NN correspondences in the voxel map.

No features, no IMU, no hand-tuned thresholds. `pip install kiss-icp`. https://arxiv.org/abs/2209.15397 · https://github.com/PRBonn/kiss-icp

### 3.2 FAST-LIO2 (Xu et al. T-RO 2022)

Tightly-coupled **iterated error-state Kalman filter**. State:

```
x = [ R ∈ SO(3), p, v, b_g, b_a, g, T_IL ]
```

Error state δx uses SO(3) retraction R = R̂ · Exp(δθ). At each scan:

1. IMU propagation between scans: δẋ = F δx + G w.
2. **Direct point-to-plane measurement** (no feature extraction): each raw point's 5-NN in the map, fit plane (n_j, d_j), residual r_j = n_j^T (T · p_j) + d_j.
3. **Iterated update**: κ ← 0…N, linearize h around x̂_κ, Kalman gain K = P H^T (H P H^T + R)^{-1}.
4. **Woodbury-identity trick** inverts a state-sized (~24×24) matrix instead of measurement-sized (thousands of points) — this is where "FAST" comes from.
5. **ikd-Tree** (Cai et al. RA-L 2021): incremental k-d tree with lazy deletes, rebalances when imbalance exceeds threshold.

https://github.com/hku-mars/FAST_LIO

### 3.3 Point-LIO (He, Xu, Zhang 2023) — recommended for Go2

Two innovations over FAST-LIO2:

- **Point-wise update.** Each LiDAR point is ingested at its own sampling time as an individual measurement, not batched per scan. Eliminates within-frame motion distortion entirely and enables 4–8 kHz odometry output.
- **Stochastic process-augmented kinematic model.** IMU specific force and angular velocity are *modeled as noisy outputs of the state* rather than as perfect inputs (a = R^T(a_W − g) + b_a + n_a, with true a_W and ω_W as random-walk states). When the cheap MEMS IMU saturates during aggressive motion (routine on a galloping quadruped), the estimator keeps running on LiDAR alone.

This is **exactly why Unitree ported Point-LIO for the L1/L2**: the L1's IMU saturates easily, and the non-repetitive scan pattern time-smears motion distortion rather than localizing it at end-of-sweep.

- Paper: https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aisy.202200459
- Original: https://github.com/hku-mars/Point-LIO
- **Unitree port (use this)**: https://github.com/unitreerobotics/point_lio_unilidar
- ROS2 port: https://github.com/dfloreaa/point_lio_ros2

### 3.4 Other options

- **DLIO (Chen et al. ICRA 2023).** Continuous-time SE(3) spline trajectory fit across the scan so every point has its own motion-corrected pose. GICP scan-to-map. Lightweight. https://github.com/vectr-ucla/direct_lidar_inertial_odometry
- **LIO-SAM (Shan et al. IROS 2020).** Factor-graph smoother with iSAM2 backend, LOAM features. Best for loop closure. **Worse on L1/L2** because LOAM features assume repetitive rings. https://github.com/TixiaoShan/LIO-SAM
- **`jizhang-cmu/autonomy_stack_go2`** — full Go2 autonomy stack wrapping Point-LIO, useful reference for rosbag→poses extraction.

### 3.5 Gotchas for L1/L2

- Don't use LOAM-style feature extraction — assumes spinning multi-ring LiDAR.
- Solid-state range is ~40m, much shorter than Velodyne → retune KISS-ICP voxel size to ~0.25–0.5 m.
- The onboard 200 Hz IMU on the LiDAR should be passed separately to Point-LIO, not externally fused.

---

## 4. 9-DOF INS fusion (accel + gyro + mag → attitude + gravity-free accel)

**Sanity check first.** The stock Go2 IMU is **6-axis**, not 9-axis. The Go2's BLDC motors produce severe, spatially-varying magnetic interference that no sim-trained MARG filter will transfer cleanly. If you're doing sim-to-real, plan to drop the mag channel. Sections below cover both modes; the recommended pipeline skips the mag.

### 4.1 Madgwick filter (2010)

Unit quaternion q ∈ S^3. Two derivatives compete:

- **Gyro rate**: q̇_gyro = ½ q ⊗ [0, ω_x, ω_y, ω_z]
- **Gradient-descent correction** from gravity (and mag if present): define f(q) where the accel should point to −z_W when stationary, compute ∇f = J(q)^T f(q):

```
q̇_corr = −β · ∇f / ‖∇f‖
q̇     = q̇_gyro + q̇_corr
q_{t+1} = normalize(q_t + q̇ · Δt)
```

- **β** trades gyro trust (low β) vs accel trust (high β). Derivation: β = √(3/4) · ω_β where ω_β is expected gyro bias magnitude. Default 0.033 (IMU) / 0.041 (MARG).
- **MARG mode** adds a mag residual with the reference-frame trick — rotate measured field into Earth frame, zero y-component (enforce h = [h_x, 0, h_z]) so magnetic dip doesn't contaminate roll/pitch.

Implementations: `ahrs.filters.Madgwick` (https://github.com/Mayitzin/ahrs), `imufusion` by xioTechnologies (https://github.com/xioTechnologies/Fusion, Madgwick's own modernized version).

### 4.2 Mahony filter (T-AC 2008)

PI correction on SO(3):

```
ω_mes = k_P · (a × â_gravity + m × m̂_ref)
b̂_dot = −k_I · ω_mes
R̂_dot = R̂ · skew(ω_gyro − b̂ + ω_mes)
```

The integral term estimates gyro bias (Madgwick doesn't do this explicitly). `ahrs.filters.Mahony`.

### 4.3 MEKF — Multiplicative EKF (Markley JGCD 2003)

Full quaternion q̂ plus **3-parameter error state** δθ. Multiplicative update q = q̂ ⊗ Exp(δθ/2) avoids 4D unit-norm covariance collapse. Propagate 3×3 orientation covariance, apply Kalman update in δθ coordinates, inject δθ into q̂, reset δθ = 0. In `filterpy` and `ahrs.filters.EKF`.

### 4.4 Invariant EKF — right tool for legged robots

For quadrupeds you want coupled attitude + velocity + position. Use the **right-invariant EKF on SE_2(3)** (Barrau & Bonnabel T-AC 2017):

```
X = [ R   v   p ]
    [ 0   1   0 ]
    [ 0   0   1 ]
```

Right-invariant error η = X̂ · X^{-1}. Key theorem: for IMU-driven kinematics with orientation and contact corrections, error dynamics are **log-linear** — ξ̇ = A · ξ + noise, A **independent of state estimate**. The linearization is exact, yielding global convergence to within a well-defined domain. No "wrong Jacobian" inconsistency after aggressive maneuvers.

Contact-aided extension (Hartley et al. IJRR 2020) adds foothold positions as additional Lie-group states. https://arxiv.org/abs/1904.09251 · https://github.com/RossHartley/invariant-ekf · https://github.com/UMich-CURLY/drift

### 4.5 Learned IMU front-ends

- **TLIO (Liu et al. RA-L 2020).** 1D ResNet-18 on 1-second gravity-aligned IMU window → 3D displacement + 3×3 covariance. Downstream stochastic-cloning EKF fuses displacements with raw IMU. https://arxiv.org/abs/2007.01867 · https://github.com/CathIAS/TLIO
- **AirIMU (Qiu et al. 2023).** Encoder-decoder outputs corrected (a, ω) plus per-sample **uncertainty** that can feed a factor graph or EKF as measurement covariance. Works across MEMS to navigation-grade. https://airimu.github.io · https://github.com/haleqiu/AirIMU
- **Brossard gyro denoiser (RA-L 2020).** Conv-net denoises gyro at input rate. Drop-in. https://github.com/mbrossar/denoise-imu-gyro

### 4.6 Recipe for fuser features

Given Madgwick output q_t:

1. R_WB = R(q_t)
2. a_W = R_WB · a_body − g_W, g_W = [0, 0, 9.81]
3. Stack `[a_W (3), ω_body (3), q_t (4)] = 10` channels per timestep

### 4.7 Magnetometer problems

- **Hard-iron**: constant offset from ferromagnetic material (battery). Fit 3D sphere min_{c, r} Σ (‖m_i − c‖ − r)², subtract c.
- **Soft-iron**: ellipsoidal distortion. Fit ellipsoid (9-DoF), compute linear transform W mapping (m − c) to sphere. Apply W · (m − c). `imucal` (https://github.com/mad-lab-fau/imucal).
- **Motor interference**: 12 BLDC motors produce time-varying fields proportional to current. No general static calibration. Practical fixes: low-pass filter mag, raise Madgwick `gain_marg`, or drop mag entirely.
- **Yaw observability** without mag: roll/pitch observable from gravity; yaw observable only via vision, LiDAR, or known-map heading. Most legged papers do exactly this (use vision/LiDAR for heading).

---

## 5. Neural fusion architectures

Five families. The preprocessed per-timestep input (with the recipe above) is roughly:

| Modality | Features | Dim |
|---|---|---|
| Leg kinematics | v_body_leg (3), foot positions (12), foot velocities (12), contact probs (4) | 31 |
| IMU | quat (4), a_world (3), ω_body (3) | 10 |
| DPVO | SE(3) delta (7) + confidence (1) | 8 |
| LiDAR (Point-LIO) | SE(3) delta (7) + confidence (1) | 8 |

Total ≈ **57** per timestep. Call this D_mod.

### 5.1 Gated / recurrent fusion (OptiState pattern)

**OptiState (Schperberg et al. ICRA 2024)** is the closest published architecture to what you're building.

- KF state x ∈ R^12 = [roll, pitch, yaw, p_x, p_y, p_z, ω_r, ω_p, ω_y, v_x, v_y, v_z]
- Depth 224×224 → ViT (patch 16, embed 128, 4 layers, 4 heads, MLP ratio 4) → latent L_depth ∈ R^128
- GRU input: KF state history over N=10 steps concatenated with L_depth → shape (B, 10, 140)
- GRU: 4 layers × 128 hidden
- Output: (B, 24) = 12 refined state + 12 per-dim uncertainty, supervised by |x̄ − x_mocap|

https://arxiv.org/abs/2401.16719 · https://github.com/AlexS28/OptiState

**For the Go2**: replace OptiState's depth-ViT latent with a small MLP over DPVO + LiDAR deltas (4 modalities → 128-D). Replace the MPC-driven KF with either Pronto-style IMU+leg propagation or Point-LIO as the strong prior. GRU eats history and corrects.

**Neural Measurement Network + InEKF (Youm et al. 2024)** — keep a classical InEKF, let a GRU-MLP output pseudo-measurements (body velocity + contact probability) that get injected as if they were real sensors. Filter stays legible; learning lives in the measurement model. Add DPVO/LiDAR as additional direct InEKF factors. https://arxiv.org/abs/2402.00366

**Buchanan et al. CoRL 2021** — 1D ResNet on 1-second IMU+kinematics window → 3D displacement + 3×3 covariance with Gaussian NLL loss `L = 0.5(Δp − μ)^T Σ^{-1}(Δp − μ) + 0.5 log|Σ|`. Calibrated covariance falls out of the supervision, which is what you need for a filter to actually use it. https://arxiv.org/abs/2111.00789

**Proprioceptive Set-Coverage (2026)** — empirical comparison of MLP/RNN/LSTM/BiGRU/TCN on proprioceptive windows. GRU won. LSTMs are bigger with no gain. Bidirectional only helps offline. https://arxiv.org/html/2603.18308

### 5.2 Transformer-based fusion

**VIFT (Kurt et al. 2024)** — causal transformer VIO, closest transformer match to your setup.

- Visual token: FlowNet-S → 512-D per frame
- Inertial token: 1D CNN → 256-D
- Concat → (B, 11, 768), T=11 (10 history + current)
- 4 layers, 6 heads, FFN 128, sinusoidal PE, **causal mask**
- Head: 6-D (3 translation + 3 rotation). Rotation uses [RPMG](https://jychen18.github.io/RPMG/) for clean backprop through manifold.
- L1 loss, rotation weighted 40× vs translation, AdamW 1e-4, 200 epochs

https://arxiv.org/abs/2409.08769 · https://github.com/ybkurt/VIFT

For the Go2: drop FlowNet (DPVO already does visual), project each of your 4 preprocessed modalities to 128-D, fuse with 4 tokens per timestep.

**BotVIO (IEEE 2024)** — edge-deployment variant. Minimalist single-layer cross-attention, 70% fewer params, 58 FPS on Jetson NX. Strong pick if real-robot deployment matters. https://github.com/wenhuiwei-ustc/BotVIO

**SS-VIO (2025), RWKV-VIO (Sensors 2025)** — Mamba (SSM) and RWKV backbones for linear-complexity temporal modeling. Worth watching if history window grows beyond ~64 steps. https://www.mdpi.com/1424-8220/25/18/5737

**TransFuser (Prakash et al. CVPR 2021)** — camera + LiDAR fusion via joint attention over concatenated token sequences at four resolution stages. Most-copied cross-modal fusion pattern of the last five years. Conceptual template for your own fusion block. https://github.com/autonomousvision/transfuser

**Perceiver IO (Jaheri et al. 2021)** — fixed-size latent array (e.g., 256 × 512). Cross-attention from input tokens into latent; self-attention inside latent; cross-attention out to arbitrary output queries. **Linear in input length** — can ingest 500 Hz IMU alongside 10 Hz LiDAR without resampling. Not applied to quadrupeds in the literature yet, which itself is a contribution angle. https://arxiv.org/abs/2107.14795

**Cross-attention patterns — what actually works:**

- **Factorized temporal-then-modal** (best default): modal self-attn over M tokens, then temporal self-attn per modality across T. Cost O(M² + T²) per block. TimeSformer factorization.
- **Joint attention over all M·T tokens**: more expressive, quadratic in M·T. Fine for M·T ≤ 200.
- **Modality-to-modality cross-attention**: pick a lead modality (leg-kin, cheapest/most-reliable) as query, cross-attend to others.
- **Perceiver latent-array**: scales to long sequences.

Minimal factorized-attention skeleton:

```python
# x: (B, T, M, D)  — M modality tokens per timestep
B, T, M, D = x.shape
x = x.reshape(B*T, M, D);       x = modal_attn(x)                # (B*T, M, D)
x = x.reshape(B, T, M, D).permute(0, 2, 1, 3).reshape(B*M, T, D)
x = temporal_attn(x, causal_mask=True)                            # (B*M, T, D)
x = x.reshape(B, M, T, D).permute(0, 2, 1, 3)
pose = head(x.mean(dim=2))                                        # (B, T, 6 or 7)
```

### 5.3 Temporal CNN

**Lee et al. Science Robotics 2020** — blind-locomotion TCN. 3 Conv1D layers, kernel 3 or 5, increasing dilation (1, 2, 4, …), channels 64/128/128, receptive field ~50 steps. Causal → real-time. Used in `rsl_rl` / `legged_gym`. https://github.com/leggedrobotics/legged_gym · https://github.com/leggedrobotics/rsl_rl

For the Go2: 57-D feature vector, conv dilations [1, 2, 4, 8, 16], kernel 3, 128 channels → ~31-step receptive field at negligible compute. Exports to TorchScript/TRT cleanly. No recurrent state to manage.

3D CNN over (time × modality) is generally a bad fit — the modality axis has no translation invariance, so conv there is just learned linear mixing. Use attention.

### 5.4 Differentiable filters

**Backprop KF (Haarnoja et al. 2016)** — full Kalman update as a differentiable graph. CNN produces (ẑ, R̂); KF propagation and update are differentiable; BPTT on pose loss. Cleanest way to combine "I trust my dynamics" with "learn observation noise." Your SRB dynamics go in PyTorch, each sensor front-end outputs its own (measurement, covariance). https://arxiv.org/abs/1605.07148 · https://github.com/stanford-iprl-lab/torchfilter (generic toolkit)

**KalmanNet (Revach et al. 2022)** — keep KF structure, replace gain with a small RNN. Useful when model is known but noise stats are hard to calibrate. https://arxiv.org/abs/2107.10043 · https://github.com/KalmanNet/KalmanNet_TSP

**α-MDF (Liu et al. CoRL 2023)** — replaces KF gain with attention over per-modality latents. 4× error reduction vs prior differentiable filters. https://github.com/ir-lab/alpha-MDF

**Theseus (FAIR)** — differentiable nonlinear least squares in PyTorch. You can express a pose graph and learn residual weights. The path if you want "learned Cerberus." https://github.com/facebookresearch/theseus

### 5.5 Quadruped-specific precedents

**Miki et al. Science Robotics 2022 (perceptive locomotion on ANYmal)** — attention-based recurrent encoder fuses proprioception with exteroceptive height map. Attention gate **learns per-cell exteroceptive trust** — when depth is unreliable (snow, reflections) attention drops and policy falls back to proprioception. This is the graceful-degradation pattern. https://leggedrobotics.github.io/rl-perceptiveloco

**rsl_rl teacher-student** — standard pattern for Go2 and ANYmal RL pipelines. A **privileged teacher** with access to ground-truth state trains first; a **history encoder student** (MLP or TCN over ~50 history steps) distills the privileged latent from observation-only features. This maps directly to your Isaac-Sim-with-GT setup: teacher sees GT pose + noiseless sensors, student sees only preprocessed features, distill. Single most field-tested pattern for quadrupeds. https://github.com/leggedrobotics/legged_gym · https://github.com/leggedrobotics/rsl_rl

---

## 6. Training considerations

### 6.1 Losses

Output: (p ∈ R^3, R ∈ SO(3), v ∈ R^3). Use all three:

- **Position**: L2 or Huber. Huber is more robust to sim contact spikes.
- **Orientation**:
  - *Chordal* (Frobenius) `L_rot = ‖R_pred − R_gt‖_F` — recommended for training (differentiable everywhere, no arccos singularities, equivalent to geodesic at small angles).
  - *Geodesic* `arccos((tr(R_pred^T R_gt) − 1)/2)` — use for evaluation.
  - Avoid Euler regression (gimbal). For quaternion regression use double-cover fix `L = min(‖q − q_gt‖, ‖q + q_gt‖)`.
  - Current best practice: 6-D continuous rotation representation (Zhou et al.) or RPMG. VIFT uses RPMG.
  - Reference: Geist et al. 2024 "Learning with 3D rotations." https://arxiv.org/abs/2404.11735
- **Velocity**: L2 in body frame.

Weights: rotation 10–40× translation (VIFT uses 40). Velocity ~1× translation.

### 6.2 Sequence length

- **≤100 ms**: underfits drift correction.
- **0.5–1 s**: sweet spot for most learned VIOs. Buchanan 1 s, VIFT ~0.5 s at 20 Hz.
- **5 s**: useful for loop-closure-like behavior; past ~2 s GT pose also drifts in sim.
- **30 s**: past the useful limit for causal encoding. Save for offline post-processing.

**Pick 1–2 s of history** (20–40 steps at 20 Hz or 40–80 at 40 Hz).

### 6.3 Normalization

- Normalize each modality independently (zero mean, unit std) over training set.
- Gravity-subtract IMU accel first, otherwise it's bimodal.
- Don't standardize quaternions (they live on S^3). Leave unit-norm or convert to 6-D rotation rep.
- DPVO confidences are non-Gaussian — use sigmoid/logit transform.

### 6.4 Variable-rate sensors (IMU 500 Hz, camera 30 Hz, LiDAR 10 Hz)

Three patterns:

1. **Resample to common rate** (50 Hz): simplest, hides rate info from network.
2. **Zero-pad + validity mask**: attention/Perceiver handle trivially; RNNs need extra is-valid channel.
3. **Per-modality (timestamp, value) embedding**: Perceiver handles natively.

Pragmatic default: **option 2 at 50 Hz**. IMU preprocess (Madgwick) + downsample; forward-fill LiDAR and DPVO between updates with a `fresh=1/0` channel that drops after the update frame.

### 6.5 Modality dropout (critical for robustness)

During training, zero each modality with per-sequence probability p_drop and set its validity mask to 0. Typical:

| Modality | p_drop |
|---|---|
| IMU | 0.0 (always available) |
| Leg-kinematics | 0.05 |
| Camera / DPVO | 0.3 |
| LiDAR / Point-LIO | 0.2 |

Plus always-on Gaussian noise (σ proportional to per-modality field std) + occasional spike corruption (σ × 10). This is what makes the Miki et al. attention-trust-gate actually work — the network needs to *see* missing sensors during training.

### 6.6 Supervision signal

Since you have Isaac Sim ground truth, you have three supervision choices stacked:

1. **Direct**: ground-truth pose/velocity → student prediction, pose loss. Simplest.
2. **Teacher-student (privileged)**: teacher takes GT pose + noiseless sensors as input and produces a latent; student takes only preprocessed features and distills the latent + predicts pose. More sample-efficient, matches rsl_rl pattern.
3. **Behavior cloning from Cerberus 2.0** or **InEKF (drift library)** run on the same sim data — use the classical estimator's trajectory as an additional target. Gives an extra loss signal grounded in legible physics.

---

## 7. Recommended end-to-end pipeline (verified April 2026)

Concretely for your setup, using only ease-of-use audited components (see §8 for the full audit):

1. **Joint angles** → **pytorch_kinematics** (🟢 `pip install pytorch_kinematics`, MIT, differentiable, accepts Go2 URDF) or Pinocchio via **Go2Py** (🟡 Docker/DDS setup, Go2-specific) → foot positions (12), body-frame foot velocities (12). Contact probabilities from **Isaac Sim GT contact forces** (4) — use these directly, skip contact-detection learning for now. Leg-estimated body velocity via §1.2 (3).
2. **IMU (6-axis; drop mag for sim-to-real)** → **imufusion** (🟢 `pip install imufusion`, MIT, C core + Python bindings, ships gravity-compensated linear-accel and earth-frame accel directly) → quaternion (4), world-frame linear accel (3), gyroscope passthrough (3). `ahrs` is fine for prototyping but imufusion is the production choice.
3. **Camera** → **DPVO** (🟡 conda + CUDA extension build — budget half a day; MIT; pretrained on TartanAir; 985 stars, last commit Oct 2024) → SE(3) delta (7) + confidence (1). Optional extra features for 3D-CNN arm: **torchvision `raft_small`** (🟢 native, no upstream RAFT repo needed) + **Metric3D v2** (🟢 pip, BSD-2, torch.hub one-liner, metric scale via canonical focal length — preferred over UniDepth which is CC-BY-NC).
4. **LiDAR** → **KISS-ICP** (🟢 `pip install kiss-icp`, MIT, consumes the standard `PointCloud2` that Isaac Sim publishes, auto-writes TUM + KITTI poses) is the primary default because **Isaac Sim does not emit Livox `CustomMsg`** — Point-LIO's Livox code path doesn't fit without a shim. Use **unitreerobotics/point_lio_unilidar** (🔴 ROS1 Noetic catkin, 476 stars) only when running on the real Go2 with a real L1/L2, which uses per-point timestamps Point-LIO can exploit.
5. **Fusion**: factorized modal-then-temporal causal transformer. 4 modality tokens per timestep (each modality → 128-D via a 2-layer MLP), T≈40 steps at 50 Hz, 2 layers, 4 heads. Pose head (6-D rotation rep + 3-D translation) and velocity head (3-D) + optional per-dim log-variance head. Study **OptiState** architecturally (🟡 MIT, research-dump quality, requires torch._six hack to install, last commit April 2024) — fork the *architecture* rather than the code.
6. **Losses**: chordal rotation (weight 20) + Huber translation (1) + Huber velocity (1) + Gaussian NLL on variance head if used.
7. **Training**: privileged teacher-student, modality dropout + noise per §6.5. Use **rsl_rl** (🟢 `pip install rsl-rl-lib`, v5.x March 2026) for the PPO trainer if you go the RL distillation route; **legged_gym** (🟡 Isaac Gym Preview 3, no stock Go2 config — use a Go2 fork like `unitreerobotics/unitree_rl_gym`) as the sim environment.
8. **Baselines to beat** on sim-held-out data: end-to-end from raw sensors (student's original attempt), KISS-ICP alone, DPVO alone, InEKF (reimplement from the algorithm; skip `RossHartley/invariant-ekf` as a dependency — dormant since 2019). Out-of-distribution test: **legkilo-dataset** (🟢 Go1 IMU+leg+LiDAR rosbags).

Total fuser: ~1–3M params, trains on a single GPU in hours, produces a calibrated-uncertainty state estimate with graceful sensor degradation.

---

## 8. Verification audit (April 2026)

All repositories in this document have been audited. Legend:

- **🟢 use**: pip-installable or trivial, actively maintained, permissive license, fits the Go2 + Isaac Sim setup.
- **🟡 use with caveats**: buildable and usable, but expect real integration work (CUDA extension compile, ROS workspace, stale deps, license restriction).
- **🔴 avoid unless necessary**: archived, abandoned, incompatible license for this project, or fundamentally mismatched to the Go2 / Isaac Sim setup.

### 8.1 Kinematics / leg odometry

| Repo | Last commit | Stars | Install | License | Python | Go2 fit | Net |
|---|---|---:|---|---|---|---|:---:|
| [pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) | Apr 2026 | 791 | 🟢 pip | MIT | native | generic URDF | 🟢 |
| [Pinocchio](https://github.com/stack-of-tasks/pinocchio) | Apr 2026 | 3275 | 🟡 conda/pip/apt | BSD-2 | bindings | generic URDF | 🟢 |
| [Go2Py](https://github.com/machines-in-motion/Go2Py) | Dec 2025 | 70 | 🟡 Docker/ROS2/DDS | MIT | native | **Go2-specific** | 🟡 |
| [PyRoki](https://github.com/chungmin99/pyroki) | Apr 2026 | 1522 | 🟢 pip -e | MIT | JAX | generic URDF | 🟢 (JAX stacks) |
| [differentiable-robot-model](https://github.com/facebookresearch/differentiable-robot-model) | Mar 2023 (archived) | 261 | 🟢 | MIT | native | generic | 🔴 **archived** |
| [deep-contact-estimator](https://github.com/UMich-CURLY/deep-contact-estimator) | Nov 2021 | 112 | 🟡 manual | **no license** | native | Mini Cheetah only | 🔴 |
| [invariant-ekf](https://github.com/RossHartley/invariant-ekf) | Aug 2019 | 540 | 🟡 CMake | BSD-3 | C++ only | landmark-based | 🟡 (algorithm, not library) |
| [DRIFT](https://github.com/UMich-CURLY/drift) | Sep 2024 | 176 | 🔴 ROS1 catkin | BSD-3 | C++ only | Mini-Cheetah/Husky/Fetch (no Go2) | 🟡 |
| [Cerberus 2.0](https://github.com/ShuoYangRobotics/Cerberus2.0) | Nov 2023 | 100 | 🔴 Docker + catkin | **AGPL-3.0** | C++ only | A1/Go1 (not Go2) | 🔴 **AGPL + wrong robot** |
| [Pronto](https://github.com/ori-drs/pronto) | Nov 2025 | 298 | 🔴 ROS2 Humble | LGPL-2.1 | C++ only | Atlas/ANYmal | 🟡 |

**Critical findings.** `differentiable-robot-model` was archived by Meta in Oct 2023 — use pytorch_kinematics or PyRoki instead. `deep-contact-estimator` has no LICENSE file, which is a legal blocker for reuse. Cerberus 2.0 is viral-copyleft AGPL-3.0 and targets Go1/A1 not Go2 — cite it for the algorithm, do not depend on it. DRIFT is the most actively maintained InEKF library but lists no Go2 config.

### 8.2 Visual odometry / depth

| Repo | Last commit | Stars | Install | License | Weights | Python | Net |
|---|---|---:|---|---|---|---|:---:|
| [DPVO](https://github.com/princeton-vl/DPVO) | Oct 2024 | 985 | 🟡 conda + CUDA ext | MIT | yes (GDrive) | CLI + importable | 🟡 |
| [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) | May 2025 | 2500 | 🟡 CUDA + lietorch + pytorch_scatter | BSD-3 | yes | CLI | 🟡 (needs 11 GB GPU) |
| [TartanVO](https://github.com/castacks/tartanvo) | Aug 2024 | 281 | 🔴 Docker-preferred (torch 1.4) | BSD-3 | yes | native | 🟡 |
| torchvision RAFT | current | — | 🟢 `pip install torchvision` | BSD-3 | yes (`Raft_*_Weights`) | native | 🟢 **use this, not the upstream repo** |
| [RAFT upstream](https://github.com/princeton-vl/RAFT) | Aug 2025 | 4000 | 🟡 stale conda pins | BSD-3 | yes | CLI | 🟡 (skip unless training) |
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | Mar 2026 | 7900 | 🟢 pip | Apache-2.0 (Small) / CC-BY-NC-4.0 (Base/Large/Giant) | yes | native | 🟢 (Small only for commercial) |
| [Metric3D](https://github.com/YvanYin/Metric3D) | Mar 2025 | 2200 | 🟢 pip | **BSD-2** | yes | torch.hub | 🟢 **best metric depth** |
| [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) | May 2025 | 1200 | 🟡 CUDA 11.8 | **CC-BY-NC-4.0** | yes | native | 🟡 (non-commercial) |
| [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) | Jul 2024 | 8500 | 🔴 C++/ROS | **GPLv3** | n/a | C++ only | 🔴 (license + C++) |
| [pySLAM](https://github.com/luigifreda/pyslam) | Mar 2026 | 3200 | 🔴 install_all.sh | **GPLv3** | partial | native | 🟡 |
| [DSO](https://github.com/JakobEngel/dso) | Feb 2024 | 2400 | 🔴 C++/CMake | GPLv3 | n/a | C++ only | 🔴 |
| [SVO Pro](https://github.com/uzh-rpg/rpg_svo_pro_open) | Jan 2024 | 1600 | 🔴 ROS1 catkin | GPLv3 | n/a | C++ only | 🔴 |

**Critical findings.** Skip the upstream RAFT repo — `torchvision.models.optical_flow.raft_small / raft_large` ships pretrained weights and avoids the stale CUDA 10.1 conda pins. For metric depth, **Metric3D v2** (BSD-2, commercial-friendly) beats UniDepth on license and Depth Anything V2 on reliability of metric output. Note Depth Anything V2's larger checkpoints (Base/Large/Giant) are CC-BY-NC-4.0; only **Depth-Anything-V2-Small** is Apache-2.0 and commercially usable. All three classical C++ SLAMs (ORB-SLAM3, DSO, SVO Pro) are GPLv3 — viral license contamination if your fusion code touches them directly.

### 8.3 LiDAR odometry

| Repo | Last commit | Stars | Install | Go2/L1-L2 | IMU required | Sim fit | Net |
|---|---|---:|---|---|---|---|:---:|
| [KISS-ICP](https://github.com/PRBonn/kiss-icp) | Jan 2026 | 2149 | 🟢 pip | generic PC2 | none | 🟢 | 🟢 **primary** |
| [KISS-SLAM](https://github.com/PRBonn/kiss-slam) | Dec 2025 | 506 | 🟢 pip | generic PC2 | none | 🟢 | 🟢 |
| [Point-LIO unilidar](https://github.com/unitreerobotics/point_lio_unilidar) | Jun 2025 | 476 | 🔴 ROS1 Noetic | **yes (L1+L2)** | required | 🟡 | 🟡 (real robot) |
| [Point-LIO original](https://github.com/hku-mars/Point-LIO) | Feb 2025 | 1176 | 🔴 ROS1 catkin | Livox generic | optional | 🟡 | 🟡 |
| [point_lio_ros2](https://github.com/dfloreaa/point_lio_ros2) | Aug 2025 | 195 | 🔴 ROS2 Humble | yes (L1+L2) | required | 🟡 | 🟡 |
| [FAST-LIO2](https://github.com/hku-mars/FAST_LIO) | Jan 2025 | 4572 | 🔴 ROS1 catkin | Livox generic | required | 🟡 | 🟡 |
| [DLIO](https://github.com/vectr-ucla/direct_lidar_inertial_odometry) | Apr 2026 | 967 | 🔴 ROS1/ROS2 catkin | Livox-branch | required (6-axis) | 🟡 | 🟡 |
| [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM) | Feb 2025 | 4695 | 🔴 ROS1 + GTSAM | no (Velodyne/Ouster) | 9-axis ≥200 Hz | 🔴 | 🔴 |
| [autonomy_stack_go2](https://github.com/jizhang-cmu/autonomy_stack_go2) | Feb 2026 | 420 | 🔴 ROS2 Humble | yes (Go2 EDU + L1) | required | 🟡 | 🟡 (real robot) |

**Critical findings — this is the biggest surprise in the audit.** **Isaac Sim's RTX LiDAR publishes standard `sensor_msgs/PointCloud2`, not Livox `CustomMsg`**. Even if the scan-pattern JSON is configured for a rosette, per-point `timestamp / line / tag / reflectivity` fields aren't in the CustomMsg layout that FAST-LIO2 / Point-LIO's Livox fast paths expect. Consequences:

- **KISS-ICP / KISS-SLAM / DLIO** eat Isaac Sim LiDAR data **directly** — no shim needed.
- **Point-LIO and FAST-LIO2** need either (a) the PointCloud2 code path (less accurate deskew because per-point timestamps are fake or missing) or (b) a shim node to synthesize CustomMsg with fabricated timestamps.
- `kiss_icp_pipeline my_sim.bag` produces `results/<seq>/<seq>_poses_tum.txt` in ~5 minutes. This is the fastest path from sim rosbag to pose CSV, and it's not close.

**Recommendation change from §3 of this doc:** For Isaac Sim preprocessing, **KISS-ICP is the primary choice, not Point-LIO**. Point-LIO becomes relevant only when moving to real-robot data with genuine per-point timestamps.

### 8.4 IMU / INS

| Repo | Last commit | Stars | Install | License | Python | Output | Net |
|---|---|---:|---|---|---|---|:---:|
| [ahrs](https://github.com/Mayitzin/ahrs) | Nov 2025 | 710 | 🟢 pip | MIT | pure Python | quaternion + 17 filters | 🟢 (prototyping) |
| [imufusion](https://github.com/xioTechnologies/Fusion) | Mar 2026 | 1579 | 🟢 pip | MIT | C + bindings | quat + gravity-compensated accel + earth-frame accel | 🟢 **production** |
| [imucal](https://github.com/mad-lab-fau/imucal) | Apr 2025 | 51 | 🟢 pip | MIT | native | scale+bias cal matrix | 🟢 (calibration step) |
| [filterpy](https://github.com/rlabbe/filterpy) | Feb 2024 | 3813 | 🟢 pip | MIT | native | generic KF | 🟡 (DIY fuser only) |
| [TLIO](https://github.com/CathIAS/TLIO) | Aug 2023 | 381 | 🔴 conda+torch | custom | native | displacement + EKF state | 🔴 (no weights; requires your own VIO-supervised dataset) |
| [AirIMU](https://github.com/haleqiu/AirIMU) | Jul 2025 | 150 | 🟡 PyPose+PyTorch | BSD-3 | native | denoised IMU + uncertainty | 🟡 (KITTI/EuRoC weights; large domain gap to Go2) |
| [Brossard gyro denoiser](https://github.com/mbrossar/denoise-imu-gyro) | Sep 2020 | 405 | 🟡 manual | MIT | native | gyro correction | 🟡 (unmaintained since 2020) |

**Critical findings.** Use **imufusion** over ahrs for production — it's the xIMU C library with maintained Python bindings (March 2026, 1.6k stars), implements Madgwick's revised chapter-7 AHRS, and exposes gravity-compensated linear-accel and earth-frame accel directly (saves you a manual R·a − g step). `ahrs` remains valuable for prototyping because you can swap between 17 filters in pure NumPy. **Skip TLIO for this project** — no pretrained weights, must generate your own supervised VIO dataset. **AirIMU** is usable only if you're willing to fine-tune on Go2 sim data; its KITTI/EuRoC weights won't transfer well to quadruped gait.

### 8.5 Fusion architectures

| Repo | Last commit | Stars | Install | License | Weights | Framework | Quality | Net |
|---|---|---:|---|---|---|---|:---:|:---:|
| [OptiState](https://github.com/AlexS28/OptiState) | Apr 2024 | 35 | 🟡 conda+torch._six hack | MIT | data only | PyTorch | 🟡 research dump | 🟢 **closest architecture match** |
| [VIFT](https://github.com/ybkurt/VIFT) | Aug 2025 | 37 | 🟢 requirements.txt | **no license** | via VSVIO | PyTorch | 🟢 clean | 🟡 (license + tight FlowNet coupling) |
| [BotVIO](https://github.com/wenhuiwei-ustc/BotVIO) | Feb 2025 | 69 | 🟡 manual pins | MIT | partial (GDrive) | PyTorch | 🟡 | 🟡 |
| [TransFuser](https://github.com/autonomousvision/transfuser) | Oct 2025 | 1537 | 🔴 CARLA 0.9.10.1 + CUDA 11.3 | MIT | yes | PyTorch | 🟢 | 🔴 (CARLA-locked) |
| [torchfilter](https://github.com/stanford-iprl-lab/torchfilter) | May 2023 | 166 | 🟢 pip -e | MIT | n/a | PyTorch | 🟢 library | 🟢 (Backprop-KF toolkit) |
| [KalmanNet](https://github.com/KalmanNet/KalmanNet_TSP) | Feb 2024 | 427 | 🟢 requirements.txt | **no license** | no | PyTorch | 🟡 | 🟡 (license blocks reuse) |
| [alpha-MDF](https://github.com/ir-lab/alpha-MDF) | Nov 2023 | 10 | 🟡 CUDA 11.1 pinned | MIT | no | PyTorch | 🟡 | 🟡 |
| [Theseus](https://github.com/facebookresearch/theseus) | Jan 2025 | 2015 | 🟢 `pip install theseus-ai` | MIT | n/a | PyTorch | 🟢 | 🟢 (diff NLS) |
| [BackpropKF_Reproduction](https://github.com/tiboat/BackpropKF_Reproduction) | Jun 2022 | 14 | 🟢 requirements.txt | Apache-2.0 | no | PyTorch | 🟡 student | 🟡 (cite, don't fork) |
| [legged_gym](https://github.com/leggedrobotics/legged_gym) | May 2025 | 2895 | 🟡 Isaac Gym Preview 3 | BSD-like | no | PyTorch/IG | 🟢 | 🟢 (but **no stock Go2 config** — use a Go2 fork) |
| [rsl_rl](https://github.com/leggedrobotics/rsl_rl) | Mar 2026 | 2500 | 🟢 `pip install rsl-rl-lib` | BSD-like | n/a | PyTorch | 🟢 | 🟢 |
| [legkilo-dataset](https://github.com/ouguangjun/legkilo-dataset) | Sep 2024 | 71 | 🟡 ROS1 Ubuntu 18.04 | **no license** | data only | C++/ROS | 🟢 | 🟢 (OOD eval) |

**Critical findings.**

- **OptiState** is the closest published architecture to what you're building, and the MIT license + working quickstart make it viable as an architectural fork — but the README instructs you to hand-patch `torch/_six.py` after install, a dependency-rot red flag. Plan a day to modernize the env and drop the patch. No pretrained weights; train from scratch on your Go2 sim data.
- **VIFT** has no LICENSE file — legal blocker for clean reuse. Also, you can't cleanly drop the FlowNet branch; visual latents are first-class inputs to the transformer fusion block, so "IMU+kinematics only" requires real rewiring.
- **Theseus** is the cleanest way into differentiable factor graphs if you want a learned-Cerberus pattern. `pip install theseus-ai`, MIT, actively maintained.
- **TransFuser** is CARLA-locked — architectural reference only, not a fork base.
- **KalmanNet** and **legkilo-dataset** both ship without LICENSE files — usable as references but legally risky to redistribute.
- **legged_gym vs rsl_rl clarification**: `rsl_rl` is the standalone PyTorch PPO trainer (pip-installable, used by Isaac Lab / Legged Gym / MuJoCo Playground). `legged_gym` is the Isaac Gym environment package — **Go2 is not in its stock robot list** (ANYmal/Cassie/A1 are). For Go2 RL, fork `unitreerobotics/unitree_rl_gym` or use Isaac Lab with Unitree configs. The upstream `legged_gym` README now points to Isaac Lab as the successor.

### 8.6 Minimum viable pipeline (pure 🟢 path)

If you want the **fastest path to a working preprocessing pipeline with zero CUDA-extension builds, zero ROS workspaces, zero license problems**, the list collapses to:

```
pip install pytorch_kinematics imufusion kiss-icp torchvision theseus-ai rsl-rl-lib
# + model weights
python -c "import torch; torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)"
```

- **Kinematics**: pytorch_kinematics (Go2 URDF) + Isaac Sim GT contact forces
- **IMU**: imufusion (Madgwick variant, gravity-compensated accel output)
- **LiDAR**: KISS-ICP (eats Isaac Sim PointCloud2 directly, auto-writes TUM)
- **Camera (optional flow + depth)**: torchvision `raft_small` + Metric3D v2
- **Fusion**: your own PyTorch transformer (factorized attention per §5.2), with Theseus available if you want differentiable NLS on top
- **RL / teacher-student trainer**: rsl_rl

This is five pip installs + one torch.hub checkpoint. No CUDA extension compile, no catkin_make, no AGPL/GPL contamination, no CARLA. Adds: **DPVO** (🟡 CUDA ext compile, half-day) when you want pose-based visual odometry instead of / in addition to flow+depth features.

What this omits and why: OptiState (fork it architecturally, not the code), Point-LIO (only useful when you move to real robot data with genuine per-point timestamps), Cerberus 2.0 (AGPL + wrong robot), TLIO (no pretrained weights), classical C++ SLAMs (GPLv3 + C++ + no Python).

---

## 9. Documented Go2 usage of these systems (April 2026 survey)

A literature and repo survey of which of the algorithms in §1–§5 have actually been deployed on a **Unitree Go2** — either real hardware or Go2 simulation (Isaac Sim, Isaac Lab, MuJoCo, Gazebo). Work on older Unitree platforms (Go1, A1, Aliengo) was excluded unless the authors explicitly ported or re-ran on Go2. **This is a surprisingly empty landscape** and is itself a contribution angle for a thesis.

### 9.1 What has been run on Go2

| System | Category | Go2 hardware | Go2 sim | Camera/sensor | Source | Reproducible |
|---|---|:---:|:---:|---|---|:---:|
| **RTAB-Map** | visual-LiDAR SLAM | ✓ | Gazebo | RealSense D435i | Naderi (2024–25); arXiv 2505.02272 | yes |
| **Point-LIO** (Unitree port) | LiDAR-inertial | ✓ | — | Unitree L1 + its IMU | unitreerobotics/point_lio_unilidar | yes |
| **Point-LIO** via autonomy stack | LiDAR-inertial | ✓ | — | Unitree L1 | jizhang-cmu/autonomy_stack_go2 | yes |
| **Invariant-EKF** (leg + IMU) | proprioceptive | ✓ | — | IMU + joints | inria-paris-robotics-lab/go2_odometry | yes |
| **X-IONet** | learned IMU-only | ✓ | — | IMU only | arXiv 2511.08277 (Nov 2025) | dataset status unclear |

Published camera-based Go2 state estimation is **one method** (RTAB-Map) on **one camera** (RealSense D435i on the EDU). The default Go2's wide-angle mono front camera — what the non-EDU ships with — has no published VO/VIO paper.

### 9.2 What has **not** been run on Go2

As of April 2026, I found zero documented deployments on Go2 (real or sim) for:

- **Monocular / stereo VO & SLAM**: ORB-SLAM3, DPVO, DROID-SLAM, TartanVO, DSO, LDSO, SVO, SVO Pro, pySLAM
- **Visual-inertial**: VINS-Mono, VINS-Fusion, OpenVINS, Kimera
- **VO preprocessing components**: RAFT (as a preproc stage), Depth Anything V2, Metric3D, UniDepth
- **Visual-inertial-leg**: Cerberus 2.0, VILENS, LVI-Q — all target A1/Go1/ANYmal, not Go2
- **Learned visual-inertial on Go2**: none found

Common misattributions to watch for:
- The NVIDIA developer-forum "Isaac Sim VIO on Unitree quadruped" thread is **A1**, not Go2.
- Cerberus / Cerberus 2.0 consistently say "Go1/A1" — not Go2.
- Kimera2's "quadruped" validation is on Boston Dynamics Spot and ANYmal.
- LVI-Q (arXiv 2510.15220, Oct 2025) — A1/Go1 K-Campus dataset, not Go2.

### 9.3 Isaac Sim Go2 camera infrastructure

Camera plumbing exists in the Go2 simulation ecosystem but is unused for state estimation:

- `Zhefan-Xu/isaac-go2-ros2` — exposes front-camera RGB + depth + semantic segmentation on ROS2 topics. No VO node.
- `sallu-786/Go2_Isaac_ros2` — similar.
- `unitreerobotics/unitree_sim_isaaclab`, `unitreerobotics/unitree_mujoco`, `abizovnuralem/go2_omniverse` — locomotion-focused, no VO runs published.

So an Isaac Sim Go2 image stream into DPVO or ORB-SLAM3 is trivially plumbable but has no published example.

### 9.4 Why this matters for the thesis

The student's pipeline is very likely the **first published application** of:

- Any learned monocular VO (DPVO / DROID-SLAM / TartanVO) on a Go2, real or simulated
- Any modern monocular-depth preprocessing (Metric3D v2 / Depth Anything V2 / UniDepth) on a Go2
- Any transformer-based multimodal state estimator on a Go2

This isn't just "we extended OptiState to a new platform." It's a genuine empty cell in the quadruped VO literature. For related-work framing, the honest claim is: Go2's published state-estimation literature is overwhelmingly LiDAR-inertial (Point-LIO) or proprioceptive (InEKF on Inria's stack). The camera — particularly the Go2's default wide-angle mono — has been treated as an auxiliary video stream, not as a state-estimation input. This work changes that.

### 9.5 Sources

- [Naderi Go2 SLAM writeup](https://h-naderi.github.io/projects/1-slam) · [repo](https://github.com/h-naderi/unitree-go2-slam-nav2-demo)
- [Robust Localization for Quadrupeds (arXiv 2505.02272)](https://arxiv.org/abs/2505.02272)
- [go2_odometry (Inria) — proprioceptive InEKF](https://github.com/inria-paris-robotics-lab/go2_odometry)
- [autonomy_stack_go2 (CMU)](https://github.com/jizhang-cmu/autonomy_stack_go2)
- [point_lio_unilidar (Unitree)](https://github.com/unitreerobotics/point_lio_unilidar)
- [X-IONet (arXiv 2511.08277) — learned IMU-only on Go2](https://arxiv.org/abs/2511.08277)
- [LVI-Q (arXiv 2510.15220) — Go1/A1 only](https://arxiv.org/abs/2510.15220)
- [Multi-IMU Fusion for Legged Robots (arXiv 2507.11447) — custom A1-class, not Go2](https://arxiv.org/abs/2507.11447)
- [Zhefan-Xu/isaac-go2-ros2 (Isaac Sim Go2 w/ cameras)](https://github.com/Zhefan-Xu/isaac-go2-ros2)
- [sallu-786/Go2_Isaac_ros2](https://github.com/sallu-786/Go2_Isaac_ros2)
- [unitreerobotics/unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab)
- [abizovnuralem/go2_omniverse](https://github.com/abizovnuralem/go2_omniverse)

---

## Appendix: annotated repository list

🟢 = safe dependency, 🟡 = buildable with effort, 🔴 = avoid.

**Kinematics / leg odometry**
- 🟢 Pinocchio — https://github.com/stack-of-tasks/pinocchio
- 🟢 pytorch_kinematics — https://github.com/UM-ARM-Lab/pytorch_kinematics
- 🟢 PyRoki (JAX) — https://github.com/chungmin99/pyroki
- 🟡 Go2Py — https://github.com/machines-in-motion/Go2Py
- 🟡 invariant-ekf (algorithm reference) — https://github.com/RossHartley/invariant-ekf
- 🟡 DRIFT — https://github.com/UMich-CURLY/drift
- 🟡 Pronto — https://github.com/ori-drs/pronto
- 🔴 differentiable-robot-model (archived) — https://github.com/facebookresearch/differentiable-robot-model
- 🔴 deep-contact-estimator (no license) — https://github.com/UMich-CURLY/deep-contact-estimator
- 🔴 Cerberus 2.0 (AGPL + Go1/A1 only) — https://github.com/ShuoYangRobotics/Cerberus2.0

**Visual odometry / depth**
- 🟢 torchvision RAFT — `torchvision.models.optical_flow.raft_small / raft_large`
- 🟢 Depth Anything V2 Small (Apache-2.0) — https://github.com/DepthAnything/Depth-Anything-V2
- 🟢 Metric3D v2 (BSD-2, metric scale) — https://github.com/YvanYin/Metric3D
- 🟡 DPVO — https://github.com/princeton-vl/DPVO
- 🟡 DROID-SLAM — https://github.com/princeton-vl/DROID-SLAM
- 🟡 TartanVO — https://github.com/castacks/tartanvo
- 🟡 UniDepth (CC-BY-NC) — https://github.com/lpiccinelli-eth/UniDepth
- 🟡 pySLAM (GPLv3) — https://github.com/luigifreda/pyslam
- 🔴 ORB-SLAM3 (GPLv3, C++) — https://github.com/UZ-SLAMLab/ORB_SLAM3
- 🔴 DSO (GPLv3, C++) — https://github.com/JakobEngel/dso
- 🔴 SVO Pro (GPLv3, ROS1) — https://github.com/uzh-rpg/rpg_svo_pro_open

**LiDAR odometry**
- 🟢 KISS-ICP (primary for Isaac Sim) — https://github.com/PRBonn/kiss-icp
- 🟢 KISS-SLAM — https://github.com/PRBonn/kiss-slam
- 🟡 Point-LIO unilidar (real robot) — https://github.com/unitreerobotics/point_lio_unilidar
- 🟡 Point-LIO original — https://github.com/hku-mars/Point-LIO
- 🟡 point_lio_ros2 — https://github.com/dfloreaa/point_lio_ros2
- 🟡 FAST-LIO2 — https://github.com/hku-mars/FAST_LIO
- 🟡 DLIO — https://github.com/vectr-ucla/direct_lidar_inertial_odometry
- 🟡 autonomy_stack_go2 (real robot) — https://github.com/jizhang-cmu/autonomy_stack_go2
- 🔴 LIO-SAM (wrong scan topology) — https://github.com/TixiaoShan/LIO-SAM

**IMU / INS**
- 🟢 imufusion (production) — https://github.com/xioTechnologies/Fusion
- 🟢 ahrs (prototyping) — https://github.com/Mayitzin/ahrs
- 🟢 imucal — https://github.com/mad-lab-fau/imucal
- 🟡 filterpy — https://github.com/rlabbe/filterpy
- 🟡 AirIMU — https://github.com/haleqiu/AirIMU
- 🟡 Brossard gyro denoiser — https://github.com/mbrossar/denoise-imu-gyro
- 🔴 TLIO (no weights, needs VIO dataset) — https://github.com/CathIAS/TLIO

**Fusion architectures**
- 🟢 OptiState (architectural fork) — https://github.com/AlexS28/OptiState
- 🟢 Theseus (differentiable NLS) — https://github.com/facebookresearch/theseus
- 🟢 torchfilter (Backprop-KF) — https://github.com/stanford-iprl-lab/torchfilter
- 🟡 VIFT (no license) — https://github.com/ybkurt/VIFT
- 🟡 BotVIO — https://github.com/wenhuiwei-ustc/BotVIO
- 🟡 KalmanNet (no license) — https://github.com/KalmanNet/KalmanNet_TSP
- 🟡 α-MDF — https://github.com/ir-lab/alpha-MDF
- 🟡 BackpropKF_Reproduction (cite, don't fork) — https://github.com/tiboat/BackpropKF_Reproduction
- 🔴 TransFuser (CARLA-locked) — https://github.com/autonomousvision/transfuser

**RL / training infrastructure**
- 🟢 rsl_rl — https://github.com/leggedrobotics/rsl_rl
- 🟡 legged_gym (no stock Go2 config) — https://github.com/leggedrobotics/legged_gym
- (likely needed) unitreerobotics/unitree_rl_gym or Isaac Lab Unitree configs

**Datasets for OOD evaluation**
- 🟢 legkilo-dataset (Go1 IMU+leg+LiDAR) — https://github.com/ouguangjun/legkilo-dataset
