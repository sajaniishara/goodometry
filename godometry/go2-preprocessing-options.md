# Preprocessing Options for Go2 End-to-End State Estimation

This is a sensor-by-sensor breakdown of freely available, well-maintained preprocessing tools suitable for a hybrid pipeline where preprocessed features are fused by a transformer or 3D CNN downstream. Recommendations favor Python/PyTorch where possible since the fusion model will be differentiable.

## A note on the Go2 sensor suite before picking tools

Worth sanity-checking with the student: the real Go2 IMU is typically advertised as 6-axis (accel + gyro), not 9-axis. Some URDFs and simulation configs add a magnetometer, and Isaac/Mujoco environments will happily simulate a MARG sensor if asked, but the hardware Go2 does not publish a magnetometer stream in `sensor_msgs/Imu` out of the box. If they're targeting sim-to-real, dropping the mag channel now saves pain later. The stock front camera is a wide-angle mono RGB (not stereo); the EDU adds a RealSense D435i. The LiDAR is Unitree's own L1 (Go2 standard) or L2 (newer Go2), both solid-state with 360°×90–96° FoV and a built-in IMU.

Which simulator they're using also matters — Isaac Lab and Mujoco (`unitree_mujoco`, `unitree_rl_mjlab`, `unitree_sim_isaaclab`) all expose the standard sensor set and a correct URDF/MJCF for the Go2.

---

## 1. Joint angles → proprioceptive state (forward kinematics, leg odometry)

This is the highest-value preprocessing step. Raw 12-DoF joint angles carry no structural prior; after FK you get foot positions in the body frame, and combined with contact detection you get body velocity in the world frame — exactly the signal a learned estimator struggles to discover from scratch.

### Differentiable FK in PyTorch (best fit for transformer fusion)

**pytorch_kinematics** (UM-ARM-Lab, MIT license) — `pip install pytorch_kinematics`. Parses URDF/SDF/MJCF, computes FK and Jacobians in parallel on GPU, fully differentiable. The Go2 URDF is freely available in `unitree_ros` and the `unitree_mujoco` repo. This is the cleanest option for a student who wants the FK step to sit inside the PyTorch computation graph. About five lines of setup — build chain from URDF, call `chain.forward_kinematics(joint_angles_batch)`.

**Differentiable Robot Model** (Meta/FAIR, MIT) — `facebookresearch/differentiable-robot-model`. Similar idea, also a `torch.nn.Module`, also supports dynamics (RNEA/ABA/CRBA) if they want ground-reaction-force-aware features. Less actively maintained than pytorch_kinematics but has proper rigid-body dynamics.

**PyRoki** (JAX, `chungmin99/pyroki`) — if they're on JAX. JIT-compiled, vmap-friendly.

### Production-grade FK (non-differentiable, more feature-complete)

**Pinocchio** (stack-of-tasks, BSD) — the de facto standard in legged robotics research. Install via conda (`conda install -c conda-forge pinocchio`). Orders of magnitude more mature than the differentiable options and used by essentially every serious legged-robot paper, including Cerberus and Pronto. Use it if they don't need gradients through FK (they probably don't — FK is usually a fixed preprocessing step and the network fuses its outputs).

**Go2Py** (`machines-in-motion/Go2Py`) — Python interface for the Go2 that wraps Pinocchio specifically for this robot. Has `Go2Model().update(q, dq, T, vel)` which does exactly what the student needs: given joint state, gives foot positions, Jacobians, and body kinematics. This is probably the fastest path to "I have Go2-specific FK running." Works in both sim (Mujoco) and real.

### Leg odometry (FK + contact detection → body velocity)

FK alone gives foot positions relative to the body. To get body velocity in the world frame (the thing that actually reduces IMU drift), you need to know which feet are in stance and assume non-slip. Options in order of complexity:

- **Cerberus / Cerberus 2.0** (`ShuoYangRobotics/Cerberus`, `Cerberus2.0`) — C++/ROS, but the leg-odometry module is cleanly separable. Originally targets A1/Go1 but the math ports to Go2 trivially (same 3-DoF-per-leg topology). Best-in-class drift numbers (<1%) and includes online kinematic parameter calibration. Ships with public Go1 datasets for validation.
- **Pronto** (`ori-drs/pronto`) — ROS package, EKF fusing IMU + leg odometry with pluggable forward-kinematics API. Less Go2-specific but very well-documented.
- **deep-contact-estimator** (`UMich-CURLY/deep-contact-estimator`) — learned contact detection from proprioception only, trained on Mini Cheetah but generalizes. Useful if they want to avoid thresholding ground-reaction-force estimates. Outputs per-foot contact probabilities.
- **Simple momentum-based contact** — if they want to roll their own, torque-based GRF estimation via the generalized momentum method is ~50 lines with Pinocchio. Good enough in simulation where the contact signal is clean.

**Recommendation:** `pytorch_kinematics` for batched FK inside the training loop, plus contact detection either from simulator ground truth (free in sim!) or a simple threshold on estimated foot force. This gives per-foot positions, per-foot velocities, and body velocity as input features — roughly 30 numbers per timestep that summarize what the 12 joint angles are "really" telling you.

---

## 2. IMU (accel + gyro + optional magnetometer) → orientation and gravity-aligned features

Raw accelerometer readings are dominated by gravity in the body frame, which makes them useless to a network unless it learns to estimate orientation itself. Running even a trivial orientation filter first decouples gravity from linear acceleration and gives the downstream model a much cleaner signal.

### The go-to library

**`ahrs`** (`Mayitzin/ahrs`, MIT) — `pip install ahrs`. Pure Python, implements Madgwick, Mahony, EKF, complementary filter, and about 15 others. Works with IMU (6-axis) or MARG (9-axis) seamlessly — relevant since the student mentioned a magnetometer. Output is a quaternion per timestep. Dead simple:

```python
from ahrs.filters import Madgwick
madgwick = Madgwick(gyr=gyro, acc=accel, mag=mag)  # or omit mag for 6-axis
orientations = madgwick.Q  # (N, 4) quaternions
```

Once you have orientation, you rotate the accelerometer reading into the world frame, subtract gravity, and feed the network (a) world-frame linear acceleration, (b) gyroscope (already body-frame, fine as-is), (c) the quaternion itself or roll/pitch (yaw is unobservable without magnetometer — drop it if there's no mag).

### Alternatives

- **`imufusion`** (`xioTechnologies/Fusion`, MIT) — pip-installable Python bindings for a modern Madgwick variant by Madgwick himself. Slightly nicer tuning defaults than the classic implementation.
- **`filterpy`** if they want to roll their own EKF — widely used, well-documented.

### For the magnetometer specifically

If they're in simulation and the mag is noiseless, it provides yaw directly. But if they're targeting sim-to-real: hard magnetic disturbances near the Go2's motors will be severe and simulated mag is basically a cheat code that won't transfer. Either (a) add realistic hard-iron + soft-iron distortion to the sim, or (b) drop the channel. `ahrs` has `gain_marg` tuning for low-confidence mag.

**Recommendation:** `ahrs.Madgwick` for a quaternion, then transform accelerometer to world frame. Three extra input channels (world-frame linear accel) replace three gravity-contaminated channels and make the downstream network's job dramatically easier.

---

## 3. Camera → visual odometry features

The Go2 has a single front-facing RGB (wide-angle) camera on the standard/Pro, plus an optional RealSense D435i on the EDU. Options split by whether they want a pose estimate, a feature map, or both.

### Classical, accurate, but C++ heavy

**ORB-SLAM3** (`UZ-SLAMLab/ORB_SLAM3`, GPL) — the reference monocular/stereo/RGB-D + VI SLAM system. Very accurate but painful to build; most people use the Docker images. Outputs 6-DoF pose at camera rate. If they just need pose estimates as inputs to the network, ORB-SLAM3 via its ROS wrapper is a good black-box choice.

**pySLAM** (`luigifreda/pyslam`) — Python-first hybrid VO/SLAM pipeline, supports monocular/stereo/RGB-D, lots of modern feature extractors built-in (SuperPoint, XFeat, ALIKED, DISK). Much easier to integrate into Python training code than ORB-SLAM3. Good choice for research iteration speed.

### Deep learning VO (PyTorch-native, fits the fusion plan)

**DPVO / DPV-SLAM** (`princeton-vl/DPVO`, MIT) — "Deep Patch Visual Odometry," current SOTA for monocular deep VO, runs at 60–120 FPS with a small memory footprint. PyTorch. Pretrained on TartanAir (synthetic, diverse environments) and generalizes well zero-shot. Outputs a 6-DoF trajectory and sparse 3D points. Best modern pick if they want a learned VO module that just works.

**DROID-SLAM** (`princeton-vl/DROID-SLAM`, BSD) — Heavier than DPVO but more accurate in some scenes and supports mono/stereo/RGB-D from a single model. Needs an 11 GB GPU for inference. PyTorch.

**TartanVO** (`castacks/tartanvo`, BSD) — Older and simpler than DPVO, one of the first zero-shot-generalizing learned VOs. Smaller model, easier to study / fine-tune.

### Raw feature-map preprocessing (if they want the network to learn its own VO)

If the student decides the 3D CNN / transformer should consume visual features directly rather than a final pose, two cheap, reliable preprocessing steps are worth running first:

- **RAFT optical flow** — built into `torchvision.models.optical_flow` (`raft_large`, `raft_small`). Two lines of code. Gives a dense 2-channel flow field between consecutive frames, which is far more useful to a state-estimation network than raw RGB — flow encodes ego-motion directly. `raft_small` is fast enough to run inline with training.
- **Monocular depth** — Depth Anything V2 (`DepthAnything/Depth-Anything-V2`) is the current best-in-class zero-shot depth model, also PyTorch. A per-pixel depth prediction combined with flow gives the network most of what a classical VO pipeline extracts.

**Recommendation depending on student's goal:**

- *"I want a pose estimate to feed the transformer as a small vector"* → **DPVO** (most modern, PyTorch, generalizes, fast).
- *"I want rich visual features for a 3D CNN"* → **RAFT optical flow** + optionally **Depth Anything V2**, both run as frozen preprocessing.
- *"I want maximum accuracy and don't mind C++"* → **ORB-SLAM3 stereo-inertial** on the EDU's RealSense.

---

## 4. LiDAR (Unitree L1/L2) → pose and geometric features

### Turnkey for Go2 specifically

**`unitreerobotics/point_lio_unilidar`** — Unitree's own port of Point-LIO tuned for the L1 and L2. LiDAR-inertial, handles the L1/L2's non-repetitive scan pattern correctly, uses the LiDAR's built-in IMU. If they need a LiDAR pose stream on the Go2, this is the path of least resistance. ROS1 version; there's a ROS2 port at `dfloreaa/point_lio_ros2` also supporting the L1/L2.

**`jizhang-cmu/autonomy_stack_go2`** — full autonomy stack for Go2 that wraps Point-LIO; essentially a ready-made way to extract SLAM poses from a Go2 rosbag.

### General-purpose alternatives

**FAST-LIO2** (`hku-mars/FAST_LIO`, GPL) — the standard baseline for LiDAR-inertial odometry. Point-LIO is basically its high-bandwidth successor. Works on Livox and spinning LiDARs. If they want to stick with the most-cited option, use this.

**KISS-ICP** (`PRBonn/kiss-icp`, MIT) — `pip install kiss-icp`. LiDAR-only (no IMU required). The "it just works" odometry pipeline — point-to-point ICP done carefully. Excellent choice for simplicity; no parameter tuning, outputs clean poses. If they don't want to fuse the IMU at this stage (because the network will do that), KISS-ICP is arguably cleaner than Point-LIO.

**KISS-SLAM** (`PRBonn/kiss-slam`, MIT) — same authors, adds loop closure. Usually overkill for state-estimation preprocessing.

### For learned point-cloud features (if they want to feed raw geometry to the 3D CNN)

**PointNet / PointNet++** (`yanx27/Pointnet_Pointnet2_pytorch`) — classic, pre-trained backbones freely available. Can be used frozen as a feature extractor: point cloud in, fixed-size feature vector out.

**torch_geometric point cloud utilities** — voxelization, farthest-point-sampling, KNN graphs; all the preprocessing primitives you'd want before a PointNet++ or point transformer, in PyTorch.

**Recommendation:** Run **Point-LIO** (Unitree's port) or **KISS-ICP** offline on each sim episode to get a 6-DoF trajectory as a preprocessed input. Don't try to feed raw point clouds into the fusion network — they're heavy, and the pose output already contains almost all the state-estimation signal the LiDAR has to offer. Keep raw point clouds only if terrain semantics or obstacle geometry matter to the downstream task.

---

## 5. End-to-end integrated options (skip the assembly)

Three projects already implement exactly the fusion the student is building, and might be worth either using directly or studying:

- **Cerberus / Cerberus 2.0** — VILO (Visual-Inertial-Leg Odometry) with IMU + leg + stereo camera, factor-graph based. Not learned, but a beautiful reference for what "all-signals-fused classical estimator" looks like for this exact sensor set. Drift <1% over hundreds of meters.
- **Pronto** — EKF-based, fuses IMU + leg + optional visual/LiDAR corrections. Open-source ROS packages.
- **legkilo-dataset** (`ouguangjun/legkilo-dataset`) — a Go1 dataset specifically designed with leg kinematics + IMU + LiDAR for testing fusion algorithms. Could be useful as an out-of-distribution test set for whatever they train.

---

## Suggested pipeline (my actual recommendation, in one place)

For a student who wants to ship this in a few weeks:

1. **Joint angles** → `pytorch_kinematics` with the Go2 URDF → foot positions (12 numbers) and body-frame foot velocities (12 numbers). If contact labels are available from the simulator, concatenate those (4 numbers).
2. **IMU** → `ahrs.Madgwick` → quaternion (4), world-frame linear acceleration (3), gyroscope pass-through (3). Drop magnetometer for sim-to-real; keep it if staying pure-sim.
3. **Camera** → either (a) **DPVO** poses as a 7-D vector per frame, or (b) **RAFT-small** 2-channel flow + Depth-Anything-V2 1-channel depth as a 3-channel low-res feature map for the 3D CNN.
4. **LiDAR** → **Point-LIO** (Unitree's port) poses as a 7-D vector. Skip raw point cloud fusion unless there's a specific reason for it.

All of this is Python except Point-LIO (which is fine as an offline preprocessing pass that writes poses to disk). The transformer/3D-CNN then consumes a modest feature vector per timestep plus, optionally, a small feature map from the camera path. Training will be dramatically faster than end-to-end from raw sensors, which matches the student's intuition.

If they want the fusion itself to be learnable and somewhat principled, **OptiState** (mentioned in the awesome-quadrupeds list — "State Estimation of Legged Robots using Gated Networks with Transformer-based Vision and Kalman Filtering") is a recent paper that does almost exactly this architecture and would be a useful reference.
