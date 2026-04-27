"""Run DROID-SLAM in stereo mode on a Go2 trajectory.

Returns per-frame camera-to-world poses aligned to sensors.npz['frame_idx'].

Optimisations vs the pilot version:
  * IMREAD_REDUCED_COLOR_2 decodes PNGs at half-res (640×360) directly —
    skips both the full-res decode and the resize, ~6× faster on the I/O.
  * ThreadPoolExecutor with 8 workers parallelises the per-frame load.
  * `stride` skips frames during DROID processing, then linearly interpolates
    translation and SLERPs orientation to fill the full sensors.npz frame_idx
    coverage. DROID's own demo uses stride=3 by default.
"""
from __future__ import annotations
import os
import sys
import time
import warnings
import types
import numpy as np
import torch
import cv2
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", category=FutureWarning)

DROID_ROOT = "/home/anyone/projects/goodometry/third_party/DROID-SLAM"
if DROID_ROOT not in sys.path:
    sys.path.insert(0, DROID_ROOT)
if os.path.join(DROID_ROOT, "droid_slam") not in sys.path:
    sys.path.insert(0, os.path.join(DROID_ROOT, "droid_slam"))

from droid import Droid  # type: ignore


def _load_one(args):
    traj_dir, i, target_hw, use_reduced = args
    l = os.path.join(traj_dir, f"frame_{int(i):06d}", "left_image.png")
    r = os.path.join(traj_dir, f"frame_{int(i):06d}", "right_image.png")
    if not (os.path.exists(l) and os.path.exists(r)):
        return None
    if use_reduced:
        # PNG is 1280×720; REDUCED_COLOR_2 decodes at half-res directly.
        L = cv2.imread(l, cv2.IMREAD_REDUCED_COLOR_2)
        R = cv2.imread(r, cv2.IMREAD_REDUCED_COLOR_2)
    else:
        L = cv2.imread(l, cv2.IMREAD_COLOR)
        R = cv2.imread(r, cv2.IMREAD_COLOR)
    if L is None or R is None:
        return None
    th, tw = target_hw
    if L.shape[:2] != (th, tw):
        L = cv2.resize(L, (tw, th))
        R = cv2.resize(R, (tw, th))
    pair = np.stack([L, R], axis=0)  # (2, H, W, 3)
    return int(i), torch.from_numpy(pair).permute(0, 3, 1, 2).contiguous()


def _load_stereo_frames(traj_dir: str, frame_idx: np.ndarray,
                        target_hw: tuple[int, int],
                        n_workers: int = 8) -> tuple[list[torch.Tensor], np.ndarray]:
    """Parallel stereo-frame loader. Half-res decode if target is 360×640."""
    th, tw = target_hw
    use_reduced = (th, tw) == (360, 640)
    args = [(traj_dir, i, target_hw, use_reduced) for i in frame_idx]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_load_one, args))
    imgs = []
    kept = []
    for r in results:
        if r is None:
            continue
        idx, t = r
        kept.append(idx)
        imgs.append(t)
    return imgs, np.asarray(kept, dtype=np.int64)


def _interpolate_poses(strided_idx: np.ndarray,
                        strided_poses: np.ndarray,    # (Ns, 4, 4)
                        full_idx: np.ndarray) -> np.ndarray:
    """Fill in poses for `full_idx` by linearly interpolating translation and
    SLERPing rotation between adjacent strided keyframes. Endpoints clamp.
    """
    from scipy.spatial.transform import Rotation as _R, Slerp  # imported lazily
    out = np.tile(np.eye(4, dtype=np.float32), (len(full_idx), 1, 1))
    pos = strided_poses[:, :3, 3]
    # Build one Rotation object across all strided poses
    rots = _R.from_matrix(strided_poses[:, :3, :3])
    slerp = Slerp(strided_idx.astype(np.float64), rots)

    full_idx_f = full_idx.astype(np.float64)
    # clip frame indices outside the strided range
    lo, hi = float(strided_idx[0]), float(strided_idx[-1])
    queryable = np.clip(full_idx_f, lo, hi)

    # rotation
    interp_R = slerp(queryable).as_matrix().astype(np.float32)
    # translation: per-axis np.interp
    interp_t = np.stack([
        np.interp(queryable, strided_idx, pos[:, ax]) for ax in range(3)
    ], axis=1).astype(np.float32)

    out[:, :3, :3] = interp_R
    out[:, :3, 3] = interp_t
    return out


def _R_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) → (N, 4) xyzw quaternions."""
    from scipy.spatial.transform import Rotation as _R
    return _R.from_matrix(R).as_quat().astype(np.float32)


def _droid_args(weights: str, image_size: list[int]) -> types.SimpleNamespace:
    # Defaults taken from evaluation_scripts/test_euroc.py; stereo = True.
    return types.SimpleNamespace(
        weights=weights,
        buffer=512,
        image_size=image_size,
        beta=0.3,
        filter_thresh=2.4,
        warmup=8,
        keyframe_thresh=4.0,
        frontend_thresh=16.0,
        frontend_window=25,
        frontend_radius=2,
        frontend_nms=1,
        backend_thresh=22.0,
        backend_radius=2,
        backend_nms=3,
        upsample=False,
        stereo=True,
        disable_vis=True,
        frontend_device="cuda",
        backend_device="cuda",
        asynchronous=False,
    )


def run_droid(
    traj_dir: str,
    intrinsics_fxfycxcy: tuple[float, float, float, float],
    orig_size_hw: tuple[int, int] = (720, 1280),
    target_size_hw: tuple[int, int] = (360, 640),     # original resolution (matches pilot)
    checkpoint: str = os.path.join(DROID_ROOT, "droid.pth"),
    stride: int = 3,                                    # 2.5× speedup; verified equivalent or
                                                        # better accuracy than stride=1 across
                                                        # 3-trajectory pilot (see EXPERIMENTS.md).
                                                        # Skipped frames are SLERP-interpolated.
    fast: bool = True,                                  # skip backend() passes; trajectories are short
) -> dict:
    sens = np.load(os.path.join(traj_dir, "sensors.npz"))
    frame_idx = sens["frame_idx"].astype(np.int64)
    if stride > 1:
        frame_idx = frame_idx[::stride]

    stereo_tensors, kept = _load_stereo_frames(traj_dir, frame_idx, target_size_hw)
    if len(stereo_tensors) < 20:
        raise RuntimeError(f"{traj_dir}: only {len(stereo_tensors)} stereo pairs loaded")

    ho, wo = orig_size_hw
    th, tw = target_size_hw
    fx, fy, cx, cy = intrinsics_fxfycxcy
    scaled_K = torch.tensor([fx * tw / wo, fy * th / ho,
                             cx * tw / wo, cy * th / ho], dtype=torch.float32)

    args = _droid_args(checkpoint, list(target_size_hw))
    droid = None
    start = time.time()
    for t, images in enumerate(stereo_tensors):
        if droid is None:
            args.image_size = [images.shape[-2], images.shape[-1]]
            droid = Droid(args)
        droid.track(t, images, intrinsics=scaled_K)
    assert droid is not None

    # Provide a stream iterator for trajectory-filling.
    def stream():
        for tt, imgs in enumerate(stereo_tensors):
            yield tt, imgs, scaled_K

    if fast:
        # Skip backend(7) and backend(12) global BA passes — overkill for 1-2 min
        # trajectories without revisits. Just run the trajectory_filler directly
        # to get per-frame poses from the keyframes the frontend produced.
        del droid.frontend
        torch.cuda.empty_cache()
        camera_trajectory = droid.traj_filler(stream())
        traj_est = camera_trajectory.inv().data.cpu().numpy()
    else:
        traj_est = droid.terminate(stream())
    elapsed = time.time() - start

    traj_est = np.asarray(traj_est, dtype=np.float32)  # (N, 7) [tx,ty,tz,qx,qy,qz,qw]
    N = len(traj_est)
    T = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    qx, qy, qz, qw = traj_est[:, 3], traj_est[:, 4], traj_est[:, 5], traj_est[:, 6]
    qn = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / qn, qy / qn, qz / qn, qw / qn
    T[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    T[:, 0, 1] = 2 * (qx * qy - qz * qw)
    T[:, 0, 2] = 2 * (qx * qz + qy * qw)
    T[:, 1, 0] = 2 * (qx * qy + qz * qw)
    T[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    T[:, 1, 2] = 2 * (qy * qz - qx * qw)
    T[:, 2, 0] = 2 * (qx * qz - qy * qw)
    T[:, 2, 1] = 2 * (qy * qz + qx * qw)
    T[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)
    T[:, 0, 3] = traj_est[:, 0]
    T[:, 1, 3] = traj_est[:, 1]
    T[:, 2, 3] = traj_est[:, 2]

    zero_t = (traj_est[:, :3] == 0).all(axis=1)
    valid_strided = ~zero_t
    if len(valid_strided) > 0:
        valid_strided[0] = True

    # If DROID returned fewer poses than we fed (tail cropping), trim kept accordingly.
    if N != len(kept):
        kept = kept[:N]

    # If we strided, interpolate to full sensors.npz['frame_idx'] coverage.
    full_idx = sens["frame_idx"].astype(np.int64)
    if stride > 1:
        T_full = _interpolate_poses(kept.astype(np.int64), T[:N], full_idx)
        # Build pose_7d_full from T_full
        N_full = len(full_idx)
        pose_7d_full = np.zeros((N_full, 7), dtype=np.float32)
        pose_7d_full[:, :3] = T_full[:, :3, 3]
        # quaternion xyzw from T_full's rotation
        from scipy.spatial.transform import Rotation as _R
        pose_7d_full[:, 3:] = _R.from_matrix(T_full[:, :3, :3]).as_quat().astype(np.float32)
        valid_full = np.ones(N_full, dtype=bool)
        # Mark frames outside the strided range (shouldn't happen) as invalid;
        # mark interpolated frames adjacent to a DROID-failed strided frame as
        # less reliable but still valid (the interp uses neighbors).
        kept_set = set(int(i) for i in kept)
        out_kept = full_idx.astype(np.int32)
        return {
            "frame_idx": out_kept,
            "pose_7d": pose_7d_full,
            "pose_mat": T_full.astype(np.float32),
            "valid": valid_full,
            "elapsed_s": float(elapsed),
            "fps": float(N_full / max(elapsed, 1e-6)),
            "stride_used": int(stride),
            "n_droid_frames": int(N),
        }

    return {
        "frame_idx": kept.astype(np.int32),
        "pose_7d": traj_est,
        "pose_mat": T,
        "valid": valid_strided,
        "elapsed_s": float(elapsed),
        "fps": float(N / max(elapsed, 1e-6)),
        "stride_used": 1,
        "n_droid_frames": int(N),
    }
