"""Run DROID-SLAM in stereo mode on a Go2 trajectory.

Returns per-frame camera-to-world poses aligned to sensors.npz['frame_idx'].
DROID-SLAM resizes images internally; we feed full-res PNGs and let its
front-end handle downsampling. Stereo mode uses left + right images stacked
on dim 0 as (2, 3, H, W).
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

warnings.filterwarnings("ignore", category=FutureWarning)

DROID_ROOT = "/home/anyone/projects/goodometry/third_party/DROID-SLAM"
if DROID_ROOT not in sys.path:
    sys.path.insert(0, DROID_ROOT)
if os.path.join(DROID_ROOT, "droid_slam") not in sys.path:
    sys.path.insert(0, os.path.join(DROID_ROOT, "droid_slam"))

from droid import Droid  # type: ignore


def _load_stereo_frames(traj_dir: str, frame_idx: np.ndarray,
                        target_hw: tuple[int, int]) -> tuple[list[torch.Tensor], np.ndarray]:
    th, tw = target_hw
    imgs = []
    kept = []
    for i in frame_idx:
        l = os.path.join(traj_dir, f"frame_{int(i):06d}", "left_image.png")
        r = os.path.join(traj_dir, f"frame_{int(i):06d}", "right_image.png")
        if not (os.path.exists(l) and os.path.exists(r)):
            continue
        L = cv2.imread(l, cv2.IMREAD_COLOR)
        R = cv2.imread(r, cv2.IMREAD_COLOR)
        if L is None or R is None:
            continue
        L = cv2.resize(L, (tw, th))
        R = cv2.resize(R, (tw, th))
        pair = np.stack([L, R], axis=0)  # (2, H, W, 3)
        t = torch.from_numpy(pair).permute(0, 3, 1, 2).contiguous()
        imgs.append(t)
        kept.append(int(i))
    return imgs, np.asarray(kept, dtype=np.int64)


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
    target_size_hw: tuple[int, int] = (360, 640),
    checkpoint: str = os.path.join(DROID_ROOT, "droid.pth"),
    stride: int = 1,
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

    # Provide a stream iterator for backend-side refinement.
    def stream():
        for tt, imgs in enumerate(stereo_tensors):
            yield tt, imgs, scaled_K

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
    valid = ~zero_t
    if len(valid) > 0:
        valid[0] = True

    # If DROID returned fewer poses than we fed (tail cropping), trim kept accordingly.
    if N != len(kept):
        kept = kept[:N]

    return {
        "frame_idx": kept.astype(np.int32),
        "pose_7d": traj_est,
        "pose_mat": T,
        "valid": valid,
        "elapsed_s": float(elapsed),
        "fps": float(N / max(elapsed, 1e-6)),
    }
