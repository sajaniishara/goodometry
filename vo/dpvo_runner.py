"""Run DPVO (monocular) on a Go2 trajectory and return per-frame poses
aligned to sensors.npz['frame_idx'].

Uses the left camera PNG frames at full resolution (1280x720). Camera frame
is Isaac Sim's (+X fwd, +Y left, +Z up); DPVO is agnostic to orientation
convention — we just convert its output camera-to-world poses to a 4x4
matrix for downstream evaluation.
"""
from __future__ import annotations
import os
import sys
import time
import warnings
import numpy as np
import torch
import cv2

warnings.filterwarnings("ignore", category=FutureWarning)

DPVO_ROOT = "/home/anyone/projects/goodometry/third_party/DPVO"
if DPVO_ROOT not in sys.path:
    sys.path.insert(0, DPVO_ROOT)

from dpvo.dpvo import DPVO  # type: ignore
from dpvo.config import cfg  # type: ignore
from dpvo.net import VONet  # type: ignore




def _load_frames(traj_dir: str, frame_idx: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Load left PNG frames at full res for all frame_idx. Returns (frames, kept_idx).

    Any frame missing from disk is dropped, with the corresponding kept_idx
    aligned so downstream code can account for it.
    """
    imgs = []
    kept = []
    for i in frame_idx:
        p = os.path.join(traj_dir, f"frame_{int(i):06d}", "left_image.png")
        if not os.path.exists(p):
            continue
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        img = img[: h - h % 16, : w - w % 16]
        imgs.append(img)
        kept.append(int(i))
    return imgs, np.asarray(kept, dtype=np.int64)


def run_dpvo(
    traj_dir: str,
    intrinsics_fxfycxcy: tuple[float, float, float, float],
    checkpoint: str = os.path.join(DPVO_ROOT, "models/dpvo.pth"),
    device: str = "cuda",
    stride: int = 1,
) -> dict:
    """Run DPVO on a single trajectory directory.

    Returns a dict with:
      frame_idx : (N,) int32
      t_sec     : (N,) float64     — per-frame timestamps copied from data.csv via gt_loader (filled by caller)
      pose_7d   : (N, 7) float32   — [tx,ty,tz,qx,qy,qz,qw] camera-to-world
      pose_mat  : (N, 4, 4) float32 — SE(3) matrix form
      valid     : (N,) bool        — False if DPVO reported zero pose / the frame was dropped
      elapsed_s : float            — wall-clock time for the VO run
      fps       : float
    """
    sens = np.load(os.path.join(traj_dir, "sensors.npz"))
    frame_idx = sens["frame_idx"].astype(np.int64)
    frames, kept = _load_frames(traj_dir, frame_idx)
    if len(frames) < 10:
        raise RuntimeError(f"{traj_dir}: only {len(frames)} frames loaded")

    if stride > 1:
        frames = frames[::stride]
        kept = kept[::stride]

    cfg.merge_from_file(os.path.join(DPVO_ROOT, "config/default.yaml"))
    cfg.BUFFER_SIZE = 2048
    cfg.LOOP_CLOSURE = False
    cfg.CLASSIC_LOOP_CLOSURE = False

    fx, fy, cx, cy = intrinsics_fxfycxcy
    intrin_np = np.array([fx, fy, cx, cy], dtype=np.float32)

    slam: DPVO | None = None
    start = time.time()
    with torch.no_grad():
        for t, img in enumerate(frames):
            image = torch.from_numpy(img).permute(2, 0, 1).to(device)
            intrinsics = torch.from_numpy(intrin_np).to(device)
            if slam is None:
                _, H, W = image.shape
                # DPVO takes either a checkpoint path (string) or a pre-built network (nn.Module).
                slam = DPVO(cfg, checkpoint, ht=H, wd=W, viz=False)
            slam(t, image, intrinsics)
        assert slam is not None
        (poses, tstamps) = slam.terminate()
    elapsed = time.time() - start

    poses = np.asarray(poses, dtype=np.float32)  # (N, 7) [tx,ty,tz,qx,qy,qz,qw]
    tstamps = np.asarray(tstamps, dtype=np.float64)

    # Build 4x4 matrices.
    N = len(poses)
    T = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    qx, qy, qz, qw = poses[:, 3], poses[:, 4], poses[:, 5], poses[:, 6]
    # Normalize
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
    T[:, 0, 3] = poses[:, 0]
    T[:, 1, 3] = poses[:, 1]
    T[:, 2, 3] = poses[:, 2]

    # Any fully-zero translations (can happen if DPVO fails to converge on a frame) flagged invalid.
    zero_t = (poses[:, :3] == 0).all(axis=1)
    valid = ~zero_t
    # The first frame is conventionally the origin; mark it valid.
    if len(valid) > 0:
        valid[0] = True

    return {
        "frame_idx": kept.astype(np.int32),
        "pose_7d": poses,
        "pose_mat": T,
        "valid": valid,
        "elapsed_s": float(elapsed),
        "fps": float(N / max(elapsed, 1e-6)),
    }
