"""VO evaluation utilities: Umeyama alignment, ATE, RPE."""
from __future__ import annotations
import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (s, R, t) so that s*R @ src.T + t ≈ dst.T.

    src, dst: (N, 3) float64. with_scale=True fits a scalar; False holds s=1.
    """
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 3
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    H = src_c.T @ dst_c / len(src)
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    if with_scale:
        var_s = (src_c ** 2).sum() / len(src)
        s = float((D * np.diag(S)).sum() / max(var_s, 1e-12))
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_sim3(positions: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (s * (R @ positions.T)).T + t


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz -> (N, 3, 3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (...,) angle in degrees via arccos((tr-1)/2)."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def ate(pred_pos: np.ndarray, gt_pos: np.ndarray, with_scale: bool) -> dict:
    """Absolute Trajectory Error after Umeyama alignment. Returns RMSE, mean, median, max (m) + scale."""
    s, R, t = umeyama(pred_pos.astype(np.float64), gt_pos.astype(np.float64), with_scale=with_scale)
    aligned = apply_sim3(pred_pos.astype(np.float64), s, R, t)
    err = np.linalg.norm(aligned - gt_pos, axis=1)
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "ate_mean_m": float(np.mean(err)),
        "ate_median_m": float(np.median(err)),
        "ate_max_m": float(np.max(err)),
        "scale": float(s),
    }


def rpe(pred_pos: np.ndarray, pred_R: np.ndarray, gt_pos: np.ndarray, gt_R: np.ndarray,
        delta_frames: int) -> dict:
    """Relative Pose Error over windows of `delta_frames` (translation RMSE in m, rotation RMSE in deg)."""
    N = len(pred_pos)
    i0 = np.arange(0, N - delta_frames)
    i1 = i0 + delta_frames

    def rel(P0, P1, R0, R1):
        # relative pose of P1 in P0's frame: R0^T (P1 - P0), R0^T R1
        dp = np.einsum("nij,nj->ni", R0.transpose(0, 2, 1), P1 - P0)
        dR = np.einsum("nij,njk->nik", R0.transpose(0, 2, 1), R1)
        return dp, dR

    dp_pred, dR_pred = rel(pred_pos[i0], pred_pos[i1], pred_R[i0], pred_R[i1])
    dp_gt, dR_gt = rel(gt_pos[i0], gt_pos[i1], gt_R[i0], gt_R[i1])
    trans_err = np.linalg.norm(dp_pred - dp_gt, axis=1)
    rel_err_R = np.einsum("nij,njk->nik", dR_pred.transpose(0, 2, 1), dR_gt)
    rot_err_deg = rotation_angle_deg(rel_err_R)
    return {
        f"rpe_trans_rmse_{delta_frames}f_m": float(np.sqrt(np.mean(trans_err ** 2))),
        f"rpe_rot_rmse_{delta_frames}f_deg": float(np.sqrt(np.mean(rot_err_deg ** 2))),
        f"rpe_trans_median_{delta_frames}f_m": float(np.median(trans_err)),
        f"rpe_rot_median_{delta_frames}f_deg": float(np.median(rot_err_deg)),
    }
