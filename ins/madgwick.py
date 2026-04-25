"""Madgwick INS preprocessing via xIOTechnologies `imufusion` (the production
variant of the Madgwick AHRS filter, maintained by Madgwick himself).

Input (all body-frame, 30 Hz):
  imu   : (N, 6)   [accel_xyz (m/s²), gyro_xyz (rad/s)]
  mag   : (N, 3)   3D magnetic field vector (any unit; Madgwick normalizes)

Output:
  quat_wxyz     : (N, 4)   body-to-world quaternion (unit-norm)
  gyro_body     : (N, 3)   pass-through of the input gyro
  accel_body    : (N, 3)   gravity-compensated, *body* frame
  accel_world   : (N, 3)   gravity-compensated, *world* frame
  euler_rpy     : (N, 3)   roll, pitch, yaw (rad) derived from quat
  flags         : (N, 4)   [initialising, angular_rate_recovery,
                            accel_recovery, magnetic_recovery]   uint8

World convention: ENU (+X=east, +Y=north, +Z=up). The robot body is +X fwd,
+Y left, +Z up — so the initial quaternion when the robot stands facing +X is
the identity when we interpret "east" as "forward of initial heading".
"""
from __future__ import annotations
import numpy as np
import imufusion


_G = 9.81


def _quat_to_R(q_wxyz: np.ndarray) -> np.ndarray:
    """(N, 4) quaternion wxyz -> (N, 3, 3) rotation matrix (body-to-world)."""
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.empty((len(q_wxyz), 3, 3), dtype=np.float32)
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


def _quat_to_euler_rpy(q_wxyz: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz -> (N, 3) intrinsic XYZ (roll, pitch, yaw) in radians."""
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    # ZYX intrinsic (Tait-Bryan) is the common ROS/robotics convention.
    # roll (x): atan2(2(wy·z + wx·y), 1 - 2(x² + y²))
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([roll, pitch, yaw], axis=1).astype(np.float32)


def run_madgwick(
    imu: np.ndarray,
    mag: np.ndarray | None,
    sample_rate_hz: float = 30.0,
    use_mag: bool = False,           # Go2 sim mag is not a realistic Earth field; disable by default.
    gain: float = 1.0,                # higher than imufusion's default (0.5); trust accel during gait impacts.
    acceleration_rejection_deg: float = 0.0,  # 0 = no rejection. Trot imposes ±4 m/s² on the sensor every step;
                                              # default 10° rejection triggers constantly and lets gyro drift.
    magnetic_rejection_deg: float = 0.0,
    convention: str = "ENU",         # "ENU" / "NED" / "NWU" — must be ENU for Z-up bodies.
    mag_axis_order: str = "ENU",     # how the input mag's three components are labelled.
                                     # The Go2 sim stores them as (N, E, D); set to "NED" to remap → ENU.
) -> dict:
    """Run imufusion Madgwick AHRS filter over a full trajectory.

    Parameters
    ----------
    imu : (N, 6)    accel (m/s²) + gyro (rad/s), body frame.
    mag : (N, 3)    magnetic field (arbitrary unit); may be None if use_mag=False.
    sample_rate_hz : per-sample rate (default 30 Hz for the Go2 v2 dataset).
    use_mag : enable MARG mode (9-axis); False = IMU-only (6-axis, yaw drifts).
    gain : Madgwick β gain. 0.5 is imufusion's default, tuned for typical IMU.

    Returns a dict of numpy arrays (see module docstring).
    """
    assert imu.ndim == 2 and imu.shape[1] == 6, imu.shape
    N = len(imu)
    if use_mag:
        assert mag is not None and mag.shape == (N, 3), f"mag shape {mag.shape if mag is not None else None}"

    conv_map = {
        "ENU": imufusion.CONVENTION_ENU,
        "NED": imufusion.CONVENTION_NED,
        "NWU": imufusion.CONVENTION_NWU,
    }
    if convention not in conv_map:
        raise ValueError(f"convention must be one of {list(conv_map)}, got {convention}")
    ahrs = imufusion.Ahrs()
    settings = imufusion.Settings(
        conv_map[convention],
        gain,
        2000.0,                                 # gyroscope range (dps)
        acceleration_rejection_deg,
        magnetic_rejection_deg if use_mag else 0.0,
        int(5 * sample_rate_hz),                # recovery trigger period
    )
    ahrs.settings = settings

    dt = 1.0 / sample_rate_hz
    quat_wxyz = np.empty((N, 4), dtype=np.float32)
    accel_body = np.empty((N, 3), dtype=np.float32)
    accel_world = np.empty((N, 3), dtype=np.float32)
    flags_out = np.empty((N, 4), dtype=np.uint8)

    # imufusion expects gyro in degrees/second, accel in g's
    gyro_dps = imu[:, 3:6].astype(np.float32) * (180.0 / np.pi)
    accel_g = imu[:, :3].astype(np.float32) / _G
    if use_mag:
        m = mag.astype(np.float32)
        if mag_axis_order == "ENU":
            mag_uT = m
        elif mag_axis_order == "NED":
            # (N, E, D) → (E, N, U) for ENU-convention filter
            mag_uT = np.stack([m[:, 1], m[:, 0], -m[:, 2]], axis=1).astype(np.float32)
        elif mag_axis_order == "NWU":
            # (N, W, U) → (E, N, U)
            mag_uT = np.stack([-m[:, 1], m[:, 0], m[:, 2]], axis=1).astype(np.float32)
        else:
            raise ValueError(f"mag_axis_order must be ENU/NED/NWU, got {mag_axis_order}")
    else:
        mag_uT = None

    for i in range(N):
        if use_mag:
            ahrs.update(gyro_dps[i], accel_g[i], mag_uT[i], dt)
        else:
            ahrs.update_no_magnetometer(gyro_dps[i], accel_g[i], dt)
        # imufusion returns quaternion as a Quaternion object with .w/.x/.y/.z;
        # linear/earth acceleration as (3,) numpy arrays; flags as a Flags object.
        q = ahrs.quaternion
        quat_wxyz[i, 0] = q.w
        quat_wxyz[i, 1] = q.x
        quat_wxyz[i, 2] = q.y
        quat_wxyz[i, 3] = q.z
        accel_body[i]  = np.asarray(ahrs.linear_acceleration, dtype=np.float32) * _G
        accel_world[i] = np.asarray(ahrs.earth_acceleration, dtype=np.float32) * _G
        fl = ahrs.flags
        flags_out[i] = [fl.initialising, fl.angular_rate_recovery,
                        fl.acceleration_recovery, fl.magnetic_recovery]

    out = {
        "quat_wxyz":   quat_wxyz,
        "gyro_body":   imu[:, 3:6].astype(np.float32),
        "accel_body":  accel_body,
        "accel_world": accel_world,
        "euler_rpy":   _quat_to_euler_rpy(quat_wxyz),
        "flags":       flags_out,
    }

    if use_mag:
        # imufusion's internal MARG mode doesn't converge well on the Go2 sim's
        # mag (we tested every reasonable rejection threshold). Two-pass stitch
        # works: IMU-only Madgwick gives reliable roll/pitch, then a standard
        # tilt-compensated heading from the body-frame mag gives drift-free yaw.
        # The user remap (NED → ENU) was applied to mag_uT above, but for the
        # tilt-compensation formula we want the original (N, E, D) labels —
        # tilt-compensation is convention-agnostic, just needs the body axes.
        roll  = out["euler_rpy"][:, 0]
        pitch = out["euler_rpy"][:, 1]
        m_orig = mag.astype(np.float32)        # original (N, E, D) per Go2 sim
        mx, my, mz = m_orig[:, 0], m_orig[:, 1], m_orig[:, 2]
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        xh = mx * cp + my * sr * sp + mz * cr * sp           # horizontal-N
        yh = my * cr - mz * sr                                # horizontal-E
        yaw_mag = np.arctan2(-yh, xh).astype(np.float32)     # heading from N

        # Stitch (roll, pitch, yaw_mag) → quaternion (body-to-world ZYX)
        cr2, sr2 = np.cos(roll/2), np.sin(roll/2)
        cp2, sp2 = np.cos(pitch/2), np.sin(pitch/2)
        cy2, sy2 = np.cos(yaw_mag/2), np.sin(yaw_mag/2)
        q_w = (cr2*cp2*cy2 + sr2*sp2*sy2).astype(np.float32)
        q_x = (sr2*cp2*cy2 - cr2*sp2*sy2).astype(np.float32)
        q_y = (cr2*sp2*cy2 + sr2*cp2*sy2).astype(np.float32)
        q_z = (cr2*cp2*sy2 - sr2*sp2*cy2).astype(np.float32)
        out["quat_wxyz"] = np.stack([q_w, q_x, q_y, q_z], axis=1)
        out["euler_rpy"] = np.stack([roll, pitch, yaw_mag], axis=1).astype(np.float32)

        # Recompute world-frame accel using the stitched quaternion.
        # body-to-world: a_world = R_wb · a_body
        R = _quat_to_R(out["quat_wxyz"])
        out["accel_world"] = np.einsum("nij,nj->ni",
                                       R, accel_body).astype(np.float32)

    return out


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.empty((len(q), 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2*(y*y + z*z); R[:, 0, 1] = 2*(x*y - z*w); R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w);   R[:, 1, 1] = 1 - 2*(x*x + z*z); R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w);   R[:, 2, 1] = 2*(y*z + x*w);   R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R
