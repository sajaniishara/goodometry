"""Forward kinematics for the Unitree Go2 using pytorch_kinematics.

Handles two index-order mismatches that bit us once:

1. `sensors.npz['joints']` uses Isaac Sim's *grouped-by-type* DOF order:
       [FL_hip, FR_hip, RL_hip, RR_hip,
        FL_thigh, FR_thigh, RL_thigh, RR_thigh,
        FL_calf, FR_calf, RL_calf, RR_calf]
   The Go2 URDF (unitree_ros/robots/go2_description/urdf/go2_description.urdf)
   uses *per-leg* order inside the kinematic chain:
       [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
        RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
   We remap using SENSORS_TO_URDF.

2. The URDF's `*_foot_joint` is a fixed-offset child of the corresponding
   calf link (offset (0, 0, -0.213) m). pytorch_kinematics' URDF loader
   collapses that fixed joint into the calf frame, so FK only reaches the
   calf origin. We apply the foot offset manually in the calf frame.
"""
from __future__ import annotations
import os
import numpy as np
import torch
import pytorch_kinematics as pk

# sensors[SENSORS_TO_URDF] → URDF joint-order array.
# URDF order: FL(hip,thigh,calf), FR(...), RL(...), RR(...)
# sensors order: 4 hips, 4 thighs, 4 calves, legs in (FL, FR, RL, RR)
# so URDF[0..2] = sensors[0, 4, 8]  (FL hip, thigh, calf)
#    URDF[3..5] = sensors[1, 5, 9]  (FR hip, thigh, calf)
#    URDF[6..8] = sensors[2, 6, 10] (RL hip, thigh, calf)
#    URDF[9..11]= sensors[3, 7, 11] (RR hip, thigh, calf)
SENSORS_TO_URDF = np.array([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11], dtype=np.int64)

# Foot offsets in their parent-calf frame (URDF-verified, all four identical on Go2).
FOOT_OFFSET_IN_CALF = np.array([0.0, 0.0, -0.213], dtype=np.float32)

# Calf frames (URDF) in the FL, FR, RL, RR order we want for the 4-leg output.
FOOT_FRAMES = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]

DEFAULT_URDF = (
    "/home/anyone/projects/goodometry/third_party/unitree_ros/"
    "robots/go2_description/urdf/go2_description.urdf"
)


class Go2FK:
    """Batched Go2 FK returning per-foot body-frame position + velocity."""

    def __init__(self, urdf_path: str = DEFAULT_URDF, device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        with open(urdf_path, "rb") as f:
            self.chain = pk.build_chain_from_urdf(f.read()).to(dtype=dtype, device=device)
        self.device = device
        self.dtype = dtype
        self.foot_offset = torch.tensor(FOOT_OFFSET_IN_CALF, dtype=dtype, device=device)

        # sanity check the URDF ordering matches our remap expectation
        expected = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
                    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
                    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
        got = self.chain.get_joint_parameter_names()
        if got != expected:
            raise RuntimeError(f"URDF joint order unexpected: got {got}")

    def foot_positions_body(self, sensor_joint_angles: torch.Tensor) -> torch.Tensor:
        """sensor_joint_angles: (N, 12) in Isaac-Sim grouped order (positions only).
        returns: (N, 4, 3) foot positions in body frame, order [FL, FR, RL, RR].
        """
        assert sensor_joint_angles.dim() == 2 and sensor_joint_angles.shape[1] == 12
        q_urdf = sensor_joint_angles[:, SENSORS_TO_URDF].to(self.dtype).to(self.device)
        transforms = self.chain.forward_kinematics(q_urdf)
        N = q_urdf.shape[0]
        foot_pos = torch.empty((N, 4, 3), dtype=self.dtype, device=self.device)
        for i, name in enumerate(FOOT_FRAMES):
            T = transforms[name].get_matrix()            # (N, 4, 4)  calf-in-base
            R = T[:, :3, :3]                              # (N, 3, 3)
            p_calf = T[:, :3, 3]                          # (N, 3)
            # foot = calf_origin + R_calf * foot_offset_in_calf
            foot_pos[:, i] = p_calf + (R @ self.foot_offset)
        return foot_pos


def foot_vel_body_from_finite_diff(foot_pos_body: torch.Tensor, dt: float) -> torch.Tensor:
    """Centered finite difference on body-frame foot positions.

    foot_pos_body: (N, 4, 3)  →  foot_vel_body: (N, 4, 3).
    Endpoints use forward / backward difference (first- and last-frame values).
    """
    N = foot_pos_body.shape[0]
    v = torch.zeros_like(foot_pos_body)
    if N >= 3:
        v[1:-1] = (foot_pos_body[2:] - foot_pos_body[:-2]) / (2.0 * dt)
        v[0]    = (foot_pos_body[1] - foot_pos_body[0]) / dt
        v[-1]   = (foot_pos_body[-1] - foot_pos_body[-2]) / dt
    elif N == 2:
        v[0] = v[1] = (foot_pos_body[1] - foot_pos_body[0]) / dt
    # N=1 case: zero velocity (already zeros).
    return v
