"""Contact probability for the four Go2 feet, from body-frame foot velocities.

Uses the velocity-gate recipe from `go2-concrete-algorithms.md` §1.3:
a foot is likely in stance when it's not moving relative to the body.
Output is a soft probability in [0, 1] per foot per timestep — Schmitt-style
hysteresis isn't needed for soft probabilities; the smooth sigmoid handles
the transition cleanly for downstream attention / weighting.
"""
from __future__ import annotations
import torch


def velocity_gate_contact(
    foot_vel_body: torch.Tensor,      # (N, 4, 3)
    swing_speed_thresh: float = 0.50,  # m/s — above this, confidently in swing
    stance_speed_thresh: float = 0.10, # m/s — below this, confidently in stance
) -> torch.Tensor:
    """Return (N, 4) soft contact probability in [0, 1].

    The intuition: a swing foot sweeps at ~0.3–1.0 m/s in body frame for a
    trot gait; a stance foot is near-stationary. We map |v_foot^B| through a
    smooth ramp from `stance_speed_thresh` to `swing_speed_thresh`:

        1.0 if |v| <= stance      (definitely stance)
        0.0 if |v| >= swing       (definitely swing)
        linear ramp between.

    Chosen over a sigmoid for interpretability — the thresholds are the
    operating points rather than a slope-and-bias pair.
    """
    assert foot_vel_body.dim() == 3 and foot_vel_body.shape[-1] == 3
    speed = foot_vel_body.norm(dim=-1)                           # (N, 4)
    denom = max(swing_speed_thresh - stance_speed_thresh, 1e-6)
    p = (swing_speed_thresh - speed) / denom
    return p.clamp(0.0, 1.0)
