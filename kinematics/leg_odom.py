"""Leg odometry: body linear velocity from stance-foot constraint.

Per `go2-concrete-algorithms.md` §1.2. For a stance foot the no-slip
assumption gives v_foot^W = 0, which means in the body frame:

    v_body^B = -omega_body^B × p_foot^B - dp_foot^B/dt

where:
  - `omega_body^B` is the gyro reading (already body-frame).
  - `p_foot^B` is the foot position in body frame from FK.
  - `dp_foot^B/dt` is the body-frame foot velocity (FK-derived).

Each foot gives an independent estimate; we combine across feet weighted by
contact probability.
"""
from __future__ import annotations
import torch


def body_velocity_from_legs(
    foot_pos_body: torch.Tensor,       # (N, 4, 3)
    foot_vel_body: torch.Tensor,       # (N, 4, 3)
    omega_body: torch.Tensor,          # (N, 3)  gyro in body frame
    contact_prob: torch.Tensor,        # (N, 4)  [0, 1]
    eps: float = 1e-3,
) -> torch.Tensor:
    """Return (N, 3) body linear velocity estimate.

    When the total contact weight Σ c_i is small (all feet in swing), the
    estimate is unobservable — we zero it out rather than divide by ~0, and
    signal that via the caller's contact-weight sum (stored separately if
    needed). The fusion network sees the zeros alongside the probability
    vector so it can learn to discount those timesteps.
    """
    # omega × p for each foot: broadcast (N, 1, 3) across (N, 4, 3)
    w = omega_body.unsqueeze(1)                              # (N, 1, 3)
    omega_cross_p = torch.cross(w.expand_as(foot_pos_body),  # (N, 4, 3)
                                foot_pos_body, dim=-1)
    v_per_foot = -omega_cross_p - foot_vel_body              # (N, 4, 3)
    c = contact_prob.unsqueeze(-1)                           # (N, 4, 1)
    weighted_sum = (v_per_foot * c).sum(dim=1)               # (N, 3)
    total_weight = c.sum(dim=1).clamp_min(eps)               # (N, 1)
    return weighted_sum / total_weight
