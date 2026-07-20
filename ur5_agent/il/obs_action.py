"""Observation and action helpers shared by IL and RL deploy."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_reach_obs(state: dict[str, Any], target_xyz: list[float]) -> np.ndarray:
    q = state.get("joint_positions_rad") or [0.0] * 6
    tcp = state.get("tcp_pose") or [0.0] * 6
    return np.array(
        list(q[:6])
        + [float(tcp[0]), float(tcp[1]), float(tcp[2])]
        + [float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])],
        dtype=np.float32,
    )


def tcp_delta_action(tcp_before: list[float], tcp_after: list[float]) -> np.ndarray:
    b = tcp_before[:3]
    a = tcp_after[:3]
    return np.array(
        [float(a[0] - b[0]), float(a[1] - b[1]), float(a[2] - b[2])],
        dtype=np.float32,
    )


def clip_action_to_step(action: np.ndarray, max_step_m: float) -> np.ndarray:
    a = np.asarray(action, dtype=np.float64).reshape(3)
    mag = float(np.linalg.norm(a))
    cap = float(max_step_m)
    if mag > cap and mag > 1e-9:
        a = a * (cap / mag)
    return a.astype(np.float32)
