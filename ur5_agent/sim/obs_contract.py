"""Shared observation/action contract for real robot, Isaac Sim, and VLA policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from il.obs_action import build_reach_obs


class ObsMode(str, Enum):
    STATE = "state"  # joints + tcp + target_xyz (12)
    RGBD_STATE = "rgbd_state"  # dict: image HxWx3, depth, state12
    VLA = "vla"  # dict: image, instruction text, optional state


@dataclass
class ReachAction:
    dx: float
    dy: float
    dz: float

    def as_list(self) -> list[float]:
        return [self.dx, self.dy, self.dz]


def state_vector_to_dict(obs12: np.ndarray) -> dict[str, Any]:
    o = np.asarray(obs12, dtype=np.float32).reshape(12)
    return {
        "joint_positions_rad": [float(v) for v in o[:6]],
        "tcp_xyz": [float(v) for v in o[6:9]],
        "target_xyz": [float(v) for v in o[9:12]],
    }


def build_state12(state: dict[str, Any], target_xyz: list[float]) -> np.ndarray:
    return build_reach_obs(state, target_xyz)


def normalize_action(action: np.ndarray, max_step_m: float) -> ReachAction:
    a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[:3], -1.0, 1.0)
    delta = a * float(max_step_m)
    return ReachAction(float(delta[0]), float(delta[1]), float(delta[2]))


def stack_policy_inputs(
    mode: ObsMode | str,
    state: dict[str, Any],
    target_xyz: list[float],
    rgb: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    instruction: str = "",
) -> dict[str, Any] | np.ndarray:
    m = mode if isinstance(mode, ObsMode) else ObsMode(str(mode))
    if m == ObsMode.STATE:
        return build_state12(state, target_xyz)
    if m == ObsMode.RGBD_STATE:
        return {
            "state": build_state12(state, target_xyz),
            "rgb": rgb,
            "depth": depth,
            "target_xyz": list(target_xyz),
        }
    if m == ObsMode.VLA:
        return {
            "rgb": rgb,
            "instruction": instruction.strip(),
            "state": build_state12(state, target_xyz),
            "target_xyz": list(target_xyz),
        }
    raise ValueError(f"Unknown obs mode: {mode}")
