"""Gymnasium reach environment (state-only baseline for PPO/SAC)."""

from __future__ import annotations

import random

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "gymnasium is required for reach env. Install with: pip install gymnasium"
    ) from e


class ReachEnv(gym.Env):
    """
    Free-space reaching baseline.
    Observation:
      [joint(6), tcp_xyz(3), target_xyz(3)]  -> 12 floats
    Action:
      [dx, dy, dz] normalized in [-1, 1], scaled by max_delta_m.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 60, max_delta_m: float = 0.01):
        super().__init__()
        self.max_steps = int(max_steps)
        self.max_delta_m = float(max_delta_m)
        self.step_count = 0
        self.joint = np.zeros(6, dtype=np.float32)
        self.tcp = np.zeros(3, dtype=np.float32)
        self.target = np.zeros(3, dtype=np.float32)
        self._last_dist = 0.0

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.joint, self.tcp, self.target]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        # Randomized free-space initialization near nominal UR5 workspace.
        self.tcp = np.array(
            [
                random.uniform(0.20, 0.45),
                random.uniform(-0.20, 0.20),
                random.uniform(0.20, 0.50),
            ],
            dtype=np.float32,
        )
        self.target = np.array(
            [
                random.uniform(0.20, 0.45),
                random.uniform(-0.20, 0.20),
                random.uniform(0.20, 0.50),
            ],
            dtype=np.float32,
        )
        self.joint = np.zeros(6, dtype=np.float32)
        self._last_dist = float(np.linalg.norm(self.target - self.tcp))
        return self._obs(), {"distance": self._last_dist}

    def step(self, action):
        self.step_count += 1
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        delta = a * self.max_delta_m
        self.tcp = self.tcp + delta

        # Keep within a simple safety bounding box.
        clipped = np.array(
            [
                np.clip(self.tcp[0], 0.15, 0.55),
                np.clip(self.tcp[1], -0.30, 0.30),
                np.clip(self.tcp[2], 0.12, 0.60),
            ],
            dtype=np.float32,
        )
        unsafe_pen = 0.0 if np.allclose(clipped, self.tcp) else 0.2
        self.tcp = clipped

        dist = float(np.linalg.norm(self.target - self.tcp))
        progress = self._last_dist - dist
        self._last_dist = dist

        done = dist < 0.015
        truncated = self.step_count >= self.max_steps

        reward = (
            2.0 * progress            # move toward target
            - 0.02 * float(np.linalg.norm(a))  # penalize large action
            - 0.01                    # time penalty
            - unsafe_pen              # unsafe zone penalty
        )
        if done:
            reward += 2.0

        info = {"distance": dist, "progress": progress}
        return self._obs(), float(reward), bool(done), bool(truncated), info
