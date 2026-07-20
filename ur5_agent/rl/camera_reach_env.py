"""Simple camera-based reaching environment scaffold for RL training."""

from __future__ import annotations

import math
import random

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "gymnasium is required for RL env. Install with: pip install gymnasium"
    ) from e


class CameraReachEnv(gym.Env):
    """
    Lightweight 2D camera-centric reach task.
    Observation:
      - image: 84x84 RGB with target + gripper dots
      - state: [gripper_x, gripper_y, target_x, target_y] in [-1, 1]
    Action:
      - [dy, dz] normalized in [-1, 1] (maps to pixel-space movement)
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, image_size: int = 84, max_steps: int = 80):
        super().__init__()
        self.image_size = int(image_size)
        self.max_steps = int(max_steps)
        self.step_count = 0
        self._gripper = np.zeros(2, dtype=np.float32)
        self._target = np.zeros(2, dtype=np.float32)
        self._last_img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.image_size, self.image_size, 3),
                    dtype=np.uint8,
                ),
                "state": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(4,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def _draw(self) -> np.ndarray:
        img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img[:, :, :] = (10, 10, 12)

        def px(v: np.ndarray) -> tuple[int, int]:
            x = int((v[0] + 1.0) * 0.5 * (self.image_size - 1))
            y = int((v[1] + 1.0) * 0.5 * (self.image_size - 1))
            return x, y

        gx, gy = px(self._gripper)
        tx, ty = px(self._target)
        rr = 3
        img[max(0, ty - rr) : min(self.image_size, ty + rr + 1), max(0, tx - rr) : min(self.image_size, tx + rr + 1)] = (0, 220, 80)
        img[max(0, gy - rr) : min(self.image_size, gy + rr + 1), max(0, gx - rr) : min(self.image_size, gx + rr + 1)] = (220, 80, 0)
        return img

    def _obs(self) -> dict:
        self._last_img = self._draw()
        state = np.array(
            [self._gripper[0], self._gripper[1], self._target[0], self._target[1]],
            dtype=np.float32,
        )
        return {"image": self._last_img, "state": state}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._gripper = np.array([random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4)], dtype=np.float32)
        self._target = np.array([random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8)], dtype=np.float32)
        return self._obs(), {}

    def step(self, action):
        self.step_count += 1
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        delta = a * 0.06
        self._gripper = np.clip(self._gripper + delta, -1.0, 1.0)

        dist = float(np.linalg.norm(self._target - self._gripper))
        done = dist < 0.06
        truncated = self.step_count >= self.max_steps
        reward = -dist - 0.01 * float(np.linalg.norm(a))
        if done:
            reward += 2.0
        return self._obs(), reward, done, truncated, {"distance": dist}

    def render(self):
        return self._last_img
