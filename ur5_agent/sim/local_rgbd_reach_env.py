"""Local RGB-D reach env (dev without Isaac). Mirrors Isaac task for RL prototyping."""

from __future__ import annotations

import random

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise RuntimeError("gymnasium required: pip install gymnasium") from e


class LocalRgbdReachEnv(gym.Env):
    """
    Lightweight 3D reach with synthetic RGB patch + state vector.
    Use for RL/VLA prototyping before Isaac Sim is wired.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, image_size: int = 84, max_steps: int = 80, max_delta_m: float = 0.01):
        super().__init__()
        self.image_size = int(image_size)
        self.max_steps = int(max_steps)
        self.max_delta_m = float(max_delta_m)
        self.step_count = 0
        self.joint = np.zeros(6, dtype=np.float32)
        self.tcp = np.zeros(3, dtype=np.float32)
        self.target = np.zeros(3, dtype=np.float32)
        self._img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self._depth = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8),
                "depth": spaces.Box(0.0, 3.0, (self.image_size, self.image_size), dtype=np.float32),
                "state": spaces.Box(-10.0, 10.0, (12,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

    def _project_target(self) -> tuple[int, int]:
        # Fake pinhole: map target offset from tcp to image center shift.
        err = self.target - self.tcp
        u = int(self.image_size * 0.5 + float(err[1]) * 180.0)
        v = int(self.image_size * 0.5 - float(err[2]) * 180.0)
        u = max(0, min(self.image_size - 1, u))
        v = max(0, min(self.image_size - 1, v))
        return u, v

    def _render_obs(self) -> dict:
        img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img[:, :] = (18, 20, 28)
        depth = np.ones((self.image_size, self.image_size), dtype=np.float32) * 0.8
        u, v = self._project_target()
        rr = 4
        dist = float(np.linalg.norm(self.target - self.tcp))
        depth_val = max(0.15, min(1.2, dist + 0.25))
        y0, y1 = max(0, v - rr), min(self.image_size, v + rr + 1)
        x0, x1 = max(0, u - rr), min(self.image_size, u + rr + 1)
        img[y0:y1, x0:x1] = (40, 220, 90)
        depth[y0:y1, x0:x1] = depth_val
        gx, gy = self.image_size // 2, self.image_size // 2
        img[gy - 2 : gy + 3, gx - 2 : gx + 3] = (230, 120, 40)
        self._img, self._depth = img, depth
        state = np.concatenate([self.joint, self.tcp, self.target]).astype(np.float32)
        return {"image": img, "depth": depth, "state": state}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.tcp = np.array(
            [random.uniform(0.22, 0.42), random.uniform(-0.15, 0.15), random.uniform(0.22, 0.45)],
            dtype=np.float32,
        )
        self.target = np.array(
            [random.uniform(0.22, 0.42), random.uniform(-0.15, 0.15), random.uniform(0.22, 0.45)],
            dtype=np.float32,
        )
        self.joint = np.zeros(6, dtype=np.float32)
        obs = self._render_obs()
        return obs, {"distance": float(np.linalg.norm(self.target - self.tcp))}

    def step(self, action):
        self.step_count += 1
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.tcp = self.tcp + a * self.max_delta_m
        self.tcp = np.clip(self.tcp, [0.15, -0.35, 0.12], [0.55, 0.35, 0.55])
        dist = float(np.linalg.norm(self.target - self.tcp))
        done = dist < 0.008
        truncated = self.step_count >= self.max_steps
        reward = -dist - 0.01 * float(np.linalg.norm(a))
        if done:
            reward += 2.0
        return self._render_obs(), reward, done, truncated, {"distance": dist}

    def render(self):
        return self._img
