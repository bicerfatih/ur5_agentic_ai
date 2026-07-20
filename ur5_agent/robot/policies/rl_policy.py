"""RL / BC policy inference wrapper with safe fallback controller."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from il.obs_action import build_reach_obs, clip_action_to_step


@dataclass
class RLStep:
    dx: float
    dy: float
    dz: float
    source: str


class ReachPolicyRunner:
    def __init__(self, policy_path: str = ""):
        self.policy_path = (policy_path or "").strip()
        self._model = None
        self._bc_W: np.ndarray | None = None
        self._bc_b: np.ndarray | None = None
        if self.policy_path:
            self._try_load()

    def _resolve_bc_path(self) -> str:
        p = self.policy_path
        if p.endswith("bc_weights.npz") and os.path.isfile(p):
            return p
        if os.path.isdir(p):
            candidate = os.path.join(p, "bc_weights.npz")
            if os.path.isfile(candidate):
                return candidate
        return ""

    def _try_load_bc(self) -> bool:
        bc_path = self._resolve_bc_path()
        if not bc_path:
            return False
        try:
            data = np.load(bc_path)
            self._bc_W = np.asarray(data["W"], dtype=np.float32)
            self._bc_b = np.asarray(data["b"], dtype=np.float32)
            self.policy_path = bc_path
            return True
        except Exception:
            self._bc_W = None
            self._bc_b = None
            return False

    def _try_load(self):
        if self._try_load_bc():
            return
        try:
            from stable_baselines3 import PPO, SAC
        except Exception:
            self._model = None
            return
        for cls in (PPO, SAC):
            try:
                self._model = cls.load(self.policy_path)
                return
            except Exception:
                continue
        self._model = None

    @staticmethod
    def _build_obs(state: dict[str, Any], target_xyz: list[float]) -> np.ndarray:
        return build_reach_obs(state, target_xyz)

    def step(self, state: dict[str, Any], target_xyz: list[float], max_step_m: float = 0.01) -> RLStep:
        tcp = state.get("tcp_pose") or [0.0, 0.0, 0.0]
        err = np.array(target_xyz[:3], dtype=np.float32) - np.array(tcp[:3], dtype=np.float32)
        dist = float(np.linalg.norm(err))
        if dist < 0.005:
            return RLStep(0.0, 0.0, 0.0, "done")

        if self._bc_W is not None and self._bc_b is not None:
            obs = self._build_obs(state, target_xyz).astype(np.float32)
            pred = obs @ self._bc_W + self._bc_b
            delta = clip_action_to_step(pred, max_step_m)
            return RLStep(float(delta[0]), float(delta[1]), float(delta[2]), "bc")

        if self._model is not None:
            obs = self._build_obs(state, target_xyz)
            act, _ = self._model.predict(obs, deterministic=True)
            a = np.clip(np.asarray(act, dtype=np.float32), -1.0, 1.0)
            delta = a[:3] * float(max_step_m)
            return RLStep(float(delta[0]), float(delta[1]), float(delta[2]), "rl")

        k = float(max_step_m) / max(dist, 1e-6)
        d = err * min(1.0, k)
        return RLStep(float(d[0]), float(d[1]), float(d[2]), "fallback-p")
