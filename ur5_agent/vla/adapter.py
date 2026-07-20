"""Vision-Language-Action adapter — image + instruction → Cartesian delta."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from config.settings import (
    VLA_ACTION_SCALE_M,
    VLA_BACKEND,
    VLA_INSTRUCTION_DEFAULT,
    VLA_MAX_STEP_M,
    VLA_SERVER_TIMEOUT_S,
    VLA_SERVER_URL,
    VLA_UNNORM_KEY,
    VLA_WIRE_FORMAT,
)
from il.obs_action import build_reach_obs, clip_action_to_step
from sim.obs_contract import ObsMode, stack_policy_inputs

REMOTE_BACKENDS = ("openvla", "pi0", "groot")


class VLAPolicyAdapter:
    """
    Modes:
      - tool_routed: parse instruction keywords, delegate to 3D target + proportional step
      - disabled: returns zero action
      - openvla | pi0 | groot: remote inference server (set VLA_SERVER_URL);
        falls back to a proportional stub when no server is configured.
    """

    def __init__(self, backend: str = "", model_path: str = "", server_url: str = ""):
        self.backend = (backend or VLA_BACKEND or "tool_routed").strip().lower()
        self.model_path = (model_path or "").strip()
        self.server_url = (server_url or VLA_SERVER_URL or "").strip()
        self._remote = None
        self._remote_error = ""
        if self.backend in REMOTE_BACKENDS and self.server_url:
            try:
                from vla.remote_client import RemoteVLAClient

                self._remote = RemoteVLAClient(
                    server_url=self.server_url,
                    backend=self.backend,
                    wire_format=VLA_WIRE_FORMAT,
                    timeout_s=VLA_SERVER_TIMEOUT_S,
                    unnorm_key=VLA_UNNORM_KEY,
                )
            except Exception as e:
                self._remote_error = str(e)

    def step(
        self,
        rgb: np.ndarray,
        instruction: str,
        state: dict[str, Any],
        target_xyz: list[float],
        max_step_m: float | None = None,
    ) -> dict[str, Any]:
        cap = float(max_step_m if max_step_m is not None else VLA_MAX_STEP_M)
        instr = (instruction or VLA_INSTRUCTION_DEFAULT).strip()
        packed = stack_policy_inputs(ObsMode.VLA, state, target_xyz, rgb=rgb, instruction=instr)

        if self.backend == "disabled":
            return {"dx": 0.0, "dy": 0.0, "dz": 0.0, "source": "vla-disabled", "instruction": instr}

        if self.backend in REMOTE_BACKENDS:
            if self._remote is not None:
                return self._step_remote(packed, instr, cap)
            return self._step_proportional_stub(packed, cap, note=self._stub_note())

        return self._step_tool_routed(state, target_xyz, instr, cap)

    def _stub_note(self) -> str:
        if self._remote_error:
            return f"remote client init failed: {self._remote_error}"
        return f"VLA_SERVER_URL not set — using proportional stub for backend '{self.backend}'"

    def _step_remote(self, packed: dict[str, Any], instruction: str, max_step_m: float) -> dict[str, Any]:
        """Query GPU-hosted VLA (OpenVLA / pi0 / GR00T), clip to safe Cartesian step."""
        from vla.remote_client import RemoteVLAError

        rgb = packed.get("rgb")
        state12 = packed.get("state")
        if rgb is None:
            return {"status": "error", "reason": "missing rgb frame for remote VLA"}
        try:
            out = self._remote.act(rgb, instruction, state12=state12)
        except RemoteVLAError as e:
            return {"status": "error", "reason": f"remote VLA failed: {e}", "source": f"{self.backend}-remote"}
        except Exception as e:
            return {"status": "error", "reason": f"remote VLA failed: {e}", "source": f"{self.backend}-remote"}

        action = np.asarray(out["action"], dtype=np.float32).reshape(-1)
        xyz = action[:3]
        # Normalized actions ([-1, 1]) get scaled to meters; metric ones pass through.
        if float(np.max(np.abs(xyz))) <= 1.0 + 1e-6 and VLA_ACTION_SCALE_M > 0:
            xyz = xyz * VLA_ACTION_SCALE_M
        delta = clip_action_to_step(xyz, max_step_m)

        result: dict[str, Any] = {
            "dx": float(delta[0]),
            "dy": float(delta[1]),
            "dz": float(delta[2]),
            "source": f"{self.backend}-remote",
            "instruction": instruction,
            "raw_action": [round(float(v), 5) for v in action.tolist()],
        }
        # Extra dims (rotation / gripper) surfaced for future use, not executed here.
        if action.size >= 7:
            result["gripper_cmd"] = float(action[6])
        raw = out.get("raw")
        if isinstance(raw, dict) and raw.get("done"):
            result["done"] = True
        return result

    def _step_tool_routed(
        self,
        state: dict[str, Any],
        target_xyz: list[float],
        instruction: str,
        max_step_m: float,
    ) -> dict[str, Any]:
        """Keyword-routed baseline until a real VLA checkpoint is wired."""
        tcp = state.get("tcp_pose") or [0.0, 0.0, 0.0]
        err = np.array(target_xyz[:3], dtype=np.float32) - np.array(tcp[:3], dtype=np.float32)
        dist = float(np.linalg.norm(err))
        if dist < 0.008:
            return {"dx": 0.0, "dy": 0.0, "dz": 0.0, "source": "vla-done", "instruction": instruction, "done": True}

        low = instruction.lower()
        scale = 1.0
        if any(w in low for w in ("slow", "careful", "gentle")):
            scale = 0.5
        if any(w in low for w in ("fast", "quick")):
            scale = 1.2

        k = max_step_m * scale / max(dist, 1e-6)
        delta = err * min(1.0, k)
        delta = clip_action_to_step(delta, max_step_m * scale)
        return {
            "dx": float(delta[0]),
            "dy": float(delta[1]),
            "dz": float(delta[2]),
            "source": "vla-tool-routed",
            "instruction": instruction,
            "dist_m": round(dist, 5),
        }

    def _step_proportional_stub(self, packed: dict[str, Any], max_step_m: float, note: str = "") -> dict[str, Any]:
        """Proportional fallback when a remote backend is selected but unreachable."""
        state12 = packed.get("state")
        if state12 is None:
            return {"status": "error", "reason": "missing state in VLA pack"}
        obs = np.asarray(state12, dtype=np.float32)
        tcp = obs[6:9]
        tgt = obs[9:12]
        err = tgt - tcp
        dist = float(np.linalg.norm(err))
        if dist < 0.008:
            return {"dx": 0.0, "dy": 0.0, "dz": 0.0, "source": f"{self.backend}-stub-done", "done": True}
        delta = clip_action_to_step(err * 0.25, max_step_m)
        return {
            "dx": float(delta[0]),
            "dy": float(delta[1]),
            "dz": float(delta[2]),
            "source": f"{self.backend}-stub",
            "note": note or "Set VLA_SERVER_URL to use real VLA inference.",
        }
