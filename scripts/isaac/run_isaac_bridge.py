#!/usr/bin/env python3
"""
Isaac Sim bridge server (run with Isaac's python.sh).

Until lab_cell.usda is built, falls back to LocalRgbdReachEnv so the training
client can be tested without full Isaac physics.

Usage (on GPU machine with Isaac, after adjusting ISAAC_PYTHON):
  $ISAAC_PYTHON scripts/isaac/run_isaac_bridge.py --port 9912

Training client (any machine):
  SIM_BACKEND=isaac python3 ur5_agent/rl/train_isaac_reach.py --timesteps 50000
"""

from __future__ import annotations

import argparse
import json
import os
import socketserver
import sys
from typing import Any

# Repo root on PYTHONPATH when launched from Isaac or dev shell
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
UR5_AGENT = os.path.join(REPO, "ur5_agent")
if UR5_AGENT not in sys.path:
    sys.path.insert(0, UR5_AGENT)

from sim.local_rgbd_reach_env import LocalRgbdReachEnv  # noqa: E402


class _BridgeEnv:
    def __init__(self, use_isaac: bool, usd_path: str):
        self.use_isaac = use_isaac
        self.usd_path = usd_path
        self._env = LocalRgbdReachEnv()
        self._isaac = None
        if use_isaac:
            self._isaac = self._try_load_isaac(usd_path)

    def _try_load_isaac(self, usd_path: str):
        """
        Replace this hook when Isaac Sim stage is ready.
        Expected: load USD, attach RGB-D sensor, step physics, return obs dict.
        """
        if not os.path.isfile(usd_path):
            print(f"[isaac] USD not found ({usd_path}); using LocalRgbdReachEnv fallback.")
            return None
        try:
            # Isaac Sim imports only exist inside NVIDIA's python.sh environment.
            import omni.isaac.core  # type: ignore  # noqa: F401
        except Exception as e:
            print(f"[isaac] Isaac modules unavailable ({e}); using local fallback.")
            return None
        print("[isaac] USD present; wire IsaacReachTask here for full physics.")
        return None

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = self._env.reset()
        return self._pack_obs(obs), info

    def step(self, action: list[float]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        import numpy as np

        a = np.asarray(action, dtype=np.float32)
        obs, reward, done, truncated, info = self._env.step(a)
        return self._pack_obs(obs), float(reward), bool(done), bool(truncated), info

    @staticmethod
    def _pack_obs(obs: dict[str, Any]) -> dict[str, Any]:
        # Send compact obs over bridge (omit full image by default for speed).
        st = obs.get("state")
        return {
            "state": [float(v) for v in st.reshape(-1).tolist()],
            "has_image": True,
            "image_shape": list(obs["image"].shape),
        }


class _Handler(socketserver.StreamRequestHandler):
    env: _BridgeEnv | None = None

    def handle(self) -> None:
        while True:
            line = self.rfile.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                self._reply({"status": "error", "reason": "invalid json"})
                continue
            cmd = msg.get("cmd")
            if cmd == "ping":
                self._reply({"status": "ok"})
            elif cmd == "close":
                self._reply({"status": "ok"})
                break
            elif cmd == "reset":
                obs, info = self.env.reset()  # type: ignore[union-attr]
                self._reply({"obs": obs, "info": info})
            elif cmd == "step":
                act = msg.get("action") or [0, 0, 0]
                obs, reward, done, truncated, info = self.env.step(act)  # type: ignore[union-attr]
                self._reply(
                    {
                        "obs": obs,
                        "reward": reward,
                        "done": done,
                        "truncated": truncated,
                        "info": info,
                    }
                )
            else:
                self._reply({"status": "error", "reason": f"unknown cmd {cmd}"})

    def _reply(self, payload: dict[str, Any]) -> None:
        data = (json.dumps(payload) + "\n").encode("utf-8")
        self.wfile.write(data)
        self.wfile.flush()


def main():
    p = argparse.ArgumentParser(description="Isaac Sim JSON bridge for RL training.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9912)
    p.add_argument("--usd", default=os.path.join(REPO, "assets/usd/lab_cell.usda"))
    p.add_argument("--use-isaac", action="store_true", help="Attempt Isaac physics (needs USD + isaac python)")
    args = p.parse_args()

    bridge_env = _BridgeEnv(use_isaac=args.use_isaac, usd_path=args.usd)
    _Handler.env = bridge_env

    class _Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    with _Server((args.host, args.port), _Handler) as srv:
        print(f"[bridge] listening on {args.host}:{args.port} usd={args.usd}")
        srv.serve_forever()


if __name__ == "__main__":
    main()
