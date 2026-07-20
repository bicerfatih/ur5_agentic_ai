#!/usr/bin/env python3
"""Train reach policy via Isaac bridge (or fall back to local env)."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import ISAAC_BRIDGE_HOST, ISAAC_BRIDGE_PORT, SIM_BACKEND
from sim.bridge import IsaacBridgeClient
from sim.local_rgbd_reach_env import LocalRgbdReachEnv


def _train_local(timesteps: int, out_path: str) -> None:
    from stable_baselines3 import PPO

    env = LocalRgbdReachEnv()
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(out_path)
    print(f"saved {out_path}")


def _train_isaac_bridge(timesteps: int, out_path: str, host: str, port: int) -> None:
    """
    Minimal online training loop over bridge state-only obs.
    Upgrade to RGB-D policy once Isaac returns images over bridge.
    """
    from stable_baselines3 import PPO
    from gymnasium import Env, spaces

    class BridgeStateEnv(Env):
        def __init__(self):
            super().__init__()
            self.client = IsaacBridgeClient(host=host, port=port)
            self.client.connect()
            self.observation_space = spaces.Box(-10, 10, (12,), dtype=np.float32)
            self.action_space = spaces.Box(-1, 1, (3,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            obs, _ = self.client.reset()
            return np.asarray(obs["state"], dtype=np.float32), {}

        def step(self, action):
            obs, reward, done, truncated, info = self.client.step(action.tolist())
            return np.asarray(obs["state"], dtype=np.float32), reward, done, truncated, info

        def close(self):
            self.client.close()

    env = BridgeStateEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(out_path)
    env.close()
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--task-id", default="reach_isaac")
    p.add_argument("--version", default="v1")
    p.add_argument("--backend", default=SIM_BACKEND, choices=["local", "isaac"])
    p.add_argument("--host", default=ISAAC_BRIDGE_HOST)
    p.add_argument("--port", type=int, default=ISAAC_BRIDGE_PORT)
    args = p.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(here, f"../data/models/policies/{args.task_id}_{args.version}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "policy.zip")

    if args.backend == "isaac":
        _train_isaac_bridge(args.timesteps, out_path, args.host, args.port)
    else:
        _train_local(args.timesteps, out_path)


if __name__ == "__main__":
    main()
