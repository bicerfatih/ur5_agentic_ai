#!/usr/bin/env python3
"""Evaluate a camera-first RL policy checkpoint in simulation scaffold."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.camera_reach_env import CameraReachEnv


def main():
    p = argparse.ArgumentParser(description="Evaluate camera RL policy")
    p.add_argument("--policy", required=True, help="Path to stable-baselines3 zip")
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()

    try:
        from stable_baselines3 import PPO
    except Exception as e:
        raise SystemExit(
            "stable-baselines3 is required. Install with: pip install stable-baselines3 gymnasium"
        ) from e

    policy_path = Path(args.policy).resolve()
    if not policy_path.is_file():
        raise SystemExit(f"Policy not found: {policy_path}")

    env = CameraReachEnv()
    model = PPO.load(str(policy_path))
    success = 0
    for _ in range(max(1, int(args.episodes))):
        obs, _ = env.reset()
        done = False
        truncated = False
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
        if done:
            success += 1
    rate = success / max(1, int(args.episodes))
    print(f"Success: {success}/{args.episodes} ({rate * 100:.1f}%)")


if __name__ == "__main__":
    main()
