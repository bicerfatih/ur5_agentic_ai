#!/usr/bin/env python3
"""Evaluate reach policy checkpoint in ReachEnv."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.envs.reach_env import ReachEnv


def main():
    p = argparse.ArgumentParser(description="Evaluate reach RL policy")
    p.add_argument("--policy", required=True, help="Path to policy.zip")
    p.add_argument("--episodes", type=int, default=30)
    args = p.parse_args()

    try:
        from stable_baselines3 import PPO, SAC
    except Exception as e:
        raise SystemExit(
            "stable-baselines3 is required. Install with: pip install stable-baselines3 gymnasium"
        ) from e

    policy = Path(args.policy).resolve()
    if not policy.is_file():
        raise SystemExit(f"Policy not found: {policy}")

    # Attempt PPO first, then SAC.
    model = None
    for cls in (PPO, SAC):
        try:
            model = cls.load(str(policy))
            break
        except Exception:
            continue
    if model is None:
        raise SystemExit("Could not load policy as PPO or SAC checkpoint.")

    env = ReachEnv()
    success = 0
    avg_dist = 0.0
    for _ in range(max(1, int(args.episodes))):
        obs, info = env.reset()
        done = False
        truncated = False
        while not done and not truncated:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(act)
        avg_dist += float(info.get("distance", 0.0))
        if done:
            success += 1
    n = max(1, int(args.episodes))
    print(f"Success: {success}/{n} ({(success / n) * 100:.1f}%)")
    print(f"Final distance mean: {avg_dist / n:.4f} m")


if __name__ == "__main__":
    main()
