#!/usr/bin/env python3
"""Train a camera-first RL baseline policy (simulation scaffold)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.camera_reach_env import CameraReachEnv


def main():
    p = argparse.ArgumentParser(description="Train camera-first RL policy (PPO baseline)")
    p.add_argument("--timesteps", type=int, default=120_000)
    p.add_argument("--run-name", default="camera_reach_v1")
    p.add_argument("--out-dir", default="../data/models/policies")
    args = p.parse_args()

    try:
        from stable_baselines3 import PPO
    except Exception as e:
        raise SystemExit(
            "stable-baselines3 is required. Install with: pip install stable-baselines3 gymnasium"
        ) from e

    out = Path(args.out_dir).resolve() / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    env = CameraReachEnv()
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
    )
    model.learn(total_timesteps=int(args.timesteps))
    ckpt = out / "policy.zip"
    model.save(str(ckpt))
    print(f"Saved policy checkpoint: {ckpt}")
    print("For runtime in execute_rl_policy, also write optional JSON gains file if needed.")


if __name__ == "__main__":
    main()
