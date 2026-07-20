#!/usr/bin/env python3
"""Train PPO/SAC on ReachEnv and save to data/models/policies/<task>_v1/."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.envs.reach_env import ReachEnv


def main():
    p = argparse.ArgumentParser(description="Train reach RL baseline (state-only)")
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--task-id", default="reach_free_space")
    p.add_argument("--version", default="v1")
    p.add_argument("--out-dir", default="../data/models/policies")
    args = p.parse_args()

    try:
        from stable_baselines3 import PPO, SAC
    except Exception as e:
        raise SystemExit(
            "stable-baselines3 is required. Install with: pip install stable-baselines3 gymnasium"
        ) from e

    env = ReachEnv()
    if args.algo == "ppo":
        model = PPO("MlpPolicy", env, verbose=1, n_steps=1024, batch_size=256)
    else:
        model = SAC("MlpPolicy", env, verbose=1, batch_size=256)

    model.learn(total_timesteps=int(args.timesteps))

    run_name = f"{args.task_id}_{args.version}"
    out = Path(args.out_dir).resolve() / run_name
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "policy.zip"
    model.save(str(ckpt))
    print(f"Saved: {ckpt}")


if __name__ == "__main__":
    main()
