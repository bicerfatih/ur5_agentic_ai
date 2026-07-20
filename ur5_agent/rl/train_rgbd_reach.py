#!/usr/bin/env python3
"""Train RGB-D reach policy on local sim env (prototype before Isaac)."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from gymnasium import spaces

from sim.local_rgbd_reach_env import LocalRgbdReachEnv


class DictFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        img_shape = observation_space["image"].shape
        n_img = int(img_shape[0] * img_shape[1] * img_shape[2])
        self.net = nn.Sequential(
            nn.Linear(n_img + 12, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations) -> torch.Tensor:
        img = observations["image"].float().reshape(observations["image"].shape[0], -1) / 255.0
        st = observations["state"].float()
        x = torch.cat([img, st], dim=1)
        return self.net(x)


def main():
    p = argparse.ArgumentParser(description="Train RGB-D reach (local sim; Isaac bridge next).")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--task-id", default="reach_rgbd")
    p.add_argument("--version", default="v1")
    p.add_argument("--algo", default="ppo", choices=["ppo"])
    args = p.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(here, f"../data/models/policies/{args.task_id}_{args.version}")
    os.makedirs(out_dir, exist_ok=True)

    def _make():
        return LocalRgbdReachEnv()

    env = make_vec_env(_make, n_envs=1)
    policy_kwargs = dict(
        features_extractor_class=DictFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    model = PPO("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=args.timesteps)
    policy_path = os.path.join(out_dir, "policy.zip")
    model.save(policy_path)
    print(f"saved {policy_path}")


if __name__ == "__main__":
    main()
