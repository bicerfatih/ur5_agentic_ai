#!/usr/bin/env python3
"""Run execute_rl_policy once via tool dispatcher (dry-run or live)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.sites import get_site
from policy.safety import PolicyEngine
from robot.factory import create_robot
from robot.tools import execute_tool


def _parse_target(raw: str) -> list[float] | None:
    txt = (raw or "").strip()
    if not txt:
        return None
    parts = [p.strip() for p in txt.split(",")]
    if len(parts) != 3:
        raise ValueError("--target must be 'x,y,z' in meters")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def main():
    p = argparse.ArgumentParser(description="Execute RL policy tool once.")
    p.add_argument("--live", action="store_true", help="Use live UR5 instead of dry-run mock")
    p.add_argument("--site", default="lab", choices=["lab", "airport_ground", "airport_cargo"])
    p.add_argument("--task-id", default="reach_free_space")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--max-step-m", type=float, default=0.01)
    p.add_argument("--policy-path", default="")
    p.add_argument("--target", default="", help="Optional target TCP xyz, e.g. 0.35,0.00,0.32")
    p.add_argument("--target-label", default="", help="Optional camera_reach label filter")
    args = p.parse_args()

    target_tcp = _parse_target(args.target)
    robot = create_robot(dry_run=not args.live)
    site = get_site(args.site)
    policy = PolicyEngine(site=site)

    robot.connect()
    try:
        inputs = {
            "task_id": args.task_id,
            "steps": args.steps,
            "max_step_m": args.max_step_m,
        }
        if args.policy_path:
            inputs["policy_path"] = args.policy_path
        if target_tcp is not None:
            inputs["target_tcp"] = target_tcp
        if args.target_label:
            inputs["target_label"] = args.target_label

        result = execute_tool("execute_rl_policy", inputs=inputs, robot=robot, policy=policy, caller="ui")
        print(result)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
