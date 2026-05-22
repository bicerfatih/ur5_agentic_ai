#!/usr/bin/env python3
"""Run one agent goal non-interactively (for testing)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.factory import create_agent
from config.sites import get_site
from robot.factory import create_robot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("goal", nargs="?", default="read the robot state and report joint angles in degrees")
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--live", action="store_true", help="Connect to real UR5")
    p.add_argument("--site", default="lab")
    p.add_argument("--llm", default="ollama", choices=["ollama", "claude"])
    p.add_argument("--model", default=None)
    args = p.parse_args()

    dry_run = not args.live
    site = get_site(args.site)

    print(f"Test: dry_run={dry_run} site={args.site} llm={args.llm}")
    print(f"Goal: {args.goal}\n")

    robot = create_robot(dry_run=dry_run)
    robot.connect()

    agent = create_agent(robot, site, llm=args.llm, ollama_model=args.model)
    agent.run(args.goal)
    robot.disconnect()
    print("\nTest finished.")


if __name__ == "__main__":
    main()
