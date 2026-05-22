#!/usr/bin/env python3
# main.py — Physical + agentic AI (Ollama default, Claude optional)

import argparse
import os
import sys

from agent.factory import create_agent
from config.settings import LLM_BACKEND, OLLAMA_MODEL, ROBOT_HOST
from config.sites import get_site
from robot.factory import create_robot


def parse_args():
    p = argparse.ArgumentParser(
        description="Agentic physical AI — UR5 + local Ollama (default)"
    )
    p.add_argument(
        "--robot",
        choices=["ur5", "openarm"],
        default=os.environ.get("ROBOT_TYPE", "ur5"),
        help="Arm driver (default: ur5)",
    )
    p.add_argument(
        "--site",
        choices=["lab", "airport_ground", "airport_cargo"],
        default=os.environ.get("SITE_ID", "lab"),
        help="Deployment profile and safety policy",
    )
    p.add_argument(
        "--llm",
        choices=["ollama", "claude"],
        default=os.environ.get("LLM_BACKEND", LLM_BACKEND),
        help="LLM backend (default: ollama)",
    )
    p.add_argument(
        "--model",
        default=None,
        help=f"Ollama model tag (default: {OLLAMA_MODEL})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate motion without connecting to hardware",
    )
    p.add_argument("--host", default=ROBOT_HOST, help="Robot controller IP (UR5)")
    return p.parse_args()


BANNER = """
╔══════════════════════════════════════════════════════════╗
║     Physical + Agentic AI — UR5 + Ollama (local LLM)     ║
╚══════════════════════════════════════════════════════════╝
"""


def main():
    args = parse_args()
    site = get_site(args.site)
    model_tag = args.model or OLLAMA_MODEL

    print(BANNER)
    print(f"  Robot:   {args.robot}" + (" (DRY-RUN)" if args.dry_run else ""))
    print(f"  Site:    {site.display_name}")
    if args.llm == "ollama":
        print(f"  LLM:     ollama ({model_tag})")
    else:
        print(f"  LLM:     claude")
    print(f"  Host:    {args.host if not args.dry_run else 'n/a'}")
    print()

    robot = create_robot(robot_type=args.robot, dry_run=args.dry_run, host=args.host)
    try:
        robot.connect()
    except (ConnectionError, ValueError) as e:
        print(f"❌ {e}")
        if not args.dry_run and args.robot == "ur5":
            print(f"Check UR5 is on and reachable at {args.host}")
        sys.exit(1)

    try:
        agent = create_agent(
            robot,
            site,
            llm=args.llm,
            ollama_model=args.model,
        )
    except Exception as e:
        print(f"❌ {e}")
        if args.llm == "ollama":
            print("\nSetup:")
            print("  curl -fsSL https://ollama.com/install.sh | sh")
            print("  ollama serve")
            print(f"  ollama pull {model_tag}")
            print("  python3 scripts/check_ollama.py")
        sys.exit(1)

    print("Ready. Commands: natural language goals, 'state', 'quit'\n")
    print("Examples:")
    print("  → read state then move up 2 centimeters")
    print("  → go to home position")
    print("  → open the gripper")
    print("  → run program fly2.urp\n")

    try:
        while True:
            try:
                goal = input("You: ").strip()
            except EOFError:
                break

            if not goal:
                continue
            if goal.lower() in ("quit", "exit", "q"):
                print("Shutting down...")
                break

            if goal.lower() == "state":
                for k, v in robot.get_full_state().items():
                    print(f"  {k}: {v}")
                continue

            try:
                agent.run(goal)
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted — stopping robot.")
                robot.stop()
            except Exception as e:
                print(f"❌ Error: {e}")

    finally:
        robot.disconnect()
        print("Goodbye!")


if __name__ == "__main__":
    main()
