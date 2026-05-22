#!/usr/bin/env python3
"""Test Dashboard load/play for a .urp program (no LLM)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.programs import is_program_allowed
from config.settings import ROBOT_HOST
from robot.dashboard import URDashboardClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("program", nargs="?", default="fly2.urp")
    p.add_argument("--host", default=ROBOT_HOST)
    p.add_argument("--load-only", action="store_true")
    args = p.parse_args()

    if not is_program_allowed(args.program):
        print(f"Not in whitelist. Set ALLOWED_URP_PROGRAMS.")
        sys.exit(1)

    dash = URDashboardClient(args.host)
    print(f"Connecting dashboard {args.host}:29999...")
    dash.connect()
    print("load:", dash.load_program(args.program))
    if not args.load_only:
        print("play:", dash.play())
        print("running:", dash.running())
    dash.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
