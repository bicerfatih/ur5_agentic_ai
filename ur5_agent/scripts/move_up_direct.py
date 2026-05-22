#!/usr/bin/env python3
"""Direct UR5 move (no LLM) — use when testing hardware quickly."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.ur5_driver import UR5Driver


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cm", type=float, default=2.0, help="Move up in cm")
    p.add_argument("--host", default=None)
    args = p.parse_args()

    robot = UR5Driver(host=args.host) if args.host else UR5Driver()
    robot.connect()
    before = robot.get_tcp_pose()
    print(f"TCP before Z: {before[2]:.4f} m")
    robot.move_tcp_relative(dz=args.cm / 100.0, speed=0.05, accel=0.05)
    after = robot.get_tcp_pose()
    print(f"TCP after  Z: {after[2]:.4f} m (delta {after[2] - before[2]:+.4f} m)")
    robot.disconnect()


if __name__ == "__main__":
    main()
