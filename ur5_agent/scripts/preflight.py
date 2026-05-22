#!/usr/bin/env python3
"""
Pre-flight checks before live agentic control.
Run: python3 scripts/preflight.py [--host 192.168.0.160]
"""

import argparse
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import ROBOT_HOST

ROBOT_MODE_RUNNING = 7
SAFETY_MODE_NORMAL = 1


def main():
    p = argparse.ArgumentParser(description="UR5 preflight (read-only RTDE)")
    p.add_argument("--host", default=ROBOT_HOST)
    args = p.parse_args()

    print("UR5 preflight")
    print(f"  Host: {args.host}\n")

    try:
        import rtde_receive
    except ImportError:
        print("FAIL: ur-rtde not installed. Run: bash setup.sh && source robot_env/bin/activate")
        sys.exit(1)

    try:
        r = rtde_receive.RTDEReceiveInterface(args.host)
    except Exception as e:
        print(f"FAIL: Cannot open RTDE receive on {args.host}: {e}")
        print("  → Robot powered? Remote control enabled? Same subnet?")
        sys.exit(1)

    mode = r.getRobotMode()
    safety = r.getSafetyMode()
    joints = [round(math.degrees(v), 2) for v in r.getActualQ()]
    tcp = [round(v, 4) for v in r.getActualTCPPose()]

    print("  robot_mode:", mode, "OK" if mode == ROBOT_MODE_RUNNING else "NOT RUNNING (need mode 7)")
    print("  safety_mode:", safety, "OK" if safety == SAFETY_MODE_NORMAL else "NOT NORMAL (need mode 1)")
    print("  joints_deg:", joints)
    print("  tcp_pose:", tcp)

    r.disconnect()

    if mode != ROBOT_MODE_RUNNING or safety != SAFETY_MODE_NORMAL:
        print("\nWARN: Fix robot state on teach pendant before agentic moves.")
        sys.exit(2)

    print("\nPASS: Robot is reachable and in RUNNING/NORMAL. Safe to proceed to dry-run then live agent.")
    sys.exit(0)


if __name__ == "__main__":
    main()
