#!/usr/bin/env python3
"""Diagnose gripper IO, dashboard play, and RTDE control."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    ALLOWED_URP_PROGRAMS,
    GRIPPER_CMD_PIN,
    GRIPPER_CMD_TARGET,
    GRIPPER_FEEDBACK_IN_CLOSED,
    GRIPPER_FEEDBACK_IN_OPEN,
    ROBOT_HOST,
)
from robot.dashboard import URDashboardClient


def main():
    host = ROBOT_HOST
    print(f"=== Diagnose UR5 @ {host} ===\n")

    print("--- Dashboard ---")
    try:
        d = URDashboardClient(host)
        d.connect()
        print("robotmode:", d.robot_mode())
        print("safetymode:", d.safety_mode())
        print("programState:", d.program_state())
        print("running:", d.running())
        print("Agent whitelist:", ALLOWED_URP_PROGRAMS)
        print("\nTrying load fly2.urp + play...")
        d.stop()
        prep = d.prepare_to_play()
        print("prepare:", prep)
        print("load:", d.load_program("fly2.urp"))
        play_resp = d.play()
        print("play:", play_resp)
        print("programState after:", d.program_state())
        print("running after:", d.running())
        if play_resp and "Failed" in play_resp:
            print(
                "\nIf play failed: enable Remote Control on pendant (Settings), "
                "stop External Control program, press PLAY manually on fly2.urp."
            )
        d.disconnect()
    except Exception as e:
        print("Dashboard error:", e)

    print("\n--- Gripper config ---")
    print(
        f"GRIPPER_CMD_TARGET={GRIPPER_CMD_TARGET} GRIPPER_CMD_PIN={GRIPPER_CMD_PIN} "
        f"feedback_in={GRIPPER_FEEDBACK_IN_OPEN},{GRIPPER_FEEDBACK_IN_CLOSED} (read only)"
    )
    print("Test outputs: python3 scripts/test_gripper_outputs.py")
    print("Run: python3 -c \"from robot.ur5_driver import UR5Driver; r=UR5Driver(); r.connect(); r.gripper_open(); input('watch jaw'); r.gripper_close(); r.disconnect()\"")

    print("\n--- RTDE control ---")
    try:
        import rtde_control

        c = rtde_control.RTDEControlInterface(host)
        print("RTDE control: connected OK")
        c.disconnect()
    except Exception as e:
        print("RTDE control:", e)


if __name__ == "__main__":
    main()
