#!/usr/bin/env python3
"""Test Robotiq gripper via URCap socket (PolyScope ID 1 → SID 9)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    GRIPPER_POLYSCOPE_ID,
    ROBOT_HOST,
    ROBOTIQ_SOCKET_PORT,
    ROBOTIQ_SOCKET_SID,
)
from robot.robotiq_client import RobotiqGripperClient


def main():
    print(
        f"Robotiq test @ {ROBOT_HOST}:{ROBOTIQ_SOCKET_PORT} "
        f"(PolyScope ID {GRIPPER_POLYSCOPE_ID}, SID {ROBOTIQ_SOCKET_SID})\n"
    )
    g = RobotiqGripperClient(
        host=ROBOT_HOST,
        port=ROBOTIQ_SOCKET_PORT,
        polyscope_id=GRIPPER_POLYSCOPE_ID,
        socket_sid=ROBOTIQ_SOCKET_SID,
    )
    g.connect()
    print("Activating...")
    g.activate()
    print("State:", g.get_state())
    input("Press Enter → OPEN...")
    g.open()
    print("After open:", g.get_state())
    input("Press Enter → CLOSE...")
    g.close()
    print("After close:", g.get_state())
    g.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
