#!/usr/bin/env python3
"""
Verify gripper I/O on the controller (read-only receive + IO set).

Run with robot powered and reachable. Watch PolyScope → I/O Tools while this runs.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import rtde_io
    import rtde_receive
except ModuleNotFoundError:
    print("Run: source robot_env/bin/activate")
    sys.exit(1)

from config.settings import ROBOT_HOST


def main():
    host = ROBOT_HOST
    print(f"Gripper pendant check @ {host}\n")
    print("WATCH PolyScope → I/O Tools → Standard Output DO 0 / DO 1\n")

    r = rtde_receive.RTDEReceiveInterface(host)
    io = rtde_io.RTDEIOInterface(host)

    def show(label):
        print(
            f"  {label}: DO0={r.getDigitalOutState(0)} DO1={r.getDigitalOutState(1)} "
            f"DI2={r.getDigitalInState(2)} DI3={r.getDigitalInState(3)}"
        )

    show("Before")
    input("Press Enter → set OPEN (DO0=ON, DO1=OFF)...")
    io.setStandardDigitalOut(0, True)
    io.setStandardDigitalOut(1, False)
    time.sleep(0.3)
    show("OPEN command")
    moved_open = input("Did DO0/DO1 toggle on pendant? Did JAW move? (y/n): ").strip().lower()

    input("Press Enter → set CLOSE (DO0=OFF, DO1=ON)...")
    io.setStandardDigitalOut(0, False)
    io.setStandardDigitalOut(1, True)
    time.sleep(0.3)
    show("CLOSE command")
    moved_close = input("Did DO0/DO1 toggle? Did JAW move? (y/n): ").strip().lower()

    io.setStandardDigitalOut(0, False)
    io.setStandardDigitalOut(1, False)
    io.disconnect()
    r.disconnect()

    print("\n=== Result ===")
    if moved_open == "y" or moved_close == "y":
        print("Gripper responds to Standard DO 0/1. Agent I/O path is OK.")
        return

    if moved_open != "y" and moved_close != "y":
        print("Software sets DO0/DO1 on controller (see True/False above).")
        print("If pendant LEDs TOGGLE but jaw does NOT move:")
        print("  → Compressed air, solenoid valves, or gripper wired to different outputs.")
        print("If pendant LEDs do NOT toggle:")
        print("  → PolyScope installation may not map I/O Tools labels to standard DO 0/1.")
        print("Try on pendant: PLAY fly2.urp — if gripper works there, use run_urp_program.")
        print("Try pulse: export GRIPPER_PULSE_MS=500 && re-test open_gripper.")


if __name__ == "__main__":
    main()
