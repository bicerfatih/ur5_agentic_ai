#!/usr/bin/env python3
"""Test digital OUTPUT pins only (not feedback inputs 2/3)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import rtde_io
except ModuleNotFoundError:
    print("Run: source robot_env/bin/activate")
    sys.exit(1)

from config.settings import ROBOT_HOST


def set_out(io, target, pin, val):
    if target == "tool":
        return io.setToolDigitalOut(pin, val)
    if target == "standard":
        return io.setStandardDigitalOut(pin, val)
    return io.setConfigurableDigitalOut(pin, val)


def main():
    print("Testing OUTPUT pins (your robot: standard DO 0/1 sinking NPN; DI 2/3 = feedback)\n")
    io = rtde_io.RTDEIOInterface(ROBOT_HOST)
    for target in ("standard", "tool", "configurable"):
        print(f"=== {target} ===")
        for pin in (0, 1):
            set_out(io, target, pin, False)
            set_out(io, target, pin, True)
            input(f"  {target} DO{pin} HIGH — jaw moved? Enter...")
            set_out(io, target, pin, False)
    io.disconnect()
    print("\nSet env for the pin that worked, e.g.:")
    print("  export GRIPPER_CMD_TARGET=standard")
    print("  export GRIPPER_CMD_PIN=0")


if __name__ == "__main__":
    main()
