# robot/mock_driver.py — dry-run / dev without hardware

import math
import time
from typing import Any

from robot.base import RobotDriver

# Default simulated home (matches UR5 home in degrees → rad)
_DEFAULT_JOINTS = [0.0, -1.5707, 0.0, -1.5707, 0.0, 0.0]
_DEFAULT_TCP = [0.3, 0.0, 0.4, 0.0, 3.14159, 0.0]


class MockDriver(RobotDriver):
    """Simulates arm state and motion for policy/agent testing without a robot."""

    def __init__(self, arm_model: str = "ur5", label: str = "mock"):
        self._arm_model = arm_model
        self._label = label
        self._connected = False
        self._joints = list(_DEFAULT_JOINTS)
        self._tcp = list(_DEFAULT_TCP)
        self._robot_mode = 7
        self._safety_mode = 1
        self._motion_log: list[dict[str, Any]] = []
        self._gripper_state = "open"
        self._loaded_program: str | None = None
        self._program_running = False

    @property
    def arm_model(self) -> str:
        return self._arm_model

    @property
    def is_simulated(self) -> bool:
        return True

    @property
    def motion_log(self) -> list[dict[str, Any]]:
        return self._motion_log

    def connect(self):
        self._connected = True
        print(f"[DRY-RUN] Mock driver ready ({self._arm_model}, {self._label}).\n")

    def disconnect(self):
        self._connected = False
        print("[DRY-RUN] Mock driver disconnected.")

    def is_connected(self) -> bool:
        return self._connected

    def _record(self, action: str, **kwargs):
        entry = {"action": action, "ts": time.time(), **kwargs}
        self._motion_log.append(entry)
        print(f"  [DRY-RUN] {action} {kwargs}")

    def get_joint_positions(self) -> list:
        return [round(v, 4) for v in self._joints]

    def get_tcp_pose(self) -> list:
        return [round(v, 4) for v in self._tcp]

    def get_tcp_force(self) -> list:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def get_robot_mode(self) -> int:
        return self._robot_mode

    def get_safety_mode(self) -> int:
        return self._safety_mode

    def get_full_state(self) -> dict:
        state = super().get_full_state()
        state["joint_positions_deg"] = [round(math.degrees(v), 2) for v in self._joints]
        state["dry_run_label"] = self._label
        state["motion_count"] = len(self._motion_log)
        return state

    def move_joint(self, joints: list, speed: float = 0.3, accel: float = 0.3):
        self._record("move_joint", joints=joints, speed=speed, accel=accel)
        self._joints = list(joints)

    def move_linear(self, tcp_pose: list, speed: float = 0.1, accel: float = 0.1):
        self._record("move_linear", tcp_pose=tcp_pose, speed=speed, accel=accel)
        self._tcp = list(tcp_pose)

    def move_home(self):
        self._record("move_home")
        self._joints = list(_DEFAULT_JOINTS)
        self._tcp = list(_DEFAULT_TCP)

    def stop(self):
        self._record("stop")

    def move_tcp_relative(
        self, dx=0.0, dy=0.0, dz=0.0, speed: float = 0.1, accel: float = 0.1
    ):
        self._record("move_tcp_relative", dx=dx, dy=dy, dz=dz, speed=speed, accel=accel)
        self._tcp[0] += dx
        self._tcp[1] += dy
        self._tcp[2] += dz

    def gripper_open(self):
        self._record("gripper_open")
        self._gripper_state = "open"

    def gripper_close(self):
        self._record("gripper_close")
        self._gripper_state = "closed"

    def get_gripper_state(self) -> dict:
        return {
            "command_state": self._gripper_state,
            "open_pin": 2,
            "close_pin": 3,
            "simulated": True,
        }

    def get_program_state(self) -> dict:
        return {
            "loaded": self._loaded_program,
            "running": self._program_running,
            "simulated": True,
        }

    def run_urp_program(self, program_name: str) -> dict:
        prog = program_name if program_name.endswith(".urp") else f"{program_name}.urp"
        self._record("run_urp_program", program=prog)
        self._loaded_program = prog
        self._program_running = True
        return {
            "status": "done",
            "program": prog,
            "load": "simulated ok",
            "play": "simulated ok",
            "running": True,
        }

    def stop_urp_program(self) -> dict:
        self._record("stop_urp_program")
        self._program_running = False
        return {"status": "stopped", "running": False}

    def release_rtde_control(self) -> dict:
        self._record("release_rtde_control")
        return {"status": "done", "notes": ["simulated"]}

    def reconnect_rtde_control(self) -> dict:
        self._record("reconnect_rtde_control")
        return {"status": "done", "message": "simulated"}
