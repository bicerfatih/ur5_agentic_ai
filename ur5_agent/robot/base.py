# robot/base.py — arm-agnostic driver contract (UR5 today, OpenArm at airports)

from abc import ABC, abstractmethod
from typing import Any


class RobotDriver(ABC):
    """
    Physical robot API shared by all arms.
    Agent tools call only this interface — never RTDE or vendor SDKs directly.
    """

    @property
    @abstractmethod
    def arm_model(self) -> str:
        """e.g. ur5, openarm"""

    @property
    @abstractmethod
    def is_simulated(self) -> bool:
        """True when no real hardware is commanded."""

    @abstractmethod
    def connect(self) -> None:
        ...

    @abstractmethod
    def disconnect(self) -> None:
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        ...

    # ── State ──────────────────────────────────────────

    @abstractmethod
    def get_joint_positions(self) -> list[float]:
        ...

    @abstractmethod
    def get_tcp_pose(self) -> list[float]:
        ...

    @abstractmethod
    def get_tcp_force(self) -> list[float]:
        ...

    @abstractmethod
    def get_robot_mode(self) -> int:
        ...

    @abstractmethod
    def get_safety_mode(self) -> int:
        ...

    def get_full_state(self) -> dict[str, Any]:
        state = {
            "arm_model": self.arm_model,
            "simulated": self.is_simulated,
            "joint_positions_rad": self.get_joint_positions(),
            "joint_positions_deg": None,  # filled by subclass if needed
            "tcp_pose": self.get_tcp_pose(),
            "tcp_force": self.get_tcp_force(),
            "robot_mode": self.get_robot_mode(),
            "safety_mode": self.get_safety_mode(),
        }
        if hasattr(self, "get_gripper_state"):
            state["gripper"] = self.get_gripper_state()
        if hasattr(self, "get_program_state"):
            state["urp_program"] = self.get_program_state()
        return state

    # ── Motion ─────────────────────────────────────────

    @abstractmethod
    def move_joint(self, joints: list[float], speed: float = 0.3, accel: float = 0.3) -> None:
        ...

    @abstractmethod
    def move_linear(self, tcp_pose: list[float], speed: float = 0.1, accel: float = 0.1) -> None:
        ...

    @abstractmethod
    def move_home(self) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @abstractmethod
    def move_tcp_relative(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        speed: float = 0.1,
        accel: float = 0.1,
    ) -> None:
        ...

    # ── Gripper & PolyScope programs (optional per driver) ─

    def gripper_open(self) -> None:
        raise NotImplementedError(f"{self.arm_model} driver has no gripper configured")

    def gripper_close(self) -> None:
        raise NotImplementedError(f"{self.arm_model} driver has no gripper configured")

    def run_urp_program(self, program_name: str) -> dict:
        raise NotImplementedError(f"{self.arm_model} driver cannot run .urp programs")

    def stop_urp_program(self) -> dict:
        raise NotImplementedError(f"{self.arm_model} driver cannot stop .urp programs")
