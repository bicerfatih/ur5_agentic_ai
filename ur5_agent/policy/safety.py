# policy/safety.py — site-aware motion policy (airport / Emirates rules)

from dataclasses import dataclass, field

from config.programs import is_program_allowed
from config.settings import ALLOWED_URP_PROGRAMS
from config.sites import SiteProfile
from robot.base import RobotDriver

MOTION_TOOLS = {
    "move_home",
    "move_joint",
    "move_linear",
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "move_forward",
    "move_backward",
}

# UR robot modes: 7 = RUNNING; safety 1 = NORMAL
ROBOT_MODE_RUNNING = 7
SAFETY_MODE_NORMAL = 1


@dataclass
class PolicyEngine:
    site: SiteProfile
    _state_read_this_goal: bool = field(default=False, init=False)

    def begin_goal(self):
        self._state_read_this_goal = False

    def record_state_read(self):
        self._state_read_this_goal = True

    def check_robot_ready(self, robot: RobotDriver) -> dict | None:
        mode = robot.get_robot_mode()
        safety = robot.get_safety_mode()
        if mode != ROBOT_MODE_RUNNING:
            return {
                "status": "error",
                "reason": f"robot_mode={mode} (expected {ROBOT_MODE_RUNNING} RUNNING). Do not move.",
            }
        if safety != SAFETY_MODE_NORMAL:
            return {
                "status": "error",
                "reason": f"safety_mode={safety} (expected {SAFETY_MODE_NORMAL} NORMAL). Do not move.",
            }
        return None

    def validate_arm(self, robot: RobotDriver) -> dict | None:
        if robot.arm_model not in self.site.allowed_arm_models:
            return {
                "status": "error",
                "reason": (
                    f"arm '{robot.arm_model}' not allowed at site '{self.site.site_id}'. "
                    f"Allowed: {self.site.allowed_arm_models}"
                ),
            }
        return None

    def clamp_speeds(
        self, joint_speed: float | None, linear_speed: float | None
    ) -> tuple[float, float]:
        js = min(joint_speed or 0.3, self.site.max_joint_speed)
        ls = min(linear_speed or 0.1, self.site.max_linear_speed)
        return js, ls

    def validate_before_move(
        self,
        robot: RobotDriver,
        tool_name: str,
        inputs: dict,
    ) -> dict | None:
        if tool_name in ("get_robot_state", "list_urp_programs"):
            return None

        if tool_name == "run_urp_program":
            name = (inputs.get("program_name") or "").strip()
            if not is_program_allowed(name):
                return {
                    "status": "error",
                    "reason": (
                        f"Program '{name}' not in whitelist. "
                        f"Allowed: {ALLOWED_URP_PROGRAMS}"
                    ),
                }

        if tool_name in MOTION_TOOLS:
            if self.site.require_state_before_move and not self._state_read_this_goal:
                return {
                    "status": "error",
                    "reason": (
                        f"Site '{self.site.site_id}' requires get_robot_state before motion. "
                        "Read state first."
                    ),
                }

        err = self.validate_arm(robot)
        if err:
            return err

        needs_ready = tool_name in MOTION_TOOLS or tool_name in (
            "run_urp_program",
            "open_gripper",
            "close_gripper",
        )
        if needs_ready:
            err = self.check_robot_ready(robot)
            if err and not robot.is_simulated:
                return err

        if tool_name == "move_down":
            dist = float(inputs.get("distance_m", 0))
            if dist > self.site.max_single_move_down:
                return {
                    "status": "error",
                    "reason": (
                        f"distance_m={dist} exceeds site limit "
                        f"{self.site.max_single_move_down}m at '{self.site.site_id}'."
                    ),
                }

        if tool_name in ("move_joint", "move_linear"):
            speed_key = "speed"
            if speed_key in inputs:
                js, ls = self.clamp_speeds(
                    inputs.get("speed") if tool_name == "move_joint" else None,
                    inputs.get("speed") if tool_name == "move_linear" else None,
                )
                cap = js if tool_name == "move_joint" else ls
                if inputs["speed"] > cap:
                    inputs["speed"] = cap

        if self.site.human_proximity_strict and tool_name in (
            "move_forward",
            "move_backward",
            "move_left",
            "move_right",
        ):
            dist = float(inputs.get("distance_m", 0))
            if dist > 0.15:
                return {
                    "status": "error",
                    "reason": (
                        f"Horizontal move {dist}m exceeds 15cm cap in human-proximity zone "
                        f"({self.site.site_id}). Split into smaller moves."
                    ),
                }

        return None
