# policy/safety.py — site-aware motion policy (airport rules)

from dataclasses import dataclass, field

from config.programs import is_program_allowed
from config.settings import ALLOWED_URP_PROGRAMS
from config.sites import SiteProfile
from robot.base import RobotDriver

MOTION_TOOLS = {
    "move_home",
    "move_joint",
    "jog_joint",
    "move_linear",
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "move_forward",
    "move_backward",
    "execute_rl_policy",
    "approach_object_once",
    "go_to_object",
}

GRIPPER_TOOLS = frozenset({"open_gripper", "close_gripper", "toggle_gripper"})

# During direction-check / approach goals the agent must not freestyle.
_AGENT_BLOCKED_TOOLS = frozenset(
    {
        "move_home",
        "move_joint",
        "jog_joint",
        "release_rtde_control",
        "run_urp_program",
    }
)

_GRIPPER_GOAL_WORDS = ("gripper", "open", "close", "pick", "place", "grasp", "release", "toggle")

# UR robot modes: 7 = RUNNING; safety 1 = NORMAL
ROBOT_MODE_RUNNING = 7
SAFETY_MODE_NORMAL = 1

# Taught UR5 home pose in degrees (agent must not command this unless UI / explicit allow).
_HOME_JOINTS_DEG = [0.0, -90.0, 0.0, -90.0, 0.0, 0.0]


@dataclass
class PolicyEngine:
    site: SiteProfile
    _state_read_this_goal: bool = field(default=False, init=False)
    _motions_this_goal: int = field(default=0, init=False)
    _actions_this_goal: int = field(default=0, init=False)
    _goal_text: str = field(default="", init=False)
    _gripper_allowed: bool = field(default=False, init=False)

    def begin_goal(self, goal: str = ""):
        self._state_read_this_goal = False
        self._motions_this_goal = 0
        self._actions_this_goal = 0
        self._goal_text = (goal or "").strip().lower()
        # Gripper only if the user clearly asked for it (not after a failed approach).
        self._gripper_allowed = any(w in self._goal_text for w in _GRIPPER_GOAL_WORDS)

    def record_state_read(self):
        self._state_read_this_goal = True

    def record_motion(self):
        self._motions_this_goal += 1
        self._actions_this_goal += 1

    def record_action(self):
        self._actions_this_goal += 1

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

    @staticmethod
    def _is_home_joint_command(joint_positions_deg: list) -> bool:
        if not joint_positions_deg or len(joint_positions_deg) < 6:
            return False
        return all(
            abs(float(joint_positions_deg[i]) - _HOME_JOINTS_DEG[i]) < 3.0 for i in range(6)
        )

    def validate_caller(self, caller: str, tool_name: str, inputs: dict) -> dict | None:
        """Block agent from homing, joint moves to home, or RTDE/URP recovery tricks."""
        if caller != "agent":
            return None
        if tool_name in _AGENT_BLOCKED_TOOLS:
            return {
                "status": "error",
                "reason": (
                    f"Tool '{tool_name}' is not allowed for Agentic AI "
                    "(prevents unwanted homing / program changes). Use Tool Console buttons "
                    "or ask the operator to fix the pendant."
                ),
            }
        if tool_name in GRIPPER_TOOLS and not self._gripper_allowed:
            return {
                "status": "error",
                "reason": (
                    "Gripper is blocked unless you explicitly say open/close/pick/place. "
                    "Direction-check goals do one move only — no gripper."
                ),
            }
        # One physical action per Run (motion or gripper) — no "and then open gripper".
        if self._actions_this_goal >= 1 and (
            tool_name in MOTION_TOOLS or tool_name in GRIPPER_TOOLS
        ):
            return {
                "status": "error",
                "reason": (
                    "Only ONE action per Run (direction check). "
                    "Click Run again for another step — no follow-up gripper/moves."
                ),
            }
        if tool_name == "move_joint":
            joints = inputs.get("joint_positions_deg") or []
            if self._is_home_joint_command(joints):
                return {
                    "status": "error",
                    "reason": (
                        "Agentic AI cannot command home joint pose. "
                        "Use the Home button in Tool Console if the operator requests home."
                    ),
                }
        return None

    def validate_before_move(
        self,
        robot: RobotDriver,
        tool_name: str,
        inputs: dict,
        caller: str = "ui",
    ) -> dict | None:
        blocked = self.validate_caller(caller, tool_name, inputs)
        if blocked:
            return blocked

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
            # One motion per Agentic Run only (UI buttons stay free).
            if caller == "agent":
                if self._motions_this_goal >= 1:
                    return {
                        "status": "error",
                        "reason": (
                            "Only ONE motion per Run is allowed right now "
                            "(direction check). Click Run again for another step."
                        ),
                    }
                if tool_name == "execute_rl_policy":
                    return {
                        "status": "error",
                        "reason": (
                            "Multi-step camera_reach is disabled during direction check. "
                            "Use approach_object_once or say 'move toward the bottle'."
                        ),
                    }
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
            "toggle_gripper",
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
