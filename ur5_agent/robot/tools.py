# robot/tools.py — agent tools (arm-agnostic) + Claude schemas

import math
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.programs import is_program_allowed
from config.settings import ALLOWED_URP_PROGRAMS, CAMERA_TYPE, MAX_SINGLE_MOVE_DOWN
from policy.safety import PolicyEngine
from robot.base import RobotDriver

if CAMERA_TYPE == "realsense":
    from camera import RealSenseCamera
else:
    RealSenseCamera = None

_camera = None


def get_robot_state(robot: RobotDriver, policy: PolicyEngine) -> dict:
    state = robot.get_full_state()
    policy.record_state_read()
    print(
        f"  [TOOL] get_robot_state → {robot.arm_model} "
        f"mode={state['robot_mode']}, safety={state['safety_mode']}"
    )
    return state


def move_home(robot: RobotDriver) -> dict:
    print("  [TOOL] move_home")
    robot.move_home()
    return {
        "status": "done",
        "position": "home",
        "joints": robot.get_joint_positions(),
    }


def move_joint(
    robot: RobotDriver,
    joint_positions_deg: list,
    speed: float = 0.3,
    acceleration: float = 0.3,
) -> dict:
    joints_rad = [math.radians(d) for d in joint_positions_deg]
    print(f"  [TOOL] move_joint → {joint_positions_deg} deg")
    robot.move_joint(joints_rad, speed, acceleration)
    return {
        "status": "done",
        "final_joints_deg": [
            round(math.degrees(v), 2) for v in robot.get_joint_positions()
        ],
    }


def move_linear(
    robot: RobotDriver,
    tcp_pose: list,
    speed: float = 0.1,
    acceleration: float = 0.1,
) -> dict:
    print(f"  [TOOL] move_linear → {tcp_pose}")
    robot.move_linear(tcp_pose, speed, acceleration)
    return {"status": "done", "final_tcp": robot.get_tcp_pose()}


def move_up(robot: RobotDriver, distance_m: float = 0.05) -> dict:
    print(f"  [TOOL] move_up {distance_m * 100:.1f}cm")
    robot.move_tcp_relative(dz=distance_m)
    return {
        "status": "done",
        "moved_up_m": distance_m,
        "final_tcp": robot.get_tcp_pose(),
    }


def move_down(robot: RobotDriver, distance_m: float = 0.05) -> dict:
    if distance_m > MAX_SINGLE_MOVE_DOWN:
        return {
            "status": "error",
            "reason": (
                f"distance_m={distance_m} exceeds global limit {MAX_SINGLE_MOVE_DOWN}m. "
                "Split into smaller moves."
            ),
        }
    print(f"  [TOOL] move_down {distance_m * 100:.1f}cm")
    robot.move_tcp_relative(dz=-distance_m)
    return {
        "status": "done",
        "moved_down_m": distance_m,
        "final_tcp": robot.get_tcp_pose(),
    }


def move_left(robot: RobotDriver, distance_m: float = 0.05) -> dict:
    print(f"  [TOOL] move_left {distance_m * 100:.1f}cm")
    robot.move_tcp_relative(dy=-distance_m)
    return {"status": "done", "final_tcp": robot.get_tcp_pose()}


def move_right(robot: RobotDriver, distance_m: float = 0.05) -> dict:
    print(f"  [TOOL] move_right {distance_m * 100:.1f}cm")
    robot.move_tcp_relative(dy=distance_m)
    return {"status": "done", "final_tcp": robot.get_tcp_pose()}


def move_forward(robot: RobotDriver, distance_m: float = 0.05) -> dict:
    print(f"  [TOOL] move_forward {distance_m * 100:.1f}cm")
    robot.move_tcp_relative(dx=distance_m)
    return {"status": "done", "final_tcp": robot.get_tcp_pose()}


def move_backward(robot: RobotDriver, distance_m: float = 0.05) -> dict:
    print(f"  [TOOL] move_backward {distance_m * 100:.1f}cm")
    robot.move_tcp_relative(dx=-distance_m)
    return {"status": "done", "final_tcp": robot.get_tcp_pose()}


def stop_robot(robot: RobotDriver) -> dict:
    print("  [TOOL] STOP")
    robot.stop()
    return {"status": "stopped"}


def open_gripper(robot: RobotDriver) -> dict:
    print("  [TOOL] open_gripper")
    robot.gripper_open()
    state = robot.get_gripper_state()
    return {
        "status": "done",
        "gripper": state,
        "important": state.get("pendant_check", ""),
        "output_readback": state.get("output_readback", {}),
    }


def close_gripper(robot: RobotDriver) -> dict:
    print("  [TOOL] close_gripper")
    robot.gripper_close()
    state = robot.get_gripper_state()
    return {
        "status": "done",
        "gripper": state,
        "important": state.get("pendant_check", ""),
        "output_readback": state.get("output_readback", {}),
    }


def list_urp_programs(robot: RobotDriver) -> dict:
    print("  [TOOL] list_urp_programs")
    current = robot.get_program_state() if hasattr(robot, "get_program_state") else {}
    return {
        "status": "done",
        "agent_whitelist": ALLOWED_URP_PROGRAMS,
        "on_robot_now": current,
        "note": (
            "Whitelist = programs the agent may run. "
            "on_robot_now = dashboard programState (loaded/running). "
            "To see all files on robot, use PolyScope Program screen or scripts/diagnose_robot.py"
        ),
    }


def release_rtde_control(robot: RobotDriver) -> dict:
    print("  [TOOL] release_rtde_control")
    if not hasattr(robot, "release_rtde_control"):
        return {"status": "error", "reason": "not supported on this driver"}
    return robot.release_rtde_control()


def reconnect_rtde_control(robot: RobotDriver) -> dict:
    print("  [TOOL] reconnect_rtde_control")
    if not hasattr(robot, "reconnect_rtde_control"):
        return {"status": "error", "reason": "not supported on this driver"}
    return robot.reconnect_rtde_control()


def run_urp_program(robot: RobotDriver, program_name: str) -> dict:
    print(f"  [TOOL] run_urp_program → {program_name}")
    if not is_program_allowed(program_name):
        return {
            "status": "error",
            "reason": f"Program not allowed. Whitelist: {ALLOWED_URP_PROGRAMS}",
        }
    return robot.run_urp_program(program_name)


def stop_urp_program(robot: RobotDriver) -> dict:
    print("  [TOOL] stop_urp_program")
    return robot.stop_urp_program()


def get_camera_frame(robot: RobotDriver, session_id: str = "lab", prefix: str = "frame") -> dict:
    del robot  # camera can be used even if robot motion is blocked
    if CAMERA_TYPE == "none":
        return {"status": "error", "reason": "Camera disabled (CAMERA_TYPE=none)."}
    if RealSenseCamera is None:
        return {"status": "error", "reason": "RealSense camera module unavailable."}

    global _camera
    if _camera is None:
        _camera = RealSenseCamera()
    print("  [TOOL] get_camera_frame")
    try:
        return _camera.save_color_frame(session_id=session_id, prefix=prefix)
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def execute_tool(
    name: str,
    inputs: dict,
    robot: RobotDriver,
    policy: PolicyEngine,
) -> dict:
    block = policy.validate_before_move(robot, name, inputs)
    if block:
        print(f"  [POLICY] blocked: {block['reason']}")
        return block

    if name == "move_joint" and "speed" in inputs:
        js, _ = policy.clamp_speeds(inputs.get("speed"), None)
        inputs["speed"] = js
    if name in ("move_linear", "move_up", "move_down", "move_left", "move_right", "move_forward", "move_backward"):
        if "speed" in inputs:
            _, ls = policy.clamp_speeds(None, inputs.get("speed"))
            inputs["speed"] = ls

    dispatch = {
        "get_robot_state": lambda: get_robot_state(robot, policy),
        "move_home": lambda: move_home(robot),
        "move_joint": lambda: move_joint(robot, **inputs),
        "move_linear": lambda: move_linear(robot, **inputs),
        "move_up": lambda: move_up(robot, **inputs),
        "move_down": lambda: move_down(robot, **inputs),
        "move_left": lambda: move_left(robot, **inputs),
        "move_right": lambda: move_right(robot, **inputs),
        "move_forward": lambda: move_forward(robot, **inputs),
        "move_backward": lambda: move_backward(robot, **inputs),
        "stop_robot": lambda: stop_robot(robot),
        "open_gripper": lambda: open_gripper(robot),
        "close_gripper": lambda: close_gripper(robot),
        "list_urp_programs": lambda: list_urp_programs(robot),
        "run_urp_program": lambda: run_urp_program(robot, **inputs),
        "stop_urp_program": lambda: stop_urp_program(robot),
        "release_rtde_control": lambda: release_rtde_control(robot),
        "reconnect_rtde_control": lambda: reconnect_rtde_control(robot),
        "get_camera_frame": lambda: get_camera_frame(robot, **inputs),
    }
    fn = dispatch.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    return fn()


TOOL_SCHEMAS = [
    {
        "name": "get_robot_state",
        "description": "Read full robot state: joints, TCP, force, modes. Required before motion at airport sites.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "move_home",
        "description": "Move to safe home. Use when position is unknown.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "move_joint",
        "description": "Joint motion; angles in DEGREES [j1..j6].",
        "input_schema": {
            "type": "object",
            "properties": {
                "joint_positions_deg": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "6 joint angles in degrees",
                },
                "speed": {"type": "number"},
                "acceleration": {"type": "number"},
            },
            "required": ["joint_positions_deg"],
        },
    },
    {
        "name": "move_linear",
        "description": "Absolute TCP pose [x, y, z, rx, ry, rz] in meters and radians.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tcp_pose": {"type": "array", "items": {"type": "number"}},
                "speed": {"type": "number"},
                "acceleration": {"type": "number"},
            },
            "required": ["tcp_pose"],
        },
    },
    {
        "name": "move_up",
        "description": "Move TCP up (positive Z) in meters.",
        "input_schema": {
            "type": "object",
            "properties": {"distance_m": {"type": "number"}},
            "required": ["distance_m"],
        },
    },
    {
        "name": "move_down",
        "description": "Move TCP down (negative Z). Site policy may limit distance per step.",
        "input_schema": {
            "type": "object",
            "properties": {"distance_m": {"type": "number"}},
            "required": ["distance_m"],
        },
    },
    {
        "name": "move_left",
        "description": "Move TCP left (negative Y) in meters.",
        "input_schema": {
            "type": "object",
            "properties": {"distance_m": {"type": "number"}},
            "required": ["distance_m"],
        },
    },
    {
        "name": "move_right",
        "description": "Move TCP right (positive Y) in meters.",
        "input_schema": {
            "type": "object",
            "properties": {"distance_m": {"type": "number"}},
            "required": ["distance_m"],
        },
    },
    {
        "name": "move_forward",
        "description": "Move TCP forward (positive X) in meters.",
        "input_schema": {
            "type": "object",
            "properties": {"distance_m": {"type": "number"}},
            "required": ["distance_m"],
        },
    },
    {
        "name": "move_backward",
        "description": "Move TCP backward (negative X) in meters.",
        "input_schema": {
            "type": "object",
            "properties": {"distance_m": {"type": "number"}},
            "required": ["distance_m"],
        },
    },
    {
        "name": "stop_robot",
        "description": "Emergency stop all motion.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "open_gripper",
        "description": "Open the Robotiq gripper (URCap socket, PolyScope ID 1).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "close_gripper",
        "description": "Close the Robotiq gripper (URCap socket, PolyScope ID 1).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_urp_programs",
        "description": "Show agent whitelist and current loaded/running program from dashboard.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "release_rtde_control",
        "description": "Release RTDE external control so dashboard can play a .urp program.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "reconnect_rtde_control",
        "description": "Reconnect RTDE control after .urp for agent move_* / gripper tools.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_urp_program",
        "description": (
            "Load and play a PolyScope program on the robot, e.g. fly2.urp. "
            "May stop RTDE external control while the URP runs. Only whitelisted names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "program_name": {
                    "type": "string",
                    "description": "Program file name, e.g. fly2.urp or fly2",
                }
            },
            "required": ["program_name"],
        },
    },
    {
        "name": "stop_urp_program",
        "description": "Stop the currently running PolyScope program.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_camera_frame",
        "description": "Capture and save one RGB frame from Intel RealSense camera.",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session tag in saved filename"},
                "prefix": {"type": "string", "description": "Filename prefix"},
            },
            "required": [],
        },
    },
]
