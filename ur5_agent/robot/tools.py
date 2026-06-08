# robot/tools.py — agent tools (arm-agnostic) + Claude schemas

import math
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.programs import is_program_allowed
from config.settings import (
    ALLOWED_URP_PROGRAMS,
    CAMERA_TYPE,
    GRIPPER_TOGGLE_PAUSE_SEC,
    GRIPPER_TYPE,
    MAX_SINGLE_MOVE_DOWN,
    MOTION_BACKWARD_VEC,
    MOTION_DOWN_VEC,
    MOTION_FORWARD_VEC,
    MOTION_LEFT_VEC,
    MOTION_RIGHT_VEC,
    MOTION_UP_VEC,
    ROBOTIQ_CLOSE_POS,
    ROBOTIQ_OPEN_POS,
)
from policy.safety import MOTION_TOOLS, PolicyEngine
from robot.base import RobotDriver

if CAMERA_TYPE == "realsense":
    from camera import ObjectDetector, RealSenseCamera
else:
    RealSenseCamera = None
    ObjectDetector = None

_camera = None
_detector = None


def _distance_magnitude(distance_m: float, tool_name: str) -> float:
    """Directional tools use positive distance_m only; sign is set by the tool itself."""
    d = abs(float(distance_m))
    if d < 1e-6:
        raise ValueError(f"{tool_name}: distance_m must be non-zero.")
    if float(distance_m) < 0:
        print(
            f"  [WARN] {tool_name}: negative distance_m ({distance_m}) "
            f"ignored — using {d} m in the correct direction."
        )
    return d


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


def _motion_result(requested_m: float, motion_report: dict | None) -> dict:
    out = {"status": "done", "requested_m": requested_m}
    if motion_report:
        out["achieved_m"] = motion_report.get("achieved_m")
        out["commanded_m"] = motion_report.get("commanded_m")
        out["final_tcp"] = motion_report.get("final_tcp") or []
    else:
        out["final_tcp"] = []
    commanded = float(out.get("commanded_m") or requested_m)
    achieved = float(out.get("achieved_m") or 0.0)
    if commanded >= 0.003 and achieved < commanded * 0.35:
        out["status"] = "error"
        out["reason"] = (
            f"Robot moved only {achieved * 100:.1f} cm of {commanded * 100:.1f} cm "
            "(controller skipped move or stale pose). Call get_robot_state and retry."
        )
    return out


def _move_along_axis(robot: RobotDriver, tool_name: str, axis: tuple[float, float, float], distance_m: float) -> dict:
    d = _distance_magnitude(distance_m, tool_name)
    dx, dy, dz = (axis[0] * d, axis[1] * d, axis[2] * d)
    print(f"  [TOOL] {tool_name} {d * 100:.1f}cm  (base Δ [{dx:.3f}, {dy:.3f}, {dz:.3f}])")
    report = robot.move_tcp_relative(dx=dx, dy=dy, dz=dz)
    result = _motion_result(d, report)
    if result.get("status") == "error":
        print(f"  [WARN] {tool_name} under-move — retrying once after settle")
        time.sleep(0.2)
        report = robot.move_tcp_relative(dx=dx, dy=dy, dz=dz)
        result = _motion_result(d, report)
    result["base_delta"] = [round(dx, 4), round(dy, 4), round(dz, 4)]
    return result


def move_up(robot: RobotDriver, distance_m: float = 0.05) -> dict:
    result = _move_along_axis(robot, "move_up", MOTION_UP_VEC, distance_m)
    result["moved_up_m"] = result["requested_m"]
    return result


def move_down(robot: RobotDriver, distance_m: float = 0.05) -> dict:
    d = _distance_magnitude(distance_m, "move_down")
    if d > MAX_SINGLE_MOVE_DOWN:
        return {
            "status": "error",
            "reason": (
                f"distance_m={d} exceeds global limit {MAX_SINGLE_MOVE_DOWN}m. "
                "Split into smaller moves."
            ),
        }
    result = _move_along_axis(robot, "move_down", MOTION_DOWN_VEC, distance_m)
    result["moved_down_m"] = result["requested_m"]
    return result


def move_left(robot: RobotDriver, distance_m: float = 0.05) -> dict:
    return _move_along_axis(robot, "move_left", MOTION_LEFT_VEC, distance_m)


def move_right(robot: RobotDriver, distance_m: float = 0.05) -> dict:
    return _move_along_axis(robot, "move_right", MOTION_RIGHT_VEC, distance_m)


def move_forward(robot: RobotDriver, distance_m: float = 0.05) -> dict:
    return _move_along_axis(robot, "move_forward", MOTION_FORWARD_VEC, distance_m)


def move_backward(robot: RobotDriver, distance_m: float = 0.05) -> dict:
    return _move_along_axis(robot, "move_backward", MOTION_BACKWARD_VEC, distance_m)


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


def _gripper_is_open(state: dict) -> bool:
    if GRIPPER_TYPE == "robotiq":
        pos = state.get("position")
        if isinstance(pos, (int, float)):
            mid = (ROBOTIQ_OPEN_POS + ROBOTIQ_CLOSE_POS) / 2
            return float(pos) < mid
    cmd = state.get("last_command") or state.get("command_state") or ""
    return str(cmd).lower() == "open"


def toggle_gripper(robot: RobotDriver) -> dict:
    """Open then close — always ends closed (exercises the gripper either way)."""
    state = robot.get_gripper_state()
    was_open = _gripper_is_open(state)
    print(
        f"  [TOOL] toggle_gripper (was {'open' if was_open else 'closed'}) → open then close"
    )
    open_gripper(robot)
    time.sleep(max(0.1, GRIPPER_TOGGLE_PAUSE_SEC))
    result = close_gripper(robot)
    result["toggled_from"] = "open" if was_open else "closed"
    result["toggled_to"] = "closed"
    result["sequence"] = ["open", "close"]
    return result


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


def _ensure_camera():
    global _camera, _detector
    if CAMERA_TYPE == "none":
        return None, {"status": "error", "reason": "Camera disabled (CAMERA_TYPE=none)."}
    if RealSenseCamera is None:
        return None, {"status": "error", "reason": "RealSense camera module unavailable."}
    if _camera is None:
        _camera = RealSenseCamera()
    if _detector is None and ObjectDetector is not None:
        _detector = ObjectDetector()
    return _camera, None


def get_camera_frame(robot: RobotDriver, session_id: str = "lab", prefix: str = "frame") -> dict:
    del robot
    cam, err = _ensure_camera()
    if err:
        return err
    print("  [TOOL] get_camera_frame")
    try:
        return cam.save_color_frame(session_id=session_id, prefix=prefix)
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def detect_objects(
    robot: RobotDriver,
    save_image: bool = False,
    session_id: str = "lab",
    prefix: str = "detect",
    label_filter: str = "",
) -> dict:
    del robot
    cam, err = _ensure_camera()
    if err:
        return err
    if _detector is None:
        return {"status": "error", "reason": "Object detector unavailable."}

    print("  [TOOL] detect_objects")
    try:
        frame = cam.capture_color_frame()
        meta = _detector.detect(frame)
        objects = meta.get("objects", [])
        if label_filter:
            needle = label_filter.strip().lower()
            objects = [o for o in objects if needle in str(o.get("label", "")).lower()]
            labels = [o["label"] for o in objects]
            meta = {
                **meta,
                "objects": objects,
                "count": len(objects),
                "labels": labels,
                "unique_labels": list(dict.fromkeys(labels)),
                "label_filter": label_filter,
            }

        result = {
            "status": "done",
            "count": meta.get("count", 0),
            "labels": meta.get("labels", []),
            "unique_labels": meta.get("unique_labels", []),
            "detector": meta.get("detector"),
            "model": meta.get("model"),
            "objects": objects,
            "image_shape": list(frame.shape),
        }
        if save_image:
            import cv2
            import datetime as dt
            import os

            from config.settings import CAMERA_OUTPUT_DIR

            drawn = _detector.draw_boxes(frame, meta)
            out_dir = os.path.abspath(CAMERA_OUTPUT_DIR)
            os.makedirs(out_dir, exist_ok=True)
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(out_dir, f"{prefix}_{session_id}_{stamp}.jpg")
            if cv2.imwrite(path, drawn):
                result["annotated_path"] = path
        return result
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def execute_tool(
    name: str,
    inputs: dict,
    robot: RobotDriver,
    policy: PolicyEngine,
    caller: str = "ui",
) -> dict:
    if (
        name in MOTION_TOOLS
        and policy.site.require_state_before_move
        and not policy._state_read_this_goal
    ):
        get_robot_state(robot, policy)

    block = policy.validate_before_move(robot, name, inputs, caller=caller)
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
        "toggle_gripper": lambda: toggle_gripper(robot),
        "list_urp_programs": lambda: list_urp_programs(robot),
        "run_urp_program": lambda: run_urp_program(robot, **inputs),
        "stop_urp_program": lambda: stop_urp_program(robot),
        "release_rtde_control": lambda: release_rtde_control(robot),
        "reconnect_rtde_control": lambda: reconnect_rtde_control(robot),
        "get_camera_frame": lambda: get_camera_frame(robot, **inputs),
        "detect_objects": lambda: detect_objects(robot, **inputs),
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
        "description": (
            "Move to taught home joint pose. ONLY when the operator explicitly asks "
            "to go home — never for error recovery or unclear commands."
        ),
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
        "description": "Move TCP up (positive Z). distance_m is positive magnitude in meters (0.02 = 2 cm).",
        "input_schema": {
            "type": "object",
            "properties": {
                "distance_m": {
                    "type": "number",
                    "description": "Positive distance in meters (never negative)",
                }
            },
            "required": ["distance_m"],
        },
    },
    {
        "name": "move_down",
        "description": "Move TCP down. distance_m is positive magnitude in meters (0.02 = 2 cm). Site may cap per step.",
        "input_schema": {
            "type": "object",
            "properties": {
                "distance_m": {
                    "type": "number",
                    "description": "Positive distance in meters (never negative)",
                }
            },
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
        "name": "toggle_gripper",
        "description": (
            "Gripper toggle cycle: always open, pause, then close. "
            "Always ends closed regardless of starting state."
        ),
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
    {
        "name": "detect_objects",
        "description": (
            "Capture a live camera frame and run object detection (YOLO or contour fallback). "
            "Returns count, label names, and bounding boxes in image pixels."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "save_image": {
                    "type": "boolean",
                    "description": "If true, save annotated JPEG with boxes drawn",
                },
                "session_id": {"type": "string", "description": "Session tag when saving image"},
                "prefix": {"type": "string", "description": "Filename prefix when saving image"},
                "label_filter": {
                    "type": "string",
                    "description": "Optional substring filter (e.g. 'cup') — only matching objects returned",
                },
            },
            "required": [],
        },
    },
]
