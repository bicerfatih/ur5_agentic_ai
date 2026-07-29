# robot/tools.py — agent tools (arm-agnostic) + Claude schemas

import math
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.programs import is_program_allowed
from config.settings import (
    ALLOWED_URP_PROGRAMS,
    APPROACH_IMAGE_INVERT_FB,
    APPROACH_IMAGE_INVERT_LR,
    CAMERA_TYPE,
    GRIPPER_TOGGLE_PAUSE_SEC,
    GRIPPER_TYPE,
    HAND_EYE_CALIB_PATH,
    MAX_SINGLE_MOVE_DOWN,
    MOTION_BACKWARD_VEC,
    MOTION_DOWN_VEC,
    MOTION_FORWARD_VEC,
    MOTION_HORIZONTAL_MODE,
    MOTION_LEFT_VEC,
    MOTION_RIGHT_VEC,
    MOTION_UP_VEC,
    REACH_APPROACH_OFFSET_M,
    REACH_DONE_DIST_M,
    ROBOTIQ_CLOSE_POS,
    ROBOTIQ_OPEN_POS,
    RL_CONTROL_DT,
    RL_OBS_MODE,
    RL_POLICY_PATH,
)
from robot.motion_math import tool_horizontal_unit
from robot.policies.rl_policy import ReachPolicyRunner
from policy.safety import GRIPPER_TOOLS, MOTION_TOOLS, PolicyEngine
from robot.base import RobotDriver

if CAMERA_TYPE == "realsense":
    from camera import ObjectDetector, RealSenseCamera
    from camera.geometry import HandEyeCalibration, estimate_object_target_base
else:
    RealSenseCamera = None
    ObjectDetector = None
    HandEyeCalibration = None
    estimate_object_target_base = None

_camera = None
_detector = None
_reach_runner = None
_hand_eye = None


def _parse_xyz_triplet(raw: str | list | None, default: list[float]) -> list[float]:
    if raw is None:
        return list(default)
    if isinstance(raw, list) and len(raw) >= 3:
        return [float(raw[0]), float(raw[1]), float(raw[2])]
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) >= 3:
            return [float(parts[0]), float(parts[1]), float(parts[2])]
    return list(default)


def _ensure_hand_eye():
    global _hand_eye
    if HandEyeCalibration is None:
        return None, {"status": "error", "reason": "Hand-eye calibration module unavailable."}
    if _hand_eye is None:
        _hand_eye = HandEyeCalibration(HAND_EYE_CALIB_PATH)
    if not _hand_eye.loaded:
        return None, {
            "status": "error",
            "reason": (
                f"Hand-eye calibration not found at {HAND_EYE_CALIB_PATH}. "
                "Copy data/calibration/hand_eye.example.json and fill measured values."
            ),
        }
    return _hand_eye, None


def _pick_detection_object(meta: dict, target_label: str) -> dict | None:
    objs = meta.get("objects", []) if isinstance(meta, dict) else []
    if not objs:
        return None
    if target_label:
        needle = target_label.strip().lower()
        for o in objs:
            if needle in str(o.get("label", "")).lower():
                return o
    return objs[0]


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


def jog_joint(
    robot: RobotDriver,
    joint: int,
    delta_deg: float,
    speed: float = 0.25,
    acceleration: float = 0.25,
) -> dict:
    """Relative joint jog: joint 1–6, delta_deg positive = + direction."""
    if joint < 1 or joint > 6:
        return {"status": "error", "reason": "joint must be 1–6"}
    if abs(delta_deg) < 0.01:
        return {"status": "error", "reason": "delta_deg too small"}
    if abs(delta_deg) > 15.0:
        return {"status": "error", "reason": "delta_deg max 15° per jog"}

    current_deg = [math.degrees(v) for v in robot.get_joint_positions()]
    idx = joint - 1
    target = list(current_deg)
    target[idx] = round(target[idx] + float(delta_deg), 3)
    print(f"  [TOOL] jog_joint J{joint} {delta_deg:+.2f}° → {target}")
    return move_joint(robot, target, speed=speed, acceleration=acceleration)


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


_HORIZONTAL_TOOL_AXES = {
    "move_left": (0.0, -1.0, 0.0),
    "move_right": (0.0, 1.0, 0.0),
    "move_forward": (1.0, 0.0, 0.0),
    "move_backward": (-1.0, 0.0, 0.0),
}

_BASE_HORIZONTAL_FALLBACK = {
    "move_left": MOTION_LEFT_VEC,
    "move_right": MOTION_RIGHT_VEC,
    "move_forward": MOTION_FORWARD_VEC,
    "move_backward": MOTION_BACKWARD_VEC,
}


def _motion_axis(robot: RobotDriver, tool_name: str, axis: tuple[float, float, float]) -> tuple[float, float, float]:
    if MOTION_HORIZONTAL_MODE != "tool_horizontal" or tool_name not in _HORIZONTAL_TOOL_AXES:
        return axis
    try:
        pose = robot.get_tcp_pose()
    except Exception:
        return _BASE_HORIZONTAL_FALLBACK.get(tool_name, axis)
    if not pose or len(pose) < 6:
        return _BASE_HORIZONTAL_FALLBACK.get(tool_name, axis)
    unit = tool_horizontal_unit((pose[3], pose[4], pose[5]), _HORIZONTAL_TOOL_AXES[tool_name])
    if unit == (0.0, 0.0, 0.0):
        return _BASE_HORIZONTAL_FALLBACK.get(tool_name, axis)
    return unit


def _move_along_axis(robot: RobotDriver, tool_name: str, axis: tuple[float, float, float], distance_m: float) -> dict:
    d = _distance_magnitude(distance_m, tool_name)
    unit = _motion_axis(robot, tool_name, axis)
    dx, dy, dz = (unit[0] * d, unit[1] * d, unit[2] * d)
    frame = "tool-horizontal" if tool_name in _HORIZONTAL_TOOL_AXES and MOTION_HORIZONTAL_MODE == "tool_horizontal" else "base"
    print(f"  [TOOL] {tool_name} {d * 100:.1f}cm  ({frame} Δ [{dx:.3f}, {dy:.3f}, {dz:.3f}])")
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
    include_3d: bool = True,
    approach_offset_m: list | None = None,
) -> dict:
    cam, err = _ensure_camera()
    if err:
        return err
    if _detector is None:
        return {"status": "error", "reason": "Object detector unavailable."}

    print("  [TOOL] detect_objects")
    try:
        rgbd = cam.capture_rgbd()
        frame = rgbd["color"]
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

        pose3d = None
        if include_3d and rgbd.get("depth") is not None and estimate_object_target_base is not None:
            calib, calib_err = _ensure_hand_eye()
            if calib_err:
                pose3d = {"status": "error", "reason": calib_err["reason"]}
            else:
                offset = _parse_xyz_triplet(
                    approach_offset_m,
                    _parse_xyz_triplet(REACH_APPROACH_OFFSET_M, [0.0, 0.0, 0.05]),
                )
                st = robot.get_full_state()
                tcp_pose = st.get("tcp_pose") if calib.mount == "eye_in_hand" else None
                enriched = []
                for obj in objects:
                    est = estimate_object_target_base(
                        obj=obj,
                        depth_image=rgbd["depth"],
                        depth_scale=float(rgbd.get("depth_scale", 0.001)),
                        intrinsics=rgbd.get("intrinsics") or {},
                        calib=calib,
                        tcp_pose=tcp_pose,
                        approach_offset_m=offset,
                    )
                    if est and "target_base_m" in est:
                        obj = {**obj, "target_base_m": est["target_base_m"], "pose3d": est}
                    elif est and est.get("error"):
                        obj = {**obj, "pose3d_error": est["error"]}
                    enriched.append(obj)
                objects = enriched
                meta["objects"] = objects
                pose3d = {"status": "ok", "approach_offset_m": offset}

        result = {
            "status": "done",
            "count": meta.get("count", 0),
            "labels": meta.get("labels", []),
            "unique_labels": meta.get("unique_labels", []),
            "detector": meta.get("detector"),
            "model": meta.get("model"),
            "objects": objects,
            "image_shape": list(frame.shape),
            "pose3d": pose3d,
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


def detect_objects_nanoowl(
    robot: RobotDriver,
    queries: list | None = None,
    query: str = "",
    threshold: float | None = None,
    save_image: bool = False,
    session_id: str = "lab",
    prefix: str = "nanoowl",
    include_3d: bool = True,
    approach_offset_m: list | None = None,
) -> dict:
    """Open-vocabulary detection via NanoOWL on Thor GPU.

    queries: list of text strings, e.g. ["cup", "screwdriver"]
    query:   single text string (shorthand for queries=["..."])
    """
    cam, err = _ensure_camera()
    if err:
        return err

    # Build query list.
    q_list: list[str] = []
    if query:
        q_list = [q.strip() for q in query.split(",") if q.strip()]
    if queries:
        q_list = list(queries)
    if not q_list:
        from config.settings import NANOOWL_DEFAULT_QUERIES
        q_list = NANOOWL_DEFAULT_QUERIES

    print(f"  [TOOL] detect_objects_nanoowl queries={q_list}")

    try:
        from camera.nanoowl_detector import NanoOwlDetector
    except ImportError as e:
        return {"status": "error", "reason": f"NanoOWL not installed: {e}"}

    # Singleton per process.
    if not hasattr(detect_objects_nanoowl, "_owl"):
        detect_objects_nanoowl._owl = NanoOwlDetector()
    owl: NanoOwlDetector = detect_objects_nanoowl._owl

    try:
        rgbd = cam.capture_rgbd()
        frame = rgbd["color"]
        meta = owl.detect(frame, queries=q_list, threshold=threshold)

        objects = meta.get("objects", [])

        pose3d = None
        if include_3d and rgbd.get("depth") is not None and estimate_object_target_base is not None:
            calib, calib_err = _ensure_hand_eye()
            if calib_err:
                pose3d = {"status": "error", "reason": calib_err["reason"]}
            else:
                offset = _parse_xyz_triplet(
                    approach_offset_m,
                    _parse_xyz_triplet(REACH_APPROACH_OFFSET_M, [0.0, 0.0, 0.05]),
                )
                st = robot.get_full_state()
                tcp_pose = st.get("tcp_pose") if calib.mount == "eye_in_hand" else None
                enriched = []
                for obj in objects:
                    est = estimate_object_target_base(
                        obj=obj,
                        depth_image=rgbd["depth"],
                        depth_scale=float(rgbd.get("depth_scale", 0.001)),
                        intrinsics=rgbd.get("intrinsics") or {},
                        calib=calib,
                        tcp_pose=tcp_pose,
                        approach_offset_m=offset,
                    )
                    if est and "target_base_m" in est:
                        obj = {**obj, "target_base_m": est["target_base_m"], "pose3d": est}
                    enriched.append(obj)
                objects = enriched
                if objects:
                    pose3d = objects[0].get("pose3d")

        result = {
            "status": "done",
            "queries": q_list,
            "count": len(objects),
            "labels": [o["label"] for o in objects],
            "unique_labels": list(dict.fromkeys(o["label"] for o in objects)),
            "detector": "nanoowl",
            "model": meta.get("model"),
            "objects": objects,
            "image_shape": list(frame.shape),
            "pose3d": pose3d,
        }
        if save_image:
            import cv2, datetime as dt, os
            from camera.nanoowl_detector import _draw_boxes
            from config.settings import CAMERA_OUTPUT_DIR

            drawn = _draw_boxes(frame, meta)
            out_dir = os.path.abspath(CAMERA_OUTPUT_DIR)
            os.makedirs(out_dir, exist_ok=True)
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(out_dir, f"{prefix}_{session_id}_{stamp}.jpg")
            if cv2.imwrite(path, drawn):
                result["annotated_path"] = path
        return result
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def go_to_object(
    robot: RobotDriver,
    target_label: str = "",
    approach_offset_m: list | None = None,
    speed: float = 0.15,
    accel: float = 0.15,
) -> dict:
    """Detect object with depth camera and move TCP directly to it in ONE move.

    Uses hand-eye calibration to compute the 3D position in robot base frame,
    then commands a single moveL to the approach point (offset in front of object).
    Much faster than repeated approach_object_once steps.
    """
    cam, err = _ensure_camera()
    if err:
        return err
    calib, calib_err = _ensure_hand_eye()
    if calib_err:
        return calib_err

    print(f"  [TOOL] go_to_object label={target_label!r}")

    try:
        rgbd = cam.capture_rgbd()
    except Exception as e:
        return {"status": "error", "reason": f"camera capture failed: {e}"}

    frame = rgbd["color"]
    depth = rgbd.get("depth")
    if depth is None:
        return {"status": "error", "reason": "Depth unavailable — set CAMERA_DEPTH_ENABLED=true."}

    # Detect with NanoOWL if available, else YOLO.
    if not hasattr(go_to_object, "_owl"):
        try:
            from camera.nanoowl_detector import NanoOwlDetector
            go_to_object._owl = NanoOwlDetector()
        except Exception:
            go_to_object._owl = None
    owl = getattr(go_to_object, "_owl", None)
    if owl is not None and owl._ensure():
        queries = [target_label] if target_label else None
        meta = owl.detect(frame, queries=queries)
    else:
        meta = _detector.detect(frame)

    tgt = _pick_detection_object(meta, target_label)
    if tgt is None:
        return {
            "status": "error",
            "reason": f"Object '{target_label}' not found in frame.",
            "labels_seen": meta.get("labels", []),
        }

    st = robot.get_full_state()
    tcp_pose = st.get("tcp_pose") or [0.3, 0.0, 0.3, 0.0, 0.0, 0.0]
    tcp_before = [float(v) for v in tcp_pose[:3]]

    offset = _parse_xyz_triplet(
        approach_offset_m,
        _parse_xyz_triplet(REACH_APPROACH_OFFSET_M, [0.0, 0.0, 0.10]),
    )

    est = estimate_object_target_base(
        obj=tgt,
        depth_image=depth,
        depth_scale=float(rgbd.get("depth_scale", 0.001)),
        intrinsics=rgbd.get("intrinsics") or {},
        calib=calib,
        tcp_pose=tcp_pose if calib.mount == "eye_in_hand" else None,
        approach_offset_m=offset,
    )
    if est is None or "target_base_m" not in est:
        return {
            "status": "error",
            "reason": (est or {}).get("error", "Could not compute 3D target from depth."),
            "label": tgt.get("label"),
            "hint": "Check hand-eye calibration and ensure depth is valid at object center.",
        }

    target = [float(v) for v in est["target_base_m"][:3]]
    dist = sum((target[i] - tcp_before[i]) ** 2 for i in range(3)) ** 0.5

    if dist < 0.01:
        return {
            "status": "done",
            "note": "Already at target — no move needed.",
            "label": tgt.get("label"),
            "dist_m": round(dist, 4),
        }

    # Build full target pose: keep current orientation, move to target xyz.
    target_pose = list(tcp_pose)
    target_pose[0] = target[0]
    target_pose[1] = target[1]
    target_pose[2] = target[2]

    try:
        robot.rtde_c.moveL(target_pose, speed, accel)
    except Exception as e:
        return {"status": "error", "reason": f"moveL failed: {e}"}

    st2 = robot.get_full_state()
    tcp_after = [float(v) for v in (st2.get("tcp_pose") or target_pose)[:3]]
    dist_after = sum((target[i] - tcp_after[i]) ** 2 for i in range(3)) ** 0.5

    return {
        "status": "done",
        "label": tgt.get("label"),
        "confidence": tgt.get("confidence"),
        "target_base_m": [round(v, 4) for v in target],
        "tcp_before": [round(v, 4) for v in tcp_before],
        "tcp_after": [round(v, 4) for v in tcp_after],
        "dist_moved_m": round(dist, 4),
        "dist_remaining_m": round(dist_after, 4),
        "depth_m": round(float(est.get("depth_m", 0)), 4),
        "note": f"Moved directly to '{tgt.get('label')}' in one moveL.",
    }


def execute_rl_policy(
    robot: RobotDriver,
    task_id: str = "reach_free_space",
    steps: int = 10,
    target_label: str = "",
    policy_path: str = "",
    target_tcp: list | None = None,
    max_step_m: float = 0.01,
    settle_sec: float = RL_CONTROL_DT,
    approach_offset_m: list | None = None,
    reach_done_dist_m: float = REACH_DONE_DIST_M,
) -> dict:
    """
    RL reach loop for safe Cartesian corrections.
    - reach_free_space: known target_tcp in base frame (meters).
    - camera_reach: detect object + depth + hand-eye -> 3D target, then RL steps in X/Y/Z.
    """
    policy_file = (policy_path or RL_POLICY_PATH or "").strip()
    n_steps = max(1, min(int(steps), 100))
    max_step_m = max(0.001, min(float(max_step_m), 0.03))
    settle_sec = max(0.02, min(float(settle_sec), 1.0))
    # target_tcp is optional for reach task. If omitted, we use +5cm in X from
    # current TCP to keep behavior deterministic.

    global _reach_runner
    if _reach_runner is None or _reach_runner.policy_path != policy_file:
        _reach_runner = ReachPolicyRunner(policy_file)

    print(f"  [TOOL] execute_rl_policy task={task_id} steps={n_steps} obs={RL_OBS_MODE}")
    trace = []
    if task_id in ("reach_free_space", "rl_reach", "reach"):
        state0 = robot.get_full_state()
        tcp0 = state0.get("tcp_pose") or [0.3, 0.0, 0.3, 0.0, 0.0, 0.0]
        if target_tcp is None:
            target_xyz = [float(tcp0[0] + 0.05), float(tcp0[1]), float(tcp0[2])]
        else:
            if not isinstance(target_tcp, list) or len(target_tcp) < 3:
                return {"status": "error", "reason": "target_tcp must be [x, y, z] in meters."}
            target_xyz = [float(target_tcp[0]), float(target_tcp[1]), float(target_tcp[2])]
        for i in range(n_steps):
            st = robot.get_full_state()
            step = _reach_runner.step(st, target_xyz, max_step_m=max_step_m)
            dist = (
                (
                    (target_xyz[0] - st["tcp_pose"][0]) ** 2
                    + (target_xyz[1] - st["tcp_pose"][1]) ** 2
                    + (target_xyz[2] - st["tcp_pose"][2]) ** 2
                )
                ** 0.5
            )
            info = {
                "step": i + 1,
                "task": "reach_free_space",
                "target_tcp": [round(v, 4) for v in target_xyz],
                "dist_m": round(float(dist), 5),
                "policy_source": step.source,
                "action": {"dx": round(step.dx, 5), "dy": round(step.dy, 5), "dz": round(step.dz, 5)},
            }
            trace.append(info)
            if step.source == "done":
                return {
                    "status": "done",
                    "task_id": task_id,
                    "steps_used": i + 1,
                    "trace": trace,
                    "note": "Reached target in free-space RL task.",
                }
            report = robot.move_tcp_relative(dx=step.dx, dy=step.dy, dz=step.dz)
            info["motion_report"] = report
            time.sleep(settle_sec)
        return {
            "status": "done",
            "task_id": task_id,
            "steps_used": n_steps,
            "trace": trace,
            "note": "Max steps reached in reach task.",
        }

    cam, err = _ensure_camera()
    if err:
        return err
    if _detector is None:
        return {"status": "error", "reason": "Object detector unavailable."}
    calib, calib_err = _ensure_hand_eye()
    if calib_err:
        return calib_err

    offset = _parse_xyz_triplet(
        approach_offset_m,
        _parse_xyz_triplet(REACH_APPROACH_OFFSET_M, [0.0, 0.0, 0.05]),
    )
    done_dist = max(0.003, min(float(reach_done_dist_m), 0.05))

    for i in range(n_steps):
        st = robot.get_full_state()
        tcp_pose = st.get("tcp_pose") or [0.3, 0.0, 0.3, 0.0, 0.0, 0.0]
        rgbd = cam.capture_rgbd()
        frame = rgbd["color"]
        depth = rgbd.get("depth")
        if depth is None:
            return {
                "status": "error",
                "reason": "Depth stream unavailable. Set CAMERA_DEPTH_ENABLED=true.",
                "trace": trace,
            }

        meta = _detector.detect(frame)
        tgt = _pick_detection_object(meta, target_label)
        if tgt is None:
            return {"status": "error", "reason": "No detected target in frame.", "trace": trace}

        est = estimate_object_target_base(
            obj=tgt,
            depth_image=depth,
            depth_scale=float(rgbd.get("depth_scale", 0.001)),
            intrinsics=rgbd.get("intrinsics") or {},
            calib=calib,
            tcp_pose=tcp_pose if calib.mount == "eye_in_hand" else None,
            approach_offset_m=offset,
        )
        if est is None or "target_base_m" not in est:
            reason = (est or {}).get("error", "Could not estimate 3D target (depth/intrinsics).")
            return {"status": "error", "reason": reason, "trace": trace}

        target_xyz = [float(v) for v in est["target_base_m"]]
        step = _reach_runner.step(st, target_xyz, max_step_m=max_step_m)
        dist = (
            (target_xyz[0] - tcp_pose[0]) ** 2
            + (target_xyz[1] - tcp_pose[1]) ** 2
            + (target_xyz[2] - tcp_pose[2]) ** 2
        ) ** 0.5
        info = {
            "step": i + 1,
            "task": "camera_reach_3d",
            "target_label": tgt.get("label"),
            "target_tcp": [round(v, 4) for v in target_xyz],
            "dist_m": round(float(dist), 5),
            "pose3d": est,
            "policy_source": step.source,
            "action": {"dx": round(step.dx, 5), "dy": round(step.dy, 5), "dz": round(step.dz, 5)},
        }
        trace.append(info)

        if step.source == "done" or dist < done_dist:
            return {
                "status": "done",
                "task_id": task_id,
                "steps_used": i + 1,
                "trace": trace,
                "note": "Reached 3D target from camera detection.",
            }

        report = robot.move_tcp_relative(dx=step.dx, dy=step.dy, dz=step.dz)
        info["motion_report"] = report
        time.sleep(settle_sec)

    return {
        "status": "done",
        "task_id": task_id,
        "steps_used": n_steps,
        "trace": trace,
        "note": "Max steps reached in 3D camera reach; re-detect or increase steps.",
    }


def approach_object_once(
    robot: RobotDriver,
    target_label: str = "",
    step_m: float = 0.03,
    approach_offset_m: list | None = None,
    mode: str = "image",
) -> dict:
    """
    Direction check — EXACTLY one short move, then stop.

    mode=image (default): move left/right/forward/back from where the object
    sits in the camera frame (does not use hand-eye). Use this while verifying
    directions.

    mode=3d: move toward target_base_m from depth + hand-eye (needs good calib).
    """
    print(f"  [TOOL] approach_object_once label={target_label!r} step_m={step_m} mode={mode}")
    cam, err = _ensure_camera()
    if err:
        return err
    if _detector is None:
        return {"status": "error", "reason": "Object detector unavailable."}

    step_cap = max(0.005, min(float(step_m), 0.05))
    mode_l = (mode or "image").strip().lower()
    if mode_l not in ("image", "3d"):
        mode_l = "image"

    st = robot.get_full_state()
    tcp_pose = st.get("tcp_pose") or [0.3, 0.0, 0.3, 0.0, 0.0, 0.0]
    tcp_before = [float(v) for v in tcp_pose[:3]]

    try:
        rgbd = cam.capture_rgbd()
    except Exception as e:
        return {"status": "error", "reason": f"camera capture failed: {e}"}

    frame = rgbd["color"]
    h, w = frame.shape[:2]
    # Use NanoOWL if available, fall back to YOLO.
    if not hasattr(approach_object_once, "_owl"):
        try:
            from camera.nanoowl_detector import NanoOwlDetector
            approach_object_once._owl = NanoOwlDetector()
        except Exception:
            approach_object_once._owl = None
    owl = getattr(approach_object_once, "_owl", None)
    if owl is not None and owl._ensure():
        queries = [target_label] if target_label else None
        meta = owl.detect(frame, queries=queries)
    else:
        meta = _detector.detect(frame)
    tgt = _pick_detection_object(meta, target_label)
    if tgt is None:
        return {
            "status": "error",
            "reason": "No matching object in frame.",
            "labels_seen": meta.get("labels") or [],
        }

    center = tgt.get("center") or {}
    u = float(center.get("x", w / 2))
    v = float(center.get("y", h / 2))
    cx = w / 2.0
    cy = h / 2.0
    du = u - cx  # + = object to the right in the image
    dv = v - cy  # + = object lower in the image
    dead = 12.0  # pixels — ignore tiny offsets
    # Prefer L/R whenever horizontal error is meaningful (bias vs F/B).
    # Stops "object on the right but a bit high" from becoming move_forward.
    prefer_lr = abs(du) >= dead and (abs(du) >= abs(dv) * 0.4 or abs(dv) < dead)

    if mode_l == "image":
        # Map image error → operator directions (uses MOTION_*_VEC).
        if abs(du) < dead and abs(dv) < dead:
            return {
                "status": "done",
                "mode": "image_direction_check",
                "note": "Object already near image center — no move.",
                "label": tgt.get("label"),
                "pixel": {"u": round(u, 1), "v": round(v, 1)},
                "image_center": {"cx": cx, "cy": cy},
                "du_px": round(du, 1),
                "dv_px": round(dv, 1),
                "tcp_before": [round(x, 4) for x in tcp_before],
                "closer": None,
            }

        if prefer_lr:
            go_right = (du > 0) ^ APPROACH_IMAGE_INVERT_LR
            tool_name = "move_right" if go_right else "move_left"
            axis = MOTION_RIGHT_VEC if go_right else MOTION_LEFT_VEC
            why = (
                f"object is {'right' if du > 0 else 'left'} of image center by {abs(du):.0f}px"
                f"{' (L/R inverted)' if APPROACH_IMAGE_INVERT_LR else ''}"
            )
        else:
            go_back = (dv > 0) ^ APPROACH_IMAGE_INVERT_FB
            tool_name = "move_backward" if go_back else "move_forward"
            axis = MOTION_BACKWARD_VEC if go_back else MOTION_FORWARD_VEC
            why = (
                f"object is {'lower' if dv > 0 else 'higher'} in image by {abs(dv):.0f}px"
                f"{' (F/B inverted)' if APPROACH_IMAGE_INVERT_FB else ''}"
            )

        dx, dy, dz = axis[0] * step_cap, axis[1] * step_cap, axis[2] * step_cap
        report = robot.move_tcp_relative(dx=dx, dy=dy, dz=dz)
        st2 = robot.get_full_state()
        tcp_after = [float(v) for v in (st2.get("tcp_pose") or tcp_pose)[:3]]

        return {
            "status": "done",
            "mode": "image_direction_check",
            "label": tgt.get("label"),
            "confidence": tgt.get("confidence"),
            "pixel": {"u": round(u, 1), "v": round(v, 1)},
            "du_px": round(du, 1),
            "dv_px": round(dv, 1),
            "decision": tool_name,
            "why": why,
            "step_commanded_cm": [round(dx * 100, 1), round(dy * 100, 1), round(dz * 100, 1)],
            "tcp_before": [round(x, 4) for x in tcp_before],
            "tcp_after": [round(x, 4) for x in tcp_after],
            "motion_report": report,
            "note": (
                "ONE image-based step (hand-eye not used). "
                "If the arm moved the wrong way vs the object on screen, tell us which way. "
                "Ask again for another step."
            ),
        }

    # ── mode=3d (hand-eye) ─────────────────────────────────
    calib, calib_err = _ensure_hand_eye()
    if calib_err:
        return calib_err
    depth = rgbd.get("depth")
    if depth is None:
        return {"status": "error", "reason": "Depth unavailable. Set CAMERA_DEPTH_ENABLED=true."}

    offset = _parse_xyz_triplet(
        approach_offset_m,
        _parse_xyz_triplet(REACH_APPROACH_OFFSET_M, [0.0, 0.0, 0.05]),
    )
    est = estimate_object_target_base(
        obj=tgt,
        depth_image=depth,
        depth_scale=float(rgbd.get("depth_scale", 0.001)),
        intrinsics=rgbd.get("intrinsics") or {},
        calib=calib,
        tcp_pose=tcp_pose if calib.mount == "eye_in_hand" else None,
        approach_offset_m=offset,
    )
    if est is None or "target_base_m" not in est:
        return {
            "status": "error",
            "reason": (est or {}).get("error", "Could not estimate 3D target."),
            "label": tgt.get("label"),
        }

    target = [float(v) for v in est["target_base_m"][:3]]
    err_vec = [target[i] - tcp_before[i] for i in range(3)]
    dist_before = sum(v * v for v in err_vec) ** 0.5
    if dist_before < 0.008:
        return {
            "status": "done",
            "mode": "3d_direction_check",
            "note": "Already within 8 mm of 3D target — no move.",
            "label": tgt.get("label"),
            "tcp_before": [round(v, 4) for v in tcp_before],
            "target_base_m": [round(v, 4) for v in target],
            "dist_before_m": round(dist_before, 5),
            "closer": None,
        }

    scale = min(1.0, step_cap / dist_before)
    dx, dy, dz = err_vec[0] * scale, err_vec[1] * scale, err_vec[2] * scale
    report = robot.move_tcp_relative(dx=dx, dy=dy, dz=dz)

    st2 = robot.get_full_state()
    tcp_after = [float(v) for v in (st2.get("tcp_pose") or tcp_pose)[:3]]
    dist_after = sum((target[i] - tcp_after[i]) ** 2 for i in range(3)) ** 0.5
    closer = dist_after < dist_before - 0.001

    return {
        "status": "done",
        "mode": "3d_direction_check",
        "label": tgt.get("label"),
        "confidence": tgt.get("confidence"),
        "tcp_before": [round(v, 4) for v in tcp_before],
        "target_base_m": [round(v, 4) for v in target],
        "delta_toward_target_cm": [round(v * 100, 1) for v in err_vec],
        "step_commanded_cm": [round(dx * 100, 1), round(dy * 100, 1), round(dz * 100, 1)],
        "tcp_after": [round(v, 4) for v in tcp_after],
        "dist_before_cm": round(dist_before * 100, 1),
        "dist_after_cm": round(dist_after * 100, 1),
        "closer": closer,
        "pose3d": est,
        "motion_report": report,
        "note": (
            "ONE 3D step. closer=true means hand-eye direction is usable. "
            "Prefer mode=image until hand-eye is recalibrated."
        ),
    }


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

    if name in ("move_joint", "jog_joint") and "speed" in inputs:
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
        "jog_joint": lambda: jog_joint(robot, **inputs),
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
        "detect_objects_nanoowl": lambda: detect_objects_nanoowl(robot, **inputs),
        "go_to_object": lambda: go_to_object(robot, **inputs),
        "approach_object_once": lambda: approach_object_once(robot, **inputs),
        "execute_rl_policy": lambda: execute_rl_policy(robot, **inputs),
    }
    fn = dispatch.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    result = fn()
    if isinstance(result, dict) and result.get("status") != "error":
        if name in MOTION_TOOLS:
            if name == "approach_object_once":
                if "step_commanded_cm" in result:
                    policy.record_motion()
            else:
                policy.record_motion()
        elif name in GRIPPER_TOOLS:
            policy.record_action()
    return result


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
        "name": "jog_joint",
        "description": "Relative joint jog for manual control: joint 1–6, delta_deg in degrees.",
        "input_schema": {
            "type": "object",
            "properties": {
                "joint": {"type": "integer", "minimum": 1, "maximum": 6},
                "delta_deg": {"type": "number", "description": "Degrees to add (+ or −)"},
                "speed": {"type": "number"},
                "acceleration": {"type": "number"},
            },
            "required": ["joint", "delta_deg"],
        },
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
        "description": "Move TCP left along the table plane (straight Cartesian move, gripper-aligned). distance_m in meters.",
        "input_schema": {
            "type": "object",
            "properties": {"distance_m": {"type": "number"}},
            "required": ["distance_m"],
        },
    },
    {
        "name": "move_right",
        "description": "Move TCP right along the table plane (straight Cartesian move, gripper-aligned). distance_m in meters.",
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
        "name": "detect_objects_nanoowl",
        "description": (
            "Open-vocabulary object detection using NVIDIA NanoOWL on Thor GPU. "
            "Unlike detect_objects (fixed YOLO classes), this accepts ANY text query — "
            "'cup', 'red screwdriver', 'robot gripper'. Use when the object is not a standard COCO class "
            "or when the user names an object that YOLO might miss. "
            "Returns labels, bounding boxes, confidence scores, and (with depth + hand-eye) target_base_m."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Single text query, e.g. 'cup' or 'red bottle, screwdriver' (comma-separated)",
                },
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of text queries, e.g. ['cup', 'wrench']",
                },
                "threshold": {
                    "type": "number",
                    "description": "Confidence threshold 0–1 (default 0.15). Lower = more detections.",
                },
                "save_image": {
                    "type": "boolean",
                    "description": "Save annotated JPEG with detection boxes",
                },
                "label_filter": {
                    "type": "string",
                    "description": "Optional substring filter on returned objects",
                },
            },
            "required": [],
        },
    },
    {
        "name": "detect_objects",
        "description": (
            "Capture a live camera frame and run object detection (YOLO or contour fallback). "
            "Returns labels, bounding boxes, and (with depth + hand-eye calib) target_base_m in meters."
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
    {
        "name": "go_to_object",
        "description": (
            "Detects the object using depth camera + hand-eye calibration and moves the TCP "
            "directly to it in ONE single moveL command. "
            "NOTE: requires accurate hand-eye calibration. If the robot moves the wrong way, "
            "use approach_object_once instead (image-based, more reliable)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_label": {
                    "type": "string",
                    "description": "Object name to move to, e.g. 'bottle', 'cup', 'red screwdriver'",
                },
                "approach_offset_m": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x, y, z] offset in meters from object center (default [0, 0, 0.10] = 10cm above)",
                },
                "speed": {
                    "type": "number",
                    "description": "Linear speed m/s (default 0.15)",
                },
            },
            "required": ["target_label"],
        },
    },
    {
        "name": "approach_object_once",
        "description": (
            "PREFERRED for 'go to bottle/cup/object', 'move toward X'. "
            "Detects the object and takes ONE step toward it using image-based direction — "
            "reliable regardless of hand-eye calibration quality. "
            "Call repeatedly (5-10 times) to walk the robot to the object. "
            "Use step_m=0.05 for faster approach. Default mode=image."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_label": {
                    "type": "string",
                    "description": "Object name substring, e.g. bottle, cup, tv",
                },
                "step_m": {
                    "type": "number",
                    "description": "Max step length in meters (default 0.03 = 3 cm, max 0.05)",
                },
                "mode": {
                    "type": "string",
                    "description": "image (default, pixel direction) or 3d (hand-eye target)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "execute_rl_policy",
        "description": (
            "Multi-step RL reach loop — only when the user explicitly asks for continuous "
            "reach / trajectory. Prefer approach_object_once for single direction checks. "
            "camera_reach detects object then steps until near target."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Policy task id, e.g. reach_free_space or camera_reach"},
                "steps": {"type": "integer", "minimum": 1, "maximum": 100},
                "target_label": {"type": "string", "description": "Preferred detection label (optional, camera mode)"},
                "policy_path": {"type": "string", "description": "Optional trained policy checkpoint path"},
                "target_tcp": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional state-reach target [x,y,z] in meters (task_id=reach_free_space)",
                },
                "max_step_m": {"type": "number", "description": "Max Cartesian step per control tick (meters)"},
                "settle_sec": {"type": "number", "description": "Pause between control steps in seconds"},
            },
            "required": [],
        },
    },
]
