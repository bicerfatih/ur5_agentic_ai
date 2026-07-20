#!/usr/bin/env python3
"""
Record human reach demonstrations while you teleop the arm (ops console or pendant).

Polls robot TCP at a fixed rate. When the tool moves, saves (observation, action=delta_tcp).
Use camera + hand-eye calib to attach a 3D target from detect_objects each sample.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CAMERA_TYPE, REACH_APPROACH_OFFSET_M
from il.demo_store import DemoEpisode, DemoTransition, new_episode_id, save_episode
from il.obs_action import build_reach_obs, tcp_delta_action
from robot.factory import create_robot


def _parse_target(raw: str) -> list[float] | None:
    txt = (raw or "").strip()
    if not txt:
        return None
    parts = [p.strip() for p in txt.split(",")]
    if len(parts) != 3:
        raise ValueError("--target must be x,y,z in meters")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def _parse_offset(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) >= 3:
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    return [0.0, 0.0, 0.05]


def _estimate_target(robot, target_label: str, approach_offset_m: list[float]) -> tuple[list[float] | None, str]:
    if CAMERA_TYPE != "realsense":
        return None, "no_camera"
    from robot import tools as toolmod

    cam, err = toolmod._ensure_camera()
    if err:
        return None, err.get("reason", "camera_error")
    calib, cerr = toolmod._ensure_hand_eye()
    if cerr:
        return None, cerr.get("reason", "calib_error")
    if toolmod._detector is None:
        return None, "detector_unavailable"

    st = robot.get_full_state()
    tcp_pose = st.get("tcp_pose") or [0.3, 0.0, 0.3, 0.0, 0.0, 0.0]
    rgbd = cam.capture_rgbd()
    depth = rgbd.get("depth")
    if depth is None:
        return None, "no_depth"
    meta = toolmod._detector.detect(rgbd["color"])
    obj = toolmod._pick_detection_object(meta, target_label)
    if obj is None:
        return None, "no_detection"
    est = toolmod.estimate_object_target_base(
        obj=obj,
        depth_image=depth,
        depth_scale=float(rgbd.get("depth_scale", 0.001)),
        intrinsics=rgbd.get("intrinsics") or {},
        calib=calib,
        tcp_pose=tcp_pose if calib.mount == "eye_in_hand" else None,
        approach_offset_m=approach_offset_m,
    )
    if est is None or "target_base_m" not in est:
        return None, (est or {}).get("error", "pose3d_failed")
    return [float(v) for v in est["target_base_m"]], str(obj.get("label", ""))


def main():
    p = argparse.ArgumentParser(description="Record human reach demos (teleop while polling TCP).")
    p.add_argument("--live", action="store_true", help="Use live UR5 (required for real demos)")
    p.add_argument("--duration-sec", type=float, default=120.0)
    p.add_argument("--poll-hz", type=float, default=10.0)
    p.add_argument("--min-delta-mm", type=float, default=1.5, help="Min TCP move to record a transition")
    p.add_argument("--target-label", default="", help="YOLO label filter for 3D target")
    p.add_argument("--target", default="", help="Fixed target x,y,z (m) if no camera/calib")
    p.add_argument("--approach-offset", default=REACH_APPROACH_OFFSET_M)
    p.add_argument("--episode-id", default="")
    p.add_argument("--out-dir", default="")
    args = p.parse_args()

    if not args.live:
        print("Use --live on the real robot. Mock mode cannot record human teleop.")
        sys.exit(1)

    fixed_target = _parse_target(args.target)
    offset = _parse_offset(args.approach_offset)
    min_delta = max(0.0005, args.min_delta_mm / 1000.0)
    period = 1.0 / max(1.0, args.poll_hz)

    robot = create_robot(dry_run=False)
    robot.connect()

    episode_id = args.episode_id or new_episode_id("reach")
    transitions: list[DemoTransition] = []
    label = args.target_label
    target_xyz: list[float] | None = fixed_target
    target_note = ""

    print("Recording demonstration.")
    print("  Teleop with ops console (arrows / joint jog) or teach pendant.")
    print("  Ctrl+C to stop early.")
    print(f"  Episode id: {episode_id}")

    try:
        st0 = robot.get_full_state()
        last_tcp = list(st0.get("tcp_pose") or [0.3, 0.0, 0.3, 0.0, 0.0, 0.0])
        t_end = time.time() + max(5.0, args.duration_sec)

        while time.time() < t_end:
            if args.target_label:
                est, note = _estimate_target(robot, args.target_label, offset)
                if est is not None:
                    target_xyz = est
                    label = note or label
                else:
                    target_note = note
            elif target_xyz is None and fixed_target is not None:
                target_xyz = fixed_target

            if target_xyz is None:
                time.sleep(period)
                continue

            st = robot.get_full_state()
            tcp = list(st.get("tcp_pose") or last_tcp)
            delta = tcp_delta_action(last_tcp, tcp)
            if float(abs(delta).sum()) < 1e-9:
                time.sleep(period)
                continue

            mag = float((delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2) ** 0.5)
            if mag >= min_delta:
                obs = build_reach_obs(st, target_xyz)
                transitions.append(
                    DemoTransition(
                        obs=[float(v) for v in obs],
                        action=[float(delta[0]), float(delta[1]), float(delta[2])],
                        tcp_pose=[float(v) for v in tcp[:6]],
                        target_xyz=[float(v) for v in target_xyz],
                        label=label,
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    )
                )
                print(
                    f"  + transition #{len(transitions)} "
                    f"delta_mm={[round(v * 1000, 2) for v in delta]} target={target_xyz}"
                )
                last_tcp = tcp
            time.sleep(period)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        robot.disconnect()

    if not transitions:
        print(
            json.dumps(
                {
                    "status": "error",
                    "reason": "No transitions recorded. Move the arm while this script runs.",
                    "target_note": target_note,
                },
                indent=2,
            )
        )
        sys.exit(1)

    episode = DemoEpisode(
        episode_id=episode_id,
        task="reach_3d" if args.target_label else "reach_fixed",
        transitions=transitions,
        meta={
            "target_label": args.target_label,
            "fixed_target": fixed_target,
            "approach_offset_m": offset,
            "n_transitions": len(transitions),
        },
    )
    path = save_episode(episode, out_dir=args.out_dir or None)
    print(json.dumps({"status": "done", "path": path, "transitions": len(transitions)}, indent=2))


if __name__ == "__main__":
    main()
