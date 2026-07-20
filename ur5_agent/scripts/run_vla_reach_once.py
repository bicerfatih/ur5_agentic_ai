#!/usr/bin/env python3
"""Run VLA-guided 3D reach on real robot (camera + language + motion loop)."""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import REACH_APPROACH_OFFSET_M, RL_CONTROL_DT, VLA_INSTRUCTION_DEFAULT
from config.sites import get_site
from policy.safety import PolicyEngine
from robot.factory import create_robot
from robot.tools import _ensure_camera, _ensure_hand_eye, _parse_xyz_triplet, _pick_detection_object, estimate_object_target_base
from robot import tools as toolmod
from vla.adapter import VLAPolicyAdapter


def main():
    p = argparse.ArgumentParser(description="VLA + 3D camera reach loop")
    p.add_argument("--live", action="store_true")
    p.add_argument("--site", default="lab")
    p.add_argument("--instruction", default=VLA_INSTRUCTION_DEFAULT)
    p.add_argument("--target-label", default="")
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--max-step-m", type=float, default=0.008)
    p.add_argument("--backend", default="", help="tool_routed | openvla | pi0 | groot (default: VLA_BACKEND env)")
    p.add_argument("--server-url", default="", help="VLA server, e.g. http://gpu-host:8000 (default: VLA_SERVER_URL env)")
    args = p.parse_args()

    robot = create_robot(dry_run=not args.live)
    site = get_site(args.site)
    policy = PolicyEngine(site=site)
    vla = VLAPolicyAdapter(backend=args.backend, server_url=args.server_url)
    print({"vla_backend": vla.backend, "server_url": vla.server_url or None, "remote_ready": vla._remote is not None})

    robot.connect()
    cam, err = _ensure_camera()
    if err:
        print(err)
        sys.exit(1)
    calib, cerr = _ensure_hand_eye()
    if cerr:
        print(cerr)
        sys.exit(1)
    if toolmod._detector is None:
        print({"status": "error", "reason": "detector unavailable"})
        sys.exit(1)

    offset = _parse_xyz_triplet(REACH_APPROACH_OFFSET_M, [0.0, 0.0, 0.05])
    trace = []

    try:
        for i in range(args.steps):
            st = robot.get_full_state()
            tcp_pose = st.get("tcp_pose") or [0.3, 0.0, 0.3, 0.0, 0.0, 0.0]
            rgbd = cam.capture_rgbd()
            frame = rgbd["color"]
            depth = rgbd.get("depth")
            if depth is None:
                print({"status": "error", "reason": "no depth"})
                break
            meta = toolmod._detector.detect(frame)
            tgt = _pick_detection_object(meta, args.target_label)
            if tgt is None:
                print({"status": "error", "reason": "no detection"})
                break
            est = estimate_object_target_base(
                obj=tgt,
                depth_image=depth,
                depth_scale=float(rgbd.get("depth_scale", 0.001)),
                intrinsics=rgbd.get("intrinsics") or {},
                calib=calib,
                tcp_pose=tcp_pose if calib.mount == "eye_in_hand" else None,
                approach_offset_m=offset,
            )
            if not est or "target_base_m" not in est:
                print({"status": "error", "reason": "pose3d failed", "est": est})
                break
            target_xyz = est["target_base_m"]
            step = vla.step(frame, args.instruction, st, target_xyz, max_step_m=args.max_step_m)
            step["step"] = i + 1
            trace.append(step)
            if step.get("done"):
                print({"status": "done", "trace": trace})
                break
            if not args.live:
                print({"status": "dry_run", "step": step})
                continue
            report = robot.move_tcp_relative(
                dx=float(step.get("dx", 0)),
                dy=float(step.get("dy", 0)),
                dz=float(step.get("dz", 0)),
            )
            step["motion_report"] = report
            time.sleep(RL_CONTROL_DT)
        else:
            print({"status": "done", "note": "max steps", "trace": trace})
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
