#!/usr/bin/env python3
"""Capture synchronized camera frames and robot state metadata."""

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera import RealSenseCamera
from robot.factory import create_robot


def parse_args():
    p = argparse.ArgumentParser(description="Capture image + robot_state dataset samples")
    p.add_argument("--count", type=int, default=50, help="Number of samples to capture")
    p.add_argument("--interval", type=float, default=0.5, help="Seconds between samples")
    p.add_argument("--session-id", default=dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--robot", choices=["ur5", "openarm"], default="ur5")
    p.add_argument("--dry-run", action="store_true", help="Use mock robot state")
    p.add_argument("--host", default=None, help="Robot host override")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Session output dir (default: data/raw/robot_sessions/<session-id>)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    default_output = project_root / "data" / "raw" / "robot_sessions" / args.session_id
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output
    images_dir = out_dir / "images"
    meta_dir = out_dir / "meta"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    print(f"Session: {args.session_id}")
    print(f"Output:  {out_dir}")

    robot = create_robot(robot_type=args.robot, dry_run=args.dry_run, host=args.host)
    cam = RealSenseCamera(output_dir=str(images_dir))
    manifest_path = out_dir / "manifest.jsonl"

    captured = 0
    try:
        robot.connect()
        cam.connect()
        print("Connected robot + camera. Starting capture...\n")

        for i in range(args.count):
            t0 = time.time()
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            sample_id = f"{i:04d}_{stamp}"

            capture = cam.save_color_frame(session_id=args.session_id, prefix=sample_id)
            state = robot.get_full_state()

            meta = {
                "sample_id": sample_id,
                "timestamp_iso": dt.datetime.now().isoformat(),
                "image_path": capture["path"],
                "image_shape": capture.get("shape"),
                "camera": capture.get("camera"),
                "camera_serial": capture.get("serial"),
                "robot_state": state,
            }
            meta_path = meta_dir / f"{sample_id}.json"
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            with manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"sample_id": sample_id, "meta_path": str(meta_path)}) + "\n")

            captured += 1
            print(f"[{i + 1}/{args.count}] saved image+meta: {sample_id}")

            elapsed = time.time() - t0
            sleep_s = max(0.0, args.interval - elapsed)
            if i < args.count - 1 and sleep_s > 0:
                time.sleep(sleep_s)

        print(f"\nDone. Captured {captured} samples.")
        print(f"Manifest: {manifest_path}")
    finally:
        cam.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
