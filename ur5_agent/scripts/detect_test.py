#!/usr/bin/env python3
"""Quick CLI test for detect_objects (camera + YOLO/fallback)."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.factory import create_robot
from robot.tools import detect_objects


def main():
    p = argparse.ArgumentParser(description="Run one detect_objects call")
    p.add_argument("--save-image", action="store_true", help="Save annotated JPEG")
    p.add_argument("--label-filter", default="", help="Filter labels by substring")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    robot = create_robot(robot_type="ur5", dry_run=args.dry_run)
    try:
        robot.connect()
        result = detect_objects(
            robot,
            save_image=args.save_image,
            label_filter=args.label_filter,
        )
        print(json.dumps(result, indent=2))
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
