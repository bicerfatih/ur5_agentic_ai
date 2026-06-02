#!/usr/bin/env python3
"""Quick Intel RealSense camera capture test."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera import RealSenseCamera


def main():
    cam = RealSenseCamera()
    try:
        cam.connect()
        result = cam.save_color_frame(session_id="camera_test", prefix="realsense")
        print("Capture OK")
        print(result)
    finally:
        cam.disconnect()


if __name__ == "__main__":
    main()
