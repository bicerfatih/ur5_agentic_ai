import datetime as dt
import os
from typing import Any

import numpy as np

from config.settings import (
    CAMERA_DEPTH_ENABLED,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_OUTPUT_DIR,
    CAMERA_SERIAL,
    CAMERA_WIDTH,
)


class RealSenseCamera:
    def __init__(
        self,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
        serial: str = CAMERA_SERIAL,
        output_dir: str = CAMERA_OUTPUT_DIR,
        depth_enabled: bool = CAMERA_DEPTH_ENABLED,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.output_dir = output_dir
        self.depth_enabled = bool(depth_enabled)
        self._pipeline = None
        self._profile = None
        self._align = None
        self._depth_scale = 0.001
        self._intrinsics: dict[str, float] = {}

    def connect(self):
        try:
            import pyrealsense2 as rs
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "pyrealsense2 not installed. Install Intel RealSense SDK Python bindings."
            ) from e

        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.depth_enabled:
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self._pipeline = rs.pipeline()
        self._profile = self._pipeline.start(config)
        if self.depth_enabled:
            self._align = rs.align(rs.stream.color)
            depth_sensor = self._profile.get_device().first_depth_sensor()
            self._depth_scale = float(depth_sensor.get_depth_scale())
        self._intrinsics = self._read_color_intrinsics()

    def disconnect(self):
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
            self._profile = None
            self._align = None

    def _ensure_connected(self):
        if self._pipeline is None:
            self.connect()

    def _read_color_intrinsics(self) -> dict[str, float]:
        if self._profile is None:
            return {}
        try:
            import pyrealsense2 as rs

            stream = self._profile.get_stream(rs.stream.color)
            intr = stream.as_video_stream_profile().get_intrinsics()
            return {
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "cx": float(intr.ppx),
                "cy": float(intr.ppy),
            }
        except Exception:
            return {}

    def get_intrinsics(self) -> dict[str, float]:
        self._ensure_connected()
        return dict(self._intrinsics)

    def capture_color_frame(self, timeout_ms: int = 5000) -> np.ndarray:
        self._ensure_connected()
        frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("No color frame received from RealSense camera.")
        return np.asanyarray(color.get_data())

    def capture_rgbd(self, timeout_ms: int = 5000) -> dict[str, Any]:
        """Aligned color + depth frame with camera intrinsics."""
        self._ensure_connected()
        if not self.depth_enabled:
            color = self.capture_color_frame(timeout_ms=timeout_ms)
            return {
                "color": color,
                "depth": None,
                "depth_scale": self._depth_scale,
                "intrinsics": self.get_intrinsics(),
            }

        frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame:
            raise RuntimeError("No color frame received from RealSense camera.")
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()) if depth_frame else None
        return {
            "color": color,
            "depth": depth,
            "depth_scale": self._depth_scale,
            "intrinsics": self.get_intrinsics(),
        }

    def save_color_frame(self, session_id: str = "session", prefix: str = "frame") -> dict[str, Any]:
        self._ensure_connected()
        frame = self.capture_color_frame()
        os.makedirs(self.output_dir, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{session_id}_{stamp}.jpg"
        path = os.path.abspath(os.path.join(self.output_dir, filename))

        try:
            import cv2
        except ModuleNotFoundError as e:
            raise RuntimeError("opencv-python is required to save camera frames as JPEG.") from e

        ok = cv2.imwrite(path, frame)
        if not ok:
            raise RuntimeError(f"Failed to save frame to {path}")
        return {
            "status": "done",
            "path": path,
            "shape": list(frame.shape),
            "camera": "intel_realsense",
            "serial": self.serial or "auto",
        }
