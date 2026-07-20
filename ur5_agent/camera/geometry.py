"""Camera geometry: depth sampling, intrinsics, and hand-eye transforms."""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


class HandEyeCalibration:
    """Load static hand-eye calibration (eye-to-hand or eye-in-hand)."""

    def __init__(self, path: str = ""):
        self.path = (path or "").strip()
        self.mount = "eye_to_hand"
        self.T_base_camera = np.eye(4, dtype=np.float64)
        self.T_tool_camera = np.eye(4, dtype=np.float64)
        self.intrinsics: dict[str, float] = {}
        self.loaded = False
        if self.path and os.path.isfile(self.path):
            self._load(self.path)

    def _load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.mount = str(data.get("mount", "eye_to_hand")).strip().lower()
        intr = data.get("intrinsics") or {}
        self.intrinsics = {
            "fx": float(intr.get("fx", 0.0)),
            "fy": float(intr.get("fy", 0.0)),
            "cx": float(intr.get("cx", 0.0)),
            "cy": float(intr.get("cy", 0.0)),
        }
        if "T_base_camera" in data:
            self.T_base_camera = np.asarray(data["T_base_camera"], dtype=np.float64).reshape(4, 4)
        elif "translation_m" in data and "rotation_rpy_rad" in data:
            t = np.asarray(data["translation_m"], dtype=np.float64).reshape(3)
            rpy = np.asarray(data["rotation_rpy_rad"], dtype=np.float64).reshape(3)
            rot = _rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
            self.T_base_camera = np.eye(4, dtype=np.float64)
            self.T_base_camera[:3, :3] = rot
            self.T_base_camera[:3, 3] = t
        if "T_tool_camera" in data:
            self.T_tool_camera = np.asarray(data["T_tool_camera"], dtype=np.float64).reshape(4, 4)
        self.loaded = True

    def merge_intrinsics(self, runtime: dict[str, float] | None) -> dict[str, float]:
        out = dict(self.intrinsics)
        if runtime:
            for k, v in runtime.items():
                if v:
                    out[k] = float(v)
        return out

    def camera_point_to_base(
        self,
        point_camera_m: np.ndarray,
        tcp_pose: list[float] | None = None,
    ) -> np.ndarray:
        p = np.ones(4, dtype=np.float64)
        p[:3] = np.asarray(point_camera_m, dtype=np.float64).reshape(3)
        if self.mount == "eye_in_hand":
            if tcp_pose is None or len(tcp_pose) < 6:
                raise ValueError("eye_in_hand calibration requires tcp_pose [x,y,z,rx,ry,rz].")
            T_base_tool = _tcp_pose_to_matrix(tcp_pose)
            T = T_base_tool @ self.T_tool_camera
        else:
            T = self.T_base_camera
        out = T @ p
        return out[:3]


def _tcp_pose_to_matrix(tcp_pose: list[float]) -> np.ndarray:
    x, y, z, rx, ry, rz = [float(v) for v in tcp_pose[:6]]
    rot = _rpy_to_matrix(rx, ry, rz)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = [x, y, z]
    return T


def pixel_depth_to_camera_xyz(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: dict[str, float],
) -> np.ndarray | None:
    if depth_m <= 0.0 or not math.isfinite(depth_m):
        return None
    fx = float(intrinsics.get("fx", 0.0))
    fy = float(intrinsics.get("fy", 0.0))
    cx = float(intrinsics.get("cx", 0.0))
    cy = float(intrinsics.get("cy", 0.0))
    if fx <= 0.0 or fy <= 0.0:
        return None
    x = (float(u) - cx) * depth_m / fx
    y = (float(v) - cy) * depth_m / fy
    z = depth_m
    return np.array([x, y, z], dtype=np.float64)


def sample_depth_m(
    depth_image: np.ndarray,
    u: int,
    v: int,
    depth_scale: float = 0.001,
    window: int = 5,
) -> float | None:
    """Median depth in meters at pixel (u, v) using a small window."""
    if depth_image is None or depth_image.size == 0:
        return None
    h, w = depth_image.shape[:2]
    u = int(max(0, min(w - 1, u)))
    v = int(max(0, min(h - 1, v)))
    r = max(1, int(window) // 2)
    y0, y1 = max(0, v - r), min(h, v + r + 1)
    x0, x1 = max(0, u - r), min(w, u + r + 1)
    patch = depth_image[y0:y1, x0:x1].astype(np.float64).reshape(-1)
    patch = patch[patch > 0]
    if patch.size == 0:
        return None
    return float(np.median(patch) * float(depth_scale))


def estimate_object_target_base(
    obj: dict[str, Any],
    depth_image: np.ndarray,
    depth_scale: float,
    intrinsics: dict[str, float],
    calib: HandEyeCalibration,
    tcp_pose: list[float] | None = None,
    approach_offset_m: list[float] | None = None,
) -> dict[str, Any] | None:
    """
    Convert detection center + depth to target XYZ in robot base frame.
    approach_offset_m is added in base frame after transform (e.g. hover above object).
    """
    center = obj.get("center") or {}
    u = int(center.get("x", 0))
    v = int(center.get("y", 0))
    depth_m = sample_depth_m(depth_image, u, v, depth_scale=depth_scale)
    if depth_m is None:
        return None

    K = calib.merge_intrinsics(intrinsics)
    p_cam = pixel_depth_to_camera_xyz(u, v, depth_m, K)
    if p_cam is None:
        return None

    try:
        p_base = calib.camera_point_to_base(p_cam, tcp_pose=tcp_pose)
    except ValueError as e:
        return {"error": str(e)}

    offset = np.zeros(3, dtype=np.float64)
    if approach_offset_m:
        offset = np.asarray(approach_offset_m[:3], dtype=np.float64)
    target = p_base + offset

    return {
        "pixel": {"u": u, "v": v},
        "depth_m": round(float(depth_m), 5),
        "point_camera_m": [round(float(v), 5) for v in p_cam],
        "point_base_m": [round(float(v), 5) for v in p_base],
        "target_base_m": [round(float(v), 5) for v in target],
    }
