"""Persistent taught home joint pose (overrides default HOME_JOINTS)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from config.settings import HOME_JOINTS, HOME_POSE_PATH


def get_home_joints() -> list[float]:
    """Return taught home joints in radians, or settings default."""
    path = HOME_POSE_PATH
    if not path or not os.path.isfile(path):
        return list(HOME_JOINTS)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        joints = data.get("joints_rad") or data.get("home_joints")
        if isinstance(joints, list) and len(joints) >= 6:
            return [float(joints[i]) for i in range(6)]
    except Exception:
        pass
    return list(HOME_JOINTS)


def save_home_joints(joints_rad: list[float], tcp_pose: list | None = None) -> dict:
    """Persist current joint pose as home. Returns saved payload."""
    if len(joints_rad) < 6:
        raise ValueError("need 6 joint angles")
    joints = [round(float(joints_rad[i]), 6) for i in range(6)]
    payload = {
        "joints_rad": joints,
        "joints_deg": [round(j * 180.0 / 3.141592653589793, 2) for j in joints],
        "tcp_pose": [round(float(v), 5) for v in (tcp_pose or [])[:6]],
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = os.path.abspath(HOME_POSE_PATH)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return {"path": out, **payload}
