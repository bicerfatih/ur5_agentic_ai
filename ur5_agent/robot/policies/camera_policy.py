"""Camera policy runner for RL checkpoints with safe fallback behavior."""

from __future__ import annotations

import json
import os
from typing import Any


def _pick_target(detection: dict[str, Any], target_label: str) -> dict[str, Any] | None:
    objs = detection.get("objects", []) if isinstance(detection, dict) else []
    if not objs:
        return None
    if target_label:
        needle = target_label.strip().lower()
        for o in objs:
            if needle in str(o.get("label", "")).lower():
                return o
    return objs[0]


def _load_policy_meta(policy_path: str) -> dict[str, Any]:
    if not policy_path:
        return {}
    if not os.path.isfile(policy_path):
        return {}
    if policy_path.endswith(".json"):
        try:
            with open(policy_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def run_camera_policy_step(
    frame,
    detection: dict[str, Any],
    policy_path: str = "",
    target_label: str = "",
    max_step_m: float = 0.01,
) -> dict[str, Any]:
    """
    Returns one policy action in robot base deltas (dx, dy, dz).
    Current baseline: visual-centering fallback with optional JSON gains.
    """
    h, w = frame.shape[:2]
    tgt = _pick_target(detection, target_label)
    if tgt is None:
        return {"status": "error", "reason": "No detected target in frame."}

    c = tgt.get("center") or {}
    cx = float(c.get("x", w * 0.5))
    cy = float(c.get("y", h * 0.5))
    ex = (cx - (w * 0.5)) / max(1.0, w * 0.5)  # right positive
    ey = (cy - (h * 0.5)) / max(1.0, h * 0.5)  # down positive

    cfg = _load_policy_meta(policy_path)
    gain_x = float(cfg.get("gain_x", 0.010))
    gain_y = float(cfg.get("gain_y", 0.010))
    deadband = float(cfg.get("deadband", 0.08))
    z_bias = float(cfg.get("z_bias", 0.0))

    done = abs(ex) < deadband and abs(ey) < deadband
    if done:
        return {
            "status": "done",
            "done": True,
            "target": tgt.get("label"),
            "pixel_error": {"ex": round(ex, 4), "ey": round(ey, 4)},
            "action": {"dx": 0.0, "dy": 0.0, "dz": 0.0},
        }

    # Camera-axis fallback mapping:
    # image x -> base y, image y -> base z (negative because down in image)
    dy = max(-max_step_m, min(max_step_m, ex * gain_x))
    dz = max(-max_step_m, min(max_step_m, -ey * gain_y + z_bias))

    return {
        "status": "active",
        "done": False,
        "target": tgt.get("label"),
        "pixel_error": {"ex": round(ex, 4), "ey": round(ey, 4)},
        "action": {"dx": 0.0, "dy": round(dy, 5), "dz": round(dz, 5)},
        "policy_source": "json-meta" if cfg else "fallback-servo",
    }
