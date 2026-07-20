# robot/urdf_config.py — URDF paths and RTDE joint name mapping

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_ASSET_DIR = _REPO_ROOT / "assets" / "robots" / "ur5_robotiq"
URDF_DIR = ROBOT_ASSET_DIR / "urdf"
JOINT_MAP_PATH = URDF_DIR / "joint_map.yaml"
REPO_URDF_PATH = URDF_DIR / "ur5_robotiq.urdf"
XACRO_PATH = URDF_DIR / "ur5_robotiq.urdf.xacro"


@lru_cache(maxsize=1)
def load_joint_map() -> dict:
    with JOINT_MAP_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def urdf_arm_joint_names() -> list[str]:
    jm = load_joint_map()
    rows = sorted(jm.get("rtde_joint_order", []), key=lambda r: r["index"])
    return [r["urdf_joint"] for r in rows]


def rtde_to_urdf_joint_names() -> list[str]:
    return urdf_arm_joint_names()


def rtde_q_to_urdf_cfg(q_rad: list[float]) -> dict[str, float]:
    """Map RTDE getActualQ() vector to URDF joint configuration dict."""
    names = urdf_arm_joint_names()
    n = min(len(names), len(q_rad))
    return {names[i]: float(q_rad[i]) for i in range(n)}


def frame_name(key: str) -> str:
    return str(load_joint_map().get("frames", {}).get(key, key))
