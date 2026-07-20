#!/usr/bin/env python3
"""Validate URDF joint order vs RTDE and optionally compare live robot joints."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_ROOT.parent


def _load_urdf_config():
    path = AGENT_ROOT / "robot" / "urdf_config.py"
    spec = importlib.util.spec_from_file_location("urdf_config", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


urdf_config = _load_urdf_config()
REPO_URDF_PATH = urdf_config.REPO_URDF_PATH
JOINT_MAP_PATH = urdf_config.JOINT_MAP_PATH
load_joint_map = urdf_config.load_joint_map
rtde_to_urdf_joint_names = urdf_config.rtde_to_urdf_joint_names
urdf_arm_joint_names = urdf_config.urdf_arm_joint_names

try:
    import yourdfpy
except ImportError:
    yourdfpy = None


def _parse_urdf_joints(urdf_path: Path) -> list[str]:
    root = ET.parse(urdf_path).getroot()
    names: list[str] = []
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        name = joint.get("name")
        if not name:
            continue
        if any(
            name.endswith(suffix)
            for suffix in (
                "_joint",
            )
        ) and name in {
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        }:
            names.append(name)
    order = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    return [n for n in order if n in names]


def _print_joint_map() -> None:
    jm = load_joint_map()
    print("RTDE getActualQ()  →  URDF joint  (PolyScope)")
    print("─" * 52)
    for row in jm.get("rtde_joint_order", []):
        idx = row["index"]
        print(f"  q[{idx}]  →  {row['urdf_joint']:<22}  ({row.get('polyscope', '')})")
    frames = jm.get("frames", {})
    print("\nKey frames:")
    for key, val in frames.items():
        print(f"  {key}: {val}")


def _compare_live(urdf_path: Path, host: str | None, dry_run: bool) -> int:
    sys.path.insert(0, str(AGENT_ROOT))
    from robot.factory import create_robot

    expected = rtde_to_urdf_joint_names()
    robot = create_robot(robot_type="ur5", dry_run=dry_run, host=host)
    robot.connect()
    q = robot.get_joint_positions()
    robot.disconnect()

    print(f"\nLive RTDE joints ({len(q)} values, rad):")
    for i, (name, val) in enumerate(zip(expected, q)):
        print(f"  q[{i}] {name}: {val:+.4f} rad  ({math.degrees(val):+.2f}°)")

    if yourdfpy is None:
        print("\nInstall yourdfpy for FK compare: pip install yourdfpy")
        return 0

    urdf = yourdfpy.URDF.load(str(urdf_path))
    cfg = {name: float(val) for name, val in zip(expected, q)}
    try:
        urdf.update_cfg(cfg)
        fk = urdf.get_frame_matrix("tool0")
        print(f"\nURDF FK tool0 position (m): {fk[0:3, 3].round(4).tolist()}")
    except Exception as e:
        print(f"\nURDF FK skipped: {e}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Validate UR5 URDF vs RTDE joint order")
    p.add_argument("--urdf", type=Path, default=REPO_URDF_PATH)
    p.add_argument("--live", action="store_true", help="Read joints from robot (or dry-run mock)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--host", default=os.environ.get("ROBOT_HOST"))
    args = p.parse_args()

    if not JOINT_MAP_PATH.is_file():
        print(f"Missing {JOINT_MAP_PATH}", file=sys.stderr)
        return 1

    _print_joint_map()

    if not args.urdf.is_file():
        print(f"\nURDF not built yet: {args.urdf}")
        print("Run: python3 ur5_agent/scripts/build_robot_urdf.py")
        return 1

    urdf_joints = _parse_urdf_joints(args.urdf)
    expected = urdf_arm_joint_names()
    print(f"\nURDF arm joints ({args.urdf.name}):")
    for i, name in enumerate(urdf_joints):
        print(f"  [{i}] {name}")

    if urdf_joints != expected:
        print("\n❌ URDF joint order mismatch vs joint_map.yaml", file=sys.stderr)
        return 1
    print("\n✓ URDF joint names match joint_map.yaml")

    if args.live:
        return _compare_live(args.urdf, args.host, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
