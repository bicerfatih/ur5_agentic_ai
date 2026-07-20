#!/usr/bin/env python3
"""Expand ur5_robotiq.urdf.xacro → flat URDF with absolute mesh paths."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR = REPO_ROOT / "assets" / "_vendor"
XACRO = REPO_ROOT / "assets" / "robots" / "ur5_robotiq" / "urdf" / "ur5_robotiq.urdf.xacro"
OUT_DIR = REPO_ROOT / "assets" / "robots" / "ur5_robotiq" / "urdf"
OUT_URDF = OUT_DIR / "ur5_robotiq.urdf"
BUILD_ROOT = OUT_DIR / "_build_vendor"

UR_DESC = VENDOR / "Universal_Robots_ROS2_Description"
ROBOTIQ_VIS = VENDOR / "robotiq" / "robotiq_2f_85_gripper_visualization"

PKG_PATHS = {
    "ur_description": UR_DESC,
    "robotiq_2f_85_gripper_visualization": ROBOTIQ_VIS,
}


def _expand_find(text: str) -> str:
    for pkg, root in PKG_PATHS.items():
        text = text.replace(f"$(find {pkg})", str(root))
    return text


def _mirror_vendor_tree() -> tuple[Path, Path]:
    """Copy vendor packages with $(find pkg) expanded (xacro includes need this)."""
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    ur_mirror = BUILD_ROOT / "ur_description"
    rq_mirror = BUILD_ROOT / "robotiq_2f_85_gripper_visualization"
    for src, dst in ((UR_DESC, ur_mirror), (ROBOTIQ_VIS, rq_mirror)):
        for path in src.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(src)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix in {".xacro", ".urdf", ".xml", ".yaml", ".yml"}:
                out.write_text(_expand_find(path.read_text(encoding="utf-8")), encoding="utf-8")
            else:
                shutil.copy2(path, out)
    return ur_mirror, rq_mirror


def _run_xacro(ur_type: str, ur_mirror: Path, rq_mirror: Path) -> str:
    src = _expand_find(XACRO.read_text(encoding="utf-8"))
    src = src.replace(str(UR_DESC), str(ur_mirror))
    src = src.replace(str(ROBOTIQ_VIS), str(rq_mirror))
    tmp = OUT_DIR / ".ur5_robotiq.build.xacro"
    tmp.write_text(src, encoding="utf-8")

    cmd = ["xacro", str(tmp), f"ur_type:={ur_type}", "force_abs_paths:=true"]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise SystemExit(
            "xacro not found. Install ROS xacro or: sudo apt install ros-${ROS_DISTRO}-xacro"
        ) from e
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr or "")
        raise SystemExit(f"xacro failed (exit {e.returncode})") from e
    return proc.stdout


def _rewrite_package_uris(urdf_text: str) -> str:
    """Convert package:// URIs to absolute file paths for Isaac / browser loaders."""

    def repl(match: re.Match) -> str:
        pkg = match.group(1)
        rel = match.group(2)
        base = PKG_PATHS.get(pkg)
        if base is None:
            return match.group(0)
        path = (base / rel).resolve()
        return f'file://{path}"'

    return re.sub(r'package://([^/]+)/(.+?)"', repl, urdf_text)


def main() -> None:
    p = argparse.ArgumentParser(description="Build flattened UR5 + Robotiq URDF")
    p.add_argument("--ur-type", default="ur5", help="UR variant (ur5, ur5e, …)")
    p.add_argument("--skip-rewrite", action="store_true", help="Keep package:// URIs")
    args = p.parse_args()

    if not XACRO.is_file():
        raise SystemExit(f"Missing {XACRO}. Run scripts/fetch_ur5_urdf.sh first.")
    if not UR_DESC.is_dir() or not ROBOTIQ_VIS.is_dir():
        raise SystemExit("Vendor URDF missing. Run scripts/fetch_ur5_urdf.sh")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ur_mirror, rq_mirror = _mirror_vendor_tree()
    urdf = _run_xacro(args.ur_type, ur_mirror, rq_mirror)
    if not args.skip_rewrite:
        urdf = _rewrite_package_uris(urdf)
    OUT_URDF.write_text(urdf, encoding="utf-8")
    print(f"Wrote {OUT_URDF} ({len(urdf)} bytes)")
    print("Validate: python3 ur5_agent/scripts/validate_urdf_joints.py")


if __name__ == "__main__":
    main()
