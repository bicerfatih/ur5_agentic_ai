#!/usr/bin/env bash
# Fetch official UR + Robotiq description packages into assets/_vendor/.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/assets/_vendor"

mkdir -p "$VENDOR"

clone_if_missing() {
  local url="$1"
  local dir="$2"
  if [[ -d "$VENDOR/$dir/.git" ]]; then
    echo "✓ $dir already cloned"
  else
    echo "Cloning $dir ..."
    git clone --depth 1 "$url" "$VENDOR/$dir"
  fi
}

clone_if_missing "https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git" \
  "Universal_Robots_ROS2_Description"
clone_if_missing "https://github.com/ROS-Industrial/robotiq.git" "robotiq"

echo ""
echo "Vendor packages ready under assets/_vendor/"
echo "Next: python3 ur5_agent/scripts/build_robot_urdf.py"
