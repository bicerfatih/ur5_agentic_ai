#!/usr/bin/env python3
"""Download Three.js into ui/web/vendor for offline Ops Console 3D twin."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "ui"
sys.path.insert(0, str(UI_DIR))

from vendor_three import download_three_vendor, three_vendor_path  # noqa: E402


def main():
    dest = three_vendor_path(ROOT / "ui" / "web")
    ok, msg = download_three_vendor(dest)
    if ok:
        print(msg)
        print(f"File: {dest}")
        return
    print(msg, file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
