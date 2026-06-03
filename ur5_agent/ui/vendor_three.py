"""Ensure Three.js is present under ui/web/vendor for the Ops Console 3D twin."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

MIN_BYTES = 100_000
# 0.163+ removed build/three.min.js (UMD); 0.160 still ships it for <script> tags.
THREE_VERSION = "0.160.0"

THREE_URLS = (
    f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.min.js",
    f"https://cdnjs.cloudflare.com/ajax/libs/three.js/r160/three.min.js",
    f"https://unpkg.com/three@{THREE_VERSION}/build/three.min.js",
    "https://raw.githubusercontent.com/mrdoob/three.js/r160/build/three.min.js",
)

FETCH_HEADERS = {
    "User-Agent": "ur5-agentic-ai/1.0 (three.js vendor fetch)",
    "Accept": "*/*",
}


def three_vendor_path(web_root: Path) -> Path:
    return web_root / "vendor" / "three.min.js"


def _download_url(url: str, dest: Path, timeout: int = 90) -> None:
    req = urllib.request.Request(url, headers=FETCH_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if len(data) < MIN_BYTES:
        raise ValueError(f"response too small ({len(data)} bytes)")
    dest.write_bytes(data)


def _download_npm(dest: Path) -> str:
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm not found on PATH")

    with tempfile.TemporaryDirectory(prefix="three_vendor_") as tmp:
        tmp_path = Path(tmp)
        print(f"  npm install three@{THREE_VERSION} …")
        subprocess.run(
            [npm, "install", f"three@{THREE_VERSION}", "--no-save", "--prefix", str(tmp_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        built = tmp_path / "node_modules" / "three" / "build" / "three.min.js"
        if not built.exists():
            raise FileNotFoundError(f"npm install did not produce {built}")
        shutil.copyfile(built, dest)
    return f"npm three@{THREE_VERSION}"


def download_three_vendor(dest: Path) -> tuple[bool, str]:
    """Try CDN mirrors, then npm. Returns (ok, message)."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size >= MIN_BYTES:
        return True, f"Three.js already present ({dest.stat().st_size:,} bytes)"

    errors: list[str] = []
    for url in THREE_URLS:
        try:
            print(f"Trying {url} …")
            _download_url(url, dest)
            return True, f"Three.js downloaded from {url} ({dest.stat().st_size:,} bytes)"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
            errors.append(f"{url}: {e}")
            print(f"  failed: {e}")

    try:
        source = _download_npm(dest)
        return True, f"Three.js via {source} ({dest.stat().st_size:,} bytes)"
    except Exception as e:
        errors.append(f"npm: {e}")
        print(f"  npm failed: {e}")

    hint = (
        "All download methods failed. Copy three.min.js manually to:\n"
        f"  {dest}\n"
        f"Working URL (use three@{THREE_VERSION}, NOT 0.163):\n"
        f"  https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.min.js"
    )
    return False, hint + "\n" + "\n".join(errors[:4])


def ensure_three_vendor(web_root: Path) -> tuple[bool, str]:
    path = three_vendor_path(web_root)
    if path.exists() and path.stat().st_size >= MIN_BYTES:
        return True, f"Three.js OK ({path.stat().st_size:,} bytes)"
    ok, msg = download_three_vendor(path)
    return ok, msg
