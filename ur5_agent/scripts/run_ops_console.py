#!/usr/bin/env python3
"""Run futuristic Robot Ops Console UI backend."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn


def main():
    host = os.environ.get("UI_BIND_HOST", "0.0.0.0")
    port = int(os.environ.get("UI_PORT", "8787"))
    print(f"Starting Robot Ops Console on http://{host}:{port}")
    print("Tip: export UI_DRY_RUN=1 for mock mode")
    print("Tip: faster Agentic AI → OLLAMA_MODEL=qwen2.5:3b UI_MODEL=qwen2.5:3b")
    try:
        from pathlib import Path
        import sys

        ui_root = Path(__file__).resolve().parents[1] / "ui"
        if str(ui_root) not in sys.path:
            sys.path.insert(0, str(ui_root))
        from vendor_three import ensure_three_vendor

        web_root = ui_root / "web"
        ok, msg = ensure_three_vendor(web_root)
        print(f"[ui] {msg}" if ok else f"[ui] WARNING: {msg}")
    except Exception as e:
        print(f"[ui] WARNING: could not prepare Three.js for 3D twin: {e}")
    try:
        import websockets  # noqa: F401
    except ImportError:
        print(
            "WARNING: websockets not installed — live WebSocket telemetry disabled.\n"
            "  Fix: pip install 'uvicorn[standard]' websockets"
        )
    uvicorn.run(
        "ui.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        ws="websockets",
    )


if __name__ == "__main__":
    main()
