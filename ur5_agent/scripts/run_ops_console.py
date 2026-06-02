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
    uvicorn.run("ui.server:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
