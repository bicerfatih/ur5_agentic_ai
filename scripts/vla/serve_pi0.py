#!/usr/bin/env python3
"""pi0 (Physical Intelligence openpi) inference server for the UR5 stack.

Wraps an openpi policy behind the same HTTP protocol as serve_openvla.py, so
the robot side only changes VLA_BACKEND=pi0.

Setup on GPU host:
  git clone https://github.com/Physical-Intelligence/openpi.git && cd openpi
  # follow openpi README (uv sync); then in that env:
  pip install fastapi uvicorn pillow
  python3 /path/to/repo/scripts/vla/serve_pi0.py --config pi0_base --checkpoint <ckpt-dir> --port 8000

Robot side:
  export VLA_BACKEND=pi0
  export VLA_SERVER_URL=http://<gpu-host>:8000

Note: pi0 expects a state vector and returns an action *chunk* (horizon of
actions). We return the first action of the chunk; first 3 dims are treated
as the Cartesian delta by the robot-side adapter.
"""

from __future__ import annotations

import argparse
import base64
import io
import time

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

app = FastAPI(title="pi0 server")
_policy = None


class ActRequest(BaseModel):
    image_jpeg_b64: str
    instruction: str
    state: list[float] | None = None
    unnorm_key: str = ""
    backend: str = "pi0"


@app.get("/health")
def health():
    return {"status": "ok", "model": "pi0", "loaded": _policy is not None}


@app.post("/act")
def act(req: ActRequest):
    t0 = time.time()
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(req.image_jpeg_b64))).convert("RGB"))
    state = np.asarray(req.state or [], dtype=np.float32)
    obs = {
        "observation/image": img,
        "observation/state": state,
        "prompt": req.instruction,
    }
    result = _policy.infer(obs)
    chunk = np.asarray(result["actions"], dtype=np.float32)
    first = chunk[0] if chunk.ndim > 1 else chunk
    return {
        "action": [float(v) for v in first.reshape(-1).tolist()],
        "chunk_len": int(chunk.shape[0]) if chunk.ndim > 1 else 1,
        "latency_s": round(time.time() - t0, 3),
    }


def main():
    global _policy
    p = argparse.ArgumentParser(description="pi0 HTTP server (openpi)")
    p.add_argument("--config", default="pi0_base", help="openpi training config name")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint dir")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    print(f"[serve_pi0] loading config={args.config} ckpt={args.checkpoint} ...")
    cfg = openpi_config.get_config(args.config)
    _policy = policy_config.create_trained_policy(cfg, args.checkpoint)
    print(f"[serve_pi0] ready on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
