#!/usr/bin/env python3
"""NVIDIA GR00T (Isaac-GR00T N1.5) inference server for the UR5 stack.

Wraps a Gr00tPolicy behind the same HTTP protocol as serve_openvla.py, so the
robot side only changes VLA_BACKEND=groot.

Setup on GPU host:
  git clone https://github.com/NVIDIA/Isaac-GR00T.git && cd Isaac-GR00T
  pip install -e .   # follow repo README (CUDA, flash-attn)
  pip install fastapi uvicorn pillow
  python3 /path/to/repo/scripts/vla/serve_groot.py \
      --model nvidia/GR00T-N1.5-3B --data-config single_panda_gripper --port 8000

Robot side:
  export VLA_BACKEND=groot
  export VLA_SERVER_URL=http://<gpu-host>:8000

Note: for best results fine-tune GR00T on UR5 demos (LeRobot format) and pass
your fine-tuned checkpoint + a matching data config / modality json.
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

app = FastAPI(title="GR00T server")
_policy = None
_video_key = "video.ego_view"
_state_key = "state.joints"
_lang_key = "annotation.human.task_description"


class ActRequest(BaseModel):
    image_jpeg_b64: str
    instruction: str
    state: list[float] | None = None
    unnorm_key: str = ""
    backend: str = "groot"


@app.get("/health")
def health():
    return {"status": "ok", "model": "groot", "loaded": _policy is not None}


@app.post("/act")
def act(req: ActRequest):
    t0 = time.time()
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(req.image_jpeg_b64))).convert("RGB"))
    state = np.asarray(req.state or [], dtype=np.float32)
    obs = {
        _video_key: img[np.newaxis, ...],  # (T=1, H, W, 3)
        _state_key: state[np.newaxis, ...],
        _lang_key: [req.instruction],
    }
    result = _policy.get_action(obs)
    # Concatenate action groups, take first step of the horizon.
    parts = []
    for key in sorted(k for k in result.keys() if k.startswith("action.")):
        arr = np.asarray(result[key], dtype=np.float32)
        parts.append(arr[0].reshape(-1) if arr.ndim > 1 else arr.reshape(-1))
    action = np.concatenate(parts) if parts else np.zeros(3, dtype=np.float32)
    return {
        "action": [float(v) for v in action.tolist()],
        "action_keys": sorted(k for k in result.keys() if k.startswith("action.")),
        "latency_s": round(time.time() - t0, 3),
    }


def main():
    global _policy, _video_key, _state_key, _lang_key
    p = argparse.ArgumentParser(description="GR00T HTTP server (Isaac-GR00T)")
    p.add_argument("--model", default="nvidia/GR00T-N1.5-3B", help="HF repo or fine-tuned checkpoint dir")
    p.add_argument("--data-config", default="single_panda_gripper", help="Isaac-GR00T data config name")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--video-key", default="video.ego_view")
    p.add_argument("--state-key", default="state.joints")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    _video_key = args.video_key
    _state_key = args.state_key
    data_config = DATA_CONFIG_MAP[args.data_config]
    print(f"[serve_groot] loading {args.model} (data_config={args.data_config}) ...")
    _policy = Gr00tPolicy(
        model_path=args.model,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
    )
    print(f"[serve_groot] ready on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
