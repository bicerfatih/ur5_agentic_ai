#!/usr/bin/env python3
"""OpenVLA inference server for the UR5 stack (run on the GPU workstation).

Protocol ("ur5" wire format, matches ur5_agent/vla/remote_client.py):
  POST /act  {"image_jpeg_b64": ..., "instruction": ..., "state": [12], "unnorm_key": ...}
  → {"action": [dx, dy, dz, droll, dpitch, dyaw, gripper]}

Setup on GPU host (needs ~16 GB VRAM for bf16, less with 4-bit):
  pip install torch transformers timm tokenizers accelerate fastapi uvicorn pillow
  python3 scripts/vla/serve_openvla.py --model openvla/openvla-7b --port 8000

Robot side:
  export VLA_BACKEND=openvla
  export VLA_SERVER_URL=http://<gpu-host>:8000
  export VLA_UNNORM_KEY=bridge_orig   # or your fine-tune dataset key
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

app = FastAPI(title="OpenVLA server")
_model = None
_processor = None
_device = "cuda"
_default_unnorm_key = ""


class ActRequest(BaseModel):
    image_jpeg_b64: str
    instruction: str
    state: list[float] | None = None
    unnorm_key: str = ""
    backend: str = "openvla"


def _decode_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


@app.get("/health")
def health():
    return {"status": "ok", "model": "openvla", "loaded": _model is not None}


@app.post("/act")
def act(req: ActRequest):
    t0 = time.time()
    image = _decode_image(req.image_jpeg_b64)
    prompt = f"In: What action should the robot take to {req.instruction.lower().rstrip('.?')}?\nOut:"
    inputs = _processor(prompt, image).to(_device, dtype=_model.dtype)
    unnorm_key = req.unnorm_key or _default_unnorm_key or None
    action = _model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    return {
        "action": [float(v) for v in action.tolist()],
        "latency_s": round(time.time() - t0, 3),
        "unnorm_key": unnorm_key,
    }


def main():
    global _model, _processor, _device, _default_unnorm_key
    p = argparse.ArgumentParser(description="OpenVLA HTTP server")
    p.add_argument("--model", default="openvla/openvla-7b", help="HF repo or local fine-tuned checkpoint dir")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--unnorm-key", default="", help="Default dataset key for action un-normalization")
    p.add_argument("--load-4bit", action="store_true", help="4-bit quantization (bitsandbytes)")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    _device = args.device
    _default_unnorm_key = args.unnorm_key
    print(f"[serve_openvla] loading {args.model} ...")
    _processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    kwargs = dict(
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if args.load_4bit:
        kwargs["load_in_4bit"] = True
    try:
        _model = AutoModelForVision2Seq.from_pretrained(args.model, **kwargs)
    except Exception:
        kwargs.pop("attn_implementation", None)
        _model = AutoModelForVision2Seq.from_pretrained(args.model, **kwargs)
    if not args.load_4bit:
        _model = _model.to(_device)
    print(f"[serve_openvla] ready on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
