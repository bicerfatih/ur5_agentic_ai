# VLA inference servers (GPU workstation)

Real vision-language-action models are too heavy for the Jetson. Run one of
these servers on the GPU host; the robot side talks to it over HTTP through
`ur5_agent/vla/remote_client.py`.

Related: [docs/SIM_VLA_RL.md](../../docs/SIM_VLA_RL.md).

## Protocol

All servers expose the same API:

| Endpoint | Body | Returns |
|----------|------|---------|
| `GET /health` | — | `{"status": "ok", "loaded": true}` |
| `POST /act` | `{"image_jpeg_b64", "instruction", "state": [12], "unnorm_key"}` | `{"action": [dx, dy, dz, ...]}` |

The robot-side adapter uses the first 3 action dims as a Cartesian delta,
clips it to `VLA_MAX_STEP_M`, and runs it through `policy/safety.py` like any
other motion — the VLA never bypasses the safety layer.

## Pick a model

| Server | Model | VRAM | Notes |
|--------|-------|------|-------|
| `serve_openvla.py` | `openvla/openvla-7b` | ~16 GB bf16 (or `--load-4bit`) | Easiest start; zero-shot is weak on unseen rigs — fine-tune on UR5 demos for real use |
| `serve_pi0.py` | pi0 (openpi) | ~24 GB | Needs openpi env + checkpoint; returns action chunks |
| `serve_groot.py` | `nvidia/GR00T-N1.5-3B` | ~12 GB | Fine-tune on LeRobot-format UR5 demos for a matching embodiment |

You can also point the client directly at the **official OpenVLA deploy
server** (`vla-scripts/deploy.py` in the openvla repo) with
`VLA_WIRE_FORMAT=openvla_native`.

## Quick start (OpenVLA example)

GPU host:

```bash
pip install torch transformers timm tokenizers accelerate fastapi uvicorn pillow
python3 scripts/vla/serve_openvla.py --model openvla/openvla-7b --port 8000
```

Robot (Jetson):

```bash
export VLA_BACKEND=openvla
export VLA_SERVER_URL=http://<gpu-host>:8000
export VLA_UNNORM_KEY=bridge_orig      # or your fine-tune dataset key
cd ur5_agent && source robot_env/bin/activate
python3 scripts/run_vla_reach_once.py --live --instruction "reach the cup" --steps 15
```

## Env vars (robot side)

| Var | Default | Meaning |
|-----|---------|---------|
| `VLA_BACKEND` | `tool_routed` | `openvla` \| `pi0` \| `groot` \| `tool_routed` \| `disabled` |
| `VLA_SERVER_URL` | — | `http://<gpu-host>:8000` |
| `VLA_WIRE_FORMAT` | `ur5` | `ur5` \| `openvla_native` |
| `VLA_UNNORM_KEY` | — | OpenVLA action un-normalization dataset key |
| `VLA_ACTION_SCALE_M` | `0.01` | Meters per unit when server returns normalized actions |
| `VLA_SERVER_TIMEOUT_S` | `10` | HTTP timeout |
| `VLA_MAX_STEP_M` | `0.008` | Hard cap on per-step Cartesian delta |

If `VLA_SERVER_URL` is unset, remote backends fall back to a proportional
stub so you can test the loop without a GPU.

## Expectations

Zero-shot VLA on a rig it has never seen (your camera pose, your table, your
gripper) is usually poor. The realistic path:

1. Verify the loop end-to-end with the stub / tool_routed backend.
2. Record UR5 demos (`scripts/record_demo.py`, see docs/IL.md).
3. Fine-tune (OpenVLA LoRA or GR00T on LeRobot-format data) on the GPU host.
4. Serve the fine-tuned checkpoint with the matching `--model` path.
