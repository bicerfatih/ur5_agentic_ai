# UR5 Agentic AI — Physical + Agentic Robot Platform

Natural-language control of a robot arm with Claude, built for **lab validation on UR5** and **future OpenArm deployment at airports (Emirates operations)**.

See [VISION.md](../VISION.md) for the long-term scope.  
Training plan: [docs/TRAINING.md](../docs/TRAINING.md) · Data folders: [data/README.md](../data/README.md)  
Gripper (Robotiq ID 1): [docs/ROBOTIQ_GRIPPER.md](../docs/ROBOTIQ_GRIPPER.md) · `.urp`: [docs/GRIPPER_PROGRAMS.md](../docs/GRIPPER_PROGRAMS.md)
Camera (Intel RealSense): [docs/CAMERA_REALSENSE.md](../docs/CAMERA_REALSENSE.md)

## Hardware (lab)

- UR5 / UR5e — RTDE at `192.168.0.160` (edit in `config/settings.py` or `--host`)
- Control machine (e.g. Jetson) — `192.168.0.85`
- Intel RealSense (D455/F455) color stream

## Project structure

```
ur5_agent/
├── main.py                 # CLI entry (--robot, --site, --dry-run)
├── config/
│   ├── settings.py         # Global defaults + env vars
│   └── sites.py            # lab | airport_ground | airport_cargo profiles
├── policy/
│   └── safety.py           # Site-aware motion policy
├── robot/
│   ├── base.py             # RobotDriver contract
│   ├── ur5_driver.py       # Live UR5 (ur-rtde)
│   ├── mock_driver.py      # Dry-run simulation
│   ├── openarm_driver.py   # Future OpenArm (stub + dry-run)
│   ├── factory.py          # create_robot()
│   └── tools.py            # Claude tools + schemas
└── agent/
    ├── factory.py          # create_agent(ollama|claude)
    ├── ollama_agent.py     # Local Ollama tool loop (default)
    ├── claude_agent.py     # Optional Anthropic backend
    └── prompts.py          # Site-specific system prompts
```

## Quick start

```bash
bash setup.sh
source robot_env/bin/activate

# Ollama (default — no API key)
ollama serve
ollama pull qwen2.5:7b
python3 scripts/check_ollama.py

# Develop without robot hardware
python3 main.py --dry-run --site lab

# Live UR5
python3 scripts/preflight.py
python3 main.py --robot ur5 --site lab

# Camera quick test
python3 scripts/test_camera.py

# Futuristic Robot Ops Console UI
export UI_DRY_RUN=1   # remove for live robot
python3 scripts/run_ops_console.py
# open http://localhost:8787

# Optional Claude backend
export ANTHROPIC_API_KEY="your-key"
python3 main.py --llm claude --robot ur5 --site lab
```

## CLI

| Flag | Description |
|------|-------------|
| `--robot ur5` | Universal Robots arm (default) |
| `--robot openarm` | OpenArm stub (hardware TBD) |
| `--site lab` | Development limits |
| `--site airport_ground` | Slower speeds, 15cm horizontal cap, Emirates context |
| `--site airport_cargo` | Cargo zone profile |
| `--dry-run` | Mock driver — no network to robot |
| `--host IP` | UR5 controller address |
| `--llm ollama` | Local Ollama agent (default) |
| `--llm claude` | Anthropic API (needs `ANTHROPIC_API_KEY`) |
| `--model TAG` | Ollama model, e.g. `qwen2.5:7b`, `llama3.1:8b` |

Environment overrides: `LLM_BACKEND`, `OLLAMA_MODEL`, `OLLAMA_HOST`, `ROBOT_TYPE`, `SITE_ID`, `DRY_RUN=1`, `ROBOT_HOST`.

## Example goals

- `read state then move up 5 centimeters`
- `go to home position`
- `move forward 8cm` (may be blocked into smaller steps at `airport_ground`)
- `what is the TCP force right now?`
- `capture a camera frame for this session`

REPL shortcuts: `state`, `quit`.

## Safety

- Global caps in `config/settings.py`; **site profiles can be stricter**
- Policy blocks motion without prior `get_robot_state` at airport sites
- Live hardware: requires robot_mode=7, safety_mode=1
- Physical e-stop always available on the pendant

## Next integration steps

1. Wire **OpenArm** vendor SDK in `robot/openarm_driver.py`
2. Add **camera / perception** tools for airport tasks
3. Supervisor **confirm** step for human-in-the-loop at airside zones
