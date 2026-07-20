# UR5 Agentic AI

Agentic control stack for **UR5** (Ollama + tools + safety policy), with **Robotiq gripper** support and path to airport / OpenArm deployment.

## Quick start

```bash
cd ur5_agent
./setup.sh
source robot_env/bin/activate
python3 scripts/check_ollama.py
python3 main.py --robot ur5 --site lab
```

## Docs

- [VISION.md](VISION.md) — roadmap (lab → airport operations)
- [docs/URDF_USD.md](docs/URDF_USD.md) — URDF + OpenUSD / Isaac Sim adoption
- [docs/SIM_VLA_RL.md](docs/SIM_VLA_RL.md) — 3D camera + Isaac + RL + VLA implementation
- [docs/RL.md](docs/RL.md) — reinforcement learning reach
- [docs/IL.md](docs/IL.md) — imitation learning (human demos)
- [KICKOFF.md](KICKOFF.md) — first live session checklist
- [docs/ROBOTIQ_GRIPPER.md](docs/ROBOTIQ_GRIPPER.md) — gripper (PolyScope ID 1, socket SID 9)
- [ur5_agent/README.md](ur5_agent/README.md) — agent CLI and layout
