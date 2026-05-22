# Kickoff — UR5 first, then vision, then sim stack

Focus on **UR5 + agentic control** for months. OpenArm, Isaac, and full airport deploy come later. This is the order that works.

---

## Phase 0 — Today (you have not tested agentic control yet)

### A. Environment (15 min)

```bash
cd /home/fatihbicer/Downloads/ur5_agentic_ai/ur5_agent
bash setup.sh
source robot_env/bin/activate

# Ollama (default LLM — offline on Jetson)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b
python3 scripts/check_ollama.py
```

### B. Read-only robot check (5 min)

On the UR teach pendant: **Remote** on, program **stopped**, no protective stop, freedrive **off**.

```bash
python3 scripts/preflight.py
```

You want `PASS` with `robot_mode: 7`, `safety_mode: 1`.

### C. Dry-run agent (no motion on hardware) (10 min)

Confirms API key, Claude tools, and airport policy without moving the arm:

```bash
python3 main.py --dry-run --site lab --llm ollama
```

Example goals:

- `read state then move up 2 centimeters`
- `go to home position`

You should see `[DRY-RUN]` lines, not RTDE errors.

### D. First **live** agent test (15 min) — one person, e-stop ready

```bash
python3 main.py --robot ur5 --site lab --host 192.168.0.160 --llm ollama
```

**First goals only** (small, predictable):

1. `read the robot state and report it` (no motion)
2. `move up 2 centimeters` (after state read)
3. `go to home position`

If anything feels wrong: Ctrl+C, physical e-stop, type `quit`.

### E. Log review

Check `logs/session.log` after each session.

---

## Phase 1 — Weeks 1–2: Trust the agent loop on UR5

| Goal | Practice |
|------|----------|
| Reliable connect/disconnect | Start every session with `preflight.py` |
| State-before-move habit | Agent should call `get_robot_state` first |
| Small Cartesian moves | 2–5 cm up/down/forward only |
| Joint moves | Single clear targets in degrees |
| Fail safely | Pull e-stop once in lab; confirm agent reports error after restart |

**Do not add camera yet** until 10+ successful live sessions with simple moves.

Optional: tighten to `--site airport_ground` occasionally to rehearse slower limits (still on UR5).

---

## Phase 2 — Weeks 3–5: Camera + visual tasks

Full training plan: [docs/TRAINING.md](docs/TRAINING.md) · Data dirs: [data/README.md](data/README.md)

Add perception **behind the same tool API** so the agent does not care if pose came from RTDE or vision.

Suggested order:

1. **Fixed USB / RealSense** — stream + snapshot tool for Claude (`capture_image`)
2. **Calibration** — camera frame ↔ robot base (checkerboard or ArUco)
3. **Simple visual tasks** — "find red object", "point TCP above marker", pick-place with known layout
4. **Policy** — no motion from vision if confidence low or marker not seen

Stack options (pick one primary):

- **ROS 2 + ur_robot_driver + image pipeline** — best if you already use ROS for UR5
- **Lightweight Python** — OpenCV + RealSense SDK + your existing `RobotDriver` (fastest next to current repo)

We will add `robot/perception/` and tools like `detect_marker`, `get_object_pose` when you start Phase 2.

---

## Phase 3 — Parallel learning: MoveIt, Isaac Sim (sim time, not blocking UR5)

Use sim to **practice motion planning**, not to replace Phase 1.

| Tool | Use for |
|------|---------|
| **MoveIt 2** | Collision-aware planning, joint limits, OMPL; later attach same goals via ROS action server |
| **Isaac Sim** | Digital twin, sensor sim, multi-step tasks before airside deploy |
| **MoveIt games / tutorials** | Joint limits, planning scenes, pick-place in sim |

Suggested weekly split (example):

- **3 sessions** — live or dry-run agent on UR5 (this repo)
- **2 sessions** — MoveIt tutorial or pick-place exercise in sim
- **1 session** — Isaac intro (UR5 asset, simple joint trajectory)

Bridge later: same high-level goals in agent → MoveIt `move_group` instead of raw `moveL` when obstacles matter.

---

## Phase 4 — Integration map (months)

```
Now:     Natural language → Claude → tools → UR5 RTDE
Next:    + camera tools + calibrated poses
Later:   + MoveIt for cluttered scenes
Later:   + Isaac for scenario rehearsal
Future:  OpenArm @ airport (same agent + policy, new driver)
```

---

## Quick reference

```bash
# Preflight
python3 scripts/preflight.py

# Dry-run agent
python3 main.py --dry-run --site lab

# Live UR5
python3 main.py --robot ur5 --site lab

# Rehearse airport limits on UR5
python3 main.py --robot ur5 --site airport_ground
```

---

## Safety checklist (every live session)

- [ ] Workspace clear, speed slider reasonable on pendant
- [ ] Remote control enabled, program stopped
- [ ] `preflight.py` PASS
- [ ] E-stop reachable
- [ ] First command is state-only or ≤ 5 cm move
- [ ] `lab` site until you are comfortable

When Phase 0D is done, you have officially **tested agentic control** — then Phase 1 is repetition and confidence, not new features.
