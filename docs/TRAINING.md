# Training — Where Physical AI Models Are Needed

This document maps **what to train**, **what to keep pretrained**, and **how data flows** from your UR5 lab through Isaac Sim to airport deployment.

Related: [VISION.md](../VISION.md), [KICKOFF.md](../KICKOFF.md), [data/README.md](../data/README.md).

---

## Summary

| Layer | Train? | Typical approach |
|-------|--------|------------------|
| Agent (Ollama + tools) | Usually **no** | Prompts, policy, optional small LoRA |
| Robot motion (RTDE / MoveIt) | **No** | Kinematics + planners |
| **Object detection** | **Yes** | Fine-tune YOLO on your images |
| **Grasp / place in clutter** | **Yes** (later) | Isaac RL + real demos (imitation) |
| **VLA / end-to-end** | Optional, late | Heavy GPU + data; only if simpler stack fails |
| Safety / site rules | **No** | Code in `policy/safety.py` |

**Rule:** Train where the **camera view and objects change**; use classical robotics + agent tools where **geometry and rules** are enough.

---

## Phase map

| Phase | Training focus | Hardware / tools |
|-------|----------------|------------------|
| **0–1** (now) | **Collect only** — no training pipeline required | UR5, Ollama agent, `logs/session.log` |
| **2** | **Perception** — YOLO (or similar) on lab + sim images | Camera, Isaac for synthetic frames |
| **3** | **Manipulation policies** — IL or RL in sim, fine-tune on real | MoveIt, Isaac Sim, UR5 demos |
| **4** | **Site redeploy** — retrain/fine-tune on airport cameras & objects | OpenArm, ground/cargo zones |
| **5** | **Fleet / monitoring** — drift checks, periodic refresh | Ops data from many runs |

---

## What you do NOT train

### 1. Agent language model (default)

The Ollama loop plans via **tools** (`get_robot_state`, `move_up`, …). The model does not output joint torques directly.

- **Use:** General instruct models (`qwen2.5:7b`, `llama3.1:8b`) + system prompt + site policy.
- **Train only if:** Consistent failures on airport SOP vocabulary → small **LoRA** on procedure docs (optional).

### 2. MoveIt 2 and RTDE motions

- **MoveIt:** Collision geometry from CAD/scan; OMPL/planners — not neural nets.
- **RTDE:** Straight moves for lab validation — no learned policy.

### 3. Safety and compliance

- Speed caps, state-before-move, zone rules stay in **code** (`config/sites.py`, `policy/safety.py`).
- Do not learn safety from data alone.

---

## What you DO train

### A. Perception (Phase 2) — first real training

**Goal:** Reliable boxes in the agent loop: “find bag”, “point above marker”, “is object in tray?”.

| Model | When | Training data |
|-------|------|----------------|
| **YOLOv8 / YOLO-OBB** | First choice for boxes + rotation | `data/labels/detection/` + `data/raw/images/` |
| **Instance segmentation** | Overlapping bags, clutter | Mask labels (CVAT, Roboflow, etc.) |
| **6D pose / grasp points** | Tight picks on known SKUs | CAD + sim renders + few real labels |

**Pipeline (typical):**

1. Capture images: real cell + Isaac synthetic (`data/synthetic/isaac/`).
2. Label → export YOLO format → `data/datasets/detection/<name>/`.
3. Train (Ultralytics example):

   ```bash
   yolo detect train model=yolov8n.pt data=data/datasets/detection/lab_v1/data.yaml epochs=100 imgsz=640
   ```

4. Export best weights → `data/models/detection/<name>/weights/best.pt`.
5. Expose to agent as tools: `detect_objects`, `get_object_pose` (reads model, returns JSON).

**Isaac Sim role:** Domain randomization (lighting, textures, camera pose) to reduce real-world label count.

**Jetson:** Train on workstation GPU if Jetson is slow; deploy **exported** `.pt` or TensorRT on device.

---

### B. Manipulation policies (Phase 3) — when MoveIt + fixed grasps fail

**Goal:** Contact-rich or variable layouts where open-loop `moveL` is not enough.

| Method | Data | Where to train |
|--------|------|----------------|
| **Imitation learning (BC)** | Human demos, kinesthetic teaching, teleop | Workstation → policy in `data/models/policies/` |
| **RL in Isaac** | Sim episodes (reward: grasp success, drop in bin) | GPU server; sim millions of steps |
| **Sim → real** | Fine-tune on 50–200 real UR5 trajectories | Lab UR5 after sim pretrain |

**Isaac Sim role:** Safe exploration, randomized object poses, sensor noise.

**Agent integration:** New tools e.g. `execute_learned_grasp(object_id)` — policy outputs waypoint or gripper command; MoveIt or driver executes.

**Do not start here** until YOLO (or markers) gives stable object poses.

---

### C. Vision-language-action (VLA) — optional, late

Models that map **image + text → actions** (e.g. research RT-style stacks).

| Consider VLA when | Skip VLA when |
|-------------------|---------------|
| Unstructured language + varied objects | MoveIt + detector + agent tools already work |
| Large GPU budget and research team | Airport ops need explainability and audits |

Default path for Emirates/airport: **detector + MoveIt + agent**, not monolithic VLA.

---

### D. Domain language (optional)

| Signal | Action |
|--------|--------|
| Ollama misunderstands “airside”, “ULD”, “gate” | Curate 50–200 Q&A examples → LoRA |
| Tool descriptions are unclear | Fix prompts/schemas first (cheaper) |

---

## Data collection (start now, Phase 0–1)

You do not need a training pipeline yet. **Collect:**

| Asset | Path | Purpose |
|-------|------|---------|
| Agent session logs | `ur5_agent/logs/session.log` | Replay, debug, future IL |
| Robot state snapshots | Copy from `state` command / preflight | Kinematics context |
| Cell photos | `data/raw/images/lab/` | Future YOLO labels |
| Short videos | `data/raw/video/lab/` | Lighting, occlusion study |
| Isaac exports | `data/synthetic/isaac/` | Synthetic pretrain |

**Per session (recommended):**

- Date, site profile (`lab` / `airport_ground`), dry-run vs live.
- Camera mount height/angle note (when camera exists).
- Object list in scene (bags, bins, markers).

---

## Dataset layout

See [data/README.md](../data/README.md). Version datasets by name:

```
data/datasets/detection/lab_v1/
data/datasets/detection/airport_cargo_v1/
```

Never overwrite `v1` when improving — create `v2` and compare metrics.

---

## Train vs deploy locations

| Task | Train | Deploy inference |
|------|-------|------------------|
| YOLO | Workstation / cloud GPU | Jetson Thor (TensorRT optional) |
| RL policy | GPU server + Isaac | Jetson if model fits; else edge server |
| Ollama agent | N/A (pull model) | Jetson (same device as robot) |
| LoRA | Workstation | Ollama `modelfile` / adapter on Jetson |

---

## Evaluation before airport use

| Model type | Metrics | Gate |
|------------|---------|------|
| Detection | mAP@50, per-class recall on **held-out real** images | No deploy if miss rate on critical objects > agreed threshold |
| Grasp policy | Success % over N trials in lab | ≥ target (e.g. 90%) before cargo zone |
| Agent | Task success on scripted goals | Log review + human sign-off |

Always test on **real** lighting at the site — sim-only metrics are not enough.

---

## Integration with the agent (target architecture)

```
User goal (natural language)
        ↓
Ollama agent (tools)
        ├── get_robot_state / move_*     ← no training
        ├── detect_objects (YOLO)        ← trained Phase 2
        ├── plan_to_pose (MoveIt)        ← no training
        └── execute_grasp_policy (opt.)  ← trained Phase 3
        ↓
PolicyEngine (site limits)
        ↓
UR5 / OpenArm driver
```

Training improves **tool backends**, not the agent loop structure.

---

## Checklist by phase

### Phase 0–1 (current)

- [ ] 10+ successful live UR5 agent sessions
- [ ] Logs archived under `data/raw/robot_sessions/` (optional copy)
- [ ] 100+ photos of future workspace → `data/raw/images/lab/`

### Phase 2 (perception)

- [ ] Camera mounted; intrinsics calibrated
- [ ] Label first dataset → `lab_v1`
- [ ] Train YOLO; export to `data/models/detection/`
- [ ] Add `detect_objects` tool; agent uses vision before pick

### Phase 3 (manipulation)

- [ ] Isaac scene matches lab cell roughly
- [ ] RL or IL baseline in sim
- [ ] 50+ real UR5 demos; sim→real fine-tune
- [ ] MoveIt collision scene includes tables/bins

### Phase 4 (airport / Emirates)

- [ ] `airport_cargo_v1` / `airport_ground_v1` datasets from site
- [ ] Retrain or fine-tune detectors on site lighting
- [ ] Policy review for airside human proximity

---

## Tools and references

| Tool | Use |
|------|-----|
| [Ultralytics YOLOv8](https://docs.ultralytics.com/) | Detection training |
| [Isaac Sim](https://developer.nvidia.com/isaac-sim) | Synthetic data + RL |
| [MoveIt 2](https://moveit.picknik.ai/) | Motion planning (no training) |
| [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling) | Agent loop (no training) |
| SPARC (reference) | YOLO + MoveIt + chat UX — not your agent architecture |

---

## Next repo steps (when you start Phase 2)

1. Add `robot/perception/` module wrapping YOLO inference.
2. Add agent tools `detect_objects`, `capture_image`.
3. Add `scripts/train_detection.sh` pointing at `data/datasets/...`.

Until then: **collect data** and run the UR5 agent; training code can wait.
