# Vision — Physical + Agentic AI for Airports

## North star

Build **agentic AI** on **physical robots** that can operate safely in real environments, starting with a **UR5 lab prototype** and evolving to **OpenArm** systems at **airports** in support of **Emirates** ground and cargo operations.

## Principles

1. **Agent logic is arm-agnostic** — tools talk to `RobotDriver`, not RTDE or a single vendor.
2. **Policy follows the site** — lab vs airport_ground vs airport_cargo changes speed, step limits, and human-proximity rules.
3. **State before act** — read robot state before motion; verify after critical moves.
4. **Dry-run first** — develop and demo agent flows without hardware (`--dry-run`).
5. **Audit everything** — session logs for replay, incident review, and compliance.

## Roadmap

| Phase | Focus | Status |
|-------|--------|--------|
| 1 | UR5 + Ollama agent + safety policy | **Current** |
| 2 | Perception (cameras, markers, bins/belts) | Planned |
| 3 | OpenArm hardware driver (replace stub) | Planned |
| 4 | Airport site integration (zones, supervisor UI) | Planned |
| 5 | Fleet ops + Emirates workflow hooks | Planned |

## Architecture

```
Operator (natural language)
        ↓
   RobotAgent (Ollama / tools)
        ↓
   PolicyEngine (site profile)
        ↓
   Tools → RobotDriver
        ├── UR5Driver (RTDE)     ← today
        ├── MockDriver           ← dry-run
        └── OpenArmDriver        ← stub → future SDK
```

Training (perception, policies): [docs/TRAINING.md](docs/TRAINING.md) · Data layout: [data/README.md](data/README.md)
