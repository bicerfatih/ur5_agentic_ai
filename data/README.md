# Data directory

Versioned datasets and trained weights for **physical AI** (perception and policies).  
Training guide: [docs/TRAINING.md](../docs/TRAINING.md).

**Do not commit large binaries** to git unless you use Git LFS. Keep raw video and full datasets on disk or object storage.

## Layout

```
data/
├── README.md                 ← you are here
├── raw/                      ← unprocessed captures
│   ├── images/
│   │   └── lab/              ← real camera stills (Phase 0+)
│   ├── video/
│   │   └── lab/              ← short cell recordings
│   └── robot_sessions/       ← copies of agent logs + notes per session
├── synthetic/
│   └── isaac/                ← renders from Isaac Sim (Phase 2+)
├── labels/
│   └── detection/            ← YOLO/COCO exports before packaging
├── datasets/                 ← training-ready packages (versioned)
│   └── detection/
│       └── lab_v1/           ← example: images/ labels/ data.yaml
└── models/                   ← trained artifacts for deployment
    ├── detection/
    │   └── lab_v1/           ← e.g. weights/best.pt
    └── policies/
        └── lab_grasp_v1/     ← Phase 3+ learned policies
```

## Naming convention

| Pattern | Example |
|---------|---------|
| Dataset version | `lab_v1`, `airport_cargo_v1` |
| Model version | Match dataset or `lab_v1_best` |
| Session folder | `2026-05-21_lab_live_01/` under `raw/robot_sessions/` |

## Session log template

Create `raw/robot_sessions/<date>_<site>_<notes>/README.txt`:

```
date: 2026-05-21
site: lab
robot: ur5
dry_run: false
camera: none
objects_in_scene: table, red_cube
agent_log: ../../../ur5_agent/logs/session.log
notes: first live move_up 2cm
```

## .gitignore

Large folders under `raw/`, `synthetic/`, `datasets/`, and `models/` should stay local. Add a root `.gitignore` entry when you start filling these directories.
