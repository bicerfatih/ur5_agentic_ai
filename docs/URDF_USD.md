# URDF + OpenUSD adoption

Week-by-week guide for simulation, training, and deployment.  
See also [docs/TRAINING.md](../docs/TRAINING.md).

## Week 1 — URDF (this repo)

### Setup

```bash
cd /home/fatihbicer/Downloads/ur5_agentic_ai
bash scripts/fetch_ur5_urdf.sh          # clone UR + Robotiq vendor packages
python3 ur5_agent/scripts/build_robot_urdf.py
python3 ur5_agent/scripts/validate_urdf_joints.py
python3 ur5_agent/scripts/validate_urdf_joints.py --live --dry-run
# On robot: python3 ur5_agent/scripts/validate_urdf_joints.py --live
```

### Outputs

| Path | Purpose |
|------|---------|
| `assets/robots/ur5_robotiq/urdf/ur5_robotiq.urdf` | Flat URDF + meshes (Isaac, MoveIt, twin) |
| `assets/robots/ur5_robotiq/urdf/joint_map.yaml` | RTDE `q[]` ↔ URDF joint names |
| `assets/cell/lab_table.urdf` | Table stub for Isaac cell |
| `assets/_vendor/` | Upstream ROS packages (gitignored) |

### RTDE joint order

| RTDE `q[i]` | URDF joint | PolyScope |
|-------------|------------|-----------|
| 0 | `shoulder_pan_joint` | Base |
| 1 | `shoulder_lift_joint` | Shoulder |
| 2 | `elbow_joint` | Elbow |
| 3 | `wrist_1_joint` | Wrist 1 |
| 4 | `wrist_2_joint` | Wrist 2 |
| 5 | `wrist_3_joint` | Wrist 3 |

Python helper: `robot.urdf_config.rtde_q_to_urdf_cfg(joint_positions_rad)`.

### Ops console

Built URDF is served at:

`http://127.0.0.1:8788/robot-assets/robots/ur5_robotiq/urdf/ur5_robotiq.urdf`

## Week 2–3 — URDF → OpenUSD (Isaac Sim)

See [scripts/isaac/README.md](../scripts/isaac/README.md).

1. Import `ur5_robotiq.urdf` in Isaac Sim  
2. Add `lab_table.urdf`, camera, bins  
3. Save `assets/usd/lab_cell.usda`  
4. Export synthetic frames → `data/synthetic/isaac/`  

## Week 3–4 — Web twin (URDF meshes)

Replace procedural `twin.js` geometry with [urdf-loader](https://github.com/gkjohnson/urdf-loaders):

- Load `/robot-assets/robots/ur5_robotiq/urdf/ur5_robotiq.urdf`
- Drive joints from telemetry `joint_positions_rad`

## Week 4+ — Training loop

| Stage | Asset |
|-------|--------|
| Detection | Isaac USD renders + `d405_lab_01` real images → YOLO `lab_v2` |
| Imitation / RL | Isaac physics + `data/models/policies/` |
| Real deploy | Agent + RTDE (no USD on robot) |

## Week 5 — Calibration

Update `camera_mount_joint` origin in `ur5_robotiq.urdf.xacro` after hand-eye calibration, then rebuild URDF.
