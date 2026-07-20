# Robot assets (URDF → USD)

## Quick start

```bash
# From repo root
bash scripts/fetch_ur5_urdf.sh
python3 ur5_agent/scripts/build_robot_urdf.py
python3 ur5_agent/scripts/validate_urdf_joints.py
```

Full guide: [docs/URDF_USD.md](../../docs/URDF_USD.md)

## Layout

```
assets/
├── robots/ur5_robotiq/urdf/   # built URDF + joint_map.yaml
├── cell/                      # table / fixtures (URDF)
├── usd/                       # Isaac Sim stages (generated)
└── _vendor/                   # cloned ROS packages (not in git)
```
