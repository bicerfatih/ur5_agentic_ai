# Isaac Sim — lab cell + RL bridge

Run on a **GPU workstation** with Isaac Sim installed. The UR5 Jetson runs deploy only.

Full guide: [docs/SIM_VLA_RL.md](../../docs/SIM_VLA_RL.md)

## 1) Build USD stage (GUI)

```bash
export REPO=/home/fatihbicer/Downloads/ur5_agentic_ai
export URDF=$REPO/assets/robots/ur5_robotiq/urdf/ur5_robotiq.urdf
```

1. **File → Import → URDF** → `ur5_robotiq.urdf`
2. Import `assets/cell/lab_table.urdf`
3. Add camera prim (640×480, match RealSense mount)
4. Add manipuland (cube/cup) on table
5. **Save As** → `assets/usd/lab_cell.usda`

## 2) Start bridge server

```bash
export ISAAC_PYTHON=/path/to/isaac-sim/python.sh   # your install
$ISAAC_PYTHON $REPO/scripts/isaac/run_isaac_bridge.py --port 9912
```

Without USD, server uses `LocalRgbdReachEnv` fallback for protocol testing.

With USD + Isaac modules:

```bash
$ISAAC_PYTHON $REPO/scripts/isaac/run_isaac_bridge.py --use-isaac --usd $REPO/assets/usd/lab_cell.usda
```

## 3) Train RL from bridge

On training machine (can be same GPU host):

```bash
cd $REPO/ur5_agent && source robot_env/bin/activate
pip install gymnasium stable-baselines3 torch

# Local dev (no Isaac)
python3 rl/train_isaac_reach.py --backend local --timesteps 50000

# Remote Isaac
export SIM_BACKEND=isaac
export ISAAC_BRIDGE_HOST=127.0.0.1
python3 rl/train_isaac_reach.py --backend isaac --timesteps 200000
```

Checkpoint: `data/models/policies/reach_isaac_v1/policy.zip`

## 4) Deploy on real UR5

Same policy format as state reach:

```bash
python3 scripts/run_rl_policy_once.py --live \
  --task-id camera_reach \
  --target-label cup \
  --policy-path ../data/models/policies/reach_isaac_v1/policy.zip
```

## 5) Synthetic data (next)

Export random poses from Isaac → `data/synthetic/isaac/` for YOLO fine-tune.  
Wire in `scripts/isaac/export_synthetic_frames.py` when stage is stable.

## Joint drives

After URDF import, set arm joints to **position control** with limits from `joint_map.yaml`.
