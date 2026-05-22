# Robotiq gripper (your robot)

## Configuration

| Setting | Value |
|---------|--------|
| PolyScope gripper **ID** | **1** |
| Socket **SID** | **9** (Robotiq maps PolyScope ID 1 → SID 9) |
| Port | **63352** on robot IP |
| Open | `POS 0` (reads ~3 on your gripper) |
| Close | `POS 229` (your 2F-85 max; was 255 in docs) |

Defaults in `config/settings.py`:

```bash
GRIPPER_TYPE=robotiq
GRIPPER_POLYSCOPE_ID=1
ROBOTIQ_SOCKET_SID=9
```

Digital I/O (standard DO 0/1) does **not** move a Robotiq gripper — that was why the jaw never moved.

---

## Test

```bash
cd ~/Downloads/ur5_agentic_ai/ur5_agent
source robot_env/bin/activate
python3 scripts/test_robotiq_gripper.py
```

Or via agent:

```bash
python3 main.py --robot ur5 --site lab
# open the gripper
```

---

## Requirements

- **Robotiq Gripper URCap** installed and enabled on pendant
- Robot powered; gripper shows **ID 1** in installation
- Port **63352** reachable from Jetson (same network as robot)

---

## Override socket SID

If PolyScope ID differs:

```bash
export GRIPPER_POLYSCOPE_ID=2
export ROBOTIQ_SOCKET_SID=10
```
