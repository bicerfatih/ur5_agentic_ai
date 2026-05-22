# Gripper wiring (your robot)

## Confirmed on PolyScope I/O Tools

| Signal | Pin | Type |
|--------|-----|------|
| **Open command** | **Standard Output DO 0** | Sinking NPN |
| **Close command** | **Standard Output DO 1** | Sinking NPN |
| **Open feedback** | **Input 2** | Read only |
| **Closed feedback** | **Input 3** | Read only |

**Sinking NPN:** when the output is **ON** in PolyScope, RTDE `setStandardDigitalOut(pin, True)` drives that line active (sinks to common). The agent uses the same ON/OFF semantics as the pendant toggle.

---

## Software defaults (`config/settings.py`)

```bash
GRIPPER_TYPE=dual_pin
GRIPPER_CMD_TARGET=standard
GRIPPER_CMD_OPEN_PIN=0
GRIPPER_CMD_CLOSE_PIN=1
GRIPPER_FEEDBACK_IN_OPEN=2
GRIPPER_FEEDBACK_IN_CLOSED=3
```

| Agent tool | DO 0 | DO 1 |
|------------|------|------|
| `open_gripper` | ON | OFF |
| `close_gripper` | OFF | ON |

---

## Test

```bash
cd ~/Downloads/ur5_agentic_ai/ur5_agent
source robot_env/bin/activate
python3 -c "
from robot.ur5_driver import UR5Driver
r = UR5Driver(); r.connect()
r.gripper_open(); input('Open? ')
r.gripper_close(); input('Close? ')
print(r.get_gripper_state())
r.disconnect()
"
```

If open/close are reversed on the hardware, swap pins:

```bash
export GRIPPER_CMD_OPEN_PIN=1
export GRIPPER_CMD_CLOSE_PIN=0
```

---

## If gripper only moves in `fly2.urp`

Use `run_urp_program` / PLAY on pendant instead of I/O tools.
