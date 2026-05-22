# Gripper configuration

## Your robot (confirmed)

- **Input 2** — open/feedback sensor (read only)
- **Input 3** — closed/feedback sensor (read only)
- **Commands** — **Standard Output DO 0** (open), **DO 1** (close), sinking NPN

---

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `GRIPPER_TYPE` | `dual_pin` | `digital_io` \| `dual_pin` \| `none` |
| `GRIPPER_CMD_TARGET` | `standard` | Standard outputs on control box (your setup) |
| `GRIPPER_CMD_OPEN_PIN` | `0` | Standard DO 0 = open |
| `GRIPPER_CMD_CLOSE_PIN` | `1` | Standard DO 1 = close |
| `GRIPPER_CMD_PIN` | `0` | Only for `digital_io` single-pin mode |
| `GRIPPER_OPEN_HIGH` | `true` | HIGH on cmd pin = open |
| `GRIPPER_FEEDBACK_IN_OPEN` | `2` | Read-only feedback |
| `GRIPPER_FEEDBACK_IN_CLOSED` | `3` | Read-only feedback |

Legacy names still work: `GRIPPER_IO_TARGET` → `GRIPPER_CMD_TARGET`, `GRIPPER_PIN` → `GRIPPER_CMD_PIN`.

---

## Examples

```bash
# Defaults (your robot) — only if you need to override:
export GRIPPER_TYPE=dual_pin
export GRIPPER_CMD_TARGET=standard
export GRIPPER_CMD_OPEN_PIN=0
export GRIPPER_CMD_CLOSE_PIN=1
```

---

## Test

```bash
python3 scripts/test_gripper_outputs.py
python3 main.py --robot ur5 --site lab
```
