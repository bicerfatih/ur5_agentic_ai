# Gripper not moving — troubleshooting

If tools return `status: done` but the **jaw does not move**, we verified on your robot (`192.168.0.160`) that **Standard DO 0/1 do change** on the controller (`do0=True do1=False` for open). The problem is **not** the agent API — it is **pendant wiring, air, or gripper only in `fly2.urp`**.

---

## Pins 2 & 3 (your robot)

| Pin | Role |
|-----|------|
| **2** | **Digital input** — feedback (read from pendant I/O Tools) |
| **3** | **Digital input** — feedback (read from pendant I/O Tools) |

**Never** use pins 2 or 3 for `open_gripper` / `close_gripper`. The agent reads them in `get_gripper_state` only.

See [GRIPPER_WIRING.md](GRIPPER_WIRING.md).

---

## Step 1 — Does the gripper work on the teach pendant?

| Test | What it means |
|------|----------------|
| **Play `fly2.urp`** and gripper moves | Gripper is in the **program**. Use `run_urp_program` / manual PLAY, not I/O tools. |
| **Manual toggle Standard Output DO 0/1** on pendant | Your setup — agent uses `GRIPPER_CMD_TARGET=standard`, pins 0/1. |
| **Never moves from pendant** | Power, air, valve, or gripper fault — fix hardware first. |

---

## Step 2 — Pendant check (do this now)

```bash
cd ~/Downloads/ur5_agentic_ai/ur5_agent
source robot_env/bin/activate
python3 scripts/gripper_pendant_check.py
```

While it runs, **watch I/O Tools** — do **Standard Output DO 0 / DO 1** toggle?

| Pendant LEDs | Jaw |
|--------------|-----|
| Toggle | No move → **air / valves / wrong device** |
| No toggle | **Wrong I/O mapping** in Installation |
| N/A | Try **PLAY `fly2.urp`** manually — if gripper works there, use program not I/O |

Try pneumatic **pulse** (500 ms then release):

```bash
export GRIPPER_PULSE_MS=500
python3 -c "from robot.ur5_driver import UR5Driver; r=UR5Driver(); r.connect(); r.gripper_open(); r.disconnect()"
```

Then:

```bash
export GRIPPER_CMD_TARGET=standard
export GRIPPER_CMD_OPEN_PIN=0
export GRIPPER_CMD_CLOSE_PIN=1
export GRIPPER_OPEN_HIGH=true
```

Feedback (fixed on your robot):

```bash
export GRIPPER_FEEDBACK_IN_OPEN=2
export GRIPPER_FEEDBACK_IN_CLOSED=3
```

---

## Step 3 — Tool flange vs control box

| Connector | Digital outs (commands) |
|-----------|-------------------------|
| Tool (wrist) | **0 and 1 only** |
| Control box configurable / standard | More pins |

Pins **2** and **3** on your setup are **inputs**, not tool-flange outputs.

---

## Step 4 — Robotiq / electric gripper

**Robotiq** needs the URCap + socket driver, not bare digital I/O.

---

## What we still need from you

1. **Gripper model**
2. **Does `fly2.urp` move the gripper** on PLAY?
3. Which **output** pin(s) on the pendant open/close the jaw (not 2/3)

---

## Agent commands by situation

| Setup | Use |
|-------|-----|
| Gripper only in `fly2.urp` | `run fly2.urp` |
| Pendant output pin works | `GRIPPER_CMD_*` + `open_gripper` / `close_gripper` |
| Robotiq | Robotiq integration (future) |
