# robot/ur5_driver.py — UR5 / UR5e via ur-rtde

import math
import os
import sys
import threading

import time

import rtde_control
import rtde_io
import rtde_receive

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    DASHBOARD_PORT,
    GRIPPER_CMD_CLOSE_PIN,
    GRIPPER_CMD_OPEN_PIN,
    GRIPPER_CMD_PIN,
    GRIPPER_ACTIVE_LOW,
    GRIPPER_CMD_TARGET,
    GRIPPER_FEEDBACK_IN_CLOSED,
    GRIPPER_FEEDBACK_IN_OPEN,
    GRIPPER_INITIAL,
    GRIPPER_OPEN_HIGH,
    GRIPPER_POLYSCOPE_ID,
    GRIPPER_PULSE_MS,
    GRIPPER_PULSE_RELEASE,
    GRIPPER_TYPE,
    GRIPPER_USE_URSCRIPT,
    HOME_JOINTS,
    MAX_JOINT_ACCEL,
    MAX_JOINT_SPEED,
    MAX_LINEAR_ACCEL,
    MAX_LINEAR_SPEED,
    ROBOT_HOST,
    ROBOTIQ_CLOSE_POS,
    ROBOTIQ_FORCE,
    ROBOTIQ_OPEN_POS,
    ROBOTIQ_SOCKET_PORT,
    ROBOTIQ_SOCKET_SID,
    ROBOTIQ_SPEED,
    URP_PLAY_WAIT_SEC,
)
from robot.base import RobotDriver
from robot.dashboard import URDashboardClient
from robot.robotiq_client import RobotiqGripperClient


class UR5Driver(RobotDriver):
    """Low-level UR5 control over RTDE with speed clamping."""

    def __init__(self, host: str = ROBOT_HOST):
        self.host = host
        self.rtde_c = None
        self.rtde_r = None
        self.rtde_io = None
        self.dashboard: URDashboardClient | None = None
        self._connected = False
        self._gripper_state = "unknown"
        self._gripper_io = ""
        self._last_output_readback: dict = {}
        self._robotiq: RobotiqGripperClient | None = None
        self._loaded_program: str | None = None
        self._rtde_lock = threading.RLock()
        self._motion_busy = False
        self._cached_state: dict | None = None
        self._receive_needs_restart = False

    @property
    def motion_busy(self) -> bool:
        return self._motion_busy

    @property
    def arm_model(self) -> str:
        return "ur5"

    @property
    def is_simulated(self) -> bool:
        return False

    def _connect_control(self):
        """RTDE control may need a retry or external-control URCap on the pendant."""
        flags_to_try = [
            ("upload script", rtde_control.RTDEControlInterface.FLAG_UPLOAD_SCRIPT),
            ("external control URCap", rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP),
        ]
        last_err = None
        for label, flag in flags_to_try:
            try:
                return rtde_control.RTDEControlInterface(self.host, flags=flag)
            except Exception as e:
                last_err = e
                print(f"  Control try ({label}) failed: {e}")
                time.sleep(0.5)
        raise last_err

    def connect(self):
        print(f"Connecting to UR5 at {self.host}...")
        try:
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.host)
        except Exception as e:
            raise ConnectionError(f"Could not connect RTDE receive at {self.host}: {e}") from e

        try:
            self.rtde_c = self._connect_control()
        except Exception as e:
            print(f"  RTDE control unavailable: {e}")
            if GRIPPER_TYPE != "robotiq":
                raise ConnectionError(
                    f"RTDE control required for {GRIPPER_TYPE} gripper and motion: {e}"
                ) from e
            print("  Continuing with Robotiq gripper only (socket 63352).")

        try:
            self.rtde_io = rtde_io.RTDEIOInterface(self.host)
        except Exception as e:
            print(f"  RTDE IO unavailable: {e}")
            self.rtde_io = None

        try:
            self.dashboard = URDashboardClient(self.host, port=DASHBOARD_PORT)
            self.dashboard.connect()
            print("  Dashboard connected (urp load/play).")
        except Exception as e:
            self.dashboard = None
            print(f"  Dashboard not available: {e}")

        if GRIPPER_TYPE == "robotiq":
            self._connect_robotiq()
        elif GRIPPER_TYPE not in ("none",):
            self._apply_gripper_initial()
        self._connected = True
        print("UR5 connected.\n")

    def _apply_gripper_initial(self):
        """Home gripper after connect (Robotiq activate leaves jaws open by default)."""
        if GRIPPER_INITIAL == "none":
            return
        if GRIPPER_INITIAL == "open":
            print("  Gripper initial: open")
            self.gripper_open()
            return
        print("  Gripper initial: close")
        self.gripper_close()

    def _connect_robotiq(self):
        print(
            f"  Robotiq gripper: PolyScope ID {GRIPPER_POLYSCOPE_ID}, "
            f"socket SID {ROBOTIQ_SOCKET_SID}, port {ROBOTIQ_SOCKET_PORT}"
        )
        self._robotiq = RobotiqGripperClient(
            host=self.host,
            port=ROBOTIQ_SOCKET_PORT,
            polyscope_id=GRIPPER_POLYSCOPE_ID,
            socket_sid=ROBOTIQ_SOCKET_SID,
            open_pos=ROBOTIQ_OPEN_POS,
            close_pos=ROBOTIQ_CLOSE_POS,
            speed=ROBOTIQ_SPEED,
            force=ROBOTIQ_FORCE,
        )
        self._robotiq.connect()
        self._robotiq.activate()
        print("  Robotiq gripper activated.")
        time.sleep(0.25)
        self._apply_gripper_initial()

    def disconnect(self):
        if self._robotiq:
            self._robotiq.disconnect()
            self._robotiq = None
        if self.dashboard:
            self.dashboard.disconnect()
            self.dashboard = None
        if self.rtde_c:
            try:
                self.rtde_c.disconnect()
            except Exception:
                pass
            self.rtde_c = None
        if self.rtde_r:
            try:
                self.rtde_r.disconnect()
            except Exception:
                pass
            self.rtde_r = None
        if self.rtde_io:
            try:
                self.rtde_io.disconnect()
            except Exception:
                pass
            self.rtde_io = None
        self._connected = False
        print("UR5 disconnected.")

    def release_rtde_control(self) -> dict:
        """Stop external-control script and free RTDE for dashboard .urp play."""
        notes = []
        if self.rtde_c and self.rtde_c.isConnected():
            try:
                self.rtde_c.stopScript()
                notes.append("stopScript ok")
            except Exception as e:
                notes.append(f"stopScript: {e}")
            try:
                self.rtde_c.disconnect()
                notes.append("rtde control disconnected")
            except Exception as e:
                notes.append(f"disconnect: {e}")
            self.rtde_c = None
        return {"status": "done", "notes": notes}

    def reconnect_rtde_control(self) -> dict:
        """Force a fresh RTDE control script (isConnected can be stale after .urp stop)."""
        notes = []
        if self.rtde_c:
            try:
                self.rtde_c.stopScript()
            except Exception as e:
                notes.append(f"stopScript: {e}")
            try:
                self.rtde_c.disconnect()
            except Exception as e:
                notes.append(f"disconnect: {e}")
            self.rtde_c = None
            time.sleep(0.3)
        try:
            self.rtde_c = self._connect_control()
            return {"status": "done", "message": "rtde control reconnected", "notes": notes}
        except Exception as e:
            return {
                "status": "error",
                "reason": str(e),
                "notes": notes,
                "hint": (
                    "On pendant: stop fly2.urp, run External Control program again. "
                    "If 'RTDE registers in use', disable EtherNet/IP/PROFINET/MODBUS on robot."
                ),
            }

    def _ensure_receive(self):
        if not self.rtde_r:
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.host)
            return
        if hasattr(self.rtde_r, "isConnected") and not self.rtde_r.isConnected():
            self._restart_receive()

    @staticmethod
    def _is_transient_rtde_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "end of file" in msg
            or "asio" in msg
            or "connection" in msg
            or "broken pipe" in msg
            or "reset by peer" in msg
            or "nonetype" in msg
        )

    def _receive_call(self, read_fn):
        """Read RTDE state under lock; reconnect receive if the socket dropped."""
        with self._rtde_lock:
            last_err = None
            for attempt in range(3):
                try:
                    self._ensure_receive()
                    if self.rtde_r is None:
                        raise RuntimeError("RTDE receive not connected")
                    return read_fn()
                except Exception as e:
                    last_err = e
                    if attempt < 2 and self._is_transient_rtde_error(e):
                        if self._motion_busy:
                            self._receive_needs_restart = True
                            raise RuntimeError(f"RTDE receive error during motion: {e}") from e
                        print(f"  RTDE receive dropped, reconnecting: {e}")
                        self._restart_receive()
                        time.sleep(0.12)
                        continue
                    raise
            raise last_err or RuntimeError("RTDE receive read failed")

    def _sync_receive_locked(self, frames: int = 2) -> None:
        """Advance RTDE receive buffer so getActualTCPPose is current."""
        self._ensure_receive()
        for _ in range(max(1, frames)):
            if self.rtde_r and hasattr(self.rtde_r, "waitForPeriod"):
                try:
                    self.rtde_r.waitForPeriod()
                except Exception:
                    time.sleep(0.008)
            else:
                time.sleep(0.008)

    def _read_tcp_pose_locked(self) -> list:
        """Fresh TCP pose for motion; caller must hold _rtde_lock."""
        self._ensure_receive()
        if self.rtde_r is None:
            raise RuntimeError("RTDE receive not connected")
        self._sync_receive_locked(2)
        return list(self.rtde_r.getActualTCPPose())

    def _finish_motion_rtde(self) -> None:
        """Reconnect receive after motion if needed; refresh cached telemetry."""
        if self._receive_needs_restart:
            with self._rtde_lock:
                try:
                    self._restart_receive()
                except Exception as e:
                    print(f"  RTDE receive reconnect after motion failed: {e}")
                finally:
                    self._receive_needs_restart = False
        try:
            self._cached_state = self._snapshot_state()
        except Exception:
            pass

    def _restart_receive(self):
        if self.rtde_r:
            try:
                self.rtde_r.disconnect()
            except Exception:
                pass
            self.rtde_r = None
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.host)
        time.sleep(0.06)

    def _fresh_tcp_pose(self) -> list:
        """TCP pose for relative moves (must not restart receive mid-motion)."""
        return self._receive_call(lambda: list(self.rtde_r.getActualTCPPose()))

    def _ensure_control(self):
        if not self.rtde_c or not self.rtde_c.isConnected():
            if GRIPPER_TYPE == "robotiq" and self._robotiq:
                raise RuntimeError(
                    "RTDE control not connected (motion disabled). "
                    "Gripper still works via Robotiq socket. "
                    "Stop other RTDE clients or disable EtherNet/IP / PROFINET on robot."
                )
            err = self.reconnect_rtde_control()
            if err.get("status") == "error":
                raise RuntimeError(err["reason"] + " — " + err.get("hint", ""))

    def _ensure_io(self):
        if not self.rtde_io:
            self.rtde_io = rtde_io.RTDEIOInterface(self.host)

    def is_connected(self) -> bool:
        return self._connected

    def get_joint_positions(self) -> list:
        return self._receive_call(
            lambda: [round(v, 4) for v in self.rtde_r.getActualQ()],
        )

    def get_tcp_pose(self) -> list:
        return self._receive_call(
            lambda: [round(v, 4) for v in self.rtde_r.getActualTCPPose()],
        )

    def get_tcp_force(self) -> list:
        return self._receive_call(
            lambda: [round(v, 4) for v in self.rtde_r.getActualTCPForce()],
        )

    def get_robot_mode(self) -> int:
        return self._receive_call(lambda: self.rtde_r.getRobotMode())

    def get_safety_mode(self) -> int:
        return self._receive_call(lambda: self.rtde_r.getSafetyMode())

    def _rtde_control_state(self) -> dict:
        rtde_ok = bool(self.rtde_c and self.rtde_c.isConnected())
        return {
            "connected": rtde_ok,
            "motion_enabled": rtde_ok,
            "hint": (
                None
                if rtde_ok
                else (
                    "Motion disabled: RTDE control not connected. On pendant run External Control "
                    "(mode 7). Stop other RTDE clients. If 'registers in use', disable "
                    "EtherNet/IP / PROFINET / MODBUS on the robot, then use reconnect_rtde_control."
                )
            ),
        }

    def _snapshot_state(self) -> dict:
        """Single RTDE receive read for all telemetry fields."""
        snap = self._receive_call(
            lambda: {
                "joint_positions_rad": [
                    round(v, 4) for v in self.rtde_r.getActualQ()
                ],
                "tcp_pose": [round(v, 4) for v in self.rtde_r.getActualTCPPose()],
                "tcp_force": [round(v, 4) for v in self.rtde_r.getActualTCPForce()],
                "robot_mode": self.rtde_r.getRobotMode(),
                "safety_mode": self.rtde_r.getSafetyMode(),
            }
        )
        state = {
            "arm_model": self.arm_model,
            "simulated": self.is_simulated,
            **snap,
            "joint_positions_deg": [
                round(math.degrees(v), 2) for v in snap["joint_positions_rad"]
            ],
            "host": self.host,
            "rtde_control": self._rtde_control_state(),
        }
        if hasattr(self, "get_gripper_state"):
            state["gripper"] = self.get_gripper_state()
        if hasattr(self, "get_program_state"):
            state["urp_program"] = self.get_program_state()
        return state

    def get_full_state(self) -> dict:
        if self._motion_busy:
            if self._cached_state:
                cached = dict(self._cached_state)
                cached["telemetry_stale"] = True
                cached["motion_in_progress"] = True
                return cached
            return {
                "arm_model": self.arm_model,
                "simulated": self.is_simulated,
                "motion_in_progress": True,
                "telemetry_stale": True,
                "rtde_control": self._rtde_control_state(),
            }
        state = self._snapshot_state()
        self._cached_state = state
        return state

    def move_joint(self, joints: list, speed: float = 0.3, accel: float = 0.3):
        self._ensure_control()
        speed = min(speed, MAX_JOINT_SPEED)
        accel = min(accel, MAX_JOINT_ACCEL)
        ok = self.rtde_c.moveJ(joints, speed, accel)
        if ok is False:
            reconnect = self.reconnect_rtde_control()
            if reconnect.get("status") != "error":
                ok = self.rtde_c.moveJ(joints, speed, accel)
        if ok is False:
            raise RuntimeError(
                "moveJ rejected by the controller. On the pendant: start External Control "
                "(or re-run reconnect_rtde_control) and ensure no other RTDE client holds control."
            )

    def _poll_tcp_motion_locked(self, start: list, target: list, timeout: float = 6.0) -> dict:
        """Poll TCP after moveL. Caller must hold _rtde_lock; never reconnects receive."""
        commanded = sum((target[i] - start[i]) ** 2 for i in range(3)) ** 0.5
        best = 0.0
        final = list(start)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self._sync_receive_locked(1)
                actual = list(self.rtde_r.getActualTCPPose())
            except Exception:
                time.sleep(0.06)
                continue
            achieved = sum((actual[i] - start[i]) ** 2 for i in range(3)) ** 0.5
            best = max(best, achieved)
            final = actual
            err = max(abs(actual[i] - target[i]) for i in range(3))
            if commanded >= 0.001 and (err <= 0.006 or achieved >= commanded * 0.85):
                break
            time.sleep(0.06)
        try:
            self._sync_receive_locked(1)
            final = list(self.rtde_r.getActualTCPPose())
            best = max(
                best,
                sum((final[i] - start[i]) ** 2 for i in range(3)) ** 0.5,
            )
        except Exception:
            pass
        return {
            "commanded_m": round(commanded, 4),
            "achieved_m": round(best, 4),
            "final_tcp": [round(v, 4) for v in final],
        }

    def move_linear(self, tcp_pose: list, speed: float = 0.1, accel: float = 0.1) -> dict:
        self._motion_busy = True
        try:
            with self._rtde_lock:
                start = self._read_tcp_pose_locked()
                return self._move_linear_from_locked(start, tcp_pose, speed, accel)
        finally:
            self._motion_busy = False
            self._finish_motion_rtde()

    def _move_linear_from_locked(
        self, start: list, tcp_pose: list, speed: float = 0.1, accel: float = 0.1
    ) -> dict:
        """Execute moveL + poll. Caller must hold _rtde_lock."""
        self._ensure_control()
        speed = min(speed, MAX_LINEAR_SPEED)
        accel = min(accel, MAX_LINEAR_ACCEL)
        ok = self.rtde_c.moveL(tcp_pose, speed, accel)
        if ok is False:
            reconnect = self.reconnect_rtde_control()
            if reconnect.get("status") != "error":
                ok = self.rtde_c.moveL(tcp_pose, speed, accel)
        if ok is False:
            raise RuntimeError(
                "moveL rejected by the controller. On the pendant: start External Control "
                "(robot_mode must be 7) and ensure no other RTDE client holds control."
            )
        return self._poll_tcp_motion_locked(start, tcp_pose)

    def _move_linear_from(
        self, start: list, tcp_pose: list, speed: float = 0.1, accel: float = 0.1
    ) -> dict:
        with self._rtde_lock:
            return self._move_linear_from_locked(start, tcp_pose, speed, accel)

    def move_home(self):
        self.move_joint(HOME_JOINTS, speed=0.3, accel=0.3)

    def stop(self):
        if self.rtde_c and self.rtde_c.isConnected():
            self.rtde_c.stopJ(2.0)

    def move_tcp_relative(
        self, dx=0.0, dy=0.0, dz=0.0, speed: float = 0.15, accel: float = 0.15
    ) -> dict:
        """Translate in robot BASE frame (x,y,z); keep current TCP orientation."""
        self._motion_busy = True
        try:
            with self._rtde_lock:
                before = self._read_tcp_pose_locked()
                target = [
                    before[0] + dx,
                    before[1] + dy,
                    before[2] + dz,
                    before[3],
                    before[4],
                    before[5],
                ]
                commanded = sum((target[i] - before[i]) ** 2 for i in range(3)) ** 0.5
                if commanded < 0.0005:
                    self._sync_receive_locked(4)
                    before = list(self.rtde_r.getActualTCPPose())
                    target = [
                        before[0] + dx,
                        before[1] + dy,
                        before[2] + dz,
                        before[3],
                        before[4],
                        before[5],
                    ]
                report = self._move_linear_from_locked(before, target, speed, accel)
                report["start_tcp"] = [round(v, 4) for v in before]
                return report
        finally:
            self._motion_busy = False
            self._finish_motion_rtde()

    # ── Gripper: commands on digital OUT, feedback on DI 2 & 3 ──

    def _cmd_io_setter(self):
        if GRIPPER_CMD_TARGET == "standard":
            return self.rtde_io.setStandardDigitalOut, "standard DO"
        if GRIPPER_CMD_TARGET == "configurable":
            return self.rtde_io.setConfigurableDigitalOut, "configurable DO"
        return self.rtde_io.setToolDigitalOut, "tool DO"

    def _out_read_pin(self, pin: int) -> int:
        """RTDEReceive pin index for getDigitalOutState."""
        if GRIPPER_CMD_TARGET == "tool":
            return 16 + pin
        return pin

    def _logical_level(self, active: bool) -> bool:
        if GRIPPER_ACTIVE_LOW:
            return not active
        return active

    def _set_digital_out(self, pin: int, level: bool) -> bool:
        setter, _ = self._cmd_io_setter()
        return setter(pin, bool(level))

    def _set_digital_out_pair(self, open_pin: int, close_pin: int, open_gripper: bool) -> bool:
        if open_gripper:
            return self._set_digital_out(
                open_pin, self._logical_level(True)
            ) and self._set_digital_out(close_pin, self._logical_level(False))
        return self._set_digital_out(
            open_pin, self._logical_level(False)
        ) and self._set_digital_out(close_pin, self._logical_level(True))

    def _gripper_via_urscript(self, open_pin: int, close_pin: int, open_gripper: bool) -> bool:
        self._ensure_control()
        if GRIPPER_CMD_TARGET == "tool":
            fn = "set_tool_digital_out"
        elif GRIPPER_CMD_TARGET == "configurable":
            fn = "set_configurable_digital_out"
        else:
            fn = "set_standard_digital_out"
        o_hi = "True" if self._logical_level(True) else "False"
        o_lo = "True" if self._logical_level(False) else "False"
        if open_gripper:
            script = (
                f"{fn}({open_pin}, {o_hi})\n"
                f"{fn}({close_pin}, {o_lo})\n"
            )
            name = "agent_gripper_open"
        else:
            script = (
                f"{fn}({open_pin}, {o_lo})\n"
                f"{fn}({close_pin}, {o_hi})\n"
            )
            name = "agent_gripper_close"
        return bool(self.rtde_c.sendCustomScriptFunction(name, script))

    def _read_cmd_outputs(self, open_pin: int, close_pin: int) -> dict:
        if self._motion_busy:
            return dict(self._last_output_readback)
        return self._receive_call(
            lambda: {
                f"do{open_pin}": bool(
                    self.rtde_r.getDigitalOutState(self._out_read_pin(open_pin))
                ),
                f"do{close_pin}": bool(
                    self.rtde_r.getDigitalOutState(self._out_read_pin(close_pin))
                ),
            }
        )

    def _read_digital_in(self, pin: int) -> bool | None:
        if self._motion_busy:
            return None
        try:
            return self._receive_call(lambda: bool(self.rtde_r.getDigitalInState(pin)))
        except Exception:
            return None

    def _read_gripper_feedback(self) -> dict:
        """Pins 2 & 3 are inputs on your pendant — read only."""
        o = self._read_digital_in(GRIPPER_FEEDBACK_IN_OPEN)
        c = self._read_digital_in(GRIPPER_FEEDBACK_IN_CLOSED)
        inferred = "unknown"
        if o is True and c is False:
            inferred = "open"
        elif c is True and o is False:
            inferred = "closed"
        elif o is False and c is False:
            inferred = "transition_or_unwired"
        return {
            "input_open_pin": GRIPPER_FEEDBACK_IN_OPEN,
            "input_closed_pin": GRIPPER_FEEDBACK_IN_CLOSED,
            "open_sensor": o,
            "closed_sensor": c,
            "inferred_jaw_state": inferred,
        }

    def _set_gripper(self, open_gripper: bool):
        if GRIPPER_TYPE == "none":
            raise RuntimeError("Gripper disabled (GRIPPER_TYPE=none)")

        _, io_desc = self._cmd_io_setter()
        if GRIPPER_TYPE == "dual_pin":
            open_pin, close_pin = GRIPPER_CMD_OPEN_PIN, GRIPPER_CMD_CLOSE_PIN
            if GRIPPER_CMD_TARGET == "tool" and (open_pin > 1 or close_pin > 1):
                raise RuntimeError(
                    "Tool flange only has DO 0–1 for commands. "
                    "Pins 2/3 are feedback inputs — use standard DO 0/1."
                )
            if GRIPPER_USE_URSCRIPT and self.rtde_c and self.rtde_c.isConnected():
                ok = self._gripper_via_urscript(open_pin, close_pin, open_gripper)
                method = "urscript"
            else:
                self._ensure_io()
                ok = self._set_digital_out_pair(open_pin, close_pin, open_gripper)
                method = "rtde_io"
            self._gripper_io = f"{io_desc} {method} open={open_pin} close={close_pin}"
        else:
            self._ensure_io()
            level = GRIPPER_OPEN_HIGH if open_gripper else (not GRIPPER_OPEN_HIGH)
            if GRIPPER_ACTIVE_LOW:
                level = not level
            pin = GRIPPER_CMD_PIN
            ok = self._set_digital_out(pin, level)
            self._gripper_io = f"{io_desc} pin={pin} level={level}"
            open_pin, close_pin = pin, pin

        if not ok:
            raise RuntimeError(f"Gripper command failed ({self._gripper_io})")

        if GRIPPER_PULSE_MS > 0:
            time.sleep(GRIPPER_PULSE_MS / 1000.0)
            if GRIPPER_TYPE == "dual_pin" and GRIPPER_PULSE_RELEASE:
                self._ensure_io()
                self._set_digital_out(open_pin, self._logical_level(False))
                self._set_digital_out(close_pin, self._logical_level(False))

        self._gripper_state = "open" if open_gripper else "closed"
        self._last_output_readback = self._read_cmd_outputs(
            GRIPPER_CMD_OPEN_PIN if GRIPPER_TYPE == "dual_pin" else GRIPPER_CMD_PIN,
            GRIPPER_CMD_CLOSE_PIN if GRIPPER_TYPE == "dual_pin" else GRIPPER_CMD_PIN,
        )

    def _ensure_robotiq(self):
        if not self._robotiq:
            self._connect_robotiq()
        elif not self._robotiq.is_active():
            self._robotiq.activate()

    def gripper_open(self):
        if GRIPPER_TYPE == "robotiq":
            self._ensure_robotiq()
            self._robotiq.open()
            self._gripper_state = "open"
            self._gripper_io = f"robotiq SID {ROBOTIQ_SOCKET_SID} POS {ROBOTIQ_OPEN_POS}"
            return
        self._set_gripper(True)

    def gripper_close(self):
        if GRIPPER_TYPE == "robotiq":
            self._ensure_robotiq()
            self._robotiq.close()
            self._gripper_state = "closed"
            self._gripper_io = f"robotiq SID {ROBOTIQ_SOCKET_SID} POS {ROBOTIQ_CLOSE_POS}"
            return
        self._set_gripper(False)

    def get_gripper_state(self) -> dict:
        if GRIPPER_TYPE == "robotiq" and self._robotiq:
            state = self._robotiq.get_state()
            state["last_command"] = self._gripper_state
            state["command_wiring"] = self._gripper_io
            return state
        feedback = self._read_gripper_feedback()
        readback = self._last_output_readback or self._read_cmd_outputs(
            GRIPPER_CMD_OPEN_PIN, GRIPPER_CMD_CLOSE_PIN
        )
        return {
            "type": GRIPPER_TYPE,
            "cmd_target": GRIPPER_CMD_TARGET,
            "last_command": self._gripper_state,
            "command_wiring": self._gripper_io,
            "output_readback": readback,
            "feedback_inputs": feedback,
            "pendant_check": (
                "If jaw did not move: on PolyScope I/O Tools watch Standard Output DO0/DO1 — "
                "they MUST toggle when open_gripper/close_gripper runs. "
                "If they toggle but jaw stays still → air supply, solenoid wiring, or gripper only in fly2.urp. "
                "If they do not toggle → wrong I/O mapping. "
                "Try: export GRIPPER_PULSE_MS=500 then test again. "
                "Or PLAY fly2.urp manually on pendant."
            ),
        }

    # ── PolyScope .urp programs (dashboard) ─────────────

    def _dashboard_required(self) -> URDashboardClient:
        if not self.dashboard:
            raise RuntimeError(
                "Dashboard server not connected. Check port 29999 and robot network."
            )
        return self.dashboard

    @staticmethod
    def _normalize_urp(name: str) -> str:
        return name if name.endswith(".urp") else f"{name}.urp"

    def get_program_state(self) -> dict:
        try:
            dash = self._dashboard_required()
            return {
                "loaded": self._loaded_program,
                "running": dash.running(),
                "program_state": dash.program_state(),
            }
        except Exception as e:
            return {"error": str(e), "loaded": self._loaded_program}

    def run_urp_program(self, program_name: str) -> dict:
        dash = self._dashboard_required()
        prog = self._normalize_urp(program_name)

        release = self.release_rtde_control()
        prep = dash.prepare_to_play()

        load_resp = dash.load_program(prog)
        if any(x in load_resp.lower() for x in ("error", "fail", "denied", "not found")):
            return {
                "status": "error",
                "step": "load",
                "program": prog,
                "response": load_resp,
                "release_rtde": release,
            }
        self._loaded_program = prog
        play_resp = dash.play()
        if any(x in play_resp.lower() for x in ("error", "fail", "denied")):
            return {
                "status": "error",
                "step": "play",
                "program": prog,
                "load": load_resp,
                "response": play_resp,
                "release_rtde": release,
                "prepare": prep,
                "program_state": dash.program_state(),
                "robot_mode": dash.robot_mode(),
                "fix_on_pendant": [
                    "Settings → enable Remote Control (required for dashboard play)",
                    "Stop the External Control / RTDE program if it is running",
                    "On Program screen press PLAY for fly2.urp if dashboard play fails",
                    "Then run tool reconnect_rtde_control before agent moves",
                ],
            }
        time.sleep(URP_PLAY_WAIT_SEC)
        return {
            "status": "done",
            "program": prog,
            "load": load_resp,
            "play": play_resp,
            "running": dash.running(),
            "program_state": dash.program_state(),
            "release_rtde": release,
            "note": "When program ends, use reconnect_rtde_control for move_* tools.",
        }

    def stop_urp_program(self) -> dict:
        dash = self._dashboard_required()
        resp = dash.stop()
        return {"status": "stopped", "response": resp, "running": dash.running()}
