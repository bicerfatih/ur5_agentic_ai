# robot/robotiq_client.py — Robotiq URCap socket (port 63352)
#
# PolyScope gripper ID 1 → socket SID 9 (Robotiq default mapping).

import socket
import threading
import time
from collections import OrderedDict


# PolyScope installation gripper ID → URCap socket slave ID
_POLYSCOPE_TO_SOCKET_SID = {1: 9, 2: 10, 3: 11, 4: 12}


class RobotiqGripperClient:
    """Control Robotiq gripper via URCap TCP socket on the UR controller."""

    ENCODING = "UTF-8"
    STA_ACTIVE = 3

    def __init__(
        self,
        host: str,
        port: int = 63352,
        polyscope_id: int = 1,
        socket_sid: int | None = None,
        open_pos: int = 0,
        close_pos: int = 255,
        speed: int = 255,
        force: int = 255,
    ):
        self.host = host
        self.port = port
        self.polyscope_id = polyscope_id
        self.socket_sid = socket_sid if socket_sid is not None else _POLYSCOPE_TO_SOCKET_SID.get(
            polyscope_id, 8 + polyscope_id
        )
        self.open_pos = open_pos
        self.close_pos = close_pos
        self.speed = speed
        self.force = force
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def connect(self):
        self._sock = socket.create_connection((self.host, self.port), timeout=5.0)
        self._sock.settimeout(3.0)
        self._select_gripper()

    def disconnect(self):
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _send_line(self, line: str) -> bytes:
        if not self._sock:
            raise RuntimeError("Robotiq socket not connected")
        with self._lock:
            self._sock.sendall((line.strip() + "\n").encode(self.ENCODING))
            return self._sock.recv(1024)

    @staticmethod
    def _is_ack(data: bytes) -> bool:
        return data.strip() == b"ack"

    def _select_gripper(self):
        if not self._is_ack(self._send_line(f"SET SID {self.socket_sid}")):
            raise RuntimeError(f"Robotiq SET SID {self.socket_sid} failed")

    def _set_vars(self, variables: OrderedDict[str, int]) -> bool:
        self._select_gripper()
        parts = " ".join(f"{k} {v}" for k, v in variables.items())
        return self._is_ack(self._send_line(f"SET {parts}"))

    def _get_var(self, name: str) -> int:
        self._select_gripper()
        data = self._send_line(f"GET {name}").decode(self.ENCODING).strip()
        var_name, value_str = data.split()
        if var_name != name:
            raise ValueError(f"Unexpected Robotiq response: {data}")
        return int(value_str)

    def is_active(self) -> bool:
        try:
            return self._get_var("STA") == self.STA_ACTIVE
        except (OSError, ValueError, socket.timeout):
            return False

    def activate(self, timeout_sec: float = 10.0):
        self._select_gripper()
        if self.is_active():
            return
        self._set_vars(OrderedDict([("ACT", 0), ("ATR", 0)]))
        time.sleep(0.2)
        self._set_vars(OrderedDict([("ACT", 1)]))
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.is_active():
                return
            time.sleep(0.1)
        raise RuntimeError(
            f"Robotiq gripper (PolyScope ID {self.polyscope_id}, SID {self.socket_sid}) "
            "did not activate — check URCap on pendant."
        )

    def move_to(self, position: int, wait: bool = True, wait_timeout: float = 8.0):
        self._select_gripper()
        ok = self._set_vars(
            OrderedDict(
                [
                    ("POS", position),
                    ("SPE", self.speed),
                    ("FOR", self.force),
                    ("GTO", 1),
                ]
            )
        )
        if not ok:
            raise RuntimeError(f"Robotiq move to {position} rejected")
        if not wait:
            return
        deadline = time.time() + wait_timeout
        while time.time() < deadline:
            pos = self._get_var("POS")
            obj = self._get_var("OBJ")
            if abs(pos - position) <= 12 and obj != 0:
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"Robotiq move to {position} timed out (last POS={self._get_var('POS')})"
        )

    def open(self):
        self.move_to(self.open_pos)

    def close(self):
        self.move_to(self.close_pos)

    def get_state(self) -> dict:
        try:
            return {
                "driver": "robotiq_socket",
                "polyscope_gripper_id": self.polyscope_id,
                "socket_sid": self.socket_sid,
                "port": self.port,
                "active": self.is_active(),
                "position": self._get_var("POS"),
                "status": self._get_var("STA"),
                "object": self._get_var("OBJ"),
            }
        except Exception as e:
            return {
                "driver": "robotiq_socket",
                "polyscope_gripper_id": self.polyscope_id,
                "socket_sid": self.socket_sid,
                "error": str(e),
            }
