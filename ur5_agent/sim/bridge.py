"""JSON/TCP bridge between training code (Jetson/workstation) and Isaac Sim (GPU host)."""

from __future__ import annotations

import json
import socket
from typing import Any


class IsaacBridgeClient:
    """
    Talks to scripts/isaac/run_isaac_bridge.py inside Isaac Sim python.

    Protocol (line-delimited JSON):
      -> {"cmd":"reset"}
      <- {"obs":{image_b64?, state:[12], ...}, "info":{}}
      -> {"cmd":"step","action":[dx,dy,dz]}
      <- {"obs":..., "reward":float, "done":bool, "truncated":bool, "info":{}}
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9912, timeout_sec: float = 30.0):
        self.host = host
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout_sec)
        self._sock.settimeout(self.timeout_sec)

    def close(self) -> None:
        if self._sock:
            try:
                self._send({"cmd": "close"})
            except Exception:
                pass
            self._sock.close()
            self._sock = None

    def _send(self, msg: dict[str, Any]) -> dict[str, Any]:
        if self._sock is None:
            raise RuntimeError("IsaacBridgeClient not connected")
        line = (json.dumps(msg) + "\n").encode("utf-8")
        self._sock.sendall(line)
        buf = b""
        while b"\n" not in buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("Isaac bridge closed connection")
            buf += chunk
        raw, _ = buf.split(b"\n", 1)
        return json.loads(raw.decode("utf-8"))

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        rep = self._send({"cmd": "reset"})
        return dict(rep.get("obs") or {}), dict(rep.get("info") or {})

    def step(self, action: list[float]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        rep = self._send({"cmd": "step", "action": [float(a) for a in action[:3]]})
        return (
            dict(rep.get("obs") or {}),
            float(rep.get("reward", 0.0)),
            bool(rep.get("done", False)),
            bool(rep.get("truncated", False)),
            dict(rep.get("info") or {}),
        )

    def ping(self) -> bool:
        try:
            rep = self._send({"cmd": "ping"})
            return rep.get("status") == "ok"
        except Exception:
            return False
