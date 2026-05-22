# robot/dashboard.py — UR Dashboard Server (port 29999) for .urp load/play

import socket
import time


class URDashboardClient:
    """Text TCP client for Universal Robots Dashboard Server."""

    def __init__(self, host: str, port: int = 29999, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None

    def connect(self):
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._sock.settimeout(self.timeout)
        self._read_line()  # welcome banner

    def disconnect(self):
        if self._sock:
            try:
                self.command("quit")
            except OSError:
                pass
            self._sock.close()
            self._sock = None

    def _read_line(self) -> str:
        buf = b""
        while b"\n" not in buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        return buf.decode("utf-8", errors="replace").strip()

    def command(self, cmd: str) -> str:
        if not self._sock:
            raise RuntimeError("Dashboard not connected")
        self._sock.sendall((cmd.strip() + "\n").encode("utf-8"))
        return self._read_line()

    def load_program(self, program: str) -> str:
        name = program if program.endswith(".urp") else f"{program}.urp"
        return self.command(f"load {name}")

    def play(self) -> str:
        return self.command("play")

    def stop(self) -> str:
        return self.command("stop")

    def pause(self) -> str:
        return self.command("pause")

    def running(self) -> bool:
        resp = self.command("running").lower()
        return "true" in resp

    def program_state(self) -> str:
        return self.command("programState")

    def robot_mode(self) -> str:
        return self.command("robotmode")

    def safety_mode(self) -> str:
        return self.command("safetymode")

    def prepare_to_play(self) -> dict:
        """Best-effort pendant prep before play (remote control must be ON)."""
        steps = {}
        for cmd in ("stop", "brake release"):
            try:
                steps[cmd] = self.command(cmd)
            except Exception as e:
                steps[cmd] = str(e)
        time.sleep(0.2)
        for cmd in ("close safety popup", "unlock protective stop"):
            try:
                steps[cmd] = self.command(cmd)
            except Exception as e:
                steps[cmd] = str(e)
        return steps
