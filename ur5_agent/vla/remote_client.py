"""HTTP client for remote VLA inference servers (OpenVLA, pi0, GR00T).

The heavy model runs on a GPU host (see scripts/vla/serve_*.py). The robot side
sends image + instruction + state and receives an action vector.

Wire formats:
  - "ur5" (default): our JSON protocol — base64 JPEG image, used by scripts/vla/serve_*.py
  - "openvla_native": json-numpy protocol of the official OpenVLA deploy server
    (openvla repo vla-scripts/deploy.py), so you can point at it directly.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class RemoteVLAError(RuntimeError):
    pass


def _encode_jpeg_b64(rgb: np.ndarray) -> str:
    if cv2 is None:
        raise RemoteVLAError("opencv required to encode image for remote VLA")
    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RemoteVLAError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class RemoteVLAClient:
    def __init__(
        self,
        server_url: str,
        backend: str = "openvla",
        wire_format: str = "ur5",
        timeout_s: float = 10.0,
        unnorm_key: str = "",
    ):
        if requests is None:
            raise RemoteVLAError("`requests` package required (pip install requests)")
        self.server_url = server_url.rstrip("/")
        self.backend = backend
        self.wire_format = wire_format
        self.timeout_s = float(timeout_s)
        self.unnorm_key = unnorm_key

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self.server_url}/health", timeout=min(self.timeout_s, 3.0))
            return r.ok
        except Exception:
            return False

    def act(
        self,
        rgb: np.ndarray,
        instruction: str,
        state12: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Return {"action": np.ndarray, "raw": server_response}.

        Action convention: first 3 dims are Cartesian delta (m or normalized —
        the adapter clips to max_step_m either way); extra dims (rotation,
        gripper) are passed through in "raw" for future use.
        """
        if self.wire_format == "openvla_native":
            payload_raw = self._post_openvla_native(rgb, instruction)
        else:
            payload_raw = self._post_ur5(rgb, instruction, state12)

        action = self._extract_action(payload_raw)
        return {"action": action, "raw": payload_raw}

    def _post_ur5(self, rgb: np.ndarray, instruction: str, state12: np.ndarray | None) -> Any:
        body: dict[str, Any] = {
            "backend": self.backend,
            "image_jpeg_b64": _encode_jpeg_b64(rgb),
            "instruction": instruction,
        }
        if state12 is not None:
            body["state"] = [float(v) for v in np.asarray(state12, dtype=np.float32).reshape(-1)]
        if self.unnorm_key:
            body["unnorm_key"] = self.unnorm_key
        r = requests.post(f"{self.server_url}/act", json=body, timeout=self.timeout_s)
        if not r.ok:
            raise RemoteVLAError(f"VLA server {r.status_code}: {r.text[:200]}")
        return r.json()

    def _post_openvla_native(self, rgb: np.ndarray, instruction: str) -> Any:
        """Official OpenVLA deploy.py uses json-numpy: raw uint8 array in JSON."""
        try:
            import json_numpy

            json_numpy.patch()
        except ImportError as e:
            raise RemoteVLAError("pip install json-numpy for openvla_native format") from e
        body = {"image": np.asarray(rgb, dtype=np.uint8), "instruction": instruction}
        if self.unnorm_key:
            body["unnorm_key"] = self.unnorm_key
        r = requests.post(
            f"{self.server_url}/act",
            data=json.dumps(body),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_s,
        )
        if not r.ok:
            raise RemoteVLAError(f"OpenVLA server {r.status_code}: {r.text[:200]}")
        return r.json()

    @staticmethod
    def _extract_action(raw: Any) -> np.ndarray:
        if isinstance(raw, dict):
            act = raw.get("action", raw.get("actions"))
        else:
            act = raw
        arr = np.asarray(act, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            raise RemoteVLAError(f"VLA action too short: shape={arr.shape}")
        return arr
