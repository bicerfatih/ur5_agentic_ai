#!/usr/bin/env python3
"""Robot Ops Console backend (telemetry + tool execution)."""

import asyncio
import datetime as dt
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.websockets import WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.sites import get_site
from config.settings import CAMERA_OUTPUT_DIR, CAMERA_TYPE
from agent.factory import create_agent
from policy.safety import PolicyEngine
from robot.factory import create_robot
from robot.tools import TOOL_SCHEMAS, execute_tool
from camera import ObjectDetector, RealSenseCamera
from speech.transcribe import speech_config, transcribe_audio

_UI_ROOT = Path(__file__).resolve().parent
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))
from vendor_three import ensure_three_vendor, three_vendor_path


def _json_safe(value: Any) -> Any:
    """Ensure telemetry payloads are JSON-serializable (numpy scalars, etc.)."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


class ToolCall(BaseModel):
    name: str
    inputs: dict[str, Any] = Field(default_factory=dict)


class GoalCall(BaseModel):
    goal: str


class RobotSession:
    def __init__(self):
        robot_type = os.environ.get("UI_ROBOT", os.environ.get("ROBOT_TYPE", "ur5"))
        site_id = os.environ.get("UI_SITE", os.environ.get("SITE_ID", "lab"))
        host = os.environ.get("UI_HOST", os.environ.get("ROBOT_HOST"))
        dry_run = os.environ.get("UI_DRY_RUN", "").lower() in ("1", "true", "yes")
        self.robot = create_robot(robot_type=robot_type, dry_run=dry_run, host=host)
        self.site = get_site(site_id)
        self.policy = PolicyEngine(self.site)
        self.connected = False
        self.last_tool_events: list[dict[str, Any]] = []
        self.last_camera_path: str | None = None
        self.llm_backend = os.environ.get("UI_LLM", os.environ.get("LLM_BACKEND", "ollama"))
        self.agent = None
        self.goal_lock = threading.Lock()
        self.goal_status: dict[str, Any] = {
            "running": False,
            "goal": None,
            "started_at": None,
            "ended_at": None,
            "result": None,
            "error": None,
        }
        self.live_camera = None
        self.camera_lock = threading.Lock()
        self.detector = ObjectDetector()
        self.last_detection: dict[str, Any] = {"count": 0, "labels": []}
        self.last_good_state: dict[str, Any] | None = None
        self.last_state_error: str | None = None

    def connect(self):
        if not self.connected:
            self.robot.connect()
            self.connected = True
            if CAMERA_TYPE == "realsense":
                self.live_camera = RealSenseCamera()
                # Share one RealSense pipeline with agent tools (avoid Device busy).
                try:
                    from robot import tools as toolmod

                    toolmod._camera = self.live_camera
                except Exception:
                    pass

    def disconnect(self):
        if self.connected:
            self.robot.disconnect()
            self.connected = False
        if self.live_camera:
            cam = self.live_camera
            cam.disconnect()
            self.live_camera = None
            try:
                from robot import tools as toolmod

                if toolmod._camera is cam:
                    toolmod._camera = None
            except Exception:
                pass

    def state(self) -> dict[str, Any]:
        if getattr(self.robot, "motion_busy", False) and self.last_good_state is not None:
            cached = dict(self.last_good_state)
            cached["telemetry_stale"] = True
            cached["motion_in_progress"] = True
            return cached
        try:
            st = self.robot.get_full_state()
            self.last_good_state = st
            self.last_state_error = None
            return st
        except Exception as e:
            self.last_state_error = str(e)
            if self.last_good_state is not None:
                cached = dict(self.last_good_state)
                cached["telemetry_stale"] = True
                cached["read_error"] = str(e)
                return cached
            raise

    def telemetry_payload(self) -> dict[str, Any]:
        """State for WebSocket / UI; never raises."""
        try:
            state = self.state()
        except Exception as e:
            state = {
                "error": str(e),
                "arm_model": getattr(self.robot, "arm_model", "ur5"),
                "simulated": getattr(self.robot, "is_simulated", False),
            }
        return _json_safe({
            "ts": time.time(),
            "state": state,
            "events": self.last_tool_events,
            "goal_status": self.goal_status,
            "detection": self.last_detection,
        })

    def run_tool(self, name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        start = time.time()
        camera_tools = {
            "get_camera_frame",
            "detect_objects",
            "approach_object_once",
            "execute_rl_policy",
            "execute_vla_policy",
        }
        if name == "get_camera_frame":
            with self.camera_lock:
                result = self.capture_and_save_frame(
                    session_id=(inputs or {}).get("session_id", "lab"),
                    prefix=(inputs or {}).get("prefix", "frame"),
                )
        elif name in camera_tools:
            with self.camera_lock:
                if self.live_camera is not None:
                    try:
                        from robot import tools as toolmod

                        toolmod._camera = self.live_camera
                    except Exception:
                        pass
                result = execute_tool(name=name, inputs=inputs, robot=self.robot, policy=self.policy)
        else:
            result = execute_tool(name=name, inputs=inputs, robot=self.robot, policy=self.policy)
        evt = {
            "ts": time.time(),
            "tool": name,
            "inputs": inputs,
            "result": result,
            "elapsed_ms": int((time.time() - start) * 1000),
        }
        self.last_tool_events.insert(0, evt)
        self.last_tool_events = self.last_tool_events[:25]
        if isinstance(result, dict) and isinstance(result.get("path"), str):
            self.last_camera_path = result["path"]
        return evt

    def _ensure_agent(self):
        # Fresh agent each goal so tool list + prompts stay current (no stale move_home).
        self.agent = create_agent(
            robot=self.robot,
            site=self.site,
            llm=self.llm_backend,
            ollama_model=os.environ.get("UI_MODEL"),
        )

    def run_goal_async(self, goal: str):
        if self.goal_status["running"]:
            raise RuntimeError("A goal is already running.")

        self.goal_status = {
            "running": True,
            "goal": goal,
            "started_at": time.time(),
            "ended_at": None,
            "result": None,
            "error": None,
        }

        def _worker():
            try:
                with self.goal_lock:
                    self._ensure_agent()
                    self.agent.run(goal)
                note = getattr(self.agent, "last_run_note", None)
                self.goal_status["result"] = "done"
                self.goal_status["note"] = note
            except Exception as e:
                self.goal_status["error"] = str(e)
            finally:
                self.goal_status["running"] = False
                self.goal_status["ended_at"] = time.time()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def live_jpeg(self) -> bytes:
        if CAMERA_TYPE == "none":
            raise RuntimeError("Camera disabled (CAMERA_TYPE=none)")
        with self.camera_lock:
            if self.live_camera is None:
                self.live_camera = RealSenseCamera()
            frame = self.live_camera.capture_color_frame()
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Failed to encode camera frame.")
        return encoded.tobytes()

    def live_jpeg_with_detection(self) -> bytes:
        if CAMERA_TYPE == "none":
            raise RuntimeError("Camera disabled (CAMERA_TYPE=none)")
        with self.camera_lock:
            if self.live_camera is None:
                self.live_camera = RealSenseCamera()
            frame = self.live_camera.capture_color_frame()
        frame, meta = self.detector.detect_and_draw(frame)
        self.last_detection = ObjectDetector.to_summary(meta)
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Failed to encode detection frame.")
        return encoded.tobytes()

    def capture_and_save_frame(self, session_id: str = "lab", prefix: str = "frame") -> dict[str, Any]:
        if CAMERA_TYPE == "none":
            return {"status": "error", "reason": "Camera disabled (CAMERA_TYPE=none)."}
        with self.camera_lock:
            if self.live_camera is None:
                self.live_camera = RealSenseCamera()
            frame = self.live_camera.capture_color_frame()
        out_dir = Path(CAMERA_OUTPUT_DIR).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{session_id}_{stamp}.jpg"
        path = (out_dir / filename).resolve()
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            return {"status": "error", "reason": f"Failed to save frame to {path}"}
        return {
            "status": "done",
            "path": str(path),
            "shape": list(frame.shape),
            "camera": "intel_realsense",
            "serial": self.live_camera.serial or "auto",
        }


class _NoCacheWebAssetsMiddleware(BaseHTTPMiddleware):
    """Prevent stale app.js / speech.js when UI updates."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/assets/") and (
            path.endswith(".js") or path.endswith(".css") or path.endswith(".html")
        ):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response


app = FastAPI(title="Robot Ops Console")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(_NoCacheWebAssetsMiddleware)

session = RobotSession()
web_root = Path(__file__).resolve().parent / "web"
repo_assets = Path(__file__).resolve().parents[2] / "assets"
app.mount("/assets", StaticFiles(directory=str(web_root)), name="assets")
if repo_assets.is_dir():
    app.mount("/robot-assets", StaticFiles(directory=str(repo_assets)), name="robot-assets")
_three_vendor_ok, _three_vendor_msg = ensure_three_vendor(web_root)


def _warmup_whisper_background():
    try:
        from speech.transcribe import warmup_local_whisper

        msg = warmup_local_whisper()
        print(f"[ui] Whisper warmup: {msg}")
    except Exception as e:
        print(f"[ui] Whisper warmup skipped: {e}")


def _warmup_ollama_background():
    backend = (session.llm_backend or "ollama").lower()
    if backend not in ("ollama", "local"):
        return
    try:
        from agent.ollama_agent import check_ollama_ready, warmup_ollama

        model = os.environ.get("UI_MODEL")
        check_ollama_ready(model)
        msg = warmup_ollama(model)
        print(f"[ui] Ollama warmup: {msg}")
    except Exception as e:
        print(f"[ui] Ollama warmup skipped: {e}")


def _print_ready_banner():
    port = int(os.environ.get("UI_PORT", "8788"))
    print(
        f"\n[ui] Console READY — open http://127.0.0.1:{port}/ in your browser.\n"
        "[ui] (Idle now — no more logs until you use the UI. Ctrl+C to stop.)\n",
        flush=True,
    )


def _warmup_all_background():
    threads = [
        threading.Thread(target=_warmup_whisper_background, daemon=True),
        threading.Thread(target=_warmup_ollama_background, daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    _print_ready_banner()


def _connect_robot_background():
    try:
        session.connect()
    except Exception as e:
        print(f"[ui] Robot connect failed — UI still works in dry-run / gripper-only: {e}", flush=True)


@app.on_event("startup")
def _startup():
    global _three_vendor_ok, _three_vendor_msg
    ok, msg = ensure_three_vendor(web_root)
    _three_vendor_ok, _three_vendor_msg = ok, msg
    print(f"[ui] {msg}")
    threading.Thread(target=_connect_robot_background, daemon=True).start()
    threading.Thread(target=_warmup_all_background, daemon=True).start()


@app.on_event("shutdown")
def _shutdown():
    session.disconnect()


_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
}


@app.get("/")
def index():
    build = str(int(time.time()))
    html = (web_root / "index.html").read_text(encoding="utf-8")
    html = html.replace("UI_BUILD_ID", build)
    html = html.replace(
        'id="ui-build-tag"></span>',
        f'id="ui-build-tag">build {build}</span>',
    )
    return HTMLResponse(content=html, headers=_NO_CACHE_HEADERS)


@app.get("/api/config")
def api_config():
    return {
        "site": session.site.site_id,
        "display_name": session.site.display_name,
        "tool_schemas": TOOL_SCHEMAS,
        "arm_model": session.robot.arm_model,
        "simulated": session.robot.is_simulated,
        "llm_backend": session.llm_backend,
        "speech": speech_config(),
        "three_js": {
            "ready": _three_vendor_ok,
            "path": str(three_vendor_path(web_root)),
            "message": _three_vendor_msg,
        },
    }


@app.get("/api/state")
def api_state():
    payload = session.telemetry_payload()
    return {
        "status": "done",
        "state": payload["state"],
        "events": payload["events"],
        "goal_status": payload["goal_status"],
        "camera_path": session.last_camera_path,
        "detection": payload["detection"],
    }


@app.post("/api/tool")
def api_tool(call: ToolCall):
    try:
        return session.run_tool(name=call.name, inputs=call.inputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/goal")
def api_goal(call: GoalCall):
    goal = (call.goal or "").strip()
    if not goal:
        raise HTTPException(status_code=400, detail="Goal cannot be empty.")
    try:
        session.run_goal_async(goal)
        return {"status": "accepted", "goal": goal}
    except Exception as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@app.get("/api/goal_status")
def api_goal_status():
    return session.goal_status


@app.get("/api/speech/config")
def api_speech_config():
    return speech_config()


@app.post("/api/speech/transcribe")
async def api_speech_transcribe(audio: UploadFile = File(...)):
    try:
        data = await audio.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read audio: {e}") from e
    filename = audio.filename or "speech.webm"
    result = await asyncio.to_thread(transcribe_audio, data, filename=filename)
    if result.get("status") == "error":
        raise HTTPException(status_code=503, detail=result.get("reason", "transcription failed"))
    return result


@app.get("/api/camera/latest")
def api_camera_latest():
    path = session.last_camera_path
    if not path:
        raise HTTPException(status_code=404, detail="No captured image yet.")
    p = Path(path).resolve()
    if not p.exists():
        raise HTTPException(status_code=404, detail="Captured image file not found.")
    return FileResponse(str(p))


@app.get("/api/detection")
def api_detection():
    return session.last_detection


@app.get("/api/camera/live.jpg")
def api_camera_live_jpg(detect: int = 0):
    try:
        jpeg = session.live_jpeg_with_detection() if int(detect) == 1 else session.live_jpeg()
        return Response(content=jpeg, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            payload = await asyncio.to_thread(session.telemetry_payload)
            await ws.send_json(payload)
            delay = 1.0 if getattr(session.robot, "motion_busy", False) else 0.5
            await asyncio.sleep(delay)
    except WebSocketDisconnect:
        return
    except Exception as e:
        print(f"[telemetry ws] closed: {e}")
        try:
            await ws.close(code=1011)
        except Exception:
            pass
