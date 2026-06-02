#!/usr/bin/env python3
"""Robot Ops Console backend (telemetry + tool execution)."""

import asyncio
import datetime as dt
import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.sites import get_site
from config.settings import (
    CAMERA_OUTPUT_DIR,
    CAMERA_TYPE,
    YOLO_CONF,
    YOLO_ENABLED,
    YOLO_MODEL_PATH,
)
from agent.factory import create_agent
from policy.safety import PolicyEngine
from robot.factory import create_robot
from robot.tools import TOOL_SCHEMAS, execute_tool
from camera import RealSenseCamera


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
        self.yolo_model = None
        self.yolo_model_name = None
        self.last_detection: dict[str, Any] = {"count": 0, "labels": []}

    def connect(self):
        if not self.connected:
            self.robot.connect()
            self.connected = True
            if CAMERA_TYPE == "realsense":
                self.live_camera = RealSenseCamera()

    def disconnect(self):
        if self.connected:
            self.robot.disconnect()
            self.connected = False
        if self.live_camera:
            self.live_camera.disconnect()
            self.live_camera = None

    def state(self) -> dict[str, Any]:
        return self.robot.get_full_state()

    def run_tool(self, name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        self.policy.begin_goal()
        start = time.time()
        if name == "get_camera_frame":
            result = self.capture_and_save_frame(
                session_id=(inputs or {}).get("session_id", "lab"),
                prefix=(inputs or {}).get("prefix", "frame"),
            )
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
        if self.agent is None:
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
                self.goal_status["result"] = "done"
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

    @staticmethod
    def _draw_detection_boxes_fallback(frame):
        """Simple CV boxes fallback when YOLO model is unavailable."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 140)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h, w = frame.shape[:2]
        min_area = max(1200, int(0.002 * w * h))
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if area < min_area:
                continue
            boxes.append((x, y, bw, bh))
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:8]

        for x, y, bw, bh in boxes:
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (110, 220, 255), 1)
        labels = [f"obj_{i}" for i in range(1, len(boxes) + 1)]
        return frame, {"detector": "contour_fallback", "boxes": len(boxes), "labels": labels}

    def _ensure_yolo(self):
        if not YOLO_ENABLED:
            return None
        if self.yolo_model is not None:
            return self.yolo_model
        try:
            from ultralytics import YOLO
        except Exception:
            return None

        model_path = YOLO_MODEL_PATH or "yolov8n.pt"
        try:
            self.yolo_model = YOLO(model_path)
            self.yolo_model_name = model_path
            return self.yolo_model
        except Exception:
            self.yolo_model = None
            self.yolo_model_name = None
            return None

    def _draw_detection_boxes_yolo(self, frame):
        model = self._ensure_yolo()
        if model is None:
            return self._draw_detection_boxes_fallback(frame)

        try:
            results = model.predict(source=frame, conf=YOLO_CONF, verbose=False)
        except Exception:
            return self._draw_detection_boxes_fallback(frame)

        det_count = 0
        labels = []
        names = results[0].names if results and hasattr(results[0], "names") else {}
        boxes = results[0].boxes if results else None
        if boxes is not None:
            for b in boxes:
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                cls_id = int(b.cls[0]) if b.cls is not None else -1
                label = names.get(cls_id, f"id_{cls_id}") if isinstance(names, dict) else f"id_{cls_id}"
                labels.append(str(label))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (110, 220, 255), 1)
                det_count += 1

        return frame, {"detector": "yolo", "boxes": det_count, "labels": labels, "model": self.yolo_model_name}

    def live_jpeg_with_detection(self) -> bytes:
        if CAMERA_TYPE == "none":
            raise RuntimeError("Camera disabled (CAMERA_TYPE=none)")
        with self.camera_lock:
            if self.live_camera is None:
                self.live_camera = RealSenseCamera()
            frame = self.live_camera.capture_color_frame()
        frame, meta = self._draw_detection_boxes_yolo(frame)
        raw_labels = list(meta.get("labels", []))
        unique_labels = list(dict.fromkeys(raw_labels))
        self.last_detection = {
            "count": int(meta.get("boxes", 0)),
            "labels": raw_labels,
            "unique_labels": unique_labels,
            "detector": meta.get("detector", "unknown"),
        }
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


app = FastAPI(title="Robot Ops Console")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = RobotSession()
web_root = Path(__file__).resolve().parent / "web"
app.mount("/assets", StaticFiles(directory=str(web_root)), name="assets")


@app.on_event("startup")
def _startup():
    session.connect()


@app.on_event("shutdown")
def _shutdown():
    session.disconnect()


@app.get("/")
def index():
    return FileResponse(str(web_root / "index.html"))


@app.get("/api/config")
def api_config():
    return {
        "site": session.site.site_id,
        "display_name": session.site.display_name,
        "tool_schemas": TOOL_SCHEMAS,
        "arm_model": session.robot.arm_model,
        "simulated": session.robot.is_simulated,
        "llm_backend": session.llm_backend,
    }


@app.get("/api/state")
def api_state():
    try:
        return {
            "status": "done",
            "state": session.state(),
            "events": session.last_tool_events,
            "camera_path": session.last_camera_path,
            "detection": session.last_detection,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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
            payload: dict[str, Any] = {
                "ts": time.time(),
                "events": session.last_tool_events,
                "goal_status": session.goal_status,
                "detection": session.last_detection,
            }
            try:
                payload["state"] = session.state()
            except Exception as e:
                payload["state"] = {"error": str(e)}
            await ws.send_json(payload)
            await asyncio.sleep(0.35)
    except Exception:
        await ws.close()
