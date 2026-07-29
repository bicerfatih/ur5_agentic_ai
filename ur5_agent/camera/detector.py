"""Object detection for agent tools and ops console (YOLO on NVIDIA Thor GPU + contour fallback)."""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from config.settings import YOLO_CONF, YOLO_DEVICE, YOLO_ENABLED, YOLO_IMGSZ, YOLO_MAX_DET, YOLO_MODEL_PATH


def _resolve_yolo_device() -> str:
    """Return a device string that Ultralytics accepts.

    On Jetson Thor, CUDA_VISIBLE_DEVICES may not be set when the process
    starts. We set it here (before torch is imported by Ultralytics) and
    return '0' for CUDA, or 'cpu' as fallback.
    """
    want = YOLO_DEVICE.lower()
    if want in ("cpu", ""):
        return "cpu"
    # Ensure the env var is set before Ultralytics/torch initialises.
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "0"
    except Exception:
        pass
    return "cpu"


class ObjectDetector:
    def __init__(self):
        self._yolo = None
        self._model_name: str | None = None
        self._device: str = _resolve_yolo_device()

    def _ensure_yolo(self):
        if not YOLO_ENABLED:
            return None
        if self._yolo is not None:
            return self._yolo
        try:
            from ultralytics import YOLO
        except Exception:
            return None

        model_path = YOLO_MODEL_PATH or "yolov8n.pt"
        try:
            self._yolo = YOLO(model_path)
            self._model_name = model_path
            # Warm up on the target device so first real inference is fast.
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self._yolo.predict(source=dummy, device=self._device, verbose=False, imgsz=64)
            print(f"[detector] YOLO {model_path} loaded on device={self._device}")
            return self._yolo
        except Exception as e:
            print(f"[detector] YOLO load failed ({e}), using contour fallback")
            self._yolo = None
            self._model_name = None
            return None

    @staticmethod
    def _detect_contour(frame: np.ndarray) -> dict[str, Any]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 140)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h, w = frame.shape[:2]
        # Allow smaller blobs so farther objects aren't discarded.
        min_area = max(400, int(0.0008 * w * h))
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh < min_area:
                continue
            boxes.append((x, y, bw, bh))
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:12]

        objects = []
        for i, (x, y, bw, bh) in enumerate(boxes, 1):
            label = f"obj_{i}"
            objects.append(
                {
                    "label": label,
                    "confidence": None,
                    "bbox": {"x1": x, "y1": y, "x2": x + bw, "y2": y + bh},
                    "center": {"x": x + bw // 2, "y": y + bh // 2},
                }
            )
        labels = [o["label"] for o in objects]
        return {
            "detector": "contour_fallback",
            "model": None,
            "count": len(objects),
            "labels": labels,
            "unique_labels": labels,
            "objects": objects,
        }

    def _detect_yolo(self, frame: np.ndarray) -> dict[str, Any]:
        model = self._ensure_yolo()
        if model is None:
            return self._detect_contour(frame)

        try:
            results = model.predict(
                source=frame,
                device=self._device,
                conf=YOLO_CONF,
                imgsz=YOLO_IMGSZ,
                max_det=YOLO_MAX_DET,
                verbose=False,
            )
        except Exception:
            return self._detect_contour(frame)

        names = results[0].names if results and hasattr(results[0], "names") else {}
        boxes = results[0].boxes if results else None
        objects = []
        if boxes is not None:
            for b in boxes:
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                cls_id = int(b.cls[0]) if b.cls is not None else -1
                conf = float(b.conf[0]) if b.conf is not None else 0.0
                label = (
                    names.get(cls_id, f"id_{cls_id}")
                    if isinstance(names, dict)
                    else f"id_{cls_id}"
                )
                objects.append(
                    {
                        "label": str(label),
                        "confidence": round(conf, 3),
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "center": {"x": (x1 + x2) // 2, "y": (y1 + y2) // 2},
                    }
                )

        labels = [o["label"] for o in objects]
        unique_labels = list(dict.fromkeys(labels))
        return {
            "detector": "yolo",
            "model": self._model_name,
            "count": len(objects),
            "labels": labels,
            "unique_labels": unique_labels,
            "objects": objects,
        }

    def detect(self, frame: np.ndarray) -> dict[str, Any]:
        """Run detection on a BGR frame; returns metadata dict (no drawing)."""
        if YOLO_ENABLED:
            meta = self._detect_yolo(frame)
        else:
            meta = self._detect_contour(frame)
        return meta

    @staticmethod
    def draw_boxes(frame: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
        out = frame.copy()
        color = (0, 220, 80)  # BGR green for boxes and labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        for obj in meta.get("objects", []):
            bb = obj.get("bbox") or {}
            x1, y1 = int(bb.get("x1", 0)), int(bb.get("y1", 0))
            x2, y2 = int(bb.get("x2", 0)), int(bb.get("y2", 0))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)

            label = str(obj.get("label", "object"))
            conf = obj.get("confidence")
            text = f"{label} {conf:.2f}" if conf is not None else label
            scale = 0.45
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
            ty = max(th + 4, y1 - 6)
            tx = x1
            # Small label background for readability
            cv2.rectangle(
                out,
                (tx, ty - th - 4),
                (tx + tw + 4, ty + baseline),
                (12, 28, 18),
                -1,
            )
            cv2.putText(
                out,
                text,
                (tx + 2, ty),
                font,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
        return out

    def detect_and_draw(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        meta = self.detect(frame)
        return self.draw_boxes(frame, meta), meta

    @staticmethod
    def to_summary(meta: dict[str, Any]) -> dict[str, Any]:
        """Compact dict for WebSocket / last_detection."""
        return {
            "count": int(meta.get("count", 0)),
            "labels": list(meta.get("labels", [])),
            "unique_labels": list(meta.get("unique_labels", [])),
            "detector": meta.get("detector", "unknown"),
            "model": meta.get("model"),
        }
