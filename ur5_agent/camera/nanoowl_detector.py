"""Open-vocabulary object detection using NVIDIA NanoOWL on Jetson Thor.

NanoOWL wraps Google OWL-ViT in a TensorRT engine optimised for Jetson.
Unlike YOLOv8 (fixed 80 COCO classes), NanoOWL accepts *any* text query at
runtime — "find the red cup", "screwdriver", "robot gripper" — no retraining.

First run:
    python3 scripts/build_nanoowl_engine.py   # ~3–5 min, once only

Then set:
    export NANOOWL_ENABLED=1
    export NANOOWL_ENGINE=data/models/owlvit_base_patch32.engine
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from config.settings import (
    NANOOWL_ENABLED,
    NANOOWL_ENGINE_PATH,
    NANOOWL_DEFAULT_QUERIES,
    NANOOWL_THRESHOLD,
)


class NanoOwlDetector:
    """Open-vocabulary detector backed by NanoOWL TensorRT engine."""

    def __init__(self):
        self._predictor = None
        self._ready = False
        self._error: str | None = None

    # ------------------------------------------------------------------ setup

    def _ensure(self) -> bool:
        """Lazy-load the predictor. Returns True if ready."""
        if self._ready:
            return True
        if self._error:
            return False
        if not NANOOWL_ENABLED:
            self._error = "NANOOWL_ENABLED=false"
            return False

        # Set CUDA visible before importing torch/nanoowl.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        try:
            from nanoowl.owl_predictor import OwlPredictor

            engine = NANOOWL_ENGINE_PATH
            # Use TRT engine if it exists, otherwise pure-PyTorch (still CUDA-accelerated).
            if engine and os.path.isfile(engine):
                self._predictor = OwlPredictor(
                    "google/owlvit-base-patch32",
                    device=device,
                    image_encoder_engine=engine,
                )
                print(f"[nanoowl] Ready — TRT engine: {engine}  device: {device}")
            else:
                self._predictor = OwlPredictor(
                    "google/owlvit-base-patch32",
                    device=device,
                )
                print(f"[nanoowl] Ready — PyTorch mode  device: {device}  (~56ms/frame on Thor)")
            self._ready = True
            return True
        except Exception as e:
            self._error = str(e)
            print(f"[nanoowl] Load failed: {e}")
            return False

    # ----------------------------------------------------------------- detect

    def detect(
        self,
        frame: np.ndarray,
        queries: list[str] | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Detect objects matching *queries* in *frame* (BGR uint8).

        Returns the same schema as ObjectDetector.detect() so it's a drop-in:
            {detector, model, count, labels, unique_labels, objects: [{label,
             confidence, bbox:{x1,y1,x2,y2}, center:{x,y}}]}
        """
        if not self._ensure():
            return self._empty(error=self._error)

        texts = queries or NANOOWL_DEFAULT_QUERIES
        if not texts:
            return self._empty(error="no queries provided")

        thresh = threshold if threshold is not None else NANOOWL_THRESHOLD

        # NanoOWL expects RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        try:
            from PIL import Image as PILImage

            pil_img = PILImage.fromarray(rgb)
            # Pre-encode text so the model can accept it.
            text_encodings = self._predictor.encode_text(texts)
            output = self._predictor.predict(
                image=pil_img,
                text=texts,
                text_encodings=text_encodings,
                threshold=thresh,
                pad_square=True,
            )
        except Exception as e:
            return self._empty(error=str(e))

        objects = []
        boxes = output.boxes
        scores = output.scores
        labels_idx = output.labels

        def _to_np(t):
            """Convert tensor or array to numpy float32, always via CPU."""
            import torch
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().float().numpy()
            return np.array(t, dtype=np.float32)

        if boxes is not None and len(boxes):
            boxes_np = _to_np(boxes)
            scores_np = _to_np(scores)
            labels_np = _to_np(labels_idx).astype(int)

            # Normalise to absolute pixels if values are in [0,1].
            if boxes_np.max() <= 1.0:
                boxes_np[:, [0, 2]] *= w
                boxes_np[:, [1, 3]] *= h

            for i, (box, score, lbl_i) in enumerate(zip(boxes_np, scores_np, labels_np)):
                x1, y1, x2, y2 = (
                    int(max(0, box[0])),
                    int(max(0, box[1])),
                    int(min(w, box[2])),
                    int(min(h, box[3])),
                )
                label = texts[int(lbl_i)] if int(lbl_i) < len(texts) else f"obj_{i}"
                objects.append(
                    {
                        "label": label,
                        "confidence": round(float(score), 3),
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "center": {"x": (x1 + x2) // 2, "y": (y1 + y2) // 2},
                    }
                )

        # Sort by confidence descending.
        objects.sort(key=lambda o: o["confidence"], reverse=True)

        # NMS — remove boxes with IoU > 0.5 against a higher-confidence box.
        objects = _nms(objects, iou_threshold=0.5)

        # Cap at 10 detections to keep UI readable.
        objects = objects[:10]
        all_labels = [o["label"] for o in objects]
        return {
            "detector": "nanoowl",
            "model": "owlvit-base-patch32-trt",
            "queries": texts,
            "count": len(objects),
            "labels": all_labels,
            "unique_labels": list(dict.fromkeys(all_labels)),
            "objects": objects,
        }

    def detect_and_draw(
        self,
        frame: np.ndarray,
        queries: list[str] | None = None,
        threshold: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        meta = self.detect(frame, queries=queries, threshold=threshold)
        return _draw_boxes(frame, meta), meta

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str | None:
        return self._error

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _empty(error: str | None = None) -> dict[str, Any]:
        return {
            "detector": "nanoowl",
            "model": None,
            "count": 0,
            "labels": [],
            "unique_labels": [],
            "objects": [],
            "error": error,
        }


def _nms(objects: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Simple greedy NMS — keeps highest-confidence box, removes overlapping ones."""
    kept = []
    for obj in objects:  # already sorted by confidence descending
        bb = obj["bbox"]
        x1, y1, x2, y2 = bb["x1"], bb["y1"], bb["x2"], bb["y2"]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        suppress = False
        for k in kept:
            kb = k["bbox"]
            ix1 = max(x1, kb["x1"])
            iy1 = max(y1, kb["y1"])
            ix2 = min(x2, kb["x2"])
            iy2 = min(y2, kb["y2"])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            k_area = max(0, kb["x2"] - kb["x1"]) * max(0, kb["y2"] - kb["y1"])
            union = area + k_area - inter
            if union > 0 and inter / union > iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(obj)
    return kept


def _draw_boxes(frame: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    """Draw detection boxes — same style as ObjectDetector.draw_boxes()."""
    out = frame.copy()
    color = (0, 165, 255)  # Orange — distinct from YOLO green
    font = cv2.FONT_HERSHEY_SIMPLEX
    for obj in meta.get("objects", []):
        bb = obj.get("bbox") or {}
        x1, y1 = int(bb.get("x1", 0)), int(bb.get("y1", 0))
        x2, y2 = int(bb.get("x2", 0)), int(bb.get("y2", 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = str(obj.get("label", "object"))
        conf = obj.get("confidence")
        text = f"{label} {conf:.2f}" if conf is not None else label
        scale, thickness = 0.48, 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        ty = max(th + 4, y1 - 6)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), (20, 60, 10), -1)
        cv2.putText(out, text, (x1 + 2, ty), font, scale, color, thickness, cv2.LINE_AA)
    return out
