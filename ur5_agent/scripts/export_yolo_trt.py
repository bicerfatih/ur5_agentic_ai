#!/usr/bin/env python3
"""Export YOLOv8 model to TensorRT engine for NVIDIA Thor GPU.

Usage:
    python3 scripts/export_yolo_trt.py                    # exports yolov8n.pt → yolov8n.engine
    python3 scripts/export_yolo_trt.py --model yolov8s.pt # larger model
    python3 scripts/export_yolo_trt.py --model yolov8n.pt --imgsz 640 --half

The .engine file is saved next to the .pt file. Set YOLO_MODEL_PATH to its
path in your environment or .env to activate TRT inference:
    export YOLO_MODEL_PATH=/path/to/yolov8n.engine

TRT is ~5-10x faster than PyTorch CPU on Thor. First export takes 2-5 min.
"""

import argparse
import os
import sys
import time

# Ensure CUDA_VISIBLE_DEVICES is set before torch/ultralytics imports.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 to TensorRT engine")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO .pt model path or name")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument(
        "--half", action="store_true", default=True, help="FP16 precision (faster on Thor)"
    )
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace GB")
    args = parser.parse_args()

    try:
        import torch

        if not torch.cuda.is_available():
            print("ERROR: CUDA not available. Check CUDA_VISIBLE_DEVICES and driver.")
            sys.exit(1)
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("ERROR: torch not installed.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. pip install ultralytics")
        sys.exit(1)

    print(f"\nLoading {args.model}...")
    model = YOLO(args.model)

    precision = "FP16" if args.half else "FP32"
    print(f"Exporting to TensorRT ({precision}, imgsz={args.imgsz}, workspace={args.workspace}GB)...")
    print("This takes 2–5 minutes on first run (TRT engine compilation).\n")

    t0 = time.time()
    engine_path = model.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half,
        device=0,
        workspace=args.workspace,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nExport done in {elapsed:.0f}s")
    print(f"Engine saved to: {engine_path}")
    print(f"\nTo use it, set:")
    print(f"  export YOLO_MODEL_PATH={engine_path}")
    print(f"  export YOLO_DEVICE=cuda")
    print(f"\nOr add to your run command:")
    print(f"  YOLO_MODEL_PATH={engine_path} YOLO_DEVICE=cuda python3 scripts/run_ops_console.py")

    # Quick benchmark
    print("\nBenchmarking TRT engine...")
    trt_model = YOLO(engine_path)
    import numpy as np

    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    trt_model.predict(source=dummy, device=0, verbose=False)  # warmup
    t0 = time.perf_counter()
    runs = 20
    for _ in range(runs):
        trt_model.predict(source=dummy, device=0, verbose=False, imgsz=args.imgsz)
    ms = (time.perf_counter() - t0) / runs * 1000
    fps = 1000 / ms
    print(f"TRT inference: {ms:.1f}ms/frame  ({fps:.0f} FPS)")
    print("\nDone. Set YOLO_MODEL_PATH to the engine path above and restart the console.")


if __name__ == "__main__":
    main()
