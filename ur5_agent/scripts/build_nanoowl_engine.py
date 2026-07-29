#!/usr/bin/env python3
"""Build NanoOWL TensorRT engine for NVIDIA Thor GPU.

Run once after installing NanoOWL:
    python3 scripts/build_nanoowl_engine.py

Takes ~3–5 minutes. The engine is saved to:
    data/models/owlvit_base_patch32.engine

Then enable NanoOWL in the ops console:
    export NANOOWL_ENABLED=1
    python3 scripts/run_ops_console.py
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available. Check CUDA_VISIBLE_DEVICES=0 and driver.")
            sys.exit(1)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("ERROR: torch not installed.")
        sys.exit(1)

    try:
        from nanoowl.owl_predictor import OwlPredictor
    except ImportError:
        print("ERROR: nanoowl not installed.")
        print("  Fix: pip install git+https://github.com/NVIDIA-AI-IOT/nanoowl.git")
        sys.exit(1)

    out_dir = Path("data/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_path = str(out_dir / "owlvit_base_patch32.engine")

    if os.path.isfile(engine_path):
        print(f"Engine already exists: {engine_path}")
        ans = input("Rebuild? [y/N] ").strip().lower()
        if ans != "y":
            print("Skipping rebuild.")
            _run_test(engine_path)
            return

    print(f"\nBuilding TRT engine → {engine_path}")
    print("This takes 3–5 minutes on first run...\n")
    t0 = time.time()

    predictor = OwlPredictor(
        "google/owlvit-base-patch32",
        image_encoder_engine=engine_path,
    )

    elapsed = time.time() - t0
    print(f"\nEngine built in {elapsed:.0f}s → {engine_path}")
    _run_test(engine_path)
    print(f"\nTo enable NanoOWL in the ops console:")
    print(f"  export NANOOWL_ENABLED=1")
    print(f"  python3 scripts/run_ops_console.py")


def _run_test(engine_path: str):
    print("\nRunning quick detection test...")
    import numpy as np
    from PIL import Image as PILImage
    from nanoowl.owl_predictor import OwlPredictor

    predictor = OwlPredictor("google/owlvit-base-patch32", image_encoder_engine=engine_path)
    dummy = PILImage.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))

    # Warmup
    predictor.predict(image=dummy, text=["object"], threshold=0.1, pad_square=True)

    import time
    t0 = time.perf_counter()
    runs = 20
    for _ in range(runs):
        predictor.predict(image=dummy, text=["cup", "bottle", "tool"], threshold=0.1, pad_square=True)
    ms = (time.perf_counter() - t0) / runs * 1000
    print(f"NanoOWL TRT inference: {ms:.1f}ms/frame  ({1000/ms:.0f} FPS)")
    print("Engine OK.")


if __name__ == "__main__":
    main()
