# Intel RealSense Camera (D455/F455)

This project now supports a camera tool: `get_camera_frame`.

Captured RGB frames are saved to:

- `data/raw/images/lab/` (default)

## 1) Install dependencies

Inside project virtualenv:

```bash
cd ~/Downloads/ur5_agentic_ai/ur5_agent
source robot_env/bin/activate
pip install -r requirements.txt
```

Install RealSense Python binding (`pyrealsense2`) based on your Jetson/Ubuntu setup.

## 2) Quick camera test

```bash
python3 scripts/test_camera.py
```

Expected: prints `Capture OK` and a saved image path.

## 3) Use from agent

```text
capture a camera frame
```

Tool output includes:

- `path`
- `shape`
- `serial`

## 4) Capture training dataset (image + robot state)

```bash
cd ~/Downloads/ur5_agentic_ai/ur5_agent
source robot_env/bin/activate
python3 scripts/capture_dataset.py --count 100 --interval 0.5 --session-id d405_lab_01
```

This writes:

- `data/raw/robot_sessions/<session-id>/images/*.jpg`
- `data/raw/robot_sessions/<session-id>/meta/*.json` (robot state per sample)
- `data/raw/robot_sessions/<session-id>/manifest.jsonl`

## 5) Optional env overrides

```bash
export CAMERA_SERIAL=<your_camera_serial>
export CAMERA_WIDTH=1280
export CAMERA_HEIGHT=720
export CAMERA_FPS=30
export CAMERA_OUTPUT_DIR=../data/raw/images/lab
```
