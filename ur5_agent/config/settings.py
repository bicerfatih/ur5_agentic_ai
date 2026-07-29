# config/settings.py — global defaults (override via env or CLI)

import os

# ── Robot network ──────────────────────────────────────
ROBOT_HOST = os.environ.get("ROBOT_HOST", "192.168.0.160")
YOUR_HOST = os.environ.get("YOUR_HOST", "192.168.0.85")

# ── Runtime (CLI can override) ───────────────────────────
ROBOT_TYPE = os.environ.get("ROBOT_TYPE", "ur5")
SITE_ID = os.environ.get("SITE_ID", "lab")
DRY_RUN = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")

# ── Safety limits (hard ceiling; site profile may be stricter) ──
MAX_JOINT_SPEED = 0.5
MAX_JOINT_ACCEL = 0.5
MAX_LINEAR_SPEED = 0.2
MAX_LINEAR_ACCEL = 0.2
MAX_SINGLE_MOVE_DOWN = 0.25

# ── Semantic move directions (robot BASE frame, not tool frame) ──
# Unit vectors as "dx,dy,dz". Lab cell: operator-facing layout where
# base -Y feels like "forward" and base -X feels like "left".
def _motion_vec(env_key: str, default: str) -> tuple[float, float, float]:
    raw = os.environ.get(env_key, default).strip()
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{env_key} must be three comma-separated numbers, got {raw!r}")
    return (parts[0], parts[1], parts[2])


MOTION_UP_VEC = _motion_vec("MOTION_UP_VEC", "0,0,1")
MOTION_DOWN_VEC = _motion_vec("MOTION_DOWN_VEC", "0,0,-1")
# Operator frame (standing behind robot) vs UR base X/Y is ~45° rotated.
# Left/Right confirmed OK; Forward/Back were inverted → swapped.
MOTION_FORWARD_VEC = _motion_vec("MOTION_FORWARD_VEC", "-0.7071,0.7071,0")
MOTION_BACKWARD_VEC = _motion_vec("MOTION_BACKWARD_VEC", "0.7071,-0.7071,0")
MOTION_LEFT_VEC = _motion_vec("MOTION_LEFT_VEC", "0.7071,0.7071,0")
MOTION_RIGHT_VEC = _motion_vec("MOTION_RIGHT_VEC", "-0.7071,-0.7071,0")

# Horizontal moves: "base" = robot base X/Y, "tool" = move along tool frame axes.
MOTION_HORIZONTAL_MODE = os.environ.get("MOTION_HORIZONTAL_MODE", "base").strip().lower()

# Image-based approach (eye-in-hand): map pixel offset → move_left/right/forward/back.
# Wrist cams are often mirrored vs operator left/right — default invert LR.
def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


APPROACH_IMAGE_INVERT_LR = _env_bool("APPROACH_IMAGE_INVERT_LR", False)
APPROACH_IMAGE_INVERT_FB = _env_bool("APPROACH_IMAGE_INVERT_FB", True)

# ── Safe home position (joint angles in radians) ───────
# Default factory pose; overridden by taught pose at HOME_POSE_PATH when present.
HOME_JOINTS = [0.0, -1.5707, 0.0, -1.5707, 0.0, 0.0]
HOME_POSE_PATH = os.environ.get(
    "HOME_POSE_PATH",
    os.path.join(os.path.dirname(__file__), "../../data/home_pose.json"),
).strip()

# ── LLM backend: ollama (default) | claude ─────────────
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")

# ── Ollama (local, offline-capable) ────────────────────
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "256"))
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))
# Keep model loaded between goals (e.g. 30m, 1h). Use -1 to hold until Ollama exits.
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m").strip()
OLLAMA_WARMUP_ENABLED = os.environ.get("OLLAMA_WARMUP_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)

# Agent (LLM) may call move_home only if user explicitly enables it (Tool Console Home button always works).
AGENT_ALLOW_MOVE_HOME = os.environ.get("AGENT_ALLOW_MOVE_HOME", "false").lower() in (
    "1",
    "true",
    "yes",
)

# ── Claude (optional) ──────────────────────────────────
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
CLAUDE_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "1024"))

# ── Speech-to-task (voice goals) ───────────────────────
# Record mode uploads audio to /api/speech/transcribe (local Whisper and/or OpenAI).
SPEECH_CLOUD_STT = os.environ.get("SPEECH_CLOUD_STT", "auto").strip().lower()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_WHISPER_MODEL = os.environ.get("OPENAI_WHISPER_MODEL", "whisper-1").strip()
# Local faster-whisper on Jetson (no Google browser STT). Set empty to disable.
# small: best accuracy/speed tradeoff on Jetson CPU (tiny hallucinates, base still weak)
SPEECH_LOCAL_WHISPER_MODEL = os.environ.get("SPEECH_LOCAL_WHISPER_MODEL", "small").strip()

# ── Gripper ────────────────────────────────────────────
# Robotiq URCap: PolyScope gripper ID 1 → socket SID 9 (port 63352)
GRIPPER_TYPE = os.environ.get("GRIPPER_TYPE", "robotiq")  # robotiq | dual_pin | digital_io | none
GRIPPER_POLYSCOPE_ID = int(os.environ.get("GRIPPER_POLYSCOPE_ID", os.environ.get("GRIPPER_ID", "1")))
ROBOTIQ_SOCKET_PORT = int(os.environ.get("ROBOTIQ_SOCKET_PORT", "63352"))
ROBOTIQ_SOCKET_SID = int(
    os.environ.get("ROBOTIQ_SOCKET_SID", str(8 + GRIPPER_POLYSCOPE_ID))
)  # ID 1 → 9
ROBOTIQ_OPEN_POS = int(os.environ.get("ROBOTIQ_OPEN_POS", "0"))
ROBOTIQ_CLOSE_POS = int(os.environ.get("ROBOTIQ_CLOSE_POS", "229"))  # 2F-85 typical max ~229
ROBOTIQ_SPEED = int(os.environ.get("ROBOTIQ_SPEED", "255"))
ROBOTIQ_FORCE = int(os.environ.get("ROBOTIQ_FORCE", "255"))
# After connect/activate: close | open | none (none = leave as-is after Robotiq activate)
GRIPPER_INITIAL = os.environ.get("GRIPPER_INITIAL", "close").strip().lower()
GRIPPER_TOGGLE_PAUSE_SEC = float(os.environ.get("GRIPPER_TOGGLE_PAUSE_SEC", "0.6"))

# Legacy pneumatic I/O (not used when GRIPPER_TYPE=robotiq):
# Pins 2 & 3 = feedback inputs only. Standard DO 0/1 = commands if using dual_pin.
GRIPPER_CMD_TARGET = os.environ.get(
    "GRIPPER_CMD_TARGET", os.environ.get("GRIPPER_IO_TARGET", "standard")
)
GRIPPER_CMD_PIN = int(os.environ.get("GRIPPER_CMD_PIN", os.environ.get("GRIPPER_PIN", "0")))
GRIPPER_CMD_OPEN_PIN = int(os.environ.get("GRIPPER_CMD_OPEN_PIN", os.environ.get("GRIPPER_OPEN_PIN", "0")))
GRIPPER_CMD_CLOSE_PIN = int(os.environ.get("GRIPPER_CMD_CLOSE_PIN", os.environ.get("GRIPPER_CLOSE_PIN", "1")))
GRIPPER_OPEN_HIGH = os.environ.get("GRIPPER_OPEN_HIGH", "true").lower() in ("1", "true", "yes")
GRIPPER_ACTIVE_LOW = os.environ.get("GRIPPER_ACTIVE_LOW", "false").lower() in ("1", "true", "yes")
# Pneumatic solenoids often need a short pulse (ms); 0 = hold DO on continuously
GRIPPER_PULSE_MS = int(os.environ.get("GRIPPER_PULSE_MS", "0"))
GRIPPER_PULSE_RELEASE = os.environ.get("GRIPPER_PULSE_RELEASE", "true").lower() in ("1", "true", "yes")
# Prefer URScript on the controller (works with External Control program running)
GRIPPER_USE_URSCRIPT = os.environ.get("GRIPPER_USE_URSCRIPT", "true").lower() in ("1", "true", "yes")

# FEEDBACK (digital inputs 2 & 3 — read only, from pendant I/O Tools):
GRIPPER_FEEDBACK_IN_OPEN = int(os.environ.get("GRIPPER_FEEDBACK_IN_OPEN", "2"))
GRIPPER_FEEDBACK_IN_CLOSED = int(os.environ.get("GRIPPER_FEEDBACK_IN_CLOSED", "3"))

# ── UR PolyScope programs (.urp on robot) ───────────────
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "29999"))
_default_programs = "fly2.urp,fly2"
ALLOWED_URP_PROGRAMS = [
    p.strip()
    for p in os.environ.get("ALLOWED_URP_PROGRAMS", _default_programs).split(",")
    if p.strip()
]
URP_PLAY_WAIT_SEC = float(os.environ.get("URP_PLAY_WAIT_SEC", "0.5"))

# ── Logging ────────────────────────────────────────────
LOG_FILE = os.environ.get("LOG_FILE", "logs/session.log")

# ── Camera (Intel RealSense) ───────────────────────────
CAMERA_TYPE = os.environ.get("CAMERA_TYPE", "realsense")  # realsense | none
CAMERA_SERIAL = os.environ.get("CAMERA_SERIAL", "").strip()
CAMERA_WIDTH = int(os.environ.get("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "480"))
CAMERA_FPS = int(os.environ.get("CAMERA_FPS", "30"))
CAMERA_OUTPUT_DIR = os.environ.get("CAMERA_OUTPUT_DIR", "../data/raw/images/lab")
CAMERA_DEPTH_ENABLED = os.environ.get("CAMERA_DEPTH_ENABLED", "true").lower() in ("1", "true", "yes")
HAND_EYE_CALIB_PATH = os.environ.get(
    "HAND_EYE_CALIB_PATH",
    os.path.join(os.path.dirname(__file__), "../../data/calibration/hand_eye.json"),
)
# Optional optical-axis flips before hand-eye (e.g. "x", "y", "xy") if reach goes the wrong way.
HAND_EYE_OPTICAL_FLIP = os.environ.get("HAND_EYE_OPTICAL_FLIP", "").strip().lower()
REACH_DONE_DIST_M = float(os.environ.get("REACH_DONE_DIST_M", "0.008"))
REACH_APPROACH_OFFSET_M = os.environ.get("REACH_APPROACH_OFFSET_M", "0,0,0.05")

# ── Vision (YOLO) ───────────────────────────────────────
YOLO_ENABLED = os.environ.get("YOLO_ENABLED", "true").lower() in ("1", "true", "yes")
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "").strip()
# Lower conf helps small / distant objects (was 0.35 — often missed ~40 cm targets).
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.20"))
# Inference size; larger = better for far/small objects (slower). 640 | 960 | 1280
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))
# Max detections per frame
YOLO_MAX_DET = int(os.environ.get("YOLO_MAX_DET", "50"))
# Device for YOLO: "cuda" uses Thor GPU (fastest), "cpu" fallback.
# Set CUDA_VISIBLE_DEVICES=0 in env before launching for Thor.
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", "cuda").strip()

# ── NanoOWL (open-vocabulary detection on Thor GPU) ─────
# Enable after running: python3 scripts/build_nanoowl_engine.py
NANOOWL_ENABLED = os.environ.get("NANOOWL_ENABLED", "true").lower() in ("1", "true", "yes")
NANOOWL_ENGINE_PATH = os.environ.get(
    "NANOOWL_ENGINE_PATH",
    os.path.join(os.path.dirname(__file__), "../../data/models/owlvit_base_patch32.engine"),
).strip()
# Comma-separated default object queries when none are specified at call time.
NANOOWL_DEFAULT_QUERIES = [
    q.strip()
    for q in os.environ.get(
        "NANOOWL_DEFAULT_QUERIES",
        "object,person,cup,bottle,box,tool,bag,phone,keyboard,chair,table"
    ).split(",")
    if q.strip()
]
NANOOWL_THRESHOLD = float(os.environ.get("NANOOWL_THRESHOLD", "0.12"))

# ── RL (camera-first policy execution) ──────────────────
RL_POLICY_PATH = os.environ.get("RL_POLICY_PATH", "").strip()
RL_OBS_MODE = os.environ.get("RL_OBS_MODE", "rgbd_state").strip()
RL_CONTROL_DT = float(os.environ.get("RL_CONTROL_DT", "0.12"))

# ── Isaac Sim bridge (GPU workstation) ─────────────────
ISAAC_BRIDGE_HOST = os.environ.get("ISAAC_BRIDGE_HOST", "127.0.0.1").strip()
ISAAC_BRIDGE_PORT = int(os.environ.get("ISAAC_BRIDGE_PORT", "9912"))
ISAAC_USD_PATH = os.environ.get(
    "ISAAC_USD_PATH",
    os.path.join(os.path.dirname(__file__), "../../assets/usd/lab_cell.usda"),
)
SIM_BACKEND = os.environ.get("SIM_BACKEND", "local").strip()  # local | isaac

# ── VLA (vision-language-action) ───────────────────────
# tool_routed | openvla | pi0 | groot | disabled
VLA_BACKEND = os.environ.get("VLA_BACKEND", "tool_routed").strip()
VLA_INSTRUCTION_DEFAULT = os.environ.get("VLA_INSTRUCTION_DEFAULT", "reach the object carefully")
VLA_MAX_STEP_M = float(os.environ.get("VLA_MAX_STEP_M", "0.008"))
VLA_SERVER_URL = os.environ.get("VLA_SERVER_URL", "").strip()  # e.g. http://gpu-host:8000
VLA_SERVER_TIMEOUT_S = float(os.environ.get("VLA_SERVER_TIMEOUT_S", "10.0"))
# "ur5" = scripts/vla/serve_*.py protocol; "openvla_native" = official openvla deploy.py
VLA_WIRE_FORMAT = os.environ.get("VLA_WIRE_FORMAT", "ur5").strip()
# Dataset key for action un-normalization (OpenVLA), e.g. "bridge_orig"
VLA_UNNORM_KEY = os.environ.get("VLA_UNNORM_KEY", "").strip()
# Scale applied when the server returns normalized actions in [-1, 1]
VLA_ACTION_SCALE_M = float(os.environ.get("VLA_ACTION_SCALE_M", "0.01"))
