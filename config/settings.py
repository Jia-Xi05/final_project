from __future__ import annotations

from pathlib import Path

# ===== Base Paths =====
BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_DIR = BASE_DIR / "frontend"
API_DIR = BASE_DIR / "api"
CONFIG_DIR = BASE_DIR / "config"
INPUT_DIR = BASE_DIR / "input"
MODEL_DIR = BASE_DIR / "models"
MODULES_DIR = BASE_DIR / "modules"
UTILS_DIR = BASE_DIR / "utils"
OUTPUT_DIR = BASE_DIR / "outputs"
RUNS_DIR = BASE_DIR / "runs"

# ===== Create runtime folders if needed =====
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ===== YOLOv8 ONNX Detector =====
YOLO_MODEL_DIR = MODEL_DIR / "yolov8"
YOLO_MODEL_PATH = YOLO_MODEL_DIR / "yolov8n.onnx"
YOLO_CLASSES_PATH = YOLO_MODEL_DIR / "classes.txt"

YOLO_CONF_THRES = 0.25
YOLO_IOU_THRES = 0.45
YOLO_INPUT_SIZE = 640
YOLO_PREFER_GPU = False

# ===== SCRFD Face Detector =====
SCRFD_MODEL_DIR = MODEL_DIR / "scrfd"
SCRFD_MODEL_PATH = SCRFD_MODEL_DIR / "scrfd_10g_bnkps.onnx"
SCRFD_CONF_THRES = 0.50
SCRFD_INPUT_SIZE = (640, 640)
SCRFD_PREFER_GPU = False

# ===== API / App (optional placeholders for later) =====
API_HOST = "127.0.0.1"
API_PORT = 5000
DEBUG = True
