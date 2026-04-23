from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

# ===== Base Paths =====
BASE_DIR = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")

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

PADDLEX_CACHE_DIR = RUNS_DIR / "paddlex_cache"
HF_HOME_DIR = RUNS_DIR / "hf_cache"
TORCH_HOME_DIR = RUNS_DIR / "torch_cache"

os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PADDLEX_CACHE_DIR))
os.environ.setdefault("HF_HOME", str(HF_HOME_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME_DIR / "hub"))
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME_DIR))

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

# ===== Module C / Scam Fusion =====
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

IMGBB_API_KEY = os.getenv("IMGBB_API_KEY", "").strip()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "").strip()

SCAM_USAGE_FILE = RUNS_DIR / "gemini_usage.json"
SCAM_DAILY_LIMIT = int(os.getenv("SCAM_DAILY_LIMIT", "1500"))
SCAM_ENABLE_OCR = os.getenv("SCAM_ENABLE_OCR", "1") == "1"
SCAM_ENABLE_VISUAL_SEARCH = os.getenv("SCAM_ENABLE_VISUAL_SEARCH", "0") == "1"
SCAM_ENABLE_LLM = os.getenv("SCAM_ENABLE_LLM", "1") == "1"
SCAM_CLIP_MODEL_NAME = os.getenv("SCAM_CLIP_MODEL_NAME", "ViT-B-32")
SCAM_CLIP_PRETRAINED = os.getenv("SCAM_CLIP_PRETRAINED", "laion2b_s34b_b79k")
SCAM_OCR_LANG = os.getenv("SCAM_OCR_LANG", "ch")
SCAM_RISK_LABELS = [
    label.strip()
    for label in os.getenv(
        "SCAM_RISK_LABELS",
        "investment profit,scam line chat,stacks of cash,sexy profile,bank receipt",
    ).split(",")
    if label.strip()
]
