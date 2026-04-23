from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from config.settings import OUTPUT_DIR
from modules.detectors.scrfd_onnx_detector import SCRFDONNXDetector
from modules.detectors.yolov8_onnx_detector import YOLOv8ONNXDetector
from utils.image_utils import load_image, save_image

_YOLO_DETECTOR: YOLOv8ONNXDetector | None = None
_SCRFD_DETECTOR: SCRFDONNXDetector | None = None


def get_yolo_detector() -> YOLOv8ONNXDetector:
    global _YOLO_DETECTOR
    if _YOLO_DETECTOR is None:
        _YOLO_DETECTOR = YOLOv8ONNXDetector()
    return _YOLO_DETECTOR


def get_scrfd_detector() -> SCRFDONNXDetector:
    global _SCRFD_DETECTOR
    if _SCRFD_DETECTOR is None:
        _SCRFD_DETECTOR = SCRFDONNXDetector()
    return _SCRFD_DETECTOR


def run_yolo_pipeline(image_path: str | Path) -> Dict[str, Any]:
    image_path = Path(image_path)
    image = load_image(image_path)

    yolo_detector = get_yolo_detector()
    scrfd_detector = get_scrfd_detector()

    detections = yolo_detector.predict(image)
    face_detections = scrfd_detector.predict(image)

    annotated = yolo_detector.annotate(image, detections)
    annotated = scrfd_detector.annotate(annotated, face_detections)

    output_name = f"yolo_{image_path.stem}_{uuid4().hex[:8]}.jpg"
    output_path = OUTPUT_DIR / output_name
    save_image(output_path, annotated)

    return {
        "status": "success",
        "module": "yolo",
        "input_filename": image_path.name,
        "num_detections": len(detections),
        "detections": detections,
        "num_faces": len(face_detections),
        "face_detections": face_detections,
        "annotated_filename": output_name,
    }
