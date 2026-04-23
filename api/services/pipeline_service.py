from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from config.settings import OUTPUT_DIR
from modules.aggregator import RiskAggregator
from modules.detectors.scrfd_onnx_detector import SCRFDONNXDetector
from modules.detectors.yolov8_onnx_detector import YOLOv8ONNXDetector
from modules.module_a import ModuleAOpenClipSerpApi
from modules.module_c import ModuleCOcrSerpApiRoi
from modules.router import VisionRouter
from utils.image_utils import load_image, save_image

_YOLO_DETECTOR: YOLOv8ONNXDetector | None = None
_SCRFD_DETECTOR: SCRFDONNXDetector | None = None
_ROUTER: VisionRouter | None = None
_MODULE_A: ModuleAOpenClipSerpApi | None = None
_MODULE_C: ModuleCOcrSerpApiRoi | None = None
_AGGREGATOR: RiskAggregator | None = None


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


def get_router() -> VisionRouter:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = VisionRouter(get_yolo_detector(), get_scrfd_detector())
    return _ROUTER


def get_module_a() -> ModuleAOpenClipSerpApi:
    global _MODULE_A
    if _MODULE_A is None:
        _MODULE_A = ModuleAOpenClipSerpApi()
    return _MODULE_A


def get_module_c() -> ModuleCOcrSerpApiRoi:
    global _MODULE_C
    if _MODULE_C is None:
        _MODULE_C = ModuleCOcrSerpApiRoi()
    return _MODULE_C


def get_aggregator() -> RiskAggregator:
    global _AGGREGATOR
    if _AGGREGATOR is None:
        _AGGREGATOR = RiskAggregator()
    return _AGGREGATOR


def run_full_pipeline(image_path: str | Path) -> Dict[str, Any]:
    image_path = Path(image_path)
    image = load_image(image_path)

    router = get_router()
    module_a = get_module_a()
    module_c = get_module_c()
    aggregator = get_aggregator()

    router_result = router.run(image)
    module_a_result = module_a.run(image_path=image_path, route_result=router_result)
    module_c_result = module_c.run(image_path=image_path, image=image, route_result=router_result)
    final_result = aggregator.aggregate(router_result, module_a_result, module_c_result)

    annotated = router.annotate(image, router_result)
    annotated = module_c.annotate(annotated, module_c_result)

    output_name = f"pipeline_{image_path.stem}_{uuid4().hex[:8]}.jpg"
    output_path = OUTPUT_DIR / output_name
    save_image(output_path, annotated)

    return {
        "status": "success",
        "module": "router_a_c_aggregator_pipeline",
        "pipeline_name": "Router + Module A + Module C + Aggregator",
        "input_filename": image_path.name,
        "annotated_filename": output_name,
        "num_detections": router_result["num_detections"],
        "detections": router_result["detections"],
        "num_faces": router_result["num_faces"],
        "face_detections": router_result["face_detections"],
        "route_label": router_result["route_label"],
        "risk_level": final_result["risk_level"],
        "risk_score": final_result["risk_score"],
        "risk_score_percent": final_result["risk_score_percent"],
        "authenticity_verdict": final_result["authenticity_verdict"],
        "authenticity_label": final_result["authenticity_label"],
        "report": final_result["report"],
        "report_source": "aggregator",
        "modules": {
            "router": router_result,
            "a": module_a_result,
            "c": module_c_result,
            "aggregator": final_result,
        },
        "summary": {
            "headline": final_result["headline"],
            "route_label": router_result["route_label"],
            "object_count": router_result["num_detections"],
            "face_count": router_result["num_faces"],
            "risk_level": final_result["risk_level"],
            "risk_score_percent": final_result["risk_score_percent"],
            "authenticity_label": final_result["authenticity_label"],
        },
    }


def run_yolo_pipeline(image_path: str | Path) -> Dict[str, Any]:
    return run_full_pipeline(image_path)


def run_scam_pipeline(image_path: str | Path) -> Dict[str, Any]:
    return run_full_pipeline(image_path)
