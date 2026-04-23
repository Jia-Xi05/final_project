from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from modules.detectors.scrfd_onnx_detector import SCRFDONNXDetector
from modules.detectors.yolov8_onnx_detector import YOLOv8ONNXDetector


class VisionRouter:
    """Front gate: YOLOv8 + SCRFD routing according to project flow."""

    def __init__(self, yolo_detector: YOLOv8ONNXDetector, scrfd_detector: SCRFDONNXDetector) -> None:
        self.yolo_detector = yolo_detector
        self.scrfd_detector = scrfd_detector

    def run(self, image: np.ndarray) -> Dict[str, Any]:
        detections = self.yolo_detector.predict(image)
        face_detections = self.scrfd_detector.predict(image)

        person_detections = [item for item in detections if item.get("class_name") == "person"]
        selected_objects = [
            item for item in detections if item.get("class_name") not in {"person"}
        ] or person_detections

        has_person = bool(person_detections)
        has_face = bool(face_detections)

        routing_flags = {
            "run_module_a": has_person and not has_face,
            "run_deepfake_branch": has_person and has_face,
            "run_module_c": (not has_person) or bool(selected_objects),
        }

        route_label = self._build_route_label(has_person, has_face, routing_flags)
        summary = self._build_summary(
            detections=detections,
            face_detections=face_detections,
            routing_flags=routing_flags,
            route_label=route_label,
        )

        return {
            "module_id": "router",
            "module_name": "YOLOv8 + SCRFD Router",
            "status": "success",
            "route_label": route_label,
            "summary": summary,
            "detections": detections,
            "face_detections": face_detections,
            "num_detections": len(detections),
            "num_faces": len(face_detections),
            "person_count": len(person_detections),
            "face_count": len(face_detections),
            "selected_object_detections": selected_objects[:4],
            "routing_flags": routing_flags,
        }

    def annotate(self, image: np.ndarray, route_result: Dict[str, Any]) -> np.ndarray:
        annotated = self.yolo_detector.annotate(image, route_result["detections"])
        annotated = self.scrfd_detector.annotate(annotated, route_result["face_detections"])
        return annotated

    @staticmethod
    def _build_route_label(has_person: bool, has_face: bool, routing_flags: Dict[str, bool]) -> str:
        if routing_flags["run_deepfake_branch"]:
            return "person_with_face"
        if has_person and not has_face:
            return "person_without_face"
        return "non_person_or_object_route"

    @staticmethod
    def _build_summary(
        detections: List[Dict[str, Any]],
        face_detections: List[Dict[str, Any]],
        routing_flags: Dict[str, bool],
        route_label: str,
    ) -> str:
        return (
            f"Router detected {len(detections)} objects and {len(face_detections)} faces. "
            f"Route selected: {route_label}. "
            f"Module A={routing_flags['run_module_a']}, "
            f"Deepfake branch={routing_flags['run_deepfake_branch']}, "
            f"Module C={routing_flags['run_module_c']}."
        )
