from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from modules.detectors.scrfd_onnx_detector import SCRFDONNXDetector


class ModuleBFaceDetector:
    """Module B: face detection wrapper around SCRFD."""

    def __init__(self, detector: SCRFDONNXDetector) -> None:
        self.detector = detector

    def run(self, image: np.ndarray) -> Dict[str, Any]:
        face_detections = self.detector.predict(image)
        strongest_score = max((float(item.get("score", 0.0)) for item in face_detections), default=0.0)

        return {
            "module_id": "B",
            "module_name": "Face Detection",
            "status": "success",
            "num_faces": len(face_detections),
            "face_detections": face_detections,
            "strongest_face_score": strongest_score,
            "summary": self._build_summary(face_detections, strongest_score),
        }

    def annotate(self, image: np.ndarray, face_detections: List[Dict[str, Any]]) -> np.ndarray:
        return self.detector.annotate(image, face_detections)

    @staticmethod
    def _build_summary(face_detections: List[Dict[str, Any]], strongest_score: float) -> str:
        if not face_detections:
            return "Module B did not detect faces in the current image."

        return (
            f"Module B detected {len(face_detections)} face(s). "
            f"The strongest face confidence was {strongest_score:.2%}."
        )
