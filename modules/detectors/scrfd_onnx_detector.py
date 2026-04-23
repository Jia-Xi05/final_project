from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    from config.settings import SCRFD_CONF_THRES, SCRFD_INPUT_SIZE, SCRFD_MODEL_PATH, SCRFD_PREFER_GPU
except Exception:
    SCRFD_MODEL_PATH = None
    SCRFD_CONF_THRES = 0.5
    SCRFD_INPUT_SIZE = (640, 640)
    SCRFD_PREFER_GPU = False

from utils.image_utils import draw_bbox
from utils.onnx_utils import create_onnx_session, run_onnx_inference

PathLike = Union[str, Path]


class SCRFDONNXDetector:
    """SCRFD face detector for ONNX models exported by InsightFace."""

    def __init__(
        self,
        model_path: Optional[PathLike] = None,
        conf_threshold: float = SCRFD_CONF_THRES,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = SCRFD_INPUT_SIZE,
        prefer_gpu: bool = SCRFD_PREFER_GPU,
    ) -> None:
        self.model_path = Path(model_path or SCRFD_MODEL_PATH) if (model_path or SCRFD_MODEL_PATH) else None
        if self.model_path is None:
            raise ValueError("SCRFD model path is not set.")

        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.session = create_onnx_session(self.model_path, prefer_gpu=prefer_gpu)

        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self._center_cache: Dict[Tuple[int, int, int], np.ndarray] = {}

    def predict(self, image: np.ndarray) -> List[Dict[str, object]]:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image is empty or invalid.")

        blob, scales = self._preprocess(image)
        raw_outputs = run_onnx_inference(self.session, blob)
        return self._postprocess(raw_outputs, image.shape[:2], scales)

    def annotate(
        self,
        image: np.ndarray,
        detections: Sequence[Dict[str, object]],
        color: Tuple[int, int, int] = (0, 165, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        annotated = image.copy()
        for det in detections:
            annotated = draw_bbox(
                image=annotated,
                bbox=det["bbox"],
                label="face",
                score=float(det.get("score", 0.0)),
                color=color,
                thickness=thickness,
            )
            for point in det.get("landmarks", []):
                x, y = int(point[0]), int(point[1])
                cv2.circle(annotated, (x, y), 2, color, -1)
        return annotated

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        input_w, input_h = self.input_size
        orig_h, orig_w = image.shape[:2]
        scale_x = input_w / float(orig_w)
        scale_y = input_h / float(orig_h)

        resized = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1.0 / 128.0,
            size=(input_w, input_h),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        return blob.astype(np.float32), (scale_x, scale_y)

    def _postprocess(
        self,
        raw_outputs: Sequence[np.ndarray],
        original_shape: Tuple[int, int],
        scales: Tuple[float, float],
    ) -> List[Dict[str, object]]:
        if len(raw_outputs) not in (6, 9):
            raise ValueError(f"Unexpected SCRFD output count: {len(raw_outputs)}")

        num_branches = len(raw_outputs) // 3
        scores_list = raw_outputs[:num_branches]
        bbox_list = raw_outputs[num_branches: num_branches * 2]
        kps_list = raw_outputs[num_branches * 2:] if len(raw_outputs) == 9 else []

        all_scores: List[np.ndarray] = []
        all_boxes: List[np.ndarray] = []
        all_landmarks: List[np.ndarray] = []

        input_h, input_w = self.input_size[1], self.input_size[0]

        for branch_index, stride in enumerate(self._feat_stride_fpn[:num_branches]):
            scores = scores_list[branch_index].reshape(-1)
            bbox_preds = bbox_list[branch_index].reshape(-1, 4) * stride
            height = input_h // stride
            width = input_w // stride

            anchor_centers = self._get_anchor_centers(height, width, stride)
            positive = np.where(scores >= self.conf_threshold)[0]
            if positive.size == 0:
                continue

            boxes = self._distance2bbox(anchor_centers, bbox_preds)
            boxes = boxes[positive]
            scores = scores[positive]

            all_boxes.append(boxes)
            all_scores.append(scores)

            if kps_list:
                kps_preds = kps_list[branch_index].reshape(-1, 10) * stride
                landmarks = self._distance2kps(anchor_centers, kps_preds)[positive]
                all_landmarks.append(landmarks)

        if not all_boxes:
            return []

        boxes = np.vstack(all_boxes)
        scores = np.concatenate(all_scores)
        landmarks = np.vstack(all_landmarks) if all_landmarks else None

        keep = self._nms(boxes, scores, self.nms_threshold)
        scale_x, scale_y = scales
        orig_h, orig_w = original_shape

        detections: List[Dict[str, object]] = []
        for idx in keep:
            box = boxes[idx].copy()
            box[0::2] /= scale_x
            box[1::2] /= scale_y
            box[[0, 2]] = np.clip(box[[0, 2]], 0, orig_w - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, orig_h - 1)

            det: Dict[str, object] = {
                "bbox": [int(round(v)) for v in box.tolist()],
                "score": float(scores[idx]),
            }

            if landmarks is not None:
                points = landmarks[idx].copy().reshape(-1, 2)
                points[:, 0] /= scale_x
                points[:, 1] /= scale_y
                points[:, 0] = np.clip(points[:, 0], 0, orig_w - 1)
                points[:, 1] = np.clip(points[:, 1], 0, orig_h - 1)
                det["landmarks"] = [[int(round(x)), int(round(y))] for x, y in points.tolist()]

            detections.append(det)

        detections.sort(key=lambda item: float(item["score"]), reverse=True)
        return detections

    def _get_anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        if key in self._center_cache:
            return self._center_cache[key]

        grid = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        centers = (grid * stride).reshape(-1, 2)
        if self._num_anchors > 1:
            centers = np.repeat(centers, self._num_anchors, axis=0)

        self._center_cache[key] = centers
        return centers

    @staticmethod
    def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        preds = []
        for idx in range(0, distance.shape[1], 2):
            px = points[:, 0] + distance[:, idx]
            py = points[:, 1] + distance[:, idx + 1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = np.maximum(0.0, x2 - x1 + 1.0) * np.maximum(0.0, y2 - y1 + 1.0)
        order = scores.argsort()[::-1]

        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1.0)
            h = np.maximum(0.0, yy2 - yy1 + 1.0)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            remaining = np.where(iou <= threshold)[0]
            order = order[remaining + 1]

        return keep
