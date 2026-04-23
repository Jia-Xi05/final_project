from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    from config.settings import (
        YOLO_CONF_THRES,
        YOLO_MODEL_PATH,
    )
except Exception:
    YOLO_MODEL_PATH = None
    YOLO_CONF_THRES = 0.25

from utils.image_utils import draw_bbox
from utils.onnx_utils import create_onnx_session, run_onnx_inference

PathLike = Union[str, Path]


class YOLOv8ONNXDetector:
    """Lightweight YOLOv8 ONNX detector wrapper.

    This class is designed for your current project structure and focuses on the
    most common YOLOv8 ONNX export format used by Ultralytics.

    Typical usage:
        detector = YOLOv8ONNXDetector()
        results = detector.predict(image)

    Returned detection format:
        [
            {
                "bbox": [x1, y1, x2, y2],
                "score": 0.91,
                "class_id": 0,
                "class_name": "person"
            }
        ]
    """

    def __init__(
        self,
        model_path: Optional[PathLike] = None,
        class_names_path: Optional[PathLike] = None,
        conf_threshold: float = YOLO_CONF_THRES,
        iou_threshold: float = 0.45,
        input_size: Union[int, Tuple[int, int]] = 640,
        prefer_gpu: bool = False,
    ) -> None:
        self.model_path = Path(model_path or YOLO_MODEL_PATH) if (model_path or YOLO_MODEL_PATH) else None
        if self.model_path is None:
            raise ValueError("YOLO model path is not set. Pass model_path or define YOLO_MODEL_PATH in config/settings.py")

        self.class_names_path = Path(class_names_path) if class_names_path else self._infer_default_classes_path(self.model_path)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.input_size = self._normalize_input_size(input_size)
        self.prefer_gpu = bool(prefer_gpu)

        self.session = create_onnx_session(self.model_path, prefer_gpu=self.prefer_gpu)
        self.class_names = self._load_class_names(self.class_names_path)

    def predict(self, image: np.ndarray) -> List[Dict[str, object]]:
        """Run detection on a single BGR image and return parsed detections."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image is empty or invalid.")

        input_tensor, scale, pad = self._preprocess(image)
        raw_outputs = run_onnx_inference(self.session, input_tensor)
        detections = self._postprocess(
            raw_outputs=raw_outputs,
            original_shape=image.shape[:2],
            scale=scale,
            pad=pad,
        )
        return detections

    def annotate(
        self,
        image: np.ndarray,
        detections: Sequence[Dict[str, object]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw all detections on a copy of the input image."""
        annotated = image.copy()
        for det in detections:
            annotated = draw_bbox(
                image=annotated,
                bbox=det["bbox"],
                label=str(det.get("class_name", "object")),
                score=float(det.get("score", 0.0)),
                color=color,
                thickness=thickness,
            )
        return annotated

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox image and convert it to NCHW float32 input tensor.

        Returns:
            input_tensor: shape (1, 3, H, W), float32, range [0,1]
            scale: resize scale used to map detections back
            pad: (pad_x, pad_y)
        """
        if image.ndim != 3:
            raise ValueError(f"Expected image shape HWC, got {image.shape}")

        input_w, input_h = self.input_size
        orig_h, orig_w = image.shape[:2]

        scale = min(input_w / orig_w, input_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        pad_x = (input_w - new_w) // 2
        pad_y = (input_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))  # HWC -> CHW
        tensor = np.expand_dims(tensor, axis=0)   # CHW -> NCHW

        return tensor, scale, (pad_x, pad_y)

    def _postprocess(
        self,
        raw_outputs: Sequence[np.ndarray],
        original_shape: Tuple[int, int],
        scale: float,
        pad: Tuple[int, int],
    ) -> List[Dict[str, object]]:
        """Parse YOLOv8 ONNX outputs and return structured detections."""
        if not raw_outputs:
            return []

        preds = self._extract_prediction_tensor(raw_outputs[0])
        if preds.size == 0:
            return []

        # Expected final shape: (num_predictions, 4 + num_classes)
        if preds.ndim != 2 or preds.shape[1] < 5:
            raise ValueError(f"Unexpected YOLO output shape after parsing: {preds.shape}")

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

        keep_mask = scores >= self.conf_threshold
        if not np.any(keep_mask):
            return []

        boxes_xywh = boxes_xywh[keep_mask]
        scores = scores[keep_mask]
        class_ids = class_ids[keep_mask]

        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        boxes_xyxy = self._scale_boxes_back(
            boxes_xyxy=boxes_xyxy,
            original_shape=original_shape,
            scale=scale,
            pad=pad,
        )

        keep_indices = self._multiclass_nms(
            boxes=boxes_xyxy,
            scores=scores,
            class_ids=class_ids,
            iou_threshold=self.iou_threshold,
        )

        detections: List[Dict[str, object]] = []
        for idx in keep_indices:
            class_id = int(class_ids[idx])
            detections.append(
                {
                    "bbox": [int(v) for v in boxes_xyxy[idx].tolist()],
                    "score": float(scores[idx]),
                    "class_id": class_id,
                    "class_name": self._get_class_name(class_id),
                }
            )

        detections.sort(key=lambda x: float(x["score"]), reverse=True)
        return detections

    def _extract_prediction_tensor(self, output: np.ndarray) -> np.ndarray:
        """Normalize common YOLOv8 ONNX output shapes.

        Common cases:
        - (1, 84, 8400)  -> transpose to (8400, 84)
        - (1, 8400, 84)  -> squeeze to (8400, 84)
        - (8400, 84)     -> already OK
        """
        preds = np.array(output)

        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]

        if preds.ndim != 2:
            raise ValueError(f"Unsupported YOLO output shape: {output.shape}")

        # Heuristic: if first dim looks like channels/features and second dim looks like num_predictions
        if preds.shape[0] < preds.shape[1] and preds.shape[0] <= 256:
            preds = preds.T

        return preds

    @staticmethod
    def _xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
        boxes = boxes_xywh.copy()
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
        return boxes

    @staticmethod
    def _scale_boxes_back(
        boxes_xyxy: np.ndarray,
        original_shape: Tuple[int, int],
        scale: float,
        pad: Tuple[int, int],
    ) -> np.ndarray:
        """Map boxes from letterboxed model space back to original image space."""
        orig_h, orig_w = original_shape
        pad_x, pad_y = pad

        boxes = boxes_xyxy.copy()
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= scale

        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h - 1)
        return boxes.round().astype(np.int32)

    def _multiclass_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        iou_threshold: float,
    ) -> List[int]:
        """Apply NMS per class and return kept indices."""
        keep: List[int] = []
        unique_classes = np.unique(class_ids)

        for class_id in unique_classes:
            class_mask = class_ids == class_id
            class_indices = np.where(class_mask)[0]
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]

            kept_local = self._nms(class_boxes, class_scores, iou_threshold=iou_threshold)
            keep.extend(class_indices[i] for i in kept_local)

        return keep

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Basic NumPy NMS implementation."""
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0].astype(np.float32)
        y1 = boxes[:, 1].astype(np.float32)
        x2 = boxes[:, 2].astype(np.float32)
        y2 = boxes[:, 3].astype(np.float32)

        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
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

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h

            union = areas[i] + areas[order[1:]] - inter
            iou = np.zeros_like(inter)
            valid = union > 0
            iou[valid] = inter[valid] / union[valid]

            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]

        return keep

    @staticmethod
    def _normalize_input_size(input_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        if isinstance(input_size, int):
            if input_size <= 0:
                raise ValueError("input_size must be positive")
            return input_size, input_size

        if len(input_size) != 2:
            raise ValueError("input_size must be int or tuple(width, height)")

        width, height = int(input_size[0]), int(input_size[1])
        if width <= 0 or height <= 0:
            raise ValueError("input_size values must be positive")
        return width, height

    @staticmethod
    def _infer_default_classes_path(model_path: Path) -> Optional[Path]:
        candidate = model_path.parent / "classes.txt"
        return candidate if candidate.exists() else None

    @staticmethod
    def _load_class_names(class_names_path: Optional[Path]) -> List[str]:
        if class_names_path is None:
            return []
        if not class_names_path.exists():
            return []

        lines = class_names_path.read_text(encoding="utf-8").splitlines()
        class_names = [line.strip() for line in lines if line.strip()]
        return class_names

    def _get_class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"


if __name__ == "__main__":
    # Minimal local smoke-test example.
    # Replace the image path below with one of your test images when needed.
    test_image_path = None

    if test_image_path is None:
        print("Set test_image_path inside yolov8_onnx_detector.py to run a local smoke test.")
    else:
        from utils.image_utils import load_image, save_image

        detector = YOLOv8ONNXDetector()
        image = load_image(test_image_path)
        detections = detector.predict(image)
        print("Detections:", detections)
        annotated = detector.annotate(image, detections)
        save_image("outputs/yolo_test_result.jpg", annotated)
        print("Saved annotated result to outputs/yolo_test_result.jpg")
