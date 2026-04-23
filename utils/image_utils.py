from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

PathLike = Union[str, Path]
BBoxLike = Sequence[Union[int, float]]


def load_image(image_path: PathLike, color_mode: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Load an image from disk with Windows Unicode-path support."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, color_mode)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


def save_image(output_path: PathLike, image: np.ndarray, create_dirs: bool = True) -> Path:
    """Save an image to disk with Windows Unicode-path support."""
    if image is None or image.size == 0:
        raise ValueError("Cannot save an empty image.")

    output_path = Path(output_path)
    if create_dirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix or ".jpg"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise ValueError(f"Failed to save image: {output_path}")

    encoded.tofile(str(output_path))
    return output_path


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB."""
    _validate_image(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to BGR."""
    _validate_image(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def resize_image(
    image: np.ndarray,
    size: Union[int, Tuple[int, int]],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize an image."""
    _validate_image(image)

    if isinstance(size, int):
        width, height = size, size
    else:
        width, height = size

    if width <= 0 or height <= 0:
        raise ValueError("Resize target must be positive.")

    return cv2.resize(image, (int(width), int(height)), interpolation=interpolation)


def crop_bbox(
    image: np.ndarray,
    bbox: BBoxLike,
    pad: int = 0,
    clip: bool = True,
) -> np.ndarray:
    """Crop a region from an image using bounding box coordinates."""
    _validate_image(image)
    x1, y1, x2, y2 = _parse_bbox(bbox)

    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad

    if clip:
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after processing: {(x1, y1, x2, y2)}")

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        raise ValueError("Cropped image is empty.")
    return cropped


def draw_bbox(
    image: np.ndarray,
    bbox: BBoxLike,
    label: Optional[str] = None,
    score: Optional[float] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a bounding box and optional label on a copy of the image."""
    _validate_image(image)
    x1, y1, x2, y2 = _parse_bbox(bbox)

    annotated = image.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

    text = _build_label_text(label=label, score=score)
    if text:
        text_y = max(20, y1 - 10)
        cv2.putText(
            annotated,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def draw_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    font_scale: float = 0.7,
    thickness: int = 2,
) -> np.ndarray:
    """Draw text on a copy of the image."""
    _validate_image(image)
    annotated = image.copy()
    cv2.putText(
        annotated,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return annotated


def _validate_image(image: np.ndarray) -> None:
    if image is None:
        raise ValueError("Image is None.")
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")
    if image.size == 0:
        raise ValueError("Image is empty.")


def _parse_bbox(bbox: BBoxLike) -> Tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly 4 values: (x1, y1, x2, y2)")
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    return x1, y1, x2, y2


def _build_label_text(label: Optional[str], score: Optional[float]) -> str:
    if label is None and score is None:
        return ""
    if label is not None and score is not None:
        return f"{label}: {score:.4f}"
    if label is not None:
        return label
    return f"{score:.4f}"
