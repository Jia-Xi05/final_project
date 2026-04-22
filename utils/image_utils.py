from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

PathLike = Union[str, Path]
BBoxLike = Sequence[Union[int, float]]


#把檔案路徑 → 變成圖片陣列
def load_image(image_path: PathLike, color_mode: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Load an image from disk.

    Args:
        image_path: Path to the image file.
        color_mode: OpenCV read mode. Defaults to cv2.IMREAD_COLOR.

    Returns:
        Image as a NumPy array.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If OpenCV fails to decode the image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), color_mode)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image

#把記憶體裡的圖片存回硬碟。
def save_image(output_path: PathLike, image: np.ndarray, create_dirs: bool = True) -> Path:
    """Save an image to disk.

    Args:
        output_path: Destination file path.
        image: Image array in OpenCV-compatible format.
        create_dirs: Whether to create parent folders automatically.

    Returns:
        Final output path.

    Raises:
        ValueError: If the image is empty or OpenCV fails to save it.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot save an empty image.")

    output_path = Path(output_path)
    if create_dirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise ValueError(f"Failed to save image: {output_path}")
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
    """Resize an image.

    Args:
        image: Input image.
        size: Either a single integer for square resize, or (width, height).
        interpolation: OpenCV interpolation method.
    """
    _validate_image(image)

    if isinstance(size, int):
        width, height = size, size
    else:
        width, height = size

    if width <= 0 or height <= 0:
        raise ValueError("Resize target must be positive.")

    return cv2.resize(image, (int(width), int(height)), interpolation=interpolation)

#根據 bounding box，把圖片某一塊切出來。
def crop_bbox(
    image: np.ndarray,
    bbox: BBoxLike,
    pad: int = 0,
    clip: bool = True,
) -> np.ndarray:
    """Crop a region from an image using bounding box coordinates.

    Args:
        image: Input image.
        bbox: Bounding box in (x1, y1, x2, y2) format.
        pad: Extra pixels to pad around the bounding box.
        clip: Whether to clip the bbox to image boundaries.

    Returns:
        Cropped image region.

    Raises:
        ValueError: If the bbox is invalid or crop is empty.
    """
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

#在圖片上畫框，必要時順便寫文字。
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

#只畫文字，不畫框。
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

#檢查傳進來的圖片是不是合法。
def _validate_image(image: np.ndarray) -> None:
    if image is None:
        raise ValueError("Image is None.")
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")
    if image.size == 0:
        raise ValueError("Image is empty.")

#把 bbox 解析成標準格式 (x1, y1, x2, y2)，而且都轉成整數。
def _parse_bbox(bbox: BBoxLike) -> Tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly 4 values: (x1, y1, x2, y2)")
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    return x1, y1, x2, y2

#把 label 和 score 組成你要顯示的文字。
def _build_label_text(label: Optional[str], score: Optional[float]) -> str:
    if label is None and score is None:
        return ""
    if label is not None and score is not None:
        return f"{label}: {score:.4f}"
    if label is not None:
        return label
    return f"{score:.4f}"
