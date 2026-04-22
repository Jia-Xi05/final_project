from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

PathLike = Union[str, Path]

#決定 PyTorch 要跑在 CPU 還是 GPU。
def get_device(prefer_gpu: bool = False) -> torch.device:
    """Get the recommended PyTorch device.

    Args:
        prefer_gpu: If True, return CUDA when available.

    Returns:
        torch.device object.
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

#把權重檔 .pth / .pt 載入到你已經建立好的 PyTorch 模型裡，然後切成推論模式。
def load_torch_model(
    model: nn.Module,
    weight_path: PathLike,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> nn.Module:
    """Load PyTorch weights into a model and switch it to eval mode.

    This helper supports common checkpoint formats such as:
    - raw state_dict
    - {"state_dict": ...}
    - {"model_state_dict": ...}

    Args:
        model: Instantiated PyTorch model.
        weight_path: Path to .pth / .pt weight file.
        device: Target device, e.g. "cpu" or "cuda".
        strict: Passed to model.load_state_dict().

    Returns:
        Loaded model in eval mode.
    """
    weight_path = Path(weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"Model weight file not found: {weight_path}")

    if device is None:
        device = get_device(prefer_gpu=False)
    device = torch.device(device)

    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    return model

#把模型輸出的 logits 轉成 softmax 機率。
@torch.no_grad()
def apply_softmax(logits: Union[torch.Tensor, np.ndarray], dim: int = -1) -> np.ndarray:
    """Apply softmax and return a NumPy array."""
    tensor = _to_tensor(logits)
    probs = torch.softmax(tensor, dim=dim)
    return probs.detach().cpu().numpy()

#把 logits 經過 sigmoid，轉成 0 到 1 之間的分數。
@torch.no_grad()
def apply_sigmoid(logits: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Apply sigmoid and return a NumPy array."""
    tensor = _to_tensor(logits)
    probs = torch.sigmoid(tensor)
    return probs.detach().cpu().numpy()

#把一個數值分數，用門檻切成類別標籤。
def classify_by_threshold(
    score: float,
    threshold: float = 0.5,
    positive_label: str = "fake",
    negative_label: str = "real",
) -> Tuple[str, float]:
    """Convert a score into a class label using a threshold.

    Returns:
        (predicted_label, score)
    """
    predicted_label = positive_label if score >= threshold else negative_label
    return predicted_label, float(score)

#把 NumPy 圖片轉成 PyTorch 模型可以吃的 tensor。
def image_to_tensor(
    image: np.ndarray,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    add_batch_dim: bool = True,
) -> torch.Tensor:
    """Convert an HWC NumPy image into a CHW float tensor.

    Notes:
        - Expects image in RGB order.
        - Converts uint8 [0,255] to float32 [0,1].
    """
    if image is None or image.size == 0:
        raise ValueError("Image is empty or None.")
    if image.ndim != 3:
        raise ValueError(f"Expected image shape HWC, got {image.shape}")

    tensor = torch.from_numpy(image).float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0

    tensor = tensor.permute(2, 0, 1)  # HWC -> CHW

    if normalize:
        mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
        std_tensor = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
        tensor = (tensor - mean_tensor) / std_tensor

    if add_batch_dim:
        tensor = tensor.unsqueeze(0)

    return tensor

#把 tensor 移到指定裝置上。
def move_tensor_to_device(tensor: torch.Tensor, device: Union[str, torch.device]) -> torch.Tensor:
    """Move a tensor to the requested device."""
    return tensor.to(torch.device(device))

#從 checkpoint 裡挖出真正可以給 model.load_state_dict() 用的那份權重。
def _extract_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            return checkpoint["model_state_dict"]
        if all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint
    raise ValueError("Unsupported checkpoint format. Expected a state_dict-like object.")

#把輸入資料統一轉成 torch.Tensor。
def _to_tensor(data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.float()
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    raise TypeError(f"Expected torch.Tensor or numpy.ndarray, got {type(data).__name__}")
