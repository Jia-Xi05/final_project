from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnxruntime as ort

PathLike = Union[str, Path]

#決定 ONNX Runtime 要用哪個 execution provider。
def get_onnx_providers(prefer_gpu: bool = False) -> List[str]:
    """Get ONNX Runtime execution providers.

    Args:
        prefer_gpu: If True, try CUDA first when available.

    Returns:
        Ordered list of providers.
    """
    available = ort.get_available_providers()

    if prefer_gpu and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

#根據 .onnx 模型檔，建立一個 InferenceSession。
def create_onnx_session(
    model_path: PathLike,
    providers: Optional[Sequence[str]] = None,
    prefer_gpu: bool = False,
) -> ort.InferenceSession:
    """Create an ONNX Runtime inference session."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    if providers is None:
        providers = get_onnx_providers(prefer_gpu=prefer_gpu)

    return ort.InferenceSession(str(model_path), providers=list(providers))

#取得 ONNX 模型的所有輸入名稱。
def get_onnx_input_names(session: ort.InferenceSession) -> List[str]:
    """Return all ONNX input names."""
    return [node.name for node in session.get_inputs()]

#取得 ONNX 模型的所有輸出名稱。
def get_onnx_output_names(session: ort.InferenceSession) -> List[str]:
    """Return all ONNX output names."""
    return [node.name for node in session.get_outputs()]

#直接拿某一個 input name，預設拿第 0 個。
def get_onnx_input_name(session: ort.InferenceSession, index: int = 0) -> str:
    """Return a single ONNX input name by index."""
    inputs = get_onnx_input_names(session)
    if index >= len(inputs):
        raise IndexError(f"Input index {index} out of range. Total inputs: {len(inputs)}")
    return inputs[index]

#真正執行 ONNX 推論。
def run_onnx_inference(
    session: ort.InferenceSession,
    input_data: np.ndarray,
    input_name: Optional[str] = None,
    output_names: Optional[Sequence[str]] = None,
) -> List[np.ndarray]:
    """Run ONNX inference and return raw outputs.

    Args:
        session: Existing ONNX Runtime session.
        input_data: Prepared model input, usually float32 NumPy array.
        input_name: Input tensor name. Defaults to the first input.
        output_names: Specific outputs to request. Defaults to all outputs.
    """
    if not isinstance(input_data, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(input_data).__name__}")

    if input_name is None:
        input_name = get_onnx_input_name(session)

    if output_names is None:
        output_names = get_onnx_output_names(session)

    if input_data.dtype != np.float32:
        input_data = input_data.astype(np.float32)

    return session.run(list(output_names), {input_name: input_data})

#把 ONNX 模型的輸入輸出資訊整理成一個可讀的字典。
def describe_onnx_session(session: ort.InferenceSession) -> Dict[str, List[Dict[str, object]]]:
    """Return basic metadata about ONNX inputs and outputs.

    Useful for debugging model wiring during early development.
    """
    inputs = [
        {"name": node.name, "shape": node.shape, "type": node.type}
        for node in session.get_inputs()
    ]
    outputs = [
        {"name": node.name, "shape": node.shape, "type": node.type}
        for node in session.get_outputs()
    ]
    return {"inputs": inputs, "outputs": outputs}
