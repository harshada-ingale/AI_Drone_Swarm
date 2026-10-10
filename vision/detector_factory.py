"""Factory function for instantiating detectors based on model file extension."""

from pathlib import Path
from typing import Any, Union
from vision.onnx_detector import OnnxYoloDetector
from vision.yolo_detector import YoloDetector

DetectorType = Union[OnnxYoloDetector, YoloDetector]


def load_detector(model_path: str, conf: float = 0.35, **kwargs: Any) -> DetectorType:
    """Loads appropriate detector based on file extension.

    Args:
        model_path: Path to model weights (.onnx, .pt, or .engine)
        conf: Confidence threshold
        **kwargs: Additional detector options

    Returns:
        OnnxYoloDetector for .onnx, YoloDetector for .pt / .engine

    Raises:
        ValueError: If model format is not supported
    """
    ext = Path(model_path).suffix.lower()
    if ext == ".onnx":
        return OnnxYoloDetector(model_path=model_path, conf=conf, **kwargs)
    elif ext in (".pt", ".engine"):
        return YoloDetector(model_path=model_path, conf=conf, **kwargs)
    else:
        raise ValueError(
            f"Unsupported model format '{ext}' for file '{model_path}'. "
            "Supported formats: .onnx, .pt, .engine"
        )
