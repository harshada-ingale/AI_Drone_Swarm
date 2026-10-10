"""YOLO detector interface and standard Detection dataclass."""

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class Detection:
    """Represents a single bounding box detection."""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels

    @property
    def center(self) -> tuple[float, float]:
        """Calculates center point (cx, cy) of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]


class YoloDetector:
    """YOLO detector wrapper that lazy-imports ultralytics only when initialized.

    Note: PyTorch and ultralytics may be restricted by system policy.
    Never import them at the top-level of any module.
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.35,
        classes: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_path = model_path
        self.conf = conf
        self.classes = classes
        self.kwargs = kwargs
        try:
            # Lazy import to avoid application control policy violations at module load
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(model_path)
        except Exception as err:
            raise RuntimeError(
                f"Failed to load YOLO model via ultralytics from {model_path}: {err}"
            ) from err

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Runs object detection on the input frame."""
        if frame is None or frame.size == 0:
            return []

        predict_kwargs = dict(self.kwargs)
        if self.classes is not None and "classes" not in predict_kwargs:
            predict_kwargs["classes"] = self.classes

        results = self.model.predict(
            frame, conf=self.conf, verbose=False, **predict_kwargs
        )
        detections: list[Detection] = []

        if not results:
            return detections

        res = results[0]
        names = getattr(res, "names", {})
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            label = names.get(cls_id, str(cls_id))
            detections.append(
                Detection(
                    label=str(label),
                    confidence=conf_val,
                    bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                )
            )

        return detections
