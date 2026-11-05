"""ONNX YOLOv8 detector implementation using onnxruntime with cv2.dnn fallback."""

import ast
import json
import os
from pathlib import Path
from typing import Any
import cv2
import numpy as np

from vision.yolo_detector import Detection

COCO_CLASSES = [
    "person",
]


def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resizes and pads an image to target shape maintaining aspect ratio.

    Returns:
        (padded_image, scale_ratio, (pad_left, pad_top))
    """
    shape = img.shape[:2]  # (height, width)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Unpadded dimensions after scaling
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2.0
    dh = (new_shape[0] - new_unpad[1]) / 2.0

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    left = int(round(dw))
    right = new_shape[1] - (new_unpad[0] + left)
    top = int(round(dh))
    bottom = new_shape[0] - (new_unpad[1] + top)

    padded_img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded_img, r, (float(left), float(top))


def decode_yolov8(
    output: np.ndarray,
    conf_thresh: float,
    iou_thresh: float,
    orig_hw: tuple[int, int],
    input_hw: tuple[int, int],
    scale: float,
    pad: tuple[float, float],
    class_names: list[str],
) -> list[Detection]:
    """Decodes raw YOLOv8 prediction tensor and returns class-aware NMS detections."""
    if output is None or output.size == 0:
        return []

    # Expected output: (1, 4 + num_classes, N) or (1, N, 4 + num_classes)
    expected_channels = 4 + len(class_names) if class_names else (4 + len(COCO_CLASSES))
    if output.ndim == 3:
        if output.shape[1] == expected_channels:
            preds = np.transpose(output[0], (1, 0))  # (N, 4 + num_classes)
        elif output.shape[2] == expected_channels:
            preds = output[0]  # (N, 4 + num_classes)
        elif output.shape[1] < output.shape[2]:
            preds = np.transpose(output[0], (1, 0))
        else:
            preds = output[0]
    elif output.ndim == 2:
        if output.shape[0] == expected_channels:
            preds = np.transpose(output, (1, 0))
        elif output.shape[1] == expected_channels:
            preds = output
        elif output.shape[0] < output.shape[1]:
            preds = np.transpose(output, (1, 0))
        else:
            preds = output
    else:
        return []

    if preds.shape[1] <= 4:
        return []

    boxes = preds[:, :4]
    scores = preds[:, 4:]

    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    mask = confidences >= conf_thresh
    if not np.any(mask):
        return []

    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    dw, dh = pad
    x1 = (x1 - dw) / scale
    y1 = (y1 - dh) / scale
    x2 = (x2 - dw) / scale
    y2 = (y2 - dh) / scale

    orig_h, orig_w = orig_hw
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    # Class-aware NMS via cv2.dnn.NMSBoxes
    nms_boxes = [
        [float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])]
        for i in range(len(x1))
    ]
    unique_classes = np.unique(class_ids)
    detections: list[Detection] = []

    for cls_id in unique_classes:
        cls_indices = np.where(class_ids == cls_id)[0]
        sub_boxes = [nms_boxes[i] for i in cls_indices]
        sub_scores = [float(confidences[i]) for i in cls_indices]

        kept = cv2.dnn.NMSBoxes(sub_boxes, sub_scores, conf_thresh, iou_thresh)
        if len(kept) > 0:
            for idx in np.array(kept).flatten():
                orig_i = cls_indices[idx]
                label = (
                    class_names[cls_id]
                    if cls_id < len(class_names)
                    else str(cls_id)
                )
                detections.append(
                    Detection(
                        label=str(label),
                        confidence=float(confidences[orig_i]),
                        bbox=(
                            float(x1[orig_i]),
                            float(y1[orig_i]),
                            float(x2[orig_i]),
                            float(y2[orig_i]),
                        ),
                    )
                )

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


class OnnxYoloDetector:
    """YOLOv8 object detector using ONNX runtime with fallback to cv2.dnn."""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: tuple[int, int] | int | None = None,
        names: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.kwargs = kwargs

        if imgsz is None:
            self.imgsz = (640, 640)
        elif isinstance(imgsz, int):
            self.imgsz = (imgsz, imgsz)
        else:
            self.imgsz = imgsz

        self.session = None
        self.net = None
        self.backend = "unknown"
        self.names = names or self._resolve_class_names()

        self._init_runtime()

    def _resolve_class_names(self) -> list[str]:
        """Resolves class names from model metadata, sidecar txt, or COCO defaults."""
        # 1. Try sidecar <model>.names.txt
        p = Path(self.model_path)
        sidecar = p.with_suffix(".names.txt")
        if sidecar.exists():
            with open(sidecar, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                return lines

        alt_sidecar = Path(f"{self.model_path}.names.txt")
        if alt_sidecar.exists():
            with open(alt_sidecar, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                return lines

        # 2. Return standard COCO defaults (metadata will be checked on session init if onnxruntime)
        return COCO_CLASSES

    def _init_runtime(self) -> None:
        """Initializes onnxruntime InferenceSession or cv2.dnn fallback."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        try:
            import onnxruntime as ort  # type: ignore

            providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.backend = "onnxruntime"
            self.input_name = self.session.get_inputs()[0].name

            # Check if metadata provides custom class names
            meta = self.session.get_modelmeta().custom_metadata_map
            if "names" in meta and (not self.names or self.names == COCO_CLASSES):
                names_raw = meta["names"]
                try:
                    loaded = json.loads(names_raw)
                    if isinstance(loaded, dict):
                        self.names = [loaded[str(i)] for i in range(len(loaded))]
                    elif isinstance(loaded, list):
                        self.names = loaded
                except Exception:
                    try:
                        loaded = ast.literal_eval(names_raw)
                        if isinstance(loaded, dict):
                            self.names = [loaded[i] for i in sorted(loaded.keys())]
                    except Exception:
                        pass
        except Exception:
            # Fallback to OpenCV DNN
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.backend = "cv2.dnn"
            except Exception as err:
                raise RuntimeError(
                    f"Failed to load ONNX model with both onnxruntime and cv2.dnn: {err}"
                ) from err

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Runs object detection on the input frame."""
        if frame is None or frame.size == 0:
            return []

        orig_hw = frame.shape[:2]
        padded_img, scale, pad = letterbox(frame, new_shape=self.imgsz, color=(114, 114, 114))

        # BGR -> RGB, / 255.0, NCHW float32
        rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        blob = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)

        if self.backend == "onnxruntime" and self.session is not None:
            outputs = self.session.run(None, {self.input_name: blob})
            raw_output = outputs[0]
        elif self.net is not None:
            self.net.setInput(blob)
            raw_output = self.net.forward()
        else:
            raise RuntimeError("No active detector runtime session.")

        return decode_yolov8(
            output=raw_output,
            conf_thresh=self.conf,
            iou_thresh=self.iou,
            orig_hw=orig_hw,
            input_hw=self.imgsz,
            scale=scale,
            pad=pad,
            class_names=self.names,
        )
