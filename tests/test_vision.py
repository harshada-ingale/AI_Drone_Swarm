"""Unit tests for vision preprocessing, YOLO decoding, and coordinate mapping."""

import numpy as np
import pytest

from vision.onnx_detector import COCO_CLASSES, decode_yolov8, letterbox
from vision.yolo_detector import Detection


def test_letterbox_and_coordinate_unmapping_1280x720() -> None:
    """A 1280x720 frame maps a box back to the correct original pixels."""
    orig_h, orig_w = 720, 1280
    dummy_frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

    padded_img, scale, pad = letterbox(dummy_frame, new_shape=(640, 640))
    # For 1280x720 to 640x640:
    # scale = 640 / 1280 = 0.5
    # scaled height = 720 * 0.5 = 360
    # pad_y = (640 - 360) / 2 = 140.0
    # pad_x = 0.0
    assert scale == pytest.approx(0.5)
    assert pad[0] == pytest.approx(0.0)
    assert pad[1] == pytest.approx(140.0)

    # Consider an object in original image at (x1=200, y1=100, x2=400, y2=300)
    # orig_cx = 300, orig_cy = 200, orig_w = 200, orig_h = 200
    # In letterbox space:
    # cx = 300 * 0.5 + 0 = 150.0
    # cy = 200 * 0.5 + 140 = 240.0
    # w = 200 * 0.5 = 100.0
    # h = 200 * 0.5 = 100.0
    num_classes = len(COCO_CLASSES)
    synthetic_pred = np.zeros((1, 4 + num_classes, 1), dtype=np.float32)
    synthetic_pred[0, 0:4, 0] = [150.0, 240.0, 100.0, 100.0]
    synthetic_pred[0, 4, 0] = 0.92  # class 0: "person"

    dets = decode_yolov8(
        output=synthetic_pred,
        conf_thresh=0.35,
        iou_thresh=0.45,
        orig_hw=(orig_h, orig_w),
        input_hw=(640, 640),
        scale=scale,
        pad=pad,
        class_names=COCO_CLASSES,
    )

    assert len(dets) == 1
    d = dets[0]
    assert d.label == "person"
    assert d.confidence == pytest.approx(0.92)
    # Check mapped bbox matches original pixels (200, 100, 400, 300)
    assert d.bbox[0] == pytest.approx(200.0, abs=1.0)
    assert d.bbox[1] == pytest.approx(100.0, abs=1.0)
    assert d.bbox[2] == pytest.approx(400.0, abs=1.0)
    assert d.bbox[3] == pytest.approx(300.0, abs=1.0)


def test_class_aware_nms_removes_duplicate_person_keeps_car() -> None:
    """NMS removes duplicate person detection but keeps car at the same position."""
    multi_classes = ["person", "car"]
    num_classes = len(multi_classes)
    # 3 predictions:
    # 0: person at [150, 240, 100, 100] with conf=0.90
    # 1: person at [152, 241, 98, 102] with conf=0.80 (duplicate box with high IoU)
    # 2: car at [150, 240, 100, 100] with conf=0.85 (different class: car is class 1)
    synthetic_pred = np.zeros((1, 4 + num_classes, 3), dtype=np.float32)

    synthetic_pred[0, :4, 0] = [150.0, 240.0, 100.0, 100.0]
    synthetic_pred[0, 4, 0] = 0.90  # person

    synthetic_pred[0, :4, 1] = [152.0, 241.0, 98.0, 102.0]
    synthetic_pred[0, 4, 1] = 0.80  # duplicate person

    synthetic_pred[0, :4, 2] = [150.0, 240.0, 100.0, 100.0]
    synthetic_pred[0, 5, 2] = 0.85  # class 1 (index 4 + 1 = 5): "car"

    dets = decode_yolov8(
        output=synthetic_pred,
        conf_thresh=0.35,
        iou_thresh=0.45,
        orig_hw=(720, 1280),
        input_hw=(640, 640),
        scale=0.5,
        pad=(0.0, 140.0),
        class_names=multi_classes,
    )

    assert len(dets) == 2
    labels = {d.label for d in dets}
    assert labels == {"person", "car"}
    # Verify higher confidence person was kept
    person_det = next(d for d in dets if d.label == "person")
    assert person_det.confidence == pytest.approx(0.90)


def test_single_class_victim_person_detection() -> None:
    """Verifies single-class victim person model decoding with COCO_CLASSES."""
    assert COCO_CLASSES == ["person"]
    num_classes = len(COCO_CLASSES)
    synthetic_pred = np.zeros((1, 4 + num_classes, 2), dtype=np.float32)

    # Box 1: high confidence person
    synthetic_pred[0, :4, 0] = [150.0, 240.0, 100.0, 100.0]
    synthetic_pred[0, 4, 0] = 0.95

    # Box 2: duplicate person with lower confidence (IoU ~ 1.0)
    synthetic_pred[0, :4, 1] = [151.0, 241.0, 100.0, 100.0]
    synthetic_pred[0, 4, 1] = 0.75

    dets = decode_yolov8(
        output=synthetic_pred,
        conf_thresh=0.35,
        iou_thresh=0.45,
        orig_hw=(720, 1280),
        input_hw=(640, 640),
        scale=0.5,
        pad=(0.0, 140.0),
        class_names=COCO_CLASSES,
    )

    assert len(dets) == 1
    assert dets[0].label == "person"
    assert dets[0].confidence == pytest.approx(0.95)


def test_low_confidence_boxes_are_filtered() -> None:
    """Detections below conf_thresh are discarded."""
    num_classes = len(COCO_CLASSES)
    synthetic_pred = np.zeros((1, 4 + num_classes, 2), dtype=np.float32)

    # Box 1: low confidence (0.25)
    synthetic_pred[0, :4, 0] = [100.0, 100.0, 50.0, 50.0]
    synthetic_pred[0, 4, 0] = 0.25

    # Box 2: high confidence (0.85)
    synthetic_pred[0, :4, 1] = [200.0, 200.0, 50.0, 50.0]
    synthetic_pred[0, 4, 1] = 0.85

    dets = decode_yolov8(
        output=synthetic_pred,
        conf_thresh=0.35,
        iou_thresh=0.45,
        orig_hw=(640, 640),
        input_hw=(640, 640),
        scale=1.0,
        pad=(0.0, 0.0),
        class_names=COCO_CLASSES,
    )

    assert len(dets) == 1
    assert dets[0].confidence == pytest.approx(0.85)
