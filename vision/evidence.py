"""Evidence extraction, thumbnail generation, and encoding for emergency alerts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import cv2
import numpy as np


def _make_blank_grey_jpeg(out_size: int = 128, jpeg_quality: int = 70) -> bytes:
    """Generates a solid grey fallback JPEG image."""
    canvas = np.full((out_size, out_size, 3), 128, dtype=np.uint8)
    _, encoded = cv2.imencode(
        ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )
    return bytes(encoded)


def make_thumbnail(
    frame: np.ndarray | None,
    bbox: Sequence[float],
    out_size: int = 128,
    pad: float = 0.5,
    min_side: int = 64,
    jpeg_quality: int = 70,
) -> bytes:
    """Crops and letterboxes a bounding box from a video frame with padding.

    - Expands box by pad (50% on each side) to preserve surrounding context.
    - Expands crop to min_side around box center if too small.
    - Clips to frame bounds and letterboxes into out_size x out_size with grey fill.
    - Never raises on degenerate/out-of-frame boxes; returns a grey JPEG.
    """
    try:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return _make_blank_grey_jpeg(out_size, jpeg_quality)

        if len(bbox) != 4:
            return _make_blank_grey_jpeg(out_size, jpeg_quality)

        x1, y1, x2, y2 = [float(v) for v in bbox]
        w = x2 - x1
        h = y2 - y1

        if w <= 0.0 or h <= 0.0 or not (np.isfinite(w) and np.isfinite(h)):
            return _make_blank_grey_jpeg(out_size, jpeg_quality)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Expand box by pad on each side
        w_pad = w * (1.0 + 2.0 * float(pad))
        h_pad = h * (1.0 + 2.0 * float(pad))

        # Grow around center to min_side if crop is smaller
        w_crop = max(w_pad, float(min_side))
        h_crop = max(h_pad, float(min_side))

        crop_x1 = cx - w_crop / 2.0
        crop_y1 = cy - h_crop / 2.0
        crop_x2 = cx + w_crop / 2.0
        crop_y2 = cy + h_crop / 2.0

        fh, fw = frame.shape[:2]
        x1_clip = max(0, min(fw, int(round(crop_x1))))
        y1_clip = max(0, min(fh, int(round(crop_y1))))
        x2_clip = max(0, min(fw, int(round(crop_x2))))
        y2_clip = max(0, min(fh, int(round(crop_y2))))

        if x2_clip <= x1_clip or y2_clip <= y1_clip:
            return _make_blank_grey_jpeg(out_size, jpeg_quality)

        cropped = frame[y1_clip:y2_clip, x1_clip:x2_clip]
        if cropped.size == 0:
            return _make_blank_grey_jpeg(out_size, jpeg_quality)

        if len(cropped.shape) == 2:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

        ch, cw = cropped.shape[:2]
        scale = min(out_size / float(cw), out_size / float(ch))
        nw = max(1, min(out_size, int(round(cw * scale))))
        nh = max(1, min(out_size, int(round(ch * scale))))

        resized = cv2.resize(
            cropped,
            (nw, nh),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )

        canvas = np.full((out_size, out_size, 3), 128, dtype=np.uint8)
        top = (out_size - nh) // 2
        left = (out_size - nw) // 2
        canvas[top : top + nh, left : left + nw] = resized

        ret, buf = cv2.imencode(
            ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        )
        if not ret or buf is None:
            return _make_blank_grey_jpeg(out_size, jpeg_quality)
        return bytes(buf)
    except Exception:
        return _make_blank_grey_jpeg(out_size, jpeg_quality)


@dataclass
class EvidenceItem:
    """Represents a single person detection evidence entry with crop."""

    track_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    thumbnail_jpeg: bytes | None


def build_evidence(
    frame: np.ndarray | None,
    tracks: Sequence[Any],
    person_labels: Sequence[str] | set[str] | None = None,
    max_items: int = 5,
    out_size: int = 128,
    jpeg_quality: int = 70,
    min_hits: int = 3,
) -> list[EvidenceItem]:
    """Builds evidence items from confirmed, currently visible person tracks.

    Sorts by confidence ASCENDING (lowest confidence first) so detections
    most in need of operator review appear first. Keeps at most max_items.
    If frame is None, thumbnail_jpeg will be None.
    """
    if person_labels is None:
        valid_labels = {"person", "pedestrian", "people"}
    else:
        valid_labels = {p.lower() for p in person_labels}

    candidates = []
    for t in tracks:
        label = getattr(t, "label", "")
        if label.lower() not in valid_labels:
            continue

        missed = getattr(t, "missed", 0)
        if missed != 0:
            continue

        if callable(getattr(t, "is_confirmed", None)):
            if not t.is_confirmed():
                continue
        else:
            hits = getattr(t, "hits", 1)
            if hits < min_hits:
                continue

        candidates.append(t)

    # Sort by confidence ASCENDING (least certain first)
    candidates.sort(key=lambda t: float(getattr(t, "confidence", 0.0)))
    selected = candidates[:max_items]

    evidence_items: list[EvidenceItem] = []
    for t in selected:
        bbox_raw = getattr(t, "bbox", (0.0, 0.0, 0.0, 0.0))
        bbox = (
            float(bbox_raw[0]),
            float(bbox_raw[1]),
            float(bbox_raw[2]),
            float(bbox_raw[3]),
        )
        conf = float(getattr(t, "confidence", 0.0))
        track_id = int(getattr(t, "track_id", 0))
        lbl = str(getattr(t, "label", "person"))

        thumb_bytes: bytes | None = None
        if frame is not None:
            thumb_bytes = make_thumbnail(
                frame=frame,
                bbox=bbox,
                out_size=out_size,
                jpeg_quality=jpeg_quality,
            )

        evidence_items.append(
            EvidenceItem(
                track_id=track_id,
                label=lbl,
                confidence=conf,
                bbox=bbox,
                thumbnail_jpeg=thumb_bytes,
            )
        )

    return evidence_items


def decode_thumbnail(jpeg: bytes) -> np.ndarray:
    """Decodes JPEG byte buffer into a BGR numpy image array."""
    if not jpeg:
        raise ValueError("Empty JPEG buffer")
    buf = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG bytes")
    return img
