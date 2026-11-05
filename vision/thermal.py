"""Thermal sensor cross-confirmation logic."""

from typing import Sequence
import cv2
import numpy as np


def thermal_confirms(
    thermal: np.ndarray | None,
    bbox: Sequence[float],
    rgb_hw: tuple[int, int],
    hot_threshold: int = 200,
    min_hot_ratio: float = 0.10,
) -> bool:
    """Verifies whether an RGB detection bounding box exhibits heat signature in thermal feed.

    Args:
        thermal: Thermal frame (grayscale or BGR).
        bbox: (x1, y1, x2, y2) bounding box in RGB frame coordinates.
        rgb_hw: (height, width) dimensions of the RGB frame.
        hot_threshold: Grayscale pixel value (0-255) considered "hot".
        min_hot_ratio: Minimum ratio of hot pixels in the ROI required for confirmation.

    Returns:
        True if heat signature is confirmed, False otherwise or for invalid inputs.
    """
    if thermal is None or thermal.size == 0:
        return False

    rgb_h, rgb_w = rgb_hw
    if rgb_h <= 0 or rgb_w <= 0:
        return False

    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return False

    # Convert thermal frame to single-channel 8-bit grayscale
    if thermal.ndim == 3:
        gray = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
    else:
        gray = thermal

    th_h, th_w = gray.shape[:2]
    if th_h <= 0 or th_w <= 0:
        return False

    # Scale RGB bbox coordinates to thermal image dimensions
    scale_x = float(th_w) / float(rgb_w)
    scale_y = float(th_h) / float(rgb_h)

    tx1 = int(round(x1 * scale_x))
    ty1 = int(round(y1 * scale_y))
    tx2 = int(round(x2 * scale_x))
    ty2 = int(round(y2 * scale_y))

    # Clamp to thermal image bounds
    tx1 = max(0, min(th_w - 1, tx1))
    ty1 = max(0, min(th_h - 1, ty1))
    tx2 = max(0, min(th_w, tx2))
    ty2 = max(0, min(th_h, ty2))

    if tx2 <= tx1 or ty2 <= ty1:
        return False

    roi = gray[ty1:ty2, tx1:tx2]
    if roi.size == 0:
        return False

    hot_count = np.count_nonzero(roi >= hot_threshold)
    ratio = float(hot_count) / float(roi.size)

    return bool(ratio >= min_hot_ratio)
