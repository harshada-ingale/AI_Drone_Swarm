"""Simple IoU-based multi-object tracker for aerial surveillance."""

from dataclasses import dataclass
from typing import Sequence
from vision.yolo_detector import Detection


def iou(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    """Computes Intersection over Union (IoU) for two xyxy bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


@dataclass
class Track:
    """Represents a tracked object across video frames."""

    track_id: int
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float
    hits: int = 1
    missed: int = 0
    prev_bbox: tuple[float, float, float, float] | None = None


class SimpleTracker:
    """IoU and center-distance multi-object tracker with constant-velocity prediction."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        min_hits: int = 3,
        max_missed: int = 10,
        max_center_dist: float = 1.5,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_missed = max_missed
        self.max_center_dist = max_center_dist
        self.tracks: list[Track] = []
        self._next_id: int = 1

    def is_confirmed(self, track: Track) -> bool:
        """Returns True if track has achieved minimum detection hits."""
        return track.hits >= self.min_hits

    def predict_bbox(self, track: Track) -> tuple[float, float, float, float]:
        """Predicts track's bounding box using constant velocity from its last two boxes."""
        if track.prev_bbox is None:
            return track.bbox
        dx1 = track.bbox[0] - track.prev_bbox[0]
        dy1 = track.bbox[1] - track.prev_bbox[1]
        dx2 = track.bbox[2] - track.prev_bbox[2]
        dy2 = track.bbox[3] - track.prev_bbox[3]
        return (
            track.bbox[0] + dx1,
            track.bbox[1] + dy1,
            track.bbox[2] + dx2,
            track.bbox[3] + dy2,
        )

    def update(self, detections: list[Detection]) -> list[Track]:
        """Updates tracks with new detections using global matching.

        Steps:
          1. Predict each track's box with constant velocity before matching.
          2. Compute all (track, detection) pairs of the same label.
          3. Score pairs by IoU, falling back to normalized center distance when IoU is low.
          4. Sort by score and assign greedily without reusing tracks or detections.
          5. Update matched tracks and spawn new tracks for unmatched detections.
          6. Keep lost tracks alive for max_missed frames and allow them to be re-matched.
        """
        # 1. Predict each track's box with constant velocity
        predicted_boxes = [self.predict_bbox(t) for t in self.tracks]

        # 2 & 3. Compute all (track, detection) pairs of the same label and score them
        candidates: list[tuple[float, int, int]] = []
        for t_idx, track in enumerate(self.tracks):
            pred_bbox = predicted_boxes[t_idx]
            trk_h = max(1e-6, pred_bbox[3] - pred_bbox[1])
            trk_cx = (pred_bbox[0] + pred_bbox[2]) / 2.0
            trk_cy = (pred_bbox[1] + pred_bbox[3]) / 2.0

            for d_idx, det in enumerate(detections):
                if track.label.lower() != det.label.lower():
                    continue

                overlap = iou(det.bbox, pred_bbox)
                if overlap >= self.iou_threshold:
                    score = overlap
                else:
                    # Fallback to normalized center distance (distance divided by box height)
                    det_h = max(1e-6, det.bbox[3] - det.bbox[1])
                    box_h = max(det_h, trk_h)
                    det_cx = (det.bbox[0] + det.bbox[2]) / 2.0
                    det_cy = (det.bbox[1] + det.bbox[3]) / 2.0
                    dist = ((det_cx - trk_cx) ** 2 + (det_cy - trk_cy) ** 2) ** 0.5
                    norm_dist = dist / box_h

                    if norm_dist <= self.max_center_dist:
                        # Map normalized distance into [0, iou_threshold) so IoU matches take precedence
                        weight = max(0.0, 1.0 - (norm_dist / self.max_center_dist))
                        if self.iou_threshold > 0.0:
                            score = self.iou_threshold * weight * 0.99
                        else:
                            score = weight
                    else:
                        continue

                candidates.append((score, t_idx, d_idx))

        # 4. Sort by score descending and assign greedily
        candidates.sort(key=lambda c: c[0], reverse=True)

        matched_track_indices: set[int] = set()
        matched_det_indices: set[int] = set()

        for score, t_idx, d_idx in candidates:
            if t_idx in matched_track_indices or d_idx in matched_det_indices:
                continue

            matched_track_indices.add(t_idx)
            matched_det_indices.add(d_idx)

            track = self.tracks[t_idx]
            det = detections[d_idx]
            track.hits += 1
            track.missed = 0
            track.prev_bbox = track.bbox
            track.bbox = det.bbox
            track.confidence = det.confidence

        # 5. Increment missed count for unmatched tracks and advance their prediction
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_track_indices:
                track.missed += 1
                track.prev_bbox = track.bbox
                track.bbox = predicted_boxes[t_idx]

        # 6. Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_indices:
                new_track = Track(
                    track_id=self._next_id,
                    label=det.label,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    hits=1,
                    missed=0,
                    prev_bbox=None,
                )
                self._next_id += 1
                self.tracks.append(new_track)

        # 7. Evict dead tracks (missed > max_missed)
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return list(self.tracks)
