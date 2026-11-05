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


class SimpleTracker:
    """Greedy IoU object tracker with confirmation and age-out logic."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        min_hits: int = 3,
        max_missed: int = 10,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_missed = max_missed
        self.tracks: list[Track] = []
        self._next_id: int = 1

    def is_confirmed(self, track: Track) -> bool:
        """Returns True if track has achieved minimum detection hits."""
        return track.hits >= self.min_hits

    def update(self, detections: list[Detection]) -> list[Track]:
        """Updates tracks with new detections using greedy IoU matching.

        Steps:
          1. Match detections (descending confidence) to best unmatched track with same label.
          2. Update matched tracks (hits+1, missed=0, update bbox/confidence).
          3. Increment missed count for unmatched tracks.
          4. Create new tracks for unmatched detections.
          5. Remove tracks where missed > max_missed.
        """
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        matched_track_ids: set[int] = set()
        unmatched_dets: list[Detection] = []

        for det in sorted_dets:
            best_track = None
            best_iou = -1.0

            for track in self.tracks:
                if track.track_id in matched_track_ids:
                    continue
                if track.label.lower() != det.label.lower():
                    continue

                overlap = iou(det.bbox, track.bbox)
                if overlap >= self.iou_threshold and overlap > best_iou:
                    best_iou = overlap
                    best_track = track

            if best_track is not None:
                best_track.hits += 1
                best_track.missed = 0
                best_track.bbox = det.bbox
                best_track.confidence = det.confidence
                matched_track_ids.add(best_track.track_id)
            else:
                unmatched_dets.append(det)

        # Increment missed count for tracks that received no update
        for track in self.tracks:
            if track.track_id not in matched_track_ids:
                track.missed += 1

        # Spawn new tracks for detections without existing matches
        for det in unmatched_dets:
            new_track = Track(
                track_id=self._next_id,
                label=det.label,
                bbox=det.bbox,
                confidence=det.confidence,
                hits=1,
                missed=0,
            )
            self._next_id += 1
            self.tracks.append(new_track)

        # Evict dead tracks
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return list(self.tracks)
