"""Core OnboardVisionNode coordinating detection, tracking, telemetry, and alerts."""

from collections.abc import Callable
from pathlib import Path
import threading
import time
from typing import Any
import cv2
import numpy as np

from vision.camera import Camera
from vision.disaster_detection import (
    PERSON_CLASSES,
    DetectionReport,
    DisasterAnalyzer,
)
from vision.emergency_alert import AlertManager, EmergencyAlert
from vision.evidence import build_evidence
from vision.telemetry import StaticTelemetry, Telemetry
from vision.thermal import thermal_confirms
from vision.tracker import SimpleTracker, Track
from vision.yolo_detector import Detection

PRIORITY_RANKS = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


class OnboardVisionNode:
    """Edge computing vision node running on the drone payload.

    Coordinates camera capture, detection, multi-object tracking,
    thermal validation, and priority alerting.
    """

    def __init__(
        self,
        detector: Any,
        camera: Any,
        drone_id: str = "D1",
        telemetry: Any | None = None,
        thermal_camera: Any | None = None,
        tracker: SimpleTracker | None = None,
        alert_manager: AlertManager | None = None,
        process_every: int = 1,
        on_alert: Callable[[EmergencyAlert], None] | None = None,
        on_frame: Callable[[np.ndarray, DetectionReport, list[Track]], None] | None = None,
        min_alert_interval_s: float = 3.0,
        reset_after_s: float = 10.0,
        log_csv: str | None = None,
        include_thumbnails: bool = True,
        max_evidence: int = 5,
        thumb_size: int = 128,
        review_below: float = 0.50,
    ) -> None:
        self.detector = detector
        self.camera = camera
        self.drone_id = drone_id
        self.telemetry = telemetry or StaticTelemetry()
        self.thermal_camera = thermal_camera
        self.tracker = tracker or SimpleTracker()
        self.include_thumbnails = include_thumbnails
        self.max_evidence = int(max_evidence)
        self.thumb_size = int(thumb_size)
        self.review_below = float(review_below)
        self.alert_manager = alert_manager or AlertManager(
            cooldown_s=0.0, save=False, review_below=self.review_below
        )
        self.process_every = max(1, int(process_every))
        self.on_alert = on_alert
        self.on_frame = on_frame
        self.min_alert_interval_s = float(min_alert_interval_s)
        self.reset_after_s = float(reset_after_s)
        self.log_csv = log_csv

        self.analyzer = DisasterAnalyzer()
        self.alerted_person_ids: set[int] = set()
        self.highest_alerted_people: int = 0
        self.last_alert_priority: str | None = None
        self.last_alert_time: float | None = None
        self.last_people_seen_time: float | None = None

        self.frames_processed: int = 0
        self.avg_inference_ms: float = 0.0
        self.alerts_raised: int = 0

        self._stop_event = threading.Event()
        self._start_time: float | None = None

        self._csv_file = None
        if self.log_csv:
            csv_path = Path(self.log_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            self._csv_file.write(
                "frame_index,track_id,label,x1,y1,x2,y2,confidence,hits,missed\n"
            )
            self._csv_file.flush()

    def _get_current_time(self, timestamp: float | None = None) -> float:
        """Determines current frame timestamp, preferring video timestamps over wall-clock."""
        if timestamp is not None:
            return float(timestamp)
        if hasattr(self.camera, "get_timestamp") and callable(self.camera.get_timestamp):
            ts = self.camera.get_timestamp()
            if ts is not None:
                return float(ts)
        if hasattr(self.camera, "current"):
            fps = getattr(self.camera, "fps", 1.0)
            if fps > 0:
                return max(0.0, (self.camera.current - 1) / float(fps))
        if hasattr(self.camera, "cap") and getattr(self.camera, "cap", None) is not None:
            try:
                msec = self.camera.cap.get(cv2.CAP_PROP_POS_MSEC)
                if msec > 0.0:
                    return msec / 1000.0
                fps = self.camera.cap.get(cv2.CAP_PROP_FPS)
                seq = getattr(self.camera, "_seq", 0)
                if fps > 0.0 and seq > 0:
                    return (seq - 1) / float(fps)
            except Exception:
                pass
        return time.time()

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float | None = None,
    ) -> tuple[DetectionReport, list[Track]]:
        """Processes a single video frame through the full vision pipeline."""
        curr_time = self._get_current_time(timestamp)

        # a. Measure detector inference time with exponential moving average
        t0 = time.perf_counter()
        detections = self.detector.detect(frame)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        if self.frames_processed == 0:
            self.avg_inference_ms = infer_ms
        else:
            self.avg_inference_ms = 0.9 * self.avg_inference_ms + 0.1 * infer_ms

        # b. Update tracker with new detections
        tracks = self.tracker.update(detections)

        # c. Visible confirmed tracks (missed == 0)
        visible = [
            t for t in tracks if self.tracker.is_confirmed(t) and t.missed == 0
        ]
        people = [t for t in visible if t.label.lower() in PERSON_CLASSES]
        num_visible_people = len(people)

        # Reset high-water mark after no confirmed people have been visible for reset_after_s
        if num_visible_people > 0:
            self.last_people_seen_time = curr_time
        else:
            if self.last_people_seen_time is not None:
                if (curr_time - self.last_people_seen_time) >= self.reset_after_s:
                    self.highest_alerted_people = 0
                    self.last_alert_priority = None
                    self.last_people_seen_time = None
                    self.alerted_person_ids.clear()

        # d. Cross-confirm with thermal feed if available
        thermal_confirmed = False
        if self.thermal_camera is not None and len(people) > 0:
            thermal_frame, _ = self.thermal_camera.read()
            if thermal_frame is not None:
                rgb_hw = (frame.shape[0], frame.shape[1])
                for p in people:
                    if thermal_confirms(thermal_frame, p.bbox, rgb_hw):
                        thermal_confirmed = True
                        break

        # e. Build DetectionReport using ONLY confirmed visible tracks
        confirmed_detections = [
            Detection(label=t.label, confidence=t.confidence, bbox=t.bbox)
            for t in visible
        ]
        telem: Telemetry = self.telemetry.get()
        report = self.analyzer.analyze_detections(
            detections=confirmed_detections,
            frame_hw=(frame.shape[0], frame.shape[1]),
            drone_id=self.drone_id,
            altitude_m=telem.altitude_m,
            drone_xy=telem.xy,
            yaw_deg=telem.yaw_deg,
            hfov_deg=telem.hfov_deg,
            thermal_confirmed=thermal_confirmed,
        )
        report.timestamp = curr_time

        # f. Escalation-based alerting:
        # Raise alert only when confirmed visible people exceeds highest count already alerted,
        # OR priority rises, AND at least min_alert_interval_s has passed since last alert.
        if report.priority in ("HIGH", "CRITICAL") and num_visible_people > 0:
            count_rose = num_visible_people > self.highest_alerted_people
            last_p_rank = (
                PRIORITY_RANKS.get(self.last_alert_priority, 0)
                if self.last_alert_priority
                else 0
            )
            curr_p_rank = PRIORITY_RANKS.get(report.priority, 0)
            priority_rose = curr_p_rank > last_p_rank

            escalated = count_rose or priority_rose

            if self.last_alert_time is None or self.min_alert_interval_s <= 0.0:
                interval_ok = True
            else:
                interval_ok = (curr_time - self.last_alert_time) >= self.min_alert_interval_s

            if escalated and interval_ok:
                ev_frame = frame if self.include_thumbnails else None
                evidence = build_evidence(
                    frame=ev_frame,
                    tracks=people,
                    person_labels=PERSON_CLASSES,
                    max_items=self.max_evidence,
                    out_size=self.thumb_size,
                )
                alert = self.alert_manager.create(report, evidence=evidence)
                if alert is not None:
                    self.highest_alerted_people = max(self.highest_alerted_people, num_visible_people)
                    self.last_alert_priority = report.priority
                    self.last_alert_time = curr_time
                    self.alerts_raised += 1
                    self.alerted_person_ids.update({p.track_id for p in people})
                    if self.on_alert is not None:
                        self.on_alert(alert)

        # g. Write to CSV log if configured (all confirmed or tentative tracks)
        if self._csv_file is not None:
            frame_idx = self.frames_processed + 1
            for trk in tracks:
                x1, y1, x2, y2 = trk.bbox
                self._csv_file.write(
                    f"{frame_idx},{trk.track_id},{trk.label},"
                    f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},"
                    f"{trk.confidence:.4f},{trk.hits},{trk.missed}\n"
                )
            self._csv_file.flush()

        # h. Update frame counters and invoke on_frame hook
        self.frames_processed += 1
        if self.on_frame is not None:
            self.on_frame(frame, report, tracks)

        return (report, tracks)

    def run(self, max_frames: int | None = None) -> None:
        """Main camera ingestion and processing loop."""
        if hasattr(self.camera, "start"):
            self.camera.start()
        if self.thermal_camera is not None and hasattr(self.thermal_camera, "start"):
            try:
                self.thermal_camera.start()
            except Exception:
                pass

        self._start_time = time.time()
        last_seq = -1

        try:
            while not self._stop_event.is_set():
                if max_frames is not None and self.frames_processed >= max_frames:
                    break

                frame, seq = self.camera.read()
                if frame is None:
                    if getattr(self.camera, "ended", False):
                        break
                    time.sleep(0.005)
                    continue

                if seq == last_seq:
                    time.sleep(0.001)
                    continue

                last_seq = seq

                # Process every N frames
                if (seq - 1) % self.process_every != 0:
                    continue

                self.process_frame(frame)
        finally:
            if hasattr(self.camera, "stop"):
                self.camera.stop()
            if self.thermal_camera is not None and hasattr(self.thermal_camera, "stop"):
                try:
                    self.thermal_camera.stop()
                except Exception:
                    pass
            if self._csv_file is not None:
                try:
                    self._csv_file.close()
                except Exception:
                    pass
                self._csv_file = None

    def stop(self) -> None:
        """Signals processing loop to stop."""
        self._stop_event.set()
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None

    def __del__(self) -> None:
        if getattr(self, "_csv_file", None) is not None:
            try:
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None

    def stats(self) -> dict[str, Any]:
        """Returns runtime execution metrics."""
        elapsed = (time.time() - self._start_time) if self._start_time else 0.0
        fps = (self.frames_processed / elapsed) if elapsed > 0 else 0.0
        return {
            "frames_processed": self.frames_processed,
            "avg_inference_ms": round(self.avg_inference_ms, 2),
            "processing_fps": round(fps, 2),
            "alerts_raised": self.alerts_raised,
        }
