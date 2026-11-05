"""Core OnboardVisionNode coordinating detection, tracking, telemetry, and alerts."""

from collections.abc import Callable
import threading
import time
from typing import Any
import numpy as np

from vision.camera import Camera
from vision.disaster_detection import (
    PERSON_CLASSES,
    DetectionReport,
    DisasterAnalyzer,
)
from vision.emergency_alert import AlertManager, EmergencyAlert
from vision.telemetry import StaticTelemetry, Telemetry
from vision.thermal import thermal_confirms
from vision.tracker import SimpleTracker, Track
from vision.yolo_detector import Detection


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
    ) -> None:
        self.detector = detector
        self.camera = camera
        self.drone_id = drone_id
        self.telemetry = telemetry or StaticTelemetry()
        self.thermal_camera = thermal_camera
        self.tracker = tracker or SimpleTracker()
        self.alert_manager = alert_manager or AlertManager(cooldown_s=0.0)
        self.process_every = max(1, int(process_every))
        self.on_alert = on_alert
        self.on_frame = on_frame

        self.analyzer = DisasterAnalyzer()
        self.alerted_person_ids: set[int] = set()
        self.frames_processed: int = 0
        self.avg_inference_ms: float = 0.0
        self.alerts_raised: int = 0

        self._stop_event = threading.Event()
        self._start_time: float | None = None

    def process_frame(self, frame: np.ndarray) -> tuple[DetectionReport, list[Track]]:
        """Processes a single video frame through the full vision pipeline."""
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

        # f. Check for new person track IDs that have not yet been alerted
        current_person_ids = {p.track_id for p in people}
        new_person_ids = current_person_ids - self.alerted_person_ids
        if new_person_ids:
            alert = self.alert_manager.create(report)
            if alert is not None:
                self.alerted_person_ids.update(new_person_ids)
                self.alerts_raised += 1
                if self.on_alert is not None:
                    self.on_alert(alert)

        # g. Update frame counters and invoke on_frame hook
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

    def stop(self) -> None:
        """Signals processing loop to stop."""
        self._stop_event.set()

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
