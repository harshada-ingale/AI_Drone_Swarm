"""Unit tests for onboard vision tracking, alerting, and camera ingestion."""

import os
import tempfile
import cv2
import numpy as np
import pytest

from vision.camera import Camera
from vision.disaster_detection import DetectionReport
from vision.emergency_alert import AlertManager, EmergencyAlert
from vision.onboard_vision import OnboardVisionNode
from vision.thermal import thermal_confirms
from vision.tracker import SimpleTracker, Track, iou
from vision.yolo_detector import Detection


class FakeCamera:
    """Simulated camera delivering a predetermined sequence of blank frames."""

    def __init__(self, num_frames: int = 10, width: int = 640, height: int = 480) -> None:
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.current = 0
        self.ended = False

    def start(self) -> "FakeCamera":
        self.current = 0
        self.ended = False
        return self

    def read(self) -> tuple[np.ndarray | None, int]:
        if self.current >= self.num_frames:
            self.ended = True
            return (None, self.current)

        self.current += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return (frame, self.current)

    def stop(self) -> None:
        self.ended = True


class FakeThermalCamera:
    """Simulated thermal camera returning a specific thermal frame."""

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.ended = False

    def start(self) -> "FakeThermalCamera":
        return self

    def read(self) -> tuple[np.ndarray | None, int]:
        return (self.frame, 1)

    def stop(self) -> None:
        self.ended = True


class ScriptedDetector:
    """Mock detector returning predefined detections per frame."""

    def __init__(self, script: list[list[Detection]]) -> None:
        self.script = script
        self.index = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.index < len(self.script):
            res = self.script[self.index]
        else:
            res = []
        self.index += 1
        return res


def test_iou_values_and_tracker_motion() -> None:
    """Verifies IoU math and confirms tracker maintains track ID across frame motion."""
    # Test identical boxes (IoU = 1.0)
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    # Test non-overlapping boxes (IoU = 0.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)

    # Test partial overlap
    # Box A: 0..10 x 0..10 -> area 100
    # Box B: 5..15 x 0..10 -> area 100
    # Intersection: 5..10 x 0..10 -> area 50, Union: 150 -> IoU = 1/3
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50.0 / 150.0)

    # Test tracker maintains single ID for slowly shifting box
    tracker = SimpleTracker(iou_threshold=0.3, min_hits=3, max_missed=5)

    det1 = [Detection(label="person", confidence=0.9, bbox=(10, 10, 30, 30))]
    det2 = [Detection(label="person", confidence=0.88, bbox=(12, 12, 32, 32))]
    det3 = [Detection(label="person", confidence=0.92, bbox=(14, 14, 34, 34))]

    t1 = tracker.update(det1)
    assert len(t1) == 1
    assert t1[0].track_id == 1
    assert t1[0].hits == 1
    assert not tracker.is_confirmed(t1[0])

    t2 = tracker.update(det2)
    assert len(t2) == 1
    assert t2[0].track_id == 1
    assert t2[0].hits == 2
    assert not tracker.is_confirmed(t2[0])

    t3 = tracker.update(det3)
    assert len(t3) == 1
    assert t3[0].track_id == 1
    assert t3[0].hits == 3
    assert tracker.is_confirmed(t3[0])


def test_person_visible_12_frames_triggers_exactly_one_alert() -> None:
    """A person continuously visible for 12 frames should only raise 1 alert."""
    person_det = Detection(label="person", confidence=0.9, bbox=(100, 100, 200, 300))
    script = [[person_det] for _ in range(12)]

    detector = ScriptedDetector(script)
    camera = FakeCamera(num_frames=12)
    alerts: list[EmergencyAlert] = []

    node = OnboardVisionNode(
        detector=detector,
        camera=camera,
        on_alert=alerts.append,
    )
    node.run()

    assert len(alerts) == 1
    assert alerts[0].persons == 1
    assert alerts[0].priority in ("HIGH", "CRITICAL")


def test_person_in_only_one_frame_raises_zero_alerts() -> None:
    """A transient single-frame false positive should not achieve min_hits and raises 0 alerts."""
    person_det = Detection(label="person", confidence=0.9, bbox=(100, 100, 200, 300))
    # 1 frame with detection, followed by 9 empty frames
    script = [[person_det]] + [[] for _ in range(9)]

    detector = ScriptedDetector(script)
    camera = FakeCamera(num_frames=10)
    alerts: list[EmergencyAlert] = []

    node = OnboardVisionNode(
        detector=detector,
        camera=camera,
        on_alert=alerts.append,
    )
    node.run()

    assert len(alerts) == 0
    assert node.alerts_raised == 0


def test_second_person_appearing_later_raises_two_alerts_total() -> None:
    """First person triggers 1st alert; later second person triggers 2nd alert with persons=2."""
    p1 = Detection(label="person", confidence=0.9, bbox=(50, 50, 100, 100))
    p2 = Detection(label="person", confidence=0.85, bbox=(300, 300, 350, 350))

    # Frames 1..5: Person 1 only
    # Frames 6..12: Person 1 and Person 2
    script = [[p1] for _ in range(5)] + [[p1, p2] for _ in range(7)]

    detector = ScriptedDetector(script)
    camera = FakeCamera(num_frames=12)
    alerts: list[EmergencyAlert] = []

    node = OnboardVisionNode(
        detector=detector,
        camera=camera,
        on_alert=alerts.append,
    )
    node.run()

    assert len(alerts) == 2
    assert alerts[0].persons == 1
    assert alerts[1].persons == 2


def test_process_every_skips_frames() -> None:
    """process_every=5 over 10 frames should process exactly 2 frames (seq 1 and seq 6)."""
    script = [[] for _ in range(10)]
    detector = ScriptedDetector(script)
    camera = FakeCamera(num_frames=10)

    node = OnboardVisionNode(
        detector=detector,
        camera=camera,
        process_every=5,
    )
    node.run()

    assert node.frames_processed == 2


def test_thermal_confirmation_and_critical_priority() -> None:
    """Thermal signature inside bbox triggers CRITICAL alert; all-cold frame fails confirmation."""
    rgb_hw = (480, 640)
    bbox = (100.0, 100.0, 200.0, 200.0)

    # 1. All-cold frame test
    cold_frame = np.zeros((480, 640), dtype=np.uint8)
    assert not thermal_confirms(cold_frame, bbox, rgb_hw)

    # 2. Hot blob frame inside bbox
    hot_frame = np.zeros((480, 640), dtype=np.uint8)
    hot_frame[100:200, 100:200] = 255  # Exceeds default hot_threshold=200
    assert thermal_confirms(hot_frame, bbox, rgb_hw)

    # 3. Node integration test: Hot thermal feed elevates alert to CRITICAL
    person_det = Detection(label="person", confidence=0.95, bbox=bbox)
    script = [[person_det] for _ in range(5)]
    detector = ScriptedDetector(script)
    camera = FakeCamera(num_frames=5)
    thermal_cam = FakeThermalCamera(hot_frame)

    alerts: list[EmergencyAlert] = []
    node = OnboardVisionNode(
        detector=detector,
        camera=camera,
        thermal_camera=thermal_cam,
        on_alert=alerts.append,
    )
    node.run()

    assert len(alerts) == 1
    assert alerts[0].priority == "CRITICAL"


def test_camera_sequential_reads_temporary_video() -> None:
    """Camera(threaded=False) sequentially reads a generated .avi file to end."""
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, "test_sequential_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(video_path, fourcc, 10.0, (160, 120))

    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter MJPG format unsupported in this environment.")

    try:
        # Write 5 distinct test frames
        for i in range(5):
            f = np.full((120, 160, 3), i * 40, dtype=np.uint8)
            writer.write(f)
        writer.release()

        cam = Camera(source=video_path, threaded=False)
        cam.start()

        frames_read = 0
        while True:
            frame, seq = cam.read()
            if frame is None:
                break
            frames_read += 1

        cam.stop()
        assert frames_read == 5
        assert cam.ended is True
    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError:
                pass
