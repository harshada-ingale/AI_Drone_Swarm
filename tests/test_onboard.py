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

    def __init__(
        self,
        num_frames: int = 10,
        width: int = 640,
        height: int = 480,
        fps: float = 1.0,
    ) -> None:
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.fps = fps
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

    def get_timestamp(self) -> float:
        if self.fps > 0:
            return (self.current - 1) / float(self.fps)
        return 0.0

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


def test_tracker_person_with_1_frame_gap_keeps_same_id() -> None:
    """A person with a 1-frame detection gap keeps the same track ID."""
    tracker = SimpleTracker(iou_threshold=0.3, min_hits=1, max_missed=5)

    # Frame 1: Initial detection at (100, 100, 150, 200)
    d1 = [Detection(label="person", confidence=0.9, bbox=(100, 100, 150, 200))]
    t1 = tracker.update(d1)
    assert len(t1) == 1
    assert t1[0].track_id == 1
    assert t1[0].missed == 0

    # Frame 2: Person moves to (105, 100, 155, 200) -> establishes velocity (+5, 0, +5, 0)
    d2 = [Detection(label="person", confidence=0.9, bbox=(105, 100, 155, 200))]
    t2 = tracker.update(d2)
    assert len(t2) == 1
    assert t2[0].track_id == 1
    assert t2[0].missed == 0

    # Frame 3: 1-frame detection gap (no detections) -> track kept alive as lost
    t3 = tracker.update([])
    assert len(t3) == 1
    assert t3[0].track_id == 1
    assert t3[0].missed == 1

    # Frame 4: Detection reappears at (115, 100, 165, 200) -> matches predicted constant velocity
    d4 = [Detection(label="person", confidence=0.9, bbox=(115, 100, 165, 200))]
    t4 = tracker.update(d4)
    assert len(t4) == 1
    assert t4[0].track_id == 1
    assert t4[0].missed == 0


def test_tracker_two_people_crossing_paths_keep_distinct_ids() -> None:
    """Two people crossing paths with constant velocity keep distinct IDs."""
    tracker = SimpleTracker(iou_threshold=0.3, min_hits=1, max_missed=5)

    # Frame 1: Person A at x=100 (moving right), Person B at x=200 (moving left)
    f1 = [
        Detection(label="person", confidence=0.9, bbox=(100, 100, 140, 200)),
        Detection(label="person", confidence=0.9, bbox=(200, 100, 240, 200)),
    ]
    t1 = tracker.update(f1)
    assert len(t1) == 2
    id_a = next(t.track_id for t in t1 if t.bbox[0] < 150)
    id_b = next(t.track_id for t in t1 if t.bbox[0] > 150)
    assert id_a != id_b

    # Frame 2: Person A at x=120, Person B at x=180 -> establishes velocity: A is +20, B is -20
    f2 = [
        Detection(label="person", confidence=0.9, bbox=(120, 100, 160, 200)),
        Detection(label="person", confidence=0.9, bbox=(180, 100, 220, 200)),
    ]
    t2 = tracker.update(f2)
    assert next(t.track_id for t in t2 if t.bbox[0] == 120) == id_a
    assert next(t.track_id for t in t2 if t.bbox[0] == 180) == id_b

    # Frame 3: Crossing point (A at x=140, B at x=160)
    f3 = [
        Detection(label="person", confidence=0.9, bbox=(140, 100, 180, 200)),
        Detection(label="person", confidence=0.9, bbox=(160, 100, 200, 200)),
    ]
    t3 = tracker.update(f3)
    assert next(t.track_id for t in t3 if t.bbox[0] == 140) == id_a
    assert next(t.track_id for t in t3 if t.bbox[0] == 160) == id_b

    # Frame 4: Crossed (A continued to x=160, B continued to x=140)
    f4 = [
        Detection(label="person", confidence=0.9, bbox=(160, 100, 200, 200)),
        Detection(label="person", confidence=0.9, bbox=(140, 100, 180, 200)),
    ]
    t4 = tracker.update(f4)
    assert next(t.track_id for t in t4 if t.bbox[0] == 160) == id_a
    assert next(t.track_id for t in t4 if t.bbox[0] == 140) == id_b

    # Frame 5: Further apart (A at x=180, B at x=120)
    f5 = [
        Detection(label="person", confidence=0.9, bbox=(180, 100, 220, 200)),
        Detection(label="person", confidence=0.9, bbox=(120, 100, 160, 200)),
    ]
    t5 = tracker.update(f5)
    assert next(t.track_id for t in t5 if t.bbox[0] == 180) == id_a
    assert next(t.track_id for t in t5 if t.bbox[0] == 120) == id_b


def test_7_people_appearing_over_60_frames_fewer_alerts_with_min_interval() -> None:
    """7 people appearing over 60 frames give far fewer alerts than 7 with min_alert_interval_s=3 at 8 fps."""
    # 60 frames at 8 fps = 7.5 seconds of video time
    people_dets = [
        Detection(label="person", confidence=0.9, bbox=(i * 70, 100, i * 70 + 40, 200))
        for i in range(7)
    ]
    # People appear gradually across the clip
    appear_frames = [0, 8, 16, 24, 32, 40, 48]

    script = []
    for f in range(60):
        frame_dets = [people_dets[i] for i in range(7) if f >= appear_frames[i]]
        script.append(frame_dets)

    detector = ScriptedDetector(script)
    camera = FakeCamera(num_frames=60, fps=8.0)
    alerts: list[EmergencyAlert] = []

    node = OnboardVisionNode(
        detector=detector,
        camera=camera,
        min_alert_interval_s=3.0,
        on_alert=alerts.append,
    )
    node.run()

    # With min_alert_interval_s=3.0 over 7.5 seconds, alerts can only occur every 3.0s (e.g. t=0.25s, t=3.25s, t=6.25s)
    # Total alerts raised should be far fewer than 7 (at most 3 alerts)
    assert len(alerts) <= 3
    assert len(alerts) < 7
    assert node.alerts_raised == len(alerts)
    assert node.stats()["alerts_raised"] == len(alerts)


def test_high_water_mark_resets_after_quiet_period() -> None:
    """The high-water mark resets after no confirmed people have been visible for reset_after_s."""
    p1 = Detection(label="person", confidence=0.9, bbox=(50, 50, 100, 100))

    # Frames 0..4 (t=0..4s): Person 1 visible (confirmed at frame 2, t=2s) -> raises alert 1
    # Frames 5..20 (t=5..20s): 0 people visible (16s > reset_after_s=10.0s) -> high-water mark resets
    # Frames 21..26 (t=21..26s): Person 1 visible again (confirmed at frame 23, t=23s) -> raises alert 2!
    script = (
        [[p1] for _ in range(5)]
        + [[] for _ in range(16)]
        + [[p1] for _ in range(6)]
    )

    detector = ScriptedDetector(script)
    camera = FakeCamera(num_frames=27, fps=1.0)
    alerts: list[EmergencyAlert] = []

    node = OnboardVisionNode(
        detector=detector,
        camera=camera,
        min_alert_interval_s=3.0,
        reset_after_s=10.0,
        on_alert=alerts.append,
    )
    node.run()

    # Re-appearance after quiet period resets high-water mark, triggering 2nd alert
    assert len(alerts) == 2
    assert alerts[0].persons == 1
    assert alerts[1].persons == 1
    assert node.alerts_raised == 2


def test_csv_file_written_with_expected_header(tmp_path: pytest.TempPathFactory) -> None:
    """CSV file is written with the expected header and track rows."""
    from pathlib import Path

    csv_path = str(tmp_path / "tracks.csv")
    p1 = Detection(label="person", confidence=0.88, bbox=(10.0, 20.0, 30.0, 40.0))
    script = [[p1] for _ in range(3)]

    detector = ScriptedDetector(script)
    camera = FakeCamera(num_frames=3, fps=1.0)

    node = OnboardVisionNode(
        detector=detector,
        camera=camera,
        log_csv=csv_path,
    )
    node.run()

    csv_file = Path(csv_path)
    assert csv_file.exists()
    lines = csv_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2

    expected_header = "frame_index,track_id,label,x1,y1,x2,y2,confidence,hits,missed"
    assert lines[0] == expected_header

    # Verify track row columns
    first_row = lines[1].split(",")
    assert len(first_row) == 10
    assert first_row[0] == "1"  # frame_index
    assert first_row[1] == "1"  # track_id
    assert first_row[2] == "person"  # label


def test_run_onboard_cli_args_min_alert_interval_and_log_csv() -> None:
    """run_onboard CLI accepts --min-alert-interval and --log-csv."""
    import sys
    from run_onboard import parse_args

    orig_argv = sys.argv
    try:
        sys.argv = [
            "run_onboard.py",
            "--source", "0",
            "--min-alert-interval", "4.5",
            "--log-csv", "output/track_log.csv",
        ]
        args = parse_args()
        assert args.min_alert_interval == 4.5
        assert args.log_csv == "output/track_log.csv"
    finally:
        sys.argv = orig_argv

