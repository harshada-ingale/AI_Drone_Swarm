"""Unit tests for alert evidence extraction, thumbnails, payload sizing, and secure channel."""

import base64
import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from vision.disaster_detection import DetectionReport
from vision.emergency_alert import AlertManager, EmergencyAlert
from vision.evidence import (
    EvidenceItem,
    build_evidence,
    decode_thumbnail,
    make_thumbnail,
)
from vision.onboard_vision import OnboardVisionNode
from vision.tracker import Track
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


def test_make_thumbnail_various_boxes() -> None:
    """make_thumbnail produces decodable JPEGs of requested size without raising."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (150, 150), 30, (0, 255, 0), -1)

    # 1. Normal box
    normal_jpeg = make_thumbnail(frame, (100, 100, 200, 250), out_size=128)
    img_normal = decode_thumbnail(normal_jpeg)
    assert img_normal.shape == (128, 128, 3)

    # 2. Box at frame corner
    corner_jpeg = make_thumbnail(frame, (0, 0, 50, 50), out_size=128)
    img_corner = decode_thumbnail(corner_jpeg)
    assert img_corner.shape == (128, 128, 3)

    # 3. Tiny 8x20 px box (should expand around center to min_side 64)
    tiny_jpeg = make_thumbnail(frame, (200, 200, 208, 220), out_size=128, min_side=64)
    img_tiny = decode_thumbnail(tiny_jpeg)
    assert img_tiny.shape == (128, 128, 3)

    # 4. Degenerate box (x2 < x1)
    degen_jpeg = make_thumbnail(frame, (200, 100, 50, 250), out_size=128)
    img_degen = decode_thumbnail(degen_jpeg)
    assert img_degen.shape == (128, 128, 3)

    # 5. Out of frame box
    oof_jpeg = make_thumbnail(frame, (1000, 1000, 1100, 1100), out_size=128)
    img_oof = decode_thumbnail(oof_jpeg)
    assert img_oof.shape == (128, 128, 3)

    # 6. None frame fallback
    none_jpeg = make_thumbnail(None, (10, 10, 50, 50), out_size=64)
    img_none = decode_thumbnail(none_jpeg)
    assert img_none.shape == (64, 64, 3)


def test_build_evidence_filtering_and_ordering() -> None:
    """build_evidence keeps at most max_items, orders ascending by confidence, and ignores unconfirmed/missing tracks."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    tracks = [
        Track(track_id=1, label="person", bbox=(10, 10, 60, 100), confidence=0.85, hits=4, missed=0),
        Track(track_id=2, label="person", bbox=(20, 20, 70, 110), confidence=0.42, hits=3, missed=0),
        Track(track_id=3, label="pedestrian", bbox=(30, 30, 80, 120), confidence=0.65, hits=5, missed=0),
        Track(track_id=4, label="person", bbox=(40, 40, 90, 130), confidence=0.95, hits=1, missed=0),  # unconfirmed (hits < 3)
        Track(track_id=5, label="person", bbox=(50, 50, 100, 140), confidence=0.20, hits=4, missed=1), # missing (missed != 0)
        Track(track_id=6, label="car", bbox=(60, 60, 150, 150), confidence=0.99, hits=5, missed=0),     # non-person
        Track(track_id=7, label="people", bbox=(70, 70, 120, 160), confidence=0.55, hits=3, missed=0),
    ]

    items = build_evidence(
        frame=frame,
        tracks=tracks,
        person_labels={"person", "pedestrian", "people"},
        max_items=3,
        out_size=128,
        min_hits=3,
    )

    # Eligible: Track 2 (0.42), Track 7 (0.55), Track 3 (0.65), Track 1 (0.85)
    # Ordered ASCENDING, top 3: Track 2, Track 7, Track 3
    assert len(items) == 3
    assert [it.track_id for it in items] == [2, 7, 3]
    assert [it.confidence for it in items] == [0.42, 0.55, 0.65]
    for it in items:
        assert it.thumbnail_jpeg is not None
        assert decode_thumbnail(it.thumbnail_jpeg).shape == (128, 128, 3)

    # If frame is None, items are returned with thumbnail_jpeg=None
    items_no_frame = build_evidence(
        frame=None,
        tracks=tracks,
        person_labels={"person", "pedestrian", "people"},
        max_items=3,
    )
    assert len(items_no_frame) == 3
    for it in items_no_frame:
        assert it.thumbnail_jpeg is None


def test_emergency_alert_evidence_and_review_flag() -> None:
    """EmergencyAlert with evidence correctly sets needs_review, min/mean confidence, and summary()."""
    report = DetectionReport(
        drone_id="D1",
        timestamp=100.0,
        persons=2,
        vehicles=0,
        hazards={},
        priority="HIGH",
        people_positions=[(10.0, 20.0)],
    )

    # 1. With confidence < review_below (0.50): needs_review is True
    mgr_review = AlertManager(cooldown_s=0.0, save=False, review_below=0.50)
    ev_items_1 = [
        EvidenceItem(track_id=1, label="person", confidence=0.45, bbox=(10, 10, 50, 50), thumbnail_jpeg=None),
        EvidenceItem(track_id=2, label="person", confidence=0.75, bbox=(60, 60, 90, 90), thumbnail_jpeg=None),
    ]
    alert1 = mgr_review.create(report, evidence=ev_items_1)
    assert alert1 is not None
    assert alert1.needs_review is True
    assert alert1.min_confidence == 0.45
    assert alert1.mean_confidence == pytest.approx(0.60, abs=0.001)
    summary1 = alert1.summary()
    assert "REVIEW" in summary1
    assert "Conf: 0.45-0.75" in summary1

    # 2. All confidences >= review_below: needs_review is False
    mgr_no_review = AlertManager(cooldown_s=0.0, save=False, review_below=0.50)
    ev_items_2 = [
        EvidenceItem(track_id=1, label="person", confidence=0.60, bbox=(10, 10, 50, 50), thumbnail_jpeg=None),
        EvidenceItem(track_id=2, label="person", confidence=0.80, bbox=(60, 60, 90, 90), thumbnail_jpeg=None),
    ]
    alert2 = mgr_no_review.create(report, evidence=ev_items_2)
    assert alert2 is not None
    assert alert2.needs_review is False
    assert alert2.min_confidence == 0.60
    assert alert2.mean_confidence == pytest.approx(0.70, abs=0.001)
    summary2 = alert2.summary()
    assert "REVIEW" not in summary2
    assert "Conf: 0.60-0.80" in summary2


def test_to_payload_size_limit_and_thumbnails_dropped() -> None:
    """to_payload drops thumbnails (most confident first) to stay under max_bytes, keeping boxes and confidences."""
    dummy_jpeg = make_thumbnail(np.zeros((100, 100, 3), dtype=np.uint8), (10, 10, 50, 50), out_size=128)
    b64_str = base64.b64encode(dummy_jpeg).decode("ascii")

    evidence = [
        {"track_id": 1, "label": "person", "confidence": 0.35, "bbox": [10, 10, 50, 50], "thumbnail_b64": b64_str},
        {"track_id": 2, "label": "person", "confidence": 0.85, "bbox": [20, 20, 60, 60], "thumbnail_b64": b64_str},
        {"track_id": 3, "label": "person", "confidence": 0.65, "bbox": [30, 30, 70, 70], "thumbnail_b64": b64_str},
    ]

    alert = EmergencyAlert(
        alert_id="ALERT-TEST01",
        drone_id="D1",
        timestamp=100.0,
        priority="HIGH",
        persons=3,
        hazards={},
        location=(15.0, 25.0),
        evidence=evidence,
        min_confidence=0.35,
        mean_confidence=0.617,
        needs_review=True,
    )

    # 1. Under default generous limit (24000 bytes)
    payload_full = alert.to_payload(max_bytes=24000)
    assert payload_full["thumbnails_dropped"] == 0
    assert all(e["thumbnail_b64"] is not None for e in payload_full["evidence"])

    # 2. Strict limit (e.g. 1500 bytes) requires dropping thumbnails
    payload_restricted = alert.to_payload(max_bytes=1500)
    encoded_size = len(json.dumps(payload_restricted).encode("utf-8"))
    assert encoded_size <= 1500
    assert payload_restricted["thumbnails_dropped"] > 0

    # Confidences and bboxes MUST be retained
    for e in payload_restricted["evidence"]:
        assert e["confidence"] in (0.35, 0.65, 0.85)
        assert len(e["bbox"]) == 4

    # The most confident item (0.85) should be dropped first
    item_85 = next(e for e in payload_restricted["evidence"] if e["confidence"] == 0.85)
    assert item_85["thumbnail_b64"] is None


def test_backward_compatibility() -> None:
    """An alert created without evidence still works and serializes properly."""
    alert = EmergencyAlert(
        alert_id="ALERT-OLD",
        drone_id="D5",
        timestamp=50.0,
        priority="CRITICAL",
        persons=1,
        hazards={"fire": 1},
        location=(5.0, 10.0),
    )

    assert alert.evidence == []
    assert alert.min_confidence is None
    assert alert.mean_confidence is None
    assert alert.needs_review is False

    summary = alert.summary()
    assert summary == "ALERT | Drone: D5 | People: 1 | Location: X=5, Y=10 | Priority: CRITICAL"

    payload = alert.to_payload()
    assert payload["alert_id"] == "ALERT-OLD"
    assert payload["thumbnails_dropped"] == 0


def test_onboard_node_integration() -> None:
    """Onboard vision node generates evidence with thumbnails or confidences-only."""
    person_det = Detection(label="person", confidence=0.45, bbox=(100, 100, 200, 300))
    script = [[person_det] for _ in range(5)]

    # 1. With thumbnails enabled (default)
    camera = FakeCamera(num_frames=5)
    detector = ScriptedDetector(script)
    alerts: list[EmergencyAlert] = []

    node = OnboardVisionNode(
        detector=detector,
        camera=camera,
        on_alert=alerts.append,
        alert_manager=AlertManager(cooldown_s=0.0, save=False),
        include_thumbnails=True,
        review_below=0.50,
    )
    node.run()

    assert len(alerts) == 1
    alert = alerts[0]
    assert len(alert.evidence) == 1
    assert alert.evidence[0]["confidence"] == 0.45
    assert alert.evidence[0]["thumbnail_b64"] is not None
    # Validate thumbnail is valid base64 and decodes to image
    raw_img = decode_thumbnail(base64.b64decode(alert.evidence[0]["thumbnail_b64"]))
    assert raw_img.shape == (128, 128, 3)
    assert alert.needs_review is True

    # 2. With include_thumbnails=False
    camera2 = FakeCamera(num_frames=5)
    detector2 = ScriptedDetector(script)
    alerts2: list[EmergencyAlert] = []

    node2 = OnboardVisionNode(
        detector=detector2,
        camera=camera2,
        on_alert=alerts2.append,
        alert_manager=AlertManager(cooldown_s=0.0, save=False),
        include_thumbnails=False,
        review_below=0.50,
    )
    node2.run()

    assert len(alerts2) == 1
    alert2 = alerts2[0]
    assert len(alert2.evidence) == 1
    assert alert2.evidence[0]["confidence"] == 0.45
    assert alert2.evidence[0]["thumbnail_b64"] is None
    assert alert2.needs_review is True


def test_alert_manager_persistence_outputs(tmp_path: Path) -> None:
    """AlertManager with save=True writes JSON and alert_<id>/p<k>.jpg into specified dir."""
    report = DetectionReport(
        drone_id="D1",
        timestamp=100.0,
        persons=1,
        vehicles=0,
        hazards={},
        priority="HIGH",
        people_positions=[(10.0, 20.0)],
    )

    dummy_thumb = make_thumbnail(np.zeros((100, 100, 3), dtype=np.uint8), (10, 10, 50, 50))
    evidence = [
        EvidenceItem(track_id=1, label="person", confidence=0.72, bbox=(10, 10, 50, 50), thumbnail_jpeg=dummy_thumb)
    ]

    manager = AlertManager(cooldown_s=0.0, save=True, output_dir=str(tmp_path))
    alert = manager.create(report, evidence=evidence)
    assert alert is not None

    # Check JSON saved and valid
    json_path = tmp_path / f"{alert.alert_id}.json"
    assert json_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["alert_id"] == alert.alert_id
    assert len(loaded["evidence"]) == 1

    # Check thumbnail saved as alert_<id>/p0.jpg
    possible_dirs = [
        tmp_path / f"alert_{alert.alert_id}",
        tmp_path / f"alert_{alert.alert_id.replace('ALERT-', '')}",
    ]
    found_thumb = any((d / "p0.jpg").exists() for d in possible_dirs)
    assert found_thumb


def test_secure_channel_round_trip() -> None:
    """Sends alert.to_payload() through SecureMessenger if security package is available."""
    try:
        from security.encryption import KeyManager, PkiRegistry, SecureMessenger
    except (ImportError, ModuleNotFoundError):
        pytest.skip("security.encryption package not available in this environment")

    pki = PkiRegistry()
    km_drone = KeyManager("drone-1", pki)
    km_station = KeyManager("ground-station", pki)

    messenger_drone = SecureMessenger(km_drone)
    messenger_station = SecureMessenger(km_station)

    dummy_thumb = make_thumbnail(np.zeros((100, 100, 3), dtype=np.uint8), (10, 10, 50, 50))
    b64_str = base64.b64encode(dummy_thumb).decode("ascii")

    alert = EmergencyAlert(
        alert_id="ALERT-SECURE",
        drone_id="D1",
        timestamp=100.0,
        priority="HIGH",
        persons=1,
        hazards={},
        location=(0.0, 0.0),
        evidence=[
            {"track_id": 1, "label": "person", "confidence": 0.88, "bbox": [10, 10, 50, 50], "thumbnail_b64": b64_str}
        ],
        min_confidence=0.88,
        mean_confidence=0.88,
        needs_review=False,
    )

    payload = alert.to_payload()
    encrypted = messenger_drone.send("ground-station", payload)
    decrypted = messenger_station.receive(encrypted)

    assert decrypted["alert_id"] == "ALERT-SECURE"
    assert decrypted["evidence"][0]["confidence"] == 0.88
    thumb_b64 = decrypted["evidence"][0]["thumbnail_b64"]
    assert thumb_b64 is not None

    decoded_img = decode_thumbnail(base64.b64decode(thumb_b64))
    assert decoded_img.shape == (128, 128, 3)


def test_run_onboard_cli_args_evidence() -> None:
    """Verifies run_onboard CLI accepts evidence arguments."""
    import sys
    from run_onboard import parse_args

    orig_argv = sys.argv
    try:
        sys.argv = [
            "run_onboard.py",
            "--source", "0",
            "--no-thumbnails",
            "--max-thumbs", "8",
            "--review-below", "0.65",
        ]
        args = parse_args()
        assert args.no_thumbnails is True
        assert args.max_thumbs == 8
        assert args.review_below == 0.65
    finally:
        sys.argv = orig_argv

