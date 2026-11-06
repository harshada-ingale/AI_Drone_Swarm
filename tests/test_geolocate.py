"""Unit tests for real-world per-person geolocation, pose providers, and MAVLink telemetry."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any
import numpy as np
import pytest

from vision.disaster_detection import DetectionReport
from vision.emergency_alert import AlertManager, EmergencyAlert
from vision.geolocate import (
    CameraModel,
    DronePose,
    ErrorModel,
    GeolocationResult,
    PoseBuffer,
    geolocate_pixel,
    latlon_to_offset,
    offset_to_latlon,
)
from vision.mavlink_telemetry import MavlinkTelemetry
from vision.onboard_vision import OnboardVisionNode
from vision.pose_providers import SimPoseProvider, StaticPoseProvider
from vision.tracker import SimpleTracker, Track
from vision.yolo_detector import Detection


class FakeCamera:
    """Mock camera producing synthetic video frames."""

    def __init__(self, num_frames: int = 5, width: int = 640, height: int = 480) -> None:
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
        return (np.zeros((self.height, self.width, 3), dtype=np.uint8), self.current)

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


# --- MAVLink Mock Message Classes ---
@dataclass
class MockGlobalPositionInt:
    lat: int  # 1e7 degrees
    lon: int  # 1e7 degrees
    alt: int  # mm MSL
    relative_alt: int  # mm AGL
    hdg: int  # centidegrees

    def get_type(self) -> str:
        return "GLOBAL_POSITION_INT"


@dataclass
class MockAttitude:
    roll: float  # radians
    pitch: float  # radians
    yaw: float  # radians

    def get_type(self) -> str:
        return "ATTITUDE"


@dataclass
class MockGpsRawInt:
    fix_type: int

    def get_type(self) -> str:
        return "GPS_RAW_INT"


@dataclass
class MockMountOrientation:
    pitch: float

    def get_type(self) -> str:
        return "MOUNT_ORIENTATION"


class FakeMavlinkConnection:
    """Mock MAVLink connection object delivering scripted messages without network/pymavlink."""

    def __init__(self, messages: list[Any]) -> None:
        self.messages = list(messages)
        self.index = 0

    def recv_match(self, blocking: bool = True, timeout: float | None = None) -> Any:
        if self.index < len(self.messages):
            msg = self.messages[self.index]
            self.index += 1
            return msg
        return None

    def close(self) -> None:
        pass


def test_mavlink_telemetry_scripted_messages() -> None:
    """FakeMavlinkConnection validates unit conversions, hdg=65535, fix types, and altitude sources."""
    # 1. Normal valid messages
    conn = FakeMavlinkConnection(
        [
            MockGpsRawInt(fix_type=3),
            MockAttitude(
                roll=math.radians(5.0),
                pitch=math.radians(-3.0),
                yaw=math.radians(90.0),
            ),
            MockGlobalPositionInt(
                lat=167051200,  # 16.70512 deg
                lon=742434100,  # 74.24341 deg
                alt=45000,  # 45.0 m MSL
                relative_alt=30000,  # 30.0 m AGL
                hdg=9000,  # 90.0 deg
            ),
        ]
    )

    telem = MavlinkTelemetry(connection=conn, camera_latency_s=0.0)
    # Process messages directly
    for _ in range(3):
        telem.handle_message(conn.recv_match())

    now = time.monotonic()
    pose = telem.pose_at(now)
    assert pose is not None
    assert pose.lat == pytest.approx(16.70512, abs=1e-5)
    assert pose.lon == pytest.approx(74.24341, abs=1e-5)
    assert pose.alt_agl_m == pytest.approx(30.0, abs=1e-2)
    assert pose.alt_msl_m == pytest.approx(45.0, abs=1e-2)
    assert pose.yaw_deg == pytest.approx(90.0, abs=1e-2)
    assert pose.roll_deg == pytest.approx(5.0, abs=1e-2)
    assert pose.pitch_deg == pytest.approx(-3.0, abs=1e-2)

    # 2. hdg = 65535 fallback to attitude yaw
    conn_hdg = FakeMavlinkConnection(
        [
            MockGpsRawInt(fix_type=3),
            MockAttitude(roll=0.0, pitch=0.0, yaw=math.radians(45.0)),
            MockGlobalPositionInt(
                lat=167051200, lon=742434100, alt=45000, relative_alt=30000, hdg=65535
            ),
        ]
    )
    telem_hdg = MavlinkTelemetry(connection=conn_hdg, camera_latency_s=0.0)
    for _ in range(3):
        telem_hdg.handle_message(conn_hdg.recv_match())

    pose_hdg = telem_hdg.pose_at(time.monotonic())
    assert pose_hdg is not None
    assert pose_hdg.yaw_deg == pytest.approx(45.0, abs=1e-2)

    # 3. fix_type < 3 gives no pose
    conn_bad_fix = FakeMavlinkConnection(
        [
            MockGpsRawInt(fix_type=2),  # 2D fix only
            MockAttitude(roll=0.0, pitch=0.0, yaw=0.0),
            MockGlobalPositionInt(
                lat=167051200, lon=742434100, alt=45000, relative_alt=30000, hdg=0
            ),
        ]
    )
    telem_bad_fix = MavlinkTelemetry(connection=conn_bad_fix, camera_latency_s=0.0)
    for _ in range(3):
        telem_bad_fix.handle_message(conn_bad_fix.recv_match())

    assert telem_bad_fix.pose_at(time.monotonic()) is None

    # 4. alt_source="msl_minus_ground" works and raises error without ground elevation
    conn_msl = FakeMavlinkConnection(
        [
            MockGpsRawInt(fix_type=3),
            MockAttitude(roll=0.0, pitch=0.0, yaw=0.0),
            MockGlobalPositionInt(
                lat=167051200, lon=742434100, alt=45000, relative_alt=30000, hdg=0
            ),
        ]
    )
    telem_msl = MavlinkTelemetry(
        connection=conn_msl,
        camera_latency_s=0.0,
        alt_source="msl_minus_ground",
        ground_elevation_msl_m=15.0,
    )
    for _ in range(3):
        telem_msl.handle_message(conn_msl.recv_match())

    pose_msl = telem_msl.pose_at(time.monotonic())
    assert pose_msl is not None
    # 45.0m MSL - 15.0m ground elevation = 30.0m AGL
    assert pose_msl.alt_agl_m == pytest.approx(30.0, abs=1e-2)

    with pytest.raises(ValueError, match="ground_elevation_msl_m"):
        MavlinkTelemetry(
            connection=conn_msl,
            alt_source="msl_minus_ground",
            ground_elevation_msl_m=None,
        )


def test_heading_applied_exactly_once() -> None:
    """Verifies heading rotation is applied once in geolocate_pixel and matches reference math."""
    origin_lat = 16.0
    origin_lon = 74.0
    cam = CameraModel(width=640, height=480, hfov_deg=90.0)

    # 1. Heading 90 deg (East), 160 px above center (Forward) -> local (115, 50)
    sim90 = SimPoseProvider(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        east_m=100.0,
        north_m=50.0,
        alt_agl_m=30.0,
        yaw_deg=90.0,
        gimbal_pitch_deg=-90.0,
    )
    pose90 = sim90.pose_at()
    res90 = geolocate_pixel((320.0, 80.0), cam, pose90)
    assert res90.status == "ok"
    dn90, de90 = latlon_to_offset(origin_lat, origin_lon, res90.lat, res90.lon)
    assert round(de90, 2) == 115.00
    assert round(dn90, 2) == 50.00

    # 2. Heading 0 deg (North), 160 px above center (Forward) -> local (100, 65)
    sim0 = SimPoseProvider(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        east_m=100.0,
        north_m=50.0,
        alt_agl_m=30.0,
        yaw_deg=0.0,
        gimbal_pitch_deg=-90.0,
    )
    pose0 = sim0.pose_at()
    res0 = geolocate_pixel((320.0, 80.0), cam, pose0)
    assert res0.status == "ok"
    dn0, de0 = latlon_to_offset(origin_lat, origin_lon, res0.lat, res0.lon)
    assert round(de0, 2) == 100.00
    assert round(dn0, 2) == 65.00

    # 3. Heading 180 deg (South), 160 px above center (Forward) -> local (100, 35)
    sim180 = SimPoseProvider(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        east_m=100.0,
        north_m=50.0,
        alt_agl_m=30.0,
        yaw_deg=180.0,
        gimbal_pitch_deg=-90.0,
    )
    pose180 = sim180.pose_at()
    res180 = geolocate_pixel((320.0, 80.0), cam, pose180)
    assert res180.status == "ok"
    dn180, de180 = latlon_to_offset(origin_lat, origin_lon, res180.lat, res180.lon)
    assert round(de180, 2) == 100.00
    assert round(dn180, 2) == 35.00

    # 4. Heading 0 deg, 160 px RIGHT of center -> local (115, 50)
    res_right = geolocate_pixel((480.0, 240.0), cam, pose0)
    assert res_right.status == "ok"
    dn_r, de_r = latlon_to_offset(origin_lat, origin_lon, res_right.lat, res_right.lon)
    assert round(de_r, 2) == 115.00
    assert round(dn_r, 2) == 50.00


def test_per_person_output_different_positions() -> None:
    """Three people at different pixels produce three distinct positions with matching track IDs."""
    origin_lat = 16.70512
    origin_lon = 74.24341
    pose_provider = StaticPoseProvider(
        lat=origin_lat,
        lon=origin_lon,
        alt_agl_m=30.0,
        yaw_deg=0.0,
        gimbal_pitch_deg=-90.0,
    )

    # Create 3 detections at distinct pixel coordinates
    det1 = Detection(label="person", confidence=0.88, bbox=(100, 100, 160, 200))
    det2 = Detection(label="person", confidence=0.75, bbox=(280, 200, 340, 300))
    det3 = Detection(label="person", confidence=0.62, bbox=(480, 300, 540, 400))

    script = [[det1, det2, det3] for _ in range(4)]
    detector = ScriptedDetector(script)
    camera = FakeCamera(num_frames=4)
    alerts: list[EmergencyAlert] = []

    node = OnboardVisionNode(
        detector=detector,
        camera=camera,
        pose_provider=pose_provider,
        camera_hfov=84.0,
        origin=(origin_lat, origin_lon, "home"),
        alert_manager=AlertManager(cooldown_s=0.0, save=False),
        on_alert=alerts.append,
    )
    node.run()

    assert len(alerts) >= 1
    alert = alerts[0]

    # Verify per-person outputs
    assert len(alert.location_gps) == 3
    assert len(alert.location_local) == 3

    track_ids_gps = [p["track_id"] for p in alert.location_gps]
    track_ids_loc = [p["track_id"] for p in alert.location_local]
    assert track_ids_gps == track_ids_loc

    # Verify all 3 positions are different (no averaging)
    coords_gps = [(p["lat"], p["lon"]) for p in alert.location_gps]
    assert len(set(coords_gps)) == 3

    coords_local = [(p["east_m"], p["north_m"]) for p in alert.location_local]
    assert len(set(coords_local)) == 3

    # Check alert drone block
    assert alert.drone is not None
    assert alert.drone["lat"] == origin_lat
    assert alert.drone["lon"] == origin_lon
    assert alert.drone["heading_deg"] == 0.0

    # Summary formatting
    summary = alert.summary()
    assert "P1" in summary or "P2" in summary or "P3" in summary
    assert "ALERT | Drone:" in summary


def test_gps_and_local_agree_within_1cm() -> None:
    """For every geolocated person, location_local equals latlon_to_offset(origin, location_gps) within 1 cm."""
    origin_lat = 16.70512
    origin_lon = 74.24341
    pose_provider = StaticPoseProvider(
        lat=16.70550,
        lon=74.24380,
        alt_agl_m=35.0,
        yaw_deg=45.0,
        gimbal_pitch_deg=-80.0,
    )

    det1 = Detection(label="person", confidence=0.85, bbox=(120, 140, 180, 240))
    det2 = Detection(label="person", confidence=0.78, bbox=(350, 220, 410, 320))
    detector = ScriptedDetector([[det1, det2] for _ in range(4)])
    camera = FakeCamera(num_frames=4)
    alerts: list[EmergencyAlert] = []

    node = OnboardVisionNode(
        detector=detector,
        camera=camera,
        pose_provider=pose_provider,
        origin={"lat": origin_lat, "lon": origin_lon, "name": "ground_station"},
        alert_manager=AlertManager(cooldown_s=0.0, save=False),
        on_alert=alerts.append,
    )
    node.run()

    assert len(alerts) >= 1
    alert = alerts[0]

    for p_gps, p_loc in zip(alert.location_gps, alert.location_local, strict=True):
        assert p_gps["track_id"] == p_loc["track_id"]
        lat = p_gps["lat"]
        lon = p_gps["lon"]
        north_calc, east_calc = latlon_to_offset(origin_lat, origin_lon, lat, lon)
        # Agreement within 1 cm (0.01 metres)
        assert abs(p_loc["east_m"] - east_calc) < 0.01
        assert abs(p_loc["north_m"] - north_calc) < 0.01


def test_reference_frame_metadata() -> None:
    """reference_frame contains WGS84, local ENU conventions, and origin metadata."""
    alert = EmergencyAlert(
        alert_id="ALERT-REF",
        drone_id="D1",
        timestamp=100.0,
        priority="HIGH",
        persons=1,
        reference_frame={
            "gps": "WGS84 degrees",
            "local": "ENU metres from origin (x=east, y=north)",
            "origin": {"name": "home", "lat": 16.705, "lon": 74.243},
        },
    )
    payload = alert.to_payload()
    ref = payload["reference_frame"]
    assert ref["gps"] == "WGS84 degrees"
    assert "ENU" in ref["local"]
    assert ref["origin"]["name"] == "home"
    assert ref["origin"]["lat"] == 16.705
    assert ref["origin"]["lon"] == 74.243


def test_failure_modes_null_positions() -> None:
    """No pose provider, stale pose, low GPS fix, or grazing ray sets null and records correct status."""
    cam = CameraModel(width=640, height=480, hfov_deg=84.0)

    # 1. No pose provider
    res_no_pose = geolocate_pixel((320, 240), cam, None)
    assert res_no_pose.lat is None
    assert res_no_pose.status == "no_pose"

    # 2. Stale pose
    stale_pose = DronePose(
        lat=16.0, lon=74.0, alt_agl_m=30.0, timestamp=10.0, gps_fix_type=3
    )
    res_stale = geolocate_pixel((320, 240), cam, stale_pose, current_time=25.0)
    assert res_stale.lat is None
    assert res_stale.status == "stale_pose"

    # 3. Bad GPS fix (< 3)
    bad_fix_pose = DronePose(
        lat=16.0, lon=74.0, alt_agl_m=30.0, timestamp=100.0, gps_fix_type=2
    )
    res_bad_fix = geolocate_pixel((320, 240), cam, bad_fix_pose, current_time=100.0)
    assert res_bad_fix.lat is None
    assert res_bad_fix.status == "no_gps_fix"

    # 4. Grazing ray (gimbal looking horizontal at 0 deg, ray pointing at horizon)
    grazing_pose = DronePose(
        lat=16.0,
        lon=74.0,
        alt_agl_m=30.0,
        timestamp=100.0,
        gps_fix_type=3,
        gimbal_pitch_deg=0.0,  # horizontal
    )
    # Center pixel looks horizontal -> depression angle 0 deg < 5 deg
    res_grazing = geolocate_pixel((320, 240), cam, grazing_pose, current_time=100.0)
    assert res_grazing.lat is None
    assert res_grazing.status == "grazing_ray"

    # 5. Alert raised with no pose still functions with null positions and status
    alert = EmergencyAlert(
        alert_id="ALERT-NOPOSE",
        drone_id="D1",
        timestamp=100.0,
        priority="HIGH",
        persons=1,
        location_gps=[
            {
                "track_id": 1,
                "lat": None,
                "lon": None,
                "uncertainty_m": None,
                "location_status": "no_pose",
            }
        ],
        location_local=[
            {"track_id": 1, "east_m": None, "north_m": None, "location_status": "no_pose"}
        ],
    )
    assert "Loc: unknown (no_pose)" in alert.summary()
    payload = alert.to_payload()
    assert payload["location_gps"][0]["lat"] is None


def test_frame_time_latency_lookup() -> None:
    """PoseBuffer retrieves pose closest to (frame_time - latency)."""
    buf = PoseBuffer()
    # Add moving poses across time
    for t_step in [10.0, 10.1, 10.2, 10.3, 10.4, 10.5]:
        buf.add(
            DronePose(
                lat=16.0 + (t_step - 10.0) * 0.001,
                lon=74.0,
                alt_agl_m=30.0,
                yaw_deg=(t_step - 10.0) * 100.0,
                timestamp=t_step,
                gps_fix_type=3,
            )
        )

    latency = 0.2
    frame_read_time = 10.4
    target = frame_read_time - latency  # 10.2

    pose = buf.pose_at(target)
    assert pose is not None
    assert pose.timestamp == pytest.approx(10.2, abs=1e-3)
    assert pose.yaw_deg == pytest.approx(20.0, abs=1e-2)


def test_hazards_honest_null() -> None:
    """When no hazard model is loaded, hazards is null, hazards_supported is False, and payload is serializable."""
    report = DetectionReport(
        drone_id="D1",
        timestamp=100.0,
        persons=1,
        vehicles=0,
        hazards={},
        priority="HIGH",
        people_positions=[(10.0, 20.0)],
    )

    mgr = AlertManager(cooldown_s=0.0, save=False)
    alert = mgr.create(report)
    assert alert is not None
    assert alert.hazards is None
    assert alert.hazards_supported is False

    payload = alert.to_payload()
    assert payload["hazards"] is None
    assert payload["hazards_supported"] is False

    serialized = json.dumps(payload)
    loaded = json.loads(serialized)
    assert loaded["hazards"] is None
    assert loaded["hazards_supported"] is False
