"""Asynchronous MAVLink telemetry consumer with thread-safe pose buffering."""

import math
import threading
import time
from typing import Any

from vision.geolocate import DronePose, PoseBuffer


class MavlinkTelemetry:
    """Consumes MAVLink telemetry messages and provides latency-compensated DronePose queries.

    pymavlink is imported lazily on start(); instances can run against mock connection
    objects without pymavlink installed.
    """

    def __init__(
        self,
        connection: Any,
        camera_latency_s: float = 0.2,
        gimbal_pitch_deg: float = -90.0,
        alt_source: str = "relative",
        ground_elevation_msl_m: float | None = None,
        max_pose_age_s: float = 2.0,
    ) -> None:
        self.connection = connection
        self.camera_latency_s = float(camera_latency_s)
        self.gimbal_pitch_deg = float(gimbal_pitch_deg)
        self.alt_source = str(alt_source).lower()
        self.ground_elevation_msl_m = (
            float(ground_elevation_msl_m) if ground_elevation_msl_m is not None else None
        )
        self.max_pose_age_s = float(max_pose_age_s)

        if self.alt_source == "msl_minus_ground":
            if self.ground_elevation_msl_m is None:
                raise ValueError(
                    "ground_elevation_msl_m must be specified when alt_source is 'msl_minus_ground'"
                )
        elif self.alt_source != "relative":
            raise ValueError(
                f"Unsupported alt_source '{self.alt_source}'. Must be 'relative' or 'msl_minus_ground'."
            )

        self.pose_buffer = PoseBuffer(max_size=2000, max_age_s=self.max_pose_age_s)
        self._mav = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Telemetry state cache
        self._lat: float | None = None
        self._lon: float | None = None
        self._alt_msl_m: float | None = None
        self._relative_alt_m: float | None = None
        self._hdg_deg: float | None = None
        self._yaw_deg: float = 0.0
        self._pitch_deg: float = 0.0
        self._roll_deg: float = 0.0
        self._current_gimbal_pitch: float = self.gimbal_pitch_deg
        self._current_fix_type: int = 0
        self._last_msg_time: float | None = None

    @property
    def latest_pose_age_s(self) -> float:
        """Returns age in seconds of latest valid pose, or infinity if none."""
        latest = self.pose_buffer.latest()
        if latest is None or latest.timestamp <= 0.0:
            return float("inf")
        return max(0.0, time.monotonic() - latest.timestamp)

    @property
    def gps_fix_type(self) -> int:
        """Returns current GPS fix type (0=None, 2=2D, 3=3D, 4=DGPS, 5=RTK)."""
        with self._lock:
            return self._current_fix_type

    def _update_pose(self) -> None:
        """Assembles and stores current state into pose buffer if valid."""
        with self._lock:
            if self._current_fix_type < 3:
                return
            if self._lat is None or self._lon is None:
                return

            if self.alt_source == "relative":
                if self._relative_alt_m is None:
                    return
                alt_agl = self._relative_alt_m
            else:
                if self._alt_msl_m is None or self.ground_elevation_msl_m is None:
                    return
                alt_agl = self._alt_msl_m - self.ground_elevation_msl_m

            heading = self._hdg_deg if self._hdg_deg is not None else self._yaw_deg

            pose = DronePose(
                lat=self._lat,
                lon=self._lon,
                alt_agl_m=alt_agl,
                alt_msl_m=self._alt_msl_m if self._alt_msl_m is not None else alt_agl,
                yaw_deg=heading,
                pitch_deg=self._pitch_deg,
                roll_deg=self._roll_deg,
                gimbal_pitch_deg=self._current_gimbal_pitch,
                timestamp=time.monotonic(),
                gps_fix_type=self._current_fix_type,
            )

        self.pose_buffer.add(pose)

    def handle_message(self, msg: Any) -> None:
        """Processes a single incoming MAVLink message (callable directly in tests)."""
        if msg is None:
            return

        msg_type = getattr(msg, "get_type", lambda: "")()
        if not msg_type and hasattr(msg, "type"):
            msg_type = getattr(msg, "type")

        self._last_msg_time = time.monotonic()

        if msg_type == "GLOBAL_POSITION_INT":
            # lat/lon in 1e7 degrees
            self._lat = float(msg.lat) / 1e7
            self._lon = float(msg.lon) / 1e7
            # alt and relative_alt in millimetres -> metres
            self._alt_msl_m = float(msg.alt) / 1000.0
            self._relative_alt_m = float(msg.relative_alt) / 1000.0
            # hdg in centidegrees (65535 indicates unknown heading)
            hdg_raw = getattr(msg, "hdg", 65535)
            if hdg_raw == 65535 or hdg_raw < 0:
                self._hdg_deg = None
            else:
                self._hdg_deg = float(hdg_raw) / 100.0
            self._update_pose()

        elif msg_type == "ATTITUDE":
            # Angles in radians -> degrees
            self._roll_deg = math.degrees(float(msg.roll))
            self._pitch_deg = math.degrees(float(msg.pitch))
            yaw_deg = math.degrees(float(msg.yaw))
            self._yaw_deg = (yaw_deg + 360.0) % 360.0
            self._update_pose()

        elif msg_type == "GPS_RAW_INT":
            self._current_fix_type = int(msg.fix_type)
            self._update_pose()

        elif msg_type == "MOUNT_ORIENTATION":
            if hasattr(msg, "pitch"):
                self._current_gimbal_pitch = float(msg.pitch)
            self._update_pose()

        elif msg_type == "GIMBAL_DEVICE_ATTITUDE_STATUS":
            if hasattr(msg, "pitch"):
                self._current_gimbal_pitch = math.degrees(float(msg.pitch))
            elif hasattr(msg, "q"):
                # Quaternion to pitch conversion if available
                q = msg.q
                sinp = 2.0 * (q[0] * q[2] - q[3] * q[1])
                pitch_rad = math.asin(max(-1.0, min(1.0, sinp)))
                self._current_gimbal_pitch = math.degrees(pitch_rad)
            self._update_pose()

    def pose_at(self, frame_monotonic_time: float) -> DronePose | None:
        """Retrieves latency-compensated drone pose for a video frame capture time."""
        target_time = frame_monotonic_time - self.camera_latency_s
        pose = self.pose_buffer.pose_at(target_time, max_tolerance_s=self.max_pose_age_s)
        if pose is None:
            return None
        if pose.gps_fix_type < 3:
            return None
        # Verify age
        if (time.monotonic() - pose.timestamp) > self.max_pose_age_s:
            return None
        return pose

    def _worker(self) -> None:
        """Background thread worker pulling MAVLink packets."""
        while not self._stop_event.is_set():
            try:
                if self._mav is None:
                    time.sleep(0.01)
                    continue
                msg = self._mav.recv_match(blocking=True, timeout=0.1)
                if msg is not None:
                    self.handle_message(msg)
            except Exception:
                time.sleep(0.01)

    def start(self) -> "MavlinkTelemetry":
        """Starts background MAVLink message reading thread."""
        if self._thread is not None and self._thread.is_alive():
            return self

        if self._mav is None:
            if isinstance(self.connection, str):
                try:
                    from pymavlink import mavutil  # Lazy import
                    self._mav = mavutil.mavlink_connection(self.connection)
                except ImportError as err:
                    raise ImportError(
                        f"pymavlink is required for MAVLink connection string '{self.connection}': {err}"
                    )
            else:
                self._mav = self.connection

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stops background reading thread and closes connection."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._mav is not None and hasattr(self._mav, "close"):
            try:
                self._mav.close()
            except Exception:
                pass
            self._mav = None
