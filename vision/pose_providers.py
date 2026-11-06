"""Pose providers for bench testing, simulation, and hardware telemetry integration."""

import math
import time
from typing import Any

from vision.geolocate import DronePose, latlon_to_offset, offset_to_latlon


class StaticPoseProvider:
    """Fixed-position pose provider for bench, desk tests, and offline replay."""

    def __init__(
        self,
        lat: float = 16.70512,
        lon: float = 74.24341,
        alt_agl_m: float = 30.0,
        yaw_deg: float = 0.0,
        gimbal_pitch_deg: float = -90.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        alt_msl_m: float | None = None,
        gps_fix_type: int = 3,
        local_east_m: float | None = 0.0,
        local_north_m: float | None = 0.0,
    ) -> None:
        self.lat = float(lat)
        self.lon = float(lon)
        self.alt_agl_m = float(alt_agl_m)
        self.yaw_deg = float(yaw_deg)
        self.gimbal_pitch_deg = float(gimbal_pitch_deg)
        self.pitch_deg = float(pitch_deg)
        self.roll_deg = float(roll_deg)
        self.alt_msl_m = float(alt_msl_m) if alt_msl_m is not None else float(alt_agl_m)
        self.gps_fix_type = int(gps_fix_type)
        self.local_east_m = local_east_m
        self.local_north_m = local_north_m

    def pose_at(self, t: float | None = None) -> DronePose:
        """Returns constant DronePose stamped with query timestamp."""
        ts = float(t) if t is not None else time.monotonic()
        return DronePose(
            lat=self.lat,
            lon=self.lon,
            alt_agl_m=self.alt_agl_m,
            alt_msl_m=self.alt_msl_m,
            yaw_deg=self.yaw_deg,
            pitch_deg=self.pitch_deg,
            roll_deg=self.roll_deg,
            gimbal_pitch_deg=self.gimbal_pitch_deg,
            timestamp=ts,
            gps_fix_type=self.gps_fix_type,
            local_east_m=self.local_east_m,
            local_north_m=self.local_north_m,
        )


class SimPoseProvider:
    """Simulation pose provider maintaining local ENU coordinates and converting to WGS84."""

    def __init__(
        self,
        origin_lat: float = 16.70512,
        origin_lon: float = 74.24341,
        east_m: float = 0.0,
        north_m: float = 0.0,
        alt_agl_m: float = 30.0,
        yaw_deg: float = 0.0,
        gimbal_pitch_deg: float = -90.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        gps_fix_type: int = 3,
    ) -> None:
        self.origin_lat = float(origin_lat)
        self.origin_lon = float(origin_lon)
        self.east_m = float(east_m)
        self.north_m = float(north_m)
        self.alt_agl_m = float(alt_agl_m)
        self.yaw_deg = float(yaw_deg)
        self.gimbal_pitch_deg = float(gimbal_pitch_deg)
        self.pitch_deg = float(pitch_deg)
        self.roll_deg = float(roll_deg)
        self.gps_fix_type = int(gps_fix_type)

    def set_position(
        self,
        east_m: float,
        north_m: float,
        alt_agl_m: float | None = None,
        yaw_deg: float | None = None,
    ) -> None:
        """Updates local ENU position and optional attitude."""
        self.east_m = float(east_m)
        self.north_m = float(north_m)
        if alt_agl_m is not None:
            self.alt_agl_m = float(alt_agl_m)
        if yaw_deg is not None:
            self.yaw_deg = float(yaw_deg)

    def pose_at(self, t: float | None = None) -> DronePose:
        """Computes WGS84 lat/lon from origin + local offsets and returns DronePose."""
        ts = float(t) if t is not None else time.monotonic()
        lat, lon = offset_to_latlon(
            self.origin_lat, self.origin_lon, self.north_m, self.east_m
        )
        return DronePose(
            lat=lat,
            lon=lon,
            alt_agl_m=self.alt_agl_m,
            alt_msl_m=self.alt_agl_m,
            yaw_deg=self.yaw_deg,
            pitch_deg=self.pitch_deg,
            roll_deg=self.roll_deg,
            gimbal_pitch_deg=self.gimbal_pitch_deg,
            timestamp=ts,
            gps_fix_type=self.gps_fix_type,
            local_east_m=self.east_m,
            local_north_m=self.north_m,
        )
