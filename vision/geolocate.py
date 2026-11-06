"""Real-world 3D camera ray projection, geodesy, and pose tracking."""

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any

# WGS-84 Ellipsoid constants
WGS84_A = 6378137.0  # Semi-major axis in metres
WGS84_F = 1.0 / 298.257223563  # Flattening
WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2  # First eccentricity squared (~0.00669437999014)


def _wgs84_radii(lat_deg: float) -> tuple[float, float]:
    """Computes meridional (M) and prime vertical (N) radii of curvature at given latitude."""
    lat_rad = math.radians(lat_deg)
    sin_lat = math.sin(lat_rad)
    denom = math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    n_prime = WGS84_A / denom
    m_merid = (WGS84_A * (1.0 - WGS84_E2)) / (denom**3)
    return (m_merid, n_prime)


def latlon_to_offset(
    origin_lat: float, origin_lon: float, lat: float, lon: float
) -> tuple[float, float]:
    """Converts WGS84 (lat, lon) to local tangent plane offsets (north_m, east_m) from origin.

    Returns:
        tuple[float, float]: (north_m, east_m) in metres.
        Note: east_m is the second value.
    """
    m_merid, n_prime = _wgs84_radii(origin_lat)
    lat_rad0 = math.radians(origin_lat)
    d_lat_rad = math.radians(lat - origin_lat)
    d_lon_rad = math.radians(lon - origin_lon)

    north_m = d_lat_rad * m_merid
    east_m = d_lon_rad * n_prime * math.cos(lat_rad0)
    return (north_m, east_m)


def offset_to_latlon(
    origin_lat: float, origin_lon: float, north_m: float, east_m: float
) -> tuple[float, float]:
    """Converts local tangent plane offsets (north_m, east_m) to WGS84 (lat, lon).

    Returns:
        tuple[float, float]: (lat, lon) in decimal degrees.
    """
    m_merid, n_prime = _wgs84_radii(origin_lat)
    lat_rad0 = math.radians(origin_lat)

    d_lat_rad = north_m / m_merid
    d_lon_rad = east_m / (n_prime * math.cos(lat_rad0))

    lat = origin_lat + math.degrees(d_lat_rad)
    lon = origin_lon + math.degrees(d_lon_rad)
    return (lat, lon)


@dataclass
class DronePose:
    """Complete 3D spatial and attitude state of the drone and gimbal."""

    lat: float
    lon: float
    alt_agl_m: float
    alt_msl_m: float = 0.0
    yaw_deg: float = 0.0  # Heading: 0 = North, 90 = East, 180 = South, 270 = West
    pitch_deg: float = 0.0  # Nose up > 0
    roll_deg: float = 0.0  # Right wing down > 0
    gimbal_pitch_deg: float = -90.0  # -90 = straight down (nadir), 0 = forward horizontal
    timestamp: float = 0.0  # Monotonic time seconds
    gps_fix_type: int = 3  # >= 3 for valid 3D GPS fix
    local_east_m: float | None = None
    local_north_m: float | None = None


@dataclass
class CameraModel:
    """Pinhole camera model with horizontal FOV and square pixels."""

    width: int
    height: int
    hfov_deg: float = 84.0
    fx: float = field(init=False)
    fy: float = field(init=False)
    cx: float = field(init=False)
    cy: float = field(init=False)

    def __post_init__(self) -> None:
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        hfov_rad = math.radians(self.hfov_deg)
        self.fx = (self.width / 2.0) / math.tan(hfov_rad / 2.0)
        self.fy = self.fx


@dataclass
class ErrorModel:
    """Estimates geolocation horizontal uncertainty (1-sigma) in metres."""

    gps_sigma_m: float = 3.0
    alt_sigma_rel: float = 0.05
    attitude_sigma_deg: float = 1.5
    pixel_sigma_px: float = 4.0

    def compute_uncertainty(
        self,
        slant_range_m: float,
        alt_agl_m: float,
        depression_rad: float,
    ) -> float:
        """Calculates combined root-sum-square geolocation uncertainty radius in metres."""
        sin_dep = max(0.01, math.sin(depression_rad))
        angular_rad = math.radians(self.attitude_sigma_deg)
        angular_err = (slant_range_m * angular_rad) / sin_dep
        tan_dep = max(0.01, math.tan(depression_rad))
        alt_err = (alt_agl_m * self.alt_sigma_rel) / tan_dep
        total = math.sqrt(self.gps_sigma_m**2 + angular_err**2 + alt_err**2)
        return round(float(total), 1)


@dataclass
class GeolocationResult:
    """Output of per-person pixel geolocation."""

    lat: float | None
    lon: float | None
    north_m: float | None
    east_m: float | None
    uncertainty_m: float | None
    status: str = "ok"  # 'ok', 'no_pose', 'stale_pose', 'no_gps_fix', 'grazing_ray'


def geolocate_pixel(
    pixel: tuple[float, float],
    camera_model: CameraModel,
    drone_pose: DronePose | None,
    error_model: ErrorModel | None = None,
    min_depression_deg: float = 5.0,
    max_pose_age_s: float = 2.0,
    current_time: float | None = None,
) -> GeolocationResult:
    """Projects a 2D camera pixel coordinate onto flat-ground WGS84 and local offsets.

    IMPORTANT: This function rotates the camera ray by the gimbal pitch, drone roll,
    drone pitch, and drone heading (yaw) into true North-East-Down (NED).
    It returns north_m and east_m measured from the point directly below the drone
    in TRUE North and East directions.
    DO NOT rotate the returned offset by heading a second time!
    Compose position as: person = drone position + (north_m, east_m).
    """
    if drone_pose is None:
        return GeolocationResult(
            lat=None, lon=None, north_m=None, east_m=None, uncertainty_m=None, status="no_pose"
        )

    # Check GPS fix
    if drone_pose.gps_fix_type < 3:
        return GeolocationResult(
            lat=None, lon=None, north_m=None, east_m=None, uncertainty_m=None, status="no_gps_fix"
        )

    # Check pose freshness if current_time provided
    if current_time is not None and drone_pose.timestamp > 0.0:
        if abs(current_time - drone_pose.timestamp) > max_pose_age_s:
            return GeolocationResult(
                lat=None, lon=None, north_m=None, east_m=None, uncertainty_m=None, status="stale_pose"
            )

    alt_agl = drone_pose.alt_agl_m
    if alt_agl <= 0.1:
        return GeolocationResult(
            lat=None, lon=None, north_m=None, east_m=None, uncertainty_m=None, status="no_pose"
        )

    u, v = pixel
    # Pinhole ray in camera coordinates: xc right, yc down, zc forward
    xc = (u - camera_model.cx) / camera_model.fx
    yc = (v - camera_model.cy) / camera_model.fy
    zc = 1.0

    # 1. Rotate by gimbal pitch (gamma_deg) into drone body frame:
    # gamma = -90 (nadir): optical axis zc points straight down (+Zb),
    # top of image (-yc) points forward (+Xb), right of image (xc) points right (+Yb).
    # gamma = 0 (horizontal forward): zc points forward (+Xb), yc points down (+Zb), xc points right (+Yb).
    gamma_rad = math.radians(drone_pose.gimbal_pitch_deg)
    sin_g = math.sin(gamma_rad)
    cos_g = math.cos(gamma_rad)

    # Body frame coordinates (Xb: Forward, Yb: Right, Zb: Down)
    xb = sin_g * yc + cos_g * zc
    yb = xc
    zb = cos_g * yc - sin_g * zc

    # 2. Rotate by drone attitude (Roll phi, Pitch theta, Yaw psi) into NED Earth frame:
    # Sequence: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    phi = math.radians(drone_pose.roll_deg)
    theta = math.radians(drone_pose.pitch_deg)
    psi = math.radians(drone_pose.yaw_deg)

    c_phi, s_phi = math.cos(phi), math.sin(phi)
    c_th, s_th = math.cos(theta), math.sin(theta)
    c_psi, s_psi = math.cos(psi), math.sin(psi)

    # Rx(roll)
    x1 = xb
    y1 = c_phi * yb - s_phi * zb
    z1 = s_phi * yb + c_phi * zb

    # Ry(pitch: nose up > 0 rotates +X towards -Z)
    x2 = c_th * x1 + s_th * z1
    y2 = y1
    z2 = -s_th * x1 + c_th * z1

    # Rz(yaw: clockwise from North)
    vn = c_psi * x2 - s_psi * y2
    ve = s_psi * x2 + c_psi * y2
    vd = z2

    norm = math.sqrt(vn * vn + ve * ve + vd * vd)
    if norm <= 1e-9:
        return GeolocationResult(
            lat=None, lon=None, north_m=None, east_m=None, uncertainty_m=None, status="grazing_ray"
        )
    vn /= norm
    ve /= norm
    vd /= norm

    # Check for grazing ray or ray pointing above horizontal
    sin_min_dep = math.sin(math.radians(min_depression_deg))
    if vd < sin_min_dep:
        return GeolocationResult(
            lat=None, lon=None, north_m=None, east_m=None, uncertainty_m=None, status="grazing_ray"
        )

    # Flat ground intersection distance along ray
    slant_range = alt_agl / vd
    north_m = slant_range * vn
    east_m = slant_range * ve

    # Convert offset to WGS84 lat/lon
    lat, lon = offset_to_latlon(drone_pose.lat, drone_pose.lon, north_m, east_m)

    # Uncertainty estimation
    dep_rad = math.asin(min(1.0, max(0.0, vd)))
    if error_model is not None:
        uncertainty = error_model.compute_uncertainty(slant_range, alt_agl, dep_rad)
    else:
        # Default baseline error model
        default_em = ErrorModel()
        uncertainty = default_em.compute_uncertainty(slant_range, alt_agl, dep_rad)

    return GeolocationResult(
        lat=round(lat, 7),
        lon=round(lon, 7),
        north_m=round(north_m, 2),
        east_m=round(east_m, 2),
        uncertainty_m=uncertainty,
        status="ok",
    )


class PoseBuffer:
    """Thread-safe circular ring buffer for timestamped DronePose objects."""

    def __init__(self, max_size: int = 2000, max_age_s: float = 5.0) -> None:
        self.max_size = int(max_size)
        self.max_age_s = float(max_age_s)
        self._poses: list[DronePose] = []
        self._lock = threading.Lock()

    def add(self, pose: DronePose, timestamp: float | None = None) -> None:
        """Inserts a new pose stamped with monotonic timestamp."""
        if timestamp is not None:
            pose.timestamp = float(timestamp)
        elif pose.timestamp == 0.0:
            pose.timestamp = time.monotonic()

        with self._lock:
            self._poses.append(pose)
            if len(self._poses) > self.max_size:
                self._poses.pop(0)

    def pose_at(self, query_time: float, max_tolerance_s: float | None = None) -> DronePose | None:
        """Finds closest pose to query_time within tolerance, or None if buffer empty or stale."""
        tol = max_tolerance_s if max_tolerance_s is not None else self.max_age_s
        with self._lock:
            if not self._poses:
                return None
            closest = min(self._poses, key=lambda p: abs(p.timestamp - query_time))
            if abs(closest.timestamp - query_time) > tol:
                return None
            return closest

    def latest(self) -> DronePose | None:
        """Returns the most recently recorded pose."""
        with self._lock:
            return self._poses[-1] if self._poses else None

    def clear(self) -> None:
        """Clears all stored poses."""
        with self._lock:
            self._poses.clear()
