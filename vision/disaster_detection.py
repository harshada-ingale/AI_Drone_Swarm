"""Disaster analysis, priority decision, and geolocation projection."""

from dataclasses import dataclass, field
import math
import time
from typing import Any
from vision.yolo_detector import Detection

PERSON_CLASSES = {"person", "pedestrian", "people"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "boat", "van", "motor"}
HAZARD_CLASSES = {"fire", "smoke", "flood", "debris", "damaged_building", "blocked_road"}


@dataclass
class DetectionReport:
    """Consolidated situational report for an aerial surveillance frame."""

    drone_id: str
    timestamp: float
    persons: int
    vehicles: int
    hazards: dict[str, int]
    priority: str
    people_positions: list[tuple[float, float]]  # (x, y) coordinates in metres
    detections: list[Detection] = field(default_factory=list)


def pixel_to_ground(
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    altitude_m: float,
    hfov_deg: float,
    drone_xy: tuple[float, float] = (0.0, 0.0),
    yaw_deg: float = 0.0,
) -> tuple[float, float]:
    """Projects image pixel center to flat-ground metres coordinates for a nadir camera.

    Formula: ground_width = 2 * altitude * tan(hfov / 2)
    """
    if img_w <= 0 or img_h <= 0 or altitude_m <= 0:
        return drone_xy

    hfov_rad = math.radians(hfov_deg)
    ground_width = 2.0 * altitude_m * math.tan(hfov_rad / 2.0)
    ground_height = ground_width * (float(img_h) / float(img_w))

    # Offset from image center in camera frame
    dx_cam = ((cx - (img_w / 2.0)) / img_w) * ground_width
    dy_cam = (((img_h / 2.0) - cy) / img_h) * ground_height

    # Rotate by yaw angle
    yaw_rad = math.radians(yaw_deg)
    dx_world = dx_cam * math.cos(yaw_rad) - dy_cam * math.sin(yaw_rad)
    dy_world = dx_cam * math.sin(yaw_rad) + dy_cam * math.cos(yaw_rad)

    return (drone_xy[0] + dx_world, drone_xy[1] + dy_world)


def decide_priority(
    persons: int,
    vehicles: int,
    hazards: dict[str, int],
    thermal_confirmed: bool = False,
) -> str:
    """Calculates threat/rescue priority based on detections and thermal status.

    Rules:
      - CRITICAL if persons >= 1 and (thermal_confirmed or fire or flood)
      - HIGH if persons >= 1
      - MEDIUM if any hazards or vehicles
      - LOW otherwise
    """
    has_fire = hazards.get("fire", 0) > 0
    has_flood = hazards.get("flood", 0) > 0

    if persons >= 1 and (thermal_confirmed or has_fire or has_flood):
        return "CRITICAL"
    if persons >= 1:
        return "HIGH"
    if sum(hazards.values()) > 0 or vehicles >= 1:
        return "MEDIUM"
    return "LOW"


class DisasterAnalyzer:
    """Analyzes raw detections and generates structured disaster reports."""

    def __init__(self, detector: Any = None, min_confidence: float = 0.35) -> None:
        self.detector = detector
        self.min_confidence = min_confidence

    def analyze_detections(
        self,
        detections: list[Detection],
        frame_hw: tuple[int, int],
        drone_id: str = "D1",
        altitude_m: float = 30.0,
        drone_xy: tuple[float, float] = (0.0, 0.0),
        yaw_deg: float = 0.0,
        hfov_deg: float = 84.0,
        thermal_confirmed: bool = False,
    ) -> DetectionReport:
        """Categorizes detections, projects coordinates, and produces a DetectionReport."""
        img_h, img_w = frame_hw
        filtered = [d for d in detections if d.confidence >= self.min_confidence]

        persons = 0
        vehicles = 0
        hazards: dict[str, int] = {h: 0 for h in HAZARD_CLASSES}
        people_positions: list[tuple[float, float]] = []

        for d in filtered:
            lbl = d.label.lower()
            if lbl in PERSON_CLASSES:
                persons += 1
                pos = pixel_to_ground(
                    cx=d.center[0],
                    cy=d.center[1],
                    img_w=img_w,
                    img_h=img_h,
                    altitude_m=altitude_m,
                    hfov_deg=hfov_deg,
                    drone_xy=drone_xy,
                    yaw_deg=yaw_deg,
                )
                people_positions.append(pos)
            elif lbl in VEHICLE_CLASSES:
                vehicles += 1

            if lbl in HAZARD_CLASSES:
                hazards[lbl] = hazards.get(lbl, 0) + 1

        priority = decide_priority(
            persons=persons,
            vehicles=vehicles,
            hazards=hazards,
            thermal_confirmed=thermal_confirmed,
        )

        return DetectionReport(
            drone_id=drone_id,
            timestamp=time.time(),
            persons=persons,
            vehicles=vehicles,
            hazards=hazards,
            priority=priority,
            people_positions=people_positions,
            detections=filtered,
        )
