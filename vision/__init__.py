"""Vision package for AI Drone Swarm edge computing."""

from vision.camera import Camera
from vision.detector_factory import load_detector
from vision.disaster_detection import (
    HAZARD_CLASSES,
    PERSON_CLASSES,
    VEHICLE_CLASSES,
    DetectionReport,
    DisasterAnalyzer,
    decide_priority,
    pixel_to_ground,
)
from vision.emergency_alert import AlertManager, EmergencyAlert
from vision.onboard_vision import OnboardVisionNode
from vision.onnx_detector import OnnxYoloDetector, decode_yolov8, letterbox
from vision.telemetry import StaticTelemetry, Telemetry
from vision.thermal import thermal_confirms
from vision.tracker import SimpleTracker, Track, iou
from vision.yolo_detector import Detection, YoloDetector

__all__ = [
    "AlertManager",
    "Camera",
    "Detection",
    "DetectionReport",
    "DisasterAnalyzer",
    "EmergencyAlert",
    "HAZARD_CLASSES",
    "OnboardVisionNode",
    "OnnxYoloDetector",
    "PERSON_CLASSES",
    "SimpleTracker",
    "StaticTelemetry",
    "Telemetry",
    "Track",
    "VEHICLE_CLASSES",
    "YoloDetector",
    "decide_priority",
    "decode_yolov8",
    "iou",
    "letterbox",
    "load_detector",
    "pixel_to_ground",
    "thermal_confirms",
]
