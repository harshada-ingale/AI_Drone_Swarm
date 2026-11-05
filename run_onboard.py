"""CLI runner for onboard drone swarm vision node."""

import argparse
import os
import sys
from typing import Any
import cv2
import numpy as np

from vision.camera import Camera
from vision.detector_factory import load_detector
from vision.emergency_alert import AlertManager, EmergencyAlert
from vision.onboard_vision import OnboardVisionNode
from vision.telemetry import StaticTelemetry
from vision.tracker import SimpleTracker, Track


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run onboard drone swarm edge vision node."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Input video source (webcam index, video file path, or stream URL).",
    )
    parser.add_argument(
        "--thermal-source",
        default=None,
        help="Optional thermal camera source.",
    )
    parser.add_argument(
        "--model",
        default="vision/models/yolov8n.onnx",
        help="Path to detection model weights (.onnx, .pt, or .engine).",
    )
    parser.add_argument(
        "--drone-id",
        default="D7",
        help="Identifier for this drone node.",
    )
    parser.add_argument(
        "--altitude",
        type=float,
        default=30.0,
        help="Flight altitude in metres.",
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=84.0,
        help="Camera horizontal field of view in degrees.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Process every Nth frame.",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="Minimum track hits before confirmation.",
    )
    parser.add_argument(
        "--min-alert-interval",
        type=float,
        default=3.0,
        help="Minimum seconds between alerts (0 disables).",
    )
    parser.add_argument(
        "--log-csv",
        default=None,
        help="Path to log tracks CSV file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display video window with annotated detections and alerts.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Force threaded real-time frame dropping capture.",
    )
    parser.add_argument(
        "--no-thumbnails",
        action="store_true",
        help="Disable cropping and attaching thumbnails to alert evidence.",
    )
    parser.add_argument(
        "--max-thumbs",
        type=int,
        default=5,
        help="Maximum number of evidence thumbnails per alert.",
    )
    parser.add_argument(
        "--review-below",
        type=float,
        default=0.5,
        help="Confidence threshold below which alert is marked for review.",
    )
    return parser.parse_args()


def is_live_source(source: str, force_live: bool = False) -> bool:
    """Determines if camera source is a live real-time stream."""
    if force_live:
        return True
    if source.isdigit():
        return True
    lower = source.lower()
    return lower.startswith("rtsp://") or lower.startswith("http://") or lower.startswith("https://")


def main() -> None:
    """Entry point for onboard vision node runner."""
    args = parse_args()

    # Verify model existence
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.", file=sys.stderr)
        print("Please export or place your YOLOv8 ONNX model at this path.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading detector from {args.model}...")
    try:
        detector = load_detector(args.model)
    except Exception as err:
        print(f"Failed to load detector: {err}", file=sys.stderr)
        sys.exit(1)

    threaded = is_live_source(args.source, args.live)
    print(f"Initializing main camera '{args.source}' (threaded={threaded})...")
    camera = Camera(source=args.source, threaded=threaded)

    thermal_camera = None
    if args.thermal_source:
        thermal_threaded = is_live_source(args.thermal_source, args.live)
        print(f"Initializing thermal camera '{args.thermal_source}'...")
        thermal_camera = Camera(source=args.thermal_source, threaded=thermal_threaded)

    telemetry = StaticTelemetry(altitude_m=args.altitude, hfov_deg=args.hfov)
    tracker = SimpleTracker(min_hits=args.min_hits)
    alert_mgr = AlertManager(cooldown_s=0.0, review_below=args.review_below)

    def handle_alert(alert: EmergencyAlert) -> None:
        print(alert.summary())

    node_ref: list[OnboardVisionNode] = []

    def handle_frame(frame: np.ndarray, report: Any, tracks: list[Track]) -> None:
        if not args.show:
            return

        disp = frame.copy()
        # Top banner
        header = f"People: {report.persons} | Priority: {report.priority} | Drone: {report.drone_id}"
        color_map = {
            "CRITICAL": (0, 0, 255),
            "HIGH": (0, 140, 255),
            "MEDIUM": (0, 215, 255),
            "LOW": (0, 255, 0),
        }
        accent = color_map.get(report.priority, (255, 255, 255))
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 36), (30, 30, 30), -1)
        cv2.putText(disp, header, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, accent, 2)

        # Draw confirmed visible tracks
        for trk in tracks:
            if tracker.is_confirmed(trk) and trk.missed == 0:
                x1, y1, x2, y2 = [int(round(v)) for v in trk.bbox]
                cv2.rectangle(disp, (x1, y1), (x2, y2), accent, 2)
                tag = f"#{trk.track_id} {trk.label} {trk.confidence:.2f}"
                cv2.putText(
                    disp,
                    tag,
                    (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        cv2.imshow("AI Drone Swarm - Onboard Vision", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            if node_ref:
                node_ref[0].stop()

    node = OnboardVisionNode(
        detector=detector,
        camera=camera,
        drone_id=args.drone_id,
        telemetry=telemetry,
        thermal_camera=thermal_camera,
        tracker=tracker,
        alert_manager=alert_mgr,
        process_every=args.every,
        on_alert=handle_alert,
        on_frame=handle_frame,
        min_alert_interval_s=args.min_alert_interval,
        log_csv=args.log_csv,
        include_thumbnails=not args.no_thumbnails,
        max_evidence=args.max_thumbs,
        review_below=args.review_below,
    )
    node_ref.append(node)

    print(f"Drone Vision Node [{args.drone_id}] starting on source: {args.source}")
    print("Press Ctrl+C (or 'q' in video window) to stop.")

    try:
        node.run()
    except KeyboardInterrupt:
        print("\nShutdown requested by user (KeyboardInterrupt).")
    finally:
        node.stop()
        if args.show:
            cv2.destroyAllWindows()
        print("\n--- Mission Summary ---")
        st = node.stats()
        print(f"Frames processed : {st['frames_processed']}")
        print(f"Avg inference    : {st['avg_inference_ms']} ms")
        print(f"Processing FPS   : {st['processing_fps']}")
        print(f"Alerts raised    : {st['alerts_raised']}")


if __name__ == "__main__":
    main()
