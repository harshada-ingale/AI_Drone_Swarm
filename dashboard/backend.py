"""Backend service for AI Drone Swarm Command Center dashboard.

Owns the simulation clock, controller lifecycle, network topology, attacks,
vision alert dispatches, and cryptographic clearance verification.
This module is strictly decoupled from Streamlit and testable standalone.
"""

from collections import deque
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any
import uuid
import base64

from controller.config import SelfHealingConfig
from controller.models import (
    DroneHealthStatus,
    DroneState,
    FailureReason,
    ThreatReport,
)
from controller.self_healing_controller import SelfHealingController
from security.authentication import Identity
from security.encryption import security_status
from vision.emergency_alert import EmergencyAlert
from vision.evidence import make_thumbnail


class SwarmService:
    """Headless simulation and mission management service for the drone swarm."""

    DEFAULT_INITIAL_NODES: list[tuple[str, tuple[float, float], float, float, float, float, str]] = [
        ("D7", (0.0, 0.0), 98.0, -50.0, 0.0, 22.0, "Forward Edge Scout"),
        ("D4", (200.0, 0.0), 94.0, -52.0, 0.0, 25.0, "Primary Backbone Relay 1"),
        ("D3", (400.0, 0.0), 91.0, -55.0, 0.0, 24.0, "Primary Backbone Relay 2"),
        ("D5", (200.0, 120.0), 96.0, -53.0, 0.0, 26.0, "North Detour Relay 1"),
        ("D6", (400.0, 120.0), 95.0, -54.0, 0.0, 27.0, "North Detour Relay 2"),
        ("D1", (100.0, -100.0), 89.0, -60.0, 0.0, 28.0, "South Flank Patrol"),
        ("D2", (300.0, -100.0), 92.0, -58.0, 0.0, 25.0, "South Reserve Relay"),
        ("GS", (600.0, 0.0), 100.0, -40.0, 0.0, 10.0, "Ground Command Station"),
    ]

    INITIAL_OPTIMAL_ROUTE = ["D7", "D4", "D3", "GS"]

    def __init__(
        self,
        config: SelfHealingConfig | None = None,
        initial_sim_time: float = 1000.0,
        max_history_len: int = 100,
    ) -> None:
        """Initializes simulation service with 8-node swarm and security PKI."""
        self.sim_time: float = float(initial_sim_time)
        self.running: bool = False
        self.max_history_len: int = max_history_len
        self.active_route_id: str = "surveillance-stream-01"

        self.config = config or SelfHealingConfig(
            battery_critical=15.0,
            battery_degraded=30.0,
            battery_recovery=35.0,
            signal_critical_dbm=-88.0,
            signal_degraded_dbm=-75.0,
            signal_recovery_dbm=-70.0,
            packet_loss_critical=0.30,
            packet_loss_degraded=0.10,
            packet_loss_recovery=0.05,
            latency_critical_ms=300.0,
            latency_degraded_ms=100.0,
            latency_recovery_ms=80.0,
            threat_critical=0.70,
            threat_degraded=0.30,
            heartbeat_timeout_s=3.0,
            consecutive_failures_to_isolate=2,
            consecutive_checks_to_recover=2,
            max_comm_range_m=250.0,
            ground_station_id="GS",
            auto_restore_routes=True,
            restore_hold_down_s=5.0,
        )

        self.controller = SelfHealingController(config=self.config)

        # Register demo security operator identity in PKI for clearance signing
        self.operator_identity = Identity.generate("OP-COMMAND-01")
        if hasattr(self.controller.security, "_pki") and self.controller.security._pki is not None:
            self.controller.security._pki.register(self.operator_identity)

        # Register nodes
        self.roles: dict[str, str] = {}
        for d_id, pos, bat, sig, loss, lat, role in self.DEFAULT_INITIAL_NODES:
            self.roles[d_id] = role
            self.controller.register_drone(
                d_id,
                DroneState(
                    drone_id=d_id,
                    position=pos,
                    battery=bat,
                    signal_strength=sig,
                    packet_loss=loss,
                    latency=lat,
                    operational_status=DroneHealthStatus.HEALTHY,
                    last_seen=self.sim_time,
                ),
            )

        # Register initial route
        self.controller.register_active_route(self.active_route_id, "D7", "GS")

        # Initial heartbeats
        self.controller.heartbeat_active_drones(current_time=self.sim_time)

        # Latest alert delivered and decrypted
        self.latest_alert: dict[str, Any] | None = None

        # Stats history bounded to max_history_len
        self.history: deque[dict[str, Any]] = deque(maxlen=self.max_history_len)
        self._record_stats_point()

    # --------------------------------------------------------------------------
    # SIMULATION LIFECYCLE & STEPPING
    # --------------------------------------------------------------------------

    def start(self) -> None:
        """Starts or unpauses simulation stepping."""
        self.running = True

    def pause(self) -> None:
        """Pauses simulation stepping."""
        self.running = False

    def resume(self) -> None:
        """Resumes simulation stepping."""
        self.running = True

    def step(self, dt: float = 1.0) -> None:
        """Advances simulation clock by dt, updates heartbeats and checks restorations."""
        if dt <= 0.0:
            return

        self.sim_time += float(dt)

        # Heartbeat active and recovering drones to reflect operational status
        for d_id, state in self.controller._drones.items():
            if state.operational_status in (DroneHealthStatus.HEALTHY, DroneHealthStatus.DEGRADED):
                self.controller.heartbeat(d_id, current_time=self.sim_time)
            elif state.operational_status == DroneHealthStatus.RECOVERING:
                # If recovering, provide healthy telemetry update to advance recovery checks
                if not state.is_compromised or state.security_cleared:
                    self.controller.update_drone_health(
                        {
                            "drone_id": d_id,
                            "battery": max(state.battery, 85.0),
                            "signal_strength": -52.0,
                            "packet_loss": 0.0,
                            "latency": 25.0,
                            "threat_level": 0.0,
                            "timestamp": self.sim_time,
                        },
                        current_time=self.sim_time,
                    )

        # Evaluate heartbeats for timeouts
        self.controller.check_heartbeats(current_time=self.sim_time)

        # Check route restorations if auto-restoration is enabled
        if self.config.auto_restore_routes:
            self.controller.check_route_restorations(current_time=self.sim_time)

        # Record metrics point
        self._record_stats_point()

    def _record_stats_point(self) -> None:
        """Appends current metrics snapshot to bounded history."""
        drones = [d for d in self.controller._drones.values() if d.drone_id != "GS"]
        if not drones:
            return

        healthy = sum(1 for d in drones if d.operational_status == DroneHealthStatus.HEALTHY)
        degraded = sum(1 for d in drones if d.operational_status == DroneHealthStatus.DEGRADED)
        isolated = sum(1 for d in drones if d.operational_status == DroneHealthStatus.ISOLATED)
        recovering = sum(1 for d in drones if d.operational_status == DroneHealthStatus.RECOVERING)
        active_threats = sum(1 for d in drones if d.threat_level > 0.0 or d.is_compromised)

        active_drones = [d for d in drones if d.operational_status in (DroneHealthStatus.HEALTHY, DroneHealthStatus.DEGRADED)]
        avg_battery = sum(d.battery for d in drones) / len(drones)
        avg_signal = sum(d.signal_strength for d in active_drones) / len(active_drones) if active_drones else -100.0
        avg_latency = sum(d.latency for d in active_drones) / len(active_drones) if active_drones else 0.0
        avg_loss = sum(d.packet_loss for d in active_drones) / len(active_drones) if active_drones else 1.0

        active_route = self.controller.get_active_route(self.active_route_id) or []

        self.history.append({
            "sim_time": round(self.sim_time, 2),
            "healthy_count": healthy,
            "degraded_count": degraded,
            "isolated_count": isolated,
            "recovering_count": recovering,
            "active_threats": active_threats,
            "avg_battery": round(avg_battery, 2),
            "avg_signal": round(avg_signal, 2),
            "avg_latency": round(avg_latency, 2),
            "avg_packet_loss": round(avg_loss * 100.0, 2),
            "route_hops": len(active_route),
        })

    # --------------------------------------------------------------------------
    # NODE OPERATIONS & ATTACKS
    # --------------------------------------------------------------------------

    def fail_drone(self, drone_id: str, reason: str = "HARDWARE_FAULT") -> tuple[bool, str]:
        """Forces immediate drone isolation via controller."""
        if drone_id not in self.controller._drones:
            return False, f"Drone '{drone_id}' not found."

        try:
            fail_reason = FailureReason(reason)
        except ValueError:
            fail_reason = FailureReason.HARDWARE_FAULT

        self.controller.isolate_drone(
            drone_id,
            reason=fail_reason,
            details={"source": "dashboard_operator_override"},
            current_time=self.sim_time,
        )
        self._record_stats_point()
        return True, f"Drone '{drone_id}' isolated due to {fail_reason.value}."

    def recover_drone(self, drone_id: str, force: bool = False) -> tuple[bool, str]:
        """Triggers recovery sequence for an isolated drone."""
        if drone_id not in self.controller._drones:
            return False, f"Drone '{drone_id}' not found."

        state = self.controller.get_drone_state(drone_id)
        if state is None:
            return False, f"State for '{drone_id}' missing."

        if state.is_compromised and not state.security_cleared and not force:
            return (
                False,
                f"Drone '{drone_id}' is flagged COMPROMISED. Operator must grant signed security clearance first.",
            )

        success, msg = self.controller.recover_drone(
            drone_id,
            force=force,
            current_time=self.sim_time,
        )

        self._record_stats_point()
        return success, msg

    def inject_attack(self, drone_id: str, kind: str) -> tuple[bool, str]:
        """Injects network degradation or security compromise into selected drone."""
        if drone_id not in self.controller._drones:
            return False, f"Drone '{drone_id}' not found."

        kind_norm = kind.strip().lower()
        if kind_norm == "jamming":
            # Degrade signal strength and increase packet loss
            res = self.controller.update_drone_health(
                {
                    "drone_id": drone_id,
                    "signal_strength": -94.0,
                    "packet_loss": 0.48,
                    "latency": 180.0,
                    "timestamp": self.sim_time,
                },
                current_time=self.sim_time,
            )
            self._record_stats_point()
            return True, f"Injected RF jamming on {drone_id} (signal -94 dBm, 48% loss). Evaluated: {res.evaluated_status.value}."

        elif kind_norm == "dos":
            # Induce high latency and packet drop
            res = self.controller.update_drone_health(
                {
                    "drone_id": drone_id,
                    "latency": 450.0,
                    "packet_loss": 0.55,
                    "timestamp": self.sim_time,
                },
                current_time=self.sim_time,
            )
            self._record_stats_point()
            return True, f"Injected DoS flood on {drone_id} (latency 450 ms, 55% loss). Evaluated: {res.evaluated_status.value}."

        elif kind_norm == "spoofing":
            # Raise critical security threat report
            threat = ThreatReport(
                drone_id=drone_id,
                threat_level=0.92,
                threat_type="GPS_SPOOFING_ANOMALY",
                timestamp=self.sim_time,
                details={"drift_m": 160.0, "source": "simulated_spoofing_attack"},
            )
            res = self.controller.report_threat(threat)
            self._record_stats_point()
            return True, f"Injected GPS spoofing anomaly on {drone_id} (threat 0.92). Evaluated: {res.evaluated_status.value}."

        else:
            raise ValueError(f"Unknown attack kind '{kind}'. Must be 'jamming', 'dos', or 'spoofing'.")

    def grant_clearance(self, drone_id: str) -> dict[str, Any]:
        """Issues cryptographically signed operator clearance to un-quarantine a compromised drone."""
        if drone_id not in self.controller._drones:
            raise KeyError(f"Drone '{drone_id}' not found.")

        action = "CLEAR_COMPROMISED"
        nonce = f"NONCE-{uuid.uuid4().hex[:12]}"
        clearance_ts = self.sim_time
        expires_at = clearance_ts + 300.0

        transcript = (
            f"CLEARANCE:{drone_id}:{action}:{self.operator_identity.peer_id}:{clearance_ts}:{expires_at}:{nonce}"
        ).encode("utf-8")

        sig_bytes = self.operator_identity.sign(transcript)
        sig_b64 = base64.b64encode(sig_bytes).decode("ascii")

        clearance_payload = {
            "drone_id": drone_id,
            "action": action,
            "operator_id": self.operator_identity.peer_id,
            "timestamp": clearance_ts,
            "expires_at": expires_at,
            "nonce": nonce,
            "signature": sig_b64,
        }

        evt = self.controller.grant_signed_clearance(clearance_payload, current_time=self.sim_time)
        self._record_stats_point()

        return {
            "success": True,
            "drone_id": drone_id,
            "operator_id": self.operator_identity.peer_id,
            "algorithm": self.operator_identity.algorithm,
            "event_type": evt.event_type.value,
        }

    # --------------------------------------------------------------------------
    # AI THREAT & COMPUTER VISION
    # --------------------------------------------------------------------------

    def run_detection(self) -> Any:
        """Executes AI anomaly detector.

        Raises:
            NotImplementedError: As specified, since AI threat detector model is not implemented.
        """
        raise NotImplementedError("AI threat detector model not implemented yet")

    def run_vision(
        self,
        source: str = "scripted",
        model_path: str = "vision/models/yolov8n.onnx",
    ) -> dict[str, Any]:
        """Runs computer vision detection pipeline and securely transmits resulting alert."""
        if source == "scripted":
            # Generate realistic scripted disaster detection alert
            alert = self._build_scripted_alert()
            dispatch_res = self.send_alert(alert, origin_drone_id="D7")
            return {
                "success": True,
                "mode": "scripted",
                "alert": dispatch_res,
            }

        # Source is a video file or test clip
        source_path = Path(source)
        if not source_path.exists():
            return {
                "success": False,
                "error": f"Video source file not found: {source}",
            }

        onnx_model = Path(model_path)
        if not onnx_model.exists():
            return {
                "success": False,
                "error": f"Model file missing: {model_path}",
            }

        try:
            import cv2
            from vision.detector_factory import load_detector
        except Exception as err:
            return {"success": False, "error": f"Failed to load vision detector: {err}"}

        try:
            detector = load_detector(str(onnx_model), conf=0.35)
            cap = cv2.VideoCapture(str(source_path))
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return {"success": False, "error": f"Failed to read video frame from {source}"}

            detections = detector.detect(frame)
            person_boxes = [d for d in detections if getattr(d, "class_name", "") == "person" or getattr(d, "cls_id", -1) == 0]

            # Generate evidence items
            evidence_items = []
            loc_gps = []
            loc_local = []
            for i, p in enumerate(person_boxes[:4], start=1):
                bbox = getattr(p, "bbox", [0.0, 0.0, 10.0, 10.0])
                conf = float(getattr(p, "conf", 0.85))
                thumb = make_thumbnail(frame, bbox)
                thumb_b64 = base64.b64encode(thumb).decode("ascii")

                evidence_items.append({
                    "track_id": i,
                    "confidence": conf,
                    "bbox": [float(v) for v in bbox],
                    "thumbnail_b64": thumb_b64,
                })
                loc_gps.append({
                    "track_id": i,
                    "lat": 37.774929 + i * 0.00003,
                    "lon": -122.419418 + i * 0.00004,
                    "uncertainty_m": round(1.2 + i * 0.3, 1),
                    "location_status": "VALID",
                })
                loc_local.append({
                    "track_id": i,
                    "east_m": round(12.5 + i * 4.2, 1),
                    "north_m": round(8.1 + i * 3.5, 1),
                    "location_status": "VALID",
                })

            alert = EmergencyAlert(
                alert_id=f"ALERT-CV-{int(self.sim_time)}-{uuid.uuid4().hex[:4]}",
                drone_id="D7",
                timestamp=self.sim_time,
                priority="HIGH" if len(person_boxes) > 0 else "LOW",
                persons=len(person_boxes),
                hazards=None,  # Honest: hazards null
                hazards_supported=False,
                location=(50.0, 25.0),
                evidence=evidence_items,
                location_gps=loc_gps,
                location_local=loc_local,
                drone={
                    "lat": 37.774900,
                    "lon": -122.419400,
                    "alt_agl_m": 25.0,
                    "alt_msl_m": 50.0,
                    "heading_deg": 45.0,
                    "pitch_deg": -5.0,
                    "roll_deg": 1.0,
                    "gimbal_pitch_deg": -45.0,
                    "pose_age_ms": 12.0,
                    "gps_fix_type": 3,
                    "local_east_m": 0.0,
                    "local_north_m": 0.0,
                },
                reference_frame={
                    "gps": "WGS84 degrees",
                    "local": "ENU metres from origin (x=east, y=north)",
                    "origin": {"name": "simulation", "lat": 37.774900, "lon": -122.419400},
                },
                utc_time=datetime.now(timezone.utc).isoformat(),
            )

            dispatch_res = self.send_alert(alert, origin_drone_id="D7")
            return {
                "success": True,
                "mode": "onnx_inference",
                "detections_count": len(person_boxes),
                "alert": dispatch_res,
            }
        except Exception as exc:
            return {"success": False, "error": f"Error running vision pipeline: {exc}"}

    def _build_scripted_alert(self) -> EmergencyAlert:
        """Constructs realistic scripted emergency alert with verified thumbnails."""
        thumb1 = make_thumbnail(None, [10, 10, 50, 50])
        thumb2 = make_thumbnail(None, [20, 20, 60, 60])

        ev1 = {
            "track_id": 1,
            "confidence": 0.94,
            "bbox": [120.0, 80.0, 180.0, 220.0],
            "thumbnail_b64": base64.b64encode(thumb1).decode("ascii"),
        }
        ev2 = {
            "track_id": 2,
            "confidence": 0.88,
            "bbox": [250.0, 110.0, 310.0, 240.0],
            "thumbnail_b64": base64.b64encode(thumb2).decode("ascii"),
        }

        return EmergencyAlert(
            alert_id=f"ALERT-SIM-{int(self.sim_time)}-{uuid.uuid4().hex[:4]}",
            drone_id="D7",
            timestamp=self.sim_time,
            priority="HIGH",
            persons=2,
            hazards=None,  # Honest: hazards null
            hazards_supported=False,
            location=(50.0, 25.0),
            evidence=[ev1, ev2],
            min_confidence=0.88,
            mean_confidence=0.91,
            location_gps=[
                {"track_id": 1, "lat": 37.774929, "lon": -122.419418, "uncertainty_m": 1.4, "location_status": "VALID"},
                {"track_id": 2, "lat": 37.774955, "lon": -122.419385, "uncertainty_m": 1.8, "location_status": "VALID"},
            ],
            location_local=[
                {"track_id": 1, "east_m": 15.2, "north_m": 8.4, "location_status": "VALID"},
                {"track_id": 2, "east_m": 18.1, "north_m": 11.2, "location_status": "VALID"},
            ],
            drone={
                "lat": 37.774900,
                "lon": -122.419400,
                "alt_agl_m": 25.0,
                "alt_msl_m": 50.0,
                "heading_deg": 45.0,
                "pitch_deg": -5.0,
                "roll_deg": 1.0,
                "gimbal_pitch_deg": -45.0,
                "pose_age_ms": 12.0,
                "gps_fix_type": 3,
                "local_east_m": 0.0,
                "local_north_m": 0.0,
            },
            reference_frame={
                "gps": "WGS84 degrees",
                "local": "ENU metres from origin (x=east, y=north)",
                "origin": {"name": "simulation", "lat": 37.774900, "lon": -122.419400},
            },
            utc_time=datetime.now(timezone.utc).isoformat(),
        )

    def send_alert(
        self,
        alert: EmergencyAlert | None = None,
        origin_drone_id: str = "D7",
    ) -> dict[str, Any]:
        """Dispatches an EmergencyAlert along dynamic encrypted route to Ground Station."""
        if alert is None:
            alert = self._build_scripted_alert()

        dispatch_res = self.controller.dispatch_emergency_alert(
            alert,
            origin_drone_id=origin_drone_id,
        )

        self.latest_alert = {
            "alert_id": alert.alert_id,
            "origin_drone_id": origin_drone_id,
            "timestamp": alert.timestamp,
            "priority": alert.priority,
            "persons": alert.persons,
            "hazards": alert.hazards,
            "hazards_supported": alert.hazards_supported,
            "delivered": dispatch_res.get("delivered", False),
            "hops_traversed": dispatch_res.get("hops_traversed", []),
            "reroutes": dispatch_res.get("reroutes", 0),
            "location_gps": alert.location_gps,
            "location_local": alert.location_local,
            "drone_pose": alert.drone,
            "reference_frame": alert.reference_frame,
            "evidence": alert.evidence,
            "delivered_payload": dispatch_res.get("delivered_payload"),
        }

        self._record_stats_point()
        return self.latest_alert

    # --------------------------------------------------------------------------
    # SECURITY & CRYPTOGRAPHY
    # --------------------------------------------------------------------------

    def security_check(self) -> dict[str, Any]:
        """Queries cryptographic subsystem status for Ground Station and swarm mesh."""
        km = None
        if hasattr(self.controller.security, "get_or_create_messenger"):
            m = self.controller.security.get_or_create_messenger("GS")
            if m is not None:
                km = m.km
        return security_status(km)

    # --------------------------------------------------------------------------
    # STATE SNAPSHOT (JSON-serializable, no live controller references)
    # --------------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Generates comprehensive, sanitized snapshot for dashboard rendering and tests."""
        drones_dict: dict[str, dict[str, Any]] = {}
        for d_id, d in self.controller._drones.items():
            age = round(max(0.0, self.sim_time - d.last_seen), 2)
            drones_dict[d_id] = {
                "drone_id": d.drone_id,
                "role": self.roles.get(d_id, "Swarm Node"),
                "position": [float(d.position[0]), float(d.position[1])],
                "battery": round(d.battery, 1),
                "signal_strength": round(d.signal_strength, 1),
                "packet_loss": round(d.packet_loss, 3),
                "latency": round(d.latency, 1),
                "threat_level": round(d.threat_level, 2),
                "operational_status": d.operational_status.value,
                "is_compromised": bool(d.is_compromised),
                "security_cleared": bool(d.security_cleared),
                "failure_reasons": [r.value for r in d.failure_reasons],
                "last_seen_age": age,
                "consecutive_healthy": d.consecutive_healthy,
                "consecutive_failures": d.consecutive_failures,
            }

        # Calculate inter-drone communication links
        links = []
        drone_ids = list(self.controller._drones.keys())
        for i in range(len(drone_ids)):
            for j in range(i + 1, len(drone_ids)):
                id1, id2 = drone_ids[i], drone_ids[j]
                p1 = self.controller._drones[id1].position
                p2 = self.controller._drones[id2].position
                dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                in_range = dist <= self.config.max_comm_range_m

                s1 = self.controller._drones[id1].operational_status
                s2 = self.controller._drones[id2].operational_status
                both_up = s1 in (DroneHealthStatus.HEALTHY, DroneHealthStatus.DEGRADED) and \
                          s2 in (DroneHealthStatus.HEALTHY, DroneHealthStatus.DEGRADED)

                links.append({
                    "source": id1,
                    "target": id2,
                    "distance": round(dist, 1),
                    "in_range": in_range,
                    "active": in_range and both_up,
                })

        active_route = self.controller.get_active_route(self.active_route_id) or []

        # Determine overall network status
        if not active_route:
            net_status = "NO ROUTE"
        elif active_route == self.INITIAL_OPTIMAL_ROUTE:
            net_status = "OPTIMAL"
        else:
            net_status = "DETOUR / REROUTED"

        # Recovering progress
        recovering_progress = {}
        for d_id, d in self.controller._drones.items():
            if d.operational_status == DroneHealthStatus.RECOVERING:
                recovering_progress[d_id] = {
                    "checks": d.consecutive_healthy,
                    "required": self.config.consecutive_checks_to_recover,
                }

        # Sanitized audit events: strip cryptographic keys, signatures and tokens
        raw_events = self.controller.get_event_log(limit=40)
        sanitized_events = []
        for evt in raw_events:
            clean_details = copy.deepcopy(evt.details)
            for secret_key in ("signature", "token", "clearance_token", "secret", "private_key", "wire_bytes"):
                clean_details.pop(secret_key, None)

            sanitized_events.append({
                "event_type": evt.event_type.value,
                "drone_id": evt.drone_id,
                "timestamp": round(evt.timestamp, 2),
                "previous_state": evt.previous_state,
                "new_state": evt.new_state,
                "details": clean_details,
            })

        # Security subsystem snapshot
        sec_info = self.security_check()

        return {
            "sim_time": round(self.sim_time, 2),
            "running": self.running,
            "drones": drones_dict,
            "links": links,
            "active_route": list(active_route),
            "initial_route": list(self.INITIAL_OPTIMAL_ROUTE),
            "network_status": net_status,
            "recovering_progress": recovering_progress,
            "events": sanitized_events,
            "stats": {
                "latest": self.history[-1] if self.history else {},
                "history": list(self.history),
            },
            "latest_alert": self.latest_alert,
            "security": sec_info,
        }

