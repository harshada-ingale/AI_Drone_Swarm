"""Data models and enums for drone health, status, and self-healing events."""

from dataclasses import asdict, dataclass, field
from enum import Enum
import time
from typing import Any
import uuid


class DroneHealthStatus(str, Enum):
    """Operational health status of a drone within the swarm."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    ISOLATED = "ISOLATED"
    RECOVERING = "RECOVERING"


class FailureReason(str, Enum):
    """Specific failure or anomaly categories identified in drone operations."""

    NONE = "NONE"
    DISCONNECTION = "DISCONNECTION"
    LOW_BATTERY = "LOW_BATTERY"
    WEAK_SIGNAL = "WEAK_SIGNAL"
    EXCESSIVE_PACKET_LOSS = "EXCESSIVE_PACKET_LOSS"
    EXCESSIVE_LATENCY = "EXCESSIVE_LATENCY"
    SECURITY_COMPROMISED = "SECURITY_COMPROMISED"
    UNRESPONSIVE = "UNRESPONSIVE"
    HARDWARE_FAULT = "HARDWARE_FAULT"


class EventType(str, Enum):
    """Self-healing lifecycle event types."""

    HEALTH_UPDATE = "HEALTH_UPDATE"
    STATUS_CHANGED = "STATUS_CHANGED"
    FAILURE_DETECTED = "FAILURE_DETECTED"
    DRONE_ISOLATED = "DRONE_ISOLATED"
    ROUTE_REROUTED = "ROUTE_REROUTED"
    ROUTING_FAILED = "ROUTING_FAILED"
    RECOVERY_STARTED = "RECOVERY_STARTED"
    DRONE_RECOVERED = "DRONE_RECOVERED"
    SECURITY_ALERT = "SECURITY_ALERT"
    SECURITY_CLEARANCE_GRANTED = "SECURITY_CLEARANCE_GRANTED"
    SECURITY_DEGRADED = "SECURITY_DEGRADED"
    ROUTE_RESTORED = "ROUTE_RESTORED"


@dataclass
class DroneState:
    """Represents the telemetry, networking, and health state of a single drone."""

    drone_id: str
    position: tuple[float, float] = (0.0, 0.0)  # (x, y) coordinates in metres
    altitude: float = 30.0                      # altitude in metres
    heading: float = 0.0                       # yaw heading in degrees (0 - 360)
    battery: float = 100.0                     # battery level percentage (0.0 - 100.0)
    signal_strength: float = -50.0             # signal strength in dBm (-100 to 0)
    packet_loss: float = 0.0                   # packet loss ratio (0.0 to 1.0)
    latency: float = 20.0                      # round-trip latency in milliseconds
    threat_level: float = 0.0                  # security threat level score (0.0 to 1.0)
    operational_status: DroneHealthStatus = DroneHealthStatus.HEALTHY
    failure_reasons: list[FailureReason] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    is_compromised: bool = False
    security_cleared: bool = False
    consecutive_failures: int = 0
    consecutive_healthy: int = 0
    previous_status: DroneHealthStatus | None = None
    isolation_timestamp: float | None = None
    recovery_timestamp: float | None = None

    def validate(self) -> None:
        """Validates metric ranges, raising ValueError if any value is out of bounds."""
        if not self.drone_id or not isinstance(self.drone_id, str):
            raise ValueError("drone_id must be a non-empty string.")

        if not (0.0 <= self.battery <= 100.0):
            raise ValueError(f"Battery must be between 0.0 and 100.0, got {self.battery}")

        if not (-150.0 <= self.signal_strength <= 0.0):
            raise ValueError(f"Signal strength must be between -150.0 and 0.0 dBm, got {self.signal_strength}")

        if not (0.0 <= self.packet_loss <= 1.0):
            raise ValueError(f"Packet loss must be between 0.0 and 1.0, got {self.packet_loss}")

        if self.latency < 0.0:
            raise ValueError(f"Latency must be non-negative, got {self.latency}")

        if not (0.0 <= self.threat_level <= 1.0):
            raise ValueError(f"Threat level must be between 0.0 and 1.0, got {self.threat_level}")

    def to_dict(self) -> dict[str, Any]:
        """Serializes drone state to dictionary."""
        data = asdict(self)
        data["operational_status"] = self.operational_status.value
        data["failure_reasons"] = [r.value for r in self.failure_reasons]
        if self.previous_status is not None:
            data["previous_status"] = self.previous_status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DroneState":
        """Constructs DroneState from dictionary."""
        copied = dict(data)
        if "operational_status" in copied and isinstance(copied["operational_status"], str):
            copied["operational_status"] = DroneHealthStatus(copied["operational_status"])
        if "failure_reasons" in copied and isinstance(copied["failure_reasons"], list):
            copied["failure_reasons"] = [
                FailureReason(r) if isinstance(r, str) else r
                for r in copied["failure_reasons"]
            ]
        if "previous_status" in copied and isinstance(copied["previous_status"], str):
            copied["previous_status"] = DroneHealthStatus(copied["previous_status"])
        if "position" in copied and isinstance(copied["position"], list):
            copied["position"] = (float(copied["position"][0]), float(copied["position"][1]))
        return cls(**copied)


@dataclass
class SelfHealingEvent:
    """Audit log entry capturing a self-healing action, state transition, or route change."""

    event_id: str = field(default_factory=lambda: f"EVT-{uuid.uuid4().hex[:8].upper()}")
    event_type: EventType = EventType.HEALTH_UPDATE
    drone_id: str = ""
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)
    previous_state: str | None = None
    new_state: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "drone_id": self.drone_id,
            "timestamp": self.timestamp,
            "details": self.details,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
        }

    def summary(self) -> str:
        """Returns human-readable one-line event summary."""
        prev_str = f" from {self.previous_state}" if self.previous_state else ""
        new_str = f" to {self.new_state}" if self.new_state else ""
        return (
            f"[{self.event_type.value}] Drone: {self.drone_id}{prev_str}{new_str} | "
            f"Details: {self.details}"
        )


@dataclass
class ThreatReport:
    """Security/anomaly threat notification from AI or security subsystem."""

    drone_id: str
    threat_level: float               # Threat score 0.0 - 1.0
    threat_type: str = "ANOMALY"      # e.g., 'GPS_SPOOFING', 'COMPROMISED_TELEMETRY', 'UNAUTHORIZED_TRAFFIC'
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)

