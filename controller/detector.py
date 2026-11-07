"""Failure and abnormal behavior detector for drone swarm."""

from dataclasses import dataclass, field
import time
from typing import Any

from controller.config import SelfHealingConfig
from controller.models import DroneHealthStatus, DroneState, FailureReason


@dataclass
class DetectionResult:
    """Outcome of evaluating a drone state against operational thresholds."""

    drone_id: str
    evaluated_status: DroneHealthStatus
    failure_reasons: list[FailureReason] = field(default_factory=list)
    degraded_reasons: list[FailureReason] = field(default_factory=list)
    metrics_exceeded: dict[str, Any] = field(default_factory=dict)
    should_isolate_immediately: bool = False

    @property
    def is_healthy(self) -> bool:
        """Returns True if state passed all healthy checks."""
        return self.evaluated_status == DroneHealthStatus.HEALTHY

    @property
    def is_degraded(self) -> bool:
        """Returns True if state exhibits degraded performance."""
        return self.evaluated_status == DroneHealthStatus.DEGRADED

    @property
    def is_failed(self) -> bool:
        """Returns True if state exhibits critical failure or security breach."""
        return self.evaluated_status in (DroneHealthStatus.FAILED, DroneHealthStatus.ISOLATED)


class FailureDetector:
    """Evaluates telemetry and networking conditions to diagnose failures and anomalies."""

    def __init__(self, config: SelfHealingConfig | None = None) -> None:
        self.config = config or SelfHealingConfig()

    def evaluate(self, state: DroneState, current_time: float | None = None) -> DetectionResult:
        """Analyzes single drone state against configured thresholds.

        Differentiates between healthy, degraded, and critical failure conditions.
        """
        now = time.time() if current_time is None else current_time
        failure_reasons: list[FailureReason] = []
        degraded_reasons: list[FailureReason] = []
        exceeded: dict[str, Any] = {}
        isolate_immediately = False

        # 1. Check Disconnection / Heartbeat Timeout
        time_since_seen = now - state.last_seen
        if time_since_seen > self.config.heartbeat_timeout_s:
            failure_reasons.append(FailureReason.DISCONNECTION)
            exceeded["disconnection_timeout_s"] = round(time_since_seen, 2)

        # 2. Check Security / Threat Level
        if state.is_compromised or state.threat_level >= self.config.threat_critical:
            failure_reasons.append(FailureReason.SECURITY_COMPROMISED)
            exceeded["threat_level"] = state.threat_level
            if self.config.immediate_isolation_on_compromise:
                isolate_immediately = True
        elif state.threat_level >= self.config.threat_degraded:
            degraded_reasons.append(FailureReason.SECURITY_COMPROMISED)
            exceeded["threat_level_degraded"] = state.threat_level

        # 3. Check Battery Level
        if state.battery <= self.config.battery_critical:
            failure_reasons.append(FailureReason.LOW_BATTERY)
            exceeded["battery"] = state.battery
        elif state.battery <= self.config.battery_degraded:
            degraded_reasons.append(FailureReason.LOW_BATTERY)
            exceeded["battery_degraded"] = state.battery

        # 4. Check Signal Strength
        if state.signal_strength <= self.config.signal_critical_dbm:
            failure_reasons.append(FailureReason.WEAK_SIGNAL)
            exceeded["signal_strength"] = state.signal_strength
        elif state.signal_strength <= self.config.signal_degraded_dbm:
            degraded_reasons.append(FailureReason.WEAK_SIGNAL)
            exceeded["signal_strength_degraded"] = state.signal_strength

        # 5. Check Packet Loss
        if state.packet_loss >= self.config.packet_loss_critical:
            failure_reasons.append(FailureReason.EXCESSIVE_PACKET_LOSS)
            exceeded["packet_loss"] = state.packet_loss
        elif state.packet_loss >= self.config.packet_loss_degraded:
            degraded_reasons.append(FailureReason.EXCESSIVE_PACKET_LOSS)
            exceeded["packet_loss_degraded"] = state.packet_loss

        # 6. Check Latency
        if state.latency >= self.config.latency_critical_ms:
            failure_reasons.append(FailureReason.EXCESSIVE_LATENCY)
            exceeded["latency_ms"] = state.latency
        elif state.latency >= self.config.latency_degraded_ms:
            degraded_reasons.append(FailureReason.EXCESSIVE_LATENCY)
            exceeded["latency_ms_degraded"] = state.latency

        # Determine overall evaluated status
        if failure_reasons:
            evaluated_status = DroneHealthStatus.FAILED
        elif degraded_reasons:
            evaluated_status = DroneHealthStatus.DEGRADED
        else:
            evaluated_status = DroneHealthStatus.HEALTHY

        return DetectionResult(
            drone_id=state.drone_id,
            evaluated_status=evaluated_status,
            failure_reasons=failure_reasons,
            degraded_reasons=degraded_reasons,
            metrics_exceeded=exceeded,
            should_isolate_immediately=isolate_immediately,
        )

    def check_recovery_eligibility(self, state: DroneState) -> tuple[bool, list[str]]:
        """Verifies if an isolated or recovering drone satisfies strict criteria to rejoin.

        Criteria:
        1. Drone must NOT be marked compromised unless security clearance was explicitly granted.
        2. Battery must be >= battery_recovery threshold.
        3. Signal strength must be >= signal_recovery_dbm threshold.
        4. Packet loss must be <= packet_loss_recovery threshold.
        5. Latency must be <= latency_recovery_ms threshold.
        6. Threat level must be < threat_degraded threshold.
        """
        rejection_reasons: list[str] = []

        if state.is_compromised and not state.security_cleared:
            rejection_reasons.append(
                "Drone is marked compromised and lacks explicit security clearance."
            )

        if state.battery < self.config.battery_recovery:
            rejection_reasons.append(
                f"Battery ({state.battery:.1f}%) below recovery threshold ({self.config.battery_recovery}%)."
            )

        if state.signal_strength < self.config.signal_recovery_dbm:
            rejection_reasons.append(
                f"Signal strength ({state.signal_strength:.1f} dBm) below recovery threshold ({self.config.signal_recovery_dbm} dBm)."
            )

        if state.packet_loss > self.config.packet_loss_recovery:
            rejection_reasons.append(
                f"Packet loss ({state.packet_loss * 100:.1f}%) exceeds recovery threshold ({self.config.packet_loss_recovery * 100:.1f}%)."
            )

        if state.latency > self.config.latency_recovery_ms:
            rejection_reasons.append(
                f"Latency ({state.latency:.1f} ms) exceeds recovery threshold ({self.config.latency_recovery_ms} ms)."
            )

        if state.threat_level >= self.config.threat_degraded:
            rejection_reasons.append(
                f"Threat level ({state.threat_level:.2f}) exceeds degraded threshold ({self.config.threat_degraded:.2f})."
            )

        is_eligible = len(rejection_reasons) == 0
        return is_eligible, rejection_reasons

