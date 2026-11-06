"""Recovery lifecycle manager for restoring failed and isolated drones."""

from dataclasses import dataclass, field
import time
from typing import Any

from controller.config import SelfHealingConfig
from controller.detector import FailureDetector
from controller.models import (
    DroneHealthStatus,
    DroneState,
    EventType,
    FailureReason,
    SelfHealingEvent,
)


@dataclass
class RecoveryRecord:
    """Historical tracking of a drone's recovery process."""

    drone_id: str
    isolation_time: float
    recovery_start_time: float | None = None
    recovered_time: float | None = None
    isolation_reasons: list[FailureReason] = field(default_factory=list)
    successful: bool = False
    notes: str = ""


class RecoveryManager:
    """Coordinates verification, safety checks, and gradual reintegration of recovering drones."""

    def __init__(
        self,
        config: SelfHealingConfig | None = None,
        detector: FailureDetector | None = None,
    ) -> None:
        self.config = config or SelfHealingConfig()
        self.detector = detector or FailureDetector(self.config)
        self._history: dict[str, list[RecoveryRecord]] = {}

    def start_recovery(self, state: DroneState, current_time: float | None = None) -> tuple[bool, str, SelfHealingEvent | None]:
        """Attempts to initiate the recovery lifecycle for an isolated drone.

        Returns:
            (success, message, event_or_none)
        """
        now = time.time() if current_time is None else current_time

        if state.operational_status != DroneHealthStatus.ISOLATED:
            return False, f"Drone {state.drone_id} is in status {state.operational_status.value}, not ISOLATED.", None

        # Check eligibility
        is_eligible, reasons = self.detector.check_recovery_eligibility(state)
        if not is_eligible:
            msg = f"Recovery rejected for {state.drone_id}: {'; '.join(reasons)}"
            return False, msg, None

        # Move to RECOVERING
        prev_status = state.operational_status
        state.previous_status = prev_status
        state.operational_status = DroneHealthStatus.RECOVERING
        state.consecutive_healthy = 1

        rec = RecoveryRecord(
            drone_id=state.drone_id,
            isolation_time=state.isolation_timestamp or now,
            recovery_start_time=now,
            isolation_reasons=list(state.failure_reasons),
        )
        self._history.setdefault(state.drone_id, []).append(rec)

        evt = SelfHealingEvent(
            event_type=EventType.RECOVERY_STARTED,
            drone_id=state.drone_id,
            timestamp=now,
            previous_state=prev_status.value,
            new_state=DroneHealthStatus.RECOVERING.value,
            details={
                "message": "Initiated verification checks for drone reintegration.",
                "battery": state.battery,
                "signal_strength": state.signal_strength,
                "packet_loss": state.packet_loss,
            },
        )
        return True, f"Drone {state.drone_id} entered RECOVERING state.", evt

    def process_recovery_check(
        self,
        state: DroneState,
        current_time: float | None = None,
    ) -> tuple[bool, str, SelfHealingEvent | None]:
        """Evaluates subsequent health updates for a drone currently in RECOVERING state.

        Returns:
            (recovered, message, event_or_none)
        """
        now = time.time() if current_time is None else current_time

        if state.operational_status != DroneHealthStatus.RECOVERING:
            return False, f"Drone {state.drone_id} is not in RECOVERING state.", None

        is_eligible, reasons = self.detector.check_recovery_eligibility(state)
        if not is_eligible:
            # Re-isolate
            prev_status = state.operational_status
            state.previous_status = prev_status
            state.operational_status = DroneHealthStatus.ISOLATED
            state.consecutive_healthy = 0

            # Record failure
            if state.drone_id in self._history and self._history[state.drone_id]:
                self._history[state.drone_id][-1].notes = f"Aborted: {'; '.join(reasons)}"

            evt = SelfHealingEvent(
                event_type=EventType.DRONE_ISOLATED,
                drone_id=state.drone_id,
                timestamp=now,
                previous_state=prev_status.value,
                new_state=DroneHealthStatus.ISOLATED.value,
                details={
                    "message": "Recovery verification failed. Re-isolating drone.",
                    "reasons": reasons,
                },
            )
            return False, f"Recovery aborted for {state.drone_id}: {'; '.join(reasons)}", evt

        state.consecutive_healthy += 1

        # Check if enough consecutive checks passed
        if state.consecutive_healthy >= self.config.consecutive_checks_to_recover:
            prev_status = state.operational_status
            state.previous_status = prev_status
            state.operational_status = DroneHealthStatus.HEALTHY
            state.failure_reasons.clear()
            state.consecutive_failures = 0
            state.consecutive_healthy = 0
            state.isolation_timestamp = None

            if state.drone_id in self._history and self._history[state.drone_id]:
                rec = self._history[state.drone_id][-1]
                rec.successful = True
                rec.recovered_time = now

            evt = SelfHealingEvent(
                event_type=EventType.DRONE_RECOVERED,
                drone_id=state.drone_id,
                timestamp=now,
                previous_state=prev_status.value,
                new_state=DroneHealthStatus.HEALTHY.value,
                details={
                    "message": "Drone successfully passed all health verifications and returned to active duty.",
                    "consecutive_passed": self.config.consecutive_checks_to_recover,
                },
            )
            return True, f"Drone {state.drone_id} fully recovered and returned to HEALTHY.", evt

        return False, (
            f"Drone {state.drone_id} progress: {state.consecutive_healthy}/"
            f"{self.config.consecutive_checks_to_recover} checks passed."
        ), None

    def grant_security_clearance(
        self,
        state: DroneState,
        clearance_token: str | None = None,
        current_time: float | None = None,
    ) -> SelfHealingEvent:
        """Explicitly grants security clearance to a compromised drone."""
        now = time.time() if current_time is None else current_time
        state.security_cleared = True
        state.is_compromised = False
        state.threat_level = 0.0

        evt = SelfHealingEvent(
            event_type=EventType.SECURITY_ALERT,
            drone_id=state.drone_id,
            timestamp=now,
            previous_state=state.operational_status.value,
            new_state=state.operational_status.value,
            details={
                "message": "Security clearance granted. Drone is now eligible for recovery evaluation.",
                "token": clearance_token or "MANUAL_OVERRIDE",
            },
        )
        return evt

