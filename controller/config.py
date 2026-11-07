"""Configuration and thresholds for the Self-Healing Swarm Controller."""

from dataclasses import dataclass, field
import re


@dataclass
class SelfHealingConfig:
    """Configurable thresholds and operational parameters for drone self-healing."""

    # Battery thresholds (percentages 0.0 - 100.0)
    battery_critical: float = 15.0
    battery_degraded: float = 30.0
    battery_recovery: float = 35.0

    # Signal strength thresholds (dBm: -100 to 0)
    signal_critical_dbm: float = -88.0
    signal_degraded_dbm: float = -75.0
    signal_recovery_dbm: float = -70.0

    # Packet loss thresholds (ratio 0.0 - 1.0)
    packet_loss_critical: float = 0.30
    packet_loss_degraded: float = 0.10
    packet_loss_recovery: float = 0.05

    # Latency thresholds (milliseconds)
    latency_critical_ms: float = 300.0
    latency_degraded_ms: float = 120.0
    latency_recovery_ms: float = 80.0

    # Threat level thresholds (0.0 - 1.0)
    threat_critical: float = 0.70
    threat_degraded: float = 0.40

    # Heartbeat & disconnection timeouts
    heartbeat_timeout_s: float = 5.0

    # Fluctuation dampening & state confirmation counts
    consecutive_failures_to_isolate: int = 3
    consecutive_checks_to_recover: int = 3
    confirmation_period_s: float = 2.0
    immediate_isolation_on_compromise: bool = True

    # Swarm network & topology configuration
    max_comm_range_m: float = 500.0
    ground_station_id: str = "GS"
    auto_restore_routes: bool = True

    # Drone ID validation pattern (default matches D1-D20, GS, or custom D<N>)
    drone_id_pattern: str = r"^(D([1-9]|1[0-9]|20)|GS)$"
    strict_drone_id_validation: bool = False

    def is_valid_drone_id(self, drone_id: str) -> bool:
        """Validates whether drone_id conforms to expected swarm naming."""
        if not drone_id or not isinstance(drone_id, str):
            return False
        if not self.strict_drone_id_validation:
            # Allows standard D<number> or GS or any alphanumeric ID if strict mode is disabled
            return bool(re.match(r"^[A-Za-z0-9_\-]+$", drone_id))
        return bool(re.match(self.drone_id_pattern, drone_id))

