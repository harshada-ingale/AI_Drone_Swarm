"""Comprehensive test suite for the AI Drone Swarm Self-Healing Mechanism.

Verifies MONITOR, DETECT, ISOLATE, REROUTE, RECOVER, and VERIFY phases across all
operational scenarios, boundary conditions, and external module integrations.
"""

import time
import pytest

from controller.adapters import (
    AIThreatAdapter,
    SecurityAdapter,
    SimulatorAdapter,
    VisionAlertAdapter,
)
from controller.config import SelfHealingConfig
from controller.detector import FailureDetector
from controller.models import (
    DroneHealthStatus,
    DroneState,
    EventType,
    FailureReason,
    SelfHealingEvent,
    ThreatReport,
)
from controller.recovery import RecoveryManager
from controller.self_healing_controller import SelfHealingController
from routing.dijkstra import DijkstraRouter
from routing.router import NoRouteFoundError
from vision.emergency_alert import EmergencyAlert


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def fast_config() -> SelfHealingConfig:
    """Config with small confirmation counts for fast, deterministic unit tests."""
    return SelfHealingConfig(
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
        latency_degraded_ms=120.0,
        latency_recovery_ms=80.0,
        threat_critical=0.70,
        threat_degraded=0.40,
        heartbeat_timeout_s=5.0,
        consecutive_failures_to_isolate=3,
        consecutive_checks_to_recover=2,
        confirmation_period_s=0.1,
        max_comm_range_m=250.0,
        ground_station_id="GS",
        auto_restore_routes=True,
    )


@pytest.fixture
def controller(fast_config: SelfHealingConfig) -> SelfHealingController:
    """Returns a SelfHealingController initialized with fast_config."""
    return SelfHealingController(config=fast_config)


@pytest.fixture
def linear_swarm_controller(fast_config: SelfHealingConfig) -> SelfHealingController:
    """Controller with a pre-configured linear mesh network:

    D7 (x=0) <-> D4 (x=200) <-> D3 (x=400) <-> GS (x=600)
    Alternative detour bypass:
    D7 (x=0) <-> D5 (x=200, y=120) <-> D6 (x=400, y=120) <-> GS (x=600)
    """
    ctrl = SelfHealingController(config=fast_config)

    nodes = [
        ("D7", (0.0, 0.0)),
        ("D4", (200.0, 0.0)),
        ("D3", (400.0, 0.0)),
        ("D5", (200.0, 120.0)),
        ("D6", (400.0, 120.0)),
        ("GS", (600.0, 0.0)),
    ]

    for d_id, pos in nodes:
        ctrl.register_drone(
            d_id,
            DroneState(
                drone_id=d_id,
                position=pos,
                battery=100.0,
                signal_strength=-50.0,
                packet_loss=0.0,
                latency=20.0,
                threat_level=0.0,
                operational_status=DroneHealthStatus.HEALTHY,
            ),
        )

    return ctrl


# ==============================================================================
# PHASE 5 CORE SCENARIOS (1 to 15)
# ==============================================================================

def test_01_all_drones_healthy(linear_swarm_controller: SelfHealingController) -> None:
    """Scenario 1: All drones healthy -> all remain HEALTHY, routing succeeds."""
    ctrl = linear_swarm_controller
    for d_id in ["D7", "D4", "D3", "D5", "D6"]:
        st = ctrl.get_drone_state(d_id)
        assert st is not None
        assert st.operational_status == DroneHealthStatus.HEALTHY

    route = ctrl.get_route("D7", "GS")
    # Should select direct linear path: D7 -> D4 -> D3 -> GS
    assert route == ["D7", "D4", "D3", "GS"]
    assert len(ctrl.get_isolated_and_failed_drones()) == 0


def test_02_one_drone_suddenly_disconnects(linear_swarm_controller: SelfHealingController) -> None:
    """Scenario 2: One drone suddenly disconnects -> detected via heartbeat timeout."""
    ctrl = linear_swarm_controller
    now = time.time()

    # D4 last seen 10 seconds ago (heartbeat timeout is 5.0s)
    st = ctrl.get_drone_state("D4")
    assert st is not None
    st.last_seen = now - 10.0

    # D4 needs consecutive_failures_to_isolate checks (3 checks)
    ctrl.check_heartbeats(current_time=now)
    ctrl.check_heartbeats(current_time=now)
    isolated = ctrl.check_heartbeats(current_time=now)

    assert "D4" in isolated
    assert st.operational_status == DroneHealthStatus.ISOLATED
    assert FailureReason.DISCONNECTION in st.failure_reasons


def test_03_one_drone_has_low_battery(linear_swarm_controller: SelfHealingController) -> None:
    """Scenario 3: One drone has low battery (<15%) -> isolates after confirmation."""
    ctrl = linear_swarm_controller
    now = time.time()

    # Feed 3 updates with critical battery (10%)
    for _ in range(3):
        ctrl.update_drone_health(
            {"drone_id": "D4", "battery": 10.0, "timestamp": now},
            current_time=now,
        )

    st = ctrl.get_drone_state("D4")
    assert st is not None
    assert st.operational_status == DroneHealthStatus.ISOLATED
    assert FailureReason.LOW_BATTERY in st.failure_reasons


def test_04_one_drone_experiences_weak_signal(linear_swarm_controller: SelfHealingController) -> None:
    """Scenario 4: One drone experiences weak signal (<= -88 dBm)."""
    ctrl = linear_swarm_controller
    now = time.time()

    for _ in range(3):
        ctrl.update_drone_health(
            {"drone_id": "D3", "signal_strength": -95.0, "timestamp": now},
            current_time=now,
        )

    st = ctrl.get_drone_state("D3")
    assert st is not None
    assert st.operational_status == DroneHealthStatus.ISOLATED
    assert FailureReason.WEAK_SIGNAL in st.failure_reasons


def test_05_one_drone_has_excessive_packet_loss(linear_swarm_controller: SelfHealingController) -> None:
    """Scenario 5: One drone experiences excessive packet loss (>= 30%)."""
    ctrl = linear_swarm_controller
    now = time.time()

    for _ in range(3):
        ctrl.update_drone_health(
            {"drone_id": "D4", "packet_loss": 0.45, "timestamp": now},
            current_time=now,
        )

    st = ctrl.get_drone_state("D4")
    assert st is not None
    assert st.operational_status == DroneHealthStatus.ISOLATED
    assert FailureReason.EXCESSIVE_PACKET_LOSS in st.failure_reasons


def test_06_one_drone_becomes_compromised(linear_swarm_controller: SelfHealingController) -> None:
    """Scenario 6: One drone becomes compromised -> isolated immediately."""
    ctrl = linear_swarm_controller
    now = time.time()

    # Immediate isolation on threat report
    ctrl.report_threat(
        ThreatReport(
            drone_id="D4",
            threat_level=0.95,
            threat_type="MALICIOUS_FIRMWARE",
            timestamp=now,
        )
    )

    st = ctrl.get_drone_state("D4")
    assert st is not None
    assert st.operational_status == DroneHealthStatus.ISOLATED
    assert st.is_compromised is True
    assert FailureReason.SECURITY_COMPROMISED in st.failure_reasons


def test_07_failed_drone_gets_isolated(linear_swarm_controller: SelfHealingController) -> None:
    """Scenario 7: Failed drone gets isolated, excluded from routing, failure recorded."""
    ctrl = linear_swarm_controller
    event = ctrl.isolate_drone("D4", reason=FailureReason.HARDWARE_FAULT)

    assert event.event_type == EventType.DRONE_ISOLATED
    assert event.drone_id == "D4"
    assert "D4" in ctrl.get_isolated_and_failed_drones()

    st = ctrl.get_drone_state("D4")
    assert st is not None
    assert st.operational_status == DroneHealthStatus.ISOLATED
    assert st.previous_status == DroneHealthStatus.HEALTHY
    assert st.isolation_timestamp is not None


def test_08_communication_gets_rerouted_through_healthy_drones(
    linear_swarm_controller: SelfHealingController,
) -> None:
    """Scenario 8: Route from D7 to GS reroutes when D4 fails."""
    ctrl = linear_swarm_controller

    # Register active route
    initial_route = ctrl.register_active_route("mission-alert-stream", "D7", "GS")
    assert initial_route == ["D7", "D4", "D3", "GS"]

    # Now D4 fails and gets isolated
    ctrl.isolate_drone("D4", reason=FailureReason.LOW_BATTERY)

    # Active route should automatically reroute through healthy drones (e.g. D5 and D6 or D3)
    updated_route = ctrl.get_active_route("mission-alert-stream")
    assert updated_route is not None
    assert "D4" not in updated_route
    assert updated_route in (["D7", "D5", "D6", "GS"], ["D7", "D5", "D3", "GS"])


def test_09_no_alternative_route_exists(fast_config: SelfHealingConfig) -> None:
    """Scenario 9: Critical bridge drone fails and no alternative path exists."""
    ctrl = SelfHealingController(config=fast_config)

    # Simple single-bridge topology: D1 <-> D2 <-> GS
    ctrl.register_drone("D1", DroneState(drone_id="D1", position=(0.0, 0.0)))
    ctrl.register_drone("D2", DroneState(drone_id="D2", position=(200.0, 0.0)))
    ctrl.register_drone("GS", DroneState(drone_id="GS", position=(400.0, 0.0)))

    # Initial route works
    route = ctrl.register_active_route("stream-1", "D1", "GS")
    assert route == ["D1", "D2", "GS"]

    # D2 fails
    ctrl.isolate_drone("D2", reason=FailureReason.HARDWARE_FAULT)

    # Now no route exists from D1 to GS
    active_path = ctrl.get_active_route("stream-1")
    assert active_path == []

    with pytest.raises(NoRouteFoundError):
        ctrl.get_route("D1", "GS", raise_on_failure=True)

    route_none = ctrl.get_route("D1", "GS", raise_on_failure=False)
    assert route_none is None


def test_10_previously_failed_drone_recovers(
    linear_swarm_controller: SelfHealingController,
) -> None:
    """Scenario 10: Failed drone recovers progressively: ISOLATED -> RECOVERING -> HEALTHY."""
    ctrl = linear_swarm_controller
    now = time.time()

    # Isolate D4
    ctrl.isolate_drone("D4", reason=FailureReason.LOW_BATTERY)
    st = ctrl.get_drone_state("D4")
    assert st is not None
    assert st.operational_status == DroneHealthStatus.ISOLATED

    # 1st healthy update: should transition to RECOVERING
    ctrl.update_drone_health(
        {"drone_id": "D4", "battery": 90.0, "signal_strength": -50.0, "packet_loss": 0.0, "latency": 20.0, "timestamp": now},
        current_time=now,
    )
    assert st.operational_status == DroneHealthStatus.RECOVERING

    # 2nd healthy update: satisfies consecutive_checks_to_recover=2 -> HEALTHY
    ctrl.update_drone_health(
        {"drone_id": "D4", "battery": 90.0, "signal_strength": -50.0, "packet_loss": 0.0, "latency": 20.0, "timestamp": now + 0.2},
        current_time=now + 0.2,
    )
    assert st.operational_status == DroneHealthStatus.HEALTHY
    assert "D4" not in ctrl.get_isolated_and_failed_drones()


def test_11_compromised_drone_not_automatically_reintroduced(
    linear_swarm_controller: SelfHealingController,
) -> None:
    """Scenario 11: Compromised drone cannot recover merely because telemetry is good."""
    ctrl = linear_swarm_controller
    now = time.time()

    # Isolate D4 due to security compromise
    ctrl.isolate_drone("D4", reason=FailureReason.SECURITY_COMPROMISED)
    st = ctrl.get_drone_state("D4")
    assert st is not None
    assert st.is_compromised is True

    # Send perfect telemetry -> recovery must be rejected!
    ctrl.update_drone_health(
        {"drone_id": "D4", "battery": 100.0, "signal_strength": -40.0, "packet_loss": 0.0, "latency": 15.0, "threat_level": 0.0, "timestamp": now},
        current_time=now,
    )
    assert st.operational_status == DroneHealthStatus.ISOLATED

    # Explicit security clearance granted
    ctrl.grant_security_clearance("D4", clearance_token="DEMO-OPERATOR-CLEARANCE-REF")
    assert st.security_cleared is True
    assert st.is_compromised is False

    # Now recovery can proceed
    ctrl.update_drone_health(
        {"drone_id": "D4", "battery": 100.0, "signal_strength": -40.0, "packet_loss": 0.0, "latency": 15.0, "threat_level": 0.0, "timestamp": now + 0.1},
        current_time=now + 0.1,
    )
    assert st.operational_status == DroneHealthStatus.RECOVERING

    ctrl.update_drone_health(
        {"drone_id": "D4", "battery": 100.0, "signal_strength": -40.0, "packet_loss": 0.0, "latency": 15.0, "threat_level": 0.0, "timestamp": now + 0.2},
        current_time=now + 0.2,
    )
    assert st.operational_status == DroneHealthStatus.HEALTHY


def test_12_temporary_network_fluctuation_does_not_trigger_isolation(
    linear_swarm_controller: SelfHealingController,
) -> None:
    """Scenario 12: Single transient packet loss spike does not isolate drone."""
    ctrl = linear_swarm_controller
    now = time.time()
    st = ctrl.get_drone_state("D4")
    assert st is not None

    # Single spike (packet loss = 0.50, critical)
    ctrl.update_drone_health(
        {"drone_id": "D4", "packet_loss": 0.50, "timestamp": now},
        current_time=now,
    )
    # Should be degraded / pending confirmation, NOT isolated
    assert st.operational_status in (DroneHealthStatus.DEGRADED, DroneHealthStatus.HEALTHY)
    assert st.consecutive_failures == 1

    # Next check normalizes (packet loss = 0.01)
    ctrl.update_drone_health(
        {"drone_id": "D4", "packet_loss": 0.01, "timestamp": now + 0.5},
        current_time=now + 0.5,
    )
    assert st.consecutive_failures == 0
    assert st.operational_status == DroneHealthStatus.HEALTHY


def test_13_multiple_drones_fail_simultaneously(
    linear_swarm_controller: SelfHealingController,
) -> None:
    """Scenario 13: Multiple drones fail simultaneously."""
    ctrl = linear_swarm_controller

    # D4 and D5 fail at once
    ctrl.isolate_drone("D4", reason=FailureReason.LOW_BATTERY)
    ctrl.isolate_drone("D5", reason=FailureReason.HARDWARE_FAULT)

    isolated = ctrl.get_isolated_and_failed_drones()
    assert "D4" in isolated
    assert "D5" in isolated

    # Route from D7 to GS is still possible via direct or remaining mesh if connected
    # In our fixture: D7 connected to D4 (failed) and D5 (failed).
    # Since both neighbors of D7 failed, D7 has no alternative path.
    with pytest.raises(NoRouteFoundError):
        ctrl.get_route("D7", "GS")


def test_14_ground_station_loses_connectivity(fast_config: SelfHealingConfig) -> None:
    """Scenario 14: Ground Station is marked isolated or unreachable."""
    ctrl = SelfHealingController(config=fast_config)
    ctrl.register_drone("D1", DroneState(drone_id="D1", position=(0.0, 0.0)))
    ctrl.register_drone("GS", DroneState(drone_id="GS", position=(200.0, 0.0)))

    # GS is isolated due to physical damage or jamming
    ctrl.isolate_drone("GS", reason=FailureReason.DISCONNECTION)

    with pytest.raises(NoRouteFoundError):
        ctrl.get_route("D1", "GS")


def test_15_controller_handles_repeated_health_updates(
    linear_swarm_controller: SelfHealingController,
) -> None:
    """Scenario 15: Controller stably handles continuous repeated telemetry streams."""
    ctrl = linear_swarm_controller
    now = time.time()

    for i in range(20):
        t = now + i * 0.1
        res = ctrl.update_drone_health(
            {
                "drone_id": "D7",
                "battery": 95.0 - i * 0.5,
                "signal_strength": -55.0,
                "packet_loss": 0.01,
                "latency": 25.0,
                "threat_level": 0.0,
                "timestamp": t,
            },
            current_time=t,
        )
        assert res.is_healthy

    st = ctrl.get_drone_state("D7")
    assert st is not None
    assert st.operational_status == DroneHealthStatus.HEALTHY
    assert st.battery == pytest.approx(85.5)


# ==============================================================================
# EDGE CASES & BOUNDARY CONDITIONS
# ==============================================================================

def test_invalid_drone_ids(controller: SelfHealingController) -> None:
    """Rejects empty, whitespace, or invalid drone IDs."""
    with pytest.raises(ValueError):
        controller.register_drone("")

    with pytest.raises(ValueError):
        controller.update_drone_health({"drone_id": ""})


def test_missing_and_malformed_telemetry(controller: SelfHealingController) -> None:
    """Handles missing telemetry fields with appropriate defaults or errors."""
    with pytest.raises(ValueError, match="Missing mandatory field 'drone_id'"):
        SimulatorAdapter.parse_telemetry({"battery": 90.0})

    # Minimal valid payload (only drone_id provided) should parse with defaults
    parsed = SimulatorAdapter.parse_telemetry({"drone_id": "D1"})
    assert parsed["drone_id"] == "D1"
    assert parsed["battery"] == 100.0
    assert parsed["signal_strength"] == -50.0


def test_invalid_health_values() -> None:
    """Rejects out-of-range metric values."""
    with pytest.raises(ValueError, match="Battery must be between"):
        st = DroneState(drone_id="D1", battery=120.0)
        st.validate()

    with pytest.raises(ValueError, match="Battery must be between"):
        st = DroneState(drone_id="D1", battery=-5.0)
        st.validate()

    with pytest.raises(ValueError, match="Packet loss must be between"):
        st = DroneState(drone_id="D1", packet_loss=1.5)
        st.validate()

    with pytest.raises(ValueError, match="Latency must be non-negative"):
        st = DroneState(drone_id="D1", latency=-10.0)
        st.validate()

    with pytest.raises(ValueError, match="Threat level must be between"):
        st = DroneState(drone_id="D1", threat_level=2.5)
        st.validate()


def test_duplicate_event_deduplication(controller: SelfHealingController) -> None:
    """Suppresses duplicate events logged within 100ms."""
    now = time.time()
    evt1 = SelfHealingEvent(
        event_type=EventType.STATUS_CHANGED,
        drone_id="D1",
        timestamp=now,
        details={"info": "test"},
    )
    evt2 = SelfHealingEvent(
        event_type=EventType.STATUS_CHANGED,
        drone_id="D1",
        timestamp=now + 0.02,
        details={"info": "test"},
    )

    controller._record_event(evt1)
    controller._record_event(evt2)

    log = controller.get_event_log(drone_id="D1")
    assert len(log) == 1


def test_recovery_abort_on_subsequent_failure(
    linear_swarm_controller: SelfHealingController,
) -> None:
    """Drone in RECOVERING that fails a check returns to ISOLATED."""
    ctrl = linear_swarm_controller
    now = time.time()

    ctrl.isolate_drone("D4", reason=FailureReason.LOW_BATTERY)
    st = ctrl.get_drone_state("D4")
    assert st is not None

    # Step 1: Start recovery
    ctrl.update_drone_health(
        {"drone_id": "D4", "battery": 90.0, "signal_strength": -50.0, "packet_loss": 0.0, "latency": 20.0, "timestamp": now},
        current_time=now,
    )
    assert st.operational_status == DroneHealthStatus.RECOVERING

    # Step 2: Sudden failure (e.g. signal drops to -95 dBm)
    ctrl.update_drone_health(
        {"drone_id": "D4", "battery": 90.0, "signal_strength": -95.0, "packet_loss": 0.0, "latency": 20.0, "timestamp": now + 0.1},
        current_time=now + 0.1,
    )
    assert st.operational_status == DroneHealthStatus.ISOLATED


def test_route_auto_restoration_after_recovery(
    linear_swarm_controller: SelfHealingController,
) -> None:
    """Original optimal route is restored after recovered drone rejoins the swarm."""
    ctrl = linear_swarm_controller
    now = time.time()

    # Active route initially: D7 -> D4 -> D3 -> GS
    route = ctrl.register_active_route("stream-a", "D7", "GS")
    assert route == ["D7", "D4", "D3", "GS"]

    # D4 fails and active route detours to D7 -> D5 -> D6 -> GS
    ctrl.isolate_drone("D4", reason=FailureReason.LOW_BATTERY)
    assert ctrl.get_active_route("stream-a") == ["D7", "D5", "D6", "GS"]

    # Force recover D4
    success, msg = ctrl.recover_drone("D4", force=True, current_time=now)
    assert success is True

    # With auto_restore_routes=True, active route should restore back to D7 -> D4 -> D3 -> GS
    restored = ctrl.get_active_route("stream-a")
    assert restored == ["D7", "D4", "D3", "GS"]


def test_vision_alert_dispatch_over_self_healed_route(
    linear_swarm_controller: SelfHealingController,
) -> None:
    """Vision EmergencyAlert is encrypted, routed through relays without decryption, and unsealed at GS."""
    ctrl = linear_swarm_controller

    # Create dummy EmergencyAlert
    alert = EmergencyAlert(
        alert_id="ALERT-SWARM-01",
        drone_id="D7",
        timestamp=time.time(),
        priority="HIGH",
        persons=2,
        hazards={"fire": 1},
        location=(150.0, 75.0),
    )

    # Dispatch alert across swarm
    report = ctrl.dispatch_emergency_alert(alert, origin_drone_id="D7")
    assert report["success"] is True
    assert report["alert_id"] == "ALERT-SWARM-01"
    assert report["route"] == ["D7", "D4", "D3", "GS"]
    assert report["hops_traversed"] == ["D7", "D4", "D3", "GS"]

    # Now isolate D4, re-dispatch alert: should automatically use detoured path
    ctrl.isolate_drone("D4", reason=FailureReason.LOW_BATTERY)
    report2 = ctrl.dispatch_emergency_alert(alert, origin_drone_id="D7")
    assert report2["success"] is True
    assert "D4" not in report2["route"]
    assert report2["route"] in (["D7", "D5", "D6", "GS"], ["D7", "D5", "D3", "GS"])


def test_relays_do_not_decrypt_payload() -> None:
    """Verifies SecurityAdapter forwards raw encrypted payload without modifying or decrypting."""
    sec = SecurityAdapter()
    sample_payload = {"alert_id": "TEST-123", "secret": "confidential"}

    encrypted = sec.seal_payload("D7", "GS", sample_payload)
    assert isinstance(encrypted, bytes)

    # Relay hop forwarding
    relayed_bytes = sec.forward_relay_hop(encrypted, "D4", "D3")
    # Must be byte-for-byte identical
    assert relayed_bytes == encrypted

    # Only destination "GS" can unseal
    decrypted = sec.unseal_payload("GS", relayed_bytes)
    assert decrypted["alert_id"] == "TEST-123"
    assert decrypted["secret"] == "confidential"


def test_swarm_status_summary(linear_swarm_controller: SelfHealingController) -> None:
    """Verifies get_swarm_status() accurately reflects swarm health overview."""
    ctrl = linear_swarm_controller
    status = ctrl.get_swarm_status()

    assert status["total_drones"] == 6
    assert status["status_counts"]["HEALTHY"] == 6
    assert status["isolated_drones"] == []

    ctrl.isolate_drone("D4", reason=FailureReason.LOW_BATTERY)
    ctrl.isolate_drone("D5", reason=FailureReason.WEAK_SIGNAL)

    updated_status = ctrl.get_swarm_status()
    assert updated_status["status_counts"]["ISOLATED"] == 2
    assert set(updated_status["isolated_drones"]) == {"D4", "D5"}


def test_degraded_drone_higher_routing_cost(fast_config: SelfHealingConfig) -> None:
    """A degraded drone is penalized with a higher edge weight, so Dijkstra prefers a healthy detour."""
    ctrl = SelfHealingController(config=fast_config)

    # Topology: D1 can reach GS via D2 (direct) or D3 (slight detour)
    # D1(0,0), D2(200,0), D3(200, 50), GS(400,0)
    ctrl.register_drone("D1", DroneState(drone_id="D1", position=(0.0, 0.0)))
    ctrl.register_drone("D2", DroneState(drone_id="D2", position=(200.0, 0.0), battery=100.0))
    ctrl.register_drone("D3", DroneState(drone_id="D3", position=(200.0, 50.0), battery=100.0))
    ctrl.register_drone("GS", DroneState(drone_id="GS", position=(400.0, 0.0)))

    # Both healthy: D2 is slightly shorter
    route1 = ctrl.get_route("D1", "GS")
    assert route1 == ["D1", "D2", "GS"]

    # Now degrade D2 (e.g. low battery 20%, high latency 150ms)
    st_d2 = ctrl.get_drone_state("D2")
    assert st_d2 is not None
    st_d2.battery = 20.0
    st_d2.latency = 150.0
    st_d2.operational_status = DroneHealthStatus.DEGRADED

    # Dijkstra should now prefer the healthier D3 path
    route2 = ctrl.get_route("D1", "GS")
    assert route2 == ["D1", "D3", "GS"]


def test_strict_drone_id_validation() -> None:
    """Verifies strict drone ID validation enforces D1-D20 and GS naming scheme."""
    strict_cfg = SelfHealingConfig(strict_drone_id_validation=True)
    ctrl = SelfHealingController(config=strict_cfg)

    # Valid IDs
    assert strict_cfg.is_valid_drone_id("D1")
    assert strict_cfg.is_valid_drone_id("D7")
    assert strict_cfg.is_valid_drone_id("D20")
    assert strict_cfg.is_valid_drone_id("GS")

    # Invalid IDs
    assert not strict_cfg.is_valid_drone_id("D21")
    assert not strict_cfg.is_valid_drone_id("D0")
    assert not strict_cfg.is_valid_drone_id("DRONE-1")
    assert not strict_cfg.is_valid_drone_id("UNKNOWN")

    with pytest.raises(ValueError):
        ctrl.register_drone("D25")


def test_event_log_filtering_and_limits(controller: SelfHealingController) -> None:
    """Verifies event log can be filtered by drone_id, event_type, and limit."""
    ctrl = controller
    now = time.time()

    ctrl.isolate_drone("D1", reason=FailureReason.LOW_BATTERY, current_time=now)
    ctrl.isolate_drone("D2", reason=FailureReason.WEAK_SIGNAL, current_time=now + 0.2)
    ctrl.isolate_drone("D3", reason=FailureReason.DISCONNECTION, current_time=now + 0.4)

    # Filter by drone
    d1_events = ctrl.get_event_log(drone_id="D1")
    assert len(d1_events) == 1
    assert d1_events[0].drone_id == "D1"

    # Filter by type
    iso_events = ctrl.get_event_log(event_type=EventType.DRONE_ISOLATED)
    assert len(iso_events) == 3

    # Limit
    limited = ctrl.get_event_log(limit=2)
    assert len(limited) == 2
    assert limited[-1].drone_id == "D3"


def test_drone_state_serialization_round_trip() -> None:
    """Verifies DroneState to_dict and from_dict preserve all properties."""
    original = DroneState(
        drone_id="D12",
        position=(123.4, 567.8),
        altitude=45.0,
        heading=180.0,
        battery=88.5,
        signal_strength=-62.0,
        packet_loss=0.02,
        latency=35.0,
        threat_level=0.15,
        operational_status=DroneHealthStatus.DEGRADED,
        failure_reasons=[FailureReason.WEAK_SIGNAL],
        previous_status=DroneHealthStatus.HEALTHY,
        isolation_timestamp=1000.0,
    )

    d = original.to_dict()
    assert d["drone_id"] == "D12"
    assert d["operational_status"] == "DEGRADED"
    assert d["failure_reasons"] == ["WEAK_SIGNAL"]
    assert d["previous_status"] == "HEALTHY"

    restored = DroneState.from_dict(d)
    assert restored.drone_id == original.drone_id
    assert restored.position == original.position
    assert restored.operational_status == DroneHealthStatus.DEGRADED
    assert restored.failure_reasons == [FailureReason.WEAK_SIGNAL]
    assert restored.previous_status == DroneHealthStatus.HEALTHY


def test_regression_isolating_failed_relay_does_not_exclude_healthy_source(
    linear_swarm_controller: SelfHealingController,
) -> None:
    """Regression test: isolating failed relay D4 must not add healthy source D7 to the excluded set."""
    ctrl = linear_swarm_controller
    now = time.time()

    # D7 is source; establish active route D7 -> D4 -> D3 -> GS
    route = ctrl.register_active_route("alert-stream", "D7", "GS")
    assert route == ["D7", "D4", "D3", "GS"]

    # Healthy drones emit keepalive heartbeats at current time
    ctrl.heartbeat_active_drones(current_time=now)

    # Relay D4 specifically goes silent and exceeds heartbeat timeout
    st_d4 = ctrl.get_drone_state("D4")
    assert st_d4 is not None
    st_d4.last_seen = now - 10.0

    # Heartbeat check isolates D4 after consecutive_failures_to_isolate checks
    for _ in range(ctrl.config.consecutive_failures_to_isolate):
        ctrl.check_heartbeats(current_time=now)

    # Verify D4 is isolated, but source D7 and destination GS are NOT excluded
    excluded = ctrl.get_isolated_and_failed_drones()
    assert "D4" in excluded
    assert "D7" not in excluded
    assert "GS" not in excluded

    st_d7 = ctrl.get_drone_state("D7")
    assert st_d7 is not None
    assert st_d7.operational_status == DroneHealthStatus.HEALTHY

    # Verify alternative route D7 -> D5 -> D6 -> GS remains available
    alt_route = ctrl.get_route("D7", "GS")
    assert alt_route is not None
    assert "D4" not in alt_route
    assert alt_route in (["D7", "D5", "D6", "GS"], ["D7", "D5", "D3", "GS"])

    # Verify alert dispatch from D7 succeeds
    alert = EmergencyAlert(
        alert_id="ALERT-REGRESSION-01",
        drone_id="D7",
        timestamp=now,
        priority="HIGH",
        persons=1,
        hazards={},
        location=(0.0, 0.0),
    )
    res = ctrl.dispatch_emergency_alert(alert, origin_drone_id="D7")
    assert res["success"] is True
    assert "D4" not in res["route"]


def test_ground_station_never_isolated_by_heartbeat_timeout(fast_config: SelfHealingConfig) -> None:
    """Ground Station is infrastructure and must never be isolated by drone flight heartbeat checks."""
    ctrl = SelfHealingController(config=fast_config)
    ctrl.register_drone("D1", DroneState(drone_id="D1"))
    ctrl.register_drone("GS", DroneState(drone_id="GS"))

    now = time.time()
    # Simulate significant time passing with no GS heartbeats
    for _ in range(5):
        ctrl.check_heartbeats(current_time=now + 20.0)

    st_gs = ctrl.get_drone_state("GS")
    assert st_gs is not None
    assert st_gs.operational_status == DroneHealthStatus.HEALTHY
    assert "GS" not in ctrl.get_isolated_and_failed_drones()


