#!/usr/bin/env python3
"""Interactive demonstration of the AI Drone Swarm Self-Healing Mechanism.

Walks through MONITOR, DETECT, ISOLATE, REROUTE, RECOVER, and VERIFY phases:
1. Swarm topology initialization and primary route selection
2. Sudden drone disconnection and heartbeat failure detection
3. Dynamic weighted Dijkstra rerouting bypassing isolated drones
4. Safe progressive recovery with multi-step health verification
5. Route auto-restoration once optimal path becomes healthy again
6. Security compromise isolation and mandatory clearance enforcement
"""

import argparse
import sys
import time
from typing import Any

from controller.config import SelfHealingConfig
from controller.models import (
    DroneHealthStatus,
    DroneState,
    EventType,
    FailureReason,
    ThreatReport,
)
from controller.self_healing_controller import SelfHealingController
from vision.emergency_alert import EmergencyAlert


# ANSI formatting helpers for terminal display
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
DIM = "\033[2m"


def print_banner(title: str) -> None:
    """Prints a styled section header banner."""
    width = 78
    print()
    print(f"{CYAN}{'=' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title.center(width - 4)}{RESET}")
    print(f"{CYAN}{'=' * width}{RESET}")
    print()


def print_substep(step: str, title: str) -> None:
    """Prints a styled step indicator."""
    print(f"\n{BOLD}{YELLOW}[{step}] {title}{RESET}")
    print(f"{DIM}{'-' * 70}{RESET}")


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Prints a clean tabular ASCII output."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            # strip ansi codes for width calculation
            clean_cell = cell.replace(GREEN, "").replace(RED, "").replace(YELLOW, "").replace(CYAN, "").replace(RESET, "").replace(BOLD, "").replace(DIM, "")
            col_widths[idx] = max(col_widths[idx], len(clean_cell))

    # Header
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(f"{BOLD}{header_line}{RESET}")
    print(f"{DIM}{sep_line}{RESET}")

    # Rows
    for row in rows:
        row_str_parts = []
        for i, cell in enumerate(row):
            clean_cell = cell.replace(GREEN, "").replace(RED, "").replace(YELLOW, "").replace(CYAN, "").replace(RESET, "").replace(BOLD, "").replace(DIM, "")
            padding = " " * (col_widths[i] - len(clean_cell))
            row_str_parts.append(cell + padding)
        print(" | ".join(row_str_parts))


def format_status(status: DroneHealthStatus) -> str:
    """Colorizes operational status string."""
    if status == DroneHealthStatus.HEALTHY:
        return f"{GREEN}{status.value}{RESET}"
    if status == DroneHealthStatus.DEGRADED:
        return f"{YELLOW}{status.value}{RESET}"
    if status == DroneHealthStatus.ISOLATED:
        return f"{RED}{status.value}{RESET}"
    if status == DroneHealthStatus.RECOVERING:
        return f"{CYAN}{status.value}{RESET}"
    return f"{DIM}{status.value}{RESET}"


def display_swarm_table(ctrl: SelfHealingController) -> None:
    """Prints formatted table of all drones in the swarm."""
    headers = ["Drone ID", "Status", "Position (X, Y)", "Battery", "Signal", "Loss", "Latency", "Threat"]
    rows = []
    for d_id, st in sorted(ctrl.get_all_drones().items()):
        rows.append([
            f"{BOLD}{d_id}{RESET}",
            format_status(st.operational_status),
            f"({st.position[0]:.0f}, {st.position[1]:.0f})",
            f"{st.battery:.1f}%",
            f"{st.signal_strength:.0f} dBm",
            f"{st.packet_loss * 100:.1f}%",
            f"{st.latency:.0f} ms",
            f"{st.threat_level:.2f}",
        ])
    print_table(headers, rows)


def run_demo(step_delay: float = 0.5, interactive: bool = False) -> None:
    """Executes the full interactive self-healing demonstration."""
    def pause(msg: str = "") -> None:
        if interactive:
            print(f"\n{DIM}{msg or 'Press ENTER to proceed to the next step...'}{RESET}", end="")
            input()
        elif step_delay > 0:
            time.sleep(step_delay)

    print_banner("AI DRONE SWARM: SELF-HEALING CONTROLLER DEMONSTRATION")
    print(f"{DIM}Simulating autonomous drone mesh operations, failure isolation, and dynamic rerouting.{RESET}\n")

    # ==========================================================================
    # PHASE 1: CONFIGURATION & TOPOLOGY INITIALIZATION
    # ==========================================================================
    print_substep("PHASE 1", "Initializing Swarm Mesh Network & Ground Station")

    config = SelfHealingConfig(
        battery_critical=15.0,
        battery_degraded=30.0,
        battery_recovery=35.0,
        signal_critical_dbm=-88.0,
        signal_recovery_dbm=-70.0,
        packet_loss_critical=0.30,
        packet_loss_recovery=0.05,
        latency_critical_ms=300.0,
        latency_recovery_ms=80.0,
        threat_critical=0.70,
        heartbeat_timeout_s=3.0,
        consecutive_failures_to_isolate=2,
        consecutive_checks_to_recover=2,
        max_comm_range_m=250.0,
        ground_station_id="GS",
        auto_restore_routes=True,
    )
    ctrl = SelfHealingController(config=config)

    # 7 Drones + 1 Ground Station
    initial_drones = [
        ("D7", (0.0, 0.0), 98.0, -50.0, 0.0, 22.0, "Forward Edge Scout"),
        ("D4", (200.0, 0.0), 94.0, -52.0, 0.0, 25.0, "Primary Backbone Relay 1"),
        ("D3", (400.0, 0.0), 91.0, -55.0, 0.0, 24.0, "Primary Backbone Relay 2"),
        ("D5", (200.0, 120.0), 96.0, -53.0, 0.0, 26.0, "North Detour Relay 1"),
        ("D6", (400.0, 120.0), 95.0, -54.0, 0.0, 27.0, "North Detour Relay 2"),
        ("D1", (100.0, -100.0), 89.0, -60.0, 0.0, 28.0, "South Flank Patrol"),
        ("D2", (300.0, -100.0), 92.0, -58.0, 0.0, 25.0, "South Reserve Relay"),
        ("GS", (600.0, 0.0), 100.0, -40.0, 0.0, 10.0, "Ground Command Station"),
    ]

    for d_id, pos, bat, sig, loss, lat, role in initial_drones:
        ctrl.register_drone(
            d_id,
            DroneState(
                drone_id=d_id,
                position=pos,
                battery=bat,
                signal_strength=sig,
                packet_loss=loss,
                latency=lat,
                operational_status=DroneHealthStatus.HEALTHY,
            ),
        )

    print(f"Registered {len(initial_drones)} swarm nodes (7 active aerial drones + Ground Station GS).")
    print(f"Transmission communication range: {config.max_comm_range_m} metres.")
    print()
    display_swarm_table(ctrl)

    # Establish active communication route
    route_id = "surveillance-stream-01"
    initial_route = ctrl.register_active_route(route_id, "D7", "GS")
    print(f"\n{BOLD}Established Active Mission Route [{route_id}]:{RESET}")
    print(f"  {CYAN}{' ──► '.join(initial_route)}{RESET}")
    print(f"  Path: Drone D7 (Edge Scout) ──► D4 (Relay 1) ──► D3 (Relay 2) ──► Ground Station (GS)")

    # Send sample vision alert over this path
    test_alert = EmergencyAlert(
        alert_id="ALERT-DEMO-001",
        drone_id="D7",
        timestamp=time.time(),
        priority="HIGH",
        persons=2,
        hazards={"fire": 1},
        location=(50.0, 25.0),
    )
    dispatch_res = ctrl.dispatch_emergency_alert(test_alert, origin_drone_id="D7")
    print(f"\n{GREEN}✔ Vision EmergencyAlert '{test_alert.alert_id}' securely transmitted to Ground Station!{RESET}")
    print(f"  Hops traversed: {' -> '.join(dispatch_res['hops_traversed'])}")
    print(f"  Security note: Intermediate relays forwarded encrypted bytes without decryption.")

    pause()

    # ==========================================================================
    # PHASE 2: FAILURE DETECTION & DRONE ISOLATION
    # ==========================================================================
    print_banner("PHASE 2: SUDDEN FAILURE INGESTION & NODE ISOLATION")
    print(f"Simulating sudden complete radio/hardware failure on intermediate relay {BOLD}D4{RESET}...\n")

    sim_time = time.time()
    # Active healthy drones continue streaming their normal flight keepalive heartbeats
    ctrl.heartbeat_active_drones(current_time=sim_time)

    st_d4 = ctrl.get_drone_state("D4")
    assert st_d4 is not None

    # Step A: Drone D4 stops transmitting telemetry
    print(f"Telemetry from D4 has ceased. Elapsed time without heartbeat exceeds {config.heartbeat_timeout_s}s...")
    st_d4.last_seen = sim_time - 6.0  # 6 seconds in the past

    # Trigger heartbeat checks
    isolated_drones = ctrl.check_heartbeats(current_time=sim_time)
    # Consecutive check 1
    isolated_drones = ctrl.check_heartbeats(current_time=sim_time + 0.1)

    print(f"\n{RED}✖ Node Failure Diagnosed!{RESET}")
    print(f"  Node ID        : {BOLD}D4{RESET}")
    print(f"  Failure Reason : {FailureReason.DISCONNECTION.value}")
    print(f"  New Status     : {format_status(st_d4.operational_status)}")
    print(f"  Quarantine     : Added to routing exclusion set.")

    # Show latest events
    events = ctrl.get_event_log(limit=2)
    print(f"\n{BOLD}Controller Audit Log:{RESET}")
    for evt in events:
        print(f"  • {evt.summary()}")

    pause()

    # ==========================================================================
    # PHASE 3: AUTONOMOUS REROUTING (DIJKSTRA ALTERNATIVE PATH)
    # ==========================================================================
    print_banner("PHASE 3: AUTOMATIC WEIGHTED DIJKSTRA REROUTING")
    print("The controller automatically detects that active route 'surveillance-stream-01' traverses isolated node D4.")
    print("Initiating dynamic graph recalculation...\n")

    ctrl.heartbeat_active_drones(current_time=time.time())
    updated_route = ctrl.get_active_route(route_id)
    assert updated_route is not None

    print(f"{BOLD}Route Comparison:{RESET}")
    print(f"  Original Route : {DIM}{' ──► '.join(initial_route)}{RESET}  {RED}[FAILED - Node D4 Isolated]{RESET}")
    print(f"  Rerouted Route : {GREEN}{BOLD}{' ──► '.join(updated_route)}{RESET}  {GREEN}[ACTIVE - Healthy Bypass]{RESET}")
    print(f"\nSwarm mesh reconfigured path through healthy North Detour Relays (D5 and D6).")

    # Transmit second alert over the rerouted path
    test_alert_2 = EmergencyAlert(
        alert_id="ALERT-DEMO-002",
        drone_id="D7",
        timestamp=time.time(),
        priority="CRITICAL",
        persons=3,
        hazards={"fire": 1, "flood": 1},
        location=(60.0, 30.0),
    )
    dispatch_res_2 = ctrl.dispatch_emergency_alert(test_alert_2, origin_drone_id="D7")
    print(f"\n{GREEN}✔ Critical EmergencyAlert '{test_alert_2.alert_id}' dispatched across healed route!{RESET}")
    print(f"  Hops traversed: {' -> '.join(dispatch_res_2['hops_traversed'])}")

    print()
    display_swarm_table(ctrl)

    pause()

    # ==========================================================================
    # PHASE 4: PROGRESSIVE RECOVERY & ROUTE RESTORATION
    # ==========================================================================
    print_banner("PHASE 4: PROGRESSIVE RECOVERY & AUTOMATIC ROUTE RESTORATION")
    print(f"Drone {BOLD}D4{RESET} re-establishes power and reconnects to the network.")
    print("Initiating multi-check recovery verification...\n")

    rec_time = time.time()
    ctrl.heartbeat_active_drones(current_time=rec_time)

    # Step 1: 1st healthy check -> transitions to RECOVERING
    print(f"Step 4.1: First healthy telemetry frame received from D4...")
    ctrl.update_drone_health(
        {
            "drone_id": "D4",
            "battery": 92.0,
            "signal_strength": -51.0,
            "packet_loss": 0.0,
            "latency": 22.0,
            "threat_level": 0.0,
            "timestamp": rec_time,
        },
        current_time=rec_time,
    )
    print(f"  D4 Status: {format_status(st_d4.operational_status)} (Verification pass 1/{config.consecutive_checks_to_recover})")

    pause()

    # Step 2: 2nd healthy check -> completes recovery -> HEALTHY
    print(f"\nStep 4.2: Second consecutive healthy telemetry frame received from D4...")
    ctrl.update_drone_health(
        {
            "drone_id": "D4",
            "battery": 92.0,
            "signal_strength": -51.0,
            "packet_loss": 0.0,
            "latency": 22.0,
            "threat_level": 0.0,
            "timestamp": rec_time + 0.2,
        },
        current_time=rec_time + 0.2,
    )
    print(f"  D4 Status: {format_status(st_d4.operational_status)} (Verification criteria satisfied!)")
    print(f"{GREEN}✔ Drone D4 successfully reinstated into active mesh routing.{RESET}")

    # Check route restoration
    restored_route = ctrl.get_active_route(route_id)
    assert restored_route is not None
    print(f"\n{BOLD}Automatic Route Optimization & Restoration:{RESET}")
    print(f"  Current Active Route: {CYAN}{BOLD}{' ──► '.join(restored_route)}{RESET}")
    print(f"  Original optimal path {initial_route} safely restored due to lower link latency.")

    pause()

    # ==========================================================================
    # PHASE 5: SECURITY COMPROMISE & MANDATORY CLEARANCE
    # ==========================================================================
    print_banner("PHASE 5: SECURITY COMPROMISE & CLEARANCE ENFORCEMENT")
    print(f"Simulating AI threat detection: Drone {BOLD}D6{RESET} exhibits anomalous behavior (GPS spoofing)...\n")

    threat_time = time.time()
    ctrl.heartbeat_active_drones(current_time=threat_time)
    ctrl.report_threat(
        ThreatReport(
            drone_id="D6",
            threat_level=0.92,
            threat_type="GPS_SPOOFING_ANOMALY",
            timestamp=threat_time,
            details={"drift_m": 145.0, "confidence": 0.98},
        )
    )

    st_d6 = ctrl.get_drone_state("D6")
    assert st_d6 is not None
    print(f"{RED}✖ Threat Detected! Node D6 immediately quarantined.{RESET}")
    print(f"  Status          : {format_status(st_d6.operational_status)}")
    print(f"  Is Compromised  : {RED}{st_d6.is_compromised}{RESET}")
    print(f"  Failure Reason  : {st_d6.failure_reasons[-1].value}")

    # Adversary tries to report perfect battery and signal
    print(f"\nAdversary attempts to re-register D6 with spoofed healthy telemetry...")
    ctrl.update_drone_health(
        {
            "drone_id": "D6",
            "battery": 100.0,
            "signal_strength": -40.0,
            "packet_loss": 0.0,
            "latency": 15.0,
            "threat_level": 0.0,
            "timestamp": threat_time + 1.0,
        },
        current_time=threat_time + 1.0,
    )
    print(f"  Recovery attempt result: {RED}REJECTED{RESET}")
    print(f"  Current status: {format_status(st_d6.operational_status)}")
    print(f"  Security Policy: Compromised drones CANNOT recover without explicit operator clearance.")

    pause()

    # Operator grants security clearance
    print(f"\nSwarm Security Officer reviews logs and performs security clearance validation.")
    clearance_ref = "DEMO-OPERATOR-CLEARANCE-REF"
    print(f"Executing: controller.grant_security_clearance('D6', clearance_token='{clearance_ref}')...")
    ctrl.grant_security_clearance("D6", clearance_token=clearance_ref)

    # Now recovery proceeds
    print(f"\nResuming health verification for D6 following security clearance...")
    ctrl.update_drone_health(
        {"drone_id": "D6", "battery": 95.0, "signal_strength": -52.0, "packet_loss": 0.0, "latency": 25.0, "threat_level": 0.0, "timestamp": threat_time + 2.0},
        current_time=threat_time + 2.0,
    )
    print(f"  Pass 1: D6 Status -> {format_status(st_d6.operational_status)}")

    ctrl.update_drone_health(
        {"drone_id": "D6", "battery": 95.0, "signal_strength": -52.0, "packet_loss": 0.0, "latency": 25.0, "threat_level": 0.0, "timestamp": threat_time + 2.2},
        current_time=threat_time + 2.2,
    )
    print(f"  Pass 2: D6 Status -> {format_status(st_d6.operational_status)}")
    print(f"{GREEN}✔ Node D6 cleared and returned to HEALTHY.{RESET}")

    pause()

    # ==========================================================================
    # PHASE 6: FINAL SWARM STATUS & AUDIT SUMMARY
    # ==========================================================================
    print_banner("PHASE 6: MISSION SUMMARY & AUDIT TRAIL")
    ctrl.heartbeat_active_drones(current_time=time.time())
    summary = ctrl.get_swarm_status()

    print(f"{BOLD}Swarm Telemetry Overview:{RESET}")
    print(f"  Total Nodes Registered   : {summary['total_drones']}")
    print(f"  Healthy Drones           : {summary['status_counts']['HEALTHY']}")
    print(f"  Isolated Drones          : {summary['status_counts']['ISOLATED']}")
    print(f"  Active Monitored Routes  : {summary['active_routes_count']}")
    print(f"  Total Self-Healing Events: {summary['total_events']}")
    print()

    display_swarm_table(ctrl)

    print(f"\n{BOLD}Recent Self-Healing Events Recorded:{RESET}")
    recent_events = ctrl.get_event_log(limit=8)
    for evt in recent_events:
        print(f"  • {evt.summary()}")

    print(f"\n{GREEN}{BOLD}Demonstration completed successfully!{RESET}\n")


def main() -> None:
    """CLI entry point for self-healing demo."""
    parser = argparse.ArgumentParser(description="AI Drone Swarm Self-Healing Interactive Demo")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.4,
        help="Delay in seconds between demonstration steps (default: 0.4s).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Pause and require pressing ENTER between steps.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run without any delay for automated testing.",
    )
    args = parser.parse_args()

    delay = 0.0 if args.fast else args.delay
    run_demo(step_delay=delay, interactive=args.interactive)


if __name__ == "__main__":
    main()

