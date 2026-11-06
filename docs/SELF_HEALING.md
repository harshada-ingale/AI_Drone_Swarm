# Self-Healing Swarm Controller Architecture & Guide

This document details the design, architecture, and operation of the **Self-Healing Mechanism** implemented for the AI Drone Swarm system (`AI_Drone_Swarm`).

---

## 1. Architecture Overview

The Self-Healing subsystem ensures autonomous operational continuity and mesh connectivity across drone fleets (`D1` through `D20`) communicating with the Ground Station (`GS`).

The system implements a continuous lifecycle:

```
MONITOR ───► DETECT ───► ISOLATE ───► REROUTE ───► RECOVER ───► VERIFY
   ▲                                                              │
   └──────────────────────────────────────────────────────────────┘
```

### Module Components

- **`controller/models.py`**:
  - `DroneHealthStatus`: `HEALTHY`, `DEGRADED`, `FAILED`, `ISOLATED`, `RECOVERING`.
  - `FailureReason`: Specific diagnosed faults (`LOW_BATTERY`, `WEAK_SIGNAL`, `EXCESSIVE_PACKET_LOSS`, `EXCESSIVE_LATENCY`, `SECURITY_COMPROMISED`, `DISCONNECTION`, etc.).
  - `DroneState`: Comprehensive health, telemetry (position, altitude, heading), and networking state with range validation.
  - `SelfHealingEvent`: Audit trail records capturing all transitions, reroutes, and alerts.
  - `ThreatReport`: Security anomaly reports from AI/security subsystems.

- **`controller/config.py`**:
  - `SelfHealingConfig`: Fully configurable operational thresholds for battery, signal (dBm), packet loss, latency (ms), threat score, heartbeat timeouts, and confirmation counts.

- **`controller/detector.py`**:
  - `FailureDetector`: Evaluates `DroneState` metrics against thresholds. Differentiates between healthy, degraded, and critical states. Implements recovery eligibility checks.

- **`controller/recovery.py`**:
  - `RecoveryManager`: Manages the multi-step verification lifecycle for isolated drones rejoining the swarm (`ISOLATED` -> `RECOVERING` -> `HEALTHY`). Requires explicit security clearance for compromised drones.

- **`controller/adapters.py`**:
  - `SimulatorAdapter`: Sanitizes telemetry dictionaries or stream frames from the flight simulator.
  - `AIThreatAdapter`: Ingests and normalizes anomaly detection events.
  - `SecurityAdapter`: Coordinates end-to-end payload encryption and guarantees relay drones forward ciphertext without decrypting.
  - `VisionAlertAdapter`: Dispatches `EmergencyAlert` instances over self-healed multi-hop paths to `GS`.

- **`controller/self_healing_controller.py`**:
  - `SelfHealingController`: The central coordinator managing health updates, heartbeat monitoring, quarantine isolation, dynamic path rerouting, route restoration, and audit logging.

- **`routing/`**:
  - `routing/router.py`: `RoutingInterface`, `RouteResult`, `NoRouteFoundError`.
  - `routing/dijkstra.py`: `DijkstraRouter` implementing weighted Dijkstra search based on distance, battery level, signal quality, packet loss, latency, and threat score.

---

## 2. Failure Detection Mechanism

The failure detector evaluates each incoming telemetry frame or status update:

| Condition | Degraded Threshold | Critical Failure Threshold | Recovery Threshold |
| :--- | :--- | :--- | :--- |
| **Battery** | $\le 30.0\%$ | $\le 15.0\%$ | $\ge 35.0\%$ |
| **Signal Strength** | $\le -75.0\text{ dBm}$ | $\le -88.0\text{ dBm}$ | $\ge -70.0\text{ dBm}$ |
| **Packet Loss** | $\ge 10.0\%$ | $\ge 30.0\%$ | $\le 5.0\%$ |
| **Latency** | $\ge 120.0\text{ ms}$ | $\ge 300.0\text{ ms}$ | $\le 80.0\text{ ms}$ |
| **Threat Score** | $\ge 0.40$ | $\ge 0.70$ | $< 0.40$ |
| **Heartbeat Timeout**| N/A | $> 5.0\text{ s}$ | Valid Telemetry |

### Fluctuation Dampening
To avoid unnecessary churn from transient RF fluctuations:
- Critical metric breaches increment `consecutive_failures`.
- A drone is transitioned to `DEGRADED` while failures are being confirmed.
- Isolation is triggered only when `consecutive_failures >= consecutive_failures_to_isolate` (default: 3).
- **Exception**: Drones marked with a security threat or malicious compromise are isolated immediately (`immediate_isolation_on_compromise = True`).

---

## 3. Drone Isolation

When a drone is identified as failed or compromised:
1. Operational status transitions to `ISOLATED`.
2. Previous operational status and isolation timestamp are archived.
3. The drone ID is immediately added to the exclusion set for new route computations.
4. Active communication streams traversing this drone are identified and dynamically rerouted.
5. A structured `DRONE_ISOLATED` event is appended to the audit log.

---

## 4. Weighted Dijkstra Rerouting

Communication routes are formatted as an ordered list of node IDs:
```python
["D7", "D4", "D3", "GS"]
```

The `DijkstraRouter` calculates edge costs $w(u, v)$ for prospective hops based on a multi-metric objective:

$$w(u, v) = w_{\text{dist}} \cdot C_{\text{dist}} + w_{\text{bat}} \cdot C_{\text{bat}} + w_{\text{sig}} \cdot C_{\text{sig}} + w_{\text{loss}} \cdot C_{\text{loss}} + w_{\text{lat}} \cdot C_{\text{lat}} + w_{\text{threat}} \cdot C_{\text{threat}} + \text{hop\_penalty}$$

- **Excluded Nodes**: Drones in `ISOLATED`, `FAILED`, `RECOVERING`, or `compromised` states are strictly excluded.
- **Detour Computation**: If node `D4` fails on active route `["D7", "D4", "D3", "GS"]`, the controller automatically finds an alternative path through healthy drones (e.g. `["D7", "D5", "D6", "GS"]`).
- **Routing Failure**: If network partition prevents any path to `GS`, the controller emits a `ROUTING_FAILED` event and raises `NoRouteFoundError`.
- **Automatic Route Restoration**: When `auto_restore_routes = True` and a previously failed node recovers, the controller evaluates active detour routes and safely restores the original, lower-cost path.

---

## 5. Drone Recovery & Verification

When an isolated drone begins reporting healthy telemetry:
1. **Eligibility Check**: All metrics must satisfy recovery thresholds. Compromised drones **must** possess explicit security clearance (`controller.grant_security_clearance(drone_id)`).
2. **`RECOVERING` Phase**: The drone enters verification mode with `consecutive_healthy = 1`.
3. **Multi-Check Verification**: Requires `consecutive_checks_to_recover` consecutive healthy telemetry updates (default: 2 or 3).
4. **Reintegration**: Once verified, status transitions to `HEALTHY`, routing eligibility is restored, and a `DRONE_RECOVERED` event is logged.
5. **Abort Condition**: If any telemetry frame fails during recovery, the process aborts immediately and the drone is re-isolated.

---

## 6. Module Integration

### Simulator
Ingest telemetry frames using `controller.update_drone_health(telemetry_dict)`:
```python
controller.update_drone_health({
    "drone_id": "D7",
    "position": (100.0, 50.0),
    "battery": 85.0,
    "signal_strength": -55.0,
    "packet_loss": 0.01,
    "latency": 25.0,
})
```

### AI Threat Detection
Integrate anomaly reports via `controller.report_threat(threat_report)`:
```python
controller.report_threat({
    "drone_id": "D4",
    "threat_level": 0.95,
    "threat_type": "GPS_SPOOFING",
})
```

### Security & Vision Alerts
Disaster alerts created by `vision.emergency_alert.EmergencyAlert` are dispatched securely:
```python
controller.dispatch_emergency_alert(alert, origin_drone_id="D7")
```
- **Relay Invariant**: Intermediate drones along the path forward the encrypted ciphertext without decrypting.
- Decryption and verification occur exclusively at the Ground Station (`GS`).

---

## 7. Running the Tests

To run the complete test suite including all 28 self-healing tests and existing vision tests:

```bash
python -m pytest tests/test_self_healing.py -v
python -m pytest
```

---

## 8. Usage Example with Simulated Swarm

```python
from controller import SelfHealingController, SelfHealingConfig, DroneState

# 1. Initialize controller
config = SelfHealingConfig(max_comm_range_m=250.0)
ctrl = SelfHealingController(config=config)

# 2. Register swarm drones
ctrl.register_drone("D7", DroneState(drone_id="D7", position=(0.0, 0.0)))
ctrl.register_drone("D4", DroneState(drone_id="D4", position=(200.0, 0.0)))
ctrl.register_drone("D3", DroneState(drone_id="D3", position=(400.0, 0.0)))
ctrl.register_drone("GS", DroneState(drone_id="GS", position=(600.0, 0.0)))

# 3. Query route
route = ctrl.get_route("D7", "GS")
print("Route:", route)  # ['D7', 'D4', 'D3', 'GS']

# 4. Simulate D4 failure
ctrl.update_drone_health({"drone_id": "D4", "battery": 10.0})
ctrl.update_drone_health({"drone_id": "D4", "battery": 10.0})
ctrl.update_drone_health({"drone_id": "D4", "battery": 10.0})

# 5. Check status
print(ctrl.get_swarm_status())
```

---

## 9. Interactive Demonstration Script

An interactive demonstration script is available in `run_self_healing_demo.py` showcasing all six operational phases (Monitor, Detect, Isolate, Dynamic Dijkstra Rerouting, Verification & Recovery, Security Clearance Enforcement):

```bash
# Standard paced demonstration
python run_self_healing_demo.py

# Interactive step-by-step mode (press Enter to advance)
python run_self_healing_demo.py --interactive

# Fast automated mode
python run_self_healing_demo.py --fast
```


