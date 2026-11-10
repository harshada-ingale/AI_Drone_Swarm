# AI Drone Swarm Command Center Dashboard

The **AI Drone Swarm Command Center** is a real-time tactical situational awareness dashboard built with Streamlit and Python 3.12. It provides live visualization and interactive control over autonomous swarm mesh topology, dynamic self-healing rerouting, adversarial threat quarantine, post-quantum cryptographic security, and computer vision disaster detection.

---

## 1. Architectural Overview

The dashboard follows a strict decoupled, testable architecture:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend (app.py)                     │
│  - 8 Interactive Tabs & Visualizers (SVG / Matplotlib)                 │
│  - Top Operational Status Strip (Truthful PQC Banner, SIMULATED badge) │
│  - Operator Sidebar Controls (Clock, Attacks, Clearance, Vision)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Calls plain methods / queries
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Backend Service (backend.py)                    │
│                        class SwarmService                              │
│  - Owned Simulation Clock (sim_time, step(dt), pause, resume)          │
│  - SelfHealingController Orchestration (Dijkstra, Health, Recovery)    │
│  - Cryptographic Identity & Operator Clearance (PkiRegistry, Ed25519)  │
│  - Computer Vision & Emergency Alert Dispatch (ONNX / Scripted)        │
│  - Bounded Metric History (Last 100 Sim Steps)                         │
│  - Sanitized State Snapshots (Zero sensitive keys/tokens/signatures)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
┌──────────────┐             ┌──────────────┐             ┌──────────────┐
│  controller/ │             │   security/  │             │   vision/    │
│ Self-Healing │             │ Post-Quantum │             │ Detection &  │
│ Routing Mesh │             │ Cryptography │             │ Geolocation  │
└──────────────┘             └──────────────┘             └──────────────┘
```

### Key Architectural Invariants:
1. **Decoupled Backend (`dashboard/backend.py`)**: `SwarmService` owns the simulation and exposes pure Python methods. It **never imports Streamlit**, ensuring 100% standalone testability.
2. **Deterministic Simulation Clock**: Simulation time (`sim_time`) advances explicitly via `step(dt)`. It does not rely on system wall-clock (`time.time()`), preventing false heartbeat timeouts when the dashboard is paused or stepped manually.
3. **Strict Import Block on PyTorch and Ultralytics**: Zero top-level or dashboard imports of PyTorch or ultralytics. Vision detection strictly uses ONNX models (`vision.detector_factory.load_detector`) or scripted telemetry.
4. **Honest Operational Reporting**:
   - **Simulated Data Badge**: Prominently marked as `[SIMULATED DATA]`.
   - **Post-Quantum Status**: Truthfully shows `ML-KEM ACTIVE` only when `liboqs` is available; otherwise displays `CLASSICAL FALLBACK (not post-quantum)`.
   - **Hazards Classification**: Reports `Not supported (null, not verified clear)` when hazard classification is unavailable, never fabricating a "clear" status.
   - **Non-Implemented Capabilities**: Features without backends (e.g. mobility simulator, AI threat detector) are either disabled or explicitly report `"not implemented yet"`.

---

## 2. Command Center Panels

The main workspace is structured into 8 comprehensive operational panels:

### 1. 🗺️ Network Map
- **Interactive SVG Engine (Default)**: Ultra-fast vector rendering with glowing active route lines, status indicators, coordinates, and hover telemetry tooltips.
- **Matplotlib Fallback**: High-fidelity plotted figures for publication or static exports.
- **Mesh Connectivity**: Renders all inter-drone radio channels within communication range ($\le 250\text{ m}$).
- **Active Route Highlighting**: Highlights the current minimum-latency route from Edge Scout `D7` to Ground Station `GS` in bright cyan (`#00e5ff`).

### 2. 📋 Drone Status Table
- Real-time tabular display of all 8 nodes (`D1`–`D7` and `GS`).
- Displays Battery (%), Signal Strength (dBm), Packet Loss (%), Latency (ms), Threat Level (0.0–1.0), Operational Status (`HEALTHY`, `DEGRADED`, `ISOLATED`, `RECOVERING`), Compromise flag, Clearance status, and Last Seen Age.

### 3. 🛡️ Threat & Quarantine Panel
- Tracks compromised nodes and isolated drones.
- Displays recorded failure reasons (`DISCONNECTION`, `SECURITY_COMPROMISED`, `WEAK_SIGNAL`, `EXCESSIVE_LATENCY`, `EXCESSIVE_PACKET_LOSS`, `HARDWARE_FAULT`).
- Enables single-click cryptographic operator clearance issuance using the demo security authority identity (`OP-COMMAND-01`).

### 4. 🩹 Self-Healing & Rerouting Panel
- Visualizes the active route path versus the primary optimal route (`D7 ──► D4 ──► D3 ──► GS`).
- Indicates routing state: `OPTIMAL`, `DETOUR / REROUTED`, `RESTORED`, or `NO ROUTE`.
- Displays progressive recovery health verification progress bars (consecutive healthy checks passed vs required).

### 5. 👁️ Disaster Detection & Geolocation Panel
- Displays latest emergency alerts delivered and decrypted at the Ground Station.
- **Per-Person Real-World Geolocation**:
  - Global Coordinates: WGS84 Latitude & Longitude with uncertainty radius ($\pm\text{m}$).
  - Local Metric Coordinates: Local East ($x$) and Local North ($y$) relative to origin.
- **Drone Telemetry Pose**: Latitude, Longitude, Altitude AGL/MSL, Heading, Pitch, Roll, Gimbal Pitch, and GPS Fix Type.
- **Evidence Thumbnails**: Displays cropped, letterboxed JPEG thumbnails of detected people extracted directly from the video stream.

### 6. 🔐 Post-Quantum Security Panel
- Validates the hybrid post-quantum state across the swarm:
  - Key Encapsulation: `ML-KEM-768` (or classical `X25519` fallback).
  - Digital Signatures: `ML-DSA-65` (or classical `Ed25519` fallback).
  - Wire Encryption: `AES-256-GCM` authenticated symmetric cipher.
  - Replay Defense: 64-packet sliding anti-replay window.
- Confirms end-to-end unsealing at Ground Station without intermediate relay decryption.

### 7. 📈 Telemetry Charts
- Responsive line charts tracking the last 100 simulation steps:
  - Swarm Average Packet Loss (%)
  - Swarm Average Latency (ms)
  - Battery Degradation (%)
  - Node Health Distributions (`HEALTHY`, `DEGRADED`, `ISOLATED`, `RECOVERING`)

### 8. 📜 Audit Event Log
- Searchable and filterable audit trail of all controller state transitions, heartbeats, reroutes, and security events.
- **Zero Token Leakage**: All cryptographic nonces are truncated, and raw signatures, private keys, and tokens are completely stripped from snapshots.

---

## 3. Sidebar Controls & Mission Actions

The sidebar provides direct operator controls:

1. **Simulation Clock**:
   - `Start` / `Pause` / `Resume`: Controls the automatic stepping loop.
   - `Step +1s` / `Step +5s`: Advances the simulation clock by 1 or 5 seconds.
   - `Reset`: Reinitializes the swarm mesh and PKI registry.
2. **Node Operations**:
   - `Fail Node`: Simulates unexpected hardware failure and triggers dynamic rerouting.
   - `Recover Node`: Initiates multi-step progressive health checks for isolated nodes.
3. **Adverse Condition & Attack Injection**:
   - **RF Jamming**: Degrades signal strength to $-94\text{ dBm}$ and elevates packet loss to $48\%$.
   - **DoS Flood**: Inundates node with high latency ($450\text{ ms}$) and $55\%$ packet loss.
   - **GPS Spoofing**: Simulates coordinate drift anomaly, elevating threat level to $0.92$ and triggering immediate isolation.
4. **Security Clearance**:
   - `Grant Signed Clearance`: Signs an authentic clearance credential using the operator's private key (`OP-COMMAND-01`) and registers it with the controller.
5. **Mission & Vision Commands**:
   - `Run Vision & Alert`: Executes detection on scripted data or video clips (`test.mp4`) using the ONNX detector (`vision/models/yolov8n.onnx`).
   - `Dispatch Alert Direct`: Dispatches a sample emergency alert along the active route.
   - `Run AI Threat Detector`: Honest placeholder informing that the model is not implemented yet.

---

## 4. Running the Dashboard

### Using the PowerShell Launcher (Windows):
```powershell
.\run_dashboard.ps1
```
Optional flags:
```powershell
# Custom port and headless mode
.\run_dashboard.ps1 -Port 8502 -Headless
```

### Using Streamlit Directly:
```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.port 8501
```

---

## 5. Automated Verification & Tests

The dashboard is backed by comprehensive automated test suites:
- **`tests/test_dashboard_backend.py`**:
  - Validates `SwarmService` clock stepping, bounds enforcement, node failures, Dijkstra rerouting, progressive recovery, attack injection (jamming, DoS, spoofing), cryptographic clearance, and alert dispatch.
- **`tests/test_dashboard_app.py`**:
  - Validates syntax compilation, AST static constraints (ensuring zero PyTorch/ultralytics imports), truthful reporting labels, and headless smoke rendering of all 8 tabs.

Run the test suite:
```powershell
.\venv\Scripts\python.exe -m pytest tests/test_dashboard_backend.py tests/test_dashboard_app.py -v
```
Run the entire project test suite:
```powershell
.\venv\Scripts\python.exe -m pytest tests -q
```
*(Result: 120 passed in ~5 seconds)*

