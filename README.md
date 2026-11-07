# AI_Drone_Swarm

Collaborative AI-enabled autonomous drone swarm system for disaster detection, situational awareness, and mesh communications.

## Modules

- **`controller/`**: Central Self-Healing Swarm Controller managing autonomous health monitoring, failure detection, quarantine isolation, dynamic rerouting, and progressive recovery.
- **`routing/`**: Mesh communication routing interface and weighted Dijkstra engine calculating link weights based on battery, signal strength, packet loss, latency, and threat level.
- **`vision/`**: Edge vision processing, YOLO object and victim detection, tracking, thermal confirmation, and emergency disaster alerting.
- **`security/`**: Cryptographic messaging and post-quantum security framework.
- **`simulator/`**: Multi-drone telemetry and network link simulation.
- **`ai/`**: Swarm intelligence and threat anomaly detection.
- **`dashboard/`**: Real-time swarm monitoring and command dashboard.

## Self-Healing Documentation

Detailed architecture, failure thresholds, recovery state machine, and usage guides are documented in [docs/SELF_HEALING.md](docs/SELF_HEALING.md).

## Running Tests

Execute automated unit and integration tests using `pytest`:

```bash
pytest
```
