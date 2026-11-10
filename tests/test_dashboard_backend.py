"""Unit tests for Command Center Dashboard SwarmService backend."""

import json
from unittest.mock import patch
import pytest

from controller.models import DroneHealthStatus
from dashboard.backend import SwarmService
from dashboard.network_map import render_network_map_matplotlib, render_network_map_svg


def test_swarm_service_initialization():
    service = SwarmService(initial_sim_time=1000.0)
    snap = service.snapshot()

    assert snap["sim_time"] == 1000.0
    assert snap["running"] is False
    assert len(snap["drones"]) == 8
    assert "GS" in snap["drones"]
    assert "D7" in snap["drones"]
    assert snap["active_route"] == ["D7", "D4", "D3", "GS"]
    assert snap["network_status"] == "OPTIMAL"
    assert len(snap["stats"]["history"]) >= 1


def test_simulation_stepping_advances_clock():
    service = SwarmService(initial_sim_time=1000.0)
    service.step(1.5)
    assert service.sim_time == 1001.5

    service.step(2.0)
    assert service.sim_time == 1003.5

    # Step with dt <= 0 does nothing
    service.step(0.0)
    assert service.sim_time == 1003.5


def test_pause_and_resume():
    service = SwarmService()
    assert service.running is False

    service.start()
    assert service.running is True

    service.pause()
    assert service.running is False

    service.resume()
    assert service.running is True


def test_fail_drone_isolates_and_reroutes():
    service = SwarmService(initial_sim_time=1000.0)
    ok, msg = service.fail_drone("D4")
    assert ok is True

    snap = service.snapshot()
    d4_state = snap["drones"]["D4"]
    assert d4_state["operational_status"] == DroneHealthStatus.ISOLATED.value

    # Primary backbone was D7 -> D4 -> D3 -> GS. With D4 down, detour through D5 -> D6
    route = snap["active_route"]
    assert "D4" not in route
    assert route == ["D7", "D5", "D6", "GS"]
    assert "DETOUR" in snap["network_status"]


def test_progressive_recovery_and_restoration():
    service = SwarmService(initial_sim_time=1000.0)
    service.fail_drone("D4")
    assert "D4" not in service.snapshot()["active_route"]

    # Trigger recovery
    ok, msg = service.recover_drone("D4")
    assert ok is True
    assert service.snapshot()["drones"]["D4"]["operational_status"] == DroneHealthStatus.RECOVERING.value

    # Step forward through recovery passes and past hold-down timer (5s)
    service.step(1.0)
    service.step(6.0)

    snap = service.snapshot()
    assert snap["drones"]["D4"]["operational_status"] == DroneHealthStatus.HEALTHY.value
    assert snap["active_route"] == ["D7", "D4", "D3", "GS"]
    assert snap["network_status"] == "OPTIMAL"


def test_inject_attack_jamming():
    service = SwarmService(initial_sim_time=1000.0)
    ok, msg = service.inject_attack("D5", "jamming")
    assert ok is True

    snap = service.snapshot()
    d5 = snap["drones"]["D5"]
    assert d5["signal_strength"] <= -90.0
    assert d5["packet_loss"] >= 0.40


def test_inject_attack_dos():
    service = SwarmService(initial_sim_time=1000.0)
    ok, msg = service.inject_attack("D5", "dos")
    assert ok is True

    snap = service.snapshot()
    d5 = snap["drones"]["D5"]
    assert d5["latency"] >= 400.0
    assert d5["packet_loss"] >= 0.50


def test_inject_attack_spoofing_quarantines_node():
    service = SwarmService(initial_sim_time=1000.0)
    ok, msg = service.inject_attack("D6", "spoofing")
    assert ok is True

    snap = service.snapshot()
    d6 = snap["drones"]["D6"]
    assert d6["is_compromised"] is True
    assert d6["operational_status"] == DroneHealthStatus.ISOLATED.value


def test_compromised_node_cannot_recover_without_clearance():
    service = SwarmService(initial_sim_time=1000.0)
    service.inject_attack("D6", "spoofing")

    # Regular recovery should be rejected because drone is compromised
    ok, msg = service.recover_drone("D6", force=False)
    assert ok is False
    assert "COMPROMISED" in msg

    # Grant cryptographic clearance
    clr = service.grant_clearance("D6")
    assert clr["success"] is True

    # Now recovery proceeds
    ok2, msg2 = service.recover_drone("D6", force=False)
    assert ok2 is True


def test_run_detection_raises_not_implemented():
    service = SwarmService()
    with pytest.raises(NotImplementedError) as exc_info:
        service.run_detection()
    assert "not implemented yet" in str(exc_info.value)


def test_send_alert_delivers_encrypted_payload():
    service = SwarmService(initial_sim_time=1000.0)
    res = service.send_alert()

    assert res["delivered"] is True
    assert res["hops_traversed"] == ["D7", "D4", "D3", "GS"]
    assert res["persons"] == 2
    assert res["delivered_payload"] is not None
    assert res["delivered_payload"]["alert_id"] == res["alert_id"]


def test_run_vision_scripted():
    service = SwarmService(initial_sim_time=1000.0)
    res = service.run_vision(source="scripted")

    assert res["success"] is True
    assert res["mode"] == "scripted"
    alert = res["alert"]
    assert alert["delivered"] is True
    assert alert["hazards"] is None  # Honest: hazards null
    assert alert["hazards_supported"] is False
    assert len(alert["location_gps"]) == 2
    assert len(alert["location_local"]) == 2


def test_run_vision_missing_files_graceful():
    service = SwarmService(initial_sim_time=1000.0)
    res = service.run_vision(source="nonexistent_video_path.mp4")
    assert res["success"] is False
    assert "Video source file not found" in res["error"]

    with patch("pathlib.Path.exists", side_effect=[True, False]):
        res2 = service.run_vision(source="mock_vid.mp4", model_path="missing.onnx")
        assert res2["success"] is False
        assert "Model file missing" in res2["error"]


def test_security_check_truthful():
    service = SwarmService()
    sec = service.security_check()

    assert "mode" in sec
    assert "is_post_quantum" in sec
    assert "kem_algorithm" in sec
    assert "sig_algorithm" in sec
    assert "verdict" in sec


def test_snapshot_sanitization():
    service = SwarmService(initial_sim_time=1000.0)
    service.inject_attack("D6", "spoofing")
    service.grant_clearance("D6")

    snap = service.snapshot()
    # Confirm JSON-serializable
    dumped = json.dumps(snap)
    assert len(dumped) > 0

    # Ensure no secret keys or signatures leaked in event details
    for evt in snap["events"]:
        details = evt.get("details", {})
        assert "signature" not in details
        assert "clearance_token" not in details
        assert "token" not in details


def test_network_map_rendering():
    service = SwarmService(initial_sim_time=1000.0)
    snap = service.snapshot()

    svg = render_network_map_svg(snap)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert "D7" in svg
    assert "GS" in svg
    assert "Active Route" in svg

    fig = render_network_map_matplotlib(snap)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
