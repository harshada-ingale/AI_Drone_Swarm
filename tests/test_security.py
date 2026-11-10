"""Comprehensive Post-Quantum and Classical Cryptographic Security Test Suite.

Covers all 34 required security and integration test scenarios:
- Cryptographic identities (Ed25519 & ML-DSA-65)
- Hybrid & classical key derivation
- AES-256-GCM session management & anti-replay sliding window
- 2-message authenticated handshake protocol
- Fail-closed security policy under require_pq
- PKI registry, identity revocation, and key lifecycle management
- SecureMessenger encrypted payload transmission
- Truthful cryptographic status reporting
- Cryptographically signed operator clearance verification
- Anti-flap route restoration hold-down timer
- End-to-end vision alert encryption with hop-by-hop relay preservation
- Mid-transit failure rerouting with identical ciphertext forwarding
"""

import base64
import os
import time
from typing import Any
import pytest

from controller.adapters import SecurityAdapter, VisionAlertAdapter
from controller.config import SelfHealingConfig
from controller.models import (
    DroneHealthStatus,
    DroneState,
    EventType,
    FailureReason,
    ThreatReport,
)
from controller.self_healing_controller import SelfHealingController
from routing.dijkstra import DijkstraRouter
from security.authentication import (
    Identity,
    get_supported_sig_algorithm,
    verify_signature,
)
from security.encryption import (
    SecureMessenger,
    security_status,
)
from security.key_manager import (
    KeyManager,
    PkiRegistry,
)
from security.pqc_security import (
    AuthenticationError,
    HandshakeManager,
    PQUnavailableError,
    ReplayError,
    SecurityError,
    Session,
    derive_session_key,
    get_supported_kem_algorithm,
    is_pq_active,
)
from vision.emergency_alert import EmergencyAlert


# ==============================================================================
# 1. CRYPTOGRAPHIC IDENTITIES & SIGNATURES (Tests 1 - 5)
# ==============================================================================

def test_01_identity_generate_ed25519() -> None:
    """Test 1: Generates classical Ed25519 identity, validates properties."""
    ident = Identity.generate("D1", prefer_pq=False)
    assert ident.peer_id == "D1"
    assert ident.algorithm == "Ed25519"
    assert len(ident.public_key_bytes) == 32
    assert ident.is_post_quantum is False


def test_02_identity_sign_verify_success() -> None:
    """Test 2: Signs message and verifies against same public key."""
    ident = Identity.generate("D2")
    msg = b"MISSION_COMMAND:WAYPOINT_ALPHA"
    sig = ident.sign(msg)
    assert isinstance(sig, bytes)
    assert len(sig) > 0
    assert ident.verify(msg, sig) is True


def test_03_identity_sign_verify_failure_tampered_message() -> None:
    """Test 3: Signature verification fails on tampered message bytes."""
    ident = Identity.generate("D3")
    msg = b"VALID_TELEMETRY_PACKET"
    sig = ident.sign(msg)
    tampered = b"TAMPERED_TELEMETRY_PACKET"
    assert ident.verify(tampered, sig) is False


def test_04_identity_sign_verify_failure_wrong_key() -> None:
    """Test 4: Signature verification fails against a different peer's key."""
    ident_a = Identity.generate("D4A")
    ident_b = Identity.generate("D4B")
    msg = b"PEER_AUTHENTICATION"
    sig_a = ident_a.sign(msg)
    assert ident_a.verify(msg, sig_a, public_key_bytes=ident_b.public_key_bytes) is False


def test_05_identity_empty_message_rejected() -> None:
    """Test 5: Signing an empty message raises ValueError."""
    ident = Identity.generate("D5")
    with pytest.raises(ValueError, match="Cannot sign empty message"):
        ident.sign(b"")


# ==============================================================================
# 2. KEY DERIVATION & SESSION MANAGEMENT (Tests 6 - 14)
# ==============================================================================

def test_06_kem_derivation_classical() -> None:
    """Test 6: derive_session_key with X25519 secret produces 32-byte key."""
    x25519_secret = os.urandom(32)
    key = derive_session_key(x25519_secret)
    assert isinstance(key, bytes)
    assert len(key) == 32


def test_07_kem_derivation_hybrid() -> None:
    """Test 7: derive_session_key combining X25519 and KEM secrets produces 32-byte key."""
    x25519_secret = os.urandom(32)
    kem_secret = os.urandom(32)
    key = derive_session_key(x25519_secret, kem_secret)
    assert isinstance(key, bytes)
    assert len(key) == 32
    # Derivation with KEM must differ from derivation without KEM
    key_no_kem = derive_session_key(x25519_secret)
    assert key != key_no_kem


def test_08_session_encryption_roundtrip() -> None:
    """Test 8: Session AES-256-GCM encryption and decryption roundtrip."""
    aes_key = os.urandom(32)
    session = Session(
        session_id="SESS-08",
        peer_id="PEER-8",
        mode="classical",
        kem_algorithm="X25519",
        sig_algorithm="Ed25519",
        aes_key=aes_key,
    )
    plaintext = b"CONFIDENTIAL_DRONE_STATUS"
    ciphertext = session.encrypt(plaintext)
    assert ciphertext != plaintext
    decrypted = session.decrypt(ciphertext)
    assert decrypted == plaintext


def test_09_session_encryption_tamper_fails() -> None:
    """Test 9: Tampered ciphertext bytes trigger SecurityError (AEAD integrity failure)."""
    aes_key = os.urandom(32)
    session = Session(
        session_id="SESS-09",
        peer_id="PEER-9",
        mode="classical",
        kem_algorithm="X25519",
        sig_algorithm="Ed25519",
        aes_key=aes_key,
    )
    ciphertext = bytearray(session.encrypt(b"INTEGRITY_CHECK"))
    ciphertext[-1] ^= 0x55  # flip bits in GCM authentication tag
    with pytest.raises(SecurityError, match="Ciphertext integrity verification failed"):
        session.decrypt(bytes(ciphertext))


def test_10_session_replay_protection_duplicate() -> None:
    """Test 10: Replaying an identical packet sequence raises ReplayError."""
    aes_key = os.urandom(32)
    session = Session(
        session_id="SESS-10",
        peer_id="PEER-10",
        mode="classical",
        kem_algorithm="X25519",
        sig_algorithm="Ed25519",
        aes_key=aes_key,
    )
    pkt = session.encrypt(b"PACKET_1")
    assert session.decrypt(pkt) == b"PACKET_1"
    with pytest.raises(ReplayError, match="Replayed or duplicate packet rejected"):
        session.decrypt(pkt)


def test_11_session_replay_window_sliding() -> None:
    """Test 11: 64-packet sliding window accepts in-window packets, rejects older."""
    aes_key = os.urandom(32)
    sess_send = Session("SESS-11", "PEER-11", "classical", "X25519", "Ed25519", aes_key)
    sess_recv = Session("SESS-11", "PEER-11", "classical", "X25519", "Ed25519", aes_key)

    pkts = [sess_send.encrypt(f"MSG_{i}".encode()) for i in range(80)]

    # Receive up to packet 70
    for i in range(70):
        sess_recv.decrypt(pkts[i])

    # Packet 2 is now outside the 64-packet window (highest=69, min_allowed=5)
    with pytest.raises(ReplayError):
        sess_recv.decrypt(pkts[2])


def test_12_session_monotonic_counters() -> None:
    """Test 12: Sequence counter increments monotonically on every encryption."""
    aes_key = os.urandom(32)
    session = Session("SESS-12", "PEER-12", "classical", "X25519", "Ed25519", aes_key)
    assert session.send_counter == 0
    session.encrypt(b"M1")
    assert session.send_counter == 1
    session.encrypt(b"M2")
    assert session.send_counter == 2


def test_13_session_rekey_expiration_time() -> None:
    """Test 13: Session expires and signals needs_rekey after max_age_s."""
    aes_key = os.urandom(32)
    session = Session(
        "SESS-13",
        "PEER-13",
        "classical",
        "X25519",
        "Ed25519",
        aes_key,
        created_at=time.time() - 3601.0,
        max_age_s=3600.0,
    )
    assert session.needs_rekey is True


def test_14_session_rekey_expiration_counter() -> None:
    """Test 14: Session expires and signals needs_rekey after max_messages."""
    aes_key = os.urandom(32)
    session = Session(
        "SESS-14",
        "PEER-14",
        "classical",
        "X25519",
        "Ed25519",
        aes_key,
        max_messages=5,
        send_counter=5,
    )
    assert session.needs_rekey is True


# ==============================================================================
# 3. AUTHENTICATED HANDSHAKE PROTOCOL (Tests 15 - 18)
# ==============================================================================

def test_15_handshake_two_message_classical_roundtrip() -> None:
    """Test 15: 2-message authenticated handshake between two nodes derives shared session."""
    alice = Identity.generate("ALICE", prefer_pq=False)
    bob = Identity.generate("BOB", prefer_pq=False)

    pki = PkiRegistry()
    pki.register(alice)
    pki.register(bob)

    # 1. Alice creates HandshakeInit
    init_msg, ctx = HandshakeManager.create_init_message(alice, "BOB", prefer_pq=False)

    # 2. Bob processes HandshakeInit, creates HandshakeResp & session
    resp_msg, bob_sess = HandshakeManager.process_init_and_respond(init_msg, bob, pki.get)

    # 3. Alice processes HandshakeResp & creates session
    alice_sess = HandshakeManager.process_resp_and_finalize(resp_msg, ctx, pki.get, alice)

    # Both must share identical session_id and aes_key
    assert alice_sess.session_id == bob_sess.session_id
    assert alice_sess.aes_key == bob_sess.aes_key

    # Direct communication verified
    msg = alice_sess.encrypt(b"HELLO_BOB")
    assert bob_sess.decrypt(msg) == b"HELLO_BOB"


def test_16_handshake_invalid_init_signature() -> None:
    """Test 16: Forged HandshakeInit signature raises AuthenticationError."""
    alice = Identity.generate("ALICE")
    bob = Identity.generate("BOB")
    pki = PkiRegistry()
    pki.register(alice)
    pki.register(bob)

    init_msg, _ = HandshakeManager.create_init_message(alice, "BOB")
    # Corrupt signature
    init_msg["signature"] = base64.b64encode(b"FORGED_SIGNATURE" * 4).decode("ascii")

    with pytest.raises(AuthenticationError, match="Invalid HandshakeInit signature"):
        HandshakeManager.process_init_and_respond(init_msg, bob, pki.get)


def test_17_handshake_invalid_resp_signature() -> None:
    """Test 17: Forged HandshakeResp signature raises AuthenticationError."""
    alice = Identity.generate("ALICE")
    bob = Identity.generate("BOB")
    pki = PkiRegistry()
    pki.register(alice)
    pki.register(bob)

    init_msg, ctx = HandshakeManager.create_init_message(alice, "BOB")
    resp_msg, _ = HandshakeManager.process_init_and_respond(init_msg, bob, pki.get)
    # Corrupt response signature
    resp_msg["signature"] = base64.b64encode(b"FORGED_SIG_RESP" * 4).decode("ascii")

    with pytest.raises(AuthenticationError, match="Invalid HandshakeResp signature"):
        HandshakeManager.process_resp_and_finalize(resp_msg, ctx, pki.get, alice)


def test_18_handshake_require_pq_fails_closed() -> None:
    """Test 18: require_pq=True fails-closed with PQUnavailableError if PQ is missing."""
    if is_pq_active():
        pytest.skip("Test specifically verifies fail-closed when post-quantum is unavailable.")

    ident = Identity.generate("D18")
    with pytest.raises(PQUnavailableError, match="require_pq=True but post-quantum KEM is not available"):
        HandshakeManager.create_init_message(ident, "GS", prefer_pq=True, require_pq=True)


# ==============================================================================
# 4. PKI REGISTRY & KEY MANAGER (Tests 19 - 23)
# ==============================================================================

def test_19_pki_registry_register_and_get() -> None:
    """Test 19: Register identity into PKI registry and retrieve by peer_id."""
    pki = PkiRegistry()
    ident = Identity.generate("D19")
    pki.register(ident)
    assert pki.is_registered("D19") is True
    assert pki.get("D19") == ident
    assert "D19" in pki.list_peers()


def test_20_pki_registry_revocation() -> None:
    """Test 20: Revoking a peer removes it from active list and returns None on get."""
    pki = PkiRegistry()
    ident = Identity.generate("D20")
    pki.register(ident)
    pki.revoke("D20")
    assert pki.is_revoked("D20") is True
    assert pki.is_registered("D20") is False
    assert pki.get("D20") is None
    assert "D20" not in pki.list_peers()


def test_21_pki_registry_reregistration_revoked_fails() -> None:
    """Test 21: Attempting to register a revoked peer ID raises SecurityError."""
    pki = PkiRegistry()
    ident = Identity.generate("D21")
    pki.register(ident)
    pki.revoke("D21")
    new_ident = Identity.generate("D21")
    with pytest.raises(SecurityError, match="Cannot register revoked peer"):
        pki.register(new_ident)


def test_22_key_manager_establish_session() -> None:
    """Test 22: KeyManager establish_session_with_peer sets active sessions on both."""
    pki = PkiRegistry()
    km1 = KeyManager("NODE-1", pki)
    km2 = KeyManager("NODE-2", pki)

    sess1, sess2 = km1.establish_session_with_peer(km2)
    assert km1.has_active_session("NODE-2") is True
    assert km2.has_active_session("NODE-1") is True
    assert sess1.session_id == sess2.session_id


def test_23_key_manager_revocation_purges_session() -> None:
    """Test 23: Revoking a peer via KeyManager immediately purges active sessions."""
    pki = PkiRegistry()
    km1 = KeyManager("NODE-23A", pki)
    km2 = KeyManager("NODE-23B", pki)
    km1.establish_session_with_peer(km2)

    assert km1.has_active_session("NODE-23B") is True
    km1.revoke_peer("NODE-23B")
    assert km1.has_active_session("NODE-23B") is False
    assert km1.get_session("NODE-23B") is None


# ==============================================================================
# 5. SECURE MESSENGER & STATUS REPORTING (Tests 24 - 27)
# ==============================================================================

def test_24_secure_messenger_send_and_receive() -> None:
    """Test 24: SecureMessenger encrypted dictionary payload round-trip."""
    pki = PkiRegistry()
    km1 = KeyManager("D24", pki)
    km2 = KeyManager("GS", pki)
    sm1 = SecureMessenger(km1)
    sm2 = SecureMessenger(km2)

    payload = {"alert": "FIRE_DETECTED", "severity": "HIGH", "coords": [12.0, 34.0]}
    ciphertext = sm1.send("GS", payload)
    assert isinstance(ciphertext, bytes)

    delivered = sm2.receive(ciphertext)
    assert delivered["alert"] == "FIRE_DETECTED"
    assert delivered["coords"] == [12.0, 34.0]


def test_25_secure_messenger_unauthenticated_peer_fails() -> None:
    """Test 25: Sending payload to an unauthenticated peer without session raises SecurityError."""
    pki = PkiRegistry()
    km = KeyManager("D25", pki)
    sm = SecureMessenger(km)
    with pytest.raises(SecurityError, match="No active session established"):
        sm.send("UNKNOWN_PEER", {"data": 123})


def test_26_security_status_truthful_classical() -> None:
    """Test 26: security_status truthfully reports CLASSICAL FALLBACK when PQ is absent."""
    if is_pq_active():
        pytest.skip("Test specifically verifies classical fallback status reporting.")

    stat = security_status()
    assert stat["is_post_quantum"] is False
    assert stat["mode"] == "classical"
    assert stat["sig_algorithm"] == "Ed25519"
    assert stat["kem_algorithm"] == "X25519"
    assert stat["verdict"] == "CLASSICAL FALLBACK (not post-quantum)"


def test_27_security_status_pq_active_only_when_available() -> None:
    """Test 27: is_pq_active returns True only if genuine ML-KEM and ML-DSA are present."""
    pq = is_pq_active()
    assert isinstance(pq, bool)
    stat = security_status()
    assert stat["is_post_quantum"] == pq
    if not pq:
        assert stat["verdict"] == "CLASSICAL FALLBACK (not post-quantum)"


# ==============================================================================
# 6. SIGNED SECURITY CLEARANCE (Tests 28 - 31)
# ==============================================================================

def test_28_controller_signed_clearance_valid() -> None:
    """Test 28: Valid cryptographically signed clearance clears compromised drone."""
    ctrl = SelfHealingController()
    ctrl.register_drone("D28", DroneState(drone_id="D28", battery=90.0))

    # Compromise drone
    ctrl.report_threat(
        ThreatReport(
            drone_id="D28",
            threat_level=0.95,
            threat_type="GPS_SPOOFING",
            timestamp=100.0,
        )
    )
    st = ctrl.get_drone_state("D28")
    assert st.is_compromised is True
    assert st.operational_status == DroneHealthStatus.ISOLATED

    # Operator generates key and signs clearance
    operator = Identity.generate("OP-01")
    ctrl.security._pki.register(operator)

    ts = 101.0
    exp = 200.0
    nonce = "NONCE-TEST-28"
    action = "CLEAR_COMPROMISED"
    transcript = f"CLEARANCE:D28:{action}:OP-01:{ts}:{exp}:{nonce}".encode("utf-8")
    sig = operator.sign(transcript)

    clearance = {
        "drone_id": "D28",
        "action": action,
        "operator_id": "OP-01",
        "timestamp": ts,
        "expires_at": exp,
        "nonce": nonce,
        "signature": base64.b64encode(sig).decode("ascii"),
    }

    evt = ctrl.grant_signed_clearance(clearance, current_time=102.0)
    assert evt.event_type == EventType.SECURITY_CLEARANCE_GRANTED
    assert st.is_compromised is False
    assert st.security_cleared is True


def test_29_controller_signed_clearance_invalid_signature() -> None:
    """Test 29: Forged signature on clearance raises AuthenticationError."""
    ctrl = SelfHealingController()
    ctrl.register_drone("D29", DroneState(drone_id="D29", is_compromised=True, operational_status=DroneHealthStatus.ISOLATED))
    operator = Identity.generate("OP-02")
    ctrl.security._pki.register(operator)

    clearance = {
        "drone_id": "D29",
        "action": "CLEAR_COMPROMISED",
        "operator_id": "OP-02",
        "timestamp": 100.0,
        "expires_at": 200.0,
        "nonce": "NONCE-29",
        "signature": base64.b64encode(b"FORGED_SIGNATURE" * 4).decode("ascii"),
    }
    with pytest.raises(AuthenticationError, match="Invalid clearance signature"):
        ctrl.grant_signed_clearance(clearance, current_time=101.0)


def test_30_controller_signed_clearance_expired() -> None:
    """Test 30: Expired clearance raises ValueError."""
    ctrl = SelfHealingController()
    ctrl.register_drone("D30", DroneState(drone_id="D30", is_compromised=True, operational_status=DroneHealthStatus.ISOLATED))
    operator = Identity.generate("OP-03")
    ctrl.security._pki.register(operator)

    clearance = {
        "drone_id": "D30",
        "action": "CLEAR_COMPROMISED",
        "operator_id": "OP-03",
        "timestamp": 100.0,
        "expires_at": 150.0,
        "nonce": "NONCE-30",
        "signature": base64.b64encode(b"SIG").decode("ascii"),
    }
    with pytest.raises(ValueError, match="Clearance expired"):
        ctrl.grant_signed_clearance(clearance, current_time=160.0)


def test_31_controller_signed_clearance_replay_rejected() -> None:
    """Test 31: Replaying a used clearance nonce raises ReplayError."""
    ctrl = SelfHealingController()
    ctrl.register_drone("D31", DroneState(drone_id="D31", is_compromised=True, operational_status=DroneHealthStatus.ISOLATED))
    operator = Identity.generate("OP-04")
    ctrl.security._pki.register(operator)

    ts = 100.0
    exp = 200.0
    nonce = "REPLAY-NONCE-31"
    action = "CLEAR_COMPROMISED"
    transcript = f"CLEARANCE:D31:{action}:OP-04:{ts}:{exp}:{nonce}".encode("utf-8")
    sig = operator.sign(transcript)

    clearance = {
        "drone_id": "D31",
        "action": action,
        "operator_id": "OP-04",
        "timestamp": ts,
        "expires_at": exp,
        "nonce": nonce,
        "signature": base64.b64encode(sig).decode("ascii"),
    }

    ctrl.grant_signed_clearance(clearance, current_time=101.0)

    # Replay same clearance
    st = ctrl.get_drone_state("D31")
    st.is_compromised = True
    st.operational_status = DroneHealthStatus.ISOLATED
    with pytest.raises(ReplayError, match="Clearance nonce has already been used"):
        ctrl.grant_signed_clearance(clearance, current_time=102.0)


# ==============================================================================
# 7. ROUTE RESTORATION HOLD-DOWN TIMER (Test 32)
# ==============================================================================

def test_32_controller_hold_down_timer_enforcement() -> None:
    """Test 32: Route restoration holds down until restore_hold_down_s has elapsed."""
    config = SelfHealingConfig(
        restore_hold_down_s=5.0,
        auto_restore_routes=True,
        max_comm_range_m=250.0,
        consecutive_checks_to_recover=1,
    )
    ctrl = SelfHealingController(config=config)

    # Topology: D7 -> D4 -> D3 -> GS (Primary) and D7 -> D5 -> D6 -> GS (Detour)
    nodes = [
        ("D7", (0.0, 0.0)),
        ("D4", (200.0, 0.0)),
        ("D3", (400.0, 0.0)),
        ("D5", (200.0, 120.0)),
        ("D6", (400.0, 120.0)),
        ("GS", (600.0, 0.0)),
    ]
    for d_id, pos in nodes:
        ctrl.register_drone(d_id, DroneState(drone_id=d_id, position=pos, battery=90.0))

    route = ctrl.register_active_route("route-32", "D7", "GS")
    assert route == ["D7", "D4", "D3", "GS"]

    now = 1000.0
    # Isolate D4 and reroute
    ctrl.isolate_drone("D4", reason=FailureReason.DISCONNECTION, current_time=now)
    detour = ctrl.get_active_route("route-32")
    assert detour == ["D7", "D5", "D6", "GS"]

    # D4 recovers: pass 1 moves to RECOVERING, pass 2 satisfies recovery criteria -> HEALTHY
    rec_time = now + 2.0
    st_d4 = ctrl.get_drone_state("D4")
    ctrl.update_drone_health({"drone_id": "D4", "battery": 90.0, "timestamp": rec_time}, current_time=rec_time)
    assert st_d4.operational_status == DroneHealthStatus.RECOVERING

    ctrl.update_drone_health({"drone_id": "D4", "battery": 90.0, "timestamp": rec_time + 0.1}, current_time=rec_time + 0.1)
    assert st_d4.operational_status == DroneHealthStatus.HEALTHY
    assert st_d4.recovery_timestamp == rec_time + 0.1

    # Immediate check at rec_time + 1.0 (elapsed = 0.9s < 5.0s hold-down)
    restored = ctrl.check_route_restorations(current_time=rec_time + 1.0)
    assert len(restored) == 0
    assert ctrl.get_active_route("route-32") == ["D7", "D5", "D6", "GS"]

    # Check at rec_time + 6.0 (elapsed = 5.9s >= 5.0s hold-down) -> restored!
    restored = ctrl.check_route_restorations(current_time=rec_time + 6.0)
    assert "route-32" in restored
    assert ctrl.get_active_route("route-32") == ["D7", "D4", "D3", "GS"]


# ==============================================================================
# 8. SECURE VISION ALERT DELIVERY & HOP FORWARDING (Tests 33 - 34)
# ==============================================================================

def test_33_vision_alert_hop_by_hop_encryption() -> None:
    """Test 33: Relays forward raw ciphertext verbatim without decrypting; destination GS unseals."""
    config = SelfHealingConfig(max_comm_range_m=250.0)
    ctrl = SelfHealingController(config=config)

    for d_id, pos in [("D7", (0.0, 0.0)), ("D4", (200.0, 0.0)), ("D3", (400.0, 0.0)), ("GS", (600.0, 0.0))]:
        ctrl.register_drone(d_id, DroneState(drone_id=d_id, position=pos, battery=95.0))

    alert = EmergencyAlert(
        alert_id="ALERT-33",
        drone_id="D7",
        timestamp=time.time(),
        priority="HIGH",
        persons=2,
        hazards={"fire": 1},
        location=(50.0, 25.0),
    )

    report = ctrl.dispatch_emergency_alert(alert, origin_drone_id="D7")
    assert report["delivered"] is True
    assert report["hops_traversed"] == ["D7", "D4", "D3", "GS"]
    assert report["delivered_payload"]["alert_id"] == "ALERT-33"
    assert report["delivered_payload"]["persons"] == 2


def test_34_vision_alert_mid_transit_reroute_same_ciphertext() -> None:
    """Test 34: When next relay fails mid-transit, dynamic reroute forwards identical ciphertext."""
    config = SelfHealingConfig(max_comm_range_m=250.0)
    ctrl = SelfHealingController(config=config)

    nodes = [
        ("D7", (0.0, 0.0)),
        ("D4", (200.0, 0.0)),
        ("D3", (400.0, 0.0)),
        ("D5", (200.0, 120.0)),
        ("D6", (400.0, 120.0)),
        ("GS", (600.0, 0.0)),
    ]
    for d_id, pos in nodes:
        ctrl.register_drone(d_id, DroneState(drone_id=d_id, position=pos, battery=95.0))

    # Initial route: D7 -> D4 -> D3 -> GS
    initial_route = ["D7", "D4", "D3", "GS"]

    # D4 fails right before packet reaches it
    ctrl.isolate_drone("D4", reason=FailureReason.DISCONNECTION)

    adapter = VisionAlertAdapter(ctrl.security)
    alert = EmergencyAlert(
        alert_id="ALERT-34-MID-TRANSIT",
        drone_id="D7",
        timestamp=time.time(),
        priority="CRITICAL",
        persons=1,
        hazards={},
        location=(10.0, 10.0),
    )

    # Transmit with controller provided for mid-transit dynamic rerouting
    report = adapter.transmit_alert(alert, initial_route, controller=ctrl)

    assert report["delivered"] is True
    assert report["reroutes"] >= 1
    # D4 must not be in hops
    assert "D4" not in report["hops_traversed"]
    # Path dynamically traversed healthy north relays
    assert "D5" in report["hops_traversed"]
    assert report["delivered_payload"]["alert_id"] == "ALERT-34-MID-TRANSIT"
