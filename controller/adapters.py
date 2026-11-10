"""Adapters for integrating Self-Healing Controller with Simulator, AI, Security, and Vision."""

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Mapping

from controller.models import DroneState, ThreatReport
from vision.emergency_alert import EmergencyAlert


class SimulatorAdapter:
    """Translates simulator frames and telemetry updates into controller updates."""

    @staticmethod
    def parse_telemetry(raw_data: dict[str, Any]) -> dict[str, Any]:
        """Sanitizes and normalizes raw telemetry dictionary from simulator.

        Raises ValueError if required fields are missing or malformed.
        """
        if not isinstance(raw_data, dict):
            raise ValueError("Telemetry data must be a dictionary.")

        if "drone_id" not in raw_data:
            raise ValueError("Missing mandatory field 'drone_id' in telemetry.")

        drone_id = str(raw_data["drone_id"])

        pos = raw_data.get("position", (0.0, 0.0))
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            position = (float(pos[0]), float(pos[1]))
        else:
            position = (0.0, 0.0)

        battery = float(raw_data.get("battery", 100.0))
        signal = float(raw_data.get("signal_strength", -50.0))
        packet_loss = float(raw_data.get("packet_loss", 0.0))
        latency = float(raw_data.get("latency", 20.0))
        threat_level = float(raw_data.get("threat_level", 0.0))
        altitude = float(raw_data.get("altitude", 30.0))
        heading = float(raw_data.get("heading", 0.0))

        return {
            "drone_id": drone_id,
            "position": position,
            "altitude": altitude,
            "heading": heading,
            "battery": battery,
            "signal_strength": signal,
            "packet_loss": packet_loss,
            "latency": latency,
            "threat_level": threat_level,
            "last_seen": float(raw_data.get("timestamp", time.time())),
        }


class AIThreatAdapter:
    """Interface adapter for AI anomaly and threat detection alerts."""

    @staticmethod
    def parse_threat(raw_report: dict[str, Any] | ThreatReport) -> ThreatReport:
        """Parses threat reports from AI module."""
        if isinstance(raw_report, ThreatReport):
            return raw_report

        if not isinstance(raw_report, dict):
            raise ValueError("Threat report must be a ThreatReport or dict.")

        if "drone_id" not in raw_report:
            raise ValueError("Missing 'drone_id' in threat report.")

        return ThreatReport(
            drone_id=str(raw_report["drone_id"]),
            threat_level=float(raw_report.get("threat_level", 1.0)),
            threat_type=str(raw_report.get("threat_type", "ANOMALY")),
            timestamp=float(raw_report.get("timestamp", time.time())),
            details=dict(raw_report.get("details", {})),
        )


class SecurityAdapter:
    """Manages cryptographic secure messaging and encrypted relay forwarding."""

    def __init__(self) -> None:
        self._has_pqc = False
        self._pki = None
        self._key_managers: dict[str, Any] = {}
        self._messengers: dict[str, Any] = {}

        try:
            from security.encryption import KeyManager, PkiRegistry, SecureMessenger
            self._pki = PkiRegistry()
            self._KeyManager = KeyManager
            self._SecureMessenger = SecureMessenger
            self._has_pqc = True
        except (ImportError, ModuleNotFoundError):
            self._has_pqc = False

    @property
    def has_native_pqc(self) -> bool:
        """Returns True if security.encryption is imported and available."""
        return self._has_pqc

    def get_or_create_messenger(self, peer_id: str) -> Any:
        """Returns SecureMessenger instance for the given peer."""
        if not self._has_pqc:
            return None
        if peer_id not in self._messengers:
            km = self._KeyManager(peer_id, self._pki)
            self._key_managers[peer_id] = km
            self._messengers[peer_id] = self._SecureMessenger(km)
        return self._messengers[peer_id]

    def seal_payload(self, sender_id: str, recipient_id: str, payload: dict[str, Any]) -> bytes:
        """Encrypts or securely seals payload for transmission across relay drones."""
        if self._has_pqc:
            messenger = self.get_or_create_messenger(sender_id)
            recipient_messenger = self.get_or_create_messenger(recipient_id)
            if not messenger.km.has_active_session(recipient_id):
                messenger.km.establish_session_with_peer(recipient_messenger.km)
            return messenger.send(recipient_id, payload)

        # Standalone cryptographic envelope fallback (AES/HMAC-SHA256 simulation)
        payload_bytes = json.dumps(payload).encode("utf-8")
        secret_key = f"SWARM-SECURE-KEY-{recipient_id}".encode("utf-8")
        signature = hmac.new(secret_key, payload_bytes, hashlib.sha256).hexdigest()
        envelope = {
            "sender": sender_id,
            "recipient": recipient_id,
            "sig": signature,
            "body": base64.b64encode(payload_bytes).decode("ascii"),
        }
        return json.dumps(envelope).encode("utf-8")

    def unseal_payload(self, receiver_id: str, encrypted_bytes: bytes) -> dict[str, Any]:
        """Decrypts and authenticates payload at the final receiving station (e.g. GS)."""
        if self._has_pqc:
            messenger = self.get_or_create_messenger(receiver_id)
            return messenger.receive(encrypted_bytes)

        envelope = json.loads(encrypted_bytes.decode("utf-8"))
        if envelope.get("recipient") != receiver_id:
            raise ValueError(f"Payload intended for {envelope.get('recipient')}, not {receiver_id}")

        body_bytes = base64.b64decode(envelope["body"])
        secret_key = f"SWARM-SECURE-KEY-{receiver_id}".encode("utf-8")
        expected_sig = hmac.new(secret_key, body_bytes, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(envelope["sig"], expected_sig):
            raise PermissionError("Encrypted payload integrity verification failed.")

        return json.loads(body_bytes.decode("utf-8"))

    @staticmethod
    def forward_relay_hop(encrypted_bytes: bytes, current_hop: str, next_hop: str) -> bytes:
        """Forwards encrypted bytes through a relay drone without decryption."""
        # Critical security invariant: relay drones must NEVER decrypt packets.
        # They pass the raw encrypted payload verbatim.
        return encrypted_bytes


class VisionAlertAdapter:
    """Coordinates dispatching emergency alerts over self-healed routes."""

    def __init__(
        self,
        security_adapter: SecurityAdapter | None = None,
        controller: Any = None,
    ) -> None:
        self.security = security_adapter or SecurityAdapter()
        self.controller = controller

    def transmit_alert(
        self,
        alert: EmergencyAlert,
        route: list[str],
        controller: Any = None,
    ) -> dict[str, Any]:
        """Transmits an EmergencyAlert along a verified path to Ground Station.

        Guarantees that raw ciphertext is forwarded hop-by-hop without relay decryption.
        If a next hop fails or is isolated mid-transit, queries the controller for a
        rerouted path and forwards the identical ciphertext without re-encrypting.

        Args:
            alert: The EmergencyAlert object.
            route: Ordered list of nodes, e.g. ["D7", "D4", "D3", "GS"].
            controller: Optional controller reference for mid-transit reroute resolution.

        Returns:
            Transmission report with delivered status, hops, reroutes, and decrypted payload.
        """
        if not route or len(route) < 2:
            return {
                "delivered": False,
                "success": False,
                "error": "Invalid route path: must contain at least source and destination.",
                "hops": [],
                "hops_traversed": [],
                "reroutes": 0,
                "delivered_payload": None,
            }

        ctrl = controller or self.controller
        source = route[0]
        destination = route[-1]

        # Origin produces payload and securely seals it once
        payload = alert.to_payload()
        sealed_bytes = self.security.seal_payload(source, destination, payload)

        hops_traversed: list[str] = [source]
        current_data = sealed_bytes
        active_route = list(route)
        current_idx = 0
        reroutes_count = 0

        while current_idx < len(active_route) - 1:
            curr_node = active_route[current_idx]
            next_node = active_route[current_idx + 1]

            # Check if next hop is isolated or failed mid-transit
            is_next_failed = False
            if ctrl is not None:
                excluded = ctrl.get_isolated_and_failed_drones()
                if next_node in excluded:
                    is_next_failed = True

            if is_next_failed:
                try:
                    # Request dynamic alternative path bypassing the failed node
                    new_subpath = ctrl.get_route(curr_node, destination, raise_on_failure=True)
                    active_route = active_route[: current_idx + 1] + new_subpath[1:]
                    next_node = active_route[current_idx + 1]
                    reroutes_count += 1
                except Exception as err:
                    return {
                        "delivered": False,
                        "success": False,
                        "error": f"Mid-transit rerouting failed at '{curr_node}': {err}",
                        "hops": hops_traversed,
                        "hops_traversed": hops_traversed,
                        "reroutes": reroutes_count,
                        "delivered_payload": None,
                    }

            # Forward identical ciphertext without decrypting at relay hop
            current_data = self.security.forward_relay_hop(current_data, curr_node, next_node)
            current_idx += 1
            hops_traversed.append(next_node)

        # Final destination unseals and authenticates payload
        try:
            delivered_payload = self.security.unseal_payload(destination, current_data)
        except Exception as err:
            return {
                "delivered": False,
                "success": False,
                "error": f"Failed to unseal payload at destination '{destination}': {err}",
                "hops": hops_traversed,
                "hops_traversed": hops_traversed,
                "reroutes": reroutes_count,
                "delivered_payload": None,
            }

        return {
            "delivered": True,
            "success": True,
            "alert_id": delivered_payload.get("alert_id") if isinstance(delivered_payload, dict) else None,
            "source": source,
            "destination": destination,
            "route": active_route,
            "hops": hops_traversed,
            "hops_traversed": hops_traversed,
            "reroutes": reroutes_count,
            "error": None,
            "delivered_payload": delivered_payload,
        }

