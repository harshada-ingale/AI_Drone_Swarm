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
            # Ensure recipient is known
            self.get_or_create_messenger(recipient_id)
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

    def __init__(self, security_adapter: SecurityAdapter | None = None) -> None:
        self.security = security_adapter or SecurityAdapter()

    def transmit_alert(
        self,
        alert: EmergencyAlert,
        route: list[str],
    ) -> dict[str, Any]:
        """Transmits an EmergencyAlert along a verified path to Ground Station.

        Args:
            alert: The EmergencyAlert object.
            route: Ordered list of nodes, e.g. ["D7", "D4", "D3", "GS"].

        Returns:
            Transmission report with delivery status, hops, and decrypted confirmation.
        """
        if not route or len(route) < 2:
            return {
                "success": False,
                "error": "Invalid route path: must contain at least source and destination.",
                "hops_traversed": [],
            }

        source = route[0]
        destination = route[-1]

        # Origin produces payload and securely seals it
        payload = alert.to_payload()
        sealed_bytes = self.security.seal_payload(source, destination, payload)

        hops_traversed: list[str] = [source]

        # Relay along path (relay hops forward without decrypting)
        current_data = sealed_bytes
        for i in range(1, len(route) - 1):
            relay_node = route[i]
            next_node = route[i + 1]
            current_data = self.security.forward_relay_hop(current_data, relay_node, next_node)
            hops_traversed.append(relay_node)

        # Final destination unseals and verifies payload
        hops_traversed.append(destination)
        delivered_payload = self.security.unseal_payload(destination, current_data)

        return {
            "success": True,
            "alert_id": delivered_payload.get("alert_id"),
            "source": source,
            "destination": destination,
            "route": route,
            "hops_traversed": hops_traversed,
            "delivered_payload": delivered_payload,
        }
