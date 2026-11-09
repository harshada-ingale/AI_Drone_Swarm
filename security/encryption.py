"""Secure messaging, encryption adapters, and security diagnostics."""

import json
from typing import Any

from security.authentication import Identity, get_supported_sig_algorithm
from security.key_manager import KeyManager, PkiRegistry
from security.pqc_security import (
    AuthenticationError,
    PQUnavailableError,
    ReplayError,
    SecurityError,
    Session,
    get_supported_kem_algorithm,
    is_pq_active,
)


def security_status(km: KeyManager | None = None) -> dict[str, Any]:
    """Returns a status dictionary detailing post-quantum cryptographic status.

    Truthful reporting: never reports post-quantum active unless genuine
    ML-KEM / ML-DSA mechanisms are loaded and active.
    """
    pq_available = is_pq_active()
    _, kem_name = get_supported_kem_algorithm(prefer_pq=True)
    _, sig_name = get_supported_sig_algorithm(prefer_pq=True)

    local_sig = km.local_identity.algorithm if (km and km.local_identity) else sig_name
    active_sess_count = 0
    if km:
        active_sess_count = sum(1 for s in km.sessions.values() if not s.needs_rekey)

    mode = "hybrid" if pq_available else "classical"
    verdict = "POST-QUANTUM ACTIVE" if pq_available else "CLASSICAL FALLBACK (not post-quantum)"

    return {
        "is_post_quantum": pq_available,
        "sig_algorithm": local_sig,
        "kem_algorithm": kem_name,
        "mode": mode,
        "active_sessions": active_sess_count,
        "verdict": verdict,
    }


class SecureMessenger:
    """Provides authenticated and encrypted payload transmission between swarm nodes."""

    def __init__(
        self,
        km: KeyManager,
        router: Any = None,
        controller: Any = None,
    ) -> None:
        if not isinstance(km, KeyManager):
            raise TypeError("km must be a KeyManager instance.")
        self.km = km
        self.router = router
        self.controller = controller

    def send(
        self,
        peer_id: str,
        payload: dict[str, Any] | str | bytes,
        aad: bytes = b"",
    ) -> bytes:
        """Encrypts payload for peer_id using the established AES-256-GCM session.

        Raises:
            SecurityError: if no valid active session exists.
        """
        session = self.km.get_session(peer_id)
        if session is None:
            raise SecurityError(f"No active session established with peer '{peer_id}'.")

        if session.needs_rekey:
            raise SecurityError(
                f"Session with peer '{peer_id}' has expired (age={session.max_age_s}s or "
                f"count={session.send_counter}) and requires rekey."
            )

        if isinstance(payload, dict):
            plaintext = json.dumps(payload, sort_keys=True).encode("utf-8")
        elif isinstance(payload, str):
            plaintext = payload.encode("utf-8")
        elif isinstance(payload, bytes):
            plaintext = payload
        else:
            raise TypeError("Payload must be a dict, str, or bytes.")

        return session.encrypt(plaintext, aad=aad)

    def receive(
        self,
        ciphertext: bytes,
        peer_id: str | None = None,
        aad: bytes = b"",
    ) -> dict[str, Any]:
        """Decrypts and authenticates ciphertext received from a peer.

        Enforces replay protection (64-packet window) and AEAD authentication.

        Returns:
            Decrypted dictionary payload.

        Raises:
            SecurityError: On tampering, missing session, or malformed framing.
            ReplayError: On duplicate or expired sequence numbers.
        """
        import struct

        if not isinstance(ciphertext, bytes) or len(ciphertext) < 22:
            raise SecurityError("Ciphertext is too short or invalid wire framing.")

        sess_len = struct.unpack(">H", ciphertext[:2])[0]
        if len(ciphertext) < 2 + sess_len + 20:
            raise SecurityError("Ciphertext is too short for session ID and headers.")

        session_id = ciphertext[2 : 2 + sess_len].decode("utf-8", errors="replace")

        session: Session | None = None

        if peer_id is not None:
            session = self.km.get_session(peer_id)

        if session is None:
            # Look up session by exact session_id match
            for s in self.km.sessions.values():
                if s.session_id == session_id:
                    session = s
                    break

        if session is None:
            raise SecurityError(f"No active session found matching session '{session_id}'.")

        plaintext = session.decrypt(ciphertext, aad=aad)

        try:
            parsed = json.loads(plaintext.decode("utf-8"))
            if isinstance(parsed, dict):
                return parsed
            return {"data": parsed}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {"raw_bytes": plaintext}


# Export common classes
__all__ = [
    "Identity",
    "KeyManager",
    "PkiRegistry",
    "SecureMessenger",
    "Session",
    "SecurityError",
    "ReplayError",
    "AuthenticationError",
    "PQUnavailableError",
    "is_pq_active",
    "security_status",
]
