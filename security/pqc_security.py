"""Post-Quantum and Classical Hybrid Key Exchange and Encrypted Session Management."""

import base64
from dataclasses import dataclass, field
import hashlib
import json
import os
import struct
import time
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from security.authentication import Identity, verify_signature


class SecurityError(Exception):
    """Base exception for security errors."""


class ReplayError(SecurityError):
    """Raised when a duplicated or expired replayed packet is received."""


class AuthenticationError(SecurityError):
    """Raised when peer signature or identity verification fails."""


class PQUnavailableError(SecurityError):
    """Raised when post-quantum security is strictly required but unavailable."""


def get_supported_kem_algorithm(prefer_pq: bool = True) -> tuple[bool, str]:
    """Detects available KEM mechanism.

    Returns:
        (is_pq, algorithm_name)
    """
    if prefer_pq:
        try:
            import oqs  # type: ignore[import-untyped]
            enabled = oqs.get_enabled_KEM_mechanisms()
            if "ML-KEM-768" in enabled:
                return True, "ML-KEM-768"
            if "Kyber768" in enabled:
                return True, "Kyber768"
        except (ImportError, Exception):
            pass
    return False, "X25519"


def is_pq_active() -> bool:
    """Checks whether both KEM and signature post-quantum algorithms are available."""
    try:
        import oqs  # type: ignore[import-untyped]
        kems = oqs.get_enabled_KEM_mechanisms()
        sigs = oqs.get_enabled_sig_mechanisms()
        has_kem = ("ML-KEM-768" in kems) or ("Kyber768" in kems)
        has_sig = ("ML-DSA-65" in sigs) or ("Dilithium3" in sigs)
        return bool(has_kem and has_sig)
    except (ImportError, Exception):
        return False


def derive_session_key(
    x25519_shared: bytes,
    kem_shared: bytes | None = None,
    context_info: bytes = b"AI_DRONE_SWARM_PQC_SESSION",
) -> bytes:
    """Derives a 256-bit AES-GCM session key using HKDF-SHA256 from combined secrets."""
    combined = x25519_shared + (kem_shared or b"")
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=context_info,
    )
    return hkdf.derive(combined)


@dataclass
class Session:
    """Manages an authenticated, encrypted symmetric communication channel with a peer.

    Implements:
    - AES-256-GCM authenticated encryption
    - Monotonic packet counter
    - 64-packet sliding replay window
    - Time-based and counter-based key rotation triggers
    """

    session_id: str
    peer_id: str
    mode: str                    # "hybrid" or "classical"
    kem_algorithm: str           # "ML-KEM-768", "Kyber768", or "X25519"
    sig_algorithm: str           # "ML-DSA-65", "Dilithium3", or "Ed25519"
    aes_key: bytes
    created_at: float = field(default_factory=time.time)
    max_age_s: float = 3600.0    # 1 hour default
    max_messages: int = 1000     # Rekey after 1000 sends
    send_counter: int = 0
    highest_recv_seq: int = -1
    _received_seqs: set[int] = field(default_factory=set)

    @property
    def is_hybrid(self) -> bool:
        """Returns True if the session was negotiated using hybrid PQ KEM."""
        return self.mode == "hybrid"

    @property
    def needs_rekey(self) -> bool:
        """Indicates whether the session has expired and requires key rotation."""
        age = time.time() - self.created_at
        return (age >= self.max_age_s) or (self.send_counter >= self.max_messages)

    def is_replay(self, seq: int) -> bool:
        """Checks whether sequence number is a duplicate or outside the 64-packet window."""
        if seq < 0:
            return True
        if self.highest_recv_seq < 0:
            return False
        # If sequence number is older than window (highest - 64)
        if seq <= self.highest_recv_seq - 64:
            return True
        # If sequence number was already received
        return seq in self._received_seqs

    def mark_received(self, seq: int) -> None:
        """Records sequence number inside the sliding replay window."""
        if seq > self.highest_recv_seq:
            self.highest_recv_seq = seq
        self._received_seqs.add(seq)
        # Prune entries falling behind the 64-packet window
        min_allowed = self.highest_recv_seq - 64
        self._received_seqs = {s for s in self._received_seqs if s > min_allowed}

    def encrypt(self, plaintext: bytes, aad: bytes = b"") -> bytes:
        """Encrypts plaintext with AES-256-GCM using monotonic counter for unique nonce."""
        if not isinstance(plaintext, bytes):
            raise TypeError("Plaintext must be bytes.")

        seq = self.send_counter
        self.send_counter += 1

        # Construct 12-byte nonce: 4 bytes random prefix + 8 bytes big-endian counter
        # Guarantees no nonce is ever reused with this AES key
        nonce_prefix = os.urandom(4)
        nonce = nonce_prefix + struct.pack(">Q", seq)

        # Additional authenticated data includes session ID and sequence number
        full_aad = self.session_id.encode("utf-8") + struct.pack(">Q", seq) + aad

        aesgcm = AESGCM(self.aes_key)
        ciphertext_and_tag = aesgcm.encrypt(nonce, plaintext, full_aad)

        # Wire framing format:
        # [2 bytes sess_len] + [sess_id] + [8 bytes seq] + [12 bytes nonce] + [ciphertext+tag]
        sess_bytes = self.session_id.encode("utf-8")
        return struct.pack(">H", len(sess_bytes)) + sess_bytes + struct.pack(">Q", seq) + nonce + ciphertext_and_tag

    def decrypt(self, framed_bytes: bytes, aad: bytes = b"") -> bytes:
        """Decrypts and authenticates packet, enforcing the 64-message sliding replay window."""
        if len(framed_bytes) < 22:  # 2 len + 8 seq + 12 nonce
            raise SecurityError("Framed packet too short.")

        sess_len = struct.unpack(">H", framed_bytes[:2])[0]
        if len(framed_bytes) < 2 + sess_len + 20:
            raise SecurityError("Framed packet too short for session ID and headers.")

        packet_sess_id = framed_bytes[2 : 2 + sess_len].decode("utf-8", errors="replace")
        if packet_sess_id != self.session_id:
            raise SecurityError(f"Session ID mismatch: expected {self.session_id}, got {packet_sess_id}")

        offset = 2 + sess_len
        seq = struct.unpack(">Q", framed_bytes[offset : offset + 8])[0]
        nonce = framed_bytes[offset + 8 : offset + 20]
        ciphertext_and_tag = framed_bytes[offset + 20 :]

        # Enforce replay protection window
        if self.is_replay(seq):
            raise ReplayError(f"Replayed or duplicate packet rejected (seq={seq}, window_highest={self.highest_recv_seq}).")

        full_aad = self.session_id.encode("utf-8") + struct.pack(">Q", seq) + aad
        aesgcm = AESGCM(self.aes_key)

        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext_and_tag, full_aad)
        except InvalidTag as err:
            raise SecurityError("Ciphertext integrity verification failed (tampered packet).") from err

        # Mark sequence number as received only after successful decryption
        self.mark_received(seq)
        return plaintext


class HandshakeManager:
    """Orchestrates 2-message authenticated key exchanges between swarm nodes."""

    @staticmethod
    def create_init_message(
        initiator: Identity,
        responder_id: str,
        prefer_pq: bool = True,
        require_pq: bool = False,
    ) -> tuple[dict[str, Any], Any]:
        """Creates HandshakeInit message.

        Returns:
            (init_message_dict, ephemeral_context)
        """
        is_pq_kem, kem_algo = get_supported_kem_algorithm(prefer_pq=prefer_pq)

        if require_pq and not is_pq_kem:
            raise PQUnavailableError("require_pq=True but post-quantum KEM is not available.")

        # 1. Ephemeral X25519
        x25519_priv = X25519PrivateKey.generate()
        x25519_pub = x25519_priv.public_key().public_bytes_raw()

        # 2. Ephemeral ML-KEM if supported
        kem_obj = None
        kem_pub_bytes = None
        if is_pq_kem:
            try:
                import oqs  # type: ignore[import-untyped]
                kem_obj = oqs.KeyEncapsulation(kem_algo)
                kem_pub_bytes = bytes(kem_obj.generate_keypair())
            except Exception:
                if require_pq:
                    raise PQUnavailableError(f"Failed to generate {kem_algo} keypair under require_pq.")
                is_pq_kem = False
                kem_algo = "X25519"

        # 3. Sign initiation transcript
        transcript = (
            b"HANDSHAKE_INIT:"
            + initiator.peer_id.encode("utf-8")
            + b":"
            + responder_id.encode("utf-8")
            + b":"
            + x25519_pub
            + b":"
            + (kem_pub_bytes or b"")
        )
        sig = initiator.sign(transcript)

        msg = {
            "type": "HANDSHAKE_INIT",
            "initiator_id": initiator.peer_id,
            "responder_id": responder_id,
            "kem_algo": kem_algo if is_pq_kem else "NONE",
            "x25519_pub": base64.b64encode(x25519_pub).decode("ascii"),
            "kem_pub": base64.b64encode(kem_pub_bytes).decode("ascii") if kem_pub_bytes else None,
            "sig_algo": initiator.algorithm,
            "signature": base64.b64encode(sig).decode("ascii"),
            "timestamp": time.time(),
        }

        context = {
            "x25519_priv": x25519_priv,
            "kem_obj": kem_obj,
            "kem_algo": kem_algo if is_pq_kem else "NONE",
            "is_pq_kem": is_pq_kem,
            "initiator_id": initiator.peer_id,
            "responder_id": responder_id,
        }
        return msg, context

    @staticmethod
    def process_init_and_respond(
        init_msg: dict[str, Any],
        responder: Identity,
        get_peer_identity_fn: Any,
        require_pq: bool = False,
    ) -> tuple[dict[str, Any], Session]:
        """Responder handles HandshakeInit and returns (HandshakeResp, Session)."""
        initiator_id = init_msg["initiator_id"]
        peer_identity = get_peer_identity_fn(initiator_id)
        if peer_identity is None:
            raise AuthenticationError(f"Unknown peer identity for '{initiator_id}'.")

        # Verify signature
        x25519_init_pub = base64.b64decode(init_msg["x25519_pub"])
        kem_pub_bytes = base64.b64decode(init_msg["kem_pub"]) if init_msg.get("kem_pub") else None
        sig = base64.b64decode(init_msg["signature"])
        sig_algo = init_msg.get("sig_algo", peer_identity.algorithm)

        transcript = (
            b"HANDSHAKE_INIT:"
            + initiator_id.encode("utf-8")
            + b":"
            + responder.peer_id.encode("utf-8")
            + b":"
            + x25519_init_pub
            + b":"
            + (kem_pub_bytes or b"")
        )
        if not verify_signature(transcript, sig, peer_identity.public_key_bytes, sig_algo):
            raise AuthenticationError(f"Invalid HandshakeInit signature from '{initiator_id}'.")

        kem_algo = init_msg.get("kem_algo", "NONE")
        is_hybrid = kem_algo in ("ML-KEM-768", "Kyber768") and kem_pub_bytes is not None

        if require_pq and not (is_hybrid and peer_identity.is_post_quantum and responder.is_post_quantum):
            raise PQUnavailableError("require_pq=True but peer or algorithms are not post-quantum.")

        # 1. Ephemeral X25519 response
        resp_x25519_priv = X25519PrivateKey.generate()
        resp_x25519_pub = resp_x25519_priv.public_key().public_bytes_raw()
        x25519_shared = resp_x25519_priv.exchange(X25519PublicKey.from_public_bytes(x25519_init_pub))

        # 2. Encapsulate ML-KEM secret if hybrid
        kem_ct = None
        kem_shared = None
        if is_hybrid:
            try:
                import oqs  # type: ignore[import-untyped]
                with oqs.KeyEncapsulation(kem_algo) as client_kem:
                    kem_ct, kem_shared = client_kem.encap_secret(kem_pub_bytes)
                    kem_ct = bytes(kem_ct)
                    kem_shared = bytes(kem_shared)
            except Exception as err:
                if require_pq:
                    raise PQUnavailableError(f"KEM encapsulation failed under require_pq: {err}") from err
                is_hybrid = False

        # 3. Derive symmetric key
        session_id = f"SESS-{initiator_id}-{responder.peer_id}-{hashlib.sha256(x25519_shared).hexdigest()[:8]}"
        aes_key = derive_session_key(x25519_shared, kem_shared, context_info=session_id.encode("utf-8"))

        # 4. Sign response
        resp_transcript = (
            b"HANDSHAKE_RESP:"
            + responder.peer_id.encode("utf-8")
            + b":"
            + initiator_id.encode("utf-8")
            + b":"
            + resp_x25519_pub
            + b":"
            + (kem_ct or b"")
        )
        resp_sig = responder.sign(resp_transcript)

        resp_msg = {
            "type": "HANDSHAKE_RESP",
            "initiator_id": initiator_id,
            "responder_id": responder.peer_id,
            "x25519_pub": base64.b64encode(resp_x25519_pub).decode("ascii"),
            "kem_ct": base64.b64encode(kem_ct).decode("ascii") if kem_ct else None,
            "sig_algo": responder.algorithm,
            "signature": base64.b64encode(resp_sig).decode("ascii"),
            "session_id": session_id,
            "timestamp": time.time(),
        }

        session = Session(
            session_id=session_id,
            peer_id=initiator_id,
            mode="hybrid" if is_hybrid else "classical",
            kem_algorithm=kem_algo if is_hybrid else "X25519",
            sig_algorithm=responder.algorithm,
            aes_key=aes_key,
        )
        return resp_msg, session

    @staticmethod
    def process_resp_and_finalize(
        resp_msg: dict[str, Any],
        context: dict[str, Any],
        get_peer_identity_fn: Any,
        initiator_identity: Identity,
        require_pq: bool = False,
    ) -> Session:
        """Initiator handles HandshakeResp and finalizes Session."""
        responder_id = resp_msg["responder_id"]
        peer_identity = get_peer_identity_fn(responder_id)
        if peer_identity is None:
            raise AuthenticationError(f"Unknown peer identity for '{responder_id}'.")

        # Verify response signature
        resp_x25519_pub = base64.b64decode(resp_msg["x25519_pub"])
        kem_ct = base64.b64decode(resp_msg["kem_ct"]) if resp_msg.get("kem_ct") else None
        sig = base64.b64decode(resp_msg["signature"])
        sig_algo = resp_msg.get("sig_algo", peer_identity.algorithm)

        transcript = (
            b"HANDSHAKE_RESP:"
            + responder_id.encode("utf-8")
            + b":"
            + initiator_identity.peer_id.encode("utf-8")
            + b":"
            + resp_x25519_pub
            + b":"
            + (kem_ct or b"")
        )
        if not verify_signature(transcript, sig, peer_identity.public_key_bytes, sig_algo):
            raise AuthenticationError(f"Invalid HandshakeResp signature from '{responder_id}'.")

        # 1. Complete X25519
        x25519_priv = context["x25519_priv"]
        x25519_shared = x25519_priv.exchange(X25519PublicKey.from_public_bytes(resp_x25519_pub))

        # 2. Complete ML-KEM decapsulation if hybrid
        kem_obj = context.get("kem_obj")
        kem_shared = None
        is_hybrid = False
        if kem_obj and kem_ct:
            try:
                kem_shared = bytes(kem_obj.decap_secret(kem_ct))
                is_hybrid = True
            except Exception as err:
                if require_pq:
                    raise PQUnavailableError(f"KEM decapsulation failed under require_pq: {err}") from err

        if require_pq and not (is_hybrid and initiator_identity.is_post_quantum and peer_identity.is_post_quantum):
            raise PQUnavailableError("require_pq=True but established session is classical.")

        session_id = resp_msg.get("session_id") or f"SESS-{initiator_identity.peer_id}-{responder_id}-{hashlib.sha256(x25519_shared).hexdigest()[:8]}"
        aes_key = derive_session_key(x25519_shared, kem_shared, context_info=session_id.encode("utf-8"))

        return Session(
            session_id=session_id,
            peer_id=responder_id,
            mode="hybrid" if is_hybrid else "classical",
            kem_algorithm=context["kem_algo"] if is_hybrid else "X25519",
            sig_algorithm=initiator_identity.algorithm,
            aes_key=aes_key,
        )
