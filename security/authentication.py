"""Authentication and identity management supporting ML-DSA-65 and Ed25519 fallback."""

from dataclasses import dataclass
from typing import Any
import os

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def get_supported_sig_algorithm(prefer_pq: bool = True) -> tuple[bool, str]:
    """Detects if liboqs signature mechanism is available.

    Returns:
        (is_pq, algorithm_name)
    """
    if prefer_pq:
        try:
            import oqs  # type: ignore[import-untyped]
            enabled = oqs.get_enabled_sig_mechanisms()
            if "ML-DSA-65" in enabled:
                return True, "ML-DSA-65"
            if "Dilithium3" in enabled:
                return True, "Dilithium3"
        except (ImportError, Exception):
            pass
    return False, "Ed25519"


def verify_signature(
    message: bytes,
    signature: bytes,
    public_key_bytes: bytes,
    algorithm: str,
) -> bool:
    """Verifies a digital signature using the specified algorithm.

    Never invents cryptography. Uses liboqs for ML-DSA-65 / Dilithium3,
    and cryptography.hazmat for Ed25519.
    """
    if not message or not signature or not public_key_bytes:
        return False

    if algorithm in ("ML-DSA-65", "Dilithium3"):
        try:
            import oqs  # type: ignore[import-untyped]
            with oqs.Signature(algorithm) as verifier:
                return bool(verifier.verify(message, signature, public_key_bytes))
        except Exception:
            return False

    if algorithm == "Ed25519":
        try:
            pub = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            pub.verify(signature, message)
            return True
        except (InvalidSignature, ValueError, TypeError, Exception):
            return False

    return False


@dataclass
class Identity:
    """Represents a drone or station cryptographic identity."""

    peer_id: str
    algorithm: str
    public_key_bytes: bytes
    _private_key: Any  # OQS Signature or Ed25519PrivateKey

    @property
    def is_post_quantum(self) -> bool:
        """Returns True only if algorithm is a genuine post-quantum mechanism."""
        return self.algorithm in ("ML-DSA-65", "Dilithium3")

    def sign(self, message: bytes) -> bytes:
        """Signs message bytes using the private key."""
        if not message:
            raise ValueError("Cannot sign empty message.")

        if self.algorithm in ("ML-DSA-65", "Dilithium3"):
            try:
                import oqs  # type: ignore[import-untyped]
                # In liboqs-python, the signature object with private key signs
                if hasattr(self._private_key, "sign"):
                    return bytes(self._private_key.sign(message))
            except Exception as err:
                raise RuntimeError(f"ML-DSA signing failed: {err}") from err

        if self.algorithm == "Ed25519":
            if isinstance(self._private_key, Ed25519PrivateKey):
                return self._private_key.sign(message)

        raise RuntimeError(f"Unsupported signing algorithm: {self.algorithm}")

    def verify(
        self,
        message: bytes,
        signature: bytes,
        public_key_bytes: bytes | None = None,
    ) -> bool:
        """Verifies signature against public key (self or provided)."""
        pub_bytes = public_key_bytes or self.public_key_bytes
        return verify_signature(
            message=message,
            signature=signature,
            public_key_bytes=pub_bytes,
            algorithm=self.algorithm,
        )

    @classmethod
    def generate(cls, peer_id: str, prefer_pq: bool = True) -> "Identity":
        """Generates a new Identity for peer_id using best available algorithm."""
        if not peer_id or not isinstance(peer_id, str):
            raise ValueError("peer_id must be a non-empty string.")

        is_pq, algo_name = get_supported_sig_algorithm(prefer_pq=prefer_pq)

        if is_pq:
            try:
                import oqs  # type: ignore[import-untyped]
                sig_obj = oqs.Signature(algo_name)
                pub_bytes = sig_obj.generate_keypair()
                return cls(
                    peer_id=peer_id,
                    algorithm=algo_name,
                    public_key_bytes=bytes(pub_bytes),
                    _private_key=sig_obj,
                )
            except Exception:
                # Fall back to Ed25519 if OQS keypair generation fails
                pass

        # Classical fallback: Ed25519
        priv = Ed25519PrivateKey.generate()
        pub_bytes = priv.public_key().public_bytes_raw()
        return cls(
            peer_id=peer_id,
            algorithm="Ed25519",
            public_key_bytes=pub_bytes,
            _private_key=priv,
        )

