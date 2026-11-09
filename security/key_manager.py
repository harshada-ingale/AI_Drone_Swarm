"""Public Key Infrastructure (PKI) registry and local KeyManager."""

from typing import Any

from security.authentication import Identity
from security.pqc_security import HandshakeManager, SecurityError, Session


class PkiRegistry:
    """Registry of trusted public identities for swarm nodes and ground stations."""

    def __init__(self) -> None:
        self._identities: dict[str, Identity] = {}
        self._key_managers: dict[str, Any] = {}
        self._revoked: set[str] = set()

    def register(self, identity: Identity) -> None:
        """Registers a peer's identity in the registry."""
        if not isinstance(identity, Identity):
            raise TypeError("Expected an Identity instance.")
        if identity.peer_id in self._revoked:
            raise SecurityError(f"Cannot register revoked peer '{identity.peer_id}'.")
        self._identities[identity.peer_id] = identity

    def get(self, peer_id: str) -> Identity | None:
        """Retrieves an identity by peer_id if not revoked."""
        if peer_id in self._revoked:
            return None
        return self._identities.get(peer_id)

    def revoke(self, peer_id: str) -> None:
        """Revokes a peer identity permanently."""
        self._revoked.add(peer_id)
        self._identities.pop(peer_id, None)

    def is_revoked(self, peer_id: str) -> bool:
        """Checks if peer_id has been revoked."""
        return peer_id in self._revoked

    def is_registered(self, peer_id: str) -> bool:
        """Checks if peer_id is currently registered and active."""
        return (peer_id in self._identities) and (peer_id not in self._revoked)

    def list_peers(self) -> list[str]:
        """Lists all active (unrevoked) registered peer IDs."""
        return [pid for pid in self._identities if pid not in self._revoked]


class KeyManager:
    """Manages local node cryptographic identity, trusted PKI registry, and active sessions."""

    def __init__(self, local_identity: Identity | str, registry: PkiRegistry | None = None) -> None:
        if isinstance(local_identity, str):
            local_identity = Identity.generate(local_identity)
        elif not isinstance(local_identity, Identity):
            raise TypeError("local_identity must be an Identity instance or str peer_id.")
        self.local_identity = local_identity
        self.registry = registry if registry is not None else PkiRegistry()
        # Always register local identity into registry if not already present
        if not self.registry.is_registered(local_identity.peer_id) and not self.registry.is_revoked(local_identity.peer_id):
            self.registry.register(local_identity)
        self.registry._key_managers[local_identity.peer_id] = self
        self.sessions: dict[str, Session] = {}

    @property
    def peer_id(self) -> str:
        """Local node ID."""
        return self.local_identity.peer_id

    def get_session(self, peer_id: str) -> Session | None:
        """Returns the active session with peer_id, or None if missing/revoked."""
        if self.registry.is_revoked(peer_id):
            self.sessions.pop(peer_id, None)
            return None
        if peer_id not in self.sessions:
            peer_km = self.registry._key_managers.get(peer_id)
            if peer_km is not None and peer_km is not self:
                self.establish_session_with_peer(peer_km)
        return self.sessions.get(peer_id)

    def set_session(self, peer_id: str, session: Session) -> None:
        """Sets an active session for peer_id."""
        if self.registry.is_revoked(peer_id):
            raise SecurityError(f"Cannot set session for revoked peer '{peer_id}'.")
        if not isinstance(session, Session):
            raise TypeError("Expected a Session instance.")
        self.sessions[peer_id] = session

    def has_active_session(self, peer_id: str) -> bool:
        """Returns True if a valid session exists that does not need rekey."""
        sess = self.get_session(peer_id)
        if sess is None:
            return False
        return not sess.needs_rekey

    def revoke_peer(self, peer_id: str) -> None:
        """Purges active sessions and revokes peer identity in PKI registry."""
        self.sessions.pop(peer_id, None)
        self.registry.revoke(peer_id)

    def clear_sessions(self) -> None:
        """Clears all active sessions."""
        self.sessions.clear()

    def establish_session_with_peer(
        self,
        peer_km: "KeyManager",
        prefer_pq: bool = True,
        require_pq: bool = False,
    ) -> tuple[Session, Session]:
        """Performs a 2-message authenticated handshake between this node and peer_km.

        Returns:
            (local_session, peer_session)
        """
        # Ensure mutual registration
        if not self.registry.is_registered(peer_km.peer_id):
            self.registry.register(peer_km.local_identity)
        if not peer_km.registry.is_registered(self.peer_id):
            peer_km.registry.register(self.local_identity)

        # 1. Initiator (self) creates init message
        init_msg, ctx = HandshakeManager.create_init_message(
            initiator=self.local_identity,
            responder_id=peer_km.peer_id,
            prefer_pq=prefer_pq,
            require_pq=require_pq,
        )

        # 2. Responder (peer_km) processes init and creates response message & session
        resp_msg, peer_sess = HandshakeManager.process_init_and_respond(
            init_msg=init_msg,
            responder=peer_km.local_identity,
            get_peer_identity_fn=peer_km.registry.get,
            require_pq=require_pq,
        )

        # 3. Initiator (self) processes response and creates initiator session
        local_sess = HandshakeManager.process_resp_and_finalize(
            resp_msg=resp_msg,
            context=ctx,
            get_peer_identity_fn=self.registry.get,
            initiator_identity=self.local_identity,
            require_pq=require_pq,
        )

        # Save both sessions
        self.set_session(peer_km.peer_id, local_sess)
        peer_km.set_session(self.peer_id, peer_sess)

        return local_sess, peer_sess
