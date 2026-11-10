# Phase 3: Post-Quantum Security & Swarm Integration Architecture

This document describes the cryptographic security architecture, self-healing coordination, vision emergency alert pipeline, and deployment guide for the **AI Drone Swarm** project.

---

## 1. Executive Summary & Security Philosophy

The AI Drone Swarm system coordinates autonomous aerial drones and Ground Control Stations (GCS) operating in contested or communications-degraded tactical environments.

### Core Principles:
1. **Never Invent Cryptography**: All cryptographic operations rely strictly on vetted standard implementations (`cryptography.hazmat` and standard `liboqs` bindings). No custom XOR or toy ciphers.
2. **Truthful Operational Status**: The system strictly reports `CLASSICAL FALLBACK (not post-quantum)` when quantum-safe primitives are absent, reserving `POST-QUANTUM ACTIVE` solely for genuine ML-KEM / ML-DSA execution.
3. **Defense-in-Depth & Zero-Trust Relays**: Intermediate mesh relays forward encrypted payloads verbatim without decrypting or retaining cryptographic keys.
4. **Fail-Closed Security Policy**: Critical configurations with `require_pq = True` abort early with `PQUnavailableError` rather than silently degrading security when quantum-safe primitives are missing.

---

## 2. Cryptographic Specifications

| Component | Post-Quantum Mechanism | Classical Fallback Mechanism | Key Size / Wire Framing |
| :--- | :--- | :--- | :--- |
| **Digital Signatures** | **ML-DSA-65** (NIST FIPS 204 / Dilithium3) | **Ed25519** (RFC 8032) | Public key: 1952 B (ML-DSA) / 32 B (Ed25519) |
| **Key Encapsulation (KEM)**| **ML-KEM-768** (NIST FIPS 203 / Kyber768) | **X25519** (RFC 7748 ECDH) | Hybrid combined shared secret |
| **Key Derivation (KDF)** | **HKDF-SHA256** (RFC 5869) | **HKDF-SHA256** (RFC 5869) | 256-bit symmetric session key |
| **Authenticated Cipher** | **AES-256-GCM** (NIST SP 800-38D) | **AES-256-GCM** (NIST SP 800-38D) | 12-byte nonce (4B random + 8B counter), 16B tag |
| **Replay Protection** | 64-packet sliding window | 64-packet sliding window | Replay rejection on duplicate or expired seq |

---

## 3. Communication Protocols & Handshake Workflow

### 3.1 Two-Message Authenticated Handshake

Nodes establish mutual session keys using a signed ephemeral exchange:

```text
Initiator (Drone)                                     Responder (GS)
       |                                                    |
       |  1. HANDSHAKE_INIT                                 |
       |  [ephemeral_x25519, ephemeral_kem, signature]      |
       |--------------------------------------------------->|
       |                                                    |
       |                                                    | (Verifies initiator signature)
       |                                                    | (Generates ephemeral X25519 & KEM ciphertext)
       |                                                    | (Derives AES-256-GCM session key via HKDF)
       |  2. HANDSHAKE_RESP                                 |
       |  [ephemeral_x25519, kem_ciphertext, signature]     |
       |<---------------------------------------------------|
       |                                                    |
       | (Verifies responder signature)                     |
       | (Decapsulates KEM secret & completes X25519)       |
       | (Derives identical AES-256-GCM session key)        |
```

### 3.2 Wire Framing Format

Every encrypted payload frame transmitted across relay drones uses the following wire framing:

```text
+-----------------------+--------------------+------------------+---------------------+-------------------+
| 2 Bytes: sess_id_len  | N Bytes: sess_id   | 8 Bytes: seq_num | 12 Bytes: GCM nonce | Ciphertext + Tag  |
+-----------------------+--------------------+------------------+---------------------+-------------------+
```

- **Monotonic Sequence Counter (`seq_num`)**: Big-endian counter incremented per message to ensure no nonce is ever reused.
- **64-Packet Anti-Replay Sliding Window**: Enforces ordering and rejects replayed or duplicate frames.

---

## 4. Vision Emergency Alert Pipeline

When an edge drone discovers a tactical emergency (e.g. fire hazard or stranded personnel), the alert is dispatched over the mesh:

1. **Origin Sealing**: Origin drone seals payload dictionary via `SecureMessenger.send("GS", payload)`.
2. **Hop-by-Hop Relay Forwarding**: Relay nodes (e.g., `D4`, `D3`, `D5`, `D6`) receive raw bytes and forward verbatim using `SecurityAdapter.forward_relay_hop()`. Relays **never decrypt** traffic.
3. **Mid-Transit Rerouting**: If a relay drone suffers sudden failure or radio disconnection while a message is in flight:
   - The controller isolates the failed node.
   - The controller calculates a detour path bypassing the failed node.
   - The **identical ciphertext** is forwarded along the new route without re-encrypting or compromising origin nonces.
4. **Destination Unsealing**: Ground Station unseals ciphertext via `SecureMessenger.receive()`, authenticating GCM integrity and verifying freshness.

---

## 5. Security Compromise & Signed Operator Clearance

When an anomaly or GPS spoofing attack is detected:
1. The drone is immediately marked `ISOLATED` and `is_compromised = True`.
2. The drone is added to Dijkstra's routing exclusion set.
3. Telemetry claiming healthy status is rejected while compromised.
4. Reintegration requires cryptographically signed authorization from an authorized Swarm Security Officer (`OP-*`).

### Clearance Token Structure:
```json
{
  "drone_id": "D6",
  "action": "CLEAR_COMPROMISED",
  "operator_id": "OP-CMD-1",
  "timestamp": 1794230800.0,
  "expires_at": 1794231100.0,
  "nonce": "CLEARANCE-NONCE-847291",
  "signature": "<Base64 encoded digital signature>"
}
```

The signature covers the canonical transcript:
`CLEARANCE:<drone_id>:<action>:<operator_id>:<timestamp>:<expires_at>:<nonce>`

Non-replayed nonces and expiration timestamps are strictly enforced. All audit logs suppress private keys, signatures, and sensitive tokens.

---

## 6. Route Restoration Hold-Down Timer

To prevent route flapping during intermittent or oscillating radio links:
- When an isolated drone recovers, its `recovery_timestamp` is recorded.
- Detour routes remain active until `restore_hold_down_s` (default: 5.0 seconds) has elapsed:
  $$\Delta t = t_{\text{current}} - t_{\text{recovery}} \ge 5.0\,\text{s}$$
- After the hold-down window elapses, the controller safely restores the optimal low-latency path.

---

## 7. Diagnostic Tools & CLI

### 7.1 Post-Quantum Diagnostics (`scripts/check_pqc.py`)

Check status:
```bash
python scripts/check_pqc.py
```

Enforce post-quantum requirement (exits with code 1 if classical):
```bash
python scripts/check_pqc.py --require-pq
```

### 7.2 Interactive Self-Healing Demonstration (`run_self_healing_demo.py`)

Run automated demonstration:
```bash
python run_self_healing_demo.py --fast
```

Run step-by-step interactive simulation:
```bash
python run_self_healing_demo.py --interactive
```

---

## 8. Post-Quantum Library Installation (Optional WSL2 / Linux)

On Debian/Ubuntu or WSL2 environments with C toolchains available:

```bash
# 1. Install liboqs build dependencies
sudo apt-get update && sudo apt-get install -y cmake gcc ninja-build libssl-dev

# 2. Build and install liboqs C library
git clone --depth=1 https://github.com/open-quantum-safe/liboqs.git
cd liboqs
mkdir build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON ..
ninja
sudo ninja install
sudo ldconfig

# 3. Install Python bindings
pip install liboqs-python
```

When `liboqs-python` is present, the system automatically activates **ML-KEM-768** and **ML-DSA-65**. Without it, the system falls back truthfully to **X25519** and **Ed25519**.

