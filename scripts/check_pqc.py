#!/usr/bin/env python3
"""Check Post-Quantum Cryptography (PQC) environment and liboqs availability.

Inspects whether liboqs-python is available, lists enabled KEM and signature
mechanisms, selects preferred algorithms, and outputs a clear verdict line.
"""

import argparse
import sys
from typing import Any


def check_pqc(require_pq: bool = False) -> int:
    """Checks liboqs status and prints diagnostics.

    Returns:
        0 if successful (or if classical fallback is permitted without --require-pq).
        1 if --require-pq is set but post-quantum cryptography is not active.
    """
    oqs_module: Any = None
    oqs_imported = False
    kem_mechanisms: list[str] = []
    sig_mechanisms: list[str] = []
    selected_kem: str | None = None
    selected_sig: str | None = None
    pq_active = False

    try:
        import oqs  # type: ignore[import-untyped]
        oqs_module = oqs
        oqs_imported = True
    except (ImportError, ModuleNotFoundError, Exception):
        oqs_imported = False

    if oqs_imported and oqs_module is not None:
        try:
            kem_mechanisms = list(oqs_module.get_enabled_KEM_mechanisms())
            sig_mechanisms = list(oqs_module.get_enabled_sig_mechanisms())

            # Select KEM: Prefer ML-KEM-768, fallback to Kyber768
            if "ML-KEM-768" in kem_mechanisms:
                selected_kem = "ML-KEM-768"
            elif "Kyber768" in kem_mechanisms:
                selected_kem = "Kyber768"

            # Select Signature: Prefer ML-DSA-65, fallback to Dilithium3
            if "ML-DSA-65" in sig_mechanisms:
                selected_sig = "ML-DSA-65"
            elif "Dilithium3" in sig_mechanisms:
                selected_sig = "Dilithium3"

            if selected_kem and selected_sig:
                pq_active = True
        except Exception:
            pq_active = False

    print("==================================================")
    print("      POST-QUANTUM CRYPTOGRAPHY STATUS CHECK      ")
    print("==================================================")
    print(f"liboqs module available : {oqs_imported}")
    if oqs_imported:
        print(f"Enabled KEM count       : {len(kem_mechanisms)}")
        print(f"Enabled Signature count : {len(sig_mechanisms)}")
        print(f"Selected KEM            : {selected_kem or 'None'}")
        print(f"Selected Signature      : {selected_sig or 'None'}")
    else:
        print("Enabled KEM mechanisms  : None (liboqs-python not installed)")
        print("Enabled Signature mechs : None (liboqs-python not installed)")
        print("Selected KEM            : X25519 (classical ECDH)")
        print("Selected Signature      : Ed25519 (classical Edwards-curve)")
    print("--------------------------------------------------")

    if pq_active:
        print("VERDICT: POST-QUANTUM ACTIVE")
        print("==================================================")
        return 0
    else:
        print("VERDICT: CLASSICAL FALLBACK (not post-quantum)")
        print("==================================================")
        if require_pq:
            print("ERROR: --require-pq was specified, but post-quantum algorithms are not active.", file=sys.stderr)
            return 1
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PQC / liboqs status for AI Drone Swarm")
    parser.add_argument(
        "--require-pq",
        action="store_true",
        help="Exit with code 1 if post-quantum cryptography is not active.",
    )
    args = parser.parse_args()
    sys.exit(check_pqc(require_pq=args.require_pq))


if __name__ == "__main__":
    main()

