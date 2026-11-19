from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Any


# =========================================================
# CONSTANTS
# =========================================================

MASTER_KEY_SIZE = 32          # 256-bit master key
SESSION_NONCE_SIZE = 16       # 128-bit public nonce
HKDF_HASH_LENGTH = 32         # SHA-256 output size


# =========================================================
# DATA STRUCTURE
# =========================================================

@dataclass(frozen=True)
class DerivedKeyMaterial:
    """
    Key material derived from one 256-bit master key and one
    public session nonce.
    """

    chaos_params: dict[str, float]
    permutation_salt: int
    baker_salt: int
    global_diffusion_salt: int
    plaintext_context_key: bytes
    authentication_key: bytes
    session_nonce: bytes


# =========================================================
# MASTER-KEY HANDLING
# =========================================================

def generate_master_key() -> bytes:
    """
    Generate a cryptographically secure random 256-bit master key.
    """
    return secrets.token_bytes(MASTER_KEY_SIZE)


def master_key_to_hex(master_key: bytes) -> str:
    """
    Convert a 32-byte master key to a 64-character hexadecimal string.
    """
    key = _validate_master_key(master_key)
    return key.hex()


def master_key_from_hex(master_key_hex: str) -> bytes:
    """
    Convert a 64-character hexadecimal string into a 32-byte key.
    """
    if not isinstance(master_key_hex, str):
        raise TypeError("master_key_hex must be a string.")

    cleaned = master_key_hex.strip()

    if len(cleaned) != MASTER_KEY_SIZE * 2:
        raise ValueError(
            "The master key must contain exactly 64 hexadecimal characters."
        )

    try:
        key = bytes.fromhex(cleaned)
    except ValueError as exc:
        raise ValueError(
            "The master key contains invalid hexadecimal characters."
        ) from exc

    return _validate_master_key(key)


def _validate_master_key(master_key: bytes) -> bytes:
    """
    Validate that the supplied master key is exactly 256 bits.
    """
    if not isinstance(master_key, bytes):
        raise TypeError("master_key must be bytes.")

    if len(master_key) != MASTER_KEY_SIZE:
        raise ValueError(
            f"master_key must be exactly {MASTER_KEY_SIZE} bytes "
            f"({MASTER_KEY_SIZE * 8} bits)."
        )

    return master_key


def generate_session_nonce() -> bytes:
    """
    Generate a public random nonce for one encryption session.

    The nonce is not secret and must be stored with the ciphertext.
    """
    return secrets.token_bytes(SESSION_NONCE_SIZE)


# =========================================================
# HKDF-SHA256
# =========================================================

def _hkdf_extract(
    salt: bytes,
    input_key_material: bytes,
) -> bytes:
    """
    HKDF-Extract using HMAC-SHA256.
    """
    if not salt:
        salt = bytes(HKDF_HASH_LENGTH)

    return hmac.new(
        salt,
        input_key_material,
        hashlib.sha256,
    ).digest()


def _hkdf_expand(
    pseudorandom_key: bytes,
    info: bytes,
    length: int,
) -> bytes:
    """
    HKDF-Expand using HMAC-SHA256.
    """
    if length <= 0:
        raise ValueError("length must be greater than zero.")

    if length > 255 * HKDF_HASH_LENGTH:
        raise ValueError("Requested HKDF output is too long.")

    output = bytearray()
    previous = b""
    counter = 1

    while len(output) < length:
        previous = hmac.new(
            pseudorandom_key,
            previous + info + bytes([counter]),
            hashlib.sha256,
        ).digest()

        output.extend(previous)
        counter += 1

    return bytes(output[:length])


def derive_bytes(
    master_key: bytes,
    session_nonce: bytes,
    context: str,
    length: int,
) -> bytes:
    """
    Derive context-separated bytes from the master key.

    Different context strings produce independent subkeys.
    """
    key = _validate_master_key(master_key)

    if not isinstance(session_nonce, bytes):
        raise TypeError("session_nonce must be bytes.")

    if len(session_nonce) != SESSION_NONCE_SIZE:
        raise ValueError(
            f"session_nonce must be exactly {SESSION_NONCE_SIZE} bytes."
        )

    if not isinstance(context, str) or not context:
        raise ValueError("context must be a non-empty string.")

    pseudorandom_key = _hkdf_extract(
        salt=session_nonce,
        input_key_material=key,
    )

    return _hkdf_expand(
        pseudorandom_key=pseudorandom_key,
        info=(
            b"MedicalImageEncryption-v1|"
            + context.encode("utf-8")
        ),
        length=length,
    )


# =========================================================
# STAGE-SPECIFIC DERIVATION
# =========================================================

def _bytes_to_unit_interval(data: bytes) -> float:
    """
    Convert bytes deterministically to a value in [0, 1).
    """
    integer = int.from_bytes(
        data,
        byteorder="big",
        signed=False,
    )

    return integer / float(1 << (8 * len(data)))


def derive_chaos_params(
    master_key: bytes,
    session_nonce: bytes,
    base_params: dict[str, Any],
) -> dict[str, float]:
    """
    Derive secret initial conditions while keeping the published
    control parameters fixed.

    The perturbations are intentionally small so the operating point
    remains close to the validated initial condition 0.3.
    """
    required = {
        "x0",
        "y0",
        "z0",
        "u0",
        "a",
        "b",
        "c",
        "gamma",
        "d",
        "alpha",
        "h",
    }

    missing = required.difference(base_params.keys())

    if missing:
        raise ValueError(
            "base_params is missing: "
            + ", ".join(sorted(missing))
        )

    raw = derive_bytes(
        master_key=master_key,
        session_nonce=session_nonce,
        context="chaotic-initial-conditions",
        length=32,
    )

    unit_values = [
        _bytes_to_unit_interval(raw[start:start + 8])
        for start in range(0, 32, 8)
    ]

    # Secret perturbations in approximately [-5e-7, +5e-7].
    # This preserves operation near the validated 0.3 point.
    scale = 1.0e-6
    offsets = [
        (value - 0.5) * scale
        for value in unit_values
    ]

    return {
        "x0": float(base_params["x0"]) + offsets[0],
        "y0": float(base_params["y0"]) + offsets[1],
        "z0": float(base_params["z0"]) + offsets[2],
        "u0": float(base_params["u0"]) + offsets[3],
        "a": float(base_params["a"]),
        "b": float(base_params["b"]),
        "c": float(base_params["c"]),
        "gamma": float(base_params["gamma"]),
        "d": float(base_params["d"]),
        "alpha": float(base_params["alpha"]),
        "h": float(base_params["h"]),
    }


def _derive_uint64(
    master_key: bytes,
    session_nonce: bytes,
    context: str,
) -> int:
    """
    Derive one unsigned 64-bit integer.
    """
    raw = derive_bytes(
        master_key=master_key,
        session_nonce=session_nonce,
        context=context,
        length=8,
    )

    return int.from_bytes(
        raw,
        byteorder="big",
        signed=False,
    )


def derive_key_material(
    master_key: bytes,
    session_nonce: bytes,
    base_chaos_params: dict[str, Any],
) -> DerivedKeyMaterial:
    """
    Derive all stage-specific key material from one master key.
    """
    chaos_params = derive_chaos_params(
        master_key=master_key,
        session_nonce=session_nonce,
        base_params=base_chaos_params,
    )

    permutation_salt = _derive_uint64(
        master_key,
        session_nonce,
        "global-permutation-salt",
    )

    baker_salt = _derive_uint64(
        master_key,
        session_nonce,
        "parallel-baker-salt",
    )

    global_diffusion_salt = _derive_uint64(
        master_key,
        session_nonce,
        "global-diffusion-salt",
    )

    plaintext_context_key = derive_bytes(
        master_key=master_key,
        session_nonce=session_nonce,
        context="plaintext-context-key",
        length=32,
    )

    authentication_key = derive_bytes(
        master_key=master_key,
        session_nonce=session_nonce,
        context="ciphertext-authentication-key",
        length=32,
    )

    return DerivedKeyMaterial(
        chaos_params=chaos_params,
        permutation_salt=permutation_salt,
        baker_salt=baker_salt,
        global_diffusion_salt=global_diffusion_salt,
        plaintext_context_key=plaintext_context_key,
        authentication_key=authentication_key,
        session_nonce=session_nonce,
    )


# =========================================================
# SELF-TEST
# =========================================================

if __name__ == "__main__":
    BASE_CHAOS_PARAMS = {
        "x0": 0.3,
        "y0": 0.3,
        "z0": 0.3,
        "u0": 0.3,
        "a": 35.0,
        "b": 3.0,
        "c": 12.0,
        "gamma": 28.0,
        "d": 0.5,
        "alpha": 0.97,
        "h": 0.001,
    }

    print("=" * 72)
    print("MASTER KEY SCHEDULE TEST")
    print("=" * 72)

    master_key = generate_master_key()
    nonce = generate_session_nonce()

    material_1 = derive_key_material(
        master_key,
        nonce,
        BASE_CHAOS_PARAMS,
    )

    material_1_repeat = derive_key_material(
        master_key,
        nonce,
        BASE_CHAOS_PARAMS,
    )

    other_nonce = generate_session_nonce()

    material_2 = derive_key_material(
        master_key,
        other_nonce,
        BASE_CHAOS_PARAMS,
    )

    print("Master key length       :", len(master_key))
    print("Session nonce length    :", len(nonce))
    print(
        "Deterministic derivation:",
        material_1 == material_1_repeat,
    )
    print(
        "Different nonce differs :",
        material_1 != material_2,
    )
    print(
        "Permutation salt        :",
        material_1.permutation_salt,
    )
    print(
        "Baker salt              :",
        material_1.baker_salt,
    )
    print(
        "Global diffusion salt   :",
        material_1.global_diffusion_salt,
    )
    print(
        "Plaintext context key bytes:",
        len(material_1.plaintext_context_key),
    )
    print(
        "Authentication key bytes:",
        len(material_1.authentication_key),
    )
    print(
        "Derived x0              :",
        f"{material_1.chaos_params['x0']:.17g}",
    )
    print("=" * 72)
    print("Key schedule test completed.")
