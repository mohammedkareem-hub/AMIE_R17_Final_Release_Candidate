from __future__ import annotations

"""
Independent reproducible Python implementation of Afzal et al. (P21):
"A hybrid encryption model for medical image security in IoT healthcare"
Discover Computing (2025) 28:321.

Implemented pipeline:
    1) Logistic-map permutation
    2) SHA-256 byte-wise XOR confusion
    3) Dynamic DNA rule selection
    4) Fibonacci-based circular DNA shift
    5) Exact inverse decryption

Important source ambiguities are exposed as configuration switches instead
of being silently "fixed":
    - The paper gives alpha in (3.9, 4) and x0 in (0, 1) but does not fix
      one concrete pair for all experiments.
    - The prose often writes SHA256(K || i), while the algorithm box first
      derives HK = SHA256(K). This implementation defaults to HK.
    - The worked ROM example formats the indexed hash as "key_i".
    - Eq. (2) and the worked example disagree on the Fibonacci starting
      convention. The default follows the worked example F(0)=1.

The implementation is intended for controlled same-hardware evaluation.
"""

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import numpy as np
from numpy.typing import NDArray


HashSource = Literal["K", "HK"]
IndexEncoding = Literal["underscore_decimal", "raw_u64_be"]
FibMode = Literal["worked_example", "equation"]


DNA_RULES = (
    ("A", "C", "G", "T"),  # R0: 00 01 10 11
    ("A", "G", "C", "T"),  # R1
    ("C", "A", "T", "G"),  # R2
    ("C", "T", "A", "G"),  # R3
    ("G", "A", "T", "C"),  # R4
    ("G", "T", "A", "C"),  # R5
    ("T", "C", "G", "A"),  # R6
    ("T", "G", "C", "A"),  # R7
)


@dataclass(frozen=True)
class AfzalP21Config:
    # Independent reproducibility choices inside the source-specified ranges.
    alpha: float = 3.99
    x0: float = 0.123456789

    # Algorithm-box interpretation.
    stream_hash_source: HashSource = "HK"
    dna_rule_hash_source: HashSource = "HK"

    # Concrete worked example uses "key_i".
    index_encoding: IndexEncoding = "underscore_decimal"

    # Concrete worked example states F(0)=1.
    fib_mode: FibMode = "worked_example"

    def validate(self) -> None:
        if not (3.9 < self.alpha < 4.0):
            raise ValueError("alpha must satisfy 3.9 < alpha < 4.0")
        if not (0.0 < self.x0 < 1.0):
            raise ValueError("x0 must satisfy 0 < x0 < 1")


@dataclass
class AfzalP21Ciphertext:
    image: NDArray[np.uint8]
    shape: tuple[int, int]


def derive_256bit_key(seed: str | bytes) -> bytes:
    """
    Paper Algorithm 1 permits a user input / entropy seed S and forms
    a 256-bit secret key using SHA-256.
    """
    if isinstance(seed, str):
        seed_bytes = seed.encode("utf-8")
    else:
        seed_bytes = bytes(seed)
    return sha256(seed_bytes).digest()


def _hash_material(K: bytes, source: HashSource) -> bytes:
    if source == "K":
        return K
    if source == "HK":
        return sha256(K).digest()
    raise ValueError(f"Unsupported hash source: {source}")


def _indexed_digest(
    material: bytes,
    index: int,
    mode: IndexEncoding,
) -> bytes:
    if mode == "underscore_decimal":
        # Follows the paper's explicit worked example:
        # SHA256("securekey123_0")
        msg = material + b"_" + str(index).encode("ascii")
    elif mode == "raw_u64_be":
        msg = material + int(index).to_bytes(8, "big", signed=False)
    else:
        raise ValueError(f"Unsupported index encoding: {mode}")

    return sha256(msg).digest()


def _keystream_and_rules(
    n: int,
    K: bytes,
    config: AfzalP21Config,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    stream_material = _hash_material(K, config.stream_hash_source)
    rule_material = _hash_material(K, config.dna_rule_hash_source)

    stream = np.empty(n, dtype=np.uint8)
    rules = np.empty(n, dtype=np.uint8)

    same_material = (
        stream_material == rule_material
        and config.stream_hash_source == config.dna_rule_hash_source
    )

    for i in range(n):
        d_stream = _indexed_digest(
            stream_material,
            i,
            config.index_encoding,
        )
        stream[i] = int.from_bytes(d_stream, "big") % 256

        if same_material:
            d_rule = d_stream
        else:
            d_rule = _indexed_digest(
                rule_material,
                i,
                config.index_encoding,
            )

        rules[i] = int.from_bytes(d_rule, "big") % len(DNA_RULES)

    return stream, rules


def logistic_permutation(
    n: int,
    alpha: float,
    x0: float,
) -> NDArray[np.int64]:
    """
    Generate x_{n+1}=alpha*x_n*(1-x_n), then use argsort(x)
    as the pixel permutation, as described in the paper.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    seq = np.empty(n, dtype=np.float64)
    x = float(x0)

    for i in range(n):
        x = alpha * x * (1.0 - x)
        seq[i] = x

    # Stable sort gives deterministic behavior if finite-precision ties occur.
    return np.argsort(seq, kind="stable").astype(np.int64, copy=False)


def inverse_permutation(
    permutation: NDArray[np.int64],
) -> NDArray[np.int64]:
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(permutation.size, dtype=np.int64)
    return inv


def fibonacci_mod4(
    n: int,
    mode: FibMode,
) -> NDArray[np.uint8]:
    """
    Return phi(i)=F(i) mod 4.

    worked_example:
        follows the paper's explicit example F(0)=1:
        1,1,2,3,1,0,...

    equation:
        follows Eq. (2) starting 0,1:
        0,1,1,2,3,1,...
    """
    out = np.empty(n, dtype=np.uint8)

    if mode == "worked_example":
        a, b = 1, 1
    elif mode == "equation":
        a, b = 0, 1
    else:
        raise ValueError(f"Unsupported Fibonacci mode: {mode}")

    for i in range(n):
        out[i] = a & 0x03
        a, b = b, (a + b) & 0x03  # only mod 4 is needed

    return out


def _build_dna_luts() -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """
    Build byte-level LUTs for:
        encode byte -> 4 DNA bases -> circular shift -> decode to byte.

    forward_lut[rule, shift, byte]
    inverse_lut[rule, shift, byte]
    """
    forward = np.empty((8, 4, 256), dtype=np.uint8)
    inverse = np.empty((8, 4, 256), dtype=np.uint8)

    for r, bases in enumerate(DNA_RULES):
        decode = {base: value for value, base in enumerate(bases)}

        for value in range(256):
            chunks = [
                (value >> 6) & 0x03,
                (value >> 4) & 0x03,
                (value >> 2) & 0x03,
                value & 0x03,
            ]
            dna = [bases[c] for c in chunks]

            for shift in range(4):
                if shift:
                    right = dna[-shift:] + dna[:-shift]
                    left = dna[shift:] + dna[:shift]
                else:
                    right = dna[:]
                    left = dna[:]

                out_f = 0
                for base in right:
                    out_f = (out_f << 2) | decode[base]
                forward[r, shift, value] = out_f

                out_i = 0
                for base in left:
                    out_i = (out_i << 2) | decode[base]
                inverse[r, shift, value] = out_i

    return forward, inverse


_DNA_FORWARD_LUT, _DNA_INVERSE_LUT = _build_dna_luts()


def encrypt(
    image: NDArray[np.uint8],
    *,
    seed: str | bytes,
    config: AfzalP21Config = AfzalP21Config(),
) -> AfzalP21Ciphertext:
    """
    Encrypt a 2D uint8 grayscale image.
    """
    config.validate()

    P = np.asarray(image)
    if P.ndim != 2:
        raise ValueError(f"image must be 2D grayscale, got shape={P.shape}")
    if P.dtype != np.uint8:
        raise ValueError(f"image dtype must be uint8, got {P.dtype}")
    if P.size == 0:
        raise ValueError("image must not be empty")

    shape = (int(P.shape[0]), int(P.shape[1]))
    flat = P.reshape(-1)
    n = flat.size

    K = derive_256bit_key(seed)

    # T1: logistic permutation
    perm = logistic_permutation(n, config.alpha, config.x0)
    scrambled = flat[perm]

    # T2: SHA-256 byte-wise XOR confusion
    stream, rules = _keystream_and_rules(n, K, config)
    xored = np.bitwise_xor(scrambled, stream)

    # T3 + T4: dynamic DNA encoding + Fibonacci right circular shift
    shifts = fibonacci_mod4(n, config.fib_mode)
    cipher_flat = _DNA_FORWARD_LUT[rules, shifts, xored]

    return AfzalP21Ciphertext(
        image=cipher_flat.reshape(shape).astype(np.uint8, copy=False),
        shape=shape,
    )


def decrypt(
    ciphertext: AfzalP21Ciphertext | NDArray[np.uint8],
    *,
    seed: str | bytes,
    config: AfzalP21Config = AfzalP21Config(),
) -> NDArray[np.uint8]:
    """
    Reverse T4 -> T3 -> T2 -> T1.
    """
    config.validate()

    if isinstance(ciphertext, AfzalP21Ciphertext):
        C = np.asarray(ciphertext.image)
        shape = ciphertext.shape
    else:
        C = np.asarray(ciphertext)
        if C.ndim != 2:
            raise ValueError("raw ciphertext array must be 2D")
        shape = (int(C.shape[0]), int(C.shape[1]))

    if C.dtype != np.uint8:
        raise ValueError(f"ciphertext dtype must be uint8, got {C.dtype}")

    cflat = C.reshape(-1)
    n = cflat.size
    K = derive_256bit_key(seed)

    stream, rules = _keystream_and_rules(n, K, config)
    shifts = fibonacci_mod4(n, config.fib_mode)

    # Inverse T4 + T3: left circular DNA shift and decode
    xored = _DNA_INVERSE_LUT[rules, shifts, cflat]

    # Inverse T2: XOR with the same stream
    scrambled = np.bitwise_xor(xored, stream)

    # Inverse T1
    perm = logistic_permutation(n, config.alpha, config.x0)
    plain_flat = np.empty_like(scrambled)
    plain_flat[perm] = scrambled

    return plain_flat.reshape(shape).astype(np.uint8, copy=False)


def exact_recovery_report(
    original: NDArray[np.uint8],
    recovered: NDArray[np.uint8],
) -> dict[str, int | bool]:
    a = np.asarray(original, dtype=np.uint8)
    b = np.asarray(recovered, dtype=np.uint8)

    if a.shape != b.shape:
        return {
            "exact_recovery": False,
            "mismatched_pixels": int(max(a.size, b.size)),
            "maximum_error": 255,
        }

    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "exact_recovery": bool(np.array_equal(a, b)),
        "mismatched_pixels": int(np.count_nonzero(diff)),
        "maximum_error": int(diff.max(initial=0)),
    }
