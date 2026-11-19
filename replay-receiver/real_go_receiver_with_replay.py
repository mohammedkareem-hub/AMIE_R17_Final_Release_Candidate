"""Phase 6 research-only real Go receiver + persistent SQLite replay state.

Does not change CFP9-v2 encryption, native .amiedcm package, or Go receiver.
Uses the actual independently-built Go binary for both HMAC layers and full
image inversion, NOT mocked authentication/decryption callbacks.

The Go CLI receives the key on its process command line. That is acceptable
ONLY for PUBLIC reproducibility test keys, not secret production keys.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from hardened_replay_cache import HardenedReplayCache, replay_key_namespace

PACKAGE_MAGIC = b"AMIE-NATIVE-DICOM-PACKAGE\x00\x01"
PACKAGE_LABELS = (
    b"dicom-prefix", b"dicom-suffix", b"dicom-context", b"amie-metadata",
    b"amie-ciphertext", b"amie-inner-tag", b"dicom-outer-tag",
)
NONCE_HEX_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
MAX_TEST_PACKAGE_BYTES = 32 * 1024 * 1024


class ReceiverVerificationError(RuntimeError):
    """Go authentication, protocol parsing or decryption failed (fail closed)."""


@dataclass(frozen=True)
class VerifiedRecovery:
    recovered_dicom: bytes
    recovered_sha256: str
    package_sha256: str
    nonce_hex: str
    authentication_verified_before_reservation: bool


def extract_nonce_from_authenticated_envelope(raw: bytes) -> bytes:
    """Parse the already-Go-authenticated snapshot, never untrusted input.

    The additional parser here extracts only nonce for durable reservation;
    the Go receiver independently validates complete framing, context and
    canonical JSON and authenticates both tags prior to this call.
    """
    if not isinstance(raw, bytes) or not (len(PACKAGE_MAGIC)+2 <= len(raw) <= MAX_TEST_PACKAGE_BYTES):
        raise ReceiverVerificationError('invalid authenticated snapshot length')
    if not raw.startswith(PACKAGE_MAGIC):
        raise ReceiverVerificationError('invalid authenticated snapshot magic')
    off = len(PACKAGE_MAGIC)

    def consume(n: int) -> bytes:
        nonlocal off
        if n < 0 or off + n > len(raw):
            raise ReceiverVerificationError('truncated authenticated snapshot')
        result = raw[off:off+n]
        off += n
        return result

    if struct.unpack('>H', consume(2))[0] != len(PACKAGE_LABELS):
        raise ReceiverVerificationError('unexpected authenticated field count')
    metadata_raw = None
    for expected_label in PACKAGE_LABELS:
        ln = struct.unpack('>H', consume(2))[0]
        if ln < 1 or ln > 128 or consume(ln) != expected_label:
            raise ReceiverVerificationError('invalid authenticated label/order')
        payload_len = struct.unpack('>Q', consume(8))[0]
        if payload_len > len(raw) - off:
            raise ReceiverVerificationError('invalid authenticated payload length')
        value = consume(payload_len)
        if expected_label == b'amie-metadata':
            metadata_raw = value
    if off != len(raw) or metadata_raw is None or len(metadata_raw) > 1_000_000:
        raise ReceiverVerificationError('unexpected authenticated trailing bytes/metadata')
    try:
        meta = json.loads(metadata_raw.decode('utf-8'))
        nonce_hex = meta['session_nonce_hex']
        if not isinstance(nonce_hex, str) or not NONCE_HEX_PATTERN.fullmatch(nonce_hex):
            raise ValueError('bad nonce encoding')
        return bytes.fromhex(nonce_hex)
    except (ValueError, KeyError, UnicodeError, TypeError) as exc:
        raise ReceiverVerificationError('invalid authenticated nonce in metadata') from exc


def _go_run(go_binary: Path, package_file: Path, master_key: bytes, *,
            auth_only: bool, out_dicom: Path | None = None,
            expected_sha256: str | None = None) -> dict:
    args = [str(go_binary), '-package', str(package_file), '-key-hex', master_key.hex()]
    if auth_only:
        args.append('-auth-only')
    if out_dicom is not None:
        args += ['-out-dicom', str(out_dicom)]
    if expected_sha256 is not None:
        args += ['-expected-sha256', expected_sha256]
    try:
        cp = subprocess.run(args, capture_output=True, text=True, timeout=45, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReceiverVerificationError('Go receiver unavailable or timed out') from exc
    if cp.returncode != 0:
        raise ReceiverVerificationError('Go receiver rejected: ' + cp.stderr.strip()[:300])
    try:
        result = json.loads(cp.stdout)
    except (ValueError, TypeError) as exc:
        raise ReceiverVerificationError('Go receiver returned invalid evidence JSON') from exc
    if (result.get('pass') is not True or result.get('inner_hmac_verified') is not True
            or result.get('outer_hmac_verified') is not True):
        raise ReceiverVerificationError('Go receiver authentication evidence incomplete')
    return result


def receive_with_persistent_replay_protection(*, raw_package: bytes,
                                              public_validation_key: bytes,
                                              go_receiver_binary: str | Path,
                                              cache: HardenedReplayCache,
                                              expected_original_sha256: str | None = None) -> VerifiedRecovery:
    """Perform real Go authentication -> SQLite reservation -> real Go decrypt.

    Takes raw package BYTES, not a mutable pathname. The exact snapshot is
    verified in Go twice (outer + inner), before and during decryption.
    Pre-existing/pending nonces always fail closed before full inversion.
    """
    if type(raw_package) is not bytes or not 0 < len(raw_package) <= MAX_TEST_PACKAGE_BYTES:
        raise ValueError('raw package must be nonempty bytes under 32 MiB test limit')
    # Public key argument name makes explicit that the Go CLI is unsuitable for
    # secret operational key material without a separate IPC/secure-key adapter.
    namespace = replay_key_namespace(public_validation_key)
    go_binary = Path(go_receiver_binary).resolve()
    if not go_binary.is_file():
        raise FileNotFoundError('Go receiver binary not found')
    if expected_original_sha256 is not None and (not isinstance(expected_original_sha256, str)
                                                    or not HASH_PATTERN.fullmatch(expected_original_sha256)):
        raise ValueError('expected original SHA-256 must be canonical hex')
    fingerprint = hashlib.sha256(raw_package).hexdigest()
    with tempfile.TemporaryDirectory(prefix='amie-phase6-receiver-') as scratch:
        package_file = Path(scratch) / 'snapshot.amiedcm'
        recovered_file = Path(scratch) / 'recovered.dcm'
        # Private temporary snapshot prevents time-of-check/use mismatch between
        # verified bytes and decryption bytes in this research harness.
        with package_file.open('xb') as out:
            if os.name == 'posix':
                os.chmod(package_file, 0o600)
            out.write(raw_package)
            out.flush()
            os.fsync(out.fileno())

        auth = _go_run(go_binary, package_file, public_validation_key, auth_only=True)
        if auth.get('serialized_package_sha256') != fingerprint:
            raise ReceiverVerificationError('authenticated snapshot SHA-256 mismatch')
        nonce = extract_nonce_from_authenticated_envelope(raw_package)
        # Hard boundary: persistent, atomic reservation before Go inverse steps.
        cache.reserve(key_namespace=namespace, nonce=nonce,
                      package_fingerprint=fingerprint)
        # A later failure leaves a pending reservation; this is intentional.
        recovered = _go_run(go_binary, package_file, public_validation_key,
                            auth_only=False, out_dicom=recovered_file,
                            expected_sha256=expected_original_sha256)
        if recovered.get('serialized_package_sha256') != fingerprint:
            raise ReceiverVerificationError('decryption consumed unexpected package snapshot')
        if recovered.get('native_dicom_decrypted') is not True:
            raise ReceiverVerificationError('Go did not report full native DICOM recovery')
        recovered_bytes = recovered_file.read_bytes()
        digest = hashlib.sha256(recovered_bytes).hexdigest()
        if digest != recovered.get('recovered_dicom_sha256'):
            raise ReceiverVerificationError('recovered DICOM evidence/hash mismatch')
        if expected_original_sha256 is not None and digest != expected_original_sha256:
            raise ReceiverVerificationError('recovered DICOM differed from required reference')
        cache.mark_accepted(key_namespace=namespace, nonce=nonce)
        return VerifiedRecovery(recovered_bytes, digest, fingerprint, nonce.hex(), True)
