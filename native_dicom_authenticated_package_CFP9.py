from __future__ import annotations

import copy
import csv
import hashlib
import hmac
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dicom_native_adapter import (
    NativePixelLayout,
    pixel_data_to_uint8_matrix,
    read_native_pixel_layout,
    uint8_matrix_to_pixel_data,
)
from dicom_native_io import (
    NativeDicomParts,
    REPRESENTATIVE_IMAGE_NAME,
    reconstruct_native_dicom,
    resolve_frozen_100_image_dataset,
    split_native_dicom_bytes,
)
from key_schedule_v4 import derive_bytes
from native_dicom_amie_roundtrip import (
    VALIDATION_MASTER_KEY,
    validation_nonce,
)
from v4_secure_pipeline import (
    ALGORITHM_VERSION,
    BLOCK_SIZE,
    decrypt_image_secure,
    encrypt_image_secure,
)


# =========================================================
# PROJECT SETTINGS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "CFP9_NEW_RESULTS"
    / "native_dicom_authenticated_package"
)

CSV_OUTPUT = (
    OUTPUT_DIR
    / "native_dicom_authenticated_package_100.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "native_dicom_authenticated_package_summary.json"
)

REPRESENTATIVE_DIR = (
    OUTPUT_DIR
    / "representative"
)

EXPECTED_SELECTED_FILES = 100
PARALLEL_WORKERS = 8

EXPERIMENT_VERSION = "NativeDicomAuthenticatedPackage-CFP9-v2"

# This profile intentionally encrypts the exact native Pixel Data bytes
# while retaining the original DICOM container/header bytes verbatim.
# The retained prefix/suffix are authenticated by the outer DICOM envelope
# tag, but are NOT confidential in this profile.
DICOM_PROFILE = "AMIE-Native-DICOM-CFP9-v1"
DICOM_ENVELOPE_HKDF_CONTEXT = (
    "native-dicom-envelope-authentication-key"
)
DICOM_ENVELOPE_DOMAIN = b"AMIE-Native-DICOM-Envelope-v1|"


# =========================================================
# ERRORS / PACKAGE TYPE
# =========================================================

class NativeDicomAuthenticationError(ValueError):
    """Raised when the authenticated DICOM envelope cannot be verified."""


@dataclass
class AuthenticatedNativeDicomPackage:
    """Self-contained authenticated native-DICOM envelope.

    encrypted_image
        AMIE-BakerChen ciphertext of the exact native Pixel Data byte matrix.

    amie_metadata / amie_authentication_tag
        Existing AMIE-BakerChen authenticated metadata and inner HMAC tag.

    dicom_prefix / dicom_suffix
        Original DICOM bytes surrounding the Pixel Data value.  They are
        retained byte-for-byte so the receiver can rebuild the original DICOM
        object without having a copy of the plaintext file.

    dicom_context
        Canonical technical description of the native DICOM layout.

    dicom_authentication_tag
        Outer HMAC-SHA256 tag covering *all* envelope components above.
        This tag is verified before any inverse AMIE image operation.
    """

    encrypted_image: NDArray[np.uint8]
    amie_metadata: dict[str, Any]
    amie_authentication_tag: bytes
    dicom_prefix: bytes
    dicom_suffix: bytes
    dicom_context: dict[str, Any]
    dicom_authentication_tag: bytes


# =========================================================
# CANONICAL ENCODING / HASH HELPERS
# =========================================================

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_sha256_hex(array: NDArray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    """Canonical JSON used by the native-DICOM envelope."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _frame(label: bytes, payload: bytes) -> bytes:
    """Length-delimit one binary envelope field unambiguously."""
    if not isinstance(label, bytes) or not label:
        raise ValueError("Envelope field label must be non-empty bytes.")
    if not isinstance(payload, bytes):
        raise TypeError("Envelope payload must be bytes.")

    return (
        len(label).to_bytes(2, "big", signed=False)
        + label
        + len(payload).to_bytes(8, "big", signed=False)
        + payload
    )


def _validate_ciphertext_array(array: NDArray) -> NDArray[np.uint8]:
    value = np.asarray(array)

    if value.ndim != 2:
        raise ValueError("encrypted_image must be a 2D array.")
    if value.size == 0:
        raise ValueError("encrypted_image must not be empty.")
    if value.dtype != np.uint8:
        raise ValueError(
            f"encrypted_image must use uint8; received {value.dtype}."
        )

    return value


def _session_nonce_from_amie_metadata(
    amie_metadata: dict[str, Any],
) -> bytes:
    """Read only the public 128-bit nonce needed to derive the envelope key."""
    try:
        nonce_hex = str(amie_metadata["session_nonce_hex"])
        nonce = bytes.fromhex(nonce_hex)
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeDicomAuthenticationError(
            "Invalid or missing session nonce in AMIE metadata."
        ) from exc

    if len(nonce) != 16:
        raise NativeDicomAuthenticationError(
            "Session nonce must contain exactly 16 bytes."
        )

    return nonce


def _derive_dicom_envelope_key(
    master_key: bytes,
    session_nonce: bytes,
) -> bytes:
    """Derive a DICOM-envelope-only HMAC key via domain-separated HKDF."""
    return derive_bytes(
        master_key=master_key,
        session_nonce=session_nonce,
        context=DICOM_ENVELOPE_HKDF_CONTEXT,
        length=32,
    )


# =========================================================
# OUTER DICOM ENVELOPE AUTHENTICATION
# =========================================================

def _dicom_envelope_message(
    package: AuthenticatedNativeDicomPackage,
) -> bytes:
    ciphertext = _validate_ciphertext_array(
        package.encrypted_image
    )

    return b"".join(
        (
            DICOM_ENVELOPE_DOMAIN,
            _frame(b"dicom-prefix", package.dicom_prefix),
            _frame(b"dicom-suffix", package.dicom_suffix),
            _frame(
                b"dicom-context",
                _canonical_json_bytes(package.dicom_context),
            ),
            _frame(
                b"amie-metadata",
                _canonical_json_bytes(package.amie_metadata),
            ),
            _frame(
                b"amie-ciphertext",
                np.ascontiguousarray(ciphertext).tobytes(order="C"),
            ),
            _frame(
                b"amie-inner-tag",
                package.amie_authentication_tag,
            ),
        )
    )


def compute_dicom_envelope_tag(
    package: AuthenticatedNativeDicomPackage,
    envelope_key: bytes,
) -> bytes:
    if not isinstance(envelope_key, bytes) or len(envelope_key) != 32:
        raise ValueError("DICOM envelope key must contain 32 bytes.")

    return hmac.new(
        envelope_key,
        _dicom_envelope_message(package),
        hashlib.sha256,
    ).digest()


def verify_dicom_envelope_before_decryption(
    package: AuthenticatedNativeDicomPackage,
    master_key: bytes,
) -> bytes:
    """Verify the entire preserved DICOM envelope before AMIE inversion.

    The only unauthenticated public field consumed before this check is the
    128-bit session nonce required for HKDF key derivation.  No DICOM prefix,
    suffix, layout field, Baker parameter, image dimension, or inverse image
    operation is trusted/executed before the outer HMAC succeeds.
    """

    nonce = _session_nonce_from_amie_metadata(
        package.amie_metadata
    )

    envelope_key = _derive_dicom_envelope_key(
        master_key,
        nonce,
    )

    expected = compute_dicom_envelope_tag(
        package,
        envelope_key,
    )

    if not hmac.compare_digest(
        expected,
        package.dicom_authentication_tag,
    ):
        raise NativeDicomAuthenticationError(
            "Native-DICOM envelope authentication failed: wrong key or "
            "modified DICOM container/ciphertext/metadata/tag."
        )

    return nonce


# =========================================================
# AUTHENTICATED DICOM CONTEXT
# =========================================================

def build_dicom_context(
    *,
    parts: NativeDicomParts,
    layout: NativePixelLayout,
    native_matrix: NDArray[np.uint8],
) -> dict[str, Any]:
    """Build the canonical context authenticated by the outer envelope."""

    return {
        "profile": DICOM_PROFILE,
        "pixel_data_policy": "encrypted-and-authenticated",
        "container_policy": "retained-byte-exact-and-authenticated",
        "container_confidentiality": "not-provided-by-this-profile",
        "transfer_syntax_uid": layout.transfer_syntax_uid,
        "pixel_data_vr": parts.pixel_data_vr,
        "pixel_data_tag_offset": int(parts.pixel_data_tag_offset),
        "pixel_data_value_offset": int(parts.pixel_data_value_offset),
        "pixel_data_length_bytes": int(parts.pixel_data_length),
        "dicom_prefix_length_bytes": len(parts.prefix),
        "dicom_suffix_length_bytes": len(parts.suffix),
        "dicom_prefix_sha256": sha256_hex(parts.prefix),
        "dicom_suffix_sha256": sha256_hex(parts.suffix),
        "rows": int(layout.rows),
        "columns": int(layout.columns),
        "samples_per_pixel": int(layout.samples_per_pixel),
        "number_of_frames": int(layout.number_of_frames),
        "bits_allocated": int(layout.bits_allocated),
        "bits_stored": int(layout.bits_stored),
        "high_bit": int(layout.high_bit),
        "pixel_representation": int(layout.pixel_representation),
        "photometric_interpretation": (
            layout.photometric_interpretation
        ),
        "bytes_per_sample": int(layout.bytes_per_sample),
        "native_matrix_rows": int(native_matrix.shape[0]),
        "native_matrix_columns": int(native_matrix.shape[1]),
        "native_matrix_dtype": "uint8",
    }


def _validate_authenticated_dicom_context(
    package: AuthenticatedNativeDicomPackage,
) -> NativePixelLayout:
    """Validate context only *after* outer HMAC authentication."""

    context = package.dicom_context

    if context.get("profile") != DICOM_PROFILE:
        raise ValueError("Unsupported native-DICOM package profile.")

    if (
        context.get("pixel_data_policy")
        != "encrypted-and-authenticated"
    ):
        raise ValueError("Unexpected Pixel Data policy.")

    if (
        context.get("container_policy")
        != "retained-byte-exact-and-authenticated"
    ):
        raise ValueError("Unexpected DICOM container policy.")

    prefix_length = int(
        context["dicom_prefix_length_bytes"]
    )
    suffix_length = int(
        context["dicom_suffix_length_bytes"]
    )

    if len(package.dicom_prefix) != prefix_length:
        raise ValueError("Authenticated DICOM prefix length mismatch.")

    if len(package.dicom_suffix) != suffix_length:
        raise ValueError("Authenticated DICOM suffix length mismatch.")

    if sha256_hex(package.dicom_prefix) != str(
        context["dicom_prefix_sha256"]
    ):
        raise ValueError("Authenticated DICOM prefix hash mismatch.")

    if sha256_hex(package.dicom_suffix) != str(
        context["dicom_suffix_sha256"]
    ):
        raise ValueError("Authenticated DICOM suffix hash mismatch.")

    rows = int(context["rows"])
    columns = int(context["columns"])
    samples_per_pixel = int(context["samples_per_pixel"])
    number_of_frames = int(context["number_of_frames"])
    bits_allocated = int(context["bits_allocated"])
    bits_stored = int(context["bits_stored"])
    high_bit = int(context["high_bit"])
    pixel_representation = int(context["pixel_representation"])
    bytes_per_sample = int(context["bytes_per_sample"])
    pixel_data_length = int(context["pixel_data_length_bytes"])
    matrix_rows = int(context["native_matrix_rows"])
    matrix_columns = int(context["native_matrix_columns"])

    if rows <= 0 or columns <= 0:
        raise ValueError("Authenticated DICOM dimensions are invalid.")
    if samples_per_pixel != 1:
        raise ValueError("Current profile requires SamplesPerPixel=1.")
    if number_of_frames != 1:
        raise ValueError("Current profile requires one DICOM frame.")
    if bits_allocated <= 0 or bits_allocated % 8 != 0:
        raise ValueError("Authenticated BitsAllocated is invalid.")
    if not (0 < bits_stored <= bits_allocated):
        raise ValueError("Authenticated BitsStored is invalid.")
    if high_bit != bits_stored - 1:
        raise ValueError("Authenticated HighBit is inconsistent.")
    if pixel_representation not in (0, 1):
        raise ValueError("Authenticated PixelRepresentation is invalid.")
    if bytes_per_sample != bits_allocated // 8:
        raise ValueError("Authenticated bytes-per-sample is inconsistent.")

    expected_pixel_bytes = (
        rows
        * columns
        * samples_per_pixel
        * number_of_frames
        * bytes_per_sample
    )

    if pixel_data_length != expected_pixel_bytes:
        raise ValueError(
            "Authenticated Pixel Data length is inconsistent with "
            "the native DICOM layout."
        )

    expected_matrix_rows = rows * number_of_frames
    expected_matrix_columns = (
        columns * samples_per_pixel * bytes_per_sample
    )

    if matrix_rows != expected_matrix_rows:
        raise ValueError("Authenticated native matrix row count is invalid.")

    if matrix_columns != expected_matrix_columns:
        raise ValueError(
            "Authenticated native matrix column count is invalid."
        )

    if context.get("native_matrix_dtype") != "uint8":
        raise ValueError("Authenticated native matrix dtype is invalid.")

    return NativePixelLayout(
        rows=rows,
        columns=columns,
        samples_per_pixel=samples_per_pixel,
        number_of_frames=number_of_frames,
        bits_allocated=bits_allocated,
        bits_stored=bits_stored,
        high_bit=high_bit,
        pixel_representation=pixel_representation,
        photometric_interpretation=str(
            context["photometric_interpretation"]
        ),
        transfer_syntax_uid=str(
            context["transfer_syntax_uid"]
        ),
        bytes_per_sample=bytes_per_sample,
        expected_pixel_data_bytes=pixel_data_length,
        matrix_rows=matrix_rows,
        matrix_columns=matrix_columns,
    )


# =========================================================
# SELF-CONTAINED NATIVE-DICOM ENCRYPTION / DECRYPTION
# =========================================================

def encrypt_native_dicom_package(
    dicom_path: str | Path,
    master_key: bytes,
    *,
    session_nonce: bytes,
    executor: ProcessPoolExecutor,
) -> tuple[AuthenticatedNativeDicomPackage, dict[str, Any]]:
    """Encrypt native Pixel Data and authenticate the preserved container."""

    path = Path(dicom_path)
    original_dicom = path.read_bytes()
    parts = split_native_dicom_bytes(original_dicom)
    layout = read_native_pixel_layout(path)

    if parts.pixel_data_length != layout.expected_pixel_data_bytes:
        raise ValueError(
            f"{path.name}: Pixel Data length does not match DICOM layout."
        )

    native_matrix = pixel_data_to_uint8_matrix(
        parts.pixel_data,
        layout,
    )

    start = time.perf_counter()

    amie_package = encrypt_image_secure(
        native_matrix,
        master_key,
        session_nonce=session_nonce,
        block_size=BLOCK_SIZE,
        max_workers=PARALLEL_WORKERS,
        executor=executor,
    )

    dicom_context = build_dicom_context(
        parts=parts,
        layout=layout,
        native_matrix=native_matrix,
    )

    # Construct with an empty outer tag first because the tag authenticates
    # all other package fields, not itself.
    package = AuthenticatedNativeDicomPackage(
        encrypted_image=np.asarray(
            amie_package["encrypted_image"],
            dtype=np.uint8,
        ),
        amie_metadata=copy.deepcopy(
            amie_package["metadata"]
        ),
        amie_authentication_tag=bytes(
            amie_package["authentication_tag"]
        ),
        dicom_prefix=parts.prefix,
        dicom_suffix=parts.suffix,
        dicom_context=dicom_context,
        dicom_authentication_tag=b"",
    )

    envelope_key = _derive_dicom_envelope_key(
        master_key,
        session_nonce,
    )

    package.dicom_authentication_tag = (
        compute_dicom_envelope_tag(
            package,
            envelope_key,
        )
    )

    elapsed = time.perf_counter() - start

    evidence = {
        "original_dicom": original_dicom,
        "original_pixel_data": parts.pixel_data,
        "native_matrix": native_matrix,
        "inner_encryption_time_sec": float(
            amie_package["encryption_time_sec"]
        ),
        "native_dicom_encryption_wall_time_sec": float(elapsed),
    }

    return package, evidence


def decrypt_native_dicom_package(
    package: AuthenticatedNativeDicomPackage,
    master_key: bytes,
    *,
    executor: ProcessPoolExecutor,
) -> tuple[bytes, dict[str, Any]]:
    """Authenticate the complete DICOM envelope, then decrypt Pixel Data."""

    start = time.perf_counter()

    # CRITICAL ORDERING: outer DICOM envelope authentication happens first.
    verify_dicom_envelope_before_decryption(
        package,
        master_key,
    )

    # Only authenticated DICOM layout/context is consumed from this point.
    layout = _validate_authenticated_dicom_context(
        package
    )

    # The existing AMIE inner HMAC is verified inside decrypt_image_secure
    # before any inverse Baker/permutation/diffusion operation.
    decrypted = decrypt_image_secure(
        package.encrypted_image,
        package.amie_metadata,
        package.amie_authentication_tag,
        master_key,
        max_workers=PARALLEL_WORKERS,
        executor=executor,
    )

    recovered_matrix = np.asarray(
        decrypted["decrypted_image"],
        dtype=np.uint8,
    )

    recovered_pixel_data = uint8_matrix_to_pixel_data(
        recovered_matrix,
        layout,
    )

    parts = NativeDicomParts(
        prefix=package.dicom_prefix,
        pixel_data=b"",
        suffix=package.dicom_suffix,
        pixel_data_vr=str(
            package.dicom_context["pixel_data_vr"]
        ),
        pixel_data_tag_offset=int(
            package.dicom_context["pixel_data_tag_offset"]
        ),
        pixel_data_value_offset=int(
            package.dicom_context["pixel_data_value_offset"]
        ),
        pixel_data_length=int(
            package.dicom_context["pixel_data_length_bytes"]
        ),
    )

    recovered_dicom = reconstruct_native_dicom(
        parts,
        recovered_pixel_data,
    )

    elapsed = time.perf_counter() - start

    evidence = {
        "recovered_matrix": recovered_matrix,
        "recovered_pixel_data": recovered_pixel_data,
        "inner_decryption_time_sec": float(
            decrypted["decryption_time_sec"]
        ),
        "native_dicom_decryption_wall_time_sec": float(elapsed),
        "outer_authentication_verified": True,
        "inner_authentication_verified": bool(
            decrypted.get("authentication_verified", True)
        ),
    }

    return recovered_dicom, evidence


# =========================================================
# PACKAGE CLONING / CONTROLLED TAMPER TESTS
# =========================================================

def clone_package(
    package: AuthenticatedNativeDicomPackage,
) -> AuthenticatedNativeDicomPackage:
    return AuthenticatedNativeDicomPackage(
        encrypted_image=package.encrypted_image.copy(),
        amie_metadata=copy.deepcopy(package.amie_metadata),
        amie_authentication_tag=bytes(
            package.amie_authentication_tag
        ),
        dicom_prefix=bytes(package.dicom_prefix),
        dicom_suffix=bytes(package.dicom_suffix),
        dicom_context=copy.deepcopy(package.dicom_context),
        dicom_authentication_tag=bytes(
            package.dicom_authentication_tag
        ),
    )


def _flip_first_byte(data: bytes) -> bytes:
    if not data:
        raise ValueError("Cannot flip a byte in an empty byte string.")
    value = bytearray(data)
    value[0] ^= 0x01
    return bytes(value)


def _wrong_key_one_bit(master_key: bytes) -> bytes:
    if len(master_key) != 32:
        raise ValueError("Expected a 256-bit validation master key.")
    changed = bytearray(master_key)
    changed[0] ^= 0x01
    return bytes(changed)


def _expect_outer_auth_rejection(
    package: AuthenticatedNativeDicomPackage,
    master_key: bytes,
    *,
    executor: ProcessPoolExecutor,
) -> bool:
    try:
        decrypt_native_dicom_package(
            package,
            master_key,
            executor=executor,
        )
    except NativeDicomAuthenticationError:
        return True
    except Exception:
        # A modified package must be rejected specifically by the outer
        # envelope authentication gate, not later by inverse processing.
        return False
    else:
        return False


def run_tamper_tests(
    package: AuthenticatedNativeDicomPackage,
    master_key: bytes,
    *,
    executor: ProcessPoolExecutor,
) -> dict[str, bool]:
    tests: dict[str, bool] = {}

    # 1. Wrong 256-bit master key (one-bit difference).
    tests["wrong_key_rejected_by_outer_auth"] = (
        _expect_outer_auth_rejection(
            clone_package(package),
            _wrong_key_one_bit(master_key),
            executor=executor,
        )
    )

    # 2. AMIE ciphertext tampering.
    modified = clone_package(package)
    modified.encrypted_image[0, 0] ^= np.uint8(1)
    tests["ciphertext_tamper_rejected_by_outer_auth"] = (
        _expect_outer_auth_rejection(
            modified,
            master_key,
            executor=executor,
        )
    )

    # 3. DICOM prefix/header/metadata tampering.
    modified = clone_package(package)
    prefix = bytearray(modified.dicom_prefix)
    prefix_index = min(132, len(prefix) - 1)
    prefix[prefix_index] ^= 0x01
    modified.dicom_prefix = bytes(prefix)
    tests["dicom_prefix_tamper_rejected_by_outer_auth"] = (
        _expect_outer_auth_rejection(
            modified,
            master_key,
            executor=executor,
        )
    )

    # 4. DICOM suffix tampering.  The present 100-image dataset happens to
    # have empty suffixes; appending one byte still tests envelope coverage.
    modified = clone_package(package)
    if modified.dicom_suffix:
        modified.dicom_suffix = _flip_first_byte(
            modified.dicom_suffix
        )
    else:
        modified.dicom_suffix = b"\x00"
    tests["dicom_suffix_tamper_rejected_by_outer_auth"] = (
        _expect_outer_auth_rejection(
            modified,
            master_key,
            executor=executor,
        )
    )

    # 5. Authenticated DICOM layout/context tampering.
    modified = clone_package(package)
    modified.dicom_context["bits_stored"] = (
        int(modified.dicom_context["bits_stored"]) + 1
    )
    tests["dicom_context_tamper_rejected_by_outer_auth"] = (
        _expect_outer_auth_rejection(
            modified,
            master_key,
            executor=executor,
        )
    )

    # 6. Existing AMIE metadata tampering.
    modified = clone_package(package)
    modified.amie_metadata["block_size"] = (
        int(modified.amie_metadata["block_size"]) + BLOCK_SIZE
    )
    tests["amie_metadata_tamper_rejected_by_outer_auth"] = (
        _expect_outer_auth_rejection(
            modified,
            master_key,
            executor=executor,
        )
    )

    # 7. Existing AMIE inner HMAC tag tampering.
    modified = clone_package(package)
    modified.amie_authentication_tag = _flip_first_byte(
        modified.amie_authentication_tag
    )
    tests["inner_tag_tamper_rejected_by_outer_auth"] = (
        _expect_outer_auth_rejection(
            modified,
            master_key,
            executor=executor,
        )
    )

    # 8. New outer DICOM envelope HMAC tag tampering.
    modified = clone_package(package)
    modified.dicom_authentication_tag = _flip_first_byte(
        modified.dicom_authentication_tag
    )
    tests["outer_tag_tamper_rejected_by_outer_auth"] = (
        _expect_outer_auth_rejection(
            modified,
            master_key,
            executor=executor,
        )
    )

    # 9. Public session nonce tampering while preserving 16-byte syntax.
    modified = clone_package(package)
    nonce = bytearray(
        bytes.fromhex(
            str(modified.amie_metadata["session_nonce_hex"])
        )
    )
    nonce[0] ^= 0x01
    modified.amie_metadata["session_nonce_hex"] = bytes(nonce).hex()
    tests["nonce_tamper_rejected_by_outer_auth"] = (
        _expect_outer_auth_rejection(
            modified,
            master_key,
            executor=executor,
        )
    )

    return tests


# =========================================================
# REPRESENTATIVE ARTIFACTS
# =========================================================

def save_representative_artifacts(
    *,
    original_dicom: bytes,
    recovered_dicom: bytes,
    package: AuthenticatedNativeDicomPackage,
    nonce: bytes,
) -> None:
    REPRESENTATIVE_DIR.mkdir(parents=True, exist_ok=True)

    (REPRESENTATIVE_DIR / "image_021_original.dcm").write_bytes(
        original_dicom
    )
    (REPRESENTATIVE_DIR / "image_021_recovered_authenticated.dcm").write_bytes(
        recovered_dicom
    )
    (REPRESENTATIVE_DIR / "image_021_dicom_prefix.bin").write_bytes(
        package.dicom_prefix
    )
    (REPRESENTATIVE_DIR / "image_021_dicom_suffix.bin").write_bytes(
        package.dicom_suffix
    )

    np.save(
        REPRESENTATIVE_DIR / "image_021_encrypted_byte_matrix.npy",
        package.encrypted_image,
        allow_pickle=False,
    )

    (REPRESENTATIVE_DIR / "image_021_amie_metadata.json").write_text(
        json.dumps(
            package.amie_metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (REPRESENTATIVE_DIR / "image_021_dicom_context.json").write_text(
        json.dumps(
            package.dicom_context,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (REPRESENTATIVE_DIR / "image_021_inner_auth_tag.txt").write_text(
        package.amie_authentication_tag.hex(),
        encoding="ascii",
    )
    (REPRESENTATIVE_DIR / "image_021_outer_dicom_auth_tag.txt").write_text(
        package.dicom_authentication_tag.hex(),
        encoding="ascii",
    )
    (REPRESENTATIVE_DIR / "image_021_validation_nonce.txt").write_text(
        nonce.hex(),
        encoding="ascii",
    )

    package_spec = {
        "profile": DICOM_PROFILE,
        "algorithm_version": ALGORITHM_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "hkdf_context": DICOM_ENVELOPE_HKDF_CONTEXT,
        "outer_hmac": "HMAC-SHA256",
        "outer_hmac_domain_ascii": (
            DICOM_ENVELOPE_DOMAIN.decode("ascii")
        ),
        "authenticated_components_in_order": [
            "dicom-prefix",
            "dicom-suffix",
            "dicom-context-canonical-json",
            "amie-metadata-canonical-json",
            "amie-ciphertext-row-major-bytes",
            "amie-inner-tag",
        ],
        "field_framing": (
            "uint16_be(label_length) || label || "
            "uint64_be(payload_length) || payload"
        ),
        "dicom_container_confidentiality": (
            "not provided in this profile; prefix/suffix are retained "
            "verbatim but authenticated"
        ),
        "pixel_data_confidentiality": (
            "native Pixel Data bytes are encrypted by AMIE-BakerChen"
        ),
        "verification_order": [
            "derive outer envelope key from master key + public nonce",
            "verify outer DICOM envelope HMAC",
            "validate authenticated DICOM context",
            "verify existing AMIE inner HMAC",
            "perform inverse AMIE processing",
            "reconstruct exact DICOM bytes",
        ],
    }

    (REPRESENTATIVE_DIR / "image_021_package_spec.json").write_text(
        json.dumps(package_spec, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# =========================================================
# ONE-FILE EVALUATION
# =========================================================

def evaluate_one_file(
    record,
    executor: ProcessPoolExecutor,
) -> dict[str, Any]:
    nonce = validation_nonce(
        record.selection_index,
        record.destination_name,
    )

    package, encrypted_evidence = encrypt_native_dicom_package(
        record.selected_path,
        VALIDATION_MASTER_KEY,
        session_nonce=nonce,
        executor=executor,
    )

    recovered_dicom, decrypted_evidence = decrypt_native_dicom_package(
        package,
        VALIDATION_MASTER_KEY,
        executor=executor,
    )

    original_dicom = bytes(
        encrypted_evidence["original_dicom"]
    )
    original_pixel_data = bytes(
        encrypted_evidence["original_pixel_data"]
    )
    recovered_pixel_data = bytes(
        decrypted_evidence["recovered_pixel_data"]
    )

    clean_dicom_exact = recovered_dicom == original_dicom
    clean_pixel_exact = (
        recovered_pixel_data == original_pixel_data
    )

    tamper = run_tamper_tests(
        package,
        VALIDATION_MASTER_KEY,
        executor=executor,
    )

    all_tamper_tests_rejected = all(tamper.values())

    if record.destination_name == REPRESENTATIVE_IMAGE_NAME:
        save_representative_artifacts(
            original_dicom=original_dicom,
            recovered_dicom=recovered_dicom,
            package=package,
            nonce=nonce,
        )

    return {
        "experiment_version": EXPERIMENT_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "dicom_profile": DICOM_PROFILE,
        "selection_index": record.selection_index,
        "destination_name": record.destination_name,
        "session_nonce_hex": nonce.hex(),
        "native_pixel_data_bytes": len(original_pixel_data),
        "dicom_prefix_bytes": len(package.dicom_prefix),
        "dicom_suffix_bytes": len(package.dicom_suffix),
        "ciphertext_rows": int(package.encrypted_image.shape[0]),
        "ciphertext_columns": int(package.encrypted_image.shape[1]),
        "ciphertext_sha256": array_sha256_hex(
            package.encrypted_image
        ),
        "inner_authentication_tag_hex": (
            package.amie_authentication_tag.hex()
        ),
        "outer_dicom_authentication_tag_hex": (
            package.dicom_authentication_tag.hex()
        ),
        "original_dicom_sha256": sha256_hex(original_dicom),
        "recovered_dicom_sha256": sha256_hex(recovered_dicom),
        "clean_pixel_data_exact": clean_pixel_exact,
        "clean_dicom_byte_exact": clean_dicom_exact,
        "clean_dicom_sha256_exact": (
            sha256_hex(original_dicom)
            == sha256_hex(recovered_dicom)
        ),
        "outer_authentication_verified": bool(
            decrypted_evidence["outer_authentication_verified"]
        ),
        "inner_authentication_verified": bool(
            decrypted_evidence["inner_authentication_verified"]
        ),
        **tamper,
        "all_tamper_tests_rejected": all_tamper_tests_rejected,
        "native_dicom_encryption_wall_time_sec": float(
            encrypted_evidence[
                "native_dicom_encryption_wall_time_sec"
            ]
        ),
        "native_dicom_decryption_wall_time_sec": float(
            decrypted_evidence[
                "native_dicom_decryption_wall_time_sec"
            ]
        ),
        "status": (
            "PASS"
            if (
                clean_pixel_exact
                and clean_dicom_exact
                and all_tamper_tests_rejected
            )
            else "FAIL"
        ),
    }


# =========================================================
# OUTPUT WRITERS
# =========================================================

def _write_csv(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No evaluation rows were supplied.")

    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with CSV_OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(summary: dict[str, Any]) -> None:
    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# =========================================================
# COMPLETE 100-IMAGE STEP-5 EVALUATION
# =========================================================

def main() -> None:
    print("=" * 104)
    print(
        "NATIVE-DICOM AUTHENTICATED SELF-CONTAINED PACKAGE — "
        "FROZEN 100-IMAGE SUBSET"
    )
    print("=" * 104)

    print()
    print("[1] Resolving and verifying the exact frozen dataset...")
    records = resolve_frozen_100_image_dataset()

    if len(records) != EXPECTED_SELECTED_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_SELECTED_FILES} files; found {len(records)}."
        )

    print(f"    PASS: verified frozen subset = {len(records)}/100")

    print()
    print("[2] Native-DICOM package policy")
    print(f"    Algorithm version:        {ALGORITHM_VERSION}")
    print(f"    Experiment version:       {EXPERIMENT_VERSION}")
    print(f"    DICOM package profile:    {DICOM_PROFILE}")
    print("    Native Pixel Data:        encrypted + authenticated")
    print("    DICOM prefix/metadata:    retained byte-exact + authenticated")
    print("    DICOM suffix:             retained byte-exact + authenticated")
    print("    Header confidentiality:   NOT provided by this Step-5 profile")
    print("    Outer authentication:     HMAC-SHA256 before AMIE inversion")
    print(f"    Outer HKDF label:         {DICOM_ENVELOPE_HKDF_CONTEXT}")
    print(f"    Parallel Baker workers:   {PARALLEL_WORKERS}")

    print()
    print("[3] Running clean recovery + wrong-key/tamper tests...")

    rows: list[dict[str, Any]] = []
    validation_start = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=PARALLEL_WORKERS
    ) as executor:
        for record in records:
            result = evaluate_one_file(
                record,
                executor,
            )
            rows.append(result)

            print(
                f"    {record.destination_name}: {result['status']} | "
                f"DICOM exact={result['clean_dicom_byte_exact']} | "
                f"tamper tests={result['all_tamper_tests_rejected']}"
            )

    total_validation_wall_time = (
        time.perf_counter() - validation_start
    )

    _write_csv(rows)

    clean_exact_count = sum(
        bool(row["clean_dicom_byte_exact"])
        for row in rows
    )
    pixel_exact_count = sum(
        bool(row["clean_pixel_data_exact"])
        for row in rows
    )
    outer_verified_count = sum(
        bool(row["outer_authentication_verified"])
        for row in rows
    )
    inner_verified_count = sum(
        bool(row["inner_authentication_verified"])
        for row in rows
    )
    all_tamper_rejected_count = sum(
        bool(row["all_tamper_tests_rejected"])
        for row in rows
    )

    tamper_fields = [
        "wrong_key_rejected_by_outer_auth",
        "ciphertext_tamper_rejected_by_outer_auth",
        "dicom_prefix_tamper_rejected_by_outer_auth",
        "dicom_suffix_tamper_rejected_by_outer_auth",
        "dicom_context_tamper_rejected_by_outer_auth",
        "amie_metadata_tamper_rejected_by_outer_auth",
        "inner_tag_tamper_rejected_by_outer_auth",
        "outer_tag_tamper_rejected_by_outer_auth",
        "nonce_tamper_rejected_by_outer_auth",
    ]

    tamper_counts = {
        field: sum(bool(row[field]) for row in rows)
        for field in tamper_fields
    }

    total_modified_packages = (
        len(rows) * len(tamper_fields)
    )
    total_modified_rejected = sum(tamper_counts.values())

    summary = {
        "experiment": (
            "Native-DICOM authenticated self-contained package evaluation"
        ),
        "experiment_version": EXPERIMENT_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "stage": (
            "CFP9 authenticated native-DICOM package validation with "
            "outer HMAC verification before AMIE inverse processing"
        ),
        "profile": DICOM_PROFILE,
        "selected_files": len(rows),
        "parallel_workers": PARALLEL_WORKERS,
        "block_size": BLOCK_SIZE,
        "pixel_data_policy": "encrypted-and-authenticated",
        "dicom_container_policy": (
            "retained-byte-exact-and-authenticated"
        ),
        "dicom_container_confidentiality": (
            "not-provided-by-this-profile"
        ),
        "outer_authentication_algorithm": "HMAC-SHA256",
        "outer_hkdf_context": DICOM_ENVELOPE_HKDF_CONTEXT,
        "outer_authentication_before_inverse_processing": True,
        "clean_pixel_data_exact_count": pixel_exact_count,
        "clean_dicom_byte_exact_count": clean_exact_count,
        "outer_authentication_verified_count": outer_verified_count,
        "inner_authentication_verified_count": inner_verified_count,
        "all_tamper_tests_rejected_count": all_tamper_rejected_count,
        "tamper_rejection_counts": tamper_counts,
        "tamper_categories_per_image": len(tamper_fields),
        "total_modified_or_wrong_key_packages": total_modified_packages,
        "total_modified_or_wrong_key_rejected_by_outer_auth": (
            total_modified_rejected
        ),
        "all_100_clean_dicom_exact": (
            clean_exact_count == EXPECTED_SELECTED_FILES
        ),
        "all_100_tamper_suites_passed": (
            all_tamper_rejected_count == EXPECTED_SELECTED_FILES
        ),
        "deterministic_validation_nonces_only_for_reproducibility": True,
        "production_requires_fresh_csprng_nonce": True,
        "validation_key_is_public_test_vector_only": True,
        "validation_master_key_hex": VALIDATION_MASTER_KEY.hex(),
        "total_validation_wall_time_sec": float(
            total_validation_wall_time
        ),
        "csv_output": str(CSV_OUTPUT),
        "representative_output_dir": str(REPRESENTATIVE_DIR),
    }

    _write_summary(summary)

    print()
    print("[4] Final result")
    print(
        f"    Clean native Pixel Data exact:          "
        f"{pixel_exact_count}/100"
    )
    print(
        f"    Clean full DICOM byte-exact recovery:   "
        f"{clean_exact_count}/100"
    )
    print(
        f"    Outer DICOM authentication verified:    "
        f"{outer_verified_count}/100"
    )
    print(
        f"    Inner AMIE authentication verified:     "
        f"{inner_verified_count}/100"
    )
    print(
        f"    Complete tamper suites passed:          "
        f"{all_tamper_rejected_count}/100"
    )
    print(
        f"    Modified/wrong-key packages rejected:   "
        f"{total_modified_rejected}/{total_modified_packages}"
    )

    for field in tamper_fields:
        print(
            f"    {field}: "
            f"{tamper_counts[field]}/100"
        )

    print(f"    Total validation wall time: {total_validation_wall_time:.3f} s")
    print(f"    CSV:     {CSV_OUTPUT}")
    print(f"    Summary: {SUMMARY_OUTPUT}")
    print(f"    Representative artifacts: {REPRESENTATIVE_DIR}")

    if clean_exact_count != EXPECTED_SELECTED_FILES:
        raise RuntimeError(
            "At least one clean authenticated DICOM package failed "
            "byte-exact recovery."
        )

    if all_tamper_rejected_count != EXPECTED_SELECTED_FILES:
        raise RuntimeError(
            "At least one wrong-key/tamper case was not rejected at the "
            "outer DICOM authentication gate."
        )

    print()
    print(
        "PASS: CFP9 authenticated the retained DICOM container/metadata "
        "and encrypted native Pixel Data as one self-contained envelope."
    )
    print(
        "READY FOR NEXT STEP: package serialization/test vectors and "
        "cross-platform Chen interoperability."
    )


if __name__ == "__main__":
    main()
