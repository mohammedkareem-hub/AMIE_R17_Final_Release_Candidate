from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from dicom_native_io import (
    REPRESENTATIVE_IMAGE_NAME,
    resolve_frozen_100_image_dataset,
)
from native_dicom_amie_roundtrip import (
    VALIDATION_MASTER_KEY,
    validation_nonce,
)
from v4_secure_pipeline import ALGORITHM_VERSION
from native_dicom_authenticated_package_CFP9 import (
    AuthenticatedNativeDicomPackage,
    DICOM_PROFILE,
    PARALLEL_WORKERS,
    decrypt_native_dicom_package,
    encrypt_native_dicom_package,
)


# =========================================================
# PROJECT SETTINGS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "CFP9_NEW_RESULTS" / "native_dicom_package_serialization"
CSV_OUTPUT = OUTPUT_DIR / "native_dicom_package_serialization_100.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "native_dicom_package_serialization_summary.json"
REPRESENTATIVE_DIR = OUTPUT_DIR / "representative"

EXPECTED_SELECTED_FILES = 100

EXPERIMENT_VERSION = "NativeDicomPackageSerialization-CFP9-v2"

# Binary package format identifier.  The final byte is the binary format
# version number (1).  A decoder MUST reject any other magic/version.
PACKAGE_MAGIC = b"AMIE-NATIVE-DICOM-PACKAGE\x00\x01"
PACKAGE_FIELD_COUNT = 7

# Each field is encoded as:
#   uint16_be(label_length) || label || uint64_be(payload_length) || payload
FIELD_ORDER = (
    b"dicom-prefix",
    b"dicom-suffix",
    b"dicom-context",
    b"amie-metadata",
    b"amie-ciphertext",
    b"amie-inner-tag",
    b"dicom-outer-tag",
)

# Defensive parser bound.  This is a format/parser limit, not a DICOM limit.
MAX_PACKAGE_BYTES = 2 * 1024 * 1024 * 1024


# =========================================================
# CANONICAL HELPERS
# =========================================================

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_sha256_hex(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def canonical_json_bytes(value: dict[str, Any]) -> bytes:
    """Exactly the JSON canonicalization used by the authenticated package."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _frame(label: bytes, payload: bytes) -> bytes:
    if not isinstance(label, bytes) or not label:
        raise ValueError("Field label must be non-empty bytes.")
    if len(label) > 0xFFFF:
        raise ValueError("Field label is too long.")
    if not isinstance(payload, bytes):
        raise TypeError("Field payload must be bytes.")

    return (
        len(label).to_bytes(2, "big", signed=False)
        + label
        + len(payload).to_bytes(8, "big", signed=False)
        + payload
    )


def _read_exact(data: bytes, offset: int, length: int) -> tuple[bytes, int]:
    if length < 0:
        raise ValueError("Negative field length is invalid.")
    end = offset + length
    if end > len(data):
        raise ValueError("Truncated serialized native-DICOM package.")
    return data[offset:end], end


def _read_frame(data: bytes, offset: int) -> tuple[bytes, bytes, int]:
    raw, offset = _read_exact(data, offset, 2)
    label_length = int.from_bytes(raw, "big", signed=False)
    if label_length <= 0:
        raise ValueError("Serialized field label must not be empty.")

    label, offset = _read_exact(data, offset, label_length)

    raw, offset = _read_exact(data, offset, 8)
    payload_length = int.from_bytes(raw, "big", signed=False)
    if payload_length > MAX_PACKAGE_BYTES:
        raise ValueError("Serialized field exceeds the configured parser limit.")

    payload, offset = _read_exact(data, offset, payload_length)
    return label, payload, offset


def _parse_canonical_json(payload: bytes, field_name: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {field_name} JSON.") from exc

    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must decode to a JSON object.")

    # Reject semantically equivalent but byte-noncanonical JSON.  This makes
    # the serialized package itself a stable interoperable test vector.
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"{field_name} JSON is not in canonical form.")

    return value


# =========================================================
# DETERMINISTIC PACKAGE SERIALIZATION
# =========================================================

def serialize_native_dicom_package(
    package: AuthenticatedNativeDicomPackage,
) -> bytes:
    encrypted = np.asarray(package.encrypted_image)

    if encrypted.ndim != 2 or encrypted.size == 0:
        raise ValueError("encrypted_image must be a non-empty 2D array.")
    if encrypted.dtype != np.uint8:
        raise ValueError("encrypted_image must use uint8.")
    if len(package.amie_authentication_tag) != 32:
        raise ValueError("AMIE inner HMAC-SHA256 tag must contain 32 bytes.")
    if len(package.dicom_authentication_tag) != 32:
        raise ValueError("DICOM outer HMAC-SHA256 tag must contain 32 bytes.")
    if package.dicom_context.get("profile") != DICOM_PROFILE:
        raise ValueError("Unexpected native-DICOM profile.")

    fields: tuple[tuple[bytes, bytes], ...] = (
        (b"dicom-prefix", bytes(package.dicom_prefix)),
        (b"dicom-suffix", bytes(package.dicom_suffix)),
        (b"dicom-context", canonical_json_bytes(package.dicom_context)),
        (b"amie-metadata", canonical_json_bytes(package.amie_metadata)),
        (
            b"amie-ciphertext",
            np.ascontiguousarray(encrypted).tobytes(order="C"),
        ),
        (b"amie-inner-tag", bytes(package.amie_authentication_tag)),
        (b"dicom-outer-tag", bytes(package.dicom_authentication_tag)),
    )

    if tuple(label for label, _ in fields) != FIELD_ORDER:
        raise AssertionError("Internal package field order changed unexpectedly.")

    encoded = bytearray(PACKAGE_MAGIC)
    encoded.extend(PACKAGE_FIELD_COUNT.to_bytes(2, "big", signed=False))

    for label, payload in fields:
        encoded.extend(_frame(label, payload))

    if len(encoded) > MAX_PACKAGE_BYTES:
        raise ValueError("Serialized package exceeds the configured parser limit.")

    return bytes(encoded)


def deserialize_native_dicom_package(
    data: bytes,
) -> AuthenticatedNativeDicomPackage:
    if not isinstance(data, bytes):
        raise TypeError("Serialized package must be bytes.")
    if len(data) > MAX_PACKAGE_BYTES:
        raise ValueError("Serialized package exceeds the configured parser limit.")

    minimum = len(PACKAGE_MAGIC) + 2
    if len(data) < minimum:
        raise ValueError("Serialized package is too short.")

    if data[: len(PACKAGE_MAGIC)] != PACKAGE_MAGIC:
        raise ValueError("Unsupported or corrupted native-DICOM package magic/version.")

    offset = len(PACKAGE_MAGIC)
    raw, offset = _read_exact(data, offset, 2)
    field_count = int.from_bytes(raw, "big", signed=False)

    if field_count != PACKAGE_FIELD_COUNT:
        raise ValueError(
            f"Expected {PACKAGE_FIELD_COUNT} serialized fields; received {field_count}."
        )

    payloads: dict[bytes, bytes] = {}

    for expected_label in FIELD_ORDER:
        label, payload, offset = _read_frame(data, offset)
        if label != expected_label:
            raise ValueError(
                "Unexpected serialized field order: "
                f"expected {expected_label!r}, received {label!r}."
            )
        if label in payloads:
            raise ValueError(f"Duplicate serialized field {label!r}.")
        payloads[label] = payload

    if offset != len(data):
        raise ValueError("Trailing bytes are not permitted after the final package field.")

    dicom_context = _parse_canonical_json(
        payloads[b"dicom-context"],
        "dicom-context",
    )
    amie_metadata = _parse_canonical_json(
        payloads[b"amie-metadata"],
        "amie-metadata",
    )

    if dicom_context.get("profile") != DICOM_PROFILE:
        raise ValueError("Unsupported native-DICOM profile in serialized package.")

    try:
        padded_shape = amie_metadata["padded_shape"]
        if not isinstance(padded_shape, list) or len(padded_shape) != 2:
            raise ValueError
        rows = int(padded_shape[0])
        columns = int(padded_shape[1])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid padded_shape in serialized AMIE metadata.") from exc

    if rows <= 0 or columns <= 0:
        raise ValueError("Serialized ciphertext dimensions must be positive.")

    ciphertext_payload = payloads[b"amie-ciphertext"]
    expected_ciphertext_bytes = rows * columns

    if expected_ciphertext_bytes != len(ciphertext_payload):
        raise ValueError(
            "Serialized ciphertext byte length does not match the AMIE padded_shape."
        )

    encrypted_image = np.frombuffer(
        ciphertext_payload,
        dtype=np.uint8,
    ).copy().reshape(rows, columns)

    inner_tag = payloads[b"amie-inner-tag"]
    outer_tag = payloads[b"dicom-outer-tag"]

    if len(inner_tag) != 32:
        raise ValueError("Serialized AMIE inner tag must contain 32 bytes.")
    if len(outer_tag) != 32:
        raise ValueError("Serialized DICOM outer tag must contain 32 bytes.")

    return AuthenticatedNativeDicomPackage(
        encrypted_image=encrypted_image,
        amie_metadata=amie_metadata,
        amie_authentication_tag=inner_tag,
        dicom_prefix=payloads[b"dicom-prefix"],
        dicom_suffix=payloads[b"dicom-suffix"],
        dicom_context=dicom_context,
        dicom_authentication_tag=outer_tag,
    )


# =========================================================
# REPRESENTATIVE INTEROPERABILITY TEST VECTOR
# =========================================================

def save_representative_vector(
    *,
    serialized_package: bytes,
    package: AuthenticatedNativeDicomPackage,
    nonce: bytes,
    original_dicom: bytes,
    recovered_dicom: bytes,
) -> None:
    REPRESENTATIVE_DIR.mkdir(parents=True, exist_ok=True)

    package_path = REPRESENTATIVE_DIR / "image_021.amiedcm"
    package_path.write_bytes(serialized_package)

    metadata_bytes = canonical_json_bytes(package.amie_metadata)
    context_bytes = canonical_json_bytes(package.dicom_context)
    ciphertext_bytes = np.ascontiguousarray(
        package.encrypted_image
    ).tobytes(order="C")

    test_vector = {
        "test_vector_version": "AMIE-Native-DICOM-Interop-Vector-CFP9-v1",
        "algorithm_version": ALGORITHM_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "profile": DICOM_PROFILE,
        "purpose": (
            "Public deterministic interoperability vector only; the key and "
            "nonce below are not production secrets."
        ),
        "serialization": {
            "magic_hex": PACKAGE_MAGIC.hex(),
            "field_count_encoding": "uint16 big-endian",
            "field_count": PACKAGE_FIELD_COUNT,
            "field_order": [label.decode("ascii") for label in FIELD_ORDER],
            "field_framing": (
                "uint16_be(label_length) || label || "
                "uint64_be(payload_length) || payload"
            ),
            "json_canonicalization": (
                "UTF-8 JSON; sort_keys=True; separators=(',', ':'); "
                "ensure_ascii=True"
            ),
            "ciphertext_encoding": "uint8 row-major C-order bytes",
            "integer_endianness": "big-endian for package framing",
            "trailing_bytes": "forbidden",
        },
        "validation_master_key_hex": VALIDATION_MASTER_KEY.hex(),
        "validation_key_is_public_test_vector_only": True,
        "validation_session_nonce_hex": nonce.hex(),
        "production_nonce_rule": "fresh 128-bit CSPRNG nonce per encryption",
        "expected_hashes": {
            "serialized_package_sha256": sha256_hex(serialized_package),
            "original_dicom_sha256": sha256_hex(original_dicom),
            "recovered_dicom_sha256": sha256_hex(recovered_dicom),
            "dicom_prefix_sha256": sha256_hex(package.dicom_prefix),
            "dicom_suffix_sha256": sha256_hex(package.dicom_suffix),
            "dicom_context_canonical_json_sha256": sha256_hex(context_bytes),
            "amie_metadata_canonical_json_sha256": sha256_hex(metadata_bytes),
            "amie_ciphertext_sha256": sha256_hex(ciphertext_bytes),
        },
        "expected_tags": {
            "amie_inner_hmac_sha256_hex": package.amie_authentication_tag.hex(),
            "dicom_outer_hmac_sha256_hex": package.dicom_authentication_tag.hex(),
        },
        "expected_ciphertext_shape": [
            int(package.encrypted_image.shape[0]),
            int(package.encrypted_image.shape[1]),
        ],
        "generator_environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy_version": np.__version__,
            "system_byteorder": sys.byteorder,
        },
        "receiver_pass_condition": (
            "Deserialize image_021.amiedcm, verify outer HMAC, verify inner "
            "HMAC, perform inverse AMIE processing, and recover a DICOM whose "
            "SHA-256 equals expected_hashes.original_dicom_sha256."
        ),
    }

    (REPRESENTATIVE_DIR / "image_021_interop_test_vector.json").write_text(
        json.dumps(test_vector, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    (REPRESENTATIVE_DIR / "image_021_original.dcm").write_bytes(
        original_dicom
    )
    (REPRESENTATIVE_DIR / "image_021_recovered_from_serialized.dcm").write_bytes(
        recovered_dicom
    )


# =========================================================
# ONE-FILE VALIDATION
# =========================================================

def evaluate_one_file(record, executor: ProcessPoolExecutor) -> dict[str, Any]:
    nonce = validation_nonce(
        record.selection_index,
        record.destination_name,
    )

    package, encryption_evidence = encrypt_native_dicom_package(
        record.selected_path,
        VALIDATION_MASTER_KEY,
        session_nonce=nonce,
        executor=executor,
    )

    serialized = serialize_native_dicom_package(package)
    parsed = deserialize_native_dicom_package(serialized)
    reserialized = serialize_native_dicom_package(parsed)

    serialization_stable = serialized == reserialized

    recovered_dicom, decryption_evidence = decrypt_native_dicom_package(
        parsed,
        VALIDATION_MASTER_KEY,
        executor=executor,
    )

    original_dicom = bytes(encryption_evidence["original_dicom"])
    dicom_exact = recovered_dicom == original_dicom
    dicom_hash_exact = sha256_hex(recovered_dicom) == sha256_hex(original_dicom)

    outer_tag_preserved = (
        parsed.dicom_authentication_tag
        == package.dicom_authentication_tag
    )
    inner_tag_preserved = (
        parsed.amie_authentication_tag
        == package.amie_authentication_tag
    )
    ciphertext_preserved = np.array_equal(
        parsed.encrypted_image,
        package.encrypted_image,
    )

    if record.destination_name == REPRESENTATIVE_IMAGE_NAME:
        save_representative_vector(
            serialized_package=serialized,
            package=parsed,
            nonce=nonce,
            original_dicom=original_dicom,
            recovered_dicom=recovered_dicom,
        )

    passed = all(
        (
            serialization_stable,
            dicom_exact,
            dicom_hash_exact,
            outer_tag_preserved,
            inner_tag_preserved,
            ciphertext_preserved,
            bool(decryption_evidence["outer_authentication_verified"]),
            bool(decryption_evidence["inner_authentication_verified"]),
        )
    )

    return {
        "experiment_version": EXPERIMENT_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "profile": DICOM_PROFILE,
        "selection_index": record.selection_index,
        "destination_name": record.destination_name,
        "session_nonce_hex": nonce.hex(),
        "serialized_package_bytes": len(serialized),
        "serialized_package_sha256": sha256_hex(serialized),
        "ciphertext_sha256": array_sha256_hex(parsed.encrypted_image),
        "inner_tag_hex": parsed.amie_authentication_tag.hex(),
        "outer_tag_hex": parsed.dicom_authentication_tag.hex(),
        "serialization_stable": serialization_stable,
        "ciphertext_preserved_after_deserialization": ciphertext_preserved,
        "inner_tag_preserved_after_deserialization": inner_tag_preserved,
        "outer_tag_preserved_after_deserialization": outer_tag_preserved,
        "outer_authentication_verified_after_deserialization": bool(
            decryption_evidence["outer_authentication_verified"]
        ),
        "inner_authentication_verified_after_deserialization": bool(
            decryption_evidence["inner_authentication_verified"]
        ),
        "original_dicom_sha256": sha256_hex(original_dicom),
        "recovered_dicom_sha256": sha256_hex(recovered_dicom),
        "dicom_byte_exact_after_serialization": dicom_exact,
        "dicom_sha256_exact_after_serialization": dicom_hash_exact,
        "status": "PASS" if passed else "FAIL",
    }


# =========================================================
# OUTPUT
# =========================================================

def write_csv(rows: list[dict[str, Any]]) -> None:
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(summary: dict[str, Any]) -> None:
    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    print("=" * 104)
    print("NATIVE-DICOM CANONICAL PACKAGE SERIALIZATION + PUBLIC TEST VECTOR — FROZEN 100 IMAGES")
    print("=" * 104)

    print()
    print("[1] Resolving and verifying the frozen dataset...")
    records = resolve_frozen_100_image_dataset()
    if len(records) != EXPECTED_SELECTED_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_SELECTED_FILES} images, found {len(records)}."
        )
    print(f"    PASS: verified frozen subset = {len(records)}/100")

    print()
    print("[2] Canonical serialized package format")
    print(f"    Algorithm version:   {ALGORITHM_VERSION}")
    print(f"    Experiment version:  {EXPERIMENT_VERSION}")
    print(f"    DICOM profile:       {DICOM_PROFILE}")
    print(f"    Magic/version:       {PACKAGE_MAGIC!r}")
    print(f"    Fields:              {PACKAGE_FIELD_COUNT}")
    print("    Field lengths:       uint16/uint64 big-endian")
    print("    JSON:                canonical UTF-8")
    print("    Ciphertext:          uint8 row-major C-order")
    print("    Trailing bytes:      rejected")

    rows: list[dict[str, Any]] = []
    start = time.perf_counter()

    print()
    print("[3] Encrypt -> serialize -> deserialize -> authenticate/decrypt -> exact DICOM...")

    with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        for record in records:
            try:
                result = evaluate_one_file(record, executor)
            except Exception as exc:
                result = {
                    "experiment_version": EXPERIMENT_VERSION,
                    "algorithm_version": ALGORITHM_VERSION,
                    "profile": DICOM_PROFILE,
                    "selection_index": record.selection_index,
                    "destination_name": record.destination_name,
                    "session_nonce_hex": validation_nonce(
                        record.selection_index,
                        record.destination_name,
                    ).hex(),
                    "serialized_package_bytes": "",
                    "serialized_package_sha256": "",
                    "ciphertext_sha256": "",
                    "inner_tag_hex": "",
                    "outer_tag_hex": "",
                    "serialization_stable": False,
                    "ciphertext_preserved_after_deserialization": False,
                    "inner_tag_preserved_after_deserialization": False,
                    "outer_tag_preserved_after_deserialization": False,
                    "outer_authentication_verified_after_deserialization": False,
                    "inner_authentication_verified_after_deserialization": False,
                    "original_dicom_sha256": "",
                    "recovered_dicom_sha256": "",
                    "dicom_byte_exact_after_serialization": False,
                    "dicom_sha256_exact_after_serialization": False,
                    "status": f"FAIL: {type(exc).__name__}: {exc}",
                }

            rows.append(result)
            print(
                f"    {record.destination_name}: {result['status']}"
                + (
                    f" | package={result['serialized_package_bytes']} bytes"
                    if result["status"] == "PASS"
                    else ""
                )
            )

    elapsed = time.perf_counter() - start
    write_csv(rows)

    pass_count = sum(row["status"] == "PASS" for row in rows)
    stable_count = sum(bool(row["serialization_stable"]) for row in rows)
    exact_count = sum(
        bool(row["dicom_byte_exact_after_serialization"])
        for row in rows
    )
    outer_count = sum(
        bool(row["outer_authentication_verified_after_deserialization"])
        for row in rows
    )
    inner_count = sum(
        bool(row["inner_authentication_verified_after_deserialization"])
        for row in rows
    )

    summary = {
        "experiment": "Native-DICOM canonical package serialization and public interoperability vector",
        "experiment_version": EXPERIMENT_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "stage": "CFP9 canonical package serialization and final public interoperability vector",
        "selected_files": len(rows),
        "profile": DICOM_PROFILE,
        "serialization_magic_hex": PACKAGE_MAGIC.hex(),
        "serialization_field_count": PACKAGE_FIELD_COUNT,
        "serialization_field_order": [
            label.decode("ascii") for label in FIELD_ORDER
        ],
        "field_framing": (
            "uint16_be(label_length) || label || "
            "uint64_be(payload_length) || payload"
        ),
        "json_canonicalization": (
            "UTF-8; sort_keys=True; separators=(',', ':'); ensure_ascii=True"
        ),
        "ciphertext_encoding": "uint8 row-major C-order bytes",
        "stable_serialization_count": stable_count,
        "outer_auth_verified_after_deserialization_count": outer_count,
        "inner_auth_verified_after_deserialization_count": inner_count,
        "dicom_byte_exact_after_serialization_count": exact_count,
        "all_100_passed": pass_count == EXPECTED_SELECTED_FILES,
        "validation_master_key_hex": VALIDATION_MASTER_KEY.hex(),
        "validation_key_is_public_test_vector_only": True,
        "production_requires_fresh_csprng_nonce": True,
        "representative_package": str(
            REPRESENTATIVE_DIR / "image_021.amiedcm"
        ),
        "representative_test_vector": str(
            REPRESENTATIVE_DIR / "image_021_interop_test_vector.json"
        ),
        "generator_environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy_version": np.__version__,
            "system_byteorder": sys.byteorder,
        },
        "total_validation_wall_time_sec": float(elapsed),
        "csv_output": str(CSV_OUTPUT),
    }
    write_summary(summary)

    print()
    print("[4] Final result")
    print(f"    Stable serialize->deserialize->serialize: {stable_count}/100")
    print(f"    Outer authentication after deserialize:   {outer_count}/100")
    print(f"    Inner authentication after deserialize:   {inner_count}/100")
    print(f"    Exact DICOM after serialized transport:   {exact_count}/100")
    print(f"    Complete serialization tests passed:      {pass_count}/100")
    print(f"    Total validation wall time:                {elapsed:.3f} s")
    print(f"    CSV:     {CSV_OUTPUT}")
    print(f"    Summary: {SUMMARY_OUTPUT}")
    print(f"    Test package: {REPRESENTATIVE_DIR / 'image_021.amiedcm'}")
    print(
        f"    Test vector:  "
        f"{REPRESENTATIVE_DIR / 'image_021_interop_test_vector.json'}"
    )

    if pass_count != EXPECTED_SELECTED_FILES:
        failed = [
            str(row["destination_name"])
            for row in rows
            if row["status"] != "PASS"
        ]
        raise RuntimeError(
            "Canonical package serialization failed for: "
            + ", ".join(failed)
        )

    print()
    print(
        "PASS: the CFP9 self-contained Native-DICOM envelope has a deterministic "
        "binary transport format and a public image_021 interoperability vector."
    )
    print(
        "READY FOR FINAL INTEROPERABILITY CHECK: move image_021.amiedcm + its JSON test vector to a "
        "materially different platform and verify byte-identical decryption / "
        "Chen-dependent interoperability."
    )


if __name__ == "__main__":
    main()
