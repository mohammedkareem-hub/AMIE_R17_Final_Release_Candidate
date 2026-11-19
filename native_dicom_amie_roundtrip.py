from __future__ import annotations

import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dicom_native_adapter import (
    pixel_data_to_uint8_matrix,
    read_native_pixel_layout,
    uint8_matrix_to_pixel_data,
)

from dicom_native_io import (
    REPRESENTATIVE_IMAGE_NAME,
    reconstruct_native_dicom,
    resolve_frozen_100_image_dataset,
    split_native_dicom_bytes,
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
    / "native_dicom_amie_roundtrip"
)

CSV_OUTPUT = (
    OUTPUT_DIR
    / "native_dicom_amie_roundtrip_100.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "native_dicom_amie_roundtrip_summary.json"
)

REPRESENTATIVE_DIR = (
    OUTPUT_DIR
    / "representative"
)

EXPECTED_SELECTED_FILES = 100
PARALLEL_WORKERS = 8

EXPERIMENT_VERSION = (
    "NativeDicomRoundTrip-CFP9-v2"
)


# =========================================================
# REPRODUCIBLE PUBLIC VALIDATION KEY
# =========================================================

# Public, reproducible TEST key for this validation only.
# It is intentionally deterministic so reviewers can reproduce
# the exact native-DICOM test vectors.
#
# It is NOT a production secret key.

VALIDATION_MASTER_KEY = hashlib.sha256(
    b"AMIE-BakerChen|Native-DICOM|Validation-Master-Key|v1"
).digest()

VALIDATION_NONCE_DOMAIN = (
    b"AMIE-BakerChen|Native-DICOM|Validation-Nonce|v1|"
)


# =========================================================
# REPRODUCIBLE VALIDATION NONCES
# =========================================================

def validation_nonce(
    selection_index: int,
    destination_name: str,
) -> bytes:
    """
    Return one deterministic 128-bit nonce for a
    reproducible validation test vector.

    Production encryption must continue to use a fresh
    cryptographically secure random nonce.
    """

    if selection_index <= 0:
        raise ValueError(
            "selection_index must be positive."
        )

    material = (
        VALIDATION_NONCE_DOMAIN
        + int(
            selection_index
        ).to_bytes(
            4,
            "big",
            signed=False,
        )
        + destination_name.encode(
            "utf-8"
        )
    )

    return hashlib.sha256(
        material
    ).digest()[:16]


# =========================================================
# HASH HELPERS
# =========================================================

def sha256_hex(
    data: bytes,
) -> str:

    return hashlib.sha256(
        data
    ).hexdigest()


def array_sha256_hex(
    array: np.ndarray,
) -> str:

    return hashlib.sha256(
        np.ascontiguousarray(
            array
        ).tobytes(
            order="C"
        )
    ).hexdigest()


# =========================================================
# REPRESENTATIVE ARTIFACTS
# =========================================================

def save_representative_artifacts(
    *,
    original_dicom: bytes,
    recovered_dicom: bytes,
    native_matrix: np.ndarray,
    encrypted_image: np.ndarray,
    recovered_matrix: np.ndarray,
    metadata: dict[str, object],
    authentication_tag: bytes,
    nonce: bytes,
) -> None:

    REPRESENTATIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        REPRESENTATIVE_DIR
        / "image_021_original.dcm"
    ).write_bytes(
        original_dicom
    )

    (
        REPRESENTATIVE_DIR
        / "image_021_recovered_amie.dcm"
    ).write_bytes(
        recovered_dicom
    )

    np.save(
        REPRESENTATIVE_DIR
        / "image_021_native_byte_matrix.npy",
        native_matrix,
        allow_pickle=False,
    )

    np.save(
        REPRESENTATIVE_DIR
        / "image_021_encrypted_byte_matrix.npy",
        encrypted_image,
        allow_pickle=False,
    )

    np.save(
        REPRESENTATIVE_DIR
        / "image_021_recovered_byte_matrix.npy",
        recovered_matrix,
        allow_pickle=False,
    )

    (
        REPRESENTATIVE_DIR
        / "image_021_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    (
        REPRESENTATIVE_DIR
        / "image_021_auth_tag.txt"
    ).write_text(
        authentication_tag.hex(),
        encoding="ascii",
    )

    (
        REPRESENTATIVE_DIR
        / "image_021_validation_nonce.txt"
    ).write_text(
        nonce.hex(),
        encoding="ascii",
    )


# =========================================================
# ONE-FILE END-TO-END AMIE NATIVE-DICOM VALIDATION
# =========================================================

def validate_one_file(
    record,
    executor: ProcessPoolExecutor,
) -> dict[str, object]:

    original_file = (
        record.selected_path.read_bytes()
    )

    parts = split_native_dicom_bytes(
        original_file
    )

    layout = read_native_pixel_layout(
        record.selected_path
    )

    native_matrix = (
        pixel_data_to_uint8_matrix(
            parts.pixel_data,
            layout,
        )
    )

    nonce = validation_nonce(
        record.selection_index,
        record.destination_name,
    )

    wall_start = time.perf_counter()

    package = encrypt_image_secure(
        native_matrix,
        VALIDATION_MASTER_KEY,
        session_nonce=nonce,
        block_size=BLOCK_SIZE,
        max_workers=PARALLEL_WORKERS,
        executor=executor,
    )

    decrypted = decrypt_image_secure(
        package[
            "encrypted_image"
        ],
        package[
            "metadata"
        ],
        package[
            "authentication_tag"
        ],
        VALIDATION_MASTER_KEY,
        max_workers=PARALLEL_WORKERS,
        executor=executor,
    )

    wall_elapsed = (
        time.perf_counter()
        - wall_start
    )

    recovered_matrix = np.asarray(
        decrypted[
            "decrypted_image"
        ],
        dtype=np.uint8,
    )

    recovered_pixel_data = (
        uint8_matrix_to_pixel_data(
            recovered_matrix,
            layout,
        )
    )

    recovered_file = (
        reconstruct_native_dicom(
            parts,
            recovered_pixel_data,
        )
    )

    native_matrix_exact = (
        np.array_equal(
            recovered_matrix,
            native_matrix,
        )
    )

    pixel_data_exact = (
        recovered_pixel_data
        == parts.pixel_data
    )

    dicom_byte_exact = (
        recovered_file
        == original_file
    )

    original_pixel_sha = (
        sha256_hex(
            parts.pixel_data
        )
    )

    recovered_pixel_sha = (
        sha256_hex(
            recovered_pixel_data
        )
    )

    original_file_sha = (
        sha256_hex(
            original_file
        )
    )

    recovered_file_sha = (
        sha256_hex(
            recovered_file
        )
    )

    pixel_sha256_exact = (
        original_pixel_sha
        == recovered_pixel_sha
    )

    dicom_sha256_exact = (
        original_file_sha
        == recovered_file_sha
    )

    ciphertext_differs = (
        not np.array_equal(
            package[
                "encrypted_image"
            ],
            package[
                "stage_padded"
            ],
        )
    )

    if (
        record.destination_name
        == REPRESENTATIVE_IMAGE_NAME
    ):
        save_representative_artifacts(
            original_dicom=
                original_file,

            recovered_dicom=
                recovered_file,

            native_matrix=
                native_matrix,

            encrypted_image=
                package[
                    "encrypted_image"
                ],

            recovered_matrix=
                recovered_matrix,

            metadata=
                package[
                    "metadata"
                ],

            authentication_tag=
                package[
                    "authentication_tag"
                ],

            nonce=
                nonce,
        )

    return {
        "experiment_version":
            EXPERIMENT_VERSION,

        "algorithm_version":
            ALGORITHM_VERSION,

        "selection_index":
            record.selection_index,

        "destination_name":
            record.destination_name,

        "rows":
            layout.rows,

        "columns":
            layout.columns,

        "bits_allocated":
            layout.bits_allocated,

        "bits_stored":
            layout.bits_stored,

        "pixel_representation":
            layout.pixel_representation,

        "native_pixel_data_bytes":
            len(
                parts.pixel_data
            ),

        "native_matrix_rows":
            int(
                native_matrix.shape[
                    0
                ]
            ),

        "native_matrix_columns":
            int(
                native_matrix.shape[
                    1
                ]
            ),

        "native_matrix_sha256":
            array_sha256_hex(
                native_matrix
            ),

        "session_nonce_hex":
            nonce.hex(),

        "ciphertext_rows":
            int(
                package[
                    "encrypted_image"
                ].shape[
                    0
                ]
            ),

        "ciphertext_columns":
            int(
                package[
                    "encrypted_image"
                ].shape[
                    1
                ]
            ),

        "ciphertext_sha256":
            array_sha256_hex(
                package[
                    "encrypted_image"
                ]
            ),

        "authentication_tag_hex":
            package[
                "authentication_tag"
            ].hex(),

        "native_matrix_exact":
            native_matrix_exact,

        "pixel_data_exact":
            pixel_data_exact,

        "pixel_sha256_exact":
            pixel_sha256_exact,

        "dicom_byte_exact":
            dicom_byte_exact,

        "dicom_sha256_exact":
            dicom_sha256_exact,

        "ciphertext_differs_from_padded_plaintext":
            ciphertext_differs,

        "original_pixel_data_sha256":
            original_pixel_sha,

        "recovered_pixel_data_sha256":
            recovered_pixel_sha,

        "original_dicom_sha256":
            original_file_sha,

        "recovered_dicom_sha256":
            recovered_file_sha,

        "encryption_time_sec_pipeline":
            float(
                package[
                    "encryption_time_sec"
                ]
            ),

        "decryption_time_sec_pipeline":
            float(
                decrypted[
                    "decryption_time_sec"
                ]
            ),

        "roundtrip_wall_time_sec":
            float(
                wall_elapsed
            ),

        "status":
            (
                "PASS"
                if (
                    native_matrix_exact
                    and pixel_data_exact
                    and pixel_sha256_exact
                    and dicom_byte_exact
                    and dicom_sha256_exact
                )
                else "FAIL"
            ),
    }


# =========================================================
# OUTPUT WRITERS
# =========================================================

def write_csv(
    rows: list[
        dict[str, object]
    ],
) -> None:

    if not rows:
        raise ValueError(
            "No rows were supplied."
        )

    CSV_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CSV_OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def write_summary(
    summary: dict[
        str,
        object,
    ],
) -> None:

    SUMMARY_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )


# =========================================================
# COMPLETE 100-IMAGE VALIDATION
# =========================================================

def main() -> None:

    print(
        "=" * 96
    )

    print(
        "NATIVE-DICOM + AMIE-BAKERCHEN CFP9 "
        "END-TO-END ROUND-TRIP — FROZEN 100 IMAGES"
    )

    print(
        "=" * 96
    )

    print()

    print(
        "[1] Resolving and verifying "
        "the exact frozen dataset..."
    )

    records = (
        resolve_frozen_100_image_dataset()
    )

    if (
        len(
            records
        )
        != EXPECTED_SELECTED_FILES
    ):
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_SELECTED_FILES} images, "
            f"found {len(records)}."
        )

    print(
        f"    PASS: verified frozen subset "
        f"= {len(records)}/100"
    )

    print()

    print(
        "[2] Validation cryptographic configuration"
    )

    print(
        f"    Algorithm version:      "
        f"{ALGORITHM_VERSION}"
    )

    print(
        f"    Experiment version:     "
        f"{EXPERIMENT_VERSION}"
    )

    print(
        f"    Baker block size:       "
        f"{BLOCK_SIZE} x {BLOCK_SIZE}"
    )

    print(
        f"    Parallel workers:       "
        f"{PARALLEL_WORKERS}"
    )

    print(
        "    Master key:             "
        "fixed PUBLIC validation key "
        "(test vector only)"
    )

    print(
        "    Session nonces:         "
        "deterministic unique validation nonces"
    )

    print(
        "    Production nonce rule:  "
        "fresh CSPRNG nonce remains required"
    )

    rows: list[
        dict[str, object]
    ] = []

    print()

    print(
        "[3] Running native Pixel Data "
        "-> AMIE encrypt "
        "-> authenticate/decrypt "
        "-> DICOM..."
    )

    total_start = (
        time.perf_counter()
    )

    with ProcessPoolExecutor(
        max_workers=
            PARALLEL_WORKERS
    ) as executor:

        for record in records:

            try:
                result = (
                    validate_one_file(
                        record,
                        executor,
                    )
                )

            except Exception as exc:

                result = {
                    "experiment_version":
                        EXPERIMENT_VERSION,

                    "algorithm_version":
                        ALGORITHM_VERSION,

                    "selection_index":
                        record.selection_index,

                    "destination_name":
                        record.destination_name,

                    "rows":
                        "",

                    "columns":
                        "",

                    "bits_allocated":
                        "",

                    "bits_stored":
                        "",

                    "pixel_representation":
                        "",

                    "native_pixel_data_bytes":
                        "",

                    "native_matrix_rows":
                        "",

                    "native_matrix_columns":
                        "",

                    "native_matrix_sha256":
                        "",

                    "session_nonce_hex":
                        validation_nonce(
                            record.selection_index,
                            record.destination_name,
                        ).hex(),

                    "ciphertext_rows":
                        "",

                    "ciphertext_columns":
                        "",

                    "ciphertext_sha256":
                        "",

                    "authentication_tag_hex":
                        "",

                    "native_matrix_exact":
                        False,

                    "pixel_data_exact":
                        False,

                    "pixel_sha256_exact":
                        False,

                    "dicom_byte_exact":
                        False,

                    "dicom_sha256_exact":
                        False,

                    "ciphertext_differs_from_padded_plaintext":
                        False,

                    "original_pixel_data_sha256":
                        "",

                    "recovered_pixel_data_sha256":
                        "",

                    "original_dicom_sha256":
                        "",

                    "recovered_dicom_sha256":
                        "",

                    "encryption_time_sec_pipeline":
                        "",

                    "decryption_time_sec_pipeline":
                        "",

                    "roundtrip_wall_time_sec":
                        "",

                    "status":
                        (
                            f"FAIL: "
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                }

            rows.append(
                result
            )

            print(
                f"    "
                f"{record.destination_name}: "
                f"{result['status']}"
                + (
                    f" | native="
                    f"{result['native_matrix_rows']}x"
                    f"{result['native_matrix_columns']} "
                    f"uint8"
                    f" | cipher="
                    f"{result['ciphertext_rows']}x"
                    f"{result['ciphertext_columns']}"
                    if (
                        result[
                            "status"
                        ]
                        == "PASS"
                    )
                    else ""
                )
            )

    total_wall = (
        time.perf_counter()
        - total_start
    )

    write_csv(
        rows
    )

    passed_rows = [
        row
        for row in rows
        if (
            row[
                "status"
            ]
            == "PASS"
        )
    ]

    native_exact_count = sum(
        bool(
            row[
                "native_matrix_exact"
            ]
        )
        for row in rows
    )

    pixel_exact_count = sum(
        bool(
            row[
                "pixel_data_exact"
            ]
        )
        for row in rows
    )

    pixel_hash_exact_count = sum(
        bool(
            row[
                "pixel_sha256_exact"
            ]
        )
        for row in rows
    )

    dicom_exact_count = sum(
        bool(
            row[
                "dicom_byte_exact"
            ]
        )
        for row in rows
    )

    dicom_hash_exact_count = sum(
        bool(
            row[
                "dicom_sha256_exact"
            ]
        )
        for row in rows
    )

    nonce_count = len(
        {
            str(
                row[
                    "session_nonce_hex"
                ]
            )
            for row in rows
        }
    )

    summary = {
        "experiment":
            (
                "Native-DICOM AMIE-BakerChen "
                "end-to-end round-trip"
            ),

        "experiment_version":
            EXPERIMENT_VERSION,

        "algorithm_version":
            ALGORITHM_VERSION,

        "stage":
            (
                "CFP9 native-DICOM exact "
                "round-trip validation"
            ),

        "selected_files":
            len(
                rows
            ),

        "parallel_workers":
            PARALLEL_WORKERS,

        "block_size":
            BLOCK_SIZE,

        "validation_master_key_hex":
            VALIDATION_MASTER_KEY.hex(),

        "validation_key_is_public_test_vector_only":
            True,

        "deterministic_validation_nonces_only_for_reproducibility":
            True,

        "production_requires_fresh_csprng_nonce":
            True,

        "unique_validation_nonces":
            nonce_count,

        "native_matrix_exact_count":
            native_exact_count,

        "pixel_data_exact_count":
            pixel_exact_count,

        "pixel_sha256_exact_count":
            pixel_hash_exact_count,

        "dicom_byte_exact_count":
            dicom_exact_count,

        "dicom_sha256_exact_count":
            dicom_hash_exact_count,

        "all_100_passed":
            (
                len(
                    passed_rows
                )
                == EXPECTED_SELECTED_FILES
            ),

        "total_validation_wall_time_sec":
            float(
                total_wall
            ),

        "csv_output":
            str(
                CSV_OUTPUT
            ),

        "representative_output_dir":
            str(
                REPRESENTATIVE_DIR
            ),

        "scope_note":
            (
                "This validation demonstrates that native "
                "Pixel Data can pass through the "
                "AMIE-BakerChen-CFP9 authenticated image "
                "pipeline and recover the original DICOM "
                "object byte-for-byte. DICOM prefix/suffix "
                "authentication is evaluated separately in "
                "the authenticated-package experiment."
            ),
    }

    write_summary(
        summary
    )

    print()

    print(
        "[4] Final result"
    )

    print(
        f"    AMIE native byte-matrix exact:      "
        f"{native_exact_count}/"
        f"{EXPECTED_SELECTED_FILES}"
    )

    print(
        f"    Native Pixel Data byte-exact:       "
        f"{pixel_exact_count}/"
        f"{EXPECTED_SELECTED_FILES}"
    )

    print(
        f"    Native Pixel Data SHA-256 exact:    "
        f"{pixel_hash_exact_count}/"
        f"{EXPECTED_SELECTED_FILES}"
    )

    print(
        f"    Full DICOM byte-exact recovery:     "
        f"{dicom_exact_count}/"
        f"{EXPECTED_SELECTED_FILES}"
    )

    print(
        f"    Full DICOM SHA-256 exact recovery:  "
        f"{dicom_hash_exact_count}/"
        f"{EXPECTED_SELECTED_FILES}"
    )

    print(
        f"    Unique validation nonces:           "
        f"{nonce_count}/"
        f"{EXPECTED_SELECTED_FILES}"
    )

    print(
        f"    Total validation wall time:          "
        f"{total_wall:.3f} s"
    )

    print(
        f"    CSV:     "
        f"{CSV_OUTPUT}"
    )

    print(
        f"    Summary: "
        f"{SUMMARY_OUTPUT}"
    )

    print(
        f"    Representative artifacts: "
        f"{REPRESENTATIVE_DIR}"
    )

    if (
        len(
            passed_rows
        )
        != EXPECTED_SELECTED_FILES
    ):
        failed = [
            str(
                row[
                    "destination_name"
                ]
            )
            for row in rows
            if (
                row[
                    "status"
                ]
                != "PASS"
            )
        ]

        raise RuntimeError(
            "Native-DICOM AMIE round-trip "
            "failed for: "
            + ", ".join(
                failed
            )
        )

    if (
        nonce_count
        != EXPECTED_SELECTED_FILES
    ):
        raise RuntimeError(
            "Validation nonce schedule "
            "is not unique across all "
            "100 images."
        )

    print()

    print(
        "PASS: all 100 native DICOM Pixel Data "
        "payloads passed through the "
        "AMIE-BakerChen-CFP9 encryption/"
        "authenticated-decryption pipeline "
        "and reconstructed the original DICOM "
        "objects byte-for-byte."
    )


if __name__ == "__main__":
    main()