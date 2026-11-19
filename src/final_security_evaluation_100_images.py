from __future__ import annotations

import copy
import csv
import inspect
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from dicom_loader import load_dicom_as_uint8
from metrics import (
    calculate_entropy,
    calculate_chi_square_uniformity,
    calculate_correlations,
    calculate_npcr_uaci,
    create_one_pixel_modified_image,
    calculate_decryption_quality,
)
from key_schedule_v4 import (
    generate_master_key,
    generate_session_nonce,
)
from select_images import calculate_sha256
from v4_secure_pipeline import (
    ALGORITHM_VERSION,
    BLOCK_SIZE,
    encrypt_image_secure,
    decrypt_image_secure,
)


# =========================================================
# SETTINGS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SELECTED_DIR = (
    PROJECT_ROOT
    / "data"
    / "selected"
)

MANIFEST_PATH = (
    SELECTED_DIR
    / "selected_images.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "CFP9_V2_FINAL_RESULTS"
    / "final_security_100"
)

EXPECTED_IMAGES = 100
PARALLEL_WORKERS = 8

FLIPPED_KEY_BYTE_INDEX = 0
FLIPPED_KEY_BIT_MASK = 0x80

EXPERIMENT_VERSION = "FinalSecurity100-CFP9-v2"


# =========================================================
# PIPELINE COMPATIBILITY
# =========================================================

def call_encrypt(
    image: np.ndarray,
    master_key: bytes,
    session_nonce: bytes,
    executor,
):
    parameters = inspect.signature(
        encrypt_image_secure
    ).parameters

    kwargs = {
        "session_nonce":
            session_nonce,

        "block_size":
            BLOCK_SIZE,

        "max_workers":
            PARALLEL_WORKERS,
    }

    if "executor" in parameters:
        kwargs[
            "executor"
        ] = executor

    return encrypt_image_secure(
        image,
        master_key,
        **kwargs,
    )


def call_decrypt(
    encrypted_image: np.ndarray,
    metadata: dict[str, Any],
    authentication_tag,
    master_key: bytes,
    executor,
):
    parameters = inspect.signature(
        decrypt_image_secure
    ).parameters

    kwargs = {
        "max_workers":
            PARALLEL_WORKERS,
    }

    if "executor" in parameters:
        kwargs[
            "executor"
        ] = executor

    return decrypt_image_secure(
        encrypted_image,
        metadata,
        authentication_tag,
        master_key,
        **kwargs,
    )


# =========================================================
# DATASET FREEZE / MANIFEST PREFLIGHT
# =========================================================

def read_manifest() -> list[dict[str, str]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(
            csv.DictReader(
                file
            )
        )


def verify_frozen_selected_dataset() -> list[Path]:
    manifest_rows = read_manifest()

    if len(
        manifest_rows
    ) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} manifest rows, "
            f"found {len(manifest_rows)}."
        )

    selected_files = sorted(
        SELECTED_DIR.glob(
            "image_*.dcm"
        )
    )

    if len(
        selected_files
    ) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} selected DICOM files, "
            f"found {len(selected_files)}."
        )

    manifest_by_name = {
        str(
            row[
                "destination_name"
            ]
        ):
            row
        for row in manifest_rows
    }

    for path in selected_files:

        row = manifest_by_name.get(
            path.name
        )

        if row is None:
            raise RuntimeError(
                f"{path.name}: not present in manifest."
            )

        actual_size = int(
            path.stat().st_size
        )

        expected_size = int(
            row[
                "source_size_bytes"
            ]
        )

        if actual_size != expected_size:
            raise RuntimeError(
                f"{path.name}: file size does not match manifest. "
                f"actual={actual_size}, expected={expected_size}"
            )

        actual_hash = calculate_sha256(
            path
        )

        expected_hash = str(
            row[
                "sha256"
            ]
        )

        if actual_hash != expected_hash:
            raise RuntimeError(
                f"{path.name}: SHA-256 does not match manifest."
            )

    return selected_files


# =========================================================
# KEY SENSITIVITY HELPERS
# =========================================================

def make_one_bit_changed_key(
    key: bytes,
) -> bytes:
    if len(
        key
    ) != 32:
        raise ValueError(
            "Expected a 256-bit (32-byte) master key."
        )

    changed = bytearray(
        key
    )

    changed[
        FLIPPED_KEY_BYTE_INDEX
    ] ^= FLIPPED_KEY_BIT_MASK

    return bytes(
        changed
    )


def key_bit_difference_count(
    key_a: bytes,
    key_b: bytes,
) -> int:
    if len(
        key_a
    ) != len(
        key_b
    ):
        raise ValueError(
            "Keys must have equal lengths."
        )

    return int(
        sum(
            (
                byte_a
                ^ byte_b
            ).bit_count()
            for byte_a, byte_b in zip(
                key_a,
                key_b,
            )
        )
    )


# =========================================================
# DIFFERENCE METRICS
# =========================================================

def bit_hamming_percent(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    if a.shape != b.shape:
        raise ValueError(
            "Bit Hamming distance requires equal shapes."
        )

    xor_bytes = np.bitwise_xor(
        a.astype(
            np.uint8,
            copy=False,
        ),
        b.astype(
            np.uint8,
            copy=False,
        ),
    )

    differing_bits = np.unpackbits(
        xor_bytes.reshape(
            -1
        )
    )

    return float(
        np.mean(
            differing_bits
        )
        * 100.0
    )


# =========================================================
# AUTHENTICATION TAMPER HELPERS
# =========================================================

def flip_first_hex_nibble(
    value: str,
) -> str:
    if not value:
        raise ValueError(
            "Cannot mutate an empty hex string."
        )

    replacement = (
        "1"
        if value[
            0
        ].lower()
        != "1"
        else "0"
    )

    return (
        replacement
        + value[
            1:
        ]
    )


def mutate_tag(
    tag,
):
    if isinstance(
        tag,
        bytes,
    ):
        if not tag:
            raise ValueError(
                "Authentication tag is empty."
            )

        changed = bytearray(
            tag
        )

        changed[
            0
        ] ^= 0x01

        return bytes(
            changed
        )

    if isinstance(
        tag,
        bytearray,
    ):
        if not tag:
            raise ValueError(
                "Authentication tag is empty."
            )

        changed = bytearray(
            tag
        )

        changed[
            0
        ] ^= 0x01

        return changed

    if isinstance(
        tag,
        str,
    ):
        return flip_first_hex_nibble(
            tag
        )

    raise TypeError(
        f"Unsupported authentication-tag type: {type(tag)!r}"
    )


def authentication_rejection_attempt(
    *,
    encrypted_image: np.ndarray,
    metadata: dict[str, Any],
    authentication_tag,
    master_key: bytes,
    executor,
) -> dict[str, Any]:
    try:
        call_decrypt(
            encrypted_image,
            metadata,
            authentication_tag,
            master_key,
            executor,
        )

        return {
            "rejected":
                False,

            "authentication_failure_message":
                False,

            "error_type":
                None,

            "error_message":
                None,
        }

    except Exception as exc:
        message = str(
            exc
        )

        return {
            "rejected":
                True,

            "authentication_failure_message":
                (
                    "Authentication failed"
                    in message
                ),

            "error_type":
                type(
                    exc
                ).__name__,

            "error_message":
                message,
        }


# =========================================================
# FINAL-CIPHERTEXT STATISTICS
# =========================================================

def ciphertext_statistics(
    encrypted: np.ndarray,
) -> dict[str, float]:
    chi = calculate_chi_square_uniformity(
        encrypted
    )

    corr = calculate_correlations(
        encrypted
    )

    h = float(
        corr[
            "corr_horizontal"
        ]
    )

    v = float(
        corr[
            "corr_vertical"
        ]
    )

    d = float(
        corr[
            "corr_diagonal"
        ]
    )

    return {
        "entropy":
            float(
                calculate_entropy(
                    encrypted
                )
            ),

        "reduced_chi_square":
            float(
                chi[
                    "reduced_chi_square"
                ]
            ),

        "corr_horizontal":
            h,

        "corr_vertical":
            v,

        "corr_diagonal":
            d,

        "mean_abs_correlation":
            float(
                (
                    abs(
                        h
                    )
                    + abs(
                        v
                    )
                    + abs(
                        d
                    )
                )
                / 3.0
            ),
    }


# =========================================================
# STATISTICS
# =========================================================

def sample_std(
    values: list[float],
) -> float:
    if len(
        values
    ) <= 1:
        return 0.0

    return float(
        statistics.stdev(
            values
        )
    )


def ci95_half_width(
    values: list[float],
) -> float:
    n = len(
        values
    )

    if n <= 1:
        return 0.0

    if n == 100:
        t_critical = 1.984
    elif n >= 30:
        t_critical = 1.96
    else:
        t_critical = 2.262

    return float(
        t_critical
        * sample_std(
            values
        )
        / math.sqrt(
            n
        )
    )


def summarize(
    values: list[float],
) -> dict[str, float]:
    return {
        "mean":
            float(
                statistics.mean(
                    values
                )
            ),

        "std":
            sample_std(
                values
            ),

        "median":
            float(
                statistics.median(
                    values
                )
            ),

        "min":
            float(
                min(
                    values
                )
            ),

        "max":
            float(
                max(
                    values
                )
            ),

        "ci95_half_width":
            ci95_half_width(
                values
            ),
    }


# =========================================================
# CSV HELPER
# =========================================================

def write_csv(
    rows: list[dict[str, Any]],
    path: Path,
):
    if not rows:
        raise ValueError(
            f"Cannot write empty CSV: {path}"
        )

    fieldnames: list[str] = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# =========================================================
# MAIN
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 104)
    print(
        "FINAL 100-IMAGE SECURITY EVALUATION"
    )
    print("=" * 104)

    # -----------------------------------------------------
    # DATASET FREEZE CHECK
    # -----------------------------------------------------

    print()
    print(
        "[0] Verifying frozen selected dataset against manifest..."
    )

    images = verify_frozen_selected_dataset()

    print(
        f"  PASS: {len(images)}/{EXPECTED_IMAGES} selected files "
        "match manifest size and SHA-256."
    )

    # -----------------------------------------------------
    # FIXED RESEARCH KEY PAIR
    # -----------------------------------------------------

    master_key_1 = generate_master_key()

    master_key_2 = make_one_bit_changed_key(
        master_key_1
    )

    changed_key_bits = key_bit_difference_count(
        master_key_1,
        master_key_2,
    )

    if changed_key_bits != 1:
        raise RuntimeError(
            f"Expected exactly one changed key bit, got {changed_key_bits}."
        )

    key_record = {
        "experiment_version":
            EXPERIMENT_VERSION,

        "purpose":
            (
                "Research-only reproducibility keys for the final "
                "100-image security evaluation."
            ),

        "master_key_1_hex":
            master_key_1.hex(),

        "master_key_2_hex":
            master_key_2.hex(),

        "master_key_bits":
            256,

        "changed_key_bits":
            changed_key_bits,

        "flipped_byte_index":
            FLIPPED_KEY_BYTE_INDEX,

        "flipped_bit_mask_hex":
            f"0x{FLIPPED_KEY_BIT_MASK:02x}",

        "warning":
            "Do not use these experimental keys for real encryption.",
    }

    keys_path = (
        OUTPUT_DIR
        / "final_security_test_keys.json"
    )

    keys_path.write_text(
        json.dumps(
            key_record,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    rows: list[
        dict[str, Any]
    ] = []

    used_nonces: set[
        bytes
    ] = set()

    with ProcessPoolExecutor(
        max_workers=PARALLEL_WORKERS
    ) as executor:

        for image_index, image_path in enumerate(
            images,
            start=1,
        ):

            print()
            print(
                f"[{image_index:03d}/{len(images):03d}] "
                f"{image_path.name}"
            )

            image = load_dicom_as_uint8(
                image_path
            )

            modified_plaintext = (
                create_one_pixel_modified_image(
                    image,
                    row=0,
                    col=0,
                    delta=1,
                )
            )

            # ---------------------------------------------
            # N1: unique nonce for all controlled comparisons
            # ---------------------------------------------

            while True:
                nonce_1 = generate_session_nonce()

                if nonce_1 not in used_nonces:
                    used_nonces.add(
                        nonce_1
                    )
                    break

            # ---------------------------------------------
            # N2: distinct fresh nonce for randomization test
            # ---------------------------------------------

            while True:
                nonce_2 = generate_session_nonce()

                if nonce_2 not in used_nonces:
                    used_nonces.add(
                        nonce_2
                    )
                    break

            # ---------------------------------------------
            # BASE ENCRYPTION: P, K1, N1
            # ---------------------------------------------

            base_package = call_encrypt(
                image,
                master_key_1,
                nonce_1,
                executor,
            )

            base_cipher = base_package[
                "encrypted_image"
            ]

            stats = ciphertext_statistics(
                base_cipher
            )

            # ---------------------------------------------
            # PLAINTEXT DIFFERENTIAL:
            # P' differs by one pixel, K1 and N1 unchanged
            # ---------------------------------------------

            modified_package = call_encrypt(
                modified_plaintext,
                master_key_1,
                nonce_1,
                executor,
            )

            differential_npcr, differential_uaci = (
                calculate_npcr_uaci(
                    base_cipher,
                    modified_package[
                        "encrypted_image"
                    ],
                )
            )

            differential_bit_hd = bit_hamming_percent(
                base_cipher,
                modified_package[
                    "encrypted_image"
                ],
            )

            # ---------------------------------------------
            # KEY SENSITIVITY:
            # P and N1 unchanged; K2 differs by exactly 1 bit
            # ---------------------------------------------

            key_package = call_encrypt(
                image,
                master_key_2,
                nonce_1,
                executor,
            )

            key_npcr, key_uaci = (
                calculate_npcr_uaci(
                    base_cipher,
                    key_package[
                        "encrypted_image"
                    ],
                )
            )

            key_bit_hd = bit_hamming_percent(
                base_cipher,
                key_package[
                    "encrypted_image"
                ],
            )

            # ---------------------------------------------
            # NONCE RANDOMIZATION:
            # P and K1 unchanged; N2 is fresh
            # ---------------------------------------------

            fresh_nonce_package = call_encrypt(
                image,
                master_key_1,
                nonce_2,
                executor,
            )

            fresh_nonce_npcr, fresh_nonce_uaci = (
                calculate_npcr_uaci(
                    base_cipher,
                    fresh_nonce_package[
                        "encrypted_image"
                    ],
                )
            )

            fresh_nonce_bit_hd = bit_hamming_percent(
                base_cipher,
                fresh_nonce_package[
                    "encrypted_image"
                ],
            )

            # ---------------------------------------------
            # CORRECT-KEY EXACT RECOVERY
            # ---------------------------------------------

            recovered_result = call_decrypt(
                base_package[
                    "encrypted_image"
                ],
                base_package[
                    "metadata"
                ],
                base_package[
                    "authentication_tag"
                ],
                master_key_1,
                executor,
            )

            quality = calculate_decryption_quality(
                image,
                recovered_result[
                    "decrypted_image"
                ],
            )

            exact_recovery = bool(
                quality[
                    "identical"
                ]
            )

            if not exact_recovery:
                raise RuntimeError(
                    f"{image_path.name}: exact recovery failed."
                )

            # ---------------------------------------------
            # AUTH TEST A: WRONG KEY
            # ---------------------------------------------

            wrong_key_result = (
                authentication_rejection_attempt(
                    encrypted_image=base_package[
                        "encrypted_image"
                    ],
                    metadata=copy.deepcopy(
                        base_package[
                            "metadata"
                        ]
                    ),
                    authentication_tag=base_package[
                        "authentication_tag"
                    ],
                    master_key=master_key_2,
                    executor=executor,
                )
            )

            # ---------------------------------------------
            # AUTH TEST B: CIPHERTEXT BIT FLIP
            # ---------------------------------------------

            tampered_cipher = base_package[
                "encrypted_image"
            ].copy()

            tampered_cipher.reshape(
                -1
            )[
                0
            ] ^= np.uint8(
                1
            )

            ciphertext_tamper_result = (
                authentication_rejection_attempt(
                    encrypted_image=tampered_cipher,
                    metadata=copy.deepcopy(
                        base_package[
                            "metadata"
                        ]
                    ),
                    authentication_tag=base_package[
                        "authentication_tag"
                    ],
                    master_key=master_key_1,
                    executor=executor,
                )
            )

            # ---------------------------------------------
            # AUTH TEST C: TAG BIT FLIP
            # ---------------------------------------------

            tag_tamper_result = (
                authentication_rejection_attempt(
                    encrypted_image=base_package[
                        "encrypted_image"
                    ],
                    metadata=copy.deepcopy(
                        base_package[
                            "metadata"
                        ]
                    ),
                    authentication_tag=mutate_tag(
                        base_package[
                            "authentication_tag"
                        ]
                    ),
                    master_key=master_key_1,
                    executor=executor,
                )
            )

            # ---------------------------------------------
            # AUTH TEST D: PLAINTEXT-CONTEXT METADATA CHANGE
            # ---------------------------------------------

            context_metadata = copy.deepcopy(
                base_package[
                    "metadata"
                ]
            )

            context_metadata[
                "plaintext_context_hex"
            ] = flip_first_hex_nibble(
                str(
                    context_metadata[
                        "plaintext_context_hex"
                    ]
                )
            )

            context_tamper_result = (
                authentication_rejection_attempt(
                    encrypted_image=base_package[
                        "encrypted_image"
                    ],
                    metadata=context_metadata,
                    authentication_tag=base_package[
                        "authentication_tag"
                    ],
                    master_key=master_key_1,
                    executor=executor,
                )
            )

            # ---------------------------------------------
            # AUTH TEST E: SESSION-NONCE METADATA CHANGE
            # ---------------------------------------------

            nonce_metadata = copy.deepcopy(
                base_package[
                    "metadata"
                ]
            )

            nonce_metadata[
                "session_nonce_hex"
            ] = flip_first_hex_nibble(
                str(
                    nonce_metadata[
                        "session_nonce_hex"
                    ]
                )
            )

            nonce_tamper_result = (
                authentication_rejection_attempt(
                    encrypted_image=base_package[
                        "encrypted_image"
                    ],
                    metadata=nonce_metadata,
                    authentication_tag=base_package[
                        "authentication_tag"
                    ],
                    master_key=master_key_1,
                    executor=executor,
                )
            )

            # ---------------------------------------------
            # RECORD
            # ---------------------------------------------

            row = {
                "image_index":
                    image_index,

                "image_name":
                    image_path.stem,

                "processed_height":
                    int(
                        image.shape[
                            0
                        ]
                    ),

                "processed_width":
                    int(
                        image.shape[
                            1
                        ]
                    ),

                "processed_payload_bytes":
                    int(
                        image.nbytes
                    ),

                "cipher_height":
                    int(
                        base_cipher.shape[
                            0
                        ]
                    ),

                "cipher_width":
                    int(
                        base_cipher.shape[
                            1
                        ]
                    ),

                "cipher_payload_bytes":
                    int(
                        base_cipher.nbytes
                    ),

                "session_nonce_1_hex":
                    nonce_1.hex(),

                "session_nonce_2_hex":
                    nonce_2.hex(),

                **stats,

                "plaintext_differential_npcr_percent":
                    float(
                        differential_npcr
                    ),

                "plaintext_differential_uaci_percent":
                    float(
                        differential_uaci
                    ),

                "plaintext_differential_bit_hamming_percent":
                    float(
                        differential_bit_hd
                    ),

                "key_npcr_percent":
                    float(
                        key_npcr
                    ),

                "key_uaci_percent":
                    float(
                        key_uaci
                    ),

                "key_ciphertext_bit_hamming_percent":
                    float(
                        key_bit_hd
                    ),

                "fresh_nonce_npcr_percent":
                    float(
                        fresh_nonce_npcr
                    ),

                "fresh_nonce_uaci_percent":
                    float(
                        fresh_nonce_uaci
                    ),

                "fresh_nonce_ciphertext_bit_hamming_percent":
                    float(
                        fresh_nonce_bit_hd
                    ),

                "exact_recovery":
                    exact_recovery,

                "mse":
                    float(
                        quality[
                            "mse"
                        ]
                    ),

                "mae":
                    float(
                        quality[
                            "mae"
                        ]
                    ),

                "wrong_key_rejected":
                    bool(
                        wrong_key_result[
                            "rejected"
                        ]
                    ),

                "wrong_key_auth_failure":
                    bool(
                        wrong_key_result[
                            "authentication_failure_message"
                        ]
                    ),

                "ciphertext_tamper_rejected":
                    bool(
                        ciphertext_tamper_result[
                            "rejected"
                        ]
                    ),

                "ciphertext_tamper_auth_failure":
                    bool(
                        ciphertext_tamper_result[
                            "authentication_failure_message"
                        ]
                    ),

                "tag_tamper_rejected":
                    bool(
                        tag_tamper_result[
                            "rejected"
                        ]
                    ),

                "tag_tamper_auth_failure":
                    bool(
                        tag_tamper_result[
                            "authentication_failure_message"
                        ]
                    ),

                "plaintext_context_tamper_rejected":
                    bool(
                        context_tamper_result[
                            "rejected"
                        ]
                    ),

                "plaintext_context_tamper_auth_failure":
                    bool(
                        context_tamper_result[
                            "authentication_failure_message"
                        ]
                    ),

                "session_nonce_tamper_rejected":
                    bool(
                        nonce_tamper_result[
                            "rejected"
                        ]
                    ),

                "session_nonce_tamper_auth_failure":
                    bool(
                        nonce_tamper_result[
                            "authentication_failure_message"
                        ]
                    ),
            }

            rows.append(
                row
            )

            print(
                "  Entropy / |corr| : "
                f"{row['entropy']:.6f} / "
                f"{row['mean_abs_correlation']:.6f}"
            )

            print(
                "  Plain NPCR/UACI  : "
                f"{row['plaintext_differential_npcr_percent']:.6f}% / "
                f"{row['plaintext_differential_uaci_percent']:.6f}%"
            )

            print(
                "  Key NPCR/UACI    : "
                f"{row['key_npcr_percent']:.6f}% / "
                f"{row['key_uaci_percent']:.6f}%"
            )

            print(
                "  Key bit-HD       : "
                f"{row['key_ciphertext_bit_hamming_percent']:.6f}%"
            )

            print(
                "  Exact recovery   : True"
            )

            print(
                "  Auth rejection   : "
                f"{int(row['wrong_key_auth_failure']) + int(row['ciphertext_tamper_auth_failure']) + int(row['tag_tamper_auth_failure']) + int(row['plaintext_context_tamper_auth_failure']) + int(row['session_nonce_tamper_auth_failure'])}/5"
            )

    # =====================================================
    # PER-IMAGE CSV
    # =====================================================

    detailed_path = (
        OUTPUT_DIR
        / "final_security_100_images.csv"
    )

    write_csv(
        rows,
        detailed_path,
    )

    # =====================================================
    # NUMERICAL AGGREGATES
    # =====================================================

    numerical_metrics = [
        (
            "Ciphertext entropy (bits/pixel)",
            "entropy",
        ),
        (
            "Reduced chi-square",
            "reduced_chi_square",
        ),
        (
            "Horizontal correlation",
            "corr_horizontal",
        ),
        (
            "Vertical correlation",
            "corr_vertical",
        ),
        (
            "Diagonal correlation",
            "corr_diagonal",
        ),
        (
            "Mean absolute adjacent correlation",
            "mean_abs_correlation",
        ),
        (
            "Plaintext differential NPCR (%)",
            "plaintext_differential_npcr_percent",
        ),
        (
            "Plaintext differential UACI (%)",
            "plaintext_differential_uaci_percent",
        ),
        (
            "Plaintext differential ciphertext bit HD (%)",
            "plaintext_differential_bit_hamming_percent",
        ),
        (
            "Independent Key-NPCR (%)",
            "key_npcr_percent",
        ),
        (
            "Independent Key-UACI (%)",
            "key_uaci_percent",
        ),
        (
            "Independent key ciphertext bit HD (%)",
            "key_ciphertext_bit_hamming_percent",
        ),
        (
            "Fresh-nonce ciphertext NPCR (%)",
            "fresh_nonce_npcr_percent",
        ),
        (
            "Fresh-nonce ciphertext UACI (%)",
            "fresh_nonce_uaci_percent",
        ),
        (
            "Fresh-nonce ciphertext bit HD (%)",
            "fresh_nonce_ciphertext_bit_hamming_percent",
        ),
    ]

    aggregate_rows = []

    aggregate_json: dict[
        str,
        Any
    ] = {}

    for display_name, field in numerical_metrics:

        values = [
            float(
                row[
                    field
                ]
            )
            for row in rows
        ]

        stats = summarize(
            values
        )

        aggregate_json[
            field
        ] = stats

        aggregate_rows.append(
            {
                "category":
                    "Numerical aggregate",

                "metric":
                    display_name,

                "field":
                    field,

                "n":
                    len(
                        values
                    ),

                "mean":
                    stats[
                        "mean"
                    ],

                "std":
                    stats[
                        "std"
                    ],

                "ci95_half_width":
                    stats[
                        "ci95_half_width"
                    ],

                "median":
                    stats[
                        "median"
                    ],

                "min":
                    stats[
                        "min"
                    ],

                "max":
                    stats[
                        "max"
                    ],

                "count":
                    "",

                "rate_percent":
                    "",
            }
        )

    # =====================================================
    # BINARY SECURITY / CORRECTNESS RATES
    # =====================================================

    binary_metrics = [
        (
            "Correct-key exact recovery",
            "exact_recovery",
        ),
        (
            "Wrong-key authentication rejection",
            "wrong_key_auth_failure",
        ),
        (
            "Ciphertext-bit tamper authentication rejection",
            "ciphertext_tamper_auth_failure",
        ),
        (
            "Authentication-tag tamper rejection",
            "tag_tamper_auth_failure",
        ),
        (
            "Protected plaintext-context metadata tamper rejection",
            "plaintext_context_tamper_auth_failure",
        ),
        (
            "Session-nonce metadata tamper rejection",
            "session_nonce_tamper_auth_failure",
        ),
    ]

    binary_json = {}

    for display_name, field in binary_metrics:

        count = sum(
            bool(
                row[
                    field
                ]
            )
            for row in rows
        )

        rate = float(
            count
            / len(
                rows
            )
            * 100.0
        )

        binary_json[
            field
        ] = {
            "count":
                count,

            "total":
                len(
                    rows
                ),

            "rate_percent":
                rate,
        }

        aggregate_rows.append(
            {
                "category":
                    "Binary security/correctness rate",

                "metric":
                    display_name,

                "field":
                    field,

                "n":
                    len(
                        rows
                    ),

                "mean":
                    "",

                "std":
                    "",

                "ci95_half_width":
                    "",

                "median":
                    "",

                "min":
                    "",

                "max":
                    "",

                "count":
                    count,

                "rate_percent":
                    rate,
            }
        )

    aggregate_path = (
        OUTPUT_DIR
        / "final_security_100_paper_ready.csv"
    )

    write_csv(
        aggregate_rows,
        aggregate_path,
    )

    # =====================================================
    # FINAL VALIDATION FLAGS
    # =====================================================

    all_exact = all(
        bool(
            row[
                "exact_recovery"
            ]
        )
        for row in rows
    )

    all_wrong_key_auth_rejected = all(
        bool(
            row[
                "wrong_key_auth_failure"
            ]
        )
        for row in rows
    )

    all_ciphertext_tamper_auth_rejected = all(
        bool(
            row[
                "ciphertext_tamper_auth_failure"
            ]
        )
        for row in rows
    )

    all_tag_tamper_auth_rejected = all(
        bool(
            row[
                "tag_tamper_auth_failure"
            ]
        )
        for row in rows
    )

    all_context_tamper_auth_rejected = all(
        bool(
            row[
                "plaintext_context_tamper_auth_failure"
            ]
        )
        for row in rows
    )

    all_nonce_tamper_auth_rejected = all(
        bool(
            row[
                "session_nonce_tamper_auth_failure"
            ]
        )
        for row in rows
    )

    # =====================================================
    # SUMMARY JSON
    # =====================================================

    summary = {
        "experiment_version":
            EXPERIMENT_VERSION,

        "algorithm_version":
            ALGORITHM_VERSION,

        "experiment":
            (
               "Final unified 100-image security evaluation "
                    f"for {ALGORITHM_VERSION}"
            ),

        "dataset_preflight":
            {
                "selected_files":
                    len(
                        images
                    ),

                "manifest_size_and_sha256_verified":
                    True,

                "dataset_status":
                    "Frozen manifest-consistent 100-image selected dataset",
            },

        "number_of_images":
            len(
                rows
            ),

        "parallel_workers":
            PARALLEL_WORKERS,

        "block_size":
            BLOCK_SIZE,

        "master_key_bits":
            256,

        "key_sensitivity_protocol":
            {
                "same_plaintext":
                    True,

                "same_nonce_within_each_K1_K2_comparison":
                    True,

                "changed_master_key_bits":
                    changed_key_bits,

                "same_fixed_key_pair_across_all_100_images":
                    True,
            },

        "plaintext_differential_protocol":
            {
                "plaintext_difference":
                    (
                        "Exactly one pixel changed by +1 modulo 256 "
                        "at row 0, column 0"
                    ),

                "same_master_key":
                    True,

                "same_session_nonce":
                    True,
            },

        "fresh_nonce_protocol":
            {
                "same_plaintext":
                    True,

                "same_master_key":
                    True,

                "different_unique_session_nonce":
                    True,
            },

        "numerical_aggregate_metrics":
            aggregate_json,

        "binary_security_and_correctness":
            binary_json,

        "validation":
            {
                "all_exact_recovery":
                    all_exact,

                "all_wrong_key_authentication_rejected":
                    all_wrong_key_auth_rejected,

                "all_ciphertext_tampering_authentication_rejected":
                    all_ciphertext_tamper_auth_rejected,

                "all_tag_tampering_authentication_rejected":
                    all_tag_tamper_auth_rejected,

                "all_plaintext_context_tampering_authentication_rejected":
                    all_context_tamper_auth_rejected,

                "all_session_nonce_tampering_authentication_rejected":
                    all_nonce_tamper_auth_rejected,
            },

        "manuscript_use_note":
            (
                "Use these aggregate results as the primary quantitative "
                "security evidence. Single-image results may be retained only "
                "as illustrative figures/examples and must not be mixed with "
                "the 100-image aggregate table."
            ),

        "key_sensitivity_note":
            (
                "Independent Key-NPCR/Key-UACI are produced by changing "
                "exactly one master-key bit while holding plaintext and nonce "
                "fixed. They are distinct from plaintext differential NPCR/UACI."
            ),

        "authentication_note":
            (
                "Authentication rates report rejection specifically through "
                "the authentication-failure path for wrong-key, ciphertext, "
                "tag, protected-context metadata, and session-nonce metadata "
                "tests."
            ),

        "replay_scope_note":
            (
                "Replay resistance is a separate stateful receiver-side "
                "freshness property and is intentionally not conflated with "
                "these stateless AMIE-BakerChen CFP9-v2 cryptographic metrics."
            ),

        "losslessness_scope":
            (
                "Exact recovery refers to the processed uint8 image, not "
                "bit-exact reconstruction of the original DICOM object."
            ),

        "security_claim_boundary":
            (
                "These are empirical statistical, differential, key-sensitivity, "
                "correctness, and authentication results over 100 images. "
                "They do not constitute a formal IND-CPA or IND-CCA proof."
            ),
    }

    summary_path = (
        OUTPUT_DIR
        / "final_security_100_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # =====================================================
    # FINAL CONSOLE REPORT
    # =====================================================

    print()
    print("=" * 104)
    print(
        "FINAL 100-IMAGE SECURITY SUMMARY"
    )
    print("=" * 104)

    for display_name, field in numerical_metrics:

        stats = aggregate_json[
            field
        ]

        print()
        print(
            display_name
        )

        print(
            "  Mean ± SD: "
            f"{stats['mean']:.6f} ± {stats['std']:.6f}"
        )

        print(
            "  95% CI   : "
            f"{stats['mean']:.6f} ± "
            f"{stats['ci95_half_width']:.6f}"
        )

    print()
    print("-" * 104)

    for display_name, field in binary_metrics:

        item = binary_json[
            field
        ]

        print(
            f"{display_name}: "
            f"{item['count']}/{item['total']} "
            f"({item['rate_percent']:.1f}%)"
        )

    print()
    print(
        "Dataset manifest/SHA-256 preflight:",
        "PASS",
    )

    print(
        "All exact recoveries:",
        all_exact,
    )

    print(
        "All five authentication attack classes rejected on all images:",
        bool(
            all_wrong_key_auth_rejected
            and all_ciphertext_tamper_auth_rejected
            and all_tag_tamper_auth_rejected
            and all_context_tamper_auth_rejected
            and all_nonce_tamper_auth_rejected
        ),
    )

    print()
    print(
        "IMPORTANT: The paper-ready table from this run is the "
        "primary final security table. Do not mix it with older "
        "single-image or pre-dataset-fix outputs."
    )

    print()
    print("Saved:")
    print(detailed_path)
    print(aggregate_path)
    print(summary_path)
    print(keys_path)

    print("=" * 104)


if __name__ == "__main__":
    main()
