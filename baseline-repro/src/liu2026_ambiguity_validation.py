from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np

from dicom_loader import load_dicom_as_uint8

from liu2026_repro import (
    Liu2026,
    Liu2026Config,
    grayscale_to_rgb,
)

from validate_liu2026_5_images import (
    VALIDATION_IMAGES,
    pad_to_square,
    crop_original,
    rgb_cipher_statistics,
    calculate_npcr_uaci_rgb,
    bit_hamming_percent,
    create_modified_plaintext,
)


# ============================================================
# SETTINGS
# ============================================================

EXPERIMENT_VERSION = "Liu2026-AmbiguityValidation-v1"

SELECTED_DIR = Path(r"data\selected")

OUTPUT_DIR = Path(
    r"output\liu2026_ambiguity_validation"
)

STREAM_MODES = [
    "fixed64_low32",
    "mantissa_low32",
]

IT_VALUES = [
    1,
    2,
    3,
    4,
]


# ============================================================
# REFERENCE BEHAVIOR
# ============================================================

# Diagnostic references only.
#
# These values are NOT used to claim reproduction.
# They only quantify how far each reasonable ambiguity
# interpretation is from the statistical behavior expected
# in the Liu paper.

IDEAL_ENTROPY = 8.0

IDEAL_NPCR_PERCENT = 99.6094

IDEAL_UACI_PERCENT = 33.4636

IDEAL_CORRELATION = 0.0


# ============================================================
# HELPERS
# ============================================================

def mean(
    values: list[float],
) -> float:

    return float(
        statistics.mean(
            values
        )
    )


def std(
    values: list[float],
) -> float:

    if len(values) <= 1:
        return 0.0

    return float(
        statistics.stdev(
            values
        )
    )


def write_csv(
    rows: list[dict],
    path: Path,
) -> None:

    if not rows:

        raise ValueError(
            f"No rows to write: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = []

    for row in rows:

        for key in row:

            if key not in fieldnames:

                fieldnames.append(
                    key
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


def validation_paths(
    selected_dir: Path,
) -> list[Path]:

    paths = []

    for filename in VALIDATION_IMAGES:

        path = (
            selected_dir
            / filename
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing image:\n{path}"
            )

        paths.append(
            path
        )

    return paths


# ============================================================
# DIAGNOSTIC SOURCE-DISTANCE
# ============================================================

def source_distance(
    entropy: float,
    npcr: float,
    uaci: float,
    correlation_abs: float,
) -> float:
    """
    Diagnostic measure only.

    NOT a security score.
    NOT evidence of exact reproduction.

    Smaller value means only:
    closer to the published/ideal statistical behavior.
    """

    entropy_difference = (
        abs(
            entropy
            - IDEAL_ENTROPY
        )
        / 8.0
    )

    npcr_difference = (
        abs(
            npcr
            - IDEAL_NPCR_PERCENT
        )
        / 100.0
    )

    uaci_difference = (
        abs(
            uaci
            - IDEAL_UACI_PERCENT
        )
        / 100.0
    )

    correlation_difference = abs(
        correlation_abs
        - IDEAL_CORRELATION
    )

    return float(
        entropy_difference
        + npcr_difference
        + uaci_difference
        + correlation_difference
    )


# ============================================================
# RUN ONE PROFILE
# ============================================================

def run_profile(
    stream_mode: str,
    it_rounds: int,
    image_paths: list[Path],
) -> tuple[
    list[dict],
    dict,
]:

    profile_name = (
        f"{stream_mode}_IT{it_rounds}"
    )

    print()
    print(
        "=" * 100
    )

    print(
        f"PROFILE: {profile_name}"
    )

    print(
        "=" * 100
    )

    config = Liu2026Config(
        epsilon=0.1,
        plm_segments=64,
        plm_mu=4.0,
        warmup=1000,
        it_rounds=it_rounds,
        stream_mode=stream_mode,
        baker_partition=None,
    )

    algorithm = Liu2026(
        config
    )

    rows = []

    for image_index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        # ----------------------------------------------------
        # Original DICOM preprocessing
        # ----------------------------------------------------

        gray = load_dicom_as_uint8(
            image_path
        )

        if gray.dtype != np.uint8:

            raise RuntimeError(
                f"{image_path.name}: "
                "expected uint8."
            )

        if gray.ndim != 2:

            raise RuntimeError(
                f"{image_path.name}: "
                "expected grayscale image."
            )

        # ----------------------------------------------------
        # Controlled square adapter
        # ----------------------------------------------------

        square, adapter_meta = (
            pad_to_square(
                gray
            )
        )

        rgb = grayscale_to_rgb(
            square
        )

        # ----------------------------------------------------
        # One-pixel differential plaintext
        # ----------------------------------------------------

        modified_gray = (
            create_modified_plaintext(
                gray
            )
        )

        modified_square, _ = (
            pad_to_square(
                modified_gray
            )
        )

        modified_rgb = (
            grayscale_to_rgb(
                modified_square
            )
        )

        # ----------------------------------------------------
        # Base encryption
        # ----------------------------------------------------

        (
            cipher,
            metadata,
            enc_time,
        ) = algorithm.encrypt_rgb(
            rgb
        )

        # ----------------------------------------------------
        # Base decryption
        # ----------------------------------------------------

        (
            recovered,
            dec_time,
        ) = algorithm.decrypt_rgb(
            cipher,
            metadata,
        )

        full_exact = bool(
            np.array_equal(
                rgb,
                recovered,
            )
        )

        # ----------------------------------------------------
        # Original-area exact recovery
        # ----------------------------------------------------

        recovered_crop = crop_original(
            recovered,
            adapter_meta,
        )

        expected_crop = grayscale_to_rgb(
            gray
        )

        crop_exact = bool(
            np.array_equal(
                expected_crop,
                recovered_crop,
            )
        )

        if not (
            full_exact
            and crop_exact
        ):

            raise RuntimeError(
                f"{profile_name} / "
                f"{image_path.name}: "
                "exact recovery failed."
            )

        # ----------------------------------------------------
        # Cipher statistics
        # ----------------------------------------------------

        stats = rgb_cipher_statistics(
            cipher
        )

        # ----------------------------------------------------
        # Differential encryption
        # ----------------------------------------------------

        (
            modified_cipher,
            _,
            _,
        ) = algorithm.encrypt_rgb(
            modified_rgb
        )

        (
            npcr,
            uaci,
        ) = calculate_npcr_uaci_rgb(
            cipher,
            modified_cipher,
        )

        bit_hd = bit_hamming_percent(
            cipher,
            modified_cipher,
        )

        row = {
            "profile":
                profile_name,

            "stream_mode":
                stream_mode,

            "it_rounds":
                it_rounds,

            "image_index":
                image_index,

            "image_name":
                image_path.name,

            "original_height":
                int(
                    gray.shape[0]
                ),

            "original_width":
                int(
                    gray.shape[1]
                ),

            "square_size":
                int(
                    adapter_meta[
                        "square_size"
                    ]
                ),

            "exact_recovery":
                True,

            "entropy":
                float(
                    stats[
                        "entropy_pooled"
                    ]
                ),

            "corr_horizontal":
                float(
                    stats[
                        "corr_horizontal_mean"
                    ]
                ),

            "corr_vertical":
                float(
                    stats[
                        "corr_vertical_mean"
                    ]
                ),

            "corr_diagonal":
                float(
                    stats[
                        "corr_diagonal_mean"
                    ]
                ),

            "mean_abs_correlation":
                float(
                    stats[
                        "mean_abs_correlation"
                    ]
                ),

            "npcr_percent":
                float(
                    npcr
                ),

            "uaci_percent":
                float(
                    uaci
                ),

            "bit_hd_percent":
                float(
                    bit_hd
                ),

            "encryption_time_sec":
                float(
                    enc_time
                ),

            "decryption_time_sec":
                float(
                    dec_time
                ),
        }

        rows.append(
            row
        )

        print(
            f"[{image_index}/5] "
            f"{image_path.name}"
        )

        print(
            f"  Exact   : PASS"
        )

        print(
            f"  Entropy : "
            f"{row['entropy']:.6f}"
        )

        print(
            f"  NPCR    : "
            f"{row['npcr_percent']:.6f}%"
        )

        print(
            f"  UACI    : "
            f"{row['uaci_percent']:.6f}%"
        )

        print(
            f"  Bit HD  : "
            f"{row['bit_hd_percent']:.6f}%"
        )

        print(
            f"  Mean|r| : "
            f"{row['mean_abs_correlation']:.6f}"
        )

    # ========================================================
    # PROFILE SUMMARY
    # ========================================================

    entropy_values = [
        row["entropy"]
        for row in rows
    ]

    npcr_values = [
        row["npcr_percent"]
        for row in rows
    ]

    uaci_values = [
        row["uaci_percent"]
        for row in rows
    ]

    bit_hd_values = [
        row["bit_hd_percent"]
        for row in rows
    ]

    correlation_values = [
        row["mean_abs_correlation"]
        for row in rows
    ]

    enc_times = [
        row["encryption_time_sec"]
        for row in rows
    ]

    dec_times = [
        row["decryption_time_sec"]
        for row in rows
    ]

    entropy_mean = mean(
        entropy_values
    )

    npcr_mean = mean(
        npcr_values
    )

    uaci_mean = mean(
        uaci_values
    )

    bit_hd_mean = mean(
        bit_hd_values
    )

    correlation_mean = mean(
        correlation_values
    )

    diagnostic_distance = source_distance(
        entropy=
            entropy_mean,

        npcr=
            npcr_mean,

        uaci=
            uaci_mean,

        correlation_abs=
            correlation_mean,
    )

    summary = {
        "profile":
            profile_name,

        "stream_mode":
            stream_mode,

        "it_rounds":
            it_rounds,

        "images":
            len(
                rows
            ),

        "exact_recovery_count":
            sum(
                bool(
                    row[
                        "exact_recovery"
                    ]
                )
                for row in rows
            ),

        "entropy_mean":
            entropy_mean,

        "entropy_std":
            std(
                entropy_values
            ),

        "npcr_mean_percent":
            npcr_mean,

        "npcr_std_percent":
            std(
                npcr_values
            ),

        "uaci_mean_percent":
            uaci_mean,

        "uaci_std_percent":
            std(
                uaci_values
            ),

        "bit_hd_mean_percent":
            bit_hd_mean,

        "bit_hd_std_percent":
            std(
                bit_hd_values
            ),

        "mean_abs_correlation":
            correlation_mean,

        "mean_abs_correlation_std":
            std(
                correlation_values
            ),

        "encryption_time_mean_sec":
            mean(
                enc_times
            ),

        "decryption_time_mean_sec":
            mean(
                dec_times
            ),

        "diagnostic_source_distance":
            diagnostic_distance,
    }

    print()

    print(
        "PROFILE SUMMARY"
    )

    print(
        "  Exact recovery : "
        f"{summary['exact_recovery_count']}/5"
    )

    print(
        "  Mean entropy   : "
        f"{entropy_mean:.6f}"
    )

    print(
        "  Mean NPCR      : "
        f"{npcr_mean:.6f}%"
    )

    print(
        "  Mean UACI      : "
        f"{uaci_mean:.6f}%"
    )

    print(
        "  Mean bit HD    : "
        f"{bit_hd_mean:.6f}%"
    )

    print(
        "  Mean |r|       : "
        f"{correlation_mean:.6f}"
    )

    print(
        "  Diagnostic dist: "
        f"{diagnostic_distance:.6f}"
    )

    return (
        rows,
        summary,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    selected_dir = (
        SELECTED_DIR.resolve()
    )

    output_dir = (
        OUTPUT_DIR.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = validation_paths(
        selected_dir
    )

    print(
        "=" * 100
    )

    print(
        "LIU ET AL. 2026 — AMBIGUITY VALIDATION"
    )

    print(
        "=" * 100
    )

    print(
        "Dataset:",
        selected_dir,
    )

    print(
        "Images :",
        ", ".join(
            path.name
            for path in paths
        ),
    )

    print()

    print(
        "Testing 8 controlled ambiguity profiles:"
    )

    for mode in STREAM_MODES:

        for it_value in IT_VALUES:

            print(
                f"  {mode}, IT={it_value}"
            )

    print()

    print(
        "NOTE:"
    )

    print(
        "All profiles are retained."
    )

    print(
        "Diagnostic distance is NOT a security score."
    )

    all_rows = []

    summaries = []

    # ========================================================
    # ALL 8 PROFILES
    # ========================================================

    for stream_mode in STREAM_MODES:

        for it_rounds in IT_VALUES:

            rows, summary = (
                run_profile(
                    stream_mode,
                    it_rounds,
                    paths,
                )
            )

            all_rows.extend(
                rows
            )

            summaries.append(
                summary
            )

    # ========================================================
    # RANK ONLY FOR DIAGNOSTIC DISPLAY
    # ========================================================

    ranked = sorted(
        summaries,
        key=lambda item:
            item[
                "diagnostic_source_distance"
            ],
    )

    for rank, summary in enumerate(
        ranked,
        start=1,
    ):

        summary[
            "diagnostic_rank"
        ] = rank

    # ========================================================
    # SAVE FILES
    # ========================================================

    per_image_path = (
        output_dir
        / "liu2026_ambiguity_per_image.csv"
    )

    summary_csv_path = (
        output_dir
        / "liu2026_ambiguity_profile_summary.csv"
    )

    summary_json_path = (
        output_dir
        / "liu2026_ambiguity_summary.json"
    )

    write_csv(
        all_rows,
        per_image_path,
    )

    write_csv(
        ranked,
        summary_csv_path,
    )

    json_data = {
        "experiment_version":
            EXPERIMENT_VERSION,

        "implementation_boundary":
            (
                "Independent controlled "
                "reimplementation; "
                "not author-exact reproduction."
            ),

        "validation_images":
            VALIDATION_IMAGES,

        "tested_stream_modes":
            STREAM_MODES,

        "tested_IT_values":
            IT_VALUES,

        "diagnostic_reference": {
            "entropy":
                IDEAL_ENTROPY,

            "npcr_percent":
                IDEAL_NPCR_PERCENT,

            "uaci_percent":
                IDEAL_UACI_PERCENT,

            "absolute_correlation":
                IDEAL_CORRELATION,
        },

        "diagnostic_warning":
            (
                "The source-distance value is not "
                "a cryptographic security score and "
                "is not evidence of exact reproduction."
            ),

        "profiles":
            ranked,
    }

    summary_json_path.write_text(
        json.dumps(
            json_data,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL COMPARISON TABLE
    # ========================================================

    print()

    print(
        "=" * 128
    )

    print(
        "FINAL LIU-2026 AMBIGUITY COMPARISON"
    )

    print(
        "=" * 128
    )

    print(
        f"{'Rank':>4} "
        f"{'Profile':<28} "
        f"{'Entropy':>9} "
        f"{'NPCR %':>10} "
        f"{'UACI %':>10} "
        f"{'Bit HD %':>10} "
        f"{'Mean|r|':>10} "
        f"{'Distance':>10}"
    )

    print(
        "-" * 128
    )

    for row in ranked:

        print(
            f"{row['diagnostic_rank']:>4} "
            f"{row['profile']:<28} "
            f"{row['entropy_mean']:>9.6f} "
            f"{row['npcr_mean_percent']:>10.6f} "
            f"{row['uaci_mean_percent']:>10.6f} "
            f"{row['bit_hd_mean_percent']:>10.6f} "
            f"{row['mean_abs_correlation']:>10.6f} "
            f"{row['diagnostic_source_distance']:>10.6f}"
        )

    print()

    print(
        "Reference behavior:"
    )

    print(
        "  Entropy -> 8.0"
    )

    print(
        "  NPCR    -> 99.6094%"
    )

    print(
        "  UACI    -> 33.4636%"
    )

    print(
        "  |r|     -> 0"
    )

    print()

    print(
        "DO NOT run the 100-image benchmark yet."
    )

    print(
        "Send this final comparison table first."
    )

    print()

    print(
        "Per-image results:",
        per_image_path,
    )

    print(
        "Profile summary  :",
        summary_csv_path,
    )

    print(
        "Summary JSON     :",
        summary_json_path,
    )

    print(
        "=" * 128
    )


if __name__ == "__main__":

    main()