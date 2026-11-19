from __future__ import annotations

import argparse
import csv
import hashlib
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
    pad_to_square,
    crop_original,
    rgb_cipher_statistics,
    calculate_npcr_uaci_rgb,
    bit_hamming_percent,
    create_modified_plaintext,
)


EXPECTED_IMAGES = 100
MEASURED_REPEATS = 3

EXPERIMENT_VERSION = (
    "Liu2026-Controlled100-Fixed64-IT3-v1"
)


# ============================================================
# HELPERS
# ============================================================

def mean_value(values):
    return float(
        statistics.mean(values)
    )


def sample_std(values):

    if len(values) <= 1:
        return 0.0

    return float(
        statistics.stdev(values)
    )


def summarize(values):

    return {
        "n":
            len(values),

        "mean":
            mean_value(values),

        "std":
            sample_std(values),

        "median":
            float(
                statistics.median(values)
            ),

        "min":
            float(
                min(values)
            ),

        "max":
            float(
                max(values)
            ),
    }


def write_csv(rows, path):

    if not rows:
        raise ValueError(
            f"No rows to write: {path}"
        )

    fields = []

    for row in rows:

        for key in row:

            if key not in fields:
                fields.append(key)

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
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(rows)


def sha256_file(path):

    hasher = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


# ============================================================
# VERIFY FROZEN DATASET
# ============================================================

def verify_dataset(
    selected_dir: Path,
):

    if not selected_dir.exists():

        raise FileNotFoundError(
            f"Dataset folder not found:\n"
            f"{selected_dir}"
        )

    images = sorted(
        selected_dir.glob(
            "image_*.dcm"
        )
    )

    if len(images) != EXPECTED_IMAGES:

        raise RuntimeError(
            f"Expected 100 images, "
            f"found {len(images)}."
        )

    expected_names = [
        f"image_{index:03d}.dcm"
        for index in range(
            1,
            101,
        )
    ]

    actual_names = [
        path.name
        for path in images
    ]

    if actual_names != expected_names:

        raise RuntimeError(
            "Image names do not match "
            "image_001.dcm ... image_100.dcm."
        )

    manifest_path = (
        selected_dir
        / "selected_images.csv"
    )

    if manifest_path.exists():

        print(
            "Manifest detected. "
            "Verifying frozen dataset..."
        )

        with manifest_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            rows = list(
                csv.DictReader(file)
            )

        if len(rows) != 100:

            raise RuntimeError(
                "Manifest does not contain "
                "exactly 100 rows."
            )

        if (
            rows
            and
            "destination_name"
            in rows[0]
        ):

            by_name = {
                row[
                    "destination_name"
                ]:
                    row
                for row in rows
            }

            for image_path in images:

                row = by_name.get(
                    image_path.name
                )

                if row is None:

                    raise RuntimeError(
                        f"{image_path.name} "
                        "missing from manifest."
                    )

                if (
                    row.get(
                        "source_size_bytes",
                        "",
                    )
                    .strip()
                ):

                    expected_size = int(
                        row[
                            "source_size_bytes"
                        ]
                    )

                    if (
                        image_path.stat().st_size
                        != expected_size
                    ):

                        raise RuntimeError(
                            f"{image_path.name}: "
                            "file-size mismatch."
                        )

                if (
                    row.get(
                        "sha256",
                        "",
                    )
                    .strip()
                ):

                    expected_sha = (
                        row[
                            "sha256"
                        ]
                        .strip()
                        .lower()
                    )

                    actual_sha = (
                        sha256_file(
                            image_path
                        )
                        .lower()
                    )

                    if (
                        actual_sha
                        != expected_sha
                    ):

                        raise RuntimeError(
                            f"{image_path.name}: "
                            "SHA-256 mismatch."
                        )

        print(
            "Frozen dataset verification: PASS"
        )

    else:

        print(
            "WARNING: selected_images.csv "
            "not found."
        )

    return images


# ============================================================
# PER-CHANNEL DIFFERENTIAL METRICS
# ============================================================

def channel_npcr_uaci(
    first,
    second,
):

    result = {}

    channel_names = [
        "r",
        "g",
        "b",
    ]

    npcr_values = []
    uaci_values = []

    for index, name in enumerate(
        channel_names
    ):

        first_channel = first[
            :,
            :,
            index
        ]

        second_channel = second[
            :,
            :,
            index
        ]

        changed = (
            first_channel
            != second_channel
        )

        npcr = float(
            np.mean(
                changed
            )
            * 100.0
        )

        difference = np.abs(
            first_channel.astype(
                np.int16
            )
            -
            second_channel.astype(
                np.int16
            )
        )

        uaci = float(
            np.mean(
                difference
                / 255.0
            )
            * 100.0
        )

        result[
            f"npcr_{name}_percent"
        ] = npcr

        result[
            f"uaci_{name}_percent"
        ] = uaci

        npcr_values.append(npcr)
        uaci_values.append(uaci)

    result[
        "npcr_channel_mean_percent"
    ] = mean_value(
        npcr_values
    )

    result[
        "uaci_channel_mean_percent"
    ] = mean_value(
        uaci_values
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "100-image controlled benchmark "
            "for Liu et al. 2026."
        )
    )

    parser.add_argument(
        "--selected-dir",
        default=r"data\selected",
    )

    parser.add_argument(
        "--output-dir",
        default=r"output\liu2026_100",
    )

    args = parser.parse_args()

    selected_dir = Path(
        args.selected_dir
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 104
    )

    print(
        "LIU ET AL. 2026 — "
        "CONTROLLED 100-IMAGE BENCHMARK"
    )

    print(
        "=" * 104
    )

    print(
        "Dataset :",
        selected_dir,
    )

    print(
        "Output  :",
        output_dir,
    )

    print(
        "Profile : fixed64_low32, IT=3"
    )

    print(
        "Repeats :",
        MEASURED_REPEATS,
    )

    print()

    print(
        "Independent controlled "
        "reimplementation;"
    )

    print(
        "not author-exact reproduction."
    )

    print()

    images = verify_dataset(
        selected_dir
    )

    # ========================================================
    # FROZEN PROFILE
    # ========================================================

    config = Liu2026Config(
        epsilon=0.1,
        plm_segments=64,
        plm_mu=4.0,
        warmup=1000,

        # Frozen after ambiguity analysis.
        it_rounds=3,

        stream_mode=
            "fixed64_low32",

        baker_partition=None,
    )

    algorithm = Liu2026(
        config
    )

    # ========================================================
    # EXCLUDED WARM-UP
    # ========================================================

    print()

    print(
        "Running excluded warm-up..."
    )

    warm_gray = load_dicom_as_uint8(
        images[0]
    )

    warm_square, warm_adapter = (
        pad_to_square(
            warm_gray
        )
    )

    warm_rgb = grayscale_to_rgb(
        warm_square
    )

    (
        warm_cipher,
        warm_meta,
        _,
    ) = algorithm.encrypt_rgb(
        warm_rgb
    )

    (
        warm_recovered,
        _,
    ) = algorithm.decrypt_rgb(
        warm_cipher,
        warm_meta,
    )

    if not np.array_equal(
        warm_rgb,
        warm_recovered,
    ):

        raise RuntimeError(
            "Warm-up exact recovery failed."
        )

    print(
        "Warm-up exact recovery: PASS"
    )

    # ========================================================
    # STORAGE
    # ========================================================

    results = []
    timings = []

    total_logical_bytes = 0
    total_liu_rgb_bytes = 0

    # ========================================================
    # ALL 100 IMAGES
    # ========================================================

    for image_index, image_path in enumerate(
        images,
        start=1,
    ):

        print()

        print(
            f"[{image_index:03d}/100] "
            f"{image_path.name}"
        )

        # ----------------------------------------------------
        # Same project preprocessing
        # ----------------------------------------------------

        gray = load_dicom_as_uint8(
            image_path
        )

        if (
            gray.dtype != np.uint8
            or
            gray.ndim != 2
        ):

            raise RuntimeError(
                f"{image_path.name}: "
                "expected 2-D uint8 image."
            )

        # ----------------------------------------------------
        # Liu adapter
        # ----------------------------------------------------

        square, adapter = (
            pad_to_square(
                gray
            )
        )

        rgb = grayscale_to_rgb(
            square
        )

        # ----------------------------------------------------
        # Differential plaintext
        # ----------------------------------------------------

        modified_gray = (
            create_modified_plaintext(
                gray
            )
        )

        modified_square, modified_adapter = (
            pad_to_square(
                modified_gray
            )
        )

        if (
            modified_adapter[
                "square_size"
            ]
            != adapter[
                "square_size"
            ]
        ):

            raise RuntimeError(
                "Differential adapter mismatch."
            )

        modified_rgb = (
            grayscale_to_rgb(
                modified_square
            )
        )

        logical_bytes = int(
            gray.size
        )

        liu_rgb_bytes = int(
            rgb.size
        )

        total_logical_bytes += (
            logical_bytes
        )

        total_liu_rgb_bytes += (
            liu_rgb_bytes
        )

        # ----------------------------------------------------
        # THREE TIMING REPEATS
        # ----------------------------------------------------

        encryption_times = []
        decryption_times = []

        base_cipher = None
        base_metadata = None

        for repeat_index in range(
            1,
            MEASURED_REPEATS + 1,
        ):

            (
                cipher,
                metadata,
                encryption_time,
            ) = algorithm.encrypt_rgb(
                rgb
            )

            (
                recovered,
                decryption_time,
            ) = algorithm.decrypt_rgb(
                cipher,
                metadata,
            )

            if not np.array_equal(
                rgb,
                recovered,
            ):

                raise RuntimeError(
                    f"{image_path.name}: "
                    f"exact recovery failed "
                    f"repeat {repeat_index}."
                )

            recovered_crop = crop_original(
                recovered,
                adapter,
            )

            expected_crop = grayscale_to_rgb(
                gray
            )

            if not np.array_equal(
                recovered_crop,
                expected_crop,
            ):

                raise RuntimeError(
                    f"{image_path.name}: "
                    "original-area recovery failed."
                )

            encryption_times.append(
                float(
                    encryption_time
                )
            )

            decryption_times.append(
                float(
                    decryption_time
                )
            )

            timings.append(
                {
                    "image_name":
                        image_path.name,

                    "repeat_index":
                        repeat_index,

                    "operation":
                        "encryption",

                    "time_sec":
                        float(
                            encryption_time
                        ),
                }
            )

            timings.append(
                {
                    "image_name":
                        image_path.name,

                    "repeat_index":
                        repeat_index,

                    "operation":
                        "decryption",

                    "time_sec":
                        float(
                            decryption_time
                        ),
                }
            )

            if repeat_index == 1:

                base_cipher = (
                    cipher.copy()
                )

                base_metadata = (
                    metadata
                )

        # ----------------------------------------------------
        # CIPHERTEXT STATISTICS
        # ----------------------------------------------------

        stats = rgb_cipher_statistics(
            base_cipher
        )

        # ----------------------------------------------------
        # DIFFERENTIAL ENCRYPTION
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
            base_cipher,
            modified_cipher,
        )

        channel_metrics = (
            channel_npcr_uaci(
                base_cipher,
                modified_cipher,
            )
        )

        bit_hd = (
            bit_hamming_percent(
                base_cipher,
                modified_cipher,
            )
        )

        # ----------------------------------------------------
        # FINAL RECOVERY CHECK
        # ----------------------------------------------------

        (
            final_recovered,
            _,
        ) = algorithm.decrypt_rgb(
            base_cipher,
            base_metadata,
        )

        exact_padded = bool(
            np.array_equal(
                rgb,
                final_recovered,
            )
        )

        final_crop = crop_original(
            final_recovered,
            adapter,
        )

        exact_original = bool(
            np.array_equal(
                final_crop,
                grayscale_to_rgb(
                    gray
                ),
            )
        )

        if not (
            exact_padded
            and exact_original
        ):

            raise RuntimeError(
                f"{image_path.name}: "
                "final exact recovery failed."
            )

        # ----------------------------------------------------
        # TIMING SUMMARY
        # ----------------------------------------------------

        enc_mean = mean_value(
            encryption_times
        )

        dec_mean = mean_value(
            decryption_times
        )

        logical_mb = (
            logical_bytes
            / 1_000_000.0
        )

        liu_rgb_mb = (
            liu_rgb_bytes
            / 1_000_000.0
        )

        row = {
            "experiment_version":
                EXPERIMENT_VERSION,

            "image_index":
                image_index,

            "image_name":
                image_path.name,

            "height":
                int(
                    gray.shape[0]
                ),

            "width":
                int(
                    gray.shape[1]
                ),

            "liu_square_size":
                int(
                    adapter[
                        "square_size"
                    ]
                ),

            "pad_bottom":
                int(
                    adapter[
                        "pad_bottom"
                    ]
                ),

            "pad_right":
                int(
                    adapter[
                        "pad_right"
                    ]
                ),

            "logical_grayscale_bytes":
                logical_bytes,

            "liu_processed_rgb_bytes":
                liu_rgb_bytes,

            "entropy_pooled":
                float(
                    stats[
                        "entropy_pooled"
                    ]
                ),

            "entropy_channel_mean":
                float(
                    stats[
                        "entropy_channel_mean"
                    ]
                ),

            "entropy_r":
                float(
                    stats[
                        "entropy_r"
                    ]
                ),

            "entropy_g":
                float(
                    stats[
                        "entropy_g"
                    ]
                ),

            "entropy_b":
                float(
                    stats[
                        "entropy_b"
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

            **channel_metrics,

            "ciphertext_bit_hamming_percent":
                float(
                    bit_hd
                ),

            "exact_recovery":
                True,

            "encryption_mean_sec":
                enc_mean,

            "encryption_std_sec":
                sample_std(
                    encryption_times
                ),

            "decryption_mean_sec":
                dec_mean,

            "decryption_std_sec":
                sample_std(
                    decryption_times
                ),

            "logical_enc_throughput_mb_s":
                float(
                    logical_mb
                    / enc_mean
                ),

            "logical_dec_throughput_mb_s":
                float(
                    logical_mb
                    / dec_mean
                ),

            "liu_rgb_enc_throughput_mb_s":
                float(
                    liu_rgb_mb
                    / enc_mean
                ),

            "liu_rgb_dec_throughput_mb_s":
                float(
                    liu_rgb_mb
                    / dec_mean
                ),

            "liu_it_rounds":
                3,

            "liu_stream_mode":
                "fixed64_low32",

            "reproduction_claim":
                (
                    "independent controlled "
                    "reimplementation; "
                    "not author-exact"
                ),
        }

        results.append(row)

        print(
            "  Exact recovery : PASS"
        )

        print(
            f"  Entropy        : "
            f"{row['entropy_pooled']:.6f}"
        )

        print(
            f"  NPCR           : "
            f"{row['npcr_percent']:.6f}%"
        )

        print(
            f"  UACI           : "
            f"{row['uaci_percent']:.6f}%"
        )

        print(
            f"  Bit HD         : "
            f"{row['ciphertext_bit_hamming_percent']:.6f}%"
        )

        print(
            f"  Mean |r|       : "
            f"{row['mean_abs_correlation']:.6f}"
        )

        print(
            f"  Enc mean       : "
            f"{enc_mean:.6f} s"
        )

        print(
            f"  Dec mean       : "
            f"{dec_mean:.6f} s"
        )

    # ========================================================
    # WRITE RAW RESULTS
    # ========================================================

    results_path = (
        output_dir
        / "liu2026_100_results.csv"
    )

    timings_path = (
        output_dir
        / "liu2026_100_timings.csv"
    )

    write_csv(
        results,
        results_path,
    )

    write_csv(
        timings,
        timings_path,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    metric_names = [
        "entropy_pooled",
        "entropy_channel_mean",
        "corr_horizontal",
        "corr_vertical",
        "corr_diagonal",
        "mean_abs_correlation",
        "npcr_percent",
        "uaci_percent",
        "npcr_channel_mean_percent",
        "uaci_channel_mean_percent",
        "ciphertext_bit_hamming_percent",
        "encryption_mean_sec",
        "decryption_mean_sec",
        "logical_enc_throughput_mb_s",
        "logical_dec_throughput_mb_s",
    ]

    metric_summary = {}

    summary_rows = []

    for metric in metric_names:

        values = [
            float(
                row[
                    metric
                ]
            )
            for row in results
        ]

        summary = summarize(
            values
        )

        metric_summary[
            metric
        ] = summary

        summary_rows.append(
            {
                "metric":
                    metric,

                **summary,
            }
        )

    exact_count = sum(
        bool(
            row[
                "exact_recovery"
            ]
        )
        for row in results
    )

    total_logical_mb = (
        total_logical_bytes
        / 1_000_000.0
    )

    total_liu_rgb_mb = (
        total_liu_rgb_bytes
        / 1_000_000.0
    )

    total_enc_time = sum(
        row[
            "encryption_mean_sec"
        ]
        for row in results
    )

    total_dec_time = sum(
        row[
            "decryption_mean_sec"
        ]
        for row in results
    )

    run_summary = {
        "experiment_version":
            EXPERIMENT_VERSION,

        "algorithm":
            (
                "Liu et al. 2026 — "
                "3-D CML + Baker map"
            ),

        "implementation":
            (
                "Independent controlled "
                "reimplementation; "
                "not author-exact reproduction."
            ),

        "images":
            100,

        "exact_recovery_count":
            exact_count,

        "exact_recovery_rate_percent":
            float(
                exact_count
            ),

        "profile": {
            "stream_mode":
                "fixed64_low32",

            "it_rounds":
                3,

            "selection_reason":
                (
                    "More literal interpretation "
                    "of the paper's last-32-bit "
                    "extraction; IT=3 retained "
                    "after documented ambiguity "
                    "analysis over IT=1..4."
                ),
        },

        "adapter": {
            "grayscale_to_rgb":
                "R=G=B",

            "rectangular_to_square":
                (
                    "bottom/right edge padding "
                    "to smallest square "
                    "divisible by 4"
                ),
        },

        "payload": {
            "logical_grayscale_mb":
                total_logical_mb,

            "liu_processed_rgb_mb":
                total_liu_rgb_mb,

            "weighted_logical_enc_mb_s":
                float(
                    total_logical_mb
                    / total_enc_time
                ),

            "weighted_logical_dec_mb_s":
                float(
                    total_logical_mb
                    / total_dec_time
                ),

            "weighted_liu_rgb_enc_mb_s":
                float(
                    total_liu_rgb_mb
                    / total_enc_time
                ),

            "weighted_liu_rgb_dec_mb_s":
                float(
                    total_liu_rgb_mb
                    / total_dec_time
                ),
        },

        "metrics":
            metric_summary,
    }

    summary_csv = (
        output_dir
        / "liu2026_100_summary.csv"
    )

    summary_json = (
        output_dir
        / "liu2026_100_summary.json"
    )

    write_csv(
        summary_rows,
        summary_csv,
    )

    summary_json.write_text(
        json.dumps(
            run_summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()

    print(
        "=" * 104
    )

    print(
        "LIU 2026 — FINAL 100-IMAGE SUMMARY"
    )

    print(
        "=" * 104
    )

    print(
        f"Exact recovery : "
        f"{exact_count}/100"
    )

    print(
        "Mean entropy   : "
        f"{metric_summary['entropy_pooled']['mean']:.6f}"
        " ± "
        f"{metric_summary['entropy_pooled']['std']:.6f}"
    )

    print(
        "Mean NPCR      : "
        f"{metric_summary['npcr_percent']['mean']:.6f}%"
        " ± "
        f"{metric_summary['npcr_percent']['std']:.6f}"
    )

    print(
        "Mean UACI      : "
        f"{metric_summary['uaci_percent']['mean']:.6f}%"
        " ± "
        f"{metric_summary['uaci_percent']['std']:.6f}"
    )

    print(
        "Mean bit HD    : "
        f"{metric_summary['ciphertext_bit_hamming_percent']['mean']:.6f}%"
        " ± "
        f"{metric_summary['ciphertext_bit_hamming_percent']['std']:.6f}"
    )

    print(
        "Mean |r|       : "
        f"{metric_summary['mean_abs_correlation']['mean']:.6f}"
        " ± "
        f"{metric_summary['mean_abs_correlation']['std']:.6f}"
    )

    print(
        "Mean enc time  : "
        f"{metric_summary['encryption_mean_sec']['mean']:.6f} s"
    )

    print(
        "Mean dec time  : "
        f"{metric_summary['decryption_mean_sec']['mean']:.6f} s"
    )

    print()

    print(
        "Weighted logical encryption throughput:"
    )

    print(
        f"{run_summary['payload']['weighted_logical_enc_mb_s']:.6f} MB/s"
    )

    print(
        "Weighted logical decryption throughput:"
    )

    print(
        f"{run_summary['payload']['weighted_logical_dec_mb_s']:.6f} MB/s"
    )

    print()

    print(
        "Results :",
        results_path,
    )

    print(
        "Timings :",
        timings_path,
    )

    print(
        "Summary :",
        summary_csv,
    )

    print(
        "JSON    :",
        summary_json,
    )

    print(
        "=" * 104
    )

    if exact_count != 100:

        raise RuntimeError(
            "100-image exact recovery "
            "did not pass."
        )


if __name__ == "__main__":

    main()