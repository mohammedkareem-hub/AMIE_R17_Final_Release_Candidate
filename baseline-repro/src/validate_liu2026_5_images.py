from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

from dicom_loader import load_dicom_as_uint8

from liu2026_repro import (
    Liu2026,
    Liu2026Config,
    grayscale_to_rgb,
)


# ============================================================
# SETTINGS
# ============================================================

VALIDATION_IMAGES = [
    "image_001.dcm",
    "image_021.dcm",
    "image_050.dcm",
    "image_075.dcm",
    "image_100.dcm",
]

MEASURED_REPEATS = 3

EXPERIMENT_VERSION = (
    "Liu2026-ControlledValidation5-v1"
)


# ============================================================
# HELPERS
# ============================================================

def sample_std(
    values: list[float],
) -> float:

    if len(values) <= 1:
        return 0.0

    return float(
        statistics.stdev(
            values
        )
    )


def mean_value(
    values: list[float],
) -> float:

    return float(
        statistics.mean(
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

        for key in row.keys():

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


# ============================================================
# SQUARE ADAPTER
# ============================================================

def pad_to_square(
    image: np.ndarray,
) -> tuple[
    np.ndarray,
    dict,
]:
    """
    Liu's discrete Baker stage requires N x N.

    Our frozen DICOM subset contains both square and
    rectangular images.

    Controlled adapter:
        * retain original top-left image placement;
        * edge-pad bottom/right;
        * choose smallest square dimension >= max(H,W);
        * ensure side length is divisible by 4.

    The original H x W area is cropped after decryption.
    """

    image = np.asarray(
        image,
        dtype=np.uint8,
    )

    if image.ndim != 2:

        raise ValueError(
            "pad_to_square expects a 2-D grayscale image."
        )

    height = int(
        image.shape[0]
    )

    width = int(
        image.shape[1]
    )

    target = max(
        height,
        width,
    )

    # Liu default controlled Baker partition uses 4 strips.
    if target % 4 != 0:

        target += (
            4
            - target % 4
        )

    pad_bottom = (
        target
        - height
    )

    pad_right = (
        target
        - width
    )

    if (
        pad_bottom == 0
        and pad_right == 0
    ):

        padded = image.copy()

    else:

        padded = np.pad(
            image,
            (
                (
                    0,
                    pad_bottom,
                ),
                (
                    0,
                    pad_right,
                ),
            ),
            mode="edge",
        )

    metadata = {
        "original_height":
            height,

        "original_width":
            width,

        "square_size":
            target,

        "pad_bottom":
            pad_bottom,

        "pad_right":
            pad_right,

        "padding_mode":
            "edge",
    }

    return (
        padded,
        metadata,
    )


def crop_original(
    image: np.ndarray,
    adapter_meta: dict,
) -> np.ndarray:

    height = int(
        adapter_meta[
            "original_height"
        ]
    )

    width = int(
        adapter_meta[
            "original_width"
        ]
    )

    return image[
        :height,
        :width,
        ...
    ]


# ============================================================
# METRICS
# ============================================================

def entropy_uint8(
    array: np.ndarray,
) -> float:

    values = np.asarray(
        array,
        dtype=np.uint8,
    ).reshape(-1)

    histogram = np.bincount(
        values,
        minlength=256,
    ).astype(
        np.float64
    )

    probabilities = (
        histogram
        / histogram.sum()
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    return float(
        -np.sum(
            probabilities
            * np.log2(
                probabilities
            )
        )
    )


def correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:

    x = np.asarray(
        first,
        dtype=np.float64,
    ).reshape(-1)

    y = np.asarray(
        second,
        dtype=np.float64,
    ).reshape(-1)

    if x.size != y.size:

        raise ValueError(
            "Correlation vectors must have equal length."
        )

    if x.size < 2:

        return 0.0

    x_std = float(
        np.std(
            x
        )
    )

    y_std = float(
        np.std(
            y
        )
    )

    if (
        x_std == 0.0
        or y_std == 0.0
    ):

        return 0.0

    return float(
        np.corrcoef(
            x,
            y,
        )[
            0,
            1
        ]
    )


def channel_correlations(
    channel: np.ndarray,
) -> tuple[
    float,
    float,
    float,
]:

    channel = np.asarray(
        channel,
        dtype=np.uint8,
    )

    horizontal = correlation(
        channel[
            :,
            :-1
        ],
        channel[
            :,
            1:
        ],
    )

    vertical = correlation(
        channel[
            :-1,
            :
        ],
        channel[
            1:,
            :
        ],
    )

    diagonal = correlation(
        channel[
            :-1,
            :-1
        ],
        channel[
            1:,
            1:
        ],
    )

    return (
        horizontal,
        vertical,
        diagonal,
    )


def rgb_cipher_statistics(
    cipher: np.ndarray,
) -> dict[str, float]:

    cipher = np.asarray(
        cipher,
        dtype=np.uint8,
    )

    if (
        cipher.ndim != 3
        or cipher.shape[2] != 3
    ):

        raise ValueError(
            "Expected RGB ciphertext."
        )

    pooled_entropy = entropy_uint8(
        cipher
    )

    channel_entropies = []

    h_values = []
    v_values = []
    d_values = []

    for channel_index in range(
        3
    ):

        channel = cipher[
            :,
            :,
            channel_index
        ]

        channel_entropies.append(
            entropy_uint8(
                channel
            )
        )

        (
            horizontal,
            vertical,
            diagonal,
        ) = channel_correlations(
            channel
        )

        h_values.append(
            horizontal
        )

        v_values.append(
            vertical
        )

        d_values.append(
            diagonal
        )

    mean_h = mean_value(
        h_values
    )

    mean_v = mean_value(
        v_values
    )

    mean_d = mean_value(
        d_values
    )

    mean_abs = float(
        (
            abs(
                mean_h
            )
            +
            abs(
                mean_v
            )
            +
            abs(
                mean_d
            )
        )
        / 3.0
    )

    return {
        "entropy_pooled":
            pooled_entropy,

        "entropy_channel_mean":
            mean_value(
                channel_entropies
            ),

        "entropy_r":
            channel_entropies[
                0
            ],

        "entropy_g":
            channel_entropies[
                1
            ],

        "entropy_b":
            channel_entropies[
                2
            ],

        "corr_horizontal_mean":
            mean_h,

        "corr_vertical_mean":
            mean_v,

        "corr_diagonal_mean":
            mean_d,

        "mean_abs_correlation":
            mean_abs,
    }


def calculate_npcr_uaci_rgb(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[
    float,
    float,
]:

    first = np.asarray(
        first,
        dtype=np.uint8,
    )

    second = np.asarray(
        second,
        dtype=np.uint8,
    )

    if first.shape != second.shape:

        raise ValueError(
            "NPCR/UACI arrays must have equal shapes."
        )

    different = (
        first
        != second
    )

    npcr = float(
        np.mean(
            different
        )
        * 100.0
    )

    difference = np.abs(
        first.astype(
            np.int16
        )
        -
        second.astype(
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

    return (
        npcr,
        uaci,
    )


def bit_hamming_percent(
    first: np.ndarray,
    second: np.ndarray,
) -> float:

    first = np.asarray(
        first,
        dtype=np.uint8,
    )

    second = np.asarray(
        second,
        dtype=np.uint8,
    )

    if first.shape != second.shape:

        raise ValueError(
            "Bit Hamming arrays must have equal shapes."
        )

    xor_bytes = np.bitwise_xor(
        first,
        second,
    )

    bits = np.unpackbits(
        xor_bytes.reshape(-1)
    )

    return float(
        np.mean(
            bits
        )
        * 100.0
    )


# ============================================================
# DIFFERENTIAL PLAINTEXT
# ============================================================

def create_modified_plaintext(
    image: np.ndarray,
) -> np.ndarray:
    """
    Same differential convention used in AMIE tests:
    modify plaintext pixel (0,0) by +1 modulo 256.
    """

    modified = np.asarray(
        image,
        dtype=np.uint8,
    ).copy()

    modified[
        0,
        0
    ] = np.uint8(
        (
            int(
                modified[
                    0,
                    0
                ]
            )
            + 1
        )
        % 256
    )

    return modified


# ============================================================
# VALIDATE REQUIRED FILES
# ============================================================

def get_validation_paths(
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
                f"Missing validation image:\n{path}"
            )

        paths.append(
            path
        )

    return paths


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Five-image validation of the independent "
            "controlled Liu et al. 2026 implementation."
        )
    )

    parser.add_argument(
        "--selected-dir",
        default=r"data\selected",
        help=(
            "Folder containing image_001.dcm ... "
            "image_100.dcm. "
            r"Default: data\selected"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=r"output\liu2026_validation_5",
        help=(
            "Output directory. "
            r"Default: output\liu2026_validation_5"
        ),
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

    image_paths = get_validation_paths(
        selected_dir
    )

    # ========================================================
    # CONTROLLED LIU CONFIG
    # ========================================================

    config = Liu2026Config(
        epsilon=0.1,
        plm_segments=64,
        plm_mu=4.0,
        warmup=1000,
        it_rounds=1,
        stream_mode="fixed64_low32",
        baker_partition=None,
    )

    algorithm = Liu2026(
        config
    )

    print(
        "=" * 96
    )

    print(
        "LIU ET AL. 2026 — FIVE-IMAGE CONTROLLED VALIDATION"
    )

    print(
        "=" * 96
    )

    print(
        "Dataset:",
        selected_dir,
    )

    print(
        "Output :",
        output_dir,
    )

    print(
        "Images :",
        ", ".join(
            path.name
            for path
            in image_paths
        ),
    )

    print(
        "Repeats:",
        MEASURED_REPEATS,
    )

    print()

    print(
        "Implementation boundary:"
    )

    print(
        "Independent controlled reimplementation;"
    )

    print(
        "not author-exact reproduction."
    )

    print()

    # ========================================================
    # GLOBAL WARM-UP — EXCLUDED
    # ========================================================

    print(
        "Running excluded warm-up..."
    )

    warm_gray = load_dicom_as_uint8(
        image_paths[
            0
        ]
    )

    warm_square, warm_adapter = pad_to_square(
        warm_gray
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
            "Excluded warm-up exact recovery failed."
        )

    print(
        "Warm-up exact recovery: PASS"
    )

    # ========================================================
    # RESULTS
    # ========================================================

    result_rows = []

    timing_rows = []

    for image_number, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print()

        print(
            "-" * 96
        )

        print(
            f"[{image_number}/"
            f"{len(image_paths)}] "
            f"{image_path.name}"
        )

        # ----------------------------------------------------
        # Same DICOM preprocessing as other project baselines.
        # ----------------------------------------------------

        original_gray = load_dicom_as_uint8(
            image_path
        )

        if (
            original_gray.dtype
            != np.uint8
        ):

            raise RuntimeError(
                f"{image_path.name}: "
                "DICOM loader did not return uint8."
            )

        if original_gray.ndim != 2:

            raise RuntimeError(
                f"{image_path.name}: "
                "expected grayscale image."
            )

        # ----------------------------------------------------
        # Square adapter for Liu Baker map.
        # ----------------------------------------------------

        square_gray, adapter_meta = pad_to_square(
            original_gray
        )

        square_rgb = grayscale_to_rgb(
            square_gray
        )

        modified_gray = create_modified_plaintext(
            original_gray
        )

        modified_square, modified_meta = pad_to_square(
            modified_gray
        )

        if (
            modified_meta[
                "square_size"
            ]
            != adapter_meta[
                "square_size"
            ]
        ):

            raise RuntimeError(
                "Base and differential square sizes differ."
            )

        modified_rgb = grayscale_to_rgb(
            modified_square
        )

        # ----------------------------------------------------
        # Timing repetitions
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
                square_rgb
            )

            (
                recovered_rgb,
                decryption_time,
            ) = algorithm.decrypt_rgb(
                cipher,
                metadata,
            )

            if not np.array_equal(
                square_rgb,
                recovered_rgb,
            ):

                raise RuntimeError(
                    f"{image_path.name}: "
                    f"full padded exact recovery failed "
                    f"at repeat {repeat_index}."
                )

            recovered_crop = crop_original(
                recovered_rgb,
                adapter_meta,
            )

            expected_crop = grayscale_to_rgb(
                original_gray
            )

            if not np.array_equal(
                expected_crop,
                recovered_crop,
            ):

                raise RuntimeError(
                    f"{image_path.name}: "
                    "cropped original-area recovery failed."
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

            timing_rows.append(
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

            timing_rows.append(
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

                base_cipher = cipher.copy()

                base_metadata = metadata

        # ----------------------------------------------------
        # Ciphertext statistics
        # ----------------------------------------------------

        stats = rgb_cipher_statistics(
            base_cipher
        )

        # ----------------------------------------------------
        # Differential test
        # One grayscale pixel +1 mod 256;
        # then duplicate grayscale into R,G,B.
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

        bit_hd = bit_hamming_percent(
            base_cipher,
            modified_cipher,
        )

        # ----------------------------------------------------
        # Final exact recovery
        # ----------------------------------------------------

        (
            final_recovered,
            _,
        ) = algorithm.decrypt_rgb(
            base_cipher,
            base_metadata,
        )

        full_exact = bool(
            np.array_equal(
                square_rgb,
                final_recovered,
            )
        )

        cropped = crop_original(
            final_recovered,
            adapter_meta,
        )

        original_rgb = grayscale_to_rgb(
            original_gray
        )

        crop_exact = bool(
            np.array_equal(
                original_rgb,
                cropped,
            )
        )

        if not (
            full_exact
            and crop_exact
        ):

            raise RuntimeError(
                f"{image_path.name}: "
                "final recovery validation failed."
            )

        # ----------------------------------------------------
        # Timing summary
        # ----------------------------------------------------

        enc_mean = mean_value(
            encryption_times
        )

        dec_mean = mean_value(
            decryption_times
        )

        enc_std = sample_std(
            encryption_times
        )

        dec_std = sample_std(
            decryption_times
        )

        logical_bytes = int(
            original_gray.size
        )

        logical_mb = (
            logical_bytes
            / 1_000_000.0
        )

        enc_throughput = float(
            logical_mb
            / enc_mean
        )

        dec_throughput = float(
            logical_mb
            / dec_mean
        )

        row = {
            "experiment_version":
                EXPERIMENT_VERSION,

            "image_name":
                image_path.name,

            "original_height":
                int(
                    original_gray.shape[
                        0
                    ]
                ),

            "original_width":
                int(
                    original_gray.shape[
                        1
                    ]
                ),

            "liu_square_size":
                int(
                    adapter_meta[
                        "square_size"
                    ]
                ),

            "pad_bottom":
                int(
                    adapter_meta[
                        "pad_bottom"
                    ]
                ),

            "pad_right":
                int(
                    adapter_meta[
                        "pad_right"
                    ]
                ),

            "padding_mode":
                "edge",

            "logical_processed_bytes":
                logical_bytes,

            "entropy_pooled":
                stats[
                    "entropy_pooled"
                ],

            "entropy_channel_mean":
                stats[
                    "entropy_channel_mean"
                ],

            "entropy_r":
                stats[
                    "entropy_r"
                ],

            "entropy_g":
                stats[
                    "entropy_g"
                ],

            "entropy_b":
                stats[
                    "entropy_b"
                ],

            "corr_horizontal_mean":
                stats[
                    "corr_horizontal_mean"
                ],

            "corr_vertical_mean":
                stats[
                    "corr_vertical_mean"
                ],

            "corr_diagonal_mean":
                stats[
                    "corr_diagonal_mean"
                ],

            "mean_abs_correlation":
                stats[
                    "mean_abs_correlation"
                ],

            "npcr_percent":
                float(
                    npcr
                ),

            "uaci_percent":
                float(
                    uaci
                ),

            "ciphertext_bit_hamming_percent":
                float(
                    bit_hd
                ),

            "full_padded_exact_recovery":
                full_exact,

            "original_area_exact_recovery":
                crop_exact,

            "encryption_mean_sec":
                enc_mean,

            "encryption_std_sec":
                enc_std,

            "decryption_mean_sec":
                dec_mean,

            "decryption_std_sec":
                dec_std,

            "encryption_throughput_mb_s":
                enc_throughput,

            "decryption_throughput_mb_s":
                dec_throughput,

            "liu_epsilon":
                config.epsilon,

            "liu_plm_segments":
                config.plm_segments,

            "liu_plm_mu":
                config.plm_mu,

            "liu_warmup":
                config.warmup,

            "liu_it_rounds":
                config.it_rounds,

            "liu_stream_mode":
                config.stream_mode,

            "liu_baker_partition":
                str(
                    base_metadata[
                        "partition"
                    ]
                ),

            "reproduction_claim":
                (
                    "independent controlled "
                    "reimplementation; "
                    "not author-exact"
                ),
        }

        result_rows.append(
            row
        )

        print(
            "  Original shape   : "
            f"{original_gray.shape[0]}"
            " x "
            f"{original_gray.shape[1]}"
        )

        print(
            "  Liu square input : "
            f"{adapter_meta['square_size']}"
            " x "
            f"{adapter_meta['square_size']}"
        )

        print(
            "  Exact recovery   : PASS"
        )

        print(
            "  Entropy pooled   : "
            f"{row['entropy_pooled']:.6f}"
        )

        print(
            "  NPCR             : "
            f"{row['npcr_percent']:.6f}%"
        )

        print(
            "  UACI             : "
            f"{row['uaci_percent']:.6f}%"
        )

        print(
            "  Bit HD           : "
            f"{row['ciphertext_bit_hamming_percent']:.6f}%"
        )

        print(
            "  Mean |r|         : "
            f"{row['mean_abs_correlation']:.6f}"
        )

        print(
            "  Enc mean         : "
            f"{enc_mean:.6f} s"
        )

        print(
            "  Dec mean         : "
            f"{dec_mean:.6f} s"
        )

    # ========================================================
    # WRITE CSV FILES
    # ========================================================

    results_path = (
        output_dir
        / "liu2026_validation_5_results.csv"
    )

    timings_path = (
        output_dir
        / "liu2026_validation_5_timings.csv"
    )

    write_csv(
        result_rows,
        results_path,
    )

    write_csv(
        timing_rows,
        timings_path,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_metrics = [
        "entropy_pooled",
        "entropy_channel_mean",
        "corr_horizontal_mean",
        "corr_vertical_mean",
        "corr_diagonal_mean",
        "mean_abs_correlation",
        "npcr_percent",
        "uaci_percent",
        "ciphertext_bit_hamming_percent",
        "encryption_mean_sec",
        "decryption_mean_sec",
        "encryption_throughput_mb_s",
        "decryption_throughput_mb_s",
    ]

    metrics_summary = {}

    for metric in summary_metrics:

        values = [
            float(
                row[
                    metric
                ]
            )
            for row
            in result_rows
        ]

        metrics_summary[
            metric
        ] = {
            "mean":
                mean_value(
                    values
                ),

            "std":
                sample_std(
                    values
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
        }

    exact_count = sum(
        bool(
            row[
                "original_area_exact_recovery"
            ]
        )
        for row
        in result_rows
    )

    summary = {
        "experiment_version":
            EXPERIMENT_VERSION,

        "algorithm":
            (
                "Liu et al. 2026 — "
                "3-D CML + Baker map"
            ),

        "implementation":
            (
                "Independent controlled reimplementation; "
                "not author-exact reproduction."
            ),

        "validation_images":
            VALIDATION_IMAGES,

        "images_tested":
            len(
                result_rows
            ),

        "exact_recovery_count":
            exact_count,

        "exact_recovery_rate_percent":
            float(
                exact_count
                / len(
                    result_rows
                )
                * 100.0
            ),

        "measured_repeats_per_image":
            MEASURED_REPEATS,

        "adapter": {
            "grayscale_to_rgb":
                "R=G=B",

            "rectangular_to_square":
                (
                    "bottom/right edge padding "
                    "to smallest square divisible by 4"
                ),

            "metrics_cipher_domain":
                (
                    "full RGB ciphertext after "
                    "controlled Liu square adaptation"
                ),

            "differential_plaintext":
                (
                    "original grayscale pixel (0,0) "
                    "+1 mod 256 before RGB duplication"
                ),
        },

        "parameters": {
            "epsilon":
                config.epsilon,

            "plm_segments":
                config.plm_segments,

            "plm_mu":
                config.plm_mu,

            "warmup":
                config.warmup,

            "it_rounds":
                config.it_rounds,

            "stream_mode":
                config.stream_mode,

            "baker_partition":
                "equal-quarter per square image",
        },

        "metrics":
            metrics_summary,
    }

    summary_path = (
        output_dir
        / "liu2026_validation_5_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL CONSOLE SUMMARY
    # ========================================================

    print()
    print(
        "=" * 96
    )

    print(
        "LIU 2026 — FINAL FIVE-IMAGE VALIDATION SUMMARY"
    )

    print(
        "=" * 96
    )

    print(
        "Exact recovery : "
        f"{exact_count}/"
        f"{len(result_rows)}"
    )

    print(
        "Mean entropy   : "
        f"{metrics_summary['entropy_pooled']['mean']:.6f}"
        " ± "
        f"{metrics_summary['entropy_pooled']['std']:.6f}"
    )

    print(
        "Mean NPCR      : "
        f"{metrics_summary['npcr_percent']['mean']:.6f}%"
        " ± "
        f"{metrics_summary['npcr_percent']['std']:.6f}"
    )

    print(
        "Mean UACI      : "
        f"{metrics_summary['uaci_percent']['mean']:.6f}%"
        " ± "
        f"{metrics_summary['uaci_percent']['std']:.6f}"
    )

    print(
        "Mean bit HD    : "
        f"{metrics_summary['ciphertext_bit_hamming_percent']['mean']:.6f}%"
        " ± "
        f"{metrics_summary['ciphertext_bit_hamming_percent']['std']:.6f}"
    )

    print(
        "Mean |r|       : "
        f"{metrics_summary['mean_abs_correlation']['mean']:.6f}"
        " ± "
        f"{metrics_summary['mean_abs_correlation']['std']:.6f}"
    )

    print(
        "Mean enc time  : "
        f"{metrics_summary['encryption_mean_sec']['mean']:.6f} s"
    )

    print(
        "Mean dec time  : "
        f"{metrics_summary['decryption_mean_sec']['mean']:.6f} s"
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
        summary_path,
    )

    print(
        "=" * 96
    )

    if exact_count != len(
        result_rows
    ):

        raise RuntimeError(
            "Five-image validation did not achieve "
            "100% exact recovery."
        )


if __name__ == "__main__":

    main()