from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from dicom_loader import load_dicom_as_uint8

from kamal2021_repro import (
    Kamal2021,
    zigzag_inverse,
    zigzag_permute,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

SRC_DIR = (
    Path(__file__).resolve().parent
)

SELECTED_DIR = (
    PROJECT_ROOT
    / "data"
    / "selected"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "kamal_baseline_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# VALIDATION DATA
# ============================================================

SAMPLE_IMAGES = [
    "image_001.dcm",
    "image_021.dcm",
    "image_050.dcm",
    "image_075.dcm",
    "image_100.dcm",
]


# ============================================================
# SOURCE-PAPER SETTINGS
# ============================================================

# Kamal et al. explicitly report the following experimental
# settings:
#
#   block size = 16
#   logistic-map parameter a = 3.9
#   transient length N0 = 1000
#
# The paper supports block sizes 16, 32 and 64.
#
# The random vectors Ri and r are described in the paper,
# but the PRNG algorithm and its seed are not specified.
# Therefore rng_seed=2021 is OUR reproducibility choice.

AUTHOR_EXPERIMENT_BLOCK_SIZE = 16

AUTHOR_EXPERIMENT_A = 3.9

AUTHOR_EXPERIMENT_N0 = 1000

REPRODUCIBILITY_SEED = 2021


# ============================================================
# CONTROLLED SENSITIVITY SETTINGS
# ============================================================

# Source-supported block sizes.
BLOCK_SIZES = [
    16,
    32,
    64,
]

# All values are inside the chaotic region discussed
# by the source paper.
A_VALUES = [
    3.70,
    3.90,
    3.99,
]

# Controlled transient-length sensitivity around N0=1000.
N0_VALUES = [
    100,
    1000,
    2000,
]

# Since the source paper does not define the PRNG seed,
# we test several deterministic seeds to verify that our
# comparison does not depend critically on seed=2021.
RNG_SEEDS = [
    7,
    42,
    2021,
    2026,
    99991,
]


# ============================================================
# BASIC HELPERS
# ============================================================

def file_sha256(
    path: Path,
) -> str:

    hasher = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            hasher.update(
                chunk
            )

    return hasher.hexdigest()


def entropy_uint8(
    image: np.ndarray,
) -> float:

    image = np.asarray(
        image,
        dtype=np.uint8,
    )

    histogram = np.bincount(
        image.reshape(-1),
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

    x = x - x.mean()
    y = y - y.mean()

    denominator = float(
        np.sqrt(
            np.dot(x, x)
            *
            np.dot(y, y)
        )
    )

    if denominator == 0.0:
        return 0.0

    return float(
        np.dot(x, y)
        / denominator
    )


def image_correlations(
    image: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
]:

    image = np.asarray(
        image,
        dtype=np.uint8,
    )

    horizontal = correlation(
        image[:, :-1],
        image[:, 1:],
    )

    vertical = correlation(
        image[:-1, :],
        image[1:, :],
    )

    diagonal = correlation(
        image[:-1, :-1],
        image[1:, 1:],
    )

    mean_abs = float(
        np.mean(
            np.abs(
                [
                    horizontal,
                    vertical,
                    diagonal,
                ]
            )
        )
    )

    return (
        horizontal,
        vertical,
        diagonal,
        mean_abs,
    )


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:

    if not rows:
        return

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
# INDEPENDENT EQUATION IMPLEMENTATIONS
# ============================================================

def independent_y0(
    image: np.ndarray,
) -> float:
    """
    Independent implementation of Kamal Eq. (2):

        Y0 = sum(P) / (M*N*255)

    This is deliberately separate from Kamal2021._calculate_y0
    so we can validate the reimplementation itself.
    """

    image = np.asarray(
        image,
        dtype=np.uint8,
    )

    rows, cols = image.shape

    pixel_sum = np.sum(
        image,
        dtype=np.float64,
    )

    return float(
        pixel_sum
        /
        (
            rows
            * cols
            * 255.0
        )
    )


def independent_logistic_key(
    *,
    image_shape: tuple[int, int],
    y0: float,
    a: float,
    n0: int,
) -> np.ndarray:
    """
    Independent implementation of:

        Y[n+1] = a * Y[n] * (1 - Y[n])

    followed by:

        K(i) = mod(floor(S(i) * 10^14), 256)

    This routine exists only for implementation validation.
    """

    rows, cols = image_shape

    number_pixels = (
        rows
        * cols
    )

    total_length = (
        n0
        + number_pixels
    )

    sequence = np.empty(
        total_length,
        dtype=np.float64,
    )

    sequence[0] = (
        y0
    )

    for index in range(
        1,
        total_length,
    ):

        previous = (
            sequence[
                index - 1
            ]
        )

        sequence[index] = (
            a
            * previous
            * (
                1.0
                - previous
            )
        )

    retained = sequence[
        n0:
        n0 + number_pixels
    ]

    key = np.mod(
        np.floor(
            retained
            * 1e14
        ),
        256,
    ).astype(
        np.uint8
    )

    return key


# ============================================================
# SOURCE-CONFORMANCE TESTS
# ============================================================

def source_conformance_checks() -> dict:

    checks = {}

    # --------------------------------------------------------
    # 1. Zigzag inverse
    # --------------------------------------------------------

    test_block = np.arange(
        64,
        dtype=np.uint8,
    ).reshape(
        8,
        8,
    )

    zigzagged = (
        zigzag_permute(
            test_block
        )
    )

    recovered = (
        zigzag_inverse(
            zigzagged
        )
    )

    checks[
        "zigzag_inverse_exact"
    ] = bool(
        np.array_equal(
            test_block,
            recovered,
        )
    )

    # --------------------------------------------------------
    # 2. Eq. (2) Y0
    # --------------------------------------------------------

    simple_image = np.asarray(
        [
            [0, 255],
            [128, 64],
        ],
        dtype=np.uint8,
    )

    algorithm = Kamal2021(
        block_size=
            AUTHOR_EXPERIMENT_BLOCK_SIZE,

        a=
            AUTHOR_EXPERIMENT_A,

        n0=
            AUTHOR_EXPERIMENT_N0,

        rng_seed=
            REPRODUCIBILITY_SEED,
    )

    implementation_y0 = (
        algorithm._calculate_y0(
            simple_image
        )
    )

    reference_y0 = (
        independent_y0(
            simple_image
        )
    )

    checks[
        "eq2_y0"
    ] = {
        "implementation":
            float(
                implementation_y0
            ),

        "independent_reference":
            float(
                reference_y0
            ),

        "matches":
            bool(
                implementation_y0
                == reference_y0
            ),
    }

    # --------------------------------------------------------
    # 3. Eq. (1) + Eq. (3)
    # --------------------------------------------------------

    key_shape = (
        16,
        16,
    )

    implementation_key = (
        algorithm._generate_key(
            key_shape,
            y0=reference_y0,
        )
    )

    reference_key = (
        independent_logistic_key(
            image_shape=
                key_shape,

            y0=
                reference_y0,

            a=
                AUTHOR_EXPERIMENT_A,

            n0=
                AUTHOR_EXPERIMENT_N0,
        )
    )

    checks[
        "eq1_eq3_logistic_key"
    ] = {
        "matches":
            bool(
                np.array_equal(
                    implementation_key,
                    reference_key,
                )
            ),

        "implementation_sha256":
            hashlib.sha256(
                implementation_key.tobytes(
                    order="C"
                )
            ).hexdigest(),

        "independent_reference_sha256":
            hashlib.sha256(
                reference_key.tobytes(
                    order="C"
                )
            ).hexdigest(),

        "first_16_key_bytes":
            implementation_key[
                :16
            ].astype(
                int
            ).tolist(),
    }

    # --------------------------------------------------------
    # 4. Parameter provenance
    # --------------------------------------------------------

    checks[
        "source_parameter_mapping"
    ] = {
        "block_size": {
            "value":
                AUTHOR_EXPERIMENT_BLOCK_SIZE,

            "status":
                (
                    "author-specified "
                    "experimental setting"
                ),
        },

        "a": {
            "value":
                AUTHOR_EXPERIMENT_A,

            "status":
                (
                    "author-specified "
                    "experimental setting"
                ),
        },

        "N0": {
            "value":
                AUTHOR_EXPERIMENT_N0,

            "status":
                (
                    "author-specified "
                    "experimental setting"
                ),
        },

        "rng_seed": {
            "value":
                REPRODUCIBILITY_SEED,

            "status":
                (
                    "reimplementation-only "
                    "reproducibility choice; "
                    "source PRNG/seed unspecified"
                ),
        },

        "source_supported_block_sizes":
            [
                16,
                32,
                64,
            ],
    }

    # --------------------------------------------------------
    # 5. Reproduction boundary
    # --------------------------------------------------------

    checks[
        "source_result_reproduction_boundary"
    ] = {
        "exact_source_ciphertext_reproduction_claimed":
            False,

        "reason":
            (
                "The publication specifies the encryption "
                "stages and experimental values block_size=16, "
                "a=3.9 and N0=1000, but does not specify the "
                "PRNG algorithm or seed used to generate the "
                "random Ri/r vectors. Therefore exact source "
                "ciphertexts cannot be uniquely regenerated "
                "from the publication alone."
            ),

        "source_reported_entropy_examples": {
            "Img1":
                7.9993,

            "Img2":
                7.9994,

            "Img3":
                7.9974,

            "Img4":
                7.9972,

            "Img5":
                7.9977,
        },

        "source_reported_img1_chi_square":
            262.9102,
    }

    return checks


# ============================================================
# DATASET
# ============================================================

def resolve_sample_images() -> list[
    Path
]:

    paths = [
        SELECTED_DIR
        / name
        for name
        in SAMPLE_IMAGES
    ]

    missing = [
        path
        for path
        in paths
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Missing validation images:\n"
            +
            "\n".join(
                str(path)
                for path
                in missing
            )
        )

    return paths


# ============================================================
# ONE CONFIGURATION
# ============================================================

def evaluate_configuration(
    *,
    label: str,
    factor: str,
    block_size: int,
    a: float,
    n0: int,
    rng_seed: int,
    image_paths: list[Path],
) -> tuple[
    list[dict],
    dict,
]:

    algorithm = Kamal2021(
        block_size=
            block_size,

        a=
            a,

        n0=
            n0,

        rng_seed=
            rng_seed,
    )

    rows = []

    for path in image_paths:

        image = (
            load_dicom_as_uint8(
                path
            )
        )

        (
            ciphertext,
            metadata,
            encryption_time,
        ) = algorithm.encrypt_array(
            image
        )

        (
            recovered,
            decryption_time,
        ) = algorithm.decrypt_array(
            ciphertext,
            metadata,
        )

        exact_recovery = bool(
            np.array_equal(
                image,
                recovered,
            )
        )

        mismatched_pixels = int(
            np.count_nonzero(
                image
                != recovered
            )
        )

        (
            horizontal,
            vertical,
            diagonal,
            mean_abs,
        ) = image_correlations(
            ciphertext
        )

        ciphertext_hash = (
            hashlib.sha256(
                ciphertext.tobytes(
                    order="C"
                )
            ).hexdigest()
        )

        rows.append(
            {
                "configuration":
                    label,

                "factor":
                    factor,

                "image":
                    path.name,

                "block_size":
                    block_size,

                "a":
                    a,

                "n0":
                    n0,

                "rng_seed":
                    rng_seed,

                "exact_recovery":
                    exact_recovery,

                "mismatched_pixels":
                    mismatched_pixels,

                "entropy":
                    entropy_uint8(
                        ciphertext
                    ),

                "horizontal_correlation":
                    horizontal,

                "vertical_correlation":
                    vertical,

                "diagonal_correlation":
                    diagonal,

                "mean_abs_correlation":
                    mean_abs,

                "encryption_time_sec":
                    float(
                        encryption_time
                    ),

                "decryption_time_sec":
                    float(
                        decryption_time
                    ),

                "ciphertext_sha256":
                    ciphertext_hash,
            }
        )

    entropy_values = [
        float(
            row[
                "entropy"
            ]
        )
        for row
        in rows
    ]

    correlation_values = [
        float(
            row[
                "mean_abs_correlation"
            ]
        )
        for row
        in rows
    ]

    encryption_times = [
        float(
            row[
                "encryption_time_sec"
            ]
        )
        for row
        in rows
    ]

    decryption_times = [
        float(
            row[
                "decryption_time_sec"
            ]
        )
        for row
        in rows
    ]

    summary = {
        "configuration":
            label,

        "factor":
            factor,

        "block_size":
            block_size,

        "a":
            a,

        "n0":
            n0,

        "rng_seed":
            rng_seed,

        "images":
            len(
                rows
            ),

        "exact_recovery_count":
            int(
                sum(
                    bool(
                        row[
                            "exact_recovery"
                        ]
                    )
                    for row
                    in rows
                )
            ),

        "all_exact_recovery":
            bool(
                all(
                    bool(
                        row[
                            "exact_recovery"
                        ]
                    )
                    for row
                    in rows
                )
            ),

        "entropy_mean":
            float(
                statistics.mean(
                    entropy_values
                )
            ),

        "entropy_std":
            float(
                statistics.stdev(
                    entropy_values
                )
                if len(
                    entropy_values
                ) > 1
                else 0.0
            ),

        "mean_abs_correlation_mean":
            float(
                statistics.mean(
                    correlation_values
                )
            ),

        "mean_abs_correlation_std":
            float(
                statistics.stdev(
                    correlation_values
                )
                if len(
                    correlation_values
                ) > 1
                else 0.0
            ),

        "encryption_time_mean_sec":
            float(
                statistics.mean(
                    encryption_times
                )
            ),

        "decryption_time_mean_sec":
            float(
                statistics.mean(
                    decryption_times
                )
            ),

        "unique_ciphertexts":
            len(
                {
                    row[
                        "ciphertext_sha256"
                    ]
                    for row
                    in rows
                }
            ),
    }

    return (
        rows,
        summary,
    )


# ============================================================
# ONE-FACTOR-AT-A-TIME PROFILES
# ============================================================

def build_profiles() -> list[
    dict
]:
    """
    Build a controlled one-factor-at-a-time sensitivity study.

    We avoid a full factorial grid because the goal is to
    determine whether one specific implementation choice
    materially changes the baseline results.
    """

    profiles = []

    seen = set()

    def add_profile(
        *,
        label: str,
        factor: str,
        block_size: int,
        a: float,
        n0: int,
        rng_seed: int,
    ) -> None:

        key = (
            int(
                block_size
            ),
            float(
                a
            ),
            int(
                n0
            ),
            int(
                rng_seed
            ),
        )

        if key in seen:
            return

        seen.add(
            key
        )

        profiles.append(
            {
                "label":
                    label,

                "factor":
                    factor,

                "block_size":
                    int(
                        block_size
                    ),

                "a":
                    float(
                        a
                    ),

                "n0":
                    int(
                        n0
                    ),

                "rng_seed":
                    int(
                        rng_seed
                    ),
            }
        )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    add_profile(
        label=
            "controlled_baseline",

        factor=
            "baseline",

        block_size=
            AUTHOR_EXPERIMENT_BLOCK_SIZE,

        a=
            AUTHOR_EXPERIMENT_A,

        n0=
            AUTHOR_EXPERIMENT_N0,

        rng_seed=
            REPRODUCIBILITY_SEED,
    )

    # --------------------------------------------------------
    # Block-size sensitivity
    # --------------------------------------------------------

    for block_size in BLOCK_SIZES:

        add_profile(
            label=
                f"block_size_{block_size}",

            factor=
                "block_size",

            block_size=
                block_size,

            a=
                AUTHOR_EXPERIMENT_A,

            n0=
                AUTHOR_EXPERIMENT_N0,

            rng_seed=
                REPRODUCIBILITY_SEED,
        )

    # --------------------------------------------------------
    # Logistic a sensitivity
    # --------------------------------------------------------

    for a in A_VALUES:

        add_profile(
            label=
                f"a_{a:.2f}",

            factor=
                "a",

            block_size=
                AUTHOR_EXPERIMENT_BLOCK_SIZE,

            a=
                a,

            n0=
                AUTHOR_EXPERIMENT_N0,

            rng_seed=
                REPRODUCIBILITY_SEED,
        )

    # --------------------------------------------------------
    # N0 sensitivity
    # --------------------------------------------------------

    for n0 in N0_VALUES:

        add_profile(
            label=
                f"n0_{n0}",

            factor=
                "n0",

            block_size=
                AUTHOR_EXPERIMENT_BLOCK_SIZE,

            a=
                AUTHOR_EXPERIMENT_A,

            n0=
                n0,

            rng_seed=
                REPRODUCIBILITY_SEED,
        )

    # --------------------------------------------------------
    # RNG-seed sensitivity
    # --------------------------------------------------------

    for seed in RNG_SEEDS:

        add_profile(
            label=
                f"rng_seed_{seed}",

            factor=
                "rng_seed",

            block_size=
                AUTHOR_EXPERIMENT_BLOCK_SIZE,

            a=
                AUTHOR_EXPERIMENT_A,

            n0=
                AUTHOR_EXPERIMENT_N0,

            rng_seed=
                seed,
        )

    return profiles


# ============================================================
# EXISTING 100-IMAGE RESULTS
# ============================================================

def find_existing_100_image_summary():
    """
    Search common project locations for the historical
    100-image Kamal benchmark summary.
    """

    candidates = [
        (
            PROJECT_ROOT
            / "kamal_100_output"
            / "kamal_100_summary.json"
        ),

        (
            PROJECT_ROOT
            / "output"
            / "kamal_100_output"
            / "kamal_100_summary.json"
        ),

        (
            PROJECT_ROOT
            / "output"
            / "Kamal_100_output"
            / "kamal_100_summary.json"
        ),

        (
            PROJECT_ROOT
            / "Kamal_2021"
            / "kamal_100_output"
            / "kamal_100_summary.json"
        ),
    ]

    for path in candidates:

        if path.exists():

            try:

                content = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )

                return {
                    "path":
                        str(
                            path
                        ),

                    "content":
                        content,
                }

            except Exception as error:

                return {
                    "path":
                        str(
                            path
                        ),

                    "read_error":
                        repr(
                            error
                        ),
                }

    return None


# ============================================================
# FACTOR RANGE ANALYSIS
# ============================================================

def factor_ranges(
    summary_rows: list[dict],
) -> dict:

    result = {}

    for factor in [
        "block_size",
        "a",
        "n0",
        "rng_seed",
    ]:

        rows = [
            row
            for row
            in summary_rows
            if row[
                "factor"
            ]
            in (
                factor,
                "baseline",
            )
        ]

        if not rows:
            continue

        entropy_values = [
            float(
                row[
                    "entropy_mean"
                ]
            )
            for row
            in rows
        ]

        correlation_values = [
            float(
                row[
                    "mean_abs_correlation_mean"
                ]
            )
            for row
            in rows
        ]

        result[
            factor
        ] = {
            "profiles":
                [
                    row[
                        "configuration"
                    ]
                    for row
                    in rows
                ],

            "entropy_mean_min":
                float(
                    min(
                        entropy_values
                    )
                ),

            "entropy_mean_max":
                float(
                    max(
                        entropy_values
                    )
                ),

            "entropy_mean_span":
                float(
                    max(
                        entropy_values
                    )
                    -
                    min(
                        entropy_values
                    )
                ),

            "mean_abs_correlation_min":
                float(
                    min(
                        correlation_values
                    )
                ),

            "mean_abs_correlation_max":
                float(
                    max(
                        correlation_values
                    )
                ),

            "mean_abs_correlation_span":
                float(
                    max(
                        correlation_values
                    )
                    -
                    min(
                        correlation_values
                    )
                ),
        }

    return result


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=" * 108
    )

    print(
        "KAMAL 2021 BASELINE VALIDATION — "
        "SOURCE CONFORMANCE + CONTROLLED "
        "PARAMETER SENSITIVITY"
    )

    print(
        "=" * 108
    )

    # --------------------------------------------------------
    # Verify implementation file
    # --------------------------------------------------------

    implementation_path = (
        SRC_DIR
        / "kamal2021_repro.py"
    )

    if not implementation_path.exists():

        raise FileNotFoundError(
            "\nExpected baseline implementation at:\n"
            f"{implementation_path}\n\n"
            "Copy kamal2021_repro.py into the project's "
            "src directory and run this validation again."
        )

    implementation_hash = (
        file_sha256(
            implementation_path
        )
    )

    image_paths = (
        resolve_sample_images()
    )

    # ========================================================
    # PART 1
    # ========================================================

    print()

    print(
        "[1] Source-equation and implementation "
        "conformance checks..."
    )

    source_checks = (
        source_conformance_checks()
    )

    source_checks[
        "baseline_implementation"
    ] = {
        "file":
            str(
                implementation_path
            ),

        "sha256":
            implementation_hash,
    }

    print(
        "    Zigzag inverse exact:           ",
        source_checks[
            "zigzag_inverse_exact"
        ],
    )

    print(
        "    Eq. (2) Y0 matches reference:   ",
        source_checks[
            "eq2_y0"
        ][
            "matches"
        ],
    )

    print(
        "    Eq. (1)/(3) key matches:        ",
        source_checks[
            "eq1_eq3_logistic_key"
        ][
            "matches"
        ],
    )

    print()

    print(
        "    Source block size:               ",
        AUTHOR_EXPERIMENT_BLOCK_SIZE,
    )

    print(
        "    Source logistic a:               ",
        AUTHOR_EXPERIMENT_A,
    )

    print(
        "    Source N0:                       ",
        AUTHOR_EXPERIMENT_N0,
    )

    print(
        "    Reimplementation RNG seed:       ",
        REPRODUCIBILITY_SEED,
        "(not source-specified)",
    )

    # ========================================================
    # PART 2
    # ========================================================

    profiles = (
        build_profiles()
    )

    print()

    print(
        "[2] One-factor-at-a-time sensitivity evaluation..."
    )

    print(
        "    Profiles:                ",
        len(
            profiles
        ),
    )

    print(
        "    Images per profile:      ",
        len(
            image_paths
        ),
    )

    print(
        "    Total evaluations:       ",
        len(
            profiles
        )
        *
        len(
            image_paths
        ),
    )

    detail_rows = []
    summary_rows = []

    for profile in profiles:

        (
            rows,
            summary,
        ) = evaluate_configuration(
            label=
                profile[
                    "label"
                ],

            factor=
                profile[
                    "factor"
                ],

            block_size=
                profile[
                    "block_size"
                ],

            a=
                profile[
                    "a"
                ],

            n0=
                profile[
                    "n0"
                ],

            rng_seed=
                profile[
                    "rng_seed"
                ],

            image_paths=
                image_paths,
        )

        detail_rows.extend(
            rows
        )

        summary_rows.append(
            summary
        )

        print(
            f"    "
            f"{profile['label']:<22} | "
            f"factor="
            f"{profile['factor']:<10} | "
            f"exact="
            f"{summary['exact_recovery_count']}"
            f"/"
            f"{summary['images']} | "
            f"H="
            f"{summary['entropy_mean']:.6f} | "
            f"|r|="
            f"{summary['mean_abs_correlation_mean']:.6f}"
        )

    # ========================================================
    # PART 3
    # SAVE CSV
    # ========================================================

    detail_csv = (
        OUTPUT_DIR
        / "kamal_sensitivity_detail.csv"
    )

    summary_csv = (
        OUTPUT_DIR
        / "kamal_sensitivity_summary.csv"
    )

    write_csv(
        detail_csv,
        detail_rows,
    )

    write_csv(
        summary_csv,
        summary_rows,
    )

    # ========================================================
    # AGGREGATE RESULTS
    # ========================================================

    total_exact = sum(
        int(
            row[
                "exact_recovery_count"
            ]
        )
        for row
        in summary_rows
    )

    total_evaluations = (
        len(
            summary_rows
        )
        *
        len(
            image_paths
        )
    )

    entropy_means = [
        float(
            row[
                "entropy_mean"
            ]
        )
        for row
        in summary_rows
    ]

    correlation_means = [
        float(
            row[
                "mean_abs_correlation_mean"
            ]
        )
        for row
        in summary_rows
    ]

    ranges = (
        factor_ranges(
            summary_rows
        )
    )

    # ========================================================
    # SAVE SOURCE-CONFORMANCE JSON
    # ========================================================

    source_json = (
        OUTPUT_DIR
        / "kamal_source_conformance.json"
    )

    source_json.write_text(
        json.dumps(
            source_checks,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    final_summary = {
        "experiment":
            (
                "Kamal 2021 controlled-baseline validation"
            ),

        "baseline_file":
            str(
                implementation_path
            ),

        "baseline_file_sha256":
            implementation_hash,

        "validation_images":
            SAMPLE_IMAGES,

        "source_parameter_mapping": {
            "block_size": {
                "value":
                    AUTHOR_EXPERIMENT_BLOCK_SIZE,

                "status":
                    (
                        "author-specified "
                        "experimental setting"
                    ),
            },

            "a": {
                "value":
                    AUTHOR_EXPERIMENT_A,

                "status":
                    (
                        "author-specified "
                        "experimental setting"
                    ),
            },

            "N0": {
                "value":
                    AUTHOR_EXPERIMENT_N0,

                "status":
                    (
                        "author-specified "
                        "experimental setting"
                    ),
            },

            "rng_seed": {
                "value":
                    REPRODUCIBILITY_SEED,

                "status":
                    (
                        "reimplementation-only "
                        "reproducibility choice; "
                        "source PRNG/seed unspecified"
                    ),
            },
        },

        "source_conformance":
            source_checks,

        "sensitivity_design": {
            "design":
                (
                    "one-factor-at-a-time"
                ),

            "profile_count":
                len(
                    profiles
                ),

            "images_per_profile":
                len(
                    image_paths
                ),

            "total_evaluations":
                total_evaluations,

            "profiles":
                profiles,
        },

        "exact_recovery": {
            "count":
                total_exact,

            "total":
                total_evaluations,

            "all_exact":
                bool(
                    total_exact
                    == total_evaluations
                ),
        },

        "all_profile_entropy_mean_range": {
            "minimum":
                float(
                    min(
                        entropy_means
                    )
                ),

            "maximum":
                float(
                    max(
                        entropy_means
                    )
                ),

            "span":
                float(
                    max(
                        entropy_means
                    )
                    -
                    min(
                        entropy_means
                    )
                ),
        },

        "all_profile_mean_abs_correlation_range": {
            "minimum":
                float(
                    min(
                        correlation_means
                    )
                ),

            "maximum":
                float(
                    max(
                        correlation_means
                    )
                ),

            "span":
                float(
                    max(
                        correlation_means
                    )
                    -
                    min(
                        correlation_means
                    )
                ),
        },

        "factor_ranges":
            ranges,

        "existing_100_image_baseline_summary":
            find_existing_100_image_summary(),

        "source_result_reproduction_boundary": {
            "exact_source_ciphertext_reproduction_claimed":
                False,

            "reason":
                (
                    "The source paper specifies the "
                    "algorithm and experimental settings "
                    "block_size=16, a=3.9 and N0=1000, "
                    "but does not specify the PRNG "
                    "implementation or seed used to generate "
                    "the random Ri/r vectors. Therefore the "
                    "exact source ciphertexts cannot be "
                    "uniquely regenerated."
                ),

            "source_reported_entropy_examples": {
                "Img1":
                    7.9993,

                "Img2":
                    7.9994,

                "Img3":
                    7.9974,

                "Img4":
                    7.9972,

                "Img5":
                    7.9977,
            },
        },

        "interpretation_boundary":
            (
                "This validation tests faithful "
                "implementation of the published equations "
                "and stages, exact reversibility, and "
                "sensitivity to supported or underspecified "
                "experimental choices. It does not claim "
                "author-code identity or exact reproduction "
                "of source ciphertexts when random-vector "
                "generation is underspecified."
            ),

        "detail_csv":
            str(
                detail_csv
            ),

        "summary_csv":
            str(
                summary_csv
            ),

        "source_json":
            str(
                source_json
            ),
    }

    summary_json = (
        OUTPUT_DIR
        / "kamal_baseline_validation_summary.json"
    )

    summary_json.write_text(
        json.dumps(
            final_summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL TERMINAL REPORT
    # ========================================================

    print()

    print(
        "=" * 108
    )

    print(
        "[3] FINAL RESULT"
    )

    print(
        "=" * 108
    )

    print(
        "    Zigzag inverse exact:                  ",
        source_checks[
            "zigzag_inverse_exact"
        ],
    )

    print(
        "    Eq. (2) Y0 independently matched:      ",
        source_checks[
            "eq2_y0"
        ][
            "matches"
        ],
    )

    print(
        "    Eq. (1)/(3) key independently matched: ",
        source_checks[
            "eq1_eq3_logistic_key"
        ][
            "matches"
        ],
    )

    print()

    print(
        "    Exact recovery:                        "
        f"{total_exact}"
        f"/"
        f"{total_evaluations}"
    )

    print(
        "    Overall entropy mean range:            "
        f"{min(entropy_means):.6f}"
        " to "
        f"{max(entropy_means):.6f}"
    )

    print(
        "    Overall mean-|r| range:                "
        f"{min(correlation_means):.6f}"
        " to "
        f"{max(correlation_means):.6f}"
    )

    print()

    for factor, values in ranges.items():

        print(
            f"    {factor} entropy span: "
            f"{values['entropy_mean_span']:.8f}"
        )

        print(
            f"    {factor} mean-|r| span: "
            f"{values['mean_abs_correlation_span']:.8f}"
        )

    print()

    print(
        "    Baseline implementation SHA-256:"
    )

    print(
        f"    {implementation_hash}"
    )

    print()

    print(
        f"    Output:  {OUTPUT_DIR}"
    )

    print(
        f"    Summary: {summary_json}"
    )

    print(
        f"    Source:  {source_json}"
    )

    print()

    historical_summary = (
        final_summary[
            "existing_100_image_baseline_summary"
        ]
    )

    if historical_summary is None:

        print(
            "    Historical 100-image summary: "
            "NOT FOUND automatically."
        )

        print(
            "    This does NOT invalidate the current "
            "validation; it only means the old JSON was "
            "not in one of the searched project paths."
        )

    else:

        print(
            "    Historical 100-image summary: FOUND"
        )

        print(
            "    ",
            historical_summary[
                "path"
            ],
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "    block_size=16, a=3.9 and N0=1000 "
        "are source-paper experimental settings."
    )

    print(
        "    rng_seed=2021 is only a deterministic "
        "reimplementation choice because the source "
        "paper does not specify the PRNG/seed."
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "    Exact source ciphertext reproduction is "
        "NOT claimed because Ri/r random-vector generation "
        "is underspecified in the publication."
    )


if __name__ == "__main__":
    main()