from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from afzal_p21 import (
    AfzalP21Config,
    DNA_RULES,
    decrypt,
    encrypt,
    exact_recovery_report,
    fibonacci_mod4,
)

from dicom_loader import load_dicom_as_uint8


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = Path(__file__).resolve().parent

SELECTED_DIR = (
    PROJECT_ROOT
    / "data"
    / "selected"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "afzal_baseline_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FIXED REPRODUCIBILITY MATERIAL
# ============================================================

SEED = (
    "Afzal-P21-same-hardware-baseline-2025"
)

SAMPLE_IMAGES = [
    "image_001.dcm",
    "image_021.dcm",
    "image_050.dcm",
    "image_075.dcm",
    "image_100.dcm",
]


# Logistic-map values used for sensitivity analysis.
# All remain inside the paper-specified chaotic interval.
ALPHA_VALUES = [
    3.91,
    3.95,
    3.99,
]

X0_VALUES = [
    0.123456789,
    0.314159265,
    0.618033989,
]


# ============================================================
# CURRENT BASELINE CONFIGURATION
# ============================================================

CURRENT_CONFIG = AfzalP21Config(
    alpha=3.99,
    x0=0.123456789,

    stream_hash_source="HK",
    dna_rule_hash_source="HK",

    index_encoding="underscore_decimal",

    fib_mode="worked_example",
)


# ============================================================
# BASIC HELPERS
# ============================================================

def file_sha256(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def entropy_uint8(
    image: np.ndarray,
) -> float:

    x = np.asarray(
        image,
        dtype=np.uint8,
    )

    hist = np.bincount(
        x.reshape(-1),
        minlength=256,
    ).astype(
        np.float64
    )

    p = hist / hist.sum()

    p = p[
        p > 0
    ]

    return float(
        -np.sum(
            p * np.log2(p)
        )
    )


def _correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    x = np.asarray(
        a,
        dtype=np.float64,
    ).reshape(-1)

    y = np.asarray(
        b,
        dtype=np.float64,
    ).reshape(-1)

    x = x - x.mean()
    y = y - y.mean()

    denom = float(
        np.sqrt(
            np.dot(x, x)
            * np.dot(y, y)
        )
    )

    if denom == 0:
        return 0.0

    return float(
        np.dot(x, y)
        / denom
    )


def correlations(
    image: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
]:

    x = np.asarray(
        image,
        dtype=np.uint8,
    )

    h = _correlation(
        x[:, :-1],
        x[:, 1:],
    )

    v = _correlation(
        x[:-1, :],
        x[1:, :],
    )

    d = _correlation(
        x[:-1, :-1],
        x[1:, 1:],
    )

    mean_abs = float(
        np.mean(
            np.abs(
                [
                    h,
                    v,
                    d,
                ]
            )
        )
    )

    return (
        h,
        v,
        d,
        mean_abs,
    )


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def right_rotate(
    values: list[str],
    shift: int,
) -> list[str]:

    shift %= len(values)

    if shift == 0:
        return values[:]

    return (
        values[-shift:]
        + values[:-shift]
    )


def summarize_values(
    values: list[float],
) -> dict[str, float]:

    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
        }

    return {
        "mean":
            float(
                statistics.mean(values)
            ),

        "std":
            float(
                statistics.stdev(values)
                if len(values) > 1
                else 0.0
            ),

        "minimum":
            float(
                min(values)
            ),

        "maximum":
            float(
                max(values)
            ),
    }


# ============================================================
# DATASET
# ============================================================

def resolve_sample_images() -> list[Path]:

    paths = [
        SELECTED_DIR / name
        for name in SAMPLE_IMAGES
    ]

    missing = [
        path
        for path in paths
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Missing sensitivity-test images:\n"
            + "\n".join(
                str(x)
                for x in missing
            )
        )

    return paths


# ============================================================
# PART A
# SOURCE-PAPER CONFORMANCE / AMBIGUITY CHECKS
# ============================================================

def source_paper_checks() -> dict:

    checks: dict = {}

    # --------------------------------------------------------
    # 1. ROM hash worked example
    #
    # Paper:
    # SHA256("securekey123_0")
    # begins:
    # 6d2ef845e6c21841d...
    #
    # and mod 256 = 161.
    # --------------------------------------------------------

    message = (
        b"securekey123_0"
    )

    digest = hashlib.sha256(
        message
    ).hexdigest()

    stream_byte = (
        int(
            digest,
            16,
        )
        % 256
    )

    checks[
        "rom_hash_example"
    ] = {
        "input":
            "securekey123_0",

        "computed_sha256":
            digest,

        "expected_prefix":
            "6d2ef845e6c21841d",

        "prefix_matches":
            bool(
                digest.startswith(
                    "6d2ef845e6c21841d"
                )
            ),

        "computed_mod256":
            int(
                stream_byte
            ),

        "expected_mod256":
            161,

        "mod256_matches":
            bool(
                stream_byte == 161
            ),
    }

    # --------------------------------------------------------
    # 2. ROM XOR example
    #
    # P[0] = 120
    # s0 = 161
    # E[0] = 217
    # --------------------------------------------------------

    xor_value = (
        120 ^ 161
    )

    checks[
        "rom_xor_example"
    ] = {
        "plaintext":
            120,

        "keystream":
            161,

        "computed_xor":
            xor_value,

        "expected_xor":
            217,

        "matches":
            bool(
                xor_value == 217
            ),

        "binary":
            format(
                xor_value,
                "08b",
            ),
    }

    # --------------------------------------------------------
    # 3. ROM DNA worked example
    #
    # The ROM section explicitly calls R0:
    # 00:A, 01:T, 10:C, 11:G
    #
    # This differs from Table 2.
    # --------------------------------------------------------

    rom_r0 = {
        "00": "A",
        "01": "T",
        "10": "C",
        "11": "G",
    }

    bits = format(
        217,
        "08b",
    )

    pairs = [
        bits[i:i + 2]
        for i in range(
            0,
            8,
            2,
        )
    ]

    dna_before_shift = [
        rom_r0[pair]
        for pair in pairs
    ]

    dna_after_shift = (
        right_rotate(
            dna_before_shift,
            1,
        )
    )

    checks[
        "rom_dna_example"
    ] = {
        "binary":
            bits,

        "pairs":
            pairs,

        "rom_r0_mapping":
            rom_r0,

        "computed_before_shift":
            "".join(
                dna_before_shift
            ),

        "expected_before_shift":
            "GTCT",

        "before_shift_matches":
            bool(
                "".join(
                    dna_before_shift
                )
                == "GTCT"
            ),

        "computed_after_shift":
            "".join(
                dna_after_shift
            ),

        "expected_after_shift":
            "TGTC",

        "after_shift_matches":
            bool(
                "".join(
                    dna_after_shift
                )
                == "TGTC"
            ),
    }

    # --------------------------------------------------------
    # 4. Table-2 DNA rule check
    #
    # Published Table 2:
    # R0 = A C G T for 00,01,10,11.
    # --------------------------------------------------------

    table2_expected_r0 = (
        "A",
        "C",
        "G",
        "T",
    )

    implementation_r0 = tuple(
        DNA_RULES[0]
    )

    checks[
        "table2_r0"
    ] = {
        "published_table2_r0":
            list(
                table2_expected_r0
            ),

        "implementation_r0":
            list(
                implementation_r0
            ),

        "implementation_matches_table2":
            bool(
                implementation_r0
                == table2_expected_r0
            ),

        "rom_example_r0":
            [
                "A",
                "T",
                "C",
                "G",
            ],

        "table2_vs_rom_consistent":
            False,
    }

    # --------------------------------------------------------
    # 5. General DNA example inconsistency
    #
    # Paper states:
    # N: 00,01,10,11 -> A,T,C,G
    # Input bits:
    # 110100101011
    #
    # Direct application of that mapping is computed here.
    # --------------------------------------------------------

    generic_mapping = {
        "00": "A",
        "01": "T",
        "10": "C",
        "11": "G",
    }

    generic_bits = (
        "110100101011"
    )

    generic_pairs = [
        generic_bits[i:i + 2]
        for i in range(
            0,
            len(generic_bits),
            2,
        )
    ]

    generic_computed = "".join(
        generic_mapping[pair]
        for pair in generic_pairs
    )

    generic_published = (
        "GTAGCG"
    )

    checks[
        "generic_dna_example"
    ] = {
        "binary":
            generic_bits,

        "stated_mapping":
            generic_mapping,

        "computed_from_stated_mapping":
            generic_computed,

        "published_sequence":
            generic_published,

        "internally_consistent":
            bool(
                generic_computed
                == generic_published
            ),
    }

    # --------------------------------------------------------
    # 6. Fibonacci conventions
    # --------------------------------------------------------

    equation_fib = (
        fibonacci_mod4(
            8,
            "equation",
        )
        .astype(int)
        .tolist()
    )

    worked_fib = (
        fibonacci_mod4(
            8,
            "worked_example",
        )
        .astype(int)
        .tolist()
    )

    checks[
        "fibonacci_conventions"
    ] = {
        "equation_mode_first_8_mod4":
            equation_fib,

        "worked_example_mode_first_8_mod4":
            worked_fib,

        "same_sequence":
            bool(
                equation_fib
                == worked_fib
            ),

        "interpretation":
            (
                "The publication's equation and worked example "
                "use different starting/index conventions."
            ),
    }

    # --------------------------------------------------------
    # 7. Source-result reproducibility boundary
    # --------------------------------------------------------

    checks[
        "table9_reproduction_boundary"
    ] = {
        "source_reports_specific_alpha_for_table9":
            False,

        "source_reports_specific_x0_for_table9":
            False,

        "source_reports_complete_table9_secret_key":
            False,

        "exact_table9_ciphertext_reproduction_claimed":
            False,

        "reason":
            (
                "The publication reports output statistics but does not "
                "provide a complete unique parameter/key specification "
                "sufficient to regenerate those exact ciphertexts."
            ),

        "source_reported_examples": {
            "Baboon": {
                "entropy": 7.9976,
                "NPCR_percent": 99.7268,
                "UACI_percent": 33.72077,
            },

            "Chest": {
                "entropy": 7.9974,
                "NPCR_percent": 99.6520,
                "UACI_percent": 33.43155,
            },
        },
    }

    return checks


# ============================================================
# CONFIGURATION EVALUATION
# ============================================================

def evaluate_configuration(
    *,
    label: str,
    config: AfzalP21Config,
    image_paths: list[Path],
) -> tuple[
    list[dict],
    dict,
]:

    config.validate()

    rows = []

    for path in image_paths:

        image = (
            load_dicom_as_uint8(
                path
            )
        )

        package = encrypt(
            image,
            seed=SEED,
            config=config,
        )

        recovered = decrypt(
            package,
            seed=SEED,
            config=config,
        )

        recovery = (
            exact_recovery_report(
                image,
                recovered,
            )
        )

        h, v, d, mean_abs = (
            correlations(
                package.image
            )
        )

        cipher_sha256 = (
            hashlib.sha256(
                package.image.tobytes(
                    order="C"
                )
            )
            .hexdigest()
        )

        rows.append(
            {
                "configuration":
                    label,

                "image":
                    path.name,

                "alpha":
                    config.alpha,

                "x0":
                    config.x0,

                "stream_hash_source":
                    config.stream_hash_source,

                "dna_rule_hash_source":
                    config.dna_rule_hash_source,

                "index_encoding":
                    config.index_encoding,

                "fib_mode":
                    config.fib_mode,

                "exact_recovery":
                    bool(
                        recovery[
                            "exact_recovery"
                        ]
                    ),

                "mismatched_pixels":
                    int(
                        recovery[
                            "mismatched_pixels"
                        ]
                    ),

                "entropy":
                    entropy_uint8(
                        package.image
                    ),

                "horizontal_correlation":
                    h,

                "vertical_correlation":
                    v,

                "diagonal_correlation":
                    d,

                "mean_abs_correlation":
                    mean_abs,

                "ciphertext_sha256":
                    cipher_sha256,
            }
        )

    entropy_values = [
        float(
            row["entropy"]
        )
        for row in rows
    ]

    corr_values = [
        float(
            row[
                "mean_abs_correlation"
            ]
        )
        for row in rows
    ]

    summary = {
        "configuration":
            label,

        "alpha":
            config.alpha,

        "x0":
            config.x0,

        "stream_hash_source":
            config.stream_hash_source,

        "dna_rule_hash_source":
            config.dna_rule_hash_source,

        "index_encoding":
            config.index_encoding,

        "fib_mode":
            config.fib_mode,

        "images":
            len(rows),

        "exact_recovery_count":
            int(
                sum(
                    bool(
                        row[
                            "exact_recovery"
                        ]
                    )
                    for row in rows
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
                    for row in rows
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
                if len(entropy_values) > 1
                else 0.0
            ),

        "mean_abs_correlation_mean":
            float(
                statistics.mean(
                    corr_values
                )
            ),

        "mean_abs_correlation_std":
            float(
                statistics.stdev(
                    corr_values
                )
                if len(corr_values) > 1
                else 0.0
            ),

        "unique_ciphertexts":
            len(
                {
                    row[
                        "ciphertext_sha256"
                    ]
                    for row in rows
                }
            ),
    }

    return (
        rows,
        summary,
    )


# ============================================================
# PART B
# LOGISTIC PARAMETER SENSITIVITY
# ============================================================

def run_parameter_sensitivity(
    image_paths: list[Path],
) -> tuple[
    list[dict],
    list[dict],
]:

    detail_rows = []
    summary_rows = []

    for alpha in ALPHA_VALUES:

        for x0 in X0_VALUES:

            label = (
                f"alpha={alpha}_x0={x0}"
            )

            config = AfzalP21Config(
                alpha=alpha,
                x0=x0,

                stream_hash_source=
                    CURRENT_CONFIG.stream_hash_source,

                dna_rule_hash_source=
                    CURRENT_CONFIG.dna_rule_hash_source,

                index_encoding=
                    CURRENT_CONFIG.index_encoding,

                fib_mode=
                    CURRENT_CONFIG.fib_mode,
            )

            rows, summary = (
                evaluate_configuration(
                    label=label,
                    config=config,
                    image_paths=image_paths,
                )
            )

            detail_rows.extend(
                rows
            )

            summary_rows.append(
                summary
            )

            print(
                f"    {label}: "
                f"exact="
                f"{summary['exact_recovery_count']}"
                f"/{summary['images']} | "
                f"H="
                f"{summary['entropy_mean']:.6f} | "
                f"|r|="
                f"{summary['mean_abs_correlation_mean']:.6f}"
            )

    return (
        detail_rows,
        summary_rows,
    )


# ============================================================
# PART C
# SOURCE-AMBIGUITY SENSITIVITY
# ============================================================

def ambiguity_profiles() -> list[
    tuple[
        str,
        AfzalP21Config,
    ]
]:

    a = CURRENT_CONFIG.alpha
    x = CURRENT_CONFIG.x0

    return [
        (
            "current_baseline",
            AfzalP21Config(
                alpha=a,
                x0=x,
                stream_hash_source="HK",
                dna_rule_hash_source="HK",
                index_encoding="underscore_decimal",
                fib_mode="worked_example",
            ),
        ),

        # Change only stream hash interpretation.
        (
            "stream_K_only",
            AfzalP21Config(
                alpha=a,
                x0=x,
                stream_hash_source="K",
                dna_rule_hash_source="HK",
                index_encoding="underscore_decimal",
                fib_mode="worked_example",
            ),
        ),

        # Change only DNA-rule hash interpretation.
        (
            "dna_K_only",
            AfzalP21Config(
                alpha=a,
                x0=x,
                stream_hash_source="HK",
                dna_rule_hash_source="K",
                index_encoding="underscore_decimal",
                fib_mode="worked_example",
            ),
        ),

        # Change only index representation.
        (
            "raw_u64_index_only",
            AfzalP21Config(
                alpha=a,
                x0=x,
                stream_hash_source="HK",
                dna_rule_hash_source="HK",
                index_encoding="raw_u64_be",
                fib_mode="worked_example",
            ),
        ),

        # Change only Fibonacci interpretation.
        (
            "equation_fibonacci_only",
            AfzalP21Config(
                alpha=a,
                x0=x,
                stream_hash_source="HK",
                dna_rule_hash_source="HK",
                index_encoding="underscore_decimal",
                fib_mode="equation",
            ),
        ),

        # Closest to the equations in the methodology:
        # stream from HK; DNA rule from K; equation Fibonacci.
        (
            "method_equation_profile",
            AfzalP21Config(
                alpha=a,
                x0=x,
                stream_hash_source="HK",
                dna_rule_hash_source="K",
                index_encoding="underscore_decimal",
                fib_mode="equation",
            ),
        ),

        # Closest to the concrete ROM example:
        # K_i using underscore-decimal and F(0)=1.
        (
            "rom_example_profile",
            AfzalP21Config(
                alpha=a,
                x0=x,
                stream_hash_source="K",
                dna_rule_hash_source="K",
                index_encoding="underscore_decimal",
                fib_mode="worked_example",
            ),
        ),
    ]


def run_ambiguity_sensitivity(
    image_paths: list[Path],
) -> tuple[
    list[dict],
    list[dict],
]:

    detail_rows = []
    summary_rows = []

    for label, config in (
        ambiguity_profiles()
    ):

        rows, summary = (
            evaluate_configuration(
                label=label,
                config=config,
                image_paths=image_paths,
            )
        )

        detail_rows.extend(
            rows
        )

        summary_rows.append(
            summary
        )

        print(
            f"    {label:<26} "
            f"exact="
            f"{summary['exact_recovery_count']}"
            f"/{summary['images']} | "
            f"H="
            f"{summary['entropy_mean']:.6f} | "
            f"|r|="
            f"{summary['mean_abs_correlation_mean']:.6f}"
        )

    return (
        detail_rows,
        summary_rows,
    )


# ============================================================
# OPTIONAL PREVIOUS 100-IMAGE RESULT
# ============================================================

def find_existing_100_image_summary():

    candidates = [
        (
            PROJECT_ROOT
            / "Afzal_P21_100_output"
            / "afzal_p21_100_summary.json"
        ),

        (
            PROJECT_ROOT
            / "output"
            / "Afzal_P21_100_output"
            / "afzal_p21_100_summary.json"
        ),
    ]

    for path in candidates:

        if path.exists():

            try:

                return {
                    "path":
                        str(path),

                    "content":
                        json.loads(
                            path.read_text(
                                encoding="utf-8"
                            )
                        ),
                }

            except Exception as exc:

                return {
                    "path":
                        str(path),

                    "read_error":
                        repr(exc),
                }

    return None


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 108)

    print(
        "AFZAL P21 BASELINE VALIDATION — "
        "SOURCE CONFORMANCE + PARAMETER SENSITIVITY + "
        "AMBIGUITY ANALYSIS"
    )

    print("=" * 108)

    image_paths = (
        resolve_sample_images()
    )

    afzal_source_path = (
        SRC_DIR
        / "afzal_p21.py"
    )

    if not afzal_source_path.exists():

        raise FileNotFoundError(
            "Expected baseline implementation at: "
            f"{afzal_source_path}"
        )

    implementation_sha256 = (
        file_sha256(
            afzal_source_path
        )
    )

    # --------------------------------------------------------
    # A. SOURCE PAPER CHECKS
    # --------------------------------------------------------

    print()
    print(
        "[1] Checking concrete source-paper examples "
        "and documented ambiguities..."
    )

    source_checks = (
        source_paper_checks()
    )

    source_checks[
        "baseline_implementation"
    ] = {
        "file":
            str(
                afzal_source_path
            ),

        "sha256":
            implementation_sha256,

        "current_alpha":
            CURRENT_CONFIG.alpha,

        "current_x0":
            CURRENT_CONFIG.x0,

        "current_stream_hash_source":
            CURRENT_CONFIG.stream_hash_source,

        "current_dna_rule_hash_source":
            CURRENT_CONFIG.dna_rule_hash_source,

        "current_index_encoding":
            CURRENT_CONFIG.index_encoding,

        "current_fib_mode":
            CURRENT_CONFIG.fib_mode,
    }

    print(
        "    ROM SHA-256 prefix:       ",
        "PASS"
        if source_checks[
            "rom_hash_example"
        ][
            "prefix_matches"
        ]
        else "FAIL",
    )

    print(
        "    ROM stream byte s0=161:  ",
        "PASS"
        if source_checks[
            "rom_hash_example"
        ][
            "mod256_matches"
        ]
        else "FAIL",
    )

    print(
        "    ROM XOR 120^161=217:     ",
        "PASS"
        if source_checks[
            "rom_xor_example"
        ][
            "matches"
        ]
        else "FAIL",
    )

    print(
        "    ROM DNA example:         ",
        "PASS"
        if (
            source_checks[
                "rom_dna_example"
            ][
                "before_shift_matches"
            ]
            and
            source_checks[
                "rom_dna_example"
            ][
                "after_shift_matches"
            ]
        )
        else "FAIL",
    )

    print(
        "    Implementation Table-2 R0:",
        "PASS"
        if source_checks[
            "table2_r0"
        ][
            "implementation_matches_table2"
        ]
        else "FAIL",
    )

    print(
        "    Table-2 / ROM R0 consistent:",
        source_checks[
            "table2_r0"
        ][
            "table2_vs_rom_consistent"
        ],
    )

    print(
        "    Generic DNA example internally consistent:",
        source_checks[
            "generic_dna_example"
        ][
            "internally_consistent"
        ],
    )

    # --------------------------------------------------------
    # B. PARAMETER SENSITIVITY
    # --------------------------------------------------------

    print()
    print(
        "[2] Logistic parameter sensitivity "
        "(9 alpha/x0 combinations x 5 frozen images)..."
    )

    (
        parameter_detail,
        parameter_summary,
    ) = run_parameter_sensitivity(
        image_paths
    )

    write_csv(
        OUTPUT_DIR
        / "afzal_parameter_sensitivity_detail.csv",

        parameter_detail,
    )

    write_csv(
        OUTPUT_DIR
        / "afzal_parameter_sensitivity_summary.csv",

        parameter_summary,
    )

    # --------------------------------------------------------
    # C. AMBIGUITY SENSITIVITY
    # --------------------------------------------------------

    print()
    print(
        "[3] Source-interpretation sensitivity "
        "(7 profiles x 5 frozen images)..."
    )

    (
        ambiguity_detail,
        ambiguity_summary,
    ) = run_ambiguity_sensitivity(
        image_paths
    )

    write_csv(
        OUTPUT_DIR
        / "afzal_ambiguity_sensitivity_detail.csv",

        ambiguity_detail,
    )

    write_csv(
        OUTPUT_DIR
        / "afzal_ambiguity_sensitivity_summary.csv",

        ambiguity_summary,
    )

    # --------------------------------------------------------
    # AGGREGATE RANGE ANALYSIS
    # --------------------------------------------------------

    parameter_entropy_means = [
        float(
            row[
                "entropy_mean"
            ]
        )
        for row in parameter_summary
    ]

    parameter_corr_means = [
        float(
            row[
                "mean_abs_correlation_mean"
            ]
        )
        for row in parameter_summary
    ]

    ambiguity_entropy_means = [
        float(
            row[
                "entropy_mean"
            ]
        )
        for row in ambiguity_summary
    ]

    ambiguity_corr_means = [
        float(
            row[
                "mean_abs_correlation_mean"
            ]
        )
        for row in ambiguity_summary
    ]

    parameter_exact = sum(
        int(
            row[
                "exact_recovery_count"
            ]
        )
        for row in parameter_summary
    )

    parameter_total = (
        len(parameter_summary)
        * len(image_paths)
    )

    ambiguity_exact = sum(
        int(
            row[
                "exact_recovery_count"
            ]
        )
        for row in ambiguity_summary
    )

    ambiguity_total = (
        len(ambiguity_summary)
        * len(image_paths)
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    final_summary = {
        "experiment":
            "Afzal P21 controlled-baseline validation",

        "baseline_file":
            str(
                afzal_source_path
            ),

        "baseline_file_sha256":
            implementation_sha256,

        "validation_images":
            SAMPLE_IMAGES,

        "source_paper_checks":
            source_checks,

        "parameter_sensitivity": {
            "alpha_values":
                ALPHA_VALUES,

            "x0_values":
                X0_VALUES,

            "configurations":
                len(
                    parameter_summary
                ),

            "image_evaluations":
                parameter_total,

            "exact_recovery_count":
                parameter_exact,

            "all_exact":
                bool(
                    parameter_exact
                    == parameter_total
                ),

            "entropy_mean_range": {
                "minimum":
                    float(
                        min(
                            parameter_entropy_means
                        )
                    ),

                "maximum":
                    float(
                        max(
                            parameter_entropy_means
                        )
                    ),

                "span":
                    float(
                        max(
                            parameter_entropy_means
                        )
                        -
                        min(
                            parameter_entropy_means
                        )
                    ),
            },

            "mean_abs_correlation_range": {
                "minimum":
                    float(
                        min(
                            parameter_corr_means
                        )
                    ),

                "maximum":
                    float(
                        max(
                            parameter_corr_means
                        )
                    ),

                "span":
                    float(
                        max(
                            parameter_corr_means
                        )
                        -
                        min(
                            parameter_corr_means
                        )
                    ),
            },
        },

        "ambiguity_sensitivity": {
            "profiles":
                [
                    row[
                        "configuration"
                    ]
                    for row
                    in ambiguity_summary
                ],

            "image_evaluations":
                ambiguity_total,

            "exact_recovery_count":
                ambiguity_exact,

            "all_exact":
                bool(
                    ambiguity_exact
                    == ambiguity_total
                ),

            "entropy_mean_range": {
                "minimum":
                    float(
                        min(
                            ambiguity_entropy_means
                        )
                    ),

                "maximum":
                    float(
                        max(
                            ambiguity_entropy_means
                        )
                    ),

                "span":
                    float(
                        max(
                            ambiguity_entropy_means
                        )
                        -
                        min(
                            ambiguity_entropy_means
                        )
                    ),
            },

            "mean_abs_correlation_range": {
                "minimum":
                    float(
                        min(
                            ambiguity_corr_means
                        )
                    ),

                "maximum":
                    float(
                        max(
                            ambiguity_corr_means
                        )
                    ),

                "span":
                    float(
                        max(
                            ambiguity_corr_means
                        )
                        -
                        min(
                            ambiguity_corr_means
                        )
                    ),
            },
        },

        "existing_100_image_baseline_summary":
            find_existing_100_image_summary(),

        "interpretation_boundary":
            (
                "The validation does not claim exact reproduction of "
                "the source paper's Table-9 ciphertext statistics because "
                "the publication does not provide a complete unique key, "
                "alpha, x0, hash-interpretation, and Fibonacci/index "
                "configuration sufficient to regenerate those exact "
                "ciphertexts. Instead, concrete published worked values "
                "are reproduced where possible, publication ambiguities "
                "are explicitly documented, and sensitivity to the "
                "ambiguous choices is measured."
            ),
    }

    source_json = (
        OUTPUT_DIR
        / "afzal_source_conformance.json"
    )

    source_json.write_text(
        json.dumps(
            source_checks,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    summary_json = (
        OUTPUT_DIR
        / "afzal_baseline_validation_summary.json"
    )

    summary_json.write_text(
        json.dumps(
            final_summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 108)
    print("[4] FINAL RESULT")
    print("=" * 108)

    print(
        "    Source ROM hash prefix reproduced:       ",
        source_checks[
            "rom_hash_example"
        ][
            "prefix_matches"
        ],
    )

    print(
        "    Source ROM s0=161 reproduced:           ",
        source_checks[
            "rom_hash_example"
        ][
            "mod256_matches"
        ],
    )

    print(
        "    Source ROM XOR example reproduced:      ",
        source_checks[
            "rom_xor_example"
        ][
            "matches"
        ],
    )

    print(
        "    Source ROM DNA/shift example reproduced:",
        bool(
            source_checks[
                "rom_dna_example"
            ][
                "before_shift_matches"
            ]
            and
            source_checks[
                "rom_dna_example"
            ][
                "after_shift_matches"
            ]
        ),
    )

    print()
    print(
        "    Published Table-2 R0 implemented exactly:",
        source_checks[
            "table2_r0"
        ][
            "implementation_matches_table2"
        ],
    )

    print(
        "    Table-2 and ROM R0 agree:               ",
        source_checks[
            "table2_r0"
        ][
            "table2_vs_rom_consistent"
        ],
    )

    print(
        "    Generic DNA example internally agrees:  ",
        source_checks[
            "generic_dna_example"
        ][
            "internally_consistent"
        ],
    )

    print()
    print(
        "    Logistic parameter exact recovery:       "
        f"{parameter_exact}/{parameter_total}"
    )

    print(
        "    Parameter entropy mean range:            "
        f"{min(parameter_entropy_means):.6f} "
        f"to "
        f"{max(parameter_entropy_means):.6f}"
    )

    print(
        "    Parameter mean-|r| range:                "
        f"{min(parameter_corr_means):.6f} "
        f"to "
        f"{max(parameter_corr_means):.6f}"
    )

    print()
    print(
        "    Ambiguity-profile exact recovery:        "
        f"{ambiguity_exact}/{ambiguity_total}"
    )

    print(
        "    Ambiguity entropy mean range:            "
        f"{min(ambiguity_entropy_means):.6f} "
        f"to "
        f"{max(ambiguity_entropy_means):.6f}"
    )

    print(
        "    Ambiguity mean-|r| range:                "
        f"{min(ambiguity_corr_means):.6f} "
        f"to "
        f"{max(ambiguity_corr_means):.6f}"
    )

    print()
    print(
        "    Baseline implementation SHA-256:"
    )

    print(
        f"    {implementation_sha256}"
    )

    print()
    print(
        f"    Output: {OUTPUT_DIR}"
    )

    print(
        f"    Summary: {summary_json}"
    )

    print(
        f"    Source checks: {source_json}"
    )

    print()

    print(
        "IMPORTANT: exact reproduction of the source "
        "paper's Table-9 ciphertext values is NOT claimed "
        "unless all unpublished experimental parameters "
        "can be recovered."
    )


if __name__ == "__main__":
    main()