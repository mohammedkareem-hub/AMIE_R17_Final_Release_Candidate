from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


# =========================================================
# INTERNAL VALIDATION
# =========================================================

def _validate_uint8_grayscale(
    image: NDArray,
    name: str = "image",
) -> NDArray[np.uint8]:
    """
    Validate that an array is a non-empty 2D uint8 grayscale image.
    """
    array = np.asarray(image)

    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D grayscale image.")

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    if array.dtype != np.uint8:
        raise ValueError(
            f"{name} must have dtype uint8. "
            f"Received dtype={array.dtype}."
        )

    return array


# =========================================================
# 1) SHANNON ENTROPY
# =========================================================

def calculate_entropy(image: NDArray) -> float:
    """
    Calculate Shannon entropy for a uint8 grayscale image.

    The theoretical maximum for an 8-bit image is 8 bits.
    """
    image_u8 = _validate_uint8_grayscale(image)

    histogram = np.bincount(
        image_u8.reshape(-1),
        minlength=256,
    ).astype(np.float64)

    probabilities = histogram / histogram.sum()
    probabilities = probabilities[probabilities > 0.0]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return float(entropy)


# =========================================================
# 2) CHI-SQUARE HISTOGRAM UNIFORMITY
# =========================================================

def calculate_chi_square_uniformity(
    image: NDArray,
) -> dict[str, float | int]:
    """
    Evaluate whether the 256-level grayscale histogram approaches
    a uniform distribution using Pearson's chi-square statistic.

    For an image containing N pixels, the expected frequency of each
    gray level under a uniform distribution is N / 256. Lower values
    indicate a histogram closer to uniformity.

    Returns
    -------
    chi_square:
        Pearson chi-square statistic.
    degrees_of_freedom:
        255 for a 256-bin grayscale histogram.
    reduced_chi_square:
        chi_square / degrees_of_freedom, useful when comparing images
        with the same number of histogram bins.
    expected_frequency:
        Expected number of pixels per gray level.

    Notes
    -----
    The statistic depends on image size. It should therefore be
    interpreted together with the image dimensions and, preferably,
    compared among images of equal size.
    """
    image_u8 = _validate_uint8_grayscale(image)

    observed = np.bincount(
        image_u8.reshape(-1),
        minlength=256,
    ).astype(np.float64)

    expected_frequency = image_u8.size / 256.0

    if expected_frequency <= 0.0:
        raise ValueError(
            "Expected histogram frequency must be positive."
        )

    chi_square = float(
        np.sum(
            (observed - expected_frequency) ** 2
            / expected_frequency
        )
    )

    degrees_of_freedom = 255

    return {
        "chi_square": chi_square,
        "degrees_of_freedom": degrees_of_freedom,
        "reduced_chi_square": (
            chi_square / degrees_of_freedom
        ),
        "expected_frequency": float(expected_frequency),
    }


# =========================================================
# 3) ADJACENT-PIXEL CORRELATION
# =========================================================

def _correlation_of_pairs(
    first: NDArray,
    second: NDArray,
) -> float:
    """
    Calculate the Pearson correlation coefficient between
    two equally sized arrays.
    """
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)

    if x.size == 0 or y.size == 0:
        raise ValueError(
            "Correlation requires at least one pixel pair."
        )

    if x.size != y.size:
        raise ValueError(
            "Correlation arrays must have the same number of elements."
        )

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    denominator = np.sqrt(
        np.sum(x_centered**2)
        * np.sum(y_centered**2)
    )

    if denominator == 0.0:
        return 0.0

    return float(
        np.sum(x_centered * y_centered) / denominator
    )


def calculate_correlations(image: NDArray) -> dict[str, float]:
    """
    Calculate horizontal, vertical, and diagonal adjacent-pixel
    correlations for a uint8 grayscale image.
    """
    image_u8 = _validate_uint8_grayscale(image)

    height, width = image_u8.shape

    if height < 2 or width < 2:
        raise ValueError(
            "The image must be at least 2 × 2 "
            "to calculate all three correlations."
        )

    horizontal = _correlation_of_pairs(
        image_u8[:, :-1],
        image_u8[:, 1:],
    )

    vertical = _correlation_of_pairs(
        image_u8[:-1, :],
        image_u8[1:, :],
    )

    diagonal = _correlation_of_pairs(
        image_u8[:-1, :-1],
        image_u8[1:, 1:],
    )

    return {
        "corr_horizontal": horizontal,
        "corr_vertical": vertical,
        "corr_diagonal": diagonal,
    }


# =========================================================
# 4) NPCR AND UACI
# =========================================================

def calculate_npcr_uaci(
    encrypted_image_1: NDArray,
    encrypted_image_2: NDArray,
) -> tuple[float, float]:
    """
    Calculate NPCR and UACI between two encrypted uint8 images.

    IMPORTANT:
    Both encrypted images must be produced using exactly the same
    secret key and encryption parameters. Their corresponding
    plaintext images should differ by only one pixel.

    Returns
    -------
    npcr_percent:
        Number of Pixels Change Rate in percent.
    uaci_percent:
        Unified Average Changing Intensity in percent.
    """
    first = _validate_uint8_grayscale(
        encrypted_image_1,
        "encrypted_image_1",
    )
    second = _validate_uint8_grayscale(
        encrypted_image_2,
        "encrypted_image_2",
    )

    if first.shape != second.shape:
        raise ValueError(
            "Encrypted images must have the same shape."
        )

    changed_pixels = first != second

    npcr = (
        np.count_nonzero(changed_pixels)
        / first.size
        * 100.0
    )

    absolute_difference = np.abs(
        first.astype(np.int16)
        - second.astype(np.int16)
    )

    uaci = (
        np.mean(absolute_difference / 255.0)
        * 100.0
    )

    return float(npcr), float(uaci)


# =========================================================
# 5) ONE-PIXEL PLAINTEXT MODIFICATION
# =========================================================

def create_one_pixel_modified_image(
    image: NDArray,
    row: int = 0,
    col: int = 0,
    delta: int = 1,
) -> NDArray[np.uint8]:
    """
    Return a copy of a uint8 image with exactly one pixel modified.

    The modification is performed modulo 256.
    """
    image_u8 = _validate_uint8_grayscale(image)

    if not 0 <= row < image_u8.shape[0]:
        raise ValueError("row is outside the image bounds.")

    if not 0 <= col < image_u8.shape[1]:
        raise ValueError("col is outside the image bounds.")

    if not isinstance(delta, (int, np.integer)):
        raise TypeError("delta must be an integer.")

    if delta % 256 == 0:
        raise ValueError(
            "delta must change the pixel value modulo 256."
        )

    modified = image_u8.copy()

    modified[row, col] = np.uint8(
        (int(modified[row, col]) + int(delta)) % 256
    )

    return modified


# =========================================================
# 6) EXACT DECRYPTION QUALITY
# =========================================================

def calculate_decryption_quality(
    original: NDArray,
    decrypted: NDArray,
) -> dict[str, float | bool | int]:
    """
    Evaluate exact reconstruction quality.

    For lossless recovery:
        identical = True
        mse = 0
        mae = 0
        maximum_absolute_error = 0
    """
    original_u8 = _validate_uint8_grayscale(
        original,
        "original",
    )
    decrypted_u8 = _validate_uint8_grayscale(
        decrypted,
        "decrypted",
    )

    if original_u8.shape != decrypted_u8.shape:
        raise ValueError(
            "Original and decrypted images must have the same shape."
        )

    difference = (
        original_u8.astype(np.int16)
        - decrypted_u8.astype(np.int16)
    )

    mse = float(np.mean(difference.astype(np.float64) ** 2))
    mae = float(np.mean(np.abs(difference)))
    maximum_error = int(np.max(np.abs(difference)))

    return {
        "identical": bool(
            np.array_equal(original_u8, decrypted_u8)
        ),
        "mse": mse,
        "mae": mae,
        "maximum_absolute_error": maximum_error,
    }


# =========================================================
# 7) IMAGE SIZE INFORMATION
# =========================================================

def get_image_size_info(image: NDArray) -> dict[str, int | float]:
    """
    Return exact image dimensions and in-memory data size.

    size_mib uses 1 MiB = 1024^2 bytes.
    size_mb uses 1 MB = 10^6 bytes.
    """
    array = np.asarray(image)

    if array.ndim != 2:
        raise ValueError(
            "The input must be a 2D grayscale image."
        )

    if array.size == 0:
        raise ValueError("The image must not be empty.")

    height, width = array.shape
    size_bytes = int(array.nbytes)

    return {
        "height": int(height),
        "width": int(width),
        "pixel_count": int(array.size),
        "bits_per_pixel": int(array.itemsize * 8),
        "size_bytes": size_bytes,
        "size_kib": size_bytes / 1024.0,
        "size_mib": size_bytes / (1024.0**2),
        "size_mb": size_bytes / 1_000_000.0,
    }


# =========================================================
# 8) THROUGHPUT
# =========================================================

def calculate_throughput(
    image_size_mb: float,
    encryption_time_sec: float,
) -> float:
    """
    Calculate throughput in decimal MB/s.

    image_size_mb should use:
        size_bytes / 1_000_000
    """
    if image_size_mb < 0.0:
        raise ValueError(
            "image_size_mb must be non-negative."
        )

    if encryption_time_sec <= 0.0:
        raise ValueError(
            "encryption_time_sec must be greater than zero."
        )

    return float(
        image_size_mb / encryption_time_sec
    )


def calculate_throughput_mib(
    image_size_mib: float,
    encryption_time_sec: float,
) -> float:
    """
    Calculate throughput in MiB/s.
    """
    if image_size_mib < 0.0:
        raise ValueError(
            "image_size_mib must be non-negative."
        )

    if encryption_time_sec <= 0.0:
        raise ValueError(
            "encryption_time_sec must be greater than zero."
        )

    return float(
        image_size_mib / encryption_time_sec
    )


# =========================================================
# 9) ESTIMATED CYCLES PER BYTE
# =========================================================

def calculate_cpb(
    cpu_freq_hz: float,
    encryption_time_sec: float,
    image_size_bytes: int,
) -> float:
    """
    Estimate cycles per byte.

    NOTE:
    This is only an estimate because modern CPU frequencies vary
    due to turbo boost, power management, and thermal throttling.
    """
    if cpu_freq_hz <= 0.0:
        raise ValueError(
            "cpu_freq_hz must be greater than zero."
        )

    if encryption_time_sec <= 0.0:
        raise ValueError(
            "encryption_time_sec must be greater than zero."
        )

    if image_size_bytes <= 0:
        raise ValueError(
            "image_size_bytes must be greater than zero."
        )

    return float(
        (cpu_freq_hz * encryption_time_sec)
        / image_size_bytes
    )


# =========================================================
# 10) SINGLE EXECUTION TIME
# =========================================================

def measure_execution_time(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, float]:
    """
    Measure one execution using time.perf_counter().

    Returns
    -------
    result:
        The function result.
    elapsed_seconds:
        Execution time in seconds.
    """
    if not callable(func):
        raise TypeError("func must be callable.")

    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start

    return result, float(elapsed)


# =========================================================
# 11) REPEATED EXECUTION-TIME BENCHMARK
# =========================================================

def benchmark_execution_time(
    func: Callable[..., Any],
    *args: Any,
    repeats: int = 5,
    warmups: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Benchmark a function over repeated executions.

    Warm-up runs are excluded from the statistics.

    Returns
    -------
    A dictionary containing:
        result
        times_sec
        mean_time_sec
        std_time_sec
        min_time_sec
        max_time_sec
        median_time_sec
    """
    if not callable(func):
        raise TypeError("func must be callable.")

    if repeats <= 0:
        raise ValueError("repeats must be greater than zero.")

    if warmups < 0:
        raise ValueError("warmups must be non-negative.")

    for _ in range(warmups):
        func(*args, **kwargs)

    times: list[float] = []
    last_result: Any = None

    for _ in range(repeats):
        start = time.perf_counter()
        last_result = func(*args, **kwargs)
        times.append(time.perf_counter() - start)

    times_array = np.asarray(times, dtype=np.float64)

    return {
        "result": last_result,
        "times_sec": times,
        "mean_time_sec": float(times_array.mean()),
        "std_time_sec": float(times_array.std(ddof=0)),
        "min_time_sec": float(times_array.min()),
        "max_time_sec": float(times_array.max()),
        "median_time_sec": float(
            np.median(times_array)
        ),
        "repeats": int(repeats),
        "warmups": int(warmups),
    }


# =========================================================
# 12) SAVE METRICS TO CSV
# =========================================================

def save_metrics_to_csv(
    metrics_rows: list[dict[str, Any]],
    save_path: str | Path,
) -> None:
    """
    Save a non-empty list of metric dictionaries to CSV.

    All rows must contain exactly the same keys.
    """
    if not metrics_rows:
        raise ValueError(
            "metrics_rows must contain at least one row."
        )

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(metrics_rows[0].keys())
    expected_keys = set(fieldnames)

    for row_index, row in enumerate(metrics_rows):
        if set(row.keys()) != expected_keys:
            raise ValueError(
                f"Row {row_index} does not contain "
                "the same metric fields as the first row."
            )

    with path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(metrics_rows)


# =========================================================
# SELF-TEST
# =========================================================

if __name__ == "__main__":
    print("=" * 70)
    print("IMAGE ENCRYPTION METRICS TEST")
    print("=" * 70)

    rng = np.random.default_rng(12345)

    original = np.tile(
        np.arange(256, dtype=np.uint8),
        (256, 1),
    )

    encrypted_1 = rng.integers(
        0,
        256,
        size=(256, 256),
        dtype=np.uint8,
    )

    encrypted_2 = encrypted_1.copy()
    encrypted_2[0, 0] = np.uint8(
        (int(encrypted_2[0, 0]) + 1) % 256
    )

    modified_original = create_one_pixel_modified_image(
        original,
        row=0,
        col=0,
    )

    npcr, uaci = calculate_npcr_uaci(
        encrypted_1,
        encrypted_2,
    )

    size_info = get_image_size_info(original)

    print(
        f"Original entropy       : "
        f"{calculate_entropy(original):.6f}"
    )
    print(
        f"Encrypted entropy      : "
        f"{calculate_entropy(encrypted_1):.6f}"
    )
    print(
        "Original correlations  :",
        calculate_correlations(original),
    )
    print(
        "Encrypted correlations :",
        calculate_correlations(encrypted_1),
    )
    print(f"NPCR test value        : {npcr:.6f}%")
    print(f"UACI test value        : {uaci:.6f}%")
    print(
        "One pixel changed      :",
        np.count_nonzero(original != modified_original) == 1,
    )
    print("Image size information :", size_info)
    print(
        "Exact recovery test    :",
        calculate_decryption_quality(
            original,
            original.copy(),
        ),
    )

    print("=" * 70)
    print("Metrics test completed.")
