from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from numpy.typing import NDArray
from pydicom.dataset import Dataset


# =========================================================
# HELPERS
# =========================================================

def _first_numeric_value(
    value: Any,
    default: float | None = None,
) -> float | None:
    """
    Convert a DICOM numeric value to float.

    MultiValue fields such as WindowCenter and WindowWidth may
    contain more than one value. In that case, the first value is used.
    """
    if value is None:
        return default

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        value = value[0]

    try:
        # pydicom MultiValue behaves like a sequence but is not
        # always a built-in list or tuple.
        if not isinstance(value, (str, bytes)) and hasattr(value, "__len__"):
            if len(value) > 0 and not np.isscalar(value):
                value = value[0]
    except TypeError:
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _validate_2d_image(image: NDArray, name: str = "image") -> NDArray:
    """
    Validate that the supplied array is a non-empty 2D image.
    """
    array = np.asarray(image)

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D grayscale image. "
            f"Received shape={array.shape}."
        )

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    return array


# =========================================================
# DICOM INTENSITY PROCESSING
# =========================================================

def apply_rescale(
    pixel_array: NDArray,
    dataset: Dataset,
) -> NDArray[np.float64]:
    """
    Apply DICOM RescaleSlope and RescaleIntercept.

    For CT images this converts stored pixel values into the
    physical intensity domain, commonly Hounsfield Units.

        output = pixel * RescaleSlope + RescaleIntercept
    """
    pixels = np.asarray(pixel_array, dtype=np.float64)

    slope = _first_numeric_value(
        getattr(dataset, "RescaleSlope", None),
        default=1.0,
    )
    intercept = _first_numeric_value(
        getattr(dataset, "RescaleIntercept", None),
        default=0.0,
    )

    return pixels * float(slope) + float(intercept)


def apply_window(
    image: NDArray,
    window_center: float,
    window_width: float,
) -> NDArray[np.float64]:
    """
    Apply linear DICOM windowing.

    Values outside the selected window are clipped.
    """
    array = np.asarray(image, dtype=np.float64)

    if window_width <= 0:
        raise ValueError("window_width must be greater than zero.")

    lower = window_center - window_width / 2.0
    upper = window_center + window_width / 2.0

    return np.clip(array, lower, upper)


def normalize_to_uint8(
    image: NDArray,
) -> NDArray[np.uint8]:
    """
    Normalize a 2D grayscale image into uint8 range [0, 255].
    """
    array = _validate_2d_image(image)
    array = np.asarray(array, dtype=np.float64)

    if not np.all(np.isfinite(array)):
        raise ValueError(
            "The image contains NaN or infinite values."
        )

    minimum = float(array.min())
    maximum = float(array.max())

    if maximum == minimum:
        return np.zeros(array.shape, dtype=np.uint8)

    normalized = (
        (array - minimum)
        / (maximum - minimum)
    )

    return np.rint(
        normalized * 255.0
    ).astype(np.uint8)


def invert_monochrome1(
    image: NDArray,
) -> NDArray:
    """
    Invert a MONOCHROME1 image so that larger values appear brighter,
    matching the usual MONOCHROME2 display convention.
    """
    array = np.asarray(image)

    minimum = array.min()
    maximum = array.max()

    return maximum + minimum - array


# =========================================================
# METADATA EXTRACTION
# =========================================================

def extract_dicom_metadata(
    dataset: Dataset,
    original_pixel_array: NDArray,
) -> dict[str, Any]:
    """
    Extract non-identifying technical metadata useful for experiments.

    Patient-identifying fields are intentionally omitted.
    """
    bits_allocated = getattr(dataset, "BitsAllocated", None)
    bits_stored = getattr(dataset, "BitsStored", None)
    high_bit = getattr(dataset, "HighBit", None)
    pixel_representation = getattr(
        dataset,
        "PixelRepresentation",
        None,
    )

    slope = _first_numeric_value(
        getattr(dataset, "RescaleSlope", None),
        default=1.0,
    )
    intercept = _first_numeric_value(
        getattr(dataset, "RescaleIntercept", None),
        default=0.0,
    )
    window_center = _first_numeric_value(
        getattr(dataset, "WindowCenter", None),
        default=None,
    )
    window_width = _first_numeric_value(
        getattr(dataset, "WindowWidth", None),
        default=None,
    )

    return {
        "rows": int(original_pixel_array.shape[-2]),
        "columns": int(original_pixel_array.shape[-1]),
        "number_of_frames": int(
            getattr(dataset, "NumberOfFrames", 1)
        ),
        "original_dtype": str(original_pixel_array.dtype),
        "bits_allocated": (
            int(bits_allocated)
            if bits_allocated is not None
            else None
        ),
        "bits_stored": (
            int(bits_stored)
            if bits_stored is not None
            else None
        ),
        "high_bit": (
            int(high_bit)
            if high_bit is not None
            else None
        ),
        "pixel_representation": (
            int(pixel_representation)
            if pixel_representation is not None
            else None
        ),
        "photometric_interpretation": str(
            getattr(
                dataset,
                "PhotometricInterpretation",
                "",
            )
        ),
        "modality": str(
            getattr(dataset, "Modality", "")
        ),
        "rescale_slope": float(slope),
        "rescale_intercept": float(intercept),
        "window_center": window_center,
        "window_width": window_width,
        "transfer_syntax_uid": str(
            getattr(
                getattr(dataset, "file_meta", None),
                "TransferSyntaxUID",
                "",
            )
        ),
    }


# =========================================================
# MAIN DICOM LOADER
# =========================================================

def load_dicom_image(
    dicom_path: str | Path,
    *,
    apply_modality_rescale: bool = True,
    apply_dicom_window: bool = False,
    invert_monochrome1_images: bool = True,
    frame_index: int = 0,
) -> tuple[NDArray[np.uint8], dict[str, Any]]:
    """
    Load a DICOM image and return:

        normalized uint8 image, technical metadata

    Processing order:
      1. Read the DICOM dataset.
      2. Extract the selected frame.
      3. Apply RescaleSlope/RescaleIntercept when requested.
      4. Apply DICOM windowing when requested and available.
      5. Invert MONOCHROME1 when requested.
      6. Normalize to uint8.

    Notes
    -----
    The returned image is an 8-bit normalized research image.
    It is not a lossless copy of the original 12-bit or 16-bit
    DICOM pixel data.
    """
    path = Path(dicom_path)

    if not path.exists():
        raise FileNotFoundError(
            f"DICOM file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"The DICOM path is not a file: {path}"
        )

    dataset = pydicom.dcmread(path)

    if "PixelData" not in dataset:
        raise ValueError(
            f"The DICOM file contains no PixelData: {path}"
        )

    pixel_array = np.asarray(dataset.pixel_array)

    metadata = extract_dicom_metadata(
        dataset,
        pixel_array,
    )

    # Multi-frame DICOM
    if pixel_array.ndim == 3:
        number_of_frames = pixel_array.shape[0]

        if not 0 <= frame_index < number_of_frames:
            raise ValueError(
                f"frame_index must be between 0 and "
                f"{number_of_frames - 1}."
            )

        image = pixel_array[frame_index]

    elif pixel_array.ndim == 2:
        image = pixel_array

    else:
        raise ValueError(
            "Only 2D grayscale or multi-frame grayscale DICOM "
            f"images are supported. Received shape={pixel_array.shape}."
        )

    image = _validate_2d_image(
        image,
        "DICOM pixel array",
    )

    if apply_modality_rescale:
        processed = apply_rescale(
            image,
            dataset,
        )
    else:
        processed = np.asarray(
            image,
            dtype=np.float64,
        )

    window_applied = False

    if apply_dicom_window:
        window_center = metadata["window_center"]
        window_width = metadata["window_width"]

        if (
            window_center is not None
            and window_width is not None
            and window_width > 0
        ):
            processed = apply_window(
                processed,
                window_center,
                window_width,
            )
            window_applied = True

    photometric = metadata[
        "photometric_interpretation"
    ].upper()

    monochrome1_inverted = False

    if (
        invert_monochrome1_images
        and photometric == "MONOCHROME1"
    ):
        processed = invert_monochrome1(
            processed
        )
        monochrome1_inverted = True

    image_uint8 = normalize_to_uint8(
        processed
    )

    metadata.update(
        {
            "selected_frame_index": int(frame_index),
            "modality_rescale_applied": bool(
                apply_modality_rescale
            ),
            "windowing_requested": bool(
                apply_dicom_window
            ),
            "windowing_applied": bool(
                window_applied
            ),
            "monochrome1_inverted": bool(
                monochrome1_inverted
            ),
            "output_dtype": str(image_uint8.dtype),
            "output_min": int(image_uint8.min()),
            "output_max": int(image_uint8.max()),
            "output_shape": tuple(
                int(value)
                for value in image_uint8.shape
            ),
        }
    )

    return image_uint8, metadata


# =========================================================
# BACKWARD-COMPATIBLE FUNCTION
# =========================================================

def load_dicom_as_uint8(
    dicom_path: str | Path,
    *,
    apply_modality_rescale: bool = True,
    apply_dicom_window: bool = False,
    invert_monochrome1_images: bool = True,
    frame_index: int = 0,
) -> NDArray[np.uint8]:
    """
    Load a DICOM file as a normalized uint8 image.

    This function preserves compatibility with the existing pipeline,
    which expects the loader to return the image only.
    """
    image, _ = load_dicom_image(
        dicom_path,
        apply_modality_rescale=apply_modality_rescale,
        apply_dicom_window=apply_dicom_window,
        invert_monochrome1_images=invert_monochrome1_images,
        frame_index=frame_index,
    )

    return image


# =========================================================
# SELF-TEST
# =========================================================

if __name__ == "__main__":
    import sys

    print("=" * 72)
    print("DICOM LOADER TEST")
    print("=" * 72)

    if len(sys.argv) < 2:
        print(
            "Usage: python dicom_loader.py path_to_image.dcm"
        )
        raise SystemExit(0)

    test_path = Path(sys.argv[1])

    image, metadata = load_dicom_image(
        test_path,
        apply_modality_rescale=True,
        apply_dicom_window=False,
        invert_monochrome1_images=True,
    )

    print(f"Loaded file          : {test_path.name}")
    print(f"Image shape          : {image.shape}")
    print(f"Image dtype          : {image.dtype}")
    print(f"Image range          : {image.min()} - {image.max()}")
    print(f"Modality             : {metadata['modality']}")
    print(
        "Photometric          : "
        f"{metadata['photometric_interpretation']}"
    )
    print(
        "Bits allocated/stored: "
        f"{metadata['bits_allocated']} / "
        f"{metadata['bits_stored']}"
    )
    print(
        "Rescale slope        : "
        f"{metadata['rescale_slope']}"
    )
    print(
        "Rescale intercept    : "
        f"{metadata['rescale_intercept']}"
    )
    print(
        "Window applied       : "
        f"{metadata['windowing_applied']}"
    )
    print(
        "MONOCHROME1 inverted : "
        f"{metadata['monochrome1_inverted']}"
    )
    print("=" * 72)
    print("DICOM loader test completed.")
