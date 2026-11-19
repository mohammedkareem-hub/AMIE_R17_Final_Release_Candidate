from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pydicom
from numpy.typing import NDArray

from dicom_native_io import (
    REPRESENTATIVE_IMAGE_NAME,
    reconstruct_native_dicom,
    resolve_frozen_100_image_dataset,
    split_native_dicom_bytes,
)


# =========================================================
# PROJECT SETTINGS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "native_dicom_byte_adapter"
)

CSV_OUTPUT = (
    OUTPUT_DIR
    / "native_dicom_byte_adapter_100.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "native_dicom_byte_adapter_summary.json"
)

REPRESENTATIVE_RECOVERED = (
    OUTPUT_DIR
    / "representative"
    / "image_021_recovered_adapter.dcm"
)

EXPECTED_TRANSFER_SYNTAX_UID = "1.2.840.10008.1.2.1"  # Explicit VR Little Endian
EXPECTED_SELECTED_FILES = 100


# =========================================================
# DATA STRUCTURE
# =========================================================

@dataclass(frozen=True)
class NativePixelLayout:
    """Technical layout needed for a lossless Pixel Data <-> byte matrix map.

    Important
    ---------
    This structure describes the *stored native bytes*.  It does not apply
    modality rescaling, DICOM windowing, MONOCHROME inversion, min-max
    normalization, or any other intensity transformation.
    """

    rows: int
    columns: int
    samples_per_pixel: int
    number_of_frames: int
    bits_allocated: int
    bits_stored: int
    high_bit: int
    pixel_representation: int
    photometric_interpretation: str
    transfer_syntax_uid: str
    bytes_per_sample: int
    expected_pixel_data_bytes: int
    matrix_rows: int
    matrix_columns: int


# =========================================================
# HASH HELPER
# =========================================================

def sha256_hex(data: bytes) -> str:
    return sha256(data).hexdigest()


# =========================================================
# HEADER-ONLY DICOM METADATA READ
# =========================================================

def read_native_pixel_layout(dicom_path: str | Path) -> NativePixelLayout:
    """Read only technical DICOM metadata required by the native adapter.

    The Pixel Data itself is *not* decoded through pydicom.  The exact raw
    Pixel Data bytes are obtained separately by ``split_native_dicom_bytes``.
    This function is therefore metadata-only and never normalizes pixels.
    """

    path = Path(dicom_path)

    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {path}")

    dataset = pydicom.dcmread(
        path,
        stop_before_pixels=True,
        force=False,
    )

    file_meta = getattr(dataset, "file_meta", None)
    transfer_syntax = str(
        getattr(file_meta, "TransferSyntaxUID", "")
    )

    if transfer_syntax != EXPECTED_TRANSFER_SYNTAX_UID:
        raise ValueError(
            f"{path.name}: unsupported Transfer Syntax UID "
            f"{transfer_syntax!r}. The current exact native mode is "
            "intentionally limited to uncompressed Explicit VR Little "
            f"Endian ({EXPECTED_TRANSFER_SYNTAX_UID})."
        )

    required = (
        "Rows",
        "Columns",
        "SamplesPerPixel",
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PixelRepresentation",
        "PhotometricInterpretation",
    )

    missing = [name for name in required if not hasattr(dataset, name)]
    if missing:
        raise ValueError(
            f"{path.name}: missing required native-pixel DICOM fields: "
            + ", ".join(missing)
        )

    rows = int(dataset.Rows)
    columns = int(dataset.Columns)
    samples_per_pixel = int(dataset.SamplesPerPixel)
    number_of_frames = int(getattr(dataset, "NumberOfFrames", 1) or 1)
    bits_allocated = int(dataset.BitsAllocated)
    bits_stored = int(dataset.BitsStored)
    high_bit = int(dataset.HighBit)
    pixel_representation = int(dataset.PixelRepresentation)
    photometric_interpretation = str(dataset.PhotometricInterpretation)

    if rows <= 0 or columns <= 0:
        raise ValueError(
            f"{path.name}: invalid image dimensions {rows} x {columns}."
        )

    if number_of_frames != 1:
        raise ValueError(
            f"{path.name}: current AMIE native-DICOM mode expects one frame; "
            f"found {number_of_frames}."
        )

    if samples_per_pixel != 1:
        raise ValueError(
            f"{path.name}: current AMIE native-DICOM mode expects "
            f"SamplesPerPixel=1; found {samples_per_pixel}."
        )

    if bits_allocated <= 0 or bits_allocated % 8 != 0:
        raise ValueError(
            f"{path.name}: BitsAllocated={bits_allocated} is not byte-aligned."
        )

    if not (0 < bits_stored <= bits_allocated):
        raise ValueError(
            f"{path.name}: invalid BitsStored={bits_stored} for "
            f"BitsAllocated={bits_allocated}."
        )

    if high_bit != bits_stored - 1:
        raise ValueError(
            f"{path.name}: unexpected HighBit={high_bit}; expected "
            f"BitsStored-1={bits_stored - 1}."
        )

    if pixel_representation not in (0, 1):
        raise ValueError(
            f"{path.name}: unsupported PixelRepresentation="
            f"{pixel_representation}."
        )

    bytes_per_sample = bits_allocated // 8

    expected_pixel_data_bytes = (
        rows
        * columns
        * samples_per_pixel
        * number_of_frames
        * bytes_per_sample
    )

    # For the current single-frame, single-sample dataset, each image row is
    # represented as the exact stored bytes for all native pixels in that row.
    # With BitsAllocated=16 this gives H x (2W), still dtype=uint8.
    matrix_rows = rows * number_of_frames
    matrix_columns = columns * samples_per_pixel * bytes_per_sample

    return NativePixelLayout(
        rows=rows,
        columns=columns,
        samples_per_pixel=samples_per_pixel,
        number_of_frames=number_of_frames,
        bits_allocated=bits_allocated,
        bits_stored=bits_stored,
        high_bit=high_bit,
        pixel_representation=pixel_representation,
        photometric_interpretation=photometric_interpretation,
        transfer_syntax_uid=transfer_syntax,
        bytes_per_sample=bytes_per_sample,
        expected_pixel_data_bytes=expected_pixel_data_bytes,
        matrix_rows=matrix_rows,
        matrix_columns=matrix_columns,
    )


# =========================================================
# LOSSLESS RAW-BYTE <-> UINT8 MATRIX ADAPTER
# =========================================================

def pixel_data_to_uint8_matrix(
    pixel_data: bytes,
    layout: NativePixelLayout,
) -> NDArray[np.uint8]:
    """Map exact native Pixel Data bytes to a 2D uint8 matrix losslessly.

    No numerical interpretation of 12-bit/16-bit pixel values occurs here.
    The original byte sequence is merely reshaped, in original file order,
    into a 2D uint8 array accepted by the existing AMIE-BakerChen pipeline.
    """

    if not isinstance(pixel_data, bytes):
        raise TypeError("pixel_data must be bytes.")

    actual_length = len(pixel_data)
    expected_length = layout.expected_pixel_data_bytes

    if actual_length != expected_length:
        raise ValueError(
            "Native Pixel Data length does not match the DICOM layout: "
            f"actual={actual_length}, expected={expected_length}. "
            "The current 100-image mode intentionally requires an exact, "
            "uncompressed, byte-aligned native payload with no extra pad "
            "byte inside Pixel Data."
        )

    matrix = np.frombuffer(
        pixel_data,
        dtype=np.uint8,
    ).copy()

    matrix = matrix.reshape(
        layout.matrix_rows,
        layout.matrix_columns,
        order="C",
    )

    if matrix.dtype != np.uint8:
        raise RuntimeError("Internal error: native byte matrix is not uint8.")

    if not matrix.flags.c_contiguous:
        matrix = np.ascontiguousarray(matrix)

    return matrix


def uint8_matrix_to_pixel_data(
    matrix: NDArray,
    layout: NativePixelLayout,
) -> bytes:
    """Invert ``pixel_data_to_uint8_matrix`` without changing any byte."""

    array = np.asarray(matrix)

    expected_shape = (
        layout.matrix_rows,
        layout.matrix_columns,
    )

    if array.ndim != 2:
        raise ValueError(
            f"Native byte matrix must be 2D; received shape={array.shape}."
        )

    if array.shape != expected_shape:
        raise ValueError(
            "Native byte matrix shape changed: "
            f"actual={array.shape}, expected={expected_shape}."
        )

    if array.dtype != np.uint8:
        raise ValueError(
            f"Native byte matrix must use uint8; received dtype={array.dtype}."
        )

    recovered = np.ascontiguousarray(array).tobytes(order="C")

    if len(recovered) != layout.expected_pixel_data_bytes:
        raise RuntimeError(
            "Recovered Pixel Data length changed unexpectedly: "
            f"actual={len(recovered)}, "
            f"expected={layout.expected_pixel_data_bytes}."
        )

    return recovered


# =========================================================
# ONE-FILE ADAPTER VALIDATION
# =========================================================

def validate_one_file(record) -> dict[str, object]:
    """Validate the lossless byte-matrix adapter for one frozen DICOM."""

    original_file = record.selected_path.read_bytes()
    parts = split_native_dicom_bytes(original_file)
    layout = read_native_pixel_layout(record.selected_path)

    if parts.pixel_data_length != layout.expected_pixel_data_bytes:
        raise ValueError(
            f"{record.destination_name}: raw Pixel Data length "
            f"({parts.pixel_data_length}) does not match metadata-derived "
            f"length ({layout.expected_pixel_data_bytes})."
        )

    native_matrix = pixel_data_to_uint8_matrix(
        parts.pixel_data,
        layout,
    )

    recovered_pixel_data = uint8_matrix_to_pixel_data(
        native_matrix,
        layout,
    )

    recovered_file = reconstruct_native_dicom(
        parts,
        recovered_pixel_data,
    )

    original_pixel_sha = sha256_hex(parts.pixel_data)
    recovered_pixel_sha = sha256_hex(recovered_pixel_data)
    original_file_sha = sha256_hex(original_file)
    recovered_file_sha = sha256_hex(recovered_file)

    pixel_data_exact = recovered_pixel_data == parts.pixel_data
    pixel_sha256_exact = recovered_pixel_sha == original_pixel_sha
    dicom_byte_exact = recovered_file == original_file
    dicom_sha256_exact = recovered_file_sha == original_file_sha

    if record.destination_name == REPRESENTATIVE_IMAGE_NAME:
        REPRESENTATIVE_RECOVERED.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        REPRESENTATIVE_RECOVERED.write_bytes(recovered_file)

    return {
        "selection_index": record.selection_index,
        "destination_name": record.destination_name,
        "rows": layout.rows,
        "columns": layout.columns,
        "samples_per_pixel": layout.samples_per_pixel,
        "number_of_frames": layout.number_of_frames,
        "bits_allocated": layout.bits_allocated,
        "bits_stored": layout.bits_stored,
        "high_bit": layout.high_bit,
        "pixel_representation": layout.pixel_representation,
        "pixel_representation_name": (
            "signed" if layout.pixel_representation == 1 else "unsigned"
        ),
        "photometric_interpretation": layout.photometric_interpretation,
        "transfer_syntax_uid": layout.transfer_syntax_uid,
        "bytes_per_sample": layout.bytes_per_sample,
        "pixel_data_length_bytes": parts.pixel_data_length,
        "native_matrix_rows": int(native_matrix.shape[0]),
        "native_matrix_columns": int(native_matrix.shape[1]),
        "native_matrix_dtype": str(native_matrix.dtype),
        "native_matrix_nbytes": int(native_matrix.nbytes),
        "original_pixel_data_sha256": original_pixel_sha,
        "recovered_pixel_data_sha256": recovered_pixel_sha,
        "pixel_data_exact": pixel_data_exact,
        "pixel_sha256_exact": pixel_sha256_exact,
        "original_dicom_sha256": original_file_sha,
        "recovered_dicom_sha256": recovered_file_sha,
        "dicom_byte_exact": dicom_byte_exact,
        "dicom_sha256_exact": dicom_sha256_exact,
    }


# =========================================================
# OUTPUT WRITERS
# =========================================================

def write_csv(rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No adapter validation rows were supplied.")

    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with CSV_OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(summary: dict[str, object]) -> None:
    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

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
# COMPLETE FROZEN-100 ADAPTER VALIDATION
# =========================================================

def main() -> None:
    print("=" * 92)
    print("NATIVE-DICOM LOSSLESS BYTE-MATRIX ADAPTER — FROZEN 100-IMAGE SUBSET")
    print("=" * 92)

    print()
    print("[1] Resolving the exact frozen dataset automatically...")
    records = resolve_frozen_100_image_dataset()

    if len(records) != EXPECTED_SELECTED_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_SELECTED_FILES} frozen images, "
            f"found {len(records)}."
        )

    print(f"    PASS: verified frozen subset = {len(records)}/100")

    print()
    print("[2] Testing Pixel Data -> uint8 byte matrix -> Pixel Data -> DICOM...")
    print("    No rescaling, windowing, normalization, or pixel-value conversion is used.")

    rows: list[dict[str, object]] = []

    for record in records:
        result = validate_one_file(record)
        rows.append(result)

        passed = (
            bool(result["pixel_data_exact"])
            and bool(result["pixel_sha256_exact"])
            and bool(result["dicom_byte_exact"])
            and bool(result["dicom_sha256_exact"])
        )

        status = "PASS" if passed else "FAIL"

        print(
            f"    {record.destination_name}: {status} | "
            f"native={result['rows']}x{result['columns']} "
            f"{result['bits_allocated']}-bit | "
            f"byte-matrix={result['native_matrix_rows']}x"
            f"{result['native_matrix_columns']} uint8"
        )

    write_csv(rows)

    pixel_exact_count = sum(bool(row["pixel_data_exact"]) for row in rows)
    pixel_hash_exact_count = sum(bool(row["pixel_sha256_exact"]) for row in rows)
    dicom_exact_count = sum(bool(row["dicom_byte_exact"]) for row in rows)
    dicom_hash_exact_count = sum(bool(row["dicom_sha256_exact"]) for row in rows)

    bits_allocated_counts: dict[str, int] = {}
    bits_stored_counts: dict[str, int] = {}
    pixel_representation_counts: dict[str, int] = {}

    for row in rows:
        key = str(row["bits_allocated"])
        bits_allocated_counts[key] = bits_allocated_counts.get(key, 0) + 1

        key = str(row["bits_stored"])
        bits_stored_counts[key] = bits_stored_counts.get(key, 0) + 1

        key = str(row["pixel_representation_name"])
        pixel_representation_counts[key] = (
            pixel_representation_counts.get(key, 0) + 1
        )

    representative_row = next(
        row
        for row in rows
        if row["destination_name"] == REPRESENTATIVE_IMAGE_NAME
    )

    summary = {
        "experiment": "Native-DICOM lossless byte-matrix adapter validation",
        "stage": "pre-encryption native Pixel Data adapter",
        "selected_files": len(rows),
        "pixel_data_exact_count": pixel_exact_count,
        "pixel_sha256_exact_count": pixel_hash_exact_count,
        "dicom_byte_exact_count": dicom_exact_count,
        "dicom_sha256_exact_count": dicom_hash_exact_count,
        "all_100_pixel_data_exact": pixel_exact_count == EXPECTED_SELECTED_FILES,
        "all_100_dicom_byte_exact": dicom_exact_count == EXPECTED_SELECTED_FILES,
        "all_100_dicom_sha256_exact": (
            dicom_hash_exact_count == EXPECTED_SELECTED_FILES
        ),
        "adapter_representation": (
            "stored native Pixel Data bytes reshaped in file order as "
            "a 2D uint8 matrix; no intensity transformation"
        ),
        "bits_allocated_counts": bits_allocated_counts,
        "bits_stored_counts": bits_stored_counts,
        "pixel_representation_counts": pixel_representation_counts,
        "representative_image": REPRESENTATIVE_IMAGE_NAME,
        "representative_native_shape": [
            representative_row["rows"],
            representative_row["columns"],
        ],
        "representative_byte_matrix_shape": [
            representative_row["native_matrix_rows"],
            representative_row["native_matrix_columns"],
        ],
        "representative_recovered_dicom": str(REPRESENTATIVE_RECOVERED),
        "csv_output": str(CSV_OUTPUT),
    }

    write_summary(summary)

    print()
    print("[3] Final result")
    print(
        f"    Native Pixel Data byte-exact:       "
        f"{pixel_exact_count}/{EXPECTED_SELECTED_FILES}"
    )
    print(
        f"    Native Pixel Data SHA-256 exact:    "
        f"{pixel_hash_exact_count}/{EXPECTED_SELECTED_FILES}"
    )
    print(
        f"    Full DICOM byte-exact recovery:     "
        f"{dicom_exact_count}/{EXPECTED_SELECTED_FILES}"
    )
    print(
        f"    Full DICOM SHA-256 exact recovery:  "
        f"{dicom_hash_exact_count}/{EXPECTED_SELECTED_FILES}"
    )
    print(f"    CSV:     {CSV_OUTPUT}")
    print(f"    Summary: {SUMMARY_OUTPUT}")
    print(
        "    Representative recovered DICOM: "
        f"{REPRESENTATIVE_RECOVERED}"
    )

    if pixel_exact_count != EXPECTED_SELECTED_FILES:
        raise RuntimeError(
            "At least one native Pixel Data payload failed exact adapter recovery."
        )

    if dicom_exact_count != EXPECTED_SELECTED_FILES:
        raise RuntimeError(
            "At least one reconstructed DICOM object changed after the adapter."
        )

    if dicom_hash_exact_count != EXPECTED_SELECTED_FILES:
        raise RuntimeError(
            "At least one reconstructed DICOM SHA-256 digest changed."
        )

    print()
    print(
        "PASS: all 100 native Pixel Data payloads survived the uint8 byte-matrix "
        "adapter exactly."
    )
    print(
        "READY FOR STEP 4: connect this exact byte matrix to the existing "
        "AMIE-BakerChen encryption/decryption pipeline."
    )


if __name__ == "__main__":
    main()
