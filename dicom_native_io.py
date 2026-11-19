from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from select_images import (
    MANIFEST_PATH,
    NUM_IMAGES,
    RANDOM_SEED,
    RAW_DIR,
    SELECTED_DIR,
    calculate_sha256,
    discover_dicom_files,
    select_files_reproducibly,
)


# =========================================================
# PROJECT SETTINGS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "native_dicom_exact_recovery"
)

CSV_OUTPUT = (
    OUTPUT_DIR
    / "native_dicom_exact_recovery_100.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "native_dicom_exact_recovery_summary.json"
)

REPRESENTATIVE_IMAGE_NAME = "image_021.dcm"
REPRESENTATIVE_RECOVERED = (
    OUTPUT_DIR
    / "representative"
    / "image_021_recovered.dcm"
)

EXPECTED_RAW_FILES = 1985
EXPECTED_SELECTED_FILES = NUM_IMAGES  # 100


# =========================================================
# DICOM BYTE FORMAT CONSTANTS
# =========================================================

PIXEL_DATA_TAG_LE = b"\xe0\x7f\x10\x00"  # (7FE0,0010)

LONG_EXPLICIT_VR = {
    b"OB",
    b"OD",
    b"OF",
    b"OL",
    b"OV",
    b"OW",
    b"SQ",
    b"UC",
    b"UN",
    b"UR",
    b"UT",
}

SUPPORTED_PIXEL_DATA_VR = {
    b"OB",
    b"OW",
}

UNDEFINED_LENGTH = 0xFFFFFFFF


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass(frozen=True)
class FrozenSelectedImage:
    """One member of the exact 100-image frozen experimental subset."""

    selection_index: int
    destination_name: str
    selected_path: Path
    raw_source_path: Path
    relative_source_path: Path
    source_size_bytes: int
    expected_sha256: str


@dataclass(frozen=True)
class NativeDicomParts:
    """Byte-exact split of one native, uncompressed DICOM object.

    prefix
        Every byte from the start of the file through the Pixel Data
        element header.

    pixel_data
        Only the original Pixel Data value bytes.

    suffix
        Every original byte after the Pixel Data value.

    Therefore:

        original_file == prefix + pixel_data + suffix

    provided Pixel Data is recovered exactly.
    """

    prefix: bytes
    pixel_data: bytes
    suffix: bytes
    pixel_data_vr: str
    pixel_data_tag_offset: int
    pixel_data_value_offset: int
    pixel_data_length: int


# =========================================================
# GENERIC HELPERS
# =========================================================

def sha256_hex(data: bytes) -> str:
    return sha256(data).hexdigest()


def _read_manifest() -> list[dict[str, str]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Selection manifest was not found: {MANIFEST_PATH}\n"
            "Run src/select_images.py once to create the frozen subset."
        )

    with MANIFEST_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


# =========================================================
# EXACT PROJECT DATASET RESOLUTION
# =========================================================

def resolve_frozen_100_image_dataset() -> list[FrozenSelectedImage]:
    """Resolve the exact 100 images used by the project automatically.

    This function deliberately requires no image path from the user.

    The project source of truth is:

        data/raw/vestibular_schwannoma_mc_rc
            -> recursively discovered DICOM files
            -> deterministic path sort
            -> random.Random(2026).sample(..., 100)
            -> data/selected/image_001.dcm ... image_100.dcm
            -> data/selected/selected_images.csv

    The function recomputes that selection from the raw dataset and checks
    that it exactly matches the existing frozen manifest. It then verifies
    every selected copy by file size and SHA-256.
    """

    # -----------------------------------------------------
    # 1. Re-discover the original raw DICOM pool.
    # -----------------------------------------------------
    raw_files = discover_dicom_files(RAW_DIR)

    if len(raw_files) != EXPECTED_RAW_FILES:
        raise RuntimeError(
            "Raw dataset count changed. "
            f"Expected {EXPECTED_RAW_FILES}, found {len(raw_files)} in:\n"
            f"{RAW_DIR}"
        )

    # -----------------------------------------------------
    # 2. Recompute the exact seed-2026 sample from raw data.
    # -----------------------------------------------------
    recomputed_selection = select_files_reproducibly(
        raw_files,
        EXPECTED_SELECTED_FILES,
        RANDOM_SEED,
    )

    # -----------------------------------------------------
    # 3. Read the frozen manifest created by select_images.py.
    # -----------------------------------------------------
    rows = _read_manifest()

    if len(rows) != EXPECTED_SELECTED_FILES:
        raise RuntimeError(
            "Manifest row count is incorrect. "
            f"Expected {EXPECTED_SELECTED_FILES}, found {len(rows)}."
        )

    # Preserve the explicit selection order recorded in the manifest.
    rows.sort(key=lambda row: int(row["selection_index"]))

    records: list[FrozenSelectedImage] = []

    for zero_index, (row, raw_source_path) in enumerate(
        zip(rows, recomputed_selection, strict=True)
    ):
        selection_index = zero_index + 1
        expected_destination_name = f"image_{selection_index:03d}.dcm"

        manifest_index = int(row["selection_index"])
        manifest_destination = str(row["destination_name"])
        manifest_relative_source = Path(str(row["relative_source_path"]))
        manifest_size = int(row["source_size_bytes"])
        manifest_sha256 = str(row["sha256"]).lower()
        manifest_seed = int(row["random_seed"])
        manifest_requested = int(row["requested_count"])
        manifest_selected = int(row["selected_count"])
        manifest_total = int(row["total_available"])

        # -------------------------------------------------
        # 4. Verify manifest metadata itself.
        # -------------------------------------------------
        if manifest_index != selection_index:
            raise RuntimeError(
                f"Manifest selection index mismatch at row {selection_index}."
            )

        if manifest_destination != expected_destination_name:
            raise RuntimeError(
                f"Unexpected destination name: {manifest_destination}; "
                f"expected {expected_destination_name}."
            )

        if manifest_seed != RANDOM_SEED:
            raise RuntimeError(
                f"{manifest_destination}: manifest random seed is "
                f"{manifest_seed}, expected {RANDOM_SEED}."
            )

        if manifest_requested != EXPECTED_SELECTED_FILES:
            raise RuntimeError(
                f"{manifest_destination}: requested_count is "
                f"{manifest_requested}, expected {EXPECTED_SELECTED_FILES}."
            )

        if manifest_selected != EXPECTED_SELECTED_FILES:
            raise RuntimeError(
                f"{manifest_destination}: selected_count is "
                f"{manifest_selected}, expected {EXPECTED_SELECTED_FILES}."
            )

        if manifest_total != EXPECTED_RAW_FILES:
            raise RuntimeError(
                f"{manifest_destination}: total_available is "
                f"{manifest_total}, expected {EXPECTED_RAW_FILES}."
            )

        # -------------------------------------------------
        # 5. Prove that seed-2026 selection still gives the
        #    same raw source path recorded in the manifest.
        # -------------------------------------------------
        recomputed_relative_source = raw_source_path.relative_to(RAW_DIR)

        if (
            recomputed_relative_source.as_posix()
            != manifest_relative_source.as_posix()
        ):
            raise RuntimeError(
                "Recomputed seed-2026 selection does not match the frozen "
                f"manifest at {manifest_destination}.\n"
                f"Recomputed: {recomputed_relative_source.as_posix()}\n"
                f"Manifest:   {manifest_relative_source.as_posix()}"
            )

        # -------------------------------------------------
        # 6. Verify the original raw source file.
        # -------------------------------------------------
        if not raw_source_path.exists():
            raise FileNotFoundError(
                f"Raw source file is missing: {raw_source_path}"
            )

        raw_size = raw_source_path.stat().st_size
        if raw_size != manifest_size:
            raise RuntimeError(
                f"{manifest_destination}: raw source size mismatch. "
                f"actual={raw_size}, manifest={manifest_size}"
            )

        raw_sha256 = calculate_sha256(raw_source_path)
        if raw_sha256.lower() != manifest_sha256:
            raise RuntimeError(
                f"{manifest_destination}: raw source SHA-256 does not "
                "match the frozen manifest."
            )

        # -------------------------------------------------
        # 7. Verify the copy used by all current experiments.
        # -------------------------------------------------
        selected_path = SELECTED_DIR / manifest_destination

        if not selected_path.exists():
            raise FileNotFoundError(
                f"Frozen selected file is missing: {selected_path}"
            )

        selected_size = selected_path.stat().st_size
        if selected_size != manifest_size:
            raise RuntimeError(
                f"{manifest_destination}: selected copy size mismatch. "
                f"actual={selected_size}, manifest={manifest_size}"
            )

        selected_sha256 = calculate_sha256(selected_path)
        if selected_sha256.lower() != manifest_sha256:
            raise RuntimeError(
                f"{manifest_destination}: selected copy SHA-256 does not "
                "match the frozen manifest."
            )

        records.append(
            FrozenSelectedImage(
                selection_index=selection_index,
                destination_name=manifest_destination,
                selected_path=selected_path,
                raw_source_path=raw_source_path,
                relative_source_path=manifest_relative_source,
                source_size_bytes=manifest_size,
                expected_sha256=manifest_sha256,
            )
        )

    return records


# =========================================================
# BYTE-EXACT PIXEL DATA SPLITTING
# =========================================================

def _candidate_pixel_data_parts(
    raw: bytes,
    tag_offset: int,
) -> NativeDicomParts | None:
    """Parse one native Explicit-VR Little-Endian Pixel Data candidate."""

    if tag_offset < 0 or tag_offset + 12 > len(raw):
        return None

    vr = raw[tag_offset + 4: tag_offset + 6]

    if vr not in SUPPORTED_PIXEL_DATA_VR:
        return None

    if vr not in LONG_EXPLICIT_VR:
        return None

    reserved = raw[tag_offset + 6: tag_offset + 8]
    if reserved != b"\x00\x00":
        return None

    value_length = int.from_bytes(
        raw[tag_offset + 8: tag_offset + 12],
        byteorder="little",
        signed=False,
    )

    # Undefined-length Pixel Data means encapsulated/compressed data.
    # The present native-DICOM path intentionally rejects such objects.
    if value_length == UNDEFINED_LENGTH:
        return None

    value_offset = tag_offset + 12
    value_end = value_offset + value_length

    if value_end > len(raw):
        return None

    return NativeDicomParts(
        prefix=raw[:value_offset],
        pixel_data=raw[value_offset:value_end],
        suffix=raw[value_end:],
        pixel_data_vr=vr.decode("ascii"),
        pixel_data_tag_offset=tag_offset,
        pixel_data_value_offset=value_offset,
        pixel_data_length=value_length,
    )


def split_native_dicom_bytes(raw: bytes) -> NativeDicomParts:
    """Split a native uncompressed DICOM object without rewriting it."""

    if not isinstance(raw, bytes):
        raise TypeError("raw must be bytes.")

    if not raw:
        raise ValueError("DICOM byte stream must not be empty.")

    candidates: list[NativeDicomParts] = []
    start = 0

    while True:
        tag_offset = raw.find(PIXEL_DATA_TAG_LE, start)

        if tag_offset < 0:
            break

        candidate = _candidate_pixel_data_parts(raw, tag_offset)

        if candidate is not None:
            candidates.append(candidate)

        start = tag_offset + 1

    if not candidates:
        raise ValueError(
            "No supported native Pixel Data element was found. "
            "Expected uncompressed Explicit-VR Little-Endian Pixel Data "
            "with VR OB or OW and a defined value length."
        )

    if len(candidates) != 1:
        raise ValueError(
            "Ambiguous DICOM byte stream: more than one valid native "
            "Pixel Data candidate was found."
        )

    return candidates[0]


def reconstruct_native_dicom(
    parts: NativeDicomParts,
    recovered_pixel_data: bytes,
) -> bytes:
    """Reconstruct the original DICOM byte stream from recovered pixels."""

    if not isinstance(recovered_pixel_data, bytes):
        raise TypeError("recovered_pixel_data must be bytes.")

    if len(recovered_pixel_data) != parts.pixel_data_length:
        raise ValueError(
            "Recovered Pixel Data length does not match the original "
            f"length: {len(recovered_pixel_data)} != "
            f"{parts.pixel_data_length}."
        )

    return parts.prefix + recovered_pixel_data + parts.suffix


# =========================================================
# ONE-FILE ROUND-TRIP TEST
# =========================================================

def verify_exact_round_trip(
    record: FrozenSelectedImage,
) -> dict[str, object]:
    """Split and reconstruct one frozen selected DICOM with no encryption."""

    original = record.selected_path.read_bytes()
    parts = split_native_dicom_bytes(original)

    # This validation gate intentionally performs no encryption yet.
    # Pixel Data passes through unchanged so that DICOM I/O is isolated.
    recovered = reconstruct_native_dicom(
        parts,
        parts.pixel_data,
    )

    original_digest = sha256_hex(original)
    recovered_digest = sha256_hex(recovered)

    raw_source_digest = calculate_sha256(record.raw_source_path)

    result = {
        "selection_index": record.selection_index,
        "destination_name": record.destination_name,
        "relative_source_path": record.relative_source_path.as_posix(),
        "file_size_bytes": len(original),
        "pixel_data_vr": parts.pixel_data_vr,
        "pixel_data_tag_offset": parts.pixel_data_tag_offset,
        "pixel_data_value_offset": parts.pixel_data_value_offset,
        "pixel_data_length_bytes": parts.pixel_data_length,
        "prefix_length_bytes": len(parts.prefix),
        "suffix_length_bytes": len(parts.suffix),
        "manifest_sha256": record.expected_sha256,
        "raw_source_sha256": raw_source_digest,
        "selected_sha256": original_digest,
        "recovered_sha256": recovered_digest,
        "raw_matches_manifest": (
            raw_source_digest.lower() == record.expected_sha256.lower()
        ),
        "selected_matches_manifest": (
            original_digest.lower() == record.expected_sha256.lower()
        ),
        "pixel_data_length_preserved": (
            len(parts.pixel_data) == parts.pixel_data_length
        ),
        "byte_exact": original == recovered,
        "sha256_exact": original_digest == recovered_digest,
    }

    if record.destination_name == REPRESENTATIVE_IMAGE_NAME:
        REPRESENTATIVE_RECOVERED.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        REPRESENTATIVE_RECOVERED.write_bytes(recovered)

    return result


# =========================================================
# OUTPUT WRITERS
# =========================================================

def _write_csv(rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No result rows were supplied.")

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
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(summary: dict[str, object]) -> None:
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
    print("=" * 88)
    print("NATIVE-DICOM BYTE-EXACT I/O VALIDATION — FROZEN 100-IMAGE SUBSET")
    print("=" * 88)

    print()
    print("[1] Resolving the project dataset automatically...")
    print(f"    Raw dataset:      {RAW_DIR}")
    print(f"    Frozen selection: {SELECTED_DIR}")
    print(f"    Manifest:         {MANIFEST_PATH}")
    print(f"    Selection rule:   random.Random({RANDOM_SEED}).sample(..., 100)")

    records = resolve_frozen_100_image_dataset()

    print(
        f"    PASS: raw pool={EXPECTED_RAW_FILES}, "
        f"frozen subset={len(records)}, manifest verified."
    )

    print()
    print("[2] Testing byte-exact DICOM split/reconstruction for all 100 files...")

    rows: list[dict[str, object]] = []

    for record in records:
        result = verify_exact_round_trip(record)
        rows.append(result)

        status = (
            "PASS"
            if bool(result["byte_exact"]) and bool(result["sha256_exact"])
            else "FAIL"
        )

        print(
            f"    {record.destination_name}: {status} | "
            f"PixelData={result['pixel_data_length_bytes']} bytes | "
            f"VR={result['pixel_data_vr']}"
        )

    _write_csv(rows)

    byte_exact_count = sum(bool(row["byte_exact"]) for row in rows)
    sha_exact_count = sum(bool(row["sha256_exact"]) for row in rows)
    manifest_match_count = sum(
        bool(row["selected_matches_manifest"])
        and bool(row["raw_matches_manifest"])
        for row in rows
    )

    total_pixel_data_bytes = sum(
        int(row["pixel_data_length_bytes"])
        for row in rows
    )

    vr_counts: dict[str, int] = {}
    for row in rows:
        vr = str(row["pixel_data_vr"])
        vr_counts[vr] = vr_counts.get(vr, 0) + 1

    summary = {
        "experiment": "Native-DICOM byte-exact I/O validation",
        "stage": "pre-encryption validation",
        "raw_dataset_directory": str(RAW_DIR),
        "selected_dataset_directory": str(SELECTED_DIR),
        "manifest_path": str(MANIFEST_PATH),
        "random_seed": RANDOM_SEED,
        "expected_raw_files": EXPECTED_RAW_FILES,
        "expected_selected_files": EXPECTED_SELECTED_FILES,
        "verified_selected_files": len(records),
        "manifest_and_raw_source_matches": manifest_match_count,
        "byte_exact_recovery_count": byte_exact_count,
        "sha256_exact_recovery_count": sha_exact_count,
        "all_100_byte_exact": byte_exact_count == EXPECTED_SELECTED_FILES,
        "all_100_sha256_exact": sha_exact_count == EXPECTED_SELECTED_FILES,
        "total_native_pixel_data_bytes": total_pixel_data_bytes,
        "pixel_data_vr_counts": vr_counts,
        "representative_recovered_file": str(REPRESENTATIVE_RECOVERED),
        "csv_output": str(CSV_OUTPUT),
    }

    _write_summary(summary)

    print()
    print("[3] Final result")
    print(
        f"    Manifest/raw/selected verification: "
        f"{manifest_match_count}/{EXPECTED_SELECTED_FILES}"
    )
    print(
        f"    Byte-exact DICOM recovery:           "
        f"{byte_exact_count}/{EXPECTED_SELECTED_FILES}"
    )
    print(
        f"    SHA-256 exact DICOM recovery:        "
        f"{sha_exact_count}/{EXPECTED_SELECTED_FILES}"
    )
    print(f"    CSV:     {CSV_OUTPUT}")
    print(f"    Summary: {SUMMARY_OUTPUT}")
    print(
        f"    Representative recovered DICOM: "
        f"{REPRESENTATIVE_RECOVERED}"
    )

    if byte_exact_count != EXPECTED_SELECTED_FILES:
        raise RuntimeError(
            "At least one DICOM object failed byte-exact reconstruction."
        )

    if sha_exact_count != EXPECTED_SELECTED_FILES:
        raise RuntimeError(
            "At least one DICOM object failed SHA-256 exact reconstruction."
        )

    print()
    print("PASS: all 100 frozen DICOM objects were reconstructed byte-for-byte.")


if __name__ == "__main__":
    main()
