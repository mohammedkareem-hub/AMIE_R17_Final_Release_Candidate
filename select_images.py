from __future__ import annotations

import csv
import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# =========================================================
# SETTINGS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "vestibular_schwannoma_mc_rc"
)

SELECTED_DIR = (
    PROJECT_ROOT
    / "data"
    / "selected"
)

MANIFEST_PATH = (
    SELECTED_DIR
    / "selected_images.csv"
)

NUM_IMAGES = 100
RANDOM_SEED = 2026

# When True, existing .dcm files and the previous manifest
# inside SELECTED_DIR are removed before creating a new sample.
CLEAR_SELECTED_DIR = False


# =========================================================
# DATA STRUCTURE
# =========================================================

@dataclass(frozen=True)
class SelectedImageRecord:
    """
    One reproducibly selected DICOM image.
    """

    selection_index: int
    source_path: Path
    relative_source_path: Path
    destination_name: str
    source_size_bytes: int
    sha256: str


# =========================================================
# VALIDATION
# =========================================================

def validate_directories(
    raw_dir: Path,
    selected_dir: Path,
) -> None:
    """
    Validate the source directory and create the output directory.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw DICOM directory was not found: {raw_dir}"
        )

    if not raw_dir.is_dir():
        raise NotADirectoryError(
            f"Raw DICOM path is not a directory: {raw_dir}"
        )

    selected_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


# =========================================================
# FILE DISCOVERY
# =========================================================

def discover_dicom_files(
    raw_dir: Path,
) -> list[Path]:
    """
    Discover all DICOM files recursively in deterministic order.

    Both .dcm and .DCM file extensions are accepted.
    """
    files = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".dcm"
    ]

    # Sort by relative POSIX path for reproducibility
    # across repeated executions on the same dataset layout.
    files.sort(
        key=lambda path: (
            path.relative_to(raw_dir)
            .as_posix()
            .lower()
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No DICOM files were found in: {raw_dir}"
        )

    return files


# =========================================================
# HASHING
# =========================================================

def calculate_sha256(
    file_path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate SHA-256 for one file without loading it all into memory.
    """
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# =========================================================
# OUTPUT CLEANUP
# =========================================================

def clear_previous_selection(
    selected_dir: Path,
    manifest_path: Path,
) -> None:
    """
    Remove only files created by this selection workflow.

    Other unrelated files are left untouched.
    """
    for file_path in selected_dir.glob("image_*.dcm"):
        if file_path.is_file():
            file_path.unlink()

    if manifest_path.exists():
        manifest_path.unlink()


# =========================================================
# REPRODUCIBLE SELECTION
# =========================================================

def select_files_reproducibly(
    all_files: list[Path],
    number_to_select: int,
    random_seed: int,
) -> list[Path]:
    """
    Select a deterministic random sample using a fixed seed.
    """
    if number_to_select <= 0:
        raise ValueError(
            "number_to_select must be greater than zero."
        )

    sample_size = min(
        number_to_select,
        len(all_files),
    )

    rng = random.Random(random_seed)

    selected = rng.sample(
        all_files,
        sample_size,
    )

    # Keep sample order deterministic and stable.
    return selected


# =========================================================
# COPY AND RECORD
# =========================================================

def copy_selected_files(
    selected_files: Iterable[Path],
    raw_dir: Path,
    selected_dir: Path,
) -> list[SelectedImageRecord]:
    """
    Copy selected DICOM files and create manifest records.
    """
    records: list[SelectedImageRecord] = []

    for index, source_path in enumerate(
        selected_files,
        start=1,
    ):
        destination_name = (
            f"image_{index:03d}.dcm"
        )

        destination_path = (
            selected_dir
            / destination_name
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        relative_source = (
            source_path.relative_to(raw_dir)
        )

        record = SelectedImageRecord(
            selection_index=index,
            source_path=source_path,
            relative_source_path=relative_source,
            destination_name=destination_name,
            source_size_bytes=source_path.stat().st_size,
            sha256=calculate_sha256(source_path),
        )

        records.append(record)

    return records


# =========================================================
# MANIFEST
# =========================================================

def save_manifest(
    records: list[SelectedImageRecord],
    manifest_path: Path,
    random_seed: int,
    requested_count: int,
    total_available: int,
) -> None:
    """
    Save the exact experimental sample to CSV.
    """
    if not records:
        raise ValueError(
            "At least one selected-image record is required."
        )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "selection_index",
        "destination_name",
        "relative_source_path",
        "source_size_bytes",
        "sha256",
        "random_seed",
        "requested_count",
        "selected_count",
        "total_available",
    ]

    with manifest_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "selection_index": (
                        record.selection_index
                    ),
                    "destination_name": (
                        record.destination_name
                    ),
                    "relative_source_path": (
                        record.relative_source_path
                        .as_posix()
                    ),
                    "source_size_bytes": (
                        record.source_size_bytes
                    ),
                    "sha256": record.sha256,
                    "random_seed": random_seed,
                    "requested_count": requested_count,
                    "selected_count": len(records),
                    "total_available": total_available,
                }
            )


# =========================================================
# MANIFEST VERIFICATION
# =========================================================

def verify_selected_files(
    selected_dir: Path,
    records: list[SelectedImageRecord],
) -> bool:
    """
    Verify that copied files match the source-file SHA-256 values.
    """
    for record in records:
        copied_path = (
            selected_dir
            / record.destination_name
        )

        if not copied_path.exists():
            return False

        if calculate_sha256(copied_path) != record.sha256:
            return False

    return True


# =========================================================
# MAIN WORKFLOW
# =========================================================

def create_reproducible_selection(
    raw_dir: Path = RAW_DIR,
    selected_dir: Path = SELECTED_DIR,
    manifest_path: Path = MANIFEST_PATH,
    number_of_images: int = NUM_IMAGES,
    random_seed: int = RANDOM_SEED,
    clear_existing: bool = CLEAR_SELECTED_DIR,
) -> list[SelectedImageRecord]:
    """
    Create a reproducible DICOM image subset for experiments.
    """
    validate_directories(
        raw_dir,
        selected_dir,
    )

    all_files = discover_dicom_files(
        raw_dir,
    )

    print(
        f"Total DICOM files found: "
        f"{len(all_files)}"
    )

    if not clear_existing and any(selected_dir.iterdir()):
        raise FileExistsError("STOP: Selected directory is not empty. Existing files are protected.")

    if clear_existing:
        clear_previous_selection(
            selected_dir,
            manifest_path,
        )

    selected_files = select_files_reproducibly(
        all_files,
        number_of_images,
        random_seed,
    )

    records = copy_selected_files(
        selected_files,
        raw_dir,
        selected_dir,
    )

    save_manifest(
        records=records,
        manifest_path=manifest_path,
        random_seed=random_seed,
        requested_count=number_of_images,
        total_available=len(all_files),
    )

    verified = verify_selected_files(
        selected_dir,
        records,
    )

    if not verified:
        raise RuntimeError(
            "Copied-file verification failed."
        )

    print(
        f"Selected and copied "
        f"{len(records)} DICOM images."
    )
    print(
        f"Random seed: {random_seed}"
    )
    print(
        f"Selected directory: "
        f"{selected_dir}"
    )
    print(
        f"Manifest: "
        f"{manifest_path}"
    )
    print(
        "SHA-256 verification: True"
    )

    return records


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    create_reproducible_selection()
