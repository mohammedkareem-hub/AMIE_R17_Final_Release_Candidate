#!/usr/bin/env python3
"""Private-data-safe launcher for the ORIGINAL AMIE 100-MR-image evaluation.

Default is validation-only. --run explicitly starts the original experiment.
This wrapper never downloads or publishes medical data, and does not alter
any of the cryptographic or measurement routines in the original evaluator.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def contained_in(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AMIE-BakerChen: validate an authorized private 100-image MR dataset or rerun its original evaluation"
    )
    parser.add_argument(
        "--selected-dir", type=Path, required=True,
        help="Private directory containing 100 image_###.dcm files and selected_images.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="New, nonexisting private output directory (mandatory with --run)",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Actually execute the original 100-image evaluation. Without this flag: read-only dataset validation.",
    )
    args = parser.parse_args()

    selected = args.selected_dir.expanduser().resolve()
    if not selected.is_dir():
        parser.error(f"Selected dataset directory not found: {selected}")
    if contained_in(selected, ROOT):
        parser.error("STOP: Medical DICOM files and their manifest must be OUTSIDE the candidate repository")
    if not (selected/'selected_images.csv').is_file():
        parser.error("STOP: selected_images.csv missing from selected data directory")

    output = args.output_dir.expanduser().resolve() if args.output_dir else None
    if args.run and output is None:
        parser.error("--output-dir is required for --run")
    if output is not None:
        if contained_in(output, ROOT):
            parser.error("STOP: Output directory must be OUTSIDE the candidate repository")
        if contained_in(output, selected) or contained_in(selected, output):
            parser.error("STOP: Output directory cannot overlap the selected DICOM directory")
        if args.run and output.exists():
            parser.error(f"STOP: Output directory already exists; refusing overwrite: {output}")

    # Execute the supplied evaluator with its ORIGINAL source unchanged.
    # Only its input/output path variables are configured before calling it.
    sys.path.insert(0, str(ROOT))
    from src import final_security_evaluation_100_images as ev
    ev.SELECTED_DIR = selected
    ev.MANIFEST_PATH = selected/'selected_images.csv'
    if output is not None:
        ev.OUTPUT_DIR = output

    images = ev.verify_frozen_selected_dataset()
    print(f"PASS: private manifest/size/SHA-256 verification: {len(images)}/{ev.EXPECTED_IMAGES}")
    if not args.run:
        print("READ-ONLY PREFLIGHT COMPLETE: no encryption or output files were created.")
        return 0

    # Check again immediately prior to executing the original writer.
    if output is None or output.exists():
        parser.error("STOP: Output destination must be a fresh nonexistent directory")
    print("RUNNING: original 100-image security evaluation; private outputs only.", flush=True)
    ev.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
