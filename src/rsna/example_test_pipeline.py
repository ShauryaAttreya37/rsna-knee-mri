"""End-to-end offline loader smoke test for the example test studies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rsna.dicom_loader import DicomLoadError, SeriesLoadSummary, load_dicom_series
from rsna.submission import (
    ID_COLUMN,
    SubmissionValidationError,
    generate_constant_submission,
)

SERIES_COLUMNS = (
    ID_COLUMN,
    "SeriesInstanceUID",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
)
PLANES = {"Axial", "Coronal", "Sagittal"}


class ExamplePipelineError(RuntimeError):
    """Raised when the example-test inference path is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class SeriesDescriptor:
    study_uid: str
    series_uid: str
    fluid_sensitive: int
    fat_suppression: int
    anatomical_plane: str


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            header = next(reader)
            rows = list(reader)
    except StopIteration as error:
        raise ExamplePipelineError(f"CSV is empty: {path}") from error
    except (OSError, UnicodeError, csv.Error) as error:
        raise ExamplePipelineError(f"Could not read CSV {path}: {error}") from error
    for row_number, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise ExamplePipelineError(
                f"CSV row {row_number} has {len(row)} fields; expected {len(header)}"
            )
    return header, rows


def load_test_metadata(
    competition_root: Path,
) -> tuple[list[str], dict[str, list[SeriesDescriptor]]]:
    test_header, test_rows = _read_csv(competition_root / "test.csv")
    if test_header != [ID_COLUMN]:
        raise ExamplePipelineError("test.csv schema is invalid")
    test_ids = [row[0] for row in test_rows]
    if not test_ids or any(not identifier for identifier in test_ids):
        raise ExamplePipelineError("test.csv contains blank or no study identifiers")
    if len(set(test_ids)) != len(test_ids):
        raise ExamplePipelineError("test.csv contains duplicate study identifiers")

    series_header, series_rows = _read_csv(competition_root / "test_series.csv")
    if tuple(series_header) != SERIES_COLUMNS:
        raise ExamplePipelineError("test_series.csv schema is invalid")
    grouped = {identifier: [] for identifier in test_ids}
    seen_series: set[str] = set()
    for row_number, row in enumerate(series_rows, start=2):
        study_uid, series_uid, fluid, fat_suppression, plane = row
        if study_uid not in grouped:
            raise ExamplePipelineError(
                f"test_series.csv row {row_number} references an unknown study"
            )
        if not series_uid or series_uid in seen_series:
            raise ExamplePipelineError(
                f"test_series.csv row {row_number} has a blank or duplicate series"
            )
        seen_series.add(series_uid)
        try:
            fluid_flag = int(fluid)
            fat_flag = int(fat_suppression)
        except ValueError as error:
            raise ExamplePipelineError(
                f"test_series.csv row {row_number} has invalid sequence flags"
            ) from error
        if fluid_flag not in (0, 1) or fat_flag not in (0, 1):
            raise ExamplePipelineError(
                f"test_series.csv row {row_number} has nonbinary sequence flags"
            )
        if plane not in PLANES:
            raise ExamplePipelineError(
                f"test_series.csv row {row_number} has an invalid anatomical plane"
            )
        grouped[study_uid].append(
            SeriesDescriptor(
                study_uid=study_uid,
                series_uid=series_uid,
                fluid_sensitive=fluid_flag,
                fat_suppression=fat_flag,
                anatomical_plane=plane,
            )
        )
    if any(not descriptors for descriptors in grouped.values()):
        raise ExamplePipelineError("At least one test study has no series metadata")
    return test_ids, grouped


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, sort_keys=True)
            file.write("\n")
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _record_summary(
    summary: SeriesLoadSummary,
    plane: str,
    counters: dict[str, Counter[str]],
    distributions: dict[str, list[float]],
) -> None:
    counters["planes"][plane] += 1
    counters["sort_strategies"][summary.sort_strategy] += 1
    counters["transfer_syntaxes"][summary.transfer_syntax_uid] += summary.slices
    counters["photometric_interpretations"][summary.photometric_interpretation] += (
        summary.slices
    )
    counters["dimensions"][f"{summary.rows}x{summary.columns}"] += summary.slices
    counters["fit_and_pad_resized_shapes"][
        f"{summary.resized_rows}x{summary.resized_columns}"
    ] += 1
    distributions["slices_per_series"].append(float(summary.slices))
    distributions["max_normal_angle_degrees"].append(summary.max_normal_angle_degrees)
    distributions["normalization_low"].append(summary.normalization_low)
    distributions["normalization_high"].append(summary.normalization_high)
    distributions["normalized_mean"].append(summary.normalized_mean)
    distributions["pixel_spacing_row_mm"].append(summary.pixel_spacing_row_mm)
    distributions["pixel_spacing_column_mm"].append(summary.pixel_spacing_column_mm)
    distributions["physical_fov_row_mm"].append(summary.physical_fov_row_mm)
    distributions["physical_fov_column_mm"].append(summary.physical_fov_column_mm)
    distributions["effective_output_spacing_mm"].append(
        summary.effective_output_spacing_mm
    )
    distributions["padded_fraction"].append(summary.padded_fraction)


def run_example_pipeline(
    competition_root: Path,
    output_csv: Path,
    report_output: Path,
    sampled_slices: int = 16,
    output_size: int = 224,
    max_normal_angle_degrees: float = 1.0,
) -> dict[str, Any]:
    competition_root = competition_root.expanduser().resolve()
    output_csv = output_csv.expanduser().resolve()
    report_output = report_output.expanduser().resolve()
    if not competition_root.is_dir():
        raise ExamplePipelineError(
            f"Competition root does not exist: {competition_root}"
        )
    for output_path in (output_csv, report_output):
        if _within(output_path, competition_root):
            raise ExamplePipelineError(
                "Generated outputs must remain outside the raw competition tree"
            )
    test_series_root = competition_root / "test_series"
    if not test_series_root.is_dir():
        raise ExamplePipelineError(
            f"Test-series directory is missing: {test_series_root}"
        )

    test_ids, grouped = load_test_metadata(competition_root)
    expected_directories = {
        (descriptor.study_uid, descriptor.series_uid)
        for descriptors in grouped.values()
        for descriptor in descriptors
    }
    actual_directories = {
        (study_directory.name, series_directory.name)
        for study_directory in test_series_root.iterdir()
        if study_directory.is_dir()
        for series_directory in study_directory.iterdir()
        if series_directory.is_dir()
    }
    if actual_directories != expected_directories:
        raise ExamplePipelineError(
            "Test-series directories do not exactly match test_series.csv"
        )

    counters = {
        "planes": Counter(),
        "sort_strategies": Counter(),
        "transfer_syntaxes": Counter(),
        "photometric_interpretations": Counter(),
        "dimensions": Counter(),
        "fit_and_pad_resized_shapes": Counter(),
    }
    distributions: dict[str, list[float]] = {
        "slices_per_series": [],
        "series_per_study": [],
        "max_normal_angle_degrees": [],
        "normalization_low": [],
        "normalization_high": [],
        "normalized_mean": [],
        "pixel_spacing_row_mm": [],
        "pixel_spacing_column_mm": [],
        "physical_fov_row_mm": [],
        "physical_fov_column_mm": [],
        "effective_output_spacing_mm": [],
        "padded_fraction": [],
    }
    started = time.monotonic()
    decoded_images = 0
    sampled_tensors = 0
    for study_index, study_uid in enumerate(test_ids, start=1):
        descriptors = grouped[study_uid]
        distributions["series_per_study"].append(float(len(descriptors)))
        for descriptor in descriptors:
            series_directory = (
                test_series_root / descriptor.study_uid / descriptor.series_uid
            )
            loaded = load_dicom_series(
                series_directory,
                descriptor.study_uid,
                descriptor.series_uid,
                sampled_slices=sampled_slices,
                output_size=output_size,
                max_normal_angle_degrees=max_normal_angle_degrees,
            )
            if loaded.sampled_images.shape != (
                sampled_slices,
                output_size,
                output_size,
            ):
                raise ExamplePipelineError("Series produced an invalid sampled tensor")
            decoded_images += loaded.summary.decoded_pixel_arrays
            sampled_tensors += 1
            _record_summary(
                loaded.summary,
                descriptor.anatomical_plane,
                counters,
                distributions,
            )
        print(
            f"Loaded test study {study_index}/{len(test_ids)}: "
            f"{len(descriptors)} series",
            flush=True,
        )
    elapsed = time.monotonic() - started
    if decoded_images <= 0 or not math.isfinite(elapsed):
        raise ExamplePipelineError("No test pixels were decoded")
    submission_validation = generate_constant_submission(
        competition_root / "test.csv",
        competition_root / "sample_submission.csv",
        output_csv,
        probability=0.5,
    )
    report = {
        "status": "pass",
        "generated_at": datetime.now(UTC).isoformat(),
        "offline": True,
        "privacy": (
            "Identifier-free aggregate loader statistics only; no study, series, "
            "SOP identifiers, pixel arrays, or report text are included."
        ),
        "configuration": {
            "sampled_slices_per_series": sampled_slices,
            "output_size": output_size,
            "spatial_transform": "physical_full_fov_aspect_fit_zero_pad",
            "anatomy_cropping": False,
            "normalization_percentiles": [1.0, 99.0],
            "max_normal_angle_degrees": max_normal_angle_degrees,
        },
        "counts": {
            "test_studies": len(test_ids),
            "series": sum(len(items) for items in grouped.values()),
            "decoded_pixel_arrays": decoded_images,
            "sampled_series_tensors": sampled_tensors,
        },
        "categorical_counts": {
            name: dict(sorted(counter.items())) for name, counter in counters.items()
        },
        "distributions": {
            name: _distribution(values) for name, values in distributions.items()
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "decoded_images_per_second": decoded_images / max(elapsed, 1e-9),
            "scope_warning": (
                "The three-study example is too small for a hidden-test runtime "
                "projection; benchmark a representative train-derived cohort next. "
                "The primary transform retains the complete physical field of view; "
                "the optional 140 mm detail view is not part of this benchmark."
            ),
        },
        "submission_validation": submission_validation.to_dict(),
    }
    _write_json_atomic(report_output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data"))
    project_root = Path(
        os.environ.get("RSNA_PROJECT_ROOT", "/home/shaurya/rsna-project")
    )
    parser = argparse.ArgumentParser(
        description=(
            "Decode and preprocess every example-test DICOM, then produce a "
            "validated placeholder submission and identifier-free report."
        )
    )
    parser.add_argument(
        "--competition-root",
        type=Path,
        default=data_root / "raw" / "competition",
    )
    parser.add_argument("--output", type=Path, default=project_root / "submission.csv")
    parser.add_argument(
        "--report-output",
        type=Path,
        default=project_root / "reports" / "example_test_pipeline" / "summary.json",
    )
    parser.add_argument("--sampled-slices", type=int, default=16)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--max-normal-angle-degrees", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_example_pipeline(
            args.competition_root,
            args.output,
            args.report_output,
            args.sampled_slices,
            args.output_size,
            args.max_normal_angle_degrees,
        )
        print(
            f"Example-test pipeline {report['status']}: "
            f"{report['counts']['decoded_pixel_arrays']:,} images in "
            f"{report['runtime']['elapsed_seconds']:.1f}s; "
            f"report={args.report_output.expanduser().resolve()}",
            flush=True,
        )
        return 0
    except (
        DicomLoadError,
        ExamplePipelineError,
        OSError,
        SubmissionValidationError,
    ) as error:
        print(f"ERROR: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
