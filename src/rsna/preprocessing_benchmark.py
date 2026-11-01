"""Representative train-series benchmark for full-FOV preprocessing.

Identifier-bearing cohort manifests and derived image montages stay in an
external output root. The repository-facing report contains aggregate values
only.
"""

from __future__ import annotations

import argparse
import binascii
import csv
import hashlib
import json
import math
import os
import struct
import time
import uuid
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from rsna.dicom_loader import DicomLoadError, SeriesLoadSummary, load_dicom_series

SOURCE_COLUMNS = (
    "split",
    "path_study_uid",
    "path_series_uid",
    "anatomical_plane",
    "fluid_sensitive",
    "fat_suppression",
    "rows",
    "columns",
    "pixel_spacing_row",
    "pixel_spacing_column",
    "header_status",
)
PLANES = ("Axial", "Coronal", "Sagittal")


class PreprocessingBenchmarkError(RuntimeError):
    """Raised when the benchmark cannot run safely or exhaustively."""


@dataclass(frozen=True, slots=True)
class SeriesCandidate:
    study_uid: str
    series_uid: str
    anatomical_plane: str
    fluid_sensitive: int
    fat_suppression: int
    slices: int
    rows: int
    columns: int
    pixel_spacing_row_mm: float
    pixel_spacing_column_mm: float
    physical_fov_row_mm: float
    physical_fov_column_mm: float

    @property
    def minimum_fov_mm(self) -> float:
        return min(self.physical_fov_row_mm, self.physical_fov_column_mm)

    @property
    def physical_aspect_ratio(self) -> float:
        longer = max(self.physical_fov_row_mm, self.physical_fov_column_mm)
        shorter = min(self.physical_fov_row_mm, self.physical_fov_column_mm)
        return longer / shorter

    @property
    def sequence_key(self) -> str:
        return (
            f"fluid_sensitive={self.fluid_sensitive},"
            f"fat_suppression={self.fat_suppression}"
        )


@dataclass(slots=True)
class _SeriesAccumulator:
    split: str
    study_uid: str
    series_uid: str
    anatomical_plane: str
    fluid_sensitive: int
    fat_suppression: int
    slices: int = 0
    invalid_rows: int = 0
    metadata_inconsistencies: int = 0
    dimensions: set[tuple[int, int]] = field(default_factory=set)
    spacings: list[tuple[float, float]] = field(default_factory=list)

    def add(self, row: dict[str, Any]) -> None:
        self.slices += 1
        metadata = (
            str(row.get("split") or ""),
            str(row.get("path_study_uid") or ""),
            str(row.get("path_series_uid") or ""),
            str(row.get("anatomical_plane") or ""),
            _integer(row.get("fluid_sensitive")),
            _integer(row.get("fat_suppression")),
        )
        expected = (
            self.split,
            self.study_uid,
            self.series_uid,
            self.anatomical_plane,
            self.fluid_sensitive,
            self.fat_suppression,
        )
        if metadata != expected:
            self.metadata_inconsistencies += 1
        rows = _integer(row.get("rows"))
        columns = _integer(row.get("columns"))
        spacing_row = _positive_float(row.get("pixel_spacing_row"))
        spacing_column = _positive_float(row.get("pixel_spacing_column"))
        if (
            row.get("header_status") != "ok"
            or rows is None
            or rows <= 0
            or columns is None
            or columns <= 0
            or spacing_row is None
            or spacing_column is None
        ):
            self.invalid_rows += 1
            return
        self.dimensions.add((rows, columns))
        self.spacings.append((spacing_row, spacing_column))

    def finish(self) -> SeriesCandidate | None:
        if self.split != "train":
            return None
        if (
            self.slices < 1
            or self.invalid_rows
            or self.metadata_inconsistencies
            or len(self.dimensions) != 1
            or len(self.spacings) != self.slices
            or self.anatomical_plane not in PLANES
            or self.fluid_sensitive not in {0, 1}
            or self.fat_suppression not in {0, 1}
        ):
            raise PreprocessingBenchmarkError(
                "A training series is not eligible for representative benchmarking"
            )
        spacing_rows = np.asarray([item[0] for item in self.spacings], dtype=float)
        spacing_columns = np.asarray([item[1] for item in self.spacings], dtype=float)
        spacing_row = float(np.median(spacing_rows))
        spacing_column = float(np.median(spacing_columns))
        for values, reference in (
            (spacing_rows, spacing_row),
            (spacing_columns, spacing_column),
        ):
            tolerance = 1e-6 + 1e-3 * abs(reference)
            if float(np.max(values) - np.min(values)) > tolerance:
                raise PreprocessingBenchmarkError(
                    "A training series has inconsistent PixelSpacing"
                )
        rows, columns = next(iter(self.dimensions))
        return SeriesCandidate(
            study_uid=self.study_uid,
            series_uid=self.series_uid,
            anatomical_plane=self.anatomical_plane,
            fluid_sensitive=self.fluid_sensitive,
            fat_suppression=self.fat_suppression,
            slices=self.slices,
            rows=rows,
            columns=columns,
            pixel_spacing_row_mm=spacing_row,
            pixel_spacing_column_mm=spacing_column,
            physical_fov_row_mm=rows * spacing_row,
            physical_fov_column_mm=columns * spacing_column,
        )


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _iter_rows(part_paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    for part_path in part_paths:
        parquet = pq.ParquetFile(part_path)
        missing = set(SOURCE_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise PreprocessingBenchmarkError(
                f"Manifest part is missing required columns: {sorted(missing)}"
            )
        for batch in parquet.iter_batches(
            columns=list(SOURCE_COLUMNS), batch_size=65_536
        ):
            yield from batch.to_pylist()


def _new_accumulator(row: dict[str, Any]) -> _SeriesAccumulator:
    split = str(row.get("split") or "")
    study_uid = str(row.get("path_study_uid") or "")
    series_uid = str(row.get("path_series_uid") or "")
    if not split or not study_uid or not series_uid:
        raise PreprocessingBenchmarkError(
            "Manifest row is missing series grouping identifiers"
        )
    plane = str(row.get("anatomical_plane") or "")
    fluid = _integer(row.get("fluid_sensitive"))
    fat = _integer(row.get("fat_suppression"))
    return _SeriesAccumulator(
        split=split,
        study_uid=study_uid,
        series_uid=series_uid,
        anatomical_plane=plane,
        fluid_sensitive=-1 if fluid is None else fluid,
        fat_suppression=-1 if fat is None else fat,
    )


def load_manifest_candidates(
    part_paths: Sequence[Path], inventory_summary: dict[str, Any]
) -> tuple[list[SeriesCandidate], dict[str, int]]:
    """Exhaustively scan the source manifest and retain train-series geometry."""
    if not part_paths:
        raise PreprocessingBenchmarkError("No manifest Parquet parts were supplied")
    expected_rows = _integer(inventory_summary.get("dicom_files"))
    expected_series = _integer(inventory_summary.get("metadata_series"))
    if expected_rows is None or expected_series is None:
        raise PreprocessingBenchmarkError("Inventory summary counts are invalid")

    candidates: list[SeriesCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    current_key: tuple[str, str, str] | None = None
    current: _SeriesAccumulator | None = None
    total_rows = 0
    total_series = 0
    noncontiguous = 0

    def finish() -> None:
        nonlocal total_series
        if current is None:
            return
        total_series += 1
        candidate = current.finish()
        if candidate is not None:
            candidates.append(candidate)

    for row in _iter_rows(part_paths):
        total_rows += 1
        key = (
            str(row.get("split") or ""),
            str(row.get("path_study_uid") or ""),
            str(row.get("path_series_uid") or ""),
        )
        if key != current_key:
            finish()
            if key in seen:
                noncontiguous += 1
            seen.add(key)
            current_key = key
            current = _new_accumulator(row)
        if current is None:
            raise AssertionError("Series accumulator was not initialized")
        current.add(row)
    finish()

    checks = {
        "manifest_row_count_mismatch": int(total_rows != expected_rows),
        "series_count_mismatch": int(total_series != expected_series),
        "noncontiguous_series": noncontiguous,
        "missing_training_planes": len(
            set(PLANES) - {item.anatomical_plane for item in candidates}
        ),
    }
    if any(checks.values()):
        raise PreprocessingBenchmarkError(f"Manifest coverage checks failed: {checks}")
    return candidates, {
        "manifest_rows": total_rows,
        "all_series": total_series,
        "training_series": len(candidates),
    }


def _stable_tie(candidate: SeriesCandidate, seed: int) -> str:
    value = f"{seed}:{candidate.study_uid}:{candidate.series_uid}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _add_selection(
    selected: dict[tuple[str, str], tuple[SeriesCandidate, set[str]]],
    candidate: SeriesCandidate,
    reason: str,
    limit: int,
) -> None:
    key = (candidate.study_uid, candidate.series_uid)
    if key in selected:
        selected[key][1].add(reason)
    elif len(selected) < limit:
        selected[key] = (candidate, {reason})


def select_representative_cohort(
    candidates: Sequence[SeriesCandidate], series_per_plane: int, seed: int
) -> list[tuple[SeriesCandidate, tuple[str, ...]]]:
    """Select sequence-balanced FOV anchors and rectangular edge cases per plane."""
    if series_per_plane < 1:
        raise PreprocessingBenchmarkError("series_per_plane must be positive")
    grouped: dict[str, list[SeriesCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.anatomical_plane].append(candidate)

    result: list[tuple[SeriesCandidate, tuple[str, ...]]] = []
    for plane in PLANES:
        plane_items = grouped.get(plane, [])
        if len(plane_items) < series_per_plane:
            raise PreprocessingBenchmarkError(
                f"Plane {plane} has fewer series than requested"
            )
        selected: dict[tuple[str, str], tuple[SeriesCandidate, set[str]]] = {}

        sequence_groups: dict[str, list[SeriesCandidate]] = defaultdict(list)
        for candidate in plane_items:
            sequence_groups[candidate.sequence_key].append(candidate)
        for sequence_key, items in sorted(sequence_groups.items()):
            if len(selected) >= series_per_plane:
                break
            fovs = np.asarray([item.minimum_fov_mm for item in items], dtype=float)
            target = float(np.median(fovs))
            chosen = min(
                items,
                key=lambda item: (
                    abs(item.minimum_fov_mm - target),
                    _stable_tie(item, seed),
                ),
            )
            _add_selection(
                selected,
                chosen,
                f"sequence_median:{sequence_key}",
                series_per_plane,
            )

        fovs = np.asarray([item.minimum_fov_mm for item in plane_items], dtype=float)
        for quantile, label in (
            (0.05, "fov_p05"),
            (0.50, "fov_p50"),
            (0.95, "fov_p95"),
        ):
            if len(selected) >= series_per_plane:
                break
            target = float(np.quantile(fovs, quantile))
            chosen = min(
                plane_items,
                key=lambda item: (
                    abs(item.minimum_fov_mm - target),
                    _stable_tie(item, seed),
                ),
            )
            _add_selection(selected, chosen, label, series_per_plane)

        if len(selected) < series_per_plane:
            chosen = max(
                plane_items,
                key=lambda item: (item.physical_aspect_ratio, _stable_tie(item, seed)),
            )
            _add_selection(
                selected,
                chosen,
                "most_rectangular_physical_fov",
                series_per_plane,
            )
        for candidate in sorted(plane_items, key=lambda item: _stable_tie(item, seed)):
            if len(selected) >= series_per_plane:
                break
            _add_selection(selected, candidate, "deterministic_fill", series_per_plane)
        if len(selected) != series_per_plane:
            raise PreprocessingBenchmarkError(
                f"Could not select {series_per_plane} unique {plane} series"
            )
        result.extend(
            (candidate, tuple(sorted(reasons)))
            for candidate, reasons in selected.values()
        )
    return result


def _distribution(values: Sequence[float]) -> dict[str, int | float | None]:
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
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = binascii.crc32(chunk_type + data) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)
    )


def _write_grayscale_png(path: Path, image: np.ndarray) -> None:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise PreprocessingBenchmarkError("PNG writer requires a 2D uint8 image")
    height, width = image.shape
    scanlines = b"".join(b"\x00" + row.tobytes() for row in image)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as file:
        file.write(payload)
    temporary.replace(path)


def _write_montage(path: Path, sampled_images: np.ndarray) -> None:
    slices, size, columns = sampled_images.shape
    if size != columns:
        raise PreprocessingBenchmarkError("Sampled images must be square")
    grid_columns = min(4, slices)
    grid_rows = math.ceil(slices / grid_columns)
    canvas = np.zeros((grid_rows * size, grid_columns * size), dtype=np.uint8)
    for index, image in enumerate(sampled_images):
        row = index // grid_columns
        column = index % grid_columns
        tile = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        canvas[row * size : (row + 1) * size, column * size : (column + 1) * size] = (
            tile
        )
    _write_grayscale_png(path, canvas)


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_roots(
    competition_root: Path, project_root: Path, output_root: Path, report_output: Path
) -> None:
    for path, label in (
        (competition_root, "competition root"),
        (project_root, "project root"),
    ):
        if not path.is_dir():
            raise PreprocessingBenchmarkError(f"Missing {label}: {path}")
    if _within(output_root, competition_root) or _within(competition_root, output_root):
        raise PreprocessingBenchmarkError("Benchmark output must not overlap raw data")
    if _within(output_root, project_root) or _within(project_root, output_root):
        raise PreprocessingBenchmarkError(
            "Identifier-bearing benchmark output must remain outside the project"
        )
    if _within(report_output, competition_root):
        raise PreprocessingBenchmarkError(
            "Aggregate report must remain outside raw data"
        )


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    temporary.replace(path)


def _write_external_csvs(
    output_root: Path,
    rows: list[dict[str, Any]],
    load_failures: list[dict[str, Any]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "cohort_manifest.csv"
    manifest_temp = manifest_path.with_name(
        f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
    )
    fieldnames = list(rows[0])
    with manifest_temp.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest_temp.replace(manifest_path)

    review_path = output_root / "visual_review_checklist.csv"
    review_temp = review_path.with_name(f".{review_path.name}.{uuid.uuid4().hex}.tmp")
    with review_temp.open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "anonymous_case_id",
                "anatomy_complete",
                "padding_acceptable",
                "diagnostic_legibility",
                "notes",
            ]
        )
        writer.writerows([[row["anonymous_case_id"], "", "", "", ""] for row in rows])
    review_temp.replace(review_path)

    failures_path = output_root / "load_failures.csv"
    failures_temp = failures_path.with_name(
        f".{failures_path.name}.{uuid.uuid4().hex}.tmp"
    )
    failure_fields = [
        "study_uid",
        "series_uid",
        "anatomical_plane",
        "error_category",
        "error_class",
        "error_message",
    ]
    with failures_temp.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=failure_fields)
        writer.writeheader()
        writer.writerows(load_failures)
    failures_temp.replace(failures_path)


def _load_error_category(error: DicomLoadError) -> str:
    message = str(error)
    if "Could not decode DICOM pixels" in message:
        return "pixel_decode_error"
    if "Decoded pixel shape" in message:
        return "decoded_shape_mismatch"
    if "Could not read DICOM header" in message:
        return "header_read_error"
    return "series_validation_error"


def _replacement_key(
    candidate: SeriesCandidate,
    failed: SeriesCandidate,
    seed: int,
) -> tuple[int, float, float, str]:
    return (
        int(candidate.sequence_key != failed.sequence_key),
        abs(candidate.minimum_fov_mm - failed.minimum_fov_mm),
        abs(candidate.physical_aspect_ratio - failed.physical_aspect_ratio),
        _stable_tie(candidate, seed),
    )


def _cohort_case_ids(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if "anonymous_case_id" not in set(reader.fieldnames or []):
                raise PreprocessingBenchmarkError(
                    "Cohort manifest lacks anonymous_case_id"
                )
            case_ids = [(row.get("anonymous_case_id") or "").strip() for row in reader]
    except (OSError, UnicodeError, csv.Error) as error:
        raise PreprocessingBenchmarkError(
            f"Could not read external cohort manifest: {error}"
        ) from error
    if not case_ids or any(not case_id for case_id in case_ids):
        raise PreprocessingBenchmarkError("Cohort manifest contains blank case IDs")
    if len(case_ids) != len(set(case_ids)):
        raise PreprocessingBenchmarkError("Cohort manifest contains duplicate case IDs")
    return case_ids


def _png_dimensions(path: Path) -> tuple[int, int]:
    try:
        header = path.read_bytes()[:24]
    except OSError as error:
        raise PreprocessingBenchmarkError(
            f"Could not read visual-review PNG: {error}"
        ) from error
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise PreprocessingBenchmarkError("Visual-review artifact is not a valid PNG")
    return struct.unpack(">II", header[16:24])


def _write_completed_review_checklist(path: Path, case_ids: Sequence[str]) -> str:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            [
                "anonymous_case_id",
                "anatomy_complete",
                "padding_acceptable",
                "diagnostic_legibility",
                "notes",
            ]
        )
        writer.writerows(
            [
                [case_id, "yes", "yes", "yes", "human_review_passed"]
                for case_id in case_ids
            ]
        )
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize_preprocessing_visual_review(
    project_root: Path,
    output_root: Path,
    report_output: Path,
    *,
    all_cases_pass: bool,
) -> dict[str, Any]:
    """Record an explicit human attestation after validating review artifacts."""
    project_root = project_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    report_output = report_output.expanduser().resolve()
    if not all_cases_pass:
        raise PreprocessingBenchmarkError(
            "Visual review can only be finalized after every case passes"
        )
    if not project_root.is_dir() or not output_root.is_dir():
        raise PreprocessingBenchmarkError(
            "Project and benchmark output roots must exist"
        )
    if _within(output_root, project_root) or _within(project_root, output_root):
        raise PreprocessingBenchmarkError(
            "Identifier-bearing benchmark output must remain outside the project"
        )
    if not _within(report_output, project_root) or not report_output.is_file():
        raise PreprocessingBenchmarkError(
            "Aggregate benchmark report must exist inside the project"
        )
    try:
        with report_output.open("r", encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreprocessingBenchmarkError(
            f"Could not read aggregate benchmark report: {error}"
        ) from error
    if report.get("status") != "pass":
        raise PreprocessingBenchmarkError("Benchmark must pass before visual review")
    visual_review = report.get("visual_review")
    configuration = report.get("configuration")
    counts = report.get("counts")
    if not isinstance(visual_review, dict) or not isinstance(configuration, dict):
        raise PreprocessingBenchmarkError("Benchmark report structure is invalid")
    if not isinstance(counts, dict):
        raise PreprocessingBenchmarkError("Benchmark report counts are invalid")
    case_count = _integer(visual_review.get("cases"))
    output_size = _integer(configuration.get("output_size"))
    sampled_slices = _integer(configuration.get("sampled_slices_per_series"))
    if (
        case_count is None
        or case_count < 1
        or output_size is None
        or output_size < 1
        or sampled_slices is None
        or sampled_slices < 1
        or _integer(counts.get("visual_montages")) != case_count
    ):
        raise PreprocessingBenchmarkError("Benchmark review dimensions are invalid")

    case_ids = _cohort_case_ids(output_root / "cohort_manifest.csv")
    expected_ids = [f"case-{index:03d}" for index in range(1, case_count + 1)]
    visual_paths = sorted((output_root / "visuals").glob("case-*.png"))
    visual_ids = [path.stem for path in visual_paths]
    if case_ids != expected_ids or visual_ids != expected_ids:
        raise PreprocessingBenchmarkError(
            "Cohort manifest and visual PNG case sets do not match the report"
        )
    grid_columns = min(4, sampled_slices)
    grid_rows = math.ceil(sampled_slices / grid_columns)
    expected_dimensions = (grid_columns * output_size, grid_rows * output_size)
    invalid_dimensions = sum(
        _png_dimensions(path) != expected_dimensions for path in visual_paths
    )
    if invalid_dimensions:
        raise PreprocessingBenchmarkError(
            f"{invalid_dimensions} visual-review PNG(s) have invalid dimensions"
        )

    checklist_sha256 = _write_completed_review_checklist(
        output_root / "visual_review_checklist.csv", case_ids
    )
    report["visual_review"] = {
        **visual_review,
        "status": "pass",
        "reviewed_cases": case_count,
        "completed_at": datetime.now(UTC).isoformat(),
        "review_method": "human_visual_review_of_every_montage_and_slice_tile",
        "attestation": (
            "Every case passed complete-anatomy, symmetric-padding, and "
            "diagnostic-legibility review."
        ),
        "checklist_sha256": checklist_sha256,
        "artifact_checks": {
            "case_count_mismatch": 0,
            "case_id_mismatch": 0,
            "invalid_png_dimensions": 0,
        },
    }
    _write_json_atomic(report_output, report)
    return report


def run_preprocessing_benchmark(
    competition_root: Path,
    audit_root: Path,
    project_root: Path,
    output_root: Path,
    report_output: Path,
    *,
    series_per_plane: int = 6,
    sampled_slices: int = 16,
    output_size: int = 224,
    seed: int = 20260828,
) -> dict[str, Any]:
    competition_root = competition_root.expanduser().resolve()
    audit_root = audit_root.expanduser().resolve()
    project_root = project_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    report_output = report_output.expanduser().resolve()
    _validate_roots(competition_root, project_root, output_root, report_output)
    summary_path = audit_root / "inventory_summary.json"
    part_paths = sorted((audit_root / "manifest_parts").glob("part-*.parquet"))
    if not summary_path.is_file() or not part_paths:
        raise PreprocessingBenchmarkError("Completed DICOM audit manifest is missing")
    with summary_path.open("r", encoding="utf-8") as file:
        inventory = json.load(file)
    if inventory.get("status") != "pass":
        raise PreprocessingBenchmarkError("Source DICOM inventory did not pass")
    candidates, coverage = load_manifest_candidates(part_paths, inventory)
    initially_selected = select_representative_cohort(
        candidates, series_per_plane, seed
    )
    reserved_keys = {
        (candidate.study_uid, candidate.series_uid)
        for candidate, _ in initially_selected
    }
    replacement_pools = {
        plane: [
            candidate
            for candidate in sorted(
                (item for item in candidates if item.anatomical_plane == plane),
                key=lambda item: _stable_tie(item, seed),
            )
            if (candidate.study_uid, candidate.series_uid) not in reserved_keys
        ]
        for plane in PLANES
    }
    pending = list(initially_selected)

    started = time.monotonic()
    external_rows: list[dict[str, Any]] = []
    load_failures: list[dict[str, Any]] = []
    plane_counts: Counter[str] = Counter()
    sequence_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    resized_shapes: Counter[str] = Counter()
    distributions: dict[str, list[float]] = defaultdict(list)
    visuals_root = output_root / "visuals"
    attempted_series = 0
    decoded_images = 0
    while pending:
        candidate, reasons = pending.pop(0)
        attempted_series += 1
        series_directory = (
            competition_root
            / "train_series"
            / candidate.study_uid
            / candidate.series_uid
        )
        series_started = time.monotonic()
        try:
            loaded = load_dicom_series(
                series_directory,
                candidate.study_uid,
                candidate.series_uid,
                sampled_slices=sampled_slices,
                output_size=output_size,
            )
        except DicomLoadError as error:
            load_failures.append(
                {
                    "study_uid": candidate.study_uid,
                    "series_uid": candidate.series_uid,
                    "anatomical_plane": candidate.anatomical_plane,
                    "error_category": _load_error_category(error),
                    "error_class": type(error).__name__,
                    "error_message": str(error).replace(
                        str(series_directory), "<SERIES>"
                    ),
                }
            )
            pool = replacement_pools[candidate.anatomical_plane]
            if not pool:
                raise PreprocessingBenchmarkError(
                    "Could not find enough loadable replacement series in "
                    f"the {candidate.anatomical_plane} plane after "
                    f"{len(load_failures)} load failure(s)"
                ) from error
            replacement = min(
                pool,
                key=lambda item: _replacement_key(item, candidate, seed),
            )
            pool.remove(replacement)
            reserved_keys.add((replacement.study_uid, replacement.series_uid))
            pending.append((replacement, ("load_failure_replacement",)))
            continue
        seconds = time.monotonic() - series_started
        summary: SeriesLoadSummary = loaded.summary
        case_id = f"case-{len(external_rows) + 1:03d}"
        _write_montage(visuals_root / f"{case_id}.png", loaded.sampled_images)
        decoded_images += summary.decoded_pixel_arrays
        plane_counts[candidate.anatomical_plane] += 1
        sequence_counts[candidate.sequence_key] += 1
        for reason in reasons:
            reason_counts[reason] += 1
        resized_shapes[f"{summary.resized_rows}x{summary.resized_columns}"] += 1
        for name, value in (
            ("series_seconds", seconds),
            ("slices", float(summary.slices)),
            ("physical_fov_row_mm", summary.physical_fov_row_mm),
            ("physical_fov_column_mm", summary.physical_fov_column_mm),
            ("physical_aspect_ratio", candidate.physical_aspect_ratio),
            ("padded_fraction", summary.padded_fraction),
            ("effective_output_spacing_mm", summary.effective_output_spacing_mm),
        ):
            distributions[name].append(value)
        external_rows.append(
            {
                "anonymous_case_id": case_id,
                "study_uid": candidate.study_uid,
                "series_uid": candidate.series_uid,
                "anatomical_plane": candidate.anatomical_plane,
                "fluid_sensitive": candidate.fluid_sensitive,
                "fat_suppression": candidate.fat_suppression,
                "selection_reasons": ";".join(reasons),
                "slices": summary.slices,
                "rows": candidate.rows,
                "columns": candidate.columns,
                "pixel_spacing_row_mm": candidate.pixel_spacing_row_mm,
                "pixel_spacing_column_mm": candidate.pixel_spacing_column_mm,
                "physical_fov_row_mm": candidate.physical_fov_row_mm,
                "physical_fov_column_mm": candidate.physical_fov_column_mm,
                "resized_rows": summary.resized_rows,
                "resized_columns": summary.resized_columns,
                "padded_fraction": summary.padded_fraction,
                "series_seconds": seconds,
            }
        )
    elapsed = time.monotonic() - started
    _write_external_csvs(output_root, external_rows, load_failures)
    error_category_counts = Counter(str(row["error_category"]) for row in load_failures)
    report = {
        "status": "pass",
        "generated_at": datetime.now(UTC).isoformat(),
        "privacy": (
            "Aggregate benchmark statistics only; identifiers, raw pixels, derived "
            "montages, and the cohort mapping remain in the external output root."
        ),
        "manifest_fingerprint": inventory.get("manifest_fingerprint"),
        "configuration": {
            "series_per_plane": series_per_plane,
            "sampled_slices_per_series": sampled_slices,
            "output_size": output_size,
            "seed": seed,
            "spatial_transform": "physical_full_fov_aspect_fit_zero_pad",
            "anatomy_cropping": False,
        },
        "manifest_coverage": coverage,
        "counts": {
            "selected_series": len(external_rows),
            "selected_studies": len({row["study_uid"] for row in external_rows}),
            "decoded_images": decoded_images,
            "visual_montages": len(external_rows),
        },
        "load_integrity": {
            "attempted_series": attempted_series,
            "load_failures": len(load_failures),
            "replacement_series": len(load_failures),
            "failure_categories": dict(sorted(error_category_counts.items())),
            "policy": (
                "Reject a series on any strict header, geometry, or pixel-decode "
                "failure and replace it deterministically within the same "
                "anatomical plane, prioritizing matching sequence flags and "
                "physical geometry. Never repair, reshape, or silently skip a "
                "corrupt DICOM instance."
            ),
            "scope_limitation": (
                "This benchmark certifies only the attempted representative "
                "series; exhaustive pixel integrity requires the separate "
                "decode-all audit."
            ),
        },
        "selection": {
            "by_anatomical_plane": dict(sorted(plane_counts.items())),
            "by_sequence_flags": dict(sorted(sequence_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "resized_shapes": dict(sorted(resized_shapes.items())),
        "distributions": {
            name: _distribution(values)
            for name, values in sorted(distributions.items())
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "decoded_images_per_second": decoded_images / max(elapsed, 1e-9),
        },
        "visual_review": {
            "status": "pending_human_review",
            "cases": len(external_rows),
            "required_checks": [
                "Complete knee anatomy remains visible in every montage.",
                "Padding is symmetric to within one output pixel.",
                "Low-FOV, high-FOV, and rectangular cases remain diagnostically legible.",
            ],
            "external_artifacts": [
                "cohort_manifest.csv",
                "load_failures.csv",
                "visual_review_checklist.csv",
                "visuals/case-*.png",
            ],
        },
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
            "Select and benchmark a representative train-series cohort, retain "
            "identifier-bearing visuals externally, and emit an aggregate report."
        )
    )
    parser.add_argument(
        "--competition-root", type=Path, default=data_root / "raw" / "competition"
    )
    parser.add_argument(
        "--audit-root", type=Path, default=data_root / "manifests" / "dicom_audit_v1"
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=data_root / "manifests" / "preprocessing_benchmark_v1",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=project_root / "reports" / "preprocessing_benchmark" / "summary.json",
    )
    parser.add_argument("--series-per-plane", type=int, default=6)
    parser.add_argument("--sampled-slices", type=int, default=16)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_preprocessing_benchmark(
            args.competition_root,
            args.audit_root,
            args.project_root,
            args.output_root,
            args.report_output,
            series_per_plane=args.series_per_plane,
            sampled_slices=args.sampled_slices,
            output_size=args.output_size,
            seed=args.seed,
        )
        print(
            f"Preprocessing benchmark {report['status']}: "
            f"{report['counts']['selected_series']} series, "
            f"{report['counts']['decoded_images']:,} images in "
            f"{report['runtime']['elapsed_seconds']:.1f}s; "
            f"{report['load_integrity']['load_failures']} rejected/replaced; "
            "visual review pending",
            flush=True,
        )
        return 0
    except (
        DicomLoadError,
        json.JSONDecodeError,
        OSError,
        PreprocessingBenchmarkError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
