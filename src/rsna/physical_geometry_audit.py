"""Identifier-free audit of DICOM pixel spacing and physical field of view.

This module consumes only the Parquet manifest produced by ``dicom_audit.py``.
It never opens a DICOM. Results are aggregated at series level so long series
do not dominate the distributions, and no study, series, SOP, or file
identifier is retained in the emitted report.
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

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
DEFAULT_CANDIDATE_CROPS_MM = (120.0, 140.0, 160.0, 180.0, 200.0)
ROW_FINDING_KEYS = (
    "non_ok_header_rows",
    "metadata_inconsistent_rows",
    "missing_spacing_rows",
    "missing_spacing_row_component_rows",
    "missing_spacing_column_component_rows",
    "invalid_spacing_rows",
    "invalid_spacing_row_component_rows",
    "invalid_spacing_column_component_rows",
    "missing_dimension_rows",
    "invalid_dimension_rows",
    "valid_physical_geometry_rows",
)
SERIES_FINDING_KEYS = (
    "series_analyzable",
    "series_excluded_from_physical_analysis",
    "series_with_non_ok_headers",
    "series_with_metadata_inconsistency",
    "series_with_invalid_group_metadata",
    "series_with_missing_spacing",
    "series_with_invalid_spacing",
    "series_with_inconsistent_row_spacing",
    "series_with_inconsistent_column_spacing",
    "series_with_inconsistent_spacing",
    "series_with_missing_dimensions",
    "series_with_invalid_dimensions",
    "series_with_inconsistent_dimensions",
    "series_with_nonfinite_derived_geometry",
)


class PhysicalGeometryAuditError(RuntimeError):
    """Raised when a manifest cannot be analyzed safely."""


@dataclass(frozen=True, slots=True)
class SeriesMeasure:
    """One identifier-free, series-weighted physical-geometry observation."""

    split: str
    anatomical_plane: str
    fluid_sensitive: int
    fat_suppression: int
    pixel_spacing_row_mm: float
    pixel_spacing_column_mm: float
    physical_fov_row_mm: float
    physical_fov_column_mm: float

    @property
    def minimum_physical_fov_mm(self) -> float:
        return min(self.physical_fov_row_mm, self.physical_fov_column_mm)


@dataclass(slots=True)
class SeriesAccumulator:
    """Bounded state for one contiguous series in the sorted manifest."""

    split: str
    anatomical_plane: str
    fluid_sensitive: int
    fat_suppression: int
    rows_seen: int = 0
    non_ok_header_rows: int = 0
    metadata_inconsistencies: int = 0
    missing_spacing_rows: int = 0
    missing_spacing_row_component_rows: int = 0
    missing_spacing_column_component_rows: int = 0
    invalid_spacing_rows: int = 0
    invalid_spacing_row_component_rows: int = 0
    invalid_spacing_column_component_rows: int = 0
    missing_dimension_rows: int = 0
    invalid_dimension_rows: int = 0
    spacings: list[tuple[float, float]] = field(default_factory=list)
    dimensions: list[tuple[int, int]] = field(default_factory=list)
    valid_physical_rows: int = 0

    def add(self, row: dict[str, Any]) -> None:
        self.rows_seen += 1
        metadata = (
            str(row.get("split") or "<missing>"),
            str(row.get("anatomical_plane") or "<missing>"),
            _integer(row.get("fluid_sensitive"), default=-1),
            _integer(row.get("fat_suppression"), default=-1),
        )
        expected = (
            self.split,
            self.anatomical_plane,
            self.fluid_sensitive,
            self.fat_suppression,
        )
        if metadata != expected:
            self.metadata_inconsistencies += 1
        if row.get("header_status") != "ok":
            self.non_ok_header_rows += 1

        spacing_row = _finite_float(row.get("pixel_spacing_row"))
        spacing_column = _finite_float(row.get("pixel_spacing_column"))
        raw_spacing_row = row.get("pixel_spacing_row")
        raw_spacing_column = row.get("pixel_spacing_column")
        spacing_missing = raw_spacing_row is None or raw_spacing_column is None
        if raw_spacing_row is None:
            self.missing_spacing_row_component_rows += 1
        if raw_spacing_column is None:
            self.missing_spacing_column_component_rows += 1
        if spacing_missing:
            self.missing_spacing_rows += 1
        else:
            row_invalid = spacing_row is None or spacing_row <= 0
            column_invalid = spacing_column is None or spacing_column <= 0
            if row_invalid:
                self.invalid_spacing_row_component_rows += 1
            if column_invalid:
                self.invalid_spacing_column_component_rows += 1
            if row_invalid or column_invalid:
                self.invalid_spacing_rows += 1
            else:
                self.spacings.append((spacing_row, spacing_column))

        raw_rows = row.get("rows")
        raw_columns = row.get("columns")
        rows = _integer(raw_rows, default=None)
        columns = _integer(raw_columns, default=None)
        dimensions_missing = raw_rows is None or raw_columns is None
        dimensions_invalid = (
            dimensions_missing
            or rows is None
            or columns is None
            or rows <= 0
            or columns <= 0
        )
        if dimensions_missing:
            self.missing_dimension_rows += 1
        elif dimensions_invalid:
            self.invalid_dimension_rows += 1
        else:
            self.dimensions.append((rows, columns))

        if (
            not spacing_missing
            and not dimensions_invalid
            and spacing_row is not None
            and spacing_column is not None
            and spacing_row > 0
            and spacing_column > 0
        ):
            self.valid_physical_rows += 1


def _integer(value: Any, default: int | None) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _distribution(values: Sequence[float]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise PhysicalGeometryAuditError("A nonfinite value reached a distribution")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _consistent(
    values: Sequence[float], relative_tolerance: float, absolute_tolerance: float
) -> bool:
    if len(values) < 2:
        return True
    array = np.asarray(values, dtype=float)
    reference = float(np.median(array))
    tolerance = absolute_tolerance + relative_tolerance * abs(reference)
    return float(np.max(array) - np.min(array)) <= tolerance


def _sequence_key(measure: SeriesMeasure | SeriesAccumulator) -> str:
    return (
        f"fluid_sensitive={measure.fluid_sensitive},"
        f"fat_suppression={measure.fat_suppression}"
    )


def _combined_key(measure: SeriesMeasure | SeriesAccumulator) -> str:
    return f"plane={measure.anatomical_plane},{_sequence_key(measure)}"


def _new_series(row: dict[str, Any]) -> SeriesAccumulator:
    return SeriesAccumulator(
        split=str(row.get("split") or "<missing>"),
        anatomical_plane=str(row.get("anatomical_plane") or "<missing>"),
        fluid_sensitive=int(_integer(row.get("fluid_sensitive"), default=-1)),
        fat_suppression=int(_integer(row.get("fat_suppression"), default=-1)),
    )


def _iter_rows(part_paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    for part_path in part_paths:
        parquet = pq.ParquetFile(part_path)
        available = set(parquet.schema_arrow.names)
        missing = set(SOURCE_COLUMNS) - available
        if missing:
            raise PhysicalGeometryAuditError(
                f"Manifest part is missing required columns: {sorted(missing)}"
            )
        for batch in parquet.iter_batches(
            columns=list(SOURCE_COLUMNS), batch_size=65_536
        ):
            yield from batch.to_pylist()


def _group_summary(
    measures: Sequence[SeriesMeasure], total_series: int
) -> dict[str, Any]:
    return {
        "total_series": total_series,
        "analyzable_series": len(measures),
        "excluded_series": total_series - len(measures),
        "pixel_spacing_mm": {
            "row": _distribution(
                [measure.pixel_spacing_row_mm for measure in measures]
            ),
            "column": _distribution(
                [measure.pixel_spacing_column_mm for measure in measures]
            ),
        },
        "physical_fov_mm": {
            "row": _distribution([measure.physical_fov_row_mm for measure in measures]),
            "column": _distribution(
                [measure.physical_fov_column_mm for measure in measures]
            ),
            "minimum_in_plane_dimension": _distribution(
                [measure.minimum_physical_fov_mm for measure in measures]
            ),
        },
    }


def _safe_fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _crop_coverage(
    candidate_mm: float,
    measures: Sequence[SeriesMeasure],
    total_series: int,
    observed_planes: Sequence[str],
    minimum_coverage: float,
) -> dict[str, Any]:
    covered = sum(
        measure.minimum_physical_fov_mm + 1e-9 >= candidate_mm for measure in measures
    )
    measures_by_plane: dict[str, list[SeriesMeasure]] = defaultdict(list)
    for measure in measures:
        measures_by_plane[measure.anatomical_plane].append(measure)

    by_plane: dict[str, dict[str, int | float | None]] = {}
    plane_coverages: list[float] = []
    every_plane_is_analyzable = True
    for plane in observed_planes:
        plane_measures = measures_by_plane.get(plane, [])
        plane_covered = sum(
            measure.minimum_physical_fov_mm + 1e-9 >= candidate_mm
            for measure in plane_measures
        )
        coverage = _safe_fraction(plane_covered, len(plane_measures))
        if coverage is None:
            every_plane_is_analyzable = False
        else:
            plane_coverages.append(coverage)
        by_plane[plane] = {
            "analyzable_series": len(plane_measures),
            "covered_without_padding": plane_covered,
            "coverage_of_analyzable_series": coverage,
        }

    overall_coverage = _safe_fraction(covered, len(measures))
    minimum_plane_coverage = min(plane_coverages) if plane_coverages else None
    meets_target = bool(
        every_plane_is_analyzable
        and overall_coverage is not None
        and overall_coverage >= minimum_coverage
        and minimum_plane_coverage is not None
        and minimum_plane_coverage >= minimum_coverage
    )
    return {
        "square_crop_mm": candidate_mm,
        "all_series": total_series,
        "analyzable_series": len(measures),
        "covered_without_padding": covered,
        "padding_required": len(measures) - covered,
        "coverage_of_analyzable_series": overall_coverage,
        "coverage_of_all_series": _safe_fraction(covered, total_series),
        "minimum_anatomical_plane_coverage": minimum_plane_coverage,
        "meets_minimum_coverage_in_every_plane": meets_target,
        "by_anatomical_plane": by_plane,
    }


def _recommendation(
    crop_coverage: Sequence[dict[str, Any]],
    minimum_coverage: float,
    target_pixels: int,
    excluded_series: int,
) -> dict[str, Any]:
    qualifying = [
        candidate
        for candidate in crop_coverage
        if candidate["meets_minimum_coverage_in_every_plane"]
    ]
    if qualifying:
        selected = max(qualifying, key=lambda item: float(item["square_crop_mm"]))
        selection_status = "coverage_target_met"
    else:
        selected = min(crop_coverage, key=lambda item: float(item["square_crop_mm"]))
        selection_status = "coverage_target_not_met"

    crop_mm = float(selected["square_crop_mm"])
    output_spacing = crop_mm / target_pixels
    return {
        "selection_status": selection_status,
        "anatomy_preservation_policy": "full_fov_primary_view_required",
        "selected_square_crop_mm": crop_mm,
        "selected_square_crop_role": "optional_detail_view_only",
        "target_output_shape_pixels": [target_pixels, target_pixels],
        "detail_view_in_plane_pixel_spacing_mm": output_spacing,
        "minimum_required_coverage": minimum_coverage,
        "selected_candidate_coverage_of_analyzable_series": selected[
            "coverage_of_analyzable_series"
        ],
        "selected_candidate_minimum_anatomical_plane_coverage": selected[
            "minimum_anatomical_plane_coverage"
        ],
        "excluded_series_requiring_explicit_handling": excluded_series,
        "selection_rule": (
            "Choose the largest candidate square detail window whose no-padding coverage "
            "meets the threshold overall and within every observed anatomical "
            "plane; if none qualifies, report the smallest candidate as an "
            "uncertified fallback."
        ),
        "preprocessing_policy": [
            (
                "Interpret DICOM PixelSpacing as physical row and column spacing and "
                "construct the in-plane image extent before resizing."
            ),
            (
                "Use the complete physical field of view as the required primary view: "
                "resize it to fit the target square while preserving aspect ratio, then "
                "pad the shorter dimension symmetrically. Do not crop this view."
            ),
            (
                "Retain the selected fixed physical window only as an optional "
                "high-resolution detail view. Never use it as the sole input because a "
                "fixed center crop cannot guarantee complete anatomy."
            ),
            (
                "Reject or route series with missing, invalid, or inconsistent spacing "
                "through an explicit fallback; never silently treat pixel units as mm."
            ),
            (
                "Continue consensus-normal slice sorting independently of this in-plane "
                "resampling policy."
            ),
            (
                "Visually review representative low-FOV, high-FOV, and off-center "
                "series before training to verify full-FOV retention, padding behavior, "
                "and any optional detail-view localization."
            ),
        ],
    }


def analyze_physical_geometry_manifest(
    part_paths: Sequence[Path],
    inventory_summary: dict[str, Any],
    *,
    candidate_crops_mm: Sequence[float] = DEFAULT_CANDIDATE_CROPS_MM,
    minimum_coverage: float = 0.90,
    target_pixels: int = 224,
    spacing_relative_tolerance: float = 1e-3,
    spacing_absolute_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Analyze every manifest series and return identifier-free aggregates."""
    if not part_paths:
        raise PhysicalGeometryAuditError("No manifest Parquet parts were supplied")
    candidates = _validated_candidates(candidate_crops_mm)
    if not math.isfinite(minimum_coverage) or not 0 < minimum_coverage <= 1:
        raise PhysicalGeometryAuditError("Minimum coverage must be in (0, 1]")
    if target_pixels <= 0:
        raise PhysicalGeometryAuditError("Target pixels must be positive")
    for name, value in (
        ("spacing relative tolerance", spacing_relative_tolerance),
        ("spacing absolute tolerance", spacing_absolute_tolerance),
    ):
        if not math.isfinite(value) or value < 0:
            raise PhysicalGeometryAuditError(f"{name} must be finite and nonnegative")

    expected_rows = _integer(inventory_summary.get("dicom_files"), default=None)
    expected_series = _integer(inventory_summary.get("metadata_series"), default=None)
    if expected_rows is None or expected_rows < 0:
        raise PhysicalGeometryAuditError(
            "Inventory summary has no valid dicom_files count"
        )
    if expected_series is None or expected_series < 0:
        raise PhysicalGeometryAuditError(
            "Inventory summary has no valid metadata_series count"
        )

    total_rows = 0
    total_series = 0
    noncontiguous_series = 0
    seen_series: set[tuple[str, str, str]] = set()
    current_key: tuple[str, str, str] | None = None
    current_series: SeriesAccumulator | None = None
    findings: Counter[str] = Counter({key: 0 for key in SERIES_FINDING_KEYS})
    row_findings: Counter[str] = Counter({key: 0 for key in ROW_FINDING_KEYS})
    all_split_counts: Counter[str] = Counter()
    all_plane_counts: Counter[str] = Counter()
    all_sequence_counts: Counter[str] = Counter()
    all_combined_counts: Counter[str] = Counter()
    measures: list[SeriesMeasure] = []

    def finish_series() -> None:
        nonlocal total_series
        if current_series is None:
            return
        total_series += 1
        series = current_series
        all_split_counts[series.split] += 1
        all_plane_counts[series.anatomical_plane] += 1
        all_sequence_counts[_sequence_key(series)] += 1
        all_combined_counts[_combined_key(series)] += 1
        row_findings["non_ok_header_rows"] += series.non_ok_header_rows
        row_findings["metadata_inconsistent_rows"] += series.metadata_inconsistencies
        row_findings["missing_spacing_rows"] += series.missing_spacing_rows
        row_findings["missing_spacing_row_component_rows"] += (
            series.missing_spacing_row_component_rows
        )
        row_findings["missing_spacing_column_component_rows"] += (
            series.missing_spacing_column_component_rows
        )
        row_findings["invalid_spacing_rows"] += series.invalid_spacing_rows
        row_findings["invalid_spacing_row_component_rows"] += (
            series.invalid_spacing_row_component_rows
        )
        row_findings["invalid_spacing_column_component_rows"] += (
            series.invalid_spacing_column_component_rows
        )
        row_findings["missing_dimension_rows"] += series.missing_dimension_rows
        row_findings["invalid_dimension_rows"] += series.invalid_dimension_rows
        row_findings["valid_physical_geometry_rows"] += series.valid_physical_rows

        spacing_rows = [value[0] for value in series.spacings]
        spacing_columns = [value[1] for value in series.spacings]
        row_spacing_inconsistent = not _consistent(
            spacing_rows, spacing_relative_tolerance, spacing_absolute_tolerance
        )
        column_spacing_inconsistent = not _consistent(
            spacing_columns, spacing_relative_tolerance, spacing_absolute_tolerance
        )
        spacing_inconsistent = row_spacing_inconsistent or column_spacing_inconsistent
        dimensions_inconsistent = len(set(series.dimensions)) > 1
        invalid_group_metadata = (
            series.split not in {"train", "test"}
            or series.anatomical_plane not in {"Axial", "Coronal", "Sagittal"}
            or series.fluid_sensitive not in {0, 1}
            or series.fat_suppression not in {0, 1}
        )

        series_flags = {
            "series_with_non_ok_headers": series.non_ok_header_rows > 0,
            "series_with_metadata_inconsistency": series.metadata_inconsistencies > 0,
            "series_with_invalid_group_metadata": invalid_group_metadata,
            "series_with_missing_spacing": series.missing_spacing_rows > 0,
            "series_with_invalid_spacing": series.invalid_spacing_rows > 0,
            "series_with_inconsistent_row_spacing": row_spacing_inconsistent,
            "series_with_inconsistent_column_spacing": column_spacing_inconsistent,
            "series_with_inconsistent_spacing": spacing_inconsistent,
            "series_with_missing_dimensions": series.missing_dimension_rows > 0,
            "series_with_invalid_dimensions": series.invalid_dimension_rows > 0,
            "series_with_inconsistent_dimensions": dimensions_inconsistent,
        }
        for name, present in series_flags.items():
            findings[name] += int(present)

        complete = (
            series.rows_seen > 0
            and series.non_ok_header_rows == 0
            and series.metadata_inconsistencies == 0
            and not invalid_group_metadata
            and series.missing_spacing_rows == 0
            and series.invalid_spacing_rows == 0
            and series.missing_dimension_rows == 0
            and series.invalid_dimension_rows == 0
            and len(series.spacings) == series.rows_seen
            and len(series.dimensions) == series.rows_seen
            and not spacing_inconsistent
            and not dimensions_inconsistent
        )
        if not complete:
            findings["series_excluded_from_physical_analysis"] += 1
            return

        spacing_row = float(np.median(spacing_rows))
        spacing_column = float(np.median(spacing_columns))
        rows, columns = series.dimensions[0]
        fov_row = rows * spacing_row
        fov_column = columns * spacing_column
        if not all(
            math.isfinite(value) and value > 0
            for value in (spacing_row, spacing_column, fov_row, fov_column)
        ):
            findings["series_excluded_from_physical_analysis"] += 1
            findings["series_with_nonfinite_derived_geometry"] += 1
            return
        findings["series_analyzable"] += 1
        measures.append(
            SeriesMeasure(
                split=series.split,
                anatomical_plane=series.anatomical_plane,
                fluid_sensitive=series.fluid_sensitive,
                fat_suppression=series.fat_suppression,
                pixel_spacing_row_mm=spacing_row,
                pixel_spacing_column_mm=spacing_column,
                physical_fov_row_mm=fov_row,
                physical_fov_column_mm=fov_column,
            )
        )

    for row in _iter_rows(part_paths):
        total_rows += 1
        split = str(row.get("split") or "")
        study_uid = str(row.get("path_study_uid") or "")
        series_uid = str(row.get("path_series_uid") or "")
        if not split or not study_uid or not series_uid:
            raise PhysicalGeometryAuditError(
                "Manifest contains a row without series grouping identifiers"
            )
        key = (split, study_uid, series_uid)
        if key != current_key:
            finish_series()
            if key in seen_series:
                noncontiguous_series += 1
            seen_series.add(key)
            current_key = key
            current_series = _new_series(row)
        if current_series is None:
            raise AssertionError("Series accumulator was not initialized")
        current_series.add(row)
    finish_series()

    checks = {
        "manifest_row_count_mismatch": int(total_rows != expected_rows),
        "series_count_mismatch": int(total_series != expected_series),
        "noncontiguous_series": noncontiguous_series,
        "metadata_inconsistent_rows": row_findings["metadata_inconsistent_rows"],
        "no_analyzable_series": int(not measures),
    }
    status = "pass" if all(value == 0 for value in checks.values()) else "fail"

    measures_by_split: dict[str, list[SeriesMeasure]] = defaultdict(list)
    measures_by_plane: dict[str, list[SeriesMeasure]] = defaultdict(list)
    measures_by_sequence: dict[str, list[SeriesMeasure]] = defaultdict(list)
    measures_by_combined: dict[str, list[SeriesMeasure]] = defaultdict(list)
    for measure in measures:
        measures_by_split[measure.split].append(measure)
        measures_by_plane[measure.anatomical_plane].append(measure)
        measures_by_sequence[_sequence_key(measure)].append(measure)
        measures_by_combined[_combined_key(measure)].append(measure)

    observed_planes = sorted(all_plane_counts)
    coverage = [
        _crop_coverage(
            candidate,
            measures,
            total_series,
            observed_planes,
            minimum_coverage,
        )
        for candidate in candidates
    ]
    excluded_series = total_series - len(measures)
    return {
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "privacy": (
            "Aggregate statistics only; no study, series, SOP, or file identifiers "
            "are included."
        ),
        "manifest_fingerprint": inventory_summary.get("manifest_fingerprint"),
        "method": {
            "aggregation_unit": "series",
            "physical_fov_definition": (
                "Rows * PixelSpacingRow and Columns * PixelSpacingColumn"
            ),
            "spacing_consistency_relative_tolerance": spacing_relative_tolerance,
            "spacing_consistency_absolute_tolerance_mm": spacing_absolute_tolerance,
            "crop_coverage_definition": (
                "Both in-plane physical FOV dimensions are at least the candidate "
                "square crop size; no padding is required."
            ),
        },
        "counts": {
            "manifest_rows": total_rows,
            "series": total_series,
            "analyzable_series": len(measures),
            "excluded_series": excluded_series,
        },
        "row_geometry_findings": dict(sorted(row_findings.items())),
        "series_geometry_findings": dict(sorted(findings.items())),
        "distributions": {
            "overall": _group_summary(measures, total_series),
            "by_split": {
                key: _group_summary(measures_by_split.get(key, []), count)
                for key, count in sorted(all_split_counts.items())
            },
            "by_anatomical_plane": {
                key: _group_summary(measures_by_plane.get(key, []), count)
                for key, count in sorted(all_plane_counts.items())
            },
            "by_sequence_flags": {
                key: _group_summary(measures_by_sequence.get(key, []), count)
                for key, count in sorted(all_sequence_counts.items())
            },
            "by_anatomical_plane_and_sequence_flags": {
                key: _group_summary(measures_by_combined.get(key, []), count)
                for key, count in sorted(all_combined_counts.items())
            },
        },
        "candidate_square_crop_coverage": coverage,
        "recommendation": _recommendation(
            coverage,
            minimum_coverage,
            target_pixels,
            excluded_series,
        ),
        "checks": checks,
    }


def _validated_candidates(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise PhysicalGeometryAuditError("At least one crop candidate is required")
    candidates: set[float] = set()
    for value in values:
        parsed = _finite_float(value)
        if parsed is None or parsed <= 0:
            raise PhysicalGeometryAuditError(
                "Crop candidates must be finite positive millimetre values"
            )
        candidates.add(parsed)
    return tuple(sorted(candidates))


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze pixel spacing and physical field of view in an existing "
            "DICOM audit manifest without reopening raw DICOMs."
        )
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=Path("/home/shaurya/rsna-data/manifests/dicom_audit_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON; defaults to <audit-root>/physical_geometry_summary.json",
    )
    parser.add_argument(
        "--candidate-crop-mm",
        type=float,
        nargs="+",
        default=list(DEFAULT_CANDIDATE_CROPS_MM),
        help="Candidate square physical crop sizes in millimetres",
    )
    parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=0.90,
        help="Required no-padding coverage overall and in every plane",
    )
    parser.add_argument(
        "--target-pixels",
        type=int,
        default=224,
        help="Square output size used to derive the recommended pixel spacing",
    )
    parser.add_argument(
        "--spacing-relative-tolerance",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--spacing-absolute-tolerance-mm",
        type=float,
        default=1e-6,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audit_root = args.audit_root.expanduser().resolve()
        summary_path = audit_root / "inventory_summary.json"
        parts_root = audit_root / "manifest_parts"
        if not summary_path.is_file():
            raise PhysicalGeometryAuditError(
                f"Missing inventory summary under audit root: {summary_path}"
            )
        part_paths = sorted(parts_root.glob("part-*.parquet"))
        if not part_paths:
            raise PhysicalGeometryAuditError(
                f"No manifest Parquet parts found under: {parts_root}"
            )
        with summary_path.open("r", encoding="utf-8") as file:
            inventory_summary = json.load(file)
        if inventory_summary.get("status") != "pass":
            raise PhysicalGeometryAuditError(
                "The source inventory did not pass; physical-geometry audit refused"
            )
        result = analyze_physical_geometry_manifest(
            part_paths,
            inventory_summary,
            candidate_crops_mm=args.candidate_crop_mm,
            minimum_coverage=args.minimum_coverage,
            target_pixels=args.target_pixels,
            spacing_relative_tolerance=args.spacing_relative_tolerance,
            spacing_absolute_tolerance=args.spacing_absolute_tolerance_mm,
        )
        output_path = (
            args.output.expanduser().resolve()
            if args.output
            else audit_root / "physical_geometry_summary.json"
        )
        _write_json_atomic(output_path, result)
        recommendation = result["recommendation"]
        print(
            f"Physical geometry audit {result['status']}: "
            f"{result['counts']['analyzable_series']:,}/"
            f"{result['counts']['series']:,} analyzable series; selected "
            f"{recommendation['selected_square_crop_mm']:g} mm optional detail window "
            f"({recommendation['selection_status']}); {output_path}",
            flush=True,
        )
        return 0 if result["status"] == "pass" else 2
    except (
        OSError,
        json.JSONDecodeError,
        PhysicalGeometryAuditError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
