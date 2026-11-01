"""Identifier-free analysis of per-slice DICOM orientation variation.

This module consumes the read-only Parquet manifest produced by
``dicom_audit.py``. It never opens a DICOM and emits aggregate statistics only;
study, series, SOP, and file identifiers are deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ORIENTATION_COLUMNS = (
    "image_orientation_row_x",
    "image_orientation_row_y",
    "image_orientation_row_z",
    "image_orientation_column_x",
    "image_orientation_column_y",
    "image_orientation_column_z",
)
POSITION_COLUMNS = (
    "image_position_x",
    "image_position_y",
    "image_position_z",
)
SOURCE_COLUMNS = (
    "split",
    "path_study_uid",
    "path_series_uid",
    "anatomical_plane",
    "fluid_sensitive",
    "fat_suppression",
    "instance_number",
    *POSITION_COLUMNS,
    *ORIENTATION_COLUMNS,
    "header_status",
    "geometry_status",
)
ANGLE_BINS = (
    (0.001, "at_most_0.001_deg"),
    (0.01, "0.001_to_0.01_deg"),
    (0.1, "0.01_to_0.1_deg"),
    (1.0, "0.1_to_1_deg"),
    (math.inf, "over_1_deg"),
)


class OrientationAuditError(RuntimeError):
    """Raised when the existing manifest cannot be analyzed safely."""


@dataclass(frozen=True, slots=True)
class SliceGeometry:
    instance_number: int
    position: np.ndarray
    raw_orientation: np.ndarray
    row: np.ndarray
    column: np.ndarray


@dataclass(slots=True)
class SeriesGeometry:
    split: str
    anatomical_plane: str
    fluid_sensitive: int
    fat_suppression: int
    slices: list[SliceGeometry] = field(default_factory=list)
    invalid_rows: int = 0
    metadata_inconsistencies: int = 0

    def add(self, row: dict[str, Any]) -> None:
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

        instance = _integer(row.get("instance_number"), default=None)
        position = _finite_vector(row, POSITION_COLUMNS)
        orientation = _finite_vector(row, ORIENTATION_COLUMNS)
        if (
            row.get("header_status") != "ok"
            or row.get("geometry_status") != "ok"
            or instance is None
            or position is None
            or orientation is None
        ):
            self.invalid_rows += 1
            return
        row_vector = _unit(orientation[:3])
        column_vector = _unit(orientation[3:])
        if row_vector is None or column_vector is None:
            self.invalid_rows += 1
            return
        self.slices.append(
            SliceGeometry(
                instance_number=instance,
                position=position,
                raw_orientation=orientation,
                row=row_vector,
                column=column_vector,
            )
        )


def _integer(value: Any, default: int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _finite_vector(row: dict[str, Any], columns: Sequence[str]) -> np.ndarray | None:
    try:
        vector = np.asarray([float(row[column]) for column in columns], dtype=float)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not np.all(np.isfinite(vector)):
        return None
    return vector


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return vector / norm


def _max_pairwise_angle_degrees(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return 0.0
    dots = np.clip(vectors @ vectors.T, -1.0, 1.0)
    return math.degrees(math.acos(float(np.min(dots))))


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


def _angle_bin(angle_degrees: float) -> str:
    for upper_bound, label in ANGLE_BINS:
        if angle_degrees <= upper_bound:
            return label
    raise AssertionError("ANGLE_BINS must end with an infinite upper bound")


def _profile_pattern(
    slice_count: int,
    unique_profiles: int,
    modal_fraction: float,
    transition_count: int,
) -> str:
    if unique_profiles == 1:
        return "single_profile"
    if modal_fraction >= 0.9:
        return "dominant_profile_with_rare_outliers"
    if transition_count <= unique_profiles:
        return "few_contiguous_profile_blocks"
    if unique_profiles >= max(3, math.ceil(0.5 * slice_count)):
        return "slice_varying_profiles"
    return "mixed_repeated_profiles"


def _series_result(series: SeriesGeometry) -> dict[str, Any]:
    ordered = sorted(series.slices, key=lambda item: item.instance_number)
    instances = [item.instance_number for item in ordered]
    duplicate_instances = len(instances) - len(set(instances))
    raw_orientations = np.asarray([item.raw_orientation for item in ordered])
    positions = np.asarray([item.position for item in ordered])
    rows = np.asarray([item.row for item in ordered])
    columns = np.asarray([item.column for item in ordered])
    normals = np.cross(rows, columns)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    reference_normal = normals[0]
    normals = np.asarray(
        [
            normal if np.dot(normal, reference_normal) >= 0 else -normal
            for normal in normals
        ]
    )

    # Match SeriesAccumulator in dicom_audit.py exactly. Python's scalar round
    # and NumPy's vectorized round can differ at binary floating-point ties.
    profiles = [
        tuple(round(float(value), 5) for value in vector) for vector in raw_orientations
    ]
    profile_counts = Counter(profiles)
    unique_profiles = len(profile_counts)
    modal_fraction = max(profile_counts.values()) / len(profiles)
    transition_count = sum(first != second for first, second in pairwise(profiles))
    transition_fraction = transition_count / max(1, len(profiles) - 1)

    adjacent_normal_angles = []
    for first, second in pairwise(normals):
        dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
        adjacent_normal_angles.append(math.degrees(math.acos(dot)))

    consensus_normal = _unit(np.mean(normals, axis=0))
    if consensus_normal is None:
        consensus_normal = reference_normal
    local_coordinates = np.sum(positions * normals, axis=1)
    consensus_coordinates = positions @ consensus_normal
    projection_delta = local_coordinates - consensus_coordinates
    projection_delta -= np.median(projection_delta)
    centered_projection_delta = float(np.max(np.abs(projection_delta)))
    local_order = tuple(np.argsort(local_coordinates, kind="stable"))
    consensus_order = tuple(np.argsort(consensus_coordinates, kind="stable"))

    max_normal_angle = _max_pairwise_angle_degrees(normals)
    return {
        "slice_count": len(ordered),
        "unique_profiles": unique_profiles,
        "modal_fraction": modal_fraction,
        "transition_count": transition_count,
        "transition_fraction": transition_fraction,
        "profile_pattern": _profile_pattern(
            len(ordered), unique_profiles, modal_fraction, transition_count
        ),
        "max_component_range": float(
            np.max(np.max(raw_orientations, axis=0) - np.min(raw_orientations, axis=0))
        ),
        "max_row_angle_degrees": _max_pairwise_angle_degrees(rows),
        "max_column_angle_degrees": _max_pairwise_angle_degrees(columns),
        "max_normal_angle_degrees": max_normal_angle,
        "max_adjacent_normal_angle_degrees": max(adjacent_normal_angles, default=0.0),
        "normal_angle_bin": _angle_bin(max_normal_angle),
        "max_centered_projection_delta_mm": centered_projection_delta,
        "consensus_projection_changes_order": local_order != consensus_order,
        "duplicate_instance_numbers": duplicate_instances,
    }


def _iter_rows(part_paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    for part_path in part_paths:
        parquet = pq.ParquetFile(part_path)
        available = set(parquet.schema_arrow.names)
        missing = set(SOURCE_COLUMNS) - available
        if missing:
            raise OrientationAuditError(
                f"Manifest part is missing required columns: {sorted(missing)}"
            )
        for batch in parquet.iter_batches(
            columns=list(SOURCE_COLUMNS), batch_size=65_536
        ):
            yield from batch.to_pylist()


def _new_series(row: dict[str, Any]) -> SeriesGeometry:
    return SeriesGeometry(
        split=str(row.get("split") or "<missing>"),
        anatomical_plane=str(row.get("anatomical_plane") or "<missing>"),
        fluid_sensitive=int(_integer(row.get("fluid_sensitive"), default=-1)),
        fat_suppression=int(_integer(row.get("fat_suppression"), default=-1)),
    )


def analyze_orientation_manifest(
    part_paths: Sequence[Path], inventory_summary: dict[str, Any]
) -> dict[str, Any]:
    """Analyze all series while retaining no identifiers in the result."""
    if not part_paths:
        raise OrientationAuditError("No manifest Parquet parts were supplied")

    total_rows = 0
    total_series = 0
    variable_series = 0
    variable_slices = 0
    invalid_rows = 0
    metadata_inconsistencies = 0
    noncontiguous_series = 0
    duplicate_instance_numbers = 0
    seen_series: set[tuple[str, str, str]] = set()
    current_key: tuple[str, str, str] | None = None
    current_series: SeriesGeometry | None = None

    variable_by_split: Counter[str] = Counter()
    variable_by_plane: Counter[str] = Counter()
    variable_by_sequence_flags: Counter[str] = Counter()
    angle_bins: Counter[str] = Counter()
    patterns: Counter[str] = Counter()
    slice_counts: list[float] = []
    unique_profile_counts: list[float] = []
    modal_fractions: list[float] = []
    transition_fractions: list[float] = []
    component_ranges: list[float] = []
    row_angles: list[float] = []
    column_angles: list[float] = []
    normal_angles: list[float] = []
    adjacent_normal_angles: list[float] = []
    projection_deltas: list[float] = []
    consensus_order_changes = 0

    def finish_series() -> None:
        nonlocal total_series
        nonlocal variable_series
        nonlocal variable_slices
        nonlocal invalid_rows
        nonlocal metadata_inconsistencies
        nonlocal duplicate_instance_numbers
        nonlocal consensus_order_changes
        if current_series is None:
            return
        total_series += 1
        invalid_rows += current_series.invalid_rows
        metadata_inconsistencies += current_series.metadata_inconsistencies
        if not current_series.slices:
            return
        result = _series_result(current_series)
        duplicate_instance_numbers += int(result["duplicate_instance_numbers"])
        if int(result["unique_profiles"]) == 1:
            return

        variable_series += 1
        variable_slices += int(result["slice_count"])
        variable_by_split[current_series.split] += 1
        variable_by_plane[current_series.anatomical_plane] += 1
        sequence_key = (
            f"fluid_sensitive={current_series.fluid_sensitive},"
            f"fat_suppression={current_series.fat_suppression}"
        )
        variable_by_sequence_flags[sequence_key] += 1
        angle_bins[str(result["normal_angle_bin"])] += 1
        patterns[str(result["profile_pattern"])] += 1
        consensus_order_changes += int(result["consensus_projection_changes_order"])
        slice_counts.append(float(result["slice_count"]))
        unique_profile_counts.append(float(result["unique_profiles"]))
        modal_fractions.append(float(result["modal_fraction"]))
        transition_fractions.append(float(result["transition_fraction"]))
        component_ranges.append(float(result["max_component_range"]))
        row_angles.append(float(result["max_row_angle_degrees"]))
        column_angles.append(float(result["max_column_angle_degrees"]))
        normal_angles.append(float(result["max_normal_angle_degrees"]))
        adjacent_normal_angles.append(
            float(result["max_adjacent_normal_angle_degrees"])
        )
        projection_deltas.append(float(result["max_centered_projection_delta_mm"]))

    for row in _iter_rows(part_paths):
        total_rows += 1
        key = (
            str(row.get("split") or ""),
            str(row.get("path_study_uid") or ""),
            str(row.get("path_series_uid") or ""),
        )
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

    expected_rows = int(inventory_summary.get("dicom_files", -1))
    expected_series = int(inventory_summary.get("metadata_series", -1))
    expected_variable = int(
        inventory_summary.get("series_geometry_audit", {}).get(
            "series_with_inconsistent_orientation", -1
        )
    )
    checks = {
        "manifest_row_count_mismatch": int(total_rows != expected_rows),
        "series_count_mismatch": int(total_series != expected_series),
        "variable_series_count_mismatch": int(variable_series != expected_variable),
        "invalid_geometry_rows": invalid_rows,
        "metadata_inconsistencies": metadata_inconsistencies,
        "noncontiguous_series": noncontiguous_series,
        "duplicate_instance_numbers": duplicate_instance_numbers,
    }
    status = "pass" if all(value == 0 for value in checks.values()) else "fail"
    return {
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "privacy": (
            "Aggregate statistics only; no study, series, SOP, or file identifiers "
            "are included."
        ),
        "manifest_fingerprint": inventory_summary.get("manifest_fingerprint"),
        "counts": {
            "manifest_rows": total_rows,
            "series": total_series,
            "constant_orientation_series": total_series - variable_series,
            "variable_orientation_series": variable_series,
            "slices_in_variable_orientation_series": variable_slices,
        },
        "variable_series_by_split": dict(sorted(variable_by_split.items())),
        "variable_series_by_anatomical_plane": dict(sorted(variable_by_plane.items())),
        "variable_series_by_sequence_flags": dict(
            sorted(variable_by_sequence_flags.items())
        ),
        "normal_angle_bins": {label: angle_bins[label] for _, label in ANGLE_BINS},
        "profile_patterns": dict(sorted(patterns.items())),
        "variable_series_distributions": {
            "slices": _distribution(slice_counts),
            "unique_rounded_orientation_profiles": _distribution(unique_profile_counts),
            "modal_profile_fraction": _distribution(modal_fractions),
            "profile_transition_fraction": _distribution(transition_fractions),
            "max_orientation_component_range": _distribution(component_ranges),
            "max_row_angle_degrees": _distribution(row_angles),
            "max_column_angle_degrees": _distribution(column_angles),
            "max_normal_angle_degrees": _distribution(normal_angles),
            "max_adjacent_normal_angle_degrees": _distribution(adjacent_normal_angles),
            "max_centered_projection_delta_mm": _distribution(projection_deltas),
        },
        "slice_sorting": {
            "consensus_projection_preserves_order": variable_series
            - consensus_order_changes,
            "consensus_projection_changes_order": consensus_order_changes,
            "interpretation": (
                "Compares sorting by each slice's own normal with projection onto "
                "the mean aligned series normal."
            ),
        },
        "definitions": {
            "variable_orientation": (
                "More than one six-component ImageOrientationPatient profile after "
                "rounding each component to five decimal places."
            ),
            "normal_angle_bins": (
                "Maximum pairwise angle between aligned slice-plane normals."
            ),
            "centered_projection_delta": (
                "Maximum absolute difference, after removing the median offset, "
                "between per-slice-normal and consensus-normal coordinates."
            ),
            "profile_patterns": {
                "dominant_profile_with_rare_outliers": (
                    "At least 90% of slices share one rounded profile."
                ),
                "few_contiguous_profile_blocks": (
                    "Profile transitions are no more numerous than unique profiles."
                ),
                "slice_varying_profiles": (
                    "At least half of slices have distinct rounded profiles, with "
                    "a minimum of three profiles."
                ),
                "mixed_repeated_profiles": (
                    "Variable series not covered by the preceding patterns."
                ),
            },
        },
        "checks": checks,
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze per-slice orientation variation in an existing DICOM audit "
            "manifest and emit identifier-free aggregate statistics."
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
        help=(
            "Output JSON path; defaults to "
            "<audit-root>/orientation_variation_summary.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audit_root = args.audit_root.expanduser().resolve()
        summary_path = audit_root / "inventory_summary.json"
        parts_root = audit_root / "manifest_parts"
        if not summary_path.is_file():
            raise OrientationAuditError(
                f"Missing inventory summary under audit root: {summary_path}"
            )
        part_paths = sorted(parts_root.glob("part-*.parquet"))
        if not part_paths:
            raise OrientationAuditError(
                f"No manifest Parquet parts found under: {parts_root}"
            )
        with summary_path.open("r", encoding="utf-8") as file:
            inventory_summary = json.load(file)
        if inventory_summary.get("status") != "pass":
            raise OrientationAuditError(
                "The source inventory did not pass; orientation analysis refused"
            )
        result = analyze_orientation_manifest(part_paths, inventory_summary)
        output_path = (
            args.output.expanduser().resolve()
            if args.output
            else audit_root / "orientation_variation_summary.json"
        )
        _write_json_atomic(output_path, result)
        print(
            f"Orientation analysis {result['status']}: "
            f"{result['counts']['variable_orientation_series']:,}/"
            f"{result['counts']['series']:,} variable series; {output_path}",
            flush=True,
        )
        return 0 if result["status"] == "pass" else 2
    except (OSError, json.JSONDecodeError, OrientationAuditError) as error:
        print(f"ERROR: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
