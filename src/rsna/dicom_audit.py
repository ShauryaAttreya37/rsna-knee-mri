"""Read-only, resumable DICOM inventory and decoder audit.

The auditor never writes to the competition-data tree. It reads a restricted
set of DICOM header tags into partitioned Parquet files, validates the
directory/CSV/UID relationships, and optionally decodes a deterministic sample
or every image. All generated artifacts live under a separate manifest root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import statistics
import sys
import time
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pydicom
from pydicom import dcmread
from pydicom.pixels import get_decoder, pixel_array
from pydicom.uid import (
    JPEG2000,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGLossless,
    JPEGLosslessSV1,
)

FORMAT_VERSION = 1
SPLIT_DIRECTORIES = {"train": "train_series", "test": "test_series"}
SERIES_CSVS = {"train": "train_series.csv", "test": "test_series.csv"}
REQUIRED_SERIES_COLUMNS = {
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
}
HEADER_TAGS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "SOPClassUID",
    "InstanceNumber",
    "Modality",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "Rows",
    "Columns",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "NumberOfFrames",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "SliceLocation",
    "RescaleSlope",
    "RescaleIntercept",
]
PACKAGE_NAMES = (
    "numpy",
    "pyarrow",
    "pydicom",
    "pylibjpeg",
    "pylibjpeg-libjpeg",
    "pylibjpeg-openjpeg",
)
EXPECTED_TRANSFER_SYNTAXES = (
    ImplicitVRLittleEndian,
    ExplicitVRLittleEndian,
    JPEGLossless,
    JPEGLosslessSV1,
    JPEG2000Lossless,
    JPEG2000,
)


class AuditConfigurationError(RuntimeError):
    """Raised when an audit cannot run safely with the supplied paths/options."""


@dataclass(frozen=True, slots=True)
class HeaderTask:
    absolute_path: str
    relative_path: str
    split: str
    file_size: int
    is_symlink: bool
    path_structure_ok: bool
    path_study_uid: str | None
    path_series_uid: str | None
    path_sop_uid: str | None


@dataclass(frozen=True, slots=True)
class SeriesMetadata:
    anatomical_plane: str
    fluid_sensitive: int
    fat_suppression: int


@dataclass(slots=True)
class SeriesAccumulator:
    """Bounded-memory geometric audit state for one contiguous series."""

    count: int = 0
    geometry_ok: int = 0
    instances: list[int] = field(default_factory=list)
    coordinates: list[float] = field(default_factory=list)
    instance_coordinate_pairs: list[tuple[int, float]] = field(default_factory=list)
    orientation_profiles: set[tuple[float, ...]] = field(default_factory=set)
    dimensions: set[tuple[int, int]] = field(default_factory=set)

    def add(self, row: dict[str, Any]) -> None:
        self.count += 1
        if row["geometry_status"] == "ok":
            self.geometry_ok += 1
        instance = _safe_int(row["instance_number"])
        coordinate = _safe_float(row["slice_coordinate"])
        if instance is not None:
            self.instances.append(instance)
        if coordinate is not None:
            self.coordinates.append(coordinate)
        if instance is not None and coordinate is not None:
            self.instance_coordinate_pairs.append((instance, coordinate))
        orientation = tuple(
            round(float(row[name]), 5)
            for name in (
                "image_orientation_row_x",
                "image_orientation_row_y",
                "image_orientation_row_z",
                "image_orientation_column_x",
                "image_orientation_column_y",
                "image_orientation_column_z",
            )
            if row[name] is not None
        )
        if len(orientation) == 6:
            self.orientation_profiles.add(orientation)
        if row["rows"] is not None and row["columns"] is not None:
            self.dimensions.add((int(row["rows"]), int(row["columns"])))

    def result(self) -> dict[str, Any]:
        rounded_coordinates = [round(value, 5) for value in self.coordinates]
        unique_coordinates = sorted(set(rounded_coordinates))
        unique_instances = set(self.instances)
        geometry_sort_ready = (
            self.geometry_ok == self.count
            and len(self.coordinates) == self.count
            and len(unique_coordinates) == self.count
        )
        instance_sort_ready = (
            len(self.instances) == self.count and len(unique_instances) == self.count
        )
        order_inconsistent = False
        if geometry_sort_ready and instance_sort_ready:
            ordered = sorted(self.instance_coordinate_pairs)
            differences = [second[1] - first[1] for first, second in pairwise(ordered)]
            if differences:
                tolerance = 1e-5
                order_inconsistent = not (
                    all(value > tolerance for value in differences)
                    or all(value < -tolerance for value in differences)
                )

        nonuniform_spacing = False
        if len(unique_coordinates) >= 3:
            spacings = [
                second - first for first, second in pairwise(unique_coordinates)
            ]
            median_spacing = float(statistics.median(spacings))
            tolerance = max(1e-3, 0.05 * abs(median_spacing))
            nonuniform_spacing = any(
                abs(value - median_spacing) > tolerance for value in spacings
            )

        if geometry_sort_ready:
            sort_strategy = "geometry"
        elif instance_sort_ready:
            sort_strategy = "instance_number_fallback"
        else:
            sort_strategy = "not_sortable"
        return {
            "count": self.count,
            "sort_strategy": sort_strategy,
            "duplicate_slice_coordinates": len(rounded_coordinates)
            - len(unique_coordinates),
            "duplicate_instance_numbers": len(self.instances) - len(unique_instances),
            "inconsistent_orientation": len(self.orientation_profiles) > 1,
            "inconsistent_dimensions": len(self.dimensions) > 1,
            "instance_coordinate_order_inconsistent": order_inconsistent,
            "nonuniform_slice_spacing": nonuniform_spacing,
        }


@dataclass(frozen=True, slots=True)
class DecodeTask:
    absolute_path: str
    relative_path: str
    split: str
    transfer_syntax_uid: str
    rows: int | None
    columns: int | None
    number_of_frames: int | None
    samples_per_pixel: int | None


MANIFEST_SCHEMA = pa.schema(
    [
        ("relative_path", pa.string()),
        ("split", pa.string()),
        ("file_size", pa.int64()),
        ("is_symlink", pa.bool_()),
        ("path_structure_ok", pa.bool_()),
        ("path_study_uid", pa.string()),
        ("path_series_uid", pa.string()),
        ("path_sop_uid", pa.string()),
        ("series_in_metadata", pa.bool_()),
        ("anatomical_plane", pa.string()),
        ("fluid_sensitive", pa.int8()),
        ("fat_suppression", pa.int8()),
        ("study_uid", pa.string()),
        ("series_uid", pa.string()),
        ("sop_uid", pa.string()),
        ("sop_class_uid", pa.string()),
        ("instance_number", pa.int64()),
        ("modality", pa.string()),
        ("manufacturer", pa.string()),
        ("manufacturer_model_name", pa.string()),
        ("magnetic_field_strength", pa.float64()),
        ("transfer_syntax_uid", pa.string()),
        ("transfer_syntax_name", pa.string()),
        ("rows", pa.int64()),
        ("columns", pa.int64()),
        ("samples_per_pixel", pa.int64()),
        ("photometric_interpretation", pa.string()),
        ("number_of_frames", pa.int64()),
        ("bits_allocated", pa.int64()),
        ("bits_stored", pa.int64()),
        ("high_bit", pa.int64()),
        ("pixel_representation", pa.int64()),
        ("pixel_spacing_row", pa.float64()),
        ("pixel_spacing_column", pa.float64()),
        ("slice_thickness", pa.float64()),
        ("spacing_between_slices", pa.float64()),
        ("image_position_x", pa.float64()),
        ("image_position_y", pa.float64()),
        ("image_position_z", pa.float64()),
        ("image_orientation_row_x", pa.float64()),
        ("image_orientation_row_y", pa.float64()),
        ("image_orientation_row_z", pa.float64()),
        ("image_orientation_column_x", pa.float64()),
        ("image_orientation_column_y", pa.float64()),
        ("image_orientation_column_z", pa.float64()),
        ("orientation_row_norm", pa.float64()),
        ("orientation_column_norm", pa.float64()),
        ("orientation_dot_product", pa.float64()),
        ("slice_location", pa.float64()),
        ("slice_coordinate", pa.float64()),
        ("geometry_status", pa.string()),
        ("rescale_slope", pa.float64()),
        ("rescale_intercept", pa.float64()),
        ("study_uid_matches_path", pa.bool_()),
        ("series_uid_matches_path", pa.bool_()),
        ("sop_uid_matches_filename", pa.bool_()),
        ("header_status", pa.string()),
        ("error_class", pa.string()),
        ("error_message", pa.string()),
    ]
)

DECODE_SCHEMA = pa.schema(
    [
        ("relative_path", pa.string()),
        ("split", pa.string()),
        ("transfer_syntax_uid", pa.string()),
        ("decode_status", pa.string()),
        ("array_shape", pa.string()),
        ("array_dtype", pa.string()),
        ("array_elements", pa.int64()),
        ("array_bytes", pa.int64()),
        ("pixel_min", pa.float64()),
        ("pixel_max", pa.float64()),
        ("all_finite", pa.bool_()),
        ("shape_matches_header", pa.bool_()),
        ("decode_seconds", pa.float64()),
        ("error_class", pa.string()),
        ("error_message", pa.string()),
    ]
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _log(message: str) -> None:
    print(
        f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}] {message}",
        flush=True,
    )


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_roots(competition_root: Path, output_root: Path) -> None:
    """Reject path relationships that could mix raw and derived data."""
    competition_root = competition_root.resolve()
    output_root = output_root.resolve()
    if not competition_root.is_dir():
        raise AuditConfigurationError(
            f"Competition root does not exist or is not a directory: {competition_root}"
        )
    if competition_root == output_root:
        raise AuditConfigurationError(
            "Output root must differ from the competition root"
        )
    if _within(output_root, competition_root):
        raise AuditConfigurationError(
            "Output root must not be inside the read-only competition-data tree"
        )
    if _within(competition_root, output_root):
        raise AuditConfigurationError(
            "Output root must not be an ancestor of the competition-data tree"
        )
    for directory in SPLIT_DIRECTORIES.values():
        split_root = competition_root / directory
        if not split_root.is_dir():
            raise AuditConfigurationError(
                f"Missing extracted DICOM directory: {split_root}"
            )
    for filename in SERIES_CSVS.values():
        csv_path = competition_root / filename
        if not csv_path.is_file():
            raise AuditConfigurationError(f"Missing series metadata CSV: {csv_path}")


def _safe_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _float_values(value: Any, expected_length: int) -> list[float] | None:
    if value is None:
        return None
    try:
        values = [_safe_float(item) for item in value]
    except TypeError:
        return None
    if len(values) != expected_length or any(item is None for item in values):
        return None
    return [float(item) for item in values if item is not None]


def _error_message(error: BaseException, absolute_path: str) -> str:
    message = " ".join(str(error).replace(absolute_path, "<DICOM>").split())
    return message[:500]


def _geometry(
    image_position: list[float] | None,
    image_orientation: list[float] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "image_position_x": None,
        "image_position_y": None,
        "image_position_z": None,
        "image_orientation_row_x": None,
        "image_orientation_row_y": None,
        "image_orientation_row_z": None,
        "image_orientation_column_x": None,
        "image_orientation_column_y": None,
        "image_orientation_column_z": None,
        "orientation_row_norm": None,
        "orientation_column_norm": None,
        "orientation_dot_product": None,
        "slice_coordinate": None,
        "geometry_status": "missing",
    }
    if image_position is not None:
        result.update(
            {
                "image_position_x": image_position[0],
                "image_position_y": image_position[1],
                "image_position_z": image_position[2],
            }
        )
    if image_orientation is not None:
        row = image_orientation[:3]
        column = image_orientation[3:]
        row_norm = math.sqrt(sum(item * item for item in row))
        column_norm = math.sqrt(sum(item * item for item in column))
        dot_product = sum(a * b for a, b in zip(row, column, strict=True))
        result.update(
            {
                "image_orientation_row_x": row[0],
                "image_orientation_row_y": row[1],
                "image_orientation_row_z": row[2],
                "image_orientation_column_x": column[0],
                "image_orientation_column_y": column[1],
                "image_orientation_column_z": column[2],
                "orientation_row_norm": row_norm,
                "orientation_column_norm": column_norm,
                "orientation_dot_product": dot_product,
            }
        )
    if image_position is None or image_orientation is None:
        return result

    row = image_orientation[:3]
    column = image_orientation[3:]
    row_norm = float(result["orientation_row_norm"])
    column_norm = float(result["orientation_column_norm"])
    dot_product = float(result["orientation_dot_product"])
    if row_norm <= 1e-8 or column_norm <= 1e-8:
        result["geometry_status"] = "invalid_zero_vector"
        return result
    if abs(row_norm - 1.0) > 1e-3 or abs(column_norm - 1.0) > 1e-3:
        result["geometry_status"] = "invalid_nonunit_vectors"
        return result
    normalized_dot = abs(dot_product / (row_norm * column_norm))
    if normalized_dot > 1e-3:
        result["geometry_status"] = "invalid_nonorthogonal"
        return result

    normal = (
        row[1] * column[2] - row[2] * column[1],
        row[2] * column[0] - row[0] * column[2],
        row[0] * column[1] - row[1] * column[0],
    )
    normal_norm = math.sqrt(sum(item * item for item in normal))
    if normal_norm <= 1e-8:
        result["geometry_status"] = "invalid_zero_normal"
        return result
    unit_normal = tuple(item / normal_norm for item in normal)
    result["slice_coordinate"] = sum(
        position * direction
        for position, direction in zip(image_position, unit_normal, strict=True)
    )
    result["geometry_status"] = "ok"
    return result


def read_header(task: HeaderTask) -> dict[str, Any]:
    """Read selected metadata from one DICOM without reading Pixel Data."""
    row: dict[str, Any] = {
        "relative_path": task.relative_path,
        "split": task.split,
        "file_size": task.file_size,
        "is_symlink": task.is_symlink,
        "path_structure_ok": task.path_structure_ok,
        "path_study_uid": task.path_study_uid,
        "path_series_uid": task.path_series_uid,
        "path_sop_uid": task.path_sop_uid,
        "header_status": "unread",
        "error_class": None,
        "error_message": None,
    }
    if task.is_symlink:
        row.update(
            {
                "header_status": "symlink_refused",
                "error_class": "SymlinkRefused",
                "error_message": "Raw DICOM path is a symbolic link; target was not opened",
            }
        )
        return row

    try:
        dataset = dcmread(
            task.absolute_path,
            stop_before_pixels=True,
            force=False,
            specific_tags=HEADER_TAGS,
        )
        transfer_syntax = getattr(dataset.file_meta, "TransferSyntaxUID", None)
        transfer_syntax_uid = _safe_string(transfer_syntax)
        transfer_syntax_name = (
            _safe_string(getattr(transfer_syntax, "name", None))
            if transfer_syntax is not None
            else None
        )
        study_uid = _safe_string(getattr(dataset, "StudyInstanceUID", None))
        series_uid = _safe_string(getattr(dataset, "SeriesInstanceUID", None))
        sop_uid = _safe_string(getattr(dataset, "SOPInstanceUID", None))
        rows = _safe_int(getattr(dataset, "Rows", None))
        columns = _safe_int(getattr(dataset, "Columns", None))
        pixel_spacing = _float_values(getattr(dataset, "PixelSpacing", None), 2)
        image_position = _float_values(
            getattr(dataset, "ImagePositionPatient", None), 3
        )
        image_orientation = _float_values(
            getattr(dataset, "ImageOrientationPatient", None), 6
        )
        row.update(
            {
                "study_uid": study_uid,
                "series_uid": series_uid,
                "sop_uid": sop_uid,
                "sop_class_uid": _safe_string(getattr(dataset, "SOPClassUID", None)),
                "instance_number": _safe_int(getattr(dataset, "InstanceNumber", None)),
                "modality": _safe_string(getattr(dataset, "Modality", None)),
                "manufacturer": _safe_string(getattr(dataset, "Manufacturer", None)),
                "manufacturer_model_name": _safe_string(
                    getattr(dataset, "ManufacturerModelName", None)
                ),
                "magnetic_field_strength": _safe_float(
                    getattr(dataset, "MagneticFieldStrength", None)
                ),
                "transfer_syntax_uid": transfer_syntax_uid,
                "transfer_syntax_name": transfer_syntax_name,
                "rows": rows,
                "columns": columns,
                "samples_per_pixel": _safe_int(
                    getattr(dataset, "SamplesPerPixel", None)
                ),
                "photometric_interpretation": _safe_string(
                    getattr(dataset, "PhotometricInterpretation", None)
                ),
                "number_of_frames": _safe_int(getattr(dataset, "NumberOfFrames", None)),
                "bits_allocated": _safe_int(getattr(dataset, "BitsAllocated", None)),
                "bits_stored": _safe_int(getattr(dataset, "BitsStored", None)),
                "high_bit": _safe_int(getattr(dataset, "HighBit", None)),
                "pixel_representation": _safe_int(
                    getattr(dataset, "PixelRepresentation", None)
                ),
                "pixel_spacing_row": pixel_spacing[0] if pixel_spacing else None,
                "pixel_spacing_column": pixel_spacing[1] if pixel_spacing else None,
                "slice_thickness": _safe_float(
                    getattr(dataset, "SliceThickness", None)
                ),
                "spacing_between_slices": _safe_float(
                    getattr(dataset, "SpacingBetweenSlices", None)
                ),
                "slice_location": _safe_float(getattr(dataset, "SliceLocation", None)),
                "rescale_slope": _safe_float(getattr(dataset, "RescaleSlope", None)),
                "rescale_intercept": _safe_float(
                    getattr(dataset, "RescaleIntercept", None)
                ),
                "study_uid_matches_path": bool(
                    study_uid
                    and task.path_study_uid
                    and study_uid == task.path_study_uid
                ),
                "series_uid_matches_path": bool(
                    series_uid
                    and task.path_series_uid
                    and series_uid == task.path_series_uid
                ),
                "sop_uid_matches_filename": bool(
                    sop_uid and task.path_sop_uid and sop_uid == task.path_sop_uid
                ),
            }
        )
        row.update(_geometry(image_position, image_orientation))

        required = {
            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": series_uid,
            "SOPInstanceUID": sop_uid,
            "TransferSyntaxUID": transfer_syntax_uid,
            "Rows": rows,
            "Columns": columns,
            "SamplesPerPixel": row.get("samples_per_pixel"),
            "PhotometricInterpretation": row.get("photometric_interpretation"),
            "BitsAllocated": row.get("bits_allocated"),
            "BitsStored": row.get("bits_stored"),
            "HighBit": row.get("high_bit"),
            "PixelRepresentation": row.get("pixel_representation"),
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            row.update(
                {
                    "header_status": "missing_required_tags",
                    "error_class": "MissingRequiredTags",
                    "error_message": "Missing required tags: " + ", ".join(missing),
                }
            )
        else:
            invalid = []
            if rows is not None and rows <= 0:
                invalid.append("Rows")
            if columns is not None and columns <= 0:
                invalid.append("Columns")
            samples = int(row["samples_per_pixel"])
            bits_allocated = int(row["bits_allocated"])
            bits_stored = int(row["bits_stored"])
            high_bit = int(row["high_bit"])
            pixel_representation = int(row["pixel_representation"])
            if samples <= 0:
                invalid.append("SamplesPerPixel")
            if bits_allocated != 1 and (bits_allocated <= 0 or bits_allocated % 8 != 0):
                invalid.append("BitsAllocated")
            if bits_stored <= 0 or bits_stored > bits_allocated:
                invalid.append("BitsStored")
            if high_bit != bits_stored - 1:
                invalid.append("HighBit")
            if pixel_representation not in (0, 1):
                invalid.append("PixelRepresentation")
            if invalid:
                row.update(
                    {
                        "header_status": "invalid_image_pixel_metadata",
                        "error_class": "InvalidImagePixelMetadata",
                        "error_message": "Invalid values for: " + ", ".join(invalid),
                    }
                )
            else:
                row["header_status"] = "ok"
    except Exception as error:  # noqa: BLE001 - record per-file audit failures
        row.update(
            {
                "header_status": "read_error",
                "error_class": type(error).__name__,
                "error_message": _error_message(error, task.absolute_path),
            }
        )
    return row


def decode_pixels(task: DecodeTask) -> dict[str, Any]:
    """Decode one DICOM to a NumPy array without changing the source file."""
    started = time.monotonic()
    row: dict[str, Any] = {
        "relative_path": task.relative_path,
        "split": task.split,
        "transfer_syntax_uid": task.transfer_syntax_uid,
        "decode_status": "unread",
        "array_shape": None,
        "array_dtype": None,
        "array_elements": None,
        "array_bytes": None,
        "pixel_min": None,
        "pixel_max": None,
        "all_finite": None,
        "shape_matches_header": None,
        "decode_seconds": None,
        "error_class": None,
        "error_message": None,
    }
    try:
        array = pixel_array(task.absolute_path, raw=True)
        shape = tuple(int(value) for value in array.shape)
        frames = task.number_of_frames or 1
        samples = task.samples_per_pixel or 1
        expected_shape: tuple[int, ...] | None = None
        if task.rows is not None and task.columns is not None:
            if frames > 1 and samples > 1:
                expected_shape = (frames, task.rows, task.columns, samples)
            elif frames > 1:
                expected_shape = (frames, task.rows, task.columns)
            elif samples > 1:
                expected_shape = (task.rows, task.columns, samples)
            else:
                expected_shape = (task.rows, task.columns)
        if array.size == 0:
            raise ValueError("Decoded pixel array is empty")
        all_finite = bool(np.isfinite(array).all())
        row.update(
            {
                "decode_status": "ok",
                "array_shape": "x".join(str(value) for value in shape),
                "array_dtype": str(array.dtype),
                "array_elements": int(array.size),
                "array_bytes": int(array.nbytes),
                "pixel_min": float(np.min(array)),
                "pixel_max": float(np.max(array)),
                "all_finite": all_finite,
                "shape_matches_header": (
                    shape == expected_shape if expected_shape is not None else None
                ),
            }
        )
    except Exception as error:  # noqa: BLE001 - record per-file audit failures
        row.update(
            {
                "decode_status": "decode_error",
                "error_class": type(error).__name__,
                "error_message": _error_message(error, task.absolute_path),
            }
        )
    row["decode_seconds"] = time.monotonic() - started
    return row


def _parse_binary_flag(value: str, column: str, csv_path: Path) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise AuditConfigurationError(
            f"Invalid {column} value in {csv_path.name}: {value!r}"
        ) from error
    if parsed not in (0, 1):
        raise AuditConfigurationError(
            f"Expected 0/1 for {column} in {csv_path.name}, got {parsed}"
        )
    return parsed


def load_series_metadata(
    competition_root: Path,
) -> dict[tuple[str, str, str], SeriesMetadata]:
    """Load only non-report series metadata and reject duplicates/schema drift."""
    metadata: dict[tuple[str, str, str], SeriesMetadata] = {}
    for split, filename in SERIES_CSVS.items():
        csv_path = competition_root / filename
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            columns = set(reader.fieldnames or [])
            missing_columns = REQUIRED_SERIES_COLUMNS - columns
            if missing_columns:
                raise AuditConfigurationError(
                    f"{filename} is missing columns: {sorted(missing_columns)}"
                )
            for row in reader:
                study_uid = (row.get("StudyInstanceUID") or "").strip()
                series_uid = (row.get("SeriesInstanceUID") or "").strip()
                if not study_uid or not series_uid:
                    raise AuditConfigurationError(
                        f"{filename} contains a blank study or series UID"
                    )
                key = (split, study_uid, series_uid)
                if key in metadata:
                    raise AuditConfigurationError(
                        f"{filename} contains a duplicate study/series pair"
                    )
                metadata[key] = SeriesMetadata(
                    anatomical_plane=(row.get("Anatomical_Plane") or "").strip(),
                    fluid_sensitive=_parse_binary_flag(
                        row.get("Fluid_Sensitive") or "",
                        "Fluid_Sensitive",
                        csv_path,
                    ),
                    fat_suppression=_parse_binary_flag(
                        row.get("Fat_Suppression") or "",
                        "Fat_Suppression",
                        csv_path,
                    ),
                )
    return metadata


def _raise_walk_error(error: OSError) -> None:
    raise error


def discover_dicoms(
    competition_root: Path,
    unexpected_writer: csv.DictWriter,
) -> Iterator[HeaderTask]:
    """Yield DICOM paths deterministically and record unexpected files."""
    for split, directory in SPLIT_DIRECTORIES.items():
        split_root = competition_root / directory
        for dirpath, dirnames, filenames in os.walk(
            split_root, topdown=True, onerror=_raise_walk_error, followlinks=False
        ):
            dirnames.sort()
            filenames.sort()
            current = Path(dirpath)
            for dirname in list(dirnames):
                directory_path = current / dirname
                if not directory_path.is_symlink():
                    continue
                dirnames.remove(dirname)
                unexpected_writer.writerow(
                    {
                        "relative_path": directory_path.relative_to(
                            competition_root
                        ).as_posix(),
                        "file_size": directory_path.lstat().st_size,
                        "is_symlink": 1,
                        "entry_kind": "directory_symlink",
                    }
                )
            for filename in filenames:
                path = current / filename
                relative_path = path.relative_to(competition_root).as_posix()
                is_symlink = path.is_symlink()
                try:
                    file_size = path.lstat().st_size
                except OSError:
                    file_size = -1
                if path.suffix.lower() != ".dcm":
                    unexpected_writer.writerow(
                        {
                            "relative_path": relative_path,
                            "file_size": file_size,
                            "is_symlink": int(is_symlink),
                            "entry_kind": "file",
                        }
                    )
                    continue

                parts = path.relative_to(split_root).parts
                structure_ok = len(parts) == 3
                yield HeaderTask(
                    absolute_path=str(path),
                    relative_path=relative_path,
                    split=split,
                    file_size=file_size,
                    is_symlink=is_symlink,
                    path_structure_ok=structure_ok,
                    path_study_uid=parts[0] if len(parts) >= 1 else None,
                    path_series_uid=parts[1] if len(parts) >= 2 else None,
                    path_sop_uid=Path(parts[-1]).stem if len(parts) >= 1 else None,
                )


def _batched(iterable: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _batch_digest(tasks: Sequence[HeaderTask | DecodeTask]) -> str:
    digest = hashlib.sha256()
    for task in tasks:
        digest.update(task.relative_path.encode("utf-8"))
        digest.update(b"\0")
        if isinstance(task, HeaderTask):
            digest.update(str(task.file_size).encode("ascii"))
        else:
            digest.update(task.transfer_syntax_uid.encode("ascii", errors="replace"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_temp_path(final_path: Path) -> Path:
    return final_path.with_name(
        f".{final_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(path)
    with temp_path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)


def _write_parquet_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    schema: pa.Schema,
    part_metadata: dict[str, str],
) -> None:
    if path.exists():
        raise AuditConfigurationError(f"Refusing to overwrite completed part: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    metadata = dict(schema.metadata or {})
    metadata.update(
        {f"rsna.{key}".encode(): value.encode() for key, value in part_metadata.items()}
    )
    table = table.replace_schema_metadata(metadata)
    temp_path = _atomic_temp_path(path)
    pq.write_table(
        table,
        temp_path,
        compression="zstd",
        version="2.6",
        use_dictionary=True,
        write_statistics=True,
    )
    os.replace(temp_path, path)


def _part_metadata(path: Path) -> tuple[pa.Schema, dict[str, str], int]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    metadata = {
        key.decode(): value.decode()
        for key, value in (schema.metadata or {}).items()
        if key.startswith(b"rsna.")
    }
    return schema.remove_metadata(), metadata, parquet.metadata.num_rows


def _validate_or_recover_part(
    part_path: Path,
    checkpoint_path: Path,
    schema: pa.Schema,
    digest: str,
    row_count: int,
    first_path: str,
    last_path: str,
) -> bool:
    """Return True for a valid completed part; recover a missing checkpoint."""
    if not part_path.exists():
        if checkpoint_path.exists():
            raise AuditConfigurationError(
                f"Checkpoint exists without its Parquet part: {checkpoint_path}"
            )
        return False

    actual_schema, metadata, actual_rows = _part_metadata(part_path)
    if not actual_schema.equals(schema, check_metadata=False):
        raise AuditConfigurationError(f"Schema mismatch in completed part: {part_path}")
    expected_metadata = {
        "rsna.format_version": str(FORMAT_VERSION),
        "rsna.batch_digest": digest,
        "rsna.row_count": str(row_count),
        "rsna.first_path": first_path,
        "rsna.last_path": last_path,
    }
    if actual_rows != row_count or any(
        metadata.get(key) != value for key, value in expected_metadata.items()
    ):
        raise AuditConfigurationError(
            f"Completed part does not match current inputs: {part_path}"
        )
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "batch_digest": digest,
        "row_count": row_count,
        "first_path": first_path,
        "last_path": last_path,
    }
    if checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as file:
            if json.load(file) != checkpoint:
                raise AuditConfigurationError(
                    f"Checkpoint does not match its Parquet part: {checkpoint_path}"
                )
    else:
        _write_json_atomic(checkpoint_path, checkpoint)
    return True


def _enrich_header_row(
    row: dict[str, Any],
    series_metadata: dict[tuple[str, str, str], SeriesMetadata],
) -> dict[str, Any]:
    key = (
        str(row.get("split") or ""),
        str(row.get("path_study_uid") or ""),
        str(row.get("path_series_uid") or ""),
    )
    metadata = series_metadata.get(key)
    row["series_in_metadata"] = metadata is not None
    row["anatomical_plane"] = metadata.anatomical_plane if metadata else None
    row["fluid_sensitive"] = metadata.fluid_sensitive if metadata else None
    row["fat_suppression"] = metadata.fat_suppression if metadata else None
    return row


def _map_rows(
    function: Any,
    tasks: list[Any],
    executor: ProcessPoolExecutor | None,
    chunksize: int,
) -> list[dict[str, Any]]:
    if executor is None:
        return [function(task) for task in tasks]
    return list(executor.map(function, tasks, chunksize=chunksize))


def _create_executor(workers: int) -> ProcessPoolExecutor | None:
    if workers == 1:
        return None
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    )


def _prepare_output_root(
    output_root: Path,
    resume: bool,
    run_config: dict[str, Any],
) -> None:
    config_path = output_root / "inventory_config.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not resume:
            raise AuditConfigurationError(
                f"Output root is not empty; use --resume or a new path: {output_root}"
            )
        if not config_path.is_file():
            raise AuditConfigurationError(
                f"Cannot resume without inventory_config.json: {output_root}"
            )
        with config_path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        if existing != run_config:
            raise AuditConfigurationError(
                "Resume configuration differs from the existing audit run"
            )
        return
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(config_path, run_config)


def run_inventory(
    competition_root: Path,
    output_root: Path,
    series_metadata: dict[tuple[str, str, str], SeriesMetadata],
    workers: int,
    batch_size: int,
    worker_chunksize: int,
) -> tuple[list[Path], int]:
    """Create or resume partitioned header manifests."""
    parts_root = output_root / "manifest_parts"
    checkpoints_root = output_root / "manifest_checkpoints"
    parts_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    unexpected_path = output_root / "unexpected_files.csv"
    unexpected_temp = _atomic_temp_path(unexpected_path)
    started = time.monotonic()
    total_rows = 0
    part_paths: list[Path] = []

    executor = _create_executor(workers)
    try:
        with unexpected_temp.open("x", encoding="utf-8", newline="") as unexpected_file:
            unexpected_writer = csv.DictWriter(
                unexpected_file,
                fieldnames=[
                    "relative_path",
                    "file_size",
                    "is_symlink",
                    "entry_kind",
                ],
            )
            unexpected_writer.writeheader()
            tasks = discover_dicoms(competition_root, unexpected_writer)
            for part_index, batch in enumerate(_batched(tasks, batch_size)):
                part_path = parts_root / f"part-{part_index:06d}.parquet"
                checkpoint_path = checkpoints_root / f"part-{part_index:06d}.json"
                digest = _batch_digest(batch)
                first_path = batch[0].relative_path
                last_path = batch[-1].relative_path
                if _validate_or_recover_part(
                    part_path,
                    checkpoint_path,
                    MANIFEST_SCHEMA,
                    digest,
                    len(batch),
                    first_path,
                    last_path,
                ):
                    action = "resumed"
                else:
                    rows = _map_rows(read_header, batch, executor, worker_chunksize)
                    rows = [_enrich_header_row(row, series_metadata) for row in rows]
                    part_metadata = {
                        "format_version": str(FORMAT_VERSION),
                        "batch_digest": digest,
                        "row_count": str(len(batch)),
                        "first_path": first_path,
                        "last_path": last_path,
                    }
                    _write_parquet_atomic(
                        part_path, rows, MANIFEST_SCHEMA, part_metadata
                    )
                    _validate_or_recover_part(
                        part_path,
                        checkpoint_path,
                        MANIFEST_SCHEMA,
                        digest,
                        len(batch),
                        first_path,
                        last_path,
                    )
                    action = "wrote"
                part_paths.append(part_path)
                total_rows += len(batch)
                elapsed = max(time.monotonic() - started, 1e-9)
                _log(
                    f"Inventory {action} {part_path.name}: {total_rows:,} files "
                    f"({total_rows / elapsed:,.0f} files/s overall)"
                )
            unexpected_file.flush()
            os.fsync(unexpected_file.fileno())
        os.replace(unexpected_temp, unexpected_path)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    existing_parts = sorted(parts_root.glob("part-*.parquet"))
    if existing_parts != part_paths:
        raise AuditConfigurationError(
            "Manifest directory contains parts outside the current deterministic run"
        )
    return part_paths, total_rows


def _iter_parquet_rows(
    part_paths: Sequence[Path], columns: Sequence[str]
) -> Iterator[dict[str, Any]]:
    for path in part_paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=list(columns), batch_size=65_536):
            yield from batch.to_pylist()


def _distribution(values: Iterable[int]) -> dict[str, int | float | None]:
    ordered = sorted(values)
    if not ordered:
        return {"min": None, "median": None, "p95": None, "max": None}
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "min": ordered[0],
        "median": float(statistics.median(ordered)),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _count_unexpected_files(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def summarize_inventory(
    part_paths: Sequence[Path],
    expected_series: set[tuple[str, str, str]],
    unexpected_files_path: Path,
    expected_dicom_count: int | None,
) -> dict[str, Any]:
    columns = [
        "split",
        "file_size",
        "path_structure_ok",
        "path_study_uid",
        "path_series_uid",
        "series_in_metadata",
        "sop_uid",
        "instance_number",
        "modality",
        "transfer_syntax_uid",
        "transfer_syntax_name",
        "rows",
        "columns",
        "geometry_status",
        "slice_coordinate",
        "image_orientation_row_x",
        "image_orientation_row_y",
        "image_orientation_row_z",
        "image_orientation_column_x",
        "image_orientation_column_y",
        "image_orientation_column_z",
        "study_uid_matches_path",
        "series_uid_matches_path",
        "sop_uid_matches_filename",
        "header_status",
        "error_class",
    ]
    total = 0
    total_bytes = 0
    split_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    transfer_counts: Counter[tuple[str, str]] = Counter()
    modality_counts: Counter[str] = Counter()
    dimension_counts: Counter[tuple[int, int]] = Counter()
    geometry_counts: Counter[str] = Counter()
    error_class_counts: Counter[str] = Counter()
    series_slice_counts: list[int] = []
    series_audit_counts: Counter[str] = Counter()
    actual_series: set[tuple[str, str, str]] = set()
    seen_sop_uids: set[str] = set()
    duplicate_sop_uids = 0
    path_structure_errors = 0
    series_not_in_metadata = 0
    study_uid_mismatches = 0
    series_uid_mismatches = 0
    sop_uid_mismatches = 0
    noncontiguous_series = 0
    current_series_key: tuple[str, str, str] | None = None
    current_series: SeriesAccumulator | None = None

    def finish_series() -> None:
        if current_series is None:
            return
        result = current_series.result()
        series_slice_counts.append(int(result["count"]))
        series_audit_counts[f"sort_{result['sort_strategy']}"] += 1
        for field_name in (
            "inconsistent_orientation",
            "inconsistent_dimensions",
            "instance_coordinate_order_inconsistent",
            "nonuniform_slice_spacing",
        ):
            if result[field_name]:
                series_audit_counts[f"series_with_{field_name}"] += 1
        for field_name in (
            "duplicate_slice_coordinates",
            "duplicate_instance_numbers",
        ):
            duplicate_count = int(result[field_name])
            if duplicate_count:
                series_audit_counts[f"series_with_{field_name}"] += 1
                series_audit_counts[f"total_{field_name}"] += duplicate_count

    for row in _iter_parquet_rows(part_paths, columns):
        total += 1
        total_bytes += int(row["file_size"] or 0)
        split = str(row["split"] or "<missing>")
        split_counts[split] += 1
        status_counts[str(row["header_status"] or "<missing>")] += 1
        modality_counts[str(row["modality"] or "<missing>")] += 1
        geometry_counts[str(row["geometry_status"] or "<missing>")] += 1
        if row["error_class"]:
            error_class_counts[str(row["error_class"])] += 1
        transfer_counts[
            (
                str(row["transfer_syntax_uid"] or "<missing>"),
                str(row["transfer_syntax_name"] or "<missing>"),
            )
        ] += 1
        if row["rows"] is not None and row["columns"] is not None:
            dimension_counts[(int(row["rows"]), int(row["columns"]))] += 1
        if not row["path_structure_ok"]:
            path_structure_errors += 1
        if not row["series_in_metadata"]:
            series_not_in_metadata += 1
        if row["study_uid_matches_path"] is False:
            study_uid_mismatches += 1
        if row["series_uid_matches_path"] is False:
            series_uid_mismatches += 1
        if row["sop_uid_matches_filename"] is False:
            sop_uid_mismatches += 1

        key = None
        if row["path_study_uid"] and row["path_series_uid"]:
            key = (split, row["path_study_uid"], row["path_series_uid"])
        if key != current_series_key:
            finish_series()
            current_series_key = key
            current_series = SeriesAccumulator() if key is not None else None
            if key is not None:
                if key in actual_series:
                    noncontiguous_series += 1
                actual_series.add(key)
        if current_series is not None:
            current_series.add(row)
        sop_uid = row["sop_uid"]
        if sop_uid:
            if sop_uid in seen_sop_uids:
                duplicate_sop_uids += 1
            else:
                seen_sop_uids.add(sop_uid)

    finish_series()

    missing_series = expected_series - actual_series
    unexpected_series = actual_series - expected_series
    unexpected_files = _count_unexpected_files(unexpected_files_path)
    checks = {
        "dicom_count_mismatch": (
            int(total != expected_dicom_count)
            if expected_dicom_count is not None
            else 0
        ),
        "unexpected_non_dicom_files": unexpected_files,
        "header_failures": total - status_counts.get("ok", 0),
        "path_structure_errors": path_structure_errors,
        "files_from_series_not_in_csv": series_not_in_metadata,
        "series_in_csv_without_files": len(missing_series),
        "series_on_disk_not_in_csv": len(unexpected_series),
        "study_uid_path_mismatches": study_uid_mismatches,
        "series_uid_path_mismatches": series_uid_mismatches,
        "sop_uid_filename_mismatches": sop_uid_mismatches,
        "duplicate_sop_uids": duplicate_sop_uids,
        "noncontiguous_series_paths": noncontiguous_series,
    }
    status = "pass" if all(value == 0 for value in checks.values()) else "fail"
    transfer_syntaxes = [
        {"uid": uid, "name": name, "count": count}
        for (uid, name), count in sorted(transfer_counts.items())
    ]
    top_dimensions = [
        {"rows": rows, "columns": columns_, "count": count}
        for (rows, columns_), count in dimension_counts.most_common(20)
    ]
    part_digests = []
    for part_path in part_paths:
        _, metadata, _ = _part_metadata(part_path)
        part_digests.append(metadata["rsna.batch_digest"])
    fingerprint = hashlib.sha256("".join(part_digests).encode("ascii")).hexdigest()
    return {
        "status": status,
        "generated_at": _now_iso(),
        "manifest_format_version": FORMAT_VERSION,
        "manifest_fingerprint": fingerprint,
        "manifest_parts": len(part_paths),
        "dicom_files": total,
        "dicom_bytes": total_bytes,
        "expected_dicom_files": expected_dicom_count,
        "split_counts": dict(sorted(split_counts.items())),
        "header_status_counts": dict(sorted(status_counts.items())),
        "header_error_class_counts": dict(sorted(error_class_counts.items())),
        "transfer_syntaxes": transfer_syntaxes,
        "modality_counts": dict(sorted(modality_counts.items())),
        "geometry_status_counts": dict(sorted(geometry_counts.items())),
        "top_image_dimensions": top_dimensions,
        "metadata_series": len(expected_series),
        "series_with_files": len(actual_series),
        "slices_per_series": _distribution(series_slice_counts),
        "series_geometry_audit": {
            "geometry_sort_ready": series_audit_counts["sort_geometry"],
            "instance_number_fallback": series_audit_counts[
                "sort_instance_number_fallback"
            ],
            "not_sortable": series_audit_counts["sort_not_sortable"],
            "series_with_duplicate_slice_coordinates": series_audit_counts[
                "series_with_duplicate_slice_coordinates"
            ],
            "total_duplicate_slice_coordinates": series_audit_counts[
                "total_duplicate_slice_coordinates"
            ],
            "series_with_duplicate_instance_numbers": series_audit_counts[
                "series_with_duplicate_instance_numbers"
            ],
            "total_duplicate_instance_numbers": series_audit_counts[
                "total_duplicate_instance_numbers"
            ],
            "series_with_inconsistent_orientation": series_audit_counts[
                "series_with_inconsistent_orientation"
            ],
            "series_with_inconsistent_dimensions": series_audit_counts[
                "series_with_inconsistent_dimensions"
            ],
            "series_with_instance_coordinate_order_inconsistent": (
                series_audit_counts[
                    "series_with_instance_coordinate_order_inconsistent"
                ]
            ),
            "series_with_nonuniform_slice_spacing": series_audit_counts[
                "series_with_nonuniform_slice_spacing"
            ],
        },
        "checks": checks,
        "scope_note": (
            "Header inventory covers every extracted .dcm file. Pixel decoding is "
            "reported separately and may be sampled or exhaustive."
        ),
    }


def _decode_task_from_row(competition_root: Path, row: dict[str, Any]) -> DecodeTask:
    return DecodeTask(
        absolute_path=str(competition_root / row["relative_path"]),
        relative_path=row["relative_path"],
        split=row["split"],
        transfer_syntax_uid=row["transfer_syntax_uid"],
        rows=_safe_int(row["rows"]),
        columns=_safe_int(row["columns"]),
        number_of_frames=_safe_int(row["number_of_frames"]),
        samples_per_pixel=_safe_int(row["samples_per_pixel"]),
    )


DECODE_SOURCE_COLUMNS = [
    "relative_path",
    "split",
    "transfer_syntax_uid",
    "rows",
    "columns",
    "number_of_frames",
    "samples_per_pixel",
    "header_status",
]


def select_sample_decode_tasks(
    competition_root: Path,
    part_paths: Sequence[Path],
    samples_per_transfer_syntax: int,
    decode_all_example_test: bool,
) -> list[DecodeTask]:
    """Select deterministic train samples per syntax plus all example-test files."""
    heaps: dict[str, list[tuple[int, str, DecodeTask]]] = {}
    selected: dict[str, DecodeTask] = {}
    for row in _iter_parquet_rows(part_paths, DECODE_SOURCE_COLUMNS):
        if row["header_status"] != "ok" or not row["transfer_syntax_uid"]:
            continue
        task = _decode_task_from_row(competition_root, row)
        if decode_all_example_test and task.split == "test":
            selected[task.relative_path] = task
        if task.split != "train":
            continue
        score = int.from_bytes(
            hashlib.blake2b(task.relative_path.encode(), digest_size=8).digest(),
            "big",
        )
        heap = heaps.setdefault(task.transfer_syntax_uid, [])
        entry = (-score, task.relative_path, task)
        if len(heap) < samples_per_transfer_syntax:
            heapq.heappush(heap, entry)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    for heap in heaps.values():
        for _, _, task in heap:
            selected[task.relative_path] = task
    return [selected[path] for path in sorted(selected)]


def iter_all_decode_tasks(
    competition_root: Path, part_paths: Sequence[Path]
) -> Iterator[DecodeTask]:
    for row in _iter_parquet_rows(part_paths, DECODE_SOURCE_COLUMNS):
        if row["header_status"] == "ok" and row["transfer_syntax_uid"]:
            yield _decode_task_from_row(competition_root, row)


def _prepare_decode_root(
    decode_root: Path, resume: bool, config: dict[str, Any]
) -> None:
    config_path = decode_root / "decode_config.json"
    if decode_root.exists() and any(decode_root.iterdir()):
        if not resume:
            raise AuditConfigurationError(
                f"Decode output exists; use --resume or a new audit root: {decode_root}"
            )
        if not config_path.is_file():
            raise AuditConfigurationError(
                f"Cannot resume decode without decode_config.json: {decode_root}"
            )
        with config_path.open("r", encoding="utf-8") as file:
            if json.load(file) != config:
                raise AuditConfigurationError(
                    "Decode resume configuration differs from the existing run"
                )
        return
    decode_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(config_path, config)


def run_decode_audit(
    competition_root: Path,
    output_root: Path,
    manifest_parts: Sequence[Path],
    inventory_summary: dict[str, Any],
    mode: str,
    samples_per_transfer_syntax: int,
    decode_all_example_test: bool,
    workers: int,
    batch_size: int,
    worker_chunksize: int,
    resume: bool,
) -> dict[str, Any] | None:
    if mode == "none":
        return None
    suffix = (
        "decode_all"
        if mode == "all"
        else f"decode_sample_{samples_per_transfer_syntax}_test-{int(decode_all_example_test)}"
    )
    decode_root = output_root / suffix
    config = {
        "format_version": FORMAT_VERSION,
        "manifest_fingerprint": inventory_summary["manifest_fingerprint"],
        "mode": mode,
        "samples_per_transfer_syntax": samples_per_transfer_syntax,
        "decode_all_example_test": decode_all_example_test,
        "batch_size": batch_size,
        "pydicom_version": pydicom.__version__,
        "numpy_version": np.__version__,
    }
    _prepare_decode_root(decode_root, resume, config)
    parts_root = decode_root / "parts"
    checkpoints_root = decode_root / "checkpoints"
    parts_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    if mode == "sample":
        candidates: Iterable[DecodeTask] = select_sample_decode_tasks(
            competition_root,
            manifest_parts,
            samples_per_transfer_syntax,
            decode_all_example_test,
        )
    else:
        candidates = iter_all_decode_tasks(competition_root, manifest_parts)

    executor = _create_executor(workers)
    decode_parts: list[Path] = []
    total = 0
    started = time.monotonic()
    try:
        for part_index, batch in enumerate(_batched(candidates, batch_size)):
            part_path = parts_root / f"part-{part_index:06d}.parquet"
            checkpoint_path = checkpoints_root / f"part-{part_index:06d}.json"
            digest = _batch_digest(batch)
            first_path = batch[0].relative_path
            last_path = batch[-1].relative_path
            if _validate_or_recover_part(
                part_path,
                checkpoint_path,
                DECODE_SCHEMA,
                digest,
                len(batch),
                first_path,
                last_path,
            ):
                action = "resumed"
            else:
                rows = _map_rows(decode_pixels, batch, executor, worker_chunksize)
                _write_parquet_atomic(
                    part_path,
                    rows,
                    DECODE_SCHEMA,
                    {
                        "format_version": str(FORMAT_VERSION),
                        "batch_digest": digest,
                        "row_count": str(len(batch)),
                        "first_path": first_path,
                        "last_path": last_path,
                    },
                )
                _validate_or_recover_part(
                    part_path,
                    checkpoint_path,
                    DECODE_SCHEMA,
                    digest,
                    len(batch),
                    first_path,
                    last_path,
                )
                action = "wrote"
            decode_parts.append(part_path)
            total += len(batch)
            elapsed = max(time.monotonic() - started, 1e-9)
            _log(
                f"Decode {action} {part_path.name}: {total:,} files "
                f"({total / elapsed:,.1f} files/s overall)"
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    existing_parts = sorted(parts_root.glob("part-*.parquet"))
    if existing_parts != decode_parts:
        raise AuditConfigurationError(
            "Decode directory contains parts outside the current deterministic run"
        )
    summary = summarize_decode(
        decode_parts,
        mode,
        inventory_summary,
        samples_per_transfer_syntax,
        decode_all_example_test,
    )
    _write_json_atomic(decode_root / "decode_summary.json", summary)
    return summary


def summarize_decode(
    part_paths: Sequence[Path],
    mode: str,
    inventory_summary: dict[str, Any],
    samples_per_transfer_syntax: int,
    decode_all_example_test: bool,
) -> dict[str, Any]:
    columns = [
        "split",
        "transfer_syntax_uid",
        "decode_status",
        "all_finite",
        "shape_matches_header",
        "decode_seconds",
    ]
    attempted = 0
    decoded = 0
    transfer_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    decode_seconds = 0.0
    nonfinite = 0
    shape_mismatches = 0
    for row in _iter_parquet_rows(part_paths, columns):
        attempted += 1
        split_counts[str(row["split"])] += 1
        transfer_counts[str(row["transfer_syntax_uid"])] += 1
        decode_seconds += float(row["decode_seconds"] or 0.0)
        if row["decode_status"] == "ok":
            decoded += 1
        if row["all_finite"] is False:
            nonfinite += 1
        if row["shape_matches_header"] is False:
            shape_mismatches += 1
    inventory_syntaxes = {
        item["uid"]
        for item in inventory_summary["transfer_syntaxes"]
        if item["uid"] != "<missing>"
    }
    missing_syntaxes = inventory_syntaxes - set(transfer_counts)
    checks = {
        "decode_failures": attempted - decoded,
        "nonfinite_arrays": nonfinite,
        "shape_mismatches": shape_mismatches,
        "transfer_syntaxes_without_decode_test": len(missing_syntaxes),
        "exhaustive_decode_count_mismatch": (
            int(attempted != inventory_summary["header_status_counts"].get("ok", 0))
            if mode == "all"
            else 0
        ),
    }
    status = "pass" if all(value == 0 for value in checks.values()) else "fail"
    return {
        "status": status,
        "generated_at": _now_iso(),
        "mode": mode,
        "scope": (
            "every header-valid DICOM"
            if mode == "all"
            else (
                f"deterministic sample of up to {samples_per_transfer_syntax} training "
                "DICOMs per transfer syntax"
                + (" plus every example-test DICOM" if decode_all_example_test else "")
            )
        ),
        "attempted": attempted,
        "decoded": decoded,
        "split_counts": dict(sorted(split_counts.items())),
        "transfer_syntax_counts": dict(sorted(transfer_counts.items())),
        "summed_worker_decode_seconds": decode_seconds,
        "checks": checks,
    }


def environment_summary() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "expected_decoder_availability": decoder_availability(),
    }


def decoder_availability() -> list[dict[str, Any]]:
    """Report pydicom decoder availability for documented competition syntaxes."""
    availability = []
    for transfer_syntax in EXPECTED_TRANSFER_SYNTAXES:
        decoder = get_decoder(transfer_syntax)
        availability.append(
            {
                "uid": str(transfer_syntax),
                "name": transfer_syntax.name,
                "available": bool(decoder.is_available),
                "available_plugins": list(decoder.available_plugins),
            }
        )
    return availability


def require_expected_decoders() -> None:
    unavailable = [
        item for item in decoder_availability() if not bool(item["available"])
    ]
    if unavailable:
        names = ", ".join(str(item["name"]) for item in unavailable)
        raise AuditConfigurationError(
            "Required pydicom pixel decoders are unavailable: " + names
        )


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data"))
    parser = argparse.ArgumentParser(
        description=(
            "Create a read-only, resumable DICOM manifest and pixel-decoder audit. "
            "Raw competition files are never modified."
        )
    )
    parser.add_argument(
        "--competition-root",
        type=Path,
        default=data_root / "raw" / "competition",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=data_root / "manifests" / "dicom_audit_v1",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
    )
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--worker-chunksize", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-dicom-count", type=int)
    parser.add_argument(
        "--decode",
        choices=("none", "sample", "all"),
        default="sample",
        help="Pixel audit scope; header inventory is always exhaustive",
    )
    parser.add_argument("--samples-per-transfer-syntax", type=int, default=64)
    parser.add_argument(
        "--decode-all-example-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _validate_positive_arguments(args: argparse.Namespace) -> None:
    for name in ("workers", "batch_size", "worker_chunksize"):
        if getattr(args, name) < 1:
            raise AuditConfigurationError(
                f"--{name.replace('_', '-')} must be positive"
            )
    if args.samples_per_transfer_syntax < 1:
        raise AuditConfigurationError("--samples-per-transfer-syntax must be positive")
    if args.expected_dicom_count is not None and args.expected_dicom_count < 1:
        raise AuditConfigurationError("--expected-dicom-count must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_positive_arguments(args)
        competition_root = args.competition_root.expanduser().resolve()
        output_root = args.output_root.expanduser().resolve()
        validate_roots(competition_root, output_root)
        if args.decode != "none":
            require_expected_decoders()
        series_metadata = load_series_metadata(competition_root)
        metadata_digests = {
            filename: _file_digest(competition_root / filename)
            for filename in SERIES_CSVS.values()
        }
        run_config = {
            "format_version": FORMAT_VERSION,
            "competition_root": str(competition_root),
            "split_directories": SPLIT_DIRECTORIES,
            "series_csv_sha256": metadata_digests,
            "batch_size": args.batch_size,
            "expected_dicom_count": args.expected_dicom_count,
            "pydicom_version": pydicom.__version__,
            "pyarrow_version": pa.__version__,
        }
        _prepare_output_root(output_root, args.resume, run_config)
        _write_json_atomic(output_root / "environment.json", environment_summary())
        _log(
            f"Starting exhaustive header inventory with {args.workers} worker(s); "
            f"raw root is read-only: {competition_root}"
        )
        manifest_parts, discovered_count = run_inventory(
            competition_root,
            output_root,
            series_metadata,
            args.workers,
            args.batch_size,
            args.worker_chunksize,
        )
        if discovered_count == 0:
            raise AuditConfigurationError("No .dcm files were discovered")
        inventory_summary = summarize_inventory(
            manifest_parts,
            set(series_metadata),
            output_root / "unexpected_files.csv",
            args.expected_dicom_count,
        )
        _write_json_atomic(output_root / "inventory_summary.json", inventory_summary)
        _log(
            f"Header inventory {inventory_summary['status']}: "
            f"{inventory_summary['dicom_files']:,} DICOMs, "
            f"{inventory_summary['checks']['header_failures']:,} header failures"
        )
        decode_summary = run_decode_audit(
            competition_root,
            output_root,
            manifest_parts,
            inventory_summary,
            args.decode,
            args.samples_per_transfer_syntax,
            args.decode_all_example_test,
            args.workers,
            args.batch_size,
            args.worker_chunksize,
            args.resume,
        )
        overall_status = inventory_summary["status"]
        if decode_summary is not None and decode_summary["status"] != "pass":
            overall_status = "fail"
        final_summary = {
            "status": overall_status,
            "generated_at": _now_iso(),
            "read_only_contract": (
                "Competition CSVs and DICOMs were opened only for reading. All writes "
                "were confined to output_root."
            ),
            "inventory": inventory_summary,
            "pixel_decode": decode_summary,
            "environment": environment_summary(),
        }
        _write_json_atomic(output_root / "audit_summary.json", final_summary)
        if decode_summary is not None:
            _log(
                f"Pixel decode {decode_summary['status']}: "
                f"{decode_summary['decoded']:,}/{decode_summary['attempted']:,} decoded"
            )
        _log(f"Audit {overall_status}; summary: {output_root / 'audit_summary.json'}")
        return 0 if overall_status == "pass" else 2
    except AuditConfigurationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
