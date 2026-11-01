"""Geometry-aware, bounded-memory DICOM series loading for inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydicom import dcmread
from pydicom.pixels import pixel_array

HEADER_TAGS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "InstanceNumber",
    "Rows",
    "Columns",
    "PixelSpacing",
    "SamplesPerPixel",
    "NumberOfFrames",
    "PhotometricInterpretation",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "RescaleSlope",
    "RescaleIntercept",
]


class DicomLoadError(RuntimeError):
    """Raised when a DICOM series cannot be loaded without ambiguity."""


@dataclass(frozen=True, slots=True)
class SliceHeader:
    path: Path
    instance_number: int | None
    rows: int
    columns: int
    pixel_spacing_row_mm: float
    pixel_spacing_column_mm: float
    position: np.ndarray | None
    row_direction: np.ndarray | None
    column_direction: np.ndarray | None
    normal: np.ndarray | None
    transfer_syntax_uid: str
    photometric_interpretation: str
    rescale_slope: float
    rescale_intercept: float


@dataclass(frozen=True, slots=True)
class SeriesLoadSummary:
    slices: int
    decoded_pixel_arrays: int
    sort_strategy: str
    rows: int
    columns: int
    pixel_spacing_row_mm: float
    pixel_spacing_column_mm: float
    physical_fov_row_mm: float
    physical_fov_column_mm: float
    resized_rows: int
    resized_columns: int
    padding_top: int
    padding_bottom: int
    padding_left: int
    padding_right: int
    padded_fraction: float
    effective_output_spacing_mm: float
    transfer_syntax_uid: str
    photometric_interpretation: str
    max_normal_angle_degrees: float
    normalization_low: float
    normalization_high: float
    normalized_minimum: float
    normalized_maximum: float
    normalized_mean: float
    sampled_slices: int
    output_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "slices": self.slices,
            "decoded_pixel_arrays": self.decoded_pixel_arrays,
            "sort_strategy": self.sort_strategy,
            "rows": self.rows,
            "columns": self.columns,
            "pixel_spacing_row_mm": self.pixel_spacing_row_mm,
            "pixel_spacing_column_mm": self.pixel_spacing_column_mm,
            "physical_fov_row_mm": self.physical_fov_row_mm,
            "physical_fov_column_mm": self.physical_fov_column_mm,
            "resized_rows": self.resized_rows,
            "resized_columns": self.resized_columns,
            "padding_top": self.padding_top,
            "padding_bottom": self.padding_bottom,
            "padding_left": self.padding_left,
            "padding_right": self.padding_right,
            "padded_fraction": self.padded_fraction,
            "effective_output_spacing_mm": self.effective_output_spacing_mm,
            "transfer_syntax_uid": self.transfer_syntax_uid,
            "photometric_interpretation": self.photometric_interpretation,
            "max_normal_angle_degrees": self.max_normal_angle_degrees,
            "normalization_low": self.normalization_low,
            "normalization_high": self.normalization_high,
            "normalized_minimum": self.normalized_minimum,
            "normalized_maximum": self.normalized_maximum,
            "normalized_mean": self.normalized_mean,
            "sampled_slices": self.sampled_slices,
            "output_size": self.output_size,
        }


@dataclass(frozen=True, slots=True)
class LoadedSeries:
    sampled_images: np.ndarray
    summary: SeriesLoadSummary


@dataclass(frozen=True, slots=True)
class FitAndPadGeometry:
    """Physical geometry of one full-FOV fit-and-pad operation."""

    input_rows: int
    input_columns: int
    pixel_spacing_row_mm: float
    pixel_spacing_column_mm: float
    physical_fov_row_mm: float
    physical_fov_column_mm: float
    resized_rows: int
    resized_columns: int
    padding_top: int
    padding_bottom: int
    padding_left: int
    padding_right: int
    output_size: int
    effective_output_spacing_mm: float

    @property
    def padded_fraction(self) -> float:
        retained_pixels = self.resized_rows * self.resized_columns
        return 1.0 - retained_pixels / (self.output_size * self.output_size)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _vector(value: Any, length: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray([float(item) for item in value], dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return vector / norm


def _read_header(
    path: Path,
    expected_study_uid: str,
    expected_series_uid: str,
) -> SliceHeader:
    try:
        dataset = dcmread(
            path,
            stop_before_pixels=True,
            force=False,
            specific_tags=HEADER_TAGS,
        )
    except Exception as error:
        raise DicomLoadError(
            f"Could not read DICOM header {path.name}: {type(error).__name__}: {error}"
        ) from error
    if str(getattr(dataset, "StudyInstanceUID", "")) != expected_study_uid:
        raise DicomLoadError(
            f"DICOM study UID does not match its directory: {path.name}"
        )
    if str(getattr(dataset, "SeriesInstanceUID", "")) != expected_series_uid:
        raise DicomLoadError(
            f"DICOM series UID does not match its directory: {path.name}"
        )
    if str(getattr(dataset, "SOPInstanceUID", "")) != path.stem:
        raise DicomLoadError(f"DICOM SOP UID does not match its filename: {path.name}")

    rows = _safe_int(getattr(dataset, "Rows", None))
    columns = _safe_int(getattr(dataset, "Columns", None))
    pixel_spacing = _vector(getattr(dataset, "PixelSpacing", None), 2)
    samples_per_pixel = _safe_int(getattr(dataset, "SamplesPerPixel", 1))
    number_of_frames = _safe_int(getattr(dataset, "NumberOfFrames", 1))
    if rows is None or rows <= 0 or columns is None or columns <= 0:
        raise DicomLoadError(f"DICOM has invalid image dimensions: {path.name}")
    if pixel_spacing is None or np.any(pixel_spacing <= 0):
        raise DicomLoadError(f"DICOM has invalid PixelSpacing: {path.name}")
    if samples_per_pixel != 1:
        raise DicomLoadError(
            f"Only single-channel MR images are supported: {path.name}"
        )
    if number_of_frames != 1:
        raise DicomLoadError(f"Only single-frame MR images are supported: {path.name}")

    position = _vector(getattr(dataset, "ImagePositionPatient", None), 3)
    orientation = _vector(getattr(dataset, "ImageOrientationPatient", None), 6)
    row_direction: np.ndarray | None = None
    column_direction: np.ndarray | None = None
    normal: np.ndarray | None = None
    if orientation is not None:
        row_direction = _unit(orientation[:3])
        column_direction = _unit(orientation[3:])
        if row_direction is None or column_direction is None:
            raise DicomLoadError(f"DICOM has a zero orientation vector: {path.name}")
        if abs(float(np.dot(row_direction, column_direction))) > 1e-3:
            raise DicomLoadError(
                f"DICOM orientation vectors are not orthogonal: {path.name}"
            )
        normal = _unit(np.cross(row_direction, column_direction))
        if normal is None:
            raise DicomLoadError(f"DICOM has an invalid slice normal: {path.name}")
    if (position is None) != (orientation is None):
        raise DicomLoadError(
            f"DICOM has only one of position/orientation geometry: {path.name}"
        )

    transfer_syntax = getattr(dataset.file_meta, "TransferSyntaxUID", None)
    if transfer_syntax is None:
        raise DicomLoadError(f"DICOM has no transfer syntax UID: {path.name}")
    photometric = str(getattr(dataset, "PhotometricInterpretation", ""))
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise DicomLoadError(
            f"Unsupported photometric interpretation {photometric!r}: {path.name}"
        )
    slope = _safe_float(getattr(dataset, "RescaleSlope", 1.0), default=1.0)
    intercept = _safe_float(getattr(dataset, "RescaleIntercept", 0.0), default=0.0)
    if slope is None or slope == 0.0 or intercept is None:
        raise DicomLoadError(f"DICOM has invalid rescale parameters: {path.name}")
    return SliceHeader(
        path=path,
        instance_number=_safe_int(getattr(dataset, "InstanceNumber", None)),
        rows=rows,
        columns=columns,
        pixel_spacing_row_mm=float(pixel_spacing[0]),
        pixel_spacing_column_mm=float(pixel_spacing[1]),
        position=position,
        row_direction=row_direction,
        column_direction=column_direction,
        normal=normal,
        transfer_syntax_uid=str(transfer_syntax),
        photometric_interpretation=photometric,
        rescale_slope=slope,
        rescale_intercept=intercept,
    )


def _max_pairwise_angle_degrees(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return 0.0
    dots = np.clip(vectors @ vectors.T, -1.0, 1.0)
    return math.degrees(math.acos(float(np.min(dots))))


def _sort_headers(
    headers: list[SliceHeader], max_normal_angle_degrees: float
) -> tuple[list[SliceHeader], str, float]:
    has_geometry = [
        header.position is not None and header.normal is not None for header in headers
    ]
    if all(has_geometry):
        normals = np.asarray([header.normal for header in headers])
        reference = normals[0]
        aligned = np.asarray(
            [
                normal if np.dot(normal, reference) >= 0 else -normal
                for normal in normals
            ]
        )
        maximum_angle = _max_pairwise_angle_degrees(aligned)
        if maximum_angle > max_normal_angle_degrees:
            raise DicomLoadError(
                "Series slice normals vary by "
                f"{maximum_angle:.6f} degrees, above the "
                f"{max_normal_angle_degrees:.6f}-degree safety limit"
            )
        consensus = _unit(np.mean(aligned, axis=0))
        if consensus is None:
            raise DicomLoadError("Series slice normals do not form a consensus")
        coordinates = [float(np.dot(header.position, consensus)) for header in headers]
        rounded = [round(value, 5) for value in coordinates]
        if len(set(rounded)) != len(rounded):
            raise DicomLoadError("Series has duplicate geometric slice coordinates")
        ordered = [
            header
            for _, header in sorted(
                zip(coordinates, headers, strict=True), key=lambda item: item[0]
            )
        ]
        return ordered, "consensus_geometry", maximum_angle
    if any(has_geometry):
        raise DicomLoadError("Series contains incomplete per-slice geometry")

    instances = [header.instance_number for header in headers]
    if any(instance is None for instance in instances):
        raise DicomLoadError("Series cannot be sorted by geometry or InstanceNumber")
    integer_instances = [
        int(instance) for instance in instances if instance is not None
    ]
    if len(set(integer_instances)) != len(integer_instances):
        raise DicomLoadError("Series has duplicate InstanceNumber values")
    ordered = [
        header
        for _, header in sorted(
            zip(integer_instances, headers, strict=True), key=lambda item: item[0]
        )
    ]
    return ordered, "instance_number_fallback", 0.0


def _decode_slice(header: SliceHeader) -> np.ndarray:
    try:
        array = np.asarray(pixel_array(str(header.path)))
    except Exception as error:
        raise DicomLoadError(
            f"Could not decode DICOM pixels {header.path.name}: "
            f"{type(error).__name__}: {error}"
        ) from error
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (header.rows, header.columns):
        raise DicomLoadError(
            f"Decoded pixel shape {array.shape} differs from header "
            f"{(header.rows, header.columns)}: {header.path.name}"
        )
    image = array.astype(np.float32, copy=False)
    if header.rescale_slope != 1.0 or header.rescale_intercept != 0.0:
        image = image * header.rescale_slope + header.rescale_intercept
    if not np.all(np.isfinite(image)):
        raise DicomLoadError(f"Decoded pixels are not finite: {header.path.name}")
    if header.photometric_interpretation == "MONOCHROME1":
        image = float(np.min(image)) + float(np.max(image)) - image
    return np.asarray(image, dtype=np.float32)


def _normalization_bounds(
    volume: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
    maximum_sample_values: int = 1_000_000,
) -> tuple[float, float]:
    flattened = volume.reshape(-1)
    stride = max(1, math.ceil(flattened.size / maximum_sample_values))
    sample = flattened[::stride]
    low, high = np.percentile(sample, (lower_percentile, upper_percentile))
    low_value = float(low)
    high_value = float(high)
    if not math.isfinite(low_value) or not math.isfinite(high_value):
        raise DicomLoadError("Series normalization bounds are not finite")
    if high_value <= low_value:
        minimum = float(np.min(flattened))
        maximum = float(np.max(flattened))
        if maximum <= minimum:
            return minimum, minimum
        return minimum, maximum
    return low_value, high_value


def _normalize_volume(
    volume: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[np.ndarray, float, float]:
    low, high = _normalization_bounds(volume, lower_percentile, upper_percentile)
    if high <= low:
        normalized = np.zeros_like(volume, dtype=np.float32)
    else:
        normalized = np.clip(volume, low, high)
        normalized = (normalized - low) / (high - low)
        normalized = np.asarray(normalized, dtype=np.float32)
        np.clip(normalized, 0.0, 1.0, out=normalized)
    if not np.all(np.isfinite(normalized)):
        raise DicomLoadError("Normalized series contains nonfinite values")
    return normalized, low, high


def _resize_bilinear(
    image: np.ndarray, output_rows: int, output_columns: int
) -> np.ndarray:
    if output_rows < 1 or output_columns < 1:
        raise DicomLoadError("Resize dimensions must be positive")
    if image.ndim != 2 or not np.all(np.isfinite(image)):
        raise DicomLoadError("Bilinear resize requires a finite two-dimensional image")
    if image.shape == (output_rows, output_columns):
        return np.asarray(image, dtype=np.float32)
    input_rows, input_columns = image.shape
    output_y = np.linspace(0.0, input_rows - 1, output_rows)
    output_x = np.linspace(0.0, input_columns - 1, output_columns)
    y0 = np.floor(output_y).astype(int)
    x0 = np.floor(output_x).astype(int)
    y1 = np.minimum(y0 + 1, input_rows - 1)
    x1 = np.minimum(x0 + 1, input_columns - 1)
    y_weight = (output_y - y0).astype(np.float32)
    x_weight = (output_x - x0).astype(np.float32)
    vertical = (
        image[y0, :] * (1.0 - y_weight[:, None]) + image[y1, :] * y_weight[:, None]
    )
    resized = (
        vertical[:, x0] * (1.0 - x_weight[None, :])
        + vertical[:, x1] * x_weight[None, :]
    )
    return np.asarray(resized, dtype=np.float32)


def fit_full_fov(
    image: np.ndarray,
    pixel_spacing_row_mm: float,
    pixel_spacing_column_mm: float,
    output_size: int,
    padding_value: float = 0.0,
) -> tuple[np.ndarray, FitAndPadGeometry]:
    """Resize a complete physical FOV to fit a square and pad without cropping."""
    image = np.asarray(image)
    if image.ndim != 2 or image.shape[0] < 1 or image.shape[1] < 1:
        raise DicomLoadError("Full-FOV fitting requires a nonempty 2D image")
    if not np.all(np.isfinite(image)):
        raise DicomLoadError("Full-FOV fitting requires finite image values")
    if output_size < 1:
        raise DicomLoadError("Full-FOV output_size must be positive")
    for name, value in (
        ("row PixelSpacing", pixel_spacing_row_mm),
        ("column PixelSpacing", pixel_spacing_column_mm),
        ("padding value", padding_value),
    ):
        if not math.isfinite(value):
            raise DicomLoadError(f"{name} must be finite")
    if pixel_spacing_row_mm <= 0 or pixel_spacing_column_mm <= 0:
        raise DicomLoadError("PixelSpacing values must be positive")

    input_rows, input_columns = image.shape
    physical_fov_row_mm = input_rows * pixel_spacing_row_mm
    physical_fov_column_mm = input_columns * pixel_spacing_column_mm
    fit_scale_pixels_per_mm = min(
        output_size / physical_fov_row_mm,
        output_size / physical_fov_column_mm,
    )
    if physical_fov_row_mm >= physical_fov_column_mm:
        resized_rows = output_size
        resized_columns = max(
            1,
            min(
                output_size,
                round(physical_fov_column_mm * fit_scale_pixels_per_mm),
            ),
        )
    else:
        resized_columns = output_size
        resized_rows = max(
            1,
            min(output_size, round(physical_fov_row_mm * fit_scale_pixels_per_mm)),
        )

    resized = _resize_bilinear(image, resized_rows, resized_columns)
    padding_rows = output_size - resized_rows
    padding_columns = output_size - resized_columns
    padding_top = padding_rows // 2
    padding_bottom = padding_rows - padding_top
    padding_left = padding_columns // 2
    padding_right = padding_columns - padding_left
    output = np.full((output_size, output_size), padding_value, dtype=np.float32)
    output[
        padding_top : output_size - padding_bottom,
        padding_left : output_size - padding_right,
    ] = resized
    geometry = FitAndPadGeometry(
        input_rows=input_rows,
        input_columns=input_columns,
        pixel_spacing_row_mm=pixel_spacing_row_mm,
        pixel_spacing_column_mm=pixel_spacing_column_mm,
        physical_fov_row_mm=physical_fov_row_mm,
        physical_fov_column_mm=physical_fov_column_mm,
        resized_rows=resized_rows,
        resized_columns=resized_columns,
        padding_top=padding_top,
        padding_bottom=padding_bottom,
        padding_left=padding_left,
        padding_right=padding_right,
        output_size=output_size,
        effective_output_spacing_mm=1.0 / fit_scale_pixels_per_mm,
    )
    if output.shape != (output_size, output_size) or not np.all(np.isfinite(output)):
        raise AssertionError("Full-FOV transform produced an invalid output")
    return output, geometry


def _uniform_sample_indices(slice_count: int, sampled_slices: int) -> np.ndarray:
    if slice_count <= 0 or sampled_slices <= 0:
        raise DicomLoadError("Slice counts must be positive")
    return np.rint(np.linspace(0, slice_count - 1, sampled_slices)).astype(int)


def _series_pixel_spacing(headers: list[SliceHeader]) -> tuple[float, float]:
    row_values = np.asarray(
        [header.pixel_spacing_row_mm for header in headers], dtype=float
    )
    column_values = np.asarray(
        [header.pixel_spacing_column_mm for header in headers], dtype=float
    )
    row_spacing = float(np.median(row_values))
    column_spacing = float(np.median(column_values))
    for name, values, reference in (
        ("row", row_values, row_spacing),
        ("column", column_values, column_spacing),
    ):
        tolerance = 1e-6 + 1e-3 * abs(reference)
        if float(np.max(values) - np.min(values)) > tolerance:
            raise DicomLoadError(
                f"Series contains inconsistent {name} PixelSpacing values"
            )
    return row_spacing, column_spacing


def load_dicom_series(
    series_directory: Path,
    expected_study_uid: str,
    expected_series_uid: str,
    sampled_slices: int = 16,
    output_size: int = 224,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    max_normal_angle_degrees: float = 1.0,
) -> LoadedSeries:
    """Load one series, normalize it, and retain a fixed-size slice sample."""
    if sampled_slices < 1 or output_size < 1:
        raise DicomLoadError("sampled_slices and output_size must be positive")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise DicomLoadError(
            "Normalization percentiles must satisfy 0 <= low < high <= 100"
        )
    if max_normal_angle_degrees <= 0.0:
        raise DicomLoadError("max_normal_angle_degrees must be positive")
    series_directory = series_directory.resolve()
    if not series_directory.is_dir():
        raise DicomLoadError(f"Series directory does not exist: {series_directory}")
    dicom_paths = sorted(series_directory.glob("*.dcm"))
    unexpected = [
        path for path in series_directory.iterdir() if not path.name.endswith(".dcm")
    ]
    if unexpected:
        raise DicomLoadError(
            f"Series directory contains {len(unexpected)} unexpected entries"
        )
    if not dicom_paths:
        raise DicomLoadError("Series directory contains no DICOM files")
    headers = [
        _read_header(path, expected_study_uid, expected_series_uid)
        for path in dicom_paths
    ]
    dimensions = {(header.rows, header.columns) for header in headers}
    syntaxes = {header.transfer_syntax_uid for header in headers}
    photometrics = {header.photometric_interpretation for header in headers}
    if len(dimensions) != 1:
        raise DicomLoadError("Series contains inconsistent image dimensions")
    if len(syntaxes) != 1:
        raise DicomLoadError("Series contains multiple transfer syntaxes")
    if len(photometrics) != 1:
        raise DicomLoadError("Series contains multiple photometric interpretations")
    pixel_spacing_row_mm, pixel_spacing_column_mm = _series_pixel_spacing(headers)
    ordered, sort_strategy, maximum_angle = _sort_headers(
        headers, max_normal_angle_degrees
    )
    decoded = [_decode_slice(header) for header in ordered]
    volume = np.stack(decoded, axis=0)
    normalized, low, high = _normalize_volume(
        volume, lower_percentile, upper_percentile
    )
    indices = _uniform_sample_indices(len(ordered), sampled_slices)
    transformed: list[np.ndarray] = []
    fit_geometry: FitAndPadGeometry | None = None
    for index in indices:
        fitted, geometry = fit_full_fov(
            normalized[index],
            pixel_spacing_row_mm,
            pixel_spacing_column_mm,
            output_size,
        )
        if fit_geometry is None:
            fit_geometry = geometry
        elif geometry != fit_geometry:
            raise AssertionError("Series slices received inconsistent fit geometry")
        transformed.append(fitted)
    if fit_geometry is None:
        raise AssertionError("No sampled image received a spatial transform")
    sampled = np.stack(transformed, axis=0)
    np.clip(sampled, 0.0, 1.0, out=sampled)
    if sampled.shape != (sampled_slices, output_size, output_size):
        raise AssertionError(f"Unexpected sampled tensor shape: {sampled.shape}")
    if (
        not np.all(np.isfinite(sampled))
        or float(np.min(sampled)) < 0.0
        or float(np.max(sampled)) > 1.0
    ):
        raise DicomLoadError("Sampled tensor violates normalized finite [0, 1] range")
    rows, columns = next(iter(dimensions))
    return LoadedSeries(
        sampled_images=sampled,
        summary=SeriesLoadSummary(
            slices=len(ordered),
            decoded_pixel_arrays=len(decoded),
            sort_strategy=sort_strategy,
            rows=rows,
            columns=columns,
            pixel_spacing_row_mm=pixel_spacing_row_mm,
            pixel_spacing_column_mm=pixel_spacing_column_mm,
            physical_fov_row_mm=fit_geometry.physical_fov_row_mm,
            physical_fov_column_mm=fit_geometry.physical_fov_column_mm,
            resized_rows=fit_geometry.resized_rows,
            resized_columns=fit_geometry.resized_columns,
            padding_top=fit_geometry.padding_top,
            padding_bottom=fit_geometry.padding_bottom,
            padding_left=fit_geometry.padding_left,
            padding_right=fit_geometry.padding_right,
            padded_fraction=fit_geometry.padded_fraction,
            effective_output_spacing_mm=fit_geometry.effective_output_spacing_mm,
            transfer_syntax_uid=next(iter(syntaxes)),
            photometric_interpretation=next(iter(photometrics)),
            max_normal_angle_degrees=maximum_angle,
            normalization_low=low,
            normalization_high=high,
            normalized_minimum=float(np.min(normalized)),
            normalized_maximum=float(np.max(normalized)),
            normalized_mean=float(np.mean(normalized, dtype=np.float64)),
            sampled_slices=sampled_slices,
            output_size=output_size,
        ),
    )
