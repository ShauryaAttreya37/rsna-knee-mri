from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.data import get_testdata_file
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from rsna.dicom_loader import DicomLoadError, fit_full_fov, load_dicom_series
from rsna.example_test_pipeline import main
from rsna.submission import ID_COLUMN, SUBMISSION_COLUMNS, TARGET_COLUMNS


def _write_dicom(
    series_directory: Path,
    study_uid: str,
    series_uid: str,
    instance_number: int,
    z_position: float,
    pixel_offset: int,
    angle_degrees: float = 0.0,
    include_geometry: bool = True,
    pixel_spacing: tuple[float, float] | None = (0.5, 0.5),
) -> Path:
    sop_uid = generate_uid()
    output_path = series_directory / f"{sop_uid}.dcm"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(
        str(output_path), {}, file_meta=file_meta, preamble=b"\0" * 128
    )
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = "MR"
    dataset.InstanceNumber = instance_number
    dataset.Rows = 4
    dataset.Columns = 5
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0
    dataset.RescaleSlope = 2.0
    dataset.RescaleIntercept = -10.0
    if pixel_spacing is not None:
        dataset.PixelSpacing = list(pixel_spacing)
    if include_geometry:
        angle = math.radians(angle_degrees)
        dataset.ImagePositionPatient = [0.0, 0.0, z_position]
        dataset.ImageOrientationPatient = [
            1.0,
            0.0,
            0.0,
            0.0,
            math.cos(angle),
            math.sin(angle),
        ]
    pixels = np.arange(20, dtype=np.uint16).reshape(4, 5) + pixel_offset
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(output_path, enforce_file_format=True)
    return output_path


def _write_csv(
    path: Path, header: list[str] | tuple[str, ...], rows: list[list[str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def test_full_fov_fit_preserves_physical_aspect_without_cropping() -> None:
    image = np.arange(1, 9, dtype=np.float32).reshape(2, 4)

    fitted, geometry = fit_full_fov(
        image,
        pixel_spacing_row_mm=1.0,
        pixel_spacing_column_mm=1.0,
        output_size=8,
    )

    assert fitted.shape == (8, 8)
    assert geometry.physical_fov_row_mm == 2.0
    assert geometry.physical_fov_column_mm == 4.0
    assert (geometry.resized_rows, geometry.resized_columns) == (4, 8)
    assert (
        geometry.padding_top,
        geometry.padding_bottom,
        geometry.padding_left,
        geometry.padding_right,
    ) == (2, 2, 0, 0)
    assert np.all(fitted[:2] == 0.0)
    assert np.all(fitted[6:] == 0.0)
    assert fitted[2, 0] == image[0, 0]
    assert fitted[2, -1] == image[0, -1]
    assert fitted[5, 0] == image[-1, 0]
    assert fitted[5, -1] == image[-1, -1]


def test_full_fov_fit_uses_pixel_spacing_not_pixel_shape() -> None:
    image = np.arange(8, dtype=np.float32).reshape(2, 4)

    fitted, geometry = fit_full_fov(
        image,
        pixel_spacing_row_mm=2.0,
        pixel_spacing_column_mm=1.0,
        output_size=8,
    )

    assert fitted.shape == (8, 8)
    assert (geometry.physical_fov_row_mm, geometry.physical_fov_column_mm) == (
        4.0,
        4.0,
    )
    assert (geometry.resized_rows, geometry.resized_columns) == (8, 8)
    assert geometry.padded_fraction == 0.0


def test_geometry_loader_sorts_normalizes_samples_and_resizes(tmp_path: Path) -> None:
    study_uid = generate_uid()
    series_uid = generate_uid()
    series_directory = tmp_path / study_uid / series_uid
    for instance, z_position, offset, angle in (
        (3, 2.0, 200, 0.0004),
        (1, 0.0, 0, 0.0),
        (2, 1.0, 100, 0.0002),
    ):
        _write_dicom(
            series_directory,
            study_uid,
            series_uid,
            instance,
            z_position,
            offset,
            angle,
        )

    loaded = load_dicom_series(
        series_directory,
        study_uid,
        series_uid,
        sampled_slices=3,
        output_size=6,
    )

    assert loaded.sampled_images.shape == (3, 6, 6)
    assert loaded.sampled_images.dtype == np.float32
    assert float(np.min(loaded.sampled_images)) >= 0.0
    assert float(np.max(loaded.sampled_images)) <= 1.0
    means = np.mean(loaded.sampled_images, axis=(1, 2))
    assert np.all(np.diff(means) > 0)
    assert loaded.summary.slices == 3
    assert loaded.summary.decoded_pixel_arrays == 3
    assert loaded.summary.sort_strategy == "consensus_geometry"
    assert loaded.summary.max_normal_angle_degrees < 0.001
    assert loaded.summary.pixel_spacing_row_mm == 0.5
    assert loaded.summary.pixel_spacing_column_mm == 0.5
    assert (loaded.summary.resized_rows, loaded.summary.resized_columns) == (5, 6)
    assert loaded.summary.padding_bottom == 1
    assert loaded.summary.padding_top == 0


def test_loader_uses_instance_number_only_when_all_geometry_is_absent(
    tmp_path: Path,
) -> None:
    study_uid = generate_uid()
    series_uid = generate_uid()
    series_directory = tmp_path / study_uid / series_uid
    for instance, offset in ((2, 100), (1, 0)):
        _write_dicom(
            series_directory,
            study_uid,
            series_uid,
            instance,
            float(instance),
            offset,
            include_geometry=False,
        )

    loaded = load_dicom_series(
        series_directory,
        study_uid,
        series_uid,
        sampled_slices=2,
        output_size=4,
    )

    assert loaded.summary.sort_strategy == "instance_number_fallback"
    assert np.mean(loaded.sampled_images[0]) < np.mean(loaded.sampled_images[1])


def test_loader_rejects_duplicate_geometric_coordinates(tmp_path: Path) -> None:
    study_uid = generate_uid()
    series_uid = generate_uid()
    series_directory = tmp_path / study_uid / series_uid
    _write_dicom(series_directory, study_uid, series_uid, 1, 0.0, 0)
    _write_dicom(series_directory, study_uid, series_uid, 2, 0.0, 100)

    with pytest.raises(DicomLoadError, match="duplicate geometric"):
        load_dicom_series(series_directory, study_uid, series_uid)


def test_loader_rejects_missing_or_inconsistent_pixel_spacing(tmp_path: Path) -> None:
    study_uid = generate_uid()
    missing_series_uid = generate_uid()
    missing_directory = tmp_path / study_uid / missing_series_uid
    _write_dicom(
        missing_directory,
        study_uid,
        missing_series_uid,
        1,
        0.0,
        0,
        pixel_spacing=None,
    )

    with pytest.raises(DicomLoadError, match="invalid PixelSpacing"):
        load_dicom_series(missing_directory, study_uid, missing_series_uid)

    inconsistent_series_uid = generate_uid()
    inconsistent_directory = tmp_path / study_uid / inconsistent_series_uid
    _write_dicom(
        inconsistent_directory,
        study_uid,
        inconsistent_series_uid,
        1,
        0.0,
        0,
        pixel_spacing=(0.5, 0.5),
    )
    _write_dicom(
        inconsistent_directory,
        study_uid,
        inconsistent_series_uid,
        2,
        1.0,
        100,
        pixel_spacing=(0.6, 0.5),
    )

    with pytest.raises(DicomLoadError, match="inconsistent row PixelSpacing"):
        load_dicom_series(
            inconsistent_directory,
            study_uid,
            inconsistent_series_uid,
        )


@pytest.mark.parametrize(
    "filename", ["MR_small_jpeg_ls_lossless.dcm", "MR_small_jp2klossless.dcm"]
)
def test_loader_decodes_offline_compressed_mr_fixtures(
    filename: str, tmp_path: Path
) -> None:
    source = Path(get_testdata_file(filename, download=False))
    dataset = pydicom.dcmread(source, stop_before_pixels=True)
    study_uid = str(dataset.StudyInstanceUID)
    series_uid = str(dataset.SeriesInstanceUID)
    sop_uid = str(dataset.SOPInstanceUID)
    series_directory = tmp_path / study_uid / series_uid
    series_directory.mkdir(parents=True)
    shutil.copy2(source, series_directory / f"{sop_uid}.dcm")

    loaded = load_dicom_series(
        series_directory,
        study_uid,
        series_uid,
        sampled_slices=1,
        output_size=32,
    )

    assert loaded.sampled_images.shape == (1, 32, 32)
    assert loaded.summary.decoded_pixel_arrays == 1
    assert loaded.summary.transfer_syntax_uid == str(
        dataset.file_meta.TransferSyntaxUID
    )


def test_complete_example_test_pipeline_is_identifier_free(tmp_path: Path) -> None:
    competition_root = tmp_path / "raw" / "competition"
    study_uid = generate_uid()
    series_uid = generate_uid()
    series_directory = competition_root / "test_series" / study_uid / series_uid
    _write_dicom(series_directory, study_uid, series_uid, 1, 0.0, 0)
    _write_dicom(series_directory, study_uid, series_uid, 2, 1.0, 100)
    _write_csv(competition_root / "test.csv", [ID_COLUMN], [[study_uid]])
    _write_csv(
        competition_root / "test_series.csv",
        [
            ID_COLUMN,
            "SeriesInstanceUID",
            "Fluid_Sensitive",
            "Fat_Suppression",
            "Anatomical_Plane",
        ],
        [[study_uid, series_uid, "1", "1", "Sagittal"]],
    )
    _write_csv(
        competition_root / "sample_submission.csv",
        SUBMISSION_COLUMNS,
        [[study_uid, *(["0.5"] * len(TARGET_COLUMNS))]],
    )
    output = tmp_path / "project" / "submission.csv"
    report = tmp_path / "project" / "reports" / "example.json"

    assert (
        main(
            [
                "--competition-root",
                str(competition_root),
                "--output",
                str(output),
                "--report-output",
                str(report),
                "--sampled-slices",
                "2",
                "--output-size",
                "4",
            ]
        )
        == 0
    )

    summary = json.loads(report.read_text(encoding="utf-8"))
    assert summary["status"] == "pass"
    assert summary["counts"] == {
        "decoded_pixel_arrays": 2,
        "sampled_series_tensors": 1,
        "series": 1,
        "test_studies": 1,
    }
    assert summary["categorical_counts"]["sort_strategies"] == {"consensus_geometry": 1}
    assert summary["configuration"]["spatial_transform"] == (
        "physical_full_fov_aspect_fit_zero_pad"
    )
    assert summary["configuration"]["anatomy_cropping"] is False
    assert summary["distributions"]["padded_fraction"]["count"] == 1
    assert summary["submission_validation"]["rows"] == 1
    serialized = json.dumps(summary)
    assert study_uid not in serialized
    assert series_uid not in serialized
