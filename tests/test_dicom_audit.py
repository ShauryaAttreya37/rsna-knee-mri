from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.data import get_testdata_file
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from rsna.dicom_audit import (
    AuditConfigurationError,
    DecodeTask,
    HeaderTask,
    decode_pixels,
    main,
    read_header,
    validate_roots,
)

SERIES_COLUMNS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
]


def _write_series_csv(
    path: Path,
    study_uid: str,
    series_uid: str,
    plane: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SERIES_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": series_uid,
                "Fluid_Sensitive": 1,
                "Fat_Suppression": 1,
                "Anatomical_Plane": plane,
            }
        )


def _write_dicom(
    competition_root: Path,
    directory: str,
    study_uid: str,
    series_uid: str,
    sop_uid: str,
    z_position: float,
) -> Path:
    output_path = (
        competition_root / directory / study_uid / series_uid / f"{sop_uid}.dcm"
    )
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
    dataset.InstanceNumber = 1
    dataset.Rows = 2
    dataset.Columns = 3
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0
    dataset.PixelSpacing = [0.5, 0.75]
    dataset.SliceThickness = 3.0
    dataset.ImagePositionPatient = [0.0, 0.0, z_position]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.PixelData = np.arange(6, dtype=np.uint16).tobytes()
    dataset.save_as(output_path, enforce_file_format=True)
    return output_path


@pytest.fixture
def tiny_competition(tmp_path: Path) -> tuple[Path, list[Path]]:
    competition_root = tmp_path / "raw" / "competition"
    competition_root.mkdir(parents=True)

    train_study = generate_uid()
    train_series = generate_uid()
    train_sop = generate_uid()
    test_study = generate_uid()
    test_series = generate_uid()
    test_sop = generate_uid()

    _write_series_csv(
        competition_root / "train_series.csv",
        train_study,
        train_series,
        "Sagittal",
    )
    _write_series_csv(
        competition_root / "test_series.csv",
        test_study,
        test_series,
        "Axial",
    )
    paths = [
        _write_dicom(
            competition_root,
            "train_series",
            train_study,
            train_series,
            train_sop,
            4.25,
        ),
        _write_dicom(
            competition_root,
            "test_series",
            test_study,
            test_series,
            test_sop,
            -2.0,
        ),
    ]
    return competition_root, paths


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_header_read_is_header_only_and_computes_geometry(
    tiny_competition: tuple[Path, list[Path]],
) -> None:
    competition_root, paths = tiny_competition
    path = paths[0]
    parts = path.relative_to(competition_root / "train_series").parts
    task = HeaderTask(
        absolute_path=str(path),
        relative_path=path.relative_to(competition_root).as_posix(),
        split="train",
        file_size=path.stat().st_size,
        is_symlink=False,
        path_structure_ok=True,
        path_study_uid=parts[0],
        path_series_uid=parts[1],
        path_sop_uid=path.stem,
    )

    before = _sha256(path)
    row = read_header(task)

    assert row["header_status"] == "ok"
    assert row["study_uid_matches_path"] is True
    assert row["series_uid_matches_path"] is True
    assert row["sop_uid_matches_filename"] is True
    assert row["geometry_status"] == "ok"
    assert row["slice_coordinate"] == pytest.approx(4.25)
    assert row["rows"] == 2
    assert row["columns"] == 3
    assert _sha256(path) == before


def test_invalid_dicom_is_reported_without_raising(tmp_path: Path) -> None:
    path = tmp_path / "invalid.dcm"
    path.write_bytes(b"not a DICOM file")
    task = HeaderTask(
        absolute_path=str(path),
        relative_path="train_series/study/series/invalid.dcm",
        split="train",
        file_size=path.stat().st_size,
        is_symlink=False,
        path_structure_ok=True,
        path_study_uid="study",
        path_series_uid="series",
        path_sop_uid="invalid",
    )

    row = read_header(task)

    assert row["header_status"] == "read_error"
    assert row["error_class"] == "InvalidDicomError"
    assert str(path) not in row["error_message"]


@pytest.mark.parametrize(
    "filename",
    ["SC_rgb_jpeg_gdcm.dcm", "MR_small_jp2klossless.dcm"],
)
def test_pinned_plugins_decode_lossless_compressed_dicoms(filename: str) -> None:
    path = Path(get_testdata_file(filename, download=False))
    dataset = pydicom.dcmread(path, stop_before_pixels=True)
    task = DecodeTask(
        absolute_path=str(path),
        relative_path=filename,
        split="test",
        transfer_syntax_uid=str(dataset.file_meta.TransferSyntaxUID),
        rows=int(dataset.Rows),
        columns=int(dataset.Columns),
        number_of_frames=int(getattr(dataset, "NumberOfFrames", 1)),
        samples_per_pixel=int(dataset.SamplesPerPixel),
    )

    row = decode_pixels(task)

    assert row["decode_status"] == "ok"
    assert row["array_elements"] > 0
    assert row["all_finite"] is True
    assert row["shape_matches_header"] is True


def test_end_to_end_audit_passes_resumes_and_preserves_raw_files(
    tiny_competition: tuple[Path, list[Path]], tmp_path: Path
) -> None:
    competition_root, raw_paths = tiny_competition
    output_root = tmp_path / "manifests" / "dicom_audit_v1"
    before = {
        path: (_sha256(path), path.stat().st_size, path.stat().st_mtime_ns)
        for path in raw_paths
    }
    arguments = [
        "--competition-root",
        str(competition_root),
        "--output-root",
        str(output_root),
        "--workers",
        "2",
        "--batch-size",
        "1",
        "--worker-chunksize",
        "1",
        "--expected-dicom-count",
        "2",
        "--decode",
        "sample",
        "--samples-per-transfer-syntax",
        "1",
        "--resume",
    ]

    assert main(arguments) == 0
    part_paths = sorted((output_root / "manifest_parts").glob("*.parquet"))
    part_timestamps = {path: path.stat().st_mtime_ns for path in part_paths}
    assert len(part_paths) == 2

    with (output_root / "audit_summary.json").open("r", encoding="utf-8") as file:
        summary = json.load(file)
    assert summary["status"] == "pass"
    assert summary["inventory"]["dicom_files"] == 2
    assert summary["inventory"]["metadata_series"] == 2
    assert summary["inventory"]["series_geometry_audit"]["geometry_sort_ready"] == 2
    assert summary["inventory"]["series_geometry_audit"]["not_sortable"] == 0
    assert summary["pixel_decode"]["attempted"] == 2
    assert summary["pixel_decode"]["decoded"] == 2
    assert all(value == 0 for value in summary["inventory"]["checks"].values())
    assert all(value == 0 for value in summary["pixel_decode"]["checks"].values())

    assert main(arguments) == 0
    assert {path: path.stat().st_mtime_ns for path in part_paths} == part_timestamps
    after = {
        path: (_sha256(path), path.stat().st_size, path.stat().st_mtime_ns)
        for path in raw_paths
    }
    assert after == before


def test_unexpected_file_fails_audit(
    tiny_competition: tuple[Path, list[Path]], tmp_path: Path
) -> None:
    competition_root, _ = tiny_competition
    (competition_root / "train_series" / "unexpected.txt").write_text(
        "unexpected", encoding="utf-8"
    )
    output_root = tmp_path / "manifests" / "with_unexpected_file"

    result = main(
        [
            "--competition-root",
            str(competition_root),
            "--output-root",
            str(output_root),
            "--workers",
            "1",
            "--expected-dicom-count",
            "2",
            "--decode",
            "none",
        ]
    )

    assert result == 2
    with (output_root / "inventory_summary.json").open("r", encoding="utf-8") as file:
        summary = json.load(file)
    assert summary["checks"]["unexpected_non_dicom_files"] == 1


def test_output_cannot_overlap_raw_tree(
    tiny_competition: tuple[Path, list[Path]], tmp_path: Path
) -> None:
    competition_root, _ = tiny_competition
    with pytest.raises(AuditConfigurationError):
        validate_roots(competition_root, competition_root / "derived")
    with pytest.raises(AuditConfigurationError):
        validate_roots(competition_root, tmp_path / "raw")
