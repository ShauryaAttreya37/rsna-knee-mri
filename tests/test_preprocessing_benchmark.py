from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

import rsna.preprocessing_benchmark as benchmark_module
from rsna.dicom_loader import DicomLoadError
from rsna.preprocessing_benchmark import (
    SOURCE_COLUMNS,
    PreprocessingBenchmarkError,
    finalize_preprocessing_visual_review,
    run_preprocessing_benchmark,
)


def _write_dicom(
    competition_root: Path,
    study_uid: str,
    series_uid: str,
    instance: int,
    rows: int,
    columns: int,
    spacing: tuple[float, float],
) -> Path:
    sop_uid = generate_uid()
    path = competition_root / "train_series" / study_uid / series_uid / f"{sop_uid}.dcm"
    path.parent.mkdir(parents=True, exist_ok=True)
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = "MR"
    dataset.InstanceNumber = instance
    dataset.Rows = rows
    dataset.Columns = columns
    dataset.PixelSpacing = list(spacing)
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0
    dataset.ImagePositionPatient = [0.0, 0.0, float(instance)]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    pixels = (
        np.arange(rows * columns, dtype=np.uint16).reshape(rows, columns)
        + instance * 10
    )
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return path


def test_representative_benchmark_keeps_identifiers_and_pixels_external(
    tmp_path: Path,
) -> None:
    competition_root = tmp_path / "raw" / "competition"
    project_root = tmp_path / "project"
    project_root.mkdir()
    audit_root = tmp_path / "manifests" / "dicom_audit_v1"
    parts_root = audit_root / "manifest_parts"
    parts_root.mkdir(parents=True)
    output_root = tmp_path / "manifests" / "preprocessing_benchmark_v1"
    report_output = project_root / "reports" / "preprocessing" / "summary.json"

    manifest_rows: list[dict[str, object]] = []
    secret_identifiers: list[str] = []
    for plane_index, plane in enumerate(("Axial", "Coronal", "Sagittal")):
        for sequence_flag in (0, 1):
            study_uid = generate_uid()
            series_uid = generate_uid()
            secret_identifiers.extend((study_uid, series_uid))
            rows = 4 + plane_index * 2
            columns = 8 if sequence_flag == 0 else rows
            spacing = (0.5 + plane_index * 0.1, 0.5)
            for instance in (1, 2):
                _write_dicom(
                    competition_root,
                    study_uid,
                    series_uid,
                    instance,
                    rows,
                    columns,
                    spacing,
                )
                row: dict[str, object] = {
                    "split": "train",
                    "path_study_uid": study_uid,
                    "path_series_uid": series_uid,
                    "anatomical_plane": plane,
                    "fluid_sensitive": sequence_flag,
                    "fat_suppression": sequence_flag,
                    "rows": rows,
                    "columns": columns,
                    "pixel_spacing_row": spacing[0],
                    "pixel_spacing_column": spacing[1],
                    "header_status": "ok",
                }
                assert set(row) == set(SOURCE_COLUMNS)
                manifest_rows.append(row)
    pq.write_table(
        pa.Table.from_pylist(manifest_rows), parts_root / "part-000000.parquet"
    )
    (audit_root / "inventory_summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "manifest_fingerprint": "synthetic-fingerprint",
                "dicom_files": len(manifest_rows),
                "metadata_series": 6,
            }
        ),
        encoding="utf-8",
    )

    report = run_preprocessing_benchmark(
        competition_root,
        audit_root,
        project_root,
        output_root,
        report_output,
        series_per_plane=2,
        sampled_slices=2,
        output_size=8,
    )

    assert report["status"] == "pass"
    assert report["manifest_coverage"] == {
        "manifest_rows": 12,
        "all_series": 6,
        "training_series": 6,
    }
    assert report["counts"] == {
        "selected_series": 6,
        "selected_studies": 6,
        "decoded_images": 12,
        "visual_montages": 6,
    }
    assert report["selection"]["by_anatomical_plane"] == {
        "Axial": 2,
        "Coronal": 2,
        "Sagittal": 2,
    }
    assert report["visual_review"]["status"] == "pending_human_review"
    assert report["load_integrity"] == {
        "attempted_series": 6,
        "load_failures": 0,
        "replacement_series": 0,
        "failure_categories": {},
        "policy": (
            "Reject a series on any strict header, geometry, or pixel-decode "
            "failure and replace it deterministically within the same anatomical "
            "plane, prioritizing matching sequence flags and physical geometry. "
            "Never repair, reshape, or silently skip a corrupt DICOM instance."
        ),
        "scope_limitation": (
            "This benchmark certifies only the attempted representative series; "
            "exhaustive pixel integrity requires the separate decode-all audit."
        ),
    }
    serialized = report_output.read_text(encoding="utf-8")
    assert all(identifier not in serialized for identifier in secret_identifiers)
    external_manifest = (output_root / "cohort_manifest.csv").read_text(
        encoding="utf-8"
    )
    assert all(identifier in external_manifest for identifier in secret_identifiers)
    assert (output_root / "visual_review_checklist.csv").is_file()
    assert (output_root / "load_failures.csv").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "study_uid,series_uid,anatomical_plane,error_category,error_class,error_message"
    ]
    montage_paths = sorted((output_root / "visuals").glob("case-*.png"))
    assert len(montage_paths) == 6
    assert all(
        path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in montage_paths
    )

    with pytest.raises(PreprocessingBenchmarkError, match="every case passes"):
        finalize_preprocessing_visual_review(
            project_root,
            output_root,
            report_output,
            all_cases_pass=False,
        )
    finalized = finalize_preprocessing_visual_review(
        project_root,
        output_root,
        report_output,
        all_cases_pass=True,
    )
    assert finalized["visual_review"]["status"] == "pass"
    assert finalized["visual_review"]["reviewed_cases"] == 6
    assert all(
        value == 0 for value in finalized["visual_review"]["artifact_checks"].values()
    )
    assert all(
        identifier not in report_output.read_text(encoding="utf-8")
        for identifier in secret_identifiers
    )
    checklist = (output_root / "visual_review_checklist.csv").read_text(
        encoding="utf-8"
    )
    assert checklist.count("human_review_passed") == 6
    assert "diagnostic_legibility" in checklist.splitlines()[0]


def test_benchmark_replaces_a_strictly_rejected_series_within_its_plane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    competition_root = tmp_path / "raw" / "competition"
    project_root = tmp_path / "project"
    project_root.mkdir()
    audit_root = tmp_path / "manifests" / "dicom_audit_v1"
    parts_root = audit_root / "manifest_parts"
    parts_root.mkdir(parents=True)
    output_root = tmp_path / "manifests" / "preprocessing_benchmark_v1"
    report_output = project_root / "reports" / "preprocessing" / "summary.json"

    manifest_rows: list[dict[str, object]] = []
    identifiers: list[str] = []
    for plane in ("Axial", "Coronal", "Sagittal"):
        for sequence_flag in (0, 1):
            study_uid = generate_uid()
            series_uid = generate_uid()
            identifiers.extend((study_uid, series_uid))
            for instance in (1, 2):
                _write_dicom(
                    competition_root,
                    study_uid,
                    series_uid,
                    instance,
                    4,
                    8,
                    (0.5, 0.5),
                )
                manifest_rows.append(
                    {
                        "split": "train",
                        "path_study_uid": study_uid,
                        "path_series_uid": series_uid,
                        "anatomical_plane": plane,
                        "fluid_sensitive": sequence_flag,
                        "fat_suppression": sequence_flag,
                        "rows": 4,
                        "columns": 8,
                        "pixel_spacing_row": 0.5,
                        "pixel_spacing_column": 0.5,
                        "header_status": "ok",
                    }
                )
    pq.write_table(
        pa.Table.from_pylist(manifest_rows), parts_root / "part-000000.parquet"
    )
    (audit_root / "inventory_summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "manifest_fingerprint": "synthetic-fingerprint",
                "dicom_files": len(manifest_rows),
                "metadata_series": 6,
            }
        ),
        encoding="utf-8",
    )

    original_loader = benchmark_module.load_dicom_series
    calls = 0

    def fail_first_series(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise DicomLoadError("Could not decode DICOM pixels secret.dcm: ValueError")
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(benchmark_module, "load_dicom_series", fail_first_series)
    report = run_preprocessing_benchmark(
        competition_root,
        audit_root,
        project_root,
        output_root,
        report_output,
        series_per_plane=1,
        sampled_slices=2,
        output_size=8,
    )

    assert report["status"] == "pass"
    assert report["counts"]["selected_series"] == 3
    assert report["load_integrity"]["attempted_series"] == 4
    assert report["load_integrity"]["load_failures"] == 1
    assert report["load_integrity"]["replacement_series"] == 1
    assert report["load_integrity"]["failure_categories"] == {"pixel_decode_error": 1}
    assert report["selection"]["reason_counts"]["load_failure_replacement"] == 1
    assert all(identifier not in json.dumps(report) for identifier in identifiers)
    failures = (output_root / "load_failures.csv").read_text(encoding="utf-8")
    assert "secret.dcm" in failures
    assert "pixel_decode_error" in failures
