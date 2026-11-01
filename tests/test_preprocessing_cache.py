from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from rsna.preprocessing_cache import (
    CACHE_MANIFEST_COLUMNS,
    PreprocessingCacheError,
    build_full_cohort_cache,
    build_full_cohort_cache_plan,
)
from rsna.submission import ID_COLUMN, TARGET_COLUMNS


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _write_dicom(
    series_directory: Path,
    study_uid: str,
    series_uid: str,
    instance: int,
) -> None:
    sop_uid = generate_uid()
    path = series_directory / f"{sop_uid}.dcm"
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
    dataset.Rows = 4
    dataset.Columns = 5
    dataset.PixelSpacing = [0.5, 0.5]
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0
    dataset.ImagePositionPatient = [0.0, 0.0, float(instance)]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    pixels = np.arange(20, dtype=np.uint16).reshape(4, 5) + instance * 10
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)


def _build_inputs(tmp_path: Path):
    project_root = tmp_path / "project"
    competition_root = tmp_path / "data" / "raw" / "competition"
    exclusions_path = tmp_path / "data" / "manifests" / "excluded_series.csv"
    cache_root = tmp_path / "data" / "processed" / "training_cache"
    report_output = project_root / "reports" / "preprocessing_cache" / "summary.json"
    project_root.mkdir()
    study_ids = [generate_uid(), generate_uid()]
    series_by_study = {
        study_uid: [generate_uid(), generate_uid()] for study_uid in study_ids
    }
    train_rows = [
        [study_uid, f"private report {index}", *([0] * len(TARGET_COLUMNS))]
        for index, study_uid in enumerate(study_ids)
    ]
    series_rows: list[list[object]] = []
    planes = ("Axial", "Coronal", "Sagittal")
    for study_index, study_uid in enumerate(study_ids):
        for series_index, series_uid in enumerate(series_by_study[study_uid]):
            series_rows.append(
                [
                    study_uid,
                    series_uid,
                    series_index % 2,
                    (series_index + 1) % 2,
                    planes[(study_index + series_index) % len(planes)],
                ]
            )
            for instance in (1, 2):
                _write_dicom(
                    competition_root / "train_series" / study_uid / series_uid,
                    study_uid,
                    series_uid,
                    instance,
                )
    _write_csv(
        competition_root / "train.csv",
        [ID_COLUMN, "Report", *TARGET_COLUMNS],
        train_rows,
    )
    _write_csv(
        competition_root / "train_series.csv",
        [
            ID_COLUMN,
            "SeriesInstanceUID",
            "Fluid_Sensitive",
            "Fat_Suppression",
            "Anatomical_Plane",
        ],
        series_rows,
    )
    excluded_uid = series_by_study[study_ids[0]][0]
    _write_csv(
        exclusions_path,
        ["split", "study_uid", "series_uid", "unsafe_reasons"],
        [["train", study_ids[0], excluded_uid, "decode_error"]],
    )
    return {
        "project_root": project_root,
        "competition_root": competition_root,
        "exclusions_path": exclusions_path,
        "cache_root": cache_root,
        "report_output": report_output,
        "study_ids": study_ids,
        "series_by_study": series_by_study,
        "excluded_uid": excluded_uid,
    }


def _run(inputs: dict[str, object], **kwargs):
    return build_full_cohort_cache(
        inputs["competition_root"],
        inputs["exclusions_path"],
        inputs["cache_root"],
        inputs["project_root"],
        inputs["report_output"],
        sampled_slices=2,
        output_size=8,
        minimum_free_gib=0.0,
        progress_every=1,
        expected_studies=2,
        expected_series=4,
        expected_excluded_series=1,
        **kwargs,
    )


def test_full_cache_plan_applies_exclusions_and_preserves_study_order(
    tmp_path: Path,
) -> None:
    inputs = _build_inputs(tmp_path)
    plan = build_full_cohort_cache_plan(
        inputs["competition_root"],
        inputs["exclusions_path"],
        expected_studies=2,
        expected_series=4,
        expected_excluded_series=1,
    )

    assert plan.studies == 2
    assert plan.total_series_before_exclusions == 4
    assert plan.excluded_series == 1
    assert len(plan.tasks) == 3
    assert [task.order for task in plan.tasks] == [0, 1, 2]
    assert plan.tasks[0].study_uid == inputs["study_ids"][0]
    assert all(task.series.series_uid != inputs["excluded_uid"] for task in plan.tasks)
    assert sum(plan.series_by_plane.values()) == 3


def test_full_cache_is_atomic_validated_private_and_reusable(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)

    first = _run(inputs, workers=1)
    second = _run(inputs, workers=1)

    assert first["status"] == "complete"
    assert first["counts"]["completed_series"] == 3
    assert first["counts"]["cache_hits"] == 0
    assert first["counts"]["cache_misses_completed"] == 3
    assert first["counts"]["decoded_images_this_run"] == 6
    assert second["status"] == "complete"
    assert second["counts"]["cache_hits"] == 3
    assert second["counts"]["cache_misses_completed"] == 0
    manifest_path = inputs["cache_root"] / "cache_manifest.csv"
    with manifest_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert tuple(rows[0]) == CACHE_MANIFEST_COLUMNS
    assert len(rows) == 3
    for row in rows:
        tensor = np.load(inputs["cache_root"] / row["tensor_relative_path"])
        assert tensor.shape == (2, 8, 8)
        assert tensor.dtype == np.float16
        metadata_path = (
            inputs["cache_root"]
            / "studies"
            / row[ID_COLUMN]
            / f"{row['SeriesInstanceUID']}.json"
        )
        assert metadata_path.is_file()
    serialized_report = inputs["report_output"].read_text(encoding="utf-8")
    for study_uid in inputs["study_ids"]:
        assert study_uid not in serialized_report
    for series_uids in inputs["series_by_study"].values():
        for series_uid in series_uids:
            assert series_uid not in serialized_report
    assert json.loads(serialized_report)["status"] == "complete"
    assert not list(inputs["cache_root"].rglob("*.tmp"))


def test_partial_cache_resumes_with_parallel_workers(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)

    partial = _run(inputs, workers=1, max_new_series=1)
    complete = _run(inputs, workers=2)

    assert partial["status"] == "partial"
    assert partial["counts"]["completed_series"] == 1
    assert partial["counts"]["remaining_series"] == 2
    assert complete["status"] == "complete"
    assert complete["counts"]["cache_hits"] == 1
    assert complete["counts"]["cache_misses_completed"] == 2


def test_cache_recomputes_orphaned_atomic_tensor(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)
    _run(inputs, workers=1)
    metadata_path = next((inputs["cache_root"] / "studies").rglob("*.json"))
    metadata_path.unlink()

    report = _run(inputs, workers=1)

    assert report["status"] == "complete"
    assert report["counts"]["orphaned_tensors_recomputed"] == 1
    assert report["counts"]["cache_hits"] == 2
    assert report["counts"]["cache_misses_completed"] == 1


def test_cache_rejects_corrupt_completed_tensor(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)
    _run(inputs, workers=1)
    tensor_path = next((inputs["cache_root"] / "studies").rglob("*.npy"))
    with tensor_path.open("wb") as file:
        np.save(file, np.full((2, 8, 8), np.nan, dtype=np.float16))

    with pytest.raises(PreprocessingCacheError, match="nonfinite"):
        _run(inputs, workers=1)


def test_cache_rejects_configuration_change(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)
    _run(inputs, workers=1)

    with pytest.raises(PreprocessingCacheError, match="configuration differs"):
        build_full_cohort_cache(
            inputs["competition_root"],
            inputs["exclusions_path"],
            inputs["cache_root"],
            inputs["project_root"],
            inputs["report_output"],
            sampled_slices=3,
            output_size=8,
            workers=1,
            minimum_free_gib=0.0,
            expected_studies=2,
            expected_series=4,
            expected_excluded_series=1,
        )


def test_cache_rejects_wrong_expected_exclusion_count(tmp_path: Path) -> None:
    inputs = _build_inputs(tmp_path)

    with pytest.raises(PreprocessingCacheError, match="Expected 3 excluded"):
        build_full_cohort_cache_plan(
            inputs["competition_root"],
            inputs["exclusions_path"],
            expected_studies=2,
            expected_series=4,
            expected_excluded_series=3,
        )
