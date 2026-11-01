from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from rsna.submission import ID_COLUMN, TARGET_COLUMNS
from rsna.training_data import (
    TrainingDataError,
    load_gold_training_cohort,
    materialize_gold_cache,
)


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


def _build_gold_inputs(tmp_path: Path):
    competition_root = tmp_path / "raw" / "competition"
    folds_path = tmp_path / "manifests" / "gold_folds.csv"
    exclusions_path = tmp_path / "manifests" / "excluded_series.csv"
    study_ids = [generate_uid() for _ in range(4)]
    series_by_study = {
        study_uid: [generate_uid(), generate_uid()] for study_uid in study_ids
    }

    train_rows: list[list[object]] = []
    series_rows: list[list[object]] = []
    for study_index, study_uid in enumerate(study_ids):
        labels = [
            int((study_index + target_index) % 2 == 0)
            for target_index in range(len(TARGET_COLUMNS))
        ]
        train_rows.append([study_uid, f"secret report {study_index}", *labels])
        for series_index, series_uid in enumerate(series_by_study[study_uid]):
            plane = ("Axial", "Coronal", "Sagittal")[(study_index + series_index) % 3]
            series_rows.append(
                [study_uid, series_uid, series_index % 2, series_index % 2, plane]
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
    _write_csv(
        folds_path, [ID_COLUMN, "fold"], [[study_uid, 0] for study_uid in study_ids]
    )
    excluded_series = series_by_study[study_ids[0]][0]
    _write_csv(
        exclusions_path,
        ["split", "study_uid", "series_uid"],
        [
            ["train", study_ids[0], excluded_series],
            ["test", generate_uid(), generate_uid()],
        ],
    )
    return competition_root, folds_path, exclusions_path, study_ids, excluded_series


def test_gold_training_cohort_applies_folds_and_whole_series_exclusions(
    tmp_path: Path,
) -> None:
    competition_root, folds_path, exclusions_path, study_ids, excluded_series = (
        _build_gold_inputs(tmp_path)
    )

    cohort = load_gold_training_cohort(
        competition_root, folds_path, exclusions_path, fold=0
    )

    assert cohort.fold == 0
    assert cohort.total_gold_studies == 4
    assert len(cohort.studies) == 4
    assert cohort.total_gold_series_before_exclusions == 8
    assert cohort.excluded_gold_series == 1
    assert sum(len(study.series) for study in cohort.studies) == 7
    assert all(study.study_uid in study_ids for study in cohort.studies)
    assert all(
        series.series_uid != excluded_series
        for study in cohort.studies
        for series in study.series
    )
    assert np.all(np.sum(cohort.labels, axis=0) == 2)
    assert set(cohort.source_sha256) == {
        "train.csv",
        "train_series.csv",
        "gold_folds.csv",
        "excluded_series.csv",
    }


def test_gold_cache_is_validated_and_reused_without_redecoding(tmp_path: Path) -> None:
    competition_root, folds_path, exclusions_path, _, _ = _build_gold_inputs(tmp_path)
    cohort = load_gold_training_cohort(
        competition_root, folds_path, exclusions_path, fold=0
    )
    cache_root = tmp_path / "processed" / "gold-cache"

    first = materialize_gold_cache(
        cohort, competition_root, cache_root, sampled_slices=2, output_size=8
    )
    second = materialize_gold_cache(
        cohort, competition_root, cache_root, sampled_slices=2, output_size=8
    )

    assert first.cache_hits == 0
    assert first.cache_misses == 7
    assert first.decoded_images == 14
    assert first.cache_bytes > 0
    assert second.cache_hits == 7
    assert second.cache_misses == 0
    assert second.decoded_images == 0
    assert second.cache_bytes == first.cache_bytes
    for study in second.studies:
        for path in study.tensor_paths:
            tensor = np.load(path, allow_pickle=False)
            assert tensor.shape == (2, 8, 8)
            assert tensor.dtype == np.float16


def test_gold_training_cohort_rejects_non_class_complete_fold(tmp_path: Path) -> None:
    competition_root, folds_path, exclusions_path, study_ids, _ = _build_gold_inputs(
        tmp_path
    )
    _write_csv(
        folds_path,
        [ID_COLUMN, "fold"],
        [[study_uid, int(index != 0)] for index, study_uid in enumerate(study_ids)],
    )

    with pytest.raises(TrainingDataError, match="not class-complete"):
        load_gold_training_cohort(competition_root, folds_path, exclusions_path, fold=0)
