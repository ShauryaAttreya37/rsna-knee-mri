"""Strict gold-study manifests and external full-FOV tensor caching."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rsna.dicom_loader import load_dicom_series
from rsna.gold_folds import GoldFoldError, load_gold_labels
from rsna.submission import ID_COLUMN

SERIES_UID_COLUMN = "SeriesInstanceUID"
PLANE_TO_INDEX = {"Axial": 0, "Coronal": 1, "Sagittal": 2}


class TrainingDataError(RuntimeError):
    """Raised when training inputs or external caches are unsafe."""


@dataclass(frozen=True, slots=True)
class TrainingSeries:
    series_uid: str
    anatomical_plane: str
    fluid_sensitive: int
    fat_suppression: int

    @property
    def metadata_vector(self) -> tuple[float, ...]:
        plane = [0.0, 0.0, 0.0]
        plane[PLANE_TO_INDEX[self.anatomical_plane]] = 1.0
        return (*plane, float(self.fluid_sensitive), float(self.fat_suppression))


@dataclass(frozen=True, slots=True)
class GoldTrainingStudy:
    study_uid: str
    fold: int
    labels: tuple[int, ...]
    series: tuple[TrainingSeries, ...]


@dataclass(frozen=True, slots=True)
class GoldTrainingCohort:
    studies: tuple[GoldTrainingStudy, ...]
    fold: int
    total_gold_studies: int
    total_gold_series_before_exclusions: int
    excluded_gold_series: int
    source_sha256: dict[str, str]

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([study.labels for study in self.studies], dtype=np.int8)


@dataclass(frozen=True, slots=True)
class CachedTrainingStudy:
    study_uid: str
    fold: int
    labels: tuple[int, ...]
    series: tuple[TrainingSeries, ...]
    tensor_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class CacheResult:
    studies: tuple[CachedTrainingStudy, ...]
    cache_hits: int
    cache_misses: int
    decoded_images: int
    cache_bytes: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise TrainingDataError(
            f"Could not fingerprint {path.name}: {error}"
        ) from error
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as error:
        raise TrainingDataError(f"Could not read {path.name}: {error}") from error
    if not fieldnames:
        raise TrainingDataError(f"CSV has no header: {path.name}")
    return fieldnames, rows


def _load_folds(path: Path, gold_ids: set[str]) -> dict[str, int]:
    fieldnames, rows = _read_csv(path)
    if fieldnames != [ID_COLUMN, "fold"]:
        raise TrainingDataError("Gold fold manifest columns are invalid")
    assignments: dict[str, int] = {}
    for row in rows:
        identifier = (row.get(ID_COLUMN) or "").strip()
        try:
            fold = int(row.get("fold") or "")
        except ValueError as error:
            raise TrainingDataError(
                "Gold fold manifest contains an invalid fold"
            ) from error
        if not identifier or identifier in assignments or fold < 0:
            raise TrainingDataError(
                "Gold fold manifest contains blank, duplicate, or invalid assignments"
            )
        assignments[identifier] = fold
    if set(assignments) != gold_ids:
        raise TrainingDataError(
            "Gold fold manifest identifiers do not exactly match gold labels"
        )
    return assignments


def _binary_flag(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise TrainingDataError(f"Series metadata contains invalid {name}") from error
    if parsed not in {0, 1}:
        raise TrainingDataError(f"Series metadata contains nonbinary {name}")
    return parsed


def _load_training_series(
    path: Path,
) -> dict[str, list[TrainingSeries]]:
    fieldnames, rows = _read_csv(path)
    required = {
        ID_COLUMN,
        SERIES_UID_COLUMN,
        "Anatomical_Plane",
        "Fluid_Sensitive",
        "Fat_Suppression",
    }
    missing = required - set(fieldnames)
    if missing:
        raise TrainingDataError(
            f"train_series.csv is missing columns: {sorted(missing)}"
        )
    by_study: dict[str, list[TrainingSeries]] = {}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        study_uid = (row.get(ID_COLUMN) or "").strip()
        series_uid = (row.get(SERIES_UID_COLUMN) or "").strip()
        plane = (row.get("Anatomical_Plane") or "").strip()
        key = (study_uid, series_uid)
        if not study_uid or not series_uid or key in seen:
            raise TrainingDataError(
                "train_series.csv contains blank or duplicate identifiers"
            )
        if plane not in PLANE_TO_INDEX:
            raise TrainingDataError("train_series.csv contains an invalid plane")
        seen.add(key)
        by_study.setdefault(study_uid, []).append(
            TrainingSeries(
                series_uid=series_uid,
                anatomical_plane=plane,
                fluid_sensitive=_binary_flag(
                    row.get("Fluid_Sensitive") or "", "Fluid_Sensitive"
                ),
                fat_suppression=_binary_flag(
                    row.get("Fat_Suppression") or "", "Fat_Suppression"
                ),
            )
        )
    for series in by_study.values():
        series.sort(key=lambda item: item.series_uid)
    return by_study


def _load_excluded_training_series(path: Path) -> set[tuple[str, str]]:
    fieldnames, rows = _read_csv(path)
    required = {"split", "study_uid", "series_uid"}
    missing = required - set(fieldnames)
    if missing:
        raise TrainingDataError(
            f"excluded_series.csv is missing columns: {sorted(missing)}"
        )
    excluded: set[tuple[str, str]] = set()
    for row in rows:
        split = (row.get("split") or "").strip()
        if split not in {"train", "test"}:
            raise TrainingDataError("Excluded-series manifest contains invalid split")
        if split != "train":
            continue
        study_uid = (row.get("study_uid") or "").strip()
        series_uid = (row.get("series_uid") or "").strip()
        key = (study_uid, series_uid)
        if not study_uid or not series_uid or key in excluded:
            raise TrainingDataError(
                "Excluded-series manifest contains blank or duplicate identifiers"
            )
        excluded.add(key)
    return excluded


def load_gold_training_cohort(
    competition_root: Path,
    folds_path: Path,
    exclusions_path: Path,
    *,
    fold: int = 0,
) -> GoldTrainingCohort:
    """Load one class-complete gold fold after strict series exclusions."""
    competition_root = competition_root.expanduser().resolve()
    folds_path = folds_path.expanduser().resolve()
    exclusions_path = exclusions_path.expanduser().resolve()
    if fold < 0:
        raise TrainingDataError("fold must be nonnegative")
    train_csv = competition_root / "train.csv"
    train_series_csv = competition_root / "train_series.csv"
    for path in (train_csv, train_series_csv, folds_path, exclusions_path):
        if not path.is_file():
            raise TrainingDataError(f"Required training input is missing: {path}")
    try:
        gold = load_gold_labels(train_csv)
    except GoldFoldError as error:
        raise TrainingDataError(str(error)) from error
    labels_by_study = {
        identifier: tuple(int(value) for value in gold.labels[index])
        for index, identifier in enumerate(gold.identifiers)
    }
    assignments = _load_folds(folds_path, set(labels_by_study))
    series_by_study = _load_training_series(train_series_csv)
    exclusions = _load_excluded_training_series(exclusions_path)
    known_series = {
        (study_uid, series.series_uid)
        for study_uid, series_items in series_by_study.items()
        for series in series_items
    }
    if exclusions - known_series:
        raise TrainingDataError(
            "Excluded training series are missing from train_series.csv"
        )

    selected_ids = sorted(
        identifier
        for identifier, assigned_fold in assignments.items()
        if assigned_fold == fold
    )
    if not selected_ids:
        raise TrainingDataError(f"Gold fold {fold} contains no studies")
    studies: list[GoldTrainingStudy] = []
    total_before = 0
    excluded_gold = 0
    for identifier in selected_ids:
        all_series = series_by_study.get(identifier, [])
        if not all_series:
            raise TrainingDataError("A selected gold study has no series metadata")
        total_before += len(all_series)
        eligible = tuple(
            series
            for series in all_series
            if (identifier, series.series_uid) not in exclusions
        )
        excluded_gold += len(all_series) - len(eligible)
        if not eligible:
            raise TrainingDataError(
                "A selected gold study has no eligible series after exclusions"
            )
        studies.append(
            GoldTrainingStudy(
                study_uid=identifier,
                fold=fold,
                labels=labels_by_study[identifier],
                series=eligible,
            )
        )
    labels = np.asarray([study.labels for study in studies], dtype=np.int8)
    positives = np.sum(labels, axis=0)
    negatives = len(labels) - positives
    if np.any(positives == 0) or np.any(negatives == 0):
        raise TrainingDataError(
            "Selected overfit fold is not class-complete for every target"
        )
    return GoldTrainingCohort(
        studies=tuple(studies),
        fold=fold,
        total_gold_studies=len(gold.identifiers),
        total_gold_series_before_exclusions=total_before,
        excluded_gold_series=excluded_gold,
        source_sha256={
            "train.csv": _sha256(train_csv),
            "train_series.csv": _sha256(train_series_csv),
            "gold_folds.csv": _sha256(folds_path),
            "excluded_series.csv": _sha256(exclusions_path),
        },
    )


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    temporary.replace(path)


def _validate_cached_tensor(
    path: Path, sampled_slices: int, output_size: int
) -> tuple[int, int]:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise TrainingDataError(f"Could not read cached tensor: {error}") from error
    if array.shape != (sampled_slices, output_size, output_size):
        raise TrainingDataError("Cached tensor has an unexpected shape")
    if array.dtype != np.float16:
        raise TrainingDataError("Cached tensor must use float16 storage")
    if (
        not np.all(np.isfinite(array))
        or float(np.min(array)) < 0.0
        or float(np.max(array)) > 1.0
    ):
        raise TrainingDataError("Cached tensor violates the finite [0, 1] contract")
    return int(array.shape[0]), path.stat().st_size


def materialize_gold_cache(
    cohort: GoldTrainingCohort,
    competition_root: Path,
    cache_root: Path,
    *,
    sampled_slices: int = 16,
    output_size: int = 224,
    progress: bool = False,
) -> CacheResult:
    """Decode eligible series once and store external float16 model inputs."""
    competition_root = competition_root.expanduser().resolve()
    cache_root = cache_root.expanduser().resolve()
    if sampled_slices < 1 or output_size < 1:
        raise TrainingDataError("Cache dimensions must be positive")
    if not competition_root.is_dir():
        raise TrainingDataError("Competition root does not exist")
    if (
        cache_root == competition_root
        or competition_root in cache_root.parents
        or cache_root in competition_root.parents
    ):
        raise TrainingDataError("Training cache must not be written under raw data")
    config = {
        "format_version": 1,
        "fold": cohort.fold,
        "sampled_slices": sampled_slices,
        "output_size": output_size,
        "spatial_transform": "physical_full_fov_aspect_fit_zero_pad",
        "normalization_percentiles": [1.0, 99.0],
        "source_sha256": cohort.source_sha256,
    }
    config_path = cache_root / "cache_config.json"
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise TrainingDataError(
                f"Could not read cache configuration: {error}"
            ) from error
        if existing != config:
            raise TrainingDataError(
                "Training cache configuration differs; use a new cache root"
            )
    else:
        _write_json_atomic(config_path, config)

    cache_hits = 0
    cache_misses = 0
    decoded_images = 0
    cache_bytes = 0
    cached_studies: list[CachedTrainingStudy] = []
    total_series = sum(len(study.series) for study in cohort.studies)
    completed_series = 0
    for study in cohort.studies:
        tensor_paths: list[Path] = []
        for series in study.series:
            tensor_path = (
                cache_root / "studies" / study.study_uid / f"{series.series_uid}.npy"
            )
            if tensor_path.exists():
                _, size = _validate_cached_tensor(
                    tensor_path, sampled_slices, output_size
                )
                cache_hits += 1
                cache_bytes += size
            else:
                series_directory = (
                    competition_root
                    / "train_series"
                    / study.study_uid
                    / series.series_uid
                )
                loaded = load_dicom_series(
                    series_directory,
                    study.study_uid,
                    series.series_uid,
                    sampled_slices=sampled_slices,
                    output_size=output_size,
                )
                array = np.asarray(loaded.sampled_images, dtype=np.float16)
                tensor_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = tensor_path.with_name(
                    f".{tensor_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
                )
                with temporary.open("xb") as file:
                    np.save(file, array, allow_pickle=False)
                temporary.replace(tensor_path)
                _, size = _validate_cached_tensor(
                    tensor_path, sampled_slices, output_size
                )
                cache_misses += 1
                decoded_images += loaded.summary.decoded_pixel_arrays
                cache_bytes += size
            tensor_paths.append(tensor_path)
            completed_series += 1
            if progress and (
                completed_series <= 3
                or completed_series % 10 == 0
                or completed_series == total_series
            ):
                print(
                    f"Gold cache: {completed_series}/{total_series} series "
                    f"({cache_hits} reused, {cache_misses} decoded)",
                    flush=True,
                )
        cached_studies.append(
            CachedTrainingStudy(
                study_uid=study.study_uid,
                fold=study.fold,
                labels=study.labels,
                series=study.series,
                tensor_paths=tuple(tensor_paths),
            )
        )
    return CacheResult(
        studies=tuple(cached_studies),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        decoded_images=decoded_images,
        cache_bytes=cache_bytes,
    )
