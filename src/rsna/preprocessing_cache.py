"""Resumable full-cohort DICOM preprocessing and tensor caching."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import time
import uuid
from collections import Counter
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rsna.dicom_loader import load_dicom_series
from rsna.submission import ID_COLUMN, TARGET_COLUMNS
from rsna.training_data import PLANE_TO_INDEX, SERIES_UID_COLUMN, TrainingSeries

CACHE_MANIFEST_COLUMNS = (
    ID_COLUMN,
    SERIES_UID_COLUMN,
    "Anatomical_Plane",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "tensor_relative_path",
    "tensor_sha256",
    "tensor_bytes",
    "source_slices",
    "sort_strategy",
    "rows",
    "columns",
    "pixel_spacing_row_mm",
    "pixel_spacing_column_mm",
    "physical_fov_row_mm",
    "physical_fov_column_mm",
    "resized_rows",
    "resized_columns",
    "padded_fraction",
    "effective_output_spacing_mm",
)

FAILURE_COLUMNS = (
    ID_COLUMN,
    SERIES_UID_COLUMN,
    "error_class",
    "error_message",
)


class PreprocessingCacheError(RuntimeError):
    """Raised when the full-cohort cache cannot be built safely."""


@dataclass(frozen=True, slots=True)
class CacheSeriesTask:
    order: int
    study_uid: str
    series: TrainingSeries


@dataclass(frozen=True, slots=True)
class FullCohortCachePlan:
    tasks: tuple[CacheSeriesTask, ...]
    studies: int
    total_series_before_exclusions: int
    excluded_series: int
    source_sha256: dict[str, str]
    series_by_plane: dict[str, int]


@dataclass(frozen=True, slots=True)
class _WorkerInput:
    task: CacheSeriesTask
    competition_root: str
    cache_root: str
    config_sha256: str
    sampled_slices: int
    output_size: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise PreprocessingCacheError(
            f"Could not fingerprint {path.name}: {error}"
        ) from error
    return digest.hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    serialized = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as file:
            json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise PreprocessingCacheError(
            f"Could not write atomic JSON artifact {path.name}: {error}"
        ) from error


def _write_csv_atomic(
    path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    except (OSError, csv.Error, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise PreprocessingCacheError(
            f"Could not write atomic CSV artifact {path.name}: {error}"
        ) from error
    return _sha256_file(path)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as error:
        raise PreprocessingCacheError(f"Could not read {path.name}: {error}") from error
    if not fieldnames:
        raise PreprocessingCacheError(f"CSV has no header: {path.name}")
    return fieldnames, rows


def _identifier(value: str | None, context: str) -> str:
    identifier = (value or "").strip()
    if (
        not identifier
        or identifier in {".", ".."}
        or "/" in identifier
        or "\\" in identifier
        or Path(identifier).name != identifier
    ):
        raise PreprocessingCacheError(f"{context} contains an unsafe identifier")
    return identifier


def _binary_flag(value: str | None, name: str) -> int:
    try:
        flag = int(value or "")
    except ValueError as error:
        raise PreprocessingCacheError(
            f"train_series.csv contains invalid {name}"
        ) from error
    if flag not in {0, 1}:
        raise PreprocessingCacheError(f"train_series.csv contains nonbinary {name}")
    return flag


def build_full_cohort_cache_plan(
    competition_root: Path,
    exclusions_path: Path,
    *,
    expected_studies: int | None = None,
    expected_series: int | None = None,
    expected_excluded_series: int | None = None,
) -> FullCohortCachePlan:
    """Validate source metadata and return every eligible training series."""
    competition_root = competition_root.expanduser().resolve()
    exclusions_path = exclusions_path.expanduser().resolve()
    train_path = competition_root / "train.csv"
    train_series_path = competition_root / "train_series.csv"
    for path in (train_path, train_series_path, exclusions_path):
        if not path.is_file():
            raise PreprocessingCacheError(f"Required input is missing: {path}")

    train_fields, train_rows = _read_csv(train_path)
    expected_train_fields = [ID_COLUMN, "Report", *TARGET_COLUMNS]
    if train_fields != expected_train_fields:
        raise PreprocessingCacheError("train.csv columns or column order are invalid")
    study_ids: list[str] = []
    seen_studies: set[str] = set()
    for row in train_rows:
        study_uid = _identifier(row.get(ID_COLUMN), "train.csv")
        if study_uid in seen_studies:
            raise PreprocessingCacheError(
                "train.csv contains duplicate study identifiers"
            )
        if not (row.get("Report") or "").strip():
            raise PreprocessingCacheError("train.csv contains a blank report")
        seen_studies.add(study_uid)
        study_ids.append(study_uid)
    if not study_ids:
        raise PreprocessingCacheError("train.csv contains no studies")
    if expected_studies is not None and len(study_ids) != expected_studies:
        raise PreprocessingCacheError(
            f"Expected {expected_studies} studies, observed {len(study_ids)}"
        )

    series_fields, series_rows = _read_csv(train_series_path)
    expected_series_fields = [
        ID_COLUMN,
        SERIES_UID_COLUMN,
        "Fluid_Sensitive",
        "Fat_Suppression",
        "Anatomical_Plane",
    ]
    if series_fields != expected_series_fields:
        raise PreprocessingCacheError(
            "train_series.csv columns or column order are invalid"
        )
    series_by_study: dict[str, list[TrainingSeries]] = {}
    known_series: set[tuple[str, str]] = set()
    for row in series_rows:
        study_uid = _identifier(row.get(ID_COLUMN), "train_series.csv")
        series_uid = _identifier(row.get(SERIES_UID_COLUMN), "train_series.csv")
        key = (study_uid, series_uid)
        if key in known_series:
            raise PreprocessingCacheError(
                "train_series.csv contains duplicate series identifiers"
            )
        plane = (row.get("Anatomical_Plane") or "").strip()
        if plane not in PLANE_TO_INDEX:
            raise PreprocessingCacheError(
                "train_series.csv contains an invalid anatomical plane"
            )
        known_series.add(key)
        series_by_study.setdefault(study_uid, []).append(
            TrainingSeries(
                series_uid=series_uid,
                anatomical_plane=plane,
                fluid_sensitive=_binary_flag(
                    row.get("Fluid_Sensitive"), "Fluid_Sensitive"
                ),
                fat_suppression=_binary_flag(
                    row.get("Fat_Suppression"), "Fat_Suppression"
                ),
            )
        )
    if expected_series is not None and len(known_series) != expected_series:
        raise PreprocessingCacheError(
            f"Expected {expected_series} series, observed {len(known_series)}"
        )
    unknown_studies = set(series_by_study) - seen_studies
    missing_studies = seen_studies - set(series_by_study)
    if unknown_studies or missing_studies:
        raise PreprocessingCacheError(
            "Study membership differs between train.csv and train_series.csv"
        )
    for series_items in series_by_study.values():
        series_items.sort(key=lambda item: item.series_uid)

    exclusion_fields, exclusion_rows = _read_csv(exclusions_path)
    required_exclusion_fields = {"split", "study_uid", "series_uid"}
    if not required_exclusion_fields.issubset(exclusion_fields):
        raise PreprocessingCacheError("excluded_series.csv columns are invalid")
    excluded: set[tuple[str, str]] = set()
    for row in exclusion_rows:
        split = (row.get("split") or "").strip()
        if split not in {"train", "test"}:
            raise PreprocessingCacheError(
                "excluded_series.csv contains an invalid split"
            )
        if split != "train":
            continue
        study_uid = _identifier(row.get("study_uid"), "excluded_series.csv")
        series_uid = _identifier(row.get("series_uid"), "excluded_series.csv")
        key = (study_uid, series_uid)
        if key in excluded:
            raise PreprocessingCacheError(
                "excluded_series.csv contains duplicate training series"
            )
        excluded.add(key)
    if excluded - known_series:
        raise PreprocessingCacheError(
            "Excluded training series are missing from train_series.csv"
        )
    if (
        expected_excluded_series is not None
        and len(excluded) != expected_excluded_series
    ):
        raise PreprocessingCacheError(
            f"Expected {expected_excluded_series} excluded training series, "
            f"observed {len(excluded)}"
        )

    train_series_root = competition_root / "train_series"
    if not train_series_root.is_dir():
        raise PreprocessingCacheError("Training series directory is missing")
    missing_series_directories = sum(
        not (train_series_root / study_uid / series_uid).is_dir()
        for study_uid, series_uid in known_series
    )
    if missing_series_directories:
        raise PreprocessingCacheError(
            f"Raw data is missing {missing_series_directories} series directories"
        )

    tasks: list[CacheSeriesTask] = []
    by_plane: Counter[str] = Counter()
    order = 0
    for study_uid in study_ids:
        eligible = [
            series
            for series in series_by_study[study_uid]
            if (study_uid, series.series_uid) not in excluded
        ]
        if not eligible:
            raise PreprocessingCacheError(
                "A training study has no eligible series after exclusions"
            )
        for series in eligible:
            tasks.append(
                CacheSeriesTask(order=order, study_uid=study_uid, series=series)
            )
            by_plane[series.anatomical_plane] += 1
            order += 1

    return FullCohortCachePlan(
        tasks=tuple(tasks),
        studies=len(study_ids),
        total_series_before_exclusions=len(known_series),
        excluded_series=len(excluded),
        source_sha256={
            "train.csv": _sha256_file(train_path),
            "train_series.csv": _sha256_file(train_series_path),
            "excluded_series.csv": _sha256_file(exclusions_path),
        },
        series_by_plane=dict(sorted(by_plane.items())),
    )


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_layout(
    competition_root: Path,
    cache_root: Path,
    project_root: Path,
    report_output: Path,
) -> None:
    if not competition_root.is_dir() or not project_root.is_dir():
        raise PreprocessingCacheError("Competition and project roots must exist")
    if _within(cache_root, competition_root) or _within(competition_root, cache_root):
        raise PreprocessingCacheError(
            "Cache root must not overlap raw competition data"
        )
    if _within(cache_root, project_root) or _within(project_root, cache_root):
        raise PreprocessingCacheError("Identifier-bearing cache must remain external")
    if not _within(report_output, project_root):
        raise PreprocessingCacheError("Aggregate report must remain inside the project")
    if _within(report_output, competition_root) or _within(report_output, cache_root):
        raise PreprocessingCacheError("Aggregate report path overlaps an unsafe root")


def _cache_paths(cache_root: Path, task: CacheSeriesTask) -> tuple[Path, Path]:
    directory = cache_root / "studies" / task.study_uid
    return (
        directory / f"{task.series.series_uid}.npy",
        directory / f"{task.series.series_uid}.json",
    )


def _inspect_tensor(
    path: Path, sampled_slices: int, output_size: int
) -> dict[str, Any]:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise PreprocessingCacheError(
            f"Could not read cached tensor: {error}"
        ) from error
    if array.shape != (sampled_slices, output_size, output_size):
        raise PreprocessingCacheError("Cached tensor has an unexpected shape")
    if array.dtype != np.float16:
        raise PreprocessingCacheError("Cached tensor must use float16 storage")
    if not np.all(np.isfinite(array)):
        raise PreprocessingCacheError("Cached tensor contains nonfinite values")
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if minimum < 0.0 or maximum > 1.0:
        raise PreprocessingCacheError("Cached tensor is outside [0, 1]")
    return {
        "tensor_bytes": path.stat().st_size,
        "tensor_sha256": _sha256_file(path),
        "tensor_minimum": minimum,
        "tensor_maximum": maximum,
    }


def _validate_record(
    record: dict[str, Any],
    task: CacheSeriesTask,
    tensor_path: Path,
    config_sha256: str,
    sampled_slices: int,
    output_size: int,
) -> dict[str, Any]:
    expected = {
        "format_version": 1,
        "config_sha256": config_sha256,
        ID_COLUMN: task.study_uid,
        SERIES_UID_COLUMN: task.series.series_uid,
        "Anatomical_Plane": task.series.anatomical_plane,
        "Fluid_Sensitive": task.series.fluid_sensitive,
        "Fat_Suppression": task.series.fat_suppression,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise PreprocessingCacheError("Cached tensor metadata differs from its plan")
    if not isinstance(record.get("summary"), dict):
        raise PreprocessingCacheError("Cached tensor metadata has no load summary")
    inspection = _inspect_tensor(tensor_path, sampled_slices, output_size)
    for key in ("tensor_bytes", "tensor_sha256", "tensor_minimum", "tensor_maximum"):
        if record.get(key) != inspection[key]:
            raise PreprocessingCacheError("Cached tensor fingerprint is inconsistent")
    return record


def _load_cached_record(
    metadata_path: Path,
    task: CacheSeriesTask,
    tensor_path: Path,
    config_sha256: str,
    sampled_slices: int,
    output_size: int,
) -> dict[str, Any]:
    try:
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreprocessingCacheError(
            f"Could not read cached tensor metadata: {error}"
        ) from error
    if not isinstance(record, dict):
        raise PreprocessingCacheError("Cached tensor metadata must be an object")
    return _validate_record(
        record,
        task,
        tensor_path,
        config_sha256,
        sampled_slices,
        output_size,
    )


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cache_one_series(worker_input: _WorkerInput) -> dict[str, Any]:
    task = worker_input.task
    competition_root = Path(worker_input.competition_root)
    cache_root = Path(worker_input.cache_root)
    tensor_path, metadata_path = _cache_paths(cache_root, task)
    series_directory = (
        competition_root / "train_series" / task.study_uid / task.series.series_uid
    )
    started = time.monotonic()
    loaded = load_dicom_series(
        series_directory,
        task.study_uid,
        task.series.series_uid,
        sampled_slices=worker_input.sampled_slices,
        output_size=worker_input.output_size,
    )
    array = np.asarray(loaded.sampled_images, dtype=np.float16)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tensor_path.with_name(
        f".{tensor_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as file:
            np.save(file, array, allow_pickle=False)
            file.flush()
            os.fsync(file.fileno())
        inspection = _inspect_tensor(
            temporary, worker_input.sampled_slices, worker_input.output_size
        )
        temporary.replace(tensor_path)
        _fsync_directory(tensor_path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    record: dict[str, Any] = {
        "format_version": 1,
        "config_sha256": worker_input.config_sha256,
        ID_COLUMN: task.study_uid,
        SERIES_UID_COLUMN: task.series.series_uid,
        "Anatomical_Plane": task.series.anatomical_plane,
        "Fluid_Sensitive": task.series.fluid_sensitive,
        "Fat_Suppression": task.series.fat_suppression,
        **inspection,
        "decoded_pixel_arrays": loaded.summary.decoded_pixel_arrays,
        "elapsed_seconds": time.monotonic() - started,
        "summary": loaded.summary.to_dict(),
    }
    _write_json_atomic(metadata_path, record)
    return record


def _record_to_manifest_row(
    task: CacheSeriesTask, record: dict[str, Any], cache_root: Path
) -> dict[str, Any]:
    tensor_path, _ = _cache_paths(cache_root, task)
    summary = record["summary"]
    return {
        ID_COLUMN: task.study_uid,
        SERIES_UID_COLUMN: task.series.series_uid,
        "Anatomical_Plane": task.series.anatomical_plane,
        "Fluid_Sensitive": task.series.fluid_sensitive,
        "Fat_Suppression": task.series.fat_suppression,
        "tensor_relative_path": tensor_path.relative_to(cache_root).as_posix(),
        "tensor_sha256": record["tensor_sha256"],
        "tensor_bytes": record["tensor_bytes"],
        "source_slices": summary["slices"],
        "sort_strategy": summary["sort_strategy"],
        "rows": summary["rows"],
        "columns": summary["columns"],
        "pixel_spacing_row_mm": summary["pixel_spacing_row_mm"],
        "pixel_spacing_column_mm": summary["pixel_spacing_column_mm"],
        "physical_fov_row_mm": summary["physical_fov_row_mm"],
        "physical_fov_column_mm": summary["physical_fov_column_mm"],
        "resized_rows": summary["resized_rows"],
        "resized_columns": summary["resized_columns"],
        "padded_fraction": summary["padded_fraction"],
        "effective_output_spacing_mm": summary["effective_output_spacing_mm"],
    }


def _process_missing_tasks(
    tasks: list[CacheSeriesTask],
    worker_template: dict[str, Any],
    workers: int,
    on_success: Callable[[CacheSeriesTask, dict[str, Any]], None],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    def worker_input(task: CacheSeriesTask) -> _WorkerInput:
        return _WorkerInput(task=task, **worker_template)

    if workers == 1:
        for task in tasks:
            try:
                record = _cache_one_series(worker_input(task))
            except Exception as error:  # noqa: BLE001 - persist worker failures
                failures.append(
                    {
                        ID_COLUMN: task.study_uid,
                        SERIES_UID_COLUMN: task.series.series_uid,
                        "error_class": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                break
            records.append(record)
            on_success(task, record)
        return records, failures

    task_iterator = iter(tasks)
    future_tasks: dict[Future[dict[str, Any]], CacheSeriesTask] = {}
    executor = ProcessPoolExecutor(max_workers=workers)
    failed = False
    try:
        for _ in range(min(len(tasks), workers * 2)):
            task = next(task_iterator, None)
            if task is None:
                break
            future_tasks[executor.submit(_cache_one_series, worker_input(task))] = task
        while future_tasks and not failed:
            completed, _ = wait(future_tasks, return_when=FIRST_COMPLETED)
            for future in completed:
                task = future_tasks.pop(future)
                try:
                    record = future.result()
                except Exception as error:  # noqa: BLE001 - persist worker failures
                    failures.append(
                        {
                            ID_COLUMN: task.study_uid,
                            SERIES_UID_COLUMN: task.series.series_uid,
                            "error_class": type(error).__name__,
                            "error_message": str(error),
                        }
                    )
                    failed = True
                    break
                records.append(record)
                on_success(task, record)
                next_task = next(task_iterator, None)
                if next_task is not None:
                    future_tasks[
                        executor.submit(_cache_one_series, worker_input(next_task))
                    ] = next_task
        if failed:
            for future in future_tasks:
                future.cancel()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return records, failures


def build_full_cohort_cache(
    competition_root: Path,
    exclusions_path: Path,
    cache_root: Path,
    project_root: Path,
    report_output: Path,
    *,
    sampled_slices: int = 16,
    output_size: int = 224,
    workers: int = 4,
    max_new_series: int | None = None,
    minimum_free_gib: float = 20.0,
    progress_every: int = 25,
    expected_studies: int | None = 4407,
    expected_series: int | None = 24371,
    expected_excluded_series: int | None = 3,
) -> dict[str, Any]:
    """Build or resume a validated, identifier-bearing external tensor cache."""
    for name, value in (
        ("sampled_slices", sampled_slices),
        ("output_size", output_size),
        ("workers", workers),
        ("progress_every", progress_every),
    ):
        if int(value) < 1:
            raise PreprocessingCacheError(f"{name} must be positive")
    if max_new_series is not None and max_new_series < 0:
        raise PreprocessingCacheError("max_new_series must be nonnegative")
    if not math.isfinite(minimum_free_gib) or minimum_free_gib < 0.0:
        raise PreprocessingCacheError("minimum_free_gib must be finite and nonnegative")

    competition_root = competition_root.expanduser().resolve()
    exclusions_path = exclusions_path.expanduser().resolve()
    cache_root = cache_root.expanduser().resolve()
    project_root = project_root.expanduser().resolve()
    report_output = report_output.expanduser().resolve()
    _validate_layout(competition_root, cache_root, project_root, report_output)
    plan = build_full_cohort_cache_plan(
        competition_root,
        exclusions_path,
        expected_studies=expected_studies,
        expected_series=expected_series,
        expected_excluded_series=expected_excluded_series,
    )
    config = {
        "format_version": 1,
        "eligible_studies": plan.studies,
        "eligible_series": len(plan.tasks),
        "excluded_series": plan.excluded_series,
        "sampled_slices": sampled_slices,
        "output_size": output_size,
        "dtype": "float16",
        "storage_format": "npy_allow_pickle_false",
        "spatial_transform": "physical_full_fov_aspect_fit_zero_pad",
        "normalization_percentiles": [1.0, 99.0],
        "normalization_maximum_sample_values": 1_000_000,
        "slice_sampling": "rounded_uniform_linspace",
        "maximum_slice_normal_angle_degrees": 1.0,
        "implementation_sha256": {
            "dicom_loader.py": _sha256_file(
                Path(__file__).with_name("dicom_loader.py")
            ),
            "preprocessing_cache.py": _sha256_file(Path(__file__)),
        },
        "dependency_versions": {
            "numpy": importlib.metadata.version("numpy"),
            "pydicom": importlib.metadata.version("pydicom"),
        },
        "source_sha256": plan.source_sha256,
    }
    config_sha256 = _canonical_sha256(config)
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / ".cache.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PreprocessingCacheError(
                "Another process is already building this cache"
            ) from error

        config_path = cache_root / "cache_config.json"
        if config_path.exists():
            try:
                existing_config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise PreprocessingCacheError(
                    f"Could not read cache configuration: {error}"
                ) from error
            if existing_config != config:
                raise PreprocessingCacheError(
                    "Cache configuration differs; use a new cache root"
                )
        else:
            studies_root = cache_root / "studies"
            if studies_root.exists() and any(studies_root.rglob("*.npy")):
                raise PreprocessingCacheError(
                    "Cache tensors exist without a cache configuration"
                )
            _write_json_atomic(config_path, config)

        expected_tensor_paths = {
            _cache_paths(cache_root, task)[0] for task in plan.tasks
        }
        expected_metadata_paths = {
            _cache_paths(cache_root, task)[1] for task in plan.tasks
        }
        studies_root = cache_root / "studies"
        observed_tensors = (
            set(studies_root.rglob("*.npy")) if studies_root.exists() else set()
        )
        observed_metadata = (
            set(studies_root.rglob("*.json")) if studies_root.exists() else set()
        )
        if (
            observed_tensors - expected_tensor_paths
            or observed_metadata - expected_metadata_paths
        ):
            raise PreprocessingCacheError("Cache contains unexpected tensor artifacts")

        started = time.monotonic()
        records_by_order: dict[int, dict[str, Any]] = {}
        cache_hits = 0
        orphaned_tensors = 0
        missing_tasks: list[CacheSeriesTask] = []
        for task in plan.tasks:
            tensor_path, metadata_path = _cache_paths(cache_root, task)
            if tensor_path.exists() and metadata_path.exists():
                records_by_order[task.order] = _load_cached_record(
                    metadata_path,
                    task,
                    tensor_path,
                    config_sha256,
                    sampled_slices,
                    output_size,
                )
                cache_hits += 1
            elif metadata_path.exists() and not tensor_path.exists():
                raise PreprocessingCacheError(
                    "Cached metadata exists without its tensor"
                )
            else:
                if tensor_path.exists():
                    orphaned_tensors += 1
                missing_tasks.append(task)

        data_bytes_per_tensor = sampled_slices * output_size * output_size * 2
        estimated_remaining_bytes = len(missing_tasks) * (data_bytes_per_tensor + 4096)
        free_bytes = shutil.disk_usage(cache_root).free
        required_free_bytes = estimated_remaining_bytes + int(
            minimum_free_gib * 1024**3
        )
        if free_bytes < required_free_bytes:
            raise PreprocessingCacheError(
                "Insufficient free space for the remaining cache and safety margin"
            )

        selected_missing = (
            missing_tasks if max_new_series is None else missing_tasks[:max_new_series]
        )
        completed_this_run = 0
        decoded_images = 0
        decoded_seconds = 0.0

        def progress_payload(status: str) -> dict[str, Any]:
            elapsed = time.monotonic() - started
            completed = len(records_by_order)
            run_rate = completed_this_run / elapsed if elapsed > 0 else 0.0
            remaining = len(plan.tasks) - completed
            return {
                "status": status,
                "updated_at": datetime.now(UTC).isoformat(),
                "eligible_series": len(plan.tasks),
                "completed_series": completed,
                "remaining_series": remaining,
                "cache_hits": cache_hits,
                "decoded_this_run": completed_this_run,
                "decoded_images_this_run": decoded_images,
                "elapsed_seconds": elapsed,
                "series_per_second": run_rate,
                "eta_seconds_at_current_rate": remaining / run_rate
                if run_rate > 0
                else None,
            }

        def on_success(task: CacheSeriesTask, record: dict[str, Any]) -> None:
            nonlocal completed_this_run, decoded_images, decoded_seconds
            records_by_order[task.order] = record
            completed_this_run += 1
            decoded_images += int(record["decoded_pixel_arrays"])
            decoded_seconds += float(record["elapsed_seconds"])
            if (
                completed_this_run <= 3
                or completed_this_run % progress_every == 0
                or len(records_by_order) == len(plan.tasks)
            ):
                payload = progress_payload("running")
                _write_json_atomic(cache_root / "progress.json", payload)
                eta = payload["eta_seconds_at_current_rate"]
                eta_text = f"{eta / 3600:.2f}h" if eta is not None else "unknown"
                print(
                    "Training cache: "
                    f"{payload['completed_series']}/{payload['eligible_series']} series "
                    f"({cache_hits} reused, {completed_this_run} decoded) | "
                    f"{payload['series_per_second']:.2f} series/s | ETA {eta_text}",
                    flush=True,
                )

        new_records, failures = _process_missing_tasks(
            selected_missing,
            {
                "competition_root": str(competition_root),
                "cache_root": str(cache_root),
                "config_sha256": config_sha256,
                "sampled_slices": sampled_slices,
                "output_size": output_size,
            },
            workers,
            on_success,
        )
        if len(new_records) != completed_this_run:
            raise AssertionError("Worker completion accounting is inconsistent")

        failure_path = cache_root / "failures.csv"
        _write_csv_atomic(failure_path, FAILURE_COLUMNS, failures)
        manifest_rows = [
            _record_to_manifest_row(task, records_by_order[task.order], cache_root)
            for task in plan.tasks
            if task.order in records_by_order
        ]
        manifest_path = cache_root / "cache_manifest.csv"
        manifest_sha256 = _write_csv_atomic(
            manifest_path, CACHE_MANIFEST_COLUMNS, manifest_rows
        )
        elapsed = time.monotonic() - started
        complete = len(records_by_order) == len(plan.tasks) and not failures
        status = "complete" if complete else "partial" if not failures else "fail"
        total_cache_bytes = sum(
            int(record["tensor_bytes"]) for record in records_by_order.values()
        )
        error_counts = Counter(row["error_class"] for row in failures)
        report = {
            "status": status,
            "generated_at": datetime.now(UTC).isoformat(),
            "privacy": (
                "Identifier-free aggregate cache statistics only; study/series IDs, "
                "tensor mappings, and failures remain in the external cache."
            ),
            "configuration": {
                "config_sha256": config_sha256,
                "sampled_slices_per_series": sampled_slices,
                "output_size": output_size,
                "dtype": "float16",
                "workers": workers,
                "spatial_transform": "physical_full_fov_aspect_fit_zero_pad",
                "normalization_percentiles": [1.0, 99.0],
                "slice_sampling": "rounded_uniform_linspace",
                "implementation_sha256": config["implementation_sha256"],
                "dependency_versions": config["dependency_versions"],
            },
            "counts": {
                "studies": plan.studies,
                "series_before_exclusions": plan.total_series_before_exclusions,
                "excluded_series": plan.excluded_series,
                "eligible_series": len(plan.tasks),
                "completed_series": len(records_by_order),
                "remaining_series": len(plan.tasks) - len(records_by_order),
                "cache_hits": cache_hits,
                "cache_misses_completed": completed_this_run,
                "orphaned_tensors_recomputed": orphaned_tensors,
                "decoded_images_this_run": decoded_images,
                "failures": len(failures),
            },
            "series_by_plane": plan.series_by_plane,
            "storage": {
                "tensor_bytes_completed": total_cache_bytes,
                "estimated_tensor_payload_bytes_full": len(plan.tasks)
                * data_bytes_per_tensor,
                "free_bytes_after_run": shutil.disk_usage(cache_root).free,
                "minimum_free_margin_bytes": int(minimum_free_gib * 1024**3),
            },
            "runtime": {
                "elapsed_seconds": elapsed,
                "decoded_series_per_second": completed_this_run / elapsed
                if elapsed > 0
                else 0.0,
                "summed_worker_decode_seconds": decoded_seconds,
            },
            "source_sha256": plan.source_sha256,
            "external_artifacts": {
                "cache_config": "cache_config.json",
                "cache_manifest": "cache_manifest.csv",
                "cache_manifest_sha256": manifest_sha256,
                "failures": "failures.csv",
                "progress": "progress.json",
                "tensor_tree": "studies/",
            },
            "error_class_counts": dict(sorted(error_counts.items())),
        }
        _write_json_atomic(report_output, report)
        _write_json_atomic(cache_root / "progress.json", progress_payload(status))
        if failures:
            raise PreprocessingCacheError(
                "Series caching failed; inspect the external failures.csv"
            )
        return report


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data"))
    project_root = Path(
        os.environ.get("RSNA_PROJECT_ROOT", "/home/shaurya/rsna-project")
    )
    parser = argparse.ArgumentParser(
        description="Build a strict, resumable full-cohort 2.5D tensor cache."
    )
    parser.add_argument(
        "--competition-root", type=Path, default=data_root / "raw" / "competition"
    )
    parser.add_argument(
        "--exclusions-path",
        type=Path,
        default=data_root / "manifests" / "pixel_integrity_v1" / "excluded_series.csv",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=data_root / "processed" / "training_cache_v1",
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=project_root / "reports" / "preprocessing_cache" / "summary.json",
    )
    parser.add_argument("--sampled-slices", type=int, default=16)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-new-series", type=int, default=None)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--expected-studies", type=int, default=4407)
    parser.add_argument("--expected-series", type=int, default=24371)
    parser.add_argument("--expected-excluded-series", type=int, default=3)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and initialize metadata without decoding new series.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_full_cohort_cache(
            args.competition_root,
            args.exclusions_path,
            args.cache_root,
            args.project_root,
            args.report_output,
            sampled_slices=args.sampled_slices,
            output_size=args.output_size,
            workers=args.workers,
            max_new_series=0 if args.dry_run else args.max_new_series,
            minimum_free_gib=args.minimum_free_gib,
            progress_every=args.progress_every,
            expected_studies=args.expected_studies,
            expected_series=args.expected_series,
            expected_excluded_series=args.expected_excluded_series,
        )
    except PreprocessingCacheError as error:
        print(f"Training cache FAIL: {error}")
        return 2
    print(
        f"Training cache {report['status'].upper()}: "
        f"{report['counts']['completed_series']}/"
        f"{report['counts']['eligible_series']} series"
    )
    return 0 if report["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
