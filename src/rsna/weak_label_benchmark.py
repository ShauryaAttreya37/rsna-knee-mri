"""Benchmark public report-derived labels against the official gold subset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rsna.evaluation import macro_roc_auc
from rsna.submission import ID_COLUMN, TARGET_COLUMNS

WEAK_LABEL_COLUMNS = tuple(f"{target}_prob" for target in TARGET_COLUMNS)


class WeakLabelBenchmarkError(RuntimeError):
    """Raised when a candidate label file violates the benchmark contract."""


@dataclass(frozen=True, slots=True)
class CandidateLabels:
    name: str
    values_by_study: Mapping[str, tuple[float, ...]]


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as error:
        raise WeakLabelBenchmarkError(f"Could not read {path}: {error}") from error
    if not fields:
        raise WeakLabelBenchmarkError(f"CSV has no header: {path}")
    return fields, rows


def _probability(value: str | None, source: str) -> float:
    try:
        result = float(value or "")
    except ValueError as error:
        raise WeakLabelBenchmarkError(f"Nonnumeric probability in {source}") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise WeakLabelBenchmarkError(f"Probability outside [0, 1] in {source}")
    return result


def load_gold(train_csv: Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Load only rows with a complete official target vector."""
    fields, rows = _read_rows(train_csv)
    required = [ID_COLUMN, *TARGET_COLUMNS]
    missing = [column for column in required if column not in fields]
    if missing:
        raise WeakLabelBenchmarkError(f"Gold CSV is missing columns: {missing}")

    identifiers: list[str] = []
    labels: list[tuple[float, ...]] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        raw = [(row.get(column) or "").strip() for column in TARGET_COLUMNS]
        if not any(raw):
            continue
        if not all(raw):
            raise WeakLabelBenchmarkError(
                f"Partially labeled gold row at {train_csv.name}:{row_number}"
            )
        uid = (row.get(ID_COLUMN) or "").strip()
        if not uid or uid in seen:
            raise WeakLabelBenchmarkError("Gold study identifiers must be unique")
        seen.add(uid)
        values = tuple(_probability(value, train_csv.name) for value in raw)
        if any(value not in {0.0, 1.0} for value in values):
            raise WeakLabelBenchmarkError("Official gold labels must be binary")
        identifiers.append(uid)
        labels.append(values)

    if not identifiers:
        raise WeakLabelBenchmarkError("Gold CSV contains no completely labeled rows")
    return tuple(identifiers), np.asarray(labels, dtype=np.float64)


def load_candidate(name: str, path: Path) -> CandidateLabels:
    fields, rows = _read_rows(path)
    required = [ID_COLUMN, *TARGET_COLUMNS]
    missing = [column for column in required if column not in fields]
    if missing:
        raise WeakLabelBenchmarkError(
            f"Candidate {name!r} is missing columns: {missing}"
        )
    values_by_study: dict[str, tuple[float, ...]] = {}
    for row_number, row in enumerate(rows, start=2):
        uid = (row.get(ID_COLUMN) or "").strip()
        if not uid or uid in values_by_study:
            raise WeakLabelBenchmarkError(
                f"Candidate {name!r} has an empty or duplicate ID at row {row_number}"
            )
        values_by_study[uid] = tuple(
            _probability(row.get(column), f"{path.name}:{row_number}:{column}")
            for column in TARGET_COLUMNS
        )
    if not values_by_study:
        raise WeakLabelBenchmarkError(f"Candidate {name!r} contains no rows")
    return CandidateLabels(name=name, values_by_study=values_by_study)


def _available_indices(candidate: CandidateLabels, ids: Sequence[str]) -> np.ndarray:
    indices = np.asarray(
        [index for index, uid in enumerate(ids) if uid in candidate.values_by_study],
        dtype=np.int64,
    )
    if len(indices) < math.ceil(0.95 * len(ids)):
        raise WeakLabelBenchmarkError(
            f"Candidate {candidate.name!r} covers only {len(indices)}/{len(ids)} "
            "gold studies"
        )
    return indices


def _matrix_for_indices(
    candidate: CandidateLabels, ids: Sequence[str], indices: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [candidate.values_by_study[ids[int(index)]] for index in indices],
        dtype=np.float64,
    )


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    auc = macro_roc_auc(labels, predictions, TARGET_COLUMNS)
    return {
        "macro_auc": auc.macro_auc,
        "per_target_auc": auc.per_target_auc,
        "binary_accuracy": float(np.mean((predictions >= 0.5) == labels)),
        "brier_score": float(np.mean(np.square(predictions - labels))),
    }


def benchmark_candidates(
    train_csv: Path,
    candidates: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    """Return identifier-free metrics for candidates and their equal-weight mean."""
    if not candidates:
        raise WeakLabelBenchmarkError("At least one candidate is required")
    names = [name for name, _path in candidates]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise WeakLabelBenchmarkError("Candidate names must be nonempty and unique")

    gold_ids, gold_labels = load_gold(train_csv)
    loaded = [load_candidate(name, path) for name, path in candidates]
    available = [_available_indices(candidate, gold_ids) for candidate in loaded]
    results: dict[str, dict[str, Any]] = {}
    for candidate, indices in zip(loaded, available, strict=True):
        matrix = _matrix_for_indices(candidate, gold_ids, indices)
        metrics = _metrics(gold_labels[indices], matrix)
        metrics["gold_studies_covered"] = len(indices)
        results[candidate.name] = metrics
    if len(loaded) > 1:
        common = np.asarray(
            sorted(set.intersection(*(set(indices.tolist()) for indices in available))),
            dtype=np.int64,
        )
        matrices = [
            _matrix_for_indices(candidate, gold_ids, common) for candidate in loaded
        ]
        mean_metrics = _metrics(
            gold_labels[common], np.mean(np.stack(matrices), axis=0)
        )
        mean_metrics["gold_studies_covered"] = len(common)
        results["equal_mean_all"] = mean_metrics

    best_name = max(results, key=lambda name: results[name]["macro_auc"])
    return {
        "status": "complete",
        "generated_at": datetime.now(UTC).isoformat(),
        "privacy": "Identifier-free aggregate metrics only.",
        "gold_studies": len(gold_ids),
        "targets": list(TARGET_COLUMNS),
        "candidates": results,
        "best_by_observed_gold_macro_auc": best_name,
        "selection_warning": (
            "The 58-study gold set is small; use these metrics as a label-source "
            "gate, not as an unbiased estimate of leaderboard performance."
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise WeakLabelBenchmarkError(f"Could not hash {path}: {error}") from error
    return digest.hexdigest()


def write_equal_mean_ensemble(
    train_csv: Path,
    candidates: Sequence[tuple[str, Path]],
    output_path: Path,
    *,
    minimum_sources: int = 2,
) -> dict[str, Any]:
    """Write an all-study equal-mean label ensemble in the training schema."""
    if not candidates:
        raise WeakLabelBenchmarkError("At least one candidate is required")
    if not 1 <= minimum_sources <= len(candidates):
        raise WeakLabelBenchmarkError("minimum_sources is outside candidate count")

    fields, rows = _read_rows(train_csv)
    if ID_COLUMN not in fields:
        raise WeakLabelBenchmarkError(f"Training CSV is missing {ID_COLUMN}")
    ids = [(row.get(ID_COLUMN) or "").strip() for row in rows]
    if any(not uid for uid in ids) or len(set(ids)) != len(ids):
        raise WeakLabelBenchmarkError("Training study identifiers must be unique")

    loaded = [load_candidate(name, path) for name, path in candidates]
    source_counts: list[int] = []
    sums = np.zeros(len(TARGET_COLUMNS), dtype=np.float64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([ID_COLUMN, *WEAK_LABEL_COLUMNS])
            for uid in ids:
                available = [
                    candidate.values_by_study[uid]
                    for candidate in loaded
                    if uid in candidate.values_by_study
                ]
                if len(available) < minimum_sources:
                    raise WeakLabelBenchmarkError(
                        f"Study has only {len(available)} candidate label sources; "
                        f"minimum is {minimum_sources}"
                    )
                values = np.mean(np.asarray(available, dtype=np.float64), axis=0)
                writer.writerow([uid, *(f"{value:.10g}" for value in values)])
                source_counts.append(len(available))
                sums += values
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "studies_written": len(ids),
        "minimum_sources_required": minimum_sources,
        "minimum_sources_observed": min(source_counts),
        "maximum_sources_observed": max(source_counts),
        "mean_target_probability": {
            target: float(value)
            for target, value in zip(TARGET_COLUMNS, sums / len(ids), strict=True)
        },
        "source_sha256": {
            name: _sha256(path) for name, path in candidates
        },
        "output_sha256": _sha256(output_path),
    }


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
