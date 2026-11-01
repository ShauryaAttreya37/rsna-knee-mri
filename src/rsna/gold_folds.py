"""Deterministic, leakage-safe multilabel folds for fully gold-labeled studies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rsna.submission import ID_COLUMN, TARGET_COLUMNS


class GoldFoldError(RuntimeError):
    """Raised when class-complete study-level folds cannot be created safely."""


@dataclass(frozen=True, slots=True)
class GoldLabels:
    identifiers: tuple[str, ...]
    labels: np.ndarray
    total_rows: int
    unlabeled_rows: int
    partial_label_rows: int


def _parse_binary(value: str, target: str) -> int:
    try:
        parsed = float(value)
    except ValueError as error:
        raise GoldFoldError(
            f"Gold target {target!r} contains a nonnumeric value"
        ) from error
    if not math.isfinite(parsed) or parsed not in {0.0, 1.0}:
        raise GoldFoldError(f"Gold target {target!r} must contain only 0/1 values")
    return int(parsed)


def load_gold_labels(train_csv: Path) -> GoldLabels:
    """Read labels without retaining or emitting report text."""
    identifiers: list[str] = []
    labels: list[list[int]] = []
    seen: set[str] = set()
    total_rows = 0
    unlabeled_rows = 0
    partial_rows = 0
    try:
        with train_csv.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required = {ID_COLUMN, "Report", *TARGET_COLUMNS}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise GoldFoldError(
                    f"train.csv is missing required columns: {sorted(missing)}"
                )
            for row in reader:
                total_rows += 1
                identifier = (row.get(ID_COLUMN) or "").strip()
                if not identifier or identifier in seen:
                    raise GoldFoldError(
                        "train.csv contains a blank or duplicate study ID"
                    )
                seen.add(identifier)
                values = [(row.get(target) or "").strip() for target in TARGET_COLUMNS]
                present = [bool(value) for value in values]
                if not any(present):
                    unlabeled_rows += 1
                    continue
                if not all(present):
                    partial_rows += 1
                    continue
                identifiers.append(identifier)
                labels.append(
                    [
                        _parse_binary(value, target)
                        for value, target in zip(values, TARGET_COLUMNS, strict=True)
                    ]
                )
    except (OSError, UnicodeError, csv.Error) as error:
        raise GoldFoldError(f"Could not read train.csv: {error}") from error
    if partial_rows:
        raise GoldFoldError(
            f"train.csv contains {partial_rows} partially labeled rows; refused"
        )
    if not identifiers:
        raise GoldFoldError("train.csv contains no fully gold-labeled studies")
    matrix = np.asarray(labels, dtype=np.int8)
    if matrix.shape != (len(identifiers), len(TARGET_COLUMNS)):
        raise AssertionError("Gold label matrix shape is invalid")
    return GoldLabels(
        identifiers=tuple(identifiers),
        labels=matrix,
        total_rows=total_rows,
        unlabeled_rows=unlabeled_rows,
        partial_label_rows=partial_rows,
    )


def _stable_number(identifier: str, seed: int, restart: int) -> int:
    digest = hashlib.sha256(f"{seed}:{restart}:{identifier}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _fold_sizes(sample_count: int, fold_count: int) -> np.ndarray:
    sizes = np.full(fold_count, sample_count // fold_count, dtype=int)
    sizes[: sample_count % fold_count] += 1
    return sizes


def _assignment_quality(
    labels: np.ndarray,
    assignments: np.ndarray,
    target_sizes: np.ndarray,
) -> tuple[int, float]:
    fold_count = len(target_sizes)
    total_positives = np.sum(labels, axis=0, dtype=float)
    desired = target_sizes[:, None] * total_positives[None, :] / len(labels)
    positives = np.stack(
        [np.sum(labels[assignments == fold], axis=0) for fold in range(fold_count)]
    ).astype(float)
    negatives = target_sizes[:, None] - positives
    violations = int(np.sum((positives == 0) | (negatives == 0)))
    balance_error = float(np.sum((positives - desired) ** 2 / (desired + 0.5)))
    return violations, balance_error


def assign_multilabel_folds(
    identifiers: Sequence[str],
    labels: np.ndarray,
    fold_count: int = 5,
    seed: int = 20260828,
    restarts: int = 512,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Use deterministic multi-start greedy balancing with hard class checks."""
    sample_count = len(identifiers)
    if labels.shape != (sample_count, len(TARGET_COLUMNS)):
        raise GoldFoldError("Label matrix shape does not match identifiers and targets")
    if fold_count < 2 or fold_count > sample_count:
        raise GoldFoldError("fold_count must be between 2 and the gold sample count")
    if restarts < 1:
        raise GoldFoldError("restarts must be positive")
    positives = np.sum(labels, axis=0)
    negatives = sample_count - positives
    if np.any(positives < fold_count) or np.any(negatives < fold_count):
        raise GoldFoldError(
            "Every target needs at least one positive and negative per requested fold"
        )

    target_sizes = _fold_sizes(sample_count, fold_count)
    desired = target_sizes[:, None] * positives[None, :] / sample_count
    rarity = labels @ (1.0 / positives.astype(float))
    best_assignments: np.ndarray | None = None
    best_quality: tuple[int, float] | None = None

    for restart in range(restarts):
        assignments = np.full(sample_count, -1, dtype=int)
        sizes = np.zeros(fold_count, dtype=int)
        fold_positives = np.zeros((fold_count, labels.shape[1]), dtype=float)
        remaining = set(range(sample_count))
        remaining_positives = positives.astype(int).copy()
        while np.any(remaining_positives > 0):
            rare_label = min(
                np.flatnonzero(remaining_positives > 0),
                key=lambda target: (
                    int(remaining_positives[target]),
                    _stable_number(str(target), seed, restart),
                ),
            )
            sample_candidates = [
                index for index in remaining if labels[index, rare_label] == 1
            ]
            if not sample_candidates:
                remaining_positives[rare_label] = 0
                continue
            index = min(
                sample_candidates,
                key=lambda sample_index: (
                    -float(rarity[sample_index]),
                    -int(np.sum(labels[sample_index])),
                    _stable_number(identifiers[sample_index], seed, restart),
                ),
            )
            available = [
                fold for fold in range(fold_count) if sizes[fold] < target_sizes[fold]
            ]
            if not available:
                raise AssertionError("No fold capacity remains during assignment")
            sample = labels[index].astype(float)
            placement_scores: list[tuple[float, float, float, int, int]] = []
            for fold in available:
                primary_need = (
                    desired[fold, rare_label] - fold_positives[fold, rare_label]
                )
                all_label_need = float(
                    np.sum(
                        np.maximum(desired[fold] - fold_positives[fold], 0.0)
                        * sample
                        / positives
                    )
                )
                remaining_capacity = target_sizes[fold] - sizes[fold]
                tie = _stable_number(identifiers[index], seed + fold, restart)
                placement_scores.append(
                    (
                        -primary_need,
                        -all_label_need,
                        -remaining_capacity,
                        tie,
                        fold,
                    )
                )
            chosen = min(placement_scores)[-1]
            assignments[index] = chosen
            sizes[chosen] += 1
            fold_positives[chosen] += sample
            remaining.remove(index)
            remaining_positives -= labels[index]

        for index in sorted(
            remaining,
            key=lambda sample_index: _stable_number(
                identifiers[sample_index], seed, restart
            ),
        ):
            available = [
                fold for fold in range(fold_count) if sizes[fold] < target_sizes[fold]
            ]
            chosen = max(
                available,
                key=lambda fold: (
                    target_sizes[fold] - sizes[fold],
                    -_stable_number(identifiers[index], seed + fold, restart),
                ),
            )
            assignments[index] = chosen
            sizes[chosen] += 1
            fold_positives[chosen] += labels[index]
        quality = _assignment_quality(labels, assignments, target_sizes)
        if best_quality is None or quality < best_quality:
            best_quality = quality
            best_assignments = assignments.copy()
            if quality[0] == 0 and quality[1] <= 1e-12:
                break

    if best_assignments is None or best_quality is None:
        raise AssertionError("No fold assignment was evaluated")
    if best_quality[0]:
        raise GoldFoldError(
            f"Could not create class-complete folds after {restarts} deterministic restarts"
        )
    return best_assignments, {
        "class_coverage_violations": best_quality[0],
        "balance_objective": best_quality[1],
        "restarts": restarts,
    }


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_roots(
    competition_root: Path,
    project_root: Path,
    output_root: Path,
    report_output: Path,
) -> None:
    if not competition_root.is_dir() or not project_root.is_dir():
        raise GoldFoldError("Competition and project roots must exist")
    if _within(output_root, competition_root) or _within(competition_root, output_root):
        raise GoldFoldError("Fold output must not overlap raw competition data")
    if _within(output_root, project_root) or _within(project_root, output_root):
        raise GoldFoldError(
            "Identifier-bearing fold output must remain outside the project"
        )
    if _within(report_output, competition_root):
        raise GoldFoldError("Aggregate report must remain outside raw data")


def _write_folds_atomic(
    path: Path, identifiers: Sequence[str], assignments: np.ndarray
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow([ID_COLUMN, "fold"])
        writer.writerows(
            sorted(
                zip(identifiers, assignments.tolist(), strict=True),
                key=lambda item: item[0],
            )
        )
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    temporary.replace(path)


def build_fold_summary(
    gold: GoldLabels,
    assignments: np.ndarray,
    fold_count: int,
    seed: int,
    assignment_metrics: dict[str, float | int],
    fold_manifest_sha256: str,
) -> dict[str, Any]:
    fold_sizes = [int(np.sum(assignments == fold)) for fold in range(fold_count)]
    target_summary: dict[str, Any] = {}
    coverage_violations = 0
    for target_index, target in enumerate(TARGET_COLUMNS):
        values = gold.labels[:, target_index]
        folds = []
        for fold in range(fold_count):
            fold_values = values[assignments == fold]
            positive = int(np.sum(fold_values))
            negative = int(len(fold_values) - positive)
            coverage_violations += int(positive == 0 or negative == 0)
            folds.append(
                {
                    "fold": fold,
                    "positive": positive,
                    "negative": negative,
                    "prevalence": positive / len(fold_values),
                }
            )
        total_positive = int(np.sum(values))
        target_summary[target] = {
            "overall_positive": total_positive,
            "overall_negative": len(values) - total_positive,
            "overall_prevalence": total_positive / len(values),
            "folds": folds,
        }
    checks = {
        "partial_label_rows": gold.partial_label_rows,
        "fold_size_difference_over_one": int(max(fold_sizes) - min(fold_sizes) > 1),
        "fold_target_class_coverage_violations": coverage_violations,
        "unassigned_gold_rows": int(np.sum(assignments < 0)),
    }
    return {
        "status": "pass" if all(value == 0 for value in checks.values()) else "fail",
        "generated_at": datetime.now(UTC).isoformat(),
        "privacy": (
            "Aggregate fold statistics only; study identifiers and row-level fold "
            "assignments remain in the external fold manifest. Report text is never "
            "loaded into an output artifact."
        ),
        "configuration": {
            "fold_count": fold_count,
            "seed": seed,
            "assignment_method": (
                "deterministic_multistart_iterative_multilabel_stratification"
            ),
            **assignment_metrics,
        },
        "leakage_policy": {
            "grouping_unit": ID_COLUMN,
            "same_study_confined_to_one_fold": True,
            "limitation": (
                "No stable patient identifier is available in the competition schema; "
                "the split cannot prove that separate study UIDs never represent the "
                "same patient. Duplicate-image/report audits remain a follow-up risk check."
            ),
        },
        "counts": {
            "train_rows": gold.total_rows,
            "fully_gold_labeled_studies": len(gold.identifiers),
            "unlabeled_studies": gold.unlabeled_rows,
            "partial_label_studies": gold.partial_label_rows,
        },
        "fold_sizes": {str(fold): size for fold, size in enumerate(fold_sizes)},
        "targets": target_summary,
        "fold_manifest_sha256": fold_manifest_sha256,
        "checks": checks,
    }


def run_gold_folds(
    competition_root: Path,
    project_root: Path,
    output_root: Path,
    report_output: Path,
    *,
    fold_count: int = 5,
    seed: int = 20260828,
    restarts: int = 512,
) -> dict[str, Any]:
    competition_root = competition_root.expanduser().resolve()
    project_root = project_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    report_output = report_output.expanduser().resolve()
    _validate_roots(competition_root, project_root, output_root, report_output)
    gold = load_gold_labels(competition_root / "train.csv")
    assignments, metrics = assign_multilabel_folds(
        gold.identifiers,
        gold.labels,
        fold_count=fold_count,
        seed=seed,
        restarts=restarts,
    )
    folds_path = output_root / "gold_folds.csv"
    fingerprint = _write_folds_atomic(folds_path, gold.identifiers, assignments)
    report = build_fold_summary(
        gold,
        assignments,
        fold_count,
        seed,
        metrics,
        fingerprint,
    )
    _write_json_atomic(report_output, report)
    if report["status"] != "pass":
        raise GoldFoldError("Generated fold report did not pass its checks")
    return report


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data"))
    project_root = Path(
        os.environ.get("RSNA_PROJECT_ROOT", "/home/shaurya/rsna-project")
    )
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic study-level multilabel folds for fully gold-labeled "
            "studies, storing identifiers externally and aggregate statistics in Git."
        )
    )
    parser.add_argument(
        "--competition-root", type=Path, default=data_root / "raw" / "competition"
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=data_root / "manifests" / "gold_folds_v1",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=project_root / "reports" / "validation_folds" / "summary.json",
    )
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--restarts", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_gold_folds(
            args.competition_root,
            args.project_root,
            args.output_root,
            args.report_output,
            fold_count=args.fold_count,
            seed=args.seed,
            restarts=args.restarts,
        )
        print(
            f"Gold folds {report['status']}: "
            f"{report['counts']['fully_gold_labeled_studies']} studies across "
            f"{report['configuration']['fold_count']} class-complete folds",
            flush=True,
        )
        return 0
    except (GoldFoldError, OSError, ValueError) as error:
        print(f"ERROR: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
