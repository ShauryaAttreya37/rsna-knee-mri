"""Offline official-metric and constant-submission smoke test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rsna.evaluation import MetricValidationError, macro_roc_auc
from rsna.submission import (
    ID_COLUMN,
    TARGET_COLUMNS,
    SubmissionValidationError,
    generate_constant_submission,
)


class SmokeTestError(RuntimeError):
    """Raised when the offline smoke test cannot complete safely."""


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _load_gold_labels(train_csv: Path) -> np.ndarray:
    expected_columns = [ID_COLUMN, "Report", *TARGET_COLUMNS]
    labels: list[list[float]] = []
    seen_identifiers: set[str] = set()
    try:
        with train_csv.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != expected_columns:
                raise SmokeTestError(
                    "train.csv columns do not match the audited competition schema"
                )
            for row_number, row in enumerate(reader, start=2):
                identifier = (row.get(ID_COLUMN) or "").strip()
                if not identifier:
                    raise SmokeTestError(
                        f"train.csv row {row_number} has a blank study identifier"
                    )
                if identifier in seen_identifiers:
                    raise SmokeTestError(
                        f"train.csv row {row_number} repeats a study identifier"
                    )
                seen_identifiers.add(identifier)
                raw_labels = [
                    (row.get(column) or "").strip() for column in TARGET_COLUMNS
                ]
                present = [bool(value) for value in raw_labels]
                if not any(present):
                    continue
                if not all(present):
                    raise SmokeTestError(
                        f"train.csv row {row_number} is only partially labeled"
                    )
                parsed: list[float] = []
                for column, value in zip(TARGET_COLUMNS, raw_labels, strict=True):
                    try:
                        label = float(value)
                    except (TypeError, ValueError, OverflowError) as error:
                        raise SmokeTestError(
                            f"train.csv row {row_number}, column {column!r} is invalid"
                        ) from error
                    if not math.isfinite(label) or label not in (0.0, 1.0):
                        raise SmokeTestError(
                            f"train.csv row {row_number}, column {column!r} is not 0/1"
                        )
                    parsed.append(label)
                labels.append(parsed)
    except (OSError, UnicodeError, csv.Error) as error:
        raise SmokeTestError(f"Could not read train.csv: {error}") from error
    if not labels:
        raise SmokeTestError("train.csv contains no fully labeled studies")
    return np.asarray(labels, dtype=float)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, sort_keys=True)
            file.write("\n")
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def run_smoke_test(
    competition_root: Path,
    output_csv: Path,
    report_output: Path,
    probability: float = 0.5,
) -> dict[str, Any]:
    competition_root = competition_root.expanduser().resolve()
    output_csv = output_csv.expanduser().resolve()
    report_output = report_output.expanduser().resolve()
    if not competition_root.is_dir():
        raise SmokeTestError(f"Competition root does not exist: {competition_root}")
    for output_path in (output_csv, report_output):
        if _within(output_path, competition_root):
            raise SmokeTestError(
                "Generated outputs must remain outside the raw competition tree"
            )
    if output_csv == report_output:
        raise SmokeTestError("Submission and report outputs must differ")

    train_csv = competition_root / "train.csv"
    test_csv = competition_root / "test.csv"
    sample_submission_csv = competition_root / "sample_submission.csv"
    for path in (train_csv, test_csv, sample_submission_csv):
        if not path.is_file():
            raise SmokeTestError(f"Required competition CSV is missing: {path}")

    gold_labels = _load_gold_labels(train_csv)
    constant_scores = np.full(gold_labels.shape, probability, dtype=float)
    metric = macro_roc_auc(gold_labels, constant_scores, TARGET_COLUMNS)
    if not math.isclose(metric.macro_auc, 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise SmokeTestError(
            f"Constant-prediction metric sanity check returned {metric.macro_auc}"
        )
    validation = generate_constant_submission(
        test_csv,
        sample_submission_csv,
        output_csv,
        probability,
    )
    summary = {
        "status": "pass",
        "generated_at": datetime.now(UTC).isoformat(),
        "offline": True,
        "privacy": (
            "Aggregate counts, target names, and metrics only; no study identifiers "
            "or report text are included."
        ),
        "metric": {
            "name": "macro_roc_auc",
            "definition": (
                "Unweighted mean of the twelve per-target binary ROC AUC values."
            ),
            "constant_prediction_sanity_check": metric.to_dict(),
            "gold_studies": int(gold_labels.shape[0]),
        },
        "submission": {
            "filename": output_csv.name,
            "sha256": _sha256(output_csv),
            "constant_probability": probability,
            "validation": validation.to_dict(),
        },
    }
    _write_json_atomic(report_output, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data"))
    project_root = Path(
        os.environ.get("RSNA_PROJECT_ROOT", "/home/shaurya/rsna-project")
    )
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the official macro ROC AUC sanity check and generate a "
            "strictly validated constant-probability submission."
        )
    )
    parser.add_argument(
        "--competition-root",
        type=Path,
        default=data_root / "raw" / "competition",
    )
    parser.add_argument("--output", type=Path, default=project_root / "submission.csv")
    parser.add_argument(
        "--report-output",
        type=Path,
        default=project_root / "reports" / "submission_smoke_test" / "summary.json",
    )
    parser.add_argument("--probability", type=float, default=0.5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_smoke_test(
            args.competition_root,
            args.output,
            args.report_output,
            args.probability,
        )
        print(
            f"Submission smoke test {summary['status']}: "
            f"metric={summary['metric']['constant_prediction_sanity_check']['macro_auc']:.3f}, "
            f"rows={summary['submission']['validation']['rows']}, "
            f"output={args.output.expanduser().resolve()}",
            flush=True,
        )
        return 0
    except (
        OSError,
        MetricValidationError,
        SmokeTestError,
        SubmissionValidationError,
    ) as error:
        print(f"ERROR: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
