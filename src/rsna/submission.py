"""Strict generation and validation of RSNA competition submissions."""

from __future__ import annotations

import csv
import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ID_COLUMN = "StudyInstanceUID"
TARGET_COLUMNS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
SUBMISSION_COLUMNS = (ID_COLUMN, *TARGET_COLUMNS)


class SubmissionValidationError(ValueError):
    """Raised when a test template or generated submission is invalid."""


@dataclass(frozen=True, slots=True)
class SubmissionValidationResult:
    rows: int
    targets: int
    minimum_probability: float
    maximum_probability: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "targets": self.targets,
            "minimum_probability": self.minimum_probability,
            "maximum_probability": self.maximum_probability,
            "identifiers_unique": True,
            "identifiers_match_test_set": True,
            "identifiers_in_sample_order": True,
            "columns_exact": True,
            "probabilities_finite": True,
            "probabilities_in_range": True,
        }


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            header = next(reader)
            rows = list(reader)
    except StopIteration as error:
        raise SubmissionValidationError(f"CSV is empty: {path}") from error
    except (OSError, UnicodeError, csv.Error) as error:
        raise SubmissionValidationError(
            f"Could not read CSV {path}: {error}"
        ) from error
    if not header or any(not column for column in header):
        raise SubmissionValidationError(f"CSV has an invalid header: {path}")
    for row_number, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise SubmissionValidationError(
                f"CSV row {row_number} has {len(row)} fields; expected {len(header)}"
            )
    return header, rows


def _validate_identifiers(identifiers: list[str], context: str) -> None:
    if any(not identifier.strip() for identifier in identifiers):
        raise SubmissionValidationError(f"{context} contains a blank study identifier")
    if len(set(identifiers)) != len(identifiers):
        raise SubmissionValidationError(
            f"{context} contains duplicate study identifiers"
        )


def _load_test_identifiers(test_csv: Path) -> list[str]:
    header, rows = _read_csv(test_csv)
    if header != [ID_COLUMN]:
        raise SubmissionValidationError(
            f"test.csv columns must be exactly {[ID_COLUMN]!r}; observed {header!r}"
        )
    identifiers = [row[0] for row in rows]
    _validate_identifiers(identifiers, "test.csv")
    if not identifiers:
        raise SubmissionValidationError("test.csv contains no studies")
    return identifiers


def _load_sample_identifiers(sample_submission_csv: Path) -> list[str]:
    header, rows = _read_csv(sample_submission_csv)
    if tuple(header) != SUBMISSION_COLUMNS:
        raise SubmissionValidationError(
            "sample_submission.csv columns do not match the official schema: "
            f"{header!r}"
        )
    identifiers = [row[0] for row in rows]
    _validate_identifiers(identifiers, "sample_submission.csv")
    return identifiers


def _validate_template_identifiers(
    test_identifiers: list[str], sample_identifiers: list[str]
) -> None:
    if set(sample_identifiers) != set(test_identifiers):
        raise SubmissionValidationError(
            "sample_submission.csv identifiers do not exactly match the test.csv set"
        )


def validate_submission(
    submission_csv: Path,
    test_csv: Path,
    sample_submission_csv: Path,
) -> SubmissionValidationResult:
    """Validate schema, IDs, order, and every probability value."""
    test_ids = _load_test_identifiers(test_csv)
    sample_ids = _load_sample_identifiers(sample_submission_csv)
    _validate_template_identifiers(test_ids, sample_ids)

    header, rows = _read_csv(submission_csv)
    if tuple(header) != SUBMISSION_COLUMNS:
        raise SubmissionValidationError(
            f"Submission columns must be exactly {list(SUBMISSION_COLUMNS)!r}"
        )
    observed_ids = [row[0] for row in rows]
    _validate_identifiers(observed_ids, "submission")
    if observed_ids != sample_ids:
        raise SubmissionValidationError(
            "Submission identifiers do not exactly match sample_submission.csv order"
        )

    probabilities: list[float] = []
    for row_number, row in enumerate(rows, start=2):
        for column, value in zip(TARGET_COLUMNS, row[1:], strict=True):
            try:
                probability = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise SubmissionValidationError(
                    f"Row {row_number}, column {column!r} is not numeric"
                ) from error
            if not math.isfinite(probability):
                raise SubmissionValidationError(
                    f"Row {row_number}, column {column!r} is not finite"
                )
            if not 0.0 <= probability <= 1.0:
                raise SubmissionValidationError(
                    f"Row {row_number}, column {column!r} is outside [0, 1]"
                )
            probabilities.append(probability)
    if not probabilities:
        raise SubmissionValidationError("Submission contains no probabilities")
    return SubmissionValidationResult(
        rows=len(rows),
        targets=len(TARGET_COLUMNS),
        minimum_probability=min(probabilities),
        maximum_probability=max(probabilities),
    )


def generate_constant_submission(
    test_csv: Path,
    sample_submission_csv: Path,
    output_csv: Path,
    probability: float = 0.5,
) -> SubmissionValidationResult:
    """Generate a constant baseline and validate the completed file."""
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise SubmissionValidationError(
            "Constant probability must be finite and inside [0, 1]"
        )
    identifiers = _load_test_identifiers(test_csv.resolve())
    probability_matrix = [
        [float(probability)] * len(TARGET_COLUMNS) for _ in identifiers
    ]
    return generate_submission(
        test_csv,
        sample_submission_csv,
        output_csv,
        identifiers,
        probability_matrix,
    )


def generate_submission(
    test_csv: Path,
    sample_submission_csv: Path,
    output_csv: Path,
    identifiers: Sequence[str],
    probabilities: Any,
) -> SubmissionValidationResult:
    """Generate a prediction submission in authoritative sample order."""
    test_csv = test_csv.resolve()
    sample_submission_csv = sample_submission_csv.resolve()
    output_csv = output_csv.resolve()
    if output_csv in {test_csv, sample_submission_csv}:
        raise SubmissionValidationError(
            "Submission output must not overwrite an input CSV"
        )
    test_ids = _load_test_identifiers(test_csv)
    sample_ids = _load_sample_identifiers(sample_submission_csv)
    _validate_template_identifiers(test_ids, sample_ids)
    prediction_ids = list(identifiers)
    _validate_identifiers(prediction_ids, "predictions")
    if prediction_ids != test_ids:
        raise SubmissionValidationError(
            "Prediction identifiers do not exactly match test.csv order"
        )
    try:
        matrix = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise SubmissionValidationError("Predictions must be numeric") from error
    expected_shape = (len(test_ids), len(TARGET_COLUMNS))
    if matrix.shape != expected_shape:
        raise SubmissionValidationError(
            f"Prediction matrix shape must be {expected_shape}; observed {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)) or not np.all((matrix >= 0.0) & (matrix <= 1.0)):
        raise SubmissionValidationError(
            "Predictions must be finite probabilities in [0, 1]"
        )
    by_identifier = dict(zip(prediction_ids, matrix, strict=True))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name(f".{output_csv.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(SUBMISSION_COLUMNS)
            for identifier in sample_ids:
                writer.writerow(
                    [
                        identifier,
                        *[repr(float(value)) for value in by_identifier[identifier]],
                    ]
                )
        temporary.replace(output_csv)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise SubmissionValidationError(
            f"Could not write submission {output_csv}: {error}"
        ) from error
    return validate_submission(output_csv, test_csv, sample_submission_csv)
