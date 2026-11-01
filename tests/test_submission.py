from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from rsna.submission import (
    ID_COLUMN,
    SUBMISSION_COLUMNS,
    TARGET_COLUMNS,
    SubmissionValidationError,
    generate_constant_submission,
    generate_submission,
    validate_submission,
)
from rsna.submission_smoke import main


def _write_csv(
    path: Path, header: list[str] | tuple[str, ...], rows: list[list[str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def competition_csvs(tmp_path: Path) -> Path:
    competition_root = tmp_path / "raw" / "competition"
    test_ids = ["test-study-1", "test-study-2", "test-study-3"]
    _write_csv(
        competition_root / "test.csv",
        [ID_COLUMN],
        [[identifier] for identifier in test_ids],
    )
    _write_csv(
        competition_root / "sample_submission.csv",
        SUBMISSION_COLUMNS,
        [[identifier, *(["0.5"] * len(TARGET_COLUMNS))] for identifier in test_ids],
    )
    train_rows = []
    for index in range(4):
        label = str(index % 2)
        train_rows.append(
            [f"train-study-{index}", "private report", *([label] * len(TARGET_COLUMNS))]
        )
    train_rows.append(
        ["unlabeled-study", "private report", *([""] * len(TARGET_COLUMNS))]
    )
    _write_csv(
        competition_root / "train.csv",
        [ID_COLUMN, "Report", *TARGET_COLUMNS],
        train_rows,
    )
    return competition_root


def test_constant_submission_generation_and_validation(
    competition_csvs: Path, tmp_path: Path
) -> None:
    output = tmp_path / "submission.csv"

    result = generate_constant_submission(
        competition_csvs / "test.csv",
        competition_csvs / "sample_submission.csv",
        output,
    )

    assert result.rows == 3
    assert result.targets == 12
    assert result.minimum_probability == 0.5
    assert result.maximum_probability == 0.5
    assert (
        validate_submission(
            output,
            competition_csvs / "test.csv",
            competition_csvs / "sample_submission.csv",
        )
        == result
    )


def test_submission_validator_rejects_wrong_identifier_order(
    competition_csvs: Path, tmp_path: Path
) -> None:
    output = tmp_path / "submission.csv"
    rows = [
        [identifier, *(["0.5"] * len(TARGET_COLUMNS))]
        for identifier in ("test-study-2", "test-study-1", "test-study-3")
    ]
    _write_csv(output, SUBMISSION_COLUMNS, rows)

    with pytest.raises(SubmissionValidationError, match="sample_submission.csv order"):
        validate_submission(
            output,
            competition_csvs / "test.csv",
            competition_csvs / "sample_submission.csv",
        )


def test_generator_uses_sample_submission_as_authoritative_order(
    competition_csvs: Path, tmp_path: Path
) -> None:
    sample_path = competition_csvs / "sample_submission.csv"
    sample_order = ("test-study-3", "test-study-1", "test-study-2")
    _write_csv(
        sample_path,
        SUBMISSION_COLUMNS,
        [[identifier, *(["0.5"] * len(TARGET_COLUMNS))] for identifier in sample_order],
    )
    output = tmp_path / "submission.csv"

    generate_constant_submission(
        competition_csvs / "test.csv",
        sample_path,
        output,
    )

    with output.open("r", encoding="utf-8", newline="") as file:
        identifiers = [row[0] for row in list(csv.reader(file))[1:]]
    assert identifiers == list(sample_order)


def test_prediction_submission_validates_shape_and_values(
    competition_csvs: Path, tmp_path: Path
) -> None:
    identifiers = ["test-study-1", "test-study-2", "test-study-3"]
    probabilities = np.linspace(
        0.1,
        0.9,
        len(identifiers) * len(TARGET_COLUMNS),
    ).reshape(len(identifiers), len(TARGET_COLUMNS))

    result = generate_submission(
        competition_csvs / "test.csv",
        competition_csvs / "sample_submission.csv",
        tmp_path / "prediction-submission.csv",
        identifiers,
        probabilities,
    )

    assert result.rows == 3
    assert result.minimum_probability == pytest.approx(0.1)
    assert result.maximum_probability == pytest.approx(0.9)


def test_prediction_submission_rejects_identifier_order(
    competition_csvs: Path, tmp_path: Path
) -> None:
    with pytest.raises(SubmissionValidationError, match="test.csv order"):
        generate_submission(
            competition_csvs / "test.csv",
            competition_csvs / "sample_submission.csv",
            tmp_path / "submission.csv",
            ["test-study-2", "test-study-1", "test-study-3"],
            np.full((3, len(TARGET_COLUMNS)), 0.5),
        )


def test_prediction_submission_maps_test_order_into_sample_order(
    competition_csvs: Path, tmp_path: Path
) -> None:
    sample_path = competition_csvs / "sample_submission.csv"
    sample_order = ["test-study-3", "test-study-1", "test-study-2"]
    _write_csv(
        sample_path,
        SUBMISSION_COLUMNS,
        [[identifier, *(["0.5"] * len(TARGET_COLUMNS))] for identifier in sample_order],
    )
    probabilities = np.repeat(
        np.asarray([[0.1], [0.2], [0.3]]), len(TARGET_COLUMNS), axis=1
    )
    output = tmp_path / "submission.csv"

    generate_submission(
        competition_csvs / "test.csv",
        sample_path,
        output,
        ["test-study-1", "test-study-2", "test-study-3"],
        probabilities,
    )

    with output.open(encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))[1:]
    assert [row[0] for row in rows] == sample_order
    assert [float(row[1]) for row in rows] == pytest.approx([0.3, 0.1, 0.2])


def test_submission_validator_rejects_nonfinite_and_out_of_range_values(
    competition_csvs: Path, tmp_path: Path
) -> None:
    for invalid_value, message in (("nan", "not finite"), ("1.01", "outside")):
        output = tmp_path / f"submission-{invalid_value}.csv"
        rows = [
            [identifier, invalid_value, *(["0.5"] * (len(TARGET_COLUMNS) - 1))]
            for identifier in ("test-study-1", "test-study-2", "test-study-3")
        ]
        _write_csv(output, SUBMISSION_COLUMNS, rows)
        with pytest.raises(SubmissionValidationError, match=message):
            validate_submission(
                output,
                competition_csvs / "test.csv",
                competition_csvs / "sample_submission.csv",
            )


def test_end_to_end_submission_smoke_test(
    competition_csvs: Path, tmp_path: Path
) -> None:
    output = tmp_path / "project" / "submission.csv"
    report = tmp_path / "project" / "reports" / "smoke.json"

    assert (
        main(
            [
                "--competition-root",
                str(competition_csvs),
                "--output",
                str(output),
                "--report-output",
                str(report),
            ]
        )
        == 0
    )

    summary = json.loads(report.read_text(encoding="utf-8"))
    assert summary["status"] == "pass"
    assert summary["offline"] is True
    assert summary["metric"]["gold_studies"] == 4
    assert summary["metric"]["constant_prediction_sanity_check"]["macro_auc"] == 0.5
    assert summary["submission"]["validation"]["rows"] == 3
    assert summary["submission"]["validation"]["targets"] == 12
    serialized = json.dumps(summary)
    assert "test-study" not in serialized
    assert "private report" not in serialized
