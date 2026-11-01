from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rsna.gold_folds import GoldFoldError, load_gold_labels, run_gold_folds
from rsna.submission import ID_COLUMN, TARGET_COLUMNS


def _write_train_csv(path: Path, *, partial: bool = False) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    identifiers = [f"secret-study-{index:03d}" for index in range(40)]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow([ID_COLUMN, "Report", *TARGET_COLUMNS])
        for index, identifier in enumerate(identifiers):
            labels = [
                int((index + 2 * target_index) % (4 + target_index % 3) == 0)
                for target_index in range(len(TARGET_COLUMNS))
            ]
            if partial and index == 0:
                labels[0] = ""
            writer.writerow([identifier, f"secret report text {index}", *labels])
        for index in range(5):
            writer.writerow(
                [
                    f"unlabeled-secret-{index}",
                    f"unlabeled secret report {index}",
                    *([""] * len(TARGET_COLUMNS)),
                ]
            )
    return identifiers


def test_gold_folds_are_deterministic_class_complete_and_identifier_free(
    tmp_path: Path,
) -> None:
    competition_root = tmp_path / "raw" / "competition"
    identifiers = _write_train_csv(competition_root / "train.csv")
    project_root = tmp_path / "project"
    project_root.mkdir()
    output_root_one = tmp_path / "manifests" / "folds-one"
    output_root_two = tmp_path / "manifests" / "folds-two"
    report_one = project_root / "reports" / "folds-one.json"
    report_two = project_root / "reports" / "folds-two.json"

    first = run_gold_folds(
        competition_root,
        project_root,
        output_root_one,
        report_one,
        fold_count=5,
        restarts=128,
    )
    second = run_gold_folds(
        competition_root,
        project_root,
        output_root_two,
        report_two,
        fold_count=5,
        restarts=128,
    )

    assert first["status"] == "pass"
    assert first["counts"] == {
        "train_rows": 45,
        "fully_gold_labeled_studies": 40,
        "unlabeled_studies": 5,
        "partial_label_studies": 0,
    }
    assert first["fold_sizes"] == {str(fold): 8 for fold in range(5)}
    assert all(value == 0 for value in first["checks"].values())
    for target in TARGET_COLUMNS:
        assert all(
            fold["positive"] > 0 and fold["negative"] > 0
            for fold in first["targets"][target]["folds"]
        )
    folds_one = (output_root_one / "gold_folds.csv").read_text(encoding="utf-8")
    folds_two = (output_root_two / "gold_folds.csv").read_text(encoding="utf-8")
    assert folds_one == folds_two
    assert all(identifier in folds_one for identifier in identifiers)
    serialized = report_one.read_text(encoding="utf-8")
    assert all(identifier not in serialized for identifier in identifiers)
    assert "secret report text" not in serialized
    assert (
        json.loads(serialized)["fold_manifest_sha256"] == first["fold_manifest_sha256"]
    )
    assert first["fold_manifest_sha256"] == second["fold_manifest_sha256"]


def test_partially_labeled_rows_are_refused(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    _write_train_csv(train_csv, partial=True)

    with pytest.raises(GoldFoldError, match="partially labeled"):
        load_gold_labels(train_csv)
