from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from rsna.imaging_training import ImagingTrainingError, InferenceResult
from rsna.oof_evaluation import combine_oof_predictions, write_oof_outputs
from rsna.submission import ID_COLUMN, TARGET_COLUMNS


def _fold_predictions() -> dict[int, InferenceResult]:
    results: dict[int, InferenceResult] = {}
    for fold in range(5):
        labels = np.asarray(
            [
                [(fold + target) % 2 for target in range(len(TARGET_COLUMNS))],
                [1 - ((fold + target) % 2) for target in range(len(TARGET_COLUMNS))],
            ],
            dtype=np.float64,
        )
        probabilities = np.where(labels == 1.0, 0.9, 0.1)
        results[fold] = InferenceResult(
            study_uids=(f"study-{fold}-a", f"study-{fold}-b"),
            labels=labels,
            probabilities=probabilities,
            mean_loss=0.1 + fold / 100,
        )
    return results


def test_combines_all_folds_and_reports_pooled_metrics(tmp_path) -> None:
    result = combine_oof_predictions(_fold_predictions(), expected_studies=10)
    predictions_path = tmp_path / "private" / "oof.csv"
    report_path = tmp_path / "reports" / "oof_summary.json"

    summary = write_oof_outputs(
        result,
        predictions_csv=predictions_path,
        aggregate_report_json=report_path,
    )

    assert summary["studies"] == 10
    assert summary["macro_auc"] == pytest.approx(1.0)
    assert summary["fold_aggregate"]["macro_auc_mean"] == pytest.approx(1.0)
    assert summary["folds"]["0"]["macro_auc"] == pytest.approx(1.0)
    assert summary["brier_score"] == pytest.approx(0.01)
    assert summary["log_loss"] == pytest.approx(-np.log(0.9))
    assert ID_COLUMN not in json.dumps(summary)
    with predictions_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    assert len(rows) == 11
    assert rows[0][:2] == [ID_COLUMN, "fold"]
    assert report_path.is_file()


def test_rejects_study_present_in_two_folds() -> None:
    predictions = _fold_predictions()
    fold_one = predictions[1]
    predictions[1] = InferenceResult(
        study_uids=(predictions[0].study_uids[0], fold_one.study_uids[1]),
        labels=fold_one.labels,
        probabilities=fold_one.probabilities,
        mean_loss=fold_one.mean_loss,
    )

    with pytest.raises(ImagingTrainingError, match="more than one OOF fold"):
        combine_oof_predictions(predictions, expected_studies=10)


def test_rejects_missing_fold() -> None:
    predictions = _fold_predictions()
    del predictions[4]

    with pytest.raises(ImagingTrainingError, match="fold set"):
        combine_oof_predictions(predictions, expected_studies=8)
