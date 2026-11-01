from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from rsna.evaluation import MetricValidationError, binary_roc_auc, macro_roc_auc


def test_binary_roc_auc_handles_rankings_and_ties() -> None:
    labels = np.asarray([0, 0, 1, 1])

    assert binary_roc_auc(labels, [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert binary_roc_auc(labels, [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert binary_roc_auc(labels, [0.0, 0.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_macro_roc_auc_is_unweighted_mean_of_targets() -> None:
    labels = np.asarray(
        [
            [0, 0],
            [0, 0],
            [1, 1],
            [1, 1],
        ]
    )
    scores = np.asarray(
        [
            [0.1, 0.5],
            [0.4, 0.5],
            [0.35, 0.5],
            [0.8, 0.5],
        ]
    )

    result = macro_roc_auc(labels, scores, ["finding_a", "finding_b"])

    assert result.per_target_auc == {
        "finding_a": pytest.approx(0.75),
        "finding_b": pytest.approx(0.5),
    }
    assert result.macro_auc == pytest.approx(0.625)
    assert result.positives == {"finding_a": 2, "finding_b": 2}
    assert result.negatives == {"finding_a": 2, "finding_b": 2}


def test_macro_roc_auc_matches_sklearn_on_random_tie_heavy_inputs() -> None:
    random = np.random.default_rng(20260827)
    for rows in (10, 31, 100):
        labels = random.integers(0, 2, size=(rows, 12))
        labels[0, :] = 0
        labels[1, :] = 1
        # Five possible score values deliberately exercise average tie ranks.
        scores = random.integers(0, 5, size=(rows, 12)) / 4.0
        target_names = [f"finding_{index}" for index in range(12)]

        result = macro_roc_auc(labels, scores, target_names)
        expected_macro = roc_auc_score(labels, scores, average="macro")

        assert result.macro_auc == pytest.approx(expected_macro, abs=1e-15)
        for index, target_name in enumerate(target_names):
            expected_target = roc_auc_score(labels[:, index], scores[:, index])
            assert result.per_target_auc[target_name] == pytest.approx(
                expected_target, abs=1e-15
            )


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([0, 0], [0.1, 0.2], "only one class"),
        ([0, 2], [0.1, 0.2], "binary 0/1"),
        ([0, 1], [0.1, np.nan], "finite"),
        ([0, 1], [0.1], "identical shapes"),
    ],
)
def test_binary_roc_auc_rejects_invalid_inputs(
    labels: list[float], scores: list[float], message: str
) -> None:
    with pytest.raises(MetricValidationError, match=message):
        binary_roc_auc(labels, scores)
