"""Competition evaluation metrics implemented without scikit-learn."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


class MetricValidationError(ValueError):
    """Raised when labels or predictions cannot be scored safely."""


@dataclass(frozen=True, slots=True)
class MacroAucResult:
    """Macro ROC AUC and the auditable per-target components."""

    macro_auc: float
    per_target_auc: dict[str, float]
    positives: dict[str, int]
    negatives: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_auc": self.macro_auc,
            "per_target_auc": self.per_target_auc,
            "positives": self.positives,
            "negatives": self.negatives,
        }


def binary_roc_auc(labels: Any, scores: Any) -> float:
    """Compute binary ROC AUC with average ranks for tied scores.

    This is the Mann-Whitney interpretation of ROC AUC and matches the usual
    competition behavior: every positive/negative pair contributes one for a
    correctly ranked pair, one half for a tie, and zero otherwise.
    """
    try:
        label_array = np.asarray(labels, dtype=float)
        score_array = np.asarray(scores, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise MetricValidationError("Labels and scores must be numeric") from error
    if label_array.ndim != 1 or score_array.ndim != 1:
        raise MetricValidationError("Binary ROC AUC expects one-dimensional arrays")
    if label_array.shape != score_array.shape:
        raise MetricValidationError("Labels and scores must have identical shapes")
    if label_array.size == 0:
        raise MetricValidationError("ROC AUC cannot score an empty target")
    if not np.all(np.isfinite(label_array)):
        raise MetricValidationError("Labels must be finite")
    if not np.all(np.isfinite(score_array)):
        raise MetricValidationError("Scores must be finite")
    if not np.all(np.isin(label_array, (0.0, 1.0))):
        raise MetricValidationError("Labels must contain only binary 0/1 values")

    positive_count = int(np.sum(label_array == 1.0))
    negative_count = int(label_array.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise MetricValidationError(
            "ROC AUC is undefined when a target contains only one class"
        )

    order = np.argsort(score_array, kind="stable")
    sorted_scores = score_array[order]
    sorted_ranks = np.empty(label_array.size, dtype=float)
    start = 0
    while start < label_array.size:
        stop = start + 1
        while stop < label_array.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        # Ranks are one-based; the tied group occupies [start + 1, stop].
        sorted_ranks[start:stop] = ((start + 1) + stop) / 2.0
        start = stop
    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    positive_rank_sum = float(np.sum(ranks[label_array == 1.0]))
    auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )
    if not math.isfinite(auc) or not 0.0 <= auc <= 1.0:
        raise AssertionError(f"Internal ROC AUC calculation produced {auc}")
    return float(auc)


def macro_roc_auc(
    labels: Any,
    scores: Any,
    target_names: list[str] | tuple[str, ...] | None = None,
) -> MacroAucResult:
    """Compute the unweighted mean of per-target binary ROC AUCs."""
    try:
        label_array = np.asarray(labels, dtype=float)
        score_array = np.asarray(scores, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise MetricValidationError("Labels and scores must be numeric") from error
    if label_array.ndim != 2 or score_array.ndim != 2:
        raise MetricValidationError("Macro ROC AUC expects two-dimensional matrices")
    if label_array.shape != score_array.shape:
        raise MetricValidationError(
            "Label and score matrices must have identical shapes"
        )
    if label_array.shape[0] == 0 or label_array.shape[1] == 0:
        raise MetricValidationError("Macro ROC AUC cannot score an empty matrix")

    if target_names is None:
        names = tuple(f"target_{index}" for index in range(label_array.shape[1]))
    else:
        names = tuple(target_names)
    if len(names) != label_array.shape[1]:
        raise MetricValidationError(
            "Target-name count must match the number of matrix columns"
        )
    if len(set(names)) != len(names) or any(not name for name in names):
        raise MetricValidationError("Target names must be nonempty and unique")

    per_target: dict[str, float] = {}
    positives: dict[str, int] = {}
    negatives: dict[str, int] = {}
    for index, name in enumerate(names):
        target_labels = label_array[:, index]
        try:
            per_target[name] = binary_roc_auc(target_labels, score_array[:, index])
        except MetricValidationError as error:
            raise MetricValidationError(f"Target {name!r}: {error}") from error
        positives[name] = int(np.sum(target_labels == 1.0))
        negatives[name] = int(np.sum(target_labels == 0.0))
    return MacroAucResult(
        macro_auc=float(np.mean(list(per_target.values()))),
        per_target_auc=per_target,
        positives=positives,
        negatives=negatives,
    )
