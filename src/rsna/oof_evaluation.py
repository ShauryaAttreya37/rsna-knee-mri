"""Strict pooled out-of-fold evaluation for imaging checkpoints."""

from __future__ import annotations

import csv
import json
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rsna.evaluation import macro_roc_auc
from rsna.imaging_training import ImagingTrainingError, InferenceResult
from rsna.submission import ID_COLUMN, TARGET_COLUMNS


@dataclass(frozen=True, slots=True)
class PooledOofResult:
    """Validated pooled predictions from one held-out pass per gold study."""

    study_uids: tuple[str, ...]
    folds: tuple[int, ...]
    labels: np.ndarray
    probabilities: np.ndarray
    fold_losses: dict[int, float]

    def aggregate_summary(self) -> dict[str, Any]:
        """Return identifier-free discrimination and calibration metrics."""
        auc = macro_roc_auc(self.labels, self.probabilities, TARGET_COLUMNS)
        fold_metrics: dict[str, dict[str, Any]] = {}
        fold_auc_values: list[float] = []
        fold_sizes: list[int] = []
        fold_array = np.asarray(self.folds)
        for fold in sorted(self.fold_losses):
            mask = fold_array == fold
            fold_auc = macro_roc_auc(
                self.labels[mask], self.probabilities[mask], TARGET_COLUMNS
            )
            fold_auc_values.append(fold_auc.macro_auc)
            fold_sizes.append(int(np.sum(mask)))
            fold_metrics[str(fold)] = {
                "studies": fold_sizes[-1],
                "loss": self.fold_losses[fold],
                "macro_auc": fold_auc.macro_auc,
                "per_target_auc": fold_auc.per_target_auc,
            }
        errors = self.probabilities - self.labels
        epsilon = np.finfo(np.float64).eps
        clipped = np.clip(self.probabilities, epsilon, 1.0 - epsilon)
        log_losses = -(
            self.labels * np.log(clipped) + (1.0 - self.labels) * np.log(1.0 - clipped)
        )
        brier_by_target = np.mean(errors**2, axis=0)
        log_loss_by_target = np.mean(log_losses, axis=0)
        observed_prevalence = np.mean(self.labels, axis=0)
        predicted_prevalence = np.mean(self.probabilities, axis=0)
        return {
            "status": "complete",
            "generated_at": datetime.now(UTC).isoformat(),
            "privacy": "Identifier-free aggregate OOF metrics only; OOF rows remain external.",
            "studies": len(self.study_uids),
            "folds": fold_metrics,
            "fold_aggregate": {
                "macro_auc_mean": float(np.mean(fold_auc_values)),
                "macro_auc_weighted_mean": float(
                    np.average(fold_auc_values, weights=fold_sizes)
                ),
                "macro_auc_sample_standard_deviation": float(
                    np.std(fold_auc_values, ddof=1)
                ),
            },
            "macro_auc": auc.macro_auc,
            "per_target_auc": auc.per_target_auc,
            "positives": auc.positives,
            "negatives": auc.negatives,
            "brier_score": float(np.mean(errors**2)),
            "per_target_brier_score": {
                target: float(brier_by_target[index])
                for index, target in enumerate(TARGET_COLUMNS)
            },
            "log_loss": float(np.mean(log_losses)),
            "per_target_log_loss": {
                target: float(log_loss_by_target[index])
                for index, target in enumerate(TARGET_COLUMNS)
            },
            "observed_prevalence": {
                target: float(observed_prevalence[index])
                for index, target in enumerate(TARGET_COLUMNS)
            },
            "predicted_prevalence": {
                target: float(predicted_prevalence[index])
                for index, target in enumerate(TARGET_COLUMNS)
            },
        }


def combine_oof_predictions(
    fold_predictions: Mapping[int, InferenceResult],
    *,
    expected_folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    expected_studies: int = 58,
) -> PooledOofResult:
    """Validate and combine exactly one held-out prediction per gold study."""
    if tuple(sorted(fold_predictions)) != tuple(sorted(expected_folds)):
        raise ImagingTrainingError("OOF fold set does not match expected folds")
    study_uids: list[str] = []
    folds: list[int] = []
    label_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    fold_losses: dict[int, float] = {}
    target_count = len(TARGET_COLUMNS)
    for fold in sorted(fold_predictions):
        result = fold_predictions[fold]
        count = len(result.study_uids)
        expected_shape = (count, target_count)
        if count == 0:
            raise ImagingTrainingError(f"OOF fold {fold} is empty")
        if result.labels.shape != expected_shape:
            raise ImagingTrainingError(f"OOF fold {fold} label shape is invalid")
        if result.probabilities.shape != expected_shape:
            raise ImagingTrainingError(f"OOF fold {fold} probability shape is invalid")
        if len(set(result.study_uids)) != count:
            raise ImagingTrainingError(f"OOF fold {fold} contains duplicate studies")
        if not np.all(np.isin(result.labels, (0.0, 1.0))):
            raise ImagingTrainingError("OOF labels must contain only binary 0/1 values")
        if not np.all(np.isfinite(result.probabilities)) or not np.all(
            (result.probabilities >= 0.0) & (result.probabilities <= 1.0)
        ):
            raise ImagingTrainingError("OOF probabilities must be finite and in [0, 1]")
        if not math.isfinite(result.mean_loss) or result.mean_loss < 0.0:
            raise ImagingTrainingError(f"OOF fold {fold} loss is invalid")
        study_uids.extend(result.study_uids)
        folds.extend([fold] * count)
        label_parts.append(result.labels.astype(np.float64, copy=False))
        probability_parts.append(result.probabilities.astype(np.float64, copy=False))
        fold_losses[fold] = result.mean_loss
    if len(study_uids) != expected_studies:
        raise ImagingTrainingError(
            f"Expected {expected_studies} OOF studies, observed {len(study_uids)}"
        )
    if len(set(study_uids)) != len(study_uids):
        raise ImagingTrainingError("A gold study appears in more than one OOF fold")
    return PooledOofResult(
        study_uids=tuple(study_uids),
        folds=tuple(folds),
        labels=np.concatenate(label_parts, axis=0),
        probabilities=np.concatenate(probability_parts, axis=0),
        fold_losses=fold_losses,
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True, allow_nan=False)
            file.write("\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_oof_outputs(
    result: PooledOofResult,
    *,
    predictions_csv: Path,
    aggregate_report_json: Path,
) -> dict[str, Any]:
    """Write private OOF rows separately from identifier-free aggregate metrics."""
    predictions_csv = predictions_csv.expanduser().resolve()
    aggregate_report_json = aggregate_report_json.expanduser().resolve()
    if predictions_csv == aggregate_report_json:
        raise ImagingTrainingError("OOF predictions and report paths must differ")
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = predictions_csv.with_name(
        f".{predictions_csv.name}.{uuid.uuid4().hex}.tmp"
    )
    fields = [ID_COLUMN, "fold"]
    fields.extend(f"{target}_label" for target in TARGET_COLUMNS)
    fields.extend(f"{target}_prob" for target in TARGET_COLUMNS)
    try:
        with temporary.open("x", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(fields)
            for index, study_uid in enumerate(result.study_uids):
                writer.writerow(
                    [
                        study_uid,
                        result.folds[index],
                        *[int(value) for value in result.labels[index]],
                        *[repr(float(value)) for value in result.probabilities[index]],
                    ]
                )
        temporary.replace(predictions_csv)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    summary = result.aggregate_summary()
    _atomic_json(aggregate_report_json, summary)
    return summary
