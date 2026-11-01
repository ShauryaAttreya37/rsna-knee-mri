"""Offline multi-checkpoint imaging inference for RSNA knee studies."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rsna.dicom_loader import load_dicom_series
from rsna.example_test_pipeline import SeriesDescriptor, load_test_metadata
from rsna.imaging_training import ImagingTrainingError, prepare_study_images
from rsna.submission import TARGET_COLUMNS
from rsna.training_data import PLANE_TO_INDEX


@dataclass(frozen=True, slots=True)
class TestEnsembleResult:
    """Identifier-bearing predictions plus aggregate runtime counters."""

    study_uids: tuple[str, ...]
    model_probabilities: np.ndarray
    probability_mean: np.ndarray
    rank_average: np.ndarray
    decoded_pixel_arrays: int
    series: int
    elapsed_seconds: float

    def aggregate_summary(self) -> dict[str, Any]:
        """Return identifier-free inference statistics."""
        return {
            "studies": len(self.study_uids),
            "series": self.series,
            "models": int(self.model_probabilities.shape[0]),
            "decoded_pixel_arrays": self.decoded_pixel_arrays,
            "elapsed_seconds": self.elapsed_seconds,
            "studies_per_second": len(self.study_uids)
            / max(self.elapsed_seconds, 1e-9),
            "ensemble": "per-target average of within-model test-study percentile ranks",
            "probability_mean_range": [
                float(np.min(self.probability_mean)),
                float(np.max(self.probability_mean)),
            ],
            "rank_average_range": [
                float(np.min(self.rank_average)),
                float(np.max(self.rank_average)),
            ],
        }


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Return average percentile ranks with deterministic tie handling."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return (ranks + 0.5) / len(values)


def rank_average_probabilities(model_probabilities: Any) -> np.ndarray:
    """Normalize each model/target score scale before ensemble averaging."""
    try:
        values = np.asarray(model_probabilities, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ImagingTrainingError("Model probabilities must be numeric") from error
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ImagingTrainingError(
            "Model probabilities must have shape [models, studies, targets]"
        )
    if values.shape[2] != len(TARGET_COLUMNS):
        raise ImagingTrainingError("Model probability target count is invalid")
    if not np.all(np.isfinite(values)) or not np.all((values >= 0.0) & (values <= 1.0)):
        raise ImagingTrainingError("Model probabilities must be finite and in [0, 1]")
    ranked = np.empty_like(values)
    for model_index in range(values.shape[0]):
        for target_index in range(values.shape[2]):
            ranked[model_index, :, target_index] = _percentile_ranks(
                values[model_index, :, target_index]
            )
    return np.mean(ranked, axis=0)


def _metadata_vector(descriptor: SeriesDescriptor) -> tuple[float, ...]:
    plane = [0.0, 0.0, 0.0]
    plane[PLANE_TO_INDEX[descriptor.anatomical_plane]] = 1.0
    return tuple(
        plane + [float(descriptor.fluid_sensitive), float(descriptor.fat_suppression)]
    )


def run_test_ensemble(
    models: list[Any],
    competition_root: Path,
    *,
    device: Any,
    amp_dtype: str,
    sampled_slices: int = 16,
    slices_per_series: int = 12,
    output_size: int = 224,
    encoder_chunk_size: int = 64,
) -> TestEnsembleResult:
    """Decode each test study once and run every fold model offline."""
    import torch

    if not models:
        raise ImagingTrainingError("At least one inference model is required")
    if amp_dtype not in {"bfloat16", "float16", "float32"}:
        raise ImagingTrainingError("amp_dtype must be bfloat16, float16, or float32")
    for name, value in (
        ("sampled_slices", sampled_slices),
        ("slices_per_series", slices_per_series),
        ("output_size", output_size),
        ("encoder_chunk_size", encoder_chunk_size),
    ):
        if value < 1:
            raise ImagingTrainingError(f"{name} must be positive")
    if slices_per_series > sampled_slices:
        raise ImagingTrainingError(
            "slices_per_series cannot exceed decoded sampled_slices"
        )
    competition_root = competition_root.expanduser().resolve()
    test_ids, grouped = load_test_metadata(competition_root)
    test_series_root = competition_root / "test_series"
    if not test_series_root.is_dir():
        raise ImagingTrainingError("Test-series directory is missing")
    expected_directories = {
        (descriptor.study_uid, descriptor.series_uid)
        for descriptors in grouped.values()
        for descriptor in descriptors
    }
    actual_directories = {
        (study_directory.name, series_directory.name)
        for study_directory in test_series_root.iterdir()
        if study_directory.is_dir()
        for series_directory in study_directory.iterdir()
        if series_directory.is_dir()
    }
    if actual_directories != expected_directories:
        raise ImagingTrainingError(
            "Test-series directories do not exactly match test_series.csv"
        )
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[amp_dtype]
    amp_enabled = device.type == "cuda" and amp_dtype != "float32"
    for model in models:
        model.eval()
    predictions: list[list[np.ndarray]] = [[] for _ in models]
    decoded_pixel_arrays = 0
    series_count = 0
    started = time.monotonic()
    with torch.inference_mode():
        for study_index, study_uid in enumerate(test_ids, start=1):
            descriptors = sorted(grouped[study_uid], key=lambda item: item.series_uid)
            volumes: list[np.ndarray] = []
            metadata_vectors: list[tuple[float, ...]] = []
            for descriptor in descriptors:
                loaded = load_dicom_series(
                    test_series_root / descriptor.study_uid / descriptor.series_uid,
                    descriptor.study_uid,
                    descriptor.series_uid,
                    sampled_slices=sampled_slices,
                    output_size=output_size,
                )
                volumes.append(loaded.sampled_images)
                metadata_vectors.append(_metadata_vector(descriptor))
                decoded_pixel_arrays += loaded.summary.decoded_pixel_arrays
                series_count += 1
            images, lengths, metadata = prepare_study_images(
                volumes,
                metadata_vectors,
                slices_per_series=slices_per_series,
                device=device,
            )
            for model_index, model in enumerate(models):
                with torch.autocast(
                    device_type=device.type, dtype=dtype, enabled=amp_enabled
                ):
                    logits = model(
                        images,
                        lengths,
                        metadata,
                        encoder_chunk_size=encoder_chunk_size,
                    )
                if (
                    logits.shape != (len(TARGET_COLUMNS),)
                    or not torch.isfinite(logits).all()
                ):
                    raise ImagingTrainingError(
                        "Inference model returned invalid logits"
                    )
                predictions[model_index].append(
                    torch.sigmoid(logits.float()).cpu().numpy()
                )
            print(
                f"Predicted test study {study_index}/{len(test_ids)} with "
                f"{len(descriptors)} series",
                flush=True,
            )
    elapsed = time.monotonic() - started
    if not math.isfinite(elapsed):
        raise ImagingTrainingError("Inference runtime is nonfinite")
    model_probabilities = np.asarray(predictions, dtype=np.float64)
    expected_shape = (len(models), len(test_ids), len(TARGET_COLUMNS))
    if model_probabilities.shape != expected_shape:
        raise ImagingTrainingError("Inference probability shape is invalid")
    return TestEnsembleResult(
        study_uids=tuple(test_ids),
        model_probabilities=model_probabilities,
        probability_mean=np.mean(model_probabilities, axis=0),
        rank_average=rank_average_probabilities(model_probabilities),
        decoded_pixel_arrays=decoded_pixel_arrays,
        series=series_count,
        elapsed_seconds=elapsed,
    )
