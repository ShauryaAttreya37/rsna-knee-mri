"""Leak-free notebook-driven imaging training for the RSNA knee task."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rsna.evaluation import macro_roc_auc
from rsna.preprocessing_cache import CACHE_MANIFEST_COLUMNS
from rsna.submission import ID_COLUMN, TARGET_COLUMNS
from rsna.training_data import PLANE_TO_INDEX, SERIES_UID_COLUMN

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Sampling and intensity augmentation only. Geometric left/right flips are
# deliberately excluded: medial and lateral compartments are distinguished by
# their position in coronal and axial knee MRI, and the cache carries no
# laterality tag, so a horizontal flip would silently relabel five targets.
SLICE_JITTER_FRACTION = 0.5
INTENSITY_SCALE_RANGE = (0.90, 1.10)
INTENSITY_SHIFT_RANGE = (-0.05, 0.05)
INTENSITY_GAMMA_RANGE = (0.90, 1.10)
WEAK_LABEL_COLUMNS = tuple(f"{target}_prob" for target in TARGET_COLUMNS)


class ImagingTrainingError(RuntimeError):
    """Raised when the training data or execution contract is unsafe."""


@dataclass(frozen=True, slots=True)
class SeriesRecord:
    tensor_path: Path
    metadata_vector: tuple[float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class StudyRecord:
    study_uid: str
    series: tuple[SeriesRecord, ...]
    targets: tuple[float, ...]
    target_weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Identifier-bearing predictions kept outside repository reports."""

    study_uids: tuple[str, ...]
    labels: np.ndarray
    probabilities: np.ndarray
    mean_loss: float


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    weak_train: tuple[StudyRecord, ...]
    weak_validation: tuple[StudyRecord, ...]
    gold_train: tuple[StudyRecord, ...]
    gold_validation: tuple[StudyRecord, ...]
    selected_fold: int
    gold_studies: int
    weak_only_studies: int
    source_sha256: dict[str, str]

    def aggregate_summary(self) -> dict[str, Any]:
        """Return identifier-free counts safe for logs and repository reports."""
        return {
            "selected_fold": self.selected_fold,
            "gold_studies": self.gold_studies,
            "weak_only_studies": self.weak_only_studies,
            "weak_train_studies": len(self.weak_train),
            "weak_validation_studies": len(self.weak_validation),
            "gold_train_studies": len(self.gold_train),
            "gold_validation_studies": len(self.gold_validation),
            "weak_pretraining_contains_gold": False,
            "gold_train_validation_overlap": False,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class StageConfig:
    name: str
    epochs: int
    head_learning_rate: float
    backbone_learning_rate: float
    weight_decay: float = 0.01
    accumulation_steps: int = 4
    slices_per_series: int = 8
    encoder_chunk_size: int = 64
    amp_dtype: str = "bfloat16"
    gradient_clip_norm: float = 1.0
    checkpoint_every_studies: int = 500
    patience: int | None = None
    seed: int = 20260905
    selection_metric: str = "loss"
    augment: bool = False

    def validate(self) -> None:
        if not self.name or any(char in self.name for char in "/\\"):
            raise ImagingTrainingError("Stage name must be a safe path component")
        if not isinstance(self.augment, bool):
            raise ImagingTrainingError("augment must be a boolean")
        for name in (
            "epochs",
            "accumulation_steps",
            "slices_per_series",
            "encoder_chunk_size",
            "checkpoint_every_studies",
        ):
            if int(getattr(self, name)) < 1:
                raise ImagingTrainingError(f"{name} must be positive")
        for name in (
            "head_learning_rate",
            "backbone_learning_rate",
            "weight_decay",
            "gradient_clip_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ImagingTrainingError(f"{name} must be finite and nonnegative")
        if self.amp_dtype not in {"bfloat16", "float16", "float32"}:
            raise ImagingTrainingError(
                "amp_dtype must be bfloat16, float16, or float32"
            )
        if self.selection_metric not in {"loss", "macro_auc"}:
            raise ImagingTrainingError("selection_metric must be loss or macro_auc")
        if self.patience is not None and self.patience < 1:
            raise ImagingTrainingError("patience must be positive when provided")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fields = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as error:
        raise ImagingTrainingError(f"Could not read {path.name}: {error}") from error
    if not fields:
        raise ImagingTrainingError(f"CSV has no header: {path}")
    return fields, rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ImagingTrainingError(f"Could not hash {path.name}: {error}") from error
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ImagingTrainingError(f"Could not read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ImagingTrainingError(f"JSON root must be an object: {path}")
    return value


def _uid(value: str | None, source: str) -> str:
    result = (value or "").strip()
    if (
        not result
        or result in {".", ".."}
        or "/" in result
        or "\\" in result
        or Path(result).name != result
    ):
        raise ImagingTrainingError(f"{source} contains an unsafe identifier")
    return result


def _float_probability(value: str | None, source: str) -> float:
    try:
        result = float(value or "")
    except ValueError as error:
        raise ImagingTrainingError(
            f"{source} contains a nonnumeric probability"
        ) from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ImagingTrainingError(
            f"{source} probabilities must be finite and in [0, 1]"
        )
    return result


def _binary_label(value: str | None, source: str) -> float:
    result = _float_probability(value, source)
    if result not in {0.0, 1.0}:
        raise ImagingTrainingError(f"{source} gold labels must be binary")
    return result


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _confidence_weights(
    targets: Sequence[float], floor: float = 0.25
) -> tuple[float, ...]:
    """Downweight uncertain pseudo-labels while retaining a small learning signal."""
    if not 0.0 <= floor <= 1.0:
        raise ImagingTrainingError("Confidence floor must be in [0, 1]")
    return tuple(max(floor, 2.0 * abs(float(value) - 0.5)) for value in targets)


def _load_series_by_study(cache_root: Path) -> dict[str, tuple[SeriesRecord, ...]]:
    manifest_path = cache_root / "cache_manifest.csv"
    fields, rows = _read_csv(manifest_path)
    if fields != list(CACHE_MANIFEST_COLUMNS):
        raise ImagingTrainingError("Cache manifest columns or order are invalid")
    if not rows:
        raise ImagingTrainingError("Cache manifest is empty")

    root = cache_root.resolve()
    grouped: dict[str, list[SeriesRecord]] = {}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        study_uid = _uid(row.get(ID_COLUMN), manifest_path.name)
        series_uid = _uid(row.get(SERIES_UID_COLUMN), manifest_path.name)
        key = (study_uid, series_uid)
        if key in seen:
            raise ImagingTrainingError("Cache manifest contains duplicate series")
        seen.add(key)
        plane = (row.get("Anatomical_Plane") or "").strip()
        if plane not in PLANE_TO_INDEX:
            raise ImagingTrainingError("Cache manifest contains an invalid plane")
        try:
            fluid = int(row.get("Fluid_Sensitive") or "")
            fat = int(row.get("Fat_Suppression") or "")
        except ValueError as error:
            raise ImagingTrainingError(
                "Cache manifest contains nonbinary metadata"
            ) from error
        if fluid not in {0, 1} or fat not in {0, 1}:
            raise ImagingTrainingError("Cache manifest contains nonbinary metadata")
        relative = Path(row.get("tensor_relative_path") or "")
        tensor_path = (root / relative).resolve()
        if relative.is_absolute() or not _inside(tensor_path, root):
            raise ImagingTrainingError("Cache manifest tensor path escapes cache root")
        if not tensor_path.is_file():
            raise ImagingTrainingError("A cache manifest tensor is missing")
        plane_vector = [0.0, 0.0, 0.0]
        plane_vector[PLANE_TO_INDEX[plane]] = 1.0
        grouped.setdefault(study_uid, []).append(
            SeriesRecord(
                tensor_path=tensor_path,
                metadata_vector=tuple(plane_vector + [float(fluid), float(fat)]),
            )
        )
    return {uid: tuple(series) for uid, series in grouped.items()}


def build_training_plan(
    competition_root: Path,
    cache_root: Path,
    weak_labels_path: Path,
    folds_path: Path,
    *,
    selected_fold: int,
    weak_validation_fraction: float = 0.1,
    seed: int = 20260905,
    expected_studies: int | None = 4407,
    expected_gold_studies: int | None = 58,
) -> TrainingPlan:
    """Validate all inputs and build disjoint weak/gold training partitions."""
    if not 0.0 < weak_validation_fraction < 1.0:
        raise ImagingTrainingError("weak_validation_fraction must be inside (0, 1)")
    competition_root = competition_root.expanduser().resolve()
    cache_root = cache_root.expanduser().resolve()
    weak_labels_path = weak_labels_path.expanduser().resolve()
    folds_path = folds_path.expanduser().resolve()
    train_path = competition_root / "train.csv"
    cache_config_path = cache_root / "cache_config.json"
    for path in (
        train_path,
        cache_root / "cache_manifest.csv",
        cache_config_path,
        weak_labels_path,
        folds_path,
    ):
        if not path.is_file():
            raise ImagingTrainingError(f"Required training input is missing: {path}")

    cache_config = _read_json(cache_config_path)
    expected_cache_contract = {
        "sampled_slices": 16,
        "output_size": 224,
        "dtype": "float16",
        "spatial_transform": "physical_full_fov_aspect_fit_zero_pad",
    }
    for name, expected in expected_cache_contract.items():
        if cache_config.get(name) != expected:
            raise ImagingTrainingError(
                f"Cache contract mismatch for {name}: expected {expected!r}"
            )

    series_by_study = _load_series_by_study(cache_root)
    cached_series = sum(len(series) for series in series_by_study.values())
    if cache_config.get("eligible_series") != cached_series:
        raise ImagingTrainingError(
            "Cache configuration and manifest series counts differ"
        )
    if cache_config.get("eligible_studies") != len(series_by_study):
        raise ImagingTrainingError(
            "Cache configuration and manifest study counts differ"
        )
    train_fields, train_rows = _read_csv(train_path)
    if train_fields != [ID_COLUMN, "Report", *TARGET_COLUMNS]:
        raise ImagingTrainingError("train.csv columns or order are invalid")
    if expected_studies is not None and len(train_rows) != expected_studies:
        raise ImagingTrainingError(
            f"Expected {expected_studies} studies, observed {len(train_rows)}"
        )
    all_ids: list[str] = []
    seen_ids: set[str] = set()
    gold_targets: dict[str, tuple[float, ...]] = {}
    for row in train_rows:
        study_uid = _uid(row.get(ID_COLUMN), train_path.name)
        if study_uid in seen_ids:
            raise ImagingTrainingError("train.csv contains duplicate studies")
        seen_ids.add(study_uid)
        all_ids.append(study_uid)
        values = [(row.get(target) or "").strip() for target in TARGET_COLUMNS]
        present = [bool(value) for value in values]
        if any(present) and not all(present):
            raise ImagingTrainingError("train.csv contains a partially labeled study")
        if all(present):
            gold_targets[study_uid] = tuple(
                _binary_label(value, train_path.name) for value in values
            )
    all_id_set = set(all_ids)
    if set(series_by_study) != all_id_set:
        raise ImagingTrainingError("Cache study membership differs from train.csv")
    if expected_gold_studies is not None and len(gold_targets) != expected_gold_studies:
        raise ImagingTrainingError(
            f"Expected {expected_gold_studies} gold studies, observed {len(gold_targets)}"
        )

    weak_fields, weak_rows = _read_csv(weak_labels_path)
    if weak_fields != [ID_COLUMN, *WEAK_LABEL_COLUMNS]:
        raise ImagingTrainingError("Weak-label columns or order are invalid")
    weak_targets: dict[str, tuple[float, ...]] = {}
    for row in weak_rows:
        study_uid = _uid(row.get(ID_COLUMN), weak_labels_path.name)
        if study_uid in weak_targets:
            raise ImagingTrainingError("Weak-label CSV contains duplicate studies")
        weak_targets[study_uid] = tuple(
            _float_probability(row.get(column), weak_labels_path.name)
            for column in WEAK_LABEL_COLUMNS
        )
    if set(weak_targets) != all_id_set:
        raise ImagingTrainingError("Weak-label study membership differs from train.csv")

    fold_fields, fold_rows = _read_csv(folds_path)
    if fold_fields != [ID_COLUMN, "fold"]:
        raise ImagingTrainingError("Gold-fold columns or order are invalid")
    fold_by_study: dict[str, int] = {}
    for row in fold_rows:
        study_uid = _uid(row.get(ID_COLUMN), folds_path.name)
        if study_uid in fold_by_study:
            raise ImagingTrainingError("Gold-fold CSV contains duplicate studies")
        try:
            fold = int(row.get("fold") or "")
        except ValueError as error:
            raise ImagingTrainingError(
                "Gold-fold CSV contains a noninteger fold"
            ) from error
        if fold < 0:
            raise ImagingTrainingError("Gold-fold CSV contains a negative fold")
        fold_by_study[study_uid] = fold
    if set(fold_by_study) != set(gold_targets):
        raise ImagingTrainingError("Gold-fold membership differs from gold labels")
    if expected_gold_studies == 58 and set(fold_by_study.values()) != set(range(5)):
        raise ImagingTrainingError(
            "The official gold split must contain folds 0 through 4"
        )
    if selected_fold not in set(fold_by_study.values()):
        raise ImagingTrainingError(f"Selected fold {selected_fold} does not exist")

    weak_only_ids = [uid for uid in all_ids if uid not in gold_targets]
    ranked_weak = sorted(
        weak_only_ids,
        key=lambda uid: hashlib.sha256(f"{seed}:{uid}".encode()).digest(),
    )
    validation_count = max(1, round(len(ranked_weak) * weak_validation_fraction))
    weak_validation_ids = set(ranked_weak[:validation_count])

    def weak_record(uid: str) -> StudyRecord:
        targets = weak_targets[uid]
        return StudyRecord(
            uid, series_by_study[uid], targets, _confidence_weights(targets)
        )

    def gold_record(uid: str) -> StudyRecord:
        targets = gold_targets[uid]
        return StudyRecord(
            uid, series_by_study[uid], targets, (1.0,) * len(TARGET_COLUMNS)
        )

    weak_train = tuple(
        weak_record(uid) for uid in weak_only_ids if uid not in weak_validation_ids
    )
    weak_validation = tuple(
        weak_record(uid) for uid in weak_only_ids if uid in weak_validation_ids
    )
    gold_train = tuple(
        gold_record(uid)
        for uid in all_ids
        if uid in gold_targets and fold_by_study[uid] != selected_fold
    )
    gold_validation = tuple(
        gold_record(uid)
        for uid in all_ids
        if uid in gold_targets and fold_by_study[uid] == selected_fold
    )
    if not weak_train or not weak_validation or not gold_train or not gold_validation:
        raise ImagingTrainingError("A required training partition is empty")
    if {record.study_uid for record in gold_train} & {
        record.study_uid for record in gold_validation
    }:
        raise AssertionError("Gold train/validation partitions overlap")
    if {record.study_uid for record in (*weak_train, *weak_validation)} & set(
        gold_targets
    ):
        raise AssertionError("Gold studies leaked into weak pretraining")
    if expected_gold_studies == 58:
        for name, records in (
            ("gold training", gold_train),
            ("gold validation", gold_validation),
        ):
            matrix = np.asarray([record.targets for record in records], dtype=np.int8)
            positives = matrix.sum(axis=0)
            if np.any(positives == 0) or np.any(positives == len(records)):
                raise ImagingTrainingError(
                    f"The selected {name} partition is not class-complete"
                )

    return TrainingPlan(
        weak_train=weak_train,
        weak_validation=weak_validation,
        gold_train=gold_train,
        gold_validation=gold_validation,
        selected_fold=selected_fold,
        gold_studies=len(gold_targets),
        weak_only_studies=len(weak_only_ids),
        source_sha256={
            "train.csv": _sha256(train_path),
            "cache_manifest.csv": _sha256(cache_root / "cache_manifest.csv"),
            "cache_config.json": _sha256(cache_config_path),
            "weak_labels.csv": _sha256(weak_labels_path),
            "gold_folds.csv": _sha256(folds_path),
        },
    )


def _sample_centers(
    slice_count: int, count: int, rng: np.random.Generator | None
) -> np.ndarray:
    """Choose slice centers, optionally jittered inside their own spacing."""
    centers = np.linspace(0, slice_count - 1, count)
    if rng is not None and count > 1:
        radius = SLICE_JITTER_FRACTION * (slice_count - 1) / (count - 1)
        centers = centers + rng.uniform(-radius, radius, size=count)
    return np.clip(np.rint(centers), 0, slice_count - 1).astype(np.int64)


def _augment_intensity(volume: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply one per-series gamma, scale, and shift, preserving the [0, 1] range."""
    gamma = rng.uniform(*INTENSITY_GAMMA_RANGE)
    scale = rng.uniform(*INTENSITY_SCALE_RANGE)
    shift = rng.uniform(*INTENSITY_SHIFT_RANGE)
    adjusted = scale * np.power(volume, gamma, dtype=np.float32) + shift
    return np.clip(adjusted, 0.0, 1.0, out=adjusted)


def load_study_tensors(
    record: StudyRecord,
    *,
    slices_per_series: int,
    device: Any,
    augment_rng: np.random.Generator | None = None,
) -> tuple[Any, tuple[int, ...], Any, Any, Any]:
    """Load one study as normalized adjacent-slice 2.5D inputs."""
    volumes: list[np.ndarray] = []
    for series in record.series:
        try:
            volume = np.load(series.tensor_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ImagingTrainingError(
                f"Could not load a cached tensor: {error}"
            ) from error
        if volume.dtype != np.float16:
            raise ImagingTrainingError("Cached tensors must use float16")
        volumes.append(volume)
    images, lengths, metadata = prepare_study_images(
        volumes,
        [series.metadata_vector for series in record.series],
        slices_per_series=slices_per_series,
        device=device,
        augment_rng=augment_rng,
    )
    try:
        import torch
    except ImportError as error:
        raise ImagingTrainingError(
            "PyTorch is required for imaging training"
        ) from error
    targets = torch.tensor(record.targets, dtype=torch.float32, device=device)
    weights = torch.tensor(record.target_weights, dtype=torch.float32, device=device)
    return images, lengths, metadata, targets, weights


def prepare_study_images(
    volumes: Sequence[np.ndarray],
    metadata_vectors: Sequence[tuple[float, float, float, float, float]],
    *,
    slices_per_series: int,
    device: Any,
    augment_rng: np.random.Generator | None = None,
) -> tuple[Any, tuple[int, ...], Any]:
    """Convert normalized series volumes into model-ready 2.5D inputs.

    Passing ``augment_rng`` enables training-only slice-sampling and intensity
    augmentation. Validation and inference must leave it unset so their inputs
    stay deterministic and identical to the packaged Kaggle path.
    """
    if slices_per_series < 1:
        raise ImagingTrainingError("slices_per_series must be positive")
    if not volumes or len(volumes) != len(metadata_vectors):
        raise ImagingTrainingError(
            "Series volumes and metadata must be nonempty and aligned"
        )
    if any(
        len(vector) != 5 or not np.all(np.isfinite(vector))
        for vector in metadata_vectors
    ):
        raise ImagingTrainingError("Every series metadata vector must contain 5 values")
    try:
        import torch
    except ImportError as error:
        raise ImagingTrainingError(
            "PyTorch is required for imaging training"
        ) from error

    image_parts: list[Any] = []
    lengths: list[int] = []
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
    for volume in volumes:
        volume = np.asarray(volume)
        if (
            volume.ndim != 3
            or volume.shape[0] < 1
            or volume.shape[1] != volume.shape[2]
        ):
            raise ImagingTrainingError(
                "Series volumes must have shape [slices, size, size]"
            )
        if not np.issubdtype(volume.dtype, np.floating):
            raise ImagingTrainingError("Series volumes must use floating-point values")
        if augment_rng is not None:
            volume = _augment_intensity(
                volume.astype(np.float32, copy=False), augment_rng
            )
        count = min(slices_per_series, int(volume.shape[0]))
        centers = _sample_centers(int(volume.shape[0]), count, augment_rng)
        previous = np.maximum(centers - 1, 0)
        following = np.minimum(centers + 1, volume.shape[0] - 1)
        triplets = np.stack(
            [volume[previous], volume[centers], volume[following]], axis=1
        ).astype(np.float32, copy=False)
        if (
            not np.isfinite(triplets).all()
            or triplets.min() < 0.0
            or triplets.max() > 1.0
        ):
            raise ImagingTrainingError(
                "Cached tensor values must be finite and in [0, 1]"
            )
        triplets = np.ascontiguousarray((triplets - mean) / std)
        image_parts.append(torch.from_numpy(triplets))
        lengths.append(count)
    images = torch.cat(image_parts, dim=0).to(device=device, non_blocking=True)
    metadata = torch.tensor(
        metadata_vectors,
        dtype=torch.float32,
        device=device,
    )
    return images, tuple(lengths), metadata


def build_multiseries_model(backbone: Any, feature_dimension: int):
    """Attach target-specific slice and series attention to a slice backbone."""
    try:
        import torch
        from torch import nn
        from torch.utils.checkpoint import checkpoint
    except ImportError as error:
        raise ImagingTrainingError(
            "PyTorch is required for imaging training"
        ) from error
    if feature_dimension < 1:
        raise ImagingTrainingError("feature_dimension must be positive")

    class TargetSpecificMultiSeriesModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.feature_dimension = feature_dimension
            self.slice_projection = nn.Sequential(
                nn.LayerNorm(feature_dimension),
                nn.Linear(feature_dimension, feature_dimension),
                nn.GELU(),
            )
            self.slice_attention = nn.Linear(feature_dimension, len(TARGET_COLUMNS))
            self.series_projection = nn.Sequential(
                nn.Linear(feature_dimension + 5, feature_dimension),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.LayerNorm(feature_dimension),
            )
            self.series_queries = nn.Parameter(
                torch.empty(len(TARGET_COLUMNS), feature_dimension)
            )
            self.head_weight = nn.Parameter(
                torch.empty(len(TARGET_COLUMNS), feature_dimension)
            )
            self.head_bias = nn.Parameter(torch.zeros(len(TARGET_COLUMNS)))
            nn.init.normal_(self.series_queries, std=0.02)
            nn.init.normal_(self.head_weight, std=0.02)

        def _encode(self, images: Any, chunk_size: int) -> Any:
            encoded: list[Any] = []
            checkpoint_backbone = self.training and any(
                parameter.requires_grad for parameter in self.backbone.parameters()
            )
            for start in range(0, images.shape[0], chunk_size):
                image_chunk = images[start : start + chunk_size]
                features = (
                    checkpoint(self.backbone, image_chunk, use_reentrant=False)
                    if checkpoint_backbone
                    else self.backbone(image_chunk)
                )
                if isinstance(features, dict):
                    features = features.get("x_norm_clstoken")
                if features is None or features.ndim != 2:
                    raise ImagingTrainingError(
                        "Backbone must return [images, features]"
                    )
                encoded.append(features)
            return torch.cat(encoded, dim=0)

        def forward(
            self,
            images: Any,
            series_lengths: Sequence[int],
            metadata: Any,
            *,
            encoder_chunk_size: int = 64,
        ) -> Any:
            if sum(series_lengths) != int(images.shape[0]):
                raise ImagingTrainingError("Series lengths do not match image count")
            if len(series_lengths) != int(metadata.shape[0]) or metadata.shape[1] != 5:
                raise ImagingTrainingError("Series metadata shape is invalid")
            features = self.slice_projection(self._encode(images, encoder_chunk_size))
            series_target_features: list[Any] = []
            offset = 0
            for length in series_lengths:
                part = features[offset : offset + length]
                scores = self.slice_attention(part)
                weights = torch.softmax(scores, dim=0)
                series_target_features.append(torch.einsum("st,sd->td", weights, part))
                offset += length
            stacked = torch.stack(series_target_features, dim=0)
            expanded_metadata = metadata[:, None, :].expand(-1, len(TARGET_COLUMNS), -1)
            enriched = self.series_projection(
                torch.cat([stacked, expanded_metadata], dim=-1)
            )
            series_scores = torch.einsum(
                "rtd,td->rt", enriched, self.series_queries
            ) / math.sqrt(feature_dimension)
            series_weights = torch.softmax(series_scores, dim=0)
            study_features = torch.einsum("rt,rtd->td", series_weights, enriched)
            return torch.sum(study_features * self.head_weight, dim=-1) + self.head_bias

    return TargetSpecificMultiSeriesModel()


def configure_backbone(model: Any, *, unfreeze_last_blocks: int) -> dict[str, int]:
    """Freeze DINOv2 except for an explicitly selected number of final blocks."""
    if unfreeze_last_blocks < 0:
        raise ImagingTrainingError("unfreeze_last_blocks must be nonnegative")
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    blocks = getattr(model.backbone, "blocks", None)
    if unfreeze_last_blocks:
        if blocks is None or unfreeze_last_blocks > len(blocks):
            raise ImagingTrainingError(
                "Backbone does not expose enough transformer blocks"
            )
        for block in blocks[-unfreeze_last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        norm = getattr(model.backbone, "norm", None)
        if norm is not None:
            for parameter in norm.parameters():
                parameter.requires_grad = True
    return {
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "unfrozen_backbone_blocks": unfreeze_last_blocks,
    }


def compute_pos_weights(
    records: Sequence[StudyRecord], *, maximum: float = 10.0
) -> np.ndarray:
    """Return tempered inverse-prevalence weights for rare positive targets."""
    if not records:
        raise ImagingTrainingError("Cannot compute target weights from no studies")
    if not math.isfinite(maximum) or maximum < 1.0:
        raise ImagingTrainingError("Maximum positive weight must be finite and >= 1")
    targets = np.asarray([record.targets for record in records], dtype=np.float64)
    positives = targets.sum(axis=0)
    negatives = len(records) - positives
    if np.any(positives <= 0.0) or np.any(negatives <= 0.0):
        raise ImagingTrainingError(
            "Every training target must contain positive and negative mass"
        )
    # Square-root tempering avoids the severe calibration shift of a raw inverse ratio.
    return np.clip(np.sqrt(negatives / positives), 1.0, maximum).astype(np.float32)


def _set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_model(
    model: Any,
    records: Sequence[StudyRecord],
    *,
    device: Any,
    slices_per_series: int,
    encoder_chunk_size: int,
    amp_dtype: str,
) -> InferenceResult:
    """Run deterministic study inference and return identifier-bearing outputs."""
    import torch
    from torch.nn import functional

    if not records:
        raise ImagingTrainingError("Cannot run inference on no studies")
    if amp_dtype not in {"bfloat16", "float16", "float32"}:
        raise ImagingTrainingError("amp_dtype must be bfloat16, float16, or float32")
    model.eval()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[amp_dtype]
    amp_enabled = device.type == "cuda" and amp_dtype != "float32"
    losses: list[float] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for record in records:
            images, lengths, metadata, targets, weights = load_study_tensors(
                record, slices_per_series=slices_per_series, device=device
            )
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=amp_enabled
            ):
                logits = model(
                    images, lengths, metadata, encoder_chunk_size=encoder_chunk_size
                )
                per_target = functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none"
                )
                loss = torch.sum(per_target * weights) / torch.sum(weights)
            losses.append(float(loss.item()))
            labels.append(targets.cpu().numpy())
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
    label_matrix = np.asarray(labels, dtype=float)
    probability_matrix = np.asarray(probabilities, dtype=float)
    return InferenceResult(
        study_uids=tuple(record.study_uid for record in records),
        labels=label_matrix,
        probabilities=probability_matrix,
        mean_loss=float(np.mean(losses)),
    )


def evaluate_model(
    model: Any,
    records: Sequence[StudyRecord],
    *,
    device: Any,
    slices_per_series: int,
    encoder_chunk_size: int,
    amp_dtype: str,
    binary_targets: bool,
) -> dict[str, Any]:
    """Evaluate a partition without retaining identifiers or per-study outputs."""
    predictions = predict_model(
        model,
        records,
        device=device,
        slices_per_series=slices_per_series,
        encoder_chunk_size=encoder_chunk_size,
        amp_dtype=amp_dtype,
    )
    result: dict[str, Any] = {
        "loss": predictions.mean_loss,
        "studies": len(records),
    }
    if binary_targets:
        auc = macro_roc_auc(
            predictions.labels, predictions.probabilities, TARGET_COLUMNS
        )
        result.update(
            {
                "macro_auc": auc.macro_auc,
                "per_target_auc": auc.per_target_auc,
                "binary_accuracy": float(
                    np.mean((predictions.probabilities >= 0.5) == predictions.labels)
                ),
            }
        )
    return result


def _atomic_torch_save(torch: Any, payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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


def run_training_stage(
    model: Any,
    train_records: Sequence[StudyRecord],
    validation_records: Sequence[StudyRecord],
    *,
    device: Any,
    output_root: Path,
    report_output: Path,
    project_root: Path,
    competition_root: Path,
    config: StageConfig,
    source_sha256: dict[str, str],
    validation_targets_are_binary: bool,
    resume: bool = True,
) -> dict[str, Any]:
    """Run one resumable stage; intended to be called from explicit notebook cells."""
    import torch
    from torch.nn import functional

    config.validate()
    if not train_records or not validation_records:
        raise ImagingTrainingError(
            "Training and validation partitions must be nonempty"
        )
    project_root = project_root.expanduser().resolve()
    competition_root = competition_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    report_output = report_output.expanduser().resolve()
    if (
        _inside(output_root, project_root)
        or _inside(project_root, output_root)
        or _inside(output_root, competition_root)
        or _inside(competition_root, output_root)
    ):
        raise ImagingTrainingError(
            "Checkpoints must remain outside project and raw data"
        )
    if not _inside(report_output, project_root):
        raise ImagingTrainingError("Aggregate report must remain inside project")

    _set_seed(torch, config.seed)
    model.to(device)
    backbone_parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    groups: list[dict[str, Any]] = []
    if head_parameters:
        groups.append({"params": head_parameters, "lr": config.head_learning_rate})
    if backbone_parameters:
        groups.append(
            {"params": backbone_parameters, "lr": config.backbone_learning_rate}
        )
    if not groups:
        raise ImagingTrainingError("Model has no trainable parameters")
    trainable_names = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    trainable_signature = hashlib.sha256(
        json.dumps(trainable_names, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    positive_weights_array = compute_pos_weights(train_records)
    positive_weights = torch.tensor(
        positive_weights_array, dtype=torch.float32, device=device
    )
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[config.amp_dtype]
    amp_enabled = device.type == "cuda" and config.amp_dtype != "float32"
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and dtype == torch.float16
    )
    latest_path = output_root / config.name / "latest.pt"
    best_path = output_root / config.name / "best.pt"
    implementation_sha256 = _sha256(Path(__file__))
    next_epoch = 1
    resume_position = 0
    resume_running_loss = 0.0
    completed_epoch = 0
    history: list[dict[str, Any]] = []
    best_value = math.inf if config.selection_metric == "loss" else -math.inf
    epochs_without_improvement = 0
    config_dict = asdict(config)
    if resume and latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint.get("format_version") != 1 or checkpoint.get(
            "target_columns"
        ) != list(TARGET_COLUMNS):
            raise ImagingTrainingError("Checkpoint contract is incompatible")
        if (
            checkpoint.get("stage_config") != config_dict
            or checkpoint.get("source_sha256") != source_sha256
            or checkpoint.get("implementation_sha256") != implementation_sha256
            or checkpoint.get("trainable_signature") != trainable_signature
        ):
            raise ImagingTrainingError(
                "Checkpoint configuration or inputs changed; use a new output root"
            )
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        checkpoint_epoch = int(checkpoint["epoch"])
        checkpoint_position = int(checkpoint.get("position", len(train_records)))
        epoch_complete = bool(checkpoint.get("epoch_complete", True))
        if epoch_complete:
            completed_epoch = checkpoint_epoch
            next_epoch = checkpoint_epoch + 1
        else:
            next_epoch = checkpoint_epoch
            resume_position = checkpoint_position
            resume_running_loss = float(checkpoint.get("running_loss", 0.0))
            if not 0 <= resume_position <= len(train_records):
                raise ImagingTrainingError("Checkpoint position is invalid")
        history = list(checkpoint["history"])
        best_value = float(checkpoint["best_value"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        print(
            f"Resuming {config.name} at epoch {next_epoch}, "
            f"study {resume_position}/{len(train_records)}",
            flush=True,
        )

    def checkpoint_payload(
        *,
        epoch: int,
        position: int,
        running_loss: float,
        epoch_complete: bool,
    ) -> dict[str, Any]:
        return {
            "format_version": 1,
            "model": "dinov2_vits14_target_specific_multiseries",
            "target_columns": list(TARGET_COLUMNS),
            "stage_config": config_dict,
            "source_sha256": source_sha256,
            "implementation_sha256": implementation_sha256,
            "trainable_signature": trainable_signature,
            "epoch": epoch,
            "position": position,
            "running_loss": running_loss,
            "epoch_complete": epoch_complete,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
            "best_value": best_value,
            "epochs_without_improvement": epochs_without_improvement,
        }

    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(next_epoch, config.epochs + 1):
        model.train()
        order = np.random.default_rng(config.seed + epoch).permutation(
            len(train_records)
        )
        start_position = resume_position if epoch == next_epoch else 0
        running_loss = resume_running_loss if epoch == next_epoch else 0.0
        next_checkpoint_position = (
            (start_position // config.checkpoint_every_studies) + 1
        ) * config.checkpoint_every_studies
        for zero_based_position in range(start_position, len(order)):
            position = zero_based_position + 1
            record_index = order[zero_based_position]
            record = train_records[int(record_index)]
            # Seeded from the study index rather than the iteration order so a
            # resumed run reproduces exactly the same augmented inputs.
            augment_rng = (
                np.random.default_rng([config.seed, epoch, int(record_index)])
                if config.augment
                else None
            )
            images, lengths, metadata, targets, weights = load_study_tensors(
                record,
                slices_per_series=config.slices_per_series,
                device=device,
                augment_rng=augment_rng,
            )
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=amp_enabled
            ):
                logits = model(
                    images,
                    lengths,
                    metadata,
                    encoder_chunk_size=config.encoder_chunk_size,
                )
                per_target = functional.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    reduction="none",
                    pos_weight=positive_weights,
                )
                loss = torch.sum(per_target * weights) / torch.sum(weights)
                group_start = (
                    (position - 1) // config.accumulation_steps
                ) * config.accumulation_steps
                group_size = min(config.accumulation_steps, len(order) - group_start)
                scaled_loss = loss / group_size
            if not torch.isfinite(loss):
                raise ImagingTrainingError("Training loss became nonfinite")
            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.item())
            update = position % config.accumulation_steps == 0 or position == len(order)
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for group in groups for parameter in group["params"]],
                    config.gradient_clip_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if position <= 3 or position % 100 == 0 or position == len(order):
                print(
                    f"{config.name} epoch {epoch}/{config.epochs} "
                    f"study {position}/{len(order)} loss={running_loss / position:.5f}",
                    flush=True,
                )
            if update and position >= next_checkpoint_position:
                _atomic_torch_save(
                    torch,
                    checkpoint_payload(
                        epoch=epoch,
                        position=position,
                        running_loss=running_loss,
                        epoch_complete=False,
                    ),
                    latest_path,
                )
                while next_checkpoint_position <= position:
                    next_checkpoint_position += config.checkpoint_every_studies

        # Persist the optimized state before validation so an interruption during the
        # validation pass repeats only validation, not the whole epoch.
        _atomic_torch_save(
            torch,
            checkpoint_payload(
                epoch=epoch,
                position=len(order),
                running_loss=running_loss,
                epoch_complete=False,
            ),
            latest_path,
        )

        validation = evaluate_model(
            model,
            validation_records,
            device=device,
            slices_per_series=config.slices_per_series,
            encoder_chunk_size=config.encoder_chunk_size,
            amp_dtype=config.amp_dtype,
            binary_targets=validation_targets_are_binary,
        )
        training_loss = running_loss / len(order)
        metric_value = float(validation[config.selection_metric])
        improved = (
            metric_value < best_value
            if config.selection_metric == "loss"
            else metric_value > best_value
        )
        if improved:
            best_value = metric_value
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        completed_epoch = epoch
        history.append(
            {"epoch": epoch, "training_loss": training_loss, "validation": validation}
        )
        payload = checkpoint_payload(
            epoch=epoch,
            position=len(order),
            running_loss=running_loss,
            epoch_complete=True,
        )
        _atomic_torch_save(torch, payload, latest_path)
        if improved:
            _atomic_torch_save(torch, payload, best_path)
        print(
            f"{config.name} epoch {epoch}: train_loss={training_loss:.5f}, "
            f"validation={validation}, best={best_value:.5f}",
            flush=True,
        )
        if (
            config.patience is not None
            and epochs_without_improvement >= config.patience
        ):
            print(f"Early stopping after {epoch} epochs", flush=True)
            break
        resume_position = 0
        resume_running_loss = 0.0

    report = {
        "status": "complete",
        "generated_at": datetime.now(UTC).isoformat(),
        "privacy": "Identifier-free aggregate training metrics only; checkpoints remain external.",
        "configuration": config_dict,
        "positive_target_weights": {
            target: float(positive_weights_array[index])
            for index, target in enumerate(TARGET_COLUMNS)
        },
        "counts": {
            "training_studies": len(train_records),
            "validation_studies": len(validation_records),
            "epochs_completed": completed_epoch,
        },
        "best_value": best_value,
        "history": history,
        "runtime_seconds_this_call": time.monotonic() - started,
        "source_sha256": source_sha256,
        "implementation_sha256": implementation_sha256,
        "trainable_signature": trainable_signature,
        "external_artifacts": [
            f"{config.name}/latest.pt",
            f"{config.name}/best.pt",
        ],
    }
    _write_json_atomic(report_output, report)
    return report


def load_checkpoint_weights(
    model: Any, checkpoint_path: Path, *, device: Any
) -> dict[str, Any]:
    """Load model weights while enforcing target and architecture contracts."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ImagingTrainingError("Unsupported checkpoint format")
    if checkpoint.get("model") != "dinov2_vits14_target_specific_multiseries":
        raise ImagingTrainingError("Checkpoint model architecture is incompatible")
    if checkpoint.get("target_columns") != list(TARGET_COLUMNS):
        raise ImagingTrainingError("Checkpoint target order is incompatible")
    model.load_state_dict(checkpoint["state_dict"])
    return {
        "epoch": int(checkpoint["epoch"]),
        "best_value": float(checkpoint["best_value"]),
        "stage_config": checkpoint["stage_config"],
        "source_sha256": checkpoint["source_sha256"],
    }
