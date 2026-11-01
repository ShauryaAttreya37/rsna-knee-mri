from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from rsna.imaging_training import (
    ImagingTrainingError,
    StageConfig,
    StudyRecord,
    _augment_intensity,
    _sample_centers,
    build_training_plan,
    compute_pos_weights,
    prepare_study_images,
)
from rsna.preprocessing_cache import CACHE_MANIFEST_COLUMNS
from rsna.submission import ID_COLUMN, TARGET_COLUMNS


def _write_csv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    competition_root = tmp_path / "raw" / "competition"
    cache_root = tmp_path / "processed" / "training_cache"
    weak_labels_path = tmp_path / "manifests" / "weak_labels.csv"
    folds_path = tmp_path / "manifests" / "gold_folds.csv"
    study_ids = [f"study-{index}" for index in range(6)]

    train_rows: list[list[object]] = []
    weak_rows: list[list[object]] = []
    manifest_rows: list[list[object]] = []
    for index, study_uid in enumerate(study_ids):
        labels: list[object] = (
            [int((index + target) % 2 == 0) for target in range(len(TARGET_COLUMNS))]
            if index < 2
            else [""] * len(TARGET_COLUMNS)
        )
        train_rows.append([study_uid, f"private report {index}", *labels])
        probabilities = [
            0.1 + 0.8 * int((index + target) % 2 == 0)
            for target in range(len(TARGET_COLUMNS))
        ]
        weak_rows.append([study_uid, *probabilities])
        relative = Path("studies") / study_uid / "series-0.npy"
        tensor_path = cache_root / relative
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        with tensor_path.open("wb") as file:
            np.save(file, np.zeros((16, 224, 224), dtype=np.float16))
        manifest_rows.append(
            [
                study_uid,
                "series-0",
                ("Axial", "Coronal", "Sagittal")[index % 3],
                index % 2,
                (index + 1) % 2,
                relative.as_posix(),
                "0" * 64,
                tensor_path.stat().st_size,
                16,
                "geometry",
                224,
                224,
                1.0,
                1.0,
                224.0,
                224.0,
                224,
                224,
                0.0,
                1.0,
            ]
        )

    _write_csv(
        competition_root / "train.csv",
        [ID_COLUMN, "Report", *TARGET_COLUMNS],
        train_rows,
    )
    _write_csv(
        weak_labels_path,
        [ID_COLUMN, *[f"{target}_prob" for target in TARGET_COLUMNS]],
        weak_rows,
    )
    _write_csv(folds_path, [ID_COLUMN, "fold"], [[study_ids[0], 0], [study_ids[1], 1]])
    _write_csv(
        cache_root / "cache_manifest.csv", list(CACHE_MANIFEST_COLUMNS), manifest_rows
    )
    (cache_root / "cache_config.json").write_text(
        json.dumps(
            {
                "eligible_studies": 6,
                "eligible_series": 6,
                "sampled_slices": 16,
                "output_size": 224,
                "dtype": "float16",
                "spatial_transform": "physical_full_fov_aspect_fit_zero_pad",
            }
        ),
        encoding="utf-8",
    )
    return {
        "competition_root": competition_root,
        "cache_root": cache_root,
        "weak_labels_path": weak_labels_path,
        "folds_path": folds_path,
    }


def test_training_plan_excludes_every_gold_study_from_weak_pretraining(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)

    plan = build_training_plan(
        **inputs,
        selected_fold=0,
        weak_validation_fraction=0.25,
        expected_studies=6,
        expected_gold_studies=2,
    )

    assert len(plan.weak_train) == 3
    assert len(plan.weak_validation) == 1
    assert len(plan.gold_train) == 1
    assert len(plan.gold_validation) == 1
    weak_ids = {
        record.study_uid for record in (*plan.weak_train, *plan.weak_validation)
    }
    gold_ids = {
        record.study_uid for record in (*plan.gold_train, *plan.gold_validation)
    }
    assert weak_ids.isdisjoint(gold_ids)
    assert plan.aggregate_summary()["weak_pretraining_contains_gold"] is False
    assert all(
        weight == pytest.approx(0.8) for weight in plan.weak_train[0].target_weights
    )


def test_training_plan_rejects_cache_contract_change(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    config_path = inputs["cache_root"] / "cache_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["output_size"] = 256
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ImagingTrainingError, match="Cache contract mismatch"):
        build_training_plan(
            **inputs,
            selected_fold=0,
            expected_studies=6,
            expected_gold_studies=2,
        )


def test_training_plan_rejects_weak_membership_mismatch(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    fields, rows = [], []
    with inputs["weak_labels_path"].open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        fields = next(reader)
        rows = list(reader)
    _write_csv(inputs["weak_labels_path"], fields, rows[:-1])

    with pytest.raises(ImagingTrainingError, match="Weak-label study membership"):
        build_training_plan(
            **inputs,
            selected_fold=0,
            expected_studies=6,
            expected_gold_studies=2,
        )


def test_tempered_positive_weights_are_finite_and_bounded(tmp_path: Path) -> None:
    del tmp_path
    records = tuple(
        StudyRecord(
            study_uid=str(index),
            series=(),
            targets=tuple(
                float((index + target) % (target + 2) == 0)
                for target in range(len(TARGET_COLUMNS))
            ),
            target_weights=(1.0,) * len(TARGET_COLUMNS),
        )
        for index in range(30)
    )

    weights = compute_pos_weights(records, maximum=5.0)

    assert weights.shape == (len(TARGET_COLUMNS),)
    assert np.isfinite(weights).all()
    assert np.all(weights >= 1.0)
    assert np.all(weights <= 5.0)


def _synthetic_volume(seed: int, slices: int = 16, size: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(slices, size, size)).astype(np.float16)


def test_slice_centers_are_deterministic_and_bounded_without_a_generator() -> None:
    baseline = _sample_centers(16, 12, None)

    assert np.array_equal(baseline, _sample_centers(16, 12, None))
    assert np.array_equal(baseline, np.rint(np.linspace(0, 15, 12)).astype(np.int64))
    assert baseline.min() >= 0 and baseline.max() <= 15


def test_jittered_slice_centers_stay_in_range_and_actually_move() -> None:
    moved = False
    for seed in range(32):
        centers = _sample_centers(16, 12, np.random.default_rng(seed))
        assert centers.min() >= 0 and centers.max() <= 15
        assert centers.shape == (12,)
        moved = moved or not np.array_equal(centers, _sample_centers(16, 12, None))
    assert moved, "Slice jitter never changed the sampled centers"


def test_intensity_augmentation_preserves_the_unit_range() -> None:
    volume = _synthetic_volume(7).astype(np.float32)

    for seed in range(16):
        adjusted = _augment_intensity(volume, np.random.default_rng(seed))
        assert adjusted.shape == volume.shape
        assert np.isfinite(adjusted).all()
        assert adjusted.min() >= 0.0 and adjusted.max() <= 1.0


def test_augmented_inputs_differ_from_clean_inputs_but_repeat_for_one_seed() -> None:
    torch = pytest.importorskip("torch")
    volumes = [_synthetic_volume(1), _synthetic_volume(2)]
    metadata = [(1.0, 0.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 0.0, 1.0)]
    device = torch.device("cpu")

    clean, lengths, _ = prepare_study_images(
        volumes, metadata, slices_per_series=12, device=device
    )
    first, augmented_lengths, _ = prepare_study_images(
        volumes,
        metadata,
        slices_per_series=12,
        device=device,
        augment_rng=np.random.default_rng([20260909, 1, 0]),
    )
    second, _, _ = prepare_study_images(
        volumes,
        metadata,
        slices_per_series=12,
        device=device,
        augment_rng=np.random.default_rng([20260909, 1, 0]),
    )

    assert lengths == augmented_lengths == (12, 12)
    assert first.shape == clean.shape
    assert torch.isfinite(first).all()
    assert torch.equal(first, second), "Augmentation must be seed-reproducible"
    assert not torch.equal(first, clean), "Augmentation must change the input"


def test_stage_config_rejects_a_nonboolean_augment_flag() -> None:
    with pytest.raises(ImagingTrainingError, match="augment must be a boolean"):
        StageConfig(
            name="gate",
            epochs=1,
            head_learning_rate=1e-4,
            backbone_learning_rate=1e-6,
            augment=1,  # type: ignore[arg-type]
        ).validate()


def test_augmentation_is_disabled_by_default() -> None:
    config = StageConfig(
        name="gate",
        epochs=1,
        head_learning_rate=1e-4,
        backbone_learning_rate=1e-6,
    )
    config.validate()

    assert config.augment is False
