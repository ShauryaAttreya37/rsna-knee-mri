"""Guard the controlled sampling/augmentation gate against silent drift."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT = PROJECT_ROOT / "scripts" / "17_run_geometry_gate.py"
V2_REPORTS = PROJECT_ROOT / "reports" / "training"


@pytest.fixture(scope="module")
def gate():
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("geometry_gate", GATE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _v2_configuration(fold: int) -> dict[str, object]:
    report = json.loads(
        (V2_REPORTS / f"imaging_v2_public_labels_fold_{fold}_summary.json").read_text()
    )
    return report["configuration"]


def test_baselines_match_the_committed_imaging_v2_reports(gate) -> None:
    for fold, expected in gate.BASELINE_MACRO_AUC.items():
        report = json.loads(
            (
                V2_REPORTS / f"imaging_v2_public_labels_fold_{fold}_summary.json"
            ).read_text()
        )
        assert report["configuration"]["name"] == f"gold_fold_{fold}"
        assert report["best_value"] == expected


def test_every_arm_changes_only_its_own_setting(gate) -> None:
    baseline = _v2_configuration(0)
    for arm, settings in gate.ARMS.items():
        config = asdict(gate._gold_config(0, "bfloat16", arm))
        gate._gold_config(0, "bfloat16", arm).validate()
        # `augment` is new, so it is absent from the imaging-v2 configuration.
        differences = {
            key
            for key, value in config.items()
            if key in baseline and baseline[key] != value
        }
        assert differences <= set(settings), (
            f"Arm {arm} changed unrelated settings: {differences - set(settings)}"
        )
        for key, value in settings.items():
            assert config[key] == value


def test_arms_are_mutually_exclusive_single_variable_experiments(gate) -> None:
    assert gate.ARMS["slices16"] == {"slices_per_series": 16, "augment": False}
    assert gate.ARMS["augment"] == {"slices_per_series": 12, "augment": True}
    baseline = _v2_configuration(0)
    assert baseline["slices_per_series"] == 12


def test_fold_seeds_match_imaging_v2(gate) -> None:
    for fold in range(5):
        expected = _v2_configuration(fold)["seed"]
        for arm in gate.ARMS:
            assert asdict(gate._gold_config(fold, "bfloat16", arm))["seed"] == expected


def test_every_arm_stays_within_the_offline_inference_slice_budget(gate) -> None:
    import inspect

    from rsna.imaging_inference import run_test_ensemble

    # Offline inference rejects slices_per_series above the decoded
    # sampled_slices, so no arm may exceed what the Kaggle path decodes.
    decoded = inspect.signature(run_test_ensemble).parameters["sampled_slices"].default
    for arm, settings in gate.ARMS.items():
        assert settings["slices_per_series"] <= decoded, (
            f"Arm {arm} cannot be served by the offline inference path"
        )
