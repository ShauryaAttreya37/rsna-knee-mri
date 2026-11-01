from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rsna.kaggle_package import (
    BOOTSTRAP,
    MODEL_NAME,
    WHEEL_DISTRIBUTIONS,
    build_package,
)
from rsna.submission import TARGET_COLUMNS


@pytest.fixture
def package_inputs(tmp_path: Path) -> dict[str, Path]:
    torch = pytest.importorskip("torch")
    paths = {
        name: tmp_path / name
        for name in (
            "project_root",
            "model_root",
            "dinov2_repo",
            "wheelhouse",
            "output_root",
        )
    }
    for name in (
        "src/rsna/__init__.py",
        "src/rsna/imaging_inference.py",
        "notebooks/06_imaging_inference.py",
    ):
        path = paths["project_root"] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic test source\n")
    for name in ("hubconf.py", "LICENSE", "dinov2/__init__.py"):
        path = paths["dinov2_repo"] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic test source\n")
    (paths["project_root"] / "kaggle.json").write_text("private sentinel")
    paths["wheelhouse"].mkdir()
    for distribution in WHEEL_DISTRIBUTIONS:
        (paths["wheelhouse"] / f"{distribution}-1.0-py3-none-any.whl").write_bytes(
            b"wheel fixture"
        )
    for fold in range(5):
        checkpoint = paths["model_root"] / f"gold_fold_{fold}" / "best.pt"
        checkpoint.parent.mkdir(parents=True)
        torch.save(
            {
                "format_version": 1,
                "model": MODEL_NAME,
                "target_columns": list(TARGET_COLUMNS),
                "stage_config": {
                    "name": f"gold_fold_{fold}",
                    "slices_per_series": 12,
                    "encoder_chunk_size": 64,
                },
                "source_sha256": {"folds": "a" * 64},
                "epoch": 2,
                "best_value": 0.8,
                "state_dict": torch.nn.Linear(3, len(TARGET_COLUMNS)).state_dict(),
                "optimizer_state_dict": {"private_training_state": True},
                "history": ["training-only"],
            },
            checkpoint,
        )
    return paths


def test_export_preserves_weights_and_sources_but_omits_training_state(package_inputs):
    import torch

    from rsna.imaging_training import load_checkpoint_weights

    originals = sorted(package_inputs["model_root"].rglob("*.pt"))
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in originals]
    manifest = build_package(**package_inputs)
    output = package_inputs["output_root"]

    assert len(manifest["checkpoints"]) == 5
    assert not (output / "kaggle.json").exists()
    assert hashes == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in originals
    ]
    for fold, original_path in enumerate(originals):
        original = torch.load(original_path, weights_only=True)
        exported_path = output / "models" / f"gold_fold_{fold}" / "best.pt"
        exported = torch.load(exported_path, weights_only=True)
        assert "optimizer_state_dict" not in exported
        assert "history" not in exported
        model = torch.nn.Linear(3, len(TARGET_COLUMNS))
        load_checkpoint_weights(model, exported_path, device="cpu")
        for key, tensor in model.state_dict().items():
            assert torch.equal(tensor, original["state_dict"][key])
    for relative, expected in manifest["files"].items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == expected
    assert json.loads((output / "rsna-package.json").read_text()) == manifest
    notebook = json.loads((output / "kernel.ipynb").read_text())
    assert notebook["nbformat"] == 4
    compile("".join(notebook["cells"][1]["source"]), "kernel.ipynb", "exec")


def test_missing_fold_fails_before_creating_output(package_inputs):
    checkpoint = package_inputs["model_root"] / "gold_fold_4" / "best.pt"
    checkpoint.rename(checkpoint.with_name("preserved.pt"))
    with pytest.raises(ValueError, match="regular file"):
        build_package(**package_inputs)
    assert not package_inputs["output_root"].exists()


@pytest.mark.parametrize(
    "failure", ["wrong_fold", "target_order", "source_drift", "nonfinite"]
)
def test_invalid_checkpoint_cannot_produce_complete_package(package_inputs, failure):
    import torch

    path = package_inputs["model_root"] / "gold_fold_4" / "best.pt"
    checkpoint = torch.load(path, weights_only=True)
    if failure == "wrong_fold":
        checkpoint["stage_config"]["name"] = "gold_fold_0"
    elif failure == "target_order":
        checkpoint["target_columns"].reverse()
    elif failure == "source_drift":
        checkpoint["source_sha256"]["folds"] = "b" * 64
    else:
        checkpoint["state_dict"]["weight"][0, 0] = float("nan")
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        build_package(**package_inputs)
    assert not (package_inputs["output_root"] / "rsna-package.json").exists()


def test_existing_output_is_preserved(package_inputs):
    output = package_inputs["output_root"]
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        build_package(**package_inputs)
    assert sentinel.read_text() == "keep"


def test_offline_bootstrap_is_valid_python():
    compile(BOOTSTRAP, "kaggle-bootstrap", "exec")


@pytest.mark.parametrize("corrupt", [False, True])
def test_bootstrap_checks_artifacts_then_autoruns_with_writable_outputs(
    tmp_path, monkeypatch, corrupt
):
    import os
    import runpy
    import subprocess
    import sys

    inputs = tmp_path / "input"
    bundle = inputs / "datasets" / "owner" / "bundle"
    bundle.mkdir(parents=True)
    source = bundle / "model-source.py"
    source.write_text("# verified source")
    manifest = {
        "format_version": 1,
        "files": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "wheel_requirements": ["pydicom==3.0.2"],
    }
    (bundle / "rsna-package.json").write_text(json.dumps(manifest))
    if corrupt:
        source.write_text("# altered source")
    competition = inputs / "competitions" / "rsna"
    (competition / "test_series").mkdir(parents=True)
    for name in ("sample_submission.csv", "test.csv", "test_series.csv"):
        (competition / name).touch()
    working = tmp_path / "working"
    monkeypatch.setenv("RSNA_KAGGLE_INPUT_ROOT", str(inputs))
    monkeypatch.setenv("RSNA_KAGGLE_WORKING_ROOT", str(working))
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: calls.append(args))
    monkeypatch.setattr(runpy, "run_path", lambda path, **kwargs: calls.append(path))
    if corrupt:
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            exec(BOOTSTRAP, {})  # noqa: S102 - execute our trusted notebook fixture
        assert calls == []
        return
    exec(BOOTSTRAP, {})  # noqa: S102 - execute our trusted notebook fixture
    assert "--no-index" in calls[0] and "--no-deps" in calls[0]
    assert calls[1] == str(bundle / "notebooks" / "06_imaging_inference.py")
    assert os.environ["RSNA_RUN_INFERENCE"] == "1"
    assert os.environ["RSNA_MODEL_ROOT"] == str(bundle / "models")
    assert os.environ["RSNA_COMPETITION_ROOT"] == str(competition)
    assert os.environ["RSNA_SUBMISSION_PATH"] == str(working / "submission.csv")
    assert os.environ["RSNA_REPORT_PATH"] == str(working / "imaging_v2_summary.json")
