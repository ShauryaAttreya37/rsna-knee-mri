"""Export the promoted five-fold model as a self-contained offline Kaggle input."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

from rsna.submission import TARGET_COLUMNS

MODEL_NAME = "dinov2_vits14_target_specific_multiseries"
CHECKPOINT_KEYS = (
    "format_version",
    "model",
    "target_columns",
    "stage_config",
    "source_sha256",
    "epoch",
    "best_value",
    "state_dict",
)
WHEEL_DISTRIBUTIONS = (
    "pydicom",
    "pylibjpeg",
    "pylibjpeg_libjpeg",
    "pylibjpeg_openjpeg",
)

# Use the same notebook source as the VM gate. Kaggle's input is read-only;
# generated reports, bytecode, and the submission belong in /kaggle/working.
BOOTSTRAP = """import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

input_root = Path(os.environ.get("RSNA_KAGGLE_INPUT_ROOT", "/kaggle/input"))
manifests = list(input_root.rglob("rsna-package.json"))
if len(manifests) != 1:
    raise RuntimeError("Attach exactly one RSNA imaging-v2 package")
bundle = manifests[0].parent
manifest = json.loads(manifests[0].read_text())
if manifest["format_version"] != 1:
    raise RuntimeError("Unsupported RSNA package format")
for relative, expected in manifest["files"].items():
    path = (bundle / relative).resolve()
    if not path.is_relative_to(bundle.resolve()):
        raise RuntimeError("Package path escapes its input directory")
    with path.open("rb") as stream:
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
    if observed != expected:
        raise RuntimeError(f"Package checksum mismatch: {relative}")

subprocess.run([
    sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
    "--disable-pip-version-check", "--find-links", str(bundle / "wheels"),
    *manifest["wheel_requirements"],
], check=True)

competition_roots = [
    path.parent for path in input_root.rglob("sample_submission.csv")
    if (path.parent / "test.csv").is_file()
    and (path.parent / "test_series.csv").is_file()
    and (path.parent / "test_series").is_dir()
]
if len(competition_roots) != 1:
    raise RuntimeError("Attach exactly one RSNA competition input")
working = Path(os.environ.get("RSNA_KAGGLE_WORKING_ROOT", "/kaggle/working"))
working.mkdir(parents=True, exist_ok=True)
sys.dont_write_bytecode = True
os.environ.update({
    "RSNA_PROJECT_ROOT": str(bundle),
    "RSNA_MODEL_ROOT": str(bundle / "models"),
    "RSNA_DINOV2_REPO": str(bundle / "dinov2_repo"),
    "RSNA_COMPETITION_ROOT": str(competition_roots[0]),
    "RSNA_SUBMISSION_PATH": str(working / "submission.csv"),
    "RSNA_REPORT_PATH": str(working / "imaging_v2_summary.json"),
    "RSNA_RUN_INFERENCE": "1",
    "XFORMERS_DISABLED": "1",
})
runpy.run_path(str(bundle / "notebooks" / "06_imaging_inference.py"), run_name="__main__")
"""


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_checkpoint(checkpoint: dict[str, Any], fold: int) -> None:
    if any(key not in checkpoint for key in CHECKPOINT_KEYS):
        raise ValueError(f"Fold {fold}: incomplete checkpoint")
    if checkpoint["format_version"] != 1 or checkpoint["model"] != MODEL_NAME:
        raise ValueError(f"Fold {fold}: incompatible checkpoint architecture")
    if checkpoint["target_columns"] != list(TARGET_COLUMNS):
        raise ValueError(f"Fold {fold}: incompatible target order")
    config = checkpoint["stage_config"]
    if config["name"] != f"gold_fold_{fold}":
        raise ValueError(f"Fold {fold}: wrong held-out fold")
    if config["slices_per_series"] != 12 or config["encoder_chunk_size"] != 64:
        raise ValueError(f"Fold {fold}: incompatible inference configuration")
    if not math.isfinite(float(checkpoint["best_value"])):
        raise ValueError(f"Fold {fold}: nonfinite validation score")
    if not checkpoint["state_dict"]:
        raise ValueError(f"Fold {fold}: empty model weights")


def build_package(
    project_root: Path,
    model_root: Path,
    dinov2_repo: Path,
    wheelhouse: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Copy code and strip optimizer state; never alter original checkpoints.

    A new destination is required. On failure a partial export is retained for
    inspection, without a completion manifest. No credentials or data are copied.
    """
    import torch

    project_root, model_root, dinov2_repo, wheelhouse, output_root = (
        path.expanduser().resolve()
        for path in (project_root, model_root, dinov2_repo, wheelhouse, output_root)
    )
    if output_root.exists():
        raise ValueError("Package destination already exists; choose a new directory")
    # Explicit source selections prevent data, tokens, logs and cached hub weights
    # from crossing into the competition input.
    sources = {
        "notebooks/06_imaging_inference.py": project_root
        / "notebooks"
        / "06_imaging_inference.py",
        **{
            f"src/rsna/{path.name}": path
            for path in sorted((project_root / "src" / "rsna").glob("*.py"))
        },
        "dinov2_repo/hubconf.py": dinov2_repo / "hubconf.py",
        **{
            f"dinov2_repo/{path.name}": path
            for path in sorted(dinov2_repo.glob("LICENSE*"))
        },
        **{
            f"dinov2_repo/{path.relative_to(dinov2_repo).as_posix()}": path
            for path in sorted((dinov2_repo / "dinov2").rglob("*.py"))
        },
    }
    for required in (
        "src/rsna/imaging_inference.py",
        "dinov2_repo/dinov2/__init__.py",
        "dinov2_repo/LICENSE",
    ):
        if required not in sources:
            raise ValueError(f"Missing required source: {required}")
    wheel_requirements = []
    for distribution in WHEEL_DISTRIBUTIONS:
        wheels = sorted(wheelhouse.glob(f"{distribution}-*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"Expected exactly one {distribution} wheel")
        wheel = wheels[0]
        wheel_requirements.append(f"{distribution}=={wheel.name.split('-')[1]}")
        sources[f"wheels/{wheel.name}"] = wheel
    checkpoints = [model_root / f"gold_fold_{fold}" / "best.pt" for fold in range(5)]
    for path in [*sources.values(), *checkpoints]:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Required input must be a regular file: {path}")

    output_root.mkdir(parents=True)
    for relative, source in sources.items():
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    summaries = []
    reference_sources = None
    for fold, path in enumerate(checkpoints):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        _validate_checkpoint(checkpoint, fold)
        if reference_sources is None:
            reference_sources = checkpoint["source_sha256"]
        elif checkpoint["source_sha256"] != reference_sources:
            raise ValueError("Fold checkpoints were trained with different source data")
        for tensor in checkpoint["state_dict"].values():
            if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                raise ValueError(f"Fold {fold}: invalid model weight tensor")
        destination = output_root / "models" / f"gold_fold_{fold}" / "best.pt"
        destination.parent.mkdir(parents=True)
        torch.save({key: checkpoint[key] for key in CHECKPOINT_KEYS}, destination)
        summaries.append(
            {
                "fold": fold,
                "epoch": int(checkpoint["epoch"]),
                "best_value": float(checkpoint["best_value"]),
                "original_sha256": _sha256(path),
                "original_bytes": path.stat().st_size,
                "inference_bytes": destination.stat().st_size,
            }
        )
        del checkpoint

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# RSNA Knee: Imaging v2\n",
                    "Offline five-fold DINOv2 inference. Enable GPU and disable internet.\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": BOOTSTRAP.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    (output_root / "kernel.ipynb").write_text(json.dumps(notebook, indent=2) + "\n")
    manifest = {
        "format_version": 1,
        "model": MODEL_NAME,
        "target_columns": list(TARGET_COLUMNS),
        "wheel_requirements": wheel_requirements,
        "checkpoints": summaries,
        "files": {
            path.relative_to(output_root).as_posix(): _sha256(path)
            for path in sorted(output_root.rglob("*"))
            if path.is_file()
        },
    }
    (output_root / "rsna-package.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--dinov2-repo", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_package(**vars(args))
    print(
        json.dumps(
            {"status": "complete", "checkpoints": manifest["checkpoints"]}, indent=2
        )
    )
