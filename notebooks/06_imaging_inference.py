# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: RSNA (Training PyTorch CUDA)
#     language: python
#     name: rsna-train
# ---

# %% [markdown]
# # RSNA Knee: Offline Five-Checkpoint Imaging Inference
#
# This notebook decodes every test study once, runs all five fold-best models,
# rank-averages their per-target predictions, and writes a strictly validated
# `submission.csv`. It performs no training and requires no internet access.

# %%
import importlib.metadata
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(
    os.environ.get("RSNA_PROJECT_ROOT", "/home/shaurya/rsna-project")
).expanduser()
DATA_ROOT = Path(
    os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data")
).expanduser()
COMPETITION_ROOT = Path(
    os.environ.get("RSNA_COMPETITION_ROOT", DATA_ROOT / "raw" / "competition")
).expanduser()
MODEL_ROOT = Path(
    os.environ.get("RSNA_MODEL_ROOT", DATA_ROOT / "models" / "imaging_v2_public_labels")
).expanduser()
DINOV2_REPO = Path(
    os.environ.get(
        "RSNA_DINOV2_REPO",
        DATA_ROOT / "models" / "torch_hub" / "facebookresearch_dinov2_main",
    )
).expanduser()
SUBMISSION_PATH = Path(
    os.environ.get("RSNA_SUBMISSION_PATH", PROJECT_ROOT / "submission.csv")
).expanduser()
REPORT_PATH = Path(
    os.environ.get(
        "RSNA_REPORT_PATH",
        PROJECT_ROOT / "reports" / "inference" / "imaging_v2_summary.json",
    )
).expanduser()

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsna.imaging_inference import run_test_ensemble
from rsna.imaging_training import build_multiseries_model, load_checkpoint_weights
from rsna.submission import generate_submission

assert torch.cuda.is_available(), "Select a CUDA kernel before continuing"
DEVICE = torch.device("cuda")
AMP_DTYPE = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
torch.set_float32_matmul_precision("high")

runtime_versions = {
    distribution: importlib.metadata.version(distribution)
    for distribution in ("numpy", "pydicom", "torch")
}
# Kaggle supplies CUDA PyTorch and NumPy. Record its runtime for the hardware
# gate instead of requiring an unrelated VM patch version before inference.
print(json.dumps({"runtime_versions": runtime_versions}, indent=2))

for path in (
    COMPETITION_ROOT / "test.csv",
    COMPETITION_ROOT / "test_series.csv",
    COMPETITION_ROOT / "sample_submission.csv",
    COMPETITION_ROOT / "test_series",
    DINOV2_REPO,
):
    assert path.exists(), f"Required offline input is missing: {path}"
for fold in range(5):
    checkpoint = MODEL_ROOT / f"gold_fold_{fold}" / "best.pt"
    assert checkpoint.is_file(), f"Checkpoint is missing: {checkpoint}"

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Mixed precision: {AMP_DTYPE}")
print(f"Competition root: {COMPETITION_ROOT}")

# %% [markdown]
# ## Load five fold-best models from offline artifacts

# %%
models = []
checkpoint_summaries = []
for fold in range(5):
    backbone = torch.hub.load(
        str(DINOV2_REPO),
        "dinov2_vits14",
        source="local",
        pretrained=False,
    )
    assert int(backbone.embed_dim) == 384
    model = build_multiseries_model(backbone, int(backbone.embed_dim)).to(DEVICE)
    checkpoint = load_checkpoint_weights(
        model,
        MODEL_ROOT / f"gold_fold_{fold}" / "best.pt",
        device=DEVICE,
    )
    assert checkpoint["stage_config"]["name"] == f"gold_fold_{fold}"
    assert checkpoint["stage_config"]["slices_per_series"] == 12
    assert checkpoint["stage_config"]["encoder_chunk_size"] == 64
    models.append(model.eval())
    checkpoint_summaries.append(
        {
            "fold": fold,
            "epoch": checkpoint["epoch"],
            "best_value": checkpoint["best_value"],
        }
    )
    print(f"Loaded fold {fold}: {checkpoint_summaries[-1]}")

assert len(models) == 5

# %% [markdown]
# ## Run offline inference and write submission
#
# Set guard only when paths and GPU are ready. Rank averaging neutralizes the large
# score-scale drift observed between fold checkpoints. The competition metric is macro
# ROC AUC, so only per-target ranking affects the score.

# %%
RUN_INFERENCE = os.environ.get("RSNA_RUN_INFERENCE", "0") == "1"

if RUN_INFERENCE:
    result = run_test_ensemble(
        models,
        COMPETITION_ROOT,
        device=DEVICE,
        amp_dtype=AMP_DTYPE,
        sampled_slices=16,
        slices_per_series=12,
        output_size=224,
        encoder_chunk_size=64,
    )
    validation = generate_submission(
        COMPETITION_ROOT / "test.csv",
        COMPETITION_ROOT / "sample_submission.csv",
        SUBMISSION_PATH,
        result.study_uids,
        result.rank_average,
    )
    report = {
        "status": "complete",
        "generated_at": datetime.now(UTC).isoformat(),
        "offline": True,
        "runtime_versions": runtime_versions,
        "privacy": "Identifier-free aggregate inference report only.",
        "checkpoints": checkpoint_summaries,
        "runtime": result.aggregate_summary(),
        "submission_validation": validation.to_dict(),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"Submission: {SUBMISSION_PATH.resolve()}")
else:
    print("Guard active: set RUN_INFERENCE=True, then run this cell once.")
