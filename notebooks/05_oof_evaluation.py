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
# # RSNA Knee: Pooled Out-of-Fold Evaluation
#
# This notebook runs each fold-best checkpoint only on its untouched gold fold.
# It writes identifier-bearing OOF rows outside Git and an identifier-free aggregate
# report inside the project. No optimization occurs.

# %%
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(
    os.environ.get("RSNA_PROJECT_ROOT", "/home/shaurya/rsna-project")
).expanduser()
DATA_ROOT = Path(os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data")).expanduser()
COMPETITION_ROOT = DATA_ROOT / "raw" / "competition"
FULL_CACHE_ROOT = DATA_ROOT / "processed" / "training_cache_v1"
WEAK_LABELS_PATH = DATA_ROOT / "manifests" / "weak_labels_v1" / "weak_labels.csv"
FOLDS_PATH = DATA_ROOT / "manifests" / "gold_folds_v1" / "gold_folds.csv"
MODEL_ROOT = DATA_ROOT / "models" / "imaging_v1"
TORCH_HUB_ROOT = DATA_ROOT / "models" / "torch_hub"
OOF_ROOT = DATA_ROOT / "manifests" / "imaging_v1_oof"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsna.imaging_training import (
    build_multiseries_model,
    build_training_plan,
    load_checkpoint_weights,
    predict_model,
)
from rsna.oof_evaluation import combine_oof_predictions, write_oof_outputs

assert torch.cuda.is_available(), "Select the rsna-train CUDA kernel before continuing"
DEVICE = torch.device("cuda")
torch.set_float32_matmul_precision("high")

expected_versions = {
    "numpy": "2.5.2",
    "pydicom": "3.0.2",
    "scikit-learn": "1.8.0",
    "torch": "2.7.0",
}
for distribution, expected in expected_versions.items():
    observed = importlib.metadata.version(distribution).split("+", maxsplit=1)[0]
    assert observed == expected, f"{distribution}: expected {expected}, found {observed}"

for path in (
    COMPETITION_ROOT / "train.csv",
    FULL_CACHE_ROOT / "cache_manifest.csv",
    WEAK_LABELS_PATH,
    FOLDS_PATH,
):
    assert path.is_file(), f"Required input is missing: {path}"
for fold in range(5):
    checkpoint_path = MODEL_ROOT / f"gold_fold_{fold}" / "best.pt"
    assert checkpoint_path.is_file(), f"Fold checkpoint is missing: {checkpoint_path}"

print(f"GPU: {torch.cuda.get_device_name(0)}")
print("Inputs: five fold-best checkpoints and complete gold cache")

# %% [markdown]
# ## Build model
#
# `pretrained=False` avoids fetching separate weights. Each checkpoint contains the
# complete backbone and head state.

# %%
TORCH_HUB_ROOT.mkdir(parents=True, exist_ok=True)
torch.hub.set_dir(str(TORCH_HUB_ROOT))
backbone = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14",
    pretrained=False,
    trust_repo=True,
)
assert int(backbone.embed_dim) == 384
model = build_multiseries_model(backbone, int(backbone.embed_dim)).to(DEVICE)

# %% [markdown]
# ## Run five held-out folds
#
# Set guard only when ready. Expected runtime is one inference pass over 58 studies.
# No gradients, optimizer, or training checkpoint writes occur.

# %%
RUN_OOF_INFERENCE = False

if RUN_OOF_INFERENCE:
    fold_predictions = {}
    fold_metrics = {}
    for fold in range(5):
        plan = build_training_plan(
            competition_root=COMPETITION_ROOT,
            cache_root=FULL_CACHE_ROOT,
            weak_labels_path=WEAK_LABELS_PATH,
            folds_path=FOLDS_PATH,
            selected_fold=fold,
            weak_validation_fraction=0.10,
        )
        checkpoint = load_checkpoint_weights(
            model,
            MODEL_ROOT / f"gold_fold_{fold}" / "best.pt",
            device=DEVICE,
        )
        assert checkpoint["stage_config"]["name"] == f"gold_fold_{fold}"
        assert checkpoint["source_sha256"] == plan.source_sha256
        config = checkpoint["stage_config"]
        predictions = predict_model(
            model,
            plan.gold_validation,
            device=DEVICE,
            slices_per_series=int(config["slices_per_series"]),
            encoder_chunk_size=int(config["encoder_chunk_size"]),
            amp_dtype=str(config["amp_dtype"]),
        )
        fold_predictions[fold] = predictions
        fold_metrics[fold] = {
            "studies": len(predictions.study_uids),
            "loss": predictions.mean_loss,
        }
        print(f"Fold {fold}: {fold_metrics[fold]}")

    pooled = combine_oof_predictions(fold_predictions)
    summary = write_oof_outputs(
        pooled,
        predictions_csv=OOF_ROOT / "oof_predictions.csv",
        aggregate_report_json=PROJECT_ROOT
        / "reports"
        / "training"
        / "imaging_v1_oof_summary.json",
    )
    assert summary["studies"] == 58
    print(json.dumps(summary, indent=2))
else:
    print("Guard active: set RUN_OOF_INFERENCE=True, then run this cell once.")
