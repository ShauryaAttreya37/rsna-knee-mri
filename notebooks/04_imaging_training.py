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
# # RSNA Knee: Leak-Free Weak Pretraining and Gold-Fold Fine-Tuning
#
# This is the first real imaging-training notebook. It deliberately keeps execution
# interactive: each expensive stage has a `RUN_... = False` guard that must be changed
# immediately before running that cell.
#
# Data boundaries:
#
# - Weak pretraining uses only the 4,349 studies with no gold labels.
# - A deterministic 10% subset of those studies selects the weak checkpoint.
# - Gold fine-tuning uses four folds; the selected fifth fold is validation-only.
# - The selected gold validation images never appear in weak pretraining.
# - Checkpoints and identifier-bearing artifacts stay outside Git.
#
# Start with `SELECTED_FOLD = 0`. Use a fresh kernel for each later fold.

# %%
import importlib.metadata
import json
import os
import shutil
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

SELECTED_FOLD = 0
assert SELECTED_FOLD in range(5)

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsna.imaging_training import (
    StageConfig,
    build_multiseries_model,
    build_training_plan,
    configure_backbone,
    evaluate_model,
    load_checkpoint_weights,
    load_study_tensors,
    run_training_stage,
)
from rsna.submission import TARGET_COLUMNS

assert torch.cuda.is_available(), "Select the rsna-train CUDA kernel before continuing"
DEVICE = torch.device("cuda")
AMP_DTYPE = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
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

assert DATA_ROOT.is_dir(), f"Data root is missing: {DATA_ROOT}"
free_gib = shutil.disk_usage(DATA_ROOT).free / 1024**3
assert free_gib >= 20.0, f"Only {free_gib:.1f} GiB free on the training volume"
cuda_probe = torch.arange(1024, device=DEVICE).square().sum()
assert torch.isfinite(cuda_probe)
torch.cuda.synchronize()

print(f"PyTorch       : {torch.__version__}")
print(f"CUDA          : {torch.version.cuda}")
print(f"GPU           : {torch.cuda.get_device_name(0)}")
print(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
print(f"Mixed precision: {AMP_DTYPE}")
print(f"Free storage   : {free_gib:.1f} GiB")
print(f"Selected fold : {SELECTED_FOLD}")

# %% [markdown]
# ## 1. Verify completed upstream artifacts and construct disjoint cohorts
#
# This reads manifests and hashes metadata files. It does not read raw DICOMs or begin
# optimization. Any count, schema, cache, or membership mismatch fails closed.

# %%
required_reports = {
    "cache": PROJECT_ROOT / "reports" / "preprocessing_cache" / "summary.json",
    "cache migration audit": PROJECT_ROOT
    / "reports"
    / "preprocessing_cache"
    / "migration_audit.json",
    "weak supervision": PROJECT_ROOT
    / "reports"
    / "report_supervision"
    / "weak_supervision_summary.json",
    "gold overfit": PROJECT_ROOT / "reports" / "training" / "gold_overfit_summary.json",
}
required_status = {
    "cache": "complete",
    "cache migration audit": "pass",
    "weak supervision": "complete",
    "gold overfit": "pass",
}
for name, path in required_reports.items():
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == required_status[name], f"{name} is not ready: {path}"
    print(f"[PASS] {name}: {payload['status']}")

plan = build_training_plan(
    competition_root=COMPETITION_ROOT,
    cache_root=FULL_CACHE_ROOT,
    weak_labels_path=WEAK_LABELS_PATH,
    folds_path=FOLDS_PATH,
    selected_fold=SELECTED_FOLD,
    weak_validation_fraction=0.10,
)
print(json.dumps(plan.aggregate_summary(), indent=2))

assert plan.weak_only_studies == 4349
assert len(plan.weak_train) + len(plan.weak_validation) == 4349
assert len(plan.gold_train) + len(plan.gold_validation) == 58
assert len(plan.gold_validation) in {11, 12}

# %% [markdown]
# ## 2. Load the official DINOv2 ViT-S/14 backbone
#
# The first execution downloads Meta's official repository and pretrained weights into
# the durable data disk. Later executions reuse the Torch Hub cache. Training cells do
# not perform this download implicitly.

# %%
TORCH_HUB_ROOT.mkdir(parents=True, exist_ok=True)
torch.hub.set_dir(str(TORCH_HUB_ROOT))
MODEL_INIT_SEED = 20260905
torch.manual_seed(MODEL_INIT_SEED)
torch.cuda.manual_seed_all(MODEL_INIT_SEED)

backbone = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14",
    pretrained=True,
    trust_repo=True,
)
feature_dimension = int(backbone.embed_dim)
assert feature_dimension == 384

model = build_multiseries_model(backbone, feature_dimension)
parameter_summary = configure_backbone(model, unfreeze_last_blocks=2)
model.to(DEVICE)
print(json.dumps(parameter_summary, indent=2))

# %% [markdown]
# ## 3. One-study forward/backward smoke test
#
# This checks tensor layout, 2.5D triplets, DINO normalization, attention dimensions,
# gradients, and GPU memory before the long run. It does not update model weights.

# %%
model.train()
smoke_record = max(plan.weak_train, key=lambda record: len(record.series))
images, lengths, metadata, targets, _weights = load_study_tensors(
    smoke_record, slices_per_series=8, device=DEVICE
)
with torch.autocast(device_type="cuda", dtype=getattr(torch, AMP_DTYPE)):
    logits = model(images, lengths, metadata, encoder_chunk_size=64)
    smoke_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
smoke_loss.backward()

assert logits.shape == (len(TARGET_COLUMNS),)
assert torch.isfinite(logits).all()
assert any(
    parameter.grad is not None and torch.isfinite(parameter.grad).all()
    for parameter in model.parameters()
    if parameter.requires_grad
)
model.zero_grad(set_to_none=True)
print(f"Smoke loss: {smoke_loss.item():.6f}")
print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")

# %% [markdown]
# ## 4. Weak-label pretraining — 4,349 non-gold studies
#
# Recommended first run: three epochs, eight 2.5D positions per series, final two DINO
# blocks trainable, and gradient accumulation over four studies. The best checkpoint is
# selected only by loss on the deterministic weak validation partition.
#
# Change `RUN_WEAK_PRETRAIN` to `True` only when you are ready. The cell resumes from
# `latest.pt` after interruption and refuses to resume if code, data, or settings changed.

# %%
WEAK_CONFIG = StageConfig(
    name="weak_pretrain",
    epochs=3,
    head_learning_rate=3e-4,
    backbone_learning_rate=1e-5,
    weight_decay=0.01,
    accumulation_steps=4,
    slices_per_series=8,
    encoder_chunk_size=64,
    amp_dtype=AMP_DTYPE,
    gradient_clip_norm=1.0,
    checkpoint_every_studies=500,
    patience=None,
    seed=20260905,
    selection_metric="loss",
)

RUN_WEAK_PRETRAIN = False
if RUN_WEAK_PRETRAIN:
    weak_report = run_training_stage(
        model,
        plan.weak_train,
        plan.weak_validation,
        device=DEVICE,
        output_root=MODEL_ROOT,
        report_output=PROJECT_ROOT
        / "reports"
        / "training"
        / "weak_pretrain_summary.json",
        project_root=PROJECT_ROOT,
        competition_root=COMPETITION_ROOT,
        config=WEAK_CONFIG,
        source_sha256=plan.source_sha256,
        validation_targets_are_binary=False,
        resume=True,
    )
    print(json.dumps(weak_report, indent=2))
else:
    print("Guard active: set RUN_WEAK_PRETRAIN=True, then run this cell to begin.")

# %% [markdown]
# ## 5. Load the best weak checkpoint before gold fine-tuning
#
# This is mandatory even immediately after pretraining: the in-memory model contains the
# last epoch, which is not necessarily the best weak-validation epoch.

# %%
WEAK_BEST = MODEL_ROOT / "weak_pretrain" / "best.pt"
assert WEAK_BEST.is_file(), "Complete weak pretraining before fine-tuning"
weak_checkpoint = load_checkpoint_weights(model, WEAK_BEST, device=DEVICE)
parameter_summary = configure_backbone(model, unfreeze_last_blocks=2)
print(json.dumps({"checkpoint": weak_checkpoint, "parameters": parameter_summary}, indent=2))

# %% [markdown]
# ## 6. Fine-tune on four gold folds; validate on the selected fold
#
# Use a fresh kernel when changing `SELECTED_FOLD`. The held-out fold is used only for
# validation and checkpoint selection. No held-out images were used during weak training.

# %%
GOLD_CONFIG = StageConfig(
    name=f"gold_fold_{SELECTED_FOLD}",
    epochs=30,
    head_learning_rate=1e-4,
    backbone_learning_rate=3e-6,
    weight_decay=0.01,
    accumulation_steps=2,
    slices_per_series=12,
    encoder_chunk_size=64,
    amp_dtype=AMP_DTYPE,
    gradient_clip_norm=1.0,
    checkpoint_every_studies=20,
    patience=7,
    seed=20260905 + SELECTED_FOLD,
    selection_metric="macro_auc",
)

RUN_GOLD_FINETUNE = False
if RUN_GOLD_FINETUNE:
    gold_report = run_training_stage(
        model,
        plan.gold_train,
        plan.gold_validation,
        device=DEVICE,
        output_root=MODEL_ROOT,
        report_output=PROJECT_ROOT
        / "reports"
        / "training"
        / f"gold_fold_{SELECTED_FOLD}_summary.json",
        project_root=PROJECT_ROOT,
        competition_root=COMPETITION_ROOT,
        config=GOLD_CONFIG,
        source_sha256=plan.source_sha256,
        validation_targets_are_binary=True,
        resume=True,
    )
    print(json.dumps(gold_report, indent=2))
else:
    print("Guard active: set RUN_GOLD_FINETUNE=True, then run this cell when ready.")

# %% [markdown]
# ## 7. Re-evaluate the best fold checkpoint
#
# This aggregate output is safe to inspect and commit. Per-study predictions and study
# identifiers are intentionally not written by this notebook.

# %%
GOLD_BEST = MODEL_ROOT / f"gold_fold_{SELECTED_FOLD}" / "best.pt"
assert GOLD_BEST.is_file(), "Complete gold fine-tuning before evaluation"
best_checkpoint = load_checkpoint_weights(model, GOLD_BEST, device=DEVICE)
best_metrics = evaluate_model(
    model,
    plan.gold_validation,
    device=DEVICE,
    slices_per_series=GOLD_CONFIG.slices_per_series,
    encoder_chunk_size=GOLD_CONFIG.encoder_chunk_size,
    amp_dtype=GOLD_CONFIG.amp_dtype,
    binary_targets=True,
)
print(json.dumps({"checkpoint": best_checkpoint, "metrics": best_metrics}, indent=2))

# %% [markdown]
# ## 8. Training curves

# %%
import matplotlib.pyplot as plt

report_path = (
    PROJECT_ROOT / "reports" / "training" / f"gold_fold_{SELECTED_FOLD}_summary.json"
)
report = json.loads(report_path.read_text(encoding="utf-8"))
history = report["history"]
epochs = [row["epoch"] for row in history]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(epochs, [row["training_loss"] for row in history], marker="o")
axes[0].set(title="Gold fine-tuning loss", xlabel="Epoch", ylabel="Training BCE")
axes[0].grid(alpha=0.3)
axes[1].plot(
    epochs,
    [row["validation"]["macro_auc"] for row in history],
    marker="o",
)
axes[1].set(title="Held-out gold macro AUC", xlabel="Epoch", ylabel="Macro AUC")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()
