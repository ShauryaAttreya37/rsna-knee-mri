"""Run the controlled sampling/augmentation gate on CUDA.

Both arms reuse the promoted imaging-v2 weak-pretraining checkpoint and change
only the gold fine-tuning stage, so each fold result is directly comparable to
the imaging-v2 fold result without spending another weak-pretraining run.

Arms:
  slices16  slices_per_series 12 -> 16, consuming the full cached depth.
  augment   training-only slice-sampling and intensity jitter at 12 slices.

Neither arm re-reads raw DICOMs, so the existing tensor cache is untouched.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsna.imaging_training import (
    StageConfig,
    build_multiseries_model,
    build_training_plan,
    configure_backbone,
    load_checkpoint_weights,
    run_training_stage,
)

# Promoted imaging-v2 fold-best macro AUC, for a like-for-like comparison.
BASELINE_MACRO_AUC = {
    0: 0.8754464285714286,
    1: 0.7981867283950618,
    2: 0.7355461860670194,
    3: 0.8496693121693122,
    4: 0.8034060846560847,
}

ARMS = {
    "slices16": {"slices_per_series": 16, "augment": False},
    "augment": {"slices_per_series": 12, "augment": True},
}


def _build_model(torch_hub_root: Path, device: torch.device):
    torch_hub_root.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_hub_root))
    torch.manual_seed(20260905)
    torch.cuda.manual_seed_all(20260905)
    backbone = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
        pretrained=True,
        trust_repo=True,
    )
    if int(backbone.embed_dim) != 384:
        raise RuntimeError("Expected DINOv2 ViT-S/14 feature dimension 384")
    model = build_multiseries_model(backbone, 384)
    configure_backbone(model, unfreeze_last_blocks=2)
    model.to(device)
    return model


def _gold_config(fold: int, amp_dtype: str, arm: str) -> StageConfig:
    """Match the imaging-v2 gold configuration except for the tested arm."""
    return StageConfig(
        name=f"gold_fold_{fold}",
        epochs=30,
        head_learning_rate=1e-4,
        backbone_learning_rate=3e-6,
        weight_decay=0.01,
        accumulation_steps=2,
        encoder_chunk_size=64,
        amp_dtype=amp_dtype,
        gradient_clip_norm=1.0,
        checkpoint_every_studies=20,
        patience=7,
        seed=20260905 + fold,
        selection_metric="macro_auc",
        **ARMS[arm],
    )


def main() -> None:
    data_root = Path(os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 3])
    parser.add_argument(
        "--weak-labels",
        type=Path,
        default=data_root
        / "manifests"
        / "weak_labels_v2_public_ensemble"
        / "weak_labels.csv",
    )
    parser.add_argument(
        "--weak-checkpoint",
        type=Path,
        default=data_root
        / "models"
        / "imaging_v2_public_labels"
        / "weak_pretrain"
        / "best.pt",
    )
    parser.add_argument("--model-root", type=Path, default=None)
    args = parser.parse_args()
    if not args.folds or any(fold not in range(5) for fold in args.folds):
        parser.error("folds must be one or more values from 0 through 4")
    if len(set(args.folds)) != len(args.folds):
        parser.error("folds must be unique")
    model_root = args.model_root or data_root / "models" / f"imaging_v3_{args.arm}"
    if not args.weak_checkpoint.is_file():
        parser.error(f"Weak checkpoint is missing: {args.weak_checkpoint}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    amp_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    torch.set_float32_matmul_precision("high")
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "amp_dtype": amp_dtype,
                "arm": args.arm,
                "arm_settings": ARMS[args.arm],
                "folds": args.folds,
                "model_root": str(model_root),
                "weak_checkpoint": str(args.weak_checkpoint),
            },
            indent=2,
        ),
        flush=True,
    )

    common = {
        "competition_root": data_root / "raw" / "competition",
        "cache_root": data_root / "processed" / "training_cache_v1",
        "weak_labels_path": args.weak_labels,
        "folds_path": data_root / "manifests" / "gold_folds_v1" / "gold_folds.csv",
        "weak_validation_fraction": 0.10,
    }

    results: dict[str, dict[str, float]] = {}
    for fold in args.folds:
        plan = build_training_plan(selected_fold=fold, **common)
        model = _build_model(data_root / "models" / "torch_hub", device)
        weak = load_checkpoint_weights(model, args.weak_checkpoint, device=device)
        if weak["stage_config"]["name"] != "weak_pretrain":
            raise RuntimeError("Expected a weak-pretraining checkpoint")
        configure_backbone(model, unfreeze_last_blocks=2)
        report = run_training_stage(
            model,
            plan.gold_train,
            plan.gold_validation,
            device=device,
            output_root=model_root,
            report_output=PROJECT_ROOT
            / "reports"
            / "training"
            / f"imaging_v3_{args.arm}_fold_{fold}_summary.json",
            project_root=PROJECT_ROOT,
            competition_root=common["competition_root"],
            config=_gold_config(fold, amp_dtype, args.arm),
            source_sha256=plan.source_sha256,
            validation_targets_are_binary=True,
            resume=True,
        )
        observed = float(report["best_value"])
        baseline = BASELINE_MACRO_AUC[fold]
        results[str(fold)] = {
            "best_macro_auc": observed,
            "baseline_macro_auc": baseline,
            "delta": observed - baseline,
        }
        print(
            f"fold {fold} complete: best_macro_auc={observed:.6f} "
            f"baseline={baseline:.6f} delta={observed - baseline:+.6f}",
            flush=True,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    deltas = [entry["delta"] for entry in results.values()]
    summary = {
        "status": "complete",
        "arm": args.arm,
        "arm_settings": ARMS[args.arm],
        "privacy": "Identifier-free aggregate gate metrics only.",
        "folds": results,
        "mean_delta": sum(deltas) / len(deltas),
        "improved_every_fold": all(delta > 0.0 for delta in deltas),
    }
    output = PROJECT_ROOT / "reports" / "training" / f"imaging_v3_{args.arm}_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
