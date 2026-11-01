"""Run the controlled imaging-v2 weak-label gate on CUDA."""

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
    summary = configure_backbone(model, unfreeze_last_blocks=2)
    model.to(device)
    print(json.dumps(summary, indent=2), flush=True)
    return model


def _gold_config(fold: int, amp_dtype: str) -> StageConfig:
    return StageConfig(
        name=f"gold_fold_{fold}",
        epochs=30,
        head_learning_rate=1e-4,
        backbone_learning_rate=3e-6,
        weight_decay=0.01,
        accumulation_steps=2,
        slices_per_series=12,
        encoder_chunk_size=64,
        amp_dtype=amp_dtype,
        gradient_clip_norm=1.0,
        checkpoint_every_studies=20,
        patience=7,
        seed=20260905 + fold,
        selection_metric="macro_auc",
    )


def main() -> None:
    data_root = Path(os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data"))
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--model-root",
        type=Path,
        default=data_root / "models" / "imaging_v2_public_labels",
    )
    parser.add_argument(
        "--report-prefix",
        default="imaging_v2_public_labels",
    )
    args = parser.parse_args()
    if not args.folds or any(fold not in range(5) for fold in args.folds):
        parser.error("folds must be one or more values from 0 through 4")
    if len(set(args.folds)) != len(args.folds):
        parser.error("folds must be unique")

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
                "weak_labels": str(args.weak_labels),
                "model_root": str(args.model_root),
                "folds": args.folds,
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
    weak_plan = build_training_plan(selected_fold=args.folds[0], **common)
    model = _build_model(data_root / "models" / "torch_hub", device)
    weak_config = StageConfig(
        name="weak_pretrain",
        epochs=3,
        head_learning_rate=3e-4,
        backbone_learning_rate=1e-5,
        weight_decay=0.01,
        accumulation_steps=4,
        slices_per_series=8,
        encoder_chunk_size=64,
        amp_dtype=amp_dtype,
        gradient_clip_norm=1.0,
        checkpoint_every_studies=500,
        patience=None,
        seed=20260905,
        selection_metric="loss",
    )
    weak_report = run_training_stage(
        model,
        weak_plan.weak_train,
        weak_plan.weak_validation,
        device=device,
        output_root=args.model_root,
        report_output=PROJECT_ROOT
        / "reports"
        / "training"
        / f"{args.report_prefix}_weak_summary.json",
        project_root=PROJECT_ROOT,
        competition_root=common["competition_root"],
        config=weak_config,
        source_sha256=weak_plan.source_sha256,
        validation_targets_are_binary=False,
        resume=True,
    )
    print(
        f"weak complete: best_loss={weak_report['best_value']:.6f}", flush=True
    )
    weak_best = args.model_root / "weak_pretrain" / "best.pt"
    del model
    gc.collect()
    torch.cuda.empty_cache()

    fold_results: dict[str, float] = {}
    for fold in args.folds:
        plan = build_training_plan(selected_fold=fold, **common)
        model = _build_model(data_root / "models" / "torch_hub", device)
        load_checkpoint_weights(model, weak_best, device=device)
        configure_backbone(model, unfreeze_last_blocks=2)
        report = run_training_stage(
            model,
            plan.gold_train,
            plan.gold_validation,
            device=device,
            output_root=args.model_root,
            report_output=PROJECT_ROOT
            / "reports"
            / "training"
            / f"{args.report_prefix}_fold_{fold}_summary.json",
            project_root=PROJECT_ROOT,
            competition_root=common["competition_root"],
            config=_gold_config(fold, amp_dtype),
            source_sha256=plan.source_sha256,
            validation_targets_are_binary=True,
            resume=True,
        )
        fold_results[str(fold)] = float(report["best_value"])
        print(
            f"fold {fold} complete: best_macro_auc={report['best_value']:.6f}",
            flush=True,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "status": "complete",
                "fold_best_macro_auc": fold_results,
                "mean_best_macro_auc": sum(fold_results.values())
                / len(fold_results),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
