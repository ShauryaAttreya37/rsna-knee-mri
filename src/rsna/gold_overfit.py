"""Same-cohort gold-label overfit test for the multi-series imaging path."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rsna.evaluation import macro_roc_auc
from rsna.submission import TARGET_COLUMNS
from rsna.training_data import (
    CachedTrainingStudy,
    CacheResult,
    TrainingDataError,
    load_gold_training_cohort,
    materialize_gold_cache,
)


class GoldOverfitError(RuntimeError):
    """Raised when the gold overfit test cannot run or meet its contract."""


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise GoldOverfitError(
            "PyTorch is not installed in this environment; use the dedicated "
            "training environment"
        ) from error
    return torch, nn


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    temporary.replace(path)


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def _build_model(torch: Any, nn: Any, feature_dimension: int = 96):
    class TinyMultiSeriesModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.slice_encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=7, stride=4, padding=3, bias=False),
                nn.GroupNorm(4, 16),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(
                    64,
                    feature_dimension,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(8, feature_dimension),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.series_projection = nn.Sequential(
                nn.Linear(feature_dimension + 5, feature_dimension),
                nn.GELU(),
            )
            self.series_attention = nn.Linear(feature_dimension, 1)
            self.head = nn.Sequential(
                nn.LayerNorm(feature_dimension),
                nn.Linear(feature_dimension, feature_dimension),
                nn.GELU(),
                nn.Linear(feature_dimension, len(TARGET_COLUMNS)),
            )

        def forward(self, images: list[Any], metadata: Any) -> Any:
            if not images:
                raise GoldOverfitError("A study cannot be modeled without series")
            slice_counts = [int(item.shape[0]) for item in images]
            stacked = torch.cat([item.unsqueeze(1) for item in images], dim=0)
            encoded_slices = self.slice_encoder(stacked).flatten(1)
            series_features = torch.stack(
                [part.mean(dim=0) for part in torch.split(encoded_slices, slice_counts)]
            )
            enriched = self.series_projection(
                torch.cat([series_features, metadata], dim=1)
            )
            weights = torch.softmax(self.series_attention(enriched).squeeze(1), dim=0)
            study_feature = torch.sum(enriched * weights.unsqueeze(1), dim=0)
            return self.head(study_feature)

    return TinyMultiSeriesModel()


def _load_study_tensors(
    torch: Any, study: CachedTrainingStudy, device: Any
) -> tuple[list[Any], Any, Any]:
    images: list[Any] = []
    for path in study.tensor_paths:
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise GoldOverfitError(
                f"Could not load cached series tensor: {error}"
            ) from error
        images.append(
            torch.from_numpy(np.asarray(array, dtype=np.float32)).to(
                device=device, non_blocking=True
            )
        )
    metadata = torch.tensor(
        [series.metadata_vector for series in study.series],
        dtype=torch.float32,
        device=device,
    )
    labels = torch.tensor(study.labels, dtype=torch.float32, device=device)
    return images, metadata, labels


def _evaluate(
    torch: Any, model: Any, studies: Sequence[CachedTrainingStudy], device: Any
):
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    losses: list[float] = []
    loss_function = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for study in studies:
            images, metadata, targets = _load_study_tensors(torch, study, device)
            logits = model(images, metadata)
            losses.append(float(loss_function(logits, targets).item()))
            labels.append(targets.detach().cpu().numpy())
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    label_matrix = np.asarray(labels, dtype=float)
    probability_matrix = np.asarray(probabilities, dtype=float)
    predictions = probability_matrix >= 0.5
    accuracy = float(np.mean(predictions == label_matrix))
    auc = macro_roc_auc(label_matrix, probability_matrix, TARGET_COLUMNS)
    return {
        "loss": float(np.mean(losses)),
        "binary_accuracy": accuracy,
        "macro_auc": auc.macro_auc,
        "minimum_target_auc": min(auc.per_target_auc.values()),
    }


def _training_environment(torch: Any, device: Any) -> dict[str, Any]:
    cuda = bool(torch.cuda.is_available())
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "cuda_available": cuda,
        "cuda_version": str(torch.version.cuda) if cuda else None,
        "gpu_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
    }


def _label_counts(studies: Sequence[CachedTrainingStudy]) -> dict[str, dict[str, int]]:
    matrix = np.asarray([study.labels for study in studies], dtype=np.int8)
    return {
        target: {
            "positive": int(np.sum(matrix[:, index])),
            "negative": int(len(matrix) - np.sum(matrix[:, index])),
        }
        for index, target in enumerate(TARGET_COLUMNS)
    }


def run_gold_overfit(
    competition_root: Path,
    folds_path: Path,
    exclusions_path: Path,
    project_root: Path,
    cache_root: Path,
    output_root: Path,
    report_output: Path,
    *,
    fold: int = 0,
    sampled_slices: int = 16,
    output_size: int = 224,
    epochs: int = 200,
    learning_rate: float = 3e-3,
    weight_decay: float = 0.0,
    seed: int = 20260828,
    device_name: str = "auto",
    target_loss: float = 0.05,
    target_accuracy: float = 1.0,
    target_macro_auc: float = 0.99,
) -> dict[str, Any]:
    """Overfit one gold fold to validate the complete image-training wiring."""
    for name, value in (
        ("epochs", epochs),
        ("sampled_slices", sampled_slices),
        ("output_size", output_size),
    ):
        if int(value) < 1:
            raise GoldOverfitError(f"{name} must be positive")
    for name, value in (
        ("learning_rate", learning_rate),
        ("weight_decay", weight_decay),
        ("target_loss", target_loss),
        ("target_accuracy", target_accuracy),
        ("target_macro_auc", target_macro_auc),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise GoldOverfitError(f"{name} must be finite and nonnegative")
    if not 0.0 <= target_accuracy <= 1.0 or not 0.0 <= target_macro_auc <= 1.0:
        raise GoldOverfitError("Accuracy and AUC targets must be inside [0, 1]")

    competition_root = competition_root.expanduser().resolve()
    project_root = project_root.expanduser().resolve()
    cache_root = cache_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    report_output = report_output.expanduser().resolve()
    if not competition_root.is_dir() or not project_root.is_dir():
        raise GoldOverfitError("Competition and project roots must exist")
    for external_root, label in ((cache_root, "cache"), (output_root, "model output")):
        if _within(external_root, project_root) or _within(project_root, external_root):
            raise GoldOverfitError(f"Identifier-bearing {label} must remain external")
        if _within(external_root, competition_root) or _within(
            competition_root, external_root
        ):
            raise GoldOverfitError(f"{label} must not overlap raw competition data")
    if _within(cache_root, output_root) or _within(output_root, cache_root):
        raise GoldOverfitError("Cache and model output roots must not overlap")
    if not _within(report_output, project_root):
        raise GoldOverfitError("Aggregate training report must remain in the project")
    if device_name not in {"auto", "cpu", "cuda"}:
        raise GoldOverfitError("device must be one of auto, cpu, or cuda")

    torch, nn = _require_torch()
    _seed_everything(torch, seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise GoldOverfitError("CUDA was requested but is unavailable")
    selected_device = (
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    )
    if selected_device == "auto":
        selected_device = "cpu"
    device = torch.device(selected_device)

    try:
        cohort = load_gold_training_cohort(
            competition_root, folds_path, exclusions_path, fold=fold
        )
        cache: CacheResult = materialize_gold_cache(
            cohort,
            competition_root,
            cache_root,
            sampled_slices=sampled_slices,
            output_size=output_size,
            progress=True,
        )
    except TrainingDataError as error:
        raise GoldOverfitError(str(error)) from error

    model = _build_model(torch, nn).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.BCEWithLogitsLoss()
    generator = np.random.default_rng(seed)
    started = time.monotonic()
    initial_metrics = _evaluate(torch, model, cache.studies, device)
    print(
        "Gold overfit initial: "
        f"loss={initial_metrics['loss']:.6f}, "
        f"accuracy={initial_metrics['binary_accuracy']:.4f}, "
        f"macro_auc={initial_metrics['macro_auc']:.4f}",
        flush=True,
    )
    best_metrics = dict(initial_metrics)
    history: list[dict[str, float | int]] = [{"epoch": 0, **initial_metrics}]
    passed = False
    completed_epochs = 0
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "tiny_multiseries_overfit.pt"
    torch.save(
        {
            "format_version": 1,
            "model": "tiny_multiseries_overfit",
            "target_columns": TARGET_COLUMNS,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        order = generator.permutation(len(cache.studies))
        for index in order:
            study = cache.studies[int(index)]
            images, metadata, targets = _load_study_tensors(torch, study, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, metadata)
            loss = loss_function(logits, targets)
            if not torch.isfinite(loss):
                raise GoldOverfitError("Training loss became nonfinite")
            loss.backward()
            optimizer.step()
        completed_epochs = epoch
        metrics = _evaluate(torch, model, cache.studies, device)
        if metrics["loss"] < best_metrics["loss"]:
            best_metrics = dict(metrics)
            torch.save(
                {
                    "format_version": 1,
                    "model": "tiny_multiseries_overfit",
                    "target_columns": TARGET_COLUMNS,
                    "state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
        if epoch <= 5 or epoch % 10 == 0:
            history.append({"epoch": epoch, **metrics})
        passed = (
            metrics["loss"] <= target_loss
            and metrics["binary_accuracy"] >= target_accuracy
            and metrics["macro_auc"] >= target_macro_auc
        )
        if passed:
            if history[-1]["epoch"] != epoch:
                history.append({"epoch": epoch, **metrics})
            best_metrics = dict(metrics)
        if epoch <= 5 or epoch % 10 == 0 or passed:
            print(
                f"Gold overfit epoch {epoch}: loss={metrics['loss']:.6f}, "
                f"accuracy={metrics['binary_accuracy']:.4f}, "
                f"macro_auc={metrics['macro_auc']:.4f}",
                flush=True,
            )
        if passed:
            break

    elapsed = time.monotonic() - started
    final_metrics = _evaluate(torch, model, cache.studies, device)
    passed = (
        final_metrics["loss"] <= target_loss
        and final_metrics["binary_accuracy"] >= target_accuracy
        and final_metrics["macro_auc"] >= target_macro_auc
    )
    report = {
        "status": "pass" if passed else "fail",
        "generated_at": datetime.now(UTC).isoformat(),
        "privacy": (
            "Aggregate same-cohort training statistics only; identifiers, derived "
            "pixels, cache mappings, and checkpoints remain external."
        ),
        "purpose": (
            "Pipeline wiring and memorization sanity test only. Metrics are measured "
            "on the same gold studies used for optimization and are not a validation "
            "or generalization estimate."
        ),
        "configuration": {
            "fold": fold,
            "epochs_requested": epochs,
            "epochs_completed": completed_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "sampled_slices_per_series": sampled_slices,
            "output_size": output_size,
            "spatial_transform": "physical_full_fov_aspect_fit_zero_pad",
            "model": "tiny_conv_slice_encoder_mean_pool_series_attention",
            "targets": len(TARGET_COLUMNS),
        },
        "acceptance_thresholds": {
            "maximum_loss": target_loss,
            "minimum_binary_accuracy": target_accuracy,
            "minimum_macro_auc": target_macro_auc,
        },
        "counts": {
            "studies": len(cache.studies),
            "series": sum(len(study.series) for study in cache.studies),
            "total_gold_studies": cohort.total_gold_studies,
            "gold_series_before_exclusions": cohort.total_gold_series_before_exclusions,
            "excluded_gold_series": cohort.excluded_gold_series,
            "cache_hits": cache.cache_hits,
            "cache_misses": cache.cache_misses,
            "decoded_images": cache.decoded_images,
            "cache_bytes": cache.cache_bytes,
        },
        "label_counts": _label_counts(cache.studies),
        "metrics": {
            "initial": initial_metrics,
            "best": best_metrics,
            "final": final_metrics,
            "history": history,
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "study_epochs_per_second": (
                len(cache.studies) * completed_epochs / max(elapsed, 1e-9)
            ),
        },
        "environment": _training_environment(torch, device),
        "source_sha256": cohort.source_sha256,
        "external_artifacts": [
            "gold_overfit_cache_v1/",
            "gold_overfit_v1/tiny_multiseries_overfit.pt",
        ],
    }
    _write_json_atomic(report_output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data"))
    project_root = Path(
        os.environ.get("RSNA_PROJECT_ROOT", "/home/shaurya/rsna-project")
    )
    parser = argparse.ArgumentParser(
        description="Overfit one class-complete gold fold as a training sanity test."
    )
    parser.add_argument(
        "--competition-root", type=Path, default=data_root / "raw" / "competition"
    )
    parser.add_argument(
        "--folds-path",
        type=Path,
        default=data_root / "manifests" / "gold_folds_v1" / "gold_folds.csv",
    )
    parser.add_argument(
        "--exclusions-path",
        type=Path,
        default=data_root / "manifests" / "pixel_integrity_v1" / "excluded_series.csv",
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=data_root / "processed" / "gold_overfit_cache_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=data_root / "models" / "gold_overfit_v1",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=project_root / "reports" / "training" / "gold_overfit_summary.json",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--sampled-slices", type=int, default=16)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--target-loss", type=float, default=0.05)
    parser.add_argument("--target-accuracy", type=float, default=1.0)
    parser.add_argument("--target-macro-auc", type=float, default=0.99)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_gold_overfit(
            args.competition_root,
            args.folds_path,
            args.exclusions_path,
            args.project_root,
            args.cache_root,
            args.output_root,
            args.report_output,
            fold=args.fold,
            sampled_slices=args.sampled_slices,
            output_size=args.output_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device_name=args.device,
            target_loss=args.target_loss,
            target_accuracy=args.target_accuracy,
            target_macro_auc=args.target_macro_auc,
        )
        print(
            f"Gold overfit {report['status']}: "
            f"loss={report['metrics']['final']['loss']:.6f}, "
            f"accuracy={report['metrics']['final']['binary_accuracy']:.4f}, "
            f"macro_auc={report['metrics']['final']['macro_auc']:.4f}, "
            f"epochs={report['configuration']['epochs_completed']}",
            flush=True,
        )
        return 0 if report["status"] == "pass" else 2
    except (GoldOverfitError, OSError, ValueError) as error:
        print(f"ERROR: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
