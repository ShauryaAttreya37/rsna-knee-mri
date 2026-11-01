"""Benchmark LLM report extraction against the 58 gold-labeled studies."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsna.evaluation import macro_roc_auc
from rsna.report_extractor import ClinicalReportExtractor
from rsna.submission import TARGET_COLUMNS


def run_gold_report_benchmark(
    competition_root: Path,
    report_output: Path,
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    device: str = "cuda",
) -> dict[str, Any]:
    train_path = competition_root / "train.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing {train_path}")

    df = pd.read_csv(train_path)
    gold_df = df[df["ACL"].notna()].copy().reset_index(drop=True)
    total_gold = len(gold_df)
    print(f"Loaded {total_gold} gold-labeled studies with reports.")

    print(f"Initializing ClinicalReportExtractor ({model_id}) on {device}...")
    extractor = ClinicalReportExtractor(model_id=model_id, device=device)

    ground_truth = gold_df[list(TARGET_COLUMNS)].values.astype(float)
    predictions = np.zeros((total_gold, len(TARGET_COLUMNS)), dtype=float)

    started = time.monotonic()
    print("Extracting predictions from reports...")
    for idx, row in gold_df.iterrows():
        report_text = str(row["Report"])
        pred_dict = extractor.extract_one(report_text)
        for col_idx, col in enumerate(TARGET_COLUMNS):
            predictions[idx, col_idx] = pred_dict[col]
        if (idx + 1) % 10 == 0 or idx == total_gold - 1:
            print(f"Processed {idx + 1}/{total_gold} reports...")

    elapsed = time.monotonic() - started
    print(f"Extraction completed in {elapsed:.1f}s ({elapsed / total_gold:.2f}s/report).")

    auc_result = macro_roc_auc(ground_truth, predictions, TARGET_COLUMNS)
    binary_preds = (predictions >= 0.5).astype(float)
    exact_acc = float(np.mean(binary_preds == ground_truth))

    print("\n--- Benchmark Results on 58 Gold Studies ---")
    print(f"Macro ROC AUC : {auc_result.macro_auc:.4f}")
    print(f"Binary Accuracy: {exact_acc * 100:.2f}%")
    print("\nPer-Target ROC AUC:")
    for target in TARGET_COLUMNS:
        print(f"  {target:18s}: {auc_result.per_target_auc[target]:.4f}")

    report_data = {
        "status": "pass",
        "generated_at": datetime.now(UTC).isoformat(),
        "model_id": model_id,
        "gold_studies_evaluated": total_gold,
        "runtime_seconds": elapsed,
        "seconds_per_report": elapsed / total_gold,
        "metrics": {
            "macro_roc_auc": auc_result.macro_auc,
            "overall_binary_accuracy": exact_acc,
            "per_target_auc": auc_result.per_target_auc,
        },
    }

    report_output.parent.mkdir(parents=True, exist_ok=True)
    with report_output.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nReport written to {report_output}")
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLM extraction on gold reports")
    parser.add_argument(
        "--competition-root",
        type=Path,
        default=Path(os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data")) / "raw" / "competition",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "report_supervision" / "gold_benchmark_summary.json",
    )
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_gold_report_benchmark(
        competition_root=args.competition_root,
        report_output=args.report_output,
        model_id=args.model_id,
        device=args.device,
    )
