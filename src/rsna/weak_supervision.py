"""Batch weak supervision extraction from radiology reports."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rsna.report_extractor import ClinicalReportExtractor
from rsna.submission import TARGET_COLUMNS


def extract_weak_supervision(
    competition_root: Path,
    output_csv: Path,
    report_summary: Path,
    batch_size: int = 8,
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    device: str = "cuda",
    max_studies: int | None = None,
) -> dict[str, Any]:
    train_path = competition_root / "train.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing {train_path}")

    df = pd.read_csv(train_path)
    total_in_train = len(df)
    print(f"Loaded {total_in_train} total studies from {train_path}.")

    if max_studies is not None and max_studies > 0:
        df = df.iloc[:max_studies].copy().reset_index(drop=True)
        print(f"Limiting execution to first {len(df)} studies (max_studies={max_studies}).")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Check existing progress for resuming
    processed_uids: set[str] = set()
    if output_csv.is_file():
        existing_df = pd.read_csv(output_csv)
        if "StudyInstanceUID" in existing_df.columns:
            processed_uids = set(existing_df["StudyInstanceUID"].astype(str))
            print(f"Found existing output at {output_csv} with {len(processed_uids)} studies.")

    pending_df = df[~df["StudyInstanceUID"].astype(str).isin(processed_uids)].copy().reset_index(drop=True)
    remaining_count = len(pending_df)
    print(f"Pending studies to extract: {remaining_count} / {len(df)}")

    if remaining_count == 0:
        print("All requested studies already extracted! Computing summary...")
        final_df = pd.read_csv(output_csv)
        return _write_summary(final_df, report_summary, model_id, total_time=0.0)

    print(f"Initializing ClinicalReportExtractor ({model_id}) on {device}...")
    extractor = ClinicalReportExtractor(model_id=model_id, device=device)

    started_all = time.monotonic()
    batch_records: list[dict[str, Any]] = []
    completed_in_run = 0

    for start_idx in range(0, remaining_count, batch_size):
        batch_slice = pending_df.iloc[start_idx : start_idx + batch_size]
        reports = [str(r) for r in batch_slice["Report"]]
        uids = [str(u) for u in batch_slice["StudyInstanceUID"]]

        extracted_batch = extractor.extract_batch(reports)

        for idx, uid in enumerate(uids):
            pred = extracted_batch[idx]
            row_data: dict[str, Any] = {"StudyInstanceUID": uid}
            # Record soft probabilities for each target
            for col in TARGET_COLUMNS:
                row_data[f"{col}_prob"] = round(float(pred.get(col, 0.0)), 4)
            batch_records.append(row_data)

        completed_in_run += len(batch_slice)

        # Write checkpoint periodically (every batch)
        if batch_records:
            chunk_df = pd.DataFrame(batch_records)
            write_header = not output_csv.is_file() or os.path.getsize(output_csv) == 0
            chunk_df.to_csv(output_csv, mode="a", index=False, header=write_header)
            batch_records.clear()

        # Progress reporting
        elapsed = time.monotonic() - started_all
        rate = completed_in_run / elapsed if elapsed > 0 else 0.0
        remaining_studies = remaining_count - completed_in_run
        eta_seconds = remaining_studies / rate if rate > 0 else 0.0
        eta_str = f"{eta_seconds / 60:.1f}m" if eta_seconds < 3600 else f"{eta_seconds / 3600:.2f}h"

        if completed_in_run % (batch_size * 5) == 0 or completed_in_run == remaining_count:
            print(
                f"Progress: [{completed_in_run}/{remaining_count}] "
                f"({completed_in_run / remaining_count * 100:.1f}%) | "
                f"Rate: {rate:.2f} studies/s ({1.0 / rate if rate > 0 else 0:.2f} s/study) | "
                f"ETA: {eta_str}"
            )

    total_time = time.monotonic() - started_all
    print(f"\nExtraction complete in {total_time:.1f}s ({total_time / 60:.1f}m).")

    final_df = pd.read_csv(output_csv)
    return _write_summary(final_df, report_summary, model_id, total_time=total_time)


def _write_summary(
    manifest_df: pd.DataFrame,
    summary_path: Path,
    model_id: str,
    total_time: float,
) -> dict[str, Any]:
    positive_rates: dict[str, float] = {}
    for col in TARGET_COLUMNS:
        pcol = f"{col}_prob"
        if pcol in manifest_df.columns:
            positive_rates[col] = float((manifest_df[pcol] >= 0.5).mean())
        else:
            positive_rates[col] = 0.0

    summary_data = {
        "status": "complete",
        "generated_at": datetime.now(UTC).isoformat(),
        "model_id": model_id,
        "total_studies_labeled": len(manifest_df),
        "runtime_seconds": total_time,
        "seconds_per_study": total_time / len(manifest_df) if len(manifest_df) > 0 else 0.0,
        "pseudo_label_positive_rates": positive_rates,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Summary written to {summary_path}")
    return summary_data
