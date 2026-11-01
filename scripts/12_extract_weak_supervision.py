"""CLI to extract weak supervision labels from clinical reports."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsna.weak_supervision import extract_weak_supervision

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract weak supervision from reports")
    parser.add_argument(
        "--competition-root",
        type=Path,
        default=Path(os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data")) / "raw" / "competition",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data")) / "manifests" / "weak_labels_v1" / "weak_labels.csv",
    )
    parser.add_argument(
        "--report-summary",
        type=Path,
        default=PROJECT_ROOT / "reports" / "report_supervision" / "weak_supervision_summary.json",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-studies", type=int, default=None)
    args = parser.parse_args()

    extract_weak_supervision(
        competition_root=args.competition_root,
        output_csv=args.output_csv,
        report_summary=args.report_summary,
        batch_size=args.batch_size,
        model_id=args.model_id,
        device=args.device,
        max_studies=args.max_studies,
    )
