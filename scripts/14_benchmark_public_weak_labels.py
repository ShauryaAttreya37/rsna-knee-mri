"""Benchmark public report-derived label files on the official gold subset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsna.weak_label_benchmark import (
    benchmark_candidates,
    write_equal_mean_ensemble,
    write_report,
)


def _candidate(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("candidate must be NAME=PATH")
    return name, Path(raw_path).expanduser()


def main() -> None:
    data_root = Path(os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=data_root / "raw" / "competition" / "train.csv",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        type=_candidate,
        help="public label source as NAME=PATH; repeat for multiple sources",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "report_supervision"
        / "public_weak_label_benchmark.json",
    )
    parser.add_argument(
        "--ensemble-output",
        type=Path,
        help="optional external weak_labels.csv destination for the equal mean",
    )
    parser.add_argument("--minimum-sources", type=int, default=2)
    args = parser.parse_args()
    payload = benchmark_candidates(args.train_csv, args.candidate)
    if args.ensemble_output is not None:
        try:
            args.ensemble_output.resolve().relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            pass
        else:
            parser.error("identifier-bearing ensemble output must remain outside Git")
        payload["ensemble"] = write_equal_mean_ensemble(
            args.train_csv,
            args.candidate,
            args.ensemble_output,
            minimum_sources=args.minimum_sources,
        )
    write_report(args.report_output, payload)
    for name, metrics in payload["candidates"].items():
        print(
            f"{name}: macro_auc={metrics['macro_auc']:.6f} "
            f"accuracy={metrics['binary_accuracy']:.6f} "
            f"brier={metrics['brier_score']:.6f}"
        )
    print(f"best: {payload['best_by_observed_gold_macro_auc']}")
    if args.ensemble_output is not None:
        print(
            f"ensemble: {args.ensemble_output} "
            f"({payload['ensemble']['studies_written']} studies)"
        )
    print(f"report: {args.report_output}")


if __name__ == "__main__":
    main()
