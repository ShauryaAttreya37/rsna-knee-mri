"""Finalize the explicit human review of preprocessing montages."""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsna.preprocessing_benchmark import (
    PreprocessingBenchmarkError,
    finalize_preprocessing_visual_review,
)


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data"))
    project_root = Path(
        os.environ.get("RSNA_PROJECT_ROOT", "/home/shaurya/rsna-project")
    )
    parser = argparse.ArgumentParser(
        description="Record a completed human review of every benchmark montage."
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=data_root / "manifests" / "preprocessing_benchmark_v1",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=project_root / "reports" / "preprocessing_benchmark" / "summary.json",
    )
    parser.add_argument(
        "--all-cases-pass",
        action="store_true",
        help="Attest that every montage and slice tile passed all required checks",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = finalize_preprocessing_visual_review(
            args.project_root,
            args.output_root,
            args.report_output,
            all_cases_pass=args.all_cases_pass,
        )
        print(
            f"Visual review {report['visual_review']['status']}: "
            f"{report['visual_review']['reviewed_cases']} cases attested",
            flush=True,
        )
        return 0
    except (
        json.JSONDecodeError,
        OSError,
        PreprocessingBenchmarkError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
