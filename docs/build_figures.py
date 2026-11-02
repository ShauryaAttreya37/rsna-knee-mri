"""Build docs/figure-data.js from the identifier-free summaries in reports/.

Page charts read only this generated module, so every plotted number traces back
to a committed aggregate report. Patient images and per-study results never enter
this file. Regenerate with: python docs/build_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "docs" / "figure-data.js"
DECIMALS = 6
PLANES = ("Axial", "Coronal", "Sagittal")
FINDING_LABELS = {
    "ACL": "ACL",
    "MCL": "MCL",
    "Medial Meniscus": "Medial meniscus",
    "Lateral Meniscus": "Lateral meniscus",
    "Medial OA": "Medial OA",
    "Lateral OA": "Lateral OA",
    "PF OA": "Patellofemoral OA",
    "Effusion": "Effusion",
    "Synovitis": "Synovitis",
    "Baker's": "Baker's cyst",
    "Contusion": "Bone contusion",
    "Fracture": "Fracture",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _rounded(value: float) -> float:
    return round(float(value), DECIMALS)


def _paired_findings(
    before: dict[str, float], after: dict[str, float]
) -> list[dict[str, Any]]:
    expected = set(FINDING_LABELS)
    missing = (expected - set(before)) | (expected - set(after))
    if missing:
        raise ValueError(f"Report is missing findings: {sorted(missing)}")
    rows = [
        {
            "label": label,
            "before": _rounded(before[key]),
            "after": _rounded(after[key]),
        }
        for key, label in FINDING_LABELS.items()
    ]
    return sorted(rows, key=lambda row: (-row["after"], row["label"]))


def _macro(before: float, after: float) -> dict[str, float]:
    return {"before": _rounded(before), "after": _rounded(after)}


def _prevalence(reports: Path) -> list[dict[str, Any]]:
    with (reports / "data_audit" / "label_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = [
            {
                "label": FINDING_LABELS[row["label"]],
                "positives": int(row["positive"]),
                "total": int(row["known"]),
            }
            for row in csv.DictReader(handle)
        ]
    return sorted(rows, key=lambda row: (-row["positives"], row["label"]))


def _crop_coverage(reports: Path) -> dict[str, Any]:
    summary = _read_json(reports / "dicom_audit" / "physical_geometry_summary.json")
    recommendation = summary["recommendation"]
    rows = [
        {
            "crop_mm": int(candidate["square_crop_mm"]),
            "overall": _rounded(candidate["coverage_of_analyzable_series"]),
            "worst_plane": _rounded(candidate["minimum_anatomical_plane_coverage"]),
            **{
                plane.lower(): _rounded(
                    candidate["by_anatomical_plane"][plane][
                        "coverage_of_analyzable_series"
                    ]
                )
                for plane in PLANES
            },
        }
        for candidate in summary["candidate_square_crop_coverage"]
    ]
    return {
        "selected_crop_mm": int(recommendation["selected_square_crop_mm"]),
        "minimum_coverage": recommendation["minimum_required_coverage"],
        "rows": sorted(rows, key=lambda row: row["crop_mm"]),
    }


def _label_quality(reports: Path) -> dict[str, Any]:
    supervision = reports / "report_supervision"
    qwen = _read_json(supervision / "gold_benchmark_summary.json")
    public = _read_json(supervision / "public_weak_label_benchmark.json")
    ensemble = public["candidates"][public["best_by_observed_gold_macro_auc"]]
    return {
        "macro": _macro(qwen["metrics"]["macro_roc_auc"], ensemble["macro_auc"]),
        "studies": {
            "before": qwen["gold_studies_evaluated"],
            "after": ensemble["gold_studies_covered"],
        },
        "rows": _paired_findings(
            qwen["metrics"]["per_target_auc"], ensemble["per_target_auc"]
        ),
    }


def _finding_auc(reports: Path) -> dict[str, Any]:
    v1 = _read_json(reports / "training" / "dinov2_vits14_oof_summary.json")
    v2 = _read_json(reports / "training" / "imaging_v2_public_labels_oof_summary.json")
    return {
        "macro": _macro(v1["pooled_oof"]["macro_auc"], v2["macro_auc"]),
        "rows": _paired_findings(
            v1["pooled_oof"]["per_target_auc"], v2["per_target_auc"]
        ),
    }


def _fold_auc(reports: Path) -> list[dict[str, Any]]:
    v1 = _read_json(reports / "training" / "dinov2_vits14_oof_summary.json")
    v2 = _read_json(reports / "training" / "imaging_v2_public_labels_oof_summary.json")
    fold_sizes = v1["protocol"]["fold_sizes"]
    return [
        {
            "label": f"Fold {index}",
            "studies": fold_sizes[index],
            "before": _rounded(before),
            "after": _rounded(v2["folds"][str(index)]["macro_auc"]),
        }
        for index, before in enumerate(v1["fold_macro_auc"])
    ]


def build_figure_data(reports: Path) -> dict[str, Any]:
    """Return every chart's data, read from the aggregate reports."""
    return {
        "prevalence": _prevalence(reports),
        "crop_coverage": _crop_coverage(reports),
        "label_quality": _label_quality(reports),
        "finding_auc": _finding_auc(reports),
        "fold_auc": _fold_auc(reports),
    }


def render_figure_module(figures: dict[str, Any]) -> str:
    """Serialize chart data as a script the static page can load directly."""
    payload = json.dumps(figures, indent=1, sort_keys=True)
    return (
        "// Generated by docs/build_figures.py from reports/. Do not edit by hand.\n"
        f"window.FIGURE_DATA = {payload};\n"
    )


def main() -> None:
    figures = build_figure_data(REPO_ROOT / "reports")
    OUTPUT_PATH.write_text(render_figure_module(figures), encoding="utf-8")


if __name__ == "__main__":
    main()
