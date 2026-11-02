from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FINDING_COUNT = 12
FOLD_COUNT = 5


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_figures", REPO_ROOT / "docs" / "build_figures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


@pytest.fixture(scope="module")
def figures(builder):
    return builder.build_figure_data(REPO_ROOT / "reports")


def test_prevalence_is_ranked_from_most_to_least_common(figures):
    rows = figures["prevalence"]

    assert len(rows) == FINDING_COUNT
    assert rows[0] == {"label": "Effusion", "positives": 35, "total": 58}
    assert rows[-1] == {"label": "MCL", "positives": 9, "total": 58}
    positives = [row["positives"] for row in rows]
    assert positives == sorted(positives, reverse=True)


def test_crop_coverage_reports_selected_window_and_threshold(figures):
    coverage = figures["crop_coverage"]

    assert coverage["selected_crop_mm"] == 140
    assert coverage["minimum_coverage"] == 0.9
    assert [row["crop_mm"] for row in coverage["rows"]] == [120, 140, 160, 180, 200]
    selected = coverage["rows"][1]
    assert selected["overall"] == pytest.approx(0.9425, abs=1e-4)
    assert selected["worst_plane"] == pytest.approx(0.9214, abs=1e-4)
    assert selected["coronal"] == pytest.approx(0.9214, abs=1e-4)


def test_label_quality_compares_both_sources_on_every_finding(figures):
    quality = figures["label_quality"]

    assert len(quality["rows"]) == FINDING_COUNT
    assert quality["macro"]["before"] == pytest.approx(0.7533, abs=1e-4)
    assert quality["macro"]["after"] == pytest.approx(0.8956, abs=1e-4)
    assert quality["studies"] == {"before": 58, "after": 57}
    assert all(row["after"] > row["before"] for row in quality["rows"])


def test_finding_auc_matches_results_table_and_exposes_declines(figures):
    results = figures["finding_auc"]
    by_label = {row["label"]: row for row in results["rows"]}

    assert len(results["rows"]) == FINDING_COUNT
    assert results["macro"]["before"] == pytest.approx(0.6863, abs=1e-4)
    assert results["macro"]["after"] == pytest.approx(0.7835, abs=1e-4)
    assert by_label["Bone contusion"]["before"] == pytest.approx(0.5209, abs=1e-4)
    assert by_label["Bone contusion"]["after"] == pytest.approx(0.8219, abs=1e-4)
    declined = {row["label"] for row in results["rows"] if row["after"] < row["before"]}
    assert declined == {"MCL", "Lateral meniscus", "Lateral OA"}


def test_fold_auc_improves_in_every_fold(figures):
    rows = figures["fold_auc"]

    assert [row["label"] for row in rows] == [f"Fold {i}" for i in range(FOLD_COUNT)]
    assert [row["studies"] for row in rows] == [12, 12, 12, 11, 11]
    assert all(row["after"] > row["before"] for row in rows)


def test_figure_data_contains_no_study_identifiers(figures):
    serialized = json.dumps(figures)

    assert "StudyInstanceUID" not in serialized
    assert "SeriesInstanceUID" not in serialized
    assert "1.2.826" not in serialized


def test_committed_figure_data_matches_reports(builder, figures):
    committed = (REPO_ROOT / "docs" / "figure-data.js").read_text(encoding="utf-8")

    assert committed == builder.render_figure_module(figures)


def test_missing_report_fails_with_named_file(builder, tmp_path):
    reports_copy = tmp_path / "reports"
    shutil.copytree(REPO_ROOT / "reports", reports_copy)
    (reports_copy / "training" / "dinov2_vits14_oof_summary.json").unlink()

    with pytest.raises(FileNotFoundError, match="dinov2_vits14_oof_summary.json"):
        builder.build_figure_data(reports_copy)
