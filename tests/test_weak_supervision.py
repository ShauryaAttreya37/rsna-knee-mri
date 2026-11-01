"""Tests for weak supervision batch extraction pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from rsna.submission import TARGET_COLUMNS
from rsna.weak_supervision import _write_summary, extract_weak_supervision


def test_write_summary(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    manifest_data = {
        "StudyInstanceUID": ["s1", "s2"],
        "ACL_prob": [1.0, 0.0],
        "MCL_prob": [0.8, 0.2],
    }
    df = pd.DataFrame(manifest_data)
    summary = _write_summary(df, summary_path, model_id="test_model", total_time=10.0)

    assert summary["status"] == "complete"
    assert summary["total_studies_labeled"] == 2
    assert summary["pseudo_label_positive_rates"]["ACL"] == 0.5
    assert summary["pseudo_label_positive_rates"]["MCL"] == 0.5
    assert summary_path.is_file()


def test_extract_weak_supervision_mock(tmp_path: Path):
    competition_root = tmp_path / "raw"
    competition_root.mkdir(parents=True)
    train_csv = competition_root / "train.csv"

    data = {
        "StudyInstanceUID": ["uid_1", "uid_2"],
        "Report": ["Report 1 text", "Report 2 text"],
        "ACL": [1.0, float("nan")],
    }
    pd.DataFrame(data).to_csv(train_csv, index=False)

    output_csv = tmp_path / "output" / "weak_labels.csv"
    summary_path = tmp_path / "summary.json"

    mock_preds = [{col: 0.2 for col in TARGET_COLUMNS}, {col: 0.8 for col in TARGET_COLUMNS}]

    with patch("rsna.weak_supervision.ClinicalReportExtractor") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.extract_batch.return_value = mock_preds
        mock_cls.return_value = mock_instance

        summary = extract_weak_supervision(
            competition_root=competition_root,
            output_csv=output_csv,
            report_summary=summary_path,
            batch_size=2,
            device="cpu",
        )

        assert output_csv.is_file()
        assert summary_path.is_file()
        out_df = pd.read_csv(output_csv)
        assert len(out_df) == 2
        assert "ACL_prob" in out_df.columns
