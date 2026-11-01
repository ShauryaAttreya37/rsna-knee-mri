"""Tests for clinical report extraction parsing and schema validation."""

import json
from rsna.report_extractor import extract_json_from_text
from rsna.submission import TARGET_COLUMNS


def test_extract_json_from_clean_output():
    payload = {col: 0.0 for col in TARGET_COLUMNS}
    payload["ACL"] = 1.0
    payload["Medial Meniscus"] = 0.8
    raw_text = json.dumps(payload)

    parsed = extract_json_from_text(raw_text)
    assert parsed["ACL"] == 1.0
    assert parsed["Medial Meniscus"] == 0.8
    assert parsed["MCL"] == 0.0


def test_extract_json_with_markdown_ticks():
    payload = {col: 0.2 for col in TARGET_COLUMNS}
    payload["Fracture"] = 0.95
    raw_text = f"```json\n{json.dumps(payload)}\n```"

    parsed = extract_json_from_text(raw_text)
    assert parsed["Fracture"] == 0.95
    assert len(parsed) == 12


def test_extract_json_clamping_and_fallback():
    raw_text = '{"ACL": 1.5, "MCL": -0.5, "Invalid": "nan"}'
    parsed = extract_json_from_text(raw_text)
    assert parsed["ACL"] == 1.0
    assert parsed["MCL"] == 0.0
    assert parsed["Medial Meniscus"] == 0.0
    assert len(parsed) == 12


def test_extract_batch_empty():
    from unittest.mock import MagicMock
    from rsna.report_extractor import ClinicalReportExtractor

    extractor = object.__new__(ClinicalReportExtractor)
    assert extractor.extract_batch([]) == []

