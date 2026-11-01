from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rsna.pixel_integrity import PixelIntegrityError, derive_pixel_integrity_policy
from rsna.submission import ID_COLUMN


def _write_series_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow([ID_COLUMN, "SeriesInstanceUID"])
        writer.writerows(rows)


def _decode_row(
    split: str,
    study_uid: str,
    series_uid: str,
    sop_uid: str,
    *,
    status: str = "ok",
    finite: bool | None = True,
    shape_matches: bool | None = True,
) -> dict[str, object]:
    directory = "train_series" if split == "train" else "test_series"
    is_error = status != "ok"
    return {
        "relative_path": f"{directory}/{study_uid}/{series_uid}/{sop_uid}.dcm",
        "split": split,
        "decode_status": status,
        "all_finite": finite,
        "shape_matches_header": shape_matches,
        "error_class": "ValueError" if is_error else None,
        "error_message": "secret per-file decode detail" if is_error else None,
    }


def _build_fixture(tmp_path: Path, *, attempted_override: int | None = None):
    competition_root = tmp_path / "raw" / "competition"
    project_root = tmp_path / "project"
    project_root.mkdir()
    audit_root = tmp_path / "manifests" / "dicom_audit_v1"
    decode_root = audit_root / "decode_all"
    parts_root = decode_root / "parts"
    parts_root.mkdir(parents=True)
    output_root = tmp_path / "manifests" / "pixel_integrity_v1"
    report_output = project_root / "reports" / "pixel-integrity.json"

    train_study = "secret-train-study"
    test_study = "secret-test-study"
    identifiers = [
        train_study,
        test_study,
        "secret-train-bad",
        "secret-train-good",
        "secret-test-nonfinite",
        "secret-test-shape",
        "secret-test-good",
    ]
    _write_series_csv(
        competition_root / "train_series.csv",
        [
            (train_study, "secret-train-bad"),
            (train_study, "secret-train-good"),
        ],
    )
    _write_series_csv(
        competition_root / "test_series.csv",
        [
            (test_study, "secret-test-nonfinite"),
            (test_study, "secret-test-shape"),
            (test_study, "secret-test-good"),
        ],
    )
    rows = [
        _decode_row(
            "train",
            train_study,
            "secret-train-bad",
            "sop-1",
            status="decode_error",
            finite=None,
            shape_matches=None,
        ),
        _decode_row("train", train_study, "secret-train-bad", "sop-2"),
        _decode_row("train", train_study, "secret-train-good", "sop-3"),
        _decode_row("test", test_study, "secret-test-nonfinite", "sop-4", finite=False),
        _decode_row(
            "test", test_study, "secret-test-shape", "sop-5", shape_matches=False
        ),
        _decode_row("test", test_study, "secret-test-good", "sop-6"),
    ]
    pq.write_table(pa.Table.from_pylist(rows), parts_root / "part-000000.parquet")
    attempted = len(rows) if attempted_override is None else attempted_override
    (decode_root / "decode_config.json").write_text(
        json.dumps({"mode": "all", "manifest_fingerprint": "opaque-fingerprint"}),
        encoding="utf-8",
    )
    (decode_root / "decode_summary.json").write_text(
        json.dumps(
            {
                "status": "fail",
                "mode": "all",
                "attempted": attempted,
                "decoded": 5,
                "checks": {
                    "decode_failures": 1,
                    "nonfinite_arrays": 1,
                    "shape_mismatches": 1,
                    "transfer_syntaxes_without_decode_test": 0,
                    "exhaustive_decode_count_mismatch": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    return (
        competition_root,
        audit_root,
        project_root,
        output_root,
        report_output,
        identifiers,
    )


def test_policy_excludes_entire_unsafe_series_and_keeps_identifiers_external(
    tmp_path: Path,
) -> None:
    roots = _build_fixture(tmp_path)
    report = derive_pixel_integrity_policy(*roots[:5])

    assert report["status"] == "pass"
    assert report["source"] == {
        "audit_mode": "all",
        "audit_status": "fail",
        "manifest_fingerprint": "opaque-fingerprint",
    }
    assert report["counts"] == {
        "attempted_instances": 6,
        "decoded_instances": 5,
        "safe_instances": 3,
        "unsafe_instances": 3,
        "unsafe_instance_fraction": 0.5,
        "total_series": 5,
        "excluded_series": 3,
        "eligible_series": 2,
        "affected_studies": 2,
    }
    assert report["checks"] == {
        "decode_part_row_count_mismatch": 0,
        "decoded_count_mismatch": 0,
        "decode_failure_count_mismatch": 0,
        "nonfinite_count_mismatch": 0,
        "shape_mismatch_count_mismatch": 0,
        "source_exhaustive_count_mismatch": 0,
        "source_transfer_syntax_coverage_failure": 0,
        "excluded_series_missing_from_metadata": 0,
        "decoded_series_missing_from_metadata": 0,
        "metadata_series_without_decode_rows": 0,
        "studies_without_eligible_series": 0,
    }
    assert report["training_policy"]["slice_skipping"] is False
    assert report["training_policy"]["pixel_repair_or_reshape"] is False
    serialized = roots[4].read_text(encoding="utf-8")
    assert all(identifier not in serialized for identifier in roots[5])
    excluded = (roots[3] / "excluded_series.csv").read_text(encoding="utf-8")
    unsafe = (roots[3] / "unsafe_instances.csv").read_text(encoding="utf-8")
    assert "secret-train-bad" in excluded
    assert "secret-test-nonfinite" in excluded
    assert "secret-test-shape" in excluded
    assert "secret per-file decode detail" in unsafe


def test_policy_rejects_decode_summary_row_count_drift(tmp_path: Path) -> None:
    roots = _build_fixture(tmp_path, attempted_override=7)

    with pytest.raises(PixelIntegrityError, match="consistency checks failed"):
        derive_pixel_integrity_policy(*roots[:5])
