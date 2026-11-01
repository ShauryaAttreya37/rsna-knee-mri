from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rsna.physical_geometry_audit import SOURCE_COLUMNS, main


def _row(
    series: str,
    *,
    instance: int,
    plane: str,
    rows: int | None,
    columns: int | None,
    spacing_row: float | None,
    spacing_column: float | None,
    fluid_sensitive: int = 1,
    fat_suppression: int = 0,
) -> dict[str, object]:
    values: dict[str, object] = {
        "split": "train",
        "path_study_uid": f"study-{series}",
        "path_series_uid": series,
        "anatomical_plane": plane,
        "fluid_sensitive": fluid_sensitive,
        "fat_suppression": fat_suppression,
        "rows": rows,
        "columns": columns,
        "pixel_spacing_row": spacing_row,
        "pixel_spacing_column": spacing_column,
        "header_status": "ok",
    }
    assert set(values) == set(SOURCE_COLUMNS)
    # Keep the helper signature slice-like even though InstanceNumber is not
    # needed by this manifest-only audit.
    assert instance > 0
    return values


def _write_audit(
    root: Path,
    rows: list[dict[str, object]],
    *,
    expected_rows: int | None = None,
    expected_series: int,
) -> Path:
    parts_root = root / "manifest_parts"
    parts_root.mkdir(parents=True)
    midpoint = max(1, len(rows) // 2)
    pq.write_table(
        pa.Table.from_pylist(rows[:midpoint]), parts_root / "part-000000.parquet"
    )
    if midpoint < len(rows):
        pq.write_table(
            pa.Table.from_pylist(rows[midpoint:]),
            parts_root / "part-000001.parquet",
        )
    inventory = {
        "status": "pass",
        "manifest_fingerprint": "synthetic-manifest-fingerprint",
        "dicom_files": len(rows) if expected_rows is None else expected_rows,
        "metadata_series": expected_series,
    }
    (root / "inventory_summary.json").write_text(
        json.dumps(inventory), encoding="utf-8"
    )
    return root


def test_cli_audits_all_series_and_emits_identifier_free_summary(
    tmp_path: Path,
) -> None:
    rows = [
        # 200 x 180 mm: covers 180 mm exactly.
        _row(
            "sagittal-secret-uid",
            instance=1,
            plane="Sagittal",
            rows=400,
            columns=300,
            spacing_row=0.5,
            spacing_column=0.6,
        ),
        _row(
            "sagittal-secret-uid",
            instance=2,
            plane="Sagittal",
            rows=400,
            columns=300,
            spacing_row=0.5,
            spacing_column=0.6,
        ),
        # 168 x 200 mm: covers 160 mm but not 180 mm.
        _row(
            "coronal-secret-uid",
            instance=1,
            plane="Coronal",
            rows=240,
            columns=250,
            spacing_row=0.7,
            spacing_column=0.8,
            fat_suppression=1,
        ),
        _row(
            "coronal-secret-uid",
            instance=2,
            plane="Coronal",
            rows=240,
            columns=250,
            spacing_row=0.7,
            spacing_column=0.8,
            fat_suppression=1,
        ),
        # Explicitly counted and excluded because spacing is incomplete.
        _row(
            "axial-missing-secret-uid",
            instance=1,
            plane="Axial",
            rows=256,
            columns=256,
            spacing_row=None,
            spacing_column=0.75,
            fluid_sensitive=0,
        ),
        # Explicitly counted and excluded because spacing varies materially.
        _row(
            "axial-variable-secret-uid",
            instance=1,
            plane="Axial",
            rows=256,
            columns=256,
            spacing_row=0.5,
            spacing_column=0.5,
            fluid_sensitive=0,
        ),
        _row(
            "axial-variable-secret-uid",
            instance=2,
            plane="Axial",
            rows=256,
            columns=256,
            spacing_row=0.55,
            spacing_column=0.5,
            fluid_sensitive=0,
        ),
    ]
    audit_root = _write_audit(tmp_path / "dicom_audit_v1", rows, expected_series=4)
    output_path = tmp_path / "physical_geometry_summary.json"

    assert (
        main(
            [
                "--audit-root",
                str(audit_root),
                "--output",
                str(output_path),
                "--candidate-crop-mm",
                "160",
                "180",
                "--minimum-coverage",
                "0.5",
                "--target-pixels",
                "200",
            ]
        )
        == 0
    )

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["status"] == "pass"
    assert summary["counts"] == {
        "manifest_rows": 7,
        "series": 4,
        "analyzable_series": 2,
        "excluded_series": 2,
    }
    assert summary["row_geometry_findings"]["missing_spacing_rows"] == 1
    assert summary["series_geometry_findings"]["series_with_missing_spacing"] == 1
    assert summary["series_geometry_findings"]["series_with_inconsistent_spacing"] == 1
    overall = summary["distributions"]["overall"]
    assert overall["pixel_spacing_mm"]["row"]["p50"] == pytest.approx(0.6)
    assert overall["physical_fov_mm"]["row"]["min"] == pytest.approx(168.0)
    assert overall["physical_fov_mm"]["column"]["max"] == pytest.approx(200.0)
    assert (
        summary["distributions"]["by_anatomical_plane"]["Axial"]["analyzable_series"]
        == 0
    )

    crop_160, crop_180 = summary["candidate_square_crop_coverage"]
    assert crop_160["covered_without_padding"] == 2
    assert crop_160["coverage_of_analyzable_series"] == 1.0
    assert crop_180["covered_without_padding"] == 1
    assert crop_180["coverage_of_analyzable_series"] == 0.5
    # Axial has no analyzable series, so no candidate can be certified across
    # every observed plane; the smallest candidate is the explicit fallback.
    assert summary["recommendation"]["selection_status"] == ("coverage_target_not_met")
    assert summary["recommendation"]["selected_square_crop_mm"] == 160.0
    assert summary["recommendation"]["detail_view_in_plane_pixel_spacing_mm"] == 0.8
    assert summary["recommendation"]["anatomy_preservation_policy"] == (
        "full_fov_primary_view_required"
    )
    assert summary["recommendation"]["selected_square_crop_role"] == (
        "optional_detail_view_only"
    )
    assert all(value == 0 for value in summary["checks"].values())

    serialized = json.dumps(summary)
    for secret in (
        "sagittal-secret-uid",
        "coronal-secret-uid",
        "axial-missing-secret-uid",
        "axial-variable-secret-uid",
    ):
        assert secret not in serialized


def test_cli_selects_largest_candidate_meeting_coverage_in_every_plane(
    tmp_path: Path,
) -> None:
    rows = [
        _row(
            "sagittal-secret",
            instance=1,
            plane="Sagittal",
            rows=400,
            columns=400,
            spacing_row=0.5,
            spacing_column=0.5,
        ),
        _row(
            "coronal-secret",
            instance=1,
            plane="Coronal",
            rows=400,
            columns=400,
            spacing_row=0.5,
            spacing_column=0.5,
        ),
        _row(
            "axial-secret",
            instance=1,
            plane="Axial",
            rows=400,
            columns=400,
            spacing_row=0.5,
            spacing_column=0.5,
        ),
    ]
    audit_root = _write_audit(tmp_path / "dicom_audit_v1", rows, expected_series=3)
    output_path = tmp_path / "summary.json"

    assert (
        main(
            [
                "--audit-root",
                str(audit_root),
                "--output",
                str(output_path),
                "--candidate-crop-mm",
                "160",
                "180",
                "220",
            ]
        )
        == 0
    )

    recommendation = json.loads(output_path.read_text(encoding="utf-8"))[
        "recommendation"
    ]
    assert recommendation["selection_status"] == "coverage_target_met"
    assert recommendation["selected_square_crop_mm"] == 180.0


def test_count_mismatch_fails_but_writes_auditable_report(tmp_path: Path) -> None:
    rows = [
        _row(
            "series-secret",
            instance=1,
            plane="Sagittal",
            rows=256,
            columns=256,
            spacing_row=0.5,
            spacing_column=0.5,
        )
    ]
    audit_root = _write_audit(
        tmp_path / "dicom_audit_v1",
        rows,
        expected_rows=2,
        expected_series=1,
    )
    output_path = tmp_path / "failed-summary.json"

    assert main(["--audit-root", str(audit_root), "--output", str(output_path)]) == 2

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["status"] == "fail"
    assert summary["checks"]["manifest_row_count_mismatch"] == 1
    assert "series-secret" not in json.dumps(summary)


def test_missing_required_manifest_column_is_refused(tmp_path: Path) -> None:
    audit_root = tmp_path / "dicom_audit_v1"
    parts_root = audit_root / "manifest_parts"
    parts_root.mkdir(parents=True)
    row = _row(
        "series-secret",
        instance=1,
        plane="Sagittal",
        rows=256,
        columns=256,
        spacing_row=0.5,
        spacing_column=0.5,
    )
    del row["pixel_spacing_column"]
    pq.write_table(pa.Table.from_pylist([row]), parts_root / "part-000000.parquet")
    (audit_root / "inventory_summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "manifest_fingerprint": "synthetic",
                "dicom_files": 1,
                "metadata_series": 1,
            }
        ),
        encoding="utf-8",
    )

    assert main(["--audit-root", str(audit_root)]) == 2
    assert not (audit_root / "physical_geometry_summary.json").exists()
