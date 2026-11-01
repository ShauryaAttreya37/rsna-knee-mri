from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rsna.orientation_audit import SOURCE_COLUMNS, main


def _row(
    series: str,
    instance: int,
    angle_degrees: float,
    plane: str = "Sagittal",
) -> dict[str, object]:
    angle = math.radians(angle_degrees)
    values: dict[str, object] = {
        "split": "train",
        "path_study_uid": f"study-{series}",
        "path_series_uid": series,
        "anatomical_plane": plane,
        "fluid_sensitive": 1,
        "fat_suppression": 0,
        "instance_number": instance,
        "image_position_x": 0.0,
        "image_position_y": 0.0,
        "image_position_z": float(instance),
        "image_orientation_row_x": 1.0,
        "image_orientation_row_y": 0.0,
        "image_orientation_row_z": 0.0,
        "image_orientation_column_x": 0.0,
        "image_orientation_column_y": math.cos(angle),
        "image_orientation_column_z": math.sin(angle),
        "header_status": "ok",
        "geometry_status": "ok",
    }
    assert set(values) == set(SOURCE_COLUMNS)
    return values


def test_cli_emits_identifier_free_orientation_summary(tmp_path: Path) -> None:
    audit_root = tmp_path / "dicom_audit_v1"
    parts_root = audit_root / "manifest_parts"
    parts_root.mkdir(parents=True)
    rows = [
        _row("constant-secret-uid", 1, 0.0),
        _row("constant-secret-uid", 2, 0.0),
        _row("constant-secret-uid", 3, 0.0),
        _row("variable-secret-uid", 1, 0.0, "Coronal"),
        _row("variable-secret-uid", 2, 0.02, "Coronal"),
        _row("variable-secret-uid", 3, 0.04, "Coronal"),
    ]
    rounding_boundary_rows = [
        _row("rounding-boundary-secret-uid", 1, 0.0, "Axial"),
        _row("rounding-boundary-secret-uid", 2, 0.0, "Axial"),
    ]
    rounding_boundary_rows[0]["image_orientation_row_x"] = -1.0
    rounding_boundary_rows[1]["image_orientation_row_x"] = -0.999995
    rows.extend(rounding_boundary_rows)
    pq.write_table(pa.Table.from_pylist(rows), parts_root / "part-000000.parquet")
    inventory = {
        "status": "pass",
        "manifest_fingerprint": "synthetic-fingerprint",
        "dicom_files": 8,
        "metadata_series": 3,
        "series_geometry_audit": {"series_with_inconsistent_orientation": 2},
    }
    (audit_root / "inventory_summary.json").write_text(
        json.dumps(inventory), encoding="utf-8"
    )
    output_path = tmp_path / "sanitized.json"

    assert (
        main(
            [
                "--audit-root",
                str(audit_root),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["status"] == "pass"
    assert summary["counts"]["series"] == 3
    assert summary["counts"]["variable_orientation_series"] == 2
    assert summary["variable_series_by_anatomical_plane"] == {
        "Axial": 1,
        "Coronal": 1,
    }
    assert summary["normal_angle_bins"]["at_most_0.001_deg"] == 1
    assert summary["normal_angle_bins"]["0.01_to_0.1_deg"] == 1
    assert summary["slice_sorting"]["consensus_projection_changes_order"] == 0
    assert all(value == 0 for value in summary["checks"].values())
    serialized = json.dumps(summary)
    assert "constant-secret-uid" not in serialized
    assert "variable-secret-uid" not in serialized
    assert "rounding-boundary-secret-uid" not in serialized
