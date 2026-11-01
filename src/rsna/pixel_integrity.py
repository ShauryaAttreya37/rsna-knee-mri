"""Derive an explicit series-exclusion policy from exhaustive pixel decoding."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow.parquet as pq

from rsna.submission import ID_COLUMN

SERIES_UID_COLUMN = "SeriesInstanceUID"
DECODE_COLUMNS = (
    "relative_path",
    "split",
    "decode_status",
    "all_finite",
    "shape_matches_header",
    "error_class",
    "error_message",
)
SPLIT_SERIES_FILES = {"train": "train_series.csv", "test": "test_series.csv"}
SPLIT_DIRECTORIES = {"train_series": "train", "test_series": "test"}


class PixelIntegrityError(RuntimeError):
    """Raised when exhaustive decode results cannot support a safe policy."""


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_roots(
    competition_root: Path,
    audit_root: Path,
    project_root: Path,
    output_root: Path,
    report_output: Path,
) -> None:
    for path, label in (
        (competition_root, "competition root"),
        (audit_root, "audit root"),
        (project_root, "project root"),
    ):
        if not path.is_dir():
            raise PixelIntegrityError(f"Missing {label}: {path}")
    if _within(output_root, competition_root) or _within(competition_root, output_root):
        raise PixelIntegrityError("Integrity output must not overlap raw data")
    if _within(output_root, project_root) or _within(project_root, output_root):
        raise PixelIntegrityError(
            "Identifier-bearing integrity output must remain outside the project"
        )
    if _within(output_root, audit_root) or _within(audit_root, output_root):
        raise PixelIntegrityError("Integrity output must not overlap the source audit")
    if not _within(report_output, project_root):
        raise PixelIntegrityError("Aggregate report must remain inside the project")


def _load_series_inventory(
    competition_root: Path,
) -> tuple[set[tuple[str, str, str]], dict[tuple[str, str], int]]:
    series: set[tuple[str, str, str]] = set()
    studies: dict[tuple[str, str], int] = defaultdict(int)
    for split, filename in SPLIT_SERIES_FILES.items():
        path = competition_root / filename
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                required = {ID_COLUMN, SERIES_UID_COLUMN}
                missing = required - set(reader.fieldnames or [])
                if missing:
                    raise PixelIntegrityError(
                        f"{filename} is missing required columns: {sorted(missing)}"
                    )
                for row in reader:
                    study_uid = (row.get(ID_COLUMN) or "").strip()
                    series_uid = (row.get(SERIES_UID_COLUMN) or "").strip()
                    key = (split, study_uid, series_uid)
                    if not study_uid or not series_uid or key in series:
                        raise PixelIntegrityError(
                            f"{filename} contains blank or duplicate series identifiers"
                        )
                    series.add(key)
                    studies[(split, study_uid)] += 1
        except (OSError, UnicodeError, csv.Error) as error:
            raise PixelIntegrityError(f"Could not read {filename}: {error}") from error
    if not series:
        raise PixelIntegrityError("Series metadata contains no series")
    return series, dict(studies)


def _iter_decode_rows(part_paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    for part_path in part_paths:
        parquet = pq.ParquetFile(part_path)
        missing = set(DECODE_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise PixelIntegrityError(
                f"Decode part is missing required columns: {sorted(missing)}"
            )
        for batch in parquet.iter_batches(
            columns=list(DECODE_COLUMNS), batch_size=65_536
        ):
            yield from batch.to_pylist()


def _parse_relative_path(
    relative_path: str, declared_split: str
) -> tuple[str, str, str, str]:
    path = PurePosixPath(relative_path)
    parts = path.parts
    if len(parts) != 4 or path.suffix.lower() != ".dcm" or ".." in parts:
        raise PixelIntegrityError("Decode output contains an invalid relative path")
    split = SPLIT_DIRECTORIES.get(parts[0])
    if split is None or split != declared_split or not all(parts[1:]):
        raise PixelIntegrityError("Decode path and split metadata are inconsistent")
    return split, parts[1], parts[2], path.stem


def _unsafe_reasons(row: dict[str, Any]) -> tuple[str, ...]:
    status = str(row.get("decode_status") or "")
    if status not in {"ok", "decode_error"}:
        raise PixelIntegrityError("Decode output contains an unknown status")
    if status == "ok" and (
        row.get("all_finite") is None or row.get("shape_matches_header") is None
    ):
        raise PixelIntegrityError("Successful decode row lacks integrity checks")
    reasons: list[str] = []
    if status != "ok":
        reasons.append("decode_error")
    if row.get("all_finite") is False:
        reasons.append("nonfinite_array")
    if row.get("shape_matches_header") is False:
        reasons.append("shape_mismatch")
    return tuple(reasons)


def _write_csv_atomic(
    path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    temporary.replace(path)


def derive_pixel_integrity_policy(
    competition_root: Path,
    audit_root: Path,
    project_root: Path,
    output_root: Path,
    report_output: Path,
) -> dict[str, Any]:
    """Create series-level exclusions and an identifier-free aggregate report."""
    competition_root = competition_root.expanduser().resolve()
    audit_root = audit_root.expanduser().resolve()
    project_root = project_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    report_output = report_output.expanduser().resolve()
    _validate_roots(
        competition_root, audit_root, project_root, output_root, report_output
    )

    decode_root = audit_root / "decode_all"
    summary_path = decode_root / "decode_summary.json"
    config_path = decode_root / "decode_config.json"
    part_paths = sorted((decode_root / "parts").glob("part-*.parquet"))
    if not summary_path.is_file() or not config_path.is_file() or not part_paths:
        raise PixelIntegrityError("Completed exhaustive decode audit is missing")
    with summary_path.open("r", encoding="utf-8") as file:
        source_summary = json.load(file)
    with config_path.open("r", encoding="utf-8") as file:
        source_config = json.load(file)
    if source_summary.get("mode") != "all" or source_config.get("mode") != "all":
        raise PixelIntegrityError("Pixel-integrity policy requires decode-all results")

    metadata_series, study_series_counts = _load_series_inventory(competition_root)
    attempted = 0
    decoded = 0
    decode_failures = 0
    nonfinite_arrays = 0
    shape_mismatches = 0
    split_attempts: Counter[str] = Counter()
    audited_series: set[tuple[str, str, str]] = set()
    unsafe_instances: list[dict[str, Any]] = []
    unsafe_by_series: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in _iter_decode_rows(part_paths):
        split, study_uid, series_uid, sop_uid = _parse_relative_path(
            str(row.get("relative_path") or ""), str(row.get("split") or "")
        )
        attempted += 1
        split_attempts[split] += 1
        audited_series.add((split, study_uid, series_uid))
        status = str(row["decode_status"])
        decoded += int(status == "ok")
        decode_failures += int(status != "ok")
        nonfinite_arrays += int(row.get("all_finite") is False)
        shape_mismatches += int(row.get("shape_matches_header") is False)
        reasons = _unsafe_reasons(row)
        if not reasons:
            continue
        instance = {
            "split": split,
            "study_uid": study_uid,
            "series_uid": series_uid,
            "sop_uid": sop_uid,
            "relative_path": str(row["relative_path"]),
            "unsafe_reasons": ";".join(reasons),
            "error_class": str(row.get("error_class") or ""),
            "error_message": str(row.get("error_message") or ""),
        }
        unsafe_instances.append(instance)
        unsafe_by_series[(split, study_uid, series_uid)].append(instance)

    expected_checks = source_summary.get("checks") or {}
    expected_attempted = _integer(source_summary.get("attempted"))
    expected_decoded = _integer(source_summary.get("decoded"))
    checks = {
        "decode_part_row_count_mismatch": int(expected_attempted != attempted),
        "decoded_count_mismatch": int(expected_decoded != decoded),
        "decode_failure_count_mismatch": int(
            _integer(expected_checks.get("decode_failures")) != decode_failures
        ),
        "nonfinite_count_mismatch": int(
            _integer(expected_checks.get("nonfinite_arrays")) != nonfinite_arrays
        ),
        "shape_mismatch_count_mismatch": int(
            _integer(expected_checks.get("shape_mismatches")) != shape_mismatches
        ),
        "source_exhaustive_count_mismatch": int(
            _integer(expected_checks.get("exhaustive_decode_count_mismatch")) != 0
        ),
        "source_transfer_syntax_coverage_failure": int(
            _integer(expected_checks.get("transfer_syntaxes_without_decode_test")) != 0
        ),
    }
    if any(checks.values()):
        raise PixelIntegrityError(f"Decode audit consistency checks failed: {checks}")
    if attempted == 0:
        raise PixelIntegrityError("Exhaustive decode audit contains no rows")

    excluded_keys = set(unsafe_by_series)
    missing_from_metadata = excluded_keys - metadata_series
    eligible_per_study = dict(study_series_counts)
    for split, study_uid, _ in excluded_keys & metadata_series:
        eligible_per_study[(split, study_uid)] -= 1
    studies_without_eligible = sum(value == 0 for value in eligible_per_study.values())
    checks.update(
        {
            "excluded_series_missing_from_metadata": len(missing_from_metadata),
            "decoded_series_missing_from_metadata": len(
                audited_series - metadata_series
            ),
            "metadata_series_without_decode_rows": len(
                metadata_series - audited_series
            ),
            "studies_without_eligible_series": studies_without_eligible,
        }
    )

    excluded_rows: list[dict[str, Any]] = []
    for (split, study_uid, series_uid), instances in sorted(unsafe_by_series.items()):
        reasons = sorted(
            {
                reason
                for instance in instances
                for reason in str(instance["unsafe_reasons"]).split(";")
            }
        )
        error_classes = sorted(
            {
                str(instance["error_class"])
                for instance in instances
                if instance["error_class"]
            }
        )
        excluded_rows.append(
            {
                "split": split,
                "study_uid": study_uid,
                "series_uid": series_uid,
                "unsafe_instances": len(instances),
                "unsafe_reasons": ";".join(reasons),
                "error_classes": ";".join(error_classes),
            }
        )

    excluded_sha256 = _write_csv_atomic(
        output_root / "excluded_series.csv",
        (
            "split",
            "study_uid",
            "series_uid",
            "unsafe_instances",
            "unsafe_reasons",
            "error_classes",
        ),
        excluded_rows,
    )
    instances_sha256 = _write_csv_atomic(
        output_root / "unsafe_instances.csv",
        (
            "split",
            "study_uid",
            "series_uid",
            "sop_uid",
            "relative_path",
            "unsafe_reasons",
            "error_class",
            "error_message",
        ),
        unsafe_instances,
    )

    reason_counts: Counter[str] = Counter()
    error_class_counts: Counter[str] = Counter()
    for instance in unsafe_instances:
        reason_counts.update(str(instance["unsafe_reasons"]).split(";"))
        if instance["error_class"]:
            error_class_counts[str(instance["error_class"])] += 1
    affected_studies = {(split, study_uid) for split, study_uid, _ in excluded_keys}
    by_split: dict[str, dict[str, int]] = {}
    for split in SPLIT_SERIES_FILES:
        split_series = {item for item in metadata_series if item[0] == split}
        split_excluded = {item for item in excluded_keys if item[0] == split}
        split_studies = {item for item in study_series_counts if item[0] == split}
        by_split[split] = {
            "attempted_instances": split_attempts[split],
            "unsafe_instances": sum(
                item["split"] == split for item in unsafe_instances
            ),
            "total_series": len(split_series),
            "excluded_series": len(split_excluded),
            "eligible_series": len(split_series - split_excluded),
            "studies": len(split_studies),
            "affected_studies": sum(item[0] == split for item in affected_studies),
        }

    status = "pass" if not any(checks.values()) else "fail"
    report = {
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "privacy": (
            "Aggregate pixel-integrity statistics only; paths, study/series/SOP "
            "identifiers, and per-instance errors remain in external manifests."
        ),
        "source": {
            "audit_mode": "all",
            "audit_status": source_summary.get("status"),
            "manifest_fingerprint": source_config.get("manifest_fingerprint"),
        },
        "counts": {
            "attempted_instances": attempted,
            "decoded_instances": decoded,
            "safe_instances": attempted - len(unsafe_instances),
            "unsafe_instances": len(unsafe_instances),
            "unsafe_instance_fraction": len(unsafe_instances) / attempted,
            "total_series": len(metadata_series),
            "excluded_series": len(excluded_keys),
            "eligible_series": len(metadata_series - excluded_keys),
            "affected_studies": len(affected_studies),
        },
        "by_split": by_split,
        "unsafe_reason_counts": dict(sorted(reason_counts.items())),
        "error_class_counts": dict(sorted(error_class_counts.items())),
        "training_policy": {
            "unit": "entire_series",
            "action": (
                "Exclude every series containing any decode error, nonfinite "
                "array, or header-shape mismatch before sampling or training."
            ),
            "slice_skipping": False,
            "pixel_repair_or_reshape": False,
            "require_at_least_one_eligible_series_per_study": True,
            "external_artifacts": [
                "excluded_series.csv",
                "unsafe_instances.csv",
            ],
        },
        "external_manifest_sha256": {
            "excluded_series.csv": excluded_sha256,
            "unsafe_instances.csv": instances_sha256,
        },
        "checks": checks,
    }
    _write_json_atomic(report_output, report)
    if status != "pass":
        raise PixelIntegrityError(
            "Pixel-integrity exclusion policy failed its safety checks"
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data"))
    project_root = Path(
        os.environ.get("RSNA_PROJECT_ROOT", "/home/shaurya/rsna-project")
    )
    parser = argparse.ArgumentParser(
        description=(
            "Derive strict series exclusions from an exhaustive DICOM pixel audit."
        )
    )
    parser.add_argument(
        "--competition-root", type=Path, default=data_root / "raw" / "competition"
    )
    parser.add_argument(
        "--audit-root", type=Path, default=data_root / "manifests" / "dicom_audit_v1"
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=data_root / "manifests" / "pixel_integrity_v1",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=project_root
        / "reports"
        / "dicom_audit"
        / "pixel_integrity_summary.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = derive_pixel_integrity_policy(
            args.competition_root,
            args.audit_root,
            args.project_root,
            args.output_root,
            args.report_output,
        )
        print(
            f"Pixel-integrity policy {report['status']}: "
            f"{report['counts']['unsafe_instances']:,} unsafe instance(s), "
            f"{report['counts']['excluded_series']:,} excluded series, "
            f"{report['counts']['eligible_series']:,} eligible series",
            flush=True,
        )
        return 0
    except (json.JSONDecodeError, OSError, PixelIntegrityError, ValueError) as error:
        print(f"ERROR: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
