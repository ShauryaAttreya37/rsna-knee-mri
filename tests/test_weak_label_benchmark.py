from __future__ import annotations

import csv
from pathlib import Path

import pytest

from rsna.submission import ID_COLUMN, TARGET_COLUMNS
from rsna.weak_label_benchmark import (
    WEAK_LABEL_COLUMNS,
    WeakLabelBenchmarkError,
    benchmark_candidates,
    write_equal_mean_ensemble,
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[ID_COLUMN, *TARGET_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)


def _row(uid: str, value: float | str) -> dict[str, object]:
    return {ID_COLUMN: uid, **{target: value for target in TARGET_COLUMNS}}


def test_benchmark_ranks_perfect_candidate_above_inverse(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    perfect = tmp_path / "perfect.csv"
    inverse = tmp_path / "inverse.csv"
    _write(train, [_row("a", 0), _row("b", 1), _row("weak", "")])
    _write(perfect, [_row("a", 0.1), _row("b", 0.9), _row("weak", 0.5)])
    _write(inverse, [_row("a", 0.9), _row("b", 0.1), _row("weak", 0.5)])

    result = benchmark_candidates(
        train, [("perfect", perfect), ("inverse", inverse)]
    )

    assert result["gold_studies"] == 2
    assert result["candidates"]["perfect"]["macro_auc"] == pytest.approx(1.0)
    assert result["candidates"]["inverse"]["macro_auc"] == pytest.approx(0.0)
    assert result["best_by_observed_gold_macro_auc"] == "perfect"


def test_benchmark_rejects_candidate_with_low_gold_coverage(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    candidate = tmp_path / "candidate.csv"
    _write(
        train,
        [_row(str(index), index % 2) for index in range(20)],
    )
    _write(candidate, [_row(str(index), index % 2) for index in range(18)])

    with pytest.raises(WeakLabelBenchmarkError, match="covers only 18/20 gold"):
        benchmark_candidates(train, [("candidate", candidate)])


def test_write_equal_mean_ensemble_uses_available_sources(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    third = tmp_path / "third.csv"
    output = tmp_path / "weak_labels.csv"
    _write(train, [_row("a", 0), _row("b", "")])
    _write(first, [_row("a", 0.0), _row("b", 0.3)])
    _write(second, [_row("a", 0.6), _row("b", 0.9)])
    _write(third, [_row("a", 0.3)])

    summary = write_equal_mean_ensemble(
        train,
        [("first", first), ("second", second), ("third", third)],
        output,
        minimum_sources=2,
    )

    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == [ID_COLUMN, *WEAK_LABEL_COLUMNS]
    assert float(rows[0][WEAK_LABEL_COLUMNS[0]]) == pytest.approx(0.3)
    assert float(rows[1][WEAK_LABEL_COLUMNS[0]]) == pytest.approx(0.6)
    assert summary["minimum_sources_observed"] == 2
    assert summary["maximum_sources_observed"] == 3
