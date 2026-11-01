from pathlib import Path

from rsna.training_preflight import _base_path_checks, _nearest_existing_parent


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_base_path_checks_accept_isolated_training_layout(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    competition_root = tmp_path / "data" / "raw" / "competition"
    folds_path = tmp_path / "data" / "manifests" / "folds.csv"
    exclusions_path = tmp_path / "data" / "manifests" / "exclusions.csv"
    cache_root = tmp_path / "data" / "processed" / "cache"
    output_root = tmp_path / "data" / "models" / "run"
    project_root.mkdir()
    (competition_root / "train_series").mkdir(parents=True)
    _touch(competition_root / "train.csv")
    _touch(competition_root / "train_series.csv")
    _touch(folds_path)
    _touch(exclusions_path)

    checks = _base_path_checks(
        project_root,
        competition_root,
        folds_path,
        exclusions_path,
        cache_root,
        output_root,
    )

    assert all(check.status == "pass" for check in checks)


def test_base_path_checks_reject_cache_inside_project(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    competition_root = tmp_path / "data" / "raw" / "competition"
    folds_path = tmp_path / "data" / "manifests" / "folds.csv"
    exclusions_path = tmp_path / "data" / "manifests" / "exclusions.csv"
    cache_root = project_root / "cache"
    output_root = tmp_path / "data" / "models" / "run"
    project_root.mkdir()

    checks = _base_path_checks(
        project_root,
        competition_root,
        folds_path,
        exclusions_path,
        cache_root,
        output_root,
    )

    isolation = next(check for check in checks if check.name == "artifact isolation")
    assert isolation.status == "fail"


def test_nearest_existing_parent_walks_up_from_missing_path(tmp_path: Path) -> None:
    destination = tmp_path / "a" / "b" / "c"

    assert _nearest_existing_parent(destination) == tmp_path
