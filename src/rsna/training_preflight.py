"""Fail-fast checks for the fold-0 GPU training gate."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """One human-readable preflight result."""

    name: str
    status: str
    detail: str


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """Complete training-readiness result."""

    checks: tuple[PreflightCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.status != "fail" for check in self.checks)


PINNED_DISTRIBUTIONS = {
    "numpy": "2.5.2",
    "pydicom": "3.0.2",
    "scikit-learn": "1.8.0",
    "torch": "2.7.0",
}


def _check(name: str, passed: bool, success: str, failure: str) -> PreflightCheck:
    return PreflightCheck(
        name, "pass" if passed else "fail", success if passed else failure
    )


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _base_path_checks(
    project_root: Path,
    competition_root: Path,
    folds_path: Path,
    exclusions_path: Path,
    cache_root: Path,
    output_root: Path,
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    checks.append(
        _check(
            "project root",
            project_root.is_dir(),
            "project directory is present",
            f"missing directory: {project_root}",
        )
    )
    required_inputs = {
        "train metadata": competition_root / "train.csv",
        "series metadata": competition_root / "train_series.csv",
        "training images": competition_root / "train_series",
        "gold folds": folds_path,
        "pixel exclusions": exclusions_path,
    }
    for name, path in required_inputs.items():
        expected = path.is_dir() if name == "training images" else path.is_file()
        checks.append(
            _check(name, expected, "present", f"missing required input: {path}")
        )

    layout_is_safe = not any(
        (
            _within(external, project_root)
            or _within(project_root, external)
            or _within(external, competition_root)
            or _within(competition_root, external)
        )
        for external in (cache_root, output_root)
    )
    layout_is_safe = (
        layout_is_safe
        and not _within(cache_root, output_root)
        and not _within(output_root, cache_root)
    )
    checks.append(
        _check(
            "artifact isolation",
            layout_is_safe,
            "raw data, project, cache, and model roots are isolated",
            "cache/model roots overlap the project, raw data, or each other",
        )
    )
    return checks


def _dependency_checks() -> tuple[list[PreflightCheck], bool]:
    checks: list[PreflightCheck] = []
    dependencies_ready = True
    for distribution, expected_version in PINNED_DISTRIBUTIONS.items():
        try:
            installed_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed_version = None
        version_matches = (
            installed_version is not None
            and installed_version.split("+", maxsplit=1)[0] == expected_version
        )
        dependencies_ready = dependencies_ready and version_matches
        checks.append(
            _check(
                f"dependency {distribution}",
                version_matches,
                f"{installed_version} matches the training lock",
                (
                    f"expected {expected_version}, found "
                    f"{installed_version or 'not installed'}"
                ),
            )
        )
    return checks, dependencies_ready


def _cuda_checks(minimum_gpu_memory_gib: float) -> tuple[list[PreflightCheck], bool]:
    checks: list[PreflightCheck] = []
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as error:
        checks.append(PreflightCheck("NVIDIA driver", "fail", str(error)))
    else:
        driver_ready = completed.returncode == 0 and bool(completed.stdout.strip())
        checks.append(
            _check(
                "NVIDIA driver",
                driver_ready,
                completed.stdout.strip(),
                (completed.stderr.strip() or "nvidia-smi returned no GPU")[:500],
            )
        )

    try:
        import torch
    except (ImportError, OSError) as error:
        checks.append(PreflightCheck("PyTorch CUDA", "fail", str(error)))
        return checks, False

    cuda_available = bool(torch.cuda.is_available()) and torch.cuda.device_count() > 0
    checks.append(
        _check(
            "PyTorch CUDA",
            cuda_available,
            f"CUDA {torch.version.cuda}; {torch.cuda.device_count()} device(s)",
            f"CUDA unavailable in PyTorch {torch.__version__}",
        )
    )
    if not cuda_available:
        return checks, False

    try:
        device = torch.device("cuda", 0)
        properties = torch.cuda.get_device_properties(device)
        total_memory_gib = properties.total_memory / (1024**3)
        tensor = torch.arange(256, device=device, dtype=torch.float32).reshape(16, 16)
        checksum = float((tensor @ tensor.T).sum().item())
        torch.cuda.synchronize(device)
        smoke_passed = math.isfinite(checksum)
    except (RuntimeError, ValueError) as error:
        checks.append(PreflightCheck("CUDA computation", "fail", str(error)))
        return checks, False

    checks.append(
        _check(
            "CUDA computation",
            smoke_passed,
            f"matrix multiply passed on {properties.name}",
            "CUDA matrix multiplication returned a nonfinite result",
        )
    )
    checks.append(
        _check(
            "GPU memory",
            total_memory_gib >= minimum_gpu_memory_gib,
            f"{total_memory_gib:.1f} GiB available on device 0",
            (
                f"{total_memory_gib:.1f} GiB is below the "
                f"{minimum_gpu_memory_gib:.1f} GiB minimum"
            ),
        )
    )
    if total_memory_gib < 20.0:
        checks.append(
            PreflightCheck(
                "baseline GPU capacity",
                "warn",
                "below the planned 20 GiB target; the overfit gate may still run",
            )
        )
    return checks, smoke_passed and total_memory_gib >= minimum_gpu_memory_gib


def _dataset_check(
    competition_root: Path,
    folds_path: Path,
    exclusions_path: Path,
    fold: int,
) -> PreflightCheck:
    try:
        from rsna.training_data import TrainingDataError, load_gold_training_cohort
    except (ImportError, OSError) as error:
        return PreflightCheck("fold cohort", "fail", str(error))
    try:
        cohort = load_gold_training_cohort(
            competition_root, folds_path, exclusions_path, fold=fold
        )
        missing_series = sum(
            not (
                competition_root / "train_series" / study.study_uid / series.series_uid
            ).is_dir()
            for study in cohort.studies
            for series in study.series
        )
        if missing_series:
            return PreflightCheck(
                "fold cohort",
                "fail",
                f"{missing_series} eligible series directories are missing",
            )
    except (OSError, ValueError, TrainingDataError) as error:
        return PreflightCheck("fold cohort", "fail", str(error))
    return PreflightCheck(
        "fold cohort",
        "pass",
        (
            f"fold {fold}: {len(cohort.studies)} class-complete studies, "
            f"{sum(len(study.series) for study in cohort.studies)} eligible series"
        ),
    )


def run_training_preflight(
    project_root: Path,
    competition_root: Path,
    folds_path: Path,
    exclusions_path: Path,
    cache_root: Path,
    output_root: Path,
    *,
    fold: int = 0,
    device: str = "cuda",
    minimum_free_gib: float = 100.0,
    minimum_gpu_memory_gib: float = 8.0,
    prepare_directories: bool = False,
) -> PreflightResult:
    """Validate the environment without reading DICOM pixel data."""
    roots = [
        project_root,
        competition_root,
        folds_path,
        exclusions_path,
        cache_root,
        output_root,
    ]
    (
        project_root,
        competition_root,
        folds_path,
        exclusions_path,
        cache_root,
        output_root,
    ) = [path.expanduser().resolve() for path in roots]
    if fold < 0:
        raise ValueError("fold must be nonnegative")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if minimum_free_gib < 0.0 or minimum_gpu_memory_gib < 0.0:
        raise ValueError("minimum capacities must be nonnegative")

    checks = _base_path_checks(
        project_root,
        competition_root,
        folds_path,
        exclusions_path,
        cache_root,
        output_root,
    )
    python_ready = sys.version_info[:2] == (3, 12)
    checks.append(
        _check(
            "Python",
            python_ready,
            f"{sys.version.split()[0]} matches the Python 3.12 environment",
            f"expected Python 3.12, found {sys.version.split()[0]}",
        )
    )
    dependency_checks, dependencies_ready = _dependency_checks()
    checks.extend(dependency_checks)

    safe_layout = next(
        check.status == "pass" for check in checks if check.name == "artifact isolation"
    )
    if prepare_directories and safe_layout:
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
            output_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            checks.append(PreflightCheck("artifact directories", "fail", str(error)))
        else:
            checks.append(
                PreflightCheck(
                    "artifact directories", "pass", "cache and model roots are ready"
                )
            )
    for name, path in (("cache writes", cache_root), ("model writes", output_root)):
        writable_parent = _nearest_existing_parent(path)
        checks.append(
            _check(
                name,
                writable_parent.is_dir() and os.access(writable_parent, os.W_OK),
                "destination is writable",
                f"destination is not writable: {writable_parent}",
            )
        )

    storage_probe = _nearest_existing_parent(cache_root)
    try:
        free_gib = shutil.disk_usage(storage_probe).free / (1024**3)
    except OSError as error:
        checks.append(PreflightCheck("free storage", "fail", str(error)))
    else:
        checks.append(
            _check(
                "free storage",
                free_gib >= minimum_free_gib,
                f"{free_gib:.1f} GiB free at the cache destination",
                f"{free_gib:.1f} GiB is below the {minimum_free_gib:.1f} GiB minimum",
            )
        )

    required_inputs_ready = all(
        check.status == "pass"
        for check in checks
        if check.name
        in {
            "train metadata",
            "series metadata",
            "training images",
            "gold folds",
            "pixel exclusions",
        }
    )
    if dependencies_ready and required_inputs_ready:
        checks.append(
            _dataset_check(competition_root, folds_path, exclusions_path, fold)
        )
    else:
        checks.append(
            PreflightCheck(
                "fold cohort",
                "fail",
                "cannot validate until pinned dependencies and inputs are present",
            )
        )

    if device == "cuda":
        cuda_results, _ = _cuda_checks(minimum_gpu_memory_gib)
        checks.extend(cuda_results)
    else:
        checks.append(
            PreflightCheck(
                "training device",
                "warn",
                "CPU mode selected; use CUDA for the actual overfit gate",
            )
        )
    return PreflightResult(tuple(checks))


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("RSNA_DATA_ROOT", "/home/shaurya/rsna-data"))
    project_root = Path(
        os.environ.get("RSNA_PROJECT_ROOT", "/home/shaurya/rsna-project")
    )
    parser = argparse.ArgumentParser(
        description="Check GPU, dependencies, storage, manifests, and fold-0 inputs."
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--competition-root", type=Path, default=data_root / "raw" / "competition"
    )
    parser.add_argument(
        "--folds-path",
        type=Path,
        default=data_root / "manifests" / "gold_folds_v1" / "gold_folds.csv",
    )
    parser.add_argument(
        "--exclusions-path",
        type=Path,
        default=data_root / "manifests" / "pixel_integrity_v1" / "excluded_series.csv",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=data_root / "processed" / "gold_overfit_cache_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=data_root / "models" / "gold_overfit_v1",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--minimum-free-gib", type=float, default=100.0)
    parser.add_argument("--minimum-gpu-memory-gib", type=float, default=8.0)
    parser.add_argument("--prepare-directories", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_training_preflight(
            args.project_root,
            args.competition_root,
            args.folds_path,
            args.exclusions_path,
            args.cache_root,
            args.output_root,
            fold=args.fold,
            device=args.device,
            minimum_free_gib=args.minimum_free_gib,
            minimum_gpu_memory_gib=args.minimum_gpu_memory_gib,
            prepare_directories=args.prepare_directories,
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", flush=True)
        return 2

    for check in result.checks:
        print(f"[{check.status.upper():4}] {check.name}: {check.detail}", flush=True)
    print(
        "Training preflight PASS" if result.ready else "Training preflight FAIL",
        flush=True,
    )
    return 0 if result.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
