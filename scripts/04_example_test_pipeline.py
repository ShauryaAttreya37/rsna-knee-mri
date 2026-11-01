"""Run the full example-test DICOM loading and preprocessing smoke test."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsna.example_test_pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
