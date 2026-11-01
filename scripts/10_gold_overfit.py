"""Run the same-cohort gold-label image-training overfit test."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsna.gold_overfit import main

if __name__ == "__main__":
    raise SystemExit(main())
