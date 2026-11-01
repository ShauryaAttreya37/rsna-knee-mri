"""Run the official-metric and constant-submission smoke test."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsna.submission_smoke import main

if __name__ == "__main__":
    raise SystemExit(main())
