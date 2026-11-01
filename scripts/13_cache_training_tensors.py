"""Build the full-cohort 2.5D training tensor cache."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsna.preprocessing_cache import main

if __name__ == "__main__":
    raise SystemExit(main())
