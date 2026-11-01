"""Analyze orientation variation in the completed DICOM manifest."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsna.orientation_audit import main

if __name__ == "__main__":
    raise SystemExit(main())
