"""Audit physical pixel spacing and field of view in the DICOM manifest."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsna.physical_geometry_audit import main

if __name__ == "__main__":
    raise SystemExit(main())
