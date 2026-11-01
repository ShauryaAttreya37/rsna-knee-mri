"""Run the read-only RSNA DICOM inventory and pixel-decoder audit."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsna.dicom_audit import main

if __name__ == "__main__":
    raise SystemExit(main())
