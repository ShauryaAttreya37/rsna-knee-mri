"""Create strict series exclusions from exhaustive pixel-decode results."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsna.pixel_integrity import main

if __name__ == "__main__":
    raise SystemExit(main())
