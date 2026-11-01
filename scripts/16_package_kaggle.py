"""Create the offline five-fold Kaggle input on the training VM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna.kaggle_package import main

if __name__ == "__main__":
    main()
