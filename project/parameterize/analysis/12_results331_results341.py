"""Entry point for Results 3.3.1 and 3.4.1 analysis."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from project.parameterize.analysis.results331_results341_analysis import main


if __name__ == "__main__":
    main()
