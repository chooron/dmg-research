"""Entry point for Results 3.3.2/3.3.3/3.4.2/3.4.3 analyses."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from project.parameterize.analysis.results332_333_342_343_analysis import main


if __name__ == "__main__":
    main()
