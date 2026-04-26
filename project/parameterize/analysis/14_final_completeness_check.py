"""Entry point for the final completeness check."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from project.parameterize.analysis.final_completeness_check_analysis import main


if __name__ == "__main__":
    main()
