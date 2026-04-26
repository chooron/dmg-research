"""Generate the full figure2 suite."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from project.parameterize.figure2.scripts._helpers import run_all


if __name__ == "__main__":
    run_all()
