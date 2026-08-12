from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.euler_convergence_utils import run_euler_convergence_validation


def main() -> int:
    run_euler_convergence_validation(write_outputs=True)
    print("Wrote Euler convergence validation artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
