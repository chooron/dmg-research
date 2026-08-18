#!/usr/bin/env python3
"""Run the frozen hydrological-model quality gate.

Default mode runs only the canonical model-integrity tests.  ``--full`` also
runs the independent paper-style optimizer tests.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from tests.model_registry import CANONICAL_MODEL_TEST_FILES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="run every pytest file, including paper-style optimizer tests",
    )
    args = parser.parse_args()
    command = [sys.executable, "-m", "pytest", "-q"]
    if not args.full:
        command.extend(CANONICAL_MODEL_TEST_FILES)
    else:
        # Restrict full mode to the configured pytest test root.  Several
        # historical scripts under scripts/ intentionally start with
        # ``test_`` but are exploratory reports, not release tests.
        command.append("tests")
    print("Running:", " ".join(command), flush=True)
    return subprocess.call(command, cwd=PROJECT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
