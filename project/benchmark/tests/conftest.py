"""Shared pytest bootstrap for project/benchmark tests.

Mirrors the sys.path setup used by the benchmark runners themselves
(see dmotpy/conftest.py for the dmotpy-side convention).
"""

import sys
from pathlib import Path

_benchmark_root = Path(__file__).resolve().parents[1]
_repo_root = _benchmark_root.parents[1]
for _p in (_repo_root, _benchmark_root, _benchmark_root / "src", _benchmark_root / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))