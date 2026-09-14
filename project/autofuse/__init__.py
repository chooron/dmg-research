"""Experiment-layer scaffolding for FUSE-78 AutoFuse studies."""

import sys
from pathlib import Path

_autofuse_dir = str(Path(__file__).resolve().parent)
if _autofuse_dir not in sys.path:
    sys.path.insert(0, _autofuse_dir)

from . import dfuse

if "dfuse" not in sys.modules:
    sys.modules["dfuse"] = dfuse
    for _sub in ("spec", "kernel", "batched", "runtime", "export_spec"):
        if hasattr(dfuse, _sub):
            sys.modules[f"dfuse.{_sub}"] = getattr(dfuse, _sub)

from .metrics import kgecomp
from .protocol import ExperimentProtocol
from .evaluator import evaluate_one

__all__ = ["ExperimentProtocol", "evaluate_one", "kgecomp", "dfuse"]
