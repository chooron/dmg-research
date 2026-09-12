"""Experiment-layer scaffolding for FUSE-78 AutoFuse studies."""

from .metrics import kgecomp
from .protocol import ExperimentProtocol
from .evaluator import evaluate_one

__all__ = ["ExperimentProtocol", "evaluate_one", "kgecomp"]
