"""Small regression gate for the repository-local dPL audit adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ADAPTER_PATH = (
    Path(__file__).resolve().parents[1]
    / "validation_results"
    / "hydro_dpl_gradient_audit_core"
    / "gradient_adapter.py"
)


def _load_adapter():
    spec = importlib.util.spec_from_file_location("hydro_dpl_audit_adapter", ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_core_sources_have_finite_applicable_gradients():
    adapter = _load_adapter()
    for model_name in adapter.AVAILABLE_MODELS:
        case = adapter.build_case(
            device="cpu",
            dtype=torch.float32,
            seed=0,
            config={"model": model_name, "case": "realistic_mixed", "length": 6},
        )
        assert torch.isfinite(case["loss"]), model_name
        case["loss"].backward()
        for target_name, target in case["targets"].items():
            if not case["applicability"].get(target_name, True):
                continue
            assert target.grad is not None, f"{model_name}: missing gradient for {target_name}"
            assert torch.isfinite(target.grad).all(), f"{model_name}: non-finite gradient for {target_name}"
