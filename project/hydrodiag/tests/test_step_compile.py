"""Test that each step kernel can be torch.compile(fullgraph=True).

This is the most critical test — it verifies that individual model step
functions can be compiled without errors, not that the full model compiles.
"""

import torch
import pytest

# Increase compile cache for grad/no-grad mode switches across tests
torch._dynamo.config.cache_size_limit = 32

from models.hbv import _hbv_step
from models.gr4j import _gr4j_step
from models.xaj import _xaj_step
from models.cemaneige import _cemaneige_hyst_step, _cemaneige_step
from models.gr4j import GR4J_UH1_MAX, GR4J_UH2_MAX


BATCH = 3
NEARZERO = 1e-8


def make_hbv_inputs(batch, device, dtype):
    return (
        torch.rand(batch, device=device, dtype=dtype) * 10.0,       # precip_t
        torch.randn(batch, device=device, dtype=dtype) * 5.0,       # temp_t
        torch.rand(batch, device=device, dtype=dtype) * 5.0,        # pet_t
        torch.rand(batch, device=device, dtype=dtype) * 5.0,        # SNOWPACK
        torch.rand(batch, device=device, dtype=dtype) * 3.0,        # MELTWATER
        torch.rand(batch, device=device, dtype=dtype) * 100.0,      # SM
        torch.rand(batch, device=device, dtype=dtype) * 10.0,       # SUZ
        torch.rand(batch, device=device, dtype=dtype) * 20.0,       # SLZ
        torch.rand(batch, device=device, dtype=dtype) * 1.0,        # parTT
        torch.rand(batch, device=device, dtype=dtype) * 5.0 + 1.0,  # parCFMAX
        torch.rand(batch, device=device, dtype=dtype) * 0.05,       # parCFR
        torch.rand(batch, device=device, dtype=dtype) * 0.1,        # parCWH
        torch.rand(batch, device=device, dtype=dtype) * 200.0 + 100.0,  # parFC
        torch.rand(batch, device=device, dtype=dtype) * 3.0 + 1.0,  # parBETA
        torch.rand(batch, device=device, dtype=dtype) * 0.5 + 0.3,  # parLP
        torch.rand(batch, device=device, dtype=dtype) * 5.0,        # parPERC
        torch.rand(batch, device=device, dtype=dtype) * 50.0,       # parUZL
        torch.rand(batch, device=device, dtype=dtype) * 0.5,        # parK0
        torch.rand(batch, device=device, dtype=dtype) * 0.3,        # parK1
        torch.rand(batch, device=device, dtype=dtype) * 0.1,        # parK2
        NEARZERO,
    )


def make_gr4j_inputs(batch, device, dtype):
    return (
        torch.rand(batch, device=device, dtype=dtype) * 10.0,       # precip_t
        torch.rand(batch, device=device, dtype=dtype) * 5.0,        # pet_t
        torch.rand(batch, device=device, dtype=dtype) * 300.0,      # s_prod
        torch.rand(batch, device=device, dtype=dtype) * 300.0,      # s_route
        torch.rand(batch, GR4J_UH1_MAX, device=device, dtype=dtype),  # uh1_buf
        torch.rand(batch, GR4J_UH2_MAX, device=device, dtype=dtype),  # uh2_buf
        torch.rand(batch, GR4J_UH1_MAX, device=device, dtype=dtype),  # uh1_ord
        torch.rand(batch, GR4J_UH2_MAX, device=device, dtype=dtype),  # uh2_ord
        torch.rand(batch, device=device, dtype=dtype) * 500.0 + 100.0,  # x1
        torch.randn(batch, device=device, dtype=dtype) * 2.0,        # x2
        torch.rand(batch, device=device, dtype=dtype) * 1000.0 + 100.0,  # x3
        NEARZERO,
    )


def make_xaj_inputs(batch, device, dtype):
    return (
        torch.rand(batch, device=device, dtype=dtype) * 10.0,       # precip_t
        torch.rand(batch, device=device, dtype=dtype) * 5.0,        # pet_t
        torch.rand(batch, device=device, dtype=dtype) * 10.0,       # wu
        torch.rand(batch, device=device, dtype=dtype) * 40.0,       # wl
        torch.rand(batch, device=device, dtype=dtype) * 20.0,       # wd
        torch.rand(batch, device=device, dtype=dtype) * 10.0,       # s
        torch.rand(batch, device=device, dtype=dtype) * 0.2,        # fr
        torch.rand(batch, device=device, dtype=dtype) * 3.0,        # qi
        torch.rand(batch, device=device, dtype=dtype) * 3.0,        # qg
        torch.rand(batch, device=device, dtype=dtype) * 0.5 + 0.8,  # k
        torch.rand(batch, device=device, dtype=dtype) * 0.5 + 0.1,  # b
        torch.rand(batch, device=device, dtype=dtype) * 0.1,        # im
        torch.rand(batch, device=device, dtype=dtype) * 20.0 + 5.0,  # um
        torch.rand(batch, device=device, dtype=dtype) * 100.0 + 20.0,  # lm
        torch.rand(batch, device=device, dtype=dtype) * 100.0 + 20.0,  # dm
        torch.rand(batch, device=device, dtype=dtype) * 0.1 + 0.05,  # c
        torch.rand(batch, device=device, dtype=dtype) * 50.0 + 10.0,  # sm
        torch.rand(batch, device=device, dtype=dtype) * 1.0 + 0.5,  # ex
        torch.rand(batch, device=device, dtype=dtype) * 0.3,        # ki
        torch.rand(batch, device=device, dtype=dtype) * 0.3,        # kg
        torch.rand(batch, device=device, dtype=dtype) * 0.5 + 0.3,  # ci
        torch.rand(batch, device=device, dtype=dtype) * 0.05 + 0.9,  # cg
        NEARZERO,
    )


def make_cemaneige_inputs(batch, device, dtype):
    return (
        torch.rand(batch, device=device, dtype=dtype) * 10.0,       # precip_t
        torch.randn(batch, device=device, dtype=dtype) * 10.0,      # temp_t
        torch.rand(batch, device=device, dtype=dtype) * 5.0,        # G
        torch.rand(batch, device=device, dtype=dtype) * (-5.0),     # eTG (<=0)
        torch.rand(batch, device=device, dtype=dtype) * 0.5,        # ctg
        torch.rand(batch, device=device, dtype=dtype) * 5.0 + 1.0,  # kf
        torch.rand(batch, device=device, dtype=dtype) * 500.0,      # G threshold
        NEARZERO,
    )


def make_cemaneige_hyst_inputs(batch, device, dtype):
    return (
        torch.rand(batch, device=device, dtype=dtype) * 10.0,
        torch.randn(batch, device=device, dtype=dtype) * 10.0,
        torch.rand(batch, device=device, dtype=dtype) * 5.0,
        torch.rand(batch, device=device, dtype=dtype) * (-5.0),
        torch.rand(batch, device=device, dtype=dtype) * 0.5,
        torch.rand(batch, device=device, dtype=dtype) * 5.0,
        torch.rand(batch, device=device, dtype=dtype) * 0.5,
        torch.rand(batch, device=device, dtype=dtype) * 5.0 + 1.0,
        torch.rand(batch, device=device, dtype=dtype) * 100.0 + 1.0,
        torch.rand(batch, device=device, dtype=dtype) * 0.2,
        torch.rand(batch, device=device, dtype=dtype) * 500.0,
        NEARZERO,
    )


def run_compile_test(step_fn, make_inputs, name, device, dtype):
    """Verify that a step function compiles with fullgraph=True and produces
    correct shapes and finite outputs."""
    inputs = make_inputs(BATCH, device, dtype)

    # First run eager
    with torch.no_grad():
        eager_out = step_fn(*inputs)

    # Compile with fullgraph=True
    compiled_step = torch.compile(step_fn, fullgraph=True)

    with torch.no_grad():
        compiled_out = compiled_step(*inputs)

    # Verify shapes match (first output is q_t [batch])
    assert eager_out[0].shape == (BATCH,), f"[{name}] Eager output shape {eager_out[0].shape} != expected ({BATCH},)"
    assert compiled_out[0].shape == (BATCH,), f"[{name}] Compiled output shape {compiled_out[0].shape} != expected ({BATCH},)"

    # Verify finite
    assert torch.isfinite(eager_out[0]).all(), f"[{name}] Eager output contains NaN/Inf"
    assert torch.isfinite(compiled_out[0]).all(), f"[{name}] Compiled output contains NaN/Inf"

    # Verify eager vs compiled match
    for i, (eo, co) in enumerate(zip(eager_out, compiled_out)):
        assert torch.allclose(eo, co, atol=1e-5, rtol=1e-4), (
            f"[{name}] Mismatch at output {i}: max_diff={torch.abs(eo - co).max().item():.2e}"
        )

    # Verify second run is consistent
    with torch.no_grad():
        compiled_out2 = compiled_step(*inputs)
    assert torch.allclose(compiled_out[0], compiled_out2[0], atol=1e-5, rtol=1e-4), (
        f"[{name}] Compiled step not deterministic"
    )

    return True


class TestStepCompile:
    """Compile each step kernel with fullgraph=True and verify correctness."""

    @pytest.mark.parametrize("device_str,dtype", [
        ("cpu", torch.float32),
        ("cpu", torch.float64),
    ])
    def test_hbv_step_compile(self, device_str, dtype):
        device = torch.device(device_str)
        assert run_compile_test(_hbv_step, make_hbv_inputs, "HBV", device, dtype)

    @pytest.mark.parametrize("device_str,dtype", [
        ("cpu", torch.float32),
        ("cpu", torch.float64),
    ])
    def test_gr4j_step_compile(self, device_str, dtype):
        device = torch.device(device_str)
        assert run_compile_test(_gr4j_step, make_gr4j_inputs, "GR4J", device, dtype)

    @pytest.mark.parametrize("device_str,dtype", [
        ("cpu", torch.float32),
        ("cpu", torch.float64),
    ])
    def test_xaj_step_compile(self, device_str, dtype):
        device = torch.device(device_str)
        assert run_compile_test(_xaj_step, make_xaj_inputs, "XAJ", device, dtype)

    @pytest.mark.parametrize("device_str,dtype", [
        ("cpu", torch.float32),
        ("cpu", torch.float64),
    ])
    def test_cemaneige_step_compile(self, device_str, dtype):
        device = torch.device(device_str)
        assert run_compile_test(_cemaneige_step, make_cemaneige_inputs, "CemaNeige", device, dtype)

    @pytest.mark.parametrize("device_str,dtype", [
        ("cpu", torch.float32),
        ("cpu", torch.float64),
    ])
    def test_cemaneige_hyst_step_compile(self, device_str, dtype):
        device = torch.device(device_str)
        assert run_compile_test(
            _cemaneige_hyst_step,
            make_cemaneige_hyst_inputs,
            "CemaNeigeHyst",
            device,
            dtype,
        )
