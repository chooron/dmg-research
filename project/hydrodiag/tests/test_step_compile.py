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
from models.xaj import _xaj_step, _xaj_step_compact
from models.simhyd import _simhyd_step, _simhyd_step_compact
from models.cemaneige import _cemaneige_hyst_step, _cemaneige_step
from models.composed import (
    _cemaneige_gr4j_fused_step,
    _cemaneige_xaj_fused_step,
    _cemaneige_simhyd_fused_step,
)
from models.tgd2 import tgd2_step
from models.precip_delay import _precip_delay_step
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


def make_xaj_compact_inputs(batch, device, dtype):
    xaj_in = make_xaj_inputs(batch, device, dtype)
    # _xaj_step_compact takes: precip_t, pet_t, wu, wl, wd, s, fr, qi, qg,
    # k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg, nearzero,
    # wm, wmm, ms, one_minus_im, one_minus_ki_kg
    precip_t, pet_t, wu, wl, wd, s, fr, qi, qg, k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg, nz = xaj_in
    wm = um + lm + dm
    wmm = wm * (1.0 + b)
    ms = sm * (1.0 + ex)
    one_minus_im = 1.0 - im
    one_minus_ki_kg = 1.0 - ki - kg
    return (
        precip_t, pet_t, wu, wl, wd, s, fr, qi, qg,
        k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg, nz,
        wm, wmm, ms, one_minus_im, one_minus_ki_kg
    )


def make_simhyd_inputs(batch, device, dtype):
    return (
        torch.rand(batch, device=device, dtype=dtype) * 10.0,
        torch.rand(batch, device=device, dtype=dtype) * 5.0,
        torch.rand(batch, device=device, dtype=dtype) * 100.0,
        torch.rand(batch, device=device, dtype=dtype) * 10.0,
        torch.rand(batch, device=device, dtype=dtype) * 5.0 + 0.1,
        torch.rand(batch, device=device, dtype=dtype) * 200.0 + 10.0,
        torch.rand(batch, device=device, dtype=dtype) * 5.0,
        torch.rand(batch, device=device, dtype=dtype) * 500.0 + 50.0,
        torch.rand(batch, device=device, dtype=dtype) * 0.5,
        torch.rand(batch, device=device, dtype=dtype) * 0.5,
        torch.rand(batch, device=device, dtype=dtype) * 0.5,
        torch.rand(batch, device=device, dtype=dtype) * 2.0 + 0.5,
        NEARZERO,
    )


def make_tgd2_inputs(batch, device, dtype):
    return (
        torch.rand(batch, device=device, dtype=dtype) * 10.0,
        torch.randn(batch, device=device, dtype=dtype) * 10.0,
        torch.rand(batch, device=device, dtype=dtype) * 5.0,
        torch.rand(batch, device=device, dtype=dtype) * 2.0 + 0.1,
        torch.rand(batch, device=device, dtype=dtype) * 50.0 + 5.0,
    )


def make_precip_delay_inputs(batch, device, dtype):
    return (
        torch.rand(batch, device=device, dtype=dtype) * 10.0,
        torch.rand(batch, device=device, dtype=dtype) * 5.0,
        torch.rand(batch, device=device, dtype=dtype) * 0.5,
        torch.rand(batch, device=device, dtype=dtype) * 10.0 + 0.1,
        NEARZERO,
    )


def make_cn_gr4j_fused_inputs(batch, device, dtype):
    return (
        torch.rand(batch, device=device, dtype=dtype) * 10.0,
        torch.randn(batch, device=device, dtype=dtype) * 10.0,
        torch.rand(batch, device=device, dtype=dtype) * 5.0,
        (torch.rand(batch, device=device, dtype=dtype) * 5.0, torch.rand(batch, device=device, dtype=dtype) * (-5.0)),
        (torch.rand(batch, device=device, dtype=dtype) * 300.0, torch.rand(batch, device=device, dtype=dtype) * 300.0,
         torch.rand(batch, GR4J_UH1_MAX, device=device, dtype=dtype), torch.rand(batch, GR4J_UH2_MAX, device=device, dtype=dtype)),
        (torch.rand(batch, device=device, dtype=dtype) * 0.5, torch.rand(batch, device=device, dtype=dtype) * 5.0 + 1.0, torch.rand(batch, device=device, dtype=dtype) * 500.0),
        (torch.rand(batch, GR4J_UH1_MAX, device=device, dtype=dtype), torch.rand(batch, GR4J_UH2_MAX, device=device, dtype=dtype),
         torch.rand(batch, device=device, dtype=dtype) * 500.0 + 100.0, torch.randn(batch, device=device, dtype=dtype) * 2.0, torch.rand(batch, device=device, dtype=dtype) * 1000.0 + 100.0),
        NEARZERO,
    )


def make_cn_xaj_fused_inputs(batch, device, dtype):
    xaj_in = make_xaj_inputs(batch, device, dtype)
    precip_t, pet_t, wu, wl, wd, s, fr, qi, qg, k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg, nz = xaj_in
    temp_t = torch.randn(batch, device=device, dtype=dtype) * 10.0
    G = torch.rand(batch, device=device, dtype=dtype) * 5.0
    eTG = torch.rand(batch, device=device, dtype=dtype) * (-5.0)
    ctg = torch.rand(batch, device=device, dtype=dtype) * 0.5
    kf = torch.rand(batch, device=device, dtype=dtype) * 5.0 + 1.0
    g_thresh = torch.rand(batch, device=device, dtype=dtype) * 500.0
    return (
        precip_t, temp_t, pet_t,
        (G, eTG),
        (wu, wl, wd, s, fr, qi, qg),
        (ctg, kf, g_thresh),
        (k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg),
        nz,
    )


def make_cn_simhyd_fused_inputs(batch, device, dtype):
    temp_t = torch.randn(batch, device=device, dtype=dtype) * 10.0
    G = torch.rand(batch, device=device, dtype=dtype) * 5.0
    eTG = torch.rand(batch, device=device, dtype=dtype) * (-5.0)
    ctg = torch.rand(batch, device=device, dtype=dtype) * 0.5
    kf = torch.rand(batch, device=device, dtype=dtype) * 5.0 + 1.0
    g_thresh = torch.rand(batch, device=device, dtype=dtype) * 500.0
    precip_t, pet_t, soil, gw, insc, coeff, sq, smsc, sub, crak, k, etmul, nz = make_simhyd_inputs(batch, device, dtype)
    return (
        precip_t, temp_t, pet_t,
        (G, eTG), (soil, gw),
        (ctg, kf, g_thresh),
        (insc, coeff, sq, smsc, sub, crak, k, etmul),
        nz,
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
    atol = 1e-2 if dtype == torch.float32 else 1e-5
    rtol = 1e-2 if dtype == torch.float32 else 1e-4
    for i, (eo, co) in enumerate(zip(eager_out, compiled_out)):
        assert torch.allclose(eo, co, atol=atol, rtol=rtol), (
            f"[{name}] Mismatch at output {i}: max_diff={torch.abs(eo - co).max().item():.2e}"
        )

    # Verify second run is consistent
    with torch.no_grad():
        compiled_out2 = compiled_step(*inputs)
    assert torch.allclose(compiled_out[0], compiled_out2[0], atol=1e-5, rtol=1e-4), (
        f"[{name}] Compiled step not deterministic"
    )

    return True


DEVICES_AND_DTYPES = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
] + ([("cuda", torch.float32), ("cuda", torch.float64)] if torch.cuda.is_available() else [])


class TestStepCompile:
    """Compile each step kernel with fullgraph=True and verify correctness."""

    @pytest.mark.parametrize("device_str,dtype", DEVICES_AND_DTYPES)
    def test_hbv_step_compile(self, device_str, dtype):
        device = torch.device(device_str)
        assert run_compile_test(_hbv_step, make_hbv_inputs, "HBV", device, dtype)

    @pytest.mark.parametrize("device_str,dtype", DEVICES_AND_DTYPES)
    def test_gr4j_step_compile(self, device_str, dtype):
        device = torch.device(device_str)
        assert run_compile_test(_gr4j_step, make_gr4j_inputs, "GR4J", device, dtype)

    @pytest.mark.parametrize("device_str,dtype", DEVICES_AND_DTYPES)
    def test_xaj_step_compile(self, device_str, dtype):
        device = torch.device(device_str)
        assert run_compile_test(_xaj_step, make_xaj_inputs, "XAJ", device, dtype)

    @pytest.mark.parametrize("device_str,dtype", DEVICES_AND_DTYPES)
    def test_xaj_compact_step_compile(self, device_str, dtype):
        device = torch.device(device_str)
        assert run_compile_test(_xaj_step_compact, make_xaj_compact_inputs, "XAJ_Compact", device, dtype)

    @pytest.mark.parametrize("device_str,dtype", DEVICES_AND_DTYPES)
    def test_simhyd_step_compile(self, device_str, dtype):
        device = torch.device(device_str)
        assert run_compile_test(_simhyd_step, make_simhyd_inputs, "SIMHYD", device, dtype)

    @pytest.mark.parametrize("device_str,dtype", DEVICES_AND_DTYPES)
    def test_simhyd_compact_step_compile(self, device_str, dtype):
        device = torch.device(device_str)
        assert run_compile_test(_simhyd_step_compact, make_simhyd_inputs, "SIMHYD_Compact", device, dtype)

    @pytest.mark.parametrize("device_str,dtype", DEVICES_AND_DTYPES)
    def test_cemaneige_step_compile(self, device_str, dtype):
        device = torch.device(device_str)
        assert run_compile_test(_cemaneige_step, make_cemaneige_inputs, "CemaNeige", device, dtype)

    @pytest.mark.parametrize("device_str,dtype", DEVICES_AND_DTYPES)
    def test_cemaneige_hyst_step_compile(self, device_str, dtype):
        device = torch.device(device_str)
        assert run_compile_test(
            _cemaneige_hyst_step,
            make_cemaneige_hyst_inputs,
            "CemaNeigeHyst",
            device,
            dtype,
        )

    @pytest.mark.parametrize("device_str,dtype", DEVICES_AND_DTYPES)
    def test_tgd2_step_compile(self, device_str, dtype):
        device = torch.device(device_str)
        assert run_compile_test(tgd2_step, make_tgd2_inputs, "TGD2", device, dtype)

    @pytest.mark.parametrize("device_str,dtype", DEVICES_AND_DTYPES)
    def test_precip_delay_step_compile(self, device_str, dtype):
        device = torch.device(device_str)
        assert run_compile_test(_precip_delay_step, make_precip_delay_inputs, "PrecipDelay", device, dtype)

    @pytest.mark.parametrize("device_str,dtype", DEVICES_AND_DTYPES)
    def test_cemaneige_gr4j_fused_step_compile(self, device_str, dtype):
        device = torch.device(device_str)
        assert run_compile_test(
            _cemaneige_gr4j_fused_step,
            make_cn_gr4j_fused_inputs,
            "CemaNeigeGR4JFused",
            device,
            dtype,
        )

    @pytest.mark.parametrize("device_str,dtype", DEVICES_AND_DTYPES)
    def test_cemaneige_xaj_fused_step_compile(self, device_str, dtype):
        device = torch.device(device_str)
        assert run_compile_test(
            _cemaneige_xaj_fused_step,
            make_cn_xaj_fused_inputs,
            "CemaNeigeXAJFused",
            device,
            dtype,
        )

    @pytest.mark.parametrize("device_str,dtype", DEVICES_AND_DTYPES)
    def test_cemaneige_simhyd_fused_step_compile(self, device_str, dtype):
        device = torch.device(device_str)
        assert run_compile_test(
            _cemaneige_simhyd_fused_step,
            make_cn_simhyd_fused_inputs,
            "CemaNeigeSIMHYDFused",
            device,
            dtype,
        )
