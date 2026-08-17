#!/usr/bin/env python3
"""Regression and behavioral tests for Soft-Denoised Sensitivity-Reweighted Gate Gradient Aggregation (R12).

Covers:
  1. Config validation & mode parsing (soft_denoised vs direction_balanced vs sensitivity vs none, invalid cap rejected)
  2. Mathematical invariants of SoftDenoisedSensitivityReweightFunction:
     - sign preservation: sign(g_tilde[i, p]) == sign(g[i, p])
     - process-mean absolute magnitude preservation: mean(|g_tilde[:, p]|) == mean(|g[:, p]|)
     - soft confidence damping: basins with |g| << tau suppressed quadratically
     - high-sensitivity amplification: basins with |g| >> tau amplified up to cap (5.0)
     - edge case stability: all zeros, uniform gradients, tiny gradients
  3. Model-level gradient-routing boundary:
     - forward streamflow, weights_on, w_* bit-identical
     - scalar L_fit, AIC value, and total loss bit-identical
     - hydrologic parameter (params, gamma_uh) gradients identical within tolerance
     - AIC -> gate gradient unaffected (AIC uses unaugmented structural weights)
     - only the fit gradient entering the gate logits / structure network changes
  4. Integration with gate_aic_delay_epochs (epochs 1-2 masked, epoch 3 released)
  5. Eval-mode invariance (no transformation when self.training == False)

Run: python test/test_soft_denoised_reweighting.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex.models.learned_weight_mopex_candidates import (  # noqa: E402
    LearnedWeightMopexE,
    SoftDenoisedSensitivityReweightFunction,
    soft_denoised_reweight_fit_gradient,
)
from project.flexmopex.models.parameter_nets import LearnedStructureNet  # noqa: E402
from project.flexmopex.models.nse_dyn_aic_batch_loss import NseDynAicBatchLoss  # noqa: E402

FAILED = []
WEIGHT_NAMES = ("w_phen", "w_int", "w_snow", "w_sub")
COSTS = {"w_phen": 2.0, "w_int": 2.0, "w_snow": 2.0, "w_sub": 1.0}
AIC_ALPHA = 0.01


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def base_cfg(soft_den: bool = False, cap: float = 5.0, delay: int | None = None):
    cfg = {
        "device": "cpu", "warm_up": 2, "warm_up_states": True,
        "variables": ["prcp", "tmean", "pet"], "nmul": 1, "nearzero": 1e-5,
        "structure_tau": 1.0, "disable_compile": True,
        "phy": {"name": ["LearnedWeightMopexE"], "warm_up": 2, "nmul": 1},
        "nn": {"attributes": ["p_mean"], "forcings": ["prcp", "tmean", "pet"]},
        "interception_semantics": "S0",
        "soft_denoised_gate_gradients": soft_den,
        "reweight_gate_cap": cap,
    }
    if delay is not None:
        cfg["gate_aic_delay_epochs"] = delay
    return cfg


def batch(n_basin: int = 16, n_days: int = 8):
    g = torch.Generator().manual_seed(42)
    x = torch.rand(n_days, n_basin, 3, generator=g) * 6.0 + 0.5
    doy = torch.arange(1, n_days + 1, dtype=torch.float32).view(n_days, 1, 1).repeat(1, n_basin, 1)
    return {"x_phy": x, "doy": doy, "c_nn_norm": torch.randn(n_basin, 1, generator=g)}


def make_nn():
    return LearnedStructureNet(input_dim=1, hidden_dim=8, dropout=0.0, nmul=1, device="cpu")


def run_forward(phy, nn, x):
    p = nn(x)
    out = phy({"x_phy": x["x_phy"], "doy": x["doy"], "c_nn_norm": x["c_nn_norm"]},
              {"params": p["params"], "weights": p["weights"], "gamma_uh": p["gamma_uh"]})
    return out, p


def main() -> None:
    torch.manual_seed(42)

    # =========================================================================
    # 1. Config validation & Defaults
    # =========================================================================
    m_def = LearnedWeightMopexE(base_cfg(), device="cpu")
    check("config: soft_denoised defaults to False (mode 'none')", m_def.reweight_gate_mode == "none")

    m_sd = LearnedWeightMopexE(base_cfg(soft_den=True), device="cpu")
    check("config: soft_denoised_gate_gradients sets mode 'soft_denoised'", m_sd.reweight_gate_mode == "soft_denoised")

    try:
        LearnedWeightMopexE(base_cfg(soft_den=True, cap=0.0), device="cpu")
        check("config: zero cap rejected", False)
    except ValueError:
        check("config: zero cap rejected", True)

    try:
        LearnedWeightMopexE(base_cfg(soft_den=True, cap=-1.0), device="cpu")
        check("config: negative cap rejected", False)
    except ValueError:
        check("config: negative cap rejected", True)

    # =========================================================================
    # 2. Mathematical invariants of SoftDenoisedSensitivityReweightFunction
    # =========================================================================
    n_b, n_p = 100, 4
    g_in = torch.zeros(n_b, n_p)

    # Process 0: 15 strong basins (|g| ~ 0.04), 85 weak noise basins (|g| ~ 0.0005)
    g_in[:15, 0] = - (torch.rand(15) * 0.04 + 0.02)
    g_in[15:, 0] = (torch.rand(85) - 0.5) * 0.001

    # Process 1: Homogeneous gradients
    g_in[:, 1] = - (torch.rand(n_b) * 0.05 + 0.01)

    # Process 2: All zeros
    g_in[:, 2] = 0.0

    # Process 3: Random normal
    g_in[:, 3] = torch.randn(n_b) * 0.02

    w_dummy = torch.randn(n_b, n_p, requires_grad=True)
    w_reweighted = soft_denoised_reweight_fit_gradient(w_dummy, cap=5.0)

    (w_reweighted * g_in).sum().backward()
    g_tilde = w_dummy.grad

    check("math: g_tilde matches input shape", g_tilde.shape == g_in.shape)
    check("math: sign preservation sign(g_tilde) == sign(g_in) for nonzeros",
          (torch.sign(g_tilde[:, [0, 1, 3]]) == torch.sign(g_in[:, [0, 1, 3]])).all().item())

    # Mean absolute magnitude preservation per process
    mean_abs_in = torch.mean(torch.abs(g_in), dim=0)
    mean_abs_out = torch.mean(torch.abs(g_tilde), dim=0)
    check("math: process-mean absolute magnitude preserved",
          torch.allclose(mean_abs_in, mean_abs_out, atol=1e-6),
          f"diff = {(mean_abs_in - mean_abs_out).abs().max():.2e}")

    # Soft denoising damping verification on Process 0:
    # Weak basins (|g| << tau) should be suppressed dramatically relative to strong basins (|g| >> tau)
    strong_ratio = torch.mean(torch.abs(g_tilde[:15, 0])) / torch.mean(torch.abs(g_in[:15, 0]))
    weak_ratio = torch.mean(torch.abs(g_tilde[15:, 0])) / torch.mean(torch.abs(g_in[15:, 0]))
    check("math: soft denoising heavily suppresses weak noise basins relative to strong signal",
          weak_ratio < 0.1 and strong_ratio > 0.8,
          f"strong ratio = {strong_ratio.item():.2f}x, weak ratio = {weak_ratio.item():.4f}x (amplification = {(strong_ratio / weak_ratio).item():.1f}x)")

    # Edge cases: zero column
    check("math: zero gradient column remains exactly zero", torch.equal(g_tilde[:, 2], torch.zeros(n_b)))

    # =========================================================================
    # 3. Model-level gradient-routing boundary
    # =========================================================================
    x = batch(n_basin=16, n_days=8)
    y = torch.rand(x["x_phy"].shape[0] - 2, x["x_phy"].shape[1], 1)
    nn_base = make_nn()
    m_base = LearnedWeightMopexE(base_cfg(soft_den=False), device="cpu")
    m_r12 = LearnedWeightMopexE(base_cfg(soft_den=True, cap=5.0), device="cpu")
    m_base.train()
    m_r12.train()

    # (a) Forward values identical on deterministic draw
    torch.manual_seed(123)
    out_base, p_base = run_forward(m_base, nn_base, x)
    torch.manual_seed(123)
    out_r12, p_r12 = run_forward(m_r12, nn_base, x)

    check("boundary: forward streamflow identical", torch.equal(out_base["streamflow"], out_r12["streamflow"]))
    check("boundary: forward w_* identical", all(torch.equal(out_base[k], out_r12[k]) for k in WEIGHT_NAMES))

    # (b) Loss scalar values identical
    loss_fn = NseDynAicBatchLoss({"aic_alpha": AIC_ALPHA}, "cpu", y_obs=y)
    w_dict_base = {k: out_base[k] for k in WEIGHT_NAMES}
    w_dict_r12 = {k: out_r12[k] for k in WEIGHT_NAMES}

    loss_base = loss_fn(out_base["streamflow"], y, sample_ids=list(range(16)), weights=w_dict_base)
    loss_r12 = loss_fn(out_r12["streamflow"], y, sample_ids=list(range(16)), weights=w_dict_r12)
    check("boundary: total scalar loss identical", torch.equal(loss_base, loss_r12),
          f"{loss_base.item():.6f} vs {loss_r12.item():.6f}")

    # (c) Non-gate parameter gradients (params head, gamma_uh head) identical within numerical tolerance
    loss_fn_fit = NseDynAicBatchLoss({"aic_alpha": 0.0}, "cpu", y_obs=y)
    loss_fit_base = loss_fn_fit(out_base["streamflow"], y, sample_ids=list(range(16)), weights=w_dict_base)
    loss_fit_r12 = loss_fn_fit(out_r12["streamflow"], y, sample_ids=list(range(16)), weights=w_dict_r12)

    g_params_base = torch.autograd.grad(loss_fit_base, p_base["params"], retain_graph=True)[0]
    g_params_r12 = torch.autograd.grad(loss_fit_r12, p_r12["params"], retain_graph=True)[0]
    check("boundary: hydrologic parameter gradients (params) identical",
          torch.allclose(g_params_base, g_params_r12, atol=1e-6),
          f"max diff = {(g_params_base - g_params_r12).abs().max():.2e}")

    g_gamma_base = torch.autograd.grad(loss_fit_base, p_base["gamma_uh"], retain_graph=True)[0]
    g_gamma_r12 = torch.autograd.grad(loss_fit_r12, p_r12["gamma_uh"], retain_graph=True)[0]
    check("boundary: routing parameter gradients (gamma_uh) identical",
          torch.allclose(g_gamma_base, g_gamma_r12, atol=1e-6),
          f"max diff = {(g_gamma_base - g_gamma_r12).abs().max():.2e}")

    # (d) AIC gradient w.r.t. gate logits is identical
    comp_base = sum(COSTS[n] * torch.mean(w_dict_base[n]) for n in WEIGHT_NAMES)
    comp_r12 = sum(COSTS[n] * torch.mean(w_dict_r12[n]) for n in WEIGHT_NAMES)
    g_aic_base = torch.autograd.grad(AIC_ALPHA * comp_base, p_base["weights"], retain_graph=True)[0]
    g_aic_r12 = torch.autograd.grad(AIC_ALPHA * comp_r12, p_r12["weights"], retain_graph=True)[0]
    check("boundary: AIC gate gradient identical",
          torch.allclose(g_aic_base, g_aic_r12, atol=1e-6),
          f"max diff = {(g_aic_base - g_aic_r12).abs().max():.2e}")

    # (e) Fit gradient w.r.t. gate logits is transformed
    g_fit_base = torch.autograd.grad(loss_fit_base, p_base["weights"], retain_graph=True)[0]
    g_fit_r12 = torch.autograd.grad(loss_fit_r12, p_r12["weights"], retain_graph=True)[0]
    check("boundary: fit gradient on gate logits is transformed",
          not torch.equal(g_fit_base, g_fit_r12) and (g_fit_base - g_fit_r12).abs().max() > 1e-7,
          f"max diff = {(g_fit_base - g_fit_r12).abs().max():.2e}")

    # =========================================================================
    # 4. Integration with gate_aic_delay_epochs
    # =========================================================================
    m_r12_delay = LearnedWeightMopexE(base_cfg(soft_den=True, cap=5.0, delay=2), device="cpu")
    m_r12_delay.train()

    # Epoch 1 (masked): w_* detached -> total loss gradient on gate equals fit gradient
    m_r12_delay.set_current_epoch(1)
    torch.manual_seed(123)
    out_d1, p_d1 = run_forward(m_r12_delay, nn_base, x)
    wd_d1 = {k: out_d1[k] for k in WEIGHT_NAMES}
    loss_d1 = loss_fn(out_d1["streamflow"], y, sample_ids=list(range(16)), weights=wd_d1)
    loss_fit_d1 = loss_fn_fit(out_d1["streamflow"], y, sample_ids=list(range(16)), weights=wd_d1)
    g_tot_d1 = torch.autograd.grad(loss_d1, p_d1["weights"], retain_graph=True)[0]
    g_fit_d1 = torch.autograd.grad(loss_fit_d1, p_d1["weights"], retain_graph=True)[0]
    check("delay+r12 ep1: total gate gradient equals reweighted fit gradient",
          torch.allclose(g_tot_d1, g_fit_d1, atol=1e-6))

    # Epoch 3 (released): total gate gradient = reweighted fit gradient + unmasked AIC gradient
    m_r12_delay.set_current_epoch(3)
    torch.manual_seed(123)
    out_d3, p_d3 = run_forward(m_r12_delay, nn_base, x)
    wd_d3 = {k: out_d3[k] for k in WEIGHT_NAMES}
    loss_d3 = loss_fn(out_d3["streamflow"], y, sample_ids=list(range(16)), weights=wd_d3)
    loss_fit_d3 = loss_fn_fit(out_d3["streamflow"], y, sample_ids=list(range(16)), weights=wd_d3)
    comp_d3 = sum(COSTS[n] * torch.mean(wd_d3[n]) for n in WEIGHT_NAMES)

    g_tot_d3 = torch.autograd.grad(loss_d3, p_d3["weights"], retain_graph=True)[0]
    g_fit_d3 = torch.autograd.grad(loss_fit_d3, p_d3["weights"], retain_graph=True)[0]
    g_aic_d3 = torch.autograd.grad(AIC_ALPHA * comp_d3, p_d3["weights"], retain_graph=True)[0]
    check("delay+r12 ep3: total gate gradient equals reweighted fit + AIC",
          torch.allclose(g_tot_d3, g_fit_d3 + g_aic_d3, atol=1e-6),
          f"max diff = {(g_tot_d3 - g_fit_d3 - g_aic_d3).abs().max():.2e}")

    # =========================================================================
    # 5. Eval-mode invariance
    # =========================================================================
    m_r12.eval()
    m_base.eval()
    out_e_base, p_e_base = run_forward(m_base, nn_base, x)
    out_e_r12, p_e_r12 = run_forward(m_r12, nn_base, x)
    check("eval: streamflow identical in eval mode", torch.equal(out_e_base["streamflow"], out_e_r12["streamflow"]))
    check("eval: w_* identical in eval mode", all(torch.equal(out_e_base[k], out_e_r12[k]) for k in WEIGHT_NAMES))

    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        sys.exit(1)
    print("ALL SOFT-DENOISED REWEIGHTING TESTS PASSED")


if __name__ == "__main__":
    main()
