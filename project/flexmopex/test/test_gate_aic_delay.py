#!/usr/bin/env python3
"""Regression tests for delayed gate-AIC gradient exposure (gate_aic_delay_epochs).

Covers, with the project's native plain-script conventions:

  1. backward compatibility: field absent / 0 -> identical forward behavior
  2. config validation: negative gate_aic_delay_epochs rejected
  3. mask window (epochs 1..N): the w_* outputs fed to the loss are detached,
     so the AIC/complexity gradient is zero through the gate logits and the
     structure network, while the predictive-fit gradient through all four
     gates stays nonzero (fit-driven gate updates continue); the AIC *value*
     is bit-identical to the unmasked forward (same Gumbel draw), so the
     reported total loss is unchanged
  4. release (epoch N+1): w_* outputs carry gradients again, the AIC gradient
     through the gates is restored, and the total gradient decomposes as
     g_total == g_fit + g_aic
  5. fit-path invariance: the masked run's fit gradient through the gate
     logits is bit-identical to the unmasked run's (the streamflow graph is
     untouched by the mask)
  6. uniformity: all four structural processes (w_phen, w_int, w_snow, w_sub)
     are masked/released together; eval mode never masks

Run: python test/test_gate_aic_delay.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex.models.learned_weight_mopex_candidates import LearnedWeightMopexE  # noqa: E402
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


def base_cfg(delay: int | None = None):
    cfg = {
        "device": "cpu", "warm_up": 2, "warm_up_states": True,
        "variables": ["prcp", "tmean", "pet"], "nmul": 1, "nearzero": 1e-5,
        "structure_tau": 1.0, "disable_compile": True,
        "phy": {"name": ["LearnedWeightMopexE"], "warm_up": 2, "nmul": 1},
        "nn": {"attributes": ["p_mean"], "forcings": ["prcp", "tmean", "pet"]},
        "interception_semantics": "S0",
    }
    if delay is not None:
        cfg["gate_aic_delay_epochs"] = delay
    return cfg


def batch(n_basin: int = 4, n_days: int = 8):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n_days, n_basin, 3, generator=g) * 6.0 + 0.5
    doy = torch.arange(1, n_days + 1, dtype=torch.float32).view(n_days, 1, 1).repeat(1, n_basin, 1)
    return {"x_phy": x, "doy": doy, "c_nn_norm": torch.randn(n_basin, 1, generator=g)}


def make_nn():
    return LearnedStructureNet(input_dim=1, hidden_dim=8, dropout=0.0, nmul=1, device="cpu")


def run_forward(phy, nn, x):
    """Real model forward: structure net -> gate logits -> weighted hydrology."""
    p = nn(x)
    out = phy({"x_phy": x["x_phy"], "doy": x["doy"], "c_nn_norm": x["c_nn_norm"]},
              {"params": p["params"], "weights": p["weights"], "gamma_uh": p["gamma_uh"]})
    return out, p


def w_dict(out):
    return {n: out[n] for n in WEIGHT_NAMES}


def complexity(wd):
    return sum(COSTS[n] * torch.mean(wd[n]) for n in WEIGHT_NAMES)


def main() -> None:
    torch.manual_seed(42)
    x = batch()
    nn0 = make_nn()
    # Model trims warm_up days from the routed output (streamflow is
    # [n_days - warm_up, n_basin, 1]); target must match the trimmed length.
    y = torch.rand(x["x_phy"].shape[0] - 2, x["x_phy"].shape[1], 1)

    # ---- 0. wiring: attribute present with default 0 ----
    m0 = LearnedWeightMopexE(base_cfg(), device="cpu")
    check("wiring: gate_aic_delay_epochs defaults to 0", m0.gate_aic_delay_epochs == 0)

    # ---- 1. backward compatibility: absent == 0 field ----
    m0b = LearnedWeightMopexE(base_cfg(0), device="cpu")
    m0.train(); m0b.train()
    torch.manual_seed(0); o0, _ = run_forward(m0, nn0, x)
    torch.manual_seed(0); o0b, _ = run_forward(m0b, nn0, x)
    check("backcompat: absent == 0 (streamflow identical)",
          torch.equal(o0["streamflow"], o0b["streamflow"]))
    check("backcompat: absent == 0 (w_* outputs identical)",
          all(torch.equal(o0[n], o0b[n]) for n in WEIGHT_NAMES))

    # ---- 2. negative config rejected ----
    try:
        LearnedWeightMopexE(base_cfg(-1), device="cpu")
        check("config: negative gate_aic_delay_epochs rejected", False)
    except ValueError:
        check("config: negative gate_aic_delay_epochs rejected", True)

    # ---- 3. mask window (delay = 2; epochs 1, 2) ----
    md = LearnedWeightMopexE(base_cfg(2), device="cpu")
    nn_d = make_nn()
    md.train()
    criterion = NseDynAicBatchLoss({"aic_alpha": AIC_ALPHA}, "cpu", y_obs=y)
    criterion_fit = NseDynAicBatchLoss({"aic_alpha": 0.0}, "cpu", y_obs=y)

    for ep in (1, 2):
        md.set_current_epoch(ep)
        torch.manual_seed(0)
        out_m, p_m = run_forward(md, nn_d, x)
        wd_m = w_dict(out_m)
        comp_m = complexity(wd_m)
        # (a) w_* outputs detached -> no AIC gradient through the gate path
        check(f"mask ep{ep}: w_* outputs detached (uniform across 4 processes)",
              all(not out_m[n].requires_grad for n in WEIGHT_NAMES))
        # (b) streamflow still requires grad through the gate logits
        check(f"mask ep{ep}: streamflow keeps grad graph",
              out_m["streamflow"].requires_grad)
        g_fit = torch.autograd.grad(out_m["streamflow"].sum(), p_m["weights"],
                                    retain_graph=True)[0]
        # (c) AIC gradient through gate logits is zero: attach a zero-weight
        #     hook (grad_fn carrier, contributes exactly 0) to the detached AIC
        #     term so autograd.grad works in both phases.
        hook = out_m["streamflow"].sum() * 0.0
        g_aic = torch.autograd.grad(AIC_ALPHA * comp_m + hook, p_m["weights"],
                                    retain_graph=True)[0]
        check(f"mask ep{ep}: AIC gradient through gate logits zero",
              float(g_aic.abs().sum()) == 0.0, f"|g_aic|={float(g_aic.abs().sum()):.3e}")
        # (d) full training loss: total gradient == fit gradient
        loss = criterion(out_m["streamflow"], y, sample_ids=list(range(x["c_nn_norm"].shape[0])),
                         weights=wd_m)
        loss_fit = criterion_fit(out_m["streamflow"], y,
                                 sample_ids=list(range(x["c_nn_norm"].shape[0])),
                                 weights=wd_m)
        g_tot = torch.autograd.grad(loss, p_m["weights"], retain_graph=True)[0]
        g_fit2 = torch.autograd.grad(loss_fit, p_m["weights"], retain_graph=True)[0]
        check(f"mask ep{ep}: total grad == fit grad (AIC contribution zero)",
              torch.allclose(g_tot, g_fit2, atol=1e-6),
              f"max|d|={float((g_tot - g_fit2).abs().max()):.3e}")
        check(f"mask ep{ep}: fit-only loss grad nonzero", float(g_fit2.abs().sum()) > 0.0)
        # (e) AIC value identical to the unmasked same-draw forward: run the
        #     same model at epoch 3 (mask released) with the same Gumbel seed
        md.set_current_epoch(3)
        torch.manual_seed(0)
        out_u, _ = run_forward(md, nn_d, x)
        comp_u = complexity(w_dict(out_u))
        md.set_current_epoch(ep)
        check(f"mask ep{ep}: AIC value bit-identical to unmasked forward",
              torch.equal(comp_m, comp_u),
              f"comp_m={float(comp_m.detach()):.6f} comp_u={float(comp_u.detach()):.6f}")
        # (f) gate head still trains via the fit gradient
        aic_grad_on_weights_head = [
            torch.autograd.grad(AIC_ALPHA * comp_m + hook, t, retain_graph=True,
                                allow_unused=True)[0]
            for t in nn_d.heads["weights"].parameters()
        ]
        check(f"mask ep{ep}: AIC gradient zero through structure-net head",
              all(g is not None and float(g.abs().sum()) == 0.0
                  for g in aic_grad_on_weights_head))
        nn_d.zero_grad(set_to_none=True)
        loss.backward()
        gw = sum(float(t.grad.abs().sum()) for t in nn_d.heads["weights"].parameters()
                 if t.grad is not None)
        gp = sum(float(t.grad.abs().sum()) for t in nn_d.heads["params"].parameters()
                 if t.grad is not None)
        check(f"mask ep{ep}: gate head receives gradient (fit-driven)", gw > 0.0)
        check(f"mask ep{ep}: parameter head keeps training", gp > 0.0)

    # ---- 4. release (epoch 3) ----
    md.set_current_epoch(3)
    torch.manual_seed(0)
    out_r, p_r = run_forward(md, nn_d, x)
    wd_r = w_dict(out_r)
    comp_r = complexity(wd_r)
    check("release: w_* outputs carry gradients again",
          all(out_r[n].requires_grad for n in WEIGHT_NAMES))
    g_aic_r = torch.autograd.grad(AIC_ALPHA * comp_r, p_r["weights"], retain_graph=True)[0]
    check("release: AIC gradient through gate logits restored",
          g_aic_r is not None and float(g_aic_r.abs().sum()) > 0.0)
    loss_r = criterion(out_r["streamflow"], y,
                       sample_ids=list(range(x["c_nn_norm"].shape[0])), weights=wd_r)
    loss_fit_r = criterion_fit(out_r["streamflow"], y,
                               sample_ids=list(range(x["c_nn_norm"].shape[0])), weights=wd_r)
    g_tot_r = torch.autograd.grad(loss_r, p_r["weights"], retain_graph=True)[0]
    g_fit_r = torch.autograd.grad(loss_fit_r, p_r["weights"], retain_graph=True)[0]
    check("release: total grad == fit grad + AIC grad",
          torch.allclose(g_tot_r, g_fit_r + g_aic_r, atol=1e-6),
          f"max|d|={float((g_tot_r - g_fit_r - g_aic_r).abs().max()):.3e}")
    check("release: total gradient nonzero", float(g_tot_r.abs().sum()) > 0.0)

    # ---- 5. fit-path invariance: masked fit grad bit-identical to unmasked ----
    md.set_current_epoch(1)
    torch.manual_seed(0)
    out_m2, p_m2 = run_forward(md, nn_d, x)
    loss_fit_m = criterion_fit(out_m2["streamflow"], y,
                               sample_ids=list(range(x["c_nn_norm"].shape[0])),
                               weights=w_dict(out_m2))
    g_fit_m = torch.autograd.grad(loss_fit_m, p_m2["weights"], retain_graph=True)[0]
    check("invariance: masked fit grad bit-identical to unmasked fit grad",
          torch.equal(g_fit_m, g_fit_r))

    # ---- 6. eval mode never masks ----
    md.set_current_epoch(1)
    md.eval()
    out_e, p_e = run_forward(md, nn_d, x)
    logits = p_e["weights"].view(x["c_nn_norm"].shape[0], 4, 2).clamp(min=-10., max=10.)
    raw = torch.softmax(logits, dim=-1)[..., 1]
    check("eval: w_* carry gradients (mask inactive in eval)",
          all(out_e[n].requires_grad for n in WEIGHT_NAMES))
    check("eval: w_* are the learned softmax gates (not detached constants)",
          all(torch.allclose(out_e[n][0, :, 0], raw[:, i]) for i, n in enumerate(WEIGHT_NAMES)))

    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        sys.exit(1)
    print("ALL GATE-AIC-DELAY TESTS PASSED")


if __name__ == "__main__":
    main()
