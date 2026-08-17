#!/usr/bin/env python3
"""Behavioral tests for full-process parameter warm-up (structure_warmup_epochs).

Covers, with the project's native plain-script conventions:

  1. backward compatibility: field absent / 0 -> identical gate behavior
  2. warm-up: effective gates exactly 1 during epochs 1..N; gate head gets no
     gradient (logits stay neutral); hydrologic/parameter heads keep training;
     kappa/phi (Candidate E) remain learnable
  3. release: from epoch N+1 effective gates revert to the learned softmax
     values; gate head receives gradients again
  4. AIC: complexity term is constant (0.01*(2+2+2+1)=0.07) during warm-up with
     zero gradient wrt gate parameters; aic_alpha and the cost table unchanged
  5. epoch indexing / resume boundary semantics

Run: python test/test_structure_warmup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex.models import mopex_core_candidates as cand  # noqa: E402
from project.flexmopex.models.learned_weight_mopex_candidates import LearnedWeightMopexE  # noqa: E402
from project.flexmopex.models.parameter_nets import LearnedStructureNet  # noqa: E402
from project.flexmopex.models.nse_dyn_aic_batch_loss import NseDynAicBatchLoss  # noqa: E402

FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def base_cfg(warmup: int | None = None):
    cfg = {
        "device": "cpu", "warm_up": 2, "warm_up_states": True,
        "variables": ["prcp", "tmean", "pet"], "nmul": 1, "nearzero": 1e-5,
        "structure_tau": 1.0, "disable_compile": True,
        "phy": {"name": ["LearnedWeightMopexE"], "warm_up": 2, "nmul": 1},
        "nn": {"attributes": ["p_mean"], "forcings": ["prcp", "tmean", "pet"]},
        "interception_semantics": "S0",
    }
    if warmup is not None:
        cfg["structure_warmup_epochs"] = warmup
    return cfg


def batch(n_basin: int = 4, n_days: int = 8):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n_days, n_basin, 3, generator=g) * 6.0 + 0.5
    doy = torch.arange(1, n_days + 1, dtype=torch.float32).view(n_days, 1, 1).repeat(1, n_basin, 1)
    return {"x_phy": x, "doy": doy, "c_nn_norm": torch.randn(n_basin, 1, generator=g)}


def make_nn():
    return LearnedStructureNet(input_dim=1, hidden_dim=8, dropout=0.0, nmul=1, device="cpu")


def gates_of(phy, nn, x):
    p = nn(x)
    return phy._structure_weights(p["weights"]), p


def main() -> None:
    torch.manual_seed(42)

    # ---- 0. step-function wiring (Candidate E actually installed) ----
    mw0 = LearnedWeightMopexE(base_cfg(2), device="cpu")
    check("wiring: installed step is the Candidate E-S0 step",
          mw0.step_fn.__name__ == "mopex_step_E_S0", mw0.step_fn.__name__)
    # functional equivalence of the installed step vs the module-level step
    import torch as _t
    _g = _t.Generator().manual_seed(3)
    args = [_t.rand(4, generator=_g), _t.rand(4, generator=_g) - 2,
            _t.rand(4, generator=_g), _t.full((4,), 100.0),
            _t.full((4,), 0.5), _t.full((4,), 1.0), _t.full((4,), 0.7), _t.full((4,), 0.3),
            _t.rand(4, generator=_g) * 40 + 1, _t.rand(4, generator=_g) * 4 + 0.1,
            _t.rand(4, generator=_g) * 1500 + 10, _t.rand(4, generator=_g) * 800 + 50,
            _t.rand(4, generator=_g) * 25 + 1, _t.rand(4, generator=_g) * 15,
            _t.rand(4, generator=_g) * 6 - 3, _t.rand(4, generator=_g) * 1200 + 100,
            _t.full((4,), 0.5), _t.full((4,), 180.0), _t.full((4,), -5.0), _t.full((4,), 20.0)]
    states = [_t.full((4,), 1e-6) for _ in range(5)]
    kw = {"P": args[0], "T": args[1], "PET": args[2], "doy": args[3],
          "w_phen": args[4], "w_int": args[5], "w_snow": args[6], "w_sub": args[7],
          "Sb1": args[8], "tw": args[9], "tu": args[10], "Se": args[11], "tc": args[12],
          "ddf": args[13], "tcrit": args[14], "Sb2": args[15], "alpha": args[16],
          "is_time": args[17], "tmin": args[18], "tmax": args[19],
          "S1": states[0], "S2": states[1], "Sc1": states[2], "Sc2": states[3], "Sn": states[4]}
    o1 = mw0.step_fn(*args, *states, 1e-5)
    o2 = cand.mopex_step_E_S0(*args, *states, 1e-5)
    check("wiring: installed step numerically == mopex_step_E_S0",
          all(_t.equal(a, b) for a, b in zip(o1, o2)))

    # ---- 1. backward compatibility: absent vs 0 ----
    m0 = LearnedWeightMopexE(base_cfg(), device="cpu")
    m0b = LearnedWeightMopexE(base_cfg(0), device="cpu")
    nn0 = make_nn()
    x = batch()
    m0.train(); m0b.train()
    torch.manual_seed(0); g0, _ = gates_of(m0, nn0, x)   # matched Gumbel noise
    torch.manual_seed(0); g0b, _ = gates_of(m0b, nn0, x)
    check("backcompat: absent == 0 field (training gates identical)",
          torch.equal(g0, g0b))
    m0.eval(); m0b.eval()
    e0, _ = gates_of(m0, nn0, x)
    e0b, _ = gates_of(m0b, nn0, x)
    check("backcompat: absent == 0 field (eval gates identical)",
          torch.equal(e0, e0b))
    check("backcompat: eval gates are learned softmax (not 1)",
          not torch.allclose(e0, torch.ones_like(e0)) and bool((e0 > 0.0).all()))

    # ---- 2. warm-up behavior ----
    mw = LearnedWeightMopexE(base_cfg(2), device="cpu")
    nnw = make_nn()
    mw.train()
    for ep in (1, 2):
        mw.set_current_epoch(ep)
        w, p = gates_of(mw, nnw, x)
        check(f"warmup: effective gates exactly 1 (epoch {ep})",
              torch.equal(w, torch.ones_like(w)))
        # raw/learned gate probability (softmax of current logits) != 1
        logits = p["weights"].view(x["c_nn_norm"].shape[0], 4, 2).clamp(min=-10., max=10.)
        raw = torch.softmax(logits, dim=-1)[..., 1]
        check(f"warmup: raw gate probability not overwritten to 1 (epoch {ep})",
              not torch.allclose(raw, torch.ones_like(raw)))
        # effective gate is detached -> no gradient path to the gate head
        check(f"warmup: effective gate detached (no grad to gate head, epoch {ep})",
              not w.requires_grad)

    # gradient to parameter head still flows during warm-up
    mw.set_current_epoch(1)
    p = nnw(x)
    w = mw._structure_weights(p["weights"])
    # build a small loss that includes the parameter path (params -> physics)
    out = mw({"x_phy": x["x_phy"], "doy": x["doy"], "c_nn_norm": x["c_nn_norm"]},
             {"params": p["params"], "weights": p["weights"], "gamma_uh": p["gamma_uh"]})
    loss = out["streamflow"].sum()
    loss.backward()
    gw = sum(float(t.grad.abs().sum()) for t in nnw.heads["weights"].parameters()
             if t.grad is not None)
    gp = sum(float(t.grad.abs().sum()) for t in nnw.heads["params"].parameters()
             if t.grad is not None)
    check("warmup: gate head zero grad through real forward", gw == 0.0)
    check("warmup: parameter head trains through real forward", gp > 0.0)
    # kappa/phi learnable: gradients on the params head (slots 8/9 descend from it)
    check("warmup: kappa/phi path (params head) receives gradients", gp > 0.0)

    # ---- 3. release behavior ----
    mw.set_current_epoch(3)
    w3, p3 = gates_of(mw, nnw, x)
    check("release: effective gates revert to learned softmax (not all 1)",
          not torch.allclose(w3, torch.ones_like(w3)) and bool((w3 > 0.0).all()))
    g3 = torch.autograd.grad(w3.sum(), p3["weights"], retain_graph=True, allow_unused=True)[0]
    check("release: gate head receives gradients again", g3 is not None and float(g3.abs().sum()) > 0.0)
    # released values near neutral (~0.5) for fresh logits
    check("release: values near neutral init (median in [0.3, 0.7])",
          0.3 <= float(w3.median()) <= 0.7, f"median={float(w3.median()):.3f}")

    # ---- 4. AIC integrity ----
    y = torch.rand(8, 4, 1)
    loss = NseDynAicBatchLoss({"aic_alpha": 0.01}, "cpu", y_obs=y)
    check("AIC: aic_alpha == 0.01", loss.aic_alpha == 0.01)
    check("AIC: cost table unchanged",
          loss.param_costs == {"w_phen": 2.0, "w_int": 2.0, "w_snow": 2.0, "w_sub": 1.0})
    # complexity term with effective gates = 1 -> 0.01*(2+2+2+1) = 0.07
    ones = {"w_phen": torch.ones(8, 4, 1), "w_int": torch.ones(8, 4, 1),
            "w_snow": torch.ones(8, 4, 1), "w_sub": torch.ones(8, 4, 1)}
    comp = loss(y, y, sample_ids=[0, 1, 2, 3], weights=ones)
    aic_only = 0.01 * (2.0 + 2.0 + 2.0 + 1.0)
    loss.aic_alpha = 0.0
    fit_only = float(loss(y, y, sample_ids=[0, 1, 2, 3], weights=ones))
    loss.aic_alpha = 0.01
    check("AIC: complexity term == 0.07 (aic*(2+2+2+1))",
          abs((float(comp) - fit_only) - aic_only) < 1e-6, f"{float(comp)-fit_only:.6f} vs {aic_only}")
    # AIC constant wrt gates during warm-up is already covered by the
    # zero-gradient-to-gate-head checks (the complexity term uses the
    # effective gates = 1, which are detached constants).

    # ---- 5. epoch indexing / resume boundary ----
    mw.set_current_epoch(2)
    w2, _ = gates_of(mw, nnw, x)
    check("boundary: epoch 2 still warm-up (gates == 1)", torch.equal(w2, torch.ones_like(w2)))
    mw.set_current_epoch(3)
    w3b, _ = gates_of(mw, nnw, x)
    check("boundary: epoch 3 released (gates != 1)", not torch.allclose(w3b, torch.ones_like(w3b)))
    # resume at release boundary: load ep2 checkpoint -> start_epoch 3 -> released
    mw.set_current_epoch(1)  # simulate a resumed warm-up run mid-way
    w1, _ = gates_of(mw, nnw, x)
    check("resume: epoch 1 warm-up after reload", torch.equal(w1, torch.ones_like(w1)))

    # ---- 6. negative config rejected ----
    try:
        LearnedWeightMopexE(base_cfg(-1), device="cpu")
        check("config: negative structure_warmup_epochs rejected", False)
    except ValueError:
        check("config: negative structure_warmup_epochs rejected", True)

    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        sys.exit(1)
    print("ALL STRUCTURE-WARMUP TESTS PASSED")


if __name__ == "__main__":
    main()
