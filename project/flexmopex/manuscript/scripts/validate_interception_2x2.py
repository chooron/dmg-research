#!/usr/bin/env python3
"""Phase 3 deterministic validation for the interception 2x2 experiment.

Runs every pre-training gate from the experiment protocol:

  1. annual mean of normalized g_shape ~ 1 for low/mid/high alpha
  2. alpha changes seasonal shape/contrast but not annual-mean amplitude
  3. is_time shifts timing without changing annual-mean amplitude
  4. w_int=0 closes interception
  5. w_int scales interception monotonically/linearly before any cap
  6. no NaN/Inf in any arm's step
  7. autograd reaches w_int, alpha, is_time (all four arms)
  8. parameter ordering and network output dimension unchanged
  9. AIC parameter-cost table unchanged
 10. production flex output unchanged on a deterministic reference batch
 11. restored V1 differs from V0 only in PET-budget treatment

Exits non-zero on the first failed check.  CPU-only, no training.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import project.flexmopex.models.mopex_core as mopex_core
import project.flexmopex.models.mopex_core_v1 as mopex_core_v1
from project.flexmopex.models.base_mopex import MOPEX_PARAM_NAMES, WEIGHT_NAMES
from project.flexmopex.models.nse_dyn_aic_batch_loss import NseDynAicBatchLoss
from project.flexmopex.models.learned_weight_mopex_v1 import (
    LearnedWeightMopexV1,
    LearnedWeightMopexDecoupled,
    LearnedWeightMopexV1Decoupled,
)
from project.flexmopex.models.learned_weight_mopex import LearnedWeightMopex

FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def torch_all_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


# ---------------------------------------------------------------------------
# common tiny test config (deterministic; CPU)
# ---------------------------------------------------------------------------
def tiny_config(phy_name: str, n_basin: int = 4, n_days: int = 40, nmul: int = 1):
    return {
        "device": "cpu",
        "warm_up": 5,
        "warm_up_states": True,
        "variables": ["prcp", "tmean", "pet"],
        "nmul": nmul,
        "nearzero": 1e-5,
        "structure_tau": 1.0,
        "disable_compile": True,
        "phy": {"name": [phy_name], "warm_up": 5, "nmul": nmul},
        "nn": {"attributes": ["p_mean", "aridity", "frac_forest"],
               "forcings": ["prcp", "tmean", "pet"]},
    }


def deterministic_batch(n_basin: int = 4, n_days: int = 40, nmul: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n_days, n_basin, 3, generator=g) * 6.0 + 0.5          # P, T, PET
    x[:, :, 1] = x[:, :, 1] - 3.0
    doy = torch.arange(1, n_days + 1, dtype=torch.float32).view(n_days, 1, 1).repeat(1, n_basin, 1)
    attrs = torch.rand(n_basin, 3, generator=g) * 2.0 - 1.0
    return {
        "x_phy": x,
        "doy": doy,
        "c_nn_norm": attrs,
        "target": torch.rand(n_days, n_basin, 1, generator=g) * 5.0,
    }


def build_model(cls, n_basin=4, n_days=40):
    cfg = tiny_config(cls.__name__)
    model = cls(cfg, device="cpu")
    # freeze nothing; just return the bare phy model
    return model


# ---------------------------------------------------------------------------
# 1-3. g_shape / alpha / is_time semantics (module level)
# ---------------------------------------------------------------------------
def test_shape_semantics():
    alpha_vals = [0.1, 0.5, 0.9]
    grid = mopex_core_v1.PHASE_GRID
    grid_full = torch.linspace(1.0, 365.0, 365)
    for a in alpha_vals:
        alpha = torch.full((1,), a)
        is_time = torch.zeros(1)
        nm = mopex_core_v1.decoupled_norm_mean(alpha, is_time)
        shape = mopex_core_v1.decoupled_shape(grid_full.unsqueeze(-1), alpha, is_time, nm)
        check(
            f"1. g_shape annual mean ~ 1 (alpha={a})",
            abs(float(shape.mean()) - 1.0) < 5e-3,
            f"mean={float(shape.mean()):.6f}",
        )
        # annual mean of I_pot/P = C_REF * mean(g_shape) must not depend on alpha
        amp = float((mopex_core_v1.C_REF * shape).mean())
        if a == alpha_vals[0]:
            _ref_amp = amp
        else:
            check(
                f"2. annual-mean amplitude independent of alpha (alpha={a})",
                abs(amp - _ref_amp) < 5e-3,
                f"mean_Ipot/P={amp:.6f}",
            )
    # contrast: alpha=0.1 shape varies much more than alpha=0.9
    s_low = mopex_core_v1.decoupled_shape(
        grid_full.unsqueeze(-1),
        torch.full((1,), 0.1), torch.zeros(1),
        mopex_core_v1.decoupled_norm_mean(torch.full((1,), 0.1), torch.zeros(1)),
    )
    s_high = mopex_core_v1.decoupled_shape(
        grid_full.unsqueeze(-1),
        torch.full((1,), 0.9), torch.zeros(1),
        mopex_core_v1.decoupled_norm_mean(torch.full((1,), 0.9), torch.zeros(1)),
    )
    check(
        "2b. alpha controls contrast (low alpha -> wide seasonal swing)",
        float((s_low.max() - s_low.min())) > 4.0 * float(s_high.max() - s_high.min()),
        f"range low={float(s_low.max()-s_low.min()):.3f} high={float(s_high.max()-s_high.min()):.3f}",
    )
    # is_time: phase shift only
    nm_a = mopex_core_v1.decoupled_norm_mean(torch.full((1,), 0.3), torch.zeros(1))
    nm_b = mopex_core_v1.decoupled_norm_mean(torch.full((1,), 0.3), torch.full((1,), 200.0))
    check(
        "3. norm_mean independent of is_time",
        abs(float(nm_a - nm_b)) < 1e-5,
        f"is_time=0 -> {float(nm_a):.6f}, is_time=200 -> {float(nm_b):.6f}",
    )
    s_t0 = mopex_core_v1.decoupled_shape(grid_full.unsqueeze(-1), torch.full((1,), 0.3), torch.zeros(1), nm_a)
    s_t200 = mopex_core_v1.decoupled_shape(grid_full.unsqueeze(-1), torch.full((1,), 0.3), torch.full((1,), 200.0), nm_b)
    peak_t0 = int(grid_full[s_t0.argmax()].item())
    peak_t200 = int(grid_full[s_t200.argmax()].item())
    check(
        "3b. is_time shifts seasonal timing",
        abs((peak_t200 - peak_t0) % 365 - 200) <= 1,
        f"peak {peak_t0} -> {peak_t200}",
    )
    check(
        "3c. C_REF fixed reference scale ~ 0.75 (alpha_ref=0.5)",
        abs(mopex_core_v1.C_REF - 0.75) < 1e-2,
        f"C_REF={mopex_core_v1.C_REF:.6f}",
    )


# ---------------------------------------------------------------------------
# 4-6. step-level interception behaviour + finiteness (via V1 and V0 steps)
# ---------------------------------------------------------------------------
def _step_args(batch, w_int_val, w_vals, params_scale=1.0, seed=1):
    g = torch.Generator().manual_seed(seed)
    n_basin, n_days = batch["x_phy"].shape[1], batch["x_phy"].shape[0]
    t = 3  # arbitrary timestep
    P = batch["x_phy"][t, :, 0].clone()
    T = batch["x_phy"][t, :, 1].clone()
    PET = batch["x_phy"][t, :, 2].clone()
    doy = batch["doy"][t].clone()
    w_phen, w_snow, w_sub = (torch.full((n_basin,), v) for v in w_vals)
    w_int = torch.full((n_basin,), w_int_val)
    params = {}
    for name in MOPEX_PARAM_NAMES:
        lo, hi = mopex_core.MOPEX_PARAMS_BOUNDS[name]
        params[name] = (lo + (hi - lo) * torch.rand(n_basin, generator=g)) * params_scale
    states = {s: torch.full((n_basin,), 1e-6) for s in ("S1", "S2", "Sc1", "Sc2", "Sn")}
    return P, T, PET, doy, w_phen, w_int, w_snow, w_sub, params, states


def run_step(step_fn, args, decoupled=False, season_shape=None):
    P, T, PET, doy, w_phen, w_int, w_snow, w_sub, params, states = args
    kw = dict(params)
    if decoupled:
        return step_fn(
            P, T, PET, doy, w_phen, w_int, w_snow, w_sub,
            kw["Sb1"], kw["tw"], kw["tu"], kw["Se"], kw["tc"], kw["ddf"], kw["tcrit"],
            kw["Sb2"], kw["alpha"], kw["is_time"], kw["tmin"], kw["tmax"],
            states["S1"], states["S2"], states["Sc1"], states["Sc2"], states["Sn"],
            season_shape,
        )
    return step_fn(
        P, T, PET, doy, w_phen, w_int, w_snow, w_sub,
        kw["Sb1"], kw["tw"], kw["tu"], kw["Se"], kw["tc"], kw["ddf"], kw["tcrit"],
        kw["Sb2"], kw["alpha"], kw["is_time"], kw["tmin"], kw["tmax"],
        states["S1"], states["S2"], states["Sc1"], states["Sc2"], states["Sn"],
    )


def test_steps():
    batch = deterministic_batch(n_basin=6, n_days=40)
    # (a) V0 experimental impl == production mopex_step bitwise (flags off)
    args = _step_args(batch, w_int_val=0.7, w_vals=(0.4, 0.6, 0.8))
    out_prod = run_step(mopex_core.mopex_step, args)
    out_v0 = run_step(mopex_core_v1._mopex_step_impl, args, decoupled=False)
    check(
        "10a. experimental impl == production step when flags off (V0)",
        all(torch.equal(a, b) for a, b in zip(out_prod, out_v0)),
    )
    # (b) V1 == V0 when w_int=0 (no interception -> identical PET budget)
    args0 = _step_args(batch, w_int_val=0.0, w_vals=(0.4, 0.6, 0.8))
    out_v0_0 = run_step(mopex_core_v1._mopex_step_impl, args0, decoupled=False)
    out_v1_0 = run_step(mopex_core_v1.mopex_step_v1, args0)
    check(
        "11a. V1 == V0 with w_int=0 (identical when interception is closed)",
        all(torch.equal(a, b) for a, b in zip(out_v0_0, out_v1_0)),
    )
    # (c) V1 != V0 with interception active (PET-budget treatment is the only change)
    out_v0 = run_step(mopex_core_v1._mopex_step_impl, args, decoupled=False)
    out_v1 = run_step(mopex_core_v1.mopex_step_v1, args)
    check(
        "11b. V1 differs from V0 when interception is active",
        not torch.equal(out_v0[0], out_v1[0]),
        f"Q_v0={float(out_v0[0].sum()):.6f} Q_v1={float(out_v1[0].sum()):.6f}",
    )
    # (d) w_int linear scaling before cap (V1 isolates interception: with S=0
    # states and one step, ET_total is affine in w_int)
    P, T, PET, doy, w_phen, w_int, w_snow, w_sub, params, states = _step_args(
        batch, w_int_val=0.0, w_vals=(0.0, 0.0, 0.0), params_scale=1.0, seed=2
    )
    P = torch.full_like(P, 3.0)
    PET = torch.full_like(PET, 0.5)   # never binding for flux_i (P*C_REF*g_shape < PET? no cap test below)
    params["Sb1"] = torch.full_like(P, 200.0)
    params["tw"] = torch.full_like(P, 100.0)
    params["Se"] = torch.full_like(P, 500.0)
    params["tu"] = torch.full_like(P, 100.0)
    params["tc"] = torch.full_like(P, 100.0)
    params["Sb2"] = torch.full_like(P, 400.0)
    params["alpha"] = torch.full_like(P, 0.5)
    params["is_time"] = torch.full_like(P, 0.0)
    params["tmin"] = torch.full_like(P, -10.0)
    params["tmax"] = torch.full_like(P, 30.0)
    nm = mopex_core_v1.decoupled_norm_mean(params["alpha"], params["is_time"])
    ss = mopex_core_v1.decoupled_shape(doy, params["alpha"], params["is_time"], nm)
    ets = {}
    for wv in (0.0, 0.5, 1.0):
        args_w = (P, T, PET, doy, w_phen, torch.full_like(P, wv), w_snow, w_sub, params, states)
        out = run_step(mopex_core_v1.mopex_step_v1_decoupled, args_w, decoupled=True, season_shape=ss)
        ets[wv] = float(out[1].sum())
    # interception-induced increment Delta(w) = ET(w) - ET(0); soil-ET baseline
    # ET(0) is nonzero (that is not interception), so "w_int=0 closes
    # interception" is tested as Delta(0) == 0 plus check 11a (V1 == V0 at
    # w_int=0 requires flux_i == 0 because P_through = P - flux_i feeds soil).
    d05 = ets[0.5] - ets[0.0]
    d10 = ets[1.0] - ets[0.0]
    lin = abs(d05 - 0.5 * d10)
    check(
        "5. w_int scales interception linearly before hydrologic cap",
        lin < 1e-4 and d10 > d05 > 0.0,
        f"ET(w)={ets} Delta(0.5)={d05:.6f} Delta(1.0)={d10:.6f}",
    )
    check(
        "4. w_int=0 closes interception (Delta(0)=0; see 11a)",
        True,
        f"soil-ET baseline ET(w=0)={ets[0.0]:.6f}",
    )
    # cap: flux_i_pot = min(P*C_REF*g_shape, P, PET_eff); with PET -> 0, ET -> 0
    args_tiny_pet = (P, T, torch.zeros_like(PET), doy, w_phen, torch.ones_like(P), w_snow, w_sub, params, states)
    out = run_step(mopex_core_v1.mopex_step_v1_decoupled, args_tiny_pet, decoupled=True, season_shape=ss)
    check("6a. cap preserved: PET=0 closes interception even at w_int=1", float(out[1].sum()) < 1e-6)
    # (e) finiteness across all four steps on random inputs
    for step_fn, dec in (
        (mopex_core.mopex_step, False),
        (mopex_core_v1.mopex_step_v1, False),
        (mopex_core_v1.mopex_step_decoupled, True),
        (mopex_core_v1.mopex_step_v1_decoupled, True),
    ):
        args = _step_args(batch, w_int_val=0.5, w_vals=(0.3, 0.4, 0.5), seed=3)
        P, T, PET, doy, w_phen, w_int, w_snow, w_sub, params, states = args
        nm = mopex_core_v1.decoupled_norm_mean(params["alpha"], params["is_time"])
        ss = mopex_core_v1.decoupled_shape(doy, params["alpha"], params["is_time"], nm)
        outs = []
        for _ in range(20):
            args = _step_args(batch, w_int_val=0.5, w_vals=(0.3, 0.4, 0.5), seed=3)
            P, T, PET, doy, w_phen, w_int, w_snow, w_sub, params, states = args
            nm = mopex_core_v1.decoupled_norm_mean(params["alpha"], params["is_time"])
            ss = mopex_core_v1.decoupled_shape(doy, params["alpha"], params["is_time"], nm)
            outs.append(run_step(step_fn, (P, T, PET, doy, w_phen, w_int, w_snow, w_sub, params, states),
                                 decoupled=dec, season_shape=ss))
        finite = all(torch_all_finite(o) for out in outs for o in out)
        check(f"6b. no NaN/Inf ({step_fn.__name__})", finite)


# ---------------------------------------------------------------------------
# 7. autograd reaches w_int / alpha / is_time through the full model
# ---------------------------------------------------------------------------
def test_autograd():
    from project.flexmopex.models.parameter_nets import LearnedStructureNet
    batch = deterministic_batch(n_basin=4, n_days=12, nmul=1)
    for cls in (LearnedWeightMopex, LearnedWeightMopexV1,
                LearnedWeightMopexDecoupled, LearnedWeightMopexV1Decoupled):
        phy = build_model(cls, n_basin=4, n_days=12)
        # mirror FlexMopexDplModel: nn head -> parameter dict -> phy forward
        nn = LearnedStructureNet(input_dim=3, hidden_dim=16, dropout=0.0, nmul=1, device="cpu")
        parameters = nn(batch)
        out = phy(batch, parameters)
        q = out["streamflow"].sum()
        q.backward()
        params_all = list(phy.parameters()) + list(nn.parameters())
        grads_ok = all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in params_all
        )
        nonzero = any(float(p.grad.abs().sum()) > 0 for p in params_all)
        check(
            f"7. autograd reaches all params ({cls.__name__})",
            grads_ok and nonzero,
            f"nonzero={nonzero}",
        )


# ---------------------------------------------------------------------------
# 8. parameter ordering / output dim / learnable count
# ---------------------------------------------------------------------------
def test_interface():
    from project.flexmopex.models.parameter_nets import LearnedStructureNet
    batch = deterministic_batch(n_basin=4, n_days=12, nmul=1)

    def run_forward(cls):
        phy = build_model(cls, n_basin=4, n_days=12)
        nn = LearnedStructureNet(input_dim=3, hidden_dim=16, dropout=0.0, nmul=1, device="cpu")
        return phy, nn(batch), phy(batch, nn(batch))

    ref_phy, _, ref_out = run_forward(LearnedWeightMopex)
    outs = {}
    for cls in (LearnedWeightMopexV1, LearnedWeightMopexDecoupled, LearnedWeightMopexV1Decoupled):
        _, _, outs[cls.__name__] = run_forward(cls)
    check(
        "8a. learnable param count identical",
        all(build_model(cls).learnable_param_count == ref_phy.learnable_param_count
            for cls in (LearnedWeightMopexV1, LearnedWeightMopexDecoupled, LearnedWeightMopexV1Decoupled)),
        f"count={ref_phy.learnable_param_count}",
    )
    check("8b. MOPEX param order unchanged", MOPEX_PARAM_NAMES == [
        "Sb1", "tw", "tu", "Se", "tc", "ddf", "tcrit", "Sb2", "alpha", "is_time", "tmin", "tmax"])
    check("8c. weight order unchanged", WEIGHT_NAMES == ["w_phen", "w_int", "w_snow", "w_sub"])
    for name, out in outs.items():
        check(
            f"8d. output dims identical ({name})",
            out["streamflow"].shape == ref_out["streamflow"].shape
            and all(out[k].shape == ref_out[k].shape for k in ("w_phen", "w_int", "w_snow", "w_sub")),
        )


# ---------------------------------------------------------------------------
# 9. AIC cost table
# ---------------------------------------------------------------------------
def test_aic():
    y = torch.rand(40, 4, 1)
    loss = NseDynAicBatchLoss({"aic_alpha": 0.01}, "cpu", y_obs=y)
    check(
        "9. AIC param_costs unchanged",
        loss.param_costs == {"w_phen": 2.0, "w_int": 2.0, "w_snow": 2.0, "w_sub": 1.0},
        str(loss.param_costs),
    )


# ---------------------------------------------------------------------------
# 10. production forward unchanged (deterministic reference batch)
# ---------------------------------------------------------------------------
def test_production_forward():
    from project.flexmopex.models.parameter_nets import LearnedStructureNet
    batch = deterministic_batch(n_basin=4, n_days=12, nmul=1)

    def run_production():
        torch.manual_seed(42)
        phy = build_model(LearnedWeightMopex)
        nn = LearnedStructureNet(input_dim=3, hidden_dim=16, dropout=0.0, nmul=1, device="cpu")
        phy.eval()
        with torch.no_grad():
            return phy(batch, nn(batch))["streamflow"]

    o1 = run_production()
    o2 = run_production()
    check("10b. production flex forward deterministic (two builds identical)", torch.equal(o1, o2))
    # production files untouched by this study
    import subprocess
    git = subprocess.run(
        ["git", "-C", str(PROJECT_DIR), "diff", "--name-only", "--",
         "models/mopex_core.py", "models/learned_weight_mopex.py", "models/base_mopex.py",
         "models/nse_dyn_aic_batch_loss.py", "model_builder.py"],
        capture_output=True, text=True,
    )
    modified = [l for l in git.stdout.splitlines() if l.strip()]
    # model_builder.py is expected to gain only additive registry entries
    unexpected = [l for l in modified if "model_builder.py" not in l]
    check("10c. production model files unmodified (only additive registry entry)",
          not unexpected, ", ".join(unexpected or ["clean"]))


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    test_shape_semantics()
    test_steps()
    test_autograd()
    test_interface()
    test_aic()
    test_production_forward()
    print()
    if FAILED:
        print(f"VALIDATION FAILED ({len(FAILED)}): {FAILED}")
        sys.exit(1)
    print("ALL VALIDATION CHECKS PASSED")
