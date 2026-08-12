#!/usr/bin/env python3
"""Final MOPEX4 T1a PET-budget closure gate before shared-DPL.

Compares E0 legacy, E1 interception-first shared-PET, and E2 soil-ET-first
shared-PET semantics under matched all-parameter calibration.  F0 and T1a-E0
are reused from the previous pre-DPL validation when the protocol is identical.
No new forcing/data and no new learnable parameters.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

BENCHMARK = Path(__file__).resolve().parents[2]
REPO = BENCHMARK.parents[1]
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src"), str(BENCHMARK / "scripts" / "diagnostics")]

import audit_mopex34_root_cause as A
from batched_cmaes import BatchedCMAES
from dmotpy.models.core.mopex4 import MOPEX4_PARAMS_BOUNDS, mopex4_step
from dmotpy.models.flux.mopex import (
    _mopex_interception_4_legacy,
    mopex_baseflow_1,
    mopex_evap_7,
    mopex_interception_4_liu,
    mopex_melt_1,
    mopex_pet_budget_limit,
    mopex_rainfall_1,
    mopex_recharge_3,
    mopex_saturation_1,
    mopex_snowfall_1,
    mopex_training_context,
)
from objective import full_kge_reference

OUT = BENCHMARK / "results" / "mopex45_phase_fix" / "root_cause_audit" / "liu_interception" / "pet_budget_closure"
OUT.mkdir(parents=True, exist_ok=True)
PREV = BENCHMARK / "results" / "mopex45_phase_fix" / "root_cause_audit" / "liu_interception" / "pre_dpl_validation"
DTYPE = torch.float64
BASIN_IDS = ["8202700", "8150800", "5507600", "11532500"]
WARMUP = SCORED = 365
START = A.START
MODES = ["legacy", "interception_first", "soil_et_first"]
ARMS = ["F0", "T1a-E0", "T1a-E1", "T1a-E2"]
CMA_STARTS = 3
CMA_POPULATION = 10
CMA_GENERATIONS = 40
GRAD_STEPS = 120
GRAD_LR = 0.04
ADAM_SEEDS = [7, 41, 73]
CMA_SEEDS = {"F0": 7, "T1a-E0": 1007, "T1a-E1": 3007, "T1a-E2": 4007}

F0_BOUNDS = [
    [-3.0, 3.0], [0.0, 20.0], [1.0, 2000.0], [0.0, 1.0],
    [0.0, 1.0], [1.0, 365.0], [0.0, 1.0], [0.05, 0.95],
    [1.0, 2000.0], [0.0, 1.0],
]
T1_BOUNDS = [list(v) for v in MOPEX4_PARAMS_BOUNDS.values()]
T1A_ACTIVE = [0, 1, 2, 3, 4, 6, 7, 8, 9]
T1A_BOUNDS = [T1_BOUNDS[i] for i in T1A_ACTIVE]


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_window() -> tuple[torch.Tensor, torch.Tensor]:
    _, xfull, yfull, _ = A.load_context()
    return xfull[START : START + WARMUP + SCORED].to(DTYPE), yfull[START : START + WARMUP + SCORED].to(DTYPE)


def arm_dimension(arm: str) -> int:
    return len(F0_BOUNDS) if arm == "F0" else len(T1A_BOUNDS)


def arm_bounds(arm: str) -> list[list[float]]:
    return F0_BOUNDS if arm == "F0" else T1A_BOUNDS


def latent_to_physical(latent: torch.Tensor, arm: str) -> torch.Tensor:
    bounds = torch.tensor(arm_bounds(arm), dtype=latent.dtype, device=latent.device)
    normalized = torch.sigmoid(latent)
    values = bounds[:, 0] + normalized * (bounds[:, 1] - bounds[:, 0])
    if arm != "F0":
        full = torch.zeros((*latent.shape[:-1], 10), dtype=latent.dtype, device=latent.device)
        full[..., T1A_ACTIVE] = values
        full[..., 5] = 1.0
        return full
    return values


def _kge_and_nse(pred: torch.Tensor, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    score = full_kge_reference(pred, obs, eps=0.1)
    mask = torch.isfinite(pred[..., 0]) & torch.isfinite(obs[:, :, None])
    obs_finite = torch.isfinite(obs)
    valid_obs = torch.where(obs_finite, obs, torch.zeros_like(obs))
    mean_obs = valid_obs.sum(dim=0) / obs_finite.to(pred.dtype).sum(dim=0).clamp_min(1.0)
    centered = torch.where(mask, pred[..., 0] - mean_obs[None, :, None], torch.zeros_like(pred[..., 0]))
    residual = torch.where(mask, pred[..., 0] - obs[:, :, None], torch.zeros_like(pred[..., 0]))
    nse = 1.0 - residual.square().sum(dim=0) / centered.square().sum(dim=0).clamp_min(1.0e-12)
    return score, nse.squeeze(-1)


def simulate(latent: torch.Tensor, arm: str, forcing: torch.Tensor, *, collect: bool = False,
             pet_limiter: str = "hard"):
    """Vectorized MOPEX4 sequential simulation honoring the PET-budget mode.

    Mirrors the production ``mopex4_step`` budget logic exactly.  F0 always
    uses the legacy semantics.  ``pet_limiter`` selects hard vs smooth budget
    limiting for the shared-budget modes.
    """
    if latent.ndim == 2:
        latent = latent[:, None, None, :]
    B, starts, population, _ = latent.shape
    groups = starts * population
    physical = latent_to_physical(latent, arm).reshape(B, groups, 10)
    budget_mode = arm_mode(arm)
    P0 = forcing[:, :, 0].to(latent.dtype)
    T0 = forcing[:, :, 1].to(latent.dtype)
    PET0 = forcing[:, :, 2].to(latent.dtype)
    doy = forcing[:, :, 3].to(latent.dtype)
    states = [torch.full((B, groups), 1.0e-6, dtype=latent.dtype, device=latent.device) for _ in range(5)]
    q_rows, et_rows = [], []
    diag = {key: [] for key in ["i", "et1", "et2", "pr", "pet", "state_sum"]} if collect else None
    with mopex_training_context(pet_budget=budget_mode, pet_limiter=pet_limiter):
        for t in range(forcing.shape[0]):
            P = P0[t, :, None].expand(B, groups)
            T = T0[t, :, None].expand(B, groups)
            PET = PET0[t, :, None].expand(B, groups)
            DOY = doy[t, :, None].expand(B, groups)
            p = [physical[..., i] for i in range(10)]
            tcrit, ddf, sb1, tw = p[0], p[1], p[2], p[3]
            int0, int1, tu, se, sb2, tc = p[4], p[5], p[6], p[7], p[8], p[9]
            sn, soil, sub, fast, slow = states
            ps = mopex_snowfall_1(P, T, tcrit)
            pr = mopex_rainfall_1(P, T, tcrit)
            qn = mopex_melt_1(ddf, tcrit, T, sn)
            sn_new = sn + ps - qn
            soil = soil + pr + qn
            if arm == "F0":
                i_pot = _mopex_interception_4_legacy(pr, DOY, int0, int1)
                flux_i = i_pot
                et1 = torch.minimum(mopex_evap_7(soil, sb1, PET, 1.0, 1e-6), soil)
                soil = soil - et1
                pet_rem = PET - et1
            else:
                i_pot = mopex_interception_4_liu(pr, int0, int1)
                if budget_mode == "interception_first":
                    flux_i = mopex_pet_budget_limit(i_pot, PET)
                    pet_after_i = PET - flux_i
                    et1 = torch.minimum(mopex_evap_7(soil, sb1, pet_after_i, 1.0, 1e-6), soil)
                    soil = soil - et1
                    pet_rem = pet_after_i - et1
                elif budget_mode == "soil_et_first":
                    et1 = torch.minimum(mopex_evap_7(soil, sb1, PET, 1.0, 1e-6), soil)
                    soil = soil - et1
                    flux_i = mopex_pet_budget_limit(i_pot, PET - et1)
                    pet_rem = PET - et1 - flux_i
                else:
                    flux_i = i_pot
                    et1 = torch.minimum(mopex_evap_7(soil, sb1, PET, 1.0, 1e-6), soil)
                    soil = soil - et1
                    pet_rem = PET - et1
            interception = torch.minimum(flux_i, soil)
            soil = soil - interception
            q1f = torch.minimum(mopex_saturation_1(pr + qn, soil, sb1, nearzero=1e-6), soil)
            soil = soil - q1f
            qw = torch.minimum(mopex_recharge_3(tw, soil), soil)
            soil_new = soil - qw
            sub = sub + qw
            q2f = torch.minimum(mopex_saturation_1(qw, sub, sb2, nearzero=1e-6), sub)
            sub = sub - q2f
            q2u = mopex_baseflow_1(tu, sub)
            sub = sub - q2u
            et2_pet = PET if budget_mode == "legacy" else pet_rem
            et2 = torch.minimum(mopex_evap_7(sub, se * sb2, et2_pet, 1.0, 1e-6), sub)
            sub_new = sub - et2
            fast = fast + q1f + q2f
            qf = mopex_baseflow_1(tc, fast)
            fast_new = fast - qf
            slow = slow + q2u
            qs = mopex_baseflow_1(tc, slow)
            slow_new = slow - qs
            q_rows.append(qf + qs)
            et_rows.append(et1 + et2 + interception)
            states = [sn_new, soil_new, sub_new, fast_new, slow_new]
            if collect:
                diag["i"].append(interception)
                diag["et1"].append(et1)
                diag["et2"].append(et2)
                diag["pr"].append(pr)
                diag["pet"].append(PET)
                diag["state_sum"].append(sum(states))
    q = torch.stack(q_rows).reshape(-1, B, starts, population)
    et = torch.stack(et_rows).reshape(-1, B, starts, population)
    if collect:
        diag = {key: torch.stack(value) for key, value in diag.items()}
    return q, et, diag


def mode_hook():
    from dmotpy.models.flux.mopex import _pet_budget_mode
    return _pet_budget_mode()


def arm_mode(arm: str) -> str:
    if arm == "F0" or arm == "T1a-E0":
        return "legacy"
    if arm == "T1a-E1":
        return "interception_first"
    if arm == "T1a-E2":
        return "soil_et_first"
    return "legacy"


def cma_evaluate(latent: torch.Tensor, arm: str, forcing: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
    q, _, _ = simulate(latent, arm, forcing)
    scores, _ = _kge_and_nse(q[WARMUP:], observations[WARMUP:])
    return scores


def protocol() -> dict:
    return {
        "basins": BASIN_IDS,
        "window": {"warmup_days": WARMUP, "scored_days": SCORED, "start_index": START},
        "device": "cpu",
        "objective": "repository streaming/full KGE with eps=0.1; NSE descriptive only",
        "cmaes": {"implementation": "project/benchmark/src/batched_cmaes.py", "starts": CMA_STARTS,
                  "population": CMA_POPULATION, "generations": CMA_GENERATIONS, "stdev_init": 0.25,
                  "active_covariance": True, "initialization": "zero latent center, matched arm protocol",
                  "solver_seed_by_arm": CMA_SEEDS, "stopping_rule": "fixed generation budget"},
        "gradient": {"optimizer": "torch.optim.Adam", "steps": GRAD_STEPS, "lr": GRAD_LR,
                     "seeds": ADAM_SEEDS, "initialization": "matched zero latent center plus seeded 0.5 latent perturbation",
                     "stopping_rule": "fixed step budget"},
        "reused_from": "pre_dpl_validation (identical protocol) for F0 and T1a-E0",
        "training_started": False, "shared_dpl_started": False, "production_default_changed": False,
    }


def stage_semantics_docs() -> None:
    (OUT / "pet_semantics_before_change.md").write_text(
        """# PET semantics before change (MOPEX4 T1a)

Audited against `dmotpy/models/core/mopex4.py` and `dmotpy/models/flux/mopex.py`.

## Exact flux/state update order

1. Snow bucket: `ps = snowfall_1(P,T,tcrit)`, `pr = rainfall_1(P,T,tcrit)`,
   `qn = melt_1(...)`; `Sn = Sn + ps - qn`.
2. Soil bucket `S1`:
   - `S1 = S1 + pr + qn`
   - `et1 = min(evap_7(S1, Sb1, PET, dt), S1)`; `S1 = S1 - et1`
   - `I = min(liu_interception_4(pr, S_eff, c), S1)`; `S1 = S1 - I`
   - `q1f = min(saturation_1(...), S1)`; `qw = min(recharge_3(tw, S1), S1)`
3. Subsurface bucket `S2`:
   - `S2 = S2 + qw`; `q2f = min(saturation_1(...), S2)`; `q2u = baseflow_1(tu, S2)`
   - `et2 = min(evap_7(S2, se*s3max, PET, dt), S2)`
4. Routing: fast/slow unit-response buckets.
5. `Q_total = qf + qs`; `ET_total = et1 + et2 + I`.

## Where PET enters

- `evap_7(S, Smax, Ep, dt) = min(Ep * clamp(S/Smax, max=1) * dt, S)`: Ep is a
  demand that is scaled by storage fullness and capped by available storage.
- ET1 and ET2 **each independently see the full daily PET**.  There is no
  shared budget between them in the current code.

## Is interception an immediate same-day evaporative loss?

Yes.  `I` is subtracted directly from the soil bucket on the same day and is
included in `ET_total`.  It is not stored in a canopy store.  Therefore a
shared-PET allocation changes only how the daily PET demand is partitioned
between interception and soil ET; it does not change the process meaning of
`I` (immediate evaporative loss).

## Does changing PET passed to ET change state semantics?

`evap_7` is demand-driven and capped by storage.  Passing a smaller PET amount
only lowers the demand offered to each store; the store update rule is
unchanged.  With the shared modes, `et1`/`et2` are additionally coupled through
the remaining-budget cascade, but state non-negativity and update order are
preserved.

## Conclusion

`I` is semantically an immediate daily evaporative loss, so shared-PET
allocation is a budget-consistency change, not a process redesign.
""",
        encoding="utf-8",
    )
    (OUT / "pet_budget_mode_spec.md").write_text(
        """# PET budget modes

All modes are selected via `mopex_training_context(pet_budget=...)`; default
is `legacy` and exactly matches current production behavior.  No public
signature changes and no new learnable parameters.

- `legacy`: ET1 uses full PET, interception is unconstrained by PET, ET2 uses
  full PET.  Current production behavior.
- `interception_first` (E1): `I = min(I_pot, PET)`; remaining `PET - I` is
  offered to ET1, then the remainder to ET2.
- `soil_et_first` (E2): ET1 keeps current full-PET priority; the residual
  `PET - et1` is offered to interception, then the remainder to ET2.

Budget limiter: `I = min(I_pot, PET_available)` (exact hard budget).  A smooth
non-trainable alternative `I = I_pot*PET/(I_pot+PET+eps)` is available via
`smooth=True` in `mopex_pet_budget_limit` for gradient comparison only.

Both E1 and E2 guarantee `I + ET1 + ET2 <= PET` (up to floating tolerance)
because each subsequent flux receives only the non-negative remainder.
""",
        encoding="utf-8",
    )


def stage_boundary() -> bool:
    rows = []
    all_pass = True
    s_grid = [1e-5, 0.5, 2.0, 4.9999]
    for mode in MODES:
        for s in s_grid:
            for pr in [0.0, 1e-4, s, 50.0]:
                for pet in [0.0, 0.01, 1.0, 20.0]:
                    P = torch.tensor(pr, dtype=DTYPE)
                    T = torch.tensor(12.0, dtype=DTYPE)
                    PET = torch.tensor(pet, dtype=DTYPE)
                    doy = torch.tensor(180.0, dtype=DTYPE)
                    p = [torch.tensor(v, dtype=DTYPE) for v in [0.0, 4.0, 200.0, 0.1, s, 1.0, 0.1, 0.5, 300.0, 0.2]]
                    with mopex_training_context(pet_budget=mode):
                        out = mopex4_step(P, T, PET, *p, *([torch.tensor(1e-6, dtype=DTYPE)] * 5),
                                          doy=doy, nearzero=1e-6)
                    # component-level mirror for flux inspection
                    i_pot = mopex_interception_4_liu(P, torch.tensor(s, dtype=DTYPE), torch.tensor(1.0, dtype=DTYPE))
                    with mopex_training_context(pet_budget=mode):
                        i, et1, et2 = _component_fluxes(P, T, PET, doy, s)
                    finite = bool(all(torch.isfinite(tensor) for tensor in out) and torch.isfinite(i) and torch.isfinite(et1) and torch.isfinite(et2))
                    i_le_pr = bool(i <= pr + 1e-9 and i >= -1e-9)
                    budget_ok = True
                    if mode != "legacy":
                        budget_ok = bool(i + et1 + et2 <= pet + 1e-6)
                    et_ge_0 = bool(et1 >= -1e-9 and et2 >= -1e-9)
                    ok = finite and i_le_pr and budget_ok and et_ge_0
                    all_pass &= bool(ok)
                    rows.append({"mode": mode, "S_eff": s, "Pr": pr, "PET": pet,
                                 "I": float(i), "ET1": float(et1), "ET2": float(et2),
                                 "I_plus_ET1_plus_ET2": float(i + et1 + et2),
                                 "I_le_Pr": i_le_pr, "budget_closed": budget_ok,
                                 "ET_nonneg": et_ge_0, "finite": finite, "pass": ok})
    write_csv("pet_budget_boundary_tests.csv", rows)
    return all_pass


def _component_fluxes(P, T, PET, doy, s_eff):
    """Single-step flux components with storage close to empty."""
    S1 = torch.tensor(1e-6, dtype=P.dtype)
    S2 = torch.tensor(1e-6, dtype=P.dtype)
    pr = mopex_rainfall_1(P, T, torch.tensor(0.0, dtype=P.dtype))
    s = torch.tensor(s_eff, dtype=P.dtype)
    c = torch.tensor(1.0, dtype=P.dtype)
    mode = mode_hook()
    i_pot = mopex_interception_4_liu(pr, s, c)
    if mode == "interception_first":
        i = mopex_pet_budget_limit(i_pot, PET)
        et1 = torch.minimum(mopex_evap_7(S1, torch.tensor(200.0, dtype=P.dtype), PET - i, 1.0, 1e-6), S1)
        et2 = torch.minimum(mopex_evap_7(S2, torch.tensor(150.0, dtype=P.dtype), PET - i - et1, 1.0, 1e-6), S2)
    elif mode == "soil_et_first":
        et1 = torch.minimum(mopex_evap_7(S1, torch.tensor(200.0, dtype=P.dtype), PET, 1.0, 1e-6), S1)
        i = mopex_pet_budget_limit(i_pot, PET - et1)
        et2 = torch.minimum(mopex_evap_7(S2, torch.tensor(150.0, dtype=P.dtype), PET - et1 - i, 1.0, 1e-6), S2)
    else:
        i = i_pot
        et1 = torch.minimum(mopex_evap_7(S1, torch.tensor(200.0, dtype=P.dtype), PET, 1.0, 1e-6), S1)
        et2 = torch.minimum(mopex_evap_7(S2, torch.tensor(150.0, dtype=P.dtype), PET, 1.0, 1e-6), S2)
    return i, et1, et2


def stage_gradient_audit(forcing: torch.Tensor) -> None:
    rows = []
    pr = mopex_rainfall_1(forcing[:, 0, 0], forcing[:, 0, 1], torch.tensor(0.0, dtype=DTYPE)).clamp_min(0)
    for mode in ["interception_first", "soil_et_first", "legacy"]:
        for s_eff in [0.5, 2.0, 4.0]:
            for pet_scale in [0.1, 1.0, 3.0]:
                s = torch.tensor(s_eff, dtype=DTYPE, requires_grad=True)
                pet = torch.tensor(pet_scale, dtype=DTYPE)
                i_pot = mopex_interception_4_liu(pr, s, torch.tensor(1.0, dtype=DTYPE))
                if mode == "interception_first":
                    i = mopex_pet_budget_limit(i_pot, pet)
                elif mode == "soil_et_first":
                    et1 = torch.minimum(mopex_evap_7(torch.tensor(50.0, dtype=DTYPE), torch.tensor(200.0, dtype=DTYPE), pet, 1.0, 1e-6), torch.tensor(50.0, dtype=DTYPE))
                    i = mopex_pet_budget_limit(i_pot, pet - et1)
                else:
                    i = i_pot
                i_smooth = mopex_pet_budget_limit(i_pot, pet, smooth=True)
                di = torch.autograd.grad(i.sum(), s, retain_graph=True)[0]
                di_smooth = torch.autograd.grad(i_smooth.sum(), s)[0]
                # analytic dI_pot/dS_eff for the Liu kernel
                x = pr / s_eff
                d_i_pot = 1.0 - torch.exp(-x) * (1.0 + x)
                below = bool((i_pot < pet).all()) if mode == "interception_first" else None
                rows.append({"mode": mode, "S_eff": s_eff, "PET": pet_scale,
                             "dI_pot_dS_eff_analytic_median": float(d_i_pot.median()),
                             "dI_dS_eff_autograd_median": float(di.median()),
                             "dI_smooth_dS_eff_autograd_median": float(di_smooth.median()),
                             "finite_hard": bool(torch.isfinite(di).all()),
                             "finite_smooth": bool(torch.isfinite(di_smooth).all()),
                             "max_abs_hard": float(di.abs().max()),
                             "max_abs_smooth": float(di_smooth.abs().max())})
    # end-to-end production step backward through raw S_eff logit
    for mode in ["legacy", "interception_first", "soil_et_first"]:
        for b in range(4):
            raw = torch.tensor(-0.5, dtype=DTYPE, requires_grad=True)
            s_eff = 1e-5 + torch.sigmoid(raw) * (5.0 - 1e-5)
            with mopex_training_context(pet_budget=mode):
                qs, ets = [], []
                states = [torch.tensor(1e-6, dtype=DTYPE) for _ in range(5)]
                for t in range(WARMUP + SCORED):
                    out = mopex4_step(forcing[t, b, 0], forcing[t, b, 1], forcing[t, b, 2],
                                      *[torch.tensor(v, dtype=DTYPE) for v in [0.0, 4.0, 200.0, 0.1]],
                                      s_eff, torch.tensor(1.0, dtype=DTYPE),
                                      *[torch.tensor(v, dtype=DTYPE) for v in [0.1, 0.5, 300.0, 0.2]],
                                      states[1], states[2], states[3], states[4], states[0],
                                      doy=forcing[t, b, 3], nearzero=1e-6)
                    qs.append(out[0]); ets.append(out[1]); states = list(out[2:7])
                q = torch.stack(qs); et = torch.stack(ets)
                score = full_kge_reference(q[WARMUP:].reshape(-1, 1, 1), forcing[WARMUP:, b, 0:1] * 0 + 1.0, eps=0.1)
                # simple finite-loss backward on streamflow mean as autograd sanity
                loss = q[WARMUP:].mean()
                loss.backward()
                rows.append({"mode": mode, "basin": BASIN_IDS[b], "end_to_end": True,
                             "raw_S_eff_grad": float(raw.grad.item()) if raw.grad is not None else float("nan"),
                             "loss_finite": bool(torch.isfinite(loss)),
                             "grad_finite": bool(raw.grad is not None and torch.isfinite(raw.grad))})
    write_csv("pet_budget_gradient_audit.csv", rows)


def step_diag(P, T, PET, tcrit, ddf, Sb1, tw, S_eff, c, tu, Se, Sb2, tc,
              S1, S2, Sc1, Sc2, Sn, *, doy, nearzero=1e-6):
    """Production-equivalent MOPEX4 step that also returns component fluxes.

    Mirrors the production ``mopex4_step`` budget logic; the PET-budget mode is
    read from the training context so the returned fluxes are exactly the ones
    used by the state update.
    """
    from dmotpy.models.flux.mopex import _pet_budget_mode as _mode
    from dmotpy.models.flux.mopex import _pet_limiter as _limiter
    pet_budget_mode = _mode()
    pet_smooth = _limiter() == "smooth"
    Sn = torch.relu(Sn); S1 = torch.relu(S1); S2 = torch.relu(S2)
    Sc1 = torch.relu(Sc1); Sc2 = torch.relu(Sc2)
    ps = mopex_snowfall_1(P, T, tcrit)
    pr = mopex_rainfall_1(P, T, tcrit)
    qn = mopex_melt_1(ddf, tcrit, T, Sn)
    Sn_new = Sn + ps - qn
    S1 = S1 + pr + qn
    i_pot = mopex_interception_4_liu(pr, S_eff, c, nearzero=nearzero)
    if pet_budget_mode == "legacy":
        et1 = torch.minimum(mopex_evap_7(S1, Sb1, PET, 1.0, nearzero), S1)
        S1 = S1 - et1
        i = i_pot
        pet_rem = PET - et1
    elif pet_budget_mode == "interception_first":
        i = mopex_pet_budget_limit(i_pot, PET, smooth=pet_smooth)
        pet_after_i = PET - i
        et1 = torch.minimum(mopex_evap_7(S1, Sb1, pet_after_i, 1.0, nearzero), S1)
        S1 = S1 - et1
        pet_rem = pet_after_i - et1
    else:  # soil_et_first
        et1 = torch.minimum(mopex_evap_7(S1, Sb1, PET, 1.0, nearzero), S1)
        S1 = S1 - et1
        i = mopex_pet_budget_limit(i_pot, PET - et1, smooth=pet_smooth)
        pet_rem = PET - et1 - i
    i = torch.minimum(i, S1)
    S1 = S1 - i
    q1f = torch.minimum(mopex_saturation_1(pr + qn, S1, Sb1, nearzero=nearzero), S1)
    S1 = S1 - q1f
    qw = torch.minimum(mopex_recharge_3(tw, S1), S1)
    S1_new = S1 - qw
    S2 = S2 + qw
    q2f = torch.minimum(mopex_saturation_1(qw, S2, Sb2, nearzero=nearzero), S2)
    S2 = S2 - q2f
    q2u = mopex_baseflow_1(tu, S2)
    S2 = S2 - q2u
    et2_pet = PET if pet_budget_mode == "legacy" else pet_rem
    et2 = torch.minimum(mopex_evap_7(S2, Se * Sb2, et2_pet, 1.0, nearzero), S2)
    S2_new = S2 - et2
    Sc1 = Sc1 + q1f + q2f
    qf = mopex_baseflow_1(tc, Sc1)
    Sc1_new = Sc1 - qf
    Sc2 = Sc2 + q2u
    qs = mopex_baseflow_1(tc, Sc2)
    Sc2_new = Sc2 - qs
    return (qf + qs, et1 + et2 + i, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new,
            i, et1, et2, pr)


def stage_water_balance(forcing: torch.Tensor, observations: torch.Tensor) -> tuple[list, list]:
    wb_rows, partition_rows = [], []
    settings = [("low", 0.05), ("mid", 0.6), ("high", 2.5)]
    for b in range(4):
        for mode in MODES:
            for name, s_eff in settings:
                s = torch.tensor(s_eff, dtype=DTYPE)
                common = [torch.tensor(v, dtype=DTYPE) for v in [0.0, 4.0, 200.0, 0.1, 0.0, 1.0, 0.1, 0.5, 300.0, 0.2]]
                common[4] = s
                i_s, et1_s, et2_s, q_s, pet_s = [], [], [], [], []
                states = [torch.tensor(1e-6, dtype=DTYPE) for _ in range(5)]
                state_sum = []
                with mopex_training_context(pet_budget=mode):
                    for t in range(WARMUP + SCORED):
                        out = step_diag(forcing[t, b, 0], forcing[t, b, 1], forcing[t, b, 2],
                                        *common, states[1], states[2], states[3], states[4], states[0],
                                        doy=forcing[t, b, 3], nearzero=1e-6)
                        q_s.append(out[0]); i_s.append(out[7]); et1_s.append(out[8]); et2_s.append(out[9])
                        pet_s.append(forcing[t, b, 2])
                        states = list(out[2:7]); state_sum.append(sum(states))
                qv = torch.stack(q_s); iv = torch.stack(i_s); et1v = torch.stack(et1_s)
                et2v = torch.stack(et2_s); petv = torch.stack(pet_s); state_sum = torch.stack(state_sum)
                p_total = float(forcing[:WARMUP + SCORED, b, 0].sum())
                et_total = float((iv + et1v + et2v).sum())
                q_total = float(qv.sum())
                resid = p_total - et_total - q_total - (float(state_sum[-1]) - 5e-6)
                state_delta = torch.empty_like(state_sum)
                state_delta[0] = state_sum[0] - 5e-6
                state_delta[1:] = state_sum[1:] - state_sum[:-1]
                daily_res = forcing[:, b, 0] - (iv + et1v + et2v) - qv - state_delta
                wb_pass = bool(daily_res.abs().max() < 1e-5)
                exceed = (iv + et1v + et2v - petv).clamp_min(0)
                scored = slice(WARMUP, WARMUP + SCORED)
                exceed_fraction = float((exceed[scored] > 0).double().mean())
                wb_rows.append({"basin_id": BASIN_IDS[b], "mode": mode, "setting": name,
                                "P_total": p_total, "ET_total": et_total, "Q_total": q_total,
                                "residual": resid, "max_daily_abs_residual": float(daily_res.abs().max()),
                                "water_balance_pass": wb_pass,
                                "exceedance_day_fraction_scored": exceed_fraction,
                                "max_exceedance": float(exceed[scored].max()),
                                "sum_exceedance": float(exceed[scored].sum())})
                partition_rows.append({"basin_id": BASIN_IDS[b], "mode": mode, "setting": name,
                                       "I/P": float(iv.sum() / max(p_total, 1e-12)),
                                       "ET1/P": float(et1v.sum() / max(p_total, 1e-12)),
                                       "ET2/P": float(et2v.sum() / max(p_total, 1e-12)),
                                       "ET/P": float((iv + et1v + et2v).sum() / max(p_total, 1e-12)),
                                       "Q/P": float(q_total / max(p_total, 1e-12)),
                                       "annual_I": float(iv.sum()), "annual_ET1": float(et1v.sum()),
                                       "annual_ET2": float(et2v.sum()), "annual_Q": q_total,
                                       "sum_I_ET1_ET2": float((iv + et1v + et2v).sum()), "sum_PET": float(petv.sum())})
    write_csv("pet_budget_water_balance.csv", wb_rows)
    write_csv("pet_budget_component_partition.csv", partition_rows)
    return wb_rows, partition_rows


def load_previous_rows() -> dict:
    """Reuse F0 and T1a-E0 results from pre_dpl_validation (identical protocol)."""
    import csv as _csv
    out = {"cma": {}, "grad": {}}
    for fn, key in [("cmaes_all_parameter_results.csv", "cma"), ("gradient_all_parameter_results.csv", "grad")]:
        with (PREV / fn).open() as handle:
            for row in _csv.DictReader(handle):
                if row["arm"] in ("F0", "T1a"):
                    out[key].setdefault(row["arm"], []).append(row)
    return out


def run_cma(forcing, observations, arm):
    dim = arm_dimension(arm)
    seed = CMA_SEEDS[arm]
    solver = BatchedCMAES(4 * CMA_STARTS, dim, CMA_POPULATION, stdev_init=0.25,
                          active=True, seed=seed, device="cpu")
    solver.set_centers(torch.zeros((4 * CMA_STARTS, dim), dtype=DTYPE))
    for _ in range(CMA_GENERATIONS):
        z, y, x = solver.ask()
        latent = x.reshape(4, CMA_STARTS, CMA_POPULATION, dim)
        score = cma_evaluate(latent, arm, forcing, observations)
        solver.tell(z, y, x, score.reshape(4 * CMA_STARTS, CMA_POPULATION))
    best_latent = solver.state.best_latent.reshape(4, CMA_STARTS, dim).detach().clone()
    return best_latent


def run_gradient(forcing, observations, arm):
    dim = arm_dimension(arm)
    results = []
    for seed in ADAM_SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        raw = (torch.randn((4, dim), generator=generator, dtype=DTYPE) * 0.5).requires_grad_(True)
        optimizer = torch.optim.Adam([raw], lr=GRAD_LR)
        best_score = torch.full((4,), -torch.inf, dtype=DTYPE)
        best_raw = raw.detach().clone()
        for _ in range(GRAD_STEPS):
            optimizer.zero_grad()
            q, _, _ = simulate(raw[:, None, None, :], arm, forcing)
            scores, _ = _kge_and_nse(q[WARMUP:], observations[WARMUP:])
            basin = scores[:, 0, 0]
            loss = 1.0 - basin.mean()
            loss.backward()
            optimizer.step()
            improved = basin.detach() > best_score
            best_score[improved] = basin.detach()[improved]
            best_raw[improved] = raw.detach()[improved]
        q, _, _ = simulate(best_raw[:, None, None, :], arm, forcing)
        scores, _ = _kge_and_nse(q[WARMUP:], observations[WARMUP:])
        for b in range(4):
            results.append((b, seed, float(scores[b, 0, 0]), best_raw[b].detach().clone()))
    return results


def optimize_arms(forcing, observations) -> dict:
    rows_cma, rows_grad = [], []
    chosen = {}
    prev = load_previous_rows()
    basin_index = {bid: i for i, bid in enumerate(BASIN_IDS)}
    for arm in ARMS:
        if arm in ("F0", "T1a-E0"):
            key = "F0" if arm == "F0" else "T1a"
            cma_rows = []
            for r in prev["cma"][key]:
                b = basin_index[r["basin_id"]]
                physical = torch.tensor([float(r.get(f"p{i}", float("nan"))) for i in range(10)], dtype=DTYPE)
                cma_rows.append((b, int(r["restart"]), float(r["KGE"]), None, physical))
            grad_rows = []
            for r in prev["grad"][key]:
                b = basin_index[r["basin_id"]]
                physical = torch.tensor([float(r.get(f"p{i}", float("nan"))) for i in range(10)], dtype=DTYPE)
                grad_rows.append((b, int(r.get("seed", 0)), float(r["KGE"]), None, physical))
                rows_grad.append({"arm": arm, "basin_id": r["basin_id"], "seed": int(r.get("seed", 0)), "KGE": float(r["KGE"])})
        else:
            cma_latents = run_cma(forcing, observations, arm)
            cma_rows = []
            for b in range(4):
                for restart in range(CMA_STARTS):
                    q, _, _ = simulate(cma_latents[b, restart][None, None, None, :], arm, forcing[:, b:b+1])
                    score, _ = _kge_and_nse(q[WARMUP:], observations[WARMUP:, b:b+1])
                    physical = latent_to_physical(cma_latents[b, restart][None, None, None, :], arm).flatten()
                    cma_rows.append((b, restart, float(score[0, 0, 0]), cma_latents[b, restart].detach().clone(), physical))
            grad_rows = []
            for (b, seed, kge, latent) in run_gradient(forcing, observations, arm):
                physical = latent_to_physical(latent[None, None, None, :], arm).flatten()
                grad_rows.append((b, seed, kge, latent, physical))
                rows_grad.append({"arm": arm, "basin_id": BASIN_IDS[b], "seed": seed, "KGE": kge})
        for row in cma_rows:
            rows_cma.append({"arm": arm, "basin_id": BASIN_IDS[row[0]], "restart": row[1], "KGE": row[2]})
        chosen[arm] = {}
        for b in range(4):
            candidates = [(sc, lat, "CMA-ES", rid, ph) for bb, rid, sc, lat, ph in cma_rows if bb == b]
            candidates += [(sc, lat, "Adam", rid, ph) for bb, rid, sc, lat, ph in grad_rows if bb == b]
            chosen[arm][b] = max(candidates, key=lambda item: item[0])
    write_csv("pet_budget_cmaes_results.csv", rows_cma)
    write_csv("pet_budget_gradient_results.csv", rows_grad)
    return chosen, rows_cma, rows_grad


def boundary_pressure(chosen, forcing) -> list:
    rows = []
    for arm in ["T1a-E0", "T1a-E1", "T1a-E2"]:
        for b in range(4):
            score, latent, method, rid, physical = chosen[arm][b]
            s = physical[4].item()
            lo, hi = T1_BOUNDS[4]
            dist = min((s - lo) / (hi - lo), (hi - s) / (hi - lo))
            raw = latent[3].item() if latent is not None and latent.numel() > 3 else float("nan")
            rows.append({"arm": arm, "basin_id": BASIN_IDS[b], "selection": method,
                         "S_eff": s, "raw_pre_activation": raw,
                         "distance_to_bound": dist, "boundary_hit": dist <= .02,
                         "transform_derivative": float(torch.sigmoid(torch.tensor(raw)) * (1 - torch.sigmoid(torch.tensor(raw)))) if math.isfinite(raw) else float("nan")})
    write_csv("pet_budget_boundary_pressure.csv", rows)
    return rows


def compensation_surface(chosen, forcing, observations) -> list:
    rows = []
    for arm in ["F0", "T1a-E0", "T1a-E1", "T1a-E2"]:
        for b in range(4):
            score, latent, method, rid, physical = chosen[arm][b]
            base = physical.clone()
            int_param = "alpha" if arm == "F0" else "S_eff"
            grid = torch.linspace(.05, .95, 7) if arm == "F0" else torch.linspace(.1, 4.9, 7)
            sb_grid = torch.linspace(max(10.0, base[2].item() * .7), min(2000.0, base[2].item() * 1.3), 7)
            for value in grid:
                for sb1 in sb_grid:
                    candidate = base.clone()
                    candidate[2] = sb1
                    candidate[4] = value
                    raw = []
                    if arm == "F0":
                        for i, (lo, hi) in enumerate(F0_BOUNDS):
                            raw.append(float(torch.logit(((candidate[i] - lo) / (hi - lo)).clamp(1e-7, 1 - 1e-7))))
                    else:
                        for i in T1A_ACTIVE:
                            lo, hi = T1_BOUNDS[i]
                            raw.append(float(torch.logit(((candidate[i] - lo) / (hi - lo)).clamp(1e-7, 1 - 1e-7))))
                    latent_candidate = torch.tensor(raw, dtype=DTYPE)
                    q, _, _ = simulate(latent_candidate[None, None, None, :], arm, forcing[:, b:b+1])
                    kge = full_kge_reference(q[WARMUP:], observations[WARMUP:, b:b+1], eps=.1).item()
                    rows.append({"arm": arm, "basin_id": BASIN_IDS[b], "interception_parameter": int_param,
                                 "interception_value": float(value), "Sb1": float(sb1), "KGE": kge})
    write_csv("pet_budget_compensation_surface.csv", rows)
    return rows


def summarize_compensation(rows) -> dict:
    result = {}
    for arm in ARMS:
        correlations = []
        for basin_id in BASIN_IDS:
            basin_rows = [r for r in rows if r["arm"] == arm and r["basin_id"] == basin_id]
            values = [float(r["KGE"]) for r in basin_rows]
            if len(values) < 2:
                continue
            threshold = np.quantile(values, .75)
            top = [r for r in basin_rows if float(r["KGE"]) >= threshold]
            if len(top) > 2 and len({r["interception_value"] for r in top}) > 1 and len({r["Sb1"] for r in top}) > 1:
                correlations.append(abs(float(np.corrcoef([float(r["interception_value"]) for r in top], [float(r["Sb1"]) for r in top])[0, 1])))
        result[arm] = float(np.mean(correlations)) if correlations else 0.0
    return result


def medians(chosen, rows_cma, rows_grad) -> dict:
    result = {}
    for arm in ARMS:
        cma = [r["KGE"] for r in rows_cma if r["arm"] == arm]
        grad = [r["KGE"] for r in rows_grad if r["arm"] == arm]
        result[arm] = {"cma_median": float(np.median(cma)) if cma else float("nan"),
                       "grad_median": float(np.median(grad)) if grad else float("nan")}
    return result


def main() -> None:
    forcing, observations = load_window()
    (OUT / "protocol_and_budget.json").write_text(json.dumps(protocol(), indent=2), encoding="utf-8")
    stage_semantics_docs()
    print("Boundary stage")
    boundary_pass = stage_boundary()
    print("Gradient audit stage")
    stage_gradient_audit(forcing)
    print("Water balance stage")
    wb_rows, partition_rows = stage_water_balance(forcing, observations)
    print("Optimization stage (E1/E2; F0/T1a-E0 reused)")
    chosen, rows_cma, rows_grad = optimize_arms(forcing, observations)
    pressure_rows = boundary_pressure(chosen, forcing)
    comp_rows = compensation_surface(chosen, forcing, observations)
    comp_summary = summarize_compensation(comp_rows)
    med = medians(chosen, rows_cma, rows_grad)
    e0_exceed = max((r["exceedance_day_fraction_scored"] for r in wb_rows if r["mode"] == "legacy"), default=float("nan"))
    e1_exceed = max((r["exceedance_day_fraction_scored"] for r in wb_rows if r["mode"] == "interception_first"), default=float("nan"))
    e2_exceed = max((r["exceedance_day_fraction_scored"] for r in wb_rows if r["mode"] == "soil_et_first"), default=float("nan"))
    wb_pass = all(bool(r["water_balance_pass"]) for r in wb_rows)
    pressure = {arm: sum(1 for r in pressure_rows if r["arm"] == arm and r["boundary_hit"]) for arm in ["T1a-E0", "T1a-E1", "T1a-E2"]}
    cma_e1 = med["T1a-E1"]["cma_median"]; cma_e2 = med["T1a-E2"]["cma_median"]
    grad_e1 = med["T1a-E1"]["grad_median"]; grad_e2 = med["T1a-E2"]["grad_median"]
    e1_vs_e0 = cma_e1 - med["T1a-E0"]["cma_median"]
    e2_vs_e0 = cma_e2 - med["T1a-E0"]["cma_median"]
    comp_e0 = comp_summary.get("T1a-E0", 0.0); comp_e1 = comp_summary.get("T1a-E1", 0.0); comp_e2 = comp_summary.get("T1a-E2", 0.0)
    closure_ok = e1_exceed < 1e-3 and e2_exceed < 1e-3
    reasonable = (cma_e1 > -0.5 and cma_e2 > -0.5)
    no_collapse = (e1_vs_e0 > -0.25 and e2_vs_e0 > -0.25)
    comp_not_worse = (comp_e1 <= comp_e0 + 0.15 and comp_e2 <= comp_e0 + 0.15)
    grad_e0 = med["T1a-E0"]["grad_median"]
    grad_degraded = (grad_e1 < grad_e0 - 0.2) or (grad_e2 < grad_e0 - 0.2)
    if closure_ok and wb_pass and reasonable and no_collapse and comp_not_worse:
        if grad_degraded:
            verdict = "CONDITIONAL PASS"
        else:
            verdict = "PASS"
        if cma_e1 >= cma_e2 - 0.02 and grad_e1 >= grad_e2 - 0.02:
            preferred = "interception_first"
        elif cma_e2 > cma_e1 + 0.05 and grad_e2 > grad_e1 + 0.05:
            preferred = "soil_et_first"
        else:
            preferred = "interception_first"
    else:
        verdict, preferred = "FAIL", "none"
    decision_rows = [
        {"component": "budget_closure_E1", "value": e1_exceed, "pass": e1_exceed < 1e-3},
        {"component": "budget_closure_E2", "value": e2_exceed, "pass": e2_exceed < 1e-3},
        {"component": "water_balance", "value": wb_pass, "pass": wb_pass},
        {"component": "CMA_median_E1_vs_E0", "value": e1_vs_e0, "pass": e1_vs_e0 > -0.25},
        {"component": "CMA_median_E2_vs_E0", "value": e2_vs_e0, "pass": e2_vs_e0 > -0.25},
        {"component": "Adam_median_E1_vs_E0", "value": grad_e1 - grad_e0, "pass": grad_e1 - grad_e0 > -0.2},
        {"component": "Adam_median_E2_vs_E0", "value": grad_e2 - grad_e0, "pass": grad_e2 - grad_e0 > -0.2},
        {"component": "compensation_E1_vs_E0", "value": comp_e1 - comp_e0, "pass": comp_e1 - comp_e0 <= 0.15},
        {"component": "compensation_E2_vs_E0", "value": comp_e2 - comp_e0, "pass": comp_e2 - comp_e0 <= 0.15},
        {"component": "PET_BUDGET_GATE", "value": verdict, "pass": verdict in ("PASS", "CONDITIONAL PASS")},
        {"component": "preferred_semantics", "value": preferred, "pass": True},
    ]
    write_csv("pet_budget_decision_matrix.csv", decision_rows)
    summary = {
        "boundary_pass": boundary_pass, "water_balance": wb_pass,
        "budget_closure_E1_max_exceed_fraction": e1_exceed,
        "budget_closure_E2_max_exceed_fraction": e2_exceed,
        "cma_median_KGE": {a: med[a]["cma_median"] for a in ARMS},
        "grad_median_KGE": {a: med[a]["grad_median"] for a in ARMS},
        "s_eff_boundary_hits": pressure, "compensation_top_quartile_corr": comp_summary,
        "gate": verdict, "preferred_semantics": preferred,
        "training_started": False, "shared_dpl_started": False, "production_default_changed": False,
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# MOPEX4 T1a PET-BUDGET FINAL GATE

Protocol: CPU; basins {', '.join(BASIN_IDS)}; 365-day warm-up + 365-day scored.
CMA-ES: starts {CMA_STARTS}, population {CMA_POPULATION}, generations {CMA_GENERATIONS}, stdev_init 0.25, seeds by arm {CMA_SEEDS}.
Adam: {GRAD_STEPS} steps, lr {GRAD_LR}, seeds {ADAM_SEEDS}; all common parameters plus S_eff jointly optimized.
F0 and T1a-E0 results reused from `pre_dpl_validation` under the identical protocol.

## Budget closure (max scored-period exceedance-day fraction)

- E0 legacy: `{e0_exceed:.6f}`
- E1 interception-first: `{e1_exceed:.6f}`
- E2 soil-ET-first: `{e2_exceed:.6f}`

## Median KGE

| arm | CMA-ES | Adam |
|---|---:|---:|
{chr(10).join(f"| {a} | {med[a]['cma_median']:.6f} | {med[a]['grad_median']:.6f} |" for a in ARMS)}

## S_eff boundary hits (out of 4)

- E0: `{pressure['T1a-E0']}`, E1: `{pressure['T1a-E1']}`, E2: `{pressure['T1a-E2']}`

## S_eff x Sb1 compensation (top-quartile abs correlation)

- F0: `{comp_summary.get('F0', float('nan')):.4f}`, E0: `{comp_e0:.4f}`, E1: `{comp_e1:.4f}`, E2: `{comp_e2:.4f}`

## Gate

**{verdict}** — preferred semantics: **{preferred}**

Water balance: **{'PASS' if wb_pass else 'FAIL'}**.
"""
    (OUT / "final_pet_budget_gate_report.md").write_text(report, encoding="utf-8")
    print("MOPEX4 T1a PET-BUDGET FINAL GATE")
    print("new forcing introduced: NO")
    print("PET role: shared daily evaporative-demand proxy")
    print("strict thermodynamic energy closure claimed: NO")
    print("T1a formula: I = S_eff * (-expm1(-Pr/S_eff)); S_eff bounds [1e-5,5] mm; c=1 fixed; Pr = liquid rainfall after snow partition")
    print("PET semantics before change: ET1/ET2 each see full PET independently; interception is an immediate same-day evaporative loss")
    print(f"Modes: E0 legacy; E1 interception-first; E2 soil-ET-first")
    print(f"Water balance: {'PASS' if wb_pass else 'FAIL'}")
    print(f"Budget closure E1: {'PASS' if e1_exceed < 1e-3 else 'FAIL'} ({e1_exceed:.6f}); E2: {'PASS' if e2_exceed < 1e-3 else 'FAIL'} ({e2_exceed:.6f})")
    print(f"CMA-ES median KGE: F0 {med['F0']['cma_median']:.6f}; T1a-E0 {med['T1a-E0']['cma_median']:.6f}; T1a-E1 {med['T1a-E1']['cma_median']:.6f}; T1a-E2 {med['T1a-E2']['cma_median']:.6f}")
    print(f"Adam median KGE: F0 {med['F0']['grad_median']:.6f}; T1a-E0 {med['T1a-E0']['grad_median']:.6f}; T1a-E1 {med['T1a-E1']['grad_median']:.6f}; T1a-E2 {med['T1a-E2']['grad_median']:.6f}")
    print(f"S_eff boundary pressure: E0 {pressure['T1a-E0']}/4; E1 {pressure['T1a-E1']}/4; E2 {pressure['T1a-E2']}/4")
    print(f"S_eff x Sb1 compensation: F0 {comp_summary.get('F0', float('nan')):.4f}; E0 {comp_e0:.4f}; E1 {comp_e1:.4f}; E2 {comp_e2:.4f}")
    print(f"PET exceedance-day fraction: E0 {e0_exceed:.6f}; E1 {e1_exceed:.6f}; E2 {e2_exceed:.6f}")
    print(f"Preferred PET semantics: {preferred}")
    print(f"PET-BUDGET GATE: {verdict}")
    print(f"Ready for 4-8 basin shared-dPL pilot: {'YES' if verdict in ('PASS', 'CONDITIONAL PASS') else 'NO'}")
    print("Ready for 531-basin training: NO")
    print("Production default changed: NO")


if __name__ == "__main__":
    main()
