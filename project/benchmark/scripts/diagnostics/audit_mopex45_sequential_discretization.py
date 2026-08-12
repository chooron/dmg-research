#!/usr/bin/env python3
"""MOPEX4/5 sequential-discretization mechanism audit.

Question: after MOPEX4/MOPEX5 added interception to the MOPEX3 soil bucket,
does the within-step sequential state update (ET1 -> interception -> q1f ->
qw each updating S1 before the next flux is computed) plus the hard state
cap (min(flux_i, S1)) explain the residual dPL-vs-IC gap?

This script is purely diagnostic.  It does not modify production MOPEX
steps, the IC path, continuation defaults, or any other model.  Variants
S1/S2 live in ``mopex45_discr_steps.py`` (benchmark-only).

Stages
------
  0  runtime flux graph + diag-vs-production equality check
  1  interception cap statistics + cap effect on alpha/is_time gradients
  2  same-timestep downstream coupling (dq1f/dalpha, dqw/dalpha, ...)
  5  fixed-parameter forward comparison (S0 vs S1 vs S2)
  6  lightweight direct-gradient comparison (G0/G1/G2)
  7  MOPEX5 mechanism sanity

Outputs land in
results/mopex45_phase_fix/sequential_discretization_audit/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BENCHMARK = Path(__file__).resolve().parents[2]
REPO = BENCHMARK.parents[1]
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src")]

from dmotpy.data_contract import add_calendar_forcing
from dmotpy.models.flux.mopex import mopex_training_context
from dmotpy.models.hydrology_model import HydrologyModel
from dmotpy.models.registry import PARAM_INFO
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from project.benchmark.scripts.run_dpl_benchmark_dmg_native import (
    load_camels_time_series,
    compute_differentiable_kge,
)
from project.benchmark.src.data_selection import load_ids
from project.benchmark.src.model_registry import model_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mopex45_discr_steps import (
    mopex4_step_diag,
    mopex4_step_samestate,
    mopex4_step_samestate_smoothcap,
    mopex5_step_diag,
    mopex5_step_samestate,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = BENCHMARK / "results" / "mopex45_phase_fix" / "sequential_discretization_audit"
WARMUP, SCORED = 365, 365
CMA_ROOT = REPO / "experiments/cmaes_36models/remote_runs/20260729_120525/checkpoints"
CONT_CKPT = (BENCHMARK / "results/mopex45_phase_fix/full_continuation/runs/seed_41"
             / "checkpoints/J2/seed_41/epoch_100.pt")

_COMPILED: dict = {}

def get_compiled(fn):
    """Cache torch.compile of a diagnostic step (12x speedup on the 8-basin
    eager loop; numerically identical, verified diff=0.0).  Passes through
    functions that are already compiled."""
    if hasattr(fn, "_torchdynamo_orig_callable"):
        return fn
    key = fn.__name__
    if key not in _COMPILED:
        _COMPILED[key] = torch.compile(fn, fullgraph=False)
    return _COMPILED[key]

M4_BOUNDS = list(PARAM_INFO["mopex4"].values())   # order = step param order
M5_BOUNDS = list(PARAM_INFO["mopex5"].values())


def csv_write(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    with (OUT / name).open("w", newline="") as handle:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def norm_to_phys(x: torch.Tensor, bounds: list) -> torch.Tensor:
    lo = torch.as_tensor([b[0] for b in bounds], dtype=x.dtype, device=x.device)
    hi = torch.as_tensor([b[1] for b in bounds], dtype=x.dtype, device=x.device)
    return lo + x * (hi - lo)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_data():
    ids = [int(v) for v in load_ids("data/531sub_id.txt")]
    tx, ty, _, _ = load_camels_time_series(ids)
    train_x = torch.as_tensor(tx, dtype=torch.float32, device=DEVICE)
    train_y = torch.as_tensor(ty, dtype=torch.float32, device=DEVICE)
    dates = pd.date_range("1980-10-01", "1995-09-30", freq="D")
    train_x, _ = add_calendar_forcing(train_x, dates, model_name="mopex4")
    return ids, train_x, train_y


def ic_theta(name: str, ids: list[int]) -> torch.Tensor:
    """Per-basin IC optimum (CMA-ES best_latent of the best start), normalized [0,1]."""
    ck = torch.load(CMA_ROOT / name / "chunk_0_gen_30.pt", map_location="cpu", weights_only=False)
    basin_ids = ck["basin_ids"]
    if isinstance(basin_ids, np.ndarray):
        basin_ids = list(map(int, basin_ids))
    else:
        basin_ids = list(basin_ids)
    assert basin_ids == ids, "basin order mismatch"
    best_latent = ck["solver"]["state"]["best_latent"].reshape(531, 5, -1)  # (B,S,D)
    best_fit = ck["solver"]["state"]["best_fitness"].reshape(531, 5)
    idx = best_fit.argmax(dim=1)                                  # best start per basin
    best = best_latent[torch.arange(531), idx]                    # latent space
    return torch.sigmoid(best).to(DEVICE)                         # normalized [0,1]


def continuation_theta(ids: list[int]) -> torch.Tensor:
    ck = torch.load(CONT_CKPT, map_location="cpu", weights_only=False)
    net = CatchmentParameterizer(35, 10, hidden_dims=[256, 256], dropout=0.05)
    net.load_state_dict(ck["network"])
    net.to(DEVICE).eval()
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
    with torch.no_grad():
        return net(attrs)                                          # normalized [0,1]


# ---------------------------------------------------------------------------
# rollout helpers (diagnostic steps, eager python loop, mopex4)
# ---------------------------------------------------------------------------

def _init_states(n_basin: int, device=None):
    d = device or DEVICE
    return (torch.full((n_basin,), 1e-6, device=d), torch.full((n_basin,), 1e-6, device=d),
            torch.full((n_basin,), 1e-6, device=d), torch.full((n_basin,), 1e-6, device=d),
            torch.full((n_basin,), 1e-6, device=d))


def rollout_m4(step_fn, x, y, theta_norm, basins, start, warmup=WARMUP, scored=SCORED,
               lambda_i=1.0, eps=None, ep_idx=None, collect=None, use_compiled=True):
    """Forward MOPEX4 over [start, start+warmup+scored) for the given basins.

    theta_norm: (n_basin, 10) normalized. If eps/ep_idx given, perturbs the
    parameter slice for finite differences (theta[..., ep_idx] += eps).
    Returns (Q, ET, states, flux_series) where flux_series is a list of per-t
    flux dicts over the scored period (warmup is detached).
    """
    b = list(basins)
    P = x[start:start + warmup + scored, b, 0]
    T = x[start:start + warmup + scored, b, 1]
    PET = x[start:start + warmup + scored, b, 2]
    doy = x[start:start + warmup + scored, b, 3]
    theta = norm_to_phys(theta_norm.clone(), M4_BOUNDS)
    if eps is not None:
        theta[..., ep_idx] = theta[..., ep_idx] + eps
    Sn, S1, S2, Sc1, Sc2 = _init_states(len(b))
    flux_series = []
    fn_eager = step_fn
    fn_scored = get_compiled(step_fn) if use_compiled else step_fn
    with mopex_training_context(lambda_i=lambda_i, lambda_p=1.0, beta=50.0):
        for t in range(warmup):
            with torch.no_grad():
                _, _, S1, S2, Sc1, Sc2, Sn, _ = fn_eager(
                    P[t], T[t], PET[t], *theta.t(), S1, S2, Sc1, Sc2, Sn,
                    doy=doy[t], nearzero=1e-6)
                S1, S2, Sc1, Sc2, Sn = S1.detach(), S2.detach(), Sc1.detach(), Sc2.detach(), Sn.detach()
        for t in range(warmup, warmup + scored):
            Q, ET, S1, S2, Sc1, Sc2, Sn, fx = fn_scored(
                P[t], T[t], PET[t], *theta.t(), S1, S2, Sc1, Sc2, Sn,
                doy=doy[t], nearzero=1e-6)
            flux_series.append({"Q": Q, "ET": ET, "flux": fx})
    return flux_series


def rollout_m5(step_fn, x, theta_norm, basins, start, warmup=WARMUP, scored=SCORED,
               lambda_i=1.0, eps=None, ep_idx=None):
    b = list(basins)
    P = x[start:start + warmup + scored, b, 0]
    T = x[start:start + warmup + scored, b, 1]
    PET = x[start:start + warmup + scored, b, 2]
    doy = x[start:start + warmup + scored, b, 3]
    theta = norm_to_phys(theta_norm.clone(), M5_BOUNDS)
    if eps is not None:
        theta[..., ep_idx] = theta[..., ep_idx] + eps
    Sn, S1, S2, Sc1, Sc2 = _init_states(len(b))
    flux_series = []
    fn_eager = step_fn
    fn_scored = get_compiled(step_fn)
    with mopex_training_context(lambda_i=lambda_i, lambda_p=1.0, beta=50.0):
        for t in range(warmup):
            with torch.no_grad():
                _, _, S1, S2, Sc1, Sc2, Sn, _ = fn_eager(
                    P[t], T[t], PET[t], *theta.t(), S1, S2, Sc1, Sc2, Sn,
                    doy=doy[t], nearzero=1e-6)
                S1, S2, Sc1, Sc2, Sn = S1.detach(), S2.detach(), Sc1.detach(), Sc2.detach(), Sn.detach()
        for t in range(warmup, warmup + scored):
            Q, ET, S1, S2, Sc1, Sc2, Sn, fx = fn_scored(
                P[t], T[t], PET[t], *theta.t(), S1, S2, Sc1, Sc2, Sn,
                doy=doy[t], nearzero=1e-6)
            flux_series.append({"Q": Q, "ET": ET, "flux": fx})
    return flux_series


def fd_series(step_fn, x, theta_norm, basins, start, ep_idx, eps=1e-3, warmup=WARMUP, scored=SCORED):
    """Central-difference per-timestep derivative of every recorded scalar w.r.t.
    normalized parameter ep_idx.  Returns dict name -> (scored,) tensor."""
    fn_c = get_compiled(step_fn)
    plus = rollout_m4(fn_c, x, None, theta_norm, basins, start, warmup, scored, eps=eps, ep_idx=ep_idx)
    minus = rollout_m4(step_fn, x, None, theta_norm, basins, start, warmup, scored, eps=-eps, ep_idx=ep_idx)
    keys = ["Q", "i", "i_raw", "q1f", "qw", "et1", "s1_before_i", "S1_new", "S2_new", "q2f", "q2u"]
    out = {}
    for k in keys:
        if k in plus[0] and k in minus[0]:
            out[k] = (torch.stack([f[k] for f in plus]) - torch.stack([f[k] for f in minus])) / (2 * eps)
        else:
            out[k] = torch.stack([f["flux"][k] for f in plus]) - torch.stack([f["flux"][k] for f in minus])
            out[k] = out[k] / (2 * eps)
    return out


# ---------------------------------------------------------------------------
# Stage 1 + 2: cap statistics and same-timestep coupling (MOPEX4, IC params)
# ---------------------------------------------------------------------------

def run_stage1_2(ids, x, theta_ic4, basins, start):
    rows_cap, rows_coup = [], []
    theta = theta_ic4[basins]
    n = len(basins)

    # reference series (unperturbed) for cap masks and rainy-day masks
    fn_c = get_compiled(mopex4_step_diag)
    base = rollout_m4(fn_c, x, None, theta, basins, start)

    P_series = torch.stack([x[start + WARMUP + t, b, 0] for b in basins for t in range(SCORED)]).reshape(n, SCORED).t()
    # P_series: (SCORED, n)

    # cap masks per basin-day
    i_raw = torch.stack([f["flux"]["i_raw"] for f in base])            # (SCORED, n)
    i = torch.stack([f["flux"]["i"] for f in base])
    s1b = torch.stack([f["flux"]["s1_before_i"] for f in base])
    cap_active = (i_raw >= s1b - 1e-9) & (i_raw > 1e-9)
    rainy = P_series > 0.1

    # finite-difference gradients (alpha=4, is_time=5)
    for name, ep in (("alpha", 4), ("is_time", 5)):
        d = fd_series(mopex4_step_diag, x, theta, basins, start, ep)
        dQ = d["Q"].abs(); di = d["i"].abs(); dq1f = d["q1f"].abs(); dqw = d["qw"].abs()
        for t in range(SCORED):
            for j in range(n):
                rows_cap.append({
                    "basin_idx": int(basins[j]), "day": t, "parameter": name,
                    "rainy": bool(rainy[t, j]), "i_raw": float(i_raw[t, j]),
                    "s1_before_i": float(s1b[t, j]), "cap_active": bool(cap_active[t, j]),
                    "i": float(i[t, j]),
                    "abs_di_dparam": float(di[t, j]), "abs_dQ_dparam": float(dQ[t, j]),
                })
                rows_coup.append({
                    "basin_idx": int(basins[j]), "day": t, "parameter": name,
                    "rainy": bool(rainy[t, j]), "cap_active": bool(cap_active[t, j]),
                    "abs_dq1f_dparam": float(dq1f[t, j]), "abs_dqw_dparam": float(dqw[t, j]),
                })

    # S1/S2 state derivatives w.r.t alpha (extra coupling rows)
    d_alpha = fd_series(mopex4_step_diag, x, theta, basins, start, 4)
    dS1 = d_alpha["S1_new"].abs(); dS2 = d_alpha["S2_new"].abs()
    for t in range(SCORED):
        for j in range(n):
            rows_coup.append({
                "basin_idx": int(basins[j]), "day": t, "parameter": "alpha",
                "rainy": bool(rainy[t, j]), "cap_active": bool(cap_active[t, j]),
                "abs_dS1next_dalpha": float(dS1[t, j]), "abs_dS2next_dalpha": float(dS2[t, j]),
            })

    # interception-off control (lambda_i=0 -> structurally MOPEX3-like soil)
    rows_ctrl = []
    base0 = rollout_m4(mopex4_step_diag, x, None, theta, basins, start, lambda_i=0.0, use_compiled=False)
    d0 = fd_series_lambda0(mopex4_step_diag, x, theta, basins, start, 4)
    for t in range(SCORED):
        for j in range(n):
            rows_ctrl.append({
                "basin_idx": int(basins[j]), "day": t,
                "lambda_i": 0.0, "abs_dq1f_dalpha": float(d0["q1f"][t, j].abs()),
                "abs_dqw_dalpha": float(d0["qw"][t, j].abs()),
                "abs_dQ_dalpha": float(d0["Q"][t, j].abs()),
            })
    csv_write("interception_cap_stats.csv", rows_cap)
    csv_write("same_timestep_coupling.csv", rows_coup)
    csv_write("interception_off_control.csv", rows_ctrl)

    # aggregate numbers for the report
    agg = {}
    for active in (True, False):
        m = cap_active & rainy
        sel = rows_cap
        sub = [r for r in sel if r["cap_active"] == active and r["rainy"]]
        agg[f"cap_{active}"] = {
            "di_alpha_mean": np.mean([r["abs_di_dparam"] for r in sub if r["parameter"] == "alpha"] or [0]),
            "di_itime_mean": np.mean([r["abs_di_dparam"] for r in sub if r["parameter"] == "is_time"] or [0]),
            "dQ_alpha_mean": np.mean([r["abs_dQ_dparam"] for r in sub if r["parameter"] == "alpha"] or [0]),
            "dQ_itime_mean": np.mean([r["abs_dQ_dparam"] for r in sub if r["parameter"] == "is_time"] or [0]),
        }
    frac = float(cap_active.float().mean())
    frac_rainy = float((cap_active & rainy).float().sum() / rainy.float().sum().clamp_min(1))
    return {
        "cap_active_fraction": frac,
        "cap_active_fraction_on_rainy": frac_rainy,
        "rainy_fraction": float(rainy.float().mean()),
        "n_basin_days": float(rainy.numel()),
        "agg": agg,
    }


def fd_series_lambda0(step_fn, x, theta_norm, basins, start, ep_idx, eps=1e-3, warmup=WARMUP, scored=SCORED):
    plus = rollout_m4(step_fn, x, None, theta_norm, basins, start, warmup, scored, lambda_i=0.0, eps=eps, ep_idx=ep_idx, use_compiled=False)
    minus = rollout_m4(step_fn, x, None, theta_norm, basins, start, warmup, scored, lambda_i=0.0, eps=-eps, ep_idx=ep_idx, use_compiled=False)
    out = {}
    for k in ("Q", "q1f", "qw"):
        out[k] = (torch.stack([f[k] if k == "Q" else f["flux"][k] for f in plus])
                  - torch.stack([f[k] if k == "Q" else f["flux"][k] for f in minus])) / (2 * eps)
    return out


# ---------------------------------------------------------------------------
# Stage 5: fixed-parameter forward comparison (S0 vs S1 vs S2)
# ---------------------------------------------------------------------------

def run_stage5(ids, x, theta_ic4, theta_cont4, basins, start):
    rows = []
    param_sets = [("IC", theta_ic4[basins]), ("continuation", theta_cont4[basins])]
    variants = [("S0_sequential", mopex4_step_diag), ("S1_samestate", mopex4_step_samestate),
                ("S2_smoothcap", mopex4_step_samestate_smoothcap)]
    for pname, theta in param_sets:
        series = {}
        for vname, fn in variants:
            series[vname] = rollout_m4(get_compiled(fn), x, None, theta, basins, start)
        for j, b in enumerate(basins):
            for vname in ("S1_samestate", "S2_smoothcap"):
                q0 = torch.stack([f["Q"][j] for f in series["S0_sequential"]])
                qv = torch.stack([f["Q"][j] for f in series[vname]])
                e0 = torch.stack([f["ET"][j] for f in series["S0_sequential"]])
                ev = torch.stack([f["ET"][j] for f in series[vname]])
                s1_0 = torch.stack([f["flux"]["S1_new"][j] for f in series["S0_sequential"]])
                s1_v = torch.stack([f["flux"]["S1_new"][j] for f in series[vname]])
                s2_0 = torch.stack([f["flux"]["S2_new"][j] for f in series["S0_sequential"]])
                s2_v = torch.stack([f["flux"]["S2_new"][j] for f in series[vname]])
                q1f_0 = torch.stack([f["flux"]["q1f"][j] for f in series["S0_sequential"]])
                q1f_v = torch.stack([f["flux"]["q1f"][j] for f in series[vname]])
                qw_0 = torch.stack([f["flux"]["qw"][j] for f in series["S0_sequential"]])
                qw_v = torch.stack([f["flux"]["qw"][j] for f in series[vname]])
                i_0 = torch.stack([f["flux"]["i"][j] for f in series["S0_sequential"]])
                i_v = torch.stack([f["flux"]["i"][j] for f in series[vname]])

                def stats(a, bv):
                    d = a - bv
                    denom = (a.square().mean().sqrt() + 1e-6)
                    return {"rmse": float(d.square().mean().sqrt()), "mae": float(d.abs().mean()),
                            "corr": float(torch.corrcoef(torch.stack([a, bv]))[0, 1]),
                            "vol_diff": float((a.sum() - bv.sum()).abs()),
                            "max_abs": float(d.abs().max())}

                for qname, (a, bv) in (("Q", (q0, qv)), ("ET", (e0, ev)), ("S1", (s1_0, s1_v)),
                                       ("S2", (s2_0, s2_v)), ("q1f", (q1f_0, q1f_v)),
                                       ("qw", (qw_0, qw_v)), ("interception", (i_0, i_v))):
                    s = stats(a, bv)
                    rows.append({"param_set": pname, "basin_idx": int(b), "variant": vname,
                                 "quantity": qname, **s})
    csv_write("fixed_parameter_forward_comparison.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# Stage 6: lightweight direct-gradient comparison (G0/G1/G2)
# ---------------------------------------------------------------------------

def run_stage6(ids, x, y, theta_ic4, basins, start, steps=100, seeds=(1, 2, 3), lr=1e-2):
    rows = []
    b = list(basins)
    init_theta = theta_ic4[basins].clone()  # start from IC (normalized)
    variants = [("G0_sequential", mopex4_step_diag), ("G1_samestate", mopex4_step_samestate),
                ("G2_smoothcap", mopex4_step_samestate_smoothcap)]
    y_win = torch.stack([y[start + WARMUP + t, j] for j in b for t in range(SCORED)]).reshape(len(b), SCORED).t()

    for vname, fn in variants:
        fn_c = get_compiled(fn)
        for seed in seeds:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            theta = nn.Parameter(init_theta.detach().clone())
            opt = torch.optim.AdamW([theta], lr=lr)
            losses, kges = [], []
            for step in range(steps):
                opt.zero_grad(set_to_none=True)
                fx = rollout_m4(fn_c, x, None, theta, b, start)
                Q = torch.stack([f["Q"] for f in fx])                     # (SCORED, n)
                loss, _ = compute_differentiable_kge(Q, y_win, warmup_days=0)
                loss.backward()
                if not torch.isfinite(loss):
                    losses.append(float("nan")); break
                g = theta.grad.detach()
                zero_frac = float((g.abs() < 1e-12).float().mean())
                gn = float(g.norm())
                nn.utils.clip_grad_norm_([theta], max_norm=1.0)
                opt.step()
                with torch.no_grad():
                    theta.clamp_(0.0, 1.0)
                losses.append(float(loss.detach()))
                if step % 20 == 0 or step == steps - 1:
                    with torch.no_grad():
                        _, kge = compute_differentiable_kge(Q.detach(), y_win, warmup_days=0)
                    kges.append((step, float(kge.median()), float(kge.mean())))
                if step == 0 or step == steps - 1:
                    rows.append({"variant": vname, "seed": seed, "step": step,
                                 "loss": float(loss.detach()),
                                 "direct_kge_median": kges[-1][1] if kges else None,
                                 "direct_kge_mean": kges[-1][2] if kges else None,
                                 "grad_zero_fraction": zero_frac, "grad_norm": gn})
            # final summary row per variant-seed
            rows.append({"variant": vname, "seed": seed, "step": "final",
                         "loss": losses[-1] if losses else None,
                         "direct_kge_median": kges[-1][1] if kges else None,
                         "direct_kge_mean": kges[-1][2] if kges else None,
                         "grad_zero_fraction": zero_frac, "grad_norm": gn,
                         "loss_first": losses[0] if losses else None,
                         "loss_reduction": (losses[0] - losses[-1]) if losses and len(losses) > 1 else None})
    csv_write("direct_gradient_sequential_vs_same_state.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# Stage 7: MOPEX5 mechanism sanity
# ---------------------------------------------------------------------------

def run_stage7(ids, x, theta_ic5, basins, start):
    rows_cap, rows_coup = [], []
    theta = theta_ic5[basins]
    n = len(basins)
    fn_c = get_compiled(mopex5_step_diag)
    base = rollout_m5(fn_c, x, theta, basins, start)
    P_series = torch.stack([x[start + WARMUP + t, b, 0] for b in basins for t in range(SCORED)]).reshape(n, SCORED).t()
    i_raw = torch.stack([f["flux"]["i_raw"] for f in base])
    s1b = torch.stack([f["flux"]["s1_before_i"] for f in base])
    cap_active = (i_raw >= s1b - 1e-9) & (i_raw > 1e-9)
    rainy = P_series > 0.1

    for name, ep in (("alpha", 4), ("is_time", 5)):
        plus = rollout_m5(fn_c, x, theta, basins, start, eps=1e-3, ep_idx=ep)
        minus = rollout_m5(fn_c, x, theta, basins, start, eps=-1e-3, ep_idx=ep)
        dQ = (torch.stack([f["Q"] for f in plus]) - torch.stack([f["Q"] for f in minus])) / 2e-3
        di = (torch.stack([f["flux"]["i"] for f in plus]) - torch.stack([f["flux"]["i"] for f in minus])) / 2e-3
        dq1f = (torch.stack([f["flux"]["q1f"] for f in plus]) - torch.stack([f["flux"]["q1f"] for f in minus])) / 2e-3
        dqw = (torch.stack([f["flux"]["qw"] for f in plus]) - torch.stack([f["flux"]["qw"] for f in minus])) / 2e-3
        for t in range(SCORED):
            for j in range(n):
                rows_cap.append({"model": "mopex5", "basin_idx": int(basins[j]), "day": t,
                                 "parameter": name, "rainy": bool(rainy[t, j]),
                                 "cap_active": bool(cap_active[t, j]),
                                 "abs_di_dparam": float(di[t, j].abs()),
                                 "abs_dQ_dparam": float(dQ[t, j].abs())})
                rows_coup.append({"model": "mopex5", "basin_idx": int(basins[j]), "day": t,
                                  "parameter": name, "rainy": bool(rainy[t, j]),
                                  "cap_active": bool(cap_active[t, j]),
                                  "abs_dq1f_dparam": float(dq1f[t, j].abs()),
                                  "abs_dqw_dparam": float(dqw[t, j].abs())})
    csv_write("mopex5_interception_cap_stats.csv", rows_cap)
    csv_write("mopex5_same_timestep_coupling.csv", rows_coup)
    return {
        "cap_active_fraction": float(cap_active.float().mean()),
        "cap_active_fraction_on_rainy": float((cap_active & rainy).float().sum() / rainy.float().sum().clamp_min(1)),
        "rainy_fraction": float(rainy.float().mean()),
    }


# ---------------------------------------------------------------------------
# Stage 0: equality check + runtime flux graph md
# ---------------------------------------------------------------------------

def run_stage0(ids, x, theta_ic4, basins, start):
    from dmotpy.models.core.mopex4 import mopex4_step as prod4
    from dmotpy.models.core.mopex5 import mopex5_step as prod5
    # equality: diag sequential == production for MOPEX4
    b = list(basins[:2])
    theta = norm_to_phys(theta_ic4[b].clone(), M4_BOUNDS)
    Sn, S1, S2, Sc1, Sc2 = _init_states(len(b))
    Sn2, S12, S22, Sc12, Sc22 = _init_states(len(b))
    maxdiff = 0.0
    with mopex_training_context(lambda_i=1.0, lambda_p=1.0, beta=50.0):
        for t in range(start, start + WARMUP + SCORED, 7):  # sample days
            P = x[t, b, 0]; T = x[t, b, 1]; PET = x[t, b, 2]; doy = x[t, b, 3]
            q1, e1, S1, S2, Sc1, Sc2, Sn, _ = mopex4_step_diag(P, T, PET, *theta.t(), S1, S2, Sc1, Sc2, Sn, doy=doy, nearzero=1e-6)
            q2, e2, S12, S22, Sc12, Sc22, Sn2 = prod4(P, T, PET, *theta.t(), S12, S22, Sc12, Sc22, Sn2, doy=doy, nearzero=1e-6)
            maxdiff = max(maxdiff, float((q1 - q2).abs().max()), float((e1 - e2).abs().max()))
    return {"diag_sequential_vs_production_max_abs_diff": maxdiff}


FLUX_GRAPH_MD = """# MOPEX3 -> MOPEX4 -> MOPEX5 runtime flux graph (soil bucket)

Derived from `dmotpy/models/core/mopex{3,4,5}.py` (runtime code, not comments)
and `dmotpy/models/flux/mopex.py`.  All fluxes below are computed sequentially:
each flux updates the storage before the next flux is evaluated.

## Soil bucket ordering per timestep (S1 = soil, MATLAB S2)

| flux | MOPEX3 | MOPEX4 | MOPEX5 | input state | cap |
|---|---|---|---|---|---|
| add inputs | S1 += pr + qn | S1 += pr + qn | S1 += pr + qn | - | - |
| ET1 | evap_7(S1) | evap_7(S1) | evap_7(S1, PET_epc) | post-input S1 | min(et1, S1) |
| interception | **absent** | interception_4(pr, doy, alpha, is_time) | same | **pr, doy, alpha, is_time (not S1)** | min(i, S1) |
| q1f | saturation_1(pr+qn, S1) | saturation_1(pr+qn, S1) | same | S1 **after ET1 and interception** | min(q1f, S1) |
| qw | recharge_3(tw, S1) | recharge_3(tw, S1) | same | S1 after ET1, interception, q1f | min(tw*S1, S1) |

MOPEX4/5 vs MOPEX3 difference: interception is inserted between ET1 and q1f,
so q1f and qw see a soil storage that has already been reduced by the same-day
interception.  This is the same-timestep coupling under audit.

## Subsurface bucket (S2, MATLAB S3)

| flux | input state | cap |
|---|---|---|
| add qw | S2 += qw | - |
| q2f | saturation_1(qw, S2) | min(q2f, S2) |
| q2u | baseflow_1(tu, S2) | min(tu*S2, S2) |
| ET2 | evap_7(S2, se*s3max, PET[,_epc]) | min(et2, S2) |

## Routing

Sc1 += q1f + q2f; qf = min(tc*Sc1, Sc1); Sc2 += q2u; qs = min(tc*Sc2, Sc2);
Q = qf + qs; ET = et1 + et2 + i (MOPEX4/5 includes interception in ET).

## Continuation context

`mopex_interception_4` multiplies the physical interception by lambda_i and
`mopex_phenology_1` mixes GSI with identity PET via lambda_p; defaults
(lambda_i=1, lambda_p=1, beta=50) reproduce the production equations exactly.
"""


# ---------------------------------------------------------------------------

def select_representative_basins(ids, x, y, theta_ic4, theta_cont4, n_each=4, start=0):
    """Rank basins by continuation-vs-IC gap on a quick 365+365 evaluation and
    pick high/median/low gap basins."""
    b_all = list(range(531))
    window = 365
    def kge_for(theta):
        out = []
        for i in range(0, 531, 100):
            bi = b_all[i:i + 100]
            fx = rollout_m4(mopex4_step_diag, x, None, theta[bi], bi, start, warmup=WARMUP, scored=window)
            Q = torch.stack([f["Q"] for f in fx])
            yw = torch.stack([y[start + WARMUP + t, j] for j in bi for t in range(window)]).reshape(len(bi), window).t()
            with torch.no_grad():
                _, kge = compute_differentiable_kge(Q, yw, warmup_days=0)
            out.append(kge)
        return torch.cat(out)
    with torch.no_grad():
        k_ic = kge_for(theta_ic4)
        k_ct = kge_for(theta_cont4)
    gap = k_ic - k_ct
    order = torch.argsort(gap, descending=True)
    picks = []
    for label, idxs in (("high_gap", order[:n_each]), ("median_gap", order[len(order)//2 - n_each//2: len(order)//2 + n_each//2]),
                        ("low_gap", order[-n_each:])):
        for i in idxs:
            picks.append({"basin_idx": int(i), "basin_id": int(ids[i]), "group": label,
                          "ic_kge": float(k_ic[i]), "cont_kge": float(k_ct[i]), "gap": float(gap[i])})
    return picks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="all")
    parser.add_argument("--basins", type=int, default=8, help="representative basins per group")
    parser.add_argument("--start-day", type=int, default=1825, help="window start in train forcing")
    parser.add_argument("--steps", type=int, default=100, help="direct-gradient steps")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    assert torch.cuda.is_available(), "CUDA required"

    ids, x, y = load_data()
    theta_ic4 = ic_theta("mopex4", ids)
    theta_ic5 = ic_theta("mopex5", ids)
    theta_cont4 = continuation_theta(ids)

    # representative basins
    picks = select_representative_basins(ids, x, y, theta_ic4, theta_cont4, n_each=args.basins // 3 + 1)
    csv_write("representative_basins.csv", picks)
    basins = [p["basin_idx"] for p in picks]
    print(f"representative basins: {basins}", flush=True)

    results = {"basins": basins, "start_day": args.start_day}

    if args.stage in ("0", "all"):
        eq = run_stage0(ids, x, theta_ic4, basins[:3], args.start_day)
        results["stage0"] = eq
        (OUT / "runtime_flux_graph.md").write_text(FLUX_GRAPH_MD +
            f"\n## Stage 0 equality check\ndiag-sequential vs production mopex4_step "
            f"max |diff| = {eq['diag_sequential_vs_production_max_abs_diff']:.3e}\n")
        print("stage0 done", eq, flush=True)

    if args.stage in ("1", "all"):
        s12 = run_stage1_2(ids, x, theta_ic4, basins, args.start_day)
        results["stage1_2"] = s12
        print("stage1/2 done:", {k: v for k, v in s12.items() if k != "agg"}, flush=True)

    if args.stage in ("5", "all"):
        rows5 = run_stage5(ids, x, theta_ic4, theta_cont4, basins, args.start_day)
        print(f"stage5 done ({len(rows5)} rows)", flush=True)

    if args.stage in ("6", "all"):
        rows6 = run_stage6(ids, x, y, theta_ic4, basins, args.start_day, steps=args.steps)
        print("stage6 done", flush=True)

    if args.stage in ("7", "all"):
        s7 = run_stage7(ids, x, theta_ic5, basins, args.start_day)
        results["stage7"] = s7
        print("stage7 done:", s7, flush=True)

    (OUT / "audit_meta.json").write_text(json.dumps({k: (v if not isinstance(v, dict) else
        {kk: (vv if not isinstance(vv, dict) else str(vv)) for kk, vv in v.items()})
        for k, v in results.items()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
