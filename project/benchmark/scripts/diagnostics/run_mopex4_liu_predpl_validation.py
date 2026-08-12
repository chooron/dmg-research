#!/usr/bin/env python3
"""Final small matched pre-dPL decision validation for MOPEX4 interception.

This runner uses the existing BatchedCMAES implementation for derivative-free
calibration and direct Adam only for the requested gradient-accessibility
comparison. It calibrates all MOPEX4 parameters per basin, uses CPU and the
fixed 365-day warm-up + 365-day scored window, and never starts a network or a
531-basin run.
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
from dmotpy.models.flux.mopex import (
    mopex_baseflow_1,
    mopex_evap_7,
    mopex_interception_4,
    mopex_interception_4_liu,
    mopex_melt_1,
    mopex_rainfall_1,
    mopex_recharge_3,
    mopex_saturation_1,
    mopex_snowfall_1,
    mopex_training_context,
)
from dmotpy.models.core.mopex4 import MOPEX4_PARAMS_BOUNDS
from dmotpy.models.core.mopex5 import MOPEX5_PARAMS_BOUNDS
from dmotpy.models.core.mopex4 import MOPEX4_LEGACY_INTERCEPTION_NAMES
from dmotpy.models.flux.mopex import _mopex_interception_4_legacy
from objective import full_kge_reference

OUT = BENCHMARK / "results" / "mopex45_phase_fix" / "root_cause_audit" / "liu_interception" / "pre_dpl_validation"
OUT.mkdir(parents=True, exist_ok=True)
DTYPE = torch.float64
BASIN_IDS = ["8202700", "8150800", "5507600", "11532500"]
WARMUP = SCORED = 365
START = A.START
ARMS = ["F0", "T1a", "T1"]
SEEDS = [7, 41, 73]
CMA_SEEDS = {"F0": 7, "T1a": 1007, "T1": 2007}
CMA_STARTS = 3
CMA_POPULATION = 10
CMA_GENERATIONS = 40
GRAD_STEPS = 120
GRAD_LR = 0.04

# Preserve the old F0 parameter bounds exactly as they existed before the Liu
# semantic replacement. The common eight slots are unchanged.
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
    return len(F0_BOUNDS) if arm in {"F0", "T1"} else len(T1A_BOUNDS)


def arm_bounds(arm: str) -> list[list[float]]:
    return F0_BOUNDS if arm == "F0" else T1A_BOUNDS if arm == "T1a" else T1_BOUNDS


def latent_to_physical(latent: torch.Tensor, arm: str) -> torch.Tensor:
    """Return full ten physical slots with the arm's fixed slots inserted."""
    bounds = torch.tensor(arm_bounds(arm), dtype=latent.dtype, device=latent.device)
    normalized = torch.sigmoid(latent)
    values = bounds[:, 0] + normalized * (bounds[:, 1] - bounds[:, 0])
    if arm == "T1a":
        full = torch.zeros((*latent.shape[:-1], 10), dtype=latent.dtype, device=latent.device)
        full[..., T1A_ACTIVE] = values
        full[..., 5] = 1.0
        return full
    return values


def _kge_and_nse(pred: torch.Tensor, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the repository KGE reduction and a matching descriptive NSE."""
    # pred: [T,B,S,P], obs: [T,B]
    score = full_kge_reference(pred, obs, eps=0.1)
    mask = torch.isfinite(pred[..., 0]) & torch.isfinite(obs[:, :, None])
    obs_finite = torch.isfinite(obs)
    valid_obs = torch.where(obs_finite, obs, torch.zeros_like(obs))
    mean_obs = valid_obs.sum(dim=0) / obs_finite.to(pred.dtype).sum(dim=0).clamp_min(1.0)
    centered = torch.where(mask, pred[..., 0] - mean_obs[None, :, None], torch.zeros_like(pred[..., 0]))
    residual = torch.where(mask, pred[..., 0] - obs[:, :, None], torch.zeros_like(pred[..., 0]))
    nse = 1.0 - residual.square().sum(dim=0) / centered.square().sum(dim=0).clamp_min(1.0e-12)
    return score, nse.squeeze(-1)


def simulate(latent: torch.Tensor, arm: str, forcing: torch.Tensor, *, collect: bool = False):
    """Vectorized sequential MOPEX4 simulation for [B,S,P,D] latent values."""
    if latent.ndim == 2:
        latent = latent[:, None, None, :]
    B, starts, population, _ = latent.shape
    groups = starts * population
    physical = latent_to_physical(latent, arm).reshape(B, groups, 10)
    P0 = forcing[:, :, 0].to(latent.dtype)
    T0 = forcing[:, :, 1].to(latent.dtype)
    PET0 = forcing[:, :, 2].to(latent.dtype)
    doy = forcing[:, :, 3].to(latent.dtype)
    states = [torch.full((B, groups), 1.0e-6, dtype=latent.dtype, device=latent.device) for _ in range(5)]
    q_rows, et_rows = [], []
    diag = {key: [] for key in ["i", "et1", "et2", "pr", "pet", "state_sum"]} if collect else None
    with mopex_training_context(lambda_i=1.0, lambda_p=1.0, beta=50.0):
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
            et1 = torch.minimum(mopex_evap_7(soil, sb1, PET, 1.0, 1e-6), soil)
            soil = soil - et1
            if arm == "F0":
                i_raw = _mopex_interception_4_legacy(pr, DOY, int0, int1)
            else:
                i_raw = mopex_interception_4_liu(pr, int0, int1)
            interception = torch.minimum(i_raw, soil)
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
            et2 = torch.minimum(mopex_evap_7(sub, se * sb2, PET, 1.0, 1e-6), sub)
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
                     "seeds": SEEDS, "initialization": "matched zero latent center plus seeded 0.5 latent perturbation",
                     "stopping_rule": "fixed step budget"},
        "training_started": False,
        "shared_dpl_started": False,
    }


def run_cma(forcing: torch.Tensor, observations: torch.Tensor):
    rows, restart_rows, best_by_arm = [], [], {}
    for arm in ARMS:
        dim = arm_dimension(arm)
        seed = CMA_SEEDS[arm]
        solver = BatchedCMAES(4 * CMA_STARTS, dim, CMA_POPULATION, stdev_init=0.25,
                              active=True, seed=seed, device="cpu")
        solver.set_centers(torch.zeros((4 * CMA_STARTS, dim), dtype=DTYPE))
        history = []
        for generation in range(CMA_GENERATIONS):
            z, y, x = solver.ask()
            latent = x.reshape(4, CMA_STARTS, CMA_POPULATION, dim)
            score = cma_evaluate(latent, arm, forcing, observations)
            solver.tell(z, y, x, score.reshape(4 * CMA_STARTS, CMA_POPULATION))
            history.append(solver.state.best_fitness.reshape(4, CMA_STARTS).detach().clone())
        best_latent = solver.state.best_latent.reshape(4, CMA_STARTS, dim).detach().clone()
        final_latent = best_latent[:, :, None, :]
        q, _, _ = simulate(final_latent, arm, forcing)
        score, nse = _kge_and_nse(q[WARMUP:], observations[WARMUP:])
        physical = latent_to_physical(final_latent, arm).squeeze(2)
        per_basin = []
        for b in range(4):
            for restart in range(CMA_STARTS):
                row = {"method": "CMA-ES", "arm": arm, "basin_id": BASIN_IDS[b], "restart": restart,
                       "KGE": score[b, restart, 0].item(), "NSE": nse[b, restart].item(),
                       "objective": score[b, restart, 0].item(), "generation": CMA_GENERATIONS}
                for i, value in enumerate(physical[b, restart]):
                    row[f"p{i}"] = value.item()
                rows.append(row)
                restart_rows.append({"arm": arm, "basin_id": BASIN_IDS[b], "restart": restart,
                                     "final_KGE": score[b, restart, 0].item(),
                                     "best_fitness": solver.state.best_fitness.reshape(4, CMA_STARTS)[b, restart].item(),
                                     "history_last_10_mean": torch.stack(history)[-min(10, CMA_GENERATIONS):, b, restart].mean().item()})
                per_basin.append((b, restart, score[b, restart, 0].item(), best_latent[b, restart]))
        best_by_arm[arm] = per_basin
    write_csv("cmaes_all_parameter_results.csv", rows)
    write_csv("cmaes_restart_distribution.csv", restart_rows)
    return best_by_arm


def run_gradient(forcing: torch.Tensor, observations: torch.Tensor):
    curve_rows, result_rows, best_by_arm = [], [], {}
    for arm in ARMS:
        dim = arm_dimension(arm)
        arm_results = []
        for seed in SEEDS:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            raw = (torch.randn((4, dim), generator=generator, dtype=DTYPE) * 0.5).requires_grad_(True)
            optimizer = torch.optim.Adam([raw], lr=GRAD_LR)
            best_score = torch.full((4,), -torch.inf, dtype=DTYPE)
            best_raw = raw.detach().clone()
            for step in range(GRAD_STEPS):
                optimizer.zero_grad()
                q, _, _ = simulate(raw[:, None, None, :], arm, forcing)
                scores, nse = _kge_and_nse(q[WARMUP:], observations[WARMUP:])
                basin_scores = scores[:, 0, 0]
                loss = 1.0 - basin_scores.mean()
                loss.backward()
                grad_norm = float(raw.grad.detach().norm()) if raw.grad is not None else float("nan")
                finite_grad = bool(raw.grad is not None and torch.isfinite(raw.grad).all() and torch.isfinite(loss))
                optimizer.step()
                improved = basin_scores.detach() > best_score
                best_score[improved] = basin_scores.detach()[improved]
                best_raw[improved] = raw.detach()[improved]
                for b in range(4):
                    curve_rows.append({"method": "Adam", "arm": arm, "seed": seed, "step": step,
                                       "basin_id": BASIN_IDS[b], "loss": loss.item(),
                                       "KGE": basin_scores[b].item(), "NSE": nse[b].item(),
                                       "grad_norm": grad_norm, "finite_grad": finite_grad})
            q, _, _ = simulate(best_raw[:, None, None, :], arm, forcing)
            scores, nse = _kge_and_nse(q[WARMUP:], observations[WARMUP:])
            physical = latent_to_physical(best_raw[:, None, None, :], arm).squeeze(1).squeeze(1)
            for b in range(4):
                row = {"method": "Adam", "arm": arm, "seed": seed, "basin_id": BASIN_IDS[b],
                       "KGE": scores[b, 0, 0].item(), "NSE": nse[b].item(), "objective": scores[b, 0, 0].item(),
                       "steps": GRAD_STEPS}
                for i, value in enumerate(physical[b]):
                    row[f"p{i}"] = value.item()
                for j, raw_value in enumerate(best_raw[b]):
                    row[f"raw{j}"] = raw_value.item()
                result_rows.append(row)
                arm_results.append((b, seed, scores[b, 0, 0].item(), best_raw[b].detach().clone()))
        best_by_arm[arm] = arm_results
    write_csv("gradient_all_parameter_results.csv", result_rows)
    write_csv("gradient_training_curves.csv", curve_rows)
    return best_by_arm


def choose_best(cma_results, grad_results):
    chosen = {}
    for arm in ARMS:
        chosen[arm] = {}
        for b in range(4):
            candidates = [(score, latent, "CMA-ES", restart) for bb, restart, score, latent in cma_results[arm] if bb == b]
            candidates += [(score, latent, "Adam", seed) for bb, seed, score, latent in grad_results[arm] if bb == b]
            chosen[arm][b] = max(candidates, key=lambda item: item[0])
    return chosen


def parameter_rows(chosen):
    rows, boundary_rows = [], []
    common_names = ["tcrit", "ddf", "s2max", "tw", "interception_0", "interception_1", "tu", "se", "s3max", "tc"]
    for arm in ARMS:
        names = common_names.copy()
        names[4:6] = ["alpha", "is_time"] if arm == "F0" else ["S_eff", "c"]
        for b in range(4):
            score, latent, method, run_id = chosen[arm][b]
            physical = latent_to_physical(latent[None, None, None, :], arm).flatten()
            row = {"arm": arm, "basin_id": BASIN_IDS[b], "selection_method": method, "selection_id": run_id, "KGE": score}
            for i, value in enumerate(physical):
                row[f"{names[i]}"] = value.item()
            rows.append(row)
            bounds = F0_BOUNDS if arm == "F0" else T1A_BOUNDS if arm == "T1a" else T1_BOUNDS
            latent_values = latent.detach().tolist()
            for i, value in enumerate(physical):
                if arm == "T1a" and i == 5:
                    lo, hi, raw = 1.0, 1.0, float("nan")
                elif arm == "T1a":
                    j = T1A_ACTIVE.index(i); lo, hi, raw = bounds[j][0], bounds[j][1], latent_values[j]
                else:
                    lo, hi, raw = bounds[i][0], bounds[i][1], latent_values[i]
                span = hi - lo
                distance = min((value.item() - lo) / max(span, 1e-12), (hi - value.item()) / max(span, 1e-12)) if span else 0.0
                boundary_rows.append({"arm": arm, "basin_id": BASIN_IDS[b], "method": method, "parameter": names[i],
                                      "physical_value": value.item(), "lower": lo, "upper": hi,
                                      "normalized_boundary_distance": distance, "boundary_hit": distance <= .02,
                                      "raw_logit": raw,
                                      "sigmoid_derivative": float(torch.sigmoid(torch.tensor(raw)) * (1 - torch.sigmoid(torch.tensor(raw)))) if math.isfinite(raw) else float("nan")})
    write_csv("optimized_parameter_summary.csv", rows)
    write_csv("parameter_boundary_audit.csv", boundary_rows)
    return rows


def identifiability_at_optimum(chosen, forcing):
    rows = []
    pr_all = mopex_rainfall_1(forcing[:, :, 0], forcing[:, :, 1], torch.tensor(0.0, dtype=DTYPE)).clamp_min(0)
    for arm in ["T1a", "T1"]:
        for b in range(4):
            score, latent, method, run_id = chosen[arm][b]
            physical = latent_to_physical(latent[None, None, None, :], arm).flatten()
            s, c = physical[4].item(), physical[5].item()
            pr = pr_all[:, b]
            x = c * pr / s
            rainy = pr > 1e-10
            ddc = pr * torch.exp(-x)
            dds = 1.0 - torch.exp(-x) * (1.0 + x)
            mass = pr.sum().item()
            rows.append({"arm": arm, "basin_id": BASIN_IDS[b], "selection_method": method, "selection_id": run_id,
                         "S_eff": s, "c": c, "P_star": s / c,
                         "median_rainy_abs_dI_dc": ddc[rainy].abs().median().item() if bool(rainy.any()) else 0.0,
                         "median_rainy_abs_dI_dS_eff": dds[rainy].abs().median().item() if bool(rainy.any()) else 0.0,
                         "mass_fraction_x_lt_1": (pr * (x < 1)).sum().item() / max(mass, 1e-12),
                         "mass_fraction_x_1_to_3": (pr * ((x >= 1) & (x <= 3))).sum().item() / max(mass, 1e-12),
                         "mass_fraction_x_gt_3": (pr * (x > 3)).sum().item() / max(mass, 1e-12),
                         "c_boundary_distance": min((c - .1) / .88, (.98 - c) / .88) if arm == "T1" else None})
    write_csv("c_identifiability_at_optimum.csv", rows)
    return rows


def compensation_surface(chosen, forcing, observations):
    rows = []
    for arm in ARMS:
        for b in range(4):
            score, latent, method, run_id = chosen[arm][b]
            base = latent_to_physical(latent[None, None, None, :], arm).flatten()
            if arm == "F0":
                values = [("alpha", a, phase) for a in torch.linspace(.05, .95, 7) for phase in [base[5].item()]]
                sb_grid = torch.linspace(max(10.0, base[2].item() * .7), min(2000.0, base[2].item() * 1.3), 7)
            else:
                values = [("S_eff", s.item(), base[5].item()) for s in torch.linspace(.1, 4.9, 7)]
                sb_grid = torch.linspace(max(10.0, base[2].item() * .7), min(2000.0, base[2].item() * 1.3), 7)
            for key, value, second in values:
                for sb1 in sb_grid:
                    candidate = base.clone()
                    candidate[2] = sb1
                    if arm == "F0": candidate[4] = value
                    else: candidate[4] = value
                    raw = []
                    bounds = F0_BOUNDS if arm == "F0" else T1A_BOUNDS if arm == "T1a" else T1_BOUNDS
                    if arm == "T1a":
                        for i in T1A_ACTIVE:
                            lo, hi = T1_BOUNDS[i]
                            raw.append(float(torch.logit(((candidate[i] - lo) / (hi - lo)).clamp(1e-7, 1 - 1e-7))))
                    else:
                        for i, (lo, hi) in enumerate(bounds):
                            raw.append(float(torch.logit(((candidate[i] - lo) / (hi - lo)).clamp(1e-7, 1 - 1e-7))))
                    latent_candidate = torch.tensor(raw, dtype=DTYPE)
                    q, _, _ = simulate(latent_candidate[None, None, None, :], arm, forcing[:, b:b+1])
                    value_kge = full_kge_reference(q[WARMUP:], observations[WARMUP:, b:b+1], eps=.1).item()
                    rows.append({"arm": arm, "basin_id": BASIN_IDS[b], "selection_method": method,
                                 "interception_parameter": key, "interception_value": value, "Sb1": sb1.item(), "KGE": value_kge})
    write_csv("liu_vs_f0_compensation_surface.csv", rows)
    return rows


def optimum_diagnostics(chosen, forcing, observations):
    pet_rows, wb_rows = [], []
    for arm in ARMS:
        for b in range(4):
            score, latent, method, run_id = chosen[arm][b]
            q, et, diag = simulate(latent[None, None, None, :], arm, forcing[:, b:b+1], collect=True)
            q = q[:, 0, 0, 0]
            et = et[:, 0, 0, 0]
            i, et1, et2, pr, pet = [diag[key][:, 0, 0] for key in ["i", "et1", "et2", "pr", "pet"]]
            total = i + et1 + et2
            exceed = (total - pet).clamp_min(0)
            for period, left, right in [("warmup", 0, WARMUP), ("scored", WARMUP, WARMUP + SCORED), ("all", 0, WARMUP + SCORED)]:
                ex = exceed[left:right]; rain = pr[left:right] > 1e-10
                pet_rows.append({"arm": arm, "basin_id": BASIN_IDS[b], "selection_method": method, "period": period,
                                 "exceedance_day_fraction_all": (ex > 0).double().mean().item(),
                                 "exceedance_day_fraction_rainy": (ex[rain] > 0).double().mean().item() if bool(rain.any()) else 0.0,
                                 "median_exceedance_days": ex[ex > 0].median().item() if bool((ex > 0).any()) else 0.0,
                                 "q95_exceedance": torch.quantile(ex, .95).item(), "max_exceedance": ex.max().item(),
                                 "sum_exceedance": ex.sum().item(), "total_I": i[left:right].sum().item(), "total_PET": pet[left:right].sum().item()})
            states0 = 5.0e-6
            state_sum = diag["state_sum"][:, 0, 0]
            flux_balance = forcing[:, b, 0].to(DTYPE) - et - q
            state_delta = torch.empty_like(state_sum)
            state_delta[0] = state_sum[0] - states0
            state_delta[1:] = state_sum[1:] - state_sum[:-1]
            daily_residual = flux_balance - state_delta
            full_residual = float(daily_residual.sum())
            scored_residual = float(daily_residual[WARMUP:].sum())
            wb_rows.append({"arm": arm, "basin_id": BASIN_IDS[b], "selection_method": method,
                            "full_residual": full_residual, "scored_residual": scored_residual,
                            "max_daily_abs_residual": daily_residual.abs().max().item(),
                            "max_state_abs": state_sum.abs().max().item(),
                            "water_balance_pass": bool(daily_residual.abs().max() < 1e-5)})
    write_csv("pet_budget_at_optimum.csv", pet_rows)
    write_csv("water_balance_at_optimum.csv", wb_rows)
    return pet_rows, wb_rows


def decision_outputs(cma_results, grad_results, chosen, ident_rows, pet_rows, wb_rows):
    rows = []
    for arm in ARMS:
        cma = [r[2] for r in cma_results[arm]]
        grad = [r[2] for r in grad_results[arm]]
        rows.append({"verdict_component": "CMA-ES basin KGE", "arm": arm, "value": float(np.median(cma)), "details": json.dumps(cma)})
        rows.append({"verdict_component": "Adam basin KGE", "arm": arm, "value": float(np.median(grad)), "details": json.dumps(grad)})
    def best_by_basin(result_rows, arm):
        return [max((r[2] for r in result_rows[arm] if r[0] == b), default=-1.0e9) for b in range(4)]
    cma_t1a = best_by_basin(cma_results, "T1a")
    cma_t1 = best_by_basin(cma_results, "T1")
    grad_t1a = best_by_basin(grad_results, "T1a")
    grad_t1 = best_by_basin(grad_results, "T1")
    cma_paired = [cma_t1[b] - cma_t1a[b] for b in range(4)]
    grad_paired = [grad_t1[b] - grad_t1a[b] for b in range(4)]
    paired = [chosen["T1"][b][0] - chosen["T1a"][b][0] for b in range(4)]
    c_boundary = [r for r in ident_rows if r["arm"] == "T1"]
    c_sensitivity = [r["median_rainy_abs_dI_dc"] for r in c_boundary]
    pet_scored = [r for r in pet_rows if r["period"] == "scored"]
    max_ex = max(r["exceedance_day_fraction_all"] for r in pet_scored)
    wb_pass = all(bool(r["water_balance_pass"]) for r in wb_rows)
    cma_gain = float(np.mean(cma_paired))
    grad_gain = float(np.mean(grad_paired))
    c_consistent = cma_gain > .01 and grad_gain > .01 and sum(v > .01 for v in cma_paired) >= 3 and sum(v > .01 for v in grad_paired) >= 3
    c_nonboundary = all((r["c_boundary_distance"] or 0.0) > .02 for r in c_boundary)
    c_ident = float(np.median(c_sensitivity)) > 1e-3 if c_sensitivity else False
    cma_medians = {arm: float(np.median([r[2] for r in cma_results[arm]])) for arm in ARMS}
    grad_medians = {arm: float(np.median([r[2] for r in grad_results[arm]])) for arm in ARMS}
    def classify(gap):
        return "YES" if gap >= -.02 else "MIXED" if gap >= -.25 else "NO"
    forward_viable = classify(cma_medians["T1a"] - cma_medians["F0"])
    gradient_viable = classify(grad_medians["T1a"] - grad_medians["F0"])
    if forward_viable != "NO" and c_consistent and c_nonboundary and c_ident:
        c_verdict = "YES"
    elif forward_viable != "NO" and not c_consistent:
        c_verdict = "NO"
    else:
        c_verdict = "NOT YET"
    s_eff_boundary_hits = 0
    for arm in ["T1a", "T1"]:
        for b in range(4):
            physical = latent_to_physical(chosen[arm][b][1][None, None, None, :], arm).flatten()
            s = physical[4].item()
            lo, hi = T1_BOUNDS[4]
            if min((s - lo) / (hi - lo), (hi - s) / (hi - lo)) <= .02:
                s_eff_boundary_hits += 1
    pet_verdict = "BLOCKER" if max_ex > .10 else "NEEDS SENSITIVITY" if max_ex > .02 else "YES"
    if forward_viable == "NO":
        next_action = "STOP-BEFORE-DPL"
    elif c_verdict == "YES" and pet_verdict == "YES" and wb_pass:
        next_action = "GO-T1"
    elif c_verdict == "NO" and pet_verdict == "YES" and wb_pass:
        next_action = "GO-T1a"
    else:
        next_action = "STOP-BEFORE-DPL"
    rows += [
        {"verdict_component": "Liu forward formulation viable", "arm": "ALL", "value": forward_viable, "details": "CMA-ES T1a best threshold"},
        {"verdict_component": "Gradient direct optimization viable", "arm": "ALL", "value": gradient_viable, "details": "Adam T1a best threshold"},
        {"verdict_component": "Second parameter c justified", "arm": "T1_vs_T1a", "value": c_verdict, "details": json.dumps(paired)},
        {"verdict_component": "PET semantics acceptable", "arm": "ALL", "value": pet_verdict, "details": str(max_ex)},
        {"verdict_component": "Water balance", "arm": "ALL", "value": wb_pass, "details": "optimized solutions"},
        {"verdict_component": "NEXT ACTION", "arm": "ALL", "value": next_action, "details": "pre-DPL gate"},
    ]
    write_csv("pre_dpl_decision_matrix.csv", rows)
    return {"forward_viable": forward_viable, "gradient_viable": gradient_viable, "c_verdict": c_verdict,
            "pet_verdict": pet_verdict, "water_balance": wb_pass, "next_action": next_action,
            "paired_deltas": paired, "cma_paired_deltas": cma_paired, "gradient_paired_deltas": grad_paired,
            "cma_medians": cma_medians, "gradient_medians": grad_medians,
            "max_pet_exceedance_day_fraction": max_ex,
            "c_boundary_hits": sum((r["c_boundary_distance"] or 0.0) <= .02 for r in c_boundary),
            "s_eff_boundary_hits": s_eff_boundary_hits,
            "median_c_sensitivity": float(np.median(c_sensitivity)) if c_sensitivity else float("nan")}


def summarize_compensation(rows):
    result = {}
    for arm in ARMS:
        correlations = []
        for basin_id in BASIN_IDS:
            basin_rows = [r for r in rows if r["arm"] == arm and r["basin_id"] == basin_id]
            values = [float(r["KGE"]) for r in basin_rows]
            threshold = np.quantile(values, .75)
            top = [r for r in basin_rows if float(r["KGE"]) >= threshold]
            if len(top) > 2 and len({r["interception_value"] for r in top}) > 1 and len({r["Sb1"] for r in top}) > 1:
                correlations.append(abs(float(np.corrcoef([float(r["interception_value"]) for r in top], [float(r["Sb1"]) for r in top])[0, 1])))
        result[arm] = float(np.mean(correlations)) if correlations else 0.0
    liu_corr = float(np.mean([result["T1a"], result["T1"]]))
    delta = liu_corr - result["F0"]
    result["liu_vs_f0"] = "STRONGER" if delta > .10 else "WEAKER" if delta < -.10 else "SIMILAR"
    result["liu_mean_abs_top_quartile_correlation"] = liu_corr
    return result


def write_report(summary, cma_results, grad_results):
    cma_median = {arm: float(np.median([r[2] for r in cma_results[arm]])) for arm in ARMS}
    grad_median = {arm: float(np.median([r[2] for r in grad_results[arm]])) for arm in ARMS}
    def basin_best(results, arm):
        return [max(r[2] for r in results[arm] if r[0] == b) for b in range(4)]
    cma_basin = {arm: basin_best(cma_results, arm) for arm in ARMS}
    grad_basin = {arm: basin_best(grad_results, arm) for arm in ARMS}
    basin_table = "\n".join(
        f"| {BASIN_IDS[b]} | {cma_basin['F0'][b]:.4f} | {cma_basin['T1a'][b]:.4f} | {cma_basin['T1'][b]:.4f} | {grad_basin['F0'][b]:.4f} | {grad_basin['T1a'][b]:.4f} | {grad_basin['T1'][b]:.4f} |"
        for b in range(4)
    )
    report = f"""# MOPEX4 LIU PRE-DPL FINAL VALIDATION

Protocol: CPU; basins `{', '.join(BASIN_IDS)}`; 365-day warm-up + 365-day scored window.
CMA-ES: existing `BatchedCMAES`, {CMA_STARTS} starts, population {CMA_POPULATION}, {CMA_GENERATIONS} generations, active covariance, stdev_init 0.25, solver seeds by arm `{CMA_SEEDS}`.
Gradient: existing model equations with direct `torch.optim.Adam`, {GRAD_STEPS} steps, lr {GRAD_LR}, seeds {SEEDS}; all common and interception parameters jointly optimized.
No shared-DPL or 531-basin training was started.

## Basin-level best KGE

| basin | F0 CMA-ES | T1a CMA-ES | T1 CMA-ES | F0 Adam | T1a Adam | T1 Adam |
|---|---:|---:|---:|---:|---:|---:|
{basin_table}

## Aggregate median KGE

- F0 CMA-ES: `{cma_median['F0']:.6f}`
- T1a CMA-ES: `{cma_median['T1a']:.6f}`
- T1 CMA-ES: `{cma_median['T1']:.6f}`
- F0 Adam: `{grad_median['F0']:.6f}`
- T1a Adam: `{grad_median['T1a']:.6f}`
- T1 Adam: `{grad_median['T1']:.6f}`

## Independent verdicts

1. Liu forward formulation viable under all-parameter calibration: **{summary['forward_viable']}**
2. Gradient direct optimization viable: **{summary['gradient_viable']}**
3. Second parameter `c` justified: **{summary['c_verdict']}**
4. PET / energy semantics acceptable for shared-DPL: **{summary['pet_verdict']}**

T1 minus T1a paired selected-solution KGE deltas: `{summary['paired_deltas']}`.
CMA-ES T1 minus T1a paired deltas: `{summary['cma_paired_deltas']}`.
Adam T1 minus T1a paired deltas: `{summary['gradient_paired_deltas']}`.
T1 `c` boundary hits: `{summary['c_boundary_hits']}/4`; T1/T1a S_eff boundary hits: `{summary['s_eff_boundary_hits']}/8`; median optimized `|dI/dc|`: `{summary['median_c_sensitivity']:.6g}`.
Maximum scored-period PET exceedance-day fraction: `{summary['max_pet_exceedance_day_fraction']:.6g}`.

## Compensation comparator

Top-quartile absolute interception-parameter/Sb1 correlation: F0 `{summary['compensation']['F0']:.4f}`, T1a `{summary['compensation']['T1a']:.4f}`, T1 `{summary['compensation']['T1']:.4f}`. Liu versus legacy F0: **{summary['compensation']['liu_vs_f0']}**.

## Final action

**{summary['next_action']}**

The optimized-solution artifacts distinguish formulation limits from gradient-accessibility limits. Results are descriptive for four basins and are not an official benchmark.
"""
    (OUT / "final_pre_dpl_validation_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    forcing, observations = load_window()
    (OUT / "protocol_and_budget.json").write_text(json.dumps(protocol(), indent=2), encoding="utf-8")
    print("Running matched all-parameter CMA-ES calibration")
    cma_by_arm = run_cma(forcing, observations)
    print("Running matched all-parameter gradient direct optimization")
    grad_by_arm = run_gradient(forcing, observations)
    chosen = choose_best(cma_by_arm, grad_by_arm)
    parameter_rows(chosen)
    ident_rows = identifiability_at_optimum(chosen, forcing)
    compensation_rows = compensation_surface(chosen, forcing, observations)
    pet_rows, wb_rows = optimum_diagnostics(chosen, forcing, observations)
    summary = decision_outputs(cma_by_arm, grad_by_arm, chosen, ident_rows, pet_rows, wb_rows)
    summary["compensation"] = summarize_compensation(compensation_rows)
    write_report(summary, cma_by_arm, grad_by_arm)
    flat_cma = {arm: [r[2] for r in cma_by_arm[arm]] for arm in ARMS}
    flat_grad = {arm: [r[2] for r in grad_by_arm[arm]] for arm in ARMS}
    final = {"protocol": protocol(), "cmaes_median_KGE": {a: float(np.median(v)) for a, v in flat_cma.items()},
             "gradient_median_KGE": {a: float(np.median(v)) for a, v in flat_grad.items()}, **summary,
             "training_started": False, "shared_dpl_started": False, "production_default_changed_further": False}
    (OUT / "audit_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("MOPEX4 LIU PRE-DPL FINAL VALIDATION")
    print("Basins: " + ", ".join(BASIN_IDS))
    print("Window: 365-day warm-up + 365-day scored; CPU")
    print(f"Optimization budgets: CMA-ES {CMA_STARTS} starts x population {CMA_POPULATION} x {CMA_GENERATIONS} generations; Adam {GRAD_STEPS} steps x seeds {SEEDS}")
    print(f"Derivative-free all-parameter calibration median KGE: F0 {summary['cma_medians']['F0']:.6f}; T1a {summary['cma_medians']['T1a']:.6f}; T1 {summary['cma_medians']['T1']:.6f}")
    print(f"Gradient all-parameter optimization median KGE: F0 {summary['gradient_medians']['F0']:.6f}; T1a {summary['gradient_medians']['T1a']:.6f}; T1 {summary['gradient_medians']['T1']:.6f}")
    print(f"CMA-ES basin best KGE: F0 {[max(r[2] for r in cma_by_arm['F0'] if r[0] == b) for b in range(4)]}; T1a {[max(r[2] for r in cma_by_arm['T1a'] if r[0] == b) for b in range(4)]}; T1 {[max(r[2] for r in cma_by_arm['T1'] if r[0] == b) for b in range(4)]}")
    print(f"Adam basin best KGE: F0 {[max(r[2] for r in grad_by_arm['F0'] if r[0] == b) for b in range(4)]}; T1a {[max(r[2] for r in grad_by_arm['T1a'] if r[0] == b) for b in range(4)]}; T1 {[max(r[2] for r in grad_by_arm['T1'] if r[0] == b) for b in range(4)]}")
    print(f"T1 vs T1a paired KGE deltas: selected {summary['paired_deltas']}; CMA-ES {summary['cma_paired_deltas']}; Adam {summary['gradient_paired_deltas']}; c boundary hits {summary['c_boundary_hits']}/4; c identifiability median |dI/dc| {summary['median_c_sensitivity']:.6g}")
    print(f"Compensation: F0 {summary['compensation']['F0']:.4f}; T1a {summary['compensation']['T1a']:.4f}; T1 {summary['compensation']['T1']:.4f}; Liu vs F0 {summary['compensation']['liu_vs_f0']}")
    print(f"PET budget: max exceedance-day fraction {summary['max_pet_exceedance_day_fraction']:.6g}; verdict {summary['pet_verdict']}")
    print(f"Water balance: {'PASS' if summary['water_balance'] else 'FAIL'}")
    print(f"Final verdicts: Liu forward {summary['forward_viable']}; gradient {summary['gradient_viable']}; c {summary['c_verdict']}; PET {summary['pet_verdict']}")
    print(f"NEXT ACTION: {summary['next_action']}")
    print("531-basin training started: NO")
    print("shared-dPL training started: NO")
    print("production default changed further: NO")


if __name__ == "__main__":
    main()
