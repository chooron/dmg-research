#!/usr/bin/env python3
"""Reproducible MOPEX4/5 interception-phase diagnosis.

This script deliberately imports the repository ``dmotpy`` package before the
historical benchmark mirror.  The mirror is retained for old artifacts but is
not the implementation under test here.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BENCHMARK = Path(__file__).resolve().parents[2]
REPO = BENCHMARK.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(BENCHMARK / "src"))

from dmotpy.data_contract import add_calendar_forcing
from dmotpy.models.flux.mopex import mopex_interception_4, mopex_interception_4_circular
from dmotpy.models.hydrology_model import HydrologyModel
from project.benchmark.src.batched_cmaes import BatchedCMAES, lhs_latent, stable_hash
from project.benchmark.src.model_registry import model_config
from project.benchmark.scripts.run_dpl_benchmark_dmg_native import (
    compute_differentiable_kge,
    load_camels_time_series,
)
from project.benchmark.src.data_selection import load_ids

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = BENCHMARK / "results" / "mopex45_phase_fix" / "circular_stage"
PERIOD = 365.25
WARMUP = 365
WINDOW = 730
SEEDS = (41, 42, 43)
# Matches the established MOPEX45 diagnostic calibration budget.  The full
# 730-day warm-up/objective window is retained; only restart candidates are
# batched rather than dispatched through Python one at a time.
DIRECT_STEPS = 15
DIRECT_LR = 1e-2
SUCCESS_GAP = 0.05


def write_csv(name: str, rows: list[dict]) -> None:
    path = OUT / name
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def kge(sim: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Native KGE wrapper, with one output per basin/parameter group."""
    _loss, score = compute_differentiable_kge(sim, obs, warmup_days=0)
    return torch.nan_to_num(score, nan=-1.0, posinf=-1.0, neginf=-1.0)


def make_model(model: str, phase: str = "scalar") -> HydrologyModel:
    cfg = model_config(model, warm_up=WARMUP, backend="python", parameter_mapping="auto", warmup_grad_mode="detach")
    cfg["phase_parameterization"] = phase
    return HydrologyModel(cfg, device=DEVICE, backend="python").to(DEVICE)


def forward(model: HydrologyModel, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return model({"x_phy": x}, (None, theta.unsqueeze(-1)))["streamflow"]


def phase_day(theta: torch.Tensor, phase: str) -> torch.Tensor:
    if phase == "scalar":
        return 1.0 + theta[..., 5] * 364.0
    cos_phi, sin_phi = 2.0 * theta[..., 5] - 1.0, 2.0 * theta[..., 6] - 1.0
    return torch.remainder(torch.atan2(sin_phi, cos_phi), 2 * math.pi) * PERIOD / (2 * math.pi)


def circular_from_scalar(theta: torch.Tensor) -> torch.Tensor:
    """Replace scalar slot five with two normalized circular network outputs."""
    day = 1.0 + theta[..., 5] * 364.0
    phi = 2 * math.pi * day / PERIOD
    circ = torch.cat((theta[..., :5], torch.zeros_like(theta[..., :1]), torch.zeros_like(theta[..., :1]), theta[..., 6:]), dim=-1)
    circ[..., 5] = (torch.cos(phi) + 1.0) / 2.0
    circ[..., 6] = (torch.sin(phi) + 1.0) / 2.0
    return circ


def optimize_direct(
    model_name: str, x: torch.Tensor, y: torch.Tensor, initial: torch.Tensor, *,
    phase: str = "scalar", freeze: dict[int, torch.Tensor] | None = None,
    steps: int = DIRECT_STEPS, seed: int, record_curve: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    torch.manual_seed(seed)
    hydro = make_model(model_name, phase)
    theta = nn.Parameter(initial.detach().clone())
    optimizer = torch.optim.AdamW([theta], lr=DIRECT_LR)
    freeze = freeze or {}
    curve = []
    best = torch.full((theta.shape[0],), -float("inf"), device=DEVICE)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for column, value in freeze.items():
                theta[:, column].copy_(value)
        previous_day = phase_day(theta.detach(), phase)
        simulated = forward(hydro, x, theta)
        score = kge(simulated, y)
        loss = (1.0 - score).mean()
        loss.backward()
        phase_grad = theta.grad[:, 5:7] if phase != "scalar" else theta.grad[:, 5:6]
        if freeze:
            for column in freeze:
                theta.grad[:, column].zero_()
        grad_norm = float(theta.grad.norm().item())
        optimizer.step()
        with torch.no_grad():
            theta.clamp_(0.0, 1.0)
            for column, value in freeze.items():
                theta[:, column].copy_(value)
        best = torch.maximum(best, score.detach())
        if record_curve:
            new_day = phase_day(theta.detach(), phase)
            curve.append({"seed": seed, "phase": phase, "step": step + 1,
                          "mean_kge": float(score.mean()), "gradient_norm": grad_norm,
                          "phase_gradient_norm": float(phase_grad.norm().item()),
                          "mean_phase_update_days": float((new_day - previous_day).abs().mean()),
                          "nonfinite": bool(not torch.isfinite(theta).all())})
    return theta.detach(), best.detach(), curve


def optimize_multistart(x: torch.Tensor, y: torch.Tensor, starts: int) -> torch.Tensor:
    """Independent AdamW starts evaluated as hydrologic parameter groups."""
    torch.manual_seed(3000 + starts)
    hydro = make_model("mopex4")
    basin_count = x.shape[1]
    theta = nn.Parameter(torch.rand(basin_count, 10, starts, device=DEVICE))
    optimizer = torch.optim.AdamW([theta], lr=DIRECT_LR)
    best = torch.full((basin_count, starts), -float("inf"), device=DEVICE)
    for _ in range(DIRECT_STEPS):
        optimizer.zero_grad(set_to_none=True)
        score = kge(hydro({"x_phy": x}, (None, theta))["streamflow"], y.repeat_interleave(starts, dim=1))
        score = score.reshape(basin_count, starts)
        (1.0 - score).mean().backward()
        optimizer.step()
        with torch.no_grad(): theta.clamp_(0.0, 1.0)
        best = torch.maximum(best, score.detach())
    return best


def run_cma_oracle(x: torch.Tensor, y: torch.Tensor, basin_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Current batched CMA-ES implementation, 4 starts x 24 population x 120 generations."""
    units, dimension, starts, population = len(basin_ids), 10, 4, 24
    solver = BatchedCMAES(units * starts, dimension, population, stdev_init=1.5,
                          active=True, seed=20260809, device=DEVICE)
    centers = torch.stack([lhs_latent(1, dimension, stable_hash(20260809, "mopex4", basin, start), DEVICE)[0]
                           for basin in basin_ids for start in range(starts)])
    solver.set_centers(centers)
    hydro = make_model("mopex4")
    with torch.no_grad():
        for _ in range(120):
            z, vector, latent = solver.ask()
            theta = torch.sigmoid(latent).reshape(units, starts * population, dimension).permute(0, 2, 1)
            sim = hydro({"x_phy": x}, (None, theta))["streamflow"]
            score = kge(sim, y.repeat_interleave(starts * population, dim=1)).reshape(units * starts, population)
            solver.tell(z, vector, latent, score)
    best = solver.state.best_fitness.reshape(units, starts)
    start_index = best.argmax(dim=1)
    latent = solver.state.best_latent.reshape(units, starts, dimension)
    theta = torch.sigmoid(latent[torch.arange(units, device=DEVICE), start_index]).float()
    return theta, best.max(dim=1).values.float()


def activity_rows(x: torch.Tensor, ids: list[int], oracle: torch.Tensor) -> list[dict]:
    rain = x[..., 0]
    doy = x[..., 3]
    alpha = oracle[:, 4]
    is_time = 1.0 + oracle[:, 5] * 364.0
    fraction = alpha.unsqueeze(0) + (1 - alpha.unsqueeze(0)) * torch.cos(2 * math.pi * (doy - is_time.unsqueeze(0)) / PERIOD)
    rainy = rain > 0.1
    gate = torch.sigmoid(50 * fraction)
    da = gate * (1 - torch.cos(2 * math.pi * (doy - is_time.unsqueeze(0)) / PERIOD)) * rain
    dt = gate * (1 - alpha.unsqueeze(0)) * (2 * math.pi / PERIOD) * torch.sin(2 * math.pi * (doy - is_time.unsqueeze(0)) / PERIOD) * rain
    rows = []
    for index, basin in enumerate(ids):
        mask = rainy[:, index]
        support = (da[:, index].abs() > 1e-6) | (dt[:, index].abs() > 1e-6)
        sign = torch.sign(dt[mask, index]); changes = (sign[1:] != sign[:-1]).float().mean() if sign.numel() > 1 else torch.tensor(0.0, device=DEVICE)
        band = ("<0.25" if alpha[index] < .25 else "0.25-0.5" if alpha[index] < .5 else
                "0.5-0.75" if alpha[index] < .75 else ">=0.75")
        denom = mask.sum().clamp_min(1)
        rows.append({"model": "mopex4", "basin_id": basin, "ic_alpha": float(alpha[index]), "alpha_band": band,
                     "all_timestep_count": int(mask.numel()), "rainy_timestep_count": int(mask.sum()),
                     "rainy_raw_fraction_le_zero": float((mask & (fraction[:, index] <= 0)).sum() / denom),
                     "rainy_softplus_fraction_near_zero": float((mask & (torch.nn.functional.softplus(50 * fraction[:, index]) / 50 <= 1e-6)).sum() / denom),
                     "rainy_dflux_dalpha_exact_zero": float((mask & (da[:, index] == 0)).sum() / denom),
                     "rainy_dflux_distime_exact_zero": float((mask & (dt[:, index] == 0)).sum() / denom),
                     "rainy_dflux_dalpha_near_zero": float((mask & (da[:, index].abs() <= 1e-6)).sum() / denom),
                     "rainy_dflux_distime_near_zero": float((mask & (dt[:, index].abs() <= 1e-6)).sum() / denom),
                     "mean_abs_dflux_dalpha": float(da[mask, index].abs().mean()),
                     "mean_abs_dflux_distime": float(dt[mask, index].abs().mean()),
                     "active_window_days": int(support.sum()), "gradient_sign_change_frequency": float(changes)})
    return rows


def landscape(x: torch.Tensor, y: torch.Tensor, basin_id: int, theta: torch.Tensor) -> tuple[list[dict], list[dict]]:
    alpha_grid = torch.linspace(0, 1, 21, device=DEVICE)
    time_grid = torch.linspace(1, 365, 48, device=DEVICE)
    aa, tt = torch.meshgrid(alpha_grid, time_grid, indexing="ij")
    candidates = theta.repeat(aa.numel(), 1)
    candidates[:, 4], candidates[:, 5] = aa.flatten(), (tt.flatten() - 1) / 364
    hydro = make_model("mopex4")
    with torch.no_grad():
        sim = hydro({"x_phy": x}, (None, candidates.T.unsqueeze(0)))["streamflow"]
        values = kge(sim, y.repeat_interleave(aa.numel(), dim=1)).reshape(21, 48)
    matrix_rows = [{"model": "mopex4", "basin_id": basin_id, "alpha": float(a), "is_time": float(t), "kge": float(values[i, j])}
                   for i, a in enumerate(alpha_grid) for j, t in enumerate(time_grid)]
    peak = torch.ones_like(values, dtype=torch.bool)
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        peak &= values >= torch.roll(values, (di, dj), (0, 1))
    summary = [{"model": "mopex4", "basin_id": basin_id, "local_optima_count": int(peak.sum()),
                "grid_best_kge": float(values.max()), "grid_worst_kge": float(values.min()),
                "flat_cell_fraction_within_0.01_of_best": float((values >= values.max() - .01).float().mean()),
                "relu_inactive_cell_fraction": float((aa + (1 - aa) * torch.cos(2 * math.pi * (tt - tt.mean()) / PERIOD) <= 0).float().mean()),
                "ic_alpha": float(theta[4]), "ic_is_time": float(1 + theta[5] * 364)}]
    return matrix_rows, summary


def regression_rows() -> list[dict]:
    doy = torch.arange(1, 366, device=DEVICE, dtype=torch.float32)
    rainfall = torch.full_like(doy, 10.0)
    alpha, day = torch.tensor([.42], device=DEVICE), torch.tensor([73.0], device=DEVICE)
    phi = 2 * math.pi * day / PERIOD
    scalar = mopex_interception_4(rainfall, doy, alpha, day)
    circular = mopex_interception_4_circular(rainfall, doy, alpha, torch.cos(phi), torch.sin(phi))
    shifted = mopex_interception_4_circular(rainfall, doy, alpha, torch.cos(phi + 2 * math.pi), torch.sin(phi + 2 * math.pi))
    c = torch.tensor([.3], device=DEVICE, requires_grad=True); s = torch.tensor([.7], device=DEVICE, requires_grad=True)
    mopex_interception_4_circular(rainfall, doy, alpha, c, s).sum().backward()
    return [{"test": "scalar_circular_forward_equivalence", "value": float((scalar - circular).abs().max()), "pass": bool((scalar - circular).abs().max() < 2e-5)},
            {"test": "circular_period_equivalence", "value": float((circular - shifted).abs().max()), "pass": bool((circular - shifted).abs().max() < 2e-5)},
            {"test": "circular_autograd_finite", "value": float(torch.isfinite(c.grad).all() and torch.isfinite(s.grad).all()), "pass": bool(torch.isfinite(c.grad).all() and torch.isfinite(s.grad).all())}]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark protocol")
    ids = [int(x) for x in load_ids("data/531sub_id.txt")]
    train_x, train_y, _, _ = load_camels_time_series(ids)
    dates = pd.date_range("1980-10-01", periods=train_x.shape[0], freq="D")
    x, _ = add_calendar_forcing(torch.as_tensor(train_x, dtype=torch.float32, device=DEVICE), dates, model_name="mopex4")
    y = torch.as_tensor(train_y, dtype=torch.float32, device=DEVICE)
    # Uniformly spaced candidates are selected before looking at any result.
    candidate_idx = torch.linspace(0, len(ids) - 1, 9, device=DEVICE).long()
    x0, y0 = x[:WINDOW, candidate_idx], y[WARMUP:WINDOW, candidate_idx]
    oracle_theta, oracle_kge = run_cma_oracle(x0, y0, [ids[i] for i in candidate_idx.tolist()])
    initial = torch.full_like(oracle_theta, .5)
    _, a0_probe, _ = optimize_direct("mopex4", x0, y0, initial, seed=SEEDS[0])
    gaps = oracle_kge - a0_probe
    high, median, low = int(gaps.argmax()), int(gaps.argsort()[len(gaps)//2]), int(gaps.argmin())
    selected_local = sorted(set((high, median, low, 0, len(gaps) - 1)))
    selected_idx = candidate_idx[torch.tensor(selected_local, device=DEVICE)]
    selected_ids = [ids[i] for i in selected_idx.tolist()]
    x1, y1, oracle_theta, oracle_kge = x[:WINDOW, selected_idx], y[WARMUP:WINDOW, selected_idx], oracle_theta[selected_local], oracle_kge[selected_local]
    write_csv("00_protocol_and_oracle.csv", [{"basin_id": basin, "selection": "high_gap" if i == high else "median_gap" if i == median else "low_gap" if i == low else "random_uniform",
                                                "ic_kge": float(oracle_kge[j]), "oracle_theta_json": json.dumps(oracle_theta[j].cpu().tolist())}
                                               for j, (i, basin) in enumerate(zip(selected_local, selected_ids))])

    # Stage 1: four constrained calibrations plus exact oracle forward.
    oracle_rows, other_rows = [], []
    cases = {"A0": {}, "A1": {5: oracle_theta[:, 5]}, "A2": {4: oracle_theta[:, 4]},
             "A3": {4: oracle_theta[:, 4], 5: oracle_theta[:, 5]}}
    for seed in SEEDS:
        outcome = {}
        for case, frozen in cases.items():
            _, score, _ = optimize_direct("mopex4", x1, y1, torch.full_like(oracle_theta, .5), freeze=frozen, seed=seed)
            outcome[case] = score
        with torch.no_grad():
            a4 = kge(forward(make_model("mopex4"), x1, oracle_theta), y1)
        # Fix both interception parameters and optimise only the remaining coordinates.
        _, other_score, _ = optimize_direct("mopex4", x1, y1, torch.full_like(oracle_theta, .5), freeze=cases["A3"], seed=seed)
        for i, basin in enumerate(selected_ids):
            base, ic = float(outcome["A0"][i]), float(a4[i])
            oracle_rows.append({"model": "mopex4", "basin_id": basin, "seed": seed, "ic_kge": ic,
                                "A0_kge": base, "A1_kge": float(outcome["A1"][i]), "A2_kge": float(outcome["A2"][i]),
                                "A3_kge": float(outcome["A3"][i]), "A4_forward_kge": ic,
                                "is_time_only_recovery": (float(outcome["A1"][i]) - base) / max(ic - base, 1e-6),
                                "alpha_only_recovery": (float(outcome["A2"][i]) - base) / max(ic - base, 1e-6),
                                "alpha_is_time_recovery": (float(outcome["A3"][i]) - base) / max(ic - base, 1e-6)})
            other_rows.append({"model": "mopex4", "basin_id": basin, "seed": seed, "fixed_alpha_is_time_other_parameters_kge": float(other_score[i]), "ic_kge": ic})
    write_csv("01_oracle_decomposition.csv", oracle_rows); write_csv("03_other_parameters_given_interception_oracle.csv", other_rows)
    write_csv("02_rainy_day_activity.csv", activity_rows(x1, selected_ids, oracle_theta))

    matrix, landscape_rows = [], []
    for i, basin in enumerate(selected_ids[:3]):
        raw, summary = landscape(x1[:, i:i+1], y1[:, i:i+1], basin, oracle_theta[i])
        matrix.extend(raw); landscape_rows.extend(summary)
    write_csv("04_alpha_is_time_landscape_matrix.csv", matrix); write_csv("04_alpha_is_time_landscape_summary.csv", landscape_rows)

    # Stage 4: independent starts, with the same optimization budget each time.
    multi_rows = []
    for count in (1, 5, 20, 50):
        scores = optimize_multistart(x1, y1, count)
        for start in range(count):
            for i, basin in enumerate(selected_ids):
                multi_rows.append({"model": "mopex4", "basin_id": basin, "n_starts": count, "start": start,
                                   "final_kge": float(scores[i, start]), "ic_kge": float(oracle_kge[i]),
                                   "success": bool(oracle_kge[i] - scores[i, start] < SUCCESS_GAP)})
    write_csv("05_multistart_adamw.csv", multi_rows)

    # Stage 5/6: A/B/C direct calibration. Circular candidates start at the
    # same canonical phase as A's midpoint, rather than at the singular origin.
    abc_rows, curves = [], []
    for label, phase in (("A_scalar", "scalar"), ("B_atan2", "atan2"), ("C_direct_circular", "circular")):
        for seed in SEEDS:
            init = torch.full_like(oracle_theta, .5) if phase == "scalar" else circular_from_scalar(torch.full_like(oracle_theta, .5))
            torch.manual_seed(seed); init = (init + .15 * torch.rand_like(init)).clamp(0.001, .999)
            final, score, curve = optimize_direct("mopex4", x1, y1, init, phase=phase, seed=seed, record_curve=True)
            curves.extend([{**row, "candidate": label} for row in curve])
            for i, basin in enumerate(selected_ids):
                abc_rows.append({"model": "mopex4", "candidate": label, "basin_id": basin, "seed": seed,
                                 "final_kge": float(score[i]), "final_loss": float(1 - score[i]), "ic_kge": float(oracle_kge[i]),
                                 "final_phase_day": float(phase_day(final, phase)[i]), "nonfinite": bool(not torch.isfinite(final[i]).all())})
    write_csv("06_abc_direct_calibration.csv", abc_rows); write_csv("06_abc_convergence.csv", curves)
    write_csv("09_regression_tests.csv", regression_rows())

    table = pd.DataFrame(abc_rows).groupby("candidate").final_kge.agg(["mean", "median", "std"])
    best_candidate = table["median"].idxmax()
    scalar_median = float(table.loc["A_scalar", "median"])
    gate = best_candidate != "A_scalar" and float(table.loc[best_candidate, "median"]) > scalar_median + .01
    write_csv("07_full_dpl_gate.csv", [{"candidate": best_candidate, "direct_median_kge": float(table.loc[best_candidate, "median"]),
                                         "scalar_median_kge": scalar_median, "passes_direct_gate": gate,
                                         "decision": "FULL_DPL_NOT_RUN: direct gate failed" if not gate else "FULL_DPL_REQUIRED"}])
    report = f"""# MOPEX4/MOPEX5 circular-phase stage report

## Runtime implementation

The tested runtime is `{REPO}/dmotpy/models`.  MOPEX4/5 call `mopex_interception_4(flux_pr, doy, alpha, is_time)`: it uses a softplus approximation to ReLU, not a hard ReLU.  `phenology_1` is `clamp((T-tmin)/trange, 0, 1) * PET`.  Older benchmark scripts import the mirror under `project/benchmark/dmotpy`; this run explicitly imports the repository implementation first.

## Gate

Best direct candidate: **{best_candidate}**, median KGE {float(table.loc[best_candidate, 'median']):.4f}; scalar A median {scalar_median:.4f}.  Full dPL gate: **{'pass' if gate else 'fail'}**.  {'A full MOPEX4 dPL comparison is required by the gate.' if gate else 'No full dPL or MOPEX5 run was authorised: the required stable direct-gradient improvement was absent.'}

## Evidence files

`01_oracle_decomposition.csv` contains A0--A4 per basin and seed. `02_rainy_day_activity.csv` reports all/rainy-day support under actual softplus. `04_*` contains the full landscape matrices. `05_multistart_adamw.csv`, `06_abc_*`, and `09_regression_tests.csv` provide raw optimization and equivalence evidence.

## Production recommendation

Do not change production defaults. The experimental circular path is default-off and only a positive direct and full-dPL gate can justify adoption. MOPEX5 is intentionally deferred until MOPEX4 passes that gate.
"""
    (OUT / "final_circular_stage_report.md").write_text(report)


if __name__ == "__main__":
    main()
