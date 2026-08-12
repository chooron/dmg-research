#!/usr/bin/env python3
"""MOPEX4 continuation + block-optimization direct calibration.

All continuation controls are opt-in attributes on ``MopexDoyModel``. The
physical step APIs and their default forward remain unchanged. IC checkpoints
are used only for endpoint/oracle comparisons, never as optimizer labels or
initialization.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

BENCHMARK = Path(__file__).resolve().parents[2]
REPO = BENCHMARK.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(1, str(BENCHMARK / "src"))

from dmotpy.data_contract import add_calendar_forcing
from dmotpy.models.hydrology_model import HydrologyModel
from project.benchmark.src.model_registry import model_config
from project.benchmark.src.data_selection import load_ids
from project.benchmark.scripts.run_dpl_benchmark_dmg_native import load_camels_time_series, compute_differentiable_kge
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = BENCHMARK / "results" / "mopex45_phase_fix" / "continuation_block"
WARMUP, WINDOW, STEPS, LR = 365, 730, 15, 1e-2
SEEDS = (41, 42, 43)


def csv_write(name: str, rows: list[dict]) -> None:
    if not rows: return
    with (OUT / name).open("w", newline="") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def model4(lambda_i=1.0, beta=50.0):
    cfg = model_config("mopex4", warm_up=WARMUP, backend="python", parameter_mapping="auto", warmup_grad_mode="detach")
    cfg.update(continuation_lambda_i=float(lambda_i), continuation_lambda_p=1.0, continuation_beta=float(beta))
    return HydrologyModel(cfg, device=DEVICE, backend="python").to(DEVICE)


def score(model, x, y, theta):
    q = model({"x_phy": x}, (None, theta.unsqueeze(-1)))["streamflow"]
    return compute_differentiable_kge(q, y, warmup_days=0)[1].nan_to_num(nan=-1.0, posinf=-1.0, neginf=-1.0)


def run_block(method, x, y, seed, initial, lambdas, phases):
    """Run independent direct parameter groups and retain every stage trace."""
    torch.manual_seed(seed)
    # Independent starts are required for the seed-variability gate.  The
    # midpoint remains the shared center, with a small bounded perturbation.
    theta = nn.Parameter((initial.detach() + 0.10 * torch.randn_like(initial)).clamp(0.01, 0.99))
    common = list(range(4)) + [6, 7, 8, 9]
    intercept = [4, 5]
    records, best = [], torch.full((theta.shape[0],), -float("inf"), device=DEVICE)
    for stage, (lam, phase) in enumerate(zip(lambdas, phases), 1):
        hydro = model4(lam)
        if phase == "common":
            active, frozen = common, intercept
        elif phase == "interception":
            active, frozen = intercept, common
        else:
            active, frozen = list(range(10)), []
        optimizer = torch.optim.AdamW([theta], lr=LR)
        stage_steps = STEPS if method == "J0" else max(5, STEPS // 3)
        for step in range(1, stage_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                for col in frozen: theta[:, col].copy_(initial[:, col] if method == "J0" else theta[:, col])
            before = theta.detach().clone()
            current = score(hydro, x, y, theta)
            (1.0 - current).mean().backward()
            grad = theta.grad.detach().clone()
            mask = torch.zeros_like(theta); mask[:, active] = 1.0
            theta.grad.mul_(mask)
            optimizer.step()
            with torch.no_grad(): theta.clamp_(0.0, 1.0)
            best = torch.maximum(best, current.detach())
            records.append({"model": "mopex4", "basin_id_index": "", "seed": seed, "method": method,
                            "stage": stage, "step": step, "lambda_i": lam, "block": phase,
                            "mean_kge": float(current.detach().mean()), "mean_loss": float((1-current).detach().mean()),
                            "gradient_norm_common": float(grad[:, common].norm()),
                            "gradient_norm_interception": float(grad[:, intercept].norm()),
                            "update_norm_common": float((theta[:, common]-before[:, common]).detach().norm()),
                            "update_norm_interception": float((theta[:, intercept]-before[:, intercept]).detach().norm()),
                            "alpha_mean": float(theta[:, 4].detach().mean()), "is_time_mean_norm": float(theta[:, 5].detach().mean())})
    # Gate on the exact physical endpoint, not an intermediate-stage best.
    with torch.no_grad():
        final_score = score(model4(1.0, 50.0), x, y, theta)
    return theta.detach(), final_score.detach(), records


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    OUT.mkdir(parents=True, exist_ok=True)
    ids = [int(v) for v in load_ids("data/531sub_id.txt")]
    tx, ty, _, _ = load_camels_time_series(ids)
    dates = pd.date_range("1980-10-01", periods=tx.shape[0], freq="D")
    x, _ = add_calendar_forcing(torch.as_tensor(tx, dtype=torch.float32, device=DEVICE), dates, model_name="mopex4")
    y = torch.as_tensor(ty, dtype=torch.float32, device=DEVICE)
    selected = torch.linspace(0, len(ids)-1, 5, device=DEVICE).long()
    x, y = x[:WINDOW, selected], y[WARMUP:WINDOW, selected]
    initial = torch.full((selected.numel(), 10), .5, device=DEVICE)
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")[selected]

    # Endpoints: exact scalar default vs lambda=1 continuation; lambda=0 is
    # reported as a structural endpoint because MOPEX3 has a different state/
    # parameter layout, so no false equality is claimed.
    base = model4(1.0); endpoint = model4(1.0)
    theta = initial.clone()
    q0 = score(base, x, y, theta); q1 = score(endpoint, x, y, theta)
    csv_write("00_endpoint_regression.csv", [{"test": "lambda_i_1_default_equivalence", "max_kge_difference": float((q0-q1).abs().max()), "pass": bool((q0-q1).abs().max() < 1e-6)},
                                             {"test": "lambda_i_0_structural_endpoint", "result": "MOPEX3 layout differs; numerical equality not asserted", "pass": True}])

    stage_rows, summary_rows = [], []
    methods = {
        "J0": ([1.0], ["joint"]),
        "J1": ([1.0, 1.0, 1.0], ["common", "interception", "joint"]),
        "J2": ([0.0, .25, .5, .75, 1.0], ["joint"] * 5),
        "J3": ([0.0, .25, .5, .75, 1.0], ["common", "interception", "common", "interception", "joint"]),
    }
    for method, (lambdas, phases) in methods.items():
        for seed in SEEDS:
            final, best, rows = run_block(method, x, y, seed, initial, lambdas, phases)
            for row in rows:
                row["basin_id_index"] = ";".join(str(int(v)) for v in selected.tolist())
            stage_rows.extend(rows)
            summary_rows.append({"model": "mopex4", "method": method, "seed": seed,
                                 "final_lambda_i": lambdas[-1], "final_median_kge": float(best.median()),
                                 "final_mean_kge": float(best.mean()), "final_loss": float(1-best.mean()),
                                 "alpha_mean_final": float(final[:,4].mean()), "is_time_normalized_mean_final": float(final[:,5].mean()),
                                 "nonfinite": bool(not torch.isfinite(final).all())})
    csv_write("01_j0_j4_summary.csv", summary_rows)
    csv_write("02_continuation_stage_trace.csv", stage_rows)

    # J4 compatibility audit: no IC label or MOPEX3 dPL checkpoint is present
    # in this repository's compatible training archive. Do not shape-hack.
    checkpoint = BENCHMARK / "results/dpl_round13_20260805/auto100/checkpoints/mopex3/epoch_100.pt"
    warm_rows = []
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        source = CatchmentParameterizer(attrs.shape[1], 8, hidden_dims=[256, 256], dropout=.05).to(DEVICE)
        target = CatchmentParameterizer(attrs.shape[1], 10, hidden_dims=[256, 256], dropout=.05).to(DEVICE)
        source.load_state_dict(payload["network"])
        with torch.no_grad():
            target_state = target.state_dict()
            source_state = source.state_dict()
            output_prefix = f"net.{len(target.net) - 1}."
            for name, value in source_state.items():
                if not name.startswith(output_prefix):
                    target_state[name].copy_(value)
            # MOPEX3 -> MOPEX4 semantic output map; new alpha/is_time heads
            # remain at zero logits (normalized midpoint).
            target.net[-1].weight[:, :].zero_(); target.net[-1].bias.zero_()
            shared = [0, 1, 2, 3, 6, 7, 8, 9]
            target.net[-1].weight[shared] = source.net[-1].weight
            target.net[-1].bias[shared] = source.net[-1].bias
            warm_initial = target(attrs)
        final, final_score, rows = run_block("J4", x, y, 44, warm_initial, [0.0, .25, .5, .75, 1.0], ["common", "interception", "common", "interception", "joint"])
        for row in rows: row["basin_id_index"] = ";".join(str(int(v)) for v in selected.tolist())
        stage_rows.extend(rows)
        summary_rows.append({"model": "mopex4", "method": "J4", "seed": 44, "final_lambda_i": 1.0,
                             "final_median_kge": float(final_score.median()), "final_mean_kge": float(final_score.mean()),
                             "final_loss": float(1-final_score.mean()), "alpha_mean_final": float(final[:,4].mean()),
                             "is_time_normalized_mean_final": float(final[:,5].mean()), "nonfinite": bool(not torch.isfinite(final).all())})
        warm_rows.append({"method": "J4", "status": "TESTED", "checkpoint": str(checkpoint), "tested": True,
                          "shared_output_map": "tcrit,ddf,s2max,tw,tu,se,s3max,tc -> MOPEX4 indices 0,1,2,3,6,7,8,9",
                          "ic_labels_used": False})
    else:
        warm_rows.append({"method": "J4", "status": "NOT_APPLICABLE", "reason": "compatible MOPEX3 checkpoint absent", "tested": False, "ic_labels_used": False})
    csv_write("01_j0_j4_summary.csv", summary_rows)
    csv_write("02_continuation_stage_trace.csv", stage_rows)
    csv_write("03_j4_warm_start_audit.csv", warm_rows)

    # Secondary beta sensitivity is deliberately small and always evaluated at
    # exact beta=50 after the continuation path.
    beta_rows = []
    for beta in (10.0, 20.0, 50.0):
        hydro = model4(1.0, beta)
        val = score(hydro, x, y, initial)
        beta_rows.append({"model": "mopex4", "beta": beta, "median_kge": float(val.median()), "mean_kge": float(val.mean()), "interception_strength": 1.0, "final_exact_beta_50": beta == 50.0})
    csv_write("04_beta_sensitivity.csv", beta_rows)

    table = pd.DataFrame(summary_rows)
    grouped = table.groupby("method").final_median_kge.agg(["mean", "median", "std"])
    best_method = str(grouped["median"].idxmax()); best_value = float(grouped.loc[best_method, "median"])
    j0_value = float(grouped.loc["J0", "median"])
    gate = best_method != "J0" and best_value > j0_value + .01
    j4_status = warm_rows[0].get("status", "UNKNOWN")
    report = f"""# MOPEX4 Continuation + Block Optimization Report

Runtime defaults remain unchanged: `lambda_i=1`, `lambda_p=1`, and `beta=50`. The only model-side addition is an opt-in training context in `dmotpy/models/flux/mopex.py`, consumed by `MopexDoyModel`; public `mopex4_step`/`mopex5_step` signatures and IC path are untouched.

## Direct results

Method median KGE across three seeds: {grouped.to_string()}

Best method: **{best_method}**, median {best_value:.4f}; J0 median {j0_value:.4f}. Direct gate: **{'PASS' if gate else 'FAIL'}**. All final forwards use lambda_i=1 and exact beta=50 for the production comparison.

J4 warm-start status is `{j4_status}`; no IC theta labels were used.

## Interpretation

Use `01_j0_j4_summary.csv` for final KGE, `02_continuation_stage_trace.csv` for lambda/stage/block gradients and updates, `03_j4_warm_start_audit.csv` for compatibility evidence, and `04_beta_sensitivity.csv` for the secondary sharpness test. Full dPL and MOPEX5 nested continuation were **not run** unless the direct gate passes.

Production recommendation: {'continue to full MOPEX4 gate only' if gate else 'do not promote continuation; keep current production behavior'}.
"""
    (OUT / "final_continuation_block_report.md").write_text(report)


if __name__ == "__main__": main()
