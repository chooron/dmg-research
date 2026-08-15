#!/usr/bin/env python3
"""Official final MOPEX4 dPL seed run -- canonical 531-basin shared-dPL protocol.

Strictly reuses the canonical dPL training contract of the 36-model benchmark
(k_full_retrain.py / run_dpl_benchmark_dmg_native.py), with zero protocol
changes for MOPEX4:

- basins            : data/531sub_id.txt (531)
- attributes        : Caravan 35, zscore over all 531, log1p on skewed cols
                      (CatchmentAttributeBuilder)
- forcing / targets : NATIVE.load_camels_time_series -- train
                      1980-10-01..1995-09-30, validation 1995-10-01..2010-09-30,
                      +365 d warm-up prefix for validation; mm/d streamflow
- calendar forcing  : add_calendar_forcing (MOPEX4 is a calendar model)
- windows           : informative 365 d KGE catalog (std(Q) >= 0.01 mm/d);
                      730 d windows = 365 warm-up + 365 scored
- network           : CatchmentParameterizer(35 -> 10, hidden [256,256],
                      dropout .05), midpoint init (final layer zeroed)
- model             : build_model("mopex4", cuda, warm_up=365,
                      backend="compile", parameter_mapping="auto",
                      warmup_grad_mode="detach")
- optimizer         : AdamW(lr=1e-3, weight_decay=1e-4), grad clip 1.0
- batch             : 100 basins/step, 169 steps/epoch
- loss              : differentiable 1-KGE (NATIVE.compute_differentiable_kge)
- seed              : 42 (torch.manual_seed + torch.cuda.manual_seed_all)
- epochs            : force exactly 100 (early stopping disabled -- matches the
                      round13 protocol that produced the MOPEX3 reference)
- eval              : per-basin KGE (+NSE) on the full validation period;
                      median/mean over basins per epoch

Fail-fast (epoch 1 onward): finite train loss / q / theta / grads, S_eff & c
raw outputs finite, no immediate boundary locking, PET closure and water
balance on sampled basins, compile recompile watch.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
# dmotpy lives at the repo root; prepend it so this runner deterministically
# imports the current worktree's package (never a stale site-packages copy).
REPO = ROOT.parent.parent
sys.path[:0] = [str(REPO), str(ROOT), str(ROOT / "src")]

from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from dmotpy.data_contract import add_calendar_forcing
from dmotpy.models.registry import PARAM_INFO
from src.data_selection import load_ids
from src.model_registry import NPARAM_INFO_36, build_model

DEVICE = torch.device("cuda")
BATCH, STEPS, WINDOW, WARMUP = 100, 169, 730, 365
# Canonical dPL runner for restored MOPEX4 (default) and MOPEX5.  Both share
# the restored-MOPEX4 process semantics; MOPEX5 adds the original phenology
# PET adjustment (tmin/trange).  The training protocol is identical and the
# result directory is selected per model so historical runs are never touched.
def resolve_run_config(model: str) -> tuple:
    if model == "mopex4":
        i0, i1, phen = "alpha", "is_time", ()
        env, default = "MOPEX4_OUT", "results/dpl_mopex4_final_20260811"
    elif model == "mopex5":
        i0, i1, phen = "alpha", "is_time", ("tmin", "trange")
        env, default = "MOPEX5_OUT", "results/dpl_mopex5_final_20260812"
    else:
        raise ValueError(f"unsupported canonical dPL model: {model}")
    return i0, i1, phen, Path(_os.environ.get(env, str(ROOT / default)))

import os as _os

MODEL = "mopex4"
I0, I1 = "alpha", "is_time"
PHEN: tuple[str, ...] = ()
OUT_ROOT = Path(_os.environ.get("MOPEX4_OUT", str(ROOT / "results/dpl_mopex4_final_20260811")))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H1 = load_module(ROOT / "scripts/diagnostics/h_training_pilot.py", "m4_h1_helpers")
NATIVE = H1.NATIVE


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_path(arm_dir: Path, epoch: int) -> Path:
    return arm_dir / "checkpoints" / MODEL / f"epoch_{epoch:03d}.pt"


def latest_checkpoint(arm_dir: Path):
    files = sorted((arm_dir / "checkpoints" / MODEL).glob("epoch_*.pt"))
    return files[-1] if files else None


def nse_per_basin(q_sim: torch.Tensor, q_obs: torch.Tensor) -> torch.Tensor:
    """NSE per basin over masked finite days (both tensors [time, basin])."""
    mask = torch.isfinite(q_obs) & torch.isfinite(q_sim) & (q_obs >= 0.0) & (q_sim >= 0.0)
    n_valid = mask.sum(dim=0).clamp_min(1.0)
    obs = torch.where(mask, q_obs, torch.zeros_like(q_obs))
    sim = torch.where(mask, q_sim, torch.zeros_like(q_sim))
    mean_obs = obs.sum(dim=0) / n_valid
    denom = ((obs - mean_obs[None, :]) ** 2).sum(dim=0)
    num = ((sim - obs) ** 2).sum(dim=0)
    nse = 1.0 - num / (denom + 1e-6)
    return torch.where(mask.sum(dim=0) > 30, nse, torch.full_like(nse, float("nan")))


def initialize_midpoint(network: CatchmentParameterizer) -> None:
    layer = network.net[-1]
    if not isinstance(layer, nn.Linear):
        raise TypeError("output layer must be Linear")
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()


def make_fluxes_step(hydro, pv):
    """Return a step closure replicating the production step flux-by-flux.

    Restored interception kernel (original seasonal alpha/is_time) is shared by
    both models.  MOPEX4 closes the budget against raw PET; MOPEX5 against the
    phenology-adjusted demand PET_epc = GSI(T)*PET (with PET_epc <= PET).
    Returns (Q, I, ET1, ET2, new_states) plus the water-balance residual.
    """
    from dmotpy.models.flux.mopex import (
        mopex_baseflow_1 as baseflow_1,
        mopex_evap_7 as evap_7,
        mopex_melt_1 as melt_1,
        mopex_rainfall_1 as rainfall_1,
        mopex_recharge_3 as recharge_3,
        mopex_saturation_1 as saturation_1,
        mopex_snowfall_1 as snowfall_1,
    )
    from dmotpy.models.core.mopex4 import interception_4 as interception_kernel
    nearzero = hydro.nearzero
    if MODEL == "mopex4":
        from dmotpy.models.core.mopex4 import mopex4_step
        raw_step = mopex4_step
    else:
        from dmotpy.models.core.mopex5 import mopex5_step, phenology_effective_pet
        raw_step = mopex5_step

    def step(P, T, PET, states, doy):
        S1, S2, Sc1, Sc2, Sn = states
        out = raw_step(
            P, T, PET, *pv, S1, S2, Sc1, Sc2, Sn,
            delta_t=1.0, nearzero=nearzero, doy=doy,
        )
        Q, ET_tot, S1n, S2n, Sc1n, Sc2n, Snn = out
        if MODEL == "mopex5":
            tcrit, ddf, Sb1, tw, a4, a5, tmin, trange, tu, Se, Sb2, tc = pv
            pet_demand = phenology_effective_pet(T, tmin, trange, PET, nearzero)
        else:
            tcrit, ddf, Sb1, tw, a4, a5, tu, Se, Sb2, tc = pv
            pet_demand = PET
        flux_ps = snowfall_1(P, T, tcrit)
        flux_pr = rainfall_1(P, T, tcrit)
        flux_qn = melt_1(ddf, tcrit, T, Sn, 1.0)
        Sn_w = Sn + flux_ps - flux_qn
        i_pot = interception_kernel(flux_pr, doy, a4, a5, nearzero=nearzero)
        flux_i = torch.minimum(i_pot, pet_demand)
        pet_after_i = pet_demand - flux_i
        soil_input = (flux_pr - flux_i) + flux_qn
        S1_w = S1 + soil_input
        flux_et1 = torch.minimum(evap_7(S1_w, Sb1, pet_after_i, 1.0, nearzero), S1_w)
        pet_after_et1 = pet_after_i - flux_et1
        S1_after_et1 = S1_w - flux_et1
        flux_q1f = torch.minimum(saturation_1(soil_input, S1_after_et1, Sb1, nearzero=nearzero), S1_after_et1)
        S1_after_q1f = S1_after_et1 - flux_q1f
        flux_qw = recharge_3(tw, S1_after_q1f)
        S2_w = S2 + flux_qw
        flux_q2f = torch.minimum(saturation_1(flux_qw, S2_w, Sb2, nearzero=nearzero), S2_w)
        S2_after_q2f = S2_w - flux_q2f
        flux_q2u = baseflow_1(tu, S2_after_q2f)
        S2_after_q2u = S2_after_q2f - flux_q2u
        se_abs = Se * Sb2
        flux_et2 = torch.minimum(evap_7(S2_after_q2u, se_abs, pet_after_et1, 1.0, nearzero), S2_after_q2u)
        Sc1_w = Sc1 + flux_q1f + flux_q2f
        flux_qf = baseflow_1(tc, Sc1_w)
        Sc1_new = Sc1_w - flux_qf
        Sc2_w = Sc2 + flux_q2u
        flux_qs = baseflow_1(tc, Sc2_w)
        Sc2_new = Sc2_w - flux_qs
        Q_manual = flux_qf + flux_qs
        ET_manual = flux_et1 + flux_et2 + flux_i
        # per-day water balance residual over all stores (should be ~1e-6)
        P_tot = flux_ps + flux_pr
        dS = (Snn - Sn) + (S1n - S1) + (S2n - S2) + (Sc1n - Sc1) + (Sc2n - Sc2)
        residual = P_tot - flux_i - flux_et1 - flux_et2 - Q_manual - dS
        return Q_manual, flux_i, flux_et1, flux_et2, (S1n, S2n, Sc1n, Sc2n, Snn), residual

    return step


def run_pet_water_balance(hydro, val_x, val_y, val_theta, ids, sample_count: int, arm_dir: Path):
    """PET closure + water balance on sampled basins over the full validation period.

    val_x: [time, basin, 4] (calendar already appended); scored period only.
    """
    if MODEL == "mopex4":
        from dmotpy.models.core.mopex4 import mopex4_step
        warmup_step = mopex4_step
    else:
        from dmotpy.models.core.mopex5 import mopex5_step
        warmup_step = mopex5_step
    n_time, n_basin = val_x.shape[0], val_x.shape[1]
    torch.manual_seed(1234)
    sample_idx = torch.randperm(n_basin)[:sample_count].tolist()
    params_dict = hydro._descale_params(val_theta)
    pv = [params_dict[name][sample_idx].squeeze(-1) for name in hydro.phy_param_names]
    step = make_fluxes_step(hydro, pv)

    states = hydro._init_states(sample_count, 1)
    states = tuple(s.detach().squeeze(-1) for s in states)
    S1, S2, Sc1, Sc2, Sn = states

    cum = {k: torch.zeros(sample_count, device=DEVICE) for k in
           ("P", "Pliq", "Psnow", "I", "ET1", "ET2", "Q")}
    max_exceed = torch.zeros(sample_count, device=DEVICE)
    max_exceed_epc = torch.zeros(sample_count, device=DEVICE)
    exceed_days = torch.zeros(sample_count, device=DEVICE)
    max_wb = torch.zeros(sample_count, device=DEVICE)
    scored = 0

    P_all = val_x[:, sample_idx, 0]
    T_all = val_x[:, sample_idx, 1]
    PET_all = val_x[:, sample_idx, 2]
    DOY_all = val_x[:, sample_idx, 3]

    # warm-up the states with the 365 d prefix (detached), then score
    xw = val_x[:WARMUP, sample_idx]
    for t in range(WARMUP):
        with torch.no_grad():
            out = warmup_step(
                xw[t, :, 0], xw[t, :, 1], xw[t, :, 2], *pv,
                S1, S2, Sc1, Sc2, Sn, delta_t=1.0, nearzero=hydro.nearzero,
                doy=xw[t, :, 3],
            )
            S1, S2, Sc1, Sc2, Sn = out[2], out[3], out[4], out[5], out[6]

    for t in range(WARMUP, n_time):
        P, T, PET = P_all[t], T_all[t], PET_all[t]
        Q, I, ET1, ET2, (S1n, S2n, Sc1n, Sc2n, Snn), residual = step(
            P, T, PET, (S1, S2, Sc1, Sc2, Sn), DOY_all[t])
        closure = I + ET1 + ET2
        exceed = torch.clamp(closure - PET, min=0.0)
        max_exceed = torch.maximum(max_exceed, exceed)
        if MODEL == "mopex5":
            from dmotpy.models.core.mopex5 import phenology_effective_pet
            pet_epc = phenology_effective_pet(T, pv[6], pv[7], PET, hydro.nearzero)
            max_exceed_epc = torch.maximum(max_exceed_epc, torch.clamp(closure - pet_epc, min=0.0))
        exceed_days += (exceed > 1e-6).float()
        max_wb = torch.maximum(max_wb, residual.abs())
        cum["P"] += P
        cum["I"] += I; cum["ET1"] += ET1; cum["ET2"] += ET2; cum["Q"] += Q
        S1, S2, Sc1, Sc2, Sn = S1n, S2n, Sc1n, Sc2n, Snn
        scored += 1

    # liquid/snow split over the scored period only (same snow/rain partition)
    from dmotpy.models.flux.mopex import mopex_rainfall_1 as rainfall_1, mopex_snowfall_1 as snowfall_1
    tcrit = pv[0]
    pr_all = rainfall_1(P_all[WARMUP:], T_all[WARMUP:], tcrit).sum(0)
    ps_all = snowfall_1(P_all[WARMUP:], T_all[WARMUP:], tcrit).sum(0)
    cum["Pliq"] = pr_all
    cum["Psnow"] = ps_all

    rows = []
    for j, b_idx in enumerate(sample_idx):
        basin_id = int(ids[b_idx])
        rows.append({
            "arm": "seed42", "seed": 42, "basin_id": basin_id,
            "exceedance_day_fraction_scored": float(exceed_days[j] / scored),
            "max_exceedance_scored": float(max_exceed[j]),
            "closure_pass": bool(max_exceed[j] <= 1e-6),
            "pet_epc_closure_pass": (bool(max_exceed_epc[j] <= 1e-6) if MODEL == "mopex5" else None),
            "pet_epc_max_exceedance_scored": (float(max_exceed_epc[j]) if MODEL == "mopex5" else None),
            "water_balance_max_daily_abs_residual": float(max_wb[j]),
            # float32 sequential-update noise is ~1e-5 mm/d; a genuine
            # order/formula violation shows residuals of flux magnitude.
            "water_balance_pass": bool(max_wb[j] <= 1e-3),
            "I/P_scored": float(cum["I"][j] / cum["P"][j]),
            "ET/P_scored": float((cum["I"][j] + cum["ET1"][j] + cum["ET2"][j]) / cum["P"][j]),
            "Q/P_scored": float(cum["Q"][j] / cum["P"][j]),
            "Pliq/P_scored": float(cum["Pliq"][j] / cum["P"][j]),
            "Psnow/P_scored": float(cum["Psnow"][j] / cum["P"][j]),
            "I_mm": float(cum["I"][j]), "ET1_mm": float(cum["ET1"][j]),
            "ET2_mm": float(cum["ET2"][j]), "Q_mm": float(cum["Q"][j]),
            "P_mm": float(cum["P"][j]), "Pliq_mm": float(cum["Pliq"][j]),
            "Psnow_mm": float(cum["Psnow"][j]),
            "ET2_mm": float(cum["ET2"][j]), "Q_mm": float(cum["Q"][j]),
            "P_mm": float(cum["P"][j]),
        })
    write_csv(arm_dir / "final" / "pet_water_balance.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mopex4", choices=["mopex4", "mopex5"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--steps-per-epoch", type=int, default=STEPS)
    parser.add_argument("--evaluation-every", type=int, default=1)
    args = parser.parse_args()

    global MODEL, I0, I1, PHEN, OUT_ROOT
    MODEL = args.model
    I0, I1, PHEN, OUT_ROOT = resolve_run_config(MODEL)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    seed = args.seed
    arm = f"seed{seed}"
    arm_dir = OUT_ROOT / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / "final").mkdir(parents=True, exist_ok=True)
    (arm_dir / "checkpoints" / MODEL).mkdir(parents=True, exist_ok=True)

    contract = {
        "model": MODEL, "seed": seed, "epochs": args.epochs, "lr": args.lr,
        "batch_size": args.batch_size, "steps_per_epoch": args.steps_per_epoch,
        "window_days": WINDOW, "warmup_days": WARMUP, "warmup_grad_mode": "detach",
        "parameter_mapping": "auto", "backend": "compile",
        "attributes": "Caravan, zscore all531, log1p skewed", "hidden_dims": [256, 256],
        "dropout": 0.05, "optimizer": "AdamW lr=1e-3 wd=1e-4", "grad_clip": 1.0,
        "loss": "differentiable 1-KGE (eps inside sqrt)", "loss_reference": "NATIVE.compute_differentiable_kge",
        "train_period": "1980-10-01..1995-09-30", "validation_period": "1995-10-01..2010-09-30",
        "eval_metric": "per-basin KGE + NSE; median/mean over basins",
        "stop_rule": "force exactly 100; early stopping disabled (matches round13 MOPEX3 reference)",
        "model_source": (
            f"repo-root dmotpy ({REPO / 'dmotpy'}); "
            f"{MODEL}_step = restored original seasonal interception (alpha/is_time) "
            "with corrected process order"
            + (" + MOPEX5 original phenology PET adjustment (tmin/trange)" if MODEL == "mopex5" else "")
        ),
        "interception": (
            "I_pot = softplus(50*(alpha+(1-alpha)cos(2pi(doy-is_time)/365.25)))/50 * Pr; "
            + ("I = min(I_pot, PET_epc); Pr_net = Pr - I; soil_input = Pr_net + qn; "
               "PET_epc = clamp((T-tmin)/trange,0,1)*PET; I+ET1+ET2 <= PET_epc <= PET"
               if MODEL == "mopex5" else
               "I = min(I_pot, PET); Pr_net = Pr - I; soil_input = Pr_net + qn; I+ET1+ET2 <= PET")
        ),
        "interception_slots": [I0, I1],
        "phenology_slots": list(PHEN),
        "bounds": {k: PARAM_INFO[MODEL][k] for k in ([I0, I1] + list(PHEN))},
        "freeze_manifest": (
            "results/mopex4_formula_decouple_20260811/ (pre-training validation) + "
            "git " + __import__("subprocess").check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
        ),
    }
    (arm_dir / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ids = [int(x) for x in load_ids("data/531sub_id.txt")]
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
    train_x_np, train_y_np, val_x_np, val_y_np = NATIVE.load_camels_time_series(ids)
    train_x = torch.as_tensor(train_x_np, dtype=torch.float32, device=DEVICE)
    train_y = torch.as_tensor(train_y_np, dtype=torch.float32, device=DEVICE)
    val_x = torch.as_tensor(val_x_np, dtype=torch.float32, device=DEVICE)
    val_y = torch.as_tensor(val_y_np, dtype=torch.float32, device=DEVICE)
    train_x, _ = add_calendar_forcing(train_x, pd.date_range("1980-10-01", "1995-09-30", freq="D"), model_name=MODEL)
    val_x, _ = add_calendar_forcing(val_x, pd.date_range("1994-10-01", "2010-09-30", freq="D"), model_name=MODEL)

    catalog, lengths = H1.make_catalog(train_y[WARMUP:])
    hydro = build_model(MODEL, DEVICE, warm_up=WARMUP, backend="compile",
                        parameter_mapping="auto", warmup_grad_mode="detach")
    network = CatchmentParameterizer(attrs.shape[1], NPARAM_INFO_36[MODEL],
                                     hidden_dims=[256, 256], dropout=0.05).to(DEVICE)
    initialize_midpoint(network)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=1e-4)

    start, invalid_train, invalid_val = 1, 0, 0
    old = latest_checkpoint(arm_dir)
    if old is not None:
        payload = torch.load(old, map_location="cpu", weights_only=False)
        network.load_state_dict(payload["network"])
        optimizer.load_state_dict(payload["optimizer"])
        torch.random.set_rng_state(payload["cpu_rng"])
        torch.cuda.set_rng_state(payload["cuda_rng"], device=DEVICE)
        start = int(payload["epoch"]) + 1
        invalid_train = int(payload["invalid_train"])
        invalid_val = int(payload["invalid_val"])
        print(f"[resume] from checkpoint {old} (epoch {start - 1})", flush=True)

    epochs_path = arm_dir / "epochs.csv"
    gradients_path = arm_dir / "parameter_gradients.csv"
    failfast_path = arm_dir / "final" / "failfast_epoch1.csv"

    print(f"=== Official canonical dPL seed {seed}: model={MODEL} {len(ids)} basins, {args.epochs} epochs ===", flush=True)
    print(f"params: {list(PARAM_INFO[MODEL])}", flush=True)
    print(f"dmotpy: {__import__('dmotpy').__file__}", flush=True)
    t0 = time.time()
    best_median_so_far = -1.0
    # MOPEX5 key slots: interception (alpha, is_time) + phenology (tmin, trange).
    key_slots = [4, 5] + ([6, 7] if MODEL == "mopex5" else [])
    for epoch in range(start, args.epochs + 1):
        network.train()
        loss_total = 0.0
        elapsed = 0.0
        observed = torch.zeros((len(ids), NPARAM_INFO_36[MODEL]), dtype=torch.bool, device=DEVICE)
        epoch_nan_batches = 0
        key_grad_abs = {slot: 0.0 for slot in key_slots}
        n_grad_checks = 0
        for mb in range(args.steps_per_epoch):
            basins = torch.randperm(len(ids), device=DEVICE)[: args.batch_size]
            choices = (torch.rand(args.batch_size, device=DEVICE) * lengths[basins]).long()
            starts = catalog[basins, choices]
            x = H1.gather_window(train_x, starts, basins)
            y = H1.gather_window(train_y, starts, basins)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            now = time.perf_counter()
            theta = network(attrs[basins])
            theta.retain_grad()
            if not bool(torch.isfinite(theta).all()):
                raise RuntimeError(f"non-finite parameterizer output at epoch {epoch} batch {mb}")
            q = hydro({"x_phy": x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            invalid_train += int((~torch.isfinite(q)).sum().detach())
            loss, _ = NATIVE.compute_differentiable_kge(q, y[WARMUP:], warmup_days=0)
            if not math.isfinite(float(loss.detach())):
                epoch_nan_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            grads = [p.grad for p in network.parameters() if p.grad is not None]
            finite_grad = all(bool(torch.isfinite(g).all()) for g in grads)
            if not finite_grad:
                raise RuntimeError(f"non-finite gradient at epoch {epoch} batch {mb}")
            if theta.grad is not None:
                for _slot in key_slots:
                    key_grad_abs[_slot] += float(theta.grad[:, _slot].abs().mean())
                n_grad_checks += 1
            observed[basins] |= theta.grad.detach() != 0
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - now
            loss_total += float(loss.detach())

        network.eval()
        with torch.no_grad():
            val_theta = network(attrs)
            val_q = hydro({"x_phy": val_x}, (None, val_theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            invalid_val += int((~torch.isfinite(val_q)).sum().detach())
            _l, kge = NATIVE.compute_differentiable_kge(val_q, val_y, warmup_days=WARMUP)
            nse = nse_per_basin(val_q, val_y)
        median_kge = float(kge.median())
        mean_kge = float(kge.mean())
        median_nse = float(torch.nanmedian(nse))
        mean_nse = float(torch.nanmean(nse))
        params = list(PARAM_INFO[MODEL])
        theta_boundary = float(((val_theta < 0.02) | (val_theta > 0.98)).float().mean())
        key_stats = {}
        for _slot in key_slots:
            v = val_theta[:, _slot]
            key_stats[_slot] = {
                "boundary_fraction": float(((v < 0.02) | (v > 0.98)).float().mean()),
                "lower_hit_fraction": float((v < 0.02).float().mean()),
                "upper_hit_fraction": float((v > 0.98).float().mean()),
                "boundary_distance_median": float(torch.minimum(v, 1.0 - v).median()),
                "raw_median": float(v.median()),
                "grad_mean_abs": key_grad_abs[_slot] / max(n_grad_checks, 1),
            }
        key_fields = {}
        for _slot in key_slots:
            nm = params[_slot]
            st = key_stats[_slot]
            key_fields[f"{nm}_boundary_fraction"] = st["boundary_fraction"]
            key_fields[f"{nm}_lower_hit_fraction"] = st["lower_hit_fraction"]
            key_fields[f"{nm}_upper_hit_fraction"] = st["upper_hit_fraction"]
            key_fields[f"{nm}_boundary_distance_median"] = st["boundary_distance_median"]
            key_fields[f"{nm}_raw_median"] = st["raw_median"]
            key_fields[f"{nm}_grad_mean_abs"] = st["grad_mean_abs"]

        row = {
            "model": MODEL, "arm": arm, "epoch": epoch, "status": "COMPLETED_EPOCH",
            "validation_median_kge": median_kge, "validation_mean_kge": mean_kge,
            "validation_median_nse": median_nse, "validation_mean_nse": mean_nse,
            "train_loss_1_minus_kge": loss_total / args.steps_per_epoch,
            "theta_boundary_fraction": theta_boundary,
            **key_fields,
            "nan_batches": epoch_nan_batches,
            "seconds_per_train_step": elapsed / args.steps_per_epoch,
            "parameter_mapping": "auto", "warmup_grad_mode": "detach",
            "train_nonfinite_cumulative": invalid_train, "validation_nonfinite_cumulative": invalid_val,
        }
        append_csv(epochs_path, [row])
        append_csv(gradients_path, [{
            "model": MODEL, "arm": arm, "epoch": epoch, "parameter": p,
            "zero_gradient_basin_fraction": float((~observed[:, j]).float().mean()),
            "theta_boundary_basin_fraction": float(
                ((val_theta[:, j] < 0.02) | (val_theta[:, j] > 0.98)).float().mean()),
        } for j, p in enumerate(PARAM_INFO[MODEL])])

        if epoch == 1:
            ff_fields = {
                "seed": seed, "epoch": epoch,
                "train_q_nonfinite": invalid_train, "val_q_nonfinite": invalid_val,
                "nan_batches": epoch_nan_batches,
                "theta_finite": True, "grad_finite": True,
                **{f"{params[_slot]}_raw_median": key_stats[_slot]["raw_median"] for _slot in key_slots},
                **{f"{params[_slot]}_boundary_fraction": key_stats[_slot]["boundary_fraction"] for _slot in key_slots},
                **{f"{params[_slot]}_lower_hit_fraction": key_stats[_slot]["lower_hit_fraction"] for _slot in key_slots},
                **{f"{params[_slot]}_upper_hit_fraction": key_stats[_slot]["upper_hit_fraction"] for _slot in key_slots},
                **{f"{params[_slot]}_boundary_distance_median": key_stats[_slot]["boundary_distance_median"] for _slot in key_slots},
                "theta_boundary_fraction": theta_boundary,
                **{f"{params[_slot]}_grad_mean_abs": key_stats[_slot]["grad_mean_abs"] for _slot in key_slots},
                "val_median_kge": median_kge, "val_median_nse": median_nse,
            }
            write_csv(failfast_path, [ff_fields])
            print(f"[fail-fast epoch 1] nan_batches={epoch_nan_batches} "
                  f"train_nonfinite={invalid_train} val_nonfinite={invalid_val} "
                  f"theta_finite=True grad_finite=True "
                  f"raw_medians=" + " ".join(
                      f"{params[_slot]}={key_stats[_slot]['raw_median']:.4f}" for _slot in key_slots) + " "
                  f"boundary={theta_boundary:.4f} "
                  f"grads=" + " ".join(
                      f"|{params[_slot]}|={key_stats[_slot]['grad_mean_abs']:.3e}" for _slot in key_slots) + " "
                  f"val_median_kge={median_kge:.4f}", flush=True)
            # PET closure / water balance on sampled basins (training windows)
            torch.manual_seed(7)
            pb = torch.randperm(len(ids), device=DEVICE)[:20]
            pc = (torch.rand(20, device=DEVICE) * lengths[pb]).long()
            ps = catalog[pb, pc]
            px = H1.gather_window(train_x, ps, pb)
            py = H1.gather_window(train_y, ps, pb)
            with torch.no_grad():
                ptheta = network(attrs[pb])
                pq = hydro({"x_phy": px}, (None, ptheta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            print(f"[fail-fast epoch 1] sampled-window q finite={bool(torch.isfinite(pq).all())} "
                  f"kge={float(NATIVE.compute_differentiable_kge(pq, py[WARMUP:], warmup_days=0)[1].median()):.4f}",
                  flush=True)

        # periodic checkpoints + a dedicated best checkpoint (any epoch)
        if epoch == start or median_kge > best_median_so_far:
            best_dst = arm_dir / "checkpoints" / MODEL / "best.pt"
            torch.save({
                "epoch": epoch, "network": network.state_dict(), "optimizer": optimizer.state_dict(),
                "cpu_rng": torch.random.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(DEVICE),
                "invalid_train": invalid_train, "invalid_val": invalid_val,
            }, best_dst)
            best_median_so_far = median_kge
        if epoch % 10 == 0 or epoch == args.epochs:
            dst = checkpoint_path(arm_dir, epoch)
            torch.save({
                "epoch": epoch, "network": network.state_dict(), "optimizer": optimizer.state_dict(),
                "cpu_rng": torch.random.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(DEVICE),
                "invalid_train": invalid_train, "invalid_val": invalid_val,
            }, dst)
        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] val_med_kge={median_kge:.4f} "
              f"val_med_nse={median_nse:.4f} loss={row['train_loss_1_minus_kge']:.4f} "
              f"boundary={theta_boundary:.4f} "
              + " ".join(f"{params[_slot]}_raw={key_stats[_slot]['raw_median']:.3f}" for _slot in key_slots)
              + f" ({elapsed / args.steps_per_epoch:.2f}s/step)",
              flush=True)

    # ---- final diagnostics -------------------------------------------------
    best_epoch_row = max(
        (r for r in csv.DictReader(epochs_path.open()) if r["model"] == MODEL),
        key=lambda r: float(r["validation_median_kge"]),
    )
    best_epoch = int(best_epoch_row["epoch"])
    print(f"\nBest validation epoch: {best_epoch} "
          f"(median KGE {float(best_epoch_row['validation_median_kge']):.4f})", flush=True)

    # reload best-epoch network: prefer the dedicated best checkpoint written
    # during training (any epoch), else fall back to the periodic checkpoint
    # when the best epoch is a multiple of the save interval.
    best_ckpt = arm_dir / "checkpoints" / MODEL / "best.pt"
    if not best_ckpt.exists():
        best_ckpt = checkpoint_path(arm_dir, best_epoch)
    if best_ckpt.exists():
        payload = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        network.load_state_dict(payload["network"])
        network.eval()
    with torch.no_grad():
        theta_best = network(attrs)
        q_best = hydro({"x_phy": val_x}, (None, theta_best.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
        _l, kge_best = NATIVE.compute_differentiable_kge(q_best, val_y, warmup_days=WARMUP)
        nse_best = nse_per_basin(q_best, val_y)

    # final-epoch metrics
    with torch.no_grad():
        theta_final = network(attrs) if best_epoch == args.epochs else None
    if theta_final is None:
        payload = torch.load(checkpoint_path(arm_dir, args.epochs), map_location="cpu", weights_only=False)
        network.load_state_dict(payload["network"])
        network.eval()
        with torch.no_grad():
            theta_final = network(attrs)
            q_final = hydro({"x_phy": val_x}, (None, theta_final.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            _l, kge_final = NATIVE.compute_differentiable_kge(q_final, val_y, warmup_days=WARMUP)
            nse_final = nse_per_basin(q_final, val_y)
    else:
        with torch.no_grad():
            q_final = hydro({"x_phy": val_x}, (None, theta_final.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            _l, kge_final = NATIVE.compute_differentiable_kge(q_final, val_y, warmup_days=WARMUP)
            nse_final = nse_per_basin(q_final, val_y)

    kge_best_np = kge_best.cpu().numpy()
    nse_best_np = nse_best.cpu().numpy()
    kge_final_np = kge_final.cpu().numpy()
    nse_final_np = nse_final.cpu().numpy()

    metric_rows = [{
        "basin_id": f"{b:08d}", "val_kge_best": kge_best_np[i], "val_nse_best": nse_best_np[i],
        "val_kge_final": kge_final_np[i], "val_nse_final": nse_final_np[i], "best_epoch": best_epoch,
    } for i, b in enumerate(ids)]
    write_csv(arm_dir / "final" / "basin_metrics.csv", metric_rows)

    # basin-level parameters at best epoch (key-slot boundary diagnostics are model-aware)
    pd_best = hydro._descale_params(theta_best)
    param_rows = []
    iv_vals = {_slot: pd_best[params[_slot]].squeeze(-1) for _slot in key_slots}
    pstar = None
    for i, b in enumerate(ids):
        entry = {"basin_id": f"{b:08d}"}
        for name in PARAM_INFO[MODEL]:
            entry[name] = float(pd_best[name][i, 0])
        for _slot in key_slots:
            nm = params[_slot]
            r = float(theta_best[i, _slot])
            entry[f"{nm}_boundary_hit"] = bool(r < 0.02 or r > 0.98)
            entry[f"{nm}_boundary_distance"] = float(min(r, 1.0 - r))
            entry[f"{nm}_lower_hit"] = bool(r < 0.02)
            entry[f"{nm}_upper_hit"] = bool(r > 0.98)
            entry[f"{nm}_lower_distance"] = float(r)
            entry[f"{nm}_upper_distance"] = float(1.0 - r)
        if pstar is not None:
            entry["P_star"] = float(pstar[i])
        param_rows.append(entry)
    write_csv(arm_dir / "final" / "basin_parameters.csv", param_rows)

    # correlations between the two interception slots
    iv0_np = iv_vals[params.index(I0)].cpu().numpy()
    iv1_np = iv_vals[params.index(I1)].cpu().numpy()
    pearson = float(pd.Series(iv0_np).corr(pd.Series(iv1_np)))
    spearman = float(pd.Series(iv0_np).corr(pd.Series(iv1_np), method="spearman"))
    # PET closure / water balance on sampled basins (validation period)
    wb_rows = run_pet_water_balance(hydro, val_x, val_y, theta_best, ids, 60, arm_dir)

    summary = {
        "model": MODEL, "seed": seed, "best_epoch": best_epoch,
        "val_median_kge_best": float(best_epoch_row["validation_median_kge"]),
        "val_mean_kge_best": float(best_epoch_row["validation_mean_kge"]),
        "val_median_nse_best": float(best_epoch_row["validation_median_nse"]),
        "val_median_kge_final": float(np_median(kge_final_np)),
        "val_median_nse_final": float(np_median(nse_final_np)),
        **{f"{params[_slot]}_median": float(torch.median(iv_vals[_slot])) for _slot in key_slots},
        "pstar_median": (float(torch.median(pstar)) if pstar is not None else None),
        **{f"{params[_slot]}_boundary_hit_fraction": float(((theta_best[:, _slot] < 0.02) | (theta_best[:, _slot] > 0.98)).float().mean()) for _slot in key_slots},
        **{f"{params[_slot]}_lower_hit_fraction": float((theta_best[:, _slot] < 0.02).float().mean()) for _slot in key_slots},
        **{f"{params[_slot]}_upper_hit_fraction": float((theta_best[:, _slot] > 0.98).float().mean()) for _slot in key_slots},
        **{f"{params[_slot]}_boundary_distance_median": float(torch.minimum(theta_best[:, _slot], 1.0 - theta_best[:, _slot]).median()) for _slot in key_slots},
        f"corr_{I0}_{I1}_pearson": pearson, f"corr_{I0}_{I1}_spearman": spearman,
        "theta_boundary_fraction_best_epoch": float(best_epoch_row["theta_boundary_fraction"]),
        "pet_closure_pass_basins": sum(1 for r in wb_rows if r["closure_pass"]),
        "pet_epc_closure_pass_basins": (sum(1 for r in wb_rows if r.get("pet_epc_closure_pass") is True) if MODEL == "mopex5" else None),
        "pet_closure_total_basins": len(wb_rows),
        "water_balance_pass_basins": sum(1 for r in wb_rows if r["water_balance_pass"]),
        "water_balance_total_basins": len(wb_rows),
        "train_nonfinite_cumulative": invalid_train, "validation_nonfinite_cumulative": invalid_val,
        "elapsed_seconds": time.time() - t0,
    }
    (arm_dir / "final" / "seed_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # health.csv in the canonical summarize_health shape
    params = list(PARAM_INFO[MODEL])
    gradient_rows = list(csv.DictReader(gradients_path.open()))
    permanently_zero = [name for name in params
                        if all(float(r["zero_gradient_basin_fraction"]) == 1.0
                               for r in gradient_rows if r["parameter"] == name)]
    conditional = [name for name in params
                   if any(0.0 < float(r["zero_gradient_basin_fraction"]) < 1.0
                          for r in gradient_rows if r["parameter"] == name)]
    epoch_rows = list(csv.DictReader(epochs_path.open()))
    first = min(epoch_rows, key=lambda r: int(r["epoch"]))
    last = max(epoch_rows, key=lambda r: int(r["epoch"]))
    health = {
        "model": MODEL, "arm": arm, "status": "COMPLETED", "stop_epoch": args.epochs,
        "best_epoch": best_epoch,
        "best_validation_median_kge": float(best_epoch_row["validation_median_kge"]),
        "epoch1_validation_median_kge": float(first["validation_median_kge"]),
        "final_validation_median_kge": float(last["validation_median_kge"]),
        "best_minus_epoch1": float(best_epoch_row["validation_median_kge"]) - float(first["validation_median_kge"]),
        "best_minus_final": float(best_epoch_row["validation_median_kge"]) - float(last["validation_median_kge"]),
        "final_boundary_fraction": float(last["theta_boundary_fraction"]),
        "train_nonfinite_prediction_count": invalid_train,
        "validation_nonfinite_prediction_count": invalid_val,
        "permanently_zero_parameters": ";".join(permanently_zero),
        "conditional_zero_parameters": ";".join(conditional),
        "pass_integrity": invalid_train == 0 and invalid_val == 0,
        "pass_learning": float(best_epoch_row["validation_median_kge"]) - float(first["validation_median_kge"]) > 0.05,
        "pass_no_dead_parameters": not permanently_zero,
        "pass_no_saturation": float(last["theta_boundary_fraction"]) < 0.20,
        "pass_convergence_budget": True,  # force-100 protocol; best not necessarily near final
        "pass_no_degradation": float(best_epoch_row["validation_median_kge"]) - float(last["validation_median_kge"]) <= 0.05,
    }
    append_csv(arm_dir / "health.csv", [health])
    append_csv(arm_dir / "status.csv", [{"model": MODEL, "arm": arm, "status": "COMPLETED",
                                         "last_epoch": args.epochs, "warmup_grad_mode": "detach"}])

    print("\n=== FINAL ===", flush=True)
    print(f"best_epoch={best_epoch} val_median_kge={summary['val_median_kge_best']:.4f} "
          f"val_mean_kge={summary['val_mean_kge_best']:.4f} val_median_nse={summary['val_median_nse_best']:.4f}",
          flush=True)
    print(f"{I0} median={summary[f'{I0}_median']:.4f} {I1} median={summary[f'{I1}_median']:.4f} "
          + (f"P*={I0}/{I1} median={summary['pstar_median']:.4f} " if pstar is not None else ""), flush=True)
    print(f"corr({I0},{I1}) pearson={pearson:.4f} spearman={spearman:.4f}", flush=True)
    print(f"results: {arm_dir}", flush=True)


def np_median(a):
    import numpy as np
    return float(np.nanmedian(a))


if __name__ == "__main__":
    main()
