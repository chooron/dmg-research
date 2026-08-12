#!/usr/bin/env python3
"""Mass balance diagnostic for modhydrolog/m36 PyTorch step function.

Evaluates water budget closure, flux/storage anomalies, and boundary violations
using synthetic forcing data (and optionally CAMELS data).

Outputs:
- validation_results/modhydrolog_debug/mass_balance_summary.csv
- validation_results/modhydrolog_debug/mass_balance_report.md
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.core.modhydrolog import (
    MODHYDROLOG_PARAMS_BOUNDS,
    create_initial_state,
    modhydrolog_step,
)

OUTPUT_DIR = REPO_ROOT / "validation_results" / "modhydrolog_debug"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
NEARZERO = 1e-6


def _named_bounds() -> dict[str, list[float]]:
    return MODHYDROLOG_PARAMS_BOUNDS


def _sample_params(bounds: dict, batch_size: int = 1) -> dict[str, torch.Tensor]:
    """Sample random physical parameters within bounds."""
    params = {}
    for name, (lo, hi) in bounds.items():
        val = lo + (hi - lo) * torch.rand(batch_size, 1)
        params[name] = val
    return params


def _median_params(bounds: dict, batch_size: int = 1) -> dict[str, torch.Tensor]:
    """Mid-point parameters."""
    params = {}
    for name, (lo, hi) in bounds.items():
        params[name] = torch.full((batch_size, 1), (lo + hi) / 2)
    return params


def _generate_synthetic_forcing(
    n_steps: int = 365,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic P, T, PET sequences."""
    rng = np.random.RandomState(seed)
    # Log-normal rainfall with dry days
    p = rng.lognormal(mean=1.5, sigma=1.0, size=n_steps)
    wet_mask = rng.rand(n_steps) < 0.35
    p = p * wet_mask
    # Temperature: sinusoidal seasonal + noise
    t = 15 + 10 * np.sin(2 * np.pi * np.arange(n_steps) / 365) + rng.randn(n_steps) * 3
    # PET: sinusoidal seasonal + noise, positive only
    pet = 3 + 2 * np.sin(2 * np.pi * np.arange(n_steps) / 365 + 1.5) + rng.rand(n_steps) * 1
    pet = np.maximum(pet, 0.1)
    return (
        torch.tensor(p, dtype=torch.float32).view(-1, 1, 1),
        torch.tensor(t, dtype=torch.float32).view(-1, 1, 1),
        torch.tensor(pet, dtype=torch.float32).view(-1, 1, 1),
    )


def _load_camels_forcing(
    basin_idx: int = 0,
    n_steps: int = 365,
    start_step: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Try to load CAMELS forcing data."""
    data_path = REPO_ROOT.parent / "data" / "camels_dataset_petv2_hargreaves.npz"
    if not data_path.exists():
        return None
    data = np.load(str(data_path), allow_pickle=True)
    forcing = data["forcing"]
    end = min(start_step + n_steps, forcing.shape[0])
    p = forcing[start_step:end, basin_idx, 0:1]
    t = forcing[start_step:end, basin_idx, 1:2]
    pet = forcing[start_step:end, basin_idx, 2:3]
    return (
        torch.tensor(p, dtype=torch.float32),
        torch.tensor(t, dtype=torch.float32),
        torch.tensor(pet, dtype=torch.float32),
    )


def _forward_with_states(
    p_seq, t_seq, pet_seq, params, n_steps, batch_size=1
) -> dict:
    """Forward the step function tracking all intermediate states and fluxes."""
    S1, S2, S3, S4, S5 = create_initial_state(batch_size, 1, DEVICE, NEARZERO)

    records = {
        "P": [], "Q": [], "Ea": [],
        "S1": [], "S2": [], "S3": [], "S4": [], "S5": [],
    }

    for t_step in range(n_steps):
        S1_old = S1.clone()
        S2_old = S2.clone()
        S3_old = S3.clone()
        S4_old = S4.clone()
        S5_old = S5.clone()

        P_t = p_seq[t_step : t_step + 1] if p_seq.dim() == 3 else p_seq[t_step]
        T_t = t_seq[t_step : t_step + 1] if t_seq.dim() == 3 else t_seq[t_step]
        PET_t = pet_seq[t_step : t_step + 1] if pet_seq.dim() == 3 else pet_seq[t_step]

        Q, Ea, S1, S2, S3, S4, S5 = modhydrolog_step(
            P_t, T_t, PET_t,
            params["insc"], params["coeff"], params["sq"], params["smsc"],
            params["sub"], params["crak"], params["em"], params["dsc"],
            params["ads"], params["md"], params["vcond"], params["dlev"],
            params["k1"], params["k2"], params["k3"],
            S1, S2, S3, S4, S5, NEARZERO,
        )

        records["P"].append(P_t.item() if isinstance(P_t, torch.Tensor) else P_t)
        records["Q"].append(Q.item())
        records["Ea"].append(Ea.item())
        records["S1"].append((S1 - S1_old).item())
        records["S2"].append((S2 - S2_old).item())
        records["S3"].append((S3 - S3_old).item())
        records["S4"].append((S4 - S4_old).item())
        records["S5"].append((S5 - S5_old).item())

    return {k: np.array(v) for k, v in records.items()}


def _forward_detailed(
    p_seq, t_seq, pet_seq, params, n_steps, batch_size=1
) -> dict:
    """Forward with detailed flux tracking via instrumented step."""
    S1, S2, S3, S4, S5 = create_initial_state(batch_size, 1, DEVICE, NEARZERO)

    # Import flux functions for instrumentation
    from models.flux.evap import evap_1, evap_2
    from models.flux.interception import interception_1
    from models.flux.infiltration import infiltration_1, infiltration_2
    from models.flux.interflow import interflow_1
    from models.flux.recharge import recharge_1
    from models.flux.depression import depression_1
    from models.flux.exchange import exchange_1, exchange_3

    records = {
        "flux_Ei": [], "flux_Ei_pot": [], "flux_EXC": [],
        "flux_INF": [], "flux_INF_pot": [],
        "flux_INT": [], "flux_INT_pot": [],
        "flux_REC": [], "flux_REC_pot": [],
        "flux_SMF": [], "flux_RUN": [],
        "flux_TRAP": [], "flux_Ed": [], "flux_Ed_pot": [],
        "flux_DINF": [], "flux_DINF_pot": [],
        "flux_GWF": [],
        "flux_SEEP": [], "flux_SEEP_pot": [],
        "flux_SRUN": [],
        "flux_FLOW": [], "flow_out": [], "flow_in": [], "pot_flow": [],
        "flux_Q": [], "flux_Q_pot": [],
        "S1_before": [], "S2_before": [], "S3_before": [],
        "S4_before": [], "S5_before": [],
        "S1_after": [], "S2_after": [], "S3_after": [],
        "S4_after": [], "S5_after": [],
        "P_t": [], "Ea_t": [],
    }

    param_names = ["insc", "coeff", "sq", "smsc", "sub", "crak", "em",
                   "dsc", "ads", "md", "vcond", "dlev", "k1", "k2", "k3"]

    for t_step in range(n_steps):
        P_t = p_seq[t_step]
        T_t = t_seq[t_step]
        PET_t = pet_seq[t_step]

        records["S1_before"].append(S1.item())
        records["S2_before"].append(S2.item())
        records["S3_before"].append(S3.item())
        records["S4_before"].append(S4.item())
        records["S5_before"].append(S5.item())
        records["P_t"].append(P_t.item())

        # S1: Evaporation
        flux_Ei_pot = evap_1(S1, PET_t)
        flux_Ei = torch.minimum(flux_Ei_pot, S1)
        S1 = S1 - flux_Ei

        # S1: Interception
        S1 = S1 + P_t
        flux_EXC_raw = interception_1(P_t, S1, params["insc"])
        flux_EXC = torch.minimum(flux_EXC_raw, S1)
        S1 = S1 - flux_EXC

        # S2: Infiltration
        flux_INF_raw = infiltration_1(params["coeff"], params["sq"], S2,
                                       params["smsc"], flux_EXC)
        flux_INF = torch.minimum(flux_INF_raw, flux_EXC)
        flux_RUN = flux_EXC - flux_INF

        # S2: Interflow
        flux_INT_raw = interflow_1(params["sub"], S2, params["smsc"], flux_INF)
        flux_INT = torch.minimum(flux_INT_raw, flux_INF)

        # S2: Recharge
        remain = flux_INF - flux_INT
        flux_REC_raw = recharge_1(params["crak"], S2, params["smsc"], remain)
        flux_REC = torch.minimum(flux_REC_raw, remain)
        flux_SMF = remain - flux_REC

        # S3: Depression
        flux_TRAP = depression_1(params["ads"], params["md"], S3,
                                  params["dsc"], flux_RUN, NEARZERO)
        S3 = S3 + flux_TRAP
        flux_SRUN = flux_RUN - flux_TRAP

        # S3: Evaporation
        flux_Ed_pot = evap_1(S3, params["ads"] * PET_t)
        flux_Ed = torch.minimum(flux_Ed_pot, S3)
        S3 = S3 - flux_Ed

        # S3: Delayed Infiltration
        flux_DINF_raw = infiltration_2(params["coeff"], params["sq"], S2,
                                        params["smsc"], flux_SMF, S3)
        flux_DINF = flux_DINF_raw * params["ads"]
        flux_DINF = torch.minimum(flux_DINF, S3)
        S3 = S3 - flux_DINF
        S3 = torch.clamp(S3, min=NEARZERO)

        # S2: Receive
        S2 = S2 + flux_SMF + flux_DINF

        # S2: Evaporation
        flux_Et_pot = evap_2(params["em"], S2, params["smsc"], PET_t)
        flux_Et = torch.minimum(flux_Et_pot, S2)
        S2 = S2 - flux_Et

        # S2: Saturation
        excess_s2 = torch.relu(S2 - params["smsc"])
        flux_GWF = excess_s2
        S2 = S2 - flux_GWF
        S2 = torch.clamp(S2, min=NEARZERO)

        # S4/S5: Accumulate
        S4 = S4 + flux_REC + flux_GWF
        S5 = S5 + flux_SRUN + flux_INT

        # S4: Seepage
        flux_SEEP_pot = exchange_3(params["vcond"], S4, params["dlev"])
        flux_SEEP = torch.minimum(flux_SEEP_pot, S4)
        S4 = S4 - flux_SEEP

        # S4/S5: Exchange
        pot_flow = exchange_1(params["k1"], params["k2"], params["k3"],
                               S4, flux_SRUN)
        flow_out = torch.relu(pot_flow)
        flow_in = torch.relu(-pot_flow)
        real_flow_out = torch.minimum(flow_out, S4)
        real_flow_in = torch.minimum(flow_in, S5)
        flux_FLOW = real_flow_out - real_flow_in
        S4 = S4 - flux_FLOW
        S5 = S5 + flux_FLOW
        S4 = torch.clamp(S4, min=NEARZERO)

        # S5: Baseflow
        flux_Q_pot = torch.tensor(1.0) * S5
        flux_Q = torch.minimum(flux_Q_pot, S5)
        S5 = S5 - flux_Q
        S5 = torch.clamp(S5, min=NEARZERO)

        Ea = flux_Ei + flux_Et + flux_Ed + flux_SEEP

        records["flux_Ei"].append(flux_Ei.item())
        records["flux_Ei_pot"].append(flux_Ei_pot.item())
        records["flux_EXC"].append(flux_EXC.item())
        records["flux_INF"].append(flux_INF.item())
        records["flux_INF_pot"].append(flux_INF_raw.item())
        records["flux_INT"].append(flux_INT.item())
        records["flux_INT_pot"].append(flux_INT_raw.item())
        records["flux_REC"].append(flux_REC.item())
        records["flux_REC_pot"].append(flux_REC_raw.item())
        records["flux_SMF"].append(flux_SMF.item())
        records["flux_RUN"].append(flux_RUN.item())
        records["flux_TRAP"].append(flux_TRAP.item())
        records["flux_Ed"].append(flux_Ed.item())
        records["flux_Ed_pot"].append(flux_Ed_pot.item())
        records["flux_DINF"].append(flux_DINF.item())
        records["flux_DINF_pot"].append(flux_DINF_raw.item())
        records["flux_GWF"].append(flux_GWF.item())
        records["flux_SEEP"].append(flux_SEEP.item())
        records["flux_SEEP_pot"].append(flux_SEEP_pot.item())
        records["flux_SRUN"].append(flux_SRUN.item())
        records["flux_FLOW"].append(flux_FLOW.item())
        records["flow_out"].append(real_flow_out.item())
        records["flow_in"].append(real_flow_in.item())
        records["pot_flow"].append(pot_flow.item())
        records["flux_Q"].append(flux_Q.item())
        records["flux_Q_pot"].append(flux_Q_pot.item())
        records["flux_Et"] = records.get("flux_Et", [])
        records["flux_Et"].append(flux_Et.item())
        records["Ea_t"].append(Ea.item())
        records["S1_after"].append(S1.item())
        records["S2_after"].append(S2.item())
        records["S3_after"].append(S3.item())
        records["S4_after"].append(S4.item())
        records["S5_after"].append(S5.item())

    return {k: np.array(v) for k, v in records.items()}


def compute_checks(rec: dict) -> dict:
    """Compute all balance checks from the recorded data."""
    n = len(rec["P_t"])
    checks = {}

    # --- water balance ---
    p_total = rec["P_t"].sum()
    q_total = rec["flux_Q"].sum()
    ea_total = rec["Ea_t"].sum()
    ds1 = rec["S1_after"][-1] - rec["S1_before"][0]
    ds2 = rec["S2_after"][-1] - rec["S2_before"][0]
    ds3 = rec["S3_after"][-1] - rec["S3_before"][0]
    ds4 = rec["S4_after"][-1] - rec["S4_before"][0]
    ds5 = rec["S5_after"][-1] - rec["S5_before"][0]
    delta_s = ds1 + ds2 + ds3 + ds4 + ds5
    residual = p_total - q_total - ea_total - delta_s

    checks["P_total"] = p_total
    checks["Q_total"] = q_total
    checks["Ea_total"] = ea_total
    checks["delta_S"] = delta_s
    checks["residual_full_period"] = residual
    checks["residual_full_relative"] = residual / (p_total + 1e-6)

    # Per-timestep residual
    residuals = np.zeros(n)
    for i in range(n):
        p_i = rec["P_t"][i]
        q_i = rec["flux_Q"][i]
        ea_i = rec["Ea_t"][i]
        ds_i = sum(rec[f"S{k}_after"][i] - rec[f"S{k}_before"][i] for k in range(1, 6))
        residuals[i] = p_i - q_i - ea_i - ds_i
    checks["residual_mean"] = float(np.mean(residuals))
    checks["residual_max_abs"] = float(np.max(np.abs(residuals)))
    checks["residual_p95_abs"] = float(np.percentile(np.abs(residuals), 95))

    # --- state anomalies ---
    for k in range(1, 6):
        s_name = f"S{k}"
        before = rec[f"{s_name}_before"]
        after = rec[f"{s_name}_after"]
        for label, arr in [("before", before), ("after", after)]:
            checks[f"{s_name}_{label}_neg"] = float(np.sum(arr < 0))
            checks[f"{s_name}_{label}_inf"] = float(np.sum(np.isinf(arr)))
            checks[f"{s_name}_{label}_nan"] = float(np.sum(np.isnan(arr)))
            checks[f"{s_name}_{label}_min"] = float(np.min(arr))
            checks[f"{s_name}_{label}_max"] = float(np.max(arr))

    # --- flux anomalies ---
    flux_names = [
        "flux_Ei", "flux_EXC", "flux_INF", "flux_INT", "flux_REC",
        "flux_SMF", "flux_Et", "flux_GWF", "flux_RUN", "flux_TRAP",
        "flux_Ed", "flux_DINF", "flux_SEEP", "flux_SRUN", "flux_Q",
    ]
    for fname in flux_names:
        arr = rec.get(fname, np.array([0]))
        checks[f"{fname}_neg"] = float(np.sum(arr < -1e-8))
        checks[f"{fname}_inf"] = float(np.sum(np.isinf(arr)))
        checks[f"{fname}_nan"] = float(np.sum(np.isnan(arr)))
        checks[f"{fname}_min"] = float(np.min(arr))
        checks[f"{fname}_max"] = float(np.max(arr))
        checks[f"{fname}_mean"] = float(np.mean(arr))

    # --- Special checks ---
    seep_pot = rec.get("flux_SEEP_pot", np.array([0]))
    seep = rec.get("flux_SEEP", np.array([0]))
    checks["SEEP_pot_neg_ratio"] = float(np.mean(seep_pot < -1e-8))
    checks["SEEP_neg_ratio"] = float(np.mean(seep < -1e-8))

    trap = rec.get("flux_TRAP", np.array([0]))
    run = rec.get("flux_RUN", np.array([0]))
    checks["TRAP_gt_RUN_ratio"] = float(np.mean(trap > run + 1e-8))

    inf = rec.get("flux_INF", np.array([0]))
    intf = rec.get("flux_INT", np.array([0]))
    recf = rec.get("flux_REC", np.array([0]))
    dinf = rec.get("flux_DINF", np.array([0]))
    srun = rec.get("flux_SRUN", np.array([0]))
    checks["INF_neg_ratio"] = float(np.mean(inf < -1e-8))
    checks["INT_neg_ratio"] = float(np.mean(intf < -1e-8))
    checks["REC_neg_ratio"] = float(np.mean(recf < -1e-8))
    checks["DINF_neg_ratio"] = float(np.mean(dinf < -1e-8))
    checks["SRUN_neg_ratio"] = float(np.mean(srun < -1e-8))

    # S5 drainage check
    s5_before = rec.get("S5_before", np.array([0]))
    s5_after = rec.get("S5_after", np.array([0]))
    flux_q = rec.get("flux_Q", np.array([0]))
    checks["S5_almost_emptied_ratio"] = float(np.mean(s5_after < 1e-5))
    checks["S5_mean_before_baseflow"] = float(np.mean(s5_before))
    checks["S5_mean_after"] = float(np.mean(s5_after))
    checks["Q_vs_S5_before_ratio"] = float(np.mean(flux_q / (s5_before + 1e-8)))

    return checks


def run_diagnostics(bounds: dict, n_steps: int = 365) -> list[dict]:
    """Run mass balance diagnostics across multiple parameter sets."""
    results = []

    # Test 1: Median params with synthetic forcing
    p_seq, t_seq, pet_seq = _generate_synthetic_forcing(n_steps, seed=42)
    for case_name, param_fn in [
        ("median_params", _median_params),
        ("random_1", _sample_params),
        ("random_2", _sample_params),
    ]:
        params = param_fn(bounds, batch_size=1)
        rec = _forward_detailed(p_seq, t_seq, pet_seq, params, n_steps)
        checks = compute_checks(rec)
        checks["test_case"] = f"synthetic_{case_name}"
        checks["n_steps"] = n_steps
        results.append(checks)
        print(f"  Completed: synthetic_{case_name}")

    # Test 2: Try CAMELS if available
    camels = _load_camels_forcing(basin_idx=0, n_steps=min(n_steps, 730))
    if camels is not None:
        p_c, t_c, pet_c = camels
        for case_name, param_fn in [
            ("median_params", _median_params),
            ("random_1", _sample_params),
            ("random_2", _sample_params),
        ]:
            n_c = min(p_c.shape[0], n_steps)
            params = param_fn(bounds, batch_size=1)
            rec = _forward_detailed(p_c[:n_c], t_c[:n_c], pet_c[:n_c],
                                     params, n_c)
            checks = compute_checks(rec)
            checks["test_case"] = f"camels_basin0_{case_name}"
            checks["n_steps"] = n_c
            results.append(checks)
            print(f"  Completed: camels_basin0_{case_name}")

    # Test 3: Extreme wet forcing (heavy rain every day)
    p_wet = torch.full((n_steps, 1, 1), 50.0)
    t_wet = torch.full((n_steps, 1, 1), 20.0)
    pet_wet = torch.full((n_steps, 1, 1), 1.0)
    params = _median_params(bounds, batch_size=1)
    rec = _forward_detailed(p_wet, t_wet, pet_wet, params, n_steps)
    checks = compute_checks(rec)
    checks["test_case"] = "synthetic_wet"
    checks["n_steps"] = n_steps
    results.append(checks)
    print(f"  Completed: synthetic_wet")

    # Test 4: Extreme dry forcing (no rain, high PET)
    p_dry = torch.zeros(n_steps, 1, 1)
    t_dry = torch.full((n_steps, 1, 1), 25.0)
    pet_dry = torch.full((n_steps, 1, 1), 8.0)
    rec = _forward_detailed(p_dry, t_dry, pet_dry, params, n_steps)
    checks = compute_checks(rec)
    checks["test_case"] = "synthetic_dry"
    checks["n_steps"] = n_steps
    results.append(checks)
    print(f"  Completed: synthetic_dry")

    return results


def save_csv(results: list[dict], path: Path) -> None:
    """Save results to CSV."""
    if not results:
        return
    keys = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV saved to {path}")


def build_report(results: list[dict]) -> str:
    """Build markdown report from results."""
    lines = [
        "# Mass Balance Diagnostic Report: modhydrolog (m36)",
        "",
        f"**Number of test cases**: {len(results)}",
        "",
        "## 1. Water Balance Closure",
        "",
        "| Test Case | P_total | Q_total | Ea_total | delta_S | Residual | Residual% |",
        "|-----------|---------|---------|----------|---------|----------|-----------|",
    ]
    for r in results:
        lines.append(
            f"| {r['test_case']} | {r['P_total']:.3f} | {r['Q_total']:.3f} | "
            f"{r['Ea_total']:.3f} | {r['delta_S']:.3f} | "
            f"{r['residual_full_period']:.3f} | {r['residual_full_relative']:.3%} |"
        )

    lines += [
        "",
        "| Test Case | Residual Mean | Residual Max Abs | Residual P95 Abs |",
        "|-----------|---------------|------------------|------------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['test_case']} | {r['residual_mean']:.6f} | "
            f"{r['residual_max_abs']:.6f} | {r['residual_p95_abs']:.6f} |"
        )

    lines += [
        "",
        "## 2. State Anomalies",
        "",
        "| Test Case | Store | Min (after) | Neg count | Inf count | NaN count |",
        "|-----------|-------|-------------|-----------|-----------|-----------|",
    ]
    for r in results:
        for k in range(1, 6):
            lines.append(
                f"| {r['test_case']} | S{k} | {r[f'S{k}_after_min']:.3f} | "
                f"{r[f'S{k}_after_neg']:.0f} | {r[f'S{k}_after_inf']:.0f} | "
                f"{r[f'S{k}_after_nan']:.0f} |"
            )

    lines += [
        "",
        "## 3. Flux Anomalies",
        "",
        "| Test Case | Flux | Min | Mean | Neg Ratio | Inf | NaN |",
        "|-----------|------|-----|------|-----------|-----|-----|",
    ]
    flux_names = [
        "flux_Ei", "flux_EXC", "flux_INF", "flux_INT", "flux_REC",
        "flux_SMF", "flux_Et", "flux_GWF", "flux_RUN", "flux_TRAP",
        "flux_Ed", "flux_DINF", "flux_SEEP", "flux_SRUN", "flux_Q",
    ]
    for r in results:
        for fn in flux_names:
            lines.append(
                f"| {r['test_case']} | {fn} | {r[f'{fn}_min']:.3f} | "
                f"{r[f'{fn}_mean']:.3f} | {r[f'{fn}_neg']:.3f} | "
                f"{r[f'{fn}_inf']:.0f} | {r[f'{fn}_nan']:.0f} |"
            )

    lines += [
        "",
        "## 4. Special Checks",
        "",
        "| Test Case | SEEP_pot<0 | SEEP<0 | TRAP>RUN | INF<0 | INT<0 | REC<0 | DINF<0 | SRUN<0 |",
        "|-----------|-----------|--------|----------|-------|-------|-------|--------|--------|",
    ]
    for r in results:
        lines.append(
            f"| {r['test_case']} | {r['SEEP_pot_neg_ratio']:.3f} | "
            f"{r['SEEP_neg_ratio']:.3f} | {r['TRAP_gt_RUN_ratio']:.3f} | "
            f"{r['INF_neg_ratio']:.3f} | {r['INT_neg_ratio']:.3f} | "
            f"{r['REC_neg_ratio']:.3f} | {r['DINF_neg_ratio']:.3f} | "
            f"{r['SRUN_neg_ratio']:.3f} |"
        )

    lines += [
        "",
        "## 5. S5 / Channel Store Behavior",
        "",
        "| Test Case | S5 Mean Before Q | S5 Mean After | Empty Ratio | Q/S5_before |",
        "|-----------|-----------------|---------------|-------------|--------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['test_case']} | {r['S5_mean_before_baseflow']:.4f} | "
            f"{r['S5_mean_after']:.4f} | {r['S5_almost_emptied_ratio']:.3f} | "
            f"{r['Q_vs_S5_before_ratio']:.3f} |"
        )

    lines += [
        "",
        "## 6. Key Findings",
        "",
        f"1. **Water Balance Residual**: Mean residual = {np.mean([r['residual_mean'] for r in results]):.6f}, "
        f"Max = {max(r['residual_max_abs'] for r in results):.6f}",
        f"2. **Flux SEEP negativity**: {np.mean([r['SEEP_neg_ratio'] for r in results]):.3%} of steps have negative seepage "
        f"({np.mean([r['SEEP_pot_neg_ratio'] for r in results]):.3%} potential negative)",
        f"3. **S5 always emptied**: {np.mean([r['S5_almost_emptied_ratio'] for r in results]):.3%} of steps have S5 near zero after baseflow",
        f"4. **TRAP > RUN**: {np.mean([r['TRAP_gt_RUN_ratio'] for r in results]):.3%} of steps have trap exceeding runoff",
        f"5. **Ea negativity**: Check if any test has Ea_total < 0",
    ]

    return "\n".join(lines)


def main():
    print("=== Modhydrolog Mass Balance Diagnostic ===")
    bounds = _named_bounds()
    print(f"Parameter count: {len(bounds)}")
    print(f"Parameters: {list(bounds.keys())}")

    results = run_diagnostics(bounds, n_steps=365)

    csv_path = OUTPUT_DIR / "mass_balance_summary.csv"
    md_path = OUTPUT_DIR / "mass_balance_report.md"

    save_csv(results, csv_path)
    report = build_report(results)
    md_path.write_text(report, encoding="utf-8")
    print(f"Report saved to {md_path}")

    # Print key findings
    print("\n=== Quick Findings ===")
    for r in results:
        print(f"\n[{r['test_case']}]")
        print(f"  P={r['P_total']:.1f} Q={r['Q_total']:.1f} Ea={r['Ea_total']:.1f} "
              f"dS={r['delta_S']:.3f} resid={r['residual_full_period']:.3f}")
        print(f"  SEEP_pot<0: {r['SEEP_pot_neg_ratio']:.1%}  SEEP<0: {r['SEEP_neg_ratio']:.1%}")
        print(f"  S5 emptied: {r['S5_almost_emptied_ratio']:.1%}")
        print(f"  TRAP>RUN: {r['TRAP_gt_RUN_ratio']:.1%}")


if __name__ == "__main__":
    main()
