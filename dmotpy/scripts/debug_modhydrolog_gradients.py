#!/usr/bin/env python3
"""Gradient trainability diagnostic for modhydrolog/m36 PyTorch step function.

Measures gradient flow through the 15 raw parameters, identifies dead gradients,
records hard-cap activation rates, and evaluates sensitivity to loss functions.

Outputs:
- validation_results/modhydrolog_debug/gradient_summary.csv
- validation_results/modhydrolog_debug/gradient_report.md
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.core.modhydrolog import (
    MODHYDROLOG_PARAMS_BOUNDS,
    create_initial_state,
)
from models.flux.evap import evap_1, evap_2
from models.flux.interception import interception_1
from models.flux.infiltration import infiltration_1, infiltration_2
from models.flux.interflow import interflow_1
from models.flux.recharge import recharge_1
from models.flux.depression import depression_1
from models.flux.exchange import exchange_1, exchange_3

OUTPUT_DIR = REPO_ROOT / "validation_results" / "modhydrolog_debug"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
NEARZERO = 1e-6

PARAM_NAMES = ["insc", "coeff", "sq", "smsc", "sub", "crak", "em",
               "dsc", "ads", "md", "vcond", "dlev", "k1", "k2", "k3"]
BOUNDS = MODHYDROLOG_PARAMS_BOUNDS


def _raw_to_physical(raw: torch.Tensor, name: str, mapping: str = "linear") -> torch.Tensor:
    """Convert raw [0,1] to physical parameter using the same mapping as HydrologyModel."""
    lo, hi = BOUNDS[name]
    return raw * (hi - lo) + lo


def _instrumented_step(
    P_t, T_t, PET_t,
    raw_params: dict[str, torch.Tensor],
    S1, S2, S3, S4, S5,
    mapping: str = "linear",
) -> tuple:
    """Run one step with instrumented intermediate values for cap analysis."""
    # Convert raw to physical
    p = {name: _raw_to_physical(raw_params[name], name, mapping) for name in PARAM_NAMES}

    caps = {}

    # S1: Evaporation
    flux_Ei_pot = evap_1(S1, PET_t)
    caps["flux_Ei_pot_gt_S1"] = (flux_Ei_pot > S1).float().mean().item()
    flux_Ei = torch.minimum(flux_Ei_pot, S1)
    S1 = S1 - flux_Ei

    # S1: Interception
    S1 = S1 + P_t
    flux_EXC_raw = interception_1(P_t, S1, p["insc"])
    flux_EXC = torch.minimum(flux_EXC_raw, S1)
    caps["flux_EXC_gt_S1"] = (flux_EXC_raw > S1).float().mean().item()
    S1 = S1 - flux_EXC

    # S2: Infiltration
    flux_INF_raw = infiltration_1(p["coeff"], p["sq"], S2, p["smsc"], flux_EXC)
    caps["flux_INF_pot_gt_EXC"] = (flux_INF_raw > flux_EXC).float().mean().item()
    flux_INF = torch.minimum(flux_INF_raw, flux_EXC)
    flux_RUN = flux_EXC - flux_INF

    # S2: Interflow
    flux_INT_raw = interflow_1(p["sub"], S2, p["smsc"], flux_INF)
    caps["flux_INT_pot_gt_INF"] = (flux_INT_raw > flux_INF).float().mean().item()
    flux_INT = torch.minimum(flux_INT_raw, flux_INF)

    # S2: Recharge
    remain = flux_INF - flux_INT
    flux_REC_raw = recharge_1(p["crak"], S2, p["smsc"], remain)
    caps["flux_REC_pot_gt_remain"] = (flux_REC_raw > remain).float().mean().item()
    flux_REC = torch.minimum(flux_REC_raw, remain)
    flux_SMF = remain - flux_REC

    # S3: Depression
    flux_TRAP = depression_1(p["ads"], p["md"], S3, p["dsc"], flux_RUN, NEARZERO)
    S3 = S3 + flux_TRAP
    flux_SRUN = flux_RUN - flux_TRAP

    # S3: Evaporation
    flux_Ed_pot = evap_1(S3, p["ads"] * PET_t)
    caps["flux_Ed_pot_gt_S3"] = (flux_Ed_pot > S3).float().mean().item()
    flux_Ed = torch.minimum(flux_Ed_pot, S3)
    S3 = S3 - flux_Ed

    # S3: Delayed Infiltration
    flux_DINF_raw = infiltration_2(p["coeff"], p["sq"], S2, p["smsc"], flux_SMF, S3)
    flux_DINF_pot = flux_DINF_raw * p["ads"]
    caps["flux_DINF_pot_x_ads_gt_S3"] = (flux_DINF_pot > S3).float().mean().item()
    flux_DINF = torch.minimum(flux_DINF_pot, S3)
    S3 = S3 - flux_DINF
    S3 = torch.clamp(S3, min=NEARZERO)

    # S2: Receive
    S2 = S2 + flux_SMF + flux_DINF

    # S2: Evaporation
    flux_Et_pot = evap_2(p["em"], S2, p["smsc"], PET_t)
    caps["flux_Et_pot_gt_S2"] = (flux_Et_pot > S2).float().mean().item()
    flux_Et = torch.minimum(flux_Et_pot, S2)
    S2 = S2 - flux_Et

    # S2: Saturation
    excess_s2 = torch.relu(S2 - p["smsc"])
    flux_GWF = excess_s2
    caps["S2_gt_smsc"] = (S2 > p["smsc"]).float().mean().item()
    S2 = S2 - flux_GWF
    S2 = torch.clamp(S2, min=NEARZERO)

    # S4/S5: Accumulate
    S4 = S4 + flux_REC + flux_GWF
    S5 = S5 + flux_SRUN + flux_INT

    # S4: Seepage
    flux_SEEP_pot = exchange_3(p["vcond"], S4, p["dlev"])
    caps["flux_SEEP_pot_gt_S4"] = (flux_SEEP_pot > S4).float().mean().item()
    flux_SEEP = torch.minimum(flux_SEEP_pot, S4)
    S4 = S4 - flux_SEEP

    # S4/S5: Exchange
    pot_flow = exchange_1(p["k1"], p["k2"], p["k3"], S4, flux_SRUN)
    flow_out = F.relu(pot_flow)
    flow_in = F.relu(-pot_flow)
    caps["flow_out_gt_S4"] = (flow_out > S4).float().mean().item()
    caps["flow_in_gt_S5"] = (flow_in > S5).float().mean().item()
    real_flow_out = torch.minimum(flow_out, S4)
    real_flow_in = torch.minimum(flow_in, S5)
    flux_FLOW = real_flow_out - real_flow_in
    S4 = S4 - flux_FLOW
    S5 = S5 + flux_FLOW
    S4 = torch.clamp(S4, min=NEARZERO)

    # S5: Baseflow
    flux_Q_pot = torch.tensor(1.0, device=S5.device) * S5
    caps["flux_Q_pot_gt_S5"] = (flux_Q_pot > S5).float().mean().item()
    flux_Q = torch.minimum(flux_Q_pot, S5)
    S5 = S5 - flux_Q
    S5 = torch.clamp(S5, min=NEARZERO)

    Ea = flux_Ei + flux_Et + flux_Ed + flux_SEEP

    return flux_Q, Ea, S1, S2, S3, S4, S5, caps


def _kge_loss(pred: torch.Tensor, obs: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    """1 - KGE loss."""
    mask = torch.isfinite(pred) & torch.isfinite(obs)
    p_sub = pred[mask]
    o_sub = obs[mask]
    if p_sub.numel() < 2:
        return torch.tensor(0.0, requires_grad=True)
    mean_p = p_sub.mean()
    mean_o = o_sub.mean()
    std_p = p_sub.std()
    std_o = o_sub.std()
    num = ((p_sub - mean_p) * (o_sub - mean_o)).sum()
    den = torch.sqrt(((p_sub - mean_p) ** 2).sum()) * torch.sqrt(
        ((o_sub - mean_o) ** 2).sum()
    )
    r = num / (den + eps)
    beta = mean_p / (mean_o + eps)
    gamma = std_p / (std_o + eps)
    kge = 1.0 - torch.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2)
    return 1.0 - kge


def _nse_loss(pred: torch.Tensor, obs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - NSE loss."""
    mask = torch.isfinite(pred) & torch.isfinite(obs)
    p_sub = pred[mask]
    o_sub = obs[mask]
    if p_sub.numel() < 2:
        return torch.tensor(0.0, requires_grad=True)
    mean_o = o_sub.mean()
    num = ((p_sub - o_sub) ** 2).sum()
    den = ((o_sub - mean_o) ** 2).sum()
    nse = 1.0 - num / (den + eps)
    return 1.0 - nse


def run_gradient_diagnostics(
    n_steps: int = 90,
    batch_size: int = 1,
    seed: int = 42,
) -> list[dict]:
    """Run gradient analysis for multiple loss functions and parameter seeds."""
    results = []

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate synthetic forcing
    rng = np.random.RandomState(seed)
    p_raw = rng.lognormal(mean=1.5, sigma=1.0, size=n_steps)
    p_raw = p_raw * (rng.rand(n_steps) < 0.35)
    p_seq = torch.tensor(p_raw, dtype=torch.float32).view(-1, 1, 1)
    t_seq = torch.full((n_steps, 1, 1), 15.0)
    pet_seq = torch.full((n_steps, 1, 1), 3.0)

    for loss_name, loss_fn in [("MSE", torch.nn.MSELoss()), ("KGE", _kge_loss), ("NSE", _nse_loss)]:
        for param_seed_idx in range(3):
            torch.manual_seed(seed + param_seed_idx)

            # Initialize raw parameters as learnable [0,1]
            raw_params = {}
            for name in PARAM_NAMES:
                lo, hi = BOUNDS[name]
                raw = torch.rand(batch_size, 1, requires_grad=True) * 0.3 + 0.35  # start in middle
                raw_params[name] = raw

            S1, S2, S3, S4, S5 = create_initial_state(batch_size, 1, DEVICE, NEARZERO)

            all_Q = []
            all_caps = {k: [] for k in [
                "flux_Ei_pot_gt_S1", "flux_EXC_gt_S1",
                "flux_INF_pot_gt_EXC", "flux_INT_pot_gt_INF",
                "flux_REC_pot_gt_remain", "flux_Ed_pot_gt_S3",
                "flux_DINF_pot_x_ads_gt_S3", "flux_Et_pot_gt_S2",
                "S2_gt_smsc", "flux_SEEP_pot_gt_S4",
                "flow_out_gt_S4", "flow_in_gt_S5",
                "flux_Q_pot_gt_S5",
            ]}

            for t in range(n_steps):
                Q_t, _, S1, S2, S3, S4, S5, caps = _instrumented_step(
                    p_seq[t], t_seq[t], pet_seq[t],
                    raw_params, S1, S2, S3, S4, S5, "linear",
                )
                all_Q.append(Q_t)
                for k, v in caps.items():
                    all_caps[k].append(v)

            Q_pred = torch.stack(all_Q).squeeze(-1).squeeze(-1)
            # Create synthetic target: mean-shifted Q with noise
            Q_obs = torch.abs(Q_pred + torch.randn_like(Q_pred) * 0.5 + 0.1)

            # Compute loss
            loss = loss_fn(Q_pred, Q_obs)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()

            # Collect gradient info
            for name in PARAM_NAMES:
                raw = raw_params[name]
                raw_grad = raw.grad.detach() if raw.grad is not None else None
                phys_val = _raw_to_physical(raw, name, "linear").detach()

                row = {
                    "loss_fn": loss_name,
                    "seed": param_seed_idx,
                    "param": name,
                    "raw_value": raw.item(),
                    "phys_value": phys_val.item(),
                    "lower_bound": BOUNDS[name][0],
                    "upper_bound": BOUNDS[name][1],
                    "near_lower": float(abs(phys_val.item() - BOUNDS[name][0]) < 1e-2 * (BOUNDS[name][1] - BOUNDS[name][0] + 1e-6)),
                    "near_upper": float(abs(phys_val.item() - BOUNDS[name][1]) < 1e-2 * (BOUNDS[name][1] - BOUNDS[name][0] + 1e-6)),
                    "raw_grad": raw_grad.item() if raw_grad is not None else 0.0,
                    "raw_grad_norm": abs(raw_grad.item()) if raw_grad is not None else 0.0,
                    "phys_grad_norm": abs(raw_grad.item()) * (BOUNDS[name][1] - BOUNDS[name][0]) if raw_grad is not None else 0.0,
                    "zero_grad": float(raw_grad is None or abs(raw_grad.item()) < 1e-15),
                    "nan_grad": float(raw_grad is not None and torch.isnan(raw_grad).any().item()),
                    "inf_grad": float(raw_grad is not None and torch.isinf(raw_grad).any().item()),
                    "loss_value": loss.item(),
                }
                results.append(row)

            # Cap statistics
            cap_stats = {}
            for k, vals in all_caps.items():
                cap_stats[f"cap_{k}_mean"] = float(np.mean(vals))
                cap_stats[f"cap_{k}_max"] = float(np.max(vals))

            for k, v in cap_stats.items():
                cap_stats_row = {
                    "loss_fn": loss_name,
                    "seed": param_seed_idx,
                    "param": "_CAPS_",
                    "raw_value": v,
                    "phys_value": v,
                    "lower_bound": 0,
                    "upper_bound": 0,
                    "near_lower": 0,
                    "near_upper": 0,
                    "raw_grad": 0,
                    "raw_grad_norm": 0,
                    "phys_grad_norm": 0,
                    "zero_grad": 0,
                    "nan_grad": 0,
                    "inf_grad": 0,
                    "loss_value": loss.item(),
                }
                cap_stats_row["test_case"] = k
                results.append(cap_stats_row)

    return results


def save_csv(results: list[dict], path: Path) -> None:
    if not results:
        return
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    keys = sorted(all_keys)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def build_report(results: list[dict]) -> str:
    lines = [
        "# Gradient Trainability Report: modhydrolog (m36)",
        "",
        "## 1. Parameter Gradient Summary",
        "",
        "| Param | Loss | Raw Grad | Phys Grad Norm | Phys Val | Near Bounds | Zero Grad? |",
        "|-------|------|----------|----------------|----------|-------------|-----------|",
    ]
    param_results = [r for r in results if r["param"] != "_CAPS_"]
    for r in param_results:
        near = "LOWER" if r["near_lower"] > 0.5 else ("UPPER" if r["near_upper"] > 0.5 else "no")
        zero = "YES" if r["zero_grad"] > 0.5 else "no"
        lines.append(
            f"| {r['param']} | {r['loss_fn']} | {r['raw_grad']:.2e} | "
            f"{r['phys_grad_norm']:.2e} | {r['phys_value']:.4f} | "
            f"{near} | {zero} |"
        )

    # Stats by param
    lines += ["", "## 2. Per-Parameter Gradient Statistics (mean over seeds/losses)", ""]
    lines.append("| Param | Mean Raw Grad | Mean Phys Grad Norm | ZeroGrad% | NaN% | Inf% |")
    lines.append("|-------|---------------|---------------------|-----------|------|------|")

    for name in PARAM_NAMES:
        subset = [r for r in param_results if r["param"] == name]
        if not subset:
            continue
        mean_rg = np.mean([abs(r["raw_grad"]) for r in subset])
        mean_pg = np.mean([r["phys_grad_norm"] for r in subset])
        zero_pct = np.mean([r["zero_grad"] for r in subset])
        nan_pct = np.mean([r["nan_grad"] for r in subset])
        inf_pct = np.mean([r["inf_grad"] for r in subset])
        lines.append(
            f"| {name} | {mean_rg:.2e} | {mean_pg:.2e} | "
            f"{zero_pct:.0%} | {nan_pct:.0%} | {inf_pct:.0%} |"
        )

    # Cap statistics
    cap_results = [r for r in results if r.get("test_case")]
    if cap_results:
        lines += ["", "## 3. Hard Cap Activation Rates", ""]
        lines.append("| Cap | Mean Activation | Max Activation |")
        lines.append("|-----|----------------|----------------|")
        for cr in cap_results:
            lines.append(
                f"| {cr['test_case']} | {cr['raw_value']:.3%} | {cr['phys_value']:.3%} |"
            )

    # Key findings
    zero_grad_params = []
    for name in PARAM_NAMES:
        subset = [r for r in param_results if r["param"] == name]
        if subset and np.mean([r["zero_grad"] for r in subset]) > 0.8:
            zero_grad_params.append(name)

    lines += [
        "",
        "## 4. Key Findings",
        "",
        f"Parameters with >80% zero gradient: {zero_grad_params or 'None'}",
        f"Total loss values range: {min(r['loss_value'] for r in param_results):.4f} to {max(r['loss_value'] for r in param_results):.4f}",
    ]

    return "\n".join(lines)


def main():
    print("=== Modhydrolog Gradient Trainability Diagnostic ===")
    results = run_gradient_diagnostics(n_steps=90)

    csv_path = OUTPUT_DIR / "gradient_summary.csv"
    md_path = OUTPUT_DIR / "gradient_report.md"

    save_csv(results, csv_path)
    report = build_report(results)
    md_path.write_text(report, encoding="utf-8")
    print(f"CSV saved to {csv_path}")
    print(f"Report saved to {md_path}")

    # Quick summary
    param_results = [r for r in results if r["param"] != "_CAPS_"]
    for name in PARAM_NAMES[:5]:
        subset = [r for r in param_results if r["param"] == name]
        if subset:
            grads = [r["raw_grad"] for r in subset]
            print(f"  {name}: grad = {np.mean([abs(g) for g in grads]):.2e} "
                  f"(zero={np.mean([r['zero_grad'] for r in subset]):.0%})")


if __name__ == "__main__":
    main()
