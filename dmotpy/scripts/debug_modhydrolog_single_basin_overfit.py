#!/usr/bin/env python3
"""Single-basin overfit smoke test + Parameter sensitivity for modhydrolog/m36.

Runs:
A) Current raw step
B) Fix interception_1 gate direction
C) Fix flux_SEEP clamp to [0,S4]
D) Fix B+C combined
E) Fix depression_1 to MATLAB formula (if alignment confirms it)
F) Parameter sensitivity via finite differences

Does NOT modify source files; inline debug variants only.

Outputs:
- validation_results/modhydrolog_debug/single_basin_overfit.csv
- validation_results/modhydrolog_debug/single_basin_overfit_report.md
- validation_results/modhydrolog_debug/parameter_sensitivity.csv
- validation_results/modhydrolog_debug/parameter_sensitivity_report.md
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.flux.evap import evap_1, evap_2
from models.flux.interception import interception_1
from models.flux.infiltration import infiltration_1, infiltration_2
from models.flux.interflow import interflow_1
from models.flux.recharge import recharge_1
from models.flux.depression import depression_1
from models.flux.exchange import exchange_1, exchange_3
from models.flux.smooth import soft_gate_storage_above
from models.core.modhydrolog import (
    MODHYDROLOG_PARAMS_BOUNDS,
    create_initial_state,
)

OUTPUT_DIR = REPO_ROOT / "validation_results" / "modhydrolog_debug"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
NEARZERO = 1e-6
PARAM_NAMES = list(MODHYDROLOG_PARAMS_BOUNDS.keys())
BOUNDS = MODHYDROLOG_PARAMS_BOUNDS

# ============================================================================
# Debug variant step functions (inline, no source modifications)
# ============================================================================

def _raw_to_phys(raw: torch.Tensor, lo: float, hi: float):
    """Scale raw [0,1] to physical [lo, hi]."""
    return raw * (hi - lo) + lo


def _variant_original(P_t, T_t, PET_t, raw_p, S1, S2, S3, S4, S5):
    """Variant A: Original step (replicated)"""
    p = {n: _raw_to_phys(raw_p[n], *BOUNDS[n]) for n in PARAM_NAMES}
    nz = NEARZERO

    flux_Ei = torch.minimum(evap_1(S1, PET_t), S1)
    S1 = S1 - flux_Ei
    S1 = S1 + P_t
    flux_EXC = torch.minimum(interception_1(P_t, S1, p["insc"]), S1)
    S1 = S1 - flux_EXC

    flux_INF = torch.minimum(infiltration_1(p["coeff"], p["sq"], S2, p["smsc"], flux_EXC), flux_EXC)
    flux_RUN = flux_EXC - flux_INF
    flux_INT = torch.minimum(interflow_1(p["sub"], S2, p["smsc"], flux_INF), flux_INF)
    remain = flux_INF - flux_INT
    flux_REC = torch.minimum(recharge_1(p["crak"], S2, p["smsc"], remain), remain)
    flux_SMF = remain - flux_REC

    flux_TRAP = depression_1(p["ads"], p["md"], S3, p["dsc"], flux_RUN, nz)
    S3 = S3 + flux_TRAP
    flux_SRUN = flux_RUN - flux_TRAP
    flux_Ed = torch.minimum(evap_1(S3, p["ads"] * PET_t), S3)
    S3 = S3 - flux_Ed
    flux_DINF_raw = infiltration_2(p["coeff"], p["sq"], S2, p["smsc"], flux_SMF, S3)
    flux_DINF = torch.minimum(flux_DINF_raw * p["ads"], S3)
    S3 = S3 - flux_DINF
    S3 = torch.clamp(S3, min=nz)

    S2 = S2 + flux_SMF + flux_DINF
    flux_Et = torch.minimum(evap_2(p["em"], S2, p["smsc"], PET_t), S2)
    S2 = S2 - flux_Et
    flux_GWF = torch.relu(S2 - p["smsc"])
    S2 = S2 - flux_GWF
    S2 = torch.clamp(S2, min=nz)

    S4 = S4 + flux_REC + flux_GWF
    S5 = S5 + flux_SRUN + flux_INT
    flux_SEEP = torch.minimum(exchange_3(p["vcond"], S4, p["dlev"]), S4)
    S4 = S4 - flux_SEEP

    pot_flow = exchange_1(p["k1"], p["k2"], p["k3"], S4, flux_SRUN)
    flow_out = F.relu(pot_flow)
    flow_in = F.relu(-pot_flow)
    flux_FLOW = torch.minimum(flow_out, S4) - torch.minimum(flow_in, S5)
    S4 = S4 - flux_FLOW
    S5 = S5 + flux_FLOW
    S4 = torch.clamp(S4, min=nz)

    flux_Q = torch.minimum(torch.ones_like(S5) * S5, S5)
    S5 = S5 - flux_Q
    S5 = torch.clamp(S5, min=nz)

    Ea = flux_Ei + flux_Et + flux_Ed + flux_SEEP
    return flux_Q, Ea, S1, S2, S3, S4, S5


def _variant_fix_interception(P_t, T_t, PET_t, raw_p, S1, S2, S3, S4, S5):
    """Variant B: Fix interception_1 gate (use sat_above instead of sat_below)"""
    p = {n: _raw_to_phys(raw_p[n], *BOUNDS[n]) for n in PARAM_NAMES}
    nz = NEARZERO

    flux_Ei = torch.minimum(evap_1(S1, PET_t), S1)
    S1 = S1 - flux_Ei
    S1 = S1 + P_t
    # FIX: use soft_gate_storage_above for interception excess
    flux_EXC_raw = P_t * soft_gate_storage_above(S1, p["insc"])
    flux_EXC = torch.minimum(flux_EXC_raw, S1)
    S1 = S1 - flux_EXC

    flux_INF = torch.minimum(infiltration_1(p["coeff"], p["sq"], S2, p["smsc"], flux_EXC), flux_EXC)
    flux_RUN = flux_EXC - flux_INF
    flux_INT = torch.minimum(interflow_1(p["sub"], S2, p["smsc"], flux_INF), flux_INF)
    remain = flux_INF - flux_INT
    flux_REC = torch.minimum(recharge_1(p["crak"], S2, p["smsc"], remain), remain)
    flux_SMF = remain - flux_REC

    flux_TRAP = depression_1(p["ads"], p["md"], S3, p["dsc"], flux_RUN, nz)
    S3 = S3 + flux_TRAP
    flux_SRUN = flux_RUN - flux_TRAP
    flux_Ed = torch.minimum(evap_1(S3, p["ads"] * PET_t), S3)
    S3 = S3 - flux_Ed
    flux_DINF_raw = infiltration_2(p["coeff"], p["sq"], S2, p["smsc"], flux_SMF, S3)
    flux_DINF = torch.minimum(flux_DINF_raw * p["ads"], S3)
    S3 = S3 - flux_DINF
    S3 = torch.clamp(S3, min=nz)

    S2 = S2 + flux_SMF + flux_DINF
    flux_Et = torch.minimum(evap_2(p["em"], S2, p["smsc"], PET_t), S2)
    S2 = S2 - flux_Et
    flux_GWF = torch.relu(S2 - p["smsc"])
    S2 = S2 - flux_GWF
    S2 = torch.clamp(S2, min=nz)

    S4 = S4 + flux_REC + flux_GWF
    S5 = S5 + flux_SRUN + flux_INT
    flux_SEEP = torch.minimum(exchange_3(p["vcond"], S4, p["dlev"]), S4)
    S4 = S4 - flux_SEEP

    pot_flow = exchange_1(p["k1"], p["k2"], p["k3"], S4, flux_SRUN)
    flow_out = F.relu(pot_flow)
    flow_in = F.relu(-pot_flow)
    flux_FLOW = torch.minimum(flow_out, S4) - torch.minimum(flow_in, S5)
    S4 = S4 - flux_FLOW
    S5 = S5 + flux_FLOW
    S4 = torch.clamp(S4, min=nz)

    flux_Q = torch.minimum(torch.ones_like(S5) * S5, S5)
    S5 = S5 - flux_Q
    S5 = torch.clamp(S5, min=nz)

    Ea = flux_Ei + flux_Et + flux_Ed + flux_SEEP
    return flux_Q, Ea, S1, S2, S3, S4, S5


def _variant_fix_seep(P_t, T_t, PET_t, raw_p, S1, S2, S3, S4, S5):
    """Variant C: Fix SEEP clamp to [0, S4] and remove from Ea"""
    p = {n: _raw_to_phys(raw_p[n], *BOUNDS[n]) for n in PARAM_NAMES}
    nz = NEARZERO

    flux_Ei = torch.minimum(evap_1(S1, PET_t), S1)
    S1 = S1 - flux_Ei
    S1 = S1 + P_t
    flux_EXC = torch.minimum(interception_1(P_t, S1, p["insc"]), S1)
    S1 = S1 - flux_EXC

    flux_INF = torch.minimum(infiltration_1(p["coeff"], p["sq"], S2, p["smsc"], flux_EXC), flux_EXC)
    flux_RUN = flux_EXC - flux_INF
    flux_INT = torch.minimum(interflow_1(p["sub"], S2, p["smsc"], flux_INF), flux_INF)
    remain = flux_INF - flux_INT
    flux_REC = torch.minimum(recharge_1(p["crak"], S2, p["smsc"], remain), remain)
    flux_SMF = remain - flux_REC

    flux_TRAP = depression_1(p["ads"], p["md"], S3, p["dsc"], flux_RUN, nz)
    S3 = S3 + flux_TRAP
    flux_SRUN = flux_RUN - flux_TRAP
    flux_Ed = torch.minimum(evap_1(S3, p["ads"] * PET_t), S3)
    S3 = S3 - flux_Ed
    flux_DINF_raw = infiltration_2(p["coeff"], p["sq"], S2, p["smsc"], flux_SMF, S3)
    flux_DINF = torch.minimum(flux_DINF_raw * p["ads"], S3)
    S3 = S3 - flux_DINF
    S3 = torch.clamp(S3, min=nz)

    S2 = S2 + flux_SMF + flux_DINF
    flux_Et = torch.minimum(evap_2(p["em"], S2, p["smsc"], PET_t), S2)
    S2 = S2 - flux_Et
    flux_GWF = torch.relu(S2 - p["smsc"])
    S2 = S2 - flux_GWF
    S2 = torch.clamp(S2, min=nz)

    S4 = S4 + flux_REC + flux_GWF
    S5 = S5 + flux_SRUN + flux_INT
    # FIX: clamp seepage to [0, S4] (no inflow from deep aquifer)
    flux_SEEP_pot = exchange_3(p["vcond"], S4, p["dlev"])
    flux_SEEP = torch.clamp(flux_SEEP_pot, min=0.0, max=S4.item() if S4.numel() == 1 else None)
    flux_SEEP = torch.minimum(flux_SEEP, S4)
    S4 = S4 - flux_SEEP

    pot_flow = exchange_1(p["k1"], p["k2"], p["k3"], S4, flux_SRUN)
    flow_out = F.relu(pot_flow)
    flow_in = F.relu(-pot_flow)
    flux_FLOW = torch.minimum(flow_out, S4) - torch.minimum(flow_in, S5)
    S4 = S4 - flux_FLOW
    S5 = S5 + flux_FLOW
    S4 = torch.clamp(S4, min=nz)

    flux_Q = torch.minimum(torch.ones_like(S5) * S5, S5)
    S5 = S5 - flux_Q
    S5 = torch.clamp(S5, min=nz)

    # FIX: SEEP not in Ea
    Ea = flux_Ei + flux_Et + flux_Ed
    return flux_Q, Ea, S1, S2, S3, S4, S5


def _variant_fix_both(P_t, T_t, PET_t, raw_p, S1, S2, S3, S4, S5):
    """Variant D: Fix B + C combined (interception gate + SEEP clamp)"""
    p = {n: _raw_to_phys(raw_p[n], *BOUNDS[n]) for n in PARAM_NAMES}
    nz = NEARZERO

    flux_Ei = torch.minimum(evap_1(S1, PET_t), S1)
    S1 = S1 - flux_Ei
    S1 = S1 + P_t
    # FIX B: use soft_gate_storage_above for interception excess
    flux_EXC_raw = P_t * soft_gate_storage_above(S1, p["insc"])
    flux_EXC = torch.minimum(flux_EXC_raw, S1)
    S1 = S1 - flux_EXC

    flux_INF = torch.minimum(infiltration_1(p["coeff"], p["sq"], S2, p["smsc"], flux_EXC), flux_EXC)
    flux_RUN = flux_EXC - flux_INF
    flux_INT = torch.minimum(interflow_1(p["sub"], S2, p["smsc"], flux_INF), flux_INF)
    remain = flux_INF - flux_INT
    flux_REC = torch.minimum(recharge_1(p["crak"], S2, p["smsc"], remain), remain)
    flux_SMF = remain - flux_REC

    flux_TRAP = depression_1(p["ads"], p["md"], S3, p["dsc"], flux_RUN, nz)
    S3 = S3 + flux_TRAP
    flux_SRUN = flux_RUN - flux_TRAP
    flux_Ed = torch.minimum(evap_1(S3, p["ads"] * PET_t), S3)
    S3 = S3 - flux_Ed
    flux_DINF_raw = infiltration_2(p["coeff"], p["sq"], S2, p["smsc"], flux_SMF, S3)
    flux_DINF = torch.minimum(flux_DINF_raw * p["ads"], S3)
    S3 = S3 - flux_DINF
    S3 = torch.clamp(S3, min=nz)

    S2 = S2 + flux_SMF + flux_DINF
    flux_Et = torch.minimum(evap_2(p["em"], S2, p["smsc"], PET_t), S2)
    S2 = S2 - flux_Et
    flux_GWF = torch.relu(S2 - p["smsc"])
    S2 = S2 - flux_GWF
    S2 = torch.clamp(S2, min=nz)

    S4 = S4 + flux_REC + flux_GWF
    S5 = S5 + flux_SRUN + flux_INT
    # FIX C: clamp seepage to [0, S4]
    flux_SEEP_pot = exchange_3(p["vcond"], S4, p["dlev"])
    flux_SEEP = torch.clamp(flux_SEEP_pot, min=0.0)
    flux_SEEP = torch.minimum(flux_SEEP, S4)
    S4 = S4 - flux_SEEP

    pot_flow = exchange_1(p["k1"], p["k2"], p["k3"], S4, flux_SRUN)
    flow_out = F.relu(pot_flow)
    flow_in = F.relu(-pot_flow)
    flux_FLOW = torch.minimum(flow_out, S4) - torch.minimum(flow_in, S5)
    S4 = S4 - flux_FLOW
    S5 = S5 + flux_FLOW
    S4 = torch.clamp(S4, min=nz)

    flux_Q = torch.minimum(torch.ones_like(S5) * S5, S5)
    S5 = S5 - flux_Q
    S5 = torch.clamp(S5, min=nz)

    Ea = flux_Ei + flux_Et + flux_Ed
    return flux_Q, Ea, S1, S2, S3, S4, S5


def _variant_fix_matlab_depression(P_t, T_t, PET_t, raw_p, S1, S2, S3, S4, S5):
    """Variant E: Fix D + depression_1 to MATLAB formula"""
    p = {n: _raw_to_phys(raw_p[n], *BOUNDS[n]) for n in PARAM_NAMES}
    nz = NEARZERO

    flux_Ei = torch.minimum(evap_1(S1, PET_t), S1)
    S1 = S1 - flux_Ei
    S1 = S1 + P_t
    flux_EXC_raw = P_t * soft_gate_storage_above(S1, p["insc"])
    flux_EXC = torch.minimum(flux_EXC_raw, S1)
    S1 = S1 - flux_EXC

    flux_INF = torch.minimum(infiltration_1(p["coeff"], p["sq"], S2, p["smsc"], flux_EXC), flux_EXC)
    flux_RUN = flux_EXC - flux_INF
    flux_INT = torch.minimum(interflow_1(p["sub"], S2, p["smsc"], flux_INF), flux_INF)
    remain = flux_INF - flux_INT
    flux_REC = torch.minimum(recharge_1(p["crak"], S2, p["smsc"], remain), remain)
    flux_SMF = remain - flux_REC

    # MATLAB depression_1: min(ads * exp(-md * S3/max(dsc-S3, eps)) * RUN, max(dsc-S3, 0))
    cap = F.relu(p["dsc"] - S3)
    denom = torch.clamp(cap, min=nz)
    exponent = -p["md"] * S3 / denom
    exponent = torch.clamp(exponent, min=-20.0, max=0.0)
    flux_TRAP_matlab = p["ads"] * torch.exp(exponent) * flux_RUN
    flux_TRAP = torch.minimum(torch.minimum(flux_TRAP_matlab, cap), flux_RUN)
    S3 = S3 + flux_TRAP
    flux_SRUN = flux_RUN - flux_TRAP

    flux_Ed = torch.minimum(evap_1(S3, p["ads"] * PET_t), S3)
    S3 = S3 - flux_Ed
    flux_DINF_raw = infiltration_2(p["coeff"], p["sq"], S2, p["smsc"], flux_SMF, S3)
    flux_DINF = torch.minimum(flux_DINF_raw * p["ads"], S3)
    S3 = S3 - flux_DINF
    S3 = torch.clamp(S3, min=nz)

    S2 = S2 + flux_SMF + flux_DINF
    flux_Et = torch.minimum(evap_2(p["em"], S2, p["smsc"], PET_t), S2)
    S2 = S2 - flux_Et
    flux_GWF = torch.relu(S2 - p["smsc"])
    S2 = S2 - flux_GWF
    S2 = torch.clamp(S2, min=nz)

    S4 = S4 + flux_REC + flux_GWF
    S5 = S5 + flux_SRUN + flux_INT
    flux_SEEP_pot = exchange_3(p["vcond"], S4, p["dlev"])
    flux_SEEP = torch.clamp(flux_SEEP_pot, min=0.0)
    flux_SEEP = torch.minimum(flux_SEEP, S4)
    S4 = S4 - flux_SEEP

    pot_flow = exchange_1(p["k1"], p["k2"], p["k3"], S4, flux_SRUN)
    flow_out = F.relu(pot_flow)
    flow_in = F.relu(-pot_flow)
    flux_FLOW = torch.minimum(flow_out, S4) - torch.minimum(flow_in, S5)
    S4 = S4 - flux_FLOW
    S5 = S5 + flux_FLOW
    S4 = torch.clamp(S4, min=nz)

    flux_Q = torch.minimum(torch.ones_like(S5) * S5, S5)
    S5 = S5 - flux_Q
    S5 = torch.clamp(S5, min=nz)

    Ea = flux_Ei + flux_Et + flux_Ed
    return flux_Q, Ea, S1, S2, S3, S4, S5


VARIANTS = {
    "A_original": _variant_original,
    "B_fix_interception": _variant_fix_interception,
    "C_fix_seep": _variant_fix_seep,
    "D_fix_both": _variant_fix_both,
    "E_fix_all_matlab_depression": _variant_fix_matlab_depression,
}


# ============================================================================
# Overfit Test
# ============================================================================

def _generate_synthetic_forcing_short(
    n_steps: int = 365, seed: int = 42,
) -> tuple:
    rng = np.random.RandomState(seed)
    p = rng.lognormal(mean=1.5, sigma=1.0, size=n_steps) * (rng.rand(n_steps) < 0.4)
    pet = np.maximum(3 + 2 * np.sin(2 * np.pi * np.arange(n_steps) / 365 + 1.5) + rng.randn(n_steps), 0.1)
    return (
        torch.tensor(p, dtype=torch.float32).view(-1, 1, 1),
        torch.full((n_steps, 1, 1), 15.0),
        torch.tensor(pet, dtype=torch.float32).view(-1, 1, 1),
    )


def _nse_loss(pred, obs, eps=1e-6):
    den = ((obs - obs.mean()) ** 2).sum() + eps
    num = ((pred - obs) ** 2).sum()
    return num / (den + eps)


def run_overfit(
    n_steps: int = 365,
    n_epochs: int = 200,
    lr: float = 0.01,
    seed: int = 42,
) -> list[dict]:
    """Run overfit test for each variant."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    p_seq, t_seq, pet_seq = _generate_synthetic_forcing_short(n_steps, seed)

    # Generate synthetic target Q that the model should learn
    rng = np.random.RandomState(seed + 999)
    fake_q = rng.lognormal(mean=0.5, sigma=0.8, size=n_steps) * (rng.rand(n_steps) < 0.5)
    Q_target = torch.tensor(fake_q, dtype=torch.float32).view(-1, 1, 1)
    Q_target.clamp_(min=0.0)

    results = []

    for variant_name, step_fn in VARIANTS.items():
        # Initialize raw parameters as leaf tensors in a list (for optimizer)
        raw_list = []
        raw_params = {}
        for name in PARAM_NAMES:
            raw = torch.rand(1, 1) * 0.3 + 0.35
            raw.requires_grad_(True)
            raw_list.append(raw)
            raw_params[name] = raw

        optimizer = Adam(raw_list, lr=lr)
        loss_history = []

        for epoch in range(n_epochs):
            # Re-dereference to ensure leaf status is maintained
            raw_params = {name: raw_list[i] for i, name in enumerate(PARAM_NAMES)}

            S1, S2, S3, S4, S5 = create_initial_state(1, 1, DEVICE, NEARZERO)
            all_Q = []

            for t in range(n_steps):
                Q_t, _, S1, S2, S3, S4, S5 = step_fn(
                    p_seq[t], t_seq[t], pet_seq[t],
                    raw_params, S1, S2, S3, S4, S5,
                )
                all_Q.append(Q_t)

            Q_pred = torch.stack(all_Q).squeeze(-1).squeeze(-1)
            Q_obs = Q_target.squeeze(-1).squeeze(-1)

            loss = _nse_loss(Q_pred, Q_obs)
            if torch.isnan(loss) or torch.isinf(loss):
                loss_history.append(float("nan"))
                break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_list, 1.0)
            optimizer.step()

            # Clamp raw to [0,1]
            with torch.no_grad():
                for raw_t in raw_list:
                    raw_t.clamp_(0.0, 1.0)

            loss_history.append(loss.item())

            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"    [{variant_name}] epoch {epoch}: loss={loss.item():.4f}", flush=True)

        # Final forward to get Q stats
        with torch.no_grad():
            raw_params = {name: raw_list[i] for i, name in enumerate(PARAM_NAMES)}
            S1, S2, S3, S4, S5 = create_initial_state(1, 1, DEVICE, NEARZERO)
            all_Q = []
            for t in range(n_steps):
                Q_t, _, S1, S2, S3, S4, S5 = step_fn(
                    p_seq[t], t_seq[t], pet_seq[t],
                    raw_params, S1, S2, S3, S4, S5,
                )
                all_Q.append(Q_t.item())
            Q_final = np.array(all_Q)

        results.append({
            "variant": variant_name,
            "initial_loss": loss_history[0] if loss_history else np.nan,
            "final_loss": loss_history[-1] if loss_history else np.nan,
            "min_loss": min(loss_history) if loss_history else np.nan,
            "Q_mean": float(np.mean(Q_final)),
            "Q_std": float(np.std(Q_final)),
            "Q_obs_mean": float(Q_target.mean().item()),
            "Q_obs_std": float(Q_target.std().item()),
            "n_epochs": n_epochs,
            "lr": lr,
            f"param_sq": raw_list[PARAM_NAMES.index("sq")].item(),
            f"param_coeff": raw_list[PARAM_NAMES.index("coeff")].item(),
            f"param_sub": raw_list[PARAM_NAMES.index("sub")].item(),
            f"param_smsc": raw_list[PARAM_NAMES.index("smsc")].item(),
        })

        print(f"  [{variant_name}] init_loss={loss_history[0] if loss_history else np.nan:.4f} "
              f"final_loss={loss_history[-1] if loss_history else np.nan:.4f} "
              f"Q_mean={np.mean(Q_final):.4f} Q_std={np.std(Q_final):.4f}")

    return results


# ============================================================================
# Parameter Sensitivity
# ============================================================================

def _forward_window(step_fn, p_seq, t_seq, pet_seq, raw_params, n_steps):
    """Forward a full window and return Q series."""
    S1, S2, S3, S4, S5 = create_initial_state(1, 1, DEVICE, NEARZERO)
    all_Q = []
    for t in range(n_steps):
        Q_t, _, S1, S2, S3, S4, S5 = step_fn(
            p_seq[t], t_seq[t], pet_seq[t],
            raw_params, S1, S2, S3, S4, S5,
        )
        all_Q.append(Q_t.item())
    return np.array(all_Q)


def run_sensitivity(
    n_steps: int = 180,
    epsilon: float = 0.01,
    seed: int = 42,
) -> list[dict]:
    """Finite difference sensitivity for each parameter."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    rng = np.random.RandomState(seed)
    p = rng.lognormal(mean=1.5, sigma=1.0, size=n_steps) * (rng.rand(n_steps) < 0.35)
    p_seq = torch.tensor(p, dtype=torch.float32).view(-1, 1, 1)
    t_seq = torch.full((n_steps, 1, 1), 15.0)
    pet_seq = torch.full((n_steps, 1, 1), 3.0)

    # Base params
    base_raw = {n: torch.tensor([[0.5]], dtype=torch.float32) for n in PARAM_NAMES}

    # Use variant D (both fixes) for sensitivity analysis
    step_fn = _variant_fix_both

    Q_base = _forward_window(step_fn, p_seq, t_seq, pet_seq, base_raw, n_steps)
    base_mean = float(np.mean(Q_base))
    base_std = float(np.std(Q_base))
    base_total = float(np.sum(Q_base))

    results = []
    for name in PARAM_NAMES:
        for sign, sign_label in [(epsilon, "+eps"), (-epsilon, "-eps")]:
            perturbed = {n: v.clone() for n, v in base_raw.items()}
            perturbed[name] = base_raw[name] + sign
            perturbed[name] = perturbed[name].clamp(0.0, 1.0)

            Q_perturbed = _forward_window(step_fn, p_seq, t_seq, pet_seq, perturbed, n_steps)
            p_mean = float(np.mean(Q_perturbed))
            p_std = float(np.std(Q_perturbed))
            p_total = float(np.sum(Q_perturbed))

            d_mean = (p_mean - base_mean) / max(epsilon, 1e-8)
            d_std = (p_std - base_std) / max(epsilon, 1e-8)
            d_total = (p_total - base_total) / max(epsilon, 1e-8)

            results.append({
                "param": name,
                "sign": sign_label,
                "base_Q_mean": base_mean,
                "perturbed_Q_mean": p_mean,
                "dQ_mean": d_mean,
                "dQ_std": d_std,
                "dQ_total": d_total,
                "insensitive": float(abs(d_mean) < 1e-5 and abs(d_total) < 1e-5),
            })

        print(f"  [{name}] dQ_mean(+eps)={results[-2]['dQ_mean']:.4e} dQ_mean(-eps)={results[-1]['dQ_mean']:.4e}")

    return results


# ============================================================================
# Main
# ============================================================================

def save_csv(results: list[dict], path: Path) -> None:
    if not results:
        return
    keys = sorted(set().union(*(r.keys() for r in results)))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def build_overfit_report(results: list[dict]) -> str:
    lines = [
        "# Single Basin Overfit Smoke Test: modhydrolog (m36)",
        "",
        "## Variants",
        "- A: Original step (baseline)",
        "- B: Fix interception_1 gate (use soft_gate_storage_above)",
        "- C: Fix flux_SEEP clamp to [0, S4], remove from Ea",
        "- D: Fix B + C combined",
        "- E: Fix D + MATLAB depression_1 formula",
        "",
        "| Variant | Init Loss | Final Loss | Min Loss | Q Mean | Q Std | Obs Mean | Obs Std |",
        "|---------|-----------|------------|----------|--------|-------|----------|---------|",
    ]
    for r in results:
        lines.append(
            f"| {r['variant']} | {r['initial_loss']:.4f} | {r['final_loss']:.4f} | "
            f"{r['min_loss']:.4f} | {r['Q_mean']:.4f} | {r['Q_std']:.4f} | "
            f"{r['Q_obs_mean']:.4f} | {r['Q_obs_std']:.4f} |"
        )
    lines += ["", "## Key Findings", ""]
    for r in results:
        lines.append(f"- **{r['variant']}**: init_loss={r['initial_loss']:.4f}, final_loss={r['final_loss']:.4f}, Q_mean={r['Q_mean']:.4f}")
    return "\n".join(lines)


def build_sensitivity_report(results: list[dict]) -> str:
    lines = [
        "# Parameter Sensitivity Report: modhydrolog (m36) [Using Variant D]",
        "",
        "| Param | dQ_mean (+eps) | dQ_mean (-eps) | dQ_std (+eps) | dQ_total (+eps) | Insensitive |",
        "|-------|---------------|---------------|---------------|-----------------|-------------|",
    ]
    param_names = sorted(set(r["param"] for r in results))
    for name in param_names:
        subset = [r for r in results if r["param"] == name]
        plus = [r for r in subset if r["sign"] == "+eps"]
        minus = [r for r in subset if r["sign"] == "-eps"]
        if plus and minus:
            lines.append(
                f"| {name} | {plus[0]['dQ_mean']:.4e} | {minus[0]['dQ_mean']:.4e} | "
                f"{plus[0]['dQ_std']:.4e} | {plus[0]['dQ_total']:.4e} | "
                f"{'YES' if plus[0]['insensitive'] > 0.5 else 'no'} |"
            )
    insensitive = [name for name in param_names
                   if any(r["insensitive"] > 0.5 for r in results if r["param"] == name)]
    lines += [
        "",
        "## Key Findings",
        f"Insensitive parameters: {insensitive or 'None'}",
    ]
    return "\n".join(lines)


def main():
    print("=== Step 4: Single Basin Overfit Smoke Test ===")
    overfit_results = run_overfit(n_steps=90, n_epochs=50, lr=0.05, seed=42)

    csv_of = OUTPUT_DIR / "single_basin_overfit.csv"
    md_of = OUTPUT_DIR / "single_basin_overfit_report.md"
    save_csv(overfit_results, csv_of)
    md_of.write_text(build_overfit_report(overfit_results), encoding="utf-8")
    print(f"Overfit results: {csv_of}")

    print("\n=== Step 5: Parameter Sensitivity ===")
    sens_results = run_sensitivity(n_steps=180, epsilon=0.01, seed=42)

    csv_sens = OUTPUT_DIR / "parameter_sensitivity.csv"
    md_sens = OUTPUT_DIR / "parameter_sensitivity_report.md"
    save_csv(sens_results, csv_sens)
    md_sens.write_text(build_sensitivity_report(sens_results), encoding="utf-8")
    print(f"Sensitivity results: {csv_sens}")


if __name__ == "__main__":
    main()
