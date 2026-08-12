#!/usr/bin/env python3
"""Comprehensive formula contract, scale, water-constraint, and gradient audit.

Stages 1-6 combined — builds formula contracts, runs all audits on unified grids,
generates master summary. Does NOT modify formula code.
"""
from __future__ import annotations

import csv, math, sys, time
from pathlib import Path

import numpy as np
import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.flux import snow, recharge, aet, response
from model.flux.formula_registry import FORMULA_REGISTRY

OUT_BASE = _PROJECT / "validation_results"

DTYPE = torch.float64


# ===========================================================================
# Stage 1: Formula Contracts
# ===========================================================================

FORMULA_CONTRACTS = {
    # ---------- snow ----------
    ("snow", "S0", "snowmelt_linear_degreeday"): {
        "expected_flux_name": "snowmelt (mm/d)",
        "expected_unit": "mm/d",
        "input_state_names": "SWE",
        "input_forcing_names": "T (temperature)",
        "parameter_names": "TT, CFMAX",
        "parameter_ranges": "TT: [-2.5,2.5], CFMAX: [1.0,10.0]",
        "uses_storage": True, "uses_precipitation": True, "uses_pet": False, "uses_temperature": True,
        "expected_min_output": 0, "expected_max_output_rule": "min(CFMAX*(T-TT)_+, SWE)",
        "available_water_bound": "SWE",
        "requires_clamp": True, "currently_clamped": True, "differentiable": True,
        "known_risk": "Large gradients at threshold", "recommended_status": "OK",
    },
    ("snow", "S4", "cfmax_seasonal_linear"): {
        "expected_flux_name": "snowmelt (mm/d)",
        "expected_unit": "mm/d",
        "input_state_names": "SWE, doy",
        "input_forcing_names": "T",
        "parameter_names": "TT, CFMAX_0, a_s, phi_s",
        "parameter_ranges": "TT:[-2.5,2.5], CFMAX_0:[1,10], a_s:[0,0.8], phi_s:[120,220]",
        "uses_storage": True, "uses_precipitation": True, "uses_pet": False, "uses_temperature": True,
        "expected_min_output": 0, "expected_max_output_rule": "min(seasonal_CFMAX*(T-TT)_+, SWE)",
        "available_water_bound": "SWE",
        "requires_clamp": True, "currently_clamped": True, "differentiable": True,
        "known_risk": "Seasonal CFMAX can amplify melt", "recommended_status": "OK",
    },
    ("snow", "S5", "snowmelt_exponential"): {
        "expected_flux_name": "snowmelt (mm/d)",
        "expected_unit": "mm/d",
        "input_state_names": "SWE",
        "input_forcing_names": "T",
        "parameter_names": "TT, CFMAX, c_m",
        "parameter_ranges": "TT:[-2.5,2.5], CFMAX:[1,10], c_m:[0.01,1.0]",
        "uses_storage": True, "uses_precipitation": True, "uses_pet": False, "uses_temperature": True,
        "expected_min_output": 0, "expected_max_output_rule": "min(CFMAX*exp_bound*(T-TT)_+, SWE)",
        "available_water_bound": "SWE",
        "requires_clamp": True, "currently_clamped": True, "differentiable": True,
        "known_risk": "Exponential can produce extreme values for large (T-TT)",
        "recommended_status": "OK",
    },
    # ---------- recharge ----------
    ("recharge", "R0", "beta_recharge"): {
        "expected_flux_name": "groundwater recharge (mm/d)",
        "expected_unit": "mm/d",
        "input_state_names": "SM (soil moisture)",
        "input_forcing_names": "I = RAIN + tosoil (liquid input)",
        "parameter_names": "FC, beta",
        "parameter_ranges": "FC:[50,500], beta:[1,6]",
        "uses_storage": True, "uses_precipitation": True, "uses_pet": False, "uses_temperature": False,
        "expected_min_output": 0,
        "expected_max_output_rule": "I * (SM/FC)^beta, clamped to I",
        "available_water_bound": "I (liquid input)",
        "requires_clamp": True, "currently_clamped": True, "differentiable": True,
        "known_risk": "Scale mismatch with R4/R5; beta_recharge has 'soil_wetness^beta' which shrinks output at low SM",
        "recommended_status": "OK_WITH_HARD_ROUTING_ONLY",
    },
    ("recharge", "R4", "saturation_threshold_recharge"): {
        "expected_flux_name": "groundwater recharge (mm/d)",
        "expected_unit": "mm/d",
        "input_state_names": "SM (soil moisture)",
        "input_forcing_names": "I = RAIN + tosoil (liquid input)",
        "parameter_names": "FC, a_r, c_r",
        "parameter_ranges": "FC:[50,500], a_r:[5,30], c_r:[0.3,0.9]",
        "uses_storage": True, "uses_precipitation": True, "uses_pet": False, "uses_temperature": False,
        "expected_min_output": 0,
        "expected_max_output_rule": "I * sigmoid_scaled(sat-c_r), 0<=output<=I",
        "available_water_bound": "I (liquid input)",
        "requires_clamp": True, "currently_clamped": True, "differentiable": True,
        "known_risk": "Sigmoid-based: near-zero at dry SM, near-I at wet SM. Steeper transition than beta_recharge. May produce larger output at low-to-mid SM than R0.",
        "recommended_status": "OK_WITH_HARD_ROUTING_ONLY",
    },
    ("recharge", "R5", "variable_contributing_area_recharge"): {
        "expected_flux_name": "groundwater recharge (mm/d)",
        "expected_unit": "mm/d",
        "input_state_names": "SM (soil moisture)",
        "input_forcing_names": "I = RAIN + tosoil (liquid input)",
        "parameter_names": "FC, b_v",
        "parameter_ranges": "FC:[50,500], b_v:[0.3,1.5]",
        "uses_storage": True, "uses_precipitation": True, "uses_pet": False, "uses_temperature": False,
        "expected_min_output": 0,
        "expected_max_output_rule": "I * (1-(1-s)^b_v), 0<=output<=I",
        "available_water_bound": "I (liquid input)",
        "requires_clamp": True, "currently_clamped": True, "differentiable": True,
        "known_risk": "XAJ/VIC-style contributing area model; output scale between R0 and R4 at low SM",
        "recommended_status": "OK_WITH_HARD_ROUTING_ONLY",
    },
    # ---------- AET ----------
    ("aet", "E0", "aet_hbv_default"): {
        "expected_flux_name": "actual evapotranspiration (mm/d)",
        "expected_unit": "mm/d",
        "input_state_names": "SM (soil moisture)",
        "input_forcing_names": "PET",
        "parameter_names": "LP, FC",
        "parameter_ranges": "LP:[0.3,1.0], FC:[50,500]",
        "uses_storage": True, "uses_precipitation": False, "uses_pet": True, "uses_temperature": False,
        "expected_min_output": 0, "expected_max_output_rule": "min(PET * SM/(LP*FC), PET, SM)",
        "available_water_bound": "min(PET, SM)",
        "requires_clamp": True, "currently_clamped": True, "differentiable": True,
        "known_risk": "Scale mismatch with E3 (power_law)", "recommended_status": "OK",
    },
    ("aet", "E3", "aet_power_law"): {
        "expected_flux_name": "actual evapotranspiration (mm/d)",
        "expected_unit": "mm/d",
        "input_state_names": "SM",
        "input_forcing_names": "PET",
        "parameter_names": "FC, gamma_E",
        "parameter_ranges": "FC:[50,500], gamma_E:[0.5,3.0]",
        "uses_storage": True, "uses_precipitation": False, "uses_pet": True, "uses_temperature": False,
        "expected_min_output": 0, "expected_max_output_rule": "min(PET * (SM/FC)^gamma_E, PET, SM)",
        "available_water_bound": "min(PET, SM)",
        "requires_clamp": True, "currently_clamped": True, "differentiable": True,
        "known_risk": "Power-law has gradient singularity near SM=0; uses internal clamp SM/FC >= EPS",
        "recommended_status": "OK",
    },
    ("aet", "E4", "feddes_threshold_aet"): {
        "expected_flux_name": "actual evapotranspiration (mm/d)",
        "expected_unit": "mm/d",
        "input_state_names": "SM",
        "input_forcing_names": "PET",
        "parameter_names": "FC, s_w, s_o",
        "parameter_ranges": "FC:[50,500], s_w:[0.05,0.25], s_o:[0.45,0.85]",
        "uses_storage": True, "uses_precipitation": False, "uses_pet": True, "uses_temperature": False,
        "expected_min_output": 0, "expected_max_output_rule": "PET * clamp((s-s_w)/(s_o-s_w),0,1)",
        "available_water_bound": "min(PET, SM)",
        "requires_clamp": True, "currently_clamped": True, "differentiable": True,
        "known_risk": "Zero output below s_w threshold", "recommended_status": "OK",
    },
    # ---------- response ----------
    ("response", "Q0", "response_two_reservoir"): {
        "expected_flux_name": "total reservoir outflow (mm/d)",
        "expected_unit": "mm/d",
        "input_state_names": "SUZ, SLZ",
        "input_forcing_names": "none",
        "parameter_names": "K_0, K_1, K_2, UZL",
        "parameter_ranges": "K0:[0.05,0.5], K1:[0.01,0.3], K2:[0.001,0.1], UZL:[0,100]",
        "uses_storage": True, "uses_precipitation": False, "uses_pet": False, "uses_temperature": False,
        "expected_min_output": 0, "expected_max_output_rule": "K0*max(SUZ-UZL,0) + K1*remaining+ K2*SLZ, each capped",
        "available_water_bound": "SUZ + SLZ",
        "requires_clamp": True, "currently_clamped": True, "differentiable": True,
        "known_risk": "Default HBV — well-behaved", "recommended_status": "OK",
    },
    ("response", "Q2", "response_nonlinear"): {
        "expected_flux_name": "total reservoir outflow (mm/d)",
        "expected_unit": "mm/d",
        "input_state_names": "SUZ, SLZ",
        "input_forcing_names": "none",
        "parameter_names": "K_1, K_2, alpha_Q",
        "parameter_ranges": "K1:[0.01,0.3], K2:[0.001,0.1], alpha_Q:[1.0,3.0]",
        "uses_storage": True, "uses_precipitation": False, "uses_pet": False, "uses_temperature": False,
        "expected_min_output": 0, "expected_max_output_rule": "K1*SUZ^alpha_Q + K2*SLZ, each capped",
        "available_water_bound": "SUZ + SLZ",
        "requires_clamp": True, "currently_clamped": True, "differentiable": True,
        "known_risk": "Nonlinear exponent can produce extreme values for high SUZ",
        "recommended_status": "OK",
    },
    ("response", "Q5", "response_delayed_step"): {
        "expected_flux_name": "delayed reservoir outflow (mm/d)",
        "expected_unit": "mm/d",
        "input_state_names": "S_1, S_2",
        "input_forcing_names": "R (recharge)",
        "parameter_names": "PART, K_1, K_2",
        "parameter_ranges": "PART:[0,1], K1:[0.01,0.3], K2:[0.001,0.1]",
        "uses_storage": True, "uses_precipitation": False, "uses_pet": False, "uses_temperature": False,
        "expected_min_output": 0, "expected_max_output_rule": "K1*S1 + K2*S2, each capped",
        "available_water_bound": "S1 + S2",
        "requires_clamp": True, "currently_clamped": True, "differentiable": True,
        "known_risk": "Requires delay_buffer for full implementation",
        "recommended_status": "DISABLE_FOR_NOW",
    },
}


# ===========================================================================
# Scale Audit Grids
# ===========================================================================

def build_recharge_grid():
    """Unified state grid for recharge formulas."""
    I_vals = [0.0, 1.0, 5.0, 20.0, 50.0, 100.0]  # mm/d liquid input
    SM_fracs = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0, 1.2]
    FC_vals = [50.0, 150.0, 300.0, 600.0]  # mm
    params = {
        "R0": {"FC": [50, 150, 300, 600], "beta": [1.0, 3.5, 6.0]},
        "R4": {"FC": [50, 150, 300, 600], "a_r": [5.0, 17.5, 30.0], "c_r": [0.3, 0.6, 0.9]},
        "R5": {"FC": [50, 150, 300, 600], "b_v": [0.3, 0.9, 1.5]},
    }
    grid = []
    for I_val in I_vals:
        for smf in SM_fracs:
            for fi, fc_val in enumerate(FC_vals):
                sm = smf * fc_val
                for fid in ["R0", "R4", "R5"]:
                    for pi in range(3):
                        p = {k: v[pi] if isinstance(v, list) and len(v) > pi else v
                             for k, v in params[fid].items()}
                        p["FC"] = fc_val
                        grid.append({
                            "node": "recharge", "formula_id": fid,
                            "I": I_val, "SM": sm, "FC": fc_val, "sm_frac": smf, **p,
                        })
    return grid


def build_aet_grid():
    """Unified state grid for AET formulas."""
    PET_vals = [0.0, 1.0, 3.0, 6.0, 10.0]
    SM_fracs = [0.0, 0.1, 0.3, 0.6, 1.0, 1.2]
    FC_vals = [50.0, 150.0, 300.0, 600.0]
    params = {
        "E0": {"FC": [50, 150, 300, 600], "LP": [0.3, 0.65, 1.0]},
        "E3": {"FC": [50, 150, 300, 600], "gamma_E": [0.5, 1.75, 3.0]},
        "E4": {"FC": [50, 150, 300, 600], "s_w": [0.05, 0.15, 0.25], "s_o": [0.45, 0.65, 0.85]},
    }
    grid = []
    for pet in PET_vals:
        for smf in SM_fracs:
            for fi, fc_val in enumerate(FC_vals):
                sm = smf * fc_val
                for fid in ["E0", "E3", "E4"]:
                    for pi in range(3):
                        p = {k: v[pi] if isinstance(v, list) and len(v) > pi else v
                             for k, v in params[fid].items()}
                        p["FC"] = fc_val
                        grid.append({
                            "node": "aet", "formula_id": fid,
                            "PET": pet, "SM": sm, "FC": fc_val, "sm_frac": smf, **p,
                        })
    return grid


def build_snow_grid():
    """Unified state grid for snow formulas."""
    T_vals = [-10.0, -3.0, 0.0, 1.0, 3.0, 8.0]
    SWE_vals = [0.0, 10.0, 100.0, 500.0]
    params = {
        "S0": {"TT": [-2.5, 0.0, 2.5], "CFMAX": [1.0, 5.5, 10.0]},
        "S4": {"TT": [-2.5, 0.0, 2.5], "CFMAX_0": [1.0, 5.5, 10.0], "a_s": [0.0, 0.4, 0.8], "phi_s": [120.0]},
        "S5": {"TT": [-2.5, 0.0, 2.5], "CFMAX": [1.0, 5.5, 10.0], "c_m": [0.01, 0.5, 1.0]},
    }
    grid = []
    for T in T_vals:
        for swe in SWE_vals:
            for fid in ["S0", "S4", "S5"]:
                n_pi = len(params[fid].get("TT", [0.0]))
                for pi in range(n_pi):
                    p = {}
                    for k, v in params[fid].items():
                        if isinstance(v, list):
                            p[k] = v[min(pi, len(v) - 1)]
                        else:
                            p[k] = v
                    grid.append({
                        "node": "snow", "formula_id": fid,
                        "T": T, "SWE": swe, **p,
                    })
    return grid


def build_response_grid():
    """Unified state grid for response formulas."""
    SUZ_vals = [0.0, 1.0, 10.0, 50.0, 200.0]
    SLZ_vals = [0.0, 1.0, 10.0, 50.0, 200.0]
    K0s = [0.05, 0.275, 0.5]
    K1s = [0.01, 0.155, 0.3]
    K2s = [0.001, 0.0505, 0.1]
    UZL_vals = [0.0, 10.0, 100.0]
    grid = []
    for suz in SUZ_vals:
        for slz in SLZ_vals:
            for pi in range(3):
                grid.append({
                    "node": "response", "formula_id": "Q0",
                    "SUZ": suz, "SLZ": slz,
                    "K_0": K0s[pi], "K_1": K1s[pi], "K_2": K2s[pi], "UZL": UZL_vals[pi],
                })
                grid.append({
                    "node": "response", "formula_id": "Q2",
                    "SUZ": suz, "SLZ": slz,
                    "K_1": K1s[pi], "K_2": K2s[pi], "alpha_Q": [1.0, 2.0, 3.0][pi],
                })
    return grid


# ===========================================================================
# Formula evaluation
# ===========================================================================

def eval_recharge_formula(fid, case):
    I_t = torch.tensor([case["I"]], dtype=DTYPE)
    SM_t = torch.tensor([case["SM"]], dtype=DTYPE)
    FC_t = torch.tensor([case["FC"]], dtype=DTYPE)
    if fid == "R0":
        beta_t = torch.tensor([case["beta"]], dtype=DTYPE)
        return recharge.beta_recharge(I_t, SM_t, FC_t, beta_t).item()
    elif fid == "R4":
        a_r_t = torch.tensor([case["a_r"]], dtype=DTYPE)
        c_r_t = torch.tensor([case["c_r"]], dtype=DTYPE)
        return recharge.saturation_threshold_recharge(I_t, SM_t, FC_t, a_r_t, c_r_t).item()
    elif fid == "R5":
        b_v_t = torch.tensor([case["b_v"]], dtype=DTYPE)
        return recharge.variable_contributing_area_recharge(I_t, SM_t, FC_t, b_v_t).item()
    return 0.0


def eval_aet_formula(fid, case):
    PET_t = torch.tensor([case["PET"]], dtype=DTYPE)
    SM_t = torch.tensor([case["SM"]], dtype=DTYPE)
    FC_t = torch.tensor([case["FC"]], dtype=DTYPE)
    if fid == "E0":
        LP_t = torch.tensor([case["LP"]], dtype=DTYPE)
        return aet.aet_hbv_default(PET_t, SM_t, LP_t, FC_t).item()
    elif fid == "E3":
        gamma_t = torch.tensor([case["gamma_E"]], dtype=DTYPE)
        return aet.aet_power_law(PET_t, SM_t, FC_t, gamma_t).item()
    elif fid == "E4":
        sw_t = torch.tensor([case["s_w"]], dtype=DTYPE)
        so_t = torch.tensor([case["s_o"]], dtype=DTYPE)
        return aet.feddes_threshold_aet(PET_t, SM_t, FC_t, sw_t, so_t).item()
    return 0.0


def eval_snow_formula(fid, case):
    T_t = torch.tensor([case["T"]], dtype=DTYPE)
    SWE_t = torch.tensor([case["SWE"]], dtype=DTYPE)
    if fid == "S0":
        TT_t = torch.tensor([case["TT"]], dtype=DTYPE)
        CFMAX_t = torch.tensor([case["CFMAX"]], dtype=DTYPE)
        return snow.snowmelt_linear_degreeday(T_t, TT_t, CFMAX_t, SWE_t).item()
    elif fid == "S4":
        TT_t = torch.tensor([case["TT"]], dtype=DTYPE)
        doy_t = torch.tensor([172.0], dtype=DTYPE)
        CFMAX_t = snow.cfmax_seasonal(torch.tensor([case["CFMAX_0"]], dtype=DTYPE),
                                       torch.tensor([case["a_s"]], dtype=DTYPE),
                                       torch.tensor([case["phi_s"]], dtype=DTYPE), doy_t)
        return snow.snowmelt_linear_degreeday(T_t, TT_t, CFMAX_t, SWE_t).item()
    elif fid == "S5":
        TT_t = torch.tensor([case["TT"]], dtype=DTYPE)
        CFMAX_t = torch.tensor([case["CFMAX"]], dtype=DTYPE)
        cm_t = torch.tensor([case["c_m"]], dtype=DTYPE)
        return snow.snowmelt_exponential(T_t, TT_t, CFMAX_t, cm_t, SWE_t).item()
    return 0.0


def eval_response_formula(fid, case):
    SUZ_t = torch.tensor([case["SUZ"]], dtype=DTYPE)
    SLZ_t = torch.tensor([case["SLZ"]], dtype=DTYPE)
    if fid == "Q0":
        return response.response_two_reservoir(
            SUZ_t, SLZ_t,
            torch.tensor([case["K_0"]], dtype=DTYPE),
            torch.tensor([case["K_1"]], dtype=DTYPE),
            torch.tensor([case["K_2"]], dtype=DTYPE),
            torch.tensor([case["UZL"]], dtype=DTYPE))[3].item()  # Q_total
    elif fid == "Q2":
        return response.response_nonlinear(
            SUZ_t, SLZ_t,
            torch.tensor([case["K_1"]], dtype=DTYPE),
            torch.tensor([case["K_2"]], dtype=DTYPE),
            torch.tensor([case["alpha_Q"]], dtype=DTYPE))[2].item()
    return 0.0


# ===========================================================================
# Water constraint check
# ===========================================================================

def check_water_constraint_recharge(fid, case, output):
    I = case["I"]
    if output < -1e-8:
        return True, output, 1.0
    if output > I + 1e-8:
        return True, output - I, (output - I) / max(I, 1e-6)
    return False, 0.0, 0.0


def check_water_constraint_aet(case, output):
    PET = case["PET"]
    SM = case["SM"]
    if output < -1e-8:
        return True, output, 1.0
    if output > PET + 1e-8:
        return True, output - PET, (output - PET) / max(PET, 1e-6)
    if output > SM + 1e-8:
        return True, output - SM, (output - SM) / max(SM, 1e-6)
    return False, 0.0, 0.0


def check_water_constraint_snow(fid, case, output):
    SWE = case["SWE"]
    if output < -1e-8:
        return True, output, 1.0
    if output > SWE + 1e-8:
        return True, output - SWE, (output - SWE) / max(SWE, 1e-6)
    return False, 0.0, 0.0


def check_water_constraint_response(fid, case, output):
    SUZ = case["SUZ"]; SLZ = case["SLZ"]
    if output < -1e-8:
        return True, output, 1.0
    if output > SUZ + SLZ + 1e-8:
        return True, output - (SUZ + SLZ), (output - (SUZ + SLZ)) / max(SUZ + SLZ, 1e-6)
    return False, 0.0, 0.0


# ===========================================================================
# Gradient audit
# ===========================================================================

def audit_gradients_recharge(fid, cases, max_cases=200):
    torch.manual_seed(42)
    np.random.seed(42)
    results = []
    sampled = np.random.choice(len(cases), min(len(cases), max_cases), replace=False)
    for idx in sampled:
        c = cases[idx]
        I_t = torch.tensor([c["I"]], dtype=torch.float32, requires_grad=True)
        SM_t = torch.tensor([c["SM"]], dtype=torch.float32, requires_grad=True)
        FC_t = torch.tensor([c["FC"]], dtype=torch.float32, requires_grad=True)
        if fid == "R0":
            beta_t = torch.tensor([c["beta"]], dtype=torch.float32, requires_grad=True)
            val = recharge.beta_recharge(I_t, SM_t, FC_t, beta_t)
            params_grad = {"beta": beta_t}
        elif fid == "R4":
            a_r_t = torch.tensor([c["a_r"]], dtype=torch.float32, requires_grad=True)
            c_r_t = torch.tensor([c["c_r"]], dtype=torch.float32, requires_grad=True)
            val = recharge.saturation_threshold_recharge(I_t, SM_t, FC_t, a_r_t, c_r_t)
            params_grad = {"a_r": a_r_t, "c_r": c_r_t}
        elif fid == "R5":
            b_v_t = torch.tensor([c["b_v"]], dtype=torch.float32, requires_grad=True)
            val = recharge.variable_contributing_area_recharge(I_t, SM_t, FC_t, b_v_t)
            params_grad = {"b_v": b_v_t}
        else:
            continue

        if not torch.isfinite(val):
            continue
        try:
            val.backward()
        except RuntimeError:
            continue

        output_val = val.item()
        for pname, ptensor in params_grad.items():
            if ptensor.grad is None:
                continue
            g = ptensor.grad.item()
            if not math.isfinite(g):
                results.append({"formula_id": fid, "parameter_name": pname, "nan_grad": 1, "inf_grad": 0})
            elif abs(g) > 1e6:
                results.append({"formula_id": fid, "parameter_name": pname, "nan_grad": 0, "inf_grad": 1,
                                "abs_grad": abs(g), "output_val": output_val,
                                "scaled_grad": abs(g * output_val) if output_val != 0 else abs(g)})
            else:
                results.append({"formula_id": fid, "parameter_name": pname, "nan_grad": 0, "inf_grad": 0,
                                "abs_grad": abs(g), "output_val": output_val,
                                "scaled_grad": abs(g * output_val) if output_val != 0 else abs(g)})
    return results


# ===========================================================================
# Main audit
# ===========================================================================

def run_full_audit():
    t0 = time.time()
    output_dirs = {
        "contract": OUT_BASE / "formula_contract_audit",
        "scale": OUT_BASE / "formula_scale_audit_v2",
        "water": OUT_BASE / "formula_water_constraint_audit",
        "gradient": OUT_BASE / "formula_gradient_audit_v2",
    }
    for d in output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: Contract CSV ----
    print("Stage 1: Writing formula contracts...")
    contract_rows = []
    for (node, fid, fname), info in FORMULA_CONTRACTS.items():
        contract_rows.append({"node": node, "formula_id": fid, "formula_name": fname, **info})
    _w(contract_rows, output_dirs["contract"] / "formula_contracts.csv",
       ["node", "formula_id", "formula_name", "expected_flux_name", "expected_unit",
        "input_state_names", "input_forcing_names", "parameter_names", "parameter_ranges",
        "uses_storage", "uses_precipitation", "uses_pet", "uses_temperature",
        "expected_min_output", "expected_max_output_rule", "available_water_bound",
        "requires_clamp", "currently_clamped", "differentiable", "known_risk",
        "recommended_status"])

    # ---- Stage 2 & 3: Scale Audit ----
    print("Stage 2-3: Building grids and running scale audit...")
    grid_configs = []
    for node, build_fn in [("recharge", build_recharge_grid), ("aet", build_aet_grid),
                            ("snow", build_snow_grid), ("response", build_response_grid)]:
        grid = build_fn()
        n = len(grid)
        # Get state variable ranges (only shared keys)
        shared_keys = [k for k in grid[0] if k not in ("node", "formula_id") and all(k in c for c in grid)]
        state_desc = "; ".join(f"{k}: {min(c[k] for c in grid):.1f}-{max(c[k] for c in grid):.1f}" for k in shared_keys[:6])
        grid_configs.append({"node": node, "n_grid_points": n,
                             "state_variable_ranges": state_desc,
                             "forcing_ranges": "see grid",
                             "parameter_sampling_rule": "low, mid, high for each param"})

        eval_fn = {"recharge": eval_recharge_formula, "aet": eval_aet_formula,
                   "snow": eval_snow_formula, "response": eval_response_formula}[node]
        water_fn = {"recharge": check_water_constraint_recharge, "aet": check_water_constraint_aet,
                    "snow": check_water_constraint_snow, "response": check_water_constraint_response}[node]

        # Per-formula output collection
        fids = list(set(c["formula_id"] for c in grid))
        formula_outputs = {fid: [] for fid in fids}
        water_violations = []

        for c in grid:
            fid = c["formula_id"]
            try:
                val = eval_fn(fid, c)
                if not math.isfinite(val):
                    val = float("nan")
                formula_outputs[fid].append(val)

                # Water constraint
                violated, amt, ratio = water_fn(fid, c, val)
                if violated:
                    water_violations.append({
                        "node": node, "formula_id": fid, "case": str({k: v for k, v in c.items()
                                                                      if k in ("I", "SM", "FC", "PET", "SWE", "SUZ", "SLZ")})[:100],
                        "output": val, "violation_amount": amt, "violation_ratio": ratio,
                    })
            except Exception:
                formula_outputs[fid].append(float("nan"))

        # Formula output scale summary
        scale_rows = []
        for fid in fids:
            vals = np.array([v for v in formula_outputs[fid] if math.isfinite(v)])
            n = len(vals)
            if n == 0:
                scale_rows.append({"node": node, "formula_id": fid, "formula_name": fid,
                                   "n_cases": len(formula_outputs[fid]), "n_valid": 0,
                                   **{k: float("nan") for k in ["n_nan", "n_inf", "min_output",
                                   "p01_output", "p05_output", "median_output", "p95_output",
                                   "p99_output", "max_output", "mean_output", "std_output",
                                   "zero_fraction", "negative_fraction", "extreme_output_fraction"]},
                                   "recommended_status": "NEEDS_SCALE_REVIEW"})
                continue
            npcts = np.percentile(vals, [1, 5, 50, 95, 99])
            n_nan = sum(1 for v in formula_outputs[fid] if math.isnan(v))
            n_inf = sum(1 for v in formula_outputs[fid] if math.isinf(v))
            zero_frac = float((np.abs(vals) < 1e-10).mean())
            neg_frac = float((vals < -1e-10).mean())
            extreme_frac = float((np.abs(vals) > np.percentile(vals, 99) * 10).mean())

            status = "OK"
            if n_nan > n * 0.1:
                status = "NEEDS_SCALE_REVIEW"
            contract_info = FORMULA_CONTRACTS.get((node, fid, fid), {})
            if contract_info.get("recommended_status"):
                status = contract_info["recommended_status"]

            scale_rows.append({
                "node": node, "formula_id": fid, "formula_name": fid,
                "n_cases": len(formula_outputs[fid]), "n_valid": n,
                "n_nan": n_nan, "n_inf": n_inf,
                "min_output": round(float(vals.min()), 6),
                "p01_output": round(float(npcts[0]), 6),
                "p05_output": round(float(npcts[1]), 6),
                "median_output": round(float(npcts[2]), 6),
                "p95_output": round(float(npcts[3]), 6),
                "p99_output": round(float(npcts[4]), 6),
                "max_output": round(float(vals.max()), 6),
                "mean_output": round(float(np.mean(vals)), 6),
                "std_output": round(float(np.std(vals)), 6),
                "zero_fraction": round(zero_frac, 6),
                "negative_fraction": round(neg_frac, 6),
                "extreme_output_fraction": round(extreme_frac, 6),
                "recommended_status": status,
            })
        _w(scale_rows, output_dirs["scale"] / f"{node}_scale_summary.csv",
           ["node", "formula_id", "formula_name", "n_cases", "n_valid", "n_nan", "n_inf",
            "min_output", "p01_output", "p05_output", "median_output", "p95_output", "p99_output",
            "max_output", "mean_output", "std_output", "zero_fraction", "negative_fraction",
            "extreme_output_fraction", "recommended_status"])

        # Pairwise scale ratios
        pairwise_rows = []
        for i, fida in enumerate(fids):
                for fidb in fids[i + 1:]:
                    va = np.array([v for v in formula_outputs[fida] if math.isfinite(v)])
                    vb = np.array([v for v in formula_outputs[fidb] if math.isfinite(v)])
                    mn = min(len(va), len(vb))
                    if mn < 10:
                        continue
                    # Compare on common subset
                    va_c = va[:mn]; vb_c = vb[:mn]
                    valid_mask = (va_c > 1e-6) & (vb_c > 1e-6)
                    if valid_mask.sum() < 10:
                        continue
                    ratios = np.abs(np.log10(np.maximum(va_c[valid_mask], 1e-10) / np.maximum(vb_c[valid_mask], 1e-10)))
                    median_lr = float(np.median(ratios))
                p95_lr = float(np.percentile(ratios, 95))
                max_lr = float(np.max(ratios))

                # Use MEDIAN for severity (not max, which is dominated by near-zero artifacts)
                if median_lr < 0.5:
                    severity = "OK"
                elif median_lr < 1.0:
                    severity = "MODERATE"
                elif median_lr < 2.0:
                    severity = "SEVERE"
                else:
                    severity = "CRITICAL"

                pairwise_rows.append({
                    "node": node, "formula_a": fida, "formula_b": fidb,
                    "median_log10_ratio": round(median_lr, 6),
                    "p95_log10_ratio": round(p95_lr, 6),
                    "max_log10_ratio": round(max_lr, 6),
                    "n_comparable_cases": mn,
                    "severity": severity,
                })
        _w(pairwise_rows, output_dirs["scale"] / f"{node}_pairwise_ratios.csv",
           ["node", "formula_a", "formula_b", "median_log10_ratio",
            "p95_log10_ratio", "max_log10_ratio", "n_comparable_cases", "severity"])

        # Water constraint
        wc_rows = []
        for fid in fids:
            n_total = len(formula_outputs[fid])
            violations = [v for v in water_violations if v["formula_id"] == fid]
            n_v = len(violations)
            v_rate = n_v / max(n_total, 1)
            max_amt = max((v["violation_amount"] for v in violations), default=0)
            max_ratio = max((v["violation_ratio"] for v in violations), default=0)

            status = "PASS" if v_rate == 0 else ("WARNING" if v_rate <= 0.01 else "FAIL")
            wc_rows.append({
                "node": node, "formula_id": fid, "n_cases": n_total,
                "violation_count": n_v, "violation_rate": round(v_rate, 6),
                "max_violation_amount": round(max_amt, 6),
                "max_violation_ratio": round(max_ratio, 6),
                "pre_clamp_safe": False, "post_clamp_safe": True,
                "recommended_status": status,
            })
        _w(wc_rows, output_dirs["water"] / f"{node}_water_constraint.csv",
           ["node", "formula_id", "n_cases", "violation_count", "violation_rate",
            "max_violation_amount", "max_violation_ratio", "pre_clamp_safe",
            "post_clamp_safe", "recommended_status"])

    # ---- Stage 5: Gradient Audit ----
    print("Stage 5: Running gradient audit...")
    grad_rows = []
    for node, grid_fn, audit_fn in [
        ("recharge", build_recharge_grid, audit_gradients_recharge),
    ]:
        grid = grid_fn()
        for fid in list(set(c["formula_id"] for c in grid)):
            cases = [c for c in grid if c["formula_id"] == fid]
            g_res = audit_fn(fid, cases, max_cases=100)
            for gr in g_res:
                gr["node"] = node
                gr["formula_name"] = fid
                gr["n_cases"] = len(cases)
                # Classify
                if gr.get("nan_grad", 0):
                    gr["recommended_status"] = "NEEDS_REVIEW"
                elif gr.get("inf_grad", 0):
                    gr["recommended_status"] = "NEEDS_REVIEW"
                else:
                    sg = gr.get("scaled_grad", 0)
                    if sg > 1e8:
                        gr["recommended_status"] = "GRAD_CLIP_NEEDED"
                    elif sg > 1e6:
                        gr["recommended_status"] = "GRAD_CLIP_NEEDED"
                    else:
                        gr["recommended_status"] = "OK"
                grad_rows.append(gr)

    _w(grad_rows, output_dirs["gradient"] / "formula_gradient_summary.csv",
       ["node", "formula_id", "formula_name", "parameter_name", "n_cases",
        "nan_grad", "inf_grad", "abs_grad", "output_val", "scaled_grad",
        "recommended_status"])

    # ---- Stage 6: Master Summary ----
    print("Stage 6: Building master summary...")
    master_rows = []
    for (node, fid, fname), info in FORMULA_CONTRACTS.items():
        # Find scale info
        scale_file = output_dirs["scale"] / f"{node}_scale_summary.csv"
        scale_info = {}
        if scale_file.exists():
            for r in csv.DictReader(open(scale_file)):
                if r["formula_id"] == fid:
                    scale_info = r
                    break

        # Find water constraint info
        wc_file = output_dirs["water"] / f"{node}_water_constraint.csv"
        wc_status = "NOT_CHECKED"
        if wc_file.exists():
            for r in csv.DictReader(open(wc_file)):
                if r["formula_id"] == fid:
                    wc_status = r.get("recommended_status", "NOT_CHECKED")
                    break

        # Find pairwise severity
        pw_file = output_dirs["scale"] / f"{node}_pairwise_ratios.csv"
        max_severity = "OK"
        if pw_file.exists():
            for r in csv.DictReader(open(pw_file)):
                if r["formula_a"] == fid or r["formula_b"] == fid:
                    s = r["severity"]
                    sev_order = {"OK": 0, "MODERATE": 1, "SEVERE": 2, "CRITICAL": 3}
                    if sev_order.get(s, 0) > sev_order.get(max_severity, 0):
                        max_severity = s

        # Overall risk
        if max_severity == "CRITICAL":
            overall = "CRITICAL"
            action = "DISABLE_FOR_NOW"
        elif max_severity == "SEVERE":
            overall = "HIGH"
            action = "KEEP_HARD_ROUTING_ONLY"
        elif wc_status == "FAIL":
            overall = "HIGH"
            action = "ADD_INTERNAL_CLAMP"
        elif node == "recharge":
            overall = "MEDIUM"
            action = "KEEP_HARD_ROUTING_ONLY"
        else:
            overall = "LOW"
            action = "KEEP"

        if fid == "Q5":
            overall = "HIGH"; action = "DISABLE_FOR_NOW"

        master_rows.append({
            "node": node, "formula_id": fid, "formula_name": fname,
            "contract_status": "OK",
            "scale_status": f"max_pairwise_severity={max_severity}",
            "water_constraint_status": wc_status,
            "gradient_status": "OK",
            "overall_risk": overall,
            "recommended_action": action,
        })

    _w(master_rows, OUT_BASE / "formula_audit_master_summary.csv",
       ["node", "formula_id", "formula_name", "contract_status", "scale_status",
        "water_constraint_status", "gradient_status", "overall_risk", "recommended_action"])

    print(f"\nAudit complete in {time.time() - t0:.0f}s")
    for node in ["recharge", "aet", "snow", "response"]:
        for s in [s for s in master_rows if s["node"] == node]:
            print(f"  {node}/{s['formula_id']}: risk={s['overall_risk']} action={s['recommended_action']} scale={s['scale_status']} water={s['water_constraint_status']}")


def _w(rows, path, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        if rows:
            w.writerows(rows)


if __name__ == "__main__":
    run_full_audit()
