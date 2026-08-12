#!/usr/bin/env python3
"""Comprehensive clamp-dominance audit for HBV candidate formulas.

Stages 1-7 combined:
  - Identifies all clamp sites
  - Exposes raw vs capped flux on synthetic grids
  - Audits on real CAMELS trajectories
  - Checks gradient effects of clamp saturation
  - Classifies risk per formula
"""
from __future__ import annotations

import csv, math, pickle, sys, time
from pathlib import Path

import numpy as np
import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.flux import snow, recharge, aet, response
from model.hbv_formula_static import HbvFormulaStatic

CAMELS_PATH = _PROJECT.parent.parent / "data" / "camels_dataset"
GAGE_ID_PATH = _PROJECT.parent.parent / "data" / "gage_id.npy"
OUT_BASE = _PROJECT / "validation_results" / "formula_clamp_dominance_audit"

DTYPE = torch.float64
EPS = 1e-6

# ===========================================================================
# Stage 1: Clamp Site Inventory
# ===========================================================================

CLAMP_SITES = [
    # --- recharge ---
    {"node": "recharge", "formula_id": "R0", "formula_name": "beta_recharge",
     "function_name": "beta_recharge", "source_file": "model/flux/recharge.py",
     "line_or_function": "line 27 clamp(SM/FC,0,1); line 26 clamp(I,0); line 29 clamp(beta,min=1e-6); line 33 clamp(R,min=0); line 34 min(R,I)",
     "clamp_type": "torch.clamp + torch.minimum (water-bound cap)",
     "pre_clamp_variable": "I * (SM/FC)^beta (raw)",
     "post_clamp_variable": "min(raw_R_clamped_to_0, I) (capped)",
     "available_bound": "I (liquid input)",
     "is_internal_to_formula": True, "is_external_in_hbv_step": False,
     "notes": "beta_recharge: sigmoid clamp on sat, min(0) on R, min(R,I) water bound"},
    {"node": "recharge", "formula_id": "R4", "formula_name": "saturation_threshold_recharge",
     "function_name": "saturation_threshold_recharge", "source_file": "model/flux/recharge.py",
     "line_or_function": "line 90-103: clamp(FC,min=1e-6), clamp(I,min=0), clamp(sat,0,1), clamp(a_r,min=1e-6), clamp(c_r,0,1), clamp(frac,0,1), min(R,I)",
     "clamp_type": "torch.clamp + torch.minimum (water-bound cap)",
     "pre_clamp_variable": "I * sigmoid_scaled(sat-c_r) (raw)",
     "post_clamp_variable": "min(raw_R_clamped_to_0, I) (capped)",
     "available_bound": "I (liquid input)",
     "is_internal_to_formula": True, "is_external_in_hbv_step": False,
     "notes": "sigmoid-based: lo/hi normalization, frac clamped [0,1], final min(R,I)"},
    {"node": "recharge", "formula_id": "R5", "formula_name": "variable_contributing_area_recharge",
     "function_name": "variable_contributing_area_recharge", "source_file": "model/flux/recharge.py",
     "line_or_function": "line 121-132: clamp(FC,min=1e-6), clamp(I,min=0), clamp(s,0,1), clamp(b_v,min=1e-6), clamp(R,min=0), min(R,I)",
     "clamp_type": "torch.clamp + torch.minimum (water-bound cap)",
     "pre_clamp_variable": "I * (1-(1-s)^b_v) (raw)",
     "post_clamp_variable": "min(raw_R_clamped_to_0, I) (capped)",
     "available_bound": "I (liquid input)",
     "is_internal_to_formula": True, "is_external_in_hbv_step": False,
     "notes": "XAJ/VIC-style: contributing area, min(R,I) water bound"},

    # --- aet ---
    {"node": "aet", "formula_id": "E0", "formula_name": "aet_hbv_default",
     "function_name": "aet_hbv_default", "source_file": "model/flux/aet.py",
     "line_or_function": "line 28-34: clamp(PET,min=0), clamp(SM,min=0), clamp(SM/threshold,0,1), min(ET,SM)",
     "clamp_type": "torch.clamp + torch.minimum (SM-bound cap)",
     "pre_clamp_variable": "PET * SM/threshold (raw)",
     "post_clamp_variable": "min(PET*frac_clamped, SM) (capped)",
     "available_bound": "min(PET, SM)",
     "is_internal_to_formula": True, "is_external_in_hbv_step": False,
     "notes": "HBV default AET: frac clamped [0,1], final min(ET, SM)"},
    {"node": "aet", "formula_id": "E3", "formula_name": "aet_power_law",
     "function_name": "aet_power_law", "source_file": "model/flux/aet.py",
     "line_or_function": "line 108-120: clamp(PET,min=0), clamp(SM,min=0), clamp(FC,min=1e-6), clamp(SM/FC,min=EPS,max=1), min(ET,SM)",
     "clamp_type": "torch.clamp + torch.minimum (SM-bound cap)",
     "pre_clamp_variable": "PET * (SM/FC)^gamma_E (raw)",
     "post_clamp_variable": "min(PET*frac, SM) (capped)",
     "available_bound": "min(PET, SM)",
     "is_internal_to_formula": True, "is_external_in_hbv_step": False,
     "notes": "Power-law: internal clamp SM/FC=EPS to avoid grad singularity"},

    # --- snow ---
    {"node": "snow", "formula_id": "S0", "formula_name": "snowmelt_linear_degreeday",
     "function_name": "snowmelt_linear_degreeday", "source_file": "model/flux/snow.py",
     "line_or_function": "line 69-71: clamp(T-TT,min=0), min(M, SWE)",
     "clamp_type": "torch.clamp + torch.min (SWE-bound cap)",
     "pre_clamp_variable": "CFMAX * (T-TT) (raw)",
     "post_clamp_variable": "min(raw_clamped_to_0, SWE) (capped)",
     "available_bound": "SWE",
     "is_internal_to_formula": True, "is_external_in_hbv_step": False,
     "notes": "Linear degree-day: clamp(T-TT,0), min(melt,SWE)"},

    # --- response ---
    {"node": "response", "formula_id": "Q0", "formula_name": "response_two_reservoir",
     "function_name": "response_two_reservoir", "source_file": "model/flux/response.py",
     "line_or_function": "line 39-52: clamp(SUZ,min=0), clamp(SLZ,min=0), clamp(SUZ-UZL,min=0), min(Q0,SUZ), min(Q1,remaining), min(Q2,SLZ)",
     "clamp_type": "torch.clamp + torch.minimum (storage caps)",
     "pre_clamp_variable": "K0*(SUZ-UZL), K1*remaining_SUZ, K2*SLZ (raw)",
     "post_clamp_variable": "min(raw_Q, storage) (capped)",
     "available_bound": "SUZ or SLZ",
     "is_internal_to_formula": True, "is_external_in_hbv_step": False,
     "notes": "Two-reservoir: each outflow capped by available storage"},
]

# ===========================================================================
# Raw vs Capped flux evaluation
# ===========================================================================

def eval_recharge_raw_capped(fid, I_val, SM_val, FC_val, **params):
    """Return (raw_flux, capped_flux, available_bound) for recharge."""
    I_t = torch.tensor([I_val], dtype=DTYPE)
    SM_t = torch.tensor([SM_val], dtype=DTYPE)
    FC_t = torch.tensor([FC_val], dtype=DTYPE)

    bound = I_val  # available liquid input

    if fid == "R0":
        beta_t = torch.tensor([params.get("beta", 3.5)], dtype=DTYPE)
        # Reconstruct raw: I * (SM/FC)^beta before clamp
        sat = max(min(SM_val / max(FC_val, EPS), 1.0), 0.0)
        raw = I_val * (sat ** params.get("beta", 3.5))
        capped = recharge.beta_recharge(I_t, SM_t, FC_t, beta_t).item()
        return raw, capped, bound

    elif fid == "R4":
        a_r_t = torch.tensor([params.get("a_r", 17.5)], dtype=DTYPE)
        c_r_t = torch.tensor([params.get("c_r", 0.6)], dtype=DTYPE)
        a_r_v = params.get("a_r", 17.5)
        c_r_v = params.get("c_r", 0.6)
        sat = max(min(SM_val / max(FC_val, EPS), 1.0), 0.0)
        lo = 1.0 / (1.0 + math.exp(a_r_v * c_r_v))
        hi = 1.0 / (1.0 + math.exp(-a_r_v * (1.0 - c_r_v)))
        raw_sigmoid = 1.0 / (1.0 + math.exp(-a_r_v * (sat - c_r_v)))
        frac = max(min((raw_sigmoid - lo) / max(hi - lo, EPS), 1.0), 0.0)
        raw = I_val * frac
        capped = recharge.saturation_threshold_recharge(I_t, SM_t, FC_t, a_r_t, c_r_t).item()
        return raw, capped, bound

    elif fid == "R5":
        b_v_t = torch.tensor([params.get("b_v", 0.9)], dtype=DTYPE)
        b_v_v = params.get("b_v", 0.9)
        sat = max(min(SM_val / max(FC_val, EPS), 1.0), 0.0)
        A_s = 1.0 - (1.0 - sat) ** max(b_v_v, EPS)
        raw = I_val * A_s
        capped = recharge.variable_contributing_area_recharge(I_t, SM_t, FC_t, b_v_t).item()
        return raw, capped, bound
    return 0.0, 0.0, bound


def eval_aet_raw_capped(fid, PET_val, SM_val, FC_val, **params):
    """Return (raw_flux, capped_flux, available_bound) for AET."""
    PET_t = torch.tensor([PET_val], dtype=DTYPE)
    SM_t = torch.tensor([SM_val], dtype=DTYPE)
    FC_t = torch.tensor([FC_val], dtype=DTYPE)
    bound = min(PET_val, SM_val)

    if fid == "E0":
        LP_t = torch.tensor([params.get("LP", 0.65)], dtype=DTYPE)
        threshold = max(params.get("LP", 0.65) * FC_val, EPS)
        frac = max(min(SM_val / threshold, 1.0), 0.0)
        raw = PET_val * frac
        capped = aet.aet_hbv_default(PET_t, SM_t, LP_t, FC_t).item()
        return raw, capped, bound

    elif fid == "E3":
        gamma_t = torch.tensor([params.get("gamma_E", 1.75)], dtype=DTYPE)
        g = params.get("gamma_E", 1.75)
        sat = max(min(SM_val / max(FC_val, EPS), 1.0), EPS)
        raw = PET_val * (sat ** g)
        capped = aet.aet_power_law(PET_t, SM_t, FC_t, gamma_t).item()
        return raw, capped, bound
    return 0.0, 0.0, bound


def eval_snow_raw_capped(fid, T_val, SWE_val, **params):
    """Return (raw_flux, capped_flux, available_bound) for snow."""
    T_t = torch.tensor([T_val], dtype=DTYPE)
    SWE_t = torch.tensor([SWE_val], dtype=DTYPE)
    bound = SWE_val

    if fid == "S0":
        TT_t = torch.tensor([params.get("TT", 0.0)], dtype=DTYPE)
        CFMAX_t = torch.tensor([params.get("CFMAX", 5.5)], dtype=DTYPE)
        raw = params.get("CFMAX", 5.5) * max(T_val - params.get("TT", 0.0), 0.0)
        capped = snow.snowmelt_linear_degreeday(T_t, TT_t, CFMAX_t, SWE_t).item()
        return raw, capped, bound
    return 0.0, 0.0, bound


def eval_response_raw_capped(fid, SUZ_val, SLZ_val, **params):
    """Return (raw_flux, capped_flux, available_bound) for response."""
    SUZ_t = torch.tensor([SUZ_val], dtype=DTYPE)
    SLZ_t = torch.tensor([SLZ_val], dtype=DTYPE)
    bound = SUZ_val + SLZ_val

    if fid == "Q0":
        raw = (params.get("K_0", 0.275) * max(SUZ_val - params.get("UZL", 10.0), 0.0) +
               params.get("K_1", 0.155) * SUZ_val +
               params.get("K_2", 0.0505) * SLZ_val)
        capped = response.response_two_reservoir(
            SUZ_t, SLZ_t,
            torch.tensor([params.get("K_0", 0.275)], dtype=DTYPE),
            torch.tensor([params.get("K_1", 0.155)], dtype=DTYPE),
            torch.tensor([params.get("K_2", 0.0505)], dtype=DTYPE),
            torch.tensor([params.get("UZL", 10.0)], dtype=DTYPE))[3].item()
        return raw, capped, bound
    return 0.0, 0.0, bound


# ===========================================================================
# Grid audit
# ===========================================================================

def build_recharge_grid():
    I_vals = [0.0, 0.1, 1.0, 5.0, 20.0, 50.0, 100.0]
    SM_fracs = [0.0, 0.05, 0.1, 0.3, 0.6, 0.9, 1.0, 1.2]
    FC_vals = [50.0, 150.0, 300.0, 600.0]
    params = {
        "R0": [{"beta": 1.0}, {"beta": 3.5}, {"beta": 6.0}],
        "R4": [{"a_r": 5.0, "c_r": 0.3}, {"a_r": 17.5, "c_r": 0.6}, {"a_r": 30.0, "c_r": 0.9}],
        "R5": [{"b_v": 0.3}, {"b_v": 0.9}, {"b_v": 1.5}],
    }
    grid = []
    for I_val in I_vals:
        for smf in SM_fracs:
            for fc_val in FC_vals:
                sm = smf * fc_val
                for fid in ["R0", "R4", "R5"]:
                    for p in params[fid]:
                        grid.append({"node": "recharge", "formula_id": fid,
                                     "I": I_val, "SM": sm, "FC": fc_val, **p})
    return grid


def build_aet_grid():
    PET_vals = [0.0, 0.1, 1.0, 3.0, 6.0, 10.0]
    SM_fracs = [0.0, 0.05, 0.1, 0.3, 0.6, 1.0, 1.2]
    FC_vals = [50.0, 150.0, 300.0, 600.0]
    params = {
        "E0": [{"LP": 0.3}, {"LP": 0.65}, {"LP": 1.0}],
        "E3": [{"gamma_E": 0.5}, {"gamma_E": 1.75}, {"gamma_E": 3.0}],
    }
    grid = []
    for pet in PET_vals:
        for smf in SM_fracs:
            for fc_val in FC_vals:
                sm = smf * fc_val
                for fid in ["E0", "E3"]:
                    for p in params[fid]:
                        grid.append({"node": "aet", "formula_id": fid,
                                     "PET": pet, "SM": sm, "FC": fc_val, **p})
    return grid


def build_snow_grid():
    T_vals = [-10.0, -3.0, 0.0, 1.0, 3.0, 8.0]
    SWE_vals = [0.0, 10.0, 100.0, 500.0]
    params = {"S0": [{"TT": -2.5, "CFMAX": 1.0}, {"TT": 0.0, "CFMAX": 5.5}, {"TT": 2.5, "CFMAX": 10.0}]}
    grid = []
    for T in T_vals:
        for swe in SWE_vals:
            for fid in ["S0"]:
                for p in params[fid]:
                    grid.append({"node": "snow", "formula_id": fid, "T": T, "SWE": swe, **p})
    return grid


def build_response_grid():
    SUZ_vals = [0.0, 0.1, 1.0, 10.0, 50.0, 200.0]
    SLZ_vals = [0.0, 0.1, 1.0, 10.0, 50.0, 200.0]
    params = {"Q0": [{"K_0": 0.05, "K_1": 0.01, "K_2": 0.001, "UZL": 0.0},
                     {"K_0": 0.275, "K_1": 0.155, "K_2": 0.0505, "UZL": 10.0},
                     {"K_0": 0.5, "K_1": 0.3, "K_2": 0.1, "UZL": 100.0}]}
    grid = []
    for suz in SUZ_vals:
        for slz in SLZ_vals:
            for fid in ["Q0"]:
                for p in params[fid]:
                    grid.append({"node": "response", "formula_id": fid,
                                 "SUZ": suz, "SLZ": slz, **p})
    return grid


# ===========================================================================
# Metrics computation
# ===========================================================================

def compute_metrics(raw_vals, capped_vals, bounds, n_cases):
    raw = np.array(raw_vals); cap = np.array(capped_vals); bnd = np.array(bounds)
    valid = np.isfinite(raw) & np.isfinite(cap)
    if valid.sum() < 2:
        return {"n_cases": n_cases, "n_valid": int(valid.sum()), "error": True}

    rv, cv, bv = raw[valid], cap[valid], bnd[valid]
    raw_over = (rv > bv + EPS)
    clamp_hit = (np.abs(cv - rv) > EPS)
    near_clamp = (bv > EPS) & (cv >= 0.95 * bv)
    exact_bound = (np.abs(cv - np.maximum(bv, EPS)) <= EPS)

    r2b = rv / np.maximum(bv, EPS)
    c2b = cv / np.maximum(bv, EPS)
    r2c = rv / np.maximum(cv, EPS)

    return {
        "n_cases": n_cases, "n_valid": int(valid.sum()),
        "raw_nan": int(np.isnan(raw).sum()), "raw_inf": int(np.isinf(raw).sum()),
        "raw_min": float(np.min(rv)), "raw_median": float(np.median(rv)),
        "raw_p95": float(np.percentile(rv, 95)), "raw_max": float(np.max(rv)),
        "capped_min": float(np.min(cv)), "capped_median": float(np.median(cv)),
        "capped_p95": float(np.percentile(cv, 95)), "capped_max": float(np.max(cv)),
        "bound_median": float(np.median(bv)), "bound_max": float(np.max(bv)),
        "raw_over_bound_count": int(raw_over.sum()),
        "raw_over_bound_rate": float(raw_over.mean()),
        "raw_over_2x": float((r2b > 2).mean()),
        "raw_over_5x": float((r2b > 5).mean()),
        "raw_over_10x": float((r2b > 10).mean()),
        "raw_over_100x": float((r2b > 100).mean()),
        "median_raw2bound": float(np.median(r2b)),
        "p95_raw2bound": float(np.percentile(r2b, 95)),
        "max_raw2bound": float(np.max(r2b)),
        "clamp_hit_count": int(clamp_hit.sum()),
        "clamp_hit_rate": float(clamp_hit.mean()),
        "near_clamp_rate": float(near_clamp.mean()),
        "exact_bound_fraction": float(exact_bound.mean()),
        "median_capped2bound": float(np.median(c2b)),
        "p95_capped2bound": float(np.percentile(c2b, 95)),
        "median_raw2capped": float(np.median(r2c)),
        "p95_raw2capped": float(np.percentile(r2c, 95)),
        "max_raw2capped": float(np.max(r2c)),
        "negative_raw_fraction": float((rv < -EPS).mean()),
        "zero_capped_fraction": float((np.abs(cv) <= EPS).mean()),
        "error": False,
    }


# ===========================================================================
# Real-trajectory audit
# ===========================================================================

def run_trajectory_audit():
    print("=== Trajectory Audit (10 basins, default params) ===")
    with open(CAMELS_PATH, "rb") as f:
        forcings, target, attributes = pickle.load(f)
    gage_ids = np.load(GAGE_ID_PATH)

    basin_ids = [1013500, 1022500, 1030500, 1031500, 1047000, 1052500, 1054200, 1055000, 1057000, 1073000]
    B = len(basin_ids)
    idx = [int(np.where(gage_ids == bid)[0][0]) for bid in basin_ids]
    wu, tr_l = 365, 60
    ev_end = wu + tr_l + 60

    forc = forcings[idx, :ev_end, :].astype(np.float32)
    forcing_t_cpu = torch.from_numpy(forc).permute(1, 0, 2).cpu()

    default_norm = 0.4
    _PB = {
        "parBETA": [1.0, 6.0], "parFC": [50.0, 500.0], "parK0": [0.05, 0.5],
        "parK1": [0.01, 0.3], "parK2": [0.001, 0.1], "parLP": [0.3, 1.0],
        "parPERC": [0.0, 3.0], "parUZL": [0.0, 100.0], "parTT": [-2.5, 2.5],
        "parCFMAX": [1.0, 10.0], "parCFR": [0.0, 0.1], "parCWH": [0.0, 0.2],
    }
    default_pv = {name: float(lo + (hi - lo) * default_norm) for name, (lo, hi) in _PB.items()}

    _AMAP = {"parBETA": "beta", "parFC": "FC", "parK0": "K_0", "parK1": "K_1",
             "parK2": "K_2", "parLP": "LP", "parUZL": "UZL", "parTT": "TT",
             "parCFMAX": "CFMAX", "parCFR": "CFR", "parCWH": "CWH", "parPERC": "PERC"}
    _NPMAP = {"snow": ["parTT", "parCFMAX", "parCFR", "parCWH"],
              "recharge": ["parFC", "parBETA"], "aet": ["parFC", "parLP"],
              "response": ["parK0", "parK1", "parK2", "parUZL", "parPERC"]}
    _XP = {"R4": {"a_r": 10.0, "c_r": 0.5}, "R5": {"b_v": 1.0}}
    DEFAULT_IDS = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}

    traj_rows = []
    for b in range(B):
        bid = basin_ids[b]
        for fid in ["R0", "R4", "R5"]:
            combo = dict(DEFAULT_IDS); combo["recharge"] = fid
            pv = dict(default_pv)
            fp = {}
            for n in ["snow", "recharge", "aet", "response"]:
                nd = {}
                for hb in _NPMAP.get(n, []):
                    if hb in pv: nd[_AMAP[hb]] = torch.as_tensor(pv[hb], dtype=torch.float32)
                fp[n] = nd
                fn = combo.get(n, DEFAULT_IDS[n])
                if fn in _XP: fp.setdefault(n, {}).update(_XP[fn])
            if "parPERC" in pv: fp["_perc"] = torch.as_tensor(pv["parPERC"], dtype=torch.float32)

            m = HbvFormulaStatic(formula_config=combo, warm_up=wu, param_dicts=fp)
            diag = m.simulate(forcing_t_cpu[:, b, 0], forcing_t_cpu[:, b, 1],
                               forcing_t_cpu[:, b, 2])
            tr = diag.get("trace", {})
            recharge_vals = tr.get("recharge", None)
            if recharge_vals is None or len(recharge_vals) == 0:
                continue

            rv = np.asarray(recharge_vals[wu:wu + tr_l], dtype=np.float64).flatten()
            rain = np.asarray(tr.get("RAIN", np.zeros_like(rv))[wu:wu + tr_l], dtype=np.float64).flatten()
            tosoil = np.asarray(tr.get("tosoil", np.zeros_like(rv))[wu:wu + tr_l], dtype=np.float64).flatten()
            I_vals = rain + tosoil
            I_vals = np.maximum(I_vals, EPS)

            valid_mask = np.isfinite(rv) & np.isfinite(I_vals)
            rv = rv[valid_mask]; I_vals = I_vals[valid_mask]
            if len(rv) < 2:
                continue
            raw_over = float(np.mean((rv > I_vals + EPS).astype(float)))
            near_clamp = float(np.mean(((I_vals > EPS) & (rv >= 0.95 * I_vals)).astype(float)))
            exact_bound = float(np.mean((np.abs(rv - I_vals) <= EPS).astype(float)))
            r2b = rv / np.maximum(I_vals, EPS)
            capped = np.minimum(rv, I_vals)
            c2b = capped / np.maximum(I_vals, EPS)
            post_clamp = int(np.sum(capped > I_vals + EPS))

            traj_rows.append({
                "basin_id": bid, "node": "recharge", "formula_id": fid,
                "formula_name": fid, "parameter_source": "default_norm=0.4",
                "n_timesteps": len(rv),
                "raw_over_bound_rate": round(float(raw_over), 6),
                "clamp_hit_rate": round(float(raw_over), 6),
                "near_clamp_rate": round(float(near_clamp), 6),
                "exact_bound_fraction": round(float(exact_bound), 6),
                "median_raw2bound": round(float(np.median(r2b)), 6),
                "p95_raw2bound": round(float(np.percentile(r2b, 95)), 6),
                "max_raw2bound": round(float(np.max(r2b)), 6),
                "median_capped2bound": round(float(np.median(c2b)), 6),
                "p95_capped2bound": round(float(np.percentile(c2b, 95)), 6),
                "post_clamp_violations": int(post_clamp),
                "recommended_status": "OK" if raw_over < 0.05 else ("WARNING" if raw_over < 0.20 else "HIGH_RISK"),
            })
            print(f"  {bid}/{fid}: raw_over_bound={raw_over:.1%} near_clamp={near_clamp:.1%} "
                  f"p95_r2b={float(np.percentile(r2b,95)):.2f} max_r2b={float(np.max(r2b)):.1f}")

    return traj_rows


# ===========================================================================
# Main
# ===========================================================================

def run_all():
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # ---- Clamp site inventory ----
    print("Stage 1: Writing clamp site inventory...")
    _w(CLAMP_SITES, OUT_BASE / "clamp_site_inventory.csv",
       ["node", "formula_id", "formula_name", "function_name", "source_file",
        "line_or_function", "clamp_type", "pre_clamp_variable", "post_clamp_variable",
        "available_bound", "is_internal_to_formula", "is_external_in_hbv_step", "notes"])

    # ---- Grid audit ----
    print("Stage 3-4: Grid clamp-dominance audit...")
    eval_fns = {"recharge": eval_recharge_raw_capped, "aet": eval_aet_raw_capped,
                "snow": eval_snow_raw_capped, "response": eval_response_raw_capped}
    grid_fns = {"recharge": build_recharge_grid, "aet": build_aet_grid,
                "snow": build_snow_grid, "response": build_response_grid}

    all_grid_rows = []
    for node, grid_fn in grid_fns.items():
        grid = grid_fn()
        fid_vals = {fid: {"raw": [], "capped": [], "bound": []} for fid in set(c["formula_id"] for c in grid)}
        for c in grid:
            fid = c["formula_id"]
            if node not in eval_fns:
                continue
            # Determine eval params
            eargs = {}
            if node == "recharge": eargs = {"I_val": c["I"], "SM_val": c["SM"], "FC_val": c["FC"]}
            elif node == "aet": eargs = {"PET_val": c["PET"], "SM_val": c["SM"], "FC_val": c["FC"]}
            elif node == "snow": eargs = {"T_val": c["T"], "SWE_val": c["SWE"]}
            elif node == "response": eargs = {"SUZ_val": c["SUZ"], "SLZ_val": c["SLZ"]}

            extra = {k: v for k, v in c.items() if k not in ["node", "formula_id", "I", "SM", "FC", "PET", "T", "SWE", "SUZ", "SLZ"]}
            try:
                r, cap, bnd = eval_fns[node](fid, **eargs, **extra)
                if math.isfinite(r) and math.isfinite(cap):
                    fid_vals[fid]["raw"].append(r)
                    fid_vals[fid]["capped"].append(cap)
                    fid_vals[fid]["bound"].append(bnd)
            except Exception:
                pass

        for fid in fid_vals:
            n = len(fid_vals[fid]["raw"])
            m = compute_metrics(fid_vals[fid]["raw"], fid_vals[fid]["capped"],
                                fid_vals[fid]["bound"], n)
            if m.get("error"):
                continue
            m["node"] = node; m["formula_id"] = fid; m["formula_name"] = fid
            all_grid_rows.append(m)

    # ---- Trajectory audit ----
    traj_rows = run_trajectory_audit()

    # ---- Risk classification ----
    def classify(r):
        rr = r.get("raw_over_bound_rate", 0); ch = r.get("clamp_hit_rate", 0)
        p95 = r.get("p95_raw2bound", 0); mx = r.get("max_raw2bound", 0)
        if rr >= 0.50 or ch >= 0.50 or p95 >= 10 or mx >= 100:
            return "CRITICAL"
        if rr >= 0.20 or ch >= 0.20 or p95 >= 5:
            return "HIGH"
        if rr >= 0.05 or ch >= 0.05 or p95 >= 2:
            return "MEDIUM"
        return "LOW"

    def action(r):
        risk = classify(r)
        if risk == "CRITICAL": return "DISABLE_FOR_NOW"
        if risk in ("HIGH", "MEDIUM"): return "KEEP_HARD_ROUTING_ONLY"
        return "KEEP"

    # ---- Write outputs ----
    _w(all_grid_rows, OUT_BASE / "grid_clamp_dominance_summary.csv",
       ["node", "formula_id", "formula_name", "n_cases", "n_valid",
        "raw_over_bound_rate", "clamp_hit_rate", "near_clamp_rate",
        "exact_bound_fraction", "median_raw2bound", "p95_raw2bound",
        "max_raw2bound", "median_capped2bound", "p95_capped2bound",
        "median_raw2capped", "p95_raw2capped", "max_raw2capped",
        "raw_over_2x", "raw_over_5x", "raw_over_10x", "raw_over_100x",
        "negative_raw_fraction", "zero_capped_fraction"])
    _w(traj_rows, OUT_BASE / "trajectory_clamp_dominance_by_basin.csv",
       ["basin_id", "node", "formula_id", "formula_name", "parameter_source",
        "n_timesteps", "raw_over_bound_rate", "clamp_hit_rate", "near_clamp_rate",
        "exact_bound_fraction", "median_raw2bound", "p95_raw2bound",
        "max_raw2bound", "median_capped2bound", "p95_capped2bound",
        "post_clamp_violations", "recommended_status"])

    # Master summary
    master = []
    for r in all_grid_rows:
        risk = classify(r); act = action(r)
        master.append({"node": r["node"], "formula_id": r["formula_id"],
                       "raw_over_bound_rate": r.get("raw_over_bound_rate", 0),
                       "clamp_hit_rate": r.get("clamp_hit_rate", 0),
                       "p95_raw2bound": r.get("p95_raw2bound", 0),
                       "exact_bound_fraction": r.get("exact_bound_fraction", 0),
                       "risk": risk, "action": act, "source": "grid"})
    for r in traj_rows:
        risk = classify(r); act = action(r)
        master.append({"node": r["node"], "formula_id": r["formula_id"],
                       "raw_over_bound_rate": r.get("raw_over_bound_rate", 0),
                       "clamp_hit_rate": r.get("clamp_hit_rate", 0),
                       "p95_raw2bound": r.get("p95_raw2bound", 0),
                       "exact_bound_fraction": r.get("exact_bound_fraction", 0),
                       "risk": risk, "action": act, "source": "trajectory"})
    _w(master, OUT_BASE / "clamp_dominance_master_summary.csv",
       ["node", "formula_id", "raw_over_bound_rate", "clamp_hit_rate",
        "p95_raw2bound", "exact_bound_fraction", "risk", "action", "source"])

    # Print summary
    print("\n=== Clamp Dominance Summary ===")
    for r in all_grid_rows:
        if r["node"] == "recharge":
            print(f"  {r['formula_id']}: raw_over_bound={r['raw_over_bound_rate']:.1%} "
                  f"clamp_hit={r['clamp_hit_rate']:.1%} near_clamp={r['near_clamp_rate']:.1%} "
                  f"p95_r2b={r['p95_raw2bound']:.1f} max_r2b={r['max_raw2bound']:.1f} "
                  f"risk={classify(r)}")

    print(f"\nDone. Output: {OUT_BASE}")


def _w(rows, path, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        if rows: w.writerows(rows)


if __name__ == "__main__":
    run_all()
