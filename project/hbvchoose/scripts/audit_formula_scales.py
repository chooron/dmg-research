#!/usr/bin/env python3
"""Candidate formula scale audit for HBV Formula-MoE.

Compares output magnitudes of different formulas within the same process node
(snow / recharge / AET / response / routing) under controlled state grids and
parameter quantile sweeps.

Run from project root:
    python scripts/audit_formula_scales.py
"""

from __future__ import annotations

import csv
import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.flux import snow, recharge, aet, response, routing
from model.flux.parameter_ranges import PARAMETER_RANGES
from model.flux.formula_registry import FORMULA_REGISTRY

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = _PROJECT / "validation_results" / "formula_scale_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
EPS = 1e-6


def _t(val, dtype=torch.float64):
    return torch.tensor(float(val), dtype=dtype)


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------
_RANGE_FLAT = {}
for _group, _entries in PARAMETER_RANGES.items():
    for _key, _info in _entries.items():
        _RANGE_FLAT[_key] = _info["range"]


def get_range(name):
    if name not in _RANGE_FLAT:
        raise KeyError(f"Parameter '{name}' not found in PARAMETER_RANGES.")
    return _RANGE_FLAT[name]


def param_quantiles(name):
    lo, hi = get_range(name)
    return {q: _t(lo + q * (hi - lo)) for q in QUANTILES}


def median_val(name):
    return param_quantiles(name)[0.50]


def _param_cases(param_names):
    """Return list of (case_name, param_dict, varied_name, varied_q)."""
    qv = {n: param_quantiles(n) for n in param_names}
    cases = []
    # baseline
    cases.append(("baseline_median", {n: qv[n][0.50] for n in param_names}, None, None))
    # oat
    for n in param_names:
        for q in QUANTILES:
            p = {pn: qv[pn][0.50] for pn in param_names}
            p[n] = qv[n][q]
            cases.append((f"oat_{n}_{q}", p, n, q))
    # corners
    for cq, cn in [(0.25, "low_all"), (0.50, "median_all"), (0.75, "high_all"),
                    (0.05, "extreme_low"), (0.95, "extreme_high")]:
        cases.append((cn, {n: qv[n][cq] for n in param_names}, None, cq))
    return cases


def _state_combos(state_spec):
    """state_spec: {name: [values], ...} -> list of {name: tensor, ...}"""
    keys = list(state_spec)
    for combo in itertools.product(*[state_spec[k] for k in keys]):
        yield {k: _t(v) for k, v in zip(keys, combo)}


# ---------------------------------------------------------------------------
# Formula definitions
# ---------------------------------------------------------------------------

def _wrap_seasonal(args):
    T, SWE, doy = args["T"], args["SWE"], args["doy"]
    TT, CFMAX, a_s, phi_s = args["TT"], args["CFMAX"], args["a_s"], args["phi_s"]
    CFMAX_t = snow.cfmax_seasonal(CFMAX, a_s, phi_s, doy)
    M = snow.snowmelt_linear_degreeday(T, TT, CFMAX_t, SWE)
    return {"M": M, "CFMAX_t": CFMAX_t}


# (func_or_wrapper, param_names, state_keys, state_grid, output_keys, post_process)
FormulaSpec = tuple  # for readability

SNOW_SPECS = [
    ("snowmelt_linear_degreeday",
     ["TT", "CFMAX"],
     {"T": [-5.0, -2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0],
      "SWE": [0.0, 5.0, 20.0, 100.0]},
     ["M"]),
    ("snowmelt_smooth_degreeday",
     ["TT", "CFMAX", "tau_M"],
     {"T": [-5.0, -2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0],
      "SWE": [0.0, 5.0, 20.0, 100.0]},
     ["M"]),
    ("cfmax_seasonal_linear",
     ["TT", "CFMAX", "a_s", "phi_s"],
     {"T": [-5.0, -2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0],
      "SWE": [0.0, 5.0, 20.0, 100.0],
      "doy": [15, 80, 172, 266, 355]},
     ["M", "CFMAX_t"]),
    ("snowmelt_exponential",
     ["TT", "CFMAX", "c_m"],
     {"T": [-5.0, -2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0],
      "SWE": [0.0, 5.0, 20.0, 100.0]},
     ["M"]),
]

RECHARGE_SPECS = [
    ("beta_recharge", ["FC", "BETA"],
     {"I": [0.0, 1.0, 5.0, 20.0, 80.0],
      "SM_frac": [0.0, 0.05, 0.20, 0.50, 0.80, 0.95, 1.0]},
     ["R"]),
    ("linear_recharge", ["FC"],
     {"I": [0.0, 1.0, 5.0, 20.0, 80.0],
      "SM_frac": [0.0, 0.05, 0.20, 0.50, 0.80, 0.95, 1.0]},
     ["R"]),
    ("strong_nonlinear_recharge", ["FC", "beta_h"],
     {"I": [0.0, 1.0, 5.0, 20.0, 80.0],
      "SM_frac": [0.0, 0.05, 0.20, 0.50, 0.80, 0.95, 1.0]},
     ["R"]),
    ("weak_nonlinear_recharge", ["FC", "beta_l"],
     {"I": [0.0, 1.0, 5.0, 20.0, 80.0],
      "SM_frac": [0.0, 0.05, 0.20, 0.50, 0.80, 0.95, 1.0]},
     ["R"]),
    ("saturation_threshold_recharge", ["FC", "a_r", "c_r"],
     {"I": [0.0, 1.0, 5.0, 20.0, 80.0],
      "SM_frac": [0.0, 0.05, 0.20, 0.50, 0.80, 0.95, 1.0]},
     ["R"]),
    ("variable_contributing_area_recharge", ["FC", "b_v"],
     {"I": [0.0, 1.0, 5.0, 20.0, 80.0],
      "SM_frac": [0.0, 0.05, 0.20, 0.50, 0.80, 0.95, 1.0]},
     ["R"]),
]

AET_SPECS = [
    # Group 1: use PET, SM (= SM_frac * FC)
    ("aet_hbv_default", ["LP", "FC"],
     {"PET": [0.0, 1.0, 3.0, 6.0, 10.0],
      "SM_frac": [0.0, 0.05, 0.20, 0.50, 0.80, 1.0]},
     ["ET"]),
    ("aet_smooth_hbv", ["LP", "FC", "tau_E"],
     {"PET": [0.0, 1.0, 3.0, 6.0, 10.0],
      "SM_frac": [0.0, 0.05, 0.20, 0.50, 0.80, 1.0]},
     ["ET"]),
    ("aet_power_law", ["FC", "gamma_E"],
     {"PET": [0.0, 1.0, 3.0, 6.0, 10.0],
      "SM_frac": [0.0, 0.05, 0.20, 0.50, 0.80, 1.0]},
     ["ET"]),
    # Group 2: temperature_corrected — more states
    ("temperature_corrected_aet", ["LP", "FC", "CET"],
     {"PET_m": [3.0], "T_t": [-5.0, 0.0, 10.0, 20.0, 30.0],
      "T_m": [0.0, 10.0, 20.0], "SM_frac": [0.0, 0.20, 0.50, 0.80, 1.0]},
     ["PET_t", "ET"]),
    ("feddes_threshold_aet", ["FC", "s_w", "s_o"],
     {"PET": [0.0, 1.0, 3.0, 6.0, 10.0],
      "SM_frac": [0.0, 0.05, 0.20, 0.50, 0.80, 1.0]},
     ["ET"]),
]

RESPONSE_SPECS = [
    ("response_two_reservoir", ["K0", "K1", "K2", "UZL"],
     {"SUZ": [0.0, 1.0, 10.0, 50.0, 150.0],
      "SLZ": [0.0, 1.0, 20.0, 100.0, 300.0]},
     ["Q0", "Q1", "Q2", "Q"]),
    ("response_smooth_threshold", ["K0", "K1", "K2", "UZL", "tau_Q"],
     {"SUZ": [0.0, 1.0, 10.0, 50.0, 150.0],
      "SLZ": [0.0, 1.0, 20.0, 100.0, 300.0]},
     ["Q0", "Q1", "Q2", "Q"]),
    ("response_nonlinear", ["K1", "K2", "alpha_Q"],
     {"SUZ": [0.0, 1.0, 10.0, 50.0, 150.0],
      "SLZ": [0.0, 1.0, 20.0, 100.0, 300.0]},
     ["Q_uz", "Q_lz", "Q"]),
    ("response_single_reservoir", ["K"],
     {"S": [0.0, 1.0, 20.0, 100.0, 300.0]},
     ["Q"]),
    ("response_two_parallel", ["K_f", "K_s", "p"],
     {"R": [0.0, 1.0, 10.0, 50.0],
      "S_f": [0.0, 1.0, 10.0, 50.0],
      "S_s": [0.0, 1.0, 20.0, 100.0]},
     ["R_f", "R_s", "Q_f", "Q_s", "Q"]),
    ("response_delayed_step", ["K1", "K2", "PART"],
     {"R_in": [0.0, 1.0, 10.0, 50.0],
      "S_1": [0.0, 1.0, 10.0, 50.0],
      "S_2": [0.0, 1.0, 20.0, 100.0]},
     ["R_imm", "R_del", "Q_1", "Q_2", "Q"]),
]

ROUTING_SPECS = [
    ("triangular_weights", ["MAXBAS"],
     {"length": [7, 15, 30]},  # length is implicit; we pick from MAXBAS
     ["weights"]),
    ("gamma_weights", ["route_a", "route_b"],
     {"length": [7, 15, 30]},
     ["weights"]),
]

# Map formula_id -> callable
_FUNC_MAP = {
    # snow
    "snowmelt_linear_degreeday": snow.snowmelt_linear_degreeday,
    "snowmelt_smooth_degreeday": snow.snowmelt_smooth_degreeday,
    "snowmelt_exponential": snow.snowmelt_exponential,
    "cfmax_seasonal_linear": _wrap_seasonal,
    # recharge
    "beta_recharge": recharge.beta_recharge,
    "linear_recharge": recharge.linear_recharge,
    "strong_nonlinear_recharge": recharge.strong_nonlinear_recharge,
    "weak_nonlinear_recharge": recharge.weak_nonlinear_recharge,
    "saturation_threshold_recharge": recharge.saturation_threshold_recharge,
    "variable_contributing_area_recharge": recharge.variable_contributing_area_recharge,
    # aet
    "aet_hbv_default": aet.aet_hbv_default,
    "aet_smooth_hbv": aet.aet_smooth_hbv,
    "temperature_corrected_aet": aet.temperature_corrected_aet,
    "aet_power_law": aet.aet_power_law,
    "feddes_threshold_aet": aet.feddes_threshold_aet,
    # response
    "response_two_reservoir": response.response_two_reservoir,
    "response_smooth_threshold": response.response_smooth_threshold,
    "response_nonlinear": response.response_nonlinear,
    "response_single_reservoir": response.response_single_reservoir,
    "response_two_parallel": response.response_two_parallel,
    "response_delayed_step": response.response_delayed_step,
    # routing
    "triangular_weights": routing.triangular_weights,
    "gamma_weights": routing.gamma_weights,
}


# ---------------------------------------------------------------------------
# Node name per formula
# ---------------------------------------------------------------------------
_NODE_MAP = {}
for _specs, _node in [(SNOW_SPECS, "snow"), (RECHARGE_SPECS, "recharge"),
                       (AET_SPECS, "aet"), (RESPONSE_SPECS, "response"),
                       (ROUTING_SPECS, "routing")]:
    for _s in _specs:
        _NODE_MAP[_s[0]] = _node

_ALL_SPECS = ([(s, "snow") for s in SNOW_SPECS] +
              [(s, "recharge") for s in RECHARGE_SPECS] +
              [(s, "aet") for s in AET_SPECS] +
              [(s, "response") for s in RESPONSE_SPECS] +
              [(s, "routing") for s in ROUTING_SPECS])


# ---------------------------------------------------------------------------
# Pool filtering from FORMULA_REGISTRY
# ---------------------------------------------------------------------------

def _build_status_map():
    """Return {(node, spec_fid): status} from FORMULA_REGISTRY."""
    smap = {}
    for node, node_data in FORMULA_REGISTRY.items():
        for entry in node_data["formulas"]:
            spec_fid = entry.get("spec_fid", entry["function"])
            smap[(node, spec_fid)] = entry["status"]
    return smap


_STATUS_MAP = _build_status_map()


def _status_for(node, spec_fid):
    return _STATUS_MAP.get((node, spec_fid), "unregistered")


def _pool_ok(node, spec_fid, pool):
    """Check whether a spec should be included for the given pool."""
    if pool == "all":
        return True
    status = _status_for(node, spec_fid)
    if pool == "main":
        return status == "main"
    if pool == "ablation":
        return status in ("main", "ablation_only", "extension_only")
    return False


def _filter_specs(pool):
    """Return (node, spec) pairs filtered by pool."""
    filtered = []
    for spec, node in _ALL_SPECS:
        fid = spec[0]
        if _pool_ok(node, fid, pool):
            filtered.append((spec, node))
    return filtered


def _pool_formulas(node, pool):
    """Return list of (fid, status) for formulas in a node/pool."""
    result = []
    for spec, _node in _ALL_SPECS:
        fid = spec[0]
        if _node == node and _pool_ok(node, fid, pool):
            result.append((fid, _status_for(node, fid)))
    return result


def _downgraded_formulas(node):
    """Return formulas not in main pool, with reasons."""
    result = []
    for spec, _node in _ALL_SPECS:
        fid = spec[0]
        if _node != node:
            continue
        status = _status_for(node, fid)
        if status == "unregistered":
            result.append((fid, "unregistered", "not present in FORMULA_REGISTRY; excluded from all non-'all' pools"))
        elif status != "main":
            reason = {
                "ablation_only": "parameter-regime variant of an HBV default formula; excluded from main MoE pool",
                "extension_only": "non-HBV extension; retained for comparison only",
            }.get(status, status)
            result.append((fid, status, reason))
    return result


# Pool-specific output paths
def _pool_paths(pool):
    """Return output subdirectory for a given pool."""
    sub = {"main": "main_pool", "all": "all_pools", "ablation": "ablation_pool"}.get(pool, pool)
    d = OUTPUT_DIR / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Audit loop — per formula, with derived state injection
# ---------------------------------------------------------------------------

def _derive_sm(state_dict, pset, fc_key="FC"):
    FC = pset.get(fc_key, median_val("FC"))
    return state_dict["SM_frac"] * FC


def _derive_length_from_maxbas(state_dict, pset):
    """For triangular_weights, use MAXBAS as the weight length."""
    return pset.get("MAXBAS", _t(5))


def run_audit(pool="all"):
    rows = []
    for spec, node in _filter_specs(pool):
        fid, param_names, state_grid, output_keys = spec
        func = _FUNC_MAP[fid]
        param_cases = _param_cases(param_names)

        for pcase_name, pset, varied_name, varied_q in param_cases:
            for sd in _state_combos(state_grid):
                # Inject derived states
                if "SM_frac" in sd:
                    fc_key = "FC"
                    if fid == "aet_hbv_default" or fid == "aet_smooth_hbv" or fid == "temperature_corrected_aet":
                        fc_key = "FC"
                    sd["SM"] = _derive_sm(sd, pset, fc_key)

                # Build positional args: first state keys, then param keys
                state_keys_ordered = [k for k in sd if k.endswith("_frac") is False and k not in ("SM",)]
                # Actually build from spec's state grid keys
                args = []
                # The state keys from the spec
                spec_state_keys = list(state_grid.keys())
                # But wrappers need different keys; let the wrapper use sd dict
                if fid == "cfmax_seasonal_linear":
                    args_sdict = sd
                elif fid == "temperature_corrected_aet":
                    args_sdict = sd
                elif fid == "response_delayed_step":
                    # needs R, S_1, S_2 — already in sd
                    args_sdict = sd
                else:
                    args_sdict = sd

                try:
                    result = _invoke(func, fid, args_sdict, pset, output_keys)
                except Exception as exc:
                    row = _make_row(node, fid, pcase_name, varied_name, varied_q, sd, pset)
                    row["output_name"] = "__error__"
                    row["output_value"] = float("nan")
                    row["error"] = str(exc)
                    rows.append(row)
                    continue

                for out_name, out_val in result.items():
                    if torch.is_tensor(out_val) and out_val.numel() > 1:
                        # Vector output: extract scalar metrics
                        scalars = _vector_to_scalars(out_name, out_val)
                        for metric_name, metric_val in scalars.items():
                            row = _make_row(node, fid, pcase_name, varied_name, varied_q, sd, pset)
                            row["output_name"] = metric_name
                            row["output_value"] = metric_val
                            rows.append(row)
                    else:
                        val = out_val.detach().cpu().item() if torch.is_tensor(out_val) else float(out_val)
                        row = _make_row(node, fid, pcase_name, varied_name, varied_q, sd, pset)
                        row["output_name"] = out_name
                        row["output_value"] = val
                        rows.append(row)

    return rows


def _to_result_dict(raw, output_keys):
    """Normalise formula output to a dict {key: tensor, ...}."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, tuple):
        return dict(zip(output_keys, raw))
    return {output_keys[0]: raw}


def _invoke(func, fid, sd, pset, output_keys):
    """Call formula and return dict {output_name: tensor, ...}."""
    if fid == "cfmax_seasonal_linear":
        return func({**sd, **pset})

    if fid == "temperature_corrected_aet":
        raw = aet.temperature_corrected_aet(
            sd["PET_m"], sd["T_t"], sd["T_m"],
            pset["CET"], sd["SM"], pset["LP"], pset["FC"])
        return _to_result_dict(raw, output_keys)

    if fid == "response_delayed_step":
        raw = response.response_delayed_step(
            sd["R_in"], sd["S_1"], sd["S_2"],
            pset["PART"], pset["K1"], pset["K2"])
        return _to_result_dict(raw, output_keys)

    arg_map = {**sd, **pset}
    keys = _get_call_keys(fid)
    pos_args = [arg_map[k] for k in keys]
    raw = func(*pos_args)
    return _to_result_dict(raw, output_keys)


def _vector_to_scalars(name, w):
    """Convert a weight vector to scalar metrics."""
    w = w.detach().cpu().to(torch.float64)
    w_sum = w.sum().item()
    n = w.numel()
    indices = torch.arange(n, dtype=torch.float64)
    if w_sum > EPS:
        mean_lag = (indices * w).sum().item() / w_sum
        var = ((indices - mean_lag) ** 2 * w).sum().item() / w_sum
        spread = math.sqrt(max(var, 0.0))
        peak_lag = w.argmax().item()
    else:
        mean_lag = 0.0
        spread = 0.0
        peak_lag = 0
    return {
        f"{name}_sum": w_sum,
        f"{name}_peak_lag": float(peak_lag),
        f"{name}_mean_lag": mean_lag,
        f"{name}_spread": spread,
        f"{name}_n": n,
    }


def _get_call_keys(fid):
    """Return ordered list of argument names for positional call."""
    sig_map = {
        "snowmelt_linear_degreeday": ["T", "TT", "CFMAX", "SWE"],
        "snowmelt_smooth_degreeday": ["T", "TT", "CFMAX", "tau_M", "SWE"],
        "snowmelt_exponential": ["T", "TT", "CFMAX", "c_m", "SWE"],
        "beta_recharge": ["I", "SM", "FC", "BETA"],
        "linear_recharge": ["I", "SM", "FC"],
        "strong_nonlinear_recharge": ["I", "SM", "FC", "beta_h"],
        "weak_nonlinear_recharge": ["I", "SM", "FC", "beta_l"],
        "saturation_threshold_recharge": ["I", "SM", "FC", "a_r", "c_r"],
        "variable_contributing_area_recharge": ["I", "SM", "FC", "b_v"],
        "aet_hbv_default": ["PET", "SM", "LP", "FC"],
        "aet_smooth_hbv": ["PET", "SM", "LP", "FC", "tau_E"],
        "aet_power_law": ["PET", "SM", "FC", "gamma_E"],
        "feddes_threshold_aet": ["PET", "SM", "FC", "s_w", "s_o"],
        "response_two_reservoir": ["SUZ", "SLZ", "K0", "K1", "K2", "UZL"],
        "response_smooth_threshold": ["SUZ", "SLZ", "K0", "K1", "K2", "UZL", "tau_Q"],
        "response_nonlinear": ["SUZ", "SLZ", "K1", "K2", "alpha_Q"],
        "response_single_reservoir": ["S", "K"],
        "response_two_parallel": ["R", "S_f", "S_s", "K_f", "K_s", "p"],
        "triangular_weights": ["MAXBAS"],
        "gamma_weights": ["route_a", "route_b", "length"],
    }
    return sig_map.get(fid, [])


def _make_row(node, fid, pcase, varied_name, varied_q, sd, pset):
    return {
        "node": node,
        "formula_id": fid,
        "scenario_id": f"{node}_{fid}_{pcase}",
        "parameter_case": pcase,
        "parameter_name_varied": varied_name or "",
        "parameter_quantile": varied_q if varied_q is not None else "",
        "state_variables": str({k: round(v.item() if torch.is_tensor(v) else v, 6) for k, v in sd.items()}),
        "parameter_values": str({k: round(v.item(), 6) for k, v in pset.items()}),
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _rows_for(rows, node, output_name):
    return [r for r in rows if r["node"] == node and r["output_name"] == output_name
            and not r.get("error") and not math.isnan(r["output_value"])]


def compute_cross_comparison(pool="all"):
    """Run all formulas at shared baseline states for direct comparison."""
    comparison_rows = []
    specs_by_node = defaultdict(list)
    for spec, node in _filter_specs(pool):
        specs_by_node[node].append(spec)

    # --- Snow: T × SWE, all formulas share these states ---
    snow_states = list(_state_combos({
        "T": [-5.0, -2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0],
        "SWE": [0.0, 5.0, 20.0, 100.0],
    }))
    for spec in specs_by_node.get("snow", []):
        fid, pnames, _, out_keys = spec
        pset = {n: median_val(n) for n in pnames}
        func = _FUNC_MAP[fid]
        for sd in snow_states:
            if fid == "cfmax_seasonal_linear":
                sd_copy = dict(sd)
                sd_copy["doy"] = _t(172.0)
                result = _invoke(func, fid, sd_copy, pset, out_keys)
            else:
                result = _invoke(func, fid, sd, pset, out_keys)
            for oname, oval in result.items():
                if torch.is_tensor(oval) and oval.numel() == 1:
                    comparison_rows.append({
                        "node": "snow", "formula_id": fid,
                        "comparison_key": f"T={sd['T'].item():.1f}_SWE={sd['SWE'].item():.1f}",
                        "output_name": oname,
                        "output_value": oval.detach().cpu().item(),
                    })

    # --- Recharge: I × SM_frac, all formulas share ---
    recharge_states = list(_state_combos({
        "I": [0.0, 1.0, 5.0, 20.0, 80.0],
        "SM_frac": [0.0, 0.05, 0.20, 0.50, 0.80, 0.95, 1.0],
    }))
    for spec in specs_by_node.get("recharge", []):
        fid, pnames, _, out_keys = spec
        pset = {n: median_val(n) for n in pnames}
        func = _FUNC_MAP[fid]
        for sd in recharge_states:
            sd["SM"] = _derive_sm(sd, pset)
            result = _invoke(func, fid, sd, pset, out_keys)
            for oname, oval in result.items():
                if torch.is_tensor(oval) and oval.numel() == 1:
                    comparison_rows.append({
                        "node": "recharge", "formula_id": fid,
                        "comparison_key": f"I={sd['I'].item():.1f}_SMf={sd['SM_frac'].item():.3f}",
                        "output_name": oname,
                        "output_value": oval.detach().cpu().item(),
                    })

    # --- AET: PET × SM_frac (skip temperature_corrected which has different interface) ---
    aet_states = list(_state_combos({
        "PET": [0.0, 1.0, 3.0, 6.0, 10.0],
        "SM_frac": [0.0, 0.05, 0.20, 0.50, 0.80, 1.0],
    }))
    for spec in specs_by_node.get("aet", []):
        if spec[0] == "temperature_corrected_aet":
            continue
        fid, pnames, _, out_keys = spec
        pset = {n: median_val(n) for n in pnames}
        func = _FUNC_MAP[fid]
        for sd in aet_states:
            sd["SM"] = _derive_sm(sd, pset)
            result = _invoke(func, fid, sd, pset, out_keys)
            for oname, oval in result.items():
                if torch.is_tensor(oval) and oval.numel() == 1:
                    comparison_rows.append({
                        "node": "aet", "formula_id": fid,
                        "comparison_key": f"PET={sd['PET'].item():.1f}_SMf={sd['SM_frac'].item():.2f}",
                        "output_name": oname,
                        "output_value": oval.detach().cpu().item(),
                    })

    # --- Response: SUZ × SLZ for reservoir formulas ---
    resp_states_2r = list(_state_combos({
        "SUZ": [0.0, 1.0, 10.0, 50.0, 150.0],
        "SLZ": [0.0, 1.0, 20.0, 100.0, 300.0],
    }))
    resp_compare_fids = {"response_two_reservoir", "response_smooth_threshold", "response_nonlinear"}
    for spec in specs_by_node.get("response", []):
        if spec[0] not in resp_compare_fids:
            continue
        fid, pnames, _, out_keys = spec
        pset = {n: median_val(n) for n in pnames}
        func = _FUNC_MAP[fid]
        for sd in resp_states_2r:
            result = _invoke(func, fid, sd, pset, out_keys)
            for oname, oval in result.items():
                if torch.is_tensor(oval) and oval.numel() == 1:
                    comparison_rows.append({
                        "node": "response", "formula_id": fid,
                        "comparison_key": f"SUZ={sd['SUZ'].item():.1f}_SLZ={sd['SLZ'].item():.1f}",
                        "output_name": oname,
                        "output_value": oval.detach().cpu().item(),
                    })

    # --- Routing: weight vectors compared at same length ---
    routing_states = list(_state_combos({"length": [7, 15, 30]}))
    for spec in specs_by_node.get("routing", []):
        fid, pnames, _, out_keys = spec
        pset = {n: median_val(n) for n in pnames}
        func = _FUNC_MAP[fid]
        for sd in routing_states:
            result = _invoke(func, fid, sd, pset, out_keys)
            for oname, oval in result.items():
                if torch.is_tensor(oval) and oval.numel() > 1:
                    scalars = _vector_to_scalars(oname, oval)
                    for mk, mv in scalars.items():
                        comparison_rows.append({
                            "node": "routing", "formula_id": fid,
                            "comparison_key": f"len={sd['length'].item():.0f}",
                            "output_name": mk,
                            "output_value": mv,
                        })
                elif torch.is_tensor(oval) and oval.numel() == 1:
                    comparison_rows.append({
                        "node": "routing", "formula_id": fid,
                        "comparison_key": f"len={sd['length'].item():.0f}",
                        "output_name": oname,
                        "output_value": oval.detach().cpu().item(),
                    })

    return comparison_rows


def compute_summary(rows, comparison_rows):
    """Compute per-node summary and per-formula flags."""
    summaries = []
    flags = []

    for node in ["snow", "recharge", "aet", "response", "routing"]:
        primary = {"snow": "M", "recharge": "R", "aet": "ET",
                   "response": "Q", "routing": "weights_mean_lag"}[node]
        cr = [r for r in comparison_rows if r["node"] == node and r["output_name"] == primary]
        nr = _rows_for(rows, node, primary)
        fids = sorted(set(r["formula_id"] for r in nr))

        # Cross-formula comparison: group by comparison_key
        by_ckey = defaultdict(list)
        for r in cr:
            by_ckey[r["comparison_key"]].append(r)

        ratios = []
        largest_cnt = defaultdict(int)
        smallest_cnt = defaultdict(int)

        for ckey, crows in by_ckey.items():
            vals = {r["formula_id"]: r["output_value"] for r in crows}
            pos = {k: v for k, v in vals.items() if v > EPS}
            if len(pos) >= 2:
                mx = max(pos.values())
                mn = min(pos.values())
                lr = math.log10(mx + EPS) - math.log10(mn + EPS)
                ratios.append((ckey, lr, mx, mn, vals))
                for fid2, v2 in vals.items():
                    if v2 == mx:
                        largest_cnt[fid2] += 1
                    if v2 == mn and mn > EPS:
                        smallest_cnt[fid2] += 1

        log10s = [r[1] for r in ratios]
        max_log10 = max(log10s) if log10s else 0.0
        med_log10 = sorted(log10s)[len(log10s) // 2] if log10s else 0.0
        n_severe = sum(1 for v in log10s if v > 1.0)
        n_moderate = sum(1 for v in log10s if 0.5 < v <= 1.0)

        n_scen = len(ratios)
        sys_large = [f for f in fids if n_scen > 0 and largest_cnt.get(f, 0) / n_scen > 0.7]
        sys_small = [f for f in fids if n_scen > 0 and smallest_cnt.get(f, 0) / n_scen > 0.7]

        summaries.append({
            "node": node, "num_scenarios": n_scen,
            "max_log10_ratio": round(max_log10, 4),
            "median_log10_ratio": round(med_log10, 4),
            "num_severe_mismatch": n_severe,
            "num_moderate_mismatch": n_moderate,
            "total_scenarios": len(by_ckey),
            "formulas_largest": ",".join(sys_large) if sys_large else "",
            "formulas_smallest": ",".join(sys_small) if sys_small else "",
        })

        for fid2 in fids:
            fvs = [r["output_value"] for r in nr if r["formula_id"] == fid2]
            n_total = len(fvs)
            n_invalid = sum(1 for v in fvs if math.isnan(v) or math.isinf(v) or v < -EPS)
            cap_ratio = _cap_ratio(rows, node, fid2)
            flags.append({
                "node": node, "formula_id": fid2,
                "severe_scale_mismatch": n_severe > 0,
                "moderate_scale_mismatch": n_moderate > 0,
                "systematic_large": fid2 in sys_large,
                "systematic_small": fid2 in sys_small,
                "high_cap_ratio": cap_ratio > 0.5,
                "invalid_output": n_invalid > 0,
                "n_invalid": n_invalid, "n_total": n_total,
                "cap_ratio": round(cap_ratio, 4),
            })

    return summaries, flags


def _cap_ratio(rows, node, fid):
    """Fraction of outputs that hit the physical upper bound."""
    primary = {"snow": "M", "recharge": "R", "aet": "ET",
               "response": "Q", "routing": "weights_sum"}[node]
    relevant = [r for r in rows if r["node"] == node and r["formula_id"] == fid
                and r["output_name"] == primary
                and not r.get("error") and not math.isnan(r["output_value"])]
    if not relevant:
        return 0.0

    capped = 0
    for r in relevant:
        sd = eval(r["state_variables"])
        val = r["output_value"]
        bound = _upper_bound(node, fid, sd)
        if bound is not None and bound > EPS and val >= 0.999 * bound:
            capped += 1
    return capped / len(relevant)


def _upper_bound(node, fid, sd):
    if node == "snow":
        return sd.get("SWE", None)
    elif node == "recharge":
        return sd.get("I", None)
    elif node == "aet":
        pet = sd.get("PET", sd.get("PET_m", float("inf")))
        sm = sd.get("SM", float("inf"))
        return min(pet, sm)
    elif node == "response":
        if fid in ("response_two_reservoir", "response_smooth_threshold", "response_nonlinear"):
            return sd.get("SUZ", 0) + sd.get("SLZ", 0)
        elif fid == "response_single_reservoir":
            return sd.get("S", 0)
        elif fid == "response_two_parallel":
            return sd.get("S_f", 0) + sd.get("S_s", 0)
        elif fid == "response_delayed_step":
            return sd.get("S_1", 0) + sd.get("S_2", 0)
        return None
    elif node == "routing":
        return 1.0
    return None


# ---------------------------------------------------------------------------
# CSV / Report writers
# ---------------------------------------------------------------------------

def write_raw_csv(rows, path):
    keys = ["node", "formula_id", "scenario_id", "parameter_case",
            "parameter_name_varied", "parameter_quantile",
            "state_variables", "parameter_values", "output_name", "output_value"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary_csv(summaries, path):
    if not summaries:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)


def write_flags_csv(flags, path):
    if not flags:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flags[0]))
        w.writeheader()
        w.writerows(flags)


def write_report(summaries, flags, path, pool):
    L = []
    L.append(f"# Formula Scale Audit Report  (pool = {pool})\n")
    L.append("## 1. Purpose\n")
    L.append("Check whether candidate formulas within the same HBV process node "
             "produce outputs in comparable magnitude ranges. "
             "Identifies scale mismatches and recommends which formulas "
             "can share a dense MoE mixture pool vs. require hard routing.\n")

    L.append("## 2. Candidate Formulas\n")
    for node in ["snow", "recharge", "aet", "response", "routing"]:
        formulas = _pool_formulas(node, pool)
        policy = FORMULA_REGISTRY.get(node, {}).get("routing_policy", "N/A") if node != "routing" else "N/A"
        L.append(f"### {node}  (routing_policy: {policy}, {len(formulas)} formulas)\n")
        for fid, status in formulas:
            L.append(f"- `{fid}`  (status: {status})")
        L.append("")

    L.append("## 3. Parameter Ranges\n")
    for group, entries in PARAMETER_RANGES.items():
        L.append(f"### {group}\n")
        L.append("| Parameter | Range | Source |")
        L.append("|---|---|---|")
        for k, v in entries.items():
            L.append(f"| {k} | [{v['range'][0]}, {v['range'][1]}] | {v['source']} |")
        L.append("")

    L.append("## 4. State Grid Summary\n")
    L.append("State grids vary by formula. Key ranges:\n")
    L.append("- Snow T: [-5, 5], SWE: [0, 100], doy: [15, 355]")
    L.append("- Recharge I: [0, 80], SM_frac: [0, 1]")
    L.append("- AET PET: [0, 10], SM_frac: [0, 1], T_t: [-5, 30]")
    L.append("- Response SUZ: [0, 150], SLZ: [0, 300]")
    L.append("- Routing length: [7, 15, 30]\n")

    L.append("## 5. Summary Results\n")
    L.append("| Node | Scenarios | Max log10 | Median log10 | Severe | Moderate | Largest | Smallest |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in summaries:
        L.append(f"| {s['node']} | {s['num_scenarios']} | {s['max_log10_ratio']} | "
                 f"{s['median_log10_ratio']} | {s['num_severe_mismatch']} | "
                 f"{s['num_moderate_mismatch']} | {s['formulas_largest']} | "
                 f"{s['formulas_smallest']} |")
    L.append("")

    L.append("### Cap Ratios\n")
    L.append("| Node | Formula | Cap Ratio | High Cap? | Invalid |")
    L.append("|---|---|---|---|---|")
    for f in sorted(flags, key=lambda x: (x["node"], x["formula_id"])):
        L.append(f"| {f['node']} | {f['formula_id']} | {f['cap_ratio']:.4f} | "
                 f"{f['high_cap_ratio']} | {f['invalid_output']} |")
    L.append("")

    L.append("## 6. Node-level Findings\n")
    for node in ["snow", "recharge", "aet", "response", "routing"]:
        nf = [f for f in flags if f["node"] == node]
        ns = [s for s in summaries if s["node"] == node]
        L.append(f"### {node}\n")
        if ns:
            s = ns[0]
            L.append(f"- Valid-output scenarios: {s['num_scenarios']}")
            L.append(f"- Max log10 ratio: {s['max_log10_ratio']}")
            L.append(f"- Median log10 ratio: {s['median_log10_ratio']}")
            sev = s["num_severe_mismatch"] > 0
            L.append(f"- **Severe mismatch: {'YES' if sev else 'NO'}** "
                     f"({s['num_severe_mismatch']} severe, {s['num_moderate_mismatch']} moderate)")
            if s["formulas_largest"]:
                L.append(f"- Systematically largest: {s['formulas_largest']}")
            if s["formulas_smallest"]:
                L.append(f"- Systematically smallest: {s['formulas_smallest']}")
            for f in nf:
                if f["high_cap_ratio"]:
                    L.append(f"- `{f['formula_id']}`: HIGH cap ratio ({f['cap_ratio']:.2f})")
                if f["invalid_output"]:
                    L.append(f"- `{f['formula_id']}`: {f['n_invalid']}/{f['n_total']} INVALID outputs")
        L.append("")

    L.append("## 7. Recommendations\n")
    L.append("### Formula Pool Status\n")
    for node in ["snow", "recharge", "aet", "response"]:
        downgraded = _downgraded_formulas(node)
        if downgraded:
            L.append(f"#### {node}\n")
            for fid, status, reason in downgraded:
                L.append(f"- `{fid}` downgraded to **{status}**: {reason}")
            L.append("")

    L.append("### Recharge Scale Mismatch Assessment\n")
    L.append("Recharge severe mismatch mainly arises from parameter-regime variants "
             "of the HBV beta formula rather than distinct empirical formulas. "
             "Therefore, linear, weak-beta, and strong-beta recharge are excluded "
             "from the main Formula-MoE pool and retained only for ablation.\n")

    L.append("### Same-pool (dense mixing OK)\n")
    L.append("- Formulas without severe mismatch and low cap ratios can share a softmax-gated mixture.\n")
    L.append("### Hard-routing only\n")
    L.append("- Formulas with systematic large/small bias or high cap ratios should use one-hot routing.\n")
    L.append("### Postpone / revise\n")
    L.append("- Formulas with invalid outputs or extreme cap ratios may need range narrowing or revision.\n")

    with open(path, "w") as f:
        f.write("\n".join(L))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="HBV formula scale audit")
    ap.add_argument("--pool", choices=["main", "all", "ablation"], default="main",
                    help="Which formula pool to audit (default: main)")
    args = ap.parse_args()
    pool = args.pool

    out_dir = _pool_paths(pool)
    print(f"Running formula scale audit [pool={pool}] -> {out_dir}")

    rows = run_audit(pool)
    print(f"Collected {len(rows)} raw output rows.")

    raw_path = out_dir / "formula_scale_raw_outputs.csv"
    write_raw_csv(rows, raw_path)
    print(f"Raw outputs: {raw_path}")

    comparison_rows = compute_cross_comparison(pool)
    print(f"Cross-comparison rows: {len(comparison_rows)}")
    summaries, flags = compute_summary(rows, comparison_rows)

    sp = out_dir / "formula_scale_summary.csv"
    write_summary_csv(summaries, sp)
    print(f"Summary: {sp}")

    fp = out_dir / "formula_scale_flags.csv"
    write_flags_csv(flags, fp)
    print(f"Flags: {fp}")

    rp = out_dir / "formula_scale_audit_report.md"
    write_report(summaries, flags, rp, pool)
    print(f"Report: {rp}")

    print("\nFormula scale audit completed.")
    print(f"Pool: {pool}")
    print(f"Raw outputs: {raw_path}")
    print(f"Summary: {sp}")
    print(f"Flags: {fp}")
    print(f"Report: {rp}")


if __name__ == "__main__":
    main()
