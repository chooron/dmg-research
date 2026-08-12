#!/usr/bin/env python3
"""Formula gradient audit for HBV Formula-MoE.

Checks gradient stability of every main-pool formula under controlled
parameter quantile sweeps and state grids.  Flags NaN/Inf gradients,
zero-gradient dominance, large gradients, and cross-formula gradient
mismatch within the same process node.

Run from project root:
    python scripts/audit_formula_gradients.py
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

from model.formula_pool import CandidateFormulaPool, _FORMULA_META
from model.flux.parameter_ranges import PARAMETER_RANGES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = _PROJECT / "validation_results" / "formula_gradient_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
EPS = 1e-8

PRIMARY_OUTPUT_INDEX = {
    "S0": None, "S4": None, "S5": None,  # single tensor
    "R0": None, "R4": None, "R5": None,
    "E0": None, "E3": None, "E4": None,
    "Q0": 3,  # (Q0, Q1, Q2, Q_total) -> index 3
    "Q2": 2,  # (Q_uz, Q_lz, Q_total) -> index 2
    "Q5": 4,  # (R_imm, R_del, Q_1, Q_2, Q_total) -> index 4
}

# Classification of each arg: "param" or "state"
_ARG_TYPE = {
    # Snow S0
    "T": "state", "TT": "param", "CFMAX": "param", "SWE": "state",
    # Snow S4
    "CFMAX_0": "param", "a_s": "param", "phi_s": "param", "doy": "state",
    # Snow S5
    "c_m": "param",
    # Recharge
    "I": "state", "SM": "state", "FC": "param", "beta": "param",
    "a_r": "param", "c_r": "param", "b_v": "param",
    # AET
    "PET": "state", "LP": "param", "gamma_E": "param",
    "s_w": "param", "s_o": "param",
    # Response
    "SUZ": "state", "SLZ": "state", "K_0": "param", "K_1": "param", "K_2": "param", "UZL": "param",
    "alpha_Q": "param", "R_in": "state", "S_1": "state", "S_2": "state", "PART": "param",
}


def _arg_class(name):
    return _ARG_TYPE.get(name, "state")


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------
_RANGE_FLAT = {}
for _group, _entries in PARAMETER_RANGES.items():
    for _key, _info in _entries.items():
        _RANGE_FLAT[_key] = _info["range"]

# Aliases: formula arg name -> PARAMETER_RANGES key
_RANGE_ALIASES = {
    "CFMAX_0": "CFMAX",
    "K_0": "K0",
    "K_1": "K1",
    "K_2": "K2",
    "beta": "BETA",
}


def get_range(name):
    key = _RANGE_ALIASES.get(name, name)
    if key not in _RANGE_FLAT:
        raise KeyError(f"Parameter '{name}' (key='{key}') not in PARAMETER_RANGES.")
    return _RANGE_FLAT[key]


def param_quantiles(name):
    lo, hi = get_range(name)
    return {q: lo + q * (hi - lo) for q in QUANTILES}


def median_val(name):
    return param_quantiles(name)[0.50]


# ---------------------------------------------------------------------------
# State grids per node (compact, to avoid explosion)
# ---------------------------------------------------------------------------
SNOW_STATE = {
    "T": [-5.0, -2.0, 0.0, 1.0, 3.0, 5.0],
    "SWE": [0.0, 5.0, 20.0, 100.0],
    "doy": [80, 172, 266],
}
RECHARGE_STATE = {
    "I": [0.0, 1.0, 5.0, 20.0, 80.0],
    "SM": [0.0, 10.0, 75.0, 200.0, 400.0],
}
AET_STATE = {
    "PET": [0.0, 1.0, 3.0, 6.0, 10.0],
    "SM": [0.0, 10.0, 100.0, 200.0, 400.0],
}
RESPONSE_STATE = {
    "SUZ": [0.0, 1.0, 10.0, 50.0, 150.0],
    "SLZ": [0.0, 1.0, 20.0, 100.0, 300.0],
    "R_in": [0.0, 1.0, 10.0, 50.0],
    "S_1": [0.0, 1.0, 10.0, 50.0],
    "S_2": [0.0, 1.0, 20.0, 100.0],
}

NODE_STATE = {"snow": SNOW_STATE, "recharge": RECHARGE_STATE,
              "aet": AET_STATE, "response": RESPONSE_STATE}

NODE_PARAM_DEFAULTS = {
    # Default fix values for state vars not varied
    "FC": 275.0, "LP": 0.65, "beta": 3.5,
}


# ---------------------------------------------------------------------------
# Parameter case generation
# ---------------------------------------------------------------------------
def _param_cases(param_names):
    qv = {n: param_quantiles(n) for n in param_names}
    cases = []
    cases.append(("median_all", {n: qv[n][0.50] for n in param_names}, None))
    for n in param_names:
        for q in QUANTILES:
            p = {pn: qv[pn][0.50] for pn in param_names}
            p[n] = qv[n][q]
            cases.append((f"oat_{n}_{q}", p, n))
    for cq, cn in [(0.25, "low_all"), (0.50, "median_all"), (0.75, "high_all"),
                    (0.05, "extreme_low"), (0.95, "extreme_high")]:
        cases.append((cn, {n: qv[n][cq] for n in param_names}, None))
    # dedup
    seen = set()
    uniq = []
    for c in cases:
        key = (c[0], tuple(sorted(c[1].items())))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def _state_combos(state_keys, node):
    grid = NODE_STATE[node]
    keys = [k for k in state_keys if k in grid]
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


# ---------------------------------------------------------------------------
# Primary output extractor
# ---------------------------------------------------------------------------
def _primary_output(fid, result):
    idx = PRIMARY_OUTPUT_INDEX.get(fid)
    if idx is not None and isinstance(result, tuple):
        return result[idx]
    if isinstance(result, tuple):
        return result[-1]
    return result


# ---------------------------------------------------------------------------
# Audit loop
# ---------------------------------------------------------------------------
def run_gradient_audit():
    pool = CandidateFormulaPool()
    rows = []
    dtype = torch.float64

    for node in ["snow", "recharge", "aet", "response"]:
        for fid in pool.formulas(node, "main"):
            meta = _FORMULA_META[fid]
            arg_names = meta["args"]
            func = meta["func"]
            param_names = [n for n in arg_names if _arg_class(n) == "param"]
            state_names = [n for n in arg_names if _arg_class(n) == "state"]

            p_cases = _param_cases(param_names)

            for pcase_name, pvals, varied_name in p_cases:
                for sd in _state_combos(state_names, node):
                    # Build tensor dict
                    tdict = {}
                    for n in arg_names:
                        if n in pvals:
                            tdict[n] = torch.tensor(pvals[n], dtype=dtype, requires_grad=True)
                        elif n in sd:
                            tdict[n] = torch.tensor(sd[n], dtype=dtype, requires_grad=True)
                        else:
                            tdict[n] = torch.tensor(NODE_PARAM_DEFAULTS.get(n, 1.0), dtype=dtype,
                                                    requires_grad=(_arg_class(n) == "param"))

                    try:
                        pos_args = [tdict[name] for name in arg_names]
                        raw = func(*pos_args)
                    except Exception as exc:
                        row = _base_row(node, fid, pcase_name, varied_name, sd, pvals)
                        row["output_value"] = float("nan")
                        row["is_finite_output"] = False
                        row["is_finite_grad"] = False
                        row["error"] = str(exc)
                        rows.append(row)
                        continue

                    y = _primary_output(fid, raw)

                    # Check output finiteness
                    y_finite = bool(torch.isfinite(y).all()) if torch.is_tensor(y) else False
                    y_val = y.detach().item() if torch.is_tensor(y) and y.numel() == 1 else float("nan")

                    if not y_finite or math.isnan(y_val):
                        row = _base_row(node, fid, pcase_name, varied_name, sd, pvals)
                        row["output_value"] = y_val
                        row["is_finite_output"] = False
                        row["is_finite_grad"] = False
                        rows.append(row)
                        continue

                    # Backward
                    try:
                        loss = y.sum() if torch.is_tensor(y) else y
                        loss.backward()
                    except Exception:
                        row = _base_row(node, fid, pcase_name, varied_name, sd, pvals)
                        row["output_value"] = y_val
                        row["is_finite_output"] = True
                        row["is_finite_grad"] = False
                        rows.append(row)
                        for n in arg_names:
                            if tdict[n].grad is not None:
                                tdict[n].grad.zero_()
                        continue

                    # Check if output is capped
                    is_capped = _check_capped(node, fid, sd, pvals, y_val)

                    # Record gradients
                    for n in arg_names:
                        t = tdict[n]
                        g = t.grad.item() if t.grad is not None else 0.0
                        t.grad = None  # zero out for next iteration

                        abs_g = abs(g)
                        if abs(y_val) > EPS and math.isfinite(abs_g):
                            scaled_g = min(abs_g * abs(t.detach().item()) / abs(y_val), 1e9)
                        else:
                            scaled_g = 0.0 if abs_g < EPS else 1e9

                        row = _base_row(node, fid, pcase_name, varied_name, sd, pvals)
                        row["var_name"] = n
                        row["var_type"] = _arg_class(n)
                        row["var_value"] = t.detach().item()
                        row["output_value"] = y_val
                        row["grad_value"] = g
                        row["abs_grad"] = abs_g
                        row["scaled_grad"] = scaled_g
                        row["is_finite_output"] = True
                        row["is_finite_grad"] = math.isfinite(g) and math.isfinite(scaled_g)
                        row["is_zero_grad"] = abs_g < EPS
                        row["is_large_grad"] = abs_g > 1e3 or scaled_g > 1e3
                        row["is_capped"] = is_capped
                        rows.append(row)

    return rows


def _check_capped(node, fid, sd, pvals, y_val):
    """Heuristic: is output at a physical ceiling?"""
    if node == "snow":
        swe = sd.get("SWE", 100.0)
        return y_val >= 0.999 * swe and swe > EPS
    elif node == "recharge":
        I = sd.get("I", 0.0)
        return y_val >= 0.999 * I and I > EPS
    elif node == "aet":
        pet = sd.get("PET", 0.0)
        sm = sd.get("SM", 0.0)
        return (y_val >= 0.999 * pet and pet > EPS) or (y_val >= 0.999 * sm and sm > EPS)
    elif node == "response":
        if fid in ("Q0", "Q2",):
            suz = sd.get("SUZ", 0)
            slz = sd.get("SLZ", 0)
            return y_val >= 0.999 * (suz + slz) and (suz + slz) > EPS
        elif fid == "Q5":
            s1 = sd.get("S_1", 0)
            s2 = sd.get("S_2", 0)
            return y_val >= 0.999 * (s1 + s2) and (s1 + s2) > EPS
    return False


def _base_row(node, fid, pcase, varied_name, sd, pvals):
    return {
        "node": node,
        "formula_id": fid,
        "scenario_id": f"{node}_{fid}_{pcase}",
        "parameter_case": pcase,
        "parameter_name_varied": varied_name or "",
        "state_variables": str({k: v for k, v in sd.items()}),
        "parameter_values": str({k: v for k, v in pvals.items()}),
    }


# ---------------------------------------------------------------------------
# Summary & flags
# ---------------------------------------------------------------------------
def compute_summary(rows):
    summaries = []
    flags = []
    for node in ["snow", "recharge", "aet", "response"]:
        nr = [r for r in rows if r["node"] == node and not r.get("error")]
        fids = sorted(set(r["formula_id"] for r in nr))
        if not nr:
            continue

        n_records = len(nr)
        n_invalid = sum(1 for r in nr if not r["is_finite_output"] or not r["is_finite_grad"])
        n_large = sum(1 for r in nr if r.get("is_large_grad"))
        max_abs = max(r["abs_grad"] for r in nr)
        med_abs = sorted([r["abs_grad"] for r in nr])[len(nr) // 2]
        max_scaled = max(r["scaled_grad"] for r in nr)
        med_scaled = sorted([r["scaled_grad"] for r in nr])[len(nr) // 2]

        summaries.append({
            "node": node, "num_records": n_records,
            "invalid_gradient_count": n_invalid,
            "large_gradient_count": n_large,
            "max_abs_grad": round(max_abs, 4),
            "median_abs_grad": round(med_abs, 4),
            "max_scaled_grad": round(max_scaled, 4),
            "median_scaled_grad": round(med_scaled, 4),
        })

        # Per-formula flags
        for fid2 in fids:
            fr = [r for r in nr if r["formula_id"] == fid2]
            nf = len(fr)
            # zero-gradient dominant (only for param vars)
            pr = [r for r in fr if r["var_type"] == "param" and r["output_value"] > EPS]
            nz = sum(1 for r in pr if r["is_zero_grad"])
            zgd = len(pr) > 5 and nz / len(pr) > 0.7 if pr else False
            cap_zero = sum(1 for r in pr if r["is_zero_grad"] and r["is_capped"])
            flags.append({
                "node": node, "formula_id": fid2,
                "invalid_gradient": n_invalid > 0,
                "has_large_gradient": any(r.get("is_large_grad") for r in fr),
                "zero_gradient_dominant": zgd,
                "cap_induced_zero_count": cap_zero,
                "n_param_grad_records": len(pr),
                "n_zero_param_grad": nz,
            })

    # Cross-formula gradient mismatch per node
    _add_mismatch_flags(rows, flags)

    return summaries, flags


def _add_mismatch_flags(rows, flags):
    """Compare gradient magnitudes across formulas within same node/scenario."""
    for node in ["snow", "recharge", "aet", "response"]:
        nr = [r for r in rows if r["node"] == node and r["var_type"] == "param"
              and r["is_finite_grad"] and not r.get("error") and r["abs_grad"] > EPS]
        by_scen = defaultdict(list)
        for r in nr:
            by_scen[r["scenario_id"]].append(r)

        severe_n = 0
        moderate_n = 0
        for sid, srows in by_scen.items():
            fids_in_scen = set(r["formula_id"] for r in srows)
            if len(fids_in_scen) < 2:
                continue
            fid_grads = {}
            for r in srows:
                fid_grads.setdefault(r["formula_id"], []).append(r["abs_grad"])
            fid_max = {f: max(v) for f, v in fid_grads.items()}
            pos = {f: v for f, v in fid_max.items() if v > EPS}
            if len(pos) >= 2:
                mx = max(pos.values())
                mn = min(pos.values())
                lr = math.log10(mx + EPS) - math.log10(mn + EPS)
                if lr > 2:
                    severe_n += 1
                elif lr > 1:
                    moderate_n += 1
        # Update flags
        for f in flags:
            if f["node"] == node:
                f["severe_gradient_mismatch"] = severe_n > 0
                f["moderate_gradient_mismatch"] = moderate_n > 0
                f["severe_mm_count"] = severe_n
                f["moderate_mm_count"] = moderate_n


# ---------------------------------------------------------------------------
# CSV / Report
# ---------------------------------------------------------------------------
_RAW_FIELDS = ["node", "formula_id", "scenario_id", "parameter_case",
               "parameter_name_varied", "state_variables", "parameter_values",
               "var_name", "var_type", "var_value", "output_value",
               "grad_value", "abs_grad", "scaled_grad",
               "is_finite_output", "is_finite_grad", "is_zero_grad",
               "is_large_grad", "is_capped"]


def write_raw_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_RAW_FIELDS, extrasaction="ignore")
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


def write_report(summaries, flags, path):
    L = []
    L.append("# Formula Gradient Audit Report\n")
    L.append("## 1. Purpose\n")
    L.append("Check gradient stability of main-pool formulas for differentiable training.\n")

    L.append("## 2. Candidate Formulas\n")
    for node in ["snow", "recharge", "aet", "response"]:
        L.append(f"### {node}\n")
        for fid in _FORMULA_META:
            if fid[0].lower() == node[0] and fid in PRIMARY_OUTPUT_INDEX:
                meta = _FORMULA_META[fid]
                params = [n for n in meta["args"] if _arg_class(n) == "param"]
                states = [n for n in meta["args"] if _arg_class(n) == "state"]
                L.append(f"- `{fid}` — params: {params} | states: {states}")
        L.append("")

    L.append("## 3. Gradient Variables\n")
    L.append("All formula parameters and key state variables tracked with `requires_grad=True`.\n")

    L.append("## 4. Summary Table\n")
    L.append("| Node | Records | Invalid | Large Grad | Max Abs | Med Abs | Max Scaled | Med Scaled |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in summaries:
        L.append(f"| {s['node']} | {s['num_records']} | {s['invalid_gradient_count']} | "
                 f"{s['large_gradient_count']} | {s['max_abs_grad']} | {s['median_abs_grad']} | "
                 f"{s['max_scaled_grad']} | {s['median_scaled_grad']} |")
    L.append("")

    L.append("### Per-Formula Flags\n")
    L.append("| Node | Formula | Invalid | LargeGrad | ZeroGradDom | CapZero | SevereMM |")
    L.append("|---|---|---|---|---|---|---|")
    for f in sorted(flags, key=lambda x: (x["node"], x["formula_id"])):
        L.append(f"| {f['node']} | {f['formula_id']} | {f.get('invalid_gradient', False)} | "
                 f"{f.get('has_large_gradient', False)} | {f.get('zero_gradient_dominant', False)} | "
                 f"{f.get('cap_induced_zero_count', 0)} | {f.get('severe_gradient_mismatch', False)} |")
    L.append("")

    L.append("## 5. Node-level Findings\n")
    for node in ["snow", "recharge", "aet", "response"]:
        nf = [f for f in flags if f["node"] == node]
        ns = [s for s in summaries if s["node"] == node]
        L.append(f"### {node}\n")
        if ns:
            s = ns[0]
            L.append(f"- Records: {s['num_records']}")
            L.append(f"- Invalid gradients: {s['invalid_gradient_count']}")
            L.append(f"- Large gradients: {s['large_gradient_count']}")
            L.append(f"- Max abs_grad: {s['max_abs_grad']}, Max scaled_grad: {s['max_scaled_grad']}")
            for f in nf:
                notes = []
                if f.get("invalid_gradient"):
                    notes.append("HAS INVALID GRADIENTS")
                if f.get("has_large_gradient"):
                    notes.append("large gradients detected")
                if f.get("zero_gradient_dominant"):
                    notes.append(f"zero-gradient dominant ({f.get('cap_induced_zero_count', 0)} cap-induced)")
                if f.get("severe_gradient_mismatch"):
                    notes.append("severe cross-formula gradient mismatch")
                if notes:
                    L.append(f"- `{f['formula_id']}`: " + "; ".join(notes))
        L.append("")

    L.append("## 6. Recommendations\n")
    L.append("| Node | Recommendation |")
    L.append("|---|---|")
    for node in ["snow", "recharge", "aet", "response"]:
        rec = _recommendation(node, summaries, flags)
        L.append(f"| {node} | {rec} |")
    L.append("")

    with open(path, "w") as f:
        f.write("\n".join(L))


def _recommendation(node, summaries, flags):
    nf = [f for f in flags if f["node"] == node]
    has_invalid = any(f.get("invalid_gradient") for f in nf)
    has_large = any(f.get("has_large_gradient") for f in nf)
    has_zgd = any(f.get("zero_gradient_dominant") for f in nf)
    has_severe = any(f.get("severe_gradient_mismatch") for f in nf)

    if node == "recharge":
        base = "hard_routing_only"
    else:
        base = "safe_for_training" if not (has_invalid or has_severe) else "train_with_caution"

    if has_invalid:
        base += " (fix invalid gradients)"
    if has_zgd:
        base += " (expect zero-grad in saturated/capped regions)"
    if has_large:
        base += " (monitor gradient clipping)"
    return base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Running formula gradient audit ...")
    rows = run_gradient_audit()
    print(f"Collected {len(rows)} gradient records.")

    raw_path = OUTPUT_DIR / "formula_gradient_raw.csv"
    write_raw_csv(rows, raw_path)
    print(f"Raw: {raw_path}")

    summaries, flags = compute_summary(rows)

    sp = OUTPUT_DIR / "formula_gradient_summary.csv"
    write_summary_csv(summaries, sp)
    print(f"Summary: {sp}")

    fp = OUTPUT_DIR / "formula_gradient_flags.csv"
    write_flags_csv(flags, fp)
    print(f"Flags: {fp}")

    rp = OUTPUT_DIR / "formula_gradient_audit_report.md"
    write_report(summaries, flags, rp)
    print(f"Report: {rp}")

    print("\nFormula gradient audit completed.")
    for p in [raw_path, sp, fp, rp]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
