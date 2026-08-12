#!/usr/bin/env python3
"""Exhaustive formula-combination forward-stability benchmark."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.hbv_formula_static import HbvFormulaStatic
from model.formula_pool import CandidateFormulaPool

OUTPUT_DIR = _PROJECT / "validation_results" / "formula_combination_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NODES = ["snow", "recharge", "aet", "response"]

SYNTHETIC_CASES = {
    "case_01_dry": {
        "P": [0.0] * 10 + [2.0, 0.0, 0.0] * 10 + [0.0] * 50 + [1.0] * 5 + [0.0] * 85,
        "T": [15.0] * 20 + [20.0] * 40 + [25.0] * 40 + [18.0] * 60,
        "PET": [4.0] * 60 + [6.0] * 40 + [5.0] * 60,
    },
    "case_02_wet": {
        "P": [5.0, 10.0, 20.0, 15.0, 8.0] * 32,
        "T": [5.0] * 40 + [10.0] * 40 + [15.0] * 40 + [12.0] * 40,
        "PET": [1.0] * 40 + [2.0] * 40 + [3.0] * 40 + [2.0] * 40,
    },
    "case_03_snow_dominated": {
        "P": [5.0] * 80 + [10.0, 20.0, 30.0, 25.0, 15.0] * 16,
        "T": [-5.0] * 60 + [-2.0] * 20 + [0.0] * 20 + [2.0] * 20 + [8.0] * 40,
        "PET": [0.5] * 80 + [1.0] * 40 + [2.0] * 40,
    },
    "case_04_rainfall_event": {
        "P": [0.0] * 40 + [80.0, 40.0, 20.0, 10.0] + [0.0] * 116,
        "T": [12.0] * 160,
        "PET": [2.0] * 80 + [1.0] * 80,
    },
    "case_05_mixed_seasonal": {
        "P": ([0.0] * 20 + [2.0, 5.0, 10.0, 8.0, 3.0, 0.0] * 4) * 3,
        "T": [0.0] * 40 + [15.0] * 40 + [3.0] * 40 + ([8.0] * 20 + [18.0] * 20) * 3,
        "PET": [1.0] * 40 + [4.0] * 40 + [2.0] * 40 + [3.0] * 40,
    },
}


def _pad(lst, target):
    if len(lst) >= target:
        return lst[:target]
    result = []
    while len(result) < target:
        result.extend(lst)
    return result[:target]


def _make_synthetic(case_def, length=160):
    P = torch.tensor(_pad(case_def["P"], length), dtype=torch.float64)
    T = torch.tensor(_pad(case_def["T"], length), dtype=torch.float64)
    PET = torch.tensor(_pad(case_def["PET"], length), dtype=torch.float64)
    return P, T, PET


def run_benchmark():
    pool = CandidateFormulaPool()
    node_formulas = {n: pool.formulas(n, "main") for n in NODES}

    combos = []
    for sn in node_formulas["snow"]:
        for rc in node_formulas["recharge"]:
            for ae in node_formulas["aet"]:
                for rs in node_formulas["response"]:
                    combo_id = f"{sn}_{rc}_{ae}_{rs}"
                    combos.append({"combo_id": combo_id, "snow_id": sn, "recharge_id": rc,
                                   "aet_id": ae, "response_id": rs,
                                   "is_default_hbv": combo_id == "S0_R0_E0_Q0"})

    forward_rows = []
    failure_rows = []
    metric_rows = []

    for combo in combos:
        combo_id = combo["combo_id"]
        fc = {n: combo[f"{n}_id"] for n in NODES}
        model = HbvFormulaStatic(formula_config=fc, warm_up=40, nearzero=1e-5)

        for case_id, case_def in SYNTHETIC_CASES.items():
            P, T, PET = _make_synthetic(case_def, length=160)
            doy = torch.arange(1, len(P) + 1, dtype=torch.float64)

            try:
                diag = model.simulate(P, T, PET, doy)
            except Exception as exc:
                failure_rows.append({"combo_id": combo_id, "case_id": case_id,
                                     "success": False, "failed_reason": f"exception: {exc}"})
                forward_rows.append({"combo_id": combo_id, "case_id": case_id, "success": False,
                                     "failed_reason": f"exception: {exc}",
                                     "nan_qsim": True, "inf_qsim": False, "negative_qsim": False,
                                     "max_qsim": 0.0, "mean_qsim": 0.0, "water_balance_error": 1.0})
                continue

            Qsim = diag["Qsim"]
            if Qsim.dim() == 0:
                Qsim = Qsim.unsqueeze(0)

            wb_err = diag["relative_water_balance_error"]
            max_q = Qsim.max().item()
            mean_q = Qsim.mean().item()
            has_nan = bool(torch.isnan(Qsim).any())
            has_inf = bool(torch.isinf(Qsim).any())
            has_neg = bool((Qsim < -1e-6).any())

            failed = False
            failed_reason = ""
            if has_nan:
                failed = True
                failed_reason = "nan_in_output_or_state"
            elif has_inf:
                failed = True
                failed_reason = "inf_in_output_or_state"
            elif has_neg:
                failed = True
                failed_reason = f"negative_state:{diag['state_negative_count']}"
            elif wb_err > 0.10:
                failed = True
                failed_reason = f"wb_error={wb_err:.4f}>0.10"
            elif max_q > 1e5:
                failed = True
                failed_reason = f"max_qsim={max_q:.1f}>1e5"

            warning = ""
            if not failed and wb_err > 0.05:
                warning = f"wb_warning:{wb_err:.4f}"
            if not failed and Qsim.sum() < 0.01 * P[40:].sum():
                warning += " low_flow"

            forward_rows.append({
                "combo_id": combo_id, "case_id": case_id,
                "success": not failed, "failed_reason": failed_reason,
                "warning": warning,
                "nan_qsim": has_nan, "inf_qsim": has_inf, "negative_qsim": has_neg,
                "max_qsim": round(max_q, 4), "mean_qsim": round(mean_q, 4),
                "water_balance_error": round(wb_err, 6),
            })

            if failed:
                failure_rows.append({"combo_id": combo_id, "case_id": case_id,
                                     "success": False, "failed_reason": failed_reason})

            metric_rows.append({
                "combo_id": combo_id, "case_id": case_id,
                "max_qsim": round(max_q, 4), "mean_qsim": round(mean_q, 4),
                "wb_error": round(wb_err, 6), "success": not failed,
            })

    _write_csv(forward_rows, OUTPUT_DIR / "combination_forward_summary.csv",
               ["combo_id", "case_id", "success", "failed_reason", "warning",
                "nan_qsim", "inf_qsim", "negative_qsim", "max_qsim", "mean_qsim",
                "water_balance_error"])
    _write_csv(failure_rows, OUTPUT_DIR / "combination_failure_log.csv",
               ["combo_id", "case_id", "success", "failed_reason"])
    _write_csv(metric_rows, OUTPUT_DIR / "combination_metrics.csv",
               ["combo_id", "case_id", "max_qsim", "mean_qsim", "wb_error", "success"])

    n_total = len(forward_rows)
    n_ok = sum(1 for r in forward_rows if r["success"])
    n_fail = n_total - n_ok
    n_warn = sum(1 for r in forward_rows if r.get("warning"))
    n_default_ok = sum(1 for r in forward_rows if r["combo_id"] == "S0_R0_E0_Q0" and r["success"])

    report = [
        "# Formula Combination Benchmark Report",
        "",
        "## 1. Purpose",
        "Exhaustive forward-stability check with full water balance.",
        "",
        f"## 2. Summary",
        f"- Total combinations: {len(combos)}",
        f"- Total test cases: {n_total}",
        f"- Successful: {n_ok}",
        f"- Failed: {n_fail}",
        f"- Warnings: {n_warn}",
        f"- Default HBV (S0_R0_E0_Q0) OK: {n_default_ok}/5",
    ]
    if failure_rows:
        report.append("\n## 3. Failed Combinations")
        seen = set()
        for fr in failure_rows:
            key = fr["combo_id"]
            if key not in seen:
                seen.add(key)
                reasons = [r["failed_reason"] for r in failure_rows if r["combo_id"] == key]
                report.append(f"- {key}: {', '.join(set(reasons))}")
    else:
        report.append("\n## 3. Failed Combinations\nNone.")

    rp = OUTPUT_DIR / "combination_benchmark_report.md"
    rp.write_text("\n".join(report))

    print(f"Benchmark: {n_ok}/{n_total} OK, {n_fail} failed, {n_warn} warnings")
    print(f"Default HBV: {'OK' if n_default_ok == 5 else 'FAIL'}")
    if failure_rows:
        print("Unique failing combos:", len(set(fr["combo_id"] for fr in failure_rows)))


def _write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    run_benchmark()
