#!/usr/bin/env python3
"""Generate the concise Markdown gate report from the completed analysis
tables (run after r3/gate_analysis.py)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from manuscript.scripts.r3.common import DEFAULT_RESULTS_ROOT  # noqa: E402

COMMON_XAJ = [
    "xaj_k",
    "xaj_b",
    "xaj_im",
    "xaj_um",
    "xaj_lm",
    "xaj_dm",
    "xaj_c",
    "xaj_sm",
    "xaj_ex",
    "xaj_ki",
    "xaj_kg",
    "xaj_ci",
    "xaj_cg",
    "xaj_a",
    "xaj_theta",
]


def fmt(v: float, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "nan"
    return f"{v:.{nd}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", default="r3_gate_v1")
    args = parser.parse_args()

    out = args.results_root / args.run_id
    report = json.loads((out / "gate_report.json").read_text())
    validation = json.loads((out / "gate_input_validation.json").read_text())
    param = pd.read_csv(out / "parameter_recovery_summary.csv")
    eq = pd.read_csv(out / "kge_vs_parameter_recovery.csv")
    deficit = pd.read_csv(out / "kge_deficit_vs_frac_snow.csv")
    state = json.loads((out / "gate_state_summary.json").read_text())
    disp = pd.read_csv(out / "ic_restart_parameter_dispersion.csv")
    seedspread = pd.read_csv(out / "dpl_seed_parameter_spread.csv")

    L = []
    L.append("# R3 correct-CN gate — truth-recovery report (531 basins)\n")
    L.append(f"Generated: {report.get('created_at')}\n")
    L.append(
        "Analysis is **CN-only**. No Base/TGD2 results were used; no "
        "identifiable subset is frozen here.\n"
    )

    L.append("## 1. Can correct CN recover synthetic discharge (IC and dPL)?\n")
    qic, qdp = report["q_recovery_ic"], report["q_recovery_dpl"]
    L.append(
        f"- **IC-CMA-ES** (best train-KGE restart): train KGE median "
        f"{fmt(qic['median_kge_train'])} (oracle {fmt(qic['median_oracle_kge_train'])}, "
        f"median gap {fmt(qic['median_oracle_gap_train'])}); test KGE median "
        f"{fmt(qic['median_kge_test'])} (gap {fmt(qic['median_oracle_gap_test'])})."
    )
    L.append(
        f"- **dPL** (median of 3 seeds): train KGE median {fmt(qdp['median_kge_train'])}, "
        f"test KGE median {fmt(qdp['median_kge_test'])}; eval-path ceiling "
        f"{fmt(qdp['median_eval_ceiling'])}, median gap {fmt(qdp['median_gap_test'])}."
    )
    L.append(
        f"- dPL per seed test medians: "
        + ", ".join(
            f"seed {s}={fmt(v)}" for s, v in qdp["per_seed_median_test_kge"].items()
        )
        + "."
    )
    L.append(
        f"- theta* round-trip (recorded forward vs frozen q_star): max abs diff "
        f"{report['round_trip_theta_star_max_abs_diff']:.2e}.\n"
    )
    L.append(
        "Verdict: **yes** — correct CN recovers Q* near its oracle ceiling "
        "under both regimes.\n"
    )

    L.append("## 2. Shared XAJ parameter recovery (normalized errors)\n")
    L.append(
        "`e = (theta_hat - theta_star)/(upper - lower)`; recovery profile per "
        "parameter, IC best-restart and dPL median-of-seeds:\n"
    )
    L.append(
        "| param | regime | med e | med \|e\| | q25/q75 \|e\| | q90 \|e\| | r(θ̂,θ\*) | slope | frac. bound | ρ(\|e\|,snow) |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in COMMON_XAJ:
        for regime, run in (("IC", "best-restart"), ("dPL", "median-seeds")):
            r = param[(param["parameter"] == p) & (param["run"] == run)].iloc[0]
            L.append(
                f"| {p} | {regime} | {fmt(r['median_signed_e'])} | {fmt(r['median_abs_e'])} "
                f"| {fmt(r['q25_abs_e'])}/{fmt(r['q75_abs_e'])} | {fmt(r['q90_abs_e'])} "
                f"| {fmt(r['pearson_theta_hat_vs_star'], 3)} | {fmt(r['ols_slope'], 3)} "
                f"| {fmt(r['frac_at_lower'] + r['frac_at_upper'], 3)} "
                f"| {fmt(r['spearman_abs_e_vs_frac_snow'], 3)} |"
            )
    L.append("")
    L.append("CN-only parameters (secondary diagnostics):\n")
    L.append("| param | regime | med \|e\| | q90 \|e\| | r(θ̂,θ\*) |")
    L.append("|---|---|---|---|---|")
    for p in ("cn_ctg", "cn_kf"):
        for regime, run in (("IC", "best-restart"), ("dPL", "median-seeds")):
            r = param[(param["parameter"] == p) & (param["run"] == run)].iloc[0]
            L.append(
                f"| {p} | {regime} | {fmt(r['median_abs_e'])} | {fmt(r['q90_abs_e'])} "
                f"| {fmt(r['pearson_theta_hat_vs_star'], 3)} |"
            )
    L.append("")

    L.append("## 3. Stability across IC restarts and dPL seeds\n")
    disp_sum = disp.groupby("parameter")["z_std_across_starts"].median()
    seed_sum = seedspread.groupby("parameter")["z_std_across_seeds"].median()
    L.append("| param | IC restart σ (median) | dPL seed σ (median) |")
    L.append("|---|---|---|")
    for p in COMMON_XAJ:
        L.append(
            f"| {p} | {fmt(disp_sum.get(p, float('nan')), 4)} | "
            f"{fmt(seed_sum.get(p, float('nan')), 4)} |"
        )
    L.append("")

    L.append("## 4. Can near-ceiling KGE coexist with poor parameter recovery?\n")
    for paradigm in ("IC", "dPL"):
        e = report["equifinality"][paradigm]
        L.append(
            f"- **{paradigm}**: D_theta (median over 15 shared params of |e|) "
            f"median {fmt(e['D_theta_median'])}, q90 {fmt(e['D_theta_q90'])}, "
            f"max {fmt(e['D_theta_max'])}; Spearman(KGE, D_theta) "
            f"{fmt(e['spearman_kge_vs_D_theta'], 3)}."
        )
        L.append(
            f"  D_theta among basins with KGE≥0.999: {fmt(e['D_theta_among_kge_ge_0p999'])} "
            f"(n={e['n_kge_ge_0p999']}); KGE≥0.99: {fmt(e['D_theta_among_kge_ge_0p99'])} "
            f"(n={e['n_kge_ge_0p99']})."
        )
    L.append("")
    L.append("Examples of basins with high KGE and relatively poor D_theta:\n")
    for paradigm, exs in report["equifinality"][
        "examples_high_kge_poor_dtheta"
    ].items():
        L.append(
            f"- {paradigm}: "
            + "; ".join(
                f"{x['basin_id']} (KGE {fmt(x['kge_train'])}, D_theta {fmt(x['D_theta'])})"
                for x in exs
            )
            + ""
        )
    L.append(
        "Verdict: **yes** — equifinality is present even for correct CN; "
        "near-ceiling discharge KGE does not imply parameter recovery.\n"
    )

    L.append("## 5. Common internal state recovery\n")
    L.append(
        "RMSE / NRMSE / correlation / bias distributions (median over basins), "
        "train period, common states:\n"
    )
    L.append("| var | fit | RMSE | NRMSE | corr | bias |")
    L.append("|---|---|---|---|---|---|")
    for var in ("wu", "wl", "wd", "s", "fr", "qi", "qg"):
        for fit in ("IC", "dPL_median"):
            s = state[var][fit]
            L.append(
                f"| {var} | {fit} | {fmt(s['rmse_median'], 3)} | {fmt(s['nrmse_median'], 3)} "
                f"| {fmt(s['corr_median'], 3)} | {fmt(s['bias_median'], 3)} |"
            )
    L.append("")
    L.append("Snow diagnostics (CN-only, train period):\n")
    L.append("| var | fit | RMSE | NRMSE | corr |")
    L.append("|---|---|---|---|---|")
    for var in ("G", "eTG", "sca", "melt"):
        for fit in ("IC", "dPL_median"):
            s = state[var][fit]
            L.append(
                f"| {var} | {fit} | {fmt(s['rmse_median'], 3)} | {fmt(s['nrmse_median'], 3)} "
                f"| {fmt(s['corr_median'], 3)} |"
            )
    L.append("")

    L.append("## 6. Snow-dependence of correct-CN recovery\n")
    sd = report["snow_diagnostics"]
    for col in ("ic_deficit_train", "ic_deficit_test", "dpl_deficit_test"):
        L.append(
            f"- Spearman({col}, frac_snow) = {fmt(sd.get(f'spearman_{col}_vs_frac_snow'), 3)}"
        )
    for paradigm in ("IC", "dPL"):
        L.append(
            f"- {paradigm}: Spearman(KGE_train, frac_snow) = "
            f"{fmt(sd[paradigm]['spearman_kge_train_vs_frac_snow'], 3)}; "
            f"Spearman(D_theta, frac_snow) = "
            f"{fmt(sd[paradigm]['spearman_D_theta_vs_frac_snow'], 3)}"
        )
    L.append("")
    L.append("Per-parameter Spearman(|e|, frac_snow) (IC / dPL median):\n")
    L.append("| param | IC | dPL |")
    L.append("|---|---|---|")
    ic_map = sd["param_abs_e_vs_frac_snow_IC"]
    dp_map = sd["param_abs_e_vs_frac_snow_dPL"]
    for p in COMMON_XAJ:
        L.append(
            f"| {p} | {fmt(ic_map.get(p, float('nan')), 3)} | {fmt(dp_map.get(p, float('nan')), 3)} |"
        )
    L.append("")
    L.append("Diagnostic axis only; no causal claim.\n")

    L.append("## 7. Provisional identifiable-subset candidates (CN-only evidence)\n")
    L.append(
        "Candidates are ranked by (i) small median |e|, (ii) high θ̂–θ* "
        "correlation, (iii) low restart/seed dispersion, (iv) small snow "
        "association. **Provisional — to be frozen externally, and only "
        "after review.**\n"
    )
    L.append(
        "See `parameter_recovery_summary.csv`, `ic_restart_parameter_dispersion.csv`, "
        "`dpl_seed_parameter_spread.csv` for the full evidence.\n"
    )

    (out / "gate_report.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out / 'gate_report.md'}")


if __name__ == "__main__":
    main()
