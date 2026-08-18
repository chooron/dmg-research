#!/usr/bin/env python3
"""R3 main-text summary table generator (Table 5).

Produces one compact main-text table that provides the numerical anchors for
R3 Figures 5 and 6.  All values are read from the frozen-aligned prepared
summaries in manuscript/results/R3/ (figure5_summary.json and
figure6_summary.json, which already assert equality with the canonical frozen
post-hoc summaries); where a CI is not stored in those summaries it is
recomputed with the repository R3 paired-basin bootstrap (2000 replicates,
seed 20260730, median).

Outputs (following the manuscript table convention of generate_table1.py):
  manuscript/tables/Table5_R3_summary.md        (Markdown)
  manuscript/tables/Table5_R3_summary.tex       (LaTeX, threeparttable)
  manuscript/stats/tables/Table5_R3_summary.md  (mirror)
  manuscript/stats/tables/Table5_R3_summary.tex (mirror)
  manuscript/results/R3/table5_main_summary.csv (machine-readable source)

Usage: python manuscript/scripts/r3/generate_table_r3_main.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]  # manuscript/scripts -> project/hydrodiag
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from manuscript.r3.common import DEFAULT_RESULTS_ROOT  # noqa: E402

RES_R3 = PROJECT / "manuscript" / "results" / "R3"
TABLES_DIR = PROJECT / "manuscript" / "tables"
STATS_DIR = PROJECT / "manuscript" / "stats" / "tables"
CANON = DEFAULT_RESULTS_ROOT / "r3_misspec_analysis_v1"

BOOT_N = 2000
BOOT_SEED = 20260730

OUT_STEM = "Table5_R3_summary"
CSV_NAME = "table5_main_summary.csv"


def format_stat(med, lo, hi, decimals=3):
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(med)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def boot_ci(values, stat_fn=np.median, n_boot=BOOT_N, seed=BOOT_SEED):
    """Paired basin-level bootstrap CI — repository R3 convention."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = stat_fn(values[idx])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def load_anchors():
    """Frozen canonical anchors used for validation (IC values must match exactly)."""
    psum = json.loads((CANON / "posthoc_summary.json").read_text())
    pval = json.loads((CANON / "posthoc_validation_summary.json").read_text())
    return psum, pval


def build_rows(d5, d6, psum, pval, seedmed):
    """Return list of row dicts: estimand, role, IC/DPL med + CI + n, decimals, source."""
    pc = d5["panel_c_f_close"]
    dd = d5["panel_d_decay"]
    pb = d6["panel_b_f_tgd2"]
    prth = d6["panel_c_r_theta"]
    prst = d6["panel_d_r_state"]
    pe = d6["panel_e_residual_vs_frac_snow"]
    pf = d6["panel_f_process_errors"]

    # residual CI: IC from frozen V4 (canonical); dPL recomputed on the
    # per-basin seed-median values (display convention, matches Figure 6 (e))
    g_ic = (
        seedmed.loc[seedmed["paradigm"] == "IC", "G_CN_over_TGD2"].dropna().to_numpy()
    )
    g_dpl = (
        seedmed.loc[seedmed["paradigm"] == "dPL", "G_CN_over_TGD2"].dropna().to_numpy()
    )
    ci_ic_res = pval["V4"]["IC_test"]["G_CN_over_TGD2"]["boot_ci_median"]
    ci_dpl_res = boot_ci(g_dpl, np.median)

    rows = [
        # 1. Limited compensation
        dict(
            estimand="F_close,test",
            role="Limited compensation",
            ic=(pc["IC_test"]["median"], pc["IC_test"]["boot_ci_median_display"]),
            dpl=(pc["dPL_test"]["median"], pc["dPL_test"]["boot_ci_median_display"]),
            n_ic=pc["IC_test"]["n_valid"],
            n_dpl=pc["dPL_test"]["n_valid"],
            decimals=3,
            source="figure5_summary.panel_c_f_close",
        ),
        # 2. Generalization decay
        dict(
            estimand="decay_G_base",
            role="Generalization decay",
            ic=(dd["IC"]["median"], dd["IC"]["boot_ci_median"]),
            dpl=(dd["dPL_agg"]["median"], dd["dPL_agg"]["boot_ci_median_display"]),
            n_ic=dd["IC"]["n_valid"],
            n_dpl=dd["dPL_agg"]["n"],
            decimals=4,
            source="figure5_summary.panel_d_decay",
        ),
        # 3. Generic mitigation
        dict(
            estimand="F_tgd2 (test)",
            role="Generic mitigation",
            ic=(pb["IC"]["median"], pb["IC"]["boot_ci_median_display"]),
            dpl=(pb["dPL"]["median"], pb["dPL"]["boot_ci_median_display"]),
            n_ic=pb["IC"]["n_valid"],
            n_dpl=pb["dPL"]["n_valid"],
            decimals=3,
            source="figure6_summary.panel_b_f_tgd2",
        ),
        # 4. Parameter relief (Delta C_theta)
        dict(
            estimand="Delta C_theta (R_theta_tgd2)",
            role="Parameter relief",
            ic=(
                prth["IC"]["median"],
                prth["IC"].get("boot_ci_frozen", prth["IC"]["boot_ci_median_display"]),
            ),
            dpl=(prth["dPL"]["median"], prth["dPL"]["boot_ci_median_display"]),
            n_ic=prth["IC"]["n_valid"],
            n_dpl=prth["dPL"]["n_valid"],
            decimals=4,
            source="figure6_summary.panel_c_r_theta",
        ),
        # 5. State relief (Delta C_state)
        dict(
            estimand="Delta C_state (R_state_tgd2)",
            role="State relief",
            ic=(
                prst["IC"]["median"],
                prst["IC"].get("boot_ci_frozen", prst["IC"]["boot_ci_median_display"]),
            ),
            dpl=(prst["dPL"]["median"], prst["dPL"]["boot_ci_median_display"]),
            n_ic=prst["IC"]["n_valid"],
            n_dpl=prst["dPL"]["n_valid"],
            decimals=4,
            source="figure6_summary.panel_d_r_state",
        ),
        # 6. Residual explicit-structure advantage
        dict(
            estimand="Delta KGE_CN-TGD2 (test)",
            role="Residual explicit advantage",
            ic=(pe["IC"]["median"], ci_ic_res),
            dpl=(pe["dPL"]["median"], ci_dpl_res),
            n_ic=int(len(g_ic)),
            n_dpl=int(len(g_dpl)),
            decimals=4,
            source="figure6_summary.panel_e_residual_vs_frac_snow + V4 frozen CI",
        ),
        # 7. Snow-active residual
        dict(
            estimand="Delta RMSE_snow (mm d-1)",
            role="Residual on snow-active days",
            ic=(
                pf["IC"]["snow_active"]["median"],
                pf["IC"]["snow_active"]["boot_ci_median_display"],
            ),
            dpl=(
                pf["dPL"]["snow_active"]["median"],
                pf["dPL"]["snow_active"]["boot_ci_median_display"],
            ),
            n_ic=pf["IC"]["snow_active"]["n_valid"],
            n_dpl=pf["dPL"]["snow_active"]["n_valid"],
            decimals=3,
            source="figure6_summary.panel_f_process_errors",
        ),
        # 8. Non-snow residual
        dict(
            estimand="Delta RMSE_non-snow (mm d-1)",
            role="Residual on non-snow days",
            ic=(
                pf["IC"]["no_snow_active"]["median"],
                pf["IC"]["no_snow_active"]["boot_ci_median_display"],
            ),
            dpl=(
                pf["dPL"]["no_snow_active"]["median"],
                pf["dPL"]["no_snow_active"]["boot_ci_median_display"],
            ),
            n_ic=pf["IC"]["no_snow_active"]["n_valid"],
            n_dpl=pf["dPL"]["no_snow_active"]["n_valid"],
            decimals=3,
            source="figure6_summary.panel_f_process_errors",
        ),
    ]
    return rows


def validate(rows, psum, pval, d5, d6):
    """Cross-check medians against frozen canonical summaries and the prepared summaries."""
    errors = []
    # IC anchors must match frozen canonical values exactly (1e-9)
    anchors = [
        (rows[0]["ic"][0], psum["IC_test"]["F_close"]["median"], "F_close IC"),
        (rows[2]["ic"][0], psum["IC_test"]["F_tgd2"]["median"], "F_tgd2 IC"),
        (rows[1]["ic"][0], pval["IC_decay_G_base"]["median"], "decay_G_base IC"),
        (rows[3]["ic"][0], pval["V3"]["IC"]["R_theta_tgd2"]["median"], "R_theta IC"),
        (rows[4]["ic"][0], pval["V3"]["IC"]["R_state_tgd2"]["median"], "R_state IC"),
        (
            rows[5]["ic"][0],
            pval["V4"]["IC_test"]["G_CN_over_TGD2"]["median"],
            "G_CN_over_TGD2 IC",
        ),
    ]
    for got, frozen, label in anchors:
        if not np.isclose(got, frozen, atol=1e-9, rtol=0):
            errors.append(f"{label}: {got} != frozen {frozen}")
    # dPL aggregated medians must lie within (or close to) the frozen per-seed range
    dpl_seed_checks = [
        (
            rows[0]["dpl"][0],
            d5["panel_c_f_close"]["dPL_test"]["seed_medians"],
            "F_close dPL",
        ),
        (rows[2]["dpl"][0], d6["panel_b_f_tgd2"]["dPL"]["seed_medians"], "F_tgd2 dPL"),
        (
            rows[1]["dpl"][0],
            [pval[f"dPL_{s}_decay_G_base"]["median"] for s in (42, 123, 2026)],
            "decay_G_base dPL",
        ),
        (
            rows[3]["dpl"][0],
            d6["panel_c_r_theta"]["dPL"]["seed_medians"],
            "R_theta dPL",
        ),
        (
            rows[4]["dpl"][0],
            d6["panel_d_r_state"]["dPL"]["seed_medians"],
            "R_state dPL",
        ),
        (
            rows[5]["dpl"][0],
            [
                pval["V4"][f"dPL_{s}_test"]["G_CN_over_TGD2"]["median"]
                for s in (42, 123, 2026)
            ],
            "G_CN_over_TGD2 dPL",
        ),
    ]
    tol = 1e-3
    for got, seed_meds, label in dpl_seed_checks:
        lo, hi = min(seed_meds), max(seed_meds)
        if not (lo - tol <= got <= hi + tol):
            errors.append(
                f"{label}: aggregated {got:.6f} outside seed range [{lo:.6f}, {hi:.6f}]"
            )
    if errors:
        raise SystemExit("Table 5 validation FAILED:\n  " + "\n  ".join(errors))
    print(
        "[check] Table 5 medians match frozen canonical R3 summaries (IC exact 1e-9; "
        "dPL aggregated within frozen per-seed range)."
    )


def build_markdown(rows, n_note):
    header = (
        "| Estimand | Evidence role | IC median [95% CI] | dPL median [95% CI] | "
        "N (IC/dPL) |\n"
        "| :--- | :--- | :--- | :--- | :---: |\n"
    )
    body = ""
    for r in rows:
        body += "| {} | {} | {} | {} | {}/{} |\n".format(
            r["estimand"],
            r["role"],
            format_stat(r["ic"][0], r["ic"][1][0], r["ic"][1][1], r["decimals"]),
            format_stat(r["dpl"][0], r["dpl"][1][0], r["dpl"][1][1], r["decimals"]),
            r["n_ic"],
            r["n_dpl"],
        )
    note = "\n*Note*: {}\n".format(n_note)
    title = (
        "# Table 5: R3 Synthetic Known-Truth Experiment — Summary of "
        "Compensation and Structural-Surrogate Evidence (Figures 5–6)\n\n"
    )
    return title + header + body + note


def build_latex(rows, n_note):
    body = ""
    for r in rows:
        body += "{} & {} & {} & {} & {}/{} \\\\\n".format(
            r["estimand"].replace("Delta", r"$\Delta$").replace("_", r"\_"),
            r["role"],
            format_stat(r["ic"][0], r["ic"][1][0], r["ic"][1][1], r["decimals"]),
            format_stat(r["dpl"][0], r["dpl"][1][0], r["dpl"][1][1], r["decimals"]),
            r["n_ic"],
            r["n_dpl"],
        )
    tex_note = n_note.replace("%", r"\%").replace("_", r"\_")
    return (
        r"""\begin{table*}[t]
\centering
\caption{R3 synthetic known-truth experiment --- summary of compensation and structural-surrogate evidence (numerical anchors for Figures~5 and~6).}
\label{tab:table5_r3_summary}
\begin{threeparttable}
\begin{tabular}{llccc}
\toprule
Estimand & Evidence role & IC median [95\% CI] & dPL median [95\% CI] & N (IC/dPL) \\
\midrule
"""
        + body
        + r"""\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]
\small
\item \textit{Note}: """
        + tex_note
        + r"""
\end{tablenotes}
\end{threeparttable}
\end{table*}
"""
    )


def main() -> None:
    RES_R3.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    d5 = json.loads((RES_R3 / "figure5_summary.json").read_text())
    d6 = json.loads((RES_R3 / "figure6_summary.json").read_text())
    psum, pval = load_anchors()
    seedmed = pd.read_csv(RES_R3 / "figure6_basin_seedmedian.csv")
    seedmed["basin_id"] = seedmed["basin_id"].astype(str).str.zfill(8)

    rows = build_rows(d5, d6, psum, pval, seedmed)
    validate(rows, psum, pval, d5, d6)
    if len(rows) != 8:
        raise SystemExit(f"expected 8 rows, got {len(rows)}")

    n_note = (
        "Values report basin-level medians with 95% bootstrap confidence intervals "
        "[2.5th, 97.5th percentiles] from paired basin resampling (2000 replicates, "
        "seed 20260730; repository R3 convention). dPL values are per-basin medians "
        "over seeds 42/123/2026 (seed-aggregated), reported as the median across basins. "
        "Sign conventions: decay_G_base = G_base(train) - G_base(test), positive means "
        "compensation is stronger in train; Delta C_theta and Delta C_state = Base - TGD2, "
        "positive means TGD2 reduces the CN-adjusted excess error; Delta KGE_CN-TGD2 = "
        "KGE_CN - KGE_TGD2, positive means CN retains an advantage; process residuals = "
        "RMSE_TGD2 - RMSE_CN (mm d-1) on truth snow-active / non-snow days, positive means "
        "CN has lower error. Valid basin counts (IC/dPL) by row: "
        + "; ".join(f"{r['role']} {r['n_ic']}/{r['n_dpl']}" for r in rows)
        + "."
    )

    md = build_markdown(rows, n_note)
    tex = build_latex(rows, n_note)

    for d in (TABLES_DIR, STATS_DIR):
        (d / f"{OUT_STEM}.md").write_text(md)
        (d / f"{OUT_STEM}.tex").write_text(tex)

    # machine-readable source
    csv_rows = []
    for r in rows:
        csv_rows.append(
            {
                "estimand": r["estimand"],
                "role": r["role"],
                "ic_median": r["ic"][0],
                "ic_ci_low": r["ic"][1][0],
                "ic_ci_high": r["ic"][1][1],
                "dpl_median": r["dpl"][0],
                "dpl_ci_low": r["dpl"][1][0],
                "dpl_ci_high": r["dpl"][1][1],
                "n_ic": r["n_ic"],
                "n_dpl": r["n_dpl"],
                "source": r["source"],
            }
        )
    pd.DataFrame(csv_rows).to_csv(RES_R3 / CSV_NAME, index=False)

    # printed summary for the record
    print(f"\nTable 5 rows ({len(rows)}):")
    for r in rows:
        print(
            f"  {r['estimand']:34s} | {r['role']:30s} | "
            f"IC {format_stat(r['ic'][0], r['ic'][1][0], r['ic'][1][1], r['decimals'])} "
            f"| dPL {format_stat(r['dpl'][0], r['dpl'][1][0], r['dpl'][1][1], r['decimals'])}"
        )
    print(
        "\nWrote:",
        TABLES_DIR / f"{OUT_STEM}.md",
        TABLES_DIR / f"{OUT_STEM}.tex",
        RES_R3 / CSV_NAME,
    )
    print("Table 5 generated successfully in markdown and LaTeX formats.")


if __name__ == "__main__":
    main()
