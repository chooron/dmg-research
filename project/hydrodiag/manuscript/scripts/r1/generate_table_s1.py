"""
Table S1 Generator Script (R1 Analysis)
Generates TableS1_paired_effects_and_sensitivity.md and TableS1_paired_effects_and_sensitivity.tex
Single compact R1 results SI table covering:
- Panel A: Paired structural ΔKGE effects in evaluation period (1995–2010) across snow regimes
- Panel B: Sensitivity of large center-of-timing errors (|ΔCT| ≥ 10, 15, 20 d) at KGE ≥ 0.60
"""

import os
import sys

import numpy as np
import pandas as pd


def format_stat(med, ci_low, ci_high, decimals=3):
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(med)} [{fmt.format(ci_low)}, {fmt.format(ci_high)}]"


def main():
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    r1_dir = os.path.join(project_root, "manuscript/results/R1")

    out_stats_dir = os.path.join(project_root, "manuscript/stats/tables")
    out_table_dir = os.path.join(project_root, "manuscript/tables")
    os.makedirs(out_stats_dir, exist_ok=True)
    os.makedirs(out_table_dir, exist_ok=True)

    # ── Panel A: Paired Structural Effects Data ──────────────────────────────
    df_p = pd.read_csv(os.path.join(r1_dir, "r1_paired_effects_summary.csv"))
    sub_p = df_p[(df_p["metric"] == "kge") & (df_p["period"] == "test")]

    scopes = [
        ("All basins", "full_sample", None, 531),
        ("S1 (0\u20130.05)", "snow_stratum", "S1", 165),
        ("S2 (0.05\u20130.15)", "snow_stratum", "S2", 156),
        ("S3 (0.15\u20130.30)", "snow_stratum", "S3", 121),
        ("S4 (0.30\u20130.50)", "snow_stratum", "S4", 34),
        ("S5 (0.50\u20131.00)", "snow_stratum", "S5", 55),
    ]

    contrasts = [
        ("CN-TGD", "IC-CMA-ES", "CN − TGD (IC)"),
        ("CN-TGD", "dPL-MLP", "CN − TGD (dPL)"),
        ("CN-Base", "IC-CMA-ES", "CN − Base (IC)"),
        ("CN-Base", "dPL-MLP", "CN − Base (dPL)"),
        ("TGD-Base", "IC-CMA-ES", "TGD − Base (IC)"),
        ("TGD-Base", "dPL-MLP", "TGD − Base (dPL)"),
    ]

    panel_a_md_rows = []
    panel_a_tex_rows = []

    for label, s_level, s_stratum, n in scopes:
        row_md = {"Snow regime": label, "n": n}
        row_tex_vals = [label, str(n)]
        for eff, p_query, c_label in contrasts:
            if s_level == "full_sample":
                sub = sub_p[
                    (sub_p["effect"] == eff)
                    & (sub_p["paradigm"] == p_query)
                    & (sub_p["summary_level"] == "full_sample")
                ]
            else:
                sub = sub_p[
                    (sub_p["effect"] == eff)
                    & (sub_p["paradigm"] == p_query)
                    & (sub_p["summary_level"] == "snow_stratum")
                    & (sub_p["snow_stratum"] == s_stratum)
                ]
            r = sub.iloc[0]
            stat_str = format_stat(
                r["median"], r["bootstrap_ci_low"], r["bootstrap_ci_high"], 3
            )
            row_md[c_label] = stat_str
            row_tex_vals.append(stat_str)
        panel_a_md_rows.append(row_md)
        panel_a_tex_rows.append(row_tex_vals)

    # ── Panel B: Timing Threshold Sensitivity Data ───────────────────────────
    perf = pd.read_csv(os.path.join(r1_dir, "r1_basin_level_performance.csv"))
    perf["basin_id"] = perf["basin_id"].astype(str).str.zfill(8)
    sig = pd.read_csv(os.path.join(r1_dir, "r1_snow_signatures_basin_level.csv"))
    sig["basin_id"] = sig["basin_id"].astype(str).str.zfill(8)

    panel_b_md_rows = []
    panel_b_tex_rows = []

    for model_name, code in [
        ("Base", "XAJ-Base"),
        ("TGD", "XAJ-TGD"),
        ("CN", "XAJ-CN"),
    ]:
        for paradigm, p_label in [("IC-CMA-ES", "IC"), ("dPL-MLP", "dPL")]:
            k_series = perf[
                (perf["paradigm"] == paradigm)
                & (perf["model"] == code)
                & (perf["period"] == "test")
            ].set_index("basin_id")["kge"]
            sub_sig = sig[
                (sig["paradigm"] == paradigm)
                & (sig["model"] == code)
                & (sig["period"] == "test")
            ]
            if paradigm == "IC-CMA-ES":
                ct_series = sub_sig[
                    sub_sig["seed_or_restart"] == "selected_restart"
                ].set_index("basin_id")["ct_error_signed"]
            else:
                ct_series = sub_sig.groupby("basin_id")["ct_error_signed"].median()

            df = pd.DataFrame({"kge": k_series, "ct_error_signed": ct_series}).dropna()
            scr = df[df["kge"] >= 0.60]
            n_screen = len(scr)

            n_10 = int((scr["ct_error_signed"].abs() >= 10.0).sum())
            pct_10 = 100.0 * n_10 / n_screen
            n_15 = int((scr["ct_error_signed"].abs() >= 15.0).sum())
            pct_15 = 100.0 * n_15 / n_screen
            n_20 = int((scr["ct_error_signed"].abs() >= 20.0).sum())
            pct_20 = 100.0 * n_20 / n_screen

            str_10 = f"{n_10}/{n_screen} ({pct_10:.1f}%)"
            str_15 = f"{n_15}/{n_screen} ({pct_15:.1f}%)"
            str_20 = f"{n_20}/{n_screen} ({pct_20:.1f}%)"

            panel_b_md_rows.append(
                {
                    "Configuration": model_name,
                    "Regime": p_label,
                    "N_screen": n_screen,
                    "|ΔCT| ≥ 10 d": str_10,
                    "|ΔCT| ≥ 15 d": str_15,
                    "|ΔCT| ≥ 20 d": str_20,
                }
            )
            panel_b_tex_rows.append(
                [model_name, p_label, str(n_screen), str_10, str_15, str_20]
            )

    # ── Build Markdown Output ────────────────────────────────────────────────
    md_content = """# Table S1: Paired Structural Effects Across Snow Regimes and Sensitivity of Large Center-of-Timing Errors

### Panel A: Evaluation Period (1995–2010) Paired Structural ΔKGE Effects Across Snow Regimes

| Snow regime | n | CN − TGD (IC) | CN − TGD (dPL) | CN − Base (IC) | CN − Base (dPL) | TGD − Base (IC) | TGD − Base (dPL) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in panel_a_md_rows:
        md_content += (
            f"| {r['Snow regime']} | {r['n']} | "
            f"{r['CN − TGD (IC)']} | {r['CN − TGD (dPL)']} | "
            f"{r['CN − Base (IC)']} | {r['CN − Base (dPL)']} | "
            f"{r['TGD − Base (IC)']} | {r['TGD − Base (dPL)']} |\n"
        )

    md_content += """
### Panel B: Sensitivity of Screened Large Center-of-Timing Errors to Threshold Definition (KGE ≥ 0.60)

| Configuration | Regime | Screened N | |ΔCT| ≥ 10 d | |ΔCT| ≥ 15 d | |ΔCT| ≥ 20 d |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for r in panel_b_md_rows:
        md_content += (
            f"| {r['Configuration']} | {r['Regime']} | {r['N_screen']} | "
            f"{r['|ΔCT| ≥ 10 d']} | {r['|ΔCT| ≥ 15 d']} | {r['|ΔCT| ≥ 20 d']} |\n"
        )

    md_content += """
*Note*: Panel A reports basin-wise median paired KGE differences [95% bootstrap CI] in the evaluation period (1995–2010) across all 531 matched basins and five snow-fraction intervals. Panel B reports the screened basin count and percentage satisfying large center-of-timing error thresholds (|ΔCT| ≥ 10, 15, 20 days) at the operational aggregate performance screen (standard KGE ≥ 0.60).
"""

    # ── Build LaTeX Output ───────────────────────────────────────────────────
    tex_a_str = ""
    for r_vals in panel_a_tex_rows:
        tex_a_str += " & ".join(r_vals) + " \\\\\n"

    tex_b_str = ""
    for r_vals in panel_b_tex_rows:
        tex_b_str += " & ".join(r_vals) + " \\\\\n"

    tex_content = (
        r"""\begin{table*}[t]
\centering
\caption{Paired structural effects across snow regimes (Panel A) and sensitivity of large center-of-timing errors to the timing threshold at $\text{KGE} \ge 0.60$ (Panel B).}
\label{tab:tables1_paired_and_sensitivity}
\begin{threeparttable}
\begin{tabular}{lcccccc}
\toprule
\multicolumn{7}{l}{\textbf{Panel A: Evaluation Period (1995--2010) Paired Structural $\Delta\text{KGE}$ Effects Across Snow Regimes}} \\
\midrule
Snow regime & $n$ & CN $-$ TGD (IC) & CN $-$ TGD (dPL) & CN $-$ Base (IC) & CN $-$ Base (dPL) & TGD $-$ Base (IC) & TGD $-$ Base (dPL) \\
\midrule
"""
        + tex_a_str
        + r"""\midrule
\multicolumn{7}{l}{\textbf{Panel B: Sensitivity of Screened Large Center-of-Timing Errors to Threshold Definition ($\text{KGE} \ge 0.60$)}} \\
\midrule
Configuration & Regime & Screened $N$ & $|\Delta\text{CT}| \ge 10$~d & $|\Delta\text{CT}| \ge 15$~d & $|\Delta\text{CT}| \ge 20$~d \\
\midrule
"""
        + tex_b_str
        + r"""\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]
\small
\item \textit{Note}: Panel A reports basin-wise median paired KGE differences [95\% bootstrap CI] across all $n = 531$ basins and five snow-fraction intervals. Panel B reports screened basin counts and percentages satisfying $|\Delta\text{CT}| \ge 10, 15, 20$ days at $\text{KGE} \ge 0.60$.
\end{tablenotes}
\end{threeparttable}
\end{table*}
"""
    )

    # Write files
    for d in (out_stats_dir, out_table_dir):
        with open(
            os.path.join(d, "TableS1_paired_effects_and_sensitivity.md"), "w"
        ) as f:
            f.write(md_content)
        with open(
            os.path.join(d, "TableS1_paired_effects_and_sensitivity.tex"), "w"
        ) as f:
            f.write(tex_content)

    print("Supplementary Table S1 generated successfully.")


if __name__ == "__main__":
    main()
