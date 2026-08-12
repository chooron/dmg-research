"""
Table 1 Generator Script (R1 Analysis)
Generates Table1_absolute_performance.md and Table1_absolute_performance.tex
Main-text predictive performance sanity check across all 531 basins.
"""

import os
import sys
import pandas as pd


def format_stat(med, ci_low, ci_high, decimals):
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(med)} [{fmt.format(ci_low)}, {fmt.format(ci_high)}]"


def main():
    project_root = "/home/jingxin/code/dmg-research/project/hydrodiag"
    r1_dir = os.path.join(project_root, "manuscript/results/R1")

    out_stats_dir = os.path.join(project_root, "manuscript/stats/tables")
    out_table_dir = os.path.join(project_root, "manuscript/tables")
    os.makedirs(out_stats_dir, exist_ok=True)
    os.makedirs(out_table_dir, exist_ok=True)

    df_abs = pd.read_csv(os.path.join(r1_dir, "r1_absolute_metrics_summary.csv"))
    fs = df_abs[df_abs["summary_level"] == "full_sample"]

    row_specs = [
        ("XAJ-Base", "IC-CMA-ES", "Base", "IC"),
        ("XAJ-Base", "dPL-MLP",   "Base", "dPL"),
        ("XAJ-TGD",  "IC-CMA-ES", "TGD",  "IC"),
        ("XAJ-TGD",  "dPL-MLP",   "TGD",  "dPL"),
        ("XAJ-CN",   "IC-CMA-ES", "CN",   "IC"),
        ("XAJ-CN",   "dPL-MLP",   "CN",   "dPL"),
        ("HBV",      "dPL-MLP",   "HBV (reference)", "dPL"),
    ]

    metrics_config = [
        ("kge", 3),
        ("nse", 3),
        ("pbias", 2),
        ("rmse", 3),
    ]

    rows_md = []
    rows_tex = []

    for model_query, p_query, model_label, p_label in row_specs:
        for period in ["train", "test"]:
            period_cap = period.capitalize()
            row_md = {
                "Configuration": model_label,
                "Regime": p_label,
                "Period": period_cap,
            }
            row_tex_vals = [model_label, p_label, period_cap]

            for metric, decs in metrics_config:
                sub = fs[
                    (fs["model"] == model_query)
                    & (fs["paradigm"] == p_query)
                    & (fs["metric"] == metric)
                    & (fs["period"] == period)
                ]
                if len(sub) == 0:
                    raise ValueError(f"Missing metric summary for {p_query} {model_query} {metric} {period}")
                r = sub.iloc[0]
                stat_str = format_stat(r["median"], r["bootstrap_ci_low"], r["bootstrap_ci_high"], decs)
                col_name = metric.upper() if metric != "rmse" else "RMSE (mm d⁻¹)"
                if metric == "pbias":
                    col_name = "PBIAS (%)"
                row_md[col_name] = stat_str
                row_tex_vals.append(stat_str)

            rows_md.append(row_md)
            rows_tex.append(row_tex_vals)

    # 1. Build Markdown Table
    md_header = (
        "| Configuration | Regime | Period | KGE | NSE | PBIAS (%) | RMSE (mm d⁻¹) |\n"
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n"
    )
    md_rows_str = ""
    for r in rows_md:
        md_rows_str += (
            f"| {r['Configuration']} | {r['Regime']} | {r['Period']} | "
            f"{r['KGE']} | {r['NSE']} | {r['PBIAS (%)']} | {r['RMSE (mm d⁻¹)']} |\n"
        )
    md_note = (
        "\n*Note*: Values report basin-wise medians with 95% bootstrap confidence intervals [2.5th, 97.5th percentiles] "
        "across all n = 531 matched basins for calibration (1981–1995) and evaluation (1995–2010) periods. "
        "Units: PBIAS (%), RMSE (mm d⁻¹). KGE and NSE are dimensionless. HBV is reported as an external "
        "dPL reference benchmark and is not part of the controlled XAJ structural progression."
    )

    full_md = "# Table 1: Streamflow Simulation Performance Across Structural Configurations and Parameter-Estimation Regimes\n\n" + md_header + md_rows_str + md_note

    # Write Markdown files
    for d in (out_stats_dir, out_table_dir):
        with open(os.path.join(d, "Table1_absolute_performance.md"), "w") as f:
            f.write(full_md)

    # 2. Build LaTeX Table
    tex_rows_str = ""
    for r_vals in rows_tex:
        tex_rows_str += " & ".join(r_vals) + " \\\\\n"

    full_tex = r"""\begin{table*}[t]
\centering
\caption{Streamflow simulation performance across structural configurations and parameter-estimation regimes ($n = 531$).}
\label{tab:table1_absolute_performance}
\begin{threeparttable}
\begin{tabular}{lllcccc}
\toprule
Configuration & Regime & Period & KGE & NSE & PBIAS (\%) & RMSE (mm d$^{-1}$) \\
\midrule
""" + tex_rows_str + r"""\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]
\small
\item \textit{Note}: Values report basin-wise medians with 95\% bootstrap confidence intervals [2.5th, 97.5th percentiles] across all $n = 531$ matched basins for calibration (1981--1995) and evaluation (1995--2010) periods. Units: PBIAS (\%), RMSE (mm d$^{-1}$). KGE and NSE are dimensionless. HBV is reported as an external dPL reference benchmark.
\end{tablenotes}
\end{threeparttable}
\end{table*}
"""

    for d in (out_stats_dir, out_table_dir):
        with open(os.path.join(d, "Table1_absolute_performance.tex"), "w") as f:
            f.write(full_tex)

    print("Table 1 generated successfully in markdown and LaTeX formats.")


if __name__ == "__main__":
    main()
