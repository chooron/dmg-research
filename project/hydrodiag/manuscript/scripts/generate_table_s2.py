"""
Table S2 Generator Script (R1 Analysis)
Generates TableS2_paired_structural_kge_differences.md and TableS2_paired_structural_kge_differences.tex
Basin-wise paired KGE differences among the controlled XAJ structures.
"""

import os
import sys
import pandas as pd


def format_val(val, decs=3):
    return f"{val:.{decs}f}"


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r1_dir = os.path.join(project_root, "manuscript/results/R1")

    out_stats_dir = os.path.join(project_root, "manuscript/stats/tables")
    out_table_dir = os.path.join(project_root, "manuscript/tables")
    os.makedirs(out_stats_dir, exist_ok=True)
    os.makedirs(out_table_dir, exist_ok=True)

    df_paired_sum = pd.read_csv(os.path.join(r1_dir, "r1_paired_effects_summary.csv"))
    kge_paired = df_paired_sum[df_paired_sum["metric"] == "kge"]

    paradigms = ["IC-CMA-ES", "dPL-MLP"]
    periods = ["train", "test"]
    scopes = [
        ("All basins", "full_sample", None),
        ("S1 (0\u20130.05)", "snow_stratum", "S1"),
        ("S2 (0.05\u20130.15)", "snow_stratum", "S2"),
        ("S3 (0.15\u20130.30)", "snow_stratum", "S3"),
        ("S4 (0.30\u20130.50)", "snow_stratum", "S4"),
        ("S5 (0.50\u20131.00)", "snow_stratum", "S5"),
    ]
    contrasts = [
        ("TGD - Base", "TGD-Base"),
        ("CN - TGD", "CN-TGD"),
        ("CN - Base", "CN-Base"),
    ]

    rows = []
    for p in paradigms:
        for per in periods:
            for scope_label, summary_lvl, s_name in scopes:
                for c_label, c_code in contrasts:
                    if summary_lvl == "full_sample":
                        sub = kge_paired[
                            (kge_paired["paradigm"] == p)
                            & (kge_paired["period"] == per)
                            & (kge_paired["effect"] == c_code)
                            & (kge_paired["summary_level"] == "full_sample")
                        ]
                    else:
                        sub = kge_paired[
                            (kge_paired["paradigm"] == p)
                            & (kge_paired["period"] == per)
                            & (kge_paired["effect"] == c_code)
                            & (kge_paired["snow_stratum"] == s_name)
                        ]

                    if len(sub) == 0:
                        raise ValueError(f"Missing row for {p} {per} {scope_label} {c_code}")

                    r = sub.iloc[0]
                    rows.append(
                        {
                            "Paradigm": p,
                            "Period": per.capitalize(),
                            "Scope": scope_label,
                            "Contrast": c_label,
                            "n": int(r["valid_basin_count"]),
                            "Median": r["median"],
                            "CI_low": r["bootstrap_ci_low"],
                            "CI_high": r["bootstrap_ci_high"],
                            "Pos_frac": r["fraction_positive"],
                        }
                    )

    # 1. Markdown Table
    md_header = (
        "# Table S2: Basin-wise Paired KGE Differences Among Controlled XAJ Structures\n\n"
        "| Paradigm | Period | Snow regime | Contrast | n | Median Paired ΔKGE | Bootstrap 95% CI | Positive-Effect Fraction |\n"
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |\n"
    )
    md_rows_str = ""
    for r in rows:
        ci_str = f"[{format_val(r['CI_low'])}, {format_val(r['CI_high'])}]"
        md_rows_str += (
            f"| {r['Paradigm']} | {r['Period']} | {r['Scope']} | {r['Contrast']} | "
            f"{r['n']} | {format_val(r['Median'])} | {ci_str} | {format_val(r['Pos_frac'])} |\n"
        )
    md_note = (
        "\n*Note*: Basin-wise paired KGE differences (TGD − Base, CN − TGD, CN − Base) evaluated across all basins "
        "and five snow-fraction intervals. Summary statistics report median paired delta KGE, 95% bootstrap confidence intervals, "
        "and the fraction of basins exhibiting a positive structural difference (ΔKGE > 0)."
    )
    full_md = md_header + md_rows_str + md_note

    md_stats_path = os.path.join(out_stats_dir, "TableS2_paired_structural_kge_differences.md")
    md_table_path = os.path.join(out_table_dir, "TableS2_paired_structural_kge_differences.md")
    with open(md_stats_path, "w") as f:
        f.write(full_md)
    with open(md_table_path, "w") as f:
        f.write(full_md)

    # 2. LaTeX Table
    tex_rows_str = ""
    for r in rows:
        ci_str = f"[{format_val(r['CI_low'])}, {format_val(r['CI_high'])}]"
        tex_rows_str += (
            f"{r['Paradigm']} & {r['Period']} & {r['Scope']} & {r['Contrast']} & "
            f"{r['n']} & {format_val(r['Median'])} & {ci_str} & {format_val(r['Pos_frac'])} \\\\\n"
        )

    full_tex = r"""\begin{table*}[t]
\centering
\caption{Basin-wise paired KGE differences among the controlled XAJ structures.}
\label{tab:s2_paired_structural_kge_differences}
\begin{threeparttable}
\begin{tabular}{lllccccccc}
\toprule
Paradigm & Period & Snow regime & Contrast & n & Median $\Delta\text{KGE}$ & Bootstrap 95\% CI & Positive Fraction \\
\midrule
""" + tex_rows_str + r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Summary statistics represent basin-wise paired structural differences evaluated across $n = 531$ matched CAMELS basins and five snow-fraction strata. Positive fraction denotes the proportion of basins with $\Delta\text{KGE} > 0$.
\end{tablenotes}
\end{threeparttable}
\end{table*}
"""

    tex_stats_path = os.path.join(out_stats_dir, "TableS2_paired_structural_kge_differences.tex")
    tex_table_path = os.path.join(out_table_dir, "TableS2_paired_structural_kge_differences.tex")
    with open(tex_stats_path, "w") as f:
        f.write(full_tex)
    with open(tex_table_path, "w") as f:
        f.write(full_tex)

    # Copy generator script to manuscript/scripts/
    cp_script_target = os.path.join(project_root, "manuscript/scripts/generate_table_s2.py")
    if os.path.abspath(__file__) != os.path.abspath(cp_script_target):
        with open(__file__, "r") as sf, open(cp_script_target, "w") as df:
            df.write(sf.read())

    print("Table S2 generated successfully!")
    print(f"Markdown: {md_stats_path}")
    print(f"LaTeX: {tex_stats_path}")


if __name__ == "__main__":
    main()
