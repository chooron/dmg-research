"""
Table S3 Generator Script (R1 Analysis)
Generates TableS3_ic_dpl_temporal_transfer.md and TableS3_ic_dpl_temporal_transfer.tex
Temporal transfer of IC relative to dPL across structures and snow strata.
"""

import os
import sys

import pandas as pd


def format_val(val, decs=3):
    return f"{val:.{decs}f}"


def format_stat(med, ci_low, ci_high, decs=3):
    return f"{format_val(med, decs)} [{format_val(ci_low, decs)}, {format_val(ci_high, decs)}]"


def main():
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    r1_dir = os.path.join(project_root, "manuscript/results/R1")

    out_stats_dir = os.path.join(project_root, "manuscript/stats/tables")
    out_table_dir = os.path.join(project_root, "manuscript/tables")
    os.makedirs(out_stats_dir, exist_ok=True)
    os.makedirs(out_table_dir, exist_ok=True)

    df_paired_sum = pd.read_csv(os.path.join(r1_dir, "r1_paired_effects_summary.csv"))
    abd = df_paired_sum[
        (df_paired_sum["paradigm"] == "IC-dPL")
        & (df_paired_sum["metric"] == "kge")
        & (
            df_paired_sum["effect"].isin(
                ["A_IC_minus_dPL", "B_IC_minus_dPL", "D_IC_minus_dPL"]
            )
        )
    ]

    scopes = [
        ("All basins", "full_sample", None),
        ("S1 (0\u20130.05)", "snow_stratum", "S1"),
        ("S2 (0.05\u20130.15)", "snow_stratum", "S2"),
        ("S3 (0.15\u20130.30)", "snow_stratum", "S3"),
        ("S4 (0.30\u20130.50)", "snow_stratum", "S4"),
        ("S5 (0.50\u20131.00)", "snow_stratum", "S5"),
    ]

    models = [("Base", "XAJ-Base"), ("TGD", "XAJ-TGD"), ("CN", "XAJ-CN")]

    rows = []
    for struct_label, m_code in models:
        for scope_label, summary_lvl, s_name in scopes:
            if summary_lvl == "full_sample":
                sub_a = abd[
                    (abd["model"] == m_code)
                    & (abd["effect"] == "A_IC_minus_dPL")
                    & (abd["summary_level"] == "full_sample")
                ].iloc[0]
                sub_b = abd[
                    (abd["model"] == m_code)
                    & (abd["effect"] == "B_IC_minus_dPL")
                    & (abd["summary_level"] == "full_sample")
                ].iloc[0]
                sub_d = abd[
                    (abd["model"] == m_code)
                    & (abd["effect"] == "D_IC_minus_dPL")
                    & (abd["summary_level"] == "full_sample")
                ].iloc[0]
            else:
                sub_a = abd[
                    (abd["model"] == m_code)
                    & (abd["effect"] == "A_IC_minus_dPL")
                    & (abd["snow_stratum"] == s_name)
                ].iloc[0]
                sub_b = abd[
                    (abd["model"] == m_code)
                    & (abd["effect"] == "B_IC_minus_dPL")
                    & (abd["snow_stratum"] == s_name)
                ].iloc[0]
                sub_d = abd[
                    (abd["model"] == m_code)
                    & (abd["effect"] == "D_IC_minus_dPL")
                    & (abd["snow_stratum"] == s_name)
                ].iloc[0]

            rows.append(
                {
                    "Structure": struct_label,
                    "Scope": scope_label,
                    "n": int(sub_a["valid_basin_count"]),
                    "A_str": format_stat(
                        sub_a["median"],
                        sub_a["bootstrap_ci_low"],
                        sub_a["bootstrap_ci_high"],
                    ),
                    "B_str": format_stat(
                        sub_b["median"],
                        sub_b["bootstrap_ci_low"],
                        sub_b["bootstrap_ci_high"],
                    ),
                    "D_str": format_stat(
                        sub_d["median"],
                        sub_d["bootstrap_ci_low"],
                        sub_d["bootstrap_ci_high"],
                    ),
                }
            )

    # 1. Markdown Table
    md_header = (
        "# Table S3: Temporal Transfer of IC Relative to dPL\n\n"
        "| Structure | Snow regime | n | A (Train: IC − dPL) | B (Test: IC − dPL) | D (Transfer Delta: B − A) |\n"
        "| :--- | :--- | :---: | :---: | :---: | :---: |\n"
    )
    md_rows_str = ""
    for r in rows:
        md_rows_str += (
            f"| {r['Structure']} | {r['Scope']} | {r['n']} | "
            f"{r['A_str']} | {r['B_str']} | {r['D_str']} |\n"
        )
    md_note = (
        "\n*Note*: Basin-wise paired differences between IC-CMA-ES and dPL-MLP for training performance "
        "(A = KGE_IC,train − KGE_dPL,train), testing performance (B = KGE_IC,test − KGE_dPL,test), "
        "and temporal transfer change (D = B − A). Summary statistics report median [95% bootstrap CI] "
        "across all n = 531 basins and five snow-fraction strata."
    )
    full_md = md_header + md_rows_str + md_note

    md_stats_path = os.path.join(out_stats_dir, "TableS3_ic_dpl_temporal_transfer.md")
    md_table_path = os.path.join(out_table_dir, "TableS3_ic_dpl_temporal_transfer.md")
    with open(md_stats_path, "w") as f:
        f.write(full_md)
    with open(md_table_path, "w") as f:
        f.write(full_md)

    # 2. LaTeX Table
    tex_rows_str = ""
    for r in rows:
        tex_rows_str += f"{r['Structure']} & {r['Scope']} & {r['n']} & {r['A_str']} & {r['B_str']} & {r['D_str']} \\\\\n"

    full_tex = (
        r"""\begin{table*}[t]
\centering
\caption{Temporal transfer of IC relative to dPL across structures and snow strata.}
\label{tab:s3_ic_dpl_temporal_transfer}
\begin{threeparttable}
\begin{tabular}{llcccc}
\toprule
Structure & Snow regime & n & A (Train: IC $-$ dPL) & B (Test: IC $-$ dPL) & D (Transfer Delta: B $-$ A) \\
\midrule
"""
        + tex_rows_str
        + r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Summary statistics represent basin-wise paired differences between IC-CMA-ES and dPL-MLP across $n = 531$ matched basins. $A = \text{KGE}_{\text{IC,train}} - \text{KGE}_{\text{dPL,train}}$, $B = \text{KGE}_{\text{IC,test}} - \text{KGE}_{\text{dPL,test}}$, and $D = B - A$. Bracketed values indicate 95\% bootstrap confidence intervals.
\end{tablenotes}
\end{threeparttable}
\end{table*}
"""
    )

    tex_stats_path = os.path.join(out_stats_dir, "TableS3_ic_dpl_temporal_transfer.tex")
    tex_table_path = os.path.join(out_table_dir, "TableS3_ic_dpl_temporal_transfer.tex")
    with open(tex_stats_path, "w") as f:
        f.write(full_tex)
    with open(tex_table_path, "w") as f:
        f.write(full_tex)

    # Copy generator script to manuscript/scripts/
    cp_script_target = os.path.join(
        project_root, "manuscript/scripts/r1/generate_table_s3.py"
    )
    if os.path.abspath(__file__) != os.path.abspath(cp_script_target):
        with open(__file__, "r") as sf, open(cp_script_target, "w") as df:
            df.write(sf.read())

    print("Table S3 generated successfully!")
    print(f"Markdown: {md_stats_path}")
    print(f"LaTeX: {tex_stats_path}")


if __name__ == "__main__":
    main()
