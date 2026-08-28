#!/usr/bin/env python3
"""Generate Main Text Table 2: Controlled recovery of the reference outlet gap.

Table 2 is the primary quantitative result table for Section 3.3 / Discussion 4.2.
It documents:
1. The limited ability of 15-parameter Base recalibration to recover the reference outlet gap;
2. The mitigation provided by the temperature-conditioned generic storage control (TGD);
3. The paired difference between TGD mitigation and Base parameter compensation.

All values are loaded directly from authoritative frozen canonical results in
`manuscript/results/discussion_audit/r3_gap_recovery_ratio_audit.csv`.
"""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_CSV = PROJECT_ROOT / "manuscript" / "results" / "discussion_audit" / "r3_gap_recovery_ratio_audit.csv"
STATS_TABLE_DIR = PROJECT_ROOT / "manuscript" / "stats" / "tables"
TABLES_DIR = PROJECT_ROOT / "manuscript" / "tables"


def fmt_ci(med: float, low: float, high: float, decimals: int = 3, is_signed: bool = False) -> str:
    fmt_str = f"{{:+0.{decimals}f}}" if is_signed else f"{{:0.{decimals}f}}"
    med_str = fmt_str.format(med)
    low_str = fmt_str.format(low)
    high_str = fmt_str.format(high)
    return f"{med_str} [{low_str}, {high_str}]"


def load_table2_data() -> pd.DataFrame:
    df_audit = pd.read_csv(AUDIT_CSV)
    df_test_full = df_audit[(df_audit["period"] == "test") & (df_audit["snow_stratum"] == "Full")].set_index("paradigm")

    ic_row = df_test_full.loc["IC"]
    dpl_row = df_test_full.loc["dPL"]

    rows = [
        {
            "Quantity": "Denominator-valid catchments, $N_{\\mathrm{valid}}$ (rate)",
            "IC": f"{int(ic_row['n_valid'])} ({ic_row['valid_rate'] * 100:.1f}%)",
            "dPL": f"{int(dpl_row['n_valid'])} ({dpl_row['valid_rate'] * 100:.1f}%)",
        },
        {
            "Quantity": "Reference outlet gap, $D_b = \\mathrm{KGE}(\\mathrm{CN}) - \\mathrm{KGE}(\\mathrm{Base}_{\\mathrm{no\\text{-}refit}})$",
            "IC": fmt_ci(ic_row["D_median"], ic_row["D_ci_low"], ic_row["D_ci_high"], 3, True),
            "dPL": fmt_ci(dpl_row["D_median"], dpl_row["D_ci_low"], dpl_row["D_ci_high"], 3, True),
        },
        {
            "Quantity": "Raw Base-refit gain, $G_{\\mathrm{Base}} = \\mathrm{KGE}(\\mathrm{Base}_{\\mathrm{refit}}) - \\mathrm{KGE}(\\mathrm{Base}_{\\mathrm{no\\text{-}refit}})$",
            "IC": fmt_ci(ic_row["G_base_median"], ic_row["G_base_ci_low"], ic_row["G_base_ci_high"], 4, True),
            "dPL": fmt_ci(dpl_row["G_base_median"], dpl_row["G_base_ci_low"], dpl_row["G_base_ci_high"], 4, True),
        },
        {
            "Quantity": "Raw TGD generic gain, $G_{\\mathrm{TGD}} = \\mathrm{KGE}(\\mathrm{TGD}) - \\mathrm{KGE}(\\mathrm{Base}_{\\mathrm{no\\text{-}refit}})$",
            "IC": fmt_ci(ic_row["G_TGD_median"], ic_row["G_TGD_ci_low"], ic_row["G_TGD_ci_high"], 4, True),
            "dPL": fmt_ci(dpl_row["G_TGD_median"], dpl_row["G_TGD_ci_low"], dpl_row["G_TGD_ci_high"], 4, True),
        },
        {
            "Quantity": "Recalibration gap-closure fraction, $F_{\\mathrm{close}} = G_{\\mathrm{Base}} / D_b$",
            "IC": fmt_ci(ic_row["F_close_median"], ic_row["F_close_ci_low"], ic_row["F_close_ci_high"], 3),
            "dPL": fmt_ci(dpl_row["F_close_median"], dpl_row["F_close_ci_low"], dpl_row["F_close_ci_high"], 3),
        },
        {
            "Quantity": "Generic-control recovery fraction, $F_{\\mathrm{TGD}} = G_{\\mathrm{TGD}} / D_b$",
            "IC": fmt_ci(ic_row["F_TGD_median"], ic_row["F_TGD_ci_low"], ic_row["F_TGD_ci_high"], 3),
            "dPL": fmt_ci(dpl_row["F_TGD_median"], dpl_row["F_TGD_ci_low"], dpl_row["F_TGD_ci_high"], 3),
        },
        {
            "Quantity": "Paired recovery-fraction difference, $\\Delta F = F_{\\mathrm{TGD}} - F_{\\mathrm{close}}$",
            "IC": fmt_ci(ic_row["delta_F_median"], ic_row["delta_F_ci_low"], ic_row["delta_F_ci_high"], 3, True),
            "dPL": fmt_ci(dpl_row["delta_F_median"], dpl_row["delta_F_ci_low"], dpl_row["delta_F_ci_high"], 3, True),
        },
        {
            "Quantity": "Positive paired fraction, $P(F_{\\mathrm{TGD}} > F_{\\mathrm{close}})$",
            "IC": f"{ic_row['delta_F_gt0_prop'] * 100:.1f}%",
            "dPL": f"{dpl_row['delta_F_gt0_prop'] * 100:.1f}%",
        },
    ]
    return pd.DataFrame(rows)


def clean_markdown_math(text: str) -> str:
    s = text.replace(r"\mathrm{valid}", "valid")
    s = s.replace(r"\mathrm{CN}", "CN")
    s = s.replace(r"\mathrm{Base}_{\mathrm{no\text{-}refit}}", "Base_norefit")
    s = s.replace(r"\mathrm{Base}_{\mathrm{refit}}", "Base_refit")
    s = s.replace(r"\mathrm{Base}", "Base")
    s = s.replace(r"\mathrm{TGD}", "TGD")
    s = s.replace(r"\mathrm{close}", "close")
    s = s.replace(r"\mathrm{", "").replace("}", "")
    s = s.replace(r"\Delta", "Δ")
    s = s.replace(r"\_", "_")
    s = s.replace(r"\text{-}", "-")
    s = s.replace("{", "").replace("}", "")
    s = s.replace("$", "")
    return s
def generate_markdown(df: pd.DataFrame) -> str:
    md = [
        "# Table 2: Controlled Recovery of the Reference Outlet Gap",
        "",
        "| Quantity | IC (Independent Calibration) | dPL (Differentiable Parameter Learning) |",
        "| :--- | :---: | :---: |",
    ]
    for _, row in df.iterrows():
        quant = clean_markdown_math(row["Quantity"])
        md.append(f"| {quant} | {row['IC']} | {row['dPL']} |")

    md.extend([
        "",
        "*Note*: Values report catchment-wise medians with marginal 95% bootstrap confidence intervals [2.5th, 97.5th percentiles] across denominator-valid catchments ($D_b = \\mathrm{KGE}(\\mathrm{CN}) - \\mathrm{KGE}(\\mathrm{Base}_{\\mathrm{no\\text{-}refit}}) > 10^{-6}$; 2,000 paired catchment resamples, seed 20260730). "
        "Here, $D_b$ represents the controlled reference outlet gap induced by the imposed snow-process omission. "
        "dPL values represent per-catchment seed medians across the three canonical training runs (seeds 42, 123, 2026). "
        "$N_{\\mathrm{valid}}$ denotes catchments satisfying the R3 reference-outlet-gap denominator criterion ($D_b > 10^{-6}$) and is unrelated to the KGE-screened subsets in Sect. 3.1. "
        "$F_{\\mathrm{close}} = G_{\\mathrm{Base}} / D_b$ and $F_{\\mathrm{TGD}} = G_{\\mathrm{TGD}} / D_b$ are computed as catchment-wise ratios prior to population summarization and are not ratios of population medians. "
        "Similarly, $\\Delta F$ is computed catchment-wise as $F_{\\mathrm{TGD}} - F_{\\mathrm{close}}$ prior to summarization; hence $\\mathrm{median}(\\Delta F) \\neq \\mathrm{median}(F_{\\mathrm{TGD}}) - \\mathrm{median}(F_{\\mathrm{close}})$. "
        "$G_{\\mathrm{Base}}$ and $G_{\\mathrm{TGD}}$ denote primary raw paired KGE gains relative to uncalibrated structural knockout. "
        "For catchments where $D_b > 0$, the sign condition $F_{\\mathrm{TGD}} > F_{\\mathrm{close}}$ is algebraically equivalent to $G_{\\mathrm{TGD}} > G_{\\mathrm{Base}}$. "
        "Denominator sensitivity across alternative cutoffs is reported in Table S2 Panel B. "
        "IC and dPL represent parallel parameter-estimation regimes evaluated under identical sample selection."
    ])
    return "\n".join(md) + "\n"


def generate_latex(df: pd.DataFrame) -> str:
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{threeparttable}",
        r"\caption{Controlled recovery of the reference outlet gap under Base parameter recalibration and temperature-conditioned generic control (TGD) in known-truth synthetic experiments.}",
        r"\label{tab:controlled_recovery}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Quantity & IC & dPL \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        quant = row["Quantity"]
        ic = row["IC"]
        dpl = row["dPL"]
        tex.append(f"{quant} & {ic} & {dpl} \\\\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item \textit{Note}: Values report catchment-wise medians with marginal 95\% bootstrap confidence intervals [2.5th, 97.5th percentiles] across denominator-valid catchments ($D_b = \mathrm{KGE}(\mathrm{CN}) - \mathrm{KGE}(\mathrm{Base}_{\mathrm{no\text{-}refit}}) > 10^{-6}$; 2,000 paired catchment resamples, seed 20260730). "
        r"Here, $D_b$ represents the controlled reference outlet gap induced by the imposed snow-process omission. "
        r"dPL values represent per-catchment seed medians across seeds 42, 123, and 2026. "
        r"$N_{\mathrm{valid}}$ denotes catchments satisfying the R3 reference-outlet-gap denominator criterion ($D_b > 10^{-6}$) and is unrelated to the KGE-screened subsets in Sect. 3.1. "
        r"$F_{\mathrm{close}} = G_{\mathrm{Base}} / D_b$ and $F_{\mathrm{TGD}} = G_{\mathrm{TGD}} / D_b$ are computed as catchment-wise ratios prior to population summarization and are not ratios of population medians. "
        r"Similarly, $\Delta F$ is computed catchment-wise as $F_{\mathrm{TGD}} - F_{\mathrm{close}}$ prior to summarization; hence $\mathrm{median}(\Delta F) \neq \mathrm{median}(F_{\mathrm{TGD}}) - \mathrm{median}(F_{\mathrm{close}})$. "
        r"$G_{\mathrm{Base}}$ and $G_{\mathrm{TGD}}$ denote primary raw paired KGE gains relative to uncalibrated structural knockout. "
        r"For catchments where $D_b > 0$, the sign condition $F_{\mathrm{TGD}} > F_{\mathrm{close}}$ is algebraically equivalent to $G_{\mathrm{TGD}} > G_{\mathrm{Base}}$. "
        r"Denominator sensitivity across alternative cutoffs is reported in Table S2 Panel B. "
        r"IC and dPL represent parallel parameter-estimation regimes evaluated under identical sample selection.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ])
    return "\n".join(tex) + "\n"


def main():
    STATS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_table2_data()

    # Write CSV
    csv_path = STATS_TABLE_DIR / "Table2_controlled_recovery.csv"
    df.to_csv(csv_path, index=False)

    # Write Markdown
    md_content = generate_markdown(df)
    (STATS_TABLE_DIR / "Table2_controlled_recovery.md").write_text(md_content, encoding="utf-8")
    (TABLES_DIR / "Table2_controlled_recovery.md").write_text(md_content, encoding="utf-8")

    # Write LaTeX
    tex_content = generate_latex(df)
    (STATS_TABLE_DIR / "Table2_controlled_recovery.tex").write_text(tex_content, encoding="utf-8")
    (TABLES_DIR / "Table2_controlled_recovery.tex").write_text(tex_content, encoding="utf-8")

    print(f"Table 2 generated successfully:\n  {csv_path}\n  {STATS_TABLE_DIR / 'Table2_controlled_recovery.md'}\n  {STATS_TABLE_DIR / 'Table2_controlled_recovery.tex'}")


if __name__ == "__main__":
    main()
