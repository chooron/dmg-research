#!/usr/bin/env python3
"""Generate Supplementary Table S2: Selected sensitivity analyses for diagnostic thresholds and reference-gap normalization.

Table S2 provides targeted sensitivity evidence across two panels:
- Panel A: R1 diagnostic-threshold sensitivity (KGE screen 0.40–0.80, CT thresholds 10/15/20 d for Base and CN).
- Panel B: R3 reference-outlet-gap denominator sensitivity (threshold cutoffs from 1e-6 to 0.10 for IC and dPL).
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATS_TABLE_DIR = PROJECT_ROOT / "manuscript" / "stats" / "tables"
SUPP_TABLE_DIR = PROJECT_ROOT / "manuscript" / "supplement" / "tables"

CT_CSV = PROJECT_ROOT / "manuscript" / "cache" / "r1_rebuild_audit_staged" / "r1_basin_level_ct.csv"
DENOM_CSV = PROJECT_ROOT / "manuscript" / "results" / "discussion_audit" / "r3_denominator_sensitivity_audit.csv"


def build_panel_a() -> pd.DataFrame:
    ct = pd.read_csv(CT_CSV)
    ct_test = ct[ct["period"] == "test"]

    rows = []
    for p in ["IC-CMA-ES", "dPL-MLP"]:
        p_label = "IC" if "IC" in p else "dPL"
        for kge_tau in [0.40, 0.50, 0.60, 0.70, 0.80]:
            for s in ["Base", "CN"]:
                sub = ct_test[(ct_test["paradigm"] == p) & (ct_test["structure"] == s)]
                scr = sub[sub["KGE"] >= kge_tau]
                n_scr = len(scr)
                n10 = int((scr["basin_median_Delta_CT"].abs() >= 10.0).sum())
                n15 = int((scr["basin_median_Delta_CT"].abs() >= 15.0).sum())
                n20 = int((scr["basin_median_Delta_CT"].abs() >= 20.0).sum())
                rows.append({
                    "Regime": p_label,
                    "KGE screening threshold (τ)": f"{kge_tau:.2f}",
                    "Configuration": s,
                    "Screened catchments (N)": n_scr,
                    "|ΔCT| ≥ 10 d": f"{n10} ({n10/n_scr*100:.1f}%)" if n_scr else "N/A",
                    "|ΔCT| ≥ 15 d": f"{n15} ({n15/n_scr*100:.1f}%)" if n_scr else "N/A",
                    "|ΔCT| ≥ 20 d": f"{n20} ({n20/n_scr*100:.1f}%)" if n_scr else "N/A",
                })
    return pd.DataFrame(rows)


def build_panel_b() -> pd.DataFrame:
    denom = pd.read_csv(DENOM_CSV)
    denom_test = denom[denom["period"] == "test"].copy()

    threshold_labels = {
        1e-06: "$D_b > 10^{-6}$ (Canonical)",
        0.0001: "$D_b > 10^{-4}$",
        0.001: "$D_b > 10^{-3}$",
        0.01: "$D_b > 0.01$",
        0.02: "$D_b > 0.02$",
        0.05: "$D_b > 0.05$",
        0.10: "$D_b > 0.10$",
    }

    rows = []
    for _, r in denom_test.iterrows():
        p_label = r["paradigm"]
        th_val = r["threshold"]
        th_str = threshold_labels.get(th_val, f"$D_b > {th_val}$")
        n_val = int(r["n_valid"])
        v_rate = float(r["valid_rate"]) * 100.0
        f_close = float(r["F_close_median"])
        f_tgd = float(r["F_TGD_median"])
        delta_f = float(r["delta_F_median"])
        p_pos = float(r["delta_F_gt0_prop"]) * 100.0

        rows.append({
            "Regime": p_label,
            "Denominator criterion": th_str,
            "Valid catchments (N, %)": f"{n_val} ({v_rate:.1f}%)",
            "F_close (median)": f"{f_close:.3f}",
            "F_TGD (median)": f"{f_tgd:.3f}",
            "ΔF (median)": f"{delta_f:+.3f}",
            "P(ΔF > 0)": f"{p_pos:.1f}%",
        })
    return pd.DataFrame(rows)


def generate_markdown(df_a: pd.DataFrame, df_b: pd.DataFrame) -> str:
    md = [
        "# Table S2: Selected Sensitivity Analyses for Diagnostic Thresholds and Reference-Gap Normalization",
        "",
        "### Panel A: R1 Diagnostic-Threshold Sensitivity (Screened Runoff Timing Errors for Base and CN)",
        "",
        "| Regime | KGE screen (τ) | Configuration | Screened N | |ΔCT| ≥ 10 d | |ΔCT| ≥ 15 d | |ΔCT| ≥ 20 d |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in df_a.iterrows():
        md.append(f"| {r['Regime']} | {r['KGE screening threshold (τ)']} | {r['Configuration']} | {r['Screened catchments (N)']} | {r['|ΔCT| ≥ 10 d']} | {r['|ΔCT| ≥ 15 d']} | {r['|ΔCT| ≥ 20 d']} |")

    md.extend([
        "",
        "### Panel B: R3 Reference-Outlet-Gap Denominator Sensitivity (Recovery Fractions across Cutoffs)",
        "",
        "| Regime | Denominator criterion | Valid N (rate) | F_close (median) | F_TGD (median) | ΔF (median) | P(ΔF > 0) |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for _, r in df_b.iterrows():
        crit = r["Denominator criterion"].replace("$", "")
        md.append(f"| {r['Regime']} | {crit} | {r['Valid catchments (N, %)']} | {r['F_close (median)']} | {r['F_TGD (median)']} | {r['ΔF (median)']} | {r['P(ΔF > 0)']} |")

    md.extend([
        "",
        "*Note*: Panel A reports the number and percentage of catchments satisfying large center-of-timing error thresholds ($|\\Delta CT| \\ge 10, 15, 20\\text{ d}$) across KGE screening cutoffs $\\tau \\in [0.40, 0.80]$ during the evaluation period (1995–2010), complementing the continuous curves in Fig. 2i. "
        "Panel B evaluates the sensitivity of catchment-wise normalized recovery fractions ($F_{\\mathrm{close}} = G_{\\mathrm{Base}} / D_b$, $F_{\\mathrm{TGD}} = G_{\\mathrm{TGD}} / D_b$, and $\\Delta F = F_{\\mathrm{TGD}} - F_{\\mathrm{close}}$) to the reference-outlet-gap denominator threshold $D_b = \\mathrm{KGE}(\\mathrm{CN}) - \\mathrm{KGE}(\\mathrm{Base}_{\\mathrm{no\\text{-}refit}})$, demonstrating that the structural contrast $\\Delta F \\approx +0.44\\text{--}+0.46$ is invariant across screening thresholds from $10^{-6}$ to $0.10$."
    ])
    return "\n".join(md) + "\n"


def generate_latex(df_a: pd.DataFrame, df_b: pd.DataFrame) -> str:
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{threeparttable}",
        r"\caption{Selected sensitivity analyses for diagnostic screening thresholds (Panel A) and reference-outlet-gap denominator normalization (Panel B).}",
        r"\label{tab:sensitivity_audits}",
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        r"\multicolumn{7}{l}{\textbf{Panel A: R1 Diagnostic-Threshold Sensitivity (Screened Timing Error Prevalence for Base and CN)}} \\",
        r"\midrule",
        r"Regime & KGE screen ($\tau$) & Configuration & Screened $N$ & $|\Delta CT| \ge 10\text{ d}$ & $|\Delta CT| \ge 15\text{ d}$ & $|\Delta CT| \ge 20\text{ d}$ \\",
        r"\midrule",
    ]
    for _, r in df_a.iterrows():
        reg = r["Regime"]
        tau = r["KGE screening threshold (τ)"]
        cfg = r["Configuration"]
        n_scr = r["Screened catchments (N)"]
        c10 = r["|ΔCT| ≥ 10 d"].replace("%", r"\%")
        c15 = r["|ΔCT| ≥ 15 d"].replace("%", r"\%")
        c20 = r["|ΔCT| ≥ 20 d"].replace("%", r"\%")
        tex.append(f"{reg} & {tau} & {cfg} & {n_scr} & {c10} & {c15} & {c20} \\\\")

    tex.extend([
        r"\midrule",
        r"\multicolumn{7}{l}{\textbf{Panel B: R3 Reference-Outlet-Gap Denominator Sensitivity (Normalized Recovery across Cutoffs)}} \\",
        r"\midrule",
        r"Regime & Denominator criterion & Valid $N$ (rate) & $F_{\mathrm{close}}$ (med.) & $F_{\mathrm{TGD}}$ (med.) & $\Delta F$ (med.) & $P(\Delta F > 0)$ \\",
        r"\midrule",
    ])
    for _, r in df_b.iterrows():
        reg = r["Regime"]
        crit = r["Denominator criterion"]
        v_n = r["Valid catchments (N, %)"].replace("%", r"\%")
        fc = r["F_close (median)"]
        ft = r["F_TGD (median)"]
        df_v = r["ΔF (median)"]
        p_pos = r["P(ΔF > 0)"].replace("%", r"\%")
        tex.append(f"{reg} & {crit} & {v_n} & {fc} & {ft} & {df_v} & {p_pos} \\\\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item \textit{Note}: Panel A reports the prevalence of large center-of-timing errors ($|\Delta CT| \ge 10, 15, 20\text{ d}$) across KGE screening cutoffs $\tau \in [0.40, 0.80]$ in the independent evaluation period (1995--2010), complementing Fig. 2i. "
        r"Panel B summarizes the sensitivity of catchment-wise normalized recovery fractions ($F_{\mathrm{close}} = G_{\mathrm{Base}} / D_b$, $F_{\mathrm{TGD}} = G_{\mathrm{TGD}} / D_b$, and $\Delta F = F_{\mathrm{TGD}} - F_{\mathrm{close}}$) to alternative reference-outlet-gap denominator thresholds $D_b = \mathrm{KGE}_{\mathrm{CN}} - \mathrm{KGE}_{\mathrm{Base\_norefit}}$, confirming that the paired contrast $\Delta F \approx +0.44\text{--}+0.46$ remains invariant from $10^{-6}$ to $0.10$.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table*}",
    ])
    return "\n".join(tex) + "\n"


def main():
    STATS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SUPP_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df_a = build_panel_a()
    df_b = build_panel_b()

    # Save combined CSV
    df_a_out = df_a.copy()
    df_a_out["Panel"] = "A_R1_threshold_sensitivity"
    df_b_out = df_b.copy()
    df_b_out["Panel"] = "B_R3_denominator_sensitivity"

    csv_path_a = STATS_TABLE_DIR / "TableS2_PanelA_threshold_sensitivity.csv"
    csv_path_b = STATS_TABLE_DIR / "TableS2_PanelB_denominator_sensitivity.csv"
    df_a.to_csv(csv_path_a, index=False)
    df_b.to_csv(csv_path_b, index=False)

    md_content = generate_markdown(df_a, df_b)
    (STATS_TABLE_DIR / "TableS2_sensitivity_audits.md").write_text(md_content, encoding="utf-8")
    (SUPP_TABLE_DIR / "TableS2_sensitivity_audits.md").write_text(md_content, encoding="utf-8")

    tex_content = generate_latex(df_a, df_b)
    (STATS_TABLE_DIR / "TableS2_sensitivity_audits.tex").write_text(tex_content, encoding="utf-8")
    (SUPP_TABLE_DIR / "TableS2_sensitivity_audits.tex").write_text(tex_content, encoding="utf-8")

    print(f"Table S2 generated successfully:\n  {STATS_TABLE_DIR / 'TableS2_sensitivity_audits.md'}\n  {STATS_TABLE_DIR / 'TableS2_sensitivity_audits.tex'}")


if __name__ == "__main__":
    main()
