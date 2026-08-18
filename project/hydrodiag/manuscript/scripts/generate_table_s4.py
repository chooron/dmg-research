#!/usr/bin/env python3
"""
Table S4 Generator (R2 Supplement)
Generates TableS4_exact_estimates_f3_f4.md and TableS4_exact_estimates_f3_f4.tex

Single compact R2 SI table with two result-only panels:
- Panel A: structure-level snow gradients underlying Figure 3 (OLS slope of
  basin-level excess distance against frac_snow) for the Base-CN and Base-TGD2
  structural contrasts under IC / dPL and Full / Excl. S5 subsets, plus the paired
  slope difference DeltaBeta = beta(Base-CN) - beta(Base-TGD2).
- Panel B: parameter-level snow gradients underlying Figure 4 (OLS slope of the
  canonical paired shift delta = z_Base - z_CN against frac_snow) for the
  highlighted parameters um, ki, ci and the qualified parameter im, under IC / dPL
  and Full / Excl. S5 subsets.

All values are read verbatim from the frozen R2 result files
(results/R2/r2_tgd2_specificity_regressions.csv,
 results/R2/r2_tgd2_slope_difference_summary.csv,
 results/R2/r2_snow_gradients_summary.csv,
 results/R2/r2_snow_gradient_robustness.csv);
no analysis is recomputed and no interpretation column is added.
"""

import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MANUSCRIPT = HERE.parent
R2 = MANUSCRIPT / "results" / "R2"
OUT_DIRS = (MANUSCRIPT / "stats" / "tables", MANUSCRIPT / "tables")


def fmt(v, decimals=3, force_sign=False):
    s = f"{float(v):.{decimals}f}"
    return f"+{s}" if force_sign and float(v) > 0 else s


def fmt_ci(lo, hi, decimals=3, force_sign=False):
    return f"[{fmt(lo, decimals, force_sign)}, {fmt(hi, decimals, force_sign)}]"


def load():
    reg = pd.read_csv(R2 / "r2_tgd2_specificity_regressions.csv")
    reg = reg[reg["dependent_var"] == "excess"]
    diff = pd.read_csv(R2 / "r2_tgd2_slope_difference_summary.csv")
    grad_full = pd.read_csv(R2 / "r2_snow_gradients_summary.csv")
    grad_rob = pd.read_csv(R2 / "r2_snow_gradient_robustness.csv")
    return reg, diff, grad_full, grad_rob


def panel_a_rows(reg, diff):
    """Return (md_rows, tex_rows): one row per regime x subset."""
    subset_map = {"Full531": ("Full", 531), "ExcludeS5": ("Excl. S5", 476)}
    out_md, out_tex = [], []
    for paradigm in ["IC", "dPL"]:
        for stratum, (label, n) in subset_map.items():
            row_md = {"Regime": paradigm, "Subset": label, "n": n}
            row_tex = [paradigm, label, str(n)]
            for contrast in ["Base-CN", "Base-TGD2"]:
                r = reg[(reg["paradigm"] == paradigm) & (reg["contrast"] == contrast)
                        & (reg["stratum"] == stratum)]
                assert len(r) == 1, f"missing {paradigm} {contrast} {stratum}"
                r = r.iloc[0]
                beta = fmt(r["slope"], 3, force_sign=True)
                ci = fmt_ci(r["slope_ci_lower"], r["slope_ci_upper"], 3, force_sign=True)
                row_md[f"beta_{contrast}"] = f"{beta} {ci}"
                row_tex.append(f"{beta} {ci}")
            d = diff[(diff["paradigm"] == paradigm) & (diff["stratum"] == stratum)]
            assert len(d) == 1, f"missing delta {paradigm} {stratum}"
            d = d.iloc[0]
            db = fmt(d["delta_beta"], 3, force_sign=True)
            dci = fmt_ci(d["delta_beta_ci_lower"], d["delta_beta_ci_upper"], 3, force_sign=True)
            row_md["delta_beta"] = f"{db} {dci}"
            row_tex.append(f"{db} {dci}")
            out_md.append(row_md)
            out_tex.append(row_tex)
    return out_md, out_tex


def panel_b_rows(grad_full, grad_rob):
    """One row per parameter x regime x subset (Full / Excl. S5)."""
    params = ["xaj_um", "xaj_ki", "xaj_ci", "xaj_im"]
    display = {"xaj_um": "$u_m$", "xaj_ki": "$k_i$", "xaj_ci": "$c_i$",
               "xaj_im": "$i_m$"}
    out_md, out_tex = [], []
    for p in params:
        for paradigm in ["IC", "dPL"]:
            for subset, n in [("full_531", 531), ("exclude_S5", 476)]:
                if subset == "full_531":
                    r = grad_full[(grad_full["paradigm"] == paradigm)
                                  & (grad_full["parameter"] == p)]
                    bcol, locol, hicol = "beta", "ci95_low", "ci95_high"
                else:
                    r = grad_rob[(grad_rob["paradigm"] == paradigm)
                                 & (grad_rob["parameter"] == p)
                                 & (grad_rob["subset"] == subset)]
                    bcol, locol, hicol = "slope", "ci95_low", "ci95_high"
                assert len(r) == 1, f"missing {p} {paradigm} {subset}"
                r = r.iloc[0]
                beta = fmt(r[bcol], 3, force_sign=True)
                ci = fmt_ci(r[locol], r[hicol], 3, force_sign=True)
                out_md.append({"Parameter": display[p], "Regime": paradigm,
                               "Subset": "Full" if subset == "full_531" else "Excl. S5",
                               "n": n, "beta": f"{beta} {ci}"})
                out_tex.append([display[p], paradigm,
                                "Full" if subset == "full_531" else "Excl. S5",
                                str(n), f"{beta} {ci}", ""])
    return out_md, out_tex


def main():
    reg, diff, grad_full, grad_rob = load()
    pa_md, pa_tex = panel_a_rows(reg, diff)
    pb_md, pb_tex = panel_b_rows(grad_full, grad_rob)

    md = """# Table S4: Exact Estimates Underlying Figures 3 and 4 (R2)

### Panel A: Structure-Level Snow Gradients Underlying Figure 3

| Regime | Subset | n | Base $-$ CN $\\beta$ [95% CI] | Base $-$ TGD2 $\\beta$ [95% CI] | $\\Delta\\beta$ [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for r in pa_md:
        md += (f"| {r['Regime']} | {r['Subset']} | {r['n']} | "
               f"{r['beta_Base-CN']} | {r['beta_Base-TGD2']} | {r['delta_beta']} |\n")

    md += """
### Panel B: Parameter-Level Snow Gradients Underlying Figure 4

| Parameter | Regime | Subset | n | $\\beta$ [95% CI] |
| :--- | :--- | :--- | :---: | :---: |
"""
    for r in pb_md:
        md += (f"| {r['Parameter']} | {r['Regime']} | {r['Subset']} | "
               f"{r['n']} | {r['beta']} |\n")

    md += """
*Note*: Panel A reports the OLS slope of the basin-level excess distance
(excess = between-all distance $-$ pooled within-structure baseline) against basin
snow fraction $f_{\\mathrm{snow}}$ for the two structural contrasts, with 95%
basin-level bootstrap confidence intervals (10,000 resamples, fixed seed) and the
paired slope difference $\\Delta\\beta = \\beta(\\mathrm{Base\\text{--}CN}) -
\\beta(\\mathrm{Base\\text{--}TGD2})$ estimated with a basin-paired bootstrap.
Panel B reports the OLS slope of the canonical paired shift
$\\Delta z = z_{\\mathrm{Base}} - z_{\\mathrm{CN}}$ against $f_{\\mathrm{snow}}$ for
the parameters highlighted in Figure 4 and the qualified parameter $i_m$, with the
same bootstrap CI protocol. All estimates are dimensionless normalized-parameter
slopes per unit $f_{\\mathrm{snow}}$; $n$ is the number of matched basins. Snow
regimes S1\u2013S5 are the fixed strata by basin snow fraction: S1 $[0, 0.05)$
($n=165$), S2 $[0.05, 0.15)$ ($n=156$), S3 $[0.15, 0.30)$ ($n=121$),
S4 $[0.30, 0.50)$ ($n=34$), S5 $[0.50, 1.00]$ ($n=55$).
"""

    pa_tex_body = "\n".join(" & ".join(r) + " \\\\" for r in pa_tex)
    pb_tex_body = "\n".join(" & ".join(r) + " \\\\" for r in pb_tex)

    tex = r"""\begin{table*}[t]
\centering
\caption{Exact estimates underlying Figures 3 and 4 (R2). Panel A: structure-level OLS snow gradients of the basin-level excess distance ($\mathrm{excess} = \mathrm{between\_all} - \mathrm{within\_pooled}$) against basin snow fraction $f_{\mathrm{snow}}$, for the Base--CN and Base--TGD2 structural contrasts under the IC and dPL parameter-constraint regimes and Full / Excl.\ S5 subsets, with 95\% basin-level bootstrap confidence intervals and the basin-paired slope difference $\Delta\beta = \beta(\mathrm{Base\text{--}CN}) - \beta(\mathrm{Base\text{--}TGD2})$. Panel B: parameter-level OLS snow gradients of the canonical paired shift $\Delta z = z_{\mathrm{Base}} - z_{\mathrm{CN}}$ against $f_{\mathrm{snow}}$ for the parameters highlighted in Figure 4 and the qualified parameter $i_m$, with the same bootstrap CI protocol.}
\label{tab:tables4_exact_estimates_f3_f4}
\begin{threeparttable}
\begin{tabular}{lccccc}
\toprule
\multicolumn{6}{l}{\textbf{Panel A: Structure-level snow gradients underlying Figure 3}} \\
\midrule
Regime & Subset & $n$ & Base $-$ CN $\beta$ [95\% CI] & Base $-$ TGD2 $\beta$ [95\% CI] & $\Delta\beta$ [95\% CI] \\
\midrule
""" + pa_tex_body + r"""
\midrule
\multicolumn{6}{l}{\textbf{Panel B: Parameter-level snow gradients underlying Figure 4}} \\
\midrule
Parameter & Regime & Subset & $n$ & $\beta$ [95\% CI] & \\
\midrule
""" + pb_tex_body + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]
\small
\item \textit{Note}: Panel A reports the OLS slope of the basin-level excess distance against $f_{\mathrm{snow}}$ for the two structural contrasts, with 95\% basin-level bootstrap CIs (10,000 resamples, fixed seed) and the basin-paired slope difference $\Delta\beta$. Panel B reports the OLS slope of the canonical paired shift $\Delta z = z_{\mathrm{Base}} - z_{\mathrm{CN}}$ against $f_{\mathrm{snow}}$, same CI protocol. All estimates are dimensionless normalized-parameter slopes per unit $f_{\mathrm{snow}}$; $n$ is the number of matched basins. Snow regimes S1--S5 are the fixed strata by basin snow fraction: S1 $[0, 0.05)$ ($n=165$), S2 $[0.05, 0.15)$ ($n=156$), S3 $[0.15, 0.30)$ ($n=121$), S4 $[0.30, 0.50)$ ($n=34$), S5 $[0.50, 1.00]$ ($n=55$).
\end{tablenotes}
\end{threeparttable}
\end{table*}
"""

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        (d / "TableS4_exact_estimates_f3_f4.md").write_text(md)
        (d / "TableS4_exact_estimates_f3_f4.tex").write_text(tex)
    print("Supplementary Table S4 generated successfully.")


if __name__ == "__main__":
    main()
