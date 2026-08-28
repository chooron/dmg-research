#!/usr/bin/env python3
"""Generate the R2 Supplement Table S4 from frozen canonical outputs.

No statistics are recomputed.  Panel A mirrors the paired slope summary used
by Figure 3(g), and Panel B mirrors the full all-15 gradient overview used by
Figure 4(a).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MANUSCRIPT = HERE.parents[1]
R2 = MANUSCRIPT / "analysis" / "R2" / "results"
OUT_DIRS = (MANUSCRIPT / "stats" / "tables",)


def fmt(value: float, decimals: int = 3) -> str:
    value = float(value)
    return f"{value:+.{decimals}f}"


def fmt_ci(low: float, high: float) -> str:
    return f"[{fmt(low)}, {fmt(high)}]"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    slope = pd.read_csv(R2 / "r2_tgd2_slope_difference_summary.csv")
    gradients = pd.read_csv(R2 / "r2_snow_gradients_summary.csv")
    assert set(slope["paradigm"]) == {"IC", "dPL"}
    assert set(slope["stratum"]) == {"Full531", "ExcludeS5"}
    assert set(gradients["contrast"]) == {"Base-CN"}
    assert len(gradients) == 30
    return slope, gradients


def main() -> None:
    slope, gradients = load()
    panel_a = []
    for paradigm in ["IC", "dPL"]:
        for subset, n in [("Full531", 531), ("ExcludeS5", 476)]:
            row = slope[(slope["paradigm"] == paradigm) & (slope["stratum"] == subset)]
            assert len(row) == 1
            row = row.iloc[0]
            panel_a.append({
                "regime": paradigm,
                "subset": subset,
                "n": n,
                "cn": f"{fmt(row['beta_Base_CN'])} {fmt_ci(row['beta_Base_CN_ci_lower'], row['beta_Base_CN_ci_upper'])}",
                "tgd": f"{fmt(row['beta_Base_TGD'])} {fmt_ci(row['beta_Base_TGD_ci_lower'], row['beta_Base_TGD_ci_upper'])}",
                "delta": f"{fmt(row['delta_beta'])} {fmt_ci(row['delta_beta_ci_lower'], row['delta_beta_ci_upper'])}",
            })

    specs = gradients[["parameter", "symbol"]].drop_duplicates().reset_index(drop=True)
    panel_b = []
    for _, spec in specs.iterrows():
        row = {"parameter": spec["symbol"]}
        for paradigm in ["IC", "dPL"]:
            hit = gradients[(gradients["parameter"] == spec["parameter"]) & (gradients["paradigm"] == paradigm)]
            assert len(hit) == 1
            hit = hit.iloc[0]
            row[paradigm] = f"{fmt(hit['beta'])} {fmt_ci(hit['slope_ci_low'], hit['slope_ci_high'])}"
        panel_b.append(row)

    md = """# Table S4: Exact Estimates Underlying Figures 3 and 4 (R2)

### Panel A: Paired slope contrast underlying Figure 3(g)

| Regime | Subset | n | Base − CN β [95% CI] | Base − TGD β [95% CI] | Paired Δβ [95% CI] |
|---|---:|---:|---:|---:|---:|
"""
    for row in panel_a:
        md += f"| {row['regime']} | {row['subset']} | {row['n']} | {row['cn']} | {row['tgd']} | {row['delta']} |\n"
    md += """
### Panel B: Full-sample all-15 parameter gradients underlying Figure 4(a)

| Parameter | IC β [95% CI] | dPL β [95% CI] |
|---|---:|---:|
"""
    for row in panel_b:
        md += f"| ${row['parameter']}$ | {row['IC']} | {row['dPL']} |\n"
    md += """
*Note:* Panel A uses the frozen basin-paired slope contrast
$\\Delta\\beta = \\beta(\\mathrm{Base-CN}) - \\beta(\\mathrm{Base-TGD})$.
Panel B uses the canonical normalized shift $\\Delta z = z_{\\mathrm{Base}} - z_{\\mathrm{CN}}$.
All quantities are dimensionless slopes per unit basin snow fraction. The full S1–S5 median-shift organization and the four recurring signatures shown in Figure 4 are read from the canonical R2 strata summary.
"""

    tex = r"""\begin{table*}[t]
\centering
\caption{Exact estimates underlying Figures 3 and 4 (R2). Panel A reports the frozen paired slope contrast $\Delta\beta = \beta(\mathrm{Base\text{--}CN}) - \beta(\mathrm{Base\text{--}TGD})$ for the Figure 3 attribution summary. Panel B reports the full-sample Base--CN snow-gradient slopes for all 15 shared parameters shown in Figure 4(a), using the canonical normalized shift $\Delta z = z_{\mathrm{Base}} - z_{\mathrm{CN}}$.}
\label{tab:tables4_exact_estimates_f3_f4}
\begin{threeparttable}
\textbf{Panel A: Paired slope contrast underlying Figure 3(g)}
\begin{tabular}{lccccc}
\toprule
Regime & Subset & $n$ & Base $-$ CN $\beta$ [95\% CI] & Base $-$ TGD $\beta$ [95\% CI] & $\Delta\beta$ [95\% CI] \\
\midrule
"""
    for row in panel_a:
        tex += f"{row['regime']} & {row['subset']} & {row['n']} & {row['cn']} & {row['tgd']} & {row['delta']} \\\\\n"
    tex += r"""\bottomrule
\end{tabular}

\medskip
\textbf{Panel B: Full-sample all-15 parameter gradients underlying Figure 4(a)}
\begin{tabular}{lcc}
\toprule
Parameter & IC $\beta$ [95\% CI] & dPL $\beta$ [95\% CI] \\
\midrule
"""
    for row in panel_b:
        tex += f"${row['parameter']}$ & {row['IC']} & {row['dPL']} \\\\\n"
    tex += r"""\bottomrule
\end{tabular}

\begin{tablenotes}[flushleft]
\small
\item All quantities are dimensionless slopes per unit basin snow fraction. The full S1--S5 median-shift organization and the four recurring signatures shown in Figure 4 are read from the canonical R2 strata summary.
\end{tablenotes}
\end{threeparttable}
\end{table*}
"""

    for directory in OUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "TableS4_exact_estimates_f3_f4.md").write_text(md, encoding="utf-8")
        (directory / "TableS4_exact_estimates_f3_f4.tex").write_text(tex, encoding="utf-8")
    print("Supplementary Table S4 generated from canonical R2 outputs.")


if __name__ == "__main__":
    main()
