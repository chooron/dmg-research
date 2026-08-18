#!/usr/bin/env python3
"""
Table S5 Generator (R2 Supplement)
Generates TableS5_boundary_point_mass.md and TableS5_boundary_point_mass.tex

Compact result-only table documenting the boundary / point-mass behaviour of the
parameters highlighted in Figure 4 (um, ki, ci) by snow regime and
parameter-constraint regime (IC / dPL), so that the F4 ridgeline features
(exact-zero point mass, exact and near boundary touching) are traceable.

Columns (all read verbatim from the frozen F4 data-audit files in
results/R2/r2_figure4_data_audit_*.csv; no analysis is recomputed):
  * frac Delta z = 0      : fraction of basins with an exact paired shift of zero
                            (pct_zero / 100 from the raw-distribution audit)
  * frac |Delta z| = 1    : fraction of basins with an exact paired shift of +1 or -1
                            (n_exact_plus1 + n_exact_minus1 over n; occurs for IC
                            only, because dPL reconstructed values are strictly
                            interior, |Delta z| < 1)
  * frac |Delta z| >= 0.95: fraction of basins within 0.05 of the +/-1 boundary
                            (|Delta z| >= 0.95 fraction from the KDE-robustness audit)

One near-boundary threshold (|Delta z| >= 0.95) is used, matching the F4 audit's
primary near-boundary criterion; no alternative thresholds are stacked.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MANUSCRIPT = HERE.parent
R2 = MANUSCRIPT / "results" / "R2"
OUT_DIRS = (MANUSCRIPT / "stats" / "tables", MANUSCRIPT / "tables")

PARAMS = ["xaj_um", "xaj_ki", "xaj_ci"]
DISPLAY = {"xaj_um": "$u_m$", "xaj_ki": "$k_i$", "xaj_ci": "$c_i$"}
REGIMES = ["S1", "S2", "S3", "S4", "S5"]
REGIME_LABELS = {
    "S1": "S1 (0\u20130.05)",
    "S2": "S2 (0.05\u20130.15)",
    "S3": "S3 (0.15\u20130.30)",
    "S4": "S4 (0.30\u20130.50)",
    "S5": "S5 (0.50\u20131.00)",
}


def fmt(v, decimals=3):
    return f"{float(v):.{decimals}f}"


def load():
    raw = pd.read_csv(R2 / "r2_figure4_data_audit_raw_distributions.csv")
    bm = pd.read_csv(R2 / "r2_figure4_data_audit_boundary_mass.csv")
    bm = bm[bm["threshold"] == 0.01]  # exact +/-1 counts are threshold-independent
    kde = pd.read_csv(R2 / "r2_figure4_data_audit_kde_robustness.csv")
    return raw, bm, kde


def rows(raw, bm, kde):
    md_rows, tex_rows = [], []
    for p in PARAMS:
        for paradigm in ["IC", "dPL"]:
            for regime in REGIMES:
                r = raw[
                    (raw["parameter"] == p)
                    & (raw["paradigm"] == paradigm)
                    & (raw["regime"] == regime)
                ]
                b = bm[
                    (bm["parameter"] == p)
                    & (bm["paradigm"] == paradigm)
                    & (bm["regime"] == regime)
                ]
                k = kde[
                    (kde["parameter"] == p)
                    & (kde["paradigm"] == paradigm)
                    & (kde["regime"] == regime)
                ]
                assert len(r) == len(b) == len(k) == 1, (
                    f"missing {p} {paradigm} {regime}"
                )
                r, b, k = r.iloc[0], b.iloc[0], k.iloc[0]
                n = int(b["n"])
                frac_zero = float(r["pct_zero"]) / 100.0
                frac_exact1 = (
                    float(b["n_exact_plus1"]) + float(b["n_exact_minus1"])
                ) / n
                frac_near = float(k["frac_absdz_ge_095"])
                md_rows.append(
                    {
                        "Parameter": DISPLAY[p],
                        "Regime": paradigm,
                        "Snow": REGIME_LABELS[regime],
                        "n": n,
                        "zero": fmt(frac_zero),
                        "exact1": fmt(frac_exact1),
                        "near": fmt(frac_near),
                    }
                )
                tex_rows.append(
                    [
                        DISPLAY[p],
                        paradigm,
                        REGIME_LABELS[regime],
                        str(n),
                        fmt(frac_zero),
                        fmt(frac_exact1),
                        fmt(frac_near),
                    ]
                )
    return md_rows, tex_rows


def main():
    raw, bm, kde = load()
    md_rows, tex_rows = rows(raw, bm, kde)

    md = """# Table S5: Boundary and Point-Mass Characteristics of Parameters Highlighted in Figure 4

| Parameter | Regime | Snow regime | n | $\\Delta z = 0$ | $|\\Delta z| = 1$ | $|\\Delta z| \\ge 0.95$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for r in md_rows:
        md += (
            f"| {r['Parameter']} | {r['Regime']} | {r['Snow']} | {r['n']} | "
            f"{r['zero']} | {r['exact1']} | {r['near']} |\n"
        )

    md += """
*Note*: The paired shift is $\\Delta z = z_{\\mathrm{Base}} - z_{\\mathrm{CN}}$
(normalized parameters, $z \\in [0,1]$). Reported fractions are basin counts over
the regime sample size $n$. Snow regimes S1\u2013S5 are the fixed strata by basin snow fraction: S1 $[0, 0.05)$ ($n=165$), S2 $[0.05, 0.15)$ ($n=156$), S3 $[0.15, 0.30)$ ($n=121$), S4 $[0.30, 0.50)$ ($n=34$), S5 $[0.50, 1.00]$ ($n=55$). Exact $\\Delta z = 0$ marks basins where the two
structures co-locate at identical normalized values (under IC, these are
predominantly cases where both structures sit at a shared parameter bound);
exact $|\\Delta z| = 1$ marks basins where one structure sits at one bound and the
other at the opposite bound (IC only; dPL reconstructed values are strictly
interior). $|\\Delta z| \\ge 0.95$ is the near-boundary fraction at the audit's
primary threshold.
"""

    body = "\n".join(" & ".join(r) + " \\\\" for r in tex_rows)

    tex = (
        r"""\begin{table}[t]
\centering
\caption{Boundary and point-mass characteristics of the parameters highlighted in Figure 4 ($u_m$, $k_i$, $c_i$) by snow regime and parameter-constraint regime, based on the canonical paired shift $\Delta z = z_{\mathrm{Base}} - z_{\mathrm{CN}}$. Snow regimes S1--S5 are the fixed strata by basin snow fraction: S1 $[0, 0.05)$ ($n=165$), S2 $[0.05, 0.15)$ ($n=156$), S3 $[0.15, 0.30)$ ($n=121$), S4 $[0.30, 0.50)$ ($n=34$), S5 $[0.50, 1.00]$ ($n=55$). Fractions are basin counts over the regime sample size $n$: exact $\Delta z = 0$ (point mass), exact $|\Delta z| = 1$ (opposite-bound co-location; IC only, because dPL reconstructed values are strictly interior), and near-boundary $|\Delta z| \ge 0.95$ at the audit's primary threshold.}
\label{tab:tables5_boundary_point_mass}
\begin{threeparttable}
\begin{tabular}{lllcccc}
\toprule
Parameter & Regime & Snow regime & $n$ & $\Delta z = 0$ & $|\Delta z| = 1$ & $|\Delta z| \ge 0.95$ \\
\midrule
"""
        + body
        + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]
\small
\item \textit{Note}: The paired shift is $\Delta z = z_{\mathrm{Base}} - z_{\mathrm{CN}}$ (normalized parameters, $z \in [0,1]$). Exact $\Delta z = 0$ marks basins where the two structures co-locate at identical normalized values (under IC, predominantly both structures at a shared parameter bound); exact $|\Delta z| = 1$ marks basins where one structure sits at one bound and the other at the opposite bound (IC only; dPL values are strictly interior). $|\Delta z| \ge 0.95$ is the near-boundary fraction at the audit's primary threshold.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    )

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        (d / "TableS5_boundary_point_mass.md").write_text(md)
        (d / "TableS5_boundary_point_mass.tex").write_text(tex)
    print("Supplementary Table S5 generated successfully.")


if __name__ == "__main__":
    main()
