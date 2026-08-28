#!/usr/bin/env python3
"""Generate Supplementary Table S1: Model parameter definitions and optimization bounds.

Table S1 is the reproducibility parameter reference table for the Methods/SI.
It defines the 15 shared XAJ host model parameters alongside the 2 additional
parameters for the generic temperature control (TGD) and the 2 additional
parameters for the explicit degree-day snow module (CN/CemaNeige), totaling 19 parameters.
"""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATS_TABLE_DIR = PROJECT_ROOT / "manuscript" / "stats" / "tables"
SUPP_TABLE_DIR = PROJECT_ROOT / "manuscript" / "supplement" / "tables"


def build_parameters_dataframe() -> pd.DataFrame:
    rows = [
        # --- 15 Shared XAJ Core Host Parameters ---
        {
            "Parameter": "$k$",
            "Identifier": "xaj_k",
            "Hydrological role": "Ratio of potential ET to reference crop evaporation",
            "Unit": "-",
            "Lower bound": 0.5,
            "Upper bound": 2.0,
            "Applies to": "Shared host",
        },
        {
            "Parameter": "$b$",
            "Identifier": "xaj_b",
            "Hydrological role": "Exponent of tension water storage capacity distribution curve",
            "Unit": "-",
            "Lower bound": 0.1,
            "Upper bound": 2.0,
            "Applies to": "Shared host",
        },
        {
            "Parameter": "$i_m$",
            "Identifier": "xaj_im",
            "Hydrological role": "Fraction of impervious and saturated direct runoff area",
            "Unit": "-",
            "Lower bound": 0.0,
            "Upper bound": 0.3,
            "Applies to": "Shared host",
        },
        {
            "Parameter": "$u_m$",
            "Identifier": "xaj_um",
            "Hydrological role": "Upper-layer soil tension water capacity",
            "Unit": "mm",
            "Lower bound": 5.0,
            "Upper bound": 50.0,
            "Applies to": "Shared host",
        },
        {
            "Parameter": "$l_m$",
            "Identifier": "xaj_lm",
            "Hydrological role": "Lower-layer soil tension water capacity",
            "Unit": "mm",
            "Lower bound": 20.0,
            "Upper bound": 200.0,
            "Applies to": "Shared host",
        },
        {
            "Parameter": "$d_m$",
            "Identifier": "xaj_dm",
            "Hydrological role": "Deep-layer soil tension water capacity",
            "Unit": "mm",
            "Lower bound": 20.0,
            "Upper bound": 200.0,
            "Applies to": "Shared host",
        },
        {
            "Parameter": "$c$",
            "Identifier": "xaj_c",
            "Hydrological role": "Deep-layer evapotranspiration coefficient",
            "Unit": "-",
            "Lower bound": 0.05,
            "Upper bound": 0.3,
            "Applies to": "Shared host",
        },
        {
            "Parameter": "$s_m$",
            "Identifier": "xaj_sm",
            "Hydrological role": "Areal mean free water capacity of surface/shallow layer",
            "Unit": "mm",
            "Lower bound": 5.0,
            "Upper bound": 100.0,
            "Applies to": "Shared host",
        },
        {
            "Parameter": "$ex$",
            "Identifier": "xaj_ex",
            "Hydrological role": "Exponent of free water capacity distribution curve",
            "Unit": "-",
            "Lower bound": 0.1,
            "Upper bound": 2.0,
            "Applies to": "Shared host",
        },
        {
            "Parameter": "$k_i$",
            "Identifier": "xaj_ki",
            "Hydrological role": "Outflow coefficient from free water storage to interflow",
            "Unit": "d⁻¹",
            "Lower bound": 0.0,
            "Upper bound": 0.7,
            "Applies to": "Shared host",
        },
        {
            "Parameter": "$k_g$",
            "Identifier": "xaj_kg",
            "Hydrological role": "Outflow coefficient from free water storage to groundwater",
            "Unit": "d⁻¹",
            "Lower bound": 0.0,
            "Upper bound": 0.7,
            "Applies to": "Shared host",
        },
        {
            "Parameter": "$c_i$",
            "Identifier": "xaj_ci",
            "Hydrological role": "Recession constant of the linear interflow reservoir",
            "Unit": "-",
            "Lower bound": 0.1,
            "Upper bound": 1.0,
            "Applies to": "Shared host",
        },
        {
            "Parameter": "$c_g$",
            "Identifier": "xaj_cg",
            "Hydrological role": "Recession constant of the linear groundwater reservoir",
            "Unit": "-",
            "Lower bound": 0.9,
            "Upper bound": 1.0,
            "Applies to": "Shared host",
        },
        {
            "Parameter": "$a$",
            "Identifier": "xaj_a",
            "Hydrological role": "Shape parameter of the Gamma unit hydrograph (Gamma-UH)",
            "Unit": "-",
            "Lower bound": 0.0,
            "Upper bound": 2.9,
            "Applies to": "Shared host",
        },
        {
            "Parameter": "$\\theta$",
            "Identifier": "xaj_theta",
            "Hydrological role": "Scale parameter of the Gamma unit hydrograph (Gamma-UH)",
            "Unit": "d",
            "Lower bound": 0.0,
            "Upper bound": 6.5,
            "Applies to": "Shared host",
        },
        # --- 2 Additional Parameters for Generic Temperature Control (TGD) ---
        {
            "Parameter": "$\\tau_{\\mathrm{warm}}$",
            "Identifier": "tgd_tau_warm",
            "Hydrological role": "Warm-condition linear reservoir residence time / baseline smoothing",
            "Unit": "d",
            "Lower bound": 0.0001,
            "Upper bound": 3.0,
            "Applies to": "TGD only",
        },
        {
            "Parameter": "$\\Delta\\tau_{\\mathrm{cold}}$",
            "Identifier": "tgd_delta_tau_cold",
            "Hydrological role": "Additional cold-condition linear reservoir residence time increment",
            "Unit": "d",
            "Lower bound": 0.1,
            "Upper bound": 180.0,
            "Applies to": "TGD only",
        },
        # --- 2 Additional Parameters for Explicit Snow Module (CN / CemaNeige) ---
        {
            "Parameter": "$C_{\\mathrm{TG}}$",
            "Identifier": "cn_ctg",
            "Hydrological role": "Snowpack thermal inertia and temperature weighting coefficient",
            "Unit": "-",
            "Lower bound": 0.0,
            "Upper bound": 1.0,
            "Applies to": "CN only",
        },
        {
            "Parameter": "$K_f$",
            "Identifier": "cn_kf",
            "Hydrological role": "Degree-day snowmelt factor ($D_f$)",
            "Unit": "mm °C⁻¹ d⁻¹",
            "Lower bound": 0.0,
            "Upper bound": 10.0,
            "Applies to": "CN only",
        },
    ]
    return pd.DataFrame(rows)


def clean_markdown_symbol(text: str) -> str:
    s = text.replace("$", "")
    s = s.replace(r"\mathrm{warm}", "warm")
    s = s.replace(r"\mathrm{cold}", "cold")
    s = s.replace(r"\mathrm{TG}", "TG")
    s = s.replace(r"\mathrm{", "").replace("}", "")
    s = s.replace(r"\theta", "θ")
    s = s.replace(r"\tau", "τ")
    s = s.replace(r"\Delta", "Δ")
    s = s.replace("{", "").replace("}", "")
    return s


def generate_markdown(df: pd.DataFrame) -> str:
    md = [
        "# Table S1: Model Parameter Definitions and Optimization Bounds",
        "",
        "| Parameter | Hydrological role | Unit | Lower bound | Upper bound | Applies to |",
        "| :---: | :--- | :---: | :---: | :---: | :--- |",
    ]
    for _, r in df.iterrows():
        sym = clean_markdown_symbol(r["Parameter"])
        md.append(f"| {sym} | {r['Hydrological role']} | {r['Unit']} | {r['Lower bound']} | {r['Upper bound']} | {r['Applies to']} |")

    md.extend([
        "",
        "*Note*: Parameters 1–15 constitute the 15-dimensional core parameter space shared identically across Base, TGD, and CN configurations within the Xinanjiang (XAJ) host framework. "
        "TGD augments the host model with two generic temperature-dependent delay parameters ($\\tau_{\\mathrm{warm}}, \\Delta\\tau_{\\mathrm{cold}}$). "
        "CN augments the host model with two degree-day snowpack accumulation and melt parameters ($C_{\\mathrm{TG}}, K_f$). "
        "Boundaries define the feasible physical search space for both independent calibration (IC-CMA-ES) and differentiable parameter learning (dPL-MLP)."
    ])
    return "\n".join(md) + "\n"


def generate_latex(df: pd.DataFrame) -> str:
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{threeparttable}",
        r"\caption{Model parameter definitions, hydrological roles, units, and optimization bounds for the shared XAJ host model and structural variants.}",
        r"\label{tab:parameter_bounds}",
        r"\begin{tabular}{cllccc}",
        r"\toprule",
        r"Parameter & Identifier & Hydrological role & Unit & Lower bound & Upper bound & Applies to \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\textbf{A. Shared XAJ Host Model Parameters (15 Core Parameters)}} \\",
    ]
    for i, r in df.iterrows():
        if i == 15:
            tex.extend([
                r"\midrule",
                r"\multicolumn{7}{l}{\textbf{B. Generic Temperature Control Additional Parameters (TGD, 2 Parameters)}} \\",
            ])
        elif i == 17:
            tex.extend([
                r"\midrule",
                r"\multicolumn{7}{l}{\textbf{C. Explicit Snow Accumulation--Melt Additional Parameters (CN / CemaNeige, 2 Parameters)}} \\",
            ])
        sym = r["Parameter"]
        code = r["Identifier"].replace("_", r"\_")
        meaning = r["Hydrological role"]
        unit = r["Unit"].replace("d⁻¹", r"\text{d}^{-1}").replace("°C⁻¹", r"^{\circ}\text{C}^{-1}").replace(" ", r"\ ")
        low = r["Lower bound"]
        up = r["Upper bound"]
        applies = r["Applies to"]
        tex.append(f"{sym} & \\texttt{{{code}}} & {meaning} & ${unit}$ & {low} & {up} & {applies} \\\\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item \textit{Note}: Parameters 1--15 define the shared host parameter space across Base, TGD, and CN within the Xinanjiang (XAJ) structure. "
        r"TGD incorporates two generic temperature-dependent delay parameters ($\tau_{\mathrm{warm}}, \Delta\tau_{\mathrm{cold}}$). "
        r"CN incorporates two degree-day snowpack accumulation and melt parameters ($C_{\mathrm{TG}}, K_f$). "
        r"Bounds define the physical search space across both IC and dPL estimation regimes.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table*}",
    ])
    return "\n".join(tex) + "\n"


def main():
    STATS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SUPP_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = build_parameters_dataframe()

    # Write CSV
    csv_path_stats = STATS_TABLE_DIR / "TableS1_parameter_bounds.csv"
    csv_path_supp = SUPP_TABLE_DIR / "TableS1_parameter_bounds.csv"
    df.to_csv(csv_path_stats, index=False)
    df.to_csv(csv_path_supp, index=False)

    # Write Markdown
    md_content = generate_markdown(df)
    (STATS_TABLE_DIR / "TableS1_parameter_bounds.md").write_text(md_content, encoding="utf-8")
    (SUPP_TABLE_DIR / "TableS1_parameter_bounds.md").write_text(md_content, encoding="utf-8")

    # Write LaTeX
    tex_content = generate_latex(df)
    (STATS_TABLE_DIR / "TableS1_parameter_bounds.tex").write_text(tex_content, encoding="utf-8")
    (SUPP_TABLE_DIR / "TableS1_parameter_bounds.tex").write_text(tex_content, encoding="utf-8")

    print(f"Table S1 generated successfully:\n  {csv_path_stats}\n  {STATS_TABLE_DIR / 'TableS1_parameter_bounds.md'}\n  {STATS_TABLE_DIR / 'TableS1_parameter_bounds.tex'}")


if __name__ == "__main__":
    main()
