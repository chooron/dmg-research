#!/usr/bin/env python3
"""Generate Main Text Table 1: Structural configurations and diagnostic roles.

Table 1 is a Methods/experimental design descriptive table that clarifies the
structural identities, parameter counts, physical mechanisms, and diagnostic
roles of Base, TGD, and CN across the manuscript.
"""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATS_TABLE_DIR = PROJECT_ROOT / "manuscript" / "stats" / "tables"
TABLES_DIR = PROJECT_ROOT / "manuscript" / "tables"


def build_table1_dataframe() -> pd.DataFrame:
    data = [
        {
            "Property / Characteristic": "Shared host parameters",
            "Base": "15",
            "TGD": "15",
            "CN": "15",
        },
        {
            "Property / Characteristic": "Calibrated parameters (shared + added)",
            "Base": "15 + 0",
            "TGD": "15 + 2",
            "CN": "15 + 2",
        },
        {
            "Property / Characteristic": "Temperature used by added structural component",
            "Base": "No",
            "TGD": "Yes",
            "CN": "Yes",
        },
        {
            "Property / Characteristic": "Additional generic temperature-conditioned storage / memory",
            "Base": "No",
            "TGD": "Yes",
            "CN": "No",
        },
        {
            "Property / Characteristic": "Explicit snow accumulation–melt sequence",
            "Base": "No",
            "TGD": "No",
            "CN": "Yes",
        },
        {
            "Property / Characteristic": "Primary diagnostic role",
            "Base": "Snow-process omission configuration",
            "TGD": "Temperature-conditioned generic control",
            "CN": "Explicit snow accumulation–melt representation",
        },
    ]
    return pd.DataFrame(data)


def generate_markdown(df: pd.DataFrame) -> str:
    md = [
        "# Table 1: Structural Configurations and Diagnostic Roles",
        "",
        "| Property / Characteristic | Base | TGD | CN |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for _, row in df.iterrows():
        md.append(f"| {row['Property / Characteristic']} | {row['Base']} | {row['TGD']} | {row['CN']} |")

    md.extend([
        "",
        "*Note*: Base represents the snow-process omission configuration lacking explicit snow dynamics (15 core host parameters). "
        "TGD is the parameter-count-matched, temperature-conditioned generic storage control that provides non-specific thermal retention/memory without snow physics (15 host + 2 temperature-smoothing/storage parameters: $\\tau_{\\mathrm{warm}}, \\Delta\\tau_{\\mathrm{cold}}$). "
        "CN incorporates an explicit degree-day snow accumulation–melt sequence comprising precipitation phase partitioning, persistent snowpack storage, and temperature-threshold melt release (15 host + 2 snow parameters: snowpack thermal inertia coefficient $C_{\\mathrm{TG}}$ and degree-day melt factor $K_f$). "
        "Matching the number of added parameters does not imply that TGD and CN encode the same structural information. "
        "TGD serves as a generic control rather than an intermediate step in a physical decomposition or an additive performance ladder. "
        "CN is the generating structure in the controlled synthetic experiment only; it is not treated as hydrological truth in real catchments. "
        "Independent calibration (IC-CMA-ES; catchment-wise independent optimization) and differentiable parameter learning (dPL-MLP; shared cross-catchment parameter mapping) represent contrasting parameter-estimation constraints evaluated in parallel, not competitive ranking benchmarks."
    ])
    return "\n".join(md) + "\n"


def generate_latex(df: pd.DataFrame) -> str:
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{threeparttable}",
        r"\caption{Structural configurations, mechanism representation, and diagnostic roles of the three investigated model structures.}",
        r"\label{tab:structural_configurations}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Property / Characteristic & Base & TGD & CN \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        prop = row["Property / Characteristic"]
        base = row["Base"]
        tgd = row["TGD"]
        cn = row["CN"]
        tex.append(f"{prop} & {base} & {tgd} & {cn} \\\\")

    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item \textit{Note}: Base represents the snow-process omission configuration lacking explicit snow dynamics (15 core host parameters). "
        r"TGD is the parameter-count-matched, temperature-conditioned generic storage control that provides non-specific thermal retention/memory without snow physics (15 host + 2 storage/smoothing parameters: $\tau_{\mathrm{warm}}, \Delta\tau_{\mathrm{cold}}$). "
        r"CN incorporates an explicit degree-day snow accumulation--melt sequence comprising precipitation phase partitioning, persistent snowpack storage, and temperature-threshold melt release (15 host + 2 snow parameters: snowpack thermal inertia coefficient $C_{\mathrm{TG}}$ and degree-day melt factor $K_f$). "
        r"Matching the number of added parameters does not imply that TGD and CN encode the same structural information. "
        r"TGD serves as a generic control rather than an intermediate step in a physical decomposition or an additive performance ladder. "
        r"CN is the generating structure in the controlled synthetic experiment only; it is not treated as hydrological truth in real catchments. "
        r"Independent calibration (IC-CMA-ES; catchment-wise optimization) and differentiable parameter learning (dPL-MLP; shared cross-catchment parameter mapping) represent contrasting parameter-estimation constraints evaluated in parallel, not competitive ranking benchmarks.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ])
    return "\n".join(tex) + "\n"


def main():
    STATS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = build_table1_dataframe()

    # Write CSV
    csv_path = STATS_TABLE_DIR / "Table1_structural_configurations.csv"
    df.to_csv(csv_path, index=False)

    # Write Markdown
    md_content = generate_markdown(df)
    (STATS_TABLE_DIR / "Table1_structural_configurations.md").write_text(md_content, encoding="utf-8")
    (TABLES_DIR / "Table1_structural_configurations.md").write_text(md_content, encoding="utf-8")

    # Write LaTeX
    tex_content = generate_latex(df)
    (STATS_TABLE_DIR / "Table1_structural_configurations.tex").write_text(tex_content, encoding="utf-8")
    (TABLES_DIR / "Table1_structural_configurations.tex").write_text(tex_content, encoding="utf-8")

    print(f"Table 1 generated successfully:\n  {csv_path}\n  {STATS_TABLE_DIR / 'Table1_structural_configurations.md'}\n  {STATS_TABLE_DIR / 'Table1_structural_configurations.tex'}")


if __name__ == "__main__":
    main()
