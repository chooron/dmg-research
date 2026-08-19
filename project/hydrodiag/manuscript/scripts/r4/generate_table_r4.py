"""Generate LaTeX and Markdown tables for Manuscript R4.

Generates:
- Table 4 (Main Manuscript): State consistency & timing diagnostics across snow regimes
- Table S4 (Supplementary): 4-phase process-conditioned breakdown & robustness regressions

Outputs:
    manuscript/tables/Table4_soil_state_consistency.tex
    manuscript/tables/Table4_soil_state_consistency.md
    manuscript/tables/TableS4_process_phase_and_robustness.tex
    manuscript/tables/TableS4_process_phase_and_robustness.md
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript.scripts.r4.common import default_results_root  # noqa: E402

TABLES_DIR = HERE.parents[1] / "tables"


def df_to_markdown_simple(df: pd.DataFrame) -> str:
    """Simple markdown table formatter without external dependencies."""
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---" if i <= 1 else ":---:" for i in range(len(headers))]) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(val) for val in row.values) + " |")
    return "\n".join(lines) + "\n"


def generate_tables(results_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    r4_dir = results_root / "r4_phase1_soil_official"

    df_quant = pd.read_csv(r4_dir / "snow_burden_quartile_summary.csv")
    df_phase = pd.read_csv(r4_dir / "robustness_process_phase_consistency.csv")
    df_perf = pd.read_csv(r4_dir / "robustness_performance_subsets.csv")

    # -----------------------------------------------------------------------
    # Table 4 (Main Manuscript): Quantile breakdown of Base vs CN consistency
    # -----------------------------------------------------------------------
    t4_rows = []
    for reg, reg_label in [("dPL_seed42", "dPL (Seed 42)"), ("dPL_seed123", "dPL (Seed 123)"), ("IC_fused", "IC Fused (5x200, Sens.)")]:
        sub_q = df_quant[df_quant["regime"] == reg].sort_values("quartile")
        for _, row in sub_q.iterrows():
            t4_rows.append({
                "Regime": reg_label,
                "Snow Regime": row["quartile"],
                "Catchments (N)": int(row["n"]),
                "Median SWE (mm)": f"{row['swe_burden_median_mm']:.1f}",
                "Base Anom Corr": f"{row['base_median_anomaly_corr']:.3f}",
                "CN Anom Corr": f"{row['cn_median_anomaly_corr']:.3f}",
                "Delta Anom Corr (CN - Base)": f"{row['delta_anomaly_corr_median']:+.3f}",
                "Base 7d Corr": f"{row['base_median_7d_corr']:.3f}",
                "CN 7d Corr": f"{row['cn_median_7d_corr']:.3f}",
                "Delta 7d Corr (CN - Base)": f"{row['delta_7d_corr_median']:+.3f}",
            })
    df_table4 = pd.DataFrame(t4_rows)
    md_t4 = (
        "# Table 4: Real-Basin Shared Soil-Water State Consistency Across Snow-Burden Quantiles\n\n"
        + df_to_markdown_simple(df_table4)
        + "\n*Note*: Values report catchment-wise medians across 531 CAMELS-US catchments evaluated against the Caravan v1.1 ERA5-Land SM100 reference (0–100 cm depth-weighted composite) over the test period (1995–2010). Model state is total tension water storage W_total = wu + wl + wd. IC Fused (5 starts x 200 generations) is reported as a sensitivity check."
    )
    (out_dir / "Table4_soil_state_consistency.md").write_text(md_t4, encoding="utf-8")
    df_table4.to_latex(out_dir / "Table4_soil_state_consistency.tex", index=False)

    # -----------------------------------------------------------------------
    # Table S4 (Supplementary): 4-Phase Process Breakdown
    # -----------------------------------------------------------------------
    s4_rows = []
    for reg, reg_label in [("dPL_seed42", "dPL (Seed 42)"), ("dPL_seed123", "dPL (Seed 123)"), ("IC_fused", "IC Fused (5x200, Sens.)")]:
        sub_ph = df_phase[df_phase["regime"] == reg]
        for p_code, p_name in [(1, "1. Snow Accumulation"), (2, "2. Active Melt / Recharge"), (3, "3. Post-Melt Transition"), (4, "4. Summer Dry-Down")]:
            sub_p = sub_ph[sub_ph["phase_code"] == p_code]
            s4_rows.append({
                "Regime": reg_label,
                "Process Phase": p_name,
                "Total Basin-Days": int(sub_p["n_days"].sum()),
                "Base Anomaly Corr": f"{sub_p['base_anomaly_corr'].median():.3f}",
                "CN Anomaly Corr": f"{sub_p['cn_anomaly_corr'].median():.3f}",
                "Delta Anomaly Corr": f"{sub_p['delta_anomaly_corr'].median():+.3f}",
                "Base Daily Corr": f"{sub_p['base_daily_corr'].median():.3f}",
                "CN Daily Corr": f"{sub_p['cn_daily_corr'].median():.3f}",
                "Delta Daily Corr": f"{sub_p['delta_daily_corr'].median():+.3f}",
            })
    df_tables4 = pd.DataFrame(s4_rows)
    md_s4 = (
        "# Table S4: Process-Phase Conditioned Soil Moisture Consistency (Snow-Active Catchments, SWE >= 20 mm)\n\n"
        + df_to_markdown_simple(df_tables4)
        + "\n*Note*: Phases are partitioned purely by external snow reference (Snow-17 / Caravan SWE): Phase 1 = Snow accumulation; Phase 2 = Active snowmelt and spring recharge; Phase 3 = Post-melt transition; Phase 4 = Summer dry-down. Catchments evaluated are all N = 352 snow-active catchments across the test period (1995–2010)."
    )
    (out_dir / "TableS4_process_phase_and_robustness.md").write_text(md_s4, encoding="utf-8")
    df_tables4.to_latex(out_dir / "TableS4_process_phase_and_robustness.tex", index=False)

    print(f"Generated Manuscript Tables:\n  {out_dir / 'Table4_soil_state_consistency.md'}\n  {out_dir / 'TableS4_process_phase_and_robustness.md'}")


if __name__ == "__main__":
    results_root = default_results_root()
    generate_tables(results_root, TABLES_DIR)
