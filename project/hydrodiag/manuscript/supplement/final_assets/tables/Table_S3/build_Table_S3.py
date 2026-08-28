#!/usr/bin/env python3
"""Build final Table S3 from frozen reviewer-2 summaries."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
TAIL = PROJECT / "results" / "reviewer2_robustness" / "p0_reporting" / "recovery_denominator_tail_audit.csv"
BREAKDOWN = PROJECT / "results" / "reviewer2_robustness" / "p0_reporting" / "invalid_denominator_strata_breakdown.csv"
ALT_SUMMARY = PROJECT / "results" / "reviewer2_robustness" / "alt_generating_field" / "alt_generating_field_summary.json"
REGISTRY = PROJECT / "results" / "reviewer2_robustness" / "summaries" / "canonical_registry.csv"


def panel_a() -> pd.DataFrame:
    tail = pd.read_csv(TAIL)
    tail = tail.loc[tail["period"].eq("test")].copy()
    breakdown = pd.read_csv(BREAKDOWN)
    breakdown = breakdown.loc[breakdown["period"].eq("test")].copy()
    rows: list[dict] = []
    common = {
        "section": "overall",
        "period": "test",
        "snow_stratum": "All",
        "N_total": None,
        "N_invalid": None,
        "N_valid": None,
        "median": None,
        "Q25": None,
        "Q75": None,
        "P05": None,
        "P95": None,
        "P_below_0": None,
        "P_above_1": None,
        "P_delta_F_gt_0": None,
        "P_F_TGD_lt_F_close": None,
        "invalid_rate_within_stratum_pct": None,
        "share_of_all_invalid_basins_pct": None,
    }
    for _, r in tail.iterrows():
        base = {**common, "paradigm": r["paradigm"], "N_total": int(r["N_total"]), "N_invalid": int(r["N_total"] - r["N_valid_gt_1e6"]), "N_valid": int(r["N_valid_gt_1e6"])}
        rows.extend([
            {**base, "metric": "D", "median": r["D_median"], "Q25": r["D_IQR_q25"], "Q75": r["D_IQR_q75"], "P05": r["D_P05"], "P95": r["D_P95"], "source": "recovery_denominator_tail_audit.csv"},
            {**base, "metric": "F_close", "median": r["F_close_median"], "Q25": r["F_close_q25"], "Q75": r["F_close_q75"], "P05": r["F_close_P05"], "P95": r["F_close_P95"], "P_below_0": r["F_close_P_lt_0"], "P_above_1": r["F_close_P_gt_1"], "source": "recovery_denominator_tail_audit.csv"},
            {**base, "metric": "F_TGD", "median": r["F_TGD_median"], "Q25": r["F_TGD_q25"], "Q75": r["F_TGD_q75"], "P05": r["F_TGD_P05"], "P95": r["F_TGD_P95"], "P_below_0": r["F_TGD_P_lt_0"], "P_above_1": r["F_TGD_P_gt_1"], "source": "recovery_denominator_tail_audit.csv"},
            {**base, "metric": "Delta F", "median": r["Delta_F_median"], "P_delta_F_gt_0": r["Delta_F_P_gt_0"], "P_F_TGD_lt_F_close": r["Delta_F_P_lt_0"], "source": "recovery_denominator_tail_audit.csv"},
        ])
    for _, r in breakdown.iterrows():
        rows.append({
            **common,
            "section": "stratum",
            "paradigm": r["paradigm"],
            "period": r["period"],
            "snow_stratum": r["snow_stratum"],
            "metric": "D validity",
            "N_total": int(r["N_stratum_total"]),
            "N_invalid": int(r["N_invalid_le_1e6"]),
            "N_valid": int(r["N_valid_gt_1e6"]),
            "invalid_rate_within_stratum_pct": r["invalid_rate_within_stratum_pct"],
            "share_of_all_invalid_basins_pct": r["share_of_all_invalid_basins_pct"],
            "source": "invalid_denominator_strata_breakdown.csv",
        })
    columns = ["section", "paradigm", "period", "snow_stratum", "metric", "N_total", "N_invalid", "N_valid", "median", "Q25", "Q75", "P05", "P95", "P_below_0", "P_above_1", "P_delta_F_gt_0", "P_F_TGD_lt_F_close", "invalid_rate_within_stratum_pct", "share_of_all_invalid_basins_pct", "source"]
    return pd.DataFrame(rows)[columns]


def registry_value(registry: pd.DataFrame, quantity: str) -> float:
    row = registry.loc[registry["quantity"].eq(quantity)]
    if len(row) != 1:
        raise ValueError(f"Expected one canonical registry row for {quantity}, got {len(row)}")
    return float(row.iloc[0]["value"])


def panel_b() -> pd.DataFrame:
    registry = pd.read_csv(REGISTRY)
    alt = json.loads(ALT_SUMMARY.read_text())
    rows = []
    for paradigm, key in (("IC", "ic"), ("dPL", "dpl")):
        rows.append({
            "generating_field": "canonical PCA/SVD-ridge field",
            "paradigm": paradigm,
            "period": "test",
            "G_Base median": registry_value(registry, f"G_Base_{key}_test"),
            "G_TGD median": registry_value(registry, f"G_TGD_{key}_test"),
            "F_close median": registry_value(registry, f"F_close_{key}_test"),
            "F_TGD* median": registry_value(registry, f"F_TGD_star_{key}_test"),
            "Delta F median": registry_value(registry, f"delta_F_{key}_test"),
            "N_total": 531,
            "N_valid": int(registry_value(registry, "valid_N_ic_test" if paradigm == "IC" else "valid_N_dpl_test_seedmedian")),
            "P(Delta F > 0)": registry_value(registry, f"P_delta_F_gt0_{key}_test"),
            "source": "reviewer2_robustness/summaries/canonical_registry.csv",
        })
        a = alt[f"{paradigm}_test"]
        rows.append({
            "generating_field": "direct basin-wise calibrated CN-IC parameter field",
            "paradigm": paradigm,
            "period": "test",
            "G_Base median": a["G_Base_median"],
            "G_TGD median": a["G_TGD_median"],
            "F_close median": a["F_close_median"],
            "F_TGD* median": a["F_TGD_median"],
            "Delta F median": a["Delta_F_median"],
            "N_total": int(a["n_total"]),
            "N_valid": int(a["n_valid"]),
            "P(Delta F > 0)": a["P_Delta_F_gt_0"],
            "source": "reviewer2_robustness/alt_generating_field/alt_generating_field_summary.json",
        })
    return pd.DataFrame(rows)


def simple_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    rows.extend("| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False, name=None))
    return "\n".join(rows)


def build_markdown(a: pd.DataFrame, b: pd.DataFrame) -> str:
    return "\n".join([
        "# Table S3 — Controlled-recovery distributions and generating-field robustness",
        "",
        "## Panel A — Denominator, recovery-fraction, and paired-distribution properties",
        "",
        "Overall rows report the test-period distribution across 531 basins (or the denominator-valid subset where applicable). Stratum rows report denominator validity for S1–S5. `share_of_all_invalid_basins_pct` has all invalid basins as its denominator; `invalid_rate_within_stratum_pct` has the stratum total as its denominator. Recovery fractions are unclipped, so values below 0 or above 1 are retained.",
        "",
        simple_table(a),
        "",
        "## Panel B — Generating-field construction sensitivity",
        "",
        "The canonical row uses the PCA/SVD-ridge field. The alternative row uses the direct basin-wise calibrated CN-IC parameter field. The latter is a generating-field construction sensitivity, not a real-catchment truth validation.",
        "",
        simple_table(b),
        "",
    ])


if __name__ == "__main__":
    a = panel_a()
    b = panel_b()
    OUT.mkdir(parents=True, exist_ok=True)
    a.to_csv(OUT / "Table_S3_panelA.csv", index=False)
    b.to_csv(OUT / "Table_S3_panelB.csv", index=False)
    (OUT / "Table_S3.md").write_text(build_markdown(a, b), encoding="utf-8")
    print(f"wrote Table S3 Panel A={len(a)} rows Panel B={len(b)} rows")
