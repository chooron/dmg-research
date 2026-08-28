#!/usr/bin/env python3
"""Build final Table S2 from frozen R1/R3 sensitivity data."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
CT_CSV = PROJECT / "manuscript" / "cache" / "r1_rebuild_audit_staged" / "r1_basin_level_ct.csv"
DENOM_CSV = PROJECT / "manuscript" / "results" / "discussion_audit" / "r3_denominator_sensitivity_audit.csv"
KGE_THRESHOLDS = (0.40, 0.50, 0.60, 0.70, 0.80)
CT_THRESHOLDS = (10.0, 15.0, 20.0)
STRUCTURES = ("Base", "TGD", "CN")


def panel_a() -> pd.DataFrame:
    ct = pd.read_csv(CT_CSV)
    ct = ct.loc[ct["period"].eq("test")].copy()
    records: list[dict] = []
    for paradigm, regime in (("IC-CMA-ES", "IC"), ("dPL-MLP", "dPL")):
        sub_p = ct.loc[ct["paradigm"].eq(paradigm)].copy()
        common = sub_p.pivot(index="basin_id", columns="structure", values=["KGE", "basin_median_Delta_CT"])
        common_mask_by_tau = {
            tau: common.loc[:, ("KGE", list(STRUCTURES))].ge(tau).all(axis=1)
            for tau in KGE_THRESHOLDS
        }
        for denominator_type in ("structure-specific", "common-pass"):
            for tau in KGE_THRESHOLDS:
                common_mask = common_mask_by_tau[tau]
                for structure in STRUCTURES:
                    if denominator_type == "structure-specific":
                        sub = sub_p.loc[sub_p["structure"].eq(structure)]
                        screened = sub.loc[sub["KGE"].ge(tau)]
                        denominator_n = len(screened)
                        denominator_description = "configuration-specific KGE pass"
                        delta_values = screened["basin_median_Delta_CT"]
                    else:
                        denominator_n = int(common_mask.sum())
                        denominator_description = "all Base/TGD/CN KGE pass"
                        delta_values = common.loc[common_mask, ("basin_median_Delta_CT", structure)]
                    for ct_tau in CT_THRESHOLDS:
                        n_large = int(delta_values.abs().ge(ct_tau).sum())
                        records.append({
                            "Regime": regime,
                            "Denominator type": denominator_type,
                            "Denominator definition": denominator_description,
                            "KGE threshold": tau,
                            "Configuration": structure,
                            "CT threshold (days)": ct_tau,
                            "Denominator N": denominator_n,
                            "Large |Delta CT| N": n_large,
                            "Large |Delta CT| fraction": (n_large / denominator_n) if denominator_n else None,
                        })
    return pd.DataFrame(records)


def panel_b() -> pd.DataFrame:
    denom = pd.read_csv(DENOM_CSV)
    denom = denom.loc[denom["period"].eq("test")].copy()
    return denom.rename(columns={
        "threshold": "D threshold",
        "n_valid": "N_valid",
        "valid_rate": "valid_rate_fraction",
        "F_close_median": "F_close median",
        "F_close_iqr": "F_close IQR",
        "F_close_p5": "F_close P05",
        "F_close_p95": "F_close P95",
        "F_TGD_median": "F_TGD median",
        "F_TGD_iqr": "F_TGD IQR",
        "F_TGD_p5": "F_TGD P05",
        "F_TGD_p95": "F_TGD P95",
        "delta_F_median": "Delta F median",
        "delta_F_gt0_prop": "P(Delta F > 0)",
    })[
        ["paradigm", "D threshold", "N_valid", "valid_rate_fraction",
         "F_close median", "F_TGD median", "Delta F median", "P(Delta F > 0)",
         "F_close IQR", "F_close P05", "F_close P95", "F_TGD IQR", "F_TGD P05", "F_TGD P95"]
    ].sort_values(["paradigm", "D threshold"])


def simple_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    rows.extend("| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False, name=None))
    return "\n".join(rows)


def build_markdown(a: pd.DataFrame, b: pd.DataFrame) -> str:
    return "\n".join([
        "# Table S2 — Threshold sensitivity audits",
        "",
        "## Panel A — R1 KGE × CT threshold sensitivity",
        "",
        "The `structure-specific` denominator is the catchment set passing the KGE threshold for the named configuration. The `common-pass` denominator is the catchment set on which Base, TGD, and CN all pass the same KGE threshold. Counts use the signed basin-median `Delta CT` and report the number and fraction with `|Delta CT|` at least the listed threshold; no runoff is recomputed.",
        "",
        simple_table(a),
        "",
        "## Panel B — R3 denominator threshold sensitivity",
        "",
        "`D` is the reference-outlet gap denominator. Values are the existing unclipped catchment-wise recovery summaries for the test period; the threshold grid is read from the source CSV rather than reconstructed from memory.",
        "",
        simple_table(b),
        "",
    ])


if __name__ == "__main__":
    a = panel_a()
    b = panel_b()
    OUT.mkdir(parents=True, exist_ok=True)
    a.to_csv(OUT / "Table_S2_panelA.csv", index=False)
    b.to_csv(OUT / "Table_S2_panelB.csv", index=False)
    (OUT / "Table_S2.md").write_text(build_markdown(a, b), encoding="utf-8")
    print(f"wrote Table S2 Panel A={len(a)} rows Panel B={len(b)} rows")
