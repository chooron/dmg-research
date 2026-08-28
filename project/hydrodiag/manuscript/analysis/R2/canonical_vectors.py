"""Stage 3: Canonical basin-level parameter vector reduction rules and export.

Reduction rules:
  - IC-CMA-ES: Complete restart with maximum stored train-period KGE (lowest start index tie-breaker).
  - dPL-MLP: Within-basin coordinate-wise median across the 3 seeds (42, 123, 2026).

Produces:
  - canonical_parameter_values.csv (3,186 rows: 531 basins x 3 structures x 2 paradigms)
  - canonical_vector_audit.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from r2_config import (
    DPL_SEEDS,
    IC_STARTS,
    PARADIGMS,
    RESULTS_DIR,
    STRUCTURES,
    TOTAL_BASINS,
)
from parameter_ledger import build_raw_parameter_ledger, load_canonical_snow_metadata
from shared_parameter_specs import (
    PARAMETER_METADATA,
    SHARED_15_PARAMETERS,
    normalize_parameters,
)


def build_canonical_parameter_vectors(
    ledger_rows: List[Dict[str, Any]] | None = None,
    output_dir: Path | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build canonical 15-parameter vectors per basin x structure x paradigm."""
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if ledger_rows is None:
        ledger_rows, _ = build_raw_parameter_ledger(output_dir=out_dir)

    snow_meta = load_canonical_snow_metadata()
    basins = sorted(snow_meta.keys())

    # Index ledger rows by (paradigm, structure, basin_id, member_id, parameter)
    ledger_df = pd.DataFrame(ledger_rows)

    canonical_rows: List[Dict[str, Any]] = []

    # 1. IC Canonical Vector: best train-period KGE
    ic_df = ledger_df[ledger_df["paradigm"] == "IC"]
    ic_selected_starts: Dict[Tuple[str, str], int] = {}

    for struct in STRUCTURES:
        s_df = ic_df[ic_df["structure"] == struct]
        for b_id in basins:
            b_sub = s_df[s_df["basin_id"] == b_id]
            # Group by member_id (start) to find start with max train_kge
            start_kges = b_sub.groupby("start_or_seed")["train_kge"].first().to_dict()
            best_start = sorted(start_kges.keys(), key=lambda s: (-start_kges[s], s))[0]
            ic_selected_starts[(struct, b_id)] = best_start

            best_rows = b_sub[b_sub["start_or_seed"] == best_start].set_index("parameter")
            frac_snow, stratum = snow_meta[b_id]

            rec = {
                "basin_id": b_id,
                "paradigm": "IC",
                "regime": "IC",
                "structure": struct,
                "reduction_rule": "best_train_kge_restart",
                "selected_member": f"start_{best_start:02d}",
                "train_kge": float(best_rows["train_kge"].iloc[0]),
                "test_kge": float(best_rows["test_kge"].iloc[0]),
                "frac_snow": frac_snow,
                "snow_stratum": stratum,
            }
            for p_name in SHARED_15_PARAMETERS:
                phys_val = float(best_rows.loc[p_name, "physical_value"])
                norm_val = float(best_rows.loc[p_name, "normalized_value"])
                rec[f"phys_{p_name}"] = phys_val
                rec[f"z_{p_name}"] = norm_val
                rec[p_name] = norm_val  # alias for normalized value

            canonical_rows.append(rec)

    # 2. dPL Canonical Vector: within-basin coordinate-wise median across seeds (42, 123, 2026)
    dpl_df = ledger_df[ledger_df["paradigm"] == "dPL"]
    for struct in STRUCTURES:
        s_df = dpl_df[dpl_df["structure"] == struct]
        for b_id in basins:
            b_sub = s_df[s_df["basin_id"] == b_id]
            frac_snow, stratum = snow_meta[b_id]

            rec = {
                "basin_id": b_id,
                "paradigm": "dPL",
                "regime": "dPL",
                "structure": struct,
                "reduction_rule": "median_across_seeds",
                "selected_member": "seed_median_42_123_2026",
                "train_kge": np.nan,
                "test_kge": np.nan,
                "frac_snow": frac_snow,
                "snow_stratum": stratum,
            }
            for p_name in SHARED_15_PARAMETERS:
                p_sub = b_sub[b_sub["parameter"] == p_name]
                phys_med = float(p_sub["physical_value"].median())
                norm_med = float(p_sub["normalized_value"].median())
                rec[f"phys_{p_name}"] = phys_med
                rec[f"z_{p_name}"] = norm_med
                rec[p_name] = norm_med

            canonical_rows.append(rec)

    expected_canonical_rows = TOTAL_BASINS * len(STRUCTURES) * len(PARADIGMS)
    if len(canonical_rows) != expected_canonical_rows:
        raise RuntimeError(f"Canonical rows count {len(canonical_rows)} != expected {expected_canonical_rows}")

    # Write output CSV
    meta_fields = [
        "basin_id", "paradigm", "regime", "structure", "reduction_rule",
        "selected_member", "train_kge", "test_kge", "frac_snow", "snow_stratum"
    ]
    param_fields = [f"z_{p}" for p in SHARED_15_PARAMETERS] + [f"phys_{p}" for p in SHARED_15_PARAMETERS]
    fields = meta_fields + param_fields + list(SHARED_15_PARAMETERS)

    out_file = out_dir / "r2_parameter_values_canonical.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in canonical_rows:
            writer.writerow(r)

    audit_summary = {
        "status": "PASS",
        "total_canonical_rows": len(canonical_rows),
        "expected_canonical_rows": expected_canonical_rows,
        "total_basins": TOTAL_BASINS,
        "ic_reduction_rule": "complete IC restart with maximum stored train-period KGE; lowest start tie-breaker",
        "dpl_reduction_rule": "within-basin coordinate-wise median across seeds (42, 123, 2026)",
        "output_file": str(out_file),
    }

    with (out_dir / "canonical_vector_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    return canonical_rows, audit_summary


if __name__ == "__main__":
    c_rows, audit = build_canonical_parameter_vectors()
    print(f"Canonical vectors built successfully: {len(c_rows)} rows (Expected: {audit['expected_canonical_rows']}).")
