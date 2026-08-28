"""Audit staged inputs and construct the canonical basin-level evaluation table.

Unit of inference: basin (531 unique basins).
Evaluation subset: 531 basins x 3 structures x 2 regimes = 3,186 rows (test period).
Aggregation rules:
  - dPL: median across seeds (42, 123, 2026) per basin x structure before inference.
  - IC: selected restart per basin x structure.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config import (
    EVAL_PERIOD,
    EXPECTED_TABLES,
    PARADIGMS,
    RESULTS_DIR,
    STAGED_DIR,
    STRATA,
    STRATA_COUNTS,
    STRUCTURES,
    TOTAL_BASINS,
)
from cuda_engine import file_sha256


def verify_staged_inputs(staged_dir: Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Verify SHA-256 digests, row counts, and schema contracts of staged inputs."""
    directory = staged_dir or STAGED_DIR
    records = {}

    for filename, expected in EXPECTED_TABLES.items():
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(f"Required staged table missing: {path}")

        actual_sha = file_sha256(path)
        if "sha256" in expected and actual_sha != expected["sha256"]:
            raise RuntimeError(f"Digest mismatch for {filename}: {actual_sha} != {expected['sha256']}")

        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            row_count = sum(1 for _ in reader)

        if expected.get("schema") and header != expected["schema"]:
            raise RuntimeError(f"Schema mismatch for {filename}: {header} != {expected['schema']}")
        if "rows" in expected and row_count != expected["rows"]:
            raise RuntimeError(f"Row count mismatch for {filename}: {row_count} != {expected['rows']}")

        records[filename] = {
            "path": str(path),
            "sha256": actual_sha,
            "row_count": row_count,
            "schema": header,
        }

    return records


def build_canonical_basin_table(
    staged_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Build the canonical 3,186-row basin-level table for the evaluation period.

    Returns:
        (test_rows, all_period_rows, audit_summary)
    """
    directory = staged_dir or STAGED_DIR
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    input_records = verify_staged_inputs(directory)

    # Read performance table
    perf_path = directory / "r1_basin_level_performance_rebuilt.csv"
    perf_rows = {}
    with perf_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            b_id = str(r["basin_id"]).zfill(8)
            key = (b_id, r["paradigm"], r["structure"], r["period"])
            if key in perf_rows:
                raise RuntimeError(f"Duplicate key in performance table: {key}")
            perf_rows[key] = r

    # Read CT table
    ct_path = directory / "r1_basin_level_ct.csv"
    ct_rows = {}
    with ct_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            b_id = str(r["basin_id"]).zfill(8)
            key = (b_id, r["paradigm"], r["structure"], r["period"])
            if key in ct_rows:
                raise RuntimeError(f"Duplicate key in CT table: {key}")
            ct_rows[key] = r

    if set(perf_rows.keys()) != set(ct_rows.keys()):
        raise RuntimeError("Keys in performance and CT tables do not match exactly")

    all_keys = sorted(perf_rows.keys())
    if len(all_keys) != 6372:
        raise RuntimeError(f"Expected 6372 total basin x structure x paradigm x period rows, got {len(all_keys)}")

    # Snow strata metadata audit across 531 unique basins
    basin_snow_meta = {}
    for key, c_row in ct_rows.items():
        b_id = key[0]
        frac_snow = float(c_row["frac_snow"])
        stratum = c_row["snow_stratum"]
        if b_id in basin_snow_meta:
            if basin_snow_meta[b_id] != (frac_snow, stratum):
                raise RuntimeError(f"Inconsistent snow metadata for basin {b_id}")
        else:
            basin_snow_meta[b_id] = (frac_snow, stratum)

    if len(basin_snow_meta) != TOTAL_BASINS:
        raise RuntimeError(f"Expected {TOTAL_BASINS} unique basins, got {len(basin_snow_meta)}")

    strata_observed = {s: sum(1 for v in basin_snow_meta.values() if v[1] == s) for s in STRATA}
    if strata_observed != STRATA_COUNTS:
        raise RuntimeError(f"Observed strata counts {strata_observed} != frozen counts {STRATA_COUNTS}")

    # Combine into canonical records
    all_period_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for key in all_keys:
        b_id, paradigm, structure, period = key
        p_row = perf_rows[key]
        c_row = ct_rows[key]

        signed_ct_str = c_row.get("basin_median_Delta_CT", "")
        signed_ct = float(signed_ct_str) if signed_ct_str != "" else float("nan")
        abs_ct = abs(signed_ct) if signed_ct == signed_ct else float("nan")

        kge_str = p_row.get("KGE", "")
        kge = float(kge_str) if kge_str != "" else float("nan")

        frac_snow = float(c_row["frac_snow"])
        stratum = c_row["snow_stratum"]
        valid_years = int(c_row.get("valid_year_count", 0))

        record = {
            "basin_id": b_id,
            "regime": paradigm,
            "paradigm": paradigm,
            "structure": structure,
            "model": p_row.get("model", f"XAJ-{structure}"),
            "period": period,
            "seed_or_restart": p_row.get("seed_or_restart", ""),
            "selected_run": p_row.get("selected_run", ""),
            "frac_snow": frac_snow,
            "snow_stratum": stratum,
            "KGE": kge,
            "signed_CT_error": signed_ct,
            "absolute_CT_error": abs_ct,
            "valid_year_count": valid_years,
            "valid_observation_count": float(p_row.get("valid_observation_count", "nan")),
            "valid_simulation_count": float(p_row.get("valid_simulation_count", "nan")),
            "valid_days": float(p_row.get("valid_days", "nan")),
            "valid_metric": p_row.get("valid_metric", "True").lower() == "true",
            "NSE": float(p_row.get("NSE", "nan")),
            "PBIAS": float(p_row.get("PBIAS", "nan")),
            "RMSE": float(p_row.get("RMSE", "nan")),
            "provenance_source": "r1_basin_level_performance_rebuilt.csv + r1_basin_level_ct.csv",
        }
        all_period_rows.append(record)

        if period == EVAL_PERIOD:
            test_rows.append(record)

    if len(test_rows) != TOTAL_BASINS * len(STRUCTURES) * len(PARADIGMS):
        raise RuntimeError(f"Expected {TOTAL_BASINS * len(STRUCTURES) * len(PARADIGMS)} test rows, got {len(test_rows)}")

    # Write output CSVs
    fields = [
        "basin_id", "regime", "paradigm", "structure", "model", "period",
        "seed_or_restart", "selected_run", "frac_snow", "snow_stratum",
        "KGE", "signed_CT_error", "absolute_CT_error", "valid_year_count",
        "valid_days", "valid_metric", "NSE", "PBIAS", "RMSE", "provenance_source",
    ]

    canonical_test_path = out_dir / "canonical_basin_level.csv"
    with canonical_test_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in test_rows:
            writer.writerow(r)

    canonical_all_path = out_dir / "canonical_basin_level_all_periods.csv"
    with canonical_all_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_period_rows:
            writer.writerow(r)

    audit_summary = {
        "status": "PASS",
        "total_unique_basins": TOTAL_BASINS,
        "paradigms": list(PARADIGMS),
        "structures": list(STRUCTURES),
        "periods": ["test", "train"],
        "canonical_eval_rows": len(test_rows),
        "all_periods_rows": len(all_period_rows),
        "strata_counts": strata_observed,
        "input_tables": input_records,
        "dpl_seed_aggregation": "median across seeds (42, 123, 2026) per basin x structure",
        "ic_restart_selection": "selected restart per basin x structure",
        "canonical_test_path": str(canonical_test_path),
        "canonical_all_path": str(canonical_all_path),
    }

    audit_path = out_dir / "canonical_basin_table_audit.json"
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    return test_rows, all_period_rows, audit_summary


if __name__ == "__main__":
    test_r, all_r, audit = build_canonical_basin_table()
    print(f"Canonical basin-level table built successfully: {len(test_r)} eval rows, {len(all_r)} all-period rows.")
    print("Strata counts:", audit["strata_counts"])
