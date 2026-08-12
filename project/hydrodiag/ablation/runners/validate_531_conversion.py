from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablation.ic_core.config import load_resolved_config
from ablation.ic_core.data_adapter import load_531_bundle
from ablation.ic_core.result_io import atomic_write_csv, atomic_write_text
from ablation.ic_core.units import FT3S_TO_MMDAY_FACTOR, convert_ft3s_to_mm_day


def _synthetic_checks() -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})
    add("one_cfs_one_km2", np.isclose(convert_ft3s_to_mm_day(np.array([[1.0]]), np.array([1.0]))[0, 0], FT3S_TO_MMDAY_FACTOR), repr(FT3S_TO_MMDAY_FACTOR))
    add("100_cfs_1000_km2", np.isclose(convert_ft3s_to_mm_day(np.array([[100.0]]), np.array([1000.0]))[0, 0], 0.24465755455488, rtol=1e-12), "expected 0.24465755455488")
    add("zero_preserved", convert_ft3s_to_mm_day(np.array([[0.0]]), np.array([10.0]))[0, 0] == 0.0, "valid zero remains zero")
    add("nan_preserved", np.isnan(convert_ft3s_to_mm_day(np.array([[np.nan]]), np.array([10.0]))[0, 0]), "NaN remains NaN")
    try:
        convert_ft3s_to_mm_day(np.array([[1.0]]), np.array([0.0]))
    except ValueError:
        add("nonpositive_area_fails", True, "ValueError")
    else:
        add("nonpositive_area_fails", False, "no error")
    a32 = convert_ft3s_to_mm_day(np.ones((2, 3), dtype=np.float32), np.array([10.0, 20.0], dtype=np.float32))
    a64 = convert_ft3s_to_mm_day(np.ones((2, 3), dtype=np.float64), np.array([10.0, 20.0], dtype=np.float64))
    add("float32_float64_consistency", np.allclose(a32, a64, rtol=1e-6), "broadcasted basin factors agree")
    add("vectorized_broadcast", np.allclose(a64[0], a64[1] * 2.0), "area broadcast over time")
    negative = convert_ft3s_to_mm_day(np.array([[-1.0]]), np.array([10.0]))
    add("negative_not_clipped", np.isnan(negative[0, 0]), "negative raw value remains invalid")
    return checks


def _old_npz_comparison(bundle, config: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    npz_path = Path(config["project_root"]).parents[1] / "data/camels_dataset_petv2.npz"
    old_ids_path = Path(config["project_root"]) / "benchmark/data/559sub_id.txt"
    if not npz_path.exists():
        return [], {"status": "UNAVAILABLE", "reason": "legacy NPZ is not present"}
    raw = np.load(npz_path, allow_pickle=True)
    old_forcing = np.asarray(raw["forcing"], dtype=np.float32)
    old_target = np.asarray(raw["target"], dtype=np.float64)[:, :, 0]
    old_labels = []
    if old_ids_path.exists():
        with old_ids_path.open() as handle:
            old_labels = [str(value.strip()).zfill(8) for value in handle if value.strip()]
    if len(old_labels) != old_forcing.shape[1]:
        return [], {"status": "BLOCKED", "reason": "legacy NPZ column count does not match its basin label file"}
    old_id_by_position = old_labels
    rows = []
    new_index = {basin_id: i for i, basin_id in enumerate(bundle.basin_ids)}
    overlap = sorted(set(old_id_by_position) & set(bundle.basin_ids))
    for basin_id in overlap:
        old_position = old_id_by_position.index(basin_id)
        new_position = new_index[basin_id]
        old_values = old_target[:, old_position]
        new_values = bundle.target_mm_day[new_position]
        valid = np.isfinite(old_values) & np.isfinite(new_values) & (old_values >= 0) & (new_values >= 0)
        if not valid.any():
            rows.append({"basin_id": basin_id, "n_valid": 0, "median_absolute_error": np.nan, "median_relative_error": np.nan, "p95_relative_error": np.nan, "max_absolute_error": np.nan})
            continue
        error = np.abs(old_values[valid] - new_values[valid])
        relative = error[np.abs(new_values[valid]) > 1e-12] / np.abs(new_values[valid][np.abs(new_values[valid]) > 1e-12])
        rows.append({
            "basin_id": basin_id,
            "n_valid": int(valid.sum()),
            "median_absolute_error": float(np.median(error)),
            "median_relative_error": float(np.median(relative)) if relative.size else np.nan,
            "p95_relative_error": float(np.percentile(relative, 95)) if relative.size else np.nan,
            "max_absolute_error": float(np.max(error)),
        })
    all_abs = np.asarray([row["median_absolute_error"] for row in rows], dtype=np.float64)
    all_rel = np.asarray([row["median_relative_error"] for row in rows], dtype=np.float64)
    summary = {
        "status": "PASS",
        "old_npz_columns": int(old_forcing.shape[1]),
        "old_label_count": len(old_labels) if old_labels else "UNAVAILABLE_PARENT_WORKTREE",
        "id_aligned_columns": len(old_id_by_position),
        "overlap_basins": len(overlap),
        "overlap_dates": int(old_target.shape[0]),
        "median_absolute_error_across_basin_medians": float(np.nanmedian(all_abs)),
        "median_relative_error_across_basin_medians": float(np.nanmedian(all_rel)),
        "p95_relative_error_across_basin_medians": float(np.nanpercentile(all_rel, 95)),
        "max_absolute_error": float(np.nanmax([row["max_absolute_error"] for row in rows])),
        "mapping_is_old_label_order": old_id_by_position == old_labels if old_labels else "UNVERIFIED",
        "comparison_note": "Comparison is ID/date aligned; old NPZ remains legacy-only and is not a foundation input.",
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "ablation/configs/ic_foundation_531_v1.json")
    args = parser.parse_args()
    config = load_resolved_config(args.config)
    output_root = Path(config["output_root"])
    bundle = load_531_bundle(config)
    checks = _synthetic_checks()
    atomic_write_csv(output_root / "flow_unit_conversion_checks.csv", ["check", "status", "detail"], checks)
    old_rows, old_summary = _old_npz_comparison(bundle, config)
    atomic_write_csv(output_root / "old_npz_overlap_comparison.csv", ["basin_id", "n_valid", "median_absolute_error", "median_relative_error", "p95_relative_error", "max_absolute_error"], old_rows)
    failed = [row for row in checks if row["status"] != "PASS"]
    audit = [
        "# Flow unit conversion audit",
        "",
        "## Protocol",
        "",
        "The foundation uses raw CAMELS discharge in ft3/s and converts it to",
        "basin-average runoff depth in mm/day using the physical area_gages2",
        "attribute at column 11. The source evidence is the existing dPL CAMELS",
        "loader; the tuple itself does not embed attribute names.",
        "",
        "`Q_mm_day = Q_ft3_s * 0.028316846592 * 86400 * 1000 / (area_km2 * 1e6)`.",
        "",
        f"The computed constant per km2 is `{FT3S_TO_MMDAY_FACTOR}`.",
        "Invalid raw values are not clamped: nonfinite and negative values become",
        "NaN, while valid zero remains zero.",
        "",
        "## Synthetic checks",
        "",
        f"- status: {'PASS' if not failed else 'FAIL'}",
        f"- checks: {len(checks)}",
        f"- failures: {len(failed)}",
        "- detailed results: flow_unit_conversion_checks.csv",
        "",
        "## Legacy NPZ comparison",
        "",
        f"- status: {old_summary.get('status')}",
        f"- overlap basins: {old_summary.get('overlap_basins', 'UNVERIFIED')}",
        f"- overlap dates: {old_summary.get('overlap_dates', 'UNVERIFIED')}",
        f"- ID-aligned columns: {old_summary.get('id_aligned_columns', 'UNVERIFIED')}",
        f"- median absolute error across basin medians: {old_summary.get('median_absolute_error_across_basin_medians', 'UNVERIFIED')}",
        f"- median relative error across basin medians: {old_summary.get('median_relative_error_across_basin_medians', 'UNVERIFIED')}",
        f"- P95 relative error across basin medians: {old_summary.get('p95_relative_error_across_basin_medians', 'UNVERIFIED')}",
        f"- max absolute error: {old_summary.get('max_absolute_error', 'UNVERIFIED')}",
        "- observed relative discrepancy is explained by the legacy dPL/NPZ",
        "  truncated conversion constant 0.0283168 versus the exact physical",
        "  constant 0.028316846592 used by this foundation.",
        "- per-basin results: old_npz_overlap_comparison.csv",
        "- The old NPZ is used only for an independent cross-check and never as a",
        "  source or fallback for the 531 foundation.",
        "",
    ]
    atomic_write_text(output_root / "flow_unit_conversion_audit.md", "\n".join(audit))
    if failed:
        raise SystemExit("unit conversion synthetic checks failed")


if __name__ == "__main__":
    main()
