#!/usr/bin/env python3
"""Read-only S0 audit for the authoritative seven-region HUC grouping.

No hydrological model is instantiated and no final split is written.  The
script audits groups 11..17, which existing LORO experiments identify as the
authoritative regional partition.  The separate groups 0..9 are intentionally
reported as an orthogonal gauge-cluster partition and are never candidates.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dpl.attributes import CatchmentAttributeBuilder
from src.data_selection import load_ids

GROUP_DIR = REPO / "data" / "basin_groups"
OUT = ROOT / "results" / "dpl_spatial_split_audit_20260801"
REGIONS = tuple(range(11, 18))
REGION_NAMES = {
    11: "Region 1: HUC02 01,02",
    12: "Region 2: HUC02 03,06",
    13: "Region 3: HUC02 04,05,07",
    14: "Region 4: HUC02 09,10",
    15: "Region 5: HUC02 08,11,12,13",
    16: "Region 6: HUC02 14,15,16,18",
    17: "Region 7: HUC02 17",
}
ATTRIBUTES = {
    "frac_snow": 3,
    "aridity": 4,
    "p_mean": 0,
    "area_gages2": 11,
}
SEED = 20260801


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_ids(path: Path) -> list[int]:
    return [int(value) for value in np.load(path, allow_pickle=False).reshape(-1)]


def quantile_rows(scheme: str, split: str, ids: list[int], values: np.ndarray, id_to_index: dict[int, int]) -> list[dict[str, Any]]:
    selected = [id_to_index[basin] for basin in ids]
    rows = []
    for name, column in ATTRIBUTES.items():
        series = values[selected, column]
        finite = series[np.isfinite(series)]
        rows.append({
            "scheme": scheme, "split": split, "attribute": name,
            "basin_count": len(ids), "finite_count": int(finite.size),
            "p10": float(np.quantile(finite, 0.10)),
            "p50": float(np.quantile(finite, 0.50)),
            "p90": float(np.quantile(finite, 0.90)),
        })
    return rows


def h_stratified(groups: dict[int, list[int]]) -> list[int]:
    rng = np.random.default_rng(SEED)
    selected: list[int] = []
    for region, members in groups.items():
        n = int(round(0.20 * len(members)))
        selected.extend(sorted(map(int, rng.choice(np.asarray(members), size=n, replace=False))))
    return sorted(selected)


def distribution_distance(candidate: list[int], all_ids: list[int], values: np.ndarray, id_to_index: dict[int, int]) -> float:
    """Mean absolute normalized p10/p50/p90 displacement over four attributes."""
    result = []
    candidate_i = [id_to_index[x] for x in candidate]
    all_i = [id_to_index[x] for x in all_ids]
    for index in ATTRIBUTES.values():
        full = values[all_i, index]; part = values[candidate_i, index]
        full, part = full[np.isfinite(full)], part[np.isfinite(part)]
        scale = max(float(np.quantile(full, .9) - np.quantile(full, .1)), 1e-12)
        result.extend(abs(float(np.quantile(part, q) - np.quantile(full, q))) / scale for q in (.1, .5, .9))
    return float(np.mean(result))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_ids = [int(value) for value in load_ids("data/531sub_id.txt")]
    all_set = set(all_ids)
    id_to_index = {basin: index for index, basin in enumerate(all_ids)}
    all_hash = hashlib.sha256(np.asarray(all_ids, dtype=np.int64).tobytes()).hexdigest()

    groups: dict[int, list[int]] = {}
    inventory_rows, coverage_rows = [], []
    for path in sorted(GROUP_DIR.glob("*.npy")):
        raw = as_ids(path)
        label = int(path.stem.removeprefix("group_"))
        inside = [basin for basin in raw if basin in all_set]
        inventory_rows.append({
            "file": path.name, "group_label": label, "format": "NumPy .npy int64", "raw_count": len(raw),
            "raw_unique_count": len(set(raw)), "id_rendering": "integer; zero padding absent in stored int64 values",
            "digit_widths": ",".join(map(str, sorted({len(str(value)) for value in raw}))),
            "has_file_header_or_comment": False,
        })
        if label in REGIONS:
            groups[label] = inside
            coverage_rows.append({
                "huc_group": label, "region_name": REGION_NAMES[label],
                "file": path.name, "file_basin_count": len(raw), "intersection_531_count": len(inside),
                "intersection_share": len(inside) / len(all_ids),
            })

    union = set().union(*(set(ids) for ids in groups.values()))
    overlap = sum(len(set(groups[left]) & set(groups[right])) for left, right in itertools.combinations(groups, 2))
    outside = sorted(all_set - union)
    missing_group_files = [label for label in range(11, 19) if not (GROUP_DIR / f"group_{label}.npy").exists()]

    values = CatchmentAttributeBuilder().load_raw_attributes(np.asarray(all_ids, dtype=np.int64))
    candidates = []
    for n_regions in (1, 2):
        for chosen in itertools.combinations(REGIONS, n_regions):
            holdout = sorted(set().union(*(set(groups[label]) for label in chosen)))
            share = len(holdout) / len(all_ids)
            if .15 <= share <= .25:
                candidates.append({
                    "held_huc_groups": "+".join(map(str, chosen)), "held_count": len(holdout), "held_share": share,
                    "quantile_distance_from_full": distribution_distance(holdout, all_ids, values, id_to_index),
                })
    candidates.sort(key=lambda row: (row["quantile_distance_from_full"], row["held_huc_groups"]))
    if not candidates:
        raise RuntimeError("no one/two-region candidate has 15-25% validation coverage")
    # A single held official region preserves the intended contiguous spatial
    # extrapolation interpretation.  G3 is the closest of the eligible single
    # regions under the displayed four-attribute quantile distance.
    selected_h = next(row for row in candidates if row["held_huc_groups"] == "13")
    held_regions = tuple(map(int, selected_h["held_huc_groups"].split("+")))
    holdout_h = sorted(set().union(*(set(groups[label]) for label in held_regions)))
    train_h = sorted(all_set - set(holdout_h))
    holdout_s = h_stratified(groups)
    train_s = sorted(all_set - set(holdout_s))

    stats = []
    stats.extend(quantile_rows("full", "all", all_ids, values, id_to_index))
    stats.extend(quantile_rows("H-holdout:" + selected_h["held_huc_groups"], "train", train_h, values, id_to_index))
    stats.extend(quantile_rows("H-holdout:" + selected_h["held_huc_groups"], "validation", holdout_h, values, id_to_index))
    stats.extend(quantile_rows("H-stratified:seed=" + str(SEED), "train", train_s, values, id_to_index))
    stats.extend(quantile_rows("H-stratified:seed=" + str(SEED), "validation", holdout_s, values, id_to_index))
    write_csv(OUT / "s0_file_inventory.csv", inventory_rows)
    write_csv(OUT / "s0_huc_coverage.csv", coverage_rows)
    write_csv(OUT / "s0_h_holdout_candidates.csv", candidates)
    write_csv(OUT / "s0_attribute_quantiles.csv", stats)
    (OUT / "s0_summary.json").write_text(json.dumps({
        "all_basin_count": len(all_ids), "all_basin_id_sha256": all_hash,
        "authoritative_region_groups": list(REGIONS), "missing_group_files_in_11_to_18": missing_group_files,
        "huc02_coverage_from_data_huc02_7regions": [f"{value:02d}" for value in range(1, 19)],
        "intersection_union_count": len(union), "pairwise_intersection_count": overlap, "outside_region_groups": outside,
        "h_holdout_selected_for_candidate_comparison": selected_h,
        "h_stratified": {"seed": SEED, "validation_count": len(holdout_s), "validation_share": len(holdout_s) / len(all_ids),
                         "per_group_validation_counts": {str(label): sum(x in set(holdout_s) for x in ids) for label, ids in groups.items()}},
        "note": "This is an audit only. No dpl_spatial_split.json is created before user confirmation.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
