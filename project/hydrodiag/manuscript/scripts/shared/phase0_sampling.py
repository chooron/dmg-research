"""Deterministic exogenous 60-basin Phase-0 sampling manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ablation.ic_core.config import load_resolved_config
from ablation.ic_core.data_adapter import ATTRIBUTE_NAMES, load_531_bundle

SELECTION_SEED = 202608
ALPHA = 0.925
MIN_VALID_DAYS = 365


def _longest_valid_segment(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values) & (values >= 0.0)
    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if not len(starts):
        return np.empty(0, dtype=np.float64)
    lengths = ends - starts
    index = int(np.argmax(lengths))
    return values[starts[index] : ends[index]]


def lyne_hollick_bfi(discharge: np.ndarray, alpha: float = ALPHA) -> float:
    """Three-pass Lyne-Hollick observed BFI with fixed transparent settings."""
    q = _longest_valid_segment(discharge)
    if q.size < MIN_VALID_DAYS or not np.isfinite(q).all() or q.sum() <= 0.0:
        return float("nan")
    q = np.maximum(q, 0.0)
    base = q.copy()
    for _ in range(3):
        forward = np.empty_like(q)
        forward[0] = q[0]
        for index in range(1, len(q)):
            forward[index] = alpha * forward[index - 1] + (1.0 + alpha) * 0.5 * (
                q[index] - q[index - 1]
            )
            forward[index] = min(max(forward[index], 0.0), q[index])
        reverse = forward[::-1].copy()
        for index in range(1, len(reverse)):
            reverse[index] = alpha * reverse[index - 1] + (1.0 + alpha) * 0.5 * (
                forward[::-1][index] - forward[::-1][index - 1]
            )
            reverse[index] = min(max(reverse[index], 0.0), forward[::-1][index])
        base = reverse[::-1]
    return float(np.sum(base) / max(np.sum(q), 1e-30))


def _rank_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(len(values) - 1, 1)


def _greedy_maximin(ids: list[str], points: np.ndarray, n: int) -> list[int]:
    remaining = list(range(len(ids)))
    centroid = points.mean(axis=0)
    first = max(
        remaining,
        key=lambda i: (float(np.linalg.norm(points[i] - centroid)), -int(ids[i])),
    )
    chosen = [first]
    remaining.remove(first)
    while remaining and len(chosen) < n:
        best = max(
            remaining,
            key=lambda i: (
                float(min(np.linalg.norm(points[i] - points[j]) for j in chosen)),
                float(np.linalg.norm(points[i] - centroid)),
                ids[i],
            ),
        )
        chosen.append(best)
        remaining.remove(best)
    return chosen


def build_manifest(config_path: str | Path, output: str | Path) -> dict:
    config = load_resolved_config(config_path, device_override="cpu")
    bundle = load_531_bundle(config)
    ids = list(bundle.basin_ids)
    precip = bundle.forcing[:, :, 0].astype(np.float64)
    pet = bundle.forcing[:, :, 2].astype(np.float64)
    valid_p = np.isfinite(precip) & (precip >= 0.0)
    valid_pet = np.isfinite(pet) & (pet >= 0.0)
    ai = np.divide(
        np.where(valid_pet, pet, np.nan).mean(axis=1),
        np.where(valid_p, precip, np.nan).mean(axis=1),
    )
    bfi = np.asarray(
        [lyne_hollick_bfi(q) for q in bundle.target_mm_day], dtype=np.float64
    )
    geo = bundle.raw_attributes[:, ATTRIBUTE_NAMES.index("geol_permeability")]
    eligible = np.isfinite(ai) & np.isfinite(bfi) & np.isfinite(geo)
    if int(eligible.sum()) < 60:
        raise RuntimeError(f"Only {int(eligible.sum())} eligible basins; need 60")
    eligible_indices = np.flatnonzero(eligible)
    quintile = np.full(len(ids), -1, dtype=np.int64)
    ai_order = np.argsort(ai[eligible_indices], kind="mergesort")
    for rank, index in enumerate(ai_order):
        quintile[eligible_indices[index]] = min(4, rank * 5 // len(eligible_indices))
    bfi_rank = _rank_percentile(bfi[eligible_indices])
    geo_rank = _rank_percentile(geo[eligible_indices])
    response_points = np.column_stack((bfi_rank, geo_rank))
    selected: list[int] = []
    strata_metadata = []
    for stratum in range(5):
        local = np.flatnonzero(quintile[eligible_indices] == stratum)
        local_ids = [ids[int(eligible_indices[i])] for i in local]
        chosen_local = _greedy_maximin(local_ids, response_points[local], 12)
        chosen_global = [int(eligible_indices[local[i]]) for i in chosen_local]
        selected.extend(chosen_global)
        strata_metadata.append(
            {
                "ai_quintile": stratum + 1,
                "candidate_count": int(len(local)),
                "selected_count": len(chosen_global),
                "selection": "greedy maximin in global rank-percentile (BFI, geol_permeability) space",
                "tie_break": "lexicographic basin ID after distance keys",
            }
        )
    selected.sort(key=lambda i: (int(quintile[i]), float(ai[i]), ids[i]))
    rows = []
    for position, index in enumerate(selected):
        rows.append(
            {
                "selected_position": position,
                "basin_id": ids[index],
                "source_index": int(bundle.source_indices[index]),
                "AI": float(ai[index]),
                "AI_quintile": int(quintile[index] + 1),
                "baseflow_index": float(bfi[index]),
                "geol_permeability": float(geo[index]),
                "selection_seed": SELECTION_SEED,
                "selection_method": "AI quintiles then within-quintile greedy maximin",
            }
        )
    manifest = {
        "protocol": "phase0_ic_only_v1",
        "selection_seed": SELECTION_SEED,
        "n_requested": 60,
        "n_selected": len(rows),
        "n_eligible": int(eligible.sum()),
        "n_excluded": int((~eligible).sum()),
        "excluded_reasons": {
            "missing_or_invalid_AI": int((~np.isfinite(ai)).sum()),
            "missing_or_invalid_baseflow_index": int((~np.isfinite(bfi)).sum()),
            "missing_or_invalid_geol_permeability": int((~np.isfinite(geo)).sum()),
        },
        "AI_definition": "mean PET / mean P over the complete current CAMELS forcing period, ignoring invalid nonnegative forcing values",
        "baseflow_index_definition": "three-pass Lyne-Hollick filter, alpha=0.925, longest valid observed discharge segment, nonnegative clamped baseflow",
        "response_attribute_standardization": "global rank percentile over eligible basins",
        "strata": strata_metadata,
        "basins": rows,
        "source": {
            "dataset": str(Path(config["dataset_path"]).resolve()),
            "gage_ids": str(Path(config["gage_ids_path"]).resolve()),
            "dates": str(Path(config["dates_path"]).resolve()),
            "forcing_period": [str(bundle.dates[0]), str(bundle.dates[-1])],
            "no_model_results_used": True,
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".csv").write_text(
        "selected_position,basin_id,source_index,AI,AI_quintile,baseflow_index,geol_permeability,selection_seed\n"
        + "\n".join(
            f"{r['selected_position']},{r['basin_id']},{r['source_index']},{r['AI']},{r['AI_quintile']},{r['baseflow_index']},{r['geol_permeability']},{r['selection_seed']}"
            for r in rows
        )
        + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="ablation/configs/ic_foundation_531_v1.json"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.config, args.output)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("n_selected", "n_eligible", "n_excluded", "strata")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
