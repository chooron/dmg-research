#!/usr/bin/env python3
"""Stage 2: Screen and select 20 real CAMELS basins for formula-MoE experiment.

Output:
  validation_results/static_router_20basin_calibrated/
    selected_basins.csv
    excluded_basins.csv
    basin_screening_report.md
"""
from __future__ import annotations

import argparse, csv, math, pickle, sys, warnings
from pathlib import Path

import numpy as np

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

CAMELS_PATH = _PROJECT.parent.parent / "data" / "camels_dataset"
GAGE_ID_PATH = _PROJECT.parent.parent / "data" / "gage_id.npy"

ATTR_NAMES = [
    "p_mean", "pet_mean", "p_seasonality", "frac_snow", "aridity",
    "high_precip_freq", "high_precip_dur", "low_precip_freq", "low_precip_dur",
    "mean_precipitation", "mean_pet", "elev_mean", "slope_mean", "area_gages2",
    "forest_frac", "lai_max", "lai_diff", "gvf_max", "gvf_diff",
    "soil_conductivity", "soil_porosity", "soil_depth_pelletier",
    "sand_frac", "silt_frac", "clay_frac",
    "soil_depth_statsgo", "soil_porosity_statsgo", "soil_conductivity_statsgo",
    "max_water_content", "geol_permeability", "geol_porosity",
    "carbonate_rocks_frac", "geol_conductivity", "geol_porosity_saturated",
    "elev_std",
]

DIVERSITY_COLS = [
    "aridity", "frac_snow", "mean_precipitation", "mean_pet",
    "elevation", "slope", "basin_area", "forest_fraction",
    "soil_conductivity", "soil_water_capacity",
]

COL_IDX = {
    "aridity": 4, "frac_snow": 3, "mean_precipitation": 9, "mean_pet": 10,
    "elevation": 11, "slope": 12, "basin_area": 13, "forest_fraction": 14,
    "soil_conductivity": 19, "soil_water_capacity": 28,
}


def flow_to_mmd(flow, area):
    return flow * 2.446575 / max(area, 1.0)


def screen_basins(forcings, target, attributes, gage_ids, warmup_d, train_d,
                  eval_d, max_b=20, strict=True):
    need = warmup_d + train_d + eval_d
    min_ratio = 0.90 if strict else 0.80
    selected = []
    excluded = []
    for idx in range(forcings.shape[0]):
        bid = int(gage_ids[idx])
        tl = forcings.shape[1]
        if tl < need:
            excluded.append({"basin_id": bid, "reason": f"Too short {tl}<{need}"})
            continue
        forc = forcings[idx]
        targ = target[idx, :, 0]
        area = attributes[idx, 11]
        fnan = int(np.isnan(forc).sum())
        finf = int(np.isinf(forc).sum())
        if fnan > 0 or finf > 0:
            excluded.append({"basin_id": bid, "reason": f"Forcing NaN={fnan} Inf={finf}"})
            continue
        targ_mmd = flow_to_mmd(targ, area)
        ev_s, ev_e = warmup_d + train_d, warmup_d + train_d + eval_d
        valid_ev = ~(np.isnan(targ[ev_s:ev_e]) | np.isinf(targ[ev_s:ev_e]))
        ev_ratio = float(valid_ev.sum() / max(len(valid_ev), 1))
        valid_all = ~(np.isnan(targ) | np.isinf(targ))
        all_ratio = float(valid_all.sum() / max(len(targ), 1))
        qinf = int(np.isinf(targ_mmd).sum())
        qzero = float((np.abs(targ_mmd) < 1e-8).mean())
        train_valid_mask = ~np.isnan(targ[warmup_d:warmup_d + train_d])
        tr_ratio = float(train_valid_mask.sum() / max(len(train_valid_mask), 1))
        rej = False
        if strict:
            if all_ratio < 0.90 or ev_ratio < 0.90 or tr_ratio < 0.90:
                rej = True
        else:
            if all_ratio < 0.80 or ev_ratio < 0.80:
                rej = True
        if qinf > 0 or qzero > 0.95:
            rej = True
        if rej:
            excluded.append({
                "basin_id": bid, "reason": "Quality check failed",
                "valid_target_ratio": round(all_ratio, 4),
                "train_valid_ratio": round(tr_ratio, 4),
                "eval_valid_ratio": round(ev_ratio, 4),
                "forcing_nan_count": fnan, "forcing_inf_count": finf,
                "q_inf_count": qinf, "q_zero_ratio": round(qzero, 4),
                "total_length": tl,
            })
            continue
        sel = {
            "basin_id": bid,
            "valid_target_ratio": round(all_ratio, 4),
            "train_valid_ratio": round(tr_ratio, 4),
            "eval_valid_ratio": round(ev_ratio, 4),
            "forcing_nan_count": fnan,
            "forcing_inf_count": finf,
            "q_nan_count": int(np.isnan(targ_mmd).sum()),
            "q_inf_count": qinf,
            "q_zero_ratio": round(qzero, 4),
            "total_length": tl,
            "warmup_days": warmup_d,
            "train_days": train_d,
            "eval_days": eval_d,
            "screening_mode": "strict" if strict else "fallback",
        }
        for dc in DIVERSITY_COLS:
            ci = COL_IDX.get(dc)
            if ci is not None and ci < attributes.shape[1]:
                sel[dc] = round(float(attributes[idx, ci]), 6)
        selected.append(sel)
    if len(selected) < max_b and strict:
        return screen_basins(forcings, target, attributes, gage_ids,
                             warmup_d, train_d, eval_d, max_b, strict=False)
    return selected, excluded


def select_diverse(screened, max_b=20):
    if len(screened) <= max_b:
        for s in screened:
            s["selection_reason"] = "quality_first"
        return screened
    result = []
    result.append(screened[0])
    screened[0]["selection_reason"] = "quality_first"
    remaining = screened[1:]
    # Greedy max-min diversity
    while len(result) < max_b and remaining:
        best_i, best_d = -1, -1.0
        for i, s in enumerate(remaining):
            dists = []
            for r in result:
                d2 = 0.0
                nf = 0
                for dc in DIVERSITY_COLS:
                    v1 = r.get(dc, 0.0)
                    v2 = s.get(dc, 0.0)
                    if not (math.isnan(v1) or math.isnan(v2)):
                        d2 += (v1 - v2) ** 2
                        nf += 1
                if nf > 0:
                    dists.append(math.sqrt(d2 / nf))
            if dists:
                min_d = min(dists)
                if min_d > best_d:
                    best_d = min_d
                    best_i = i
        if best_i < 0:
            break
        sel = remaining.pop(best_i)
        sel["selection_reason"] = f"diversity_d={best_d:.4f}"
        result.append(sel)
    for s in remaining:
        s["selection_reason"] = "backup"
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup-days", type=int, default=365)
    ap.add_argument("--train-days", type=int, default=365)
    ap.add_argument("--eval-days", type=int, default=365)
    ap.add_argument("--max-basins", type=int, default=20)
    ap.add_argument("--output-dir",
                    default=str(_PROJECT / "validation_results/static_router_20basin_calibrated"))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(CAMELS_PATH, "rb") as f:
        forcings, target, attributes = pickle.load(f)
    gage_ids = np.load(GAGE_ID_PATH)

    print(f"Loaded {forcings.shape[0]} basins, {forcings.shape[1]} steps each")
    print(f"Window: warmup={args.warmup_days}d, train={args.train_days}d, eval={args.eval_days}d")

    selected, excluded = screen_basins(
        forcings, target, attributes, gage_ids,
        args.warmup_days, args.train_days, args.eval_days,
        args.max_basins, strict=True)

    print(f"Strict screening: {len(selected)} candidates, {len(excluded)} excluded")

    mode = selected[0]["screening_mode"] if selected else "fallback"
    print(f"Screening mode: {mode}")

    if len(selected) > args.max_basins:
        selected = select_diverse(selected, args.max_basins)
        print(f"Diversity selection: {len(selected)} basins")
    elif len(selected) < args.max_basins:
        print(f"WARNING: Only {len(selected)} basins qualify (need {args.max_basins})")
        print("Falling back to relaxed rules...")
        selected, excluded2 = screen_basins(
            forcings, target, attributes, gage_ids,
            args.warmup_days, args.train_days, args.eval_days,
            args.max_basins, strict=False)
        mode = "fallback"
        print(f"Fallback screening: {len(selected)} candidates")
        if len(selected) > args.max_basins:
            selected = select_diverse(selected, args.max_basins)

    # Write outputs
    sel_fields = ["basin_id", "valid_target_ratio", "train_valid_ratio",
                  "eval_valid_ratio", "forcing_nan_count", "forcing_inf_count",
                  "q_nan_count", "q_inf_count", "q_zero_ratio", "total_length",
                  "warmup_days", "train_days", "eval_days", "screening_mode",
                  "selection_reason"] + DIVERSITY_COLS
    _write_csv(selected, out_dir / "selected_basins.csv", sel_fields)

    if excluded:
        exc_fields = ["basin_id", "reason", "valid_target_ratio",
                      "train_valid_ratio", "eval_valid_ratio",
                      "forcing_nan_count", "forcing_inf_count",
                      "q_inf_count", "q_zero_ratio", "total_length"]
        _write_csv(excluded, out_dir / "excluded_basins.csv", exc_fields)

    # Report
    basin_ids = [s["basin_id"] for s in selected]
    lines = [
        "# 20-Basin Screening Report",
        "",
        f"## Configuration",
        f"- warmup_days: {args.warmup_days}",
        f"- train_days: {args.train_days}",
        f"- eval_days: {args.eval_days}",
        f"- max_basins: {args.max_basins}",
        f"- screening_mode: {mode}",
        "",
        f"## Results",
        f"- Selected: {len(selected)} basins",
        f"- Excluded: {len(excluded)} basins",
        f"- Basin IDs: {', '.join(str(b) for b in basin_ids)}",
        "",
        f"## Diversity Coverage",
    ]
    if selected:
        for dc in DIVERSITY_COLS:
            vals = [s.get(dc, float("nan")) for s in selected if not math.isnan(s.get(dc, float("nan")))]
            if vals:
                lines.append(f"- {dc}: min={min(vals):.4f}, max={max(vals):.4f}, mean={np.mean(vals):.4f}")
    (out_dir / "basin_screening_report.md").write_text("\n".join(lines))

    print(f"\nSelected {len(selected)} basins:")
    for i, s in enumerate(selected):
        vals = ", ".join(f"{dc}={s.get(dc, 'NA')}" for dc in DIVERSITY_COLS[:5])
        print(f"  [{i+1}] {s['basin_id']}  {vals}")
    print(f"\nOutput: {out_dir}")


def _write_csv(rows, path, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        if rows:
            w.writerows(rows)


if __name__ == "__main__":
    main()
