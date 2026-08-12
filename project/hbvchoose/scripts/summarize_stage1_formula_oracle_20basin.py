#!/usr/bin/env python3
"""Summarize 20-basin formula oracle results."""
import csv, math, sys
from pathlib import Path
import numpy as np

_BASE = Path(__file__).resolve().parent.parent / "validation_results" / "stage1_formula_oracle_20basin"

def main():
    RECHARGE_FIDS = ["R0", "R4", "R5"]
    seeds = [0, 1, 2]

    all_seed = []
    all_basin = []
    consistency = {}

    for seed in seeds:
        d = _BASE / f"seed{seed}"
        oracle = list(csv.DictReader(open(d / "oracle_labels_train.csv")))
        metrics = list(csv.DictReader(open(d / "formula_metrics_all.csv")))
        router = list(csv.DictReader(open(d / "router_selection_summary.csv")))

        if not oracle or not metrics or not router:
            print(f"WARNING: Missing data for seed {seed}")
            continue

        # Oracle distribution
        r0 = sum(1 for r in oracle if r["best_train_formula"] == "R0")
        r4 = sum(1 for r in oracle if r["best_train_formula"] == "R4")
        r5 = sum(1 for r in oracle if r["best_train_formula"] == "R5")

        # Eval metrics per formula
        ev_r0 = [float(r["eval_nse"]) for r in metrics if r["formula_id"] == "R0" if not math.isnan(float(r["eval_nse"]))]
        ev_r4 = [float(r["eval_nse"]) for r in metrics if r["formula_id"] == "R4" if not math.isnan(float(r["eval_nse"]))]
        ev_r5 = [float(r["eval_nse"]) for r in metrics if r["formula_id"] == "R5" if not math.isnan(float(r["eval_nse"]))]

        # Router accuracy
        correct = sum(1 for r in router if r.get("match_oracle") == "True")
        r0_sel = sum(1 for r in router if r["selected_formula"] == "R0")
        r4_sel = sum(1 for r in router if r["selected_formula"] == "R4")
        r5_sel = sum(1 for r in router if r["selected_formula"] == "R5")

        all_seed.append({
            "seed": seed, "n_basins": len(oracle),
            "oracle_R0": r0, "oracle_R4": r4, "oracle_R5": r5,
            "mean_eval_nse_R0": round(np.mean(ev_r0), 4) if ev_r0 else float("nan"),
            "mean_eval_nse_R4": round(np.mean(ev_r4), 4) if ev_r4 else float("nan"),
            "mean_eval_nse_R5": round(np.mean(ev_r5), 4) if ev_r5 else float("nan"),
            "router_accuracy": f"{correct}/{len(router)}",
            "router_R0_sel": r0_sel, "router_R4_sel": r4_sel, "router_R5_sel": r5_sel,
            "nondefault_rate": round((r4_sel + r5_sel) / len(router), 4) if router else 0,
        })

        for i, o in enumerate(oracle):
            m = next((r for r in router if r["basin_id"] == o["basin_id"]), {})
            bid = int(o["basin_id"])
            all_basin.append({
                "basin_id": bid, "seed": seed,
                "oracle_best": o["best_train_formula"],
                "router_selected": m.get("selected_formula", "?"),
                "match": m.get("match_oracle") == "True",
            })

            if bid not in consistency: consistency[bid] = {}
            if seed not in consistency[bid]: consistency[bid][seed] = o["best_train_formula"]

    # Selection consistency
    cons_rows = []
    for bid, seed_dict in consistency.items():
        formulas = [seed_dict.get(s, "?") for s in seeds]
        from collections import Counter
        c = Counter(formulas)
        majority = c.most_common(1)[0][0] if c else "?"
        cons_rows.append({
            "basin_id": bid,
            "seed0": formulas[0] if len(formulas) > 0 else "?",
            "seed1": formulas[1] if len(formulas) > 1 else "?",
            "seed2": formulas[2] if len(formulas) > 2 else "?",
            "majority": majority,
            "consistency": round(c[majority] / 3, 2) if c else 0,
        })

    out = _BASE
    _w(all_seed, out / "summary_by_seed.csv",
       ["seed", "n_basins", "oracle_R0", "oracle_R4", "oracle_R5",
        "mean_eval_nse_R0", "mean_eval_nse_R4", "mean_eval_nse_R5",
        "router_accuracy", "router_R0_sel", "router_R4_sel", "router_R5_sel",
        "nondefault_rate"])
    _w(all_basin, out / "summary_by_basin.csv",
       ["basin_id", "seed", "oracle_best", "router_selected", "match"])
    _w(cons_rows, out / "selection_consistency.csv",
       ["basin_id", "seed0", "seed1", "seed2", "majority", "consistency"])

    # Average consistency
    avg_cons = np.mean([r["consistency"] for r in cons_rows]) if cons_rows else 0
    print(f"Average oracle consistency: {avg_cons:.2f}")
    for s in all_seed:
        print(f"  Seed {s['seed']}: oracle R0={s['oracle_R0']} R4={s['oracle_R4']} R5={s['oracle_R5']} "
              f"nondefault_rate={s['nondefault_rate']}")
    print(f"Output: {out}")


def _w(rows, path, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
