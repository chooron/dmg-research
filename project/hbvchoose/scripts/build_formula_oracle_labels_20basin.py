#!/usr/bin/env python3
"""Stage 4: Build train-window oracle formula labels from calibrated results.

Reads formula_metrics_train.csv and formula_metrics_eval.csv from all seeds,
computes oracle labels (best train-NSE formula per basin per seed), and audits
eval generalization.
"""
from __future__ import annotations

import argparse, csv, math, sys
from pathlib import Path

import numpy as np

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dirs", nargs="+", required=True,
                    help="Directories containing formula_metrics_train.csv and formula_metrics_eval.csv")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all metrics
    all_train, all_eval = [], []
    seeds_seen = set()
    for d in args.input_dirs:
        dp = Path(d)
        tr_file = dp / "formula_metrics_train.csv"
        ev_file = dp / "formula_metrics_eval.csv"
        if tr_file.exists():
            rows = list(csv.DictReader(open(tr_file)))
            for r in rows:
                r["source_dir"] = str(dp)
                seeds_seen.add(int(r.get("seed", -1)))
            all_train.extend(rows)
        if ev_file.exists():
            rows = list(csv.DictReader(open(ev_file)))
            for r in rows:
                r["source_dir"] = str(dp)
            all_eval.extend(rows)

    if not all_train:
        print("ERROR: No training metrics found")
        return False

    # Group by basin_id + seed
    from collections import defaultdict
    train_by_key = defaultdict(list)
    for r in all_train:
        bid = int(r["basin_id"])
        seed = int(r["seed"])
        train_by_key[(bid, seed)].append(r)

    eval_by_key = defaultdict(list)
    for r in all_eval:
        bid = int(r["basin_id"])
        seed = int(r["seed"])
        eval_by_key[(bid, seed)].append(r)

    # Compute oracle labels
    oracle_rows = []
    oracle_eval_rows = []
    oracle_summary = []
    label_counts = {"R0": defaultdict(int), "R4": defaultdict(int), "R5": defaultdict(int)}

    for (bid, seed), tr_rows in sorted(train_by_key.items()):
        # Find best by train NSE
        ranked = sorted(tr_rows, key=lambda x: -float(x["train_nse"])
                        if not math.isnan(float(x["train_nse"])) else -1e9)
        if not ranked:
            continue

        best_fid = ranked[0]["formula_id"]
        rank_ids = [r["formula_id"] for r in ranked[:3]]
        while len(rank_ids) < 3:
            rank_ids.append("?")

        tr_nse_map = {}
        for r in tr_rows:
            fid = r["formula_id"]
            tr_nse_map[f"train_nse_{fid}"] = round(float(r["train_nse"]), 6)
            tr_nse_map[f"train_mse_{fid}"] = round(float(r["train_mse"]), 8)

        oracle_rows.append({
            "basin_id": bid,
            "seed": seed,
            "best_train_formula": best_fid,
            "best_train_formula_name": best_fid,
            "rank1_formula": rank_ids[0],
            "rank2_formula": rank_ids[1],
            "rank3_formula": rank_ids[2],
            **{f"train_nse_{fid}": tr_nse_map.get(f"train_nse_{fid}", float("nan")) for fid in ["R0", "R4", "R5"]},
            **{f"train_mse_{fid}": tr_nse_map.get(f"train_mse_{fid}", float("nan")) for fid in ["R0", "R4", "R5"]},
            "label_source": "train_window_fixed_formula_calibration",
            "eval_used_for_label": "False",
        })

        label_counts[best_fid][seed] += 1

        # Eval audit
        ev_rows = [r for r in all_eval if int(r["basin_id"]) == bid and int(r["seed"]) == seed]
        ev_nse_map = {}
        ev_best_fid = None
        ev_best_nse = -float("inf")
        for r in ev_rows:
            fid = r["formula_id"]
            ev_nse = float(r["eval_nse"]) if not math.isnan(float(r["eval_nse"])) else float("nan")
            ev_nse_map[f"eval_nse_{fid}"] = round(ev_nse, 6)
            if not math.isnan(ev_nse) and ev_nse > ev_best_nse:
                ev_best_nse = ev_nse
                ev_best_fid = fid

        best_tr_eval = ev_nse_map.get(f"eval_nse_{best_fid}", float("nan"))
        generalizes = (best_fid == ev_best_fid) if ev_best_fid else "unknown"

        oracle_eval_rows.append({
            "basin_id": bid,
            "seed": seed,
            "best_train_formula": best_fid,
            "eval_nse_of_best_train_formula": best_tr_eval if not math.isnan(float(best_tr_eval)) else float("nan"),
            "best_eval_formula": ev_best_fid or "?",
            **ev_nse_map,
            "generalizes_to_eval": str(generalizes),
            "eval_used_for_label": "False",
        })

    # Oracle summary
    for seed in sorted(label_counts["R0"].keys()):
        oracle_summary.append({
            "seed": seed,
            "oracle_R0": label_counts["R0"].get(seed, 0),
            "oracle_R4": label_counts["R4"].get(seed, 0),
            "oracle_R5": label_counts["R5"].get(seed, 0),
            "total_basins": sum(label_counts[fid].get(seed, 0) for fid in ["R0", "R4", "R5"]),
        })

    # Write
    _w(oracle_rows, out_dir / "oracle_labels_train.csv",
       ["basin_id", "seed", "best_train_formula", "best_train_formula_name",
        "rank1_formula", "rank2_formula", "rank3_formula",
        "train_nse_R0", "train_nse_R4", "train_nse_R5",
        "train_mse_R0", "train_mse_R4", "train_mse_R5",
        "label_source", "eval_used_for_label"])

    _w(oracle_eval_rows, out_dir / "oracle_eval_audit.csv",
       ["basin_id", "seed", "best_train_formula", "eval_nse_of_best_train_formula",
        "best_eval_formula", "eval_nse_R0", "eval_nse_R4", "eval_nse_R5",
        "generalizes_to_eval", "eval_used_for_label"])

    _w(oracle_summary, out_dir / "oracle_summary.csv",
       ["seed", "oracle_R0", "oracle_R4", "oracle_R5", "total_basins"])

    print("Oracle label distribution:")
    for s in oracle_summary:
        print(f"  Seed {s['seed']}: R0={s['oracle_R0']}, R4={s['oracle_R4']}, R5={s['oracle_R5']} / {s['total_basins']}")

    gen_count = sum(1 for r in oracle_eval_rows if r["generalizes_to_eval"] == "True")
    total = len(oracle_eval_rows)
    print(f"\nEval generalization: {gen_count}/{total} ({gen_count/total*100:.1f}%)" if total else "No data")

    print(f"\nDone. Output: {out_dir}")


def _w(rows, path, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        if rows:
            w.writerows(rows)


if __name__ == "__main__":
    main()
