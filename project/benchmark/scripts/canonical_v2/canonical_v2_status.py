#!/usr/bin/env python3
"""Canonical dPL v2 Real-Time Status Inspection CLI.

Usage:
    python canonical_v2_status.py --root project/benchmark/results/dpl_canonical_v2_20260831
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import yaml


def inspect_canonical_status(output_dir: Path) -> None:
    manifest_path = output_dir / "canonical_manifest.yaml"
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        return

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    jobs = manifest["jobs"]
    runs_dir = output_dir / "runs"

    rows = []
    status_counts = {"DONE": 0, "RUNNING": 0, "QUEUED": 0, "FAILED": 0}

    for j in jobs:
        m = j["model"]
        n_params = j.get("n_params", "N/A")
        horizon = j.get("horizon_days", 1095)

        run_dir = runs_dir / m
        if not run_dir.exists():
            status = "QUEUED"
        elif (run_dir / "DONE").exists():
            status = "DONE"
        elif (run_dir / "FAILED").exists():
            status = "FAILED"
        elif (run_dir / ".lock").exists():
            status = "RUNNING"
        else:
            status = "QUEUED"

        status_counts[status] = status_counts.get(status, 0) + 1

        cur_ep = "N/A"
        best_loss = "N/A"
        test_med = "N/A"
        test_q25 = "N/A"
        runtime_s = "N/A"

        ep_csv = run_dir / "epoch_metrics.csv"
        if ep_csv.exists():
            try:
                df_ep = pd.read_csv(ep_csv)
                if not df_ep.empty:
                    cur_ep = str(int(df_ep["epoch"].iloc[-1]))
            except Exception:
                pass

        best_meta = run_dir / "best_metadata.json"
        if best_meta.exists():
            try:
                bm = json.loads(best_meta.read_text())
                best_loss = f"{float(bm.get('best_selection_value', 0)):.4f}"
            except Exception:
                pass

        test_file = run_dir / "test_evaluation.json"
        if test_file.exists():
            try:
                tv = json.loads(test_file.read_text())
                test_med = f"{float(tv.get('median', 0)):.4f}"
                test_q25 = f"{float(tv.get('q25', 0)):.4f}"
            except Exception:
                pass

        rt_file = run_dir / "runtime.json"
        if rt_file.exists():
            try:
                rt = json.loads(rt_file.read_text())
                runtime_s = f"{float(rt.get('total_runtime_s', 0)):.1f}s"
            except Exception:
                pass

        rows.append({
            "model": m,
            "params": n_params,
            "horizon": f"{horizon}d",
            "status": status,
            "epoch": cur_ep,
            "best_train_loss": best_loss,
            "test_median_kge": test_med,
            "test_q25_kge": test_q25,
            "runtime": runtime_s,
        })

    df = pd.DataFrame(rows)
    print("=" * 110)
    print(f"CANONICAL dPL v2 BENCHMARK STATUS: {output_dir.name}")
    print(f"Summary: DONE={status_counts['DONE']}/{len(jobs)} | RUNNING={status_counts['RUNNING']} | QUEUED={status_counts['QUEUED']} | FAILED={status_counts['FAILED']}")
    print("=" * 110)

    # Format table
    print(f"{'Model':<15} | {'Params':<6} | {'Horizon':<7} | {'Status':<12} | {'Epoch':<5} | {'Train Loss':<10} | {'Test Med KGE':<12} | {'Test Q25':<8} | {'Runtime':<8}")
    print("-" * 110)
    for _, r in df.iterrows():
        print(f"{r['model']:<15} | {str(r['params']):<6} | {r['horizon']:<7} | {r['status']:<12} | {r['epoch']:<5} | {r['best_train_loss']:<10} | {r['test_median_kge']:<12} | {r['test_q25_kge']:<8} | {r['runtime']:<8}")
    print("=" * 110)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="project/benchmark/results/dpl_canonical_v2_20260831", help="Path to canonical v2 root")
    args = parser.parse_args()
    inspect_canonical_status(Path(args.root).resolve())


if __name__ == "__main__":
    main()
