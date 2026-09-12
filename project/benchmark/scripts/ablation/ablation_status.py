#!/usr/bin/env python3
"""Ablation Study Real-Time Status Inspection CLI.

Usage:
    python ablation_status.py --root results/dpl_protocol_ablation_v2_20260831
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import yaml


def inspect_status(ablation_root: Path) -> None:
    manifest_path = ablation_root / "experiment_manifest.yaml"
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        return

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    jobs = manifest["jobs"]
    runs_dir = ablation_root / "runs"

    rows = []
    status_counts = {"DONE": 0, "RUNNING": 0, "QUEUED": 0, "FAILED": 0, "BLOCKED_BY_GATE": 0}

    for j in jobs:
        job_id = j["job_id"]
        phase = j.get("phase", "")
        model = j.get("model", "")
        horizon = j.get("horizon_days", 730)
        gate_cond = j.get("gate_condition", "none")

        run_dir = runs_dir / job_id
        if not run_dir.exists():
            status = "BLOCKED_BY_GATE" if j.get("status") == "BLOCKED_BY_GATE" else "QUEUED"
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
        med_kge = "N/A"
        q25_kge = "N/A"
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

        inner_val = run_dir / "inner_validation_metrics.json"
        if inner_val.exists():
            try:
                iv = json.loads(inner_val.read_text())
                med_kge = f"{float(iv.get('median', 0)):.4f}"
                q25_kge = f"{float(iv.get('q25', 0)):.4f}"
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
            "job_id": job_id,
            "phase": phase,
            "model": model,
            "horizon": f"{horizon}d",
            "status": status,
            "epoch": cur_ep,
            "best_train_loss": best_loss,
            "inner_val_med_kge": med_kge,
            "inner_val_q25_kge": q25_kge,
            "runtime": runtime_s,
        })

    df = pd.DataFrame(rows)
    print("=" * 110)
    print(f"ABLATION STUDY STATUS: {ablation_root.name}")
    print(f"Summary: DONE={status_counts['DONE']} | RUNNING={status_counts['RUNNING']} | QUEUED={status_counts['QUEUED']} | FAILED={status_counts['FAILED']} | BLOCKED={status_counts['BLOCKED_BY_GATE']} | TOTAL={len(jobs)}")
    print("=" * 110)

    # Format and display table
    print(f"{'Job ID':<20} | {'Ph':<2} | {'Model':<10} | {'Horizon':<7} | {'Status':<15} | {'Epoch':<5} | {'Train Loss':<10} | {'Med KGE':<8} | {'Q25 KGE':<8} | {'Runtime':<8}")
    print("-" * 110)
    for _, r in df.iterrows():
        print(f"{r['job_id']:<20} | {r['phase']:<2} | {r['model']:<10} | {r['horizon']:<7} | {r['status']:<15} | {r['epoch']:<5} | {r['best_train_loss']:<10} | {r['inner_val_med_kge']:<8} | {r['inner_val_q25_kge']:<8} | {r['runtime']:<8}")
    print("=" * 110)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="project/benchmark/results/dpl_protocol_ablation_v2_20260831", help="Path to ablation root")
    args = parser.parse_args()
    inspect_status(Path(args.root).resolve())


if __name__ == "__main__":
    main()
