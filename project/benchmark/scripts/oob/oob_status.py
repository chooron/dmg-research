#!/usr/bin/env python3
"""Report local PRIMARY-8 OOB job status without touching running jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from oob_common import PRIMARY8, N_FOLDS, load_oob_config, resolve_repo_path, write_json
from run_oob_queue import status_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="oob_primary8_5fold_20260902.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    config = load_oob_config(args.config)
    output_root = resolve_repo_path(args.out or config["outputs"]["root"])
    rows = []
    for fold in range(N_FOLDS):
        for model in PRIMARY8:
            run_dir = output_root / "runs" / model / f"fold_{fold}"
            rows.append({"job_id": f"{model}_fold{fold}", "model": model, "fold": fold, "status": status_for(run_dir)})
    counts = {status: sum(row["status"] == status for row in rows) for status in ("DONE", "RUNNING", "QUEUED", "FAILED")}
    payload = {"output_root": str(output_root), "jobs_total": len(rows), "counts": counts, "jobs": rows}
    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        for row in rows:
            print(f"{row['job_id']:24s} {row['status']}")
        print("SUMMARY " + " ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
