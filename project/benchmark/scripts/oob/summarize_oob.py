#!/usr/bin/env python3
"""Generate the post-run PRIMARY-8 five-fold OOB ledger and summaries."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from oob_common import (
    EXPECTED_BASINS,
    KGE_EPS,
    N_FOLDS,
    PRIMARY8,
    combine_statistics,
    kge_from_statistics,
    load_ids,
    load_oob_config,
    read_fold_assignment,
    resolve_repo_path,
    validate_fold_contract,
    write_json,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="oob_primary8_5fold_20260902.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    config = load_oob_config(args.config)
    output_root = resolve_repo_path(args.out or config["outputs"]["root"])
    assignment_path = resolve_repo_path(config["folds"]["assignment_file"])
    ids = load_ids(config["data"]["basin_ids"])
    validate_fold_contract(ids, read_fold_assignment(assignment_path), expected_count=int(config["folds"].get("expected_basin_count", EXPECTED_BASINS)))

    jobs: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for model in PRIMARY8:
        fold_scores: list[float] = []
        fold_stats: list[dict[str, Any]] = []
        best_epochs: list[int] = []
        clip_fractions: list[float] = []
        model_failures = 0
        model_incomplete = False
        for fold in range(N_FOLDS):
            run_dir = output_root / "runs" / model / f"fold_{fold}"
            job_id = f"{model}_fold{fold}"
            evaluation_path = run_dir / "heldout_test_evaluation.json"
            health_path = run_dir / "training_health.json"
            if not evaluation_path.exists() or not health_path.exists() or not (run_dir / "DONE").exists():
                incomplete.append(job_id)
                model_incomplete = True
                model_failures += 1
                jobs.append({"job_id": job_id, "model": model, "fold": fold, "status": "INCOMPLETE"})
                continue
            evaluation = read_json(evaluation_path)
            health = read_json(health_path)
            fold_scores.append(float(evaluation["foldwise_median_basin_kge"]))
            fold_stats.append(evaluation["pooled_statistics"])
            best_epochs.append(int(health["best_epoch"]))
            clip_fractions.append(float(health["clip_fraction"]))
            jobs.append(
                {
                    "job_id": job_id,
                    "model": model,
                    "fold": fold,
                    "status": health["status"],
                    "best_epoch": health["best_epoch"],
                    "best_train_loss": health["best_train_loss"],
                    "heldout_median_basin_kge": evaluation["foldwise_median_basin_kge"],
                    "heldout_mean_basin_kge": evaluation["foldwise_mean_basin_kge"],
                    "pooled_fold_kge": evaluation["pooled_fold_kge"],
                }
            )
        if model_incomplete and not args.allow_incomplete:
            continue
        combined = combine_statistics(fold_stats) if fold_stats and not model_incomplete else None
        pooled_kge = float(kge_from_statistics(combined, eps=KGE_EPS)) if combined else float("nan")
        model_rows.append(
            {
                "model": model,
                "folds_complete": len(fold_scores),
                "failure_count": model_failures,
                "heldout_fold_median_kge_mean": float(np.mean(fold_scores)) if fold_scores else float("nan"),
                "heldout_fold_median_kge_median": float(np.median(fold_scores)) if fold_scores else float("nan"),
                "heldout_fold_median_kge_std": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else float("nan"),
                "pooled_531_basin_oob_kge": pooled_kge,
                "best_epoch_mean": float(np.mean(best_epochs)) if best_epochs else float("nan"),
                "best_epoch_min": int(min(best_epochs)) if best_epochs else "",
                "best_epoch_max": int(max(best_epochs)) if best_epochs else "",
                "clip_fraction_mean": float(np.mean(clip_fractions)) if clip_fractions else float("nan"),
                "KGE_IC": float("nan"),
                "KGE_dPL_seen": float("nan"),
                "KGE_dPL_OOB": pooled_kge,
                "G_seen": float("nan"),
                "G_OOB": float("nan"),
                "OOB_degradation": float("nan"),
            }
        )

    if incomplete and not args.allow_incomplete:
        raise SystemExit("cannot summarize incomplete queue; missing: " + ", ".join(incomplete))

    summary_path = output_root / "OOB_PRIMARY8_5FOLD_SUMMARY.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        if model_rows:
            writer = csv.DictWriter(handle, fieldnames=list(model_rows[0]))
            writer.writeheader()
            writer.writerows(model_rows)

    ledger_lines = [
        "# PRIMARY 8 5-Fold OOB Ledger",
        "",
        f"- Experiment: `{config['experiment_id']}`",
        "- Protocol: 5-fold basin-held-out; outer held-out targets are post-hoc only",
        "- Selection: `train_loss` only; exact `best.pt` restored before OOB evaluation",
        "- Normalization: fold-specific, train-basins-only",
        "",
        "| Job | Status | Best epoch | Best train loss | Fold median KGE | Fold mean KGE | Pooled fold KGE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in jobs:
        ledger_lines.append(
            f"| `{row['job_id']}` | {row['status']} | {row.get('best_epoch', '')} | "
            f"{row.get('best_train_loss', '')} | {row.get('heldout_median_basin_kge', '')} | "
            f"{row.get('heldout_mean_basin_kge', '')} | {row.get('pooled_fold_kge', '')} |"
        )
    (output_root / "OOB_PRIMARY8_5FOLD_LEDGER.md").write_text("\n".join(ledger_lines) + "\n", encoding="utf-8")

    report = f"""# PRIMARY 8 5-Fold OOB Report

- Jobs: {sum(row['status'] != 'INCOMPLETE' for row in jobs)}/40 complete.
- Fold assignment: `{assignment_path}`.
- Training selection: `train_loss`; no held-out target is used for selection or early stopping.
- Fold normalization: train basins only; each run stores `attribute_mean.npy` and `attribute_std.npy`.
- `KGE_IC` and `KGE_dPL_seen` are intentionally blank because IC is not rerun in this experiment.
- `KGE_dPL_OOB` is reported only for models with all five completed folds; it is the exact pooled KGE from combined time/basin sufficient statistics.
- This report is a descriptive experiment summary, not a paper-level conclusion.

See `OOB_PRIMARY8_5FOLD_SUMMARY.csv` and `OOB_PRIMARY8_5FOLD_LEDGER.md`.
"""
    (output_root / "OOB_PRIMARY8_5FOLD_REPORT.md").write_text(report, encoding="utf-8")
    write_json(output_root / "OOB_PRIMARY8_5FOLD_SUMMARY_METADATA.json", {"jobs": jobs, "incomplete_jobs": incomplete, "models": list(PRIMARY8)})
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
