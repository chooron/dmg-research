#!/usr/bin/env python3
"""Rebuild and audit a dPL benchmark without loading the GPU.

The original dPL launcher writes the per-model artifacts before attempting to
write the shared summary CSV.  This script treats the per-model artifacts and
logs as the source of truth, rebuilds a deterministic summary, and separates
real training failures from post-processing failures.

Example:
    python scripts/rebuild_dpl_summary.py \
        --root /autodl-fs/data/dpl_run_20260814/project/benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

EXPECTED_MODELS = [
    "alpine1", "alpine2", "australia", "collie1", "collie2", "collie3",
    "flexb", "flexi", "flexis", "gr4j", "gsfb", "hbv96", "hillslope",
    "hymod", "ihacres", "modhydrolog", "mopex1", "mopex2", "mopex3",
    "mopex4", "mopex5", "newzealand1", "newzealand2", "penman", "plateau",
    "simhyd", "smar", "susannah1", "susannah2", "tank", "tcm", "topmodel",
    "us1", "vic", "wetland", "xinanjiang",
]

EPOCH_RE = re.compile(r"epoch[_-](\d+)", re.IGNORECASE)
MASTER_FAILED_RE = re.compile(r"FAILED\s*\[([^]]+)\]", re.IGNORECASE)
MASTER_COMPLETED_RE = re.compile(r"COMPLETED\s*\[([^]]+)\]", re.IGNORECASE)
EPOCH_LINE_RE = re.compile(
    r"Epoch\s*\[\s*(\d+)\s*/\s*(\d+)\s*\].*?"
    r"Train Loss[^:]*:\s*([-+0-9.eE]+).*?"
    r"Val KGE:\s*([-+0-9.eE]+)",
    re.IGNORECASE,
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def find_first(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name)) if root.exists() else []
    return matches[0] if matches else None


def number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def parse_log(log_path: Path, launcher_failed: bool) -> dict[str, Any]:
    if not log_path.exists():
        return {
            "log_exists": False,
            "validation_complete": False,
            "exception": "",
            "loss_plateau": False,
            "sentinel_val_kge": False,
            "nan_or_inf": False,
            "epoch_count": 0,
            "best_logged_val_kge": "",
            "last_logged_loss": "",
            "last_logged_val_kge": "",
            "failure_kind": "missing_log" if launcher_failed else "",
        }

    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    epoch_rows: list[tuple[int, float, float]] = []
    for line in lines:
        match = EPOCH_LINE_RE.search(line)
        if match:
            try:
                epoch_rows.append(
                    (int(match.group(1)), float(match.group(3)), float(match.group(4)))
                )
            except ValueError:
                pass

    # The dPL logs use -0.4142 as the invalid/empty-KGE sentinel.  A sustained
    # loss of exactly 1.0 is also a useful symptom of a broken gradient path.
    loss_plateau = False
    run = 0
    for _, loss, _ in epoch_rows:
        if loss >= 0.999:
            run += 1
            loss_plateau = loss_plateau or run >= 3
        else:
            run = 0
    sentinel_count = sum(kge <= -0.4 for _, _, kge in epoch_rows)
    nan_or_inf = bool(re.search(r"\b(?:nan|inf)\b", text, re.IGNORECASE))

    exception = ""
    for line in reversed(lines):
        if re.search(
            r"(?:OSError|RuntimeError|ValueError|KeyError|CUDA out of memory|"
            r"FileNotFoundError|Exception):",
            line,
            re.IGNORECASE,
        ):
            exception = line.strip()
            break

    validation_complete = "DMG Validation Complete" in text
    if launcher_failed and exception and "_summary" in exception:
        failure_kind = "postprocess_summary_write"
    elif exception:
        failure_kind = "training_or_validation_exception"
    elif launcher_failed:
        failure_kind = "launcher_failure_unknown"
    else:
        failure_kind = ""

    return {
        "log_exists": True,
        "validation_complete": validation_complete,
        "exception": exception,
        "loss_plateau": loss_plateau,
        "sentinel_val_kge": sentinel_count >= 2,
        "nan_or_inf": nan_or_inf,
        "epoch_count": len(epoch_rows),
        "best_logged_val_kge": number(max((x[2] for x in epoch_rows), default=None)),
        "last_logged_loss": number(epoch_rows[-1][1] if epoch_rows else None),
        "last_logged_val_kge": number(epoch_rows[-1][2] if epoch_rows else None),
        "failure_kind": failure_kind,
    }


def collect_master_status(log_dir: Path) -> tuple[set[str], set[str]]:
    failed: set[str] = set()
    completed: set[str] = set()
    for path in sorted(log_dir.glob("master*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        failed.update(m.group(1).lower() for m in MASTER_FAILED_RE.finditer(text))
        completed.update(m.group(1).lower() for m in MASTER_COMPLETED_RE.finditer(text))
    return failed, completed


def model_row(root: Path, model: str, failed: set[str], completed: set[str], logs_dir: Path) -> dict[str, Any]:
    ckpt_dir = root / "checkpoints" / "dpl" / model
    result_dir = root / "results" / "dpl" / model / "1-kge" / "seed42"
    summary_path = result_dir / "summary.json"
    summary = read_json(summary_path)
    checkpoint_paths = list(ckpt_dir.rglob("*.pt")) if ckpt_dir.exists() else []
    epoch_numbers = [
        int(match.group(1))
        for path in checkpoint_paths
        if (match := EPOCH_RE.search(path.name))
    ]
    best_path = find_first(ckpt_dir, "best.pt") if ckpt_dir.exists() else None
    log_path = logs_dir / f"dpl_{model}_100ep.log"
    log = parse_log(log_path, model in failed)

    if not ckpt_dir.exists() and not result_dir.exists():
        status = "not_run"
    elif model in failed and log["failure_kind"] == "postprocess_summary_write":
        status = "postprocess_failure_artifacts_valid"
    elif model in failed:
        status = "failed"
    elif checkpoint_paths and summary_path.exists():
        status = "complete"
    else:
        status = "incomplete"

    warnings: list[str] = []
    if log["loss_plateau"]:
        warnings.append("loss_plateau_at_1")
    if log["sentinel_val_kge"]:
        warnings.append("invalid_val_kge_sentinel")
    if log["nan_or_inf"]:
        warnings.append("nan_or_inf_in_log")
    if model in failed and log["failure_kind"] == "postprocess_summary_write":
        warnings.append("launcher_exit_after_artifacts")

    return {
        "model": model,
        "status": status,
        "launcher_failed": model in failed,
        "launcher_completed": model in completed,
        "checkpoint_count": len(checkpoint_paths),
        "last_checkpoint_epoch": max(epoch_numbers) if epoch_numbers else "",
        "best_checkpoint": bool(best_path),
        "summary_json": summary_path.exists(),
        "basin_metrics_csv": (result_dir / "final" / "basin_metrics.csv").exists(),
        "epochs_csv": (result_dir / "epochs.csv").exists(),
        "actual_epochs": summary.get("actual_epochs", ""),
        "best_epoch": summary.get("best_epoch", ""),
        "train_loss_final": summary.get("train_loss_final", ""),
        "val_kge_median_best": summary.get("val_kge_median_best", summary.get("val_kge_median", "")),
        "val_kge_mean": summary.get("val_kge_mean", ""),
        "stop_reason": summary.get("stop_reason", ""),
        "validation_complete": log["validation_complete"],
        "epoch_log_count": log["epoch_count"],
        "last_logged_loss": log["last_logged_loss"],
        "last_logged_val_kge": log["last_logged_val_kge"],
        "best_logged_val_kge": log["best_logged_val_kge"],
        "failure_kind": log["failure_kind"],
        "exception": log["exception"],
        "warnings": ";".join(warnings),
        "log_path": str(log_path),
        "summary_path": str(summary_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Remote/local benchmark root")
    parser.add_argument("--output-dir", type=Path, help="Output directory; defaults to results/dpl/_summary")
    parser.add_argument("--logs-dir", type=Path, help="Log directory; defaults to logs/dpl_pool")
    args = parser.parse_args()

    root = args.root
    output_dir = args.output_dir or root / "results" / "dpl" / "_summary"
    logs_dir = args.logs_dir or root / "logs" / "dpl_pool"
    failed, completed = collect_master_status(logs_dir)

    model_dirs = root.joinpath("checkpoints", "dpl")
    discovered = {p.name.lower() for p in model_dirs.iterdir() if p.is_dir()} if model_dirs.exists() else set()
    models = list(dict.fromkeys(EXPECTED_MODELS + sorted(discovered - set(EXPECTED_MODELS))))
    rows = [model_row(root, model, failed, completed, logs_dir) for model in models]

    fields = [
        "model", "status", "launcher_failed", "launcher_completed", "checkpoint_count",
        "last_checkpoint_epoch", "best_checkpoint", "summary_json", "basin_metrics_csv",
        "epochs_csv", "actual_epochs", "best_epoch", "train_loss_final", "val_kge_median_best",
        "val_kge_mean", "stop_reason", "validation_complete", "epoch_log_count",
        "last_logged_loss", "last_logged_val_kge", "best_logged_val_kge", "failure_kind",
        "exception", "warnings", "log_path", "summary_path",
    ]
    summary_path = output_dir / "dpl_model_summary_rebuilt.csv"
    write_csv(summary_path, rows, fields)

    audit_rows = [row for row in rows if row["status"] not in {"complete"} or row["warnings"]]
    audit_fields = [
        "model", "status", "failure_kind", "exception", "warnings", "checkpoint_count",
        "last_checkpoint_epoch", "best_checkpoint", "summary_json", "basin_metrics_csv",
        "validation_complete", "actual_epochs", "best_epoch", "train_loss_final",
        "val_kge_median_best", "last_logged_loss", "last_logged_val_kge", "log_path",
    ]
    audit_path = output_dir / "dpl_failure_audit.csv"
    write_csv(audit_path, audit_rows, audit_fields)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    report = {
        "root": str(root),
        "output_dir": str(output_dir),
        "models": len(rows),
        "master_failed_models": sorted(failed),
        "master_completed_models": sorted(completed),
        "status_counts": counts,
        "postprocess_failure_models": [
            row["model"] for row in rows if row["status"] == "postprocess_failure_artifacts_valid"
        ],
        "suspicious_training_models": [
            row["model"] for row in rows if "loss_plateau_at_1" in row["warnings"]
        ],
        "note": "No torch/GPU is used; this is a filesystem/log aggregation audit.",
    }
    report_path = output_dir / "dpl_audit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"summary={summary_path}")
    print(f"audit={audit_path}")
    print(f"report={report_path}")
    print(f"status_counts={json.dumps(counts, sort_keys=True)}")
    print(f"postprocess_failures={','.join(report['postprocess_failure_models']) or '-'}")
    print(f"suspicious_training={','.join(report['suspicious_training_models']) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
