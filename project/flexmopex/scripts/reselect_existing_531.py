"""Re-select a formal 531 checkpoint from an existing 100-epoch run.

This never mutates the source run. It uses the logged training objective only,
then evaluates the selected copied checkpoint into a separate output directory.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent


def select_epoch(log_path: Path, min_epochs: int, patience: int, min_delta: float) -> tuple[int, dict]:
    rows: list[tuple[int, float]] = []
    patterns = [
        re.compile(r"(?:R15 )?Epoch\s+(\d+)[^|]*\|.*?(?:Loss_total=|loss=)([-+0-9.eE]+)"),
        re.compile(r"Epoch\s+(\d+)[^|]*?loss=([-+0-9.eE]+)"),
    ]
    for line in log_path.read_text(errors="replace").splitlines():
        match = next((pattern.search(line) for pattern in patterns if pattern.search(line)), None)
        if match:
            rows.append((int(match.group(1)), float(match.group(2))))
    if not rows:
        raise ValueError(f"No train-loss rows found in {log_path}")

    best_value: float | None = None
    best_epoch: int | None = None
    wait = 0
    stop_epoch = rows[-1][0]
    reason = "max_epochs_reached"
    history = []
    for epoch, value in rows:
        eligible = epoch >= min_epochs
        improved = False
        if eligible:
            if best_value is None or value < best_value - min_delta:
                best_value, best_epoch, wait = value, epoch, 0
                improved = True
            else:
                wait += 1
        history.append(
            {
                "epoch": epoch,
                "monitor": "train_loss",
                "value": value,
                "eligible": eligible,
                "improved": improved,
                "best_value": best_value,
                "best_epoch": best_epoch,
                "wait_count": wait,
            }
        )
        if eligible and best_epoch is not None and wait >= patience:
            stop_epoch = epoch
            reason = "patience_exhausted"
            break
    if best_epoch is None:
        raise ValueError("No checkpoint at or after min_epochs")
    return best_epoch, {
        "monitor": "train_loss",
        "mode": "min",
        "min_epochs": min_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "early_stop_reason": reason,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-file", required=True, help="Epoch checkpoint filename prefix, e.g. learnedweightmopexe")
    parser.add_argument("--output-root", type=Path, default=PROJECT_DIR / "results" / "formal_531_reused")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    source_run = args.source_run.resolve()
    log_path = args.log.resolve()
    output_dir = (args.output_root / args.run_name).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_epoch, metadata = select_epoch(log_path, args.min_epochs, args.patience, args.min_delta)
    source_model = source_run / "model" / f"{args.model_file}_ep{best_epoch}.pt"
    if not source_model.exists():
        raise FileNotFoundError(source_model)
    shutil.copy2(source_model, model_dir / source_model.name)
    normalization = source_run / "model" / "normalization_statistics.json"
    if normalization.exists():
        shutil.copy2(normalization, model_dir / normalization.name)

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            str(PROJECT_DIR / "run_model.py"),
            "--config",
            str(config_path),
            "--mode",
            "test",
            "--test-epoch",
            str(best_epoch),
            "--gpu-id",
            str(args.gpu_id),
            "--output-root",
            str(args.output_root),
            "--run-name",
            args.run_name,
        ],
        cwd=PROJECT_DIR,
        check=True,
    )
    (output_dir / "early_stopping.json").write_text(
        json.dumps(
            {
                "protocol": "reselected_existing_100_epoch_run",
                "source_run": str(source_run),
                "source_log": str(log_path),
                "selected_checkpoint": "best_checkpoint.pt",
                **metadata,
            },
            indent=2,
        )
        + "\n"
    )
    shutil.copy2(model_dir / source_model.name, model_dir / "best_checkpoint.pt")
    shutil.copy2(model_dir / source_model.name, model_dir / "final_checkpoint.pt")
    shutil.copy2(model_dir / source_model.name, model_dir / "last_checkpoint.pt")
    print(json.dumps({"output": str(output_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
