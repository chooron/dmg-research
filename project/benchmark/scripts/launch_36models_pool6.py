"""Run the dPL model registry with a locked, constant-size worker pool."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src"), str(BENCHMARK_ROOT.parents[1])]

from src.model_registry import NPARAM_INFO_36

LOGS_DIR = BENCHMARK_ROOT / "logs" / "dpl_pool"
CHECKPOINTS_DIR = BENCHMARK_ROOT / "checkpoints" / "dpl"
RESULTS_DIR = BENCHMARK_ROOT / "results" / "dpl"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

ALL_MODELS = list(NPARAM_INFO_36.keys())


def run_constant_pool_of_6(
    models_queue: list[str],
    epochs: int = 100,
    min_epochs: int = 50,
    patience: int = 10,
    batch_size: int = 100,
    device: str = "cuda",
    max_workers: int = 6,
    skip_completed: bool = True,
) -> None:
    master_log = LOGS_DIR / f"master_pool_{epochs}ep.log"
    lock_path = LOGS_DIR / f".pool_{epochs}ep.lock"
    lock_file = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Another {epochs}-epoch master pool is already running.")
        lock_file.close()
        return

    completed_checkpoints: list[str] = []
    pending_models: list[str] = []
    for model in models_queue:
        # Completion is signalled by the canonical DONE marker
        # (results/dpl/{model}/1-kge/seed42/DONE), not by a fixed-epoch file:
        # early stopping may legitimately finish before the epoch budget.
        done = RESULTS_DIR / model / "1-kge" / "seed42" / "DONE"
        if skip_completed and done.exists():
            completed_checkpoints.append(model)
        else:
            pending_models.append(model)

    print(
        f"=== Starting Worker Pool ({max_workers} workers, batch_size={batch_size}) "
        f"for {len(pending_models)} pending models ==="
    )
    print(f"Master Log: {master_log}")

    remaining_queue = pending_models.copy()
    active_processes: list[dict] = []
    completed_models: list[str] = []
    failed_models: list[str] = []

    try:
        with open(master_log, "a", encoding="utf-8", buffering=1) as master_f:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            master_f.write(
                f"\n[{now}] Started Master Pool: total={len(models_queue)} "
                f"pending={len(pending_models)} epochs={epochs} "
                f"batch_size={batch_size} workers={max_workers}\n"
            )
            if completed_checkpoints:
                master_f.write(
                    f"[{now}] Skipped existing checkpoints: "
                    f"{', '.join(completed_checkpoints)}\n"
                )

            while remaining_queue or active_processes:
                while len(active_processes) < max_workers and remaining_queue:
                    model = remaining_queue.pop(0)
                    log_path = LOGS_DIR / f"dpl_{model}_{epochs}ep.log"
                    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
                    cmd = [
                        sys.executable,
                        "-u",
                        str(BENCHMARK_ROOT / "scripts" / "run_dpl_benchmark_dmg_native.py"),
                        "--model", model,
                        "--epochs", str(epochs),
                        "--min-epochs", str(min_epochs),
                        "--patience", str(patience),
                        "--batch_size", str(batch_size),
                        "--device", device,
                    ]
                    process = subprocess.Popen(
                        cmd,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        cwd=BENCHMARK_ROOT,
                        env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    )
                    active_processes.append({
                        "model": model,
                        "process": process,
                        "log_file": log_file,
                        "start_time": time.time(),
                    })
                    msg = (
                        f"[{time.strftime('%H:%M:%S')}] Spawned [{model.upper()}] "
                        f"PID={process.pid} pool={len(active_processes)}/{max_workers} "
                        f"queue_left={len(remaining_queue)} log={log_path.name}\n"
                    )
                    print(msg.strip())
                    master_f.write(msg)

                still_active: list[dict] = []
                for item in active_processes:
                    process = item["process"]
                    return_code = process.poll()
                    if return_code is None:
                        still_active.append(item)
                        continue

                    item["log_file"].close()
                    elapsed = time.time() - item["start_time"]
                    model = item["model"]
                    if return_code == 0:
                        completed_models.append(model)
                        msg = (
                            f"[{time.strftime('%H:%M:%S')}] COMPLETED [{model.upper()}] "
                            f"elapsed={elapsed:.1f}s total={len(completed_models)}/{len(pending_models)}\n"
                        )
                    else:
                        failed_models.append(model)
                        msg = (
                            f"[{time.strftime('%H:%M:%S')}] FAILED [{model.upper()}] "
                            f"exit_code={return_code} elapsed={elapsed:.1f}s\n"
                        )
                    print(msg.strip())
                    master_f.write(msg)

                active_processes = still_active
                time.sleep(2)

            summary = (
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pool finished: "
                f"completed={len(completed_models)} failed={len(failed_models)}\n"
            )
            print(summary.strip())
            master_f.write(summary)
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Constant worker pool for all dPL models")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs budget (early stopping may finish earlier)")
    parser.add_argument("--min-epochs", type=int, default=50, help="Early stopping never triggers before this epoch")
    parser.add_argument("--patience", type=int, default=10, help="Stop after N epochs without validation-KGE improvement (after min-epochs)")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_workers", type=int, default=6)
    parser.add_argument("--models", default=",".join(ALL_MODELS))
    parser.add_argument("--no-skip-completed", action="store_true")
    args = parser.parse_args()

    models = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    unknown = sorted(set(models) - set(NPARAM_INFO_36))
    if unknown:
        parser.error(f"Unknown model(s): {', '.join(unknown)}")

    run_constant_pool_of_6(
        models,
        epochs=args.epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
        max_workers=args.max_workers,
        skip_completed=not args.no_skip_completed,
    )


if __name__ == "__main__":
    main()
