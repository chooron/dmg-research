#!/usr/bin/env python3
"""Resilient 2x2 Factorial Runner for Flex-MOPEX SSH experiment.

Executes 12 formal runs (Seeds 42, 43, 44 x Conditions E1, E2, E3, E4) with
controlled concurrency (max 2 parallel on 12GB RTX 3080 Ti), streams logs,
updates TSV status table, and triggers analysis upon completion.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent
RESULT_ROOT = PROJECT_DIR / "results/ssh_2x2"
LOG_ROOT = RESULT_ROOT / "logs"
STATUS_FILE = RESULT_ROOT / "matrix_status.tsv"

CONDITIONS = ["E1", "E2", "E3", "E4"]
SEEDS = [42, 43, 44]


def write_status(rows: list[dict]) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        f.write("condition\tseed\tgpu\tstatus\tpid\tconfig\toutput\tlog\tstarted\tended\n")
        for r in rows:
            f.write(
                f"{r['condition']}\t{r['seed']}\t{r['gpu']}\t{r['status']}\t{r.get('pid','')}\t"
                f"{r['config']}\t{r['output']}\t{r['log']}\t{r.get('started','')}\t{r.get('ended','')}\n"
            )


def run_single(item: tuple[str, int, int, str]) -> dict:
    condition, seed, gpu_id, python_bin = item
    config_path = PROJECT_DIR / f"conf/ssh_2x2/config_{condition}_pure_x35_531_lambda0007.yaml"
    output_dir = RESULT_ROOT / condition / f"seed_{seed}"
    log_file = LOG_ROOT / f"{condition}_seed{seed}.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cmd = [
        python_bin,
        str(PROJECT_DIR / "run_model.py"),
        "--config", str(config_path),
        "--mode", "train_test",
        "--seed", str(seed),
        "--gpu-id", str(gpu_id),
        "--epochs", "10",
        "--test-epoch", "10",
        "--disable-early-stopping",
        "--output-root", str(RESULT_ROOT),
        "--run-name", f"{condition}/seed_{seed}",
        "--verbose",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["FLEXMOPEX_DATA_DIR"] = env.get("FLEXMOPEX_DATA_DIR", "/root/autodl-fs/data")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TORCH_COMPILE_DEBUG"] = "0"

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] START {condition} seed={seed} -> {log_file}", flush=True)

    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(REPO_ROOT),
        )
        pid = proc.pid
        rc = proc.wait()

    ended = datetime.datetime.now(datetime.timezone.utc).isoformat()
    status = "COMPLETED" if rc == 0 else "FAILED"
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] END {condition} seed={seed} status={status} (rc={rc})", flush=True)

    return {
        "condition": condition,
        "seed": seed,
        "gpu": gpu_id,
        "status": status,
        "pid": pid,
        "config": str(config_path),
        "output": str(output_dir),
        "log": str(log_file),
        "started": started,
        "ended": ended,
        "rc": rc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=2, help="Number of parallel runs on GPU")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--python-bin", default="/root/miniconda3/bin/python")
    args = parser.parse_args()

    python_bin = args.python_bin
    if not os.path.exists(python_bin):
        python_bin = sys.executable

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    # All 12 experiment items
    tasks = []
    for seed in SEEDS:
        for cond in CONDITIONS:
            tasks.append((cond, seed, args.gpu_id, python_bin))

    print(f"=== Starting Flex-MOPEX 2x2 Factorial Matrix ===")
    print(f"Total tasks: {len(tasks)} | Concurrency: {args.concurrency} | GPU: {args.gpu_id}")
    print(f"Result root: {RESULT_ROOT}")

    # Initial status
    status_rows = [
        {
            "condition": t[0],
            "seed": t[1],
            "gpu": t[2],
            "status": "PENDING",
            "config": str(PROJECT_DIR / f"conf/ssh_2x2/config_{t[0]}_pure_x35_531_lambda0007.yaml"),
            "output": str(RESULT_ROOT / t[0] / f"seed_{t[1]}"),
            "log": str(LOG_ROOT / f"{t[0]}_seed{t[1]}.log"),
        }
        for t in tasks
    ]
    write_status(status_rows)

    results = []
    start_all = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(run_single, t): t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            # Update status file
            for row in status_rows:
                if row["condition"] == res["condition"] and row["seed"] == res["seed"]:
                    row.update(res)
            write_status(status_rows)

    elapsed = time.time() - start_all
    print(f"\n=== All 12 runs completed in {elapsed/60:.2f} minutes ===")

    # Check for any failures
    n_failed = sum(1 for r in results if r["status"] != "COMPLETED")
    if n_failed > 0:
        print(f"WARNING: {n_failed} runs failed! Check logs in {LOG_ROOT}")
    else:
        print("All 12 runs COMPLETED successfully!")

    # Run analysis script
    print("\n=== Running analyze_ssh_2x2_matrix.py ===")
    analysis_cmd = [
        python_bin,
        str(PROJECT_DIR / "scripts/analyze_ssh_2x2_matrix.py"),
        "--root", str(RESULT_ROOT),
        "--gpu-id", str(args.gpu_id),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["FLEXMOPEX_DATA_DIR"] = env.get("FLEXMOPEX_DATA_DIR", "/root/autodl-fs/data")
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    proc = subprocess.run(analysis_cmd, env=env, cwd=str(REPO_ROOT))
    if proc.returncode == 0:
        print("Analysis completed successfully!")
    else:
        print(f"Analysis failed with returncode {proc.returncode}")


if __name__ == "__main__":
    main()
