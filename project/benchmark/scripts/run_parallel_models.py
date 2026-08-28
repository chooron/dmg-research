#!/usr/bin/env python3
"""Run benchmark models in a bounded process pool.

Each model owns one independent CUDA process.  Completed models are skipped by
checking their DONE marker, and a newly free worker is immediately filled from
the pending model queue.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).with_name("run_36model_benchmark.py")
sys.path[:0] = [str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src")]

from src.model_registry import NPARAM_INFO_36  # noqa: E402


def _done_path(run_id: str, model_name: str) -> Path:
    return BENCHMARK_ROOT / "checkpoints" / run_id / model_name / "DONE"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded parallel 36-model benchmark runner")
    parser.add_argument("--run-id", required=True, help="Shared checkpoint run identifier")
    parser.add_argument(
        "--config",
        default="configs/full_run_10starts_300gen_dpl_aligned_1980_1995.yaml",
        help="Benchmark config relative to project/benchmark or an absolute path",
    )
    parser.add_argument("--workers", type=int, default=4, help="Maximum concurrent model processes (1-4)")
    parser.add_argument("--backend", choices=["compile"], default="compile")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names, or 'all' (default)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Per-model log directory; defaults to project/benchmark/logs/<run-id>",
    )
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser


def _start_model(
    model_name: str,
    args: argparse.Namespace,
    log_dir: Path,
    env: dict[str, str],
) -> tuple[subprocess.Popen[bytes], object, Path]:
    log_path = log_dir / f"{model_name}.log"
    log_handle = log_path.open("ab", buffering=0)
    config = args.config if os.path.isabs(args.config) else args.config
    command = [
        sys.executable,
        str(RUNNER),
        "--model",
        model_name,
        "--run-id",
        args.run_id,
        "--config",
        config,
        "--backend",
        args.backend,
        "--device",
        args.device,
    ]
    process = subprocess.Popen(
        command,
        cwd=BENCHMARK_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log_handle, log_path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be between 1 and 4")
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")

    requested = list(NPARAM_INFO_36) if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in requested if m not in NPARAM_INFO_36]
    if unknown:
        raise SystemExit(f"Unknown model(s): {', '.join(unknown)}")

    log_dir = Path(args.log_dir) if args.log_dir else BENCHMARK_ROOT / "logs" / args.run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            # Four CUDA workers should not multiply host-side BLAS/OpenMP threads.
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )

    pending = deque(requested)
    running: dict[str, tuple[subprocess.Popen[bytes], object, Path]] = {}
    failures: dict[str, int] = {}

    print(f"=== Parallel benchmark [{args.run_id}] workers={args.workers} models={len(requested)} ===", flush=True)
    while pending or running:
        while pending and len(running) < args.workers:
            model_name = pending.popleft()
            if _done_path(args.run_id, model_name).is_file():
                print(f"[{model_name}] DONE exists; skipping", flush=True)
                continue
            process, log_handle, log_path = _start_model(model_name, args, log_dir, env)
            running[model_name] = (process, log_handle, log_path)
            print(f"[{model_name}] started pid={process.pid} log={log_path}", flush=True)

        finished = []
        for model_name, (process, log_handle, log_path) in running.items():
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            finished.append(model_name)
            if return_code == 0:
                print(f"[{model_name}] completed", flush=True)
            else:
                failures[model_name] = return_code
                print(f"[{model_name}] failed returncode={return_code} log={log_path}", flush=True)
        for model_name in finished:
            del running[model_name]

        if running and not finished:
            time.sleep(args.poll_seconds)

    if failures:
        print(f"=== Parallel benchmark failed models: {failures} ===", flush=True)
        return 1
    print("=== Parallel benchmark complete ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
