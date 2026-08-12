"""
Parallel Launcher Script for Remote GPU Node.
Launches 6 hydrological models concurrently in parallel on CUDA.
Usage: python3 scripts/launch_parallel_dpl_remote.py --models simhyd hbv96 gr4j collie1 wetland alpine1
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = BENCHMARK_ROOT / "logs" / "dpl_parallel"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_6_MODELS = ["simhyd", "hbv96", "gr4j", "collie1", "wetland", "alpine1"]


def launch_parallel_models(models: list[str], epochs: int = 20, device: str = "cuda"):
    print(f"=== Launching {len(models)} Parallel dPL Training Tasks on GPU ({device}) ===")
    print(f"Target Models: {models}")
    print(f"Epochs per model: {epochs}")

    processes = []
    log_files = []

    for m in models:
        log_path = LOGS_DIR / f"dpl_{m}.log"
        log_f = open(log_path, "w", encoding="utf-8")
        log_files.append(log_f)

        cmd = [
            sys.executable,
            str(BENCHMARK_ROOT / "scripts" / "run_dpl_benchmark_20ep.py"),
            "--model", m,
            "--epochs", str(epochs),
            "--device", device,
        ]

        print(f"Starting [{m}] -> Log: {log_path}")
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=BENCHMARK_ROOT)
        processes.append((m, p, log_path))

    print(f"\nAll {len(models)} models spawned! Monitoring initial startup...")
    time.sleep(3)

    for m, p, log_path in processes:
        ret = p.poll()
        if ret is not None and ret != 0:
            print(f"ERROR: Process [{m}] exited prematurely with code {ret}!")
            with open(log_path) as f:
                print(f"--- Log contents for {m} ---\n{f.read()}")
        else:
            print(f"Status [{m}]: Running (PID {p.pid})")

    print("\nParallel dPL Training Launched Successfully!")


def main():
    parser = argparse.ArgumentParser(description="Launch Parallel dPL Training Tasks")
    parser.add_argument("--models", nargs="+", default=DEFAULT_6_MODELS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    launch_parallel_models(args.models, epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    main()
