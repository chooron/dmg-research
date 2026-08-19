"""
Parallel launcher for the native dPL runner.
Each model gets an independent process and log on the selected CUDA device.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = BENCHMARK_ROOT / "logs" / "dpl_parallel"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_ROOT = BENCHMARK_ROOT / "checkpoints" / "dpl_production_20260730"
DEFAULT_MODELS = ["flexb", "flexi", "flexis", "mopex4", "mopex5"]


def launch_parallel_models(
    models: list[str],
    epochs: int = 100,
    min_epochs: int = 50,
    patience: int = 10,
    min_delta: float = 1.0e-4,
    batch_size: int = 100,
    lr: float = 1.0e-3,
    rho: int = 730,
    warmup: int = 365,
    device: str = "cuda",
    backend: str = "compile",
    detect_anomaly: bool = False,
    resume: bool = False,
):
    print(f"=== Launching {len(models)} native dPL tasks on GPU ({device}) ===")
    print(f"Models: {models}")
    print(
        f"Max epochs: {epochs}; min epochs: {min_epochs}; patience: {patience}; "
        f"min_delta: {min_delta}; batch: {batch_size}; resume: {resume}"
    )
    processes = []
    log_files = []

    for model_name in models:
        log_path = LOGS_DIR / f"dpl_{model_name}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        log_files.append(log_file)
        cmd = [
            sys.executable,
            str(BENCHMARK_ROOT / "scripts" / "run_dpl_benchmark_dmg_native.py"),
            "--model", model_name,
            "--epochs", str(epochs),
            "--min_epochs", str(min_epochs),
            "--patience", str(patience),
            "--min_delta", str(min_delta),
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            "--rho", str(rho),
            "--warmup", str(warmup),
            "--backend", backend,
            "--device", device,
        ]
        if detect_anomaly:
            cmd.append("--detect_anomaly")
        if resume:
            cmd.extend(["--resume_checkpoint", str(CHECKPOINT_ROOT / model_name / "best.pt")])
        print(f"Starting [{model_name}] -> {log_path}", flush=True)
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=BENCHMARK_ROOT,
        )
        processes.append((model_name, process, log_path))

    print("All model processes spawned; checking startup after 5 seconds...", flush=True)
    time.sleep(5)
    for model_name, process, log_path in processes:
        return_code = process.poll()
        if return_code is not None:
            print(f"ERROR: [{model_name}] exited during startup with code {return_code}; log={log_path}")
        else:
            print(f"RUNNING: [{model_name}] pid={process.pid} log={log_path}")
    return processes, log_files


def main():
    parser = argparse.ArgumentParser(description="Launch parallel native dPL training tasks")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1.0e-4)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--rho", type=int, default=730)
    parser.add_argument("--warmup", type=int, default=365)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("compile", "eager"), default="compile")
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume each model from its best.pt")
    args = parser.parse_args()
    launch_parallel_models(
        args.models,
        epochs=args.epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        batch_size=args.batch_size,
        lr=args.lr,
        rho=args.rho,
        warmup=args.warmup,
        device=args.device,
        backend=args.backend,
        detect_anomaly=args.detect_anomaly,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
