#!/usr/bin/env python3
"""R5 Production Pipeline: Serial IC (6 models) followed by Parallel dPL (6 models).

Execution order:
1. IC Batched CMA-ES on 531 basins (10 starts, 300 generations) executed strictly serially:
   - GR4J
   - GR4J_CN
   - GR4J_TGD2
   - SIMHYD
   - SIMHYD_CN
   - SIMHYD_TGD2
2. dPL training (100 epochs, batch 128, Lite mode) executed with 6 parallel processes:
   - GR4J, GR4J_CN, GR4J_TGD2, SIMHYD, SIMHYD_CN, SIMHYD_TGD2
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent

IC_RUNNER = PROJECT_DIR / "training/ic/run_tgd2_batched_cmaes_531.py"
DPL_RUNNER = PROJECT_DIR / "training/dpl/run_dpl_model.py"
DPL_BASE_CFG = PROJECT_DIR / "training/dpl/base_config_camels_531_autodl.json"
if not DPL_BASE_CFG.exists():
    DPL_BASE_CFG = PROJECT_DIR / "training/dpl/base_config_camels_531.json"

LOG_DIR = PROJECT_DIR / "logs"
STATUS_FILE = LOG_DIR / "r5_pipeline_status.json"

IC_MODELS = [
    ("GR4J", 12, 100),
    ("GR4J_CN", 12, 90),
    ("GR4J_TGD2", 12, 100),
    ("SIMHYD", 15, 85),
    ("SIMHYD_CN", 18, 75),
    ("SIMHYD_TGD2", 18, 75),
]

DPL_MODELS = ["GR4J", "GR4J_CN", "GR4J_TGD2", "SIMHYD", "SIMHYD_CN", "SIMHYD_TGD2"]


def log(msg: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with (LOG_DIR / "r5_pipeline.log").open("a") as f:
        f.write(formatted + "\n")


def update_status(data: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_file = STATUS_FILE.with_suffix(".tmp")
    with temp_file.open("w") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_file, STATUS_FILE)


def run_ic_phase(args: argparse.Namespace) -> None:
    log("=" * 60)
    log("PHASE 1: STARTING SERIAL IC BATCHED CMA-ES CALIBRATIONS")
    log("=" * 60)

    for index, (model, pop, chunk) in enumerate(IC_MODELS):
        out_dir = PROJECT_DIR / "results" / f"{model.lower()}_cmaes_531_batched_v1"
        log(f"--- IC Task {index + 1}/{len(IC_MODELS)}: {model} ---")
        log(f"Output directory: {out_dir}")

        if (out_dir / "DONE.json").exists() and not getattr(args, "force", False):
            log(f"IC Task {model} ALREADY COMPLETE (DONE.json exists), skipping to next task.")
            continue

        cmd = [
            sys.executable, str(IC_RUNNER),
            "--model", model,
            "--output", str(out_dir),
            "--starts", str(args.ic_starts),
            "--population", str(pop),
            "--generations", str(args.ic_generations),
            "--chunk-basins", str(chunk),
            "--checkpoint-interval", str(args.ic_checkpoint_interval),
            "--device", args.device,
        ]

        status_data = {
            "current_phase": "IC",
            "current_model": model,
            "ic_model_index": index + 1,
            "ic_total_models": len(IC_MODELS),
            "command": " ".join(cmd),
            "start_time": datetime.datetime.now().isoformat(),
            "status": "running",
        }
        update_status(status_data)

        log(f"Executing: {' '.join(cmd)}")
        ic_log_file = LOG_DIR / f"ic_{model.lower()}.log"
        with ic_log_file.open("a") as log_f:
            log_f.write(f"=== STARTING {model} AT {datetime.datetime.now().isoformat()} ===\n")
            log_f.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_DIR),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )

        if proc.returncode != 0:
            log(f"ERROR: IC Task {model} failed with exit code {proc.returncode}!")
            status_data["status"] = "failed"
            status_data["failed_model"] = model
            status_data["error_code"] = proc.returncode
            update_status(status_data)
            sys.exit(proc.returncode)

        log(f"IC Task {model} COMPLETED SUCCESSFULLY.")

    log("=" * 60)
    log("PHASE 1 COMPLETE: ALL 6 IC TASKS FINISHED SUCCESSFULLY.")
    log("=" * 60)


def run_dpl_phase(args: argparse.Namespace) -> None:
    log("=" * 60)
    log("PHASE 2: STARTING 6 PARALLEL dPL TRAINING PROCESSES")
    log("=" * 60)

    base_cfg = json.loads(DPL_BASE_CFG.read_text())
    base_cfg["runtime"]["device"] = args.device
    base_cfg["training"]["epochs"] = args.dpl_epochs
    base_cfg["training"]["batch_size"] = args.dpl_batch_size
    base_cfg["training"]["seed"] = args.dpl_seed

    active_procs: list[dict] = []
    dpl_cfg_dir = PROJECT_DIR / "training/dpl/generated_configs"
    dpl_cfg_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })

    for model in DPL_MODELS:
        cfg = json.loads(json.dumps(base_cfg))
        out_dir = PROJECT_DIR / "results" / f"dpl_camels_531_lite_v3" / model / f"seed_{args.dpl_seed}"
        cfg["model_name"] = model
        cfg["output_dir"] = str(out_dir.relative_to(PROJECT_DIR) if out_dir.is_relative_to(PROJECT_DIR) else out_dir)
        cfg["lite_mode"] = True
        if model.endswith("_TGD2"):
            cfg["network"]["parameter_mapping"] = "sigmoid_to_physical_range_with_log_tgd2_residence_times"
            cfg["network"]["tgd_structure_version"] = "temperature_dependent_generic_delay2_v1"

        cfg_path = dpl_cfg_dir / f"r5_{model.lower()}_seed_{args.dpl_seed}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

        cmd = [
            sys.executable, str(DPL_RUNNER),
            "--model", model,
            "--config", str(cfg_path),
            "--output", str(out_dir),
            "--lite",
            "--resume",
        ]

        log_path = LOG_DIR / f"dpl_{model.lower()}_seed_{args.dpl_seed}.log"
        log_handle = log_path.open("a")
        log_handle.write(f"=== LAUNCHING dPL {model} AT {datetime.datetime.now().isoformat()} ===\n")
        log_handle.write(f"COMMAND: {' '.join(cmd)}\n")
        log_handle.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_DIR),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

        active_procs.append({
            "model": model,
            "pid": proc.pid,
            "proc": proc,
            "log_file": str(log_path),
            "log_handle": log_handle,
            "output_dir": str(out_dir),
            "start_time": datetime.datetime.now().isoformat(),
        })
        log(f"Launched dPL Process: model={model:<15} PID={proc.pid} log={log_path}")

    update_status({
        "current_phase": "dPL",
        "dpl_processes": [
            {"model": p["model"], "pid": p["pid"], "log_file": p["log_file"], "output_dir": p["output_dir"]}
            for p in active_procs
        ],
        "start_time": datetime.datetime.now().isoformat(),
        "status": "running",
    })

    log("=" * 60)
    log(f"ALL {len(active_procs)} dPL PROCESSES RUNNING CONCURRENTLY.")
    log("=" * 60)

    # Monitor dPL processes
    while active_procs:
        time.sleep(10)
        still_running = []
        for item in active_procs:
            ret = item["proc"].poll()
            if ret is None:
                still_running.append(item)
            else:
                item["log_handle"].close()
                log(f"dPL Process Finished: model={item['model']} PID={item['pid']} returncode={ret}")
                if ret != 0:
                    log(f"WARNING: dPL model={item['model']} failed with code {ret}! Check {item['log_file']}")
        active_procs = still_running

    log("PHASE 2 COMPLETE: ALL dPL PROCESSES FINISHED.")
    update_status({
        "current_phase": "COMPLETED",
        "status": "success",
        "end_time": datetime.datetime.now().isoformat(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ic-starts", type=int, default=10)
    parser.add_argument("--ic-generations", type=int, default=300)
    parser.add_argument("--ic-checkpoint-interval", type=int, default=5)
    parser.add_argument("--dpl-epochs", type=int, default=100)
    parser.add_argument("--dpl-batch-size", type=int, default=128)
    parser.add_argument("--dpl-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-ic", action="store_true", help="Skip IC phase if already completed")
    parser.add_argument("--force", action="store_true", help="Force rerun even if DONE.json exists")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Starting R5 Production Pipeline on device={args.device}")

    if not args.skip_ic:
        run_ic_phase(args)

    run_dpl_phase(args)
    log("R5 PRODUCTION PIPELINE EXECUTION COMPLETED.")


if __name__ == "__main__":
    main()
