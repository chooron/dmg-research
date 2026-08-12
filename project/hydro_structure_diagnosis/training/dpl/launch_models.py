#!/usr/bin/env python3
"""Launch the active unified dPL jobs with bounded concurrency."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
KNOWN_REPO_ROOTS = (REPO_ROOT,)
BASE_CONFIG = HERE / "base_config.json"
RUNNER = HERE / "run_dpl_model.py"
GENERATED = HERE / "generated_configs"
LOG_DIR = HERE / "logs"

MODELS = (
    "HBV", "GR4J", "XAJ", "GR4J_CN", "XAJ_CN", "SIMHYD", "SIMHYD_CN",
    "GR4J_PD", "XAJ_PD", "SIMHYD_PD",
    "XAJ_TGD2",
    "XAJ_2S", "XAJ_RWPE",
)


def replace_repo_root(value: object, source_root: Path, target_root: Path) -> object:
    if isinstance(value, str):
        source = str(source_root)
        return value.replace(source, str(target_root)) if value.startswith(source) else value
    if isinstance(value, dict):
        return {key: replace_repo_root(item, source_root, target_root) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_repo_root(item, source_root, target_root) for item in value]
    return value


def prepare_config(base: dict, model: str, repo_root: Path,
                   output_prefix: str) -> tuple[Path, Path]:
    config = json.loads(json.dumps(base))
    for source_root in KNOWN_REPO_ROOTS:
        config = replace_repo_root(config, source_root, repo_root)
    config["model_name"] = model
    config["output_dir"] = f"{output_prefix.rstrip('/')}/{model}"
    GENERATED.mkdir(parents=True, exist_ok=True)
    path = GENERATED / f"{model}.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path, repo_root / "project" / "hydro_structure_diagnosis" / config["output_dir"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-prefix", default="outputs/dpl_unified_365d_v1")
    parser.add_argument("--max-basins", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        parser.error("--gpus must not be empty")

    base = json.loads(BASE_CONFIG.read_text())
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    commands = []
    for index, model in enumerate(MODELS):
        config_path, output_dir = prepare_config(base, model, args.repo_root, args.output_prefix)
        if (output_dir / "COMPLETE").exists() and not args.force:
            print(f"SKIP {model} COMPLETE exists")
            continue
        command = [sys.executable, str(RUNNER), "--config", str(config_path), "--model", model]
        if args.max_basins is not None:
            command.extend(["--max-basins", str(args.max_basins)])
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus[index % len(gpus)]
        commands.append((model, command, env, output_dir))
        print(f"READY {model} GPU={env['CUDA_VISIBLE_DEVICES']}")
        print("  " + " ".join(command))

    if args.dry_run or not commands:
        return

    pending = list(commands)
    active: list[tuple[str, subprocess.Popen, object]] = []
    while pending or active:
        while pending and len(active) < args.jobs:
            model, command, env, _output_dir = pending.pop(0)
            log_path = LOG_DIR / f"{model}.log"
            log_handle = log_path.open("w")
            log_handle.write("COMMAND: " + " ".join(command) + "\n")
            log_handle.write("CUDA_VISIBLE_DEVICES: " + env["CUDA_VISIBLE_DEVICES"] + "\n\n")
            log_handle.flush()
            process = subprocess.Popen(command, cwd=PROJECT_DIR, env=env,
                                       stdout=log_handle, stderr=subprocess.STDOUT, text=True)
            active.append((model, process, log_handle))
            print(f"START {model} pid={process.pid} log={log_path}")

        remaining = []
        for model, process, log_handle in active:
            code = process.poll()
            if code is None:
                remaining.append((model, process, log_handle))
            else:
                log_handle.close()
                print(f"DONE {model} {'OK' if code == 0 else f'FAIL({code})'}")
        active = remaining
        if active:
            time.sleep(2)


if __name__ == "__main__":
    main()
