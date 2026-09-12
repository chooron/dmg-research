#!/usr/bin/env python3
"""Run the frozen PRIMARY-8 x 5-fold OOB queue with four GPU workers."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from oob_common import (
    EXPECTED_BASINS,
    N_FOLDS,
    PRIMARY8,
    fold_ids,
    load_ids,
    load_oob_config,
    read_fold_assignment,
    resolve_repo_path,
    validate_fold_contract,
    validate_oob_protocol,
    write_json,
)


RUNNER = Path(__file__).resolve().with_name("run_oob_job.py")


@dataclass
class RunningJob:
    model: str
    fold: int
    process: subprocess.Popen
    stdout_handle: Any
    stderr_handle: Any
    attempt: int
    device: str
    slot: int

    @property
    def job_id(self) -> str:
        return f"{self.model}_fold{self.fold}"


def job_list() -> list[tuple[str, int]]:
    # Model-interleaved ordering: all model fold-0 jobs first, then fold-1, ...
    return [(model, fold) for fold in range(N_FOLDS) for model in PRIMARY8]


def status_for(run_dir: Path) -> str:
    if (run_dir / "DONE").exists():
        return "DONE"
    if (run_dir / "FAILED").exists():
        return "FAILED"
    if (run_dir / ".lock").exists():
        return "RUNNING"
    return "QUEUED"


def recover_stale_lock(lock_path: Path, exited_pid: int) -> bool:
    """Remove only a lock proven to belong to the exited child.

    A queue restart must never delete a live worker's lock and launch a second
    copy of the same job.  Malformed or live-owner locks are left untouched.
    """
    if not lock_path.exists():
        return True
    try:
        owner_line = next(line for line in lock_path.read_text(encoding="utf-8").splitlines() if line.startswith("pid="))
        owner_pid = int(owner_line.split("=", 1)[1])
    except (OSError, StopIteration, ValueError):
        return False
    if owner_pid != exited_pid:
        try:
            os.kill(owner_pid, 0)
        except ProcessLookupError:
            pass
        except PermissionError:
            return False
        else:
            return False
    lock_path.unlink(missing_ok=True)
    return True


def resolve_devices(raw: str | None, concurrency: int) -> list[str]:
    """Resolve four worker slots, allowing deliberate GPU slot sharing.

    Four processes may share one physical GPU.  Each child is still pinned to
    ``cuda:0`` inside its own ``CUDA_VISIBLE_DEVICES`` namespace, so duplicate
    physical IDs are intentional rather than a configuration error.
    """
    if raw:
        devices = [value.strip() for value in raw.split(",") if value.strip()]
        if len(devices) != concurrency:
            raise ValueError(f"--devices must provide exactly {concurrency} devices")
        if any(not (value.startswith("cuda:") or value.isdigit()) for value in devices):
            raise ValueError("formal OOB queue accepts CUDA devices only")
        normalized = [value if value.startswith("cuda:") else f"cuda:{value}" for value in devices]
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("formal OOB queue requires CUDA; no GPU is available")
        if torch.cuda.device_count() < 1:
            raise RuntimeError("formal OOB queue requires at least one GPU")
        normalized = ["cuda:0"] * concurrency
    available = torch.cuda.device_count()
    for device in normalized:
        physical_index = int(device.split(":", 1)[1])
        if physical_index >= available:
            raise ValueError(f"requested {device}, but only {available} physical GPU(s) are available")
    return normalized

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="oob_primary8_5fold_20260902.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--fold-assignment", default=None)
    parser.add_argument("--data-cache-root", default=None)
    parser.add_argument("--devices", default=None, help="Comma-separated CUDA devices for four slots; repeated IDs share one GPU")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.concurrency != 4:
        raise SystemExit("formal OOB concurrency is frozen at 4")

    config = load_oob_config(args.config)
    validate_oob_protocol(config)
    output_root = resolve_repo_path(args.out or config["outputs"]["root"])
    assignment_path = resolve_repo_path(args.fold_assignment or config["folds"]["assignment_file"])
    data_cache_root = resolve_repo_path(args.data_cache_root or config["outputs"]["data_cache_root"])
    ids = load_ids(config["data"]["basin_ids"])
    rows = read_fold_assignment(assignment_path)
    contract = validate_fold_contract(
        ids,
        rows,
        n_folds=N_FOLDS,
        expected_count=int(config["folds"].get("expected_basin_count", EXPECTED_BASINS)),
    )
    if args.dry_run:
        devices = (
            [value.strip() for value in args.devices.split(",") if value.strip()]
            if args.devices else ["cuda:0"] * args.concurrency
        )
    else:
        devices = resolve_devices(args.devices, args.concurrency)
    jobs = job_list()
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "queue_contract.json",
        {
            "experiment_id": config["experiment_id"],
            "jobs_total": len(jobs),
            "models": list(PRIMARY8),
            "folds": N_FOLDS,
            "concurrency": args.concurrency,
            "devices": devices,
            "distinct_physical_devices": len(set(devices)),
            "retry_limit": int(config["outputs"].get("retry_limit", 1)),
            "fold_sizes": list(contract.fold_sizes),
            "assignment": str(assignment_path),
        },
    )
    if args.dry_run:
        for index, (model, fold) in enumerate(jobs, 1):
            print(f"{index:02d}/40 {model}_fold{fold}")
        print(f"DRY_RUN jobs=40 concurrency=4 devices={','.join(devices)}")
        return

    retry_limit = int(config["outputs"].get("retry_limit", 1))
    pending = list(jobs)
    attempts: dict[tuple[str, int], int] = {}
    running: list[RunningJob] = []
    completed: set[tuple[str, int]] = set()
    failed: set[tuple[str, int]] = set()
    free_slots = list(range(args.concurrency))

    def launch(model: str, fold: int, slot: int) -> RunningJob:
        run_dir = output_root / "runs" / model / f"fold_{fold}"
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_handle = (run_dir / "stdout.log").open("a", encoding="utf-8")
        stderr_handle = (run_dir / "stderr.log").open("a", encoding="utf-8")
        attempt = attempts.get((model, fold), 0) + 1
        attempts[(model, fold)] = attempt
        physical_device = devices[slot]
        process_device = "cuda:0"
        command = [
            sys.executable,
            str(RUNNER),
            "--model",
            model,
            "--fold",
            str(fold),
            "--config",
            str(args.config),
            "--out",
            str(output_root),
            "--fold-assignment",
            str(assignment_path),
            "--data-cache-root",
            str(data_cache_root),
            "--device",
            process_device,
        ]
        (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = physical_device.split(":", 1)[1]
        process = subprocess.Popen(command, stdout=stdout_handle, stderr=stderr_handle, env=env)
        print(f"[{time.strftime('%H:%M:%S')}] START {model}_fold{fold} attempt={attempt} physical={physical_device} pid={process.pid}")
        return RunningJob(model, fold, process, stdout_handle, stderr_handle, attempt, physical_device, slot)

    while pending or running:
        while pending and free_slots:
            candidate = pending.pop(0)
            model, fold = candidate
            run_dir = output_root / "runs" / model / f"fold_{fold}"
            if (run_dir / "DONE").exists():
                completed.add(candidate)
                continue
            slot = free_slots.pop(0)
            running.append(launch(model, fold, slot))
        if not running:
            continue
        time.sleep(max(args.poll_seconds, 0.2))
        survivors: list[RunningJob] = []
        for job in running:
            return_code = job.process.poll()
            if return_code is None:
                survivors.append(job)
                continue
            job.stdout_handle.close()
            job.stderr_handle.close()
            free_slots.append(job.slot)
            free_slots.sort()
            candidate = (job.model, job.fold)
            stale_lock = output_root / "runs" / job.model / f"fold_{job.fold}" / ".lock"
            if not recover_stale_lock(stale_lock, job.process.pid):
                failed.add((job.model, job.fold))
                print(f"[{time.strftime('%H:%M:%S')}] FAILED {job.job_id}: live or malformed lock retained")
                continue
            if return_code == 0 and (output_root / "runs" / job.model / f"fold_{job.fold}" / "DONE").exists():
                completed.add(candidate)
                print(f"[{time.strftime('%H:%M:%S')}] DONE {job.job_id} ({len(completed)}/40)")
            elif attempts[candidate] <= retry_limit:
                pending.append(candidate)
                print(f"[{time.strftime('%H:%M:%S')}] RETRY {job.job_id} after exit={return_code}")
            else:
                failed.add(candidate)
                print(f"[{time.strftime('%H:%M:%S')}] FAILED {job.job_id} exit={return_code}")
        running = survivors
        write_json(
            output_root / "queue_status.json",
            {
                "completed": len(completed),
                "running": [job.job_id for job in running],
                "queued": len(pending),
                "failed": len(failed),
                "failed_jobs": [f"{model}_fold{fold}" for model, fold in sorted(failed)],
            },
        )

    print(f"QUEUE_STOP completed={len(completed)} failed={len(failed)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
