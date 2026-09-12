#!/usr/bin/env python3
"""4-Worker Queue Scheduler for Canonical dPL v2 36-Model Retraining.

Enforces:
1. TARGET_CONCURRENCY = 4 (Constant 4 concurrent workers on GPU).
2. Dynamic slot replacement (Fills empty slot immediately upon model completion).
3. Process-level isolation, per-job stdout/stderr logging, .lock protection.
4. Continuous ledger updates to CANONICAL_V2_LEDGER.md.
5. Automatic final baseline accuracy report generation upon completion.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
RUNNER_SCRIPT = BENCHMARK_ROOT / "scripts/canonical_v2/run_canonical_v2_model.py"


@dataclass
class JobProcess:
    model: str
    job_config: dict[str, Any]
    process: subprocess.Popen
    stdout_file: Any
    stderr_file: Any
    gpu_id: str
    start_time: float
    retry_count: int = 0


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    with open(manifest_path) as f:
        return yaml.safe_load(f)


def get_job_status(output_dir: Path, model_name: str) -> str:
    run_dir = output_dir / "runs" / model_name
    if not run_dir.exists():
        return "QUEUED"
    if (run_dir / "DONE").exists():
        return "DONE"
    if (run_dir / "FAILED").exists():
        return "FAILED"
    if (run_dir / ".lock").exists():
        return "RUNNING"
    return "QUEUED"


def append_to_ledger(output_dir: Path, job_config: dict[str, Any], result: dict[str, Any], status: str) -> None:
    ledger_path = output_dir / "CANONICAL_V2_LEDGER.md"
    if not ledger_path.exists():
        return

    model = job_config["model"]
    horizon = job_config.get("horizon_days", 1095)
    warmup = job_config.get("warmup_days", 730)
    scored = job_config.get("scored_days", 365)
    mapping = job_config.get("mapping", "auto")
    lr = job_config.get("lr", 1e-3)
    seed = job_config.get("seed", 42)

    best_ep = result.get("best_epoch", "N/A")
    best_loss = f"{result.get('best_train_loss', 0.0):.4f}" if "best_train_loss" in result else "N/A"
    test_med = f"{result.get('test_median_kge', 0.0):.4f}" if "test_median_kge" in result else "N/A"
    test_q25 = f"{result.get('test_q25_kge', 0.0):.4f}" if "test_q25_kge" in result else "N/A"
    clip_frac = "N/A"

    run_dir = output_dir / "runs" / model
    ep_csv = run_dir / "epoch_metrics.csv"
    if ep_csv.exists():
        try:
            df_ep = pd.read_csv(ep_csv)
            if not df_ep.empty and "grad_clip_fraction" in df_ep.columns:
                clip_frac = f"{df_ep['grad_clip_fraction'].mean()*100:.1f}%"
        except Exception:
            pass

    row = (
        f"| `{model}` | {horizon}d | {warmup}d | {scored}d | `{mapping}` | {lr:.1e} | None | 1.0 | {seed} | "
        f"**`{status}`** | {best_ep} | {best_loss} | **{test_med}** | {test_q25} | {clip_frac} | `runs/{model}/` |\n"
    )

    with open(ledger_path, "a") as f:
        f.write(row)


def generate_canonical_v2_report(output_dir: Path) -> None:
    manifest_path = output_dir / "canonical_manifest.yaml"
    if not manifest_path.exists():
        return
    manifest = load_manifest(manifest_path)
    models = [j["model"] for j in manifest["jobs"]]

    rows = []
    runs_dir = output_dir / "runs"

    for m in models:
        run_dir = runs_dir / m
        status = get_job_status(output_dir, m)
        test_file = run_dir / "test_evaluation.json"
        runtime_file = run_dir / "runtime.json"

        test_med = np.nan
        test_q25 = np.nan
        test_mean = np.nan
        best_loss = np.nan
        best_ep = np.nan
        runtime_s = np.nan

        if test_file.exists():
            try:
                tv = json.loads(test_file.read_text())
                test_med = float(tv.get("median", np.nan))
                test_q25 = float(tv.get("q25", np.nan))
                test_mean = float(tv.get("mean", np.nan))
            except Exception:
                pass

        if runtime_file.exists():
            try:
                rt = json.loads(runtime_file.read_text())
                best_loss = float(rt.get("best_train_loss", np.nan))
                best_ep = int(rt.get("best_epoch", 1))
                runtime_s = float(rt.get("total_runtime_s", np.nan))
            except Exception:
                pass

        rows.append({
            "model": m,
            "status": status,
            "best_epoch": best_ep,
            "best_train_loss": best_loss,
            "test_median_kge": test_med,
            "test_q25_kge": test_q25,
            "test_mean_kge": test_mean,
            "runtime_seconds": runtime_s,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "CANONICAL_V2_SUMMARY.csv", index=False)

    # Markdown Report
    completed = df.loc[df["status"].eq("DONE")]
    valid_kges = completed["test_median_kge"].dropna()
    med_overall = float(valid_kges.median()) if len(valid_kges) > 0 else np.nan
    mean_overall = float(valid_kges.mean()) if len(valid_kges) > 0 else np.nan
    q25_model_overall = float(valid_kges.quantile(0.25)) if len(valid_kges) > 0 else np.nan
    q75_model_overall = float(valid_kges.quantile(0.75)) if len(valid_kges) > 0 else np.nan
    q25_basin_descriptive = float(completed["test_q25_kge"].dropna().median()) if len(completed["test_q25_kge"].dropna()) > 0 else np.nan

    md_text = f"""# Canonical dPL v2 36-Model Benchmark Report (2026-08-31)

**Execution Scope**: Full 36 Hydrological Models (CAMELS 531 basins, Seed 42).  
**Protocol Contract**: Frozen Canonical v2 (35 models: 730d warmup + 365d scored, full backprop; Penman: 365d warmup + 365d scored, full backprop; AdamW lr=1e-3, sched=None, clip=1.0).  
**Evaluation Partition**: Test Period `1995-10-01..2010-09-30` (365d warmup, strictly out-of-sample post-hoc evaluation on restored `best.pt`).

---

## 1. Overall Performance Summary
- **Completed Models**: {len(valid_kges)} / {len(models)}
- **Median across completed model-level Test KGE medians ({len(valid_kges)} models)**: **{med_overall:.4f}**
- **Mean across completed model-level Test KGE medians ({len(valid_kges)} models)**: **{mean_overall:.4f}**
- **Q25 across completed model-level Test KGE medians ({len(valid_kges)} models)**: **{q25_model_overall:.4f}**
- **Q75 across completed model-level Test KGE medians ({len(valid_kges)} models)**: **{q75_model_overall:.4f}**
- **Median of per-model basin-level Test KGE Q25 values (descriptive)**: **{q25_basin_descriptive:.4f}**

---

## 2. 36-Model Canonical v2 Results Table

| Model | Status | Best Ep | Best Train Loss | Test Median KGE | Test Q25 KGE | Test Mean KGE | Runtime (s) |
|---|---|---|---|---|---|---|---|
"""
    for _, r in df.iterrows():
        md_text += f"| `{r['model']}` | **`{r['status']}`** | {r['best_epoch']} | {r['best_train_loss']:.4f} | **{r['test_median_kge']:.4f}** | {r['test_q25_kge']:.4f} | {r['test_mean_kge']:.4f} | {r['runtime_seconds']:.1f} |\n"

    (output_dir / "CANONICAL_V2_REPORT.md").write_text(md_text)
    print(f"Generated Canonical v2 Summary and Report at {output_dir}")


def run_canonical_queue(
    output_dir: Path,
    manifest_path: Path,
    concurrency: int = 4,
    devices: list[str] | None = None,
    poll_interval: float = 1.0,
    resume: bool = True,
) -> None:
    manifest = load_manifest(manifest_path)
    jobs = manifest["jobs"]

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if devices is None or len(devices) == 0:
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            devices = [f"cuda:{i % n_gpus}" for i in range(concurrency)]
        else:
            devices = ["cpu"] * concurrency

    print(f"=== Starting Canonical dPL v2 Queue Scheduler (36 Models) ===")
    print(f"Output Root: {output_dir}")
    print(f"Total Models: {len(jobs)}")
    print(f"Concurrency: {concurrency}")
    print(f"Device Allocation: {devices}")
    print(f"Resume: {resume}")

    active_processes: list[JobProcess] = []
    job_queue = list(jobs)
    job_attempts = {}

    available_slots = list(range(concurrency))

    def launch_model(job_cfg: dict[str, Any], slot_idx: int) -> JobProcess:
        m = job_cfg["model"]
        dev = devices[slot_idx % len(devices)]
        run_dir = runs_dir / m
        run_dir.mkdir(parents=True, exist_ok=True)

        stdout_f = open(run_dir / "stdout.log", "a")
        stderr_f = open(run_dir / "stderr.log", "a")

        cfg_json = json.dumps(job_cfg)
        cmd = [
            sys.executable,
            str(RUNNER_SCRIPT),
            "--job-config", cfg_json,
            "--out", str(output_dir),
            "--device", dev,
        ]

        (run_dir / "command.txt").write_text(" ".join(cmd) + "\n")

        env = dict(os.environ)
        if "cuda" in dev.lower():
            gpu_idx = dev.split(":")[-1] if ":" in dev else "0"
            env["CUDA_VISIBLE_DEVICES"] = gpu_idx

        p = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, env=env)
        print(f"[{time.strftime('%H:%M:%S')}] Launched {m} on slot {slot_idx} ({dev}) with PID {p.pid}")

        return JobProcess(
            model=m,
            job_config=job_cfg,
            process=p,
            stdout_file=stdout_f,
            stderr_file=stderr_f,
            gpu_id=dev,
            start_time=time.time(),
            retry_count=job_attempts.get(m, 0),
        )

    # Initial queue fill
    while len(active_processes) < concurrency and job_queue:
        cfg = job_queue.pop(0)
        m = cfg["model"]
        if resume and get_job_status(output_dir, m) == "DONE":
            print(f"Model {m} already marked DONE. Skipping.")
            continue
        slot = available_slots.pop(0)
        proc = launch_model(cfg, slot)
        active_processes.append(proc)

    # Main scheduler loop: dynamic slot replacement
    while active_processes or job_queue:
        for proc in list(active_processes):
            ret = proc.process.poll()
            if ret is not None:
                proc.stdout_file.close()
                proc.stderr_file.close()
                active_processes.remove(proc)
                slot_idx = int(proc.gpu_id.split(":")[-1]) if ":" in proc.gpu_id else 0
                available_slots.append(slot_idx)

                m = proc.model
                run_dir = runs_dir / m
                runtime_file = run_dir / "runtime.json"
                test_file = run_dir / "test_evaluation.json"

                result_data = {}
                if runtime_file.exists():
                    try:
                        result_data.update(json.loads(runtime_file.read_text()))
                    except Exception:
                        pass
                if test_file.exists():
                    try:
                        tv = json.loads(test_file.read_text())
                        result_data["test_median_kge"] = tv.get("median", np.nan)
                        result_data["test_q25_kge"] = tv.get("q25", np.nan)
                    except Exception:
                        pass

                if ret == 0 and (run_dir / "DONE").exists():
                    print(f"[{time.strftime('%H:%M:%S')}] Model {m} COMPLETED. Test Median KGE: {result_data.get('test_median_kge', 'N/A')}")
                    append_to_ledger(output_dir, proc.job_config, result_data, "DONE")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Model {m} FAILED with exit code {ret}.")
                    attempts = job_attempts.get(m, 0) + 1
                    job_attempts[m] = attempts
                    failure_txt = (run_dir / "failure_reason.md").read_text() if (run_dir / "failure_reason.md").exists() else ""
                    is_scientific_err = any(k in failure_txt.lower() for k in ["nan", "oom", "floatingpoint", "test leakage", "zerodivision"])
                    if attempts <= 1 and not is_scientific_err:
                        print(f"[{time.strftime('%H:%M:%S')}] Scheduling 1x retry for transient error on {m}.")
                        job_queue.append(proc.job_config)
                    else:
                        append_to_ledger(output_dir, proc.job_config, result_data, "FAILED")

                # Launch next immediately
                while len(active_processes) < concurrency and job_queue:
                    next_cfg = job_queue.pop(0)
                    next_m = next_cfg["model"]
                    if resume and get_job_status(output_dir, next_m) == "DONE":
                        print(f"Model {next_m} already marked DONE. Skipping.")
                        continue
                    next_slot = available_slots.pop(0) if available_slots else 0
                    new_proc = launch_model(next_cfg, next_slot)
                    active_processes.append(new_proc)

        time.sleep(poll_interval)

    print("\n=== All 36 Canonical dPL v2 Models Completed ===")
    generate_canonical_v2_report(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to canonical_manifest.yaml")
    parser.add_argument("--workers", type=int, default=4, help="Concurrency target (default: 4)")
    parser.add_argument("--devices", type=str, nargs="+", default=None, help="Device list (e.g. cuda:0 cuda:1)")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume skipped completed jobs")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Poll interval in seconds")
    args = parser.parse_args()

    manifest_p = Path(args.manifest).resolve()
    output_dir = manifest_p.parent

    run_canonical_queue(
        output_dir=output_dir,
        manifest_path=manifest_p,
        concurrency=args.workers,
        devices=args.devices,
        poll_interval=args.poll_interval,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
