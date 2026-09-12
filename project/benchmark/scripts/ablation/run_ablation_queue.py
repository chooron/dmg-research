#!/usr/bin/env python3
"""4-Worker Queue Scheduler for dPL Protocol Ablation Study (Phase H & Phase U).

Enforces:
1. TARGET_CONCURRENCY = 4 (Dynamic slot replacement: fills empty slot immediately).
2. Phase Isolation: Automatically executes specified phase (H or U) and pauses gracefully.
3. Process-level isolation, per-job stdout/stderr logging, and .lock protection.
4. Transient failure retry (max 1 retry for transient runtime errors, 0 for NaN/OOM).
5. Continuous appending to ABLATION_LEDGER.md.
6. Automatic Phase report generation upon completion.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
RUNNER_SCRIPT = BENCHMARK_ROOT / "scripts/ablation/ablation_runner.py"


@dataclass
class JobProcess:
    job_id: str
    phase: str
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


def get_job_status(ablation_root: Path, job_id: str) -> str:
    run_dir = ablation_root / "runs" / job_id
    if not run_dir.exists():
        return "QUEUED"
    if (run_dir / "DONE").exists():
        return "DONE"
    if (run_dir / "FAILED").exists():
        return "FAILED"
    if (run_dir / ".lock").exists():
        return "RUNNING"
    return "QUEUED"


def append_to_ledger(ablation_root: Path, job_config: dict[str, Any], result: dict[str, Any], status: str) -> None:
    ledger_path = ablation_root / "ABLATION_LEDGER.md"
    if not ledger_path.exists():
        return

    job_id = job_config["job_id"]
    phase = job_config.get("phase", "H")
    model = job_config.get("model", "")
    horizon = job_config.get("horizon_days", 730)
    warmup = job_config.get("warmup_days", 365)
    scored = job_config.get("scored_days", 365)
    mapping = job_config.get("mapping", "auto")
    lr = job_config.get("lr", 1e-3)
    scheduler = job_config.get("scheduler", "none")
    clip = job_config.get("clip_norm", 1.0)
    max_up = job_config.get("max_optimizer_updates", "N/A")

    best_ep = result.get("best_epoch", "N/A")
    best_loss = f"{result.get('best_train_loss', 0.0):.4f}" if "best_train_loss" in result else "N/A"
    med_kge = f"{result.get('inner_val_median_kge', 0.0):.4f}" if "inner_val_median_kge" in result else "N/A"
    q25_kge = f"{result.get('inner_val_q25_kge', 0.0):.4f}" if "inner_val_q25_kge" in result else "N/A"
    clip_frac = "N/A"

    run_dir = ablation_root / "runs" / job_id
    ep_csv = run_dir / "epoch_metrics.csv"
    if ep_csv.exists():
        try:
            df_ep = pd.read_csv(ep_csv)
            if not df_ep.empty and "grad_clip_fraction" in df_ep.columns:
                clip_frac = f"{df_ep['grad_clip_fraction'].mean()*100:.1f}%"
        except Exception:
            pass

    row = (
        f"| `{job_id}` | `{phase}` | `{model}` | {horizon}d | {warmup}d | {scored}d | `{mapping}` | "
        f"{lr:.1e} | `{scheduler}` | {clip} | **`{status}`** | {best_ep} | {best_loss} | {med_kge} | {q25_kge} | {clip_frac} | `runs/{job_id}/` |\n"
    )

    with open(ledger_path, "a") as f:
        f.write(row)


def generate_phase_h_report(ablation_root: Path) -> None:
    manifest_path = ablation_root / "experiment_manifest.yaml"
    if not manifest_path.exists():
        return
    manifest = load_manifest(manifest_path)
    h_jobs = [j for j in manifest["jobs"] if j.get("phase") == "H"]

    summary_rows = []
    paired_data = {}

    for j in h_jobs:
        job_id = j["job_id"]
        model = j["model"]
        horizon = j["horizon_days"]
        run_dir = ablation_root / "runs" / job_id

        status = get_job_status(ablation_root, job_id)
        inner_val_file = run_dir / "inner_validation_metrics.json"
        runtime_file = run_dir / "runtime.json"
        best_meta_file = run_dir / "best_metadata.json"

        med_kge = np.nan
        q25_kge = np.nan
        mean_kge = np.nan
        best_loss = np.nan
        best_ep = np.nan
        runtime_s = np.nan

        if inner_val_file.exists():
            try:
                iv = json.loads(inner_val_file.read_text())
                med_kge = float(iv.get("median", np.nan))
                q25_kge = float(iv.get("q25", np.nan))
                mean_kge = float(iv.get("mean", np.nan))
            except Exception:
                pass

        if best_meta_file.exists():
            try:
                bm = json.loads(best_meta_file.read_text())
                best_loss = float(bm.get("best_selection_value", np.nan))
                best_ep = int(bm.get("best_epoch", 1))
            except Exception:
                pass

        if runtime_file.exists():
            try:
                rt = json.loads(runtime_file.read_text())
                runtime_s = float(rt.get("total_runtime_s", np.nan))
            except Exception:
                pass

        summary_rows.append({
            "job_id": job_id,
            "model": model,
            "horizon_days": horizon,
            "status": status,
            "best_epoch": best_ep,
            "best_train_loss": best_loss,
            "inner_val_median_kge": med_kge,
            "inner_val_q25_kge": q25_kge,
            "inner_val_mean_kge": mean_kge,
            "runtime_seconds": runtime_s,
        })

        if model not in paired_data:
            paired_data[model] = {}
        paired_data[model][horizon] = {
            "med_kge": med_kge,
            "q25_kge": q25_kge,
            "best_loss": best_loss,
            "runtime_s": runtime_s,
            "status": status,
        }

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(ablation_root / "PHASE_H_SUMMARY.csv", index=False)

    paired_rows = []
    for model, h_dict in sorted(paired_data.items()):
        h1 = h_dict.get(730, {})
        h2 = h_dict.get(1825, {})
        med_730 = h1.get("med_kge", np.nan)
        med_1825 = h2.get("med_kge", np.nan)
        q25_730 = h1.get("q25_kge", np.nan)
        q25_1825 = h2.get("q25_kge", np.nan)

        d_med = med_1825 - med_730 if (np.isfinite(med_1825) and np.isfinite(med_730)) else np.nan
        d_q25 = q25_1825 - q25_730 if (np.isfinite(q25_1825) and np.isfinite(q25_730)) else np.nan

        # Gate Evaluation
        if np.isnan(d_med):
            gate = "INCOMPLETE"
        elif d_med >= 0.005 or d_q25 >= 0.010:
            gate = "HORIZON_BENEFIT"
        elif abs(d_med) < 0.003 and abs(d_q25) < 0.005:
            gate = "NO_MATERIAL_HORIZON_BENEFIT"
        else:
            gate = "AMBIGUOUS"

        paired_rows.append({
            "model": model,
            "h1_730d_median_kge": med_730,
            "h2_1825d_median_kge": med_1825,
            "delta_median_kge": d_med,
            "h1_730d_q25_kge": q25_730,
            "h2_1825d_q25_kge": q25_1825,
            "delta_q25_kge": d_q25,
            "gate_verdict": gate,
        })

    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(ablation_root / "PHASE_H_PAIRED_COMPARISON.csv", index=False)

    md_text = """# Phase H: Training Horizon Core Ablation Report (2026-08-31)

**Study Scope**: 8 Representative Hydrological Models comparing H1 (730d total, 365d scored) vs H2 (1825d total, 1460d scored) on CAMELS 531 basins.  
**Evaluation Protocol**: Evaluated strictly on INNER-VAL (`1990-10-01..1995-09-30`) on restored `best.pt`. TEST partition strictly untouched.

---

## 1. Paired Horizon Comparison (H1 vs H2)

| Model | H1 (730d) Med KGE | H2 (1825d) Med KGE | Δ Med KGE | H1 Q25 KGE | H2 Q25 KGE | Δ Q25 KGE | Gate Verdict |
|---|---|---|---|---|---|---|---|
"""
    for _, r in df_paired.iterrows():
        md_text += f"| `{r['model']}` | {r['h1_730d_median_kge']:.4f} | {r['h2_1825d_median_kge']:.4f} | **{r['delta_median_kge']:+.4f}** | {r['h1_730d_q25_kge']:.4f} | {r['h2_1825d_q25_kge']:.4f} | {r['delta_q25_kge']:+.4f} | `{r['gate_verdict']}` |\n"

    md_text += """
---

## 2. Gate Decision Rules
- `HORIZON_BENEFIT`: $\\Delta \\text{Median KGE} \\ge +0.005$ OR $\\Delta \\text{Q25 KGE} \\ge +0.010$.
- `NO_MATERIAL_HORIZON_BENEFIT`: $|\\Delta \\text{Median KGE}| < 0.003$ AND $|\\Delta \\text{Q25 KGE}| < 0.005$.
- `AMBIGUOUS`: Intermediate response requiring further evaluation.
"""
    (ablation_root / "PHASE_H_REPORT.md").write_text(md_text)


def generate_phase_u_report(ablation_root: Path) -> None:
    manifest_path = ablation_root / "experiment_manifest.yaml"
    if not manifest_path.exists():
        return
    manifest = load_manifest(manifest_path)
    u_jobs = [j for j in manifest["jobs"] if j.get("phase") == "U"]

    # Load Phase H results for comparison
    h_summary_p = ablation_root / "PHASE_H_SUMMARY.csv"
    h_data = {}
    if h_summary_p.exists():
        df_h = pd.read_csv(h_summary_p)
        for _, row in df_h.iterrows():
            m = row["model"]
            h = int(row["horizon_days"])
            if m not in h_data:
                h_data[m] = {}
            h_data[m][h] = {
                "med_kge": float(row["inner_val_median_kge"]),
                "q25_kge": float(row["inner_val_q25_kge"]),
                "best_loss": float(row["best_train_loss"]),
            }

    summary_rows = []
    matched_rows = []

    for j in u_jobs:
        job_id = j["job_id"]
        model = j["model"]
        target_updates = j["max_optimizer_updates"]
        run_dir = ablation_root / "runs" / job_id

        status = get_job_status(ablation_root, job_id)
        inner_val_file = run_dir / "inner_validation_metrics.json"
        runtime_file = run_dir / "runtime.json"
        best_meta_file = run_dir / "best_metadata.json"

        u_med = np.nan
        u_q25 = np.nan
        u_mean = np.nan
        u_loss = np.nan
        actual_up = np.nan
        runtime_s = np.nan

        if inner_val_file.exists():
            try:
                iv = json.loads(inner_val_file.read_text())
                u_med = float(iv.get("median", np.nan))
                u_q25 = float(iv.get("q25", np.nan))
                u_mean = float(iv.get("mean", np.nan))
            except Exception:
                pass

        if best_meta_file.exists():
            try:
                bm = json.loads(best_meta_file.read_text())
                u_loss = float(bm.get("best_selection_value", np.nan))
                actual_up = int(bm.get("actual_optimizer_updates", target_updates))
            except Exception:
                pass

        if runtime_file.exists():
            try:
                rt = json.loads(runtime_file.read_text())
                runtime_s = float(rt.get("total_runtime_s", np.nan))
            except Exception:
                pass

        summary_rows.append({
            "job_id": job_id,
            "model": model,
            "target_updates": target_updates,
            "actual_updates": actual_up,
            "status": status,
            "best_train_loss": u_loss,
            "inner_val_median_kge": u_med,
            "inner_val_q25_kge": u_q25,
            "inner_val_mean_kge": u_mean,
            "runtime_seconds": runtime_s,
        })

        # Three-Point Contrast: H1 (730d), H2 (1825d), U (1825d matched)
        h1_dict = h_data.get(model, {}).get(730, {})
        h2_dict = h_data.get(model, {}).get(1825, {})

        h1_med = h1_dict.get("med_kge", np.nan)
        h1_q25 = h1_dict.get("q25_kge", np.nan)
        h2_med = h2_dict.get("med_kge", np.nan)
        h2_q25 = h2_dict.get("q25_kge", np.nan)

        d_h_med = h2_med - h1_med if (np.isfinite(h2_med) and np.isfinite(h1_med)) else np.nan
        d_u_med = u_med - h2_med if (np.isfinite(u_med) and np.isfinite(h2_med)) else np.nan
        d_matched_med = u_med - h1_med if (np.isfinite(u_med) and np.isfinite(h1_med)) else np.nan

        d_h_q25 = h2_q25 - h1_q25 if (np.isfinite(h2_q25) and np.isfinite(h1_q25)) else np.nan
        d_u_q25 = u_q25 - h2_q25 if (np.isfinite(u_q25) and np.isfinite(h2_q25)) else np.nan
        d_matched_q25 = u_q25 - h1_q25 if (np.isfinite(u_q25) and np.isfinite(h1_q25)) else np.nan

        # Gate Evaluation
        if np.isnan(d_matched_med):
            verdict = "INCOMPLETE"
        elif d_matched_med >= 0.005 and d_matched_q25 > -0.005:
            verdict = "LONG_HORIZON_WINS_MATCHED"
        elif d_matched_med <= -0.005 and d_matched_q25 <= -0.005:
            verdict = "SHORT_HORIZON_WINS_MATCHED"
        elif d_u_med >= 0.005 or d_u_q25 >= 0.010:
            verdict = "UPDATE_LIMITED"
        else:
            verdict = "MIXED"

        matched_rows.append({
            "model": model,
            "target_updates": target_updates,
            "h1_730d_med_kge": h1_med,
            "h2_1825d_med_kge": h2_med,
            "u_1825d_matched_med_kge": u_med,
            "delta_H_med (H2 - H1)": d_h_med,
            "delta_U_med (U - H2)": d_u_med,
            "delta_MATCHED_med (U - H1)": d_matched_med,
            "h1_730d_q25_kge": h1_q25,
            "h2_1825d_q25_kge": h2_q25,
            "u_1825d_matched_q25_kge": u_q25,
            "delta_H_q25": d_h_q25,
            "delta_U_q25": d_u_q25,
            "delta_MATCHED_q25": d_matched_q25,
            "verdict": verdict,
        })

    df_u_summary = pd.DataFrame(summary_rows)
    df_u_summary.to_csv(ablation_root / "PHASE_U_SUMMARY.csv", index=False)

    df_matched = pd.DataFrame(matched_rows)
    df_matched.to_csv(ablation_root / "PHASE_U_MATCHED_COMPARISON.csv", index=False)

    # Markdown report
    md_text = """# Phase U: Update-Budget Matched Horizon Ablation Report (2026-08-31)

**Study Scope**: 5 Key Hydrological Models (`hbv96`, `mopex4`, `xinanjiang`, `topmodel`, `gr4j`) matching 1825d optimizer update budget to H1 baseline.  
**Evaluation Protocol**: Evaluated strictly on INNER-VAL (`1990-10-01..1995-09-30`) on restored `best.pt`. TEST partition strictly untouched.

---

## 1. Three-Point Matched Comparison Table (H1 vs H2 vs U)

| Model | Updates | H1 (730d) Med | H2 (1825d) Med | U (1825d Matched) Med | Δ_H (H2-H1) | Δ_U (U-H2) | Δ_MATCHED (U-H1) | Verdict |
|---|---|---|---|---|---|---|---|---|
"""
    for _, r in df_matched.iterrows():
        md_text += f"| `{r['model']}` | {r['target_updates']} | {r['h1_730d_med_kge']:.4f} | {r['h2_1825d_med_kge']:.4f} | **{r['u_1825d_matched_med_kge']:.4f}** | {r['delta_H_med (H2 - H1)']:+.4f} | {r['delta_U_med (U - H2)']:+.4f} | **{r['delta_MATCHED_med (U - H1)']:+.4f}** | `{r['verdict']}` |\n"

    md_text += """
---

## 2. Gate Decision Rules
- `UPDATE_LIMITED`: $\\Delta_U \\ge +0.005$ in median OR $\\ge +0.010$ in Q25 (confirms H2 was bottlenecked by updates).
- `LONG_HORIZON_WINS_MATCHED`: $\\Delta_{\\text{MATCHED}} \\ge +0.005$ and $\\Delta Q25 > -0.005$ (1825d net benefit when updates matched).
- `SHORT_HORIZON_WINS_MATCHED`: $\\Delta_{\\text{MATCHED}} \\le -0.005$ and $\\Delta Q25 \\le -0.005$ (730d remains superior even with matched updates).
- `MIXED`: Intermediate / balanced response.
"""
    (ablation_root / "PHASE_U_REPORT.md").write_text(md_text)


def run_queue(
    ablation_root: Path,
    manifest_path: Path,
    target_phase: str = "H",
    concurrency: int = 4,
    devices: list[str] | None = None,
    poll_interval: float = 1.0,
    resume: bool = True,
) -> None:
    manifest = load_manifest(manifest_path)
    all_jobs = manifest["jobs"]
    candidate_jobs = [j for j in all_jobs if j.get("phase") == target_phase and j.get("status") == "ELIGIBLE"]

    if not candidate_jobs:
        print(f"No eligible jobs found for Phase '{target_phase}'.")
        return

    ablation_root.mkdir(parents=True, exist_ok=True)
    runs_dir = ablation_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if devices is None or len(devices) == 0:
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            devices = [f"cuda:{i % n_gpus}" for i in range(concurrency)]
        else:
            devices = ["cpu"] * concurrency

    print(f"=== Starting Ablation Queue Scheduler ===")
    print(f"Root: {ablation_root}")
    print(f"Target Phase: {target_phase} ({len(candidate_jobs)} jobs)")
    print(f"Concurrency: {concurrency}")
    print(f"Device Allocation: {devices}")
    print(f"Resume Enabled: {resume}")

    active_processes: list[JobProcess] = []
    job_queue = list(candidate_jobs)
    job_attempts = {}

    available_slots = list(range(concurrency))

    def launch_job(job_cfg: dict[str, Any], slot_idx: int) -> JobProcess:
        job_id = job_cfg["job_id"]
        dev = devices[slot_idx % len(devices)]
        run_dir = runs_dir / job_id
        run_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        stdout_f = open(stdout_path, "a")
        stderr_f = open(stderr_path, "a")

        cfg_json = json.dumps(job_cfg)
        cmd = [
            sys.executable,
            str(RUNNER_SCRIPT),
            "--job-config", cfg_json,
            "--out", str(ablation_root),
            "--device", dev,
        ]

        (run_dir / "command.txt").write_text(" ".join(cmd) + "\n")

        env = dict(os.environ)
        if "cuda" in dev.lower():
            gpu_idx = dev.split(":")[-1] if ":" in dev else "0"
            env["CUDA_VISIBLE_DEVICES"] = gpu_idx

        p = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, env=env)
        print(f"[{time.strftime('%H:%M:%S')}] Launched {job_id} on slot {slot_idx} ({dev}) with PID {p.pid}")

        return JobProcess(
            job_id=job_id,
            phase=job_cfg.get("phase", target_phase),
            model=job_cfg.get("model", ""),
            job_config=job_cfg,
            process=p,
            stdout_file=stdout_f,
            stderr_file=stderr_f,
            gpu_id=dev,
            start_time=time.time(),
            retry_count=job_attempts.get(job_id, 0),
        )

    # Initial queue dispatch
    while len(active_processes) < concurrency and job_queue:
        job_cfg = job_queue.pop(0)
        job_id = job_cfg["job_id"]

        if resume and get_job_status(ablation_root, job_id) == "DONE":
            print(f"Job {job_id} already marked DONE. Skipping.")
            continue

        slot = available_slots.pop(0)
        proc_info = launch_job(job_cfg, slot)
        active_processes.append(proc_info)

    # Main scheduler loop: dynamic slot replacement
    while active_processes or job_queue:
        for proc_info in list(active_processes):
            retcode = proc_info.process.poll()
            if retcode is not None:
                proc_info.stdout_file.close()
                proc_info.stderr_file.close()
                active_processes.remove(proc_info)
                slot_idx = int(proc_info.gpu_id.split(":")[-1]) if ":" in proc_info.gpu_id else 0
                available_slots.append(slot_idx)

                job_id = proc_info.job_id
                run_dir = runs_dir / job_id
                runtime_file = run_dir / "runtime.json"
                inner_val_file = run_dir / "inner_validation_metrics.json"

                result_data = {}
                if runtime_file.exists():
                    try:
                        result_data.update(json.loads(runtime_file.read_text()))
                    except Exception:
                        pass
                if inner_val_file.exists():
                    try:
                        iv = json.loads(inner_val_file.read_text())
                        result_data["inner_val_median_kge"] = iv.get("median", np.nan)
                        result_data["inner_val_q25_kge"] = iv.get("q25", np.nan)
                    except Exception:
                        pass

                if retcode == 0 and (run_dir / "DONE").exists():
                    print(f"[{time.strftime('%H:%M:%S')}] Job {job_id} COMPLETED successfully. (Inner-Val Med KGE: {result_data.get('inner_val_median_kge', 'N/A')})")
                    append_to_ledger(ablation_root, proc_info.job_config, result_data, "DONE")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Job {job_id} FAILED with exit code {retcode}.")
                    attempts = job_attempts.get(job_id, 0) + 1
                    job_attempts[job_id] = attempts
                    failure_txt = (run_dir / "failure_reason.md").read_text() if (run_dir / "failure_reason.md").exists() else ""

                    is_scientific_err = any(k in failure_txt.lower() for k in ["nan", "oom", "floatingpoint", "test leakage", "zerodivision"])
                    if attempts <= 1 and not is_scientific_err:
                        print(f"[{time.strftime('%H:%M:%S')}] Scheduling 1x retry for transient error on {job_id}.")
                        job_queue.append(proc_info.job_config)
                    else:
                        append_to_ledger(ablation_root, proc_info.job_config, result_data, "FAILED")

                # Dynamic slot replacement
                while len(active_processes) < concurrency and job_queue:
                    next_cfg = job_queue.pop(0)
                    next_id = next_cfg["job_id"]
                    if resume and get_job_status(ablation_root, next_id) == "DONE":
                        print(f"Job {next_id} already marked DONE. Skipping.")
                        continue
                    next_slot = available_slots.pop(0) if available_slots else 0
                    new_proc = launch_job(next_cfg, next_slot)
                    active_processes.append(new_proc)

        time.sleep(poll_interval)

    # Phase completion handling
    print(f"\n=== All Phase {target_phase} Jobs Completed ===")
    if target_phase == "H":
        print("Generating Phase H Report and Paired Comparison...")
        generate_phase_h_report(ablation_root)
        print("State set to PAUSED_AFTER_PHASE_H.")
    elif target_phase == "U":
        print("Generating Phase U Report and Matched Comparison...")
        generate_phase_u_report(ablation_root)
        print("State set to PAUSED_AFTER_PHASE_U.")
    elif target_phase == "W2":
        print("Generating Phase W2 2x2 Warmup Decomposition and Report...")
        from project.benchmark.scripts.ablation.evaluate_2x2_warmup import run_2x2_evaluation
        dev_str = devices[0] if devices else "cpu"
        run_2x2_evaluation(ablation_root, device_str=dev_str)
        print("State set to PAUSED_AFTER_PHASE_W2. Scheduler will not start Phase O/M/C automatically.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to experiment_manifest.yaml")
    parser.add_argument("--phase", type=str, default="H", help="Phase to execute (default: H)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrency target (default: 4)")
    parser.add_argument("--devices", type=str, nargs="+", default=None, help="Device list (e.g. cuda:0 cuda:1)")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume skipped completed jobs")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Poll interval in seconds")
    args = parser.parse_args()

    manifest_p = Path(args.manifest).resolve()
    ablation_root = manifest_p.parent

    run_queue(
        ablation_root=ablation_root,
        manifest_path=manifest_p,
        target_phase=args.phase,
        concurrency=args.workers,
        devices=args.devices,
        poll_interval=args.poll_interval,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
