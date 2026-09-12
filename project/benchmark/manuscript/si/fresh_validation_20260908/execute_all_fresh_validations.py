#!/usr/bin/env python3
"""Master execution script for fresh validation suite, capturing raw logs and ledger."""
import subprocess, time, datetime, hashlib, os
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
OUT_DIR = Path(__file__).resolve().parent
LOGS_DIR = OUT_DIR / "fresh_raw_logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

commands = [
    {
        "command_id": "CMD-01",
        "category": "Registry & Parameter Contract",
        "command": "./.venv/bin/python project/benchmark/manuscript/si/fresh_validation_20260908/run_fresh_registry_contract.py",
        "raw_log": "cmd01_registry_contract.log",
        "result_artifact": "fresh_36model_registry_results.csv"
    },
    {
        "command_id": "CMD-02",
        "category": "Water Balance Closure",
        "command": "./.venv/bin/python project/benchmark/manuscript/si/fresh_validation_20260908/run_fresh_water_balance.py",
        "raw_log": "cmd02_water_balance.log",
        "result_artifact": "fresh_water_balance_results.csv"
    },
    {
        "command_id": "CMD-03",
        "category": "Forward Numerical Stability",
        "command": "./.venv/bin/python project/benchmark/manuscript/si/fresh_validation_20260908/run_fresh_forward_stability.py",
        "raw_log": "cmd03_forward_stability.log",
        "result_artifact": "fresh_forward_stability_results.csv"
    },
    {
        "command_id": "CMD-04",
        "category": "End-to-End Autograd",
        "command": "PYTHONPATH=dmotpy ./.venv/bin/python -m pytest dmotpy/tests/test_model_gradient_end_to_end.py -v",
        "raw_log": "cmd04_autograd_end_to_end.log",
        "result_artifact": "fresh_gradient_end_to_end_results.csv"
    },
    {
        "command_id": "CMD-05",
        "category": "FP64 Representative Gradcheck",
        "command": "PYTHONPATH=dmotpy ./.venv/bin/python -m pytest dmotpy/tests/test_model_gradcheck_representative.py -v",
        "raw_log": "cmd05_gradcheck_representative.log",
        "result_artifact": "fresh_gradcheck_results.csv"
    },
    {
        "command_id": "CMD-06",
        "category": "Routing & UH Mass Conservation",
        "command": "PYTHONPATH=dmotpy ./.venv/bin/python -m pytest dmotpy/tests/test_uh_tail_mass_balance.py dmotpy/tests/test_unithydro_consistency.py -v",
        "raw_log": "cmd06_routing_unithydro.log",
        "result_artifact": "fresh_routing_results.csv"
    },
    {
        "command_id": "CMD-07",
        "category": "Euler Substep Convergence",
        "command": "./.venv/bin/python project/benchmark/manuscript/si/fresh_validation_20260908/run_fresh_euler_convergence.py",
        "raw_log": "cmd07_euler_convergence.log",
        "result_artifact": "fresh_euler_convergence_results.csv"
    },
    {
        "command_id": "CMD-08",
        "category": "36-Model Gradient Sparsity & Multi-Metric Suite",
        "command": "PYTHONPATH=dmotpy ./.venv/bin/python dmotpy/scripts/standalone_validation_36.py",
        "raw_log": "cmd08_standalone_validation_36.log",
        "result_artifact": "fresh_standalone_validation_summary.log"
    }
]

ledger_rows = []

print("================================================================================")
print("STARTING FRESH REVALIDATION EXECUTION OF 36-MODEL DIFFERENTIABLE HYDROLOGICAL SUITE")
print("================================================================================")

for item in commands:
    cid = item["command_id"]
    cat = item["category"]
    cmd = item["command"]
    log_name = item["raw_log"]
    log_path = LOGS_DIR / log_name
    art_name = item["result_artifact"]
    
    start_dt = datetime.datetime.now(datetime.timezone.utc)
    start_iso = start_dt.isoformat()
    print(f"\n[{cid}] Executing: {cat} ...")
    print(f"  Command: {cmd}")
    print(f"  Start:   {start_iso}")
    
    t0 = time.time()
    res = subprocess.run(
        cmd,
        shell=True,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    t_elapsed = time.time() - t0
    end_dt = datetime.datetime.now(datetime.timezone.utc)
    end_iso = end_dt.isoformat()
    
    # Save raw log
    with open(log_path, "w") as f:
        f.write(f"=== FRESH EXECUTION RAW LOG: {cid} ({cat}) ===\n")
        f.write(f"Command: {cmd}\n")
        f.write(f"Working Directory: {REPO_ROOT}\n")
        f.write(f"Start Time (UTC): {start_iso}\n")
        f.write(f"End Time (UTC):   {end_iso}\n")
        f.write(f"Duration: {t_elapsed:.3f}s\n")
        f.write(f"Exit Code: {res.returncode}\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(res.stdout)
        f.write("\n--------------------------------------------------------------------------------\n")
        f.write(f"EXECUTION FINISHED WITH EXIT CODE: {res.returncode}\n")
        
    print(f"  End:     {end_iso} ({t_elapsed:.2f}s) | Exit code: {res.returncode}")
    print(f"  Saved raw log -> {log_path.relative_to(REPO_ROOT)}")
    
    ledger_rows.append({
        "command_id": cid,
        "category": cat,
        "command": cmd,
        "cwd": str(REPO_ROOT),
        "start_time": start_iso,
        "end_time": end_iso,
        "duration_s": round(t_elapsed, 3),
        "exit_code": res.returncode,
        "raw_log": str(log_path.relative_to(REPO_ROOT)),
        "result_artifact": art_name
    })

ledger_df = pd.DataFrame(ledger_rows)
ledger_path = OUT_DIR / "fresh_execution_ledger.csv"
ledger_df.to_csv(ledger_path, index=False)

print("\n================================================================================")
print(f"ALL FRESH VALIDATION COMMANDS COMPLETED. LEDGER SAVED TO: {ledger_path}")
print("================================================================================")
print(ledger_df[["command_id", "category", "duration_s", "exit_code", "raw_log"]])
