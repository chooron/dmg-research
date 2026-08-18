#!/usr/bin/env python3
"""Launch local three-seed XAJ-TGD2 dPL retraining under the lite-v2 protocol."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REPO = PROJECT.parents[1]
SOURCE_ROOT = PROJECT / "results/dpl_camels_531_lite_v2/XAJ_TGD"
OUTPUT_ROOT = "results/dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2"
SEEDS = (42, 123, 2026)


def config_for(seed: int) -> Path:
    config = json.loads((SOURCE_ROOT / f"seed_{seed}/config.json").read_text())
    data = REPO / "data"
    config["_protocol"] = "dpl_camels_531_lite_v3_tgd2_dpl_audited"
    config["_note"] = (
        "Copied Lite-v2 dPL protocol: random contiguous 730-day windows; "
        "first 365 days warm-up only, final 365 days KGE loss. Only TGD2 differs."
    )
    config["gage_ids_path"] = str(data / "gage_id.npy")
    config["dates_path"] = str(data / "camels_dates.npy")
    config["data_pkl_dataset"] = str(data / "camels_dataset")
    config["data_basin_ids"] = str(data / "531sub_id.txt")
    config["output_dir"] = f"{OUTPUT_ROOT}/seed_{seed}"
    config["model_name"] = "XAJ_TGD2"
    config["network"]["parameter_mapping"] = (
        "sigmoid_to_physical_range_with_log_tgd2_residence_times"
    )
    config["network"]["tgd_structure_version"] = (
        "temperature_dependent_generic_delay2_v1"
    )
    config.pop("parameter_names", None)
    config.pop("parameter_specs", None)
    target = HERE / "generated_configs" / f"xaj_tgd2_lite_v3_seed_{seed}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, indent=2) + "\n")
    return target


def main() -> None:
    logs = HERE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    active = []
    for seed in SEEDS:
        output = PROJECT / f"{OUTPUT_ROOT}/seed_{seed}"
        if (output / "COMPLETE").exists():
            print(f"SKIP seed={seed}: COMPLETE", flush=True)
            continue
        command = [
            sys.executable,
            str(HERE / "run_dpl_model.py"),
            "--config",
            str(config_for(seed)),
            "--model",
            "XAJ_TGD2",
            "--lite",
            "--resume",
        ]
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "TORCHINDUCTOR_COMPILE_THREADS": "1",
            }
        )
        handle = (logs / f"xaj_tgd2_lite_v3_seed_{seed}.out").open("a")
        handle.write("COMMAND: " + " ".join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        active.append((seed, process, handle))
        print(f"START seed={seed} pid={process.pid}", flush=True)
    while active:
        remaining = []
        for seed, process, handle in active:
            status = process.poll()
            if status is None:
                remaining.append((seed, process, handle))
            else:
                handle.close()
                print(f"DONE seed={seed} code={status}", flush=True)
        active = remaining
        if active:
            time.sleep(5)
    if any(
        not (PROJECT / f"{OUTPUT_ROOT}/seed_{seed}/COMPLETE").exists() for seed in SEEDS
    ):
        raise SystemExit("one or more dPL seeds did not complete")


if __name__ == "__main__":
    main()
