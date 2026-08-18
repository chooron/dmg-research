#!/usr/bin/env python3
"""Launch the 531-basin CN-dPL gate (D2) with three parallel seed processes.

Each seed runs the frozen R1/R2 dPL protocol over all 531 basins with
``target_override_npz = q_star`` (synthetic target) and the canonical
``cn_psol_annual``.  Three processes run concurrently on one GPU; thread
counts are capped to keep host CPU/RAM usage bounded.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from manuscript.r3.common import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_RESULTS_ROOT,
)

SEEDS = (42, 123, 2026)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--truth-run-id", default="r3_synthetic_truth_v1")
    parser.add_argument("--dpl-run-prefix", default="r3_gate_dpl_xaj_cn_seed_")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_dir = args.results_root / "r3_gate_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    q_star = args.results_root / args.truth_run_id / "q_star.npz"
    if not q_star.exists():
        parser.error(f"truth target not found: {q_star}")

    processes = []
    for seed in args.seeds:
        config = json.loads(
            (PROJECT / "training/dpl/base_config_camels_531.json").read_text()
        )
        config["output_dir"] = str(args.results_root / f"{args.dpl_run_prefix}{seed}")
        config["data_basin_ids"] = str(args.data_root / "531sub_id.txt")
        config["target_override_npz"] = str(q_star)
        config["_protocol"] = "r3_gate_531_dpl_synthetic_target_v1"
        config["_note"] = (
            "531-basin correct-CN dPL gate: canonical attribute normalization, "
            "canonical cn_psol_annual, target = Q*."
        )
        cfg_file = config_dir / f"dpl_xaj_cn_seed_{seed}.json"
        cfg_file.write_text(json.dumps(config, indent=2) + "\n")
        command = [
            sys.executable,
            str(PROJECT / "training/dpl/run_dpl_model.py"),
            "--config",
            str(cfg_file),
            "--model",
            "XAJ_CN",
            "--lite",
            "--seed",
            str(seed),
        ]
        print("COMMAND:", " ".join(command), flush=True)
        if args.dry_run:
            continue
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
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
        handle = (output_dir / "train.log").open("a")
        handle.write("COMMAND: " + " ".join(command) + "\n")
        handle.flush()
        proc = subprocess.Popen(
            command,
            cwd=PROJECT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((seed, proc, handle))
        print(f"launched seed {seed} pid={proc.pid}", flush=True)

    if args.dry_run:
        return
    print(f"all seeds launched; waiting ...", flush=True)
    for seed, proc, handle in processes:
        proc.wait()
        handle.close()
        print(f"seed {seed} exit={proc.returncode}", flush=True)
    print("D2 parallel launch complete", flush=True)


if __name__ == "__main__":
    main()
