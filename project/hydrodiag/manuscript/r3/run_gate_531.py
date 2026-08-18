#!/usr/bin/env python3
"""Phase D launch chain: wait for the 12-basin pilot, then run the 531-basin
correct-CN gate (D1: IC-CMA-ES; D2: dPL x 3 seeds).

- D1: ``training/ic/run_tgd2_batched_cmaes_531.py --model XAJ_CN`` over all
  531 basins, default protocol (10 starts, 300 generations), target = Q*
  (``--target-npz``, which also enables the canonical ``cn_psol_annual``);
- D2: ``training/dpl/run_dpl_model.py --model XAJ_CN --lite`` over all 531
  basins with the default config, seeds 42/123/2026, target = Q*
  (``target_override_npz``).

Run products: ``results/r3_gate_ic_xaj_cn_531_v1/`` and
``results/r3_gate_dpl_xaj_cn_seed_<s>/``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from manuscript.r3.common import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_RESULTS_ROOT,
    git_commit,
    write_json,
)

SEEDS = (42, 123, 2026)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def pilot_complete(results_root: Path, run_id: str = "r3_pilot_v1") -> bool:
    manifest_path = results_root / run_id / "pilot_manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    stages = manifest.get("stages", {})
    if not stages:
        return False
    for key, stage in stages.items():
        kind = key.split(":")[0]
        if kind == "ic":
            if not (Path(stage["output"]) / "DONE.json").exists():
                return False
        else:
            if not (Path(stage["output_dir"]) / "COMPLETE").exists():
                return False
    return True


def wait_for_pilot(results_root: Path, run_id: str, poll_s: int = 300) -> None:
    while not pilot_complete(results_root, run_id):
        print(f"[gate-chain] {_utcnow()} waiting for pilot ...", flush=True)
        time.sleep(poll_s)


def run_ic_gate(args) -> dict:
    output = args.results_root / args.ic_run_id
    q_star = args.results_root / args.truth_run_id / "q_star.npz"
    command = [
        sys.executable,
        str(PROJECT / "training/ic/run_tgd2_batched_cmaes_531.py"),
        "--model",
        "XAJ_CN",
        "--output",
        str(output),
        "--starts",
        str(args.starts),
        "--generations",
        str(args.generations),
        "--target-npz",
        str(q_star),
        "--device",
        args.device,
    ]
    print(f"[gate-chain] {_utcnow()} D1 IC: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)
    return {"command": " ".join(command), "output": str(output)}


def run_dpl_gate(args, seed: int) -> dict:
    config = json.loads(
        (PROJECT / "training/dpl/base_config_camels_531.json").read_text()
    )
    config["output_dir"] = str(args.results_root / f"{args.dpl_run_prefix}{seed}")
    config["data_basin_ids"] = str(args.data_root / "531sub_id.txt")
    config["target_override_npz"] = str(
        args.results_root / args.truth_run_id / "q_star.npz"
    )
    config["_protocol"] = "r3_gate_531_dpl_synthetic_target_v1"
    config["_note"] = (
        "531-basin correct-CN dPL gate: canonical attribute normalization, "
        "canonical cn_psol_annual, target = Q*."
    )
    config_path = args.results_root / args.ic_run_id / ".." / "r3_gate_configs"
    config_path = (args.results_root / "r3_gate_configs").resolve()
    config_path.mkdir(parents=True, exist_ok=True)
    cfg_file = config_path / f"dpl_xaj_cn_seed_{seed}.json"
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
    print(
        f"[gate-chain] {_utcnow()} D2 dPL seed {seed}: {' '.join(command)}", flush=True
    )
    subprocess.run(command, cwd=PROJECT, check=True)
    return {"command": " ".join(command), "output": config["output_dir"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--truth-run-id", default="r3_synthetic_truth_v1")
    parser.add_argument("--pilot-run-id", default="r3_pilot_v1")
    parser.add_argument("--ic-run-id", default="r3_gate_ic_xaj_cn_531_v1")
    parser.add_argument("--dpl-run-prefix", default="r3_gate_dpl_xaj_cn_seed_")
    parser.add_argument("--starts", type=int, default=10)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-dpl",
        action="store_true",
        help="Run only D1 (531 CN-IC); D2 is launched separately in parallel.",
    )
    parser.add_argument("--wait-poll-s", type=int, default=300)
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()

    report = {
        "protocol": "r3_gate_531_chain_v1",
        "created_at": _utcnow(),
        "code": git_commit(args.project_root),
        "truth_run_id": args.truth_run_id,
        "ic_run_id": args.ic_run_id,
        "dpl_run_prefix": args.dpl_run_prefix,
        "stages": {},
    }
    if not args.skip_wait:
        wait_for_pilot(args.results_root, args.pilot_run_id, args.wait_poll_s)

    report["stages"]["d1_ic"] = run_ic_gate(args)
    if args.skip_dpl:
        print("[gate-chain] D2 skipped (launched separately in parallel)", flush=True)
        args.seeds = []
    for seed in args.seeds:
        key = f"d2_dpl_seed_{seed}"
        report["stages"][key] = run_dpl_gate(args, seed)
    report["completed_at"] = _utcnow()
    write_json(args.results_root / "r3_gate_chain_report.json", report)
    print(
        f"[gate-chain] COMPLETE -> {args.results_root / 'r3_gate_chain_report.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
