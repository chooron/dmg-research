#!/usr/bin/env python3
"""R3 pilot orchestration (Phase 4 + Phase 5, engineering gate only).

Reuses the repository's production pipelines unchanged:

- IC-CMA-ES: ``training/ic/run_tgd2_batched_cmaes_531.py`` (batched
  CMA-ES, default 10 starts / 300 generations, population rule
  max(12, round(25*d/17)), KGE(train) objective, normalized [0,1]
  coordinates clipped to bounds);
- dPL: ``training/dpl/run_dpl_model.py`` (35 -> 256^3 -> n_params MLP,
  balanced_valid_kge_windows sampling, default epochs/optimizer config,
  seeds 42/123/2026).

The only synthetic adaptation is the calibration target: Q* replaces the
observed discharge (IC: ``--target-npz``; dPL: ``target_override_npz``
config key).  The pilot basin subset is a deterministic frac_snow-tercile
stratified sample (``r3/common.pilot_basin_subset``) used only as an
engineering gate, never as a scientific sample.

Stages:

- cn-ic      : CN + IC-CMA-ES            (Phase 4)
- cn-dpl     : CN + dPL                  (Phase 4)
- comp-ic    : Base + IC, TGD2 + IC      (Phase 5)
- comp-dpl   : Base + dPL, TGD2 + dPL    (Phase 5)
- all        : every stage above
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[0]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from r3.common import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_RESULTS_ROOT,
    frac_snow_series,
    git_commit,
    pilot_basin_subset,
    reordered_531_list,
    write_json,
)

IC_MODELS = {"cn-ic": ["XAJ_CN"], "comp-ic": ["XAJ", "XAJ_TGD2"]}
DPL_MODELS = {"cn-dpl": ["XAJ_CN"], "comp-dpl": ["XAJ", "XAJ_TGD2"]}
DPL_SEEDS = (42, 123, 2026)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["cn-ic", "cn-dpl", "comp-ic", "comp-dpl", "all"],
        default="all",
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--truth-run-id", default="r3_synthetic_truth_v1")
    parser.add_argument("--run-id", default="r3_pilot_v1")
    parser.add_argument("--n-basins", type=int, default=12)
    parser.add_argument("--ic-starts", type=int, default=10)
    parser.add_argument("--ic-generations", type=int, default=300)
    parser.add_argument("--dpl-seeds", type=int, nargs="+", default=list(DPL_SEEDS))
    parser.add_argument("--device", default="cuda" if _cuda() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.n_basins < 3 or args.n_basins % 3 != 0:
        parser.error(
            "--n-basins must be a positive multiple of 3 (tercile stratification)"
        )
    per_tercile = args.n_basins // 3
    truth_dir = args.results_root / args.truth_run_id
    if not (truth_dir / "q_star.npz").exists():
        parser.error(
            f"truth run not found: {truth_dir}/q_star.npz (run generate_truth.py first)"
        )

    bundle, _config = load_bundle(args.project_root, args.data_root)
    snow = frac_snow_series(bundle)
    frac_map = dict(zip(snow["basin_id"], snow["frac_snow"]))
    pilot = pilot_basin_subset(
        bundle.basin_ids,
        np.asarray([frac_map[b] for b in bundle.basin_ids]),
        per_tercile=per_tercile,
    )
    if len(pilot) != args.n_basins:
        raise RuntimeError(
            f"pilot subset size mismatch: {len(pilot)} != {args.n_basins}"
        )

    run_dir = args.results_root / args.run_id
    configs_dir = run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    order_file = configs_dir / "pilot_basin_order_531.json"
    order_file.write_text(
        json.dumps(reordered_531_list(bundle.basin_ids, pilot), indent=1) + "\n"
    )
    (configs_dir / "pilot_basins.json").write_text(
        json.dumps(
            {
                "basin_ids": pilot,
                "frac_snow": {b: frac_map[b] for b in pilot},
                "selection": "deterministic tercile-stratified spread; engineering gate only",
            },
            indent=2,
        )
        + "\n"
    )

    q_star_path = truth_dir / "q_star.npz"
    manifest = {
        "protocol": "r3_pilot_v1",
        "created_at": _utcnow(),
        "code": git_commit(args.project_root),
        "truth_run_id": args.truth_run_id,
        "pilot_basins": pilot,
        "per_tercile": per_tercile,
        "ic_defaults": {
            "starts": args.ic_starts,
            "generations": args.ic_generations,
            "objective": "KGE(Q*), maximize, train only",
            "population_rule": "max(12, round(25*dimension/17))",
        },
        "dpl_defaults": {
            "seeds": list(args.dpl_seeds),
            "network": "35->256->256->256->n_params",
            "epochs": "config default (100)",
            "sampling": "balanced_valid_kge_windows",
        },
        "stages": {},
        "engineering_only": True,
        "engineering_only_note": (
            "12-basin deterministic frac_snow-stratified subset; validates "
            "plumbing only. Never used for scientific identifiability or "
            "relationship conclusions."
        ),
    }

    if args.stage == "all":
        ic_models = IC_MODELS["cn-ic"] + IC_MODELS["comp-ic"]
        dpl_models = DPL_MODELS["cn-dpl"] + DPL_MODELS["comp-dpl"]
    elif args.stage in IC_MODELS:
        ic_models, dpl_models = IC_MODELS[args.stage], []
    elif args.stage in DPL_MODELS:
        ic_models, dpl_models = [], DPL_MODELS[args.stage]
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(f"unknown stage {args.stage}")

    for model in ic_models:
        output = args.results_root / f"{args.run_id}_ic_{model.lower()}"
        command = [
            sys.executable,
            str(PROJECT / "training/ic/run_tgd2_batched_cmaes_531.py"),
            "--model",
            model,
            "--output",
            str(output),
            "--starts",
            str(args.ic_starts),
            "--generations",
            str(args.ic_generations),
            "--basin-ids",
            ",".join(pilot),
            "--target-npz",
            str(q_star_path),
            "--device",
            args.device,
        ]
        manifest["stages"][f"ic:{model}"] = {
            "command": " ".join(command),
            "output": str(output),
            "checkpoint": str(output / "checkpoints" / f"{model.lower()}_batched.pt"),
        }
        print("COMMAND:", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT, check=True)

    for model in dpl_models:
        for seed in args.dpl_seeds:
            config = json.loads(
                (PROJECT / "training/dpl/base_config_camels_531.json").read_text()
            )
            config["output_dir"] = str(
                args.results_root / f"{args.run_id}_dpl_{model.lower()}_seed_{seed}"
            )
            config["data_basin_ids"] = str(order_file)
            config["target_override_npz"] = str(q_star_path)
            config["_protocol"] = "r3_pilot_v1_dpl_synthetic_target"
            config["_note"] = (
                "Pilot dPL with the synthetic target Q*; basin list is the full "
                "531 with the pilot subset first (--max-basins N); attribute "
                "normalization uses the full 531-basin statistics (canonical)."
            )
            config_path = configs_dir / f"dpl_{model.lower()}_seed_{seed}.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            command = [
                sys.executable,
                str(PROJECT / "training/dpl/run_dpl_model.py"),
                "--config",
                str(config_path),
                "--model",
                model,
                "--lite",
                "--max-basins",
                str(args.n_basins),
                "--seed",
                str(seed),
            ]
            manifest["stages"][f"dpl:{model}:{seed}"] = {
                "command": " ".join(command),
                "config": str(config_path),
                "output_dir": config["output_dir"],
            }
            print("COMMAND:", " ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=PROJECT, check=True)

    # merge with any existing manifest so stage-wise runs accumulate
    manifest_path = run_dir / "pilot_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing.setdefault("stages", {}).update(manifest["stages"])
        existing["updated_at"] = _utcnow()
        existing["code"] = git_commit(args.project_root)
        manifest = existing
    write_json(manifest_path, manifest)
    print(f"PILOT manifest -> {manifest_path}", flush=True)


def load_bundle(project_root: Path, data_root: Path):
    from r3.common import load_bundle as _load

    return _load(project_root, data_root)


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    main()
