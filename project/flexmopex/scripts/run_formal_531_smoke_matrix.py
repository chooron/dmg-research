"""Run the local CAMELS-531 smoke matrix before any remote full run.

Every runnable formal job receives a real CLI preflight.  The four LOPO
removed-process variants additionally run one local training epoch, because
that path exercises counterfactual target construction and fixed-off masking.
No formal result directory is touched.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def job_specs() -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = []

    for seed in (43, 44):
        jobs.append(
            (
                f"reference/seed_{seed}",
                [
                    "--config", "conf/config_formal_531_flex_lambda0007_seed%s.yaml" % seed,
                    "--model-type", "flex", "--alpha", "0.007", "--seed", str(seed),
                ],
            )
        )
    for process in ("w_snow", "w_sub", "w_phen", "w_int"):
        for seed in (42, 43, 44):
            jobs.append(
                (
                    f"lopo/{process}/seed_{seed}",
                    [
                        "--config", "conf/config_formal_531_loro.yaml",
                        "--model-type", "flex", "--alpha", "0.007", "--seed", str(seed),
                        "--removed-process", process,
                    ],
                )
            )
    for nmul in (1, 8, 16, 32):
        for seed in (42, 43, 44):
            jobs.append(
                (
                    f"nmul/{nmul}/seed_{seed}",
                    [
                        "--config", "conf/config_formal_531_flex_lambda0007.yaml",
                        "--model-type", "flex", "--alpha", "0.007", "--nmul", str(nmul),
                        "--seed", str(seed),
                    ],
                )
            )
    for region in range(7):
        for model_type in ("base", "full", "flex"):
            for seed in (42, 43, 44):
                alpha = "0.007" if model_type == "flex" else "0.0"
                jobs.append(
                    (
                        f"loro/{model_type}/region{region}/seed_{seed}",
                        [
                            "--config", "conf/config_formal_531_loro.yaml",
                            "--model-type", model_type, "--alpha", alpha, "--seed", str(seed),
                            "--loro-holdout-region", str(region),
                        ],
                    )
                )
    assert len(jobs) == 89, len(jobs)
    return jobs


def run_one(
    label: str,
    args: list[str],
    output_root: Path,
    log_root: Path,
    *,
    train_epoch: bool = False,
) -> dict:
    run_name = label.replace("/", "__")
    log_path = log_root / f"{run_name}.log"
    command = [
        str(PYTHON), "run_model.py",
        "--mode", "train",
        "--gpu-id", "0",
        "--output-root", str(output_root),
        "--run-name", run_name,
        *args,
    ]
    if train_epoch:
        command += ["--epochs", "1", "--min-epochs", "1", "--disable-early-stopping"]
    else:
        command += ["--preflight-only"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as stream:
        result = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            timeout=900 if train_epoch else 600,
            check=False,
        )
    return {
        "label": label,
        "kind": "one_epoch_train" if train_epoch else "preflight",
        "returncode": result.returncode,
        "log": str(log_path),
        "command": command,
    }


def check_loro_sampler() -> list[dict]:
    sys.path.insert(0, str(REPO_ROOT))
    from project.flexmopex import load_config
    from project.flexmopex.models.pub_sampler import PubSampler
    from project.flexmopex.run_model import _resolve_config, apply_runtime_overrides, parse_args

    expected = (92, 95, 93, 51, 68, 60, 72)
    checks = []
    for region, expected_val in enumerate(expected):
        args = parse_args(
            [
                "--config", "conf/config_formal_531_loro.yaml", "--mode", "train",
                "--model-type", "flex", "--alpha", "0.007", "--seed", "42",
                "--loro-holdout-region", str(region), "--disable-early-stopping",
            ]
        )
        config_path = _resolve_config(args.config)
        config = load_config(config_path)
        apply_runtime_overrides(config, args, config_path=config_path)
        sampler = PubSampler(config)
        actual = len(sampler.val_indices)
        checks.append({"region": region, "train": len(sampler.train_indices), "holdout": actual})
        if actual != expected_val or len(sampler.train_indices) + actual != 531:
            raise AssertionError(f"LORO region {region}: got {len(sampler.train_indices)}/{actual}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=PROJECT_DIR / "results" / "smoke_formal_531_matrix")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty smoke root: {root}")
    preflight_root = root / "preflight"
    train_root = root / "one_epoch"
    preflight_logs = root / "logs" / "preflight"
    train_logs = root / "logs" / "one_epoch"
    root.mkdir(parents=True)

    results = []
    for label, command_args in job_specs():
        results.append(run_one(label, command_args, preflight_root, preflight_logs))

    for process in ("w_snow", "w_sub", "w_phen", "w_int"):
        results.append(
            run_one(
                f"lopo/{process}/one_epoch",
                [
                    "--config", "conf/config_formal_531_loro.yaml",
                    "--model-type", "flex", "--alpha", "0.007", "--seed", "42",
                    "--removed-process", process,
                ],
                train_root,
                train_logs,
                train_epoch=True,
            )
        )

    summary = {"preflight_jobs": 89, "one_epoch_lopo_jobs": 4, "results": results}
    summary["loro_sampler"] = check_loro_sampler()
    summary["failed"] = [r for r in results if r["returncode"] != 0]
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"root": str(root), "preflight": 89, "one_epoch_lopo": 4, "failed": len(summary["failed"])}, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
