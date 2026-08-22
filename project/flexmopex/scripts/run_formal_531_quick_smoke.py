"""Lightweight local smoke matrix for all distinct formal execution paths.

Each representative command runs one training epoch. Seeds do not change
command wiring, so the smoke matrix uses seed 42 except for one reference
seed43 check. Formal full results are never touched.
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


def specs() -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = [
        (
            "reference_seed43",
            [
                "--config", "conf/config_formal_531_flex_lambda0007_seed43.yaml",
                "--model-type", "flex", "--alpha", "0.007", "--seed", "43",
            ],
        )
    ]
    for process in ("w_snow", "w_sub", "w_phen", "w_int"):
        jobs.append(
            (
                f"lopo_{process}",
                [
                    "--config", "conf/config_formal_531_loro.yaml",
                    "--model-type", "flex", "--alpha", "0.007", "--seed", "42",
                    "--removed-process", process,
                ],
            )
        )
    for nmul in (1, 8, 16, 32):
        jobs.append(
            (
                f"nmul_{nmul}",
                [
                    "--config", "conf/config_formal_531_flex_lambda0007.yaml",
                    "--model-type", "flex", "--alpha", "0.007", "--nmul", str(nmul),
                    "--seed", "42",
                ],
            )
        )
    for region in range(7):
        for model_type in ("base", "full", "flex"):
            alpha = "0.007" if model_type == "flex" else "0.0"
            jobs.append(
                (
                    f"loro_{model_type}_region{region}",
                    [
                        "--config", "conf/config_formal_531_loro.yaml",
                        "--model-type", model_type, "--alpha", alpha, "--seed", "42",
                        "--loro-holdout-region", str(region),
                    ],
                )
            )
    assert len(jobs) == 30, len(jobs)
    return jobs


def run_job(label: str, args: list[str], root: Path, log_dir: Path) -> dict:
    run_name = label.replace("/", "__")
    log_path = log_dir / f"{run_name}.log"
    command = [
        str(PYTHON), "run_model.py", "--mode", "train", "--gpu-id", "0",
        "--output-root", str(root), "--run-name", run_name,
        "--epochs", "1", "--min-epochs", "1", "--disable-early-stopping",
        *args,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=PROJECT_DIR,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                timeout=900,
                check=False,
            )
        return {"label": label, "returncode": result.returncode, "log": str(log_path)}
    except subprocess.TimeoutExpired:
        return {"label": label, "returncode": 124, "error": "timeout", "log": str(log_path)}
    except OSError as exc:
        return {"label": label, "returncode": 125, "error": repr(exc), "log": str(log_path)}


def check_loro_groups() -> list[dict]:
    sys.path.insert(0, str(REPO_ROOT))
    from project.flexmopex import load_config
    from project.flexmopex.models.pub_sampler import PubSampler
    from project.flexmopex.run_model import _resolve_config, apply_runtime_overrides, parse_args

    expected_holdout = (92, 95, 93, 51, 68, 60, 72)
    checks = []
    for region, expected in enumerate(expected_holdout):
        args = parse_args([
            "--config", "conf/config_formal_531_loro.yaml", "--mode", "train",
            "--model-type", "flex", "--alpha", "0.007", "--seed", "42",
            "--loro-holdout-region", str(region), "--disable-early-stopping",
        ])
        config_path = _resolve_config(args.config)
        config = load_config(config_path)
        apply_runtime_overrides(config, args, config_path=config_path)
        sampler = PubSampler(config)
        actual = {"region": region, "train": len(sampler.train_indices), "holdout": len(sampler.val_indices)}
        checks.append(actual)
        if actual["train"] + actual["holdout"] != 531 or actual["holdout"] != expected:
            raise AssertionError(actual)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=PROJECT_DIR / "results" / "smoke_formal_531_quick")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty smoke root: {root}")
    root.mkdir(parents=True)
    results = []
    for label, command_args in specs():
        print(f"SMOKE_START {label}", flush=True)
        result = run_job(label, command_args, root / "runs", root / "logs")
        results.append(result)
        print(f"SMOKE_DONE {label} rc={result['returncode']}", flush=True)
    groups = check_loro_groups()
    summary = {"jobs": len(results), "failed": [r for r in results if r["returncode"] != 0], "loro_groups": groups, "results": results}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"jobs": len(results), "failed": len(summary["failed"]), "root": str(root)}, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
