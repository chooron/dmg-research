from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

MULTISEED_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MULTISEED_DIR.parent
REPO_ROOT = PROJECT_DIR.parent.parent

for path in (REPO_ROOT, PROJECT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

DEFAULT_SEEDS = (111, 222, 333, 444, 555)
TARGET_METRICS = ("nse", "kge")


@dataclass(frozen=True)
class ModelRun:
    config_path: Path
    model: str
    output_root: Path
    split_epoch: int | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize bettermodel multiseed NSE/KGE results into CSV files.",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help=(
            "Config paths to include. Defaults to configs inferred from "
            "project/bettermodel/scripts/multiseed."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seed list to summarize. Default: 111 222 333 444 555.",
    )
    parser.add_argument(
        "--split",
        choices=("test", "train"),
        default="test",
        help="Which evaluation split directory to summarize. Default: test.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(MULTISEED_DIR / "results"),
        help="Directory for CSV and figure outputs.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip generating the summary boxplot figure.",
    )
    return parser.parse_args(argv)


def discover_default_configs() -> list[Path]:
    scripts_dir = PROJECT_DIR / "scripts" / "multiseed"
    config_paths: list[Path] = []
    for script_path in sorted(scripts_dir.glob("run_*_multiseed.sh")):
        stem = script_path.stem
        if not stem.startswith("run_") or not stem.endswith("_multiseed"):
            continue
        config_name = f"config_{stem[len('run_'):-len('_multiseed')]}.yaml"
        config_path = PROJECT_DIR / "conf" / config_name
        if config_path.exists():
            config_paths.append(config_path.resolve())
    return config_paths


def resolve_config_paths(config_args: list[str] | None) -> list[Path]:
    if not config_args:
        config_paths = discover_default_configs()
        if not config_paths:
            raise FileNotFoundError("No multiseed configs discovered under scripts/multiseed.")
        return config_paths

    resolved: list[Path] = []
    for config_arg in config_args:
        path = Path(config_arg).expanduser()
        if not path.is_absolute():
            candidates = (
                Path.cwd() / path,
                PROJECT_DIR / path,
                PROJECT_DIR / "conf" / path,
            )
            for candidate in candidates:
                if candidate.exists():
                    path = candidate
                    break
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_arg}")
        resolved.append(path.resolve())
    return resolved


def load_model_runs(config_paths: list[Path], split: str) -> list[ModelRun]:
    model_runs: list[ModelRun] = []
    for config_path in config_paths:
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        if not isinstance(config, dict):
            raise TypeError(f"Expected mapping config, got {type(config)!r}: {config_path}")
        model = str(config["paper"]["variant"])
        output_root = (config_path.parent.parent / str(config["output_dir"])).resolve().parent
        split_cfg = config.get(split, {})
        split_epoch = split_cfg.get("test_epoch") if isinstance(split_cfg, dict) else None
        model_runs.append(
            ModelRun(
                config_path=config_path,
                model=model,
                output_root=output_root,
                split_epoch=split_epoch,
            )
        )
    return model_runs


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(payload)!r}")
    return payload


def clean_values(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def compute_stats(values: Any) -> dict[str, float | int]:
    array = clean_values(values)
    if array.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
    }


def pick_metrics_dir(seed_dir: Path, split: str, split_epoch: int | None) -> tuple[Path | None, str | None]:
    if not seed_dir.exists():
        return None, "seed_dir_missing"

    candidates = sorted(
        path
        for path in seed_dir.iterdir()
        if path.is_dir() and path.name.startswith(split)
    )
    if split_epoch is not None:
        epoch_candidates = [path for path in candidates if f"Ep{split_epoch}" in path.name]
        if epoch_candidates:
            candidates = epoch_candidates

    candidates = [
        path
        for path in candidates
        if (path / "metrics.json").exists() and (path / "metrics_agg.json").exists()
    ]
    if not candidates:
        return None, "metrics_files_missing"
    if len(candidates) > 1:
        return candidates[-1], f"multiple_{split}_dirs"
    return candidates[0], None


def flatten_row(row: dict[str, Any], metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    flat = dict(row)
    for metric_name, metric_stats in metrics.items():
        for stat_name, value in metric_stats.items():
            flat[f"{metric_name}_{stat_name}"] = value
    return flat


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot_boxplots(
    per_model_values: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib_not_available"

    models = [
        model
        for model in sorted(per_model_values)
        if all(per_model_values[model][metric].size > 0 for metric in TARGET_METRICS)
    ]
    if not models:
        return "no_plot_data"

    fig_width = max(10, 1.8 * len(models))
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5.2), constrained_layout=True)
    for axis, metric in zip(axes, TARGET_METRICS):
        data = [per_model_values[model][metric] for model in models]
        axis.boxplot(data, showfliers=False)
        axis.set_title(metric.upper())
        axis.set_ylabel(metric.upper())
        axis.set_xticks(range(1, len(models) + 1))
        axis.set_xticklabels(models, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.3)

    fig.suptitle("Combined multiseed basin-level distributions", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return None


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results_dir = Path(args.results_dir).expanduser().resolve()
    config_paths = resolve_config_paths(args.configs)
    model_runs = load_model_runs(config_paths, split=args.split)

    per_seed_rows: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    basin_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for model_run in model_runs:
        seed_metric_values: dict[str, list[np.ndarray]] = {metric: [] for metric in TARGET_METRICS}
        metric_by_seed: dict[str, list[np.ndarray]] = {metric: [] for metric in TARGET_METRICS}
        basin_count: int | None = None
        completed_seed_count = 0

        for seed in args.seeds:
            seed_dir = model_run.output_root / f"seed_{seed}"
            metrics_dir, issue = pick_metrics_dir(seed_dir, split=args.split, split_epoch=model_run.split_epoch)
            if metrics_dir is None:
                missing_rows.append(
                    {
                        "model": model_run.model,
                        "seed": seed,
                        "split": args.split,
                        "issue": issue,
                    }
                )
                continue

            if issue is not None:
                warnings.append(
                    {
                        "model": model_run.model,
                        "seed": seed,
                        "split": args.split,
                        "issue": issue,
                    }
                )

            metrics_agg_path = metrics_dir / "metrics_agg.json"
            metrics_path = metrics_dir / "metrics.json"
            metrics_agg = read_json(metrics_agg_path)
            metrics = read_json(metrics_path)

            metric_lengths: dict[str, int] = {}
            per_seed_metric_stats: dict[str, dict[str, Any]] = {}
            missing_metric = False
            for metric in TARGET_METRICS:
                if metric not in metrics_agg or metric not in metrics:
                    missing_rows.append(
                        {
                            "model": model_run.model,
                            "seed": seed,
                            "split": args.split,
                            "issue": f"{metric}_missing_in_metrics",
                        }
                    )
                    missing_metric = True
                    break

                values = clean_values(metrics[metric])
                metric_lengths[metric] = int(values.size)
                per_seed_metric_stats[metric] = {
                    "mean": metrics_agg[metric]["mean"],
                    "median": metrics_agg[metric]["median"],
                    "std": metrics_agg[metric]["std"],
                    "count": int(values.size),
                }
                seed_metric_values[metric].append(values)
                metric_by_seed[metric].append(values)

            if missing_metric:
                continue

            completed_seed_count += 1
            basin_count = metric_lengths[TARGET_METRICS[0]]
            row = {
                "model": model_run.model,
                "seed": seed,
                "split": args.split,
                "basin_count": basin_count,
            }
            per_seed_rows.append(flatten_row(row, per_seed_metric_stats))

        if completed_seed_count == 0:
            continue

        combined_metric_stats: dict[str, dict[str, Any]] = {}
        for metric in TARGET_METRICS:
            combined_values = np.concatenate(seed_metric_values[metric])
            combined_metric_stats[metric] = compute_stats(combined_values)
            seed_metric_values[metric] = [combined_values]

        combined_rows.append(
            flatten_row(
                {
                    "model": model_run.model,
                    "split": args.split,
                    "seed_count": completed_seed_count,
                    "total_records": combined_metric_stats[TARGET_METRICS[0]]["count"],
                    "basin_count_per_seed": basin_count,
                },
                combined_metric_stats,
            )
        )

        aligned_lengths = {metric: {len(values) for values in metric_by_seed[metric]} for metric in TARGET_METRICS}
        if all(len(lengths) == 1 for lengths in aligned_lengths.values()):
            stacked_by_metric = {
                metric: np.vstack(metric_by_seed[metric]) for metric in TARGET_METRICS
            }
            n_basins = stacked_by_metric[TARGET_METRICS[0]].shape[1]
            for basin_index in range(n_basins):
                basin_metric_stats = {
                    metric: compute_stats(stacked_by_metric[metric][:, basin_index])
                    for metric in TARGET_METRICS
                }
                basin_rows.append(
                    flatten_row(
                        {
                            "model": model_run.model,
                            "split": args.split,
                            "basin_index": basin_index,
                            "seed_count": completed_seed_count,
                        },
                        basin_metric_stats,
                    )
                )
        else:
            warnings.append(
                {
                    "model": model_run.model,
                    "seed": "all",
                    "split": args.split,
                    "issue": "inconsistent_basin_count_across_seeds",
                }
            )

    per_seed_rows.sort(key=lambda row: (row["model"], int(row["seed"])))
    combined_rows.sort(key=lambda row: row["model"])
    basin_rows.sort(key=lambda row: (row["model"], int(row["basin_index"])))
    missing_rows.sort(key=lambda row: (row["model"], int(row["seed"])))
    warnings.sort(key=lambda row: (row["model"], str(row["seed"])))

    per_seed_path = results_dir / f"{args.split}_per_seed_metrics.csv"
    combined_path = results_dir / f"{args.split}_combined_multiseed_metrics.csv"
    basin_path = results_dir / f"{args.split}_per_basin_multiseed_metrics.csv"
    missing_path = results_dir / f"{args.split}_missing_runs.csv"
    warnings_path = results_dir / f"{args.split}_warnings.csv"

    write_csv(per_seed_path, per_seed_rows)
    write_csv(combined_path, combined_rows)
    write_csv(basin_path, basin_rows)
    write_csv(missing_path, missing_rows)
    write_csv(warnings_path, warnings)

    plot_issue: str | None = None
    plot_path = results_dir / f"{args.split}_combined_boxplots.png"
    if not args.skip_plot:
        per_model_values = {}
        for row in combined_rows:
            model = row["model"]
            model_values = {
                metric: np.array([], dtype=float)
                for metric in TARGET_METRICS
            }
            per_model_values[model] = model_values

        for row in basin_rows:
            model = row["model"]
            for metric in TARGET_METRICS:
                per_model_values.setdefault(
                    model,
                    {name: np.array([], dtype=float) for name in TARGET_METRICS},
                )

        for model_run in model_runs:
            model = model_run.model
            metric_values = {metric: [] for metric in TARGET_METRICS}
            for seed in args.seeds:
                seed_dir = model_run.output_root / f"seed_{seed}"
                metrics_dir, issue = pick_metrics_dir(seed_dir, split=args.split, split_epoch=model_run.split_epoch)
                if metrics_dir is None:
                    continue
                metrics = read_json(metrics_dir / "metrics.json")
                for metric in TARGET_METRICS:
                    if metric in metrics:
                        metric_values[metric].append(clean_values(metrics[metric]))
            if all(metric_values[metric] for metric in TARGET_METRICS):
                per_model_values[model] = {
                    metric: np.concatenate(metric_values[metric]) for metric in TARGET_METRICS
                }

        plot_issue = maybe_plot_boxplots(per_model_values, plot_path)

    print(f"per-seed summary: {per_seed_path}")
    print(f"combined summary: {combined_path}")
    print(f"per-basin summary: {basin_path}")
    print(f"missing runs: {missing_path}")
    print(f"warnings: {warnings_path}")
    if not args.skip_plot:
        if plot_issue is None:
            print(f"boxplot: {plot_path}")
        else:
            print(f"boxplot skipped: {plot_issue}")


if __name__ == "__main__":
    main()
