from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path("/workspace/autoresearch/project/flexmopex/outputs/flex_mopex_v1")
OUTPUT_DIR = Path(
    "/workspace/autoresearch/project/flexmopex/analysis/flex_mopex_v1_npz_diagnostics"
)
MODEL_VERSION = "flex_mopex_v1"
NPZ_NAME = "model_outputs.npz"
WEIGHT_NAMES = ["w_phen", "w_int", "w_snow", "w_sub"]
REQUIRED_KEYS = ["streamflow", *WEIGHT_NAMES]
ALPHA_PATTERN = re.compile(r"^alpha_([0-9]+(?:[._][0-9]+)*)$")
DATA_RANGE_PATTERN = re.compile(r"^(?:test|train|valid)\d{4}-\d{4}(?:_Ep\d+)?$")
METRIC_NAMES = ["nse", "kge", "r2", "rmse", "pbias", "corr"]


def parse_alpha(path: Path) -> float | None:
    for part in path.parts:
        match = ALPHA_PATTERN.match(part)
        if match:
            return float(match.group(1).replace("_", "."))
    return None


def parse_data_range(path: Path) -> str:
    for part in reversed(path.parts):
        if DATA_RANGE_PATTERN.match(part):
            return part
    return str(path)


def find_npz_results(root_dir: Path) -> list[dict[str, object]]:
    records = []
    seen = set()
    for npz_path in sorted(root_dir.rglob(NPZ_NAME)):
        result_dir = npz_path.parent.resolve()
        if result_dir in seen:
            continue
        alpha = parse_alpha(result_dir)
        if alpha is None:
            continue
        seen.add(result_dir)
        records.append(
            {
                "model_version": MODEL_VERSION,
                "alpha": alpha,
                "result_dir": result_dir,
                "npz_path": npz_path.resolve(),
                "metrics_agg_path": result_dir / "metrics_agg.json",
                "data_range": parse_data_range(result_dir),
                "run_name": str(result_dir.relative_to(root_dir)),
            }
        )
    return sorted(records, key=lambda item: (float(item["alpha"]), str(item["run_name"])))


def reduce_to_station(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return values.mean(axis=0)
    if values.ndim == 3 and values.shape[2] == 1:
        return values.mean(axis=0).squeeze(axis=1)
    raise ValueError(f"Unsupported shape for station reduction: {values.shape}")


def temporal_deviation(values: np.ndarray) -> float:
    if values.ndim == 1:
        return 0.0
    if values.ndim == 2:
        reference = values[0, :]
        return float(np.nanmax(np.abs(values - reference)))
    if values.ndim == 3 and values.shape[2] == 1:
        reference = values[0, :, :]
        return float(np.nanmax(np.abs(values - reference)))
    raise ValueError(f"Unsupported shape for temporal diagnostics: {values.shape}")


def load_metric_rows(record: dict[str, object]) -> list[dict[str, object]]:
    metrics_path = Path(record["metrics_agg_path"])
    if not metrics_path.is_file():
        return []

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    rows = []
    for metric_name in METRIC_NAMES:
        if metric_name in metrics:
            metric_values = metrics[metric_name]
            rows.append(
                {
                    "model_version": record["model_version"],
                    "alpha": record["alpha"],
                    "run_name": record["run_name"],
                    "metric_name": metric_name,
                    "median": metric_values.get("median", np.nan),
                    "mean": metric_values.get("mean", np.nan),
                    "std": metric_values.get("std", np.nan),
                }
            )
    return rows


def inspect_npz_record(
    record: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    inventory_rows = []
    weight_rows = []
    temporal_rows = []

    with np.load(record["npz_path"]) as data:
        keys = list(data.files)
        missing_keys = sorted(set(REQUIRED_KEYS) - set(keys))
        unexpected_keys = sorted(set(keys) - set(REQUIRED_KEYS))

        for key in keys:
            values = data[key]
            finite_mask = np.isfinite(values)
            n_total = int(values.size)
            n_nonfinite = int(n_total - finite_mask.sum())
            finite_values = values[finite_mask]
            if finite_values.size:
                min_value = float(finite_values.min())
                max_value = float(finite_values.max())
                mean_value = float(finite_values.mean())
                std_value = float(finite_values.std())
            else:
                min_value = np.nan
                max_value = np.nan
                mean_value = np.nan
                std_value = np.nan

            inventory_rows.append(
                {
                    "model_version": record["model_version"],
                    "alpha": record["alpha"],
                    "run_name": record["run_name"],
                    "data_range": record["data_range"],
                    "npz_path": str(record["npz_path"]),
                    "key": key,
                    "shape": "x".join(str(dim) for dim in values.shape),
                    "dtype": str(values.dtype),
                    "n_total": n_total,
                    "n_nonfinite": n_nonfinite,
                    "min": min_value,
                    "max": max_value,
                    "mean": mean_value,
                    "std": std_value,
                    "missing_required_keys": ",".join(missing_keys),
                    "unexpected_keys": ",".join(unexpected_keys),
                    "file_size_mb": Path(record["npz_path"]).stat().st_size / 1024**2,
                }
            )

            if key in WEIGHT_NAMES:
                temporal_rows.append(
                    {
                        "model_version": record["model_version"],
                        "alpha": record["alpha"],
                        "run_name": record["run_name"],
                        "weight_name": key,
                        "shape": "x".join(str(dim) for dim in values.shape),
                        "max_abs_temporal_deviation": temporal_deviation(values),
                        "n_nonfinite": n_nonfinite,
                        "n_lt_0": int((values < 0).sum()),
                        "n_gt_1": int((values > 1).sum()),
                    }
                )

                station_values = np.asarray(reduce_to_station(values), dtype=float)
                if station_values.ndim != 1:
                    raise ValueError(
                        f"{record['npz_path']}:{key} reduced to {station_values.shape}, expected [B]"
                    )
                for station_index, weight_value in enumerate(station_values):
                    weight_rows.append(
                        {
                            "model_version": record["model_version"],
                            "alpha": record["alpha"],
                            "run_name": record["run_name"],
                            "result_dir": str(record["result_dir"]),
                            "npz_path": str(record["npz_path"]),
                            "weight_name": key,
                            "station_index": station_index,
                            "weight_value": float(weight_value),
                        }
                    )

    return inventory_rows, weight_rows, temporal_rows


def build_summary_tables(
    weights_long: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = weights_long.groupby(["model_version", "alpha", "weight_name"], sort=True)[
        "weight_value"
    ]
    summary = grouped.agg(
        n_station="count",
        mean="mean",
        median="median",
        std="std",
        q05=lambda values: values.quantile(0.05),
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
        q95=lambda values: values.quantile(0.95),
        min="min",
        max="max",
        frac_lt_0p01=lambda values: (values < 0.01).mean(),
        frac_lt_0p05=lambda values: (values < 0.05).mean(),
        frac_lt_0p10=lambda values: (values < 0.10).mean(),
        frac_gt_0p90=lambda values: (values > 0.90).mean(),
        frac_gt_0p95=lambda values: (values > 0.95).mean(),
        frac_gt_0p99=lambda values: (values > 0.99).mean(),
    ).reset_index()

    basin = (
        weights_long.pivot_table(
            index=["model_version", "alpha", "run_name", "station_index"],
            columns="weight_name",
            values="weight_value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    basin = basin[["model_version", "alpha", "run_name", "station_index", *WEIGHT_NAMES]]
    basin["mean_weight"] = basin[WEIGHT_NAMES].mean(axis=1)
    basin["sum_weight"] = basin[WEIGHT_NAMES].sum(axis=1)
    basin["n_active_0p5"] = (basin[WEIGHT_NAMES] > 0.5).sum(axis=1)
    basin["n_active_0p8"] = (basin[WEIGHT_NAMES] > 0.8).sum(axis=1)
    basin["n_inactive_0p1"] = (basin[WEIGHT_NAMES] < 0.1).sum(axis=1)

    complexity = (
        basin.groupby(["model_version", "alpha"], sort=True)
        .agg(
            n_station=("station_index", "count"),
            mean_sum_weight=("sum_weight", "mean"),
            median_sum_weight=("sum_weight", "median"),
            q05_sum_weight=("sum_weight", lambda values: values.quantile(0.05)),
            q25_sum_weight=("sum_weight", lambda values: values.quantile(0.25)),
            q75_sum_weight=("sum_weight", lambda values: values.quantile(0.75)),
            q95_sum_weight=("sum_weight", lambda values: values.quantile(0.95)),
            mean_weight=("mean_weight", "mean"),
            mean_n_active_0p5=("n_active_0p5", "mean"),
            mean_n_active_0p8=("n_active_0p8", "mean"),
            mean_n_inactive_0p1=("n_inactive_0p1", "mean"),
        )
        .reset_index()
    )
    return summary, basin, complexity


def format_alpha(alpha: float) -> str:
    return f"{alpha:g}"


def set_alpha_axis(ax: plt.Axes, alphas: pd.Series) -> None:
    if len(alphas) > 0 and (alphas > 0).all():
        ax.set_xscale("log")


def plot_weight_metric(summary: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for weight_name in WEIGHT_NAMES:
        group = summary[summary["weight_name"] == weight_name].sort_values("alpha")
        ax.plot(group["alpha"], group[metric], marker="o", linewidth=1.8, label=weight_name)
    positive_alpha = summary[summary["alpha"] > 0]["alpha"]
    set_alpha_axis(ax, positive_alpha)
    ax.set_xlabel("AIC penalty alpha")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{MODEL_VERSION}: {ylabel} by alpha")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_complexity(complexity: pd.DataFrame, output_path: Path) -> None:
    group = complexity.sort_values("alpha")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(group["alpha"], group["mean_sum_weight"], marker="o", label="mean sum_weight")
    ax.plot(
        group["alpha"],
        group["median_sum_weight"],
        marker="s",
        label="median sum_weight",
    )
    positive_alpha = group[group["alpha"] > 0]["alpha"]
    set_alpha_axis(ax, positive_alpha)
    ax.set_xlabel("AIC penalty alpha")
    ax.set_ylabel("sum of four structure weights")
    ax.set_title(f"{MODEL_VERSION}: basin-level complexity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_weight_distribution(weights_long: pd.DataFrame, output_path: Path) -> None:
    alphas = sorted(weights_long["alpha"].unique())
    labels = [format_alpha(alpha) for alpha in alphas]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey=True)
    for ax, weight_name in zip(axes.ravel(), WEIGHT_NAMES):
        group = weights_long[weights_long["weight_name"] == weight_name]
        values = [
            group[group["alpha"] == alpha]["weight_value"].to_numpy()
            for alpha in alphas
        ]
        ax.boxplot(values, tick_labels=labels, showfliers=False)
        ax.set_title(weight_name)
        ax.set_xlabel("alpha")
        ax.set_ylabel("station weight")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"{MODEL_VERSION}: station weight distributions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric_name in zip(axes, ["nse", "kge"]):
        group = metrics[metrics["metric_name"] == metric_name].sort_values("alpha")
        ax.plot(group["alpha"], group["median"], marker="o", label=f"{metric_name} median")
        positive_alpha = group[group["alpha"] > 0]["alpha"]
        set_alpha_axis(ax, positive_alpha)
        ax.set_xlabel("AIC penalty alpha")
        ax.set_ylabel(metric_name)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"{MODEL_VERSION}: selected performance diagnostics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def describe_endpoint_trend(frame: pd.DataFrame, value_col: str) -> str:
    group = frame.sort_values("alpha")
    if len(group) < 2:
        return "only one alpha level"
    first_alpha = float(group["alpha"].iloc[0])
    last_alpha = float(group["alpha"].iloc[-1])
    first_value = float(group[value_col].iloc[0])
    last_value = float(group[value_col].iloc[-1])
    deltas = np.diff(group[value_col].to_numpy(dtype=float))
    decreases = int((deltas < 0).sum())
    increases = int((deltas > 0).sum())
    direction = "decreases" if last_value < first_value else "increases or stays higher"
    return (
        f"{direction}; {format_alpha(first_alpha)}->{format_alpha(last_alpha)}: "
        f"{first_value:.4g}->{last_value:.4g}, decreases={decreases}, increases={increases}"
    )


def write_report(
    output_path: Path,
    records: list[dict[str, object]],
    inventory: pd.DataFrame,
    summary: pd.DataFrame,
    complexity: pd.DataFrame,
    temporal: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    alpha_text = ", ".join(format_alpha(alpha) for alpha in sorted(summary["alpha"].unique()))
    lines = [
        "# Flex-MOPEX V1 NPZ Data Diagnostic Report",
        "",
        f"- Input root: `{ROOT_DIR}`",
        f"- Output directory: `{OUTPUT_DIR}`",
        f"- Discovered `model_outputs.npz` files with alpha: {len(records)}",
        f"- Alpha values: {alpha_text}",
        f"- Required keys: {', '.join(REQUIRED_KEYS)}",
        "",
        "## NPZ Inventory",
        "",
    ]

    missing = inventory[inventory["missing_required_keys"].astype(str) != ""]
    unexpected = inventory[inventory["unexpected_keys"].astype(str) != ""]
    nonfinite = inventory[inventory["n_nonfinite"] > 0]
    out_of_range = temporal[(temporal["n_lt_0"] > 0) | (temporal["n_gt_1"] > 0)]

    lines.append(f"- Inventory rows: {len(inventory)}")
    lines.append(f"- Missing required-key rows: {len(missing)}")
    lines.append(f"- Unexpected-key rows: {len(unexpected)}")
    lines.append(f"- Non-finite value rows: {len(nonfinite)}")
    lines.append(f"- Weight out-of-[0,1] rows: {len(out_of_range)}")
    lines.append("")
    lines.append("## Shape Diagnostics")
    lines.append("")
    for key, group in inventory.groupby("key", sort=True):
        shapes = ", ".join(sorted(group["shape"].unique()))
        dtypes = ", ".join(sorted(group["dtype"].unique()))
        lines.append(f"- {key}: shapes={shapes}; dtypes={dtypes}")

    lines.extend(["", "## Temporal Constancy Diagnostics", ""])
    max_temporal = float(temporal["max_abs_temporal_deviation"].max())
    lines.append(
        f"- Maximum absolute temporal deviation among weight arrays: {max_temporal:.6g}"
    )
    if max_temporal == 0:
        lines.append("- All weight arrays are exactly constant along the time dimension.")
    else:
        worst = temporal.sort_values("max_abs_temporal_deviation", ascending=False).head(5)
        for _, row in worst.iterrows():
            lines.append(
                f"- alpha={format_alpha(row['alpha'])} {row['weight_name']}: "
                f"max_abs_temporal_deviation={row['max_abs_temporal_deviation']:.6g}"
            )

    lines.extend(["", "## Alpha Trends", ""])
    for weight_name in WEIGHT_NAMES:
        group = summary[summary["weight_name"] == weight_name]
        lines.append(f"- {weight_name} mean: {describe_endpoint_trend(group, 'mean')}")
        high_alpha = group.sort_values("alpha").iloc[-1]
        lines.append(
            f"  - highest alpha mean={high_alpha['mean']:.4g}, "
            f"frac_lt_0p10={high_alpha['frac_lt_0p10']:.4g}, "
            f"frac_gt_0p90={high_alpha['frac_gt_0p90']:.4g}"
        )

    lines.extend(["", "## Basin Complexity", ""])
    lines.append(f"- sum_weight trend: {describe_endpoint_trend(complexity, 'mean_sum_weight')}")
    for _, row in complexity.sort_values("alpha").iterrows():
        lines.append(
            f"- alpha={format_alpha(row['alpha'])}: mean_sum_weight={row['mean_sum_weight']:.4g}, "
            f"median_sum_weight={row['median_sum_weight']:.4g}, "
            f"mean_n_active_0p5={row['mean_n_active_0p5']:.4g}, "
            f"mean_n_inactive_0p1={row['mean_n_inactive_0p1']:.4g}"
        )

    lines.extend(["", "## Selected Metric Diagnostics", ""])
    if metrics.empty:
        lines.append("- No metrics_agg.json rows were loaded.")
    else:
        for metric_name in ["nse", "kge", "r2", "rmse"]:
            group = metrics[metrics["metric_name"] == metric_name]
            if not group.empty:
                lines.append(
                    f"- {metric_name} median: {describe_endpoint_trend(group, 'median')}"
                )

    lines.extend(["", "## Potential Data Issues", ""])
    issue_lines = []
    if len(records) == 0:
        issue_lines.append("- No alpha-specific npz files were found.")
    if len(missing) > 0:
        issue_lines.append("- Some npz rows are missing required keys; inspect `npz_inventory.csv`.")
    if len(nonfinite) > 0:
        issue_lines.append("- Non-finite values detected; inspect `npz_inventory.csv`.")
    if len(out_of_range) > 0:
        issue_lines.append("- Weight values outside [0, 1] detected; inspect `temporal_weight_diagnostics.csv`.")
    near_zero = summary[(summary["alpha"] >= 0.1) & (summary["mean"] < 1e-4)]
    if not near_zero.empty:
        names = ", ".join(
            f"alpha={format_alpha(row['alpha'])}:{row['weight_name']}"
            for _, row in near_zero.iterrows()
        )
        issue_lines.append(f"- High-alpha weights are effectively zero for: {names}.")
    if max_temporal > 0:
        issue_lines.append("- At least one weight array varies along time; V1 normally repeats station weights over time.")
    if not issue_lines:
        issue_lines.append("- No missing keys, non-finite values, out-of-range weights, or shape mismatches were detected.")
    lines.extend(issue_lines)

    lines.extend(["", "## Output Files", ""])
    for file_name in [
        "npz_inventory.csv",
        "weights_long.csv",
        "weight_alpha_summary.csv",
        "basin_complexity.csv",
        "complexity_alpha_summary.csv",
        "temporal_weight_diagnostics.csv",
        "metrics_alpha_summary.csv",
        "fig_alpha_mean_weights.png",
        "fig_alpha_median_weights.png",
        "fig_alpha_zero_fraction.png",
        "fig_alpha_one_fraction.png",
        "fig_alpha_complexity.png",
        "fig_weight_distribution_by_alpha.png",
        "fig_metrics_nse_kge_by_alpha.png",
        "data_diagnostic_report.md",
    ]:
        lines.append(f"- `{file_name}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = find_npz_results(ROOT_DIR)
    inventory_rows = []
    weight_rows = []
    temporal_rows = []
    metric_rows = []

    for record in records:
        inv, weights, temporal = inspect_npz_record(record)
        inventory_rows.extend(inv)
        weight_rows.extend(weights)
        temporal_rows.extend(temporal)
        metric_rows.extend(load_metric_rows(record))

    inventory = pd.DataFrame(inventory_rows)
    weights_long = pd.DataFrame(weight_rows)
    temporal = pd.DataFrame(temporal_rows)
    metrics = pd.DataFrame(metric_rows)
    summary, basin, complexity = build_summary_tables(weights_long)

    inventory.to_csv(OUTPUT_DIR / "npz_inventory.csv", index=False)
    weights_long.to_csv(OUTPUT_DIR / "weights_long.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "weight_alpha_summary.csv", index=False)
    basin.to_csv(OUTPUT_DIR / "basin_complexity.csv", index=False)
    complexity.to_csv(OUTPUT_DIR / "complexity_alpha_summary.csv", index=False)
    temporal.to_csv(OUTPUT_DIR / "temporal_weight_diagnostics.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "metrics_alpha_summary.csv", index=False)

    plot_weight_metric(summary, "mean", "mean station weight", OUTPUT_DIR / "fig_alpha_mean_weights.png")
    plot_weight_metric(
        summary,
        "median",
        "median station weight",
        OUTPUT_DIR / "fig_alpha_median_weights.png",
    )
    plot_weight_metric(
        summary,
        "frac_lt_0p10",
        "fraction below 0.10",
        OUTPUT_DIR / "fig_alpha_zero_fraction.png",
    )
    plot_weight_metric(
        summary,
        "frac_gt_0p90",
        "fraction above 0.90",
        OUTPUT_DIR / "fig_alpha_one_fraction.png",
    )
    plot_complexity(complexity, OUTPUT_DIR / "fig_alpha_complexity.png")
    plot_weight_distribution(weights_long, OUTPUT_DIR / "fig_weight_distribution_by_alpha.png")
    plot_metrics(metrics, OUTPUT_DIR / "fig_metrics_nse_kge_by_alpha.png")

    write_report(
        OUTPUT_DIR / "data_diagnostic_report.md",
        records,
        inventory,
        summary,
        complexity,
        temporal,
        metrics,
    )

    print(f"Discovered npz files: {len(records)}")
    print("Alpha values:")
    print("  " + ", ".join(format_alpha(record["alpha"]) for record in records))
    print(f"Station-level weight rows: {len(weights_long)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Complexity trend:")
    print(f"  {describe_endpoint_trend(complexity, 'mean_sum_weight')}")
    print("Diagnostic report:")
    print(f"  {OUTPUT_DIR / 'data_diagnostic_report.md'}")


if __name__ == "__main__":
    main()
