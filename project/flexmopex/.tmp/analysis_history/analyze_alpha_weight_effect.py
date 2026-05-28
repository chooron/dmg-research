from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIRS = [
    Path("/workspace/autoresearch/project/flexmopex/output/flex_mopex_v1"),
    Path("/workspace/autoresearch/project/flexmopex/output/flex_mopex_v2"),
    Path("/workspace/autoresearch/project/flexmopex/output/flex_mopex_v3"),
]
OUTPUT_DIR = Path("/workspace/autoresearch/project/flexmopex/analysis/alpha_weight_effect")
WEIGHT_NAMES = ["w_int", "w_phen", "w_snow", "w_sub"]
WEIGHT_FILES = [f"{name}.npy" for name in WEIGHT_NAMES]
MODEL_VERSIONS = [root.name for root in ROOT_DIRS]
ALPHA_PATTERN = re.compile(r"^alpha_([0-9]+(?:[._][0-9]+)*)$")
DATA_RANGE_PATTERN = re.compile(r"^(?:test|train|valid)\d{4}-\d{4}(?:_Ep\d+)?$")


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


def find_weight_result_dirs(root_dirs: list[Path]) -> list[dict[str, object]]:
    records = []
    seen_dirs = set()

    for root in root_dirs:
        for weight_path in sorted(root.rglob("w_int.npy")):
            result_dir = weight_path.parent.resolve()
            if result_dir in seen_dirs:
                continue
            if not all((result_dir / file_name).is_file() for file_name in WEIGHT_FILES):
                continue

            alpha = parse_alpha(result_dir)
            if alpha is None:
                continue

            seen_dirs.add(result_dir)
            records.append(
                {
                    "result_dir": result_dir,
                    "model_version": root.name,
                    "alpha": alpha,
                    "data_range": parse_data_range(result_dir),
                    "run_name": str(result_dir.relative_to(root)),
                }
            )

    return sorted(
        records,
        key=lambda item: (
            str(item["model_version"]),
            float(item["alpha"]),
            str(item["run_name"]),
        ),
    )


def reduce_to_station(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return values.mean(axis=0)
    if values.ndim == 3 and values.shape[2] == 1:
        return values.mean(axis=0).squeeze(axis=1)
    raise ValueError(f"Unsupported weight shape: {values.shape}")


def load_weight_result(result_dir: Path, metadata: dict[str, object]) -> dict[str, object]:
    weights = {}
    n_station = None

    for weight_name in WEIGHT_NAMES:
        values = np.load(result_dir / f"{weight_name}.npy", mmap_mode="r")
        station_values = np.asarray(reduce_to_station(values), dtype=float)
        if station_values.ndim != 1:
            raise ValueError(
                f"{result_dir / f'{weight_name}.npy'} reduced to {station_values.shape}, expected [B]"
            )
        if n_station is None:
            n_station = station_values.shape[0]
        if station_values.shape[0] != n_station:
            raise ValueError(
                f"Station count mismatch in {result_dir}: {weight_name} has {station_values.shape[0]}, expected {n_station}"
            )
        weights[weight_name] = station_values

    return {**metadata, "weights": weights, "n_station": n_station}


def build_weights_long(records: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for record in records:
        for weight_name, station_values in record["weights"].items():
            for station_index, weight_value in enumerate(station_values):
                rows.append(
                    {
                        "model_version": record["model_version"],
                        "alpha": record["alpha"],
                        "run_name": record["run_name"],
                        "result_dir": str(record["result_dir"]),
                        "weight_name": weight_name,
                        "station_index": station_index,
                        "weight_value": float(weight_value),
                    }
                )

    columns = [
        "model_version",
        "alpha",
        "run_name",
        "result_dir",
        "weight_name",
        "station_index",
        "weight_value",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_summary_tables(weights_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if weights_long.empty:
        summary_columns = [
            "model_version",
            "alpha",
            "weight_name",
            "n_station",
            "mean",
            "median",
            "std",
            "q05",
            "q25",
            "q75",
            "q95",
            "min",
            "max",
            "frac_lt_0p05",
            "frac_lt_0p10",
            "frac_gt_0p90",
            "frac_gt_0p95",
        ]
        basin_columns = [
            "model_version",
            "alpha",
            "run_name",
            "station_index",
            *WEIGHT_NAMES,
            "mean_weight",
            "sum_weight",
            "n_active_0p5",
            "n_active_0p8",
        ]
        complexity_columns = [
            "model_version",
            "alpha",
            "n_station",
            "mean_sum_weight",
            "median_sum_weight",
            "q25_sum_weight",
            "q75_sum_weight",
            "mean_n_active_0p5",
            "mean_n_active_0p8",
            "mean_weight",
        ]
        return (
            pd.DataFrame(columns=summary_columns),
            pd.DataFrame(columns=basin_columns),
            pd.DataFrame(columns=complexity_columns),
        )

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
        frac_lt_0p05=lambda values: (values < 0.05).mean(),
        frac_lt_0p10=lambda values: (values < 0.10).mean(),
        frac_gt_0p90=lambda values: (values > 0.90).mean(),
        frac_gt_0p95=lambda values: (values > 0.95).mean(),
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

    complexity = (
        basin.groupby(["model_version", "alpha"], sort=True)
        .agg(
            n_station=("station_index", "count"),
            mean_sum_weight=("sum_weight", "mean"),
            median_sum_weight=("sum_weight", "median"),
            q25_sum_weight=("sum_weight", lambda values: values.quantile(0.25)),
            q75_sum_weight=("sum_weight", lambda values: values.quantile(0.75)),
            mean_n_active_0p5=("n_active_0p5", "mean"),
            mean_n_active_0p8=("n_active_0p8", "mean"),
            mean_weight=("mean_weight", "mean"),
        )
        .reset_index()
    )

    return summary, basin, complexity


def format_alpha(alpha: float) -> str:
    return f"{alpha:g}"


def set_alpha_axis_scale(ax: plt.Axes, alphas: pd.Series) -> None:
    if len(alphas) > 0 and (alphas > 0).all():
        ax.set_xscale("log")


def plot_metric_panels(
    summary_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(MODEL_VERSIONS), figsize=(5.8 * len(MODEL_VERSIONS), 4.6), sharey=True)
    if len(MODEL_VERSIONS) == 1:
        axes = [axes]

    for ax, model_version in zip(axes, MODEL_VERSIONS):
        model_df = summary_df[summary_df["model_version"] == model_version]
        if model_df.empty:
            ax.text(0.5, 0.5, "No complete weight result", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(model_version)
            ax.set_xlabel("alpha")
            ax.set_ylabel(ylabel)
            continue

        for weight_name in WEIGHT_NAMES:
            line_df = model_df[model_df["weight_name"] == weight_name].sort_values("alpha")
            ax.plot(line_df["alpha"], line_df[metric], marker="o", linewidth=1.8, label=weight_name)
        set_alpha_axis_scale(ax, model_df["alpha"])
        ax.set_title(model_version)
        ax.set_xlabel("alpha")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_complexity(complexity_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    plotted = False

    for model_version in MODEL_VERSIONS:
        model_df = complexity_df[complexity_df["model_version"] == model_version].sort_values("alpha")
        if model_df.empty:
            continue
        plotted = True
        ax.plot(
            model_df["alpha"],
            model_df["mean_sum_weight"],
            marker="o",
            linewidth=1.9,
            label=model_version,
        )

    if plotted:
        set_alpha_axis_scale(ax, complexity_df["alpha"])
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No complete weight result", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Alpha Effect on Basin-Level Complexity")
    ax.set_xlabel("alpha")
    ax.set_ylabel("mean sum weight")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_by_model(weights_long: pd.DataFrame, model_version: str, output_path: Path) -> None:
    model_df = weights_long[weights_long["model_version"] == model_version]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=False, sharey=True)
    flat_axes = axes.ravel()

    if model_df.empty:
        for ax in flat_axes:
            ax.axis("off")
        fig.text(0.5, 0.5, f"No complete weight result for {model_version}", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    alpha_values = sorted(model_df["alpha"].unique())
    labels = [format_alpha(alpha) for alpha in alpha_values]

    for ax, weight_name in zip(flat_axes, WEIGHT_NAMES):
        weight_df = model_df[model_df["weight_name"] == weight_name]
        box_values = [
            weight_df[weight_df["alpha"] == alpha]["weight_value"].to_numpy()
            for alpha in alpha_values
        ]
        ax.boxplot(box_values, tick_labels=labels, showfliers=False)
        ax.set_title(weight_name)
        ax.set_xlabel("alpha")
        ax.set_ylabel("weight value")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"Weight Distribution by Alpha: {model_version}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_trends(summary_df: pd.DataFrame, complexity_df: pd.DataFrame, output_dir: Path) -> None:
    plot_metric_panels(
        summary_df,
        "mean",
        "mean weight",
        "Mean Structure Weights by Alpha",
        output_dir / "fig_alpha_mean_weights.png",
    )
    plot_metric_panels(
        summary_df,
        "median",
        "median weight",
        "Median Structure Weights by Alpha",
        output_dir / "fig_alpha_median_weights.png",
    )
    plot_metric_panels(
        summary_df,
        "frac_lt_0p10",
        "fraction below 0.10",
        "Near-Zero Fraction by Alpha",
        output_dir / "fig_alpha_zero_fraction.png",
    )
    plot_metric_panels(
        summary_df,
        "frac_gt_0p90",
        "fraction above 0.90",
        "High-Activation Fraction by Alpha",
        output_dir / "fig_alpha_one_fraction.png",
    )
    plot_complexity(complexity_df, output_dir / "fig_alpha_complexity.png")


def describe_trend(alpha_values: pd.Series, values: pd.Series) -> str:
    order = np.argsort(alpha_values.to_numpy(dtype=float))
    sorted_alpha = alpha_values.to_numpy(dtype=float)[order]
    sorted_values = values.to_numpy(dtype=float)[order]

    if len(sorted_values) < 2:
        return "only one alpha level"

    deltas = np.diff(sorted_values)
    total_delta = sorted_values[-1] - sorted_values[0]
    n_decrease = int((deltas < 0).sum())
    n_increase = int((deltas > 0).sum())
    if abs(total_delta) < 1e-6:
        direction = "flat by endpoint"
    elif total_delta < 0:
        direction = "lower at higher alpha"
    else:
        direction = "higher at higher alpha"

    return (
        f"{direction}; {format_alpha(sorted_alpha[0])}->{format_alpha(sorted_alpha[-1])}: "
        f"{sorted_values[0]:.4f}->{sorted_values[-1]:.4f} "
        f"(delta {total_delta:+.4f}, step decreases {n_decrease}, increases {n_increase})"
    )


def complexity_decrease_flags(complexity_df: pd.DataFrame) -> dict[str, str]:
    flags = {}
    for model_version in MODEL_VERSIONS:
        model_df = complexity_df[complexity_df["model_version"] == model_version].sort_values("alpha")
        if model_df.empty:
            flags[model_version] = "no complete weight result"
            continue
        values = model_df["mean_sum_weight"].to_numpy(dtype=float)
        if len(values) < 2:
            flags[model_version] = "only one alpha level"
            continue
        deltas = np.diff(values)
        decreases = int((deltas < 0).sum())
        increases = int((deltas > 0).sum())
        endpoint_down = values[-1] <= values[0]
        flags[model_version] = (
            "roughly decreases"
            if endpoint_down and decreases >= increases
            else "does not roughly decrease"
        )
    return flags


def alpha_dir_status(root_dirs: list[Path], valid_records: list[dict[str, object]]) -> pd.DataFrame:
    valid_counts = {}
    for record in valid_records:
        key = (record["model_version"], float(record["alpha"]))
        valid_counts[key] = valid_counts.get(key, 0) + 1

    rows = []
    for root in root_dirs:
        for alpha_dir in sorted(path for path in root.iterdir() if path.is_dir() and parse_alpha(path) is not None):
            alpha = parse_alpha(alpha_dir)
            n_valid = valid_counts.get((root.name, float(alpha)), 0)
            rows.append(
                {
                    "model_version": root.name,
                    "alpha": float(alpha),
                    "alpha_dir": str(alpha_dir),
                    "complete_weight_files": n_valid > 0,
                    "valid_result_dirs": n_valid,
                }
            )
    return pd.DataFrame(rows)


def write_markdown_report(
    output_path: Path,
    result_metadata: list[dict[str, object]],
    summary_df: pd.DataFrame,
    complexity_df: pd.DataFrame,
    status_df: pd.DataFrame,
    complexity_flags: dict[str, str],
) -> None:
    lines = [
        "# Alpha Weight Effect Report",
        "",
        "This report summarizes existing `w_int.npy`, `w_phen.npy`, `w_snow.npy`, and `w_sub.npy` files only. No model training or reruns were performed.",
        "",
        "## Discovery",
        "",
        f"- Valid result_dir count: {len(result_metadata)}",
        f"- Detected model versions with complete weights: {', '.join(sorted(summary_df['model_version'].unique())) if not summary_df.empty else 'none'}",
        f"- Detected alpha values with complete weights: {', '.join(format_alpha(alpha) for alpha in sorted(summary_df['alpha'].unique())) if not summary_df.empty else 'none'}",
        "",
        "## Completeness by Alpha Directory",
        "",
    ]

    if status_df.empty:
        lines.append("- No alpha directories were detected under the configured roots.")
    else:
        for _, row in status_df.sort_values(["model_version", "alpha"]).iterrows():
            status = "complete" if row["complete_weight_files"] else "missing complete weight result"
            lines.append(
                f"- {row['model_version']} alpha={format_alpha(row['alpha'])}: {status}; valid_result_dirs={int(row['valid_result_dirs'])}"
            )

    lines.extend(["", "## Mean Weight Trends", ""])
    if summary_df.empty:
        lines.append("- No complete weight results were available for trend analysis.")
    else:
        for model_version in MODEL_VERSIONS:
            model_df = summary_df[summary_df["model_version"] == model_version]
            if model_df.empty:
                lines.append(f"- {model_version}: no complete weight result.")
                continue
            for weight_name in WEIGHT_NAMES:
                weight_df = model_df[model_df["weight_name"] == weight_name].sort_values("alpha")
                lines.append(
                    f"- {model_version} {weight_name}: {describe_trend(weight_df['alpha'], weight_df['mean'])}"
                )

    lines.extend(["", "## Complexity Trend", ""])
    for model_version in MODEL_VERSIONS:
        model_df = complexity_df[complexity_df["model_version"] == model_version].sort_values("alpha")
        if model_df.empty:
            lines.append(f"- {model_version}: no complete weight result.")
            continue
        lines.append(
            f"- {model_version}: {complexity_flags[model_version]}; "
            f"{describe_trend(model_df['alpha'], model_df['mean_sum_weight'])}"
        )

    lines.extend(["", "## Valid Result Directories", ""])
    if result_metadata:
        for record in result_metadata:
            lines.append(
                f"- {record['model_version']} alpha={format_alpha(record['alpha'])}: "
                f"data_range={record['data_range']}; run_name={record['run_name']}"
            )
    else:
        lines.append("- None.")

    lines.extend(["", "## Low and High Activation Summary", ""])
    if summary_df.empty:
        lines.append("- No complete weight results were available.")
    else:
        high_alpha = summary_df["alpha"].max()
        high_df = summary_df[summary_df["alpha"] == high_alpha].copy()
        low_rank = high_df.sort_values(["frac_lt_0p10", "mean"], ascending=[False, True]).head(4)
        high_rank = high_df.sort_values(["frac_gt_0p90", "mean"], ascending=[False, False]).head(4)
        lines.append(f"- Highest detected alpha with complete weights: {format_alpha(high_alpha)}")
        lines.append("- Weights most easily pushed low at highest alpha:")
        for _, row in low_rank.iterrows():
            lines.append(
                f"  - {row['model_version']} {row['weight_name']}: mean={row['mean']:.4f}, frac_lt_0p10={row['frac_lt_0p10']:.4f}"
            )
        lines.append("- Weights retaining high activation at highest alpha:")
        if high_rank["frac_gt_0p90"].max() == 0:
            lines.append("  - None had values above 0.90 at the highest detected alpha.")
        else:
            for _, row in high_rank.iterrows():
                lines.append(
                    f"  - {row['model_version']} {row['weight_name']}: mean={row['mean']:.4f}, frac_gt_0p90={row['frac_gt_0p90']:.4f}"
                )

    lines.extend(["", "## Potential Anomalies", ""])
    anomaly_lines = []
    for model_version in MODEL_VERSIONS:
        if summary_df[summary_df["model_version"] == model_version].empty:
            anomaly_lines.append(f"- {model_version}: no complete weight result found under the configured root.")

    if not summary_df.empty:
        for (model_version, weight_name), group in summary_df.groupby(["model_version", "weight_name"]):
            if len(group) > 1 and group["mean"].max() - group["mean"].min() < 1e-6:
                anomaly_lines.append(f"- {model_version} {weight_name}: mean weight is nearly unchanged across alpha.")
            near_half = group[(group["mean"] - 0.5).abs() < 0.02]
            if not near_half.empty:
                alpha_list = ", ".join(format_alpha(alpha) for alpha in near_half["alpha"])
                anomaly_lines.append(f"- {model_version} {weight_name}: mean weight is near 0.5 at alpha {alpha_list}.")

        trend_signs = {}
        for (model_version, weight_name), group in summary_df.groupby(["model_version", "weight_name"]):
            group = group.sort_values("alpha")
            if len(group) > 1:
                delta = float(group["mean"].iloc[-1] - group["mean"].iloc[0])
                trend_signs.setdefault(weight_name, []).append((model_version, np.sign(delta)))
        for weight_name, signs in trend_signs.items():
            non_zero_signs = {sign for _, sign in signs if sign != 0}
            if len(non_zero_signs) > 1:
                details = ", ".join(f"{model_version}:{int(sign):+d}" for model_version, sign in signs)
                anomaly_lines.append(f"- {weight_name}: model versions show opposite endpoint trend signs ({details}).")

    if anomaly_lines:
        lines.extend(anomaly_lines)
    else:
        lines.append("- No shape anomalies or cross-version trend reversals were detected in completed analysis.")

    lines.extend(["", "## Output Files", ""])
    for file_name in [
        "weights_long.csv",
        "weight_alpha_summary.csv",
        "basin_complexity.csv",
        "complexity_alpha_summary.csv",
        "fig_alpha_mean_weights.png",
        "fig_alpha_median_weights.png",
        "fig_alpha_zero_fraction.png",
        "fig_alpha_one_fraction.png",
        "fig_alpha_complexity.png",
        *[f"fig_weight_distribution_by_alpha_{model_version}.png" for model_version in MODEL_VERSIONS],
    ]:
        lines.append(f"- `{file_name}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = find_weight_result_dirs(ROOT_DIRS)
    records = [
        load_weight_result(Path(item["result_dir"]), item)
        for item in metadata
    ]
    weights_long = build_weights_long(records)
    summary_df, basin_df, complexity_df = build_summary_tables(weights_long)
    status_df = alpha_dir_status(ROOT_DIRS, metadata)

    weights_long.to_csv(OUTPUT_DIR / "weights_long.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "weight_alpha_summary.csv", index=False)
    basin_df.to_csv(OUTPUT_DIR / "basin_complexity.csv", index=False)
    complexity_df.to_csv(OUTPUT_DIR / "complexity_alpha_summary.csv", index=False)

    plot_alpha_trends(summary_df, complexity_df, OUTPUT_DIR)
    for model_version in MODEL_VERSIONS:
        plot_distribution_by_model(
            weights_long,
            model_version,
            OUTPUT_DIR / f"fig_weight_distribution_by_alpha_{model_version}.png",
        )

    flags = complexity_decrease_flags(complexity_df)
    write_markdown_report(
        OUTPUT_DIR / "alpha_weight_effect_report.md",
        metadata,
        summary_df,
        complexity_df,
        status_df,
        flags,
    )

    print(f"Valid result_dir count: {len(metadata)}")
    print("Alpha values by model_version:")
    for model_version in MODEL_VERSIONS:
        alphas = sorted(summary_df[summary_df["model_version"] == model_version]["alpha"].unique())
        alpha_text = ", ".join(format_alpha(alpha) for alpha in alphas) if alphas else "none"
        print(f"  {model_version}: {alpha_text}")

    print("Output files:")
    for output_path in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {output_path}")

    print("Complexity trend by model_version:")
    for model_version in MODEL_VERSIONS:
        print(f"  {model_version}: {flags[model_version]}")


if __name__ == "__main__":
    main()
