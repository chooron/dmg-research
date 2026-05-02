from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.bettermodel.parameters.plot_multibasin_parameter_variability_two_metrics import (  # noqa: E402
    CORE_MODEL_ORDER,
    OTHER_MODEL_ORDER,
    MODEL_COLORS,
    MODEL_DISPLAY_LABELS,
    MODEL_LABELS,
    PARAMETER_DISPLAY_LABELS,
    PARAMETER_ORDER,
    RESULTS_DIR,
    _discover_npz,
    _load_payload,
    _select_median_component_trajectory,
    _set_plot_style,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modify the current multi-basin summary figure into a 3x3 three-metric paper figure.",
    )
    parser.add_argument("--script-path", default=str(Path(__file__).resolve()))
    parser.add_argument("--lstm-path", default=str(RESULTS_DIR / "lstm_seed111_normalized_dynamic_parameters.npz"))
    parser.add_argument("--s4d-path", default=str(RESULTS_DIR / "s4d_seed111_normalized_dynamic_parameters.npz"))
    parser.add_argument("--s5dv1-path", default=str(RESULTS_DIR / "s5dv1_seed111_normalized_dynamic_parameters.npz"))
    parser.add_argument("--s5dv2-path", default=str(RESULTS_DIR / "s5dv2_seed111_normalized_dynamic_parameters.npz"))
    parser.add_argument("--transformer-path", default=str(RESULTS_DIR / "transformer_seed111_normalized_dynamic_parameters.npz"))
    parser.add_argument("--timemixer-path", default=str(RESULTS_DIR / "timemixer_seed111_normalized_dynamic_parameters.npz"))
    parser.add_argument("--tcn-path", default=str(RESULTS_DIR / "tcn_seed111_normalized_dynamic_parameters.npz"))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--base-name", default="multibasin_parameter_summary_3metrics")
    parser.add_argument("--point-alpha", type=float, default=0.55)
    parser.add_argument("--box-width", type=float, default=0.40)
    parser.add_argument("--box-alpha", type=float, default=0.32)
    parser.add_argument("--boundary-eps", type=float, default=0.05)
    parser.add_argument(
        "--group",
        choices=("core", "other", "both"),
        default="both",
        help="Which model group(s) to render.",
    )
    return parser.parse_args()


def _median_abs_day_to_day_change(series: np.ndarray) -> tuple[float, int]:
    cleaned = series[np.isfinite(series)]
    if cleaned.size < 2:
        return float("nan"), int(cleaned.size)
    diff = np.abs(np.diff(cleaned))
    if diff.size == 0:
        return float("nan"), int(cleaned.size)
    return float(np.nanmedian(diff)), int(cleaned.size)


def _temporal_iqr(series: np.ndarray) -> tuple[float, int]:
    cleaned = series[np.isfinite(series)]
    if cleaned.size == 0:
        return float("nan"), 0
    q75 = float(np.nanpercentile(cleaned, 75))
    q25 = float(np.nanpercentile(cleaned, 25))
    return q75 - q25, int(cleaned.size)


def _boundary_saturation_ratio(series: np.ndarray, eps: float) -> tuple[float, int]:
    cleaned = series[np.isfinite(series)]
    if cleaned.size == 0:
        return float("nan"), 0
    saturated = (cleaned < eps) | (cleaned > 1.0 - eps)
    return float(np.mean(saturated)), int(cleaned.size)


def _darken_color(color: str, factor: float = 0.72) -> tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(np.clip(rgb * factor, 0.0, 1.0))


def _build_summary(payloads: list[dict[str, Any]], boundary_eps: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        tensor = payload["normalized_parameters"]
        parameter_names = payload["parameter_names"]
        basin_ids = payload["basin_ids"]

        parameter_pairs = [
            (parameter, parameter_names.index(parameter))
            for parameter in PARAMETER_ORDER
            if parameter in parameter_names
        ]
        if not parameter_pairs:
            continue
        selected_indices = [index for _, index in parameter_pairs]

        for basin_idx, basin_id in enumerate(basin_ids):
            collapsed = _select_median_component_trajectory(tensor[:, basin_idx, selected_indices, :])
            for local_idx, (parameter, _) in enumerate(parameter_pairs):
                series = collapsed[:, local_idx]
                day_change, n_valid_steps = _median_abs_day_to_day_change(series)
                temporal_iqr, _ = _temporal_iqr(series)
                boundary_ratio, _ = _boundary_saturation_ratio(series, boundary_eps)
                rows.append(
                    {
                        "model": payload["model"],
                        "basin_id": int(basin_id),
                        "parameter": parameter,
                        "median_abs_day_to_day_change": day_change,
                        "temporal_iqr": temporal_iqr,
                        "boundary_saturation_ratio": boundary_ratio,
                        "n_valid_steps": n_valid_steps,
                        "source_npz": payload["path"].name,
                    }
                )
    return pd.DataFrame(rows)


def _metric_ylabel(metric: str) -> str:
    if metric == "median_abs_day_to_day_change":
        return "Median absolute day-to-day change"
    if metric == "temporal_iqr":
        return "Temporal IQR"
    if metric == "boundary_saturation_ratio":
        return "Boundary saturation ratio"
    raise ValueError(f"Unknown metric: {metric}")


def _draw_panel(
    ax: plt.Axes,
    panel_df: pd.DataFrame,
    *,
    model_order: tuple[str, ...],
    metric: str,
    parameter: str,
    panel_label: str,
    show_xticklabels: bool,
    show_ylabel: bool,
    point_alpha: float,
    box_width: float,
    box_alpha: float,
) -> None:
    data = [
        panel_df.loc[panel_df["model"] == MODEL_LABELS[model_key], metric].dropna().values
        for model_key in model_order
    ]
    positions = np.arange(1, len(model_order) + 1)

    bp = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        notch=True,
        showfliers=False,
        widths=box_width,
        medianprops={"color": "#222222", "linewidth": 1.4},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
        boxprops={"linewidth": 1.0, "edgecolor": "#444444"},
    )
    for patch, model_key in zip(bp["boxes"], model_order):
        patch.set_facecolor(MODEL_COLORS[model_key])
        patch.set_alpha(box_alpha)
        patch.set_zorder(1)

    rng = np.random.default_rng(20260429 + abs(hash((metric, parameter))) % 10000)
    for pos, model_key in zip(positions, model_order):
        values = panel_df.loc[
            panel_df["model"] == MODEL_LABELS[model_key], metric
        ].dropna().values
        if values.size == 0:
            continue
        jitter = rng.uniform(-0.10, 0.10, size=values.size)
        ax.scatter(
            np.full(values.size, pos) + jitter,
            values,
            s=7,
            color=_darken_color(MODEL_COLORS[model_key]),
            alpha=point_alpha,
            linewidths=0,
            zorder=4,
        )

    ax.set_xticks(positions)
    if show_xticklabels:
        ax.set_xticklabels(
            [MODEL_DISPLAY_LABELS[model_key] for model_key in model_order],
            fontsize=12,
        )
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

    ax.set_ylabel(_metric_ylabel(metric) if show_ylabel else "", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.margins(y=0.05)
    ax.text(
        0.03,
        0.97,
        f"{panel_label} {PARAMETER_DISPLAY_LABELS.get(parameter, parameter)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13.5,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.80,
            "pad": 1.3,
        },
        zorder=5,
    )


def _plot_figure(
    summary_df: pd.DataFrame,
    output_base: Path,
    *,
    model_order: tuple[str, ...],
    point_alpha: float,
    box_width: float,
    box_alpha: float,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14.0, 10.0), constrained_layout=True)

    panels = [
        ("median_abs_day_to_day_change", "parBETA", "(a)", 0, 0),
        ("median_abs_day_to_day_change", "parK0", "(b)", 0, 1),
        ("median_abs_day_to_day_change", "parBETAET", "(c)", 0, 2),
        ("temporal_iqr", "parBETA", "(d)", 1, 0),
        ("temporal_iqr", "parK0", "(e)", 1, 1),
        ("temporal_iqr", "parBETAET", "(f)", 1, 2),
        ("boundary_saturation_ratio", "parBETA", "(g)", 2, 0),
        ("boundary_saturation_ratio", "parK0", "(h)", 2, 1),
        ("boundary_saturation_ratio", "parBETAET", "(i)", 2, 2),
    ]

    for metric, parameter, panel_label, row, col in panels:
        panel_df = summary_df[summary_df["parameter"] == parameter].copy()
        _draw_panel(
            axes[row, col],
            panel_df,
            model_order=model_order,
            metric=metric,
            parameter=parameter,
            panel_label=panel_label,
            show_xticklabels=(row == 2),
            show_ylabel=(col == 0),
            point_alpha=point_alpha,
            box_width=box_width,
            box_alpha=box_alpha,
        )

    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _print_report(
    summary_df: pd.DataFrame,
    payloads: list[dict[str, Any]],
    summary_path: Path,
    output_bases: list[Path],
    font_family: str,
    box_width: float,
    box_alpha: float,
) -> None:
    print(f"Font used: {font_family}")
    for payload in payloads:
        print(
            f"{payload['model']}: loaded {payload['path']} "
            f"shape={tuple(payload['normalized_parameters'].shape)}"
        )

    counts = (
        summary_df.groupby(["model", "parameter"])["basin_id"]
        .nunique()
        .reset_index(name="n_basins")
        .sort_values(["parameter", "model"])
    )
    print("Basin counts by model/parameter:")
    print(counts.to_string(index=False))

    for column in [
        "median_abs_day_to_day_change",
        "temporal_iqr",
        "boundary_saturation_ratio",
    ]:
        nan_count = int(summary_df[column].isna().sum())
        print(f"{column}: computed, NaN rows={nan_count}")

    print(
        "Readability optimization: "
        f"box width={box_width}, box alpha={box_alpha}, showfliers=False, "
        "jitter points enabled, scatter zorder=4, smaller point size and darker point color."
    )
    print(f"Summary CSV: {summary_path}")
    figure_files = [str(output_base.with_suffix(".png")) for output_base in output_bases]
    print("Figure files: " + ", ".join(figure_files))


def main() -> None:
    args = _parse_args()
    font_family = _set_plot_style()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    path_map = {
        "lstm": args.lstm_path,
        "s4d": args.s4d_path,
        "s5dv1": args.s5dv1_path,
        "s5dv2": args.s5dv2_path,
        "transformer": args.transformer_path,
        "timemixer": args.timemixer_path,
        "tcn": args.tcn_path,
    }
    required_models = CORE_MODEL_ORDER if args.group == "core" else (
        OTHER_MODEL_ORDER if args.group == "other" else CORE_MODEL_ORDER + tuple(
            key for key in OTHER_MODEL_ORDER if key not in CORE_MODEL_ORDER
        )
    )
    payloads = [
        _load_payload(_discover_npz(path_map[model_key], model_key), model_key)
        for model_key in required_models
    ]

    summary_df = _build_summary(payloads, args.boundary_eps)
    summary_path = output_dir / f"{args.base_name}_summary.csv"
    summary_df[
        [
            "model",
            "basin_id",
            "parameter",
            "median_abs_day_to_day_change",
            "temporal_iqr",
            "boundary_saturation_ratio",
        ]
    ].to_csv(summary_path, index=False)

    output_bases: list[Path] = []
    if args.group in {"core", "both"}:
        output_base = output_dir / f"{args.base_name}_core_models"
        _plot_figure(
            summary_df[summary_df["model"].isin([MODEL_LABELS[key] for key in CORE_MODEL_ORDER])].copy(),
            output_base,
            model_order=CORE_MODEL_ORDER,
            point_alpha=args.point_alpha,
            box_width=args.box_width,
            box_alpha=args.box_alpha,
        )
        output_bases.append(output_base)
    if args.group in {"other", "both"}:
        output_base = output_dir / f"{args.base_name}_other_models"
        _plot_figure(
            summary_df[summary_df["model"].isin([MODEL_LABELS[key] for key in OTHER_MODEL_ORDER])].copy(),
            output_base,
            model_order=OTHER_MODEL_ORDER,
            point_alpha=args.point_alpha,
            box_width=args.box_width,
            box_alpha=args.box_alpha,
        )
        output_bases.append(output_base)
    _print_report(
        summary_df,
        payloads,
        summary_path,
        output_bases,
        font_family,
        args.box_width,
        args.box_alpha,
    )


if __name__ == "__main__":
    main()
