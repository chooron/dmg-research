from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "parameters" / "results"
MSTTCOREFONTS_DIR = Path("/usr/share/fonts/truetype/msttcorefonts")
CORE_MODEL_ORDER = ("lstm", "s4d", "s5dv1", "s5dv2")
OTHER_MODEL_ORDER = ("lstm", "transformer", "timemixer", "tcn")
ALL_MODEL_ORDER = ("lstm", "s4d", "s5dv1", "s5dv2", "transformer", "timemixer", "tcn")
PARAMETER_ORDER = ("parBETA", "parK0", "parBETAET")

MODEL_LABELS = {
    "lstm": "LSTM",
    "s4d": "S4D",
    "s5dv1": "S5Dv1",
    "s5dv2": "S5Dv2",
    "transformer": "Transformer",
    "timemixer": "TimeMixer",
    "tcn": "TCN",
}

MODEL_DISPLAY_LABELS = {
    "lstm": r"$\delta\,\mathrm{MG}_{\mathrm{LSTM}}$",
    "s4d": r"$\delta\,\mathrm{MG}_{\mathrm{S4D}}$",
    "s5dv1": r"$\delta\,\mathrm{MG}_{\mathrm{S5Dv1}}$",
    "s5dv2": r"$\delta\,\mathrm{MG}_{\mathrm{S5Dv2}}$",
    "transformer": r"$\delta\,\mathrm{MG}_{\mathrm{Transformer}}$",
    "timemixer": r"$\delta\,\mathrm{MG}_{\mathrm{TimeMixer}}$",
    "tcn": r"$\delta\,\mathrm{MG}_{\mathrm{TCN}}$",
}

PARAMETER_DISPLAY_LABELS = {
    "parBETA": r"$\beta$",
    "parK0": r"$K_0$",
    "parBETAET": r"$\gamma$",
}

MODEL_COLORS = {
    "lstm": "#5B84B1FF",
    "s4d": "#D94F4FFF",
    "s5dv1": "#5BA85FFF",
    "s5dv2": "#8C6BB1",
    "transformer": "#D94F4FFF",
    "timemixer": "#5BA85FFF",
    "tcn": "#8C6BB1",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 2x3 multi-basin dynamic-parameter variability figure with two metrics.",
    )
    parser.add_argument("--lstm-path", default=str(RESULTS_DIR / "lstm_seed111_normalized_dynamic_parameters.npz"))
    parser.add_argument("--s4d-path", default=str(RESULTS_DIR / "s4d_seed111_normalized_dynamic_parameters.npz"))
    parser.add_argument("--s5dv1-path", default=str(RESULTS_DIR / "s5dv1_seed111_normalized_dynamic_parameters.npz"))
    parser.add_argument("--s5dv2-path", default=str(RESULTS_DIR / "s5dv2_seed111_normalized_dynamic_parameters.npz"))
    parser.add_argument("--transformer-path", default=str(RESULTS_DIR / "transformer_seed111_normalized_dynamic_parameters.npz"))
    parser.add_argument("--timemixer-path", default=str(RESULTS_DIR / "timemixer_seed111_normalized_dynamic_parameters.npz"))
    parser.add_argument("--tcn-path", default=str(RESULTS_DIR / "tcn_seed111_normalized_dynamic_parameters.npz"))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--point-alpha", type=float, default=0.10)
    parser.add_argument("--base-name", default="multibasin_parameter_variability_two_metrics")
    parser.add_argument(
        "--group",
        choices=("core", "other", "both"),
        default="both",
        help="Which model group(s) to render.",
    )
    return parser.parse_args()


def _set_plot_style() -> str:
    for path in sorted(MSTTCOREFONTS_DIR.glob("*.TTF")) + sorted(MSTTCOREFONTS_DIR.glob("*.ttf")):
        try:
            font_manager.fontManager.addfont(str(path))
        except Exception:
            pass
    font_manager._load_fontmanager(try_read_cache=False)

    preferred_fonts = [
        "Times New Roman",
        "TimesNewRomanPSMT",
        "Nimbus Roman No9 L",
        "Nimbus Roman",
        "STIXGeneral",
        "DejaVu Serif",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    font_family = next((name for name in preferred_fonts if name in available), "serif")

    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.serif": [font_family],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    if font_family == "Times New Roman":
        plt.rcParams.update(
            {
                "mathtext.fontset": "custom",
                "mathtext.rm": font_family,
                "mathtext.it": f"{font_family}:italic",
                "mathtext.bf": f"{font_family}:bold",
                "mathtext.cal": font_family,
                "mathtext.fallback": "stix",
            }
        )
    else:
        plt.rcParams.update({"mathtext.fontset": "stix"})
    return font_family


def _discover_npz(path_or_dir: str, model_key: str) -> Path:
    path = Path(path_or_dir).expanduser().resolve()
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"{MODEL_LABELS[model_key]} input path not found: {path}")

    candidates = sorted(path.rglob("*.npz"))
    for candidate in candidates:
        try:
            with np.load(candidate, allow_pickle=True) as npz:
                if "normalized_parameters" in npz:
                    return candidate
        except Exception:
            continue
    raise FileNotFoundError(
        f"No normalized dynamic-parameter NPZ found under {path} for {MODEL_LABELS[model_key]}"
    )


def _load_payload(path: Path, model_key: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as npz:
        metadata = json.loads(str(npz["metadata_json"].item())) if "metadata_json" in npz else {}
        parameter_names = [str(x) for x in npz["parameter_names"].tolist()]
        return {
            "path": path,
            "model_key": model_key,
            "model": MODEL_LABELS[model_key],
            "metadata": metadata,
            "normalized_parameters": np.asarray(npz["normalized_parameters"], dtype=np.float32),
            "basin_ids": np.asarray(npz["basin_ids"]),
            "parameter_names": parameter_names,
        }


def _select_median_component_trajectory(param_tensor: np.ndarray) -> np.ndarray:
    if param_tensor.ndim != 3:
        raise ValueError(f"Expected (time, parameter, component), got {param_tensor.shape}")

    time_steps, n_params, _ = param_tensor.shape
    selected = np.full((time_steps, n_params), np.nan, dtype=np.float32)
    for p_idx in range(n_params):
        series = param_tensor[:, p_idx, :]
        component_means = np.nanmean(series, axis=0)
        if np.all(np.isnan(component_means)):
            continue
        median_mean = np.nanmedian(component_means)
        component_index = int(np.nanargmin(np.abs(component_means - median_mean)))
        selected[:, p_idx] = series[:, component_index]
    return selected


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


def _build_summary(payloads: list[dict[str, Any]]) -> pd.DataFrame:
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
        selected_indices = [index for _, index in parameter_pairs]

        for basin_idx, basin_id in enumerate(basin_ids):
            collapsed = _select_median_component_trajectory(tensor[:, basin_idx, selected_indices, :])
            for local_idx, (parameter, _) in enumerate(parameter_pairs):
                series = collapsed[:, local_idx]
                day_change, n_valid_steps = _median_abs_day_to_day_change(series)
                amplitude, _ = _temporal_iqr(series)
                rows.append(
                    {
                        "model": payload["model"],
                        "basin_id": int(basin_id),
                        "parameter": parameter,
                        "median_abs_day_to_day_change": day_change,
                        "temporal_iqr": amplitude,
                        "n_valid_steps": n_valid_steps,
                        "source_npz": payload["path"].name,
                    }
                )
    return pd.DataFrame(rows)


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
        widths=0.55,
        medianprops={"color": "black", "linewidth": 1.3},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
        boxprops={"linewidth": 1.0, "edgecolor": "#555555"},
    )
    for patch, model_key in zip(bp["boxes"], model_order):
        patch.set_facecolor(MODEL_COLORS[model_key])
        patch.set_alpha(0.9)

    rng = np.random.default_rng(20260429 + hash((metric, parameter)) % 1000)
    for pos, model_key in zip(positions, model_order):
        values = panel_df.loc[
            panel_df["model"] == MODEL_LABELS[model_key], metric
        ].dropna().values
        if values.size == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=values.size)
        ax.scatter(
            np.full(values.size, pos) + jitter,
            values,
            s=5,
            color=MODEL_COLORS[model_key],
            alpha=point_alpha,
            linewidths=0,
            zorder=2,
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

    if metric == "median_abs_day_to_day_change":
        ylabel = "Median absolute day-to-day change"
    else:
        ylabel = "Temporal IQR"
    ax.set_ylabel(ylabel if show_ylabel else "", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.text(
        0.03,
        0.97,
        f"{panel_label} {PARAMETER_DISPLAY_LABELS.get(parameter, parameter)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.8,
            "pad": 1.5,
        },
        zorder=5,
    )


def _plot_figure(
    summary_df: pd.DataFrame,
    output_png: Path,
    *,
    model_order: tuple[str, ...],
    point_alpha: float,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.2), constrained_layout=True)

    panels = [
        ("median_abs_day_to_day_change", "parBETA", "(a)", 0, 0),
        ("median_abs_day_to_day_change", "parK0", "(b)", 0, 1),
        ("median_abs_day_to_day_change", "parBETAET", "(c)", 0, 2),
        ("temporal_iqr", "parBETA", "(d)", 1, 0),
        ("temporal_iqr", "parK0", "(e)", 1, 1),
        ("temporal_iqr", "parBETAET", "(f)", 1, 2),
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
            show_xticklabels=(row == 1),
            show_ylabel=(col == 0),
            point_alpha=point_alpha,
        )

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _print_report(
    summary_df: pd.DataFrame,
    payloads: list[dict[str, Any]],
    summary_path: Path,
    output_pngs: list[Path],
    font_family: str,
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

    for parameter in PARAMETER_ORDER:
        sub = summary_df[summary_df["parameter"] == parameter]
        medians = sub.groupby("model")["median_abs_day_to_day_change"].median().to_dict()
        s4d = medians.get("S4D", float("nan"))
        s5dv1 = medians.get("S5Dv1", float("nan"))
        s5dv2 = medians.get("S5Dv2", float("nan"))
        print(
            f"{parameter} high-frequency variability medians: "
            f"S4D={s4d:.6f}, S5Dv1={s5dv1:.6f}, S5Dv2={s5dv2:.6f}"
        )
        print(
            f"{parameter} variability decreases vs S4D: "
            f"S5Dv1={bool(s5dv1 < s4d)}, S5Dv2={bool(s5dv2 < s4d)}"
        )

    for parameter in PARAMETER_ORDER:
        sub = summary_df[summary_df["parameter"] == parameter]
        iqr_medians = sub.groupby("model")["temporal_iqr"].median().to_dict()
        print(
            f"{parameter} temporal IQR medians: "
            f"S4D={iqr_medians.get('S4D', float('nan')):.6f}, "
            f"S5Dv1={iqr_medians.get('S5Dv1', float('nan')):.6f}, "
            f"S5Dv2={iqr_medians.get('S5Dv2', float('nan')):.6f}"
        )
        print(
            f"{parameter} S5D temporal IQR non-zero: "
            f"S5Dv1={bool(iqr_medians.get('S5Dv1', 0.0) > 0)}, "
            f"S5Dv2={bool(iqr_medians.get('S5Dv2', 0.0) > 0)}"
        )

    print(f"Summary CSV: {summary_path}")
    print("Figure files: " + ", ".join(str(path) for path in output_pngs))


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

    payloads = [
        _load_payload(_discover_npz(path_map[model_key], model_key), model_key)
        for model_key in ALL_MODEL_ORDER
    ]
    summary_df = _build_summary(payloads)
    summary_path = output_dir / f"{args.base_name}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    output_pngs: list[Path] = []
    if args.group in {"core", "both"}:
        output_png = output_dir / f"{args.base_name}_core_models.png"
        _plot_figure(
            summary_df[summary_df["model"].isin([MODEL_LABELS[key] for key in CORE_MODEL_ORDER])].copy(),
            output_png,
            model_order=CORE_MODEL_ORDER,
            point_alpha=args.point_alpha,
        )
        output_pngs.append(output_png)
    if args.group in {"other", "both"}:
        output_png = output_dir / f"{args.base_name}_other_models.png"
        _plot_figure(
            summary_df[summary_df["model"].isin([MODEL_LABELS[key] for key in OTHER_MODEL_ORDER])].copy(),
            output_png,
            model_order=OTHER_MODEL_ORDER,
            point_alpha=args.point_alpha,
        )
        output_pngs.append(output_png)

    _print_report(summary_df, payloads, summary_path, output_pngs, font_family)


if __name__ == "__main__":
    main()
