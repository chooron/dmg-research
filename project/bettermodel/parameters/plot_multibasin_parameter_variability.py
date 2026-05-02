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
SUMMARY_FILENAME = "multibasin_parameter_variability_summary.csv"
MSTTCOREFONTS_DIR = Path("/usr/share/fonts/truetype/msttcorefonts")
CORE_MODELS = ("lstm", "s4d", "s5dv1", "s5dv2")
OTHER_MODELS = ("lstm", "transformer", "timemixer", "tcn")
PREFERRED_PARAMETERS = ("parBETA", "parFC", "parK0")

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
        description="Create multi-basin dynamic-parameter variability boxplots from extracted NPZ files.",
    )
    parser.add_argument("--input-dir", default=str(RESULTS_DIR))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument(
        "--collapse-strategy",
        choices=("median_component",),
        default="median_component",
        help="How to reduce the component dimension to one trajectory per basin-parameter pair.",
    )
    parser.add_argument("--point-alpha", type=float, default=0.12)
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
                "mathtext.fallback": "stix",
            }
        )
    else:
        plt.rcParams.update({"mathtext.fontset": "stix"})
    return font_family


def _load_npz_files(input_dir: Path) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for path in sorted(input_dir.glob("*_normalized_dynamic_parameters.npz")):
        with np.load(path, allow_pickle=True) as npz:
            metadata = json.loads(str(npz["metadata_json"].item()))
            model_key = str(npz["model_key"].item())
            loaded[model_key] = {
                "path": path,
                "metadata": metadata,
                "normalized_parameters": np.asarray(npz["normalized_parameters"], dtype=np.float32),
                "basin_ids": np.asarray(npz["basin_ids"]),
                "parameter_names": [str(x) for x in npz["parameter_names"].tolist()],
            }
    return loaded


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


def _collapse_trajectory(param_tensor: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "median_component":
        return _select_median_component_trajectory(param_tensor)
    raise ValueError(f"Unsupported collapse strategy: {strategy}")


def _median_abs_day_to_day_change(series: np.ndarray) -> tuple[float, int]:
    valid = np.isfinite(series)
    cleaned = series[valid]
    if cleaned.size < 2:
        return float("nan"), int(cleaned.size)
    diff = np.abs(np.diff(cleaned))
    if diff.size == 0:
        return float("nan"), int(cleaned.size)
    return float(np.nanmedian(diff)), int(cleaned.size)


def _resolve_panel_parameters(loaded: dict[str, dict[str, Any]]) -> list[str]:
    available: list[str] = []
    for payload in loaded.values():
        for name in payload["parameter_names"]:
            if name not in available:
                available.append(name)

    selected = [name for name in PREFERRED_PARAMETERS if name in available]
    for name in available:
        if name not in selected:
            selected.append(name)
        if len(selected) == 3:
            break
    return selected[:3]


def _build_summary(
    loaded: dict[str, dict[str, Any]],
    panel_parameters: list[str],
    collapse_strategy: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_key, payload in loaded.items():
        parameter_names = payload["parameter_names"]
        tensor = payload["normalized_parameters"]
        basin_ids = payload["basin_ids"]
        selected_pairs = [
            (name, parameter_names.index(name))
            for name in panel_parameters
            if name in parameter_names
        ]
        for basin_idx, basin_id in enumerate(basin_ids):
            selected_indices = [index for _, index in selected_pairs]
            collapsed = _collapse_trajectory(tensor[:, basin_idx, selected_indices, :], collapse_strategy)
            for local_idx, (parameter, _) in enumerate(selected_pairs):
                metric, n_valid_steps = _median_abs_day_to_day_change(collapsed[:, local_idx])
                rows.append(
                    {
                        "model_key": model_key,
                        "model": MODEL_LABELS.get(model_key, model_key),
                        "parameter": parameter,
                        "basin_id": int(basin_id),
                        "variability_metric": metric,
                        "n_valid_steps": n_valid_steps,
                        "seed": int(payload["metadata"]["seed"]),
                        "window_start": int(payload["metadata"]["window_start"]),
                        "window_stop": int(payload["metadata"]["window_stop"]),
                        "collapse_strategy": collapse_strategy,
                        "source_npz": payload["path"].name,
                    }
                )
    return pd.DataFrame(rows)


def _plot_group(
    summary_df: pd.DataFrame,
    *,
    model_keys: tuple[str, ...],
    panel_parameters: list[str],
    output_base: Path,
    point_alpha: float,
) -> None:
    fig, axes = plt.subplots(1, len(panel_parameters), figsize=(13.5, 4.2), constrained_layout=True)
    if len(panel_parameters) == 1:
        axes = [axes]

    for panel_idx, parameter in enumerate(panel_parameters):
        ax = axes[panel_idx]
        panel_df = summary_df[
            summary_df["model_key"].isin(model_keys) & (summary_df["parameter"] == parameter)
        ].copy()

        data = [
            panel_df.loc[panel_df["model_key"] == model_key, "variability_metric"].dropna().values
            for model_key in model_keys
        ]
        positions = np.arange(1, len(model_keys) + 1)
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

        for patch, model_key in zip(bp["boxes"], model_keys):
            patch.set_facecolor(MODEL_COLORS[model_key])
            patch.set_alpha(0.9)

        rng = np.random.default_rng(20260429 + panel_idx)
        for pos, model_key in zip(positions, model_keys):
            values = panel_df.loc[panel_df["model_key"] == model_key, "variability_metric"].dropna().values
            if values.size == 0:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=values.size)
            ax.scatter(
                np.full(values.size, pos) + jitter,
                values,
                s=6,
                color=MODEL_COLORS[model_key],
                alpha=point_alpha,
                linewidths=0,
                zorder=2,
            )

        panel_letter = chr(ord("a") + panel_idx)
        panel_label = PARAMETER_DISPLAY_LABELS.get(parameter, parameter)
        ax.set_xticks(positions)
        ax.set_xticklabels([MODEL_DISPLAY_LABELS.get(key, MODEL_LABELS[key]) for key in model_keys], fontsize=12)
        ax.set_ylabel("Median absolute day-to-day change" if panel_idx == 0 else "", fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.text(
            0.03,
            0.97,
            f"({panel_letter}) {panel_label}",
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

    for suffix in (".pdf", ".svg", ".png"):
        save_kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            save_kwargs["dpi"] = 300
        fig.savefig(output_base.with_suffix(suffix), **save_kwargs)
    plt.close(fig)


def _trend_line(summary_df: pd.DataFrame, model_keys: tuple[str, ...], parameter: str) -> str:
    medians = []
    for key in model_keys:
        values = summary_df.loc[
            (summary_df["model_key"] == key) & (summary_df["parameter"] == parameter),
            "variability_metric",
        ].dropna()
        median = float(values.median()) if not values.empty else float("nan")
        medians.append((MODEL_LABELS[key], median))
    return ", ".join(
        f"{label}={value:.6f}" if np.isfinite(value) else f"{label}=nan"
        for label, value in medians
    )


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    font_family = _set_plot_style()
    loaded = _load_npz_files(input_dir)
    if not loaded:
        raise FileNotFoundError(f"No extracted NPZ files found in {input_dir}")

    panel_parameters = _resolve_panel_parameters(loaded)
    summary_df = _build_summary(loaded, panel_parameters, args.collapse_strategy)
    summary_path = output_dir / SUMMARY_FILENAME
    summary_df.to_csv(summary_path, index=False)

    _plot_group(
        summary_df,
        model_keys=CORE_MODELS,
        panel_parameters=panel_parameters,
        output_base=output_dir / "multibasin_parameter_variability_core_models",
        point_alpha=args.point_alpha,
    )
    _plot_group(
        summary_df,
        model_keys=OTHER_MODELS,
        panel_parameters=panel_parameters,
        output_base=output_dir / "multibasin_parameter_variability_other_models",
        point_alpha=args.point_alpha,
    )

    print(f"Wrote summary CSV: {summary_path}")
    print(f"Font used: {font_family}")
    print(f"Panel parameters: {panel_parameters}")
    for parameter in panel_parameters:
        print(
            f"[core] {parameter}: "
            + _trend_line(summary_df, CORE_MODELS, parameter)
        )
        print(
            f"[other] {parameter}: "
            + _trend_line(summary_df, OTHER_MODELS, parameter)
        )


if __name__ == "__main__":
    main()
