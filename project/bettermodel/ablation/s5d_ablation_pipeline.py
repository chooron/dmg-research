from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "ablation" / "results"

for path in (REPO_ROOT, PROJECT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from dmg.core.utils import import_data_loader, set_randomseed  # noqa: E402
from project.bettermodel import load_config  # noqa: E402
from project.bettermodel.local_model_handler import LocalModelHandler  # noqa: E402
from project.bettermodel.run_experiment import apply_runtime_overrides  # noqa: E402


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    config: str
    add_conv: bool
    normalization: str
    activation: str


VARIANTS = [
    Variant(
        "s4d_baseline",
        "S4D-baseline",
        "conf/config_dhbv_hopev1.yaml",
        False,
        "BatchNorm",
        "Sigmoid",
    ),
    Variant(
        "s4d_ln",
        "S4D-LN",
        "conf/config_dhbv_ablation_s4d_ln.yaml",
        False,
        "LayerNorm",
        "Sigmoid",
    ),
    Variant(
        "s4d_softsign",
        "S4D-Softsign",
        "conf/config_dhbv_ablation_s4d_softsign.yaml",
        False,
        "BatchNorm",
        "Softsign",
    ),
    Variant(
        "s4d_ln_softsign",
        "S4D-LN-Softsign",
        "conf/config_dhbv_ablation_s4d_ln_softsign.yaml",
        False,
        "LayerNorm",
        "Softsign",
    ),
    Variant(
        "s5d_conv_only",
        "S5D-ConvOnly",
        "conf/config_dhbv_ablation_s5d_conv_only.yaml",
        True,
        "BatchNorm",
        "Sigmoid",
    ),
    Variant(
        "s5d_conv_bn_softsign",
        "S5D-ConvBN-Softsign",
        "conf/config_dhbv_ablation_s5d_conv_bn_softsign.yaml",
        True,
        "BatchNorm",
        "Softsign",
    ),
    Variant(
        "s5d_conv_ln_sigmoid",
        "S5D-ConvLN-Sigmoid",
        "conf/config_dhbv_ablation_s5d_conv_ln_sigmoid.yaml",
        True,
        "LayerNorm",
        "Sigmoid",
    ),
    Variant(
        "s5d_full",
        "S5D-full",
        "conf/config_dhbv_hopev3.yaml",
        True,
        "LayerNorm",
        "Softsign",
    ),
]

VARIANT_ORDER = [variant.label for variant in VARIANTS]
VARIANT_PALETTE = {
    "S4D-baseline": "#4C78A8",
    "S4D-LN": "#72B7B2",
    "S4D-Softsign": "#F58518",
    "S4D-LN-Softsign": "#54A24B",
    "S5D-ConvOnly": "#B279A2",
    "S5D-ConvBN-Softsign": "#FF9DA6",
    "S5D-ConvLN-Sigmoid": "#9D755D",
    "S5D-full": "#E45756",
}
METRIC_PALETTE = {"nse": "#4C78A8", "kge": "#F58518"}
ACTIVATION_PALETTE = {"Sigmoid": "#4C78A8", "Softsign": "#F58518"}
TRAJECTORY_GROUPS = [
    ("BatchNorm + Sigmoid", ["S4D-baseline", "S5D-ConvOnly"]),
    ("BatchNorm + Softsign", ["S4D-Softsign", "S5D-ConvBN-Softsign"]),
    ("LayerNorm + Sigmoid", ["S4D-LN", "S5D-ConvLN-Sigmoid"]),
    ("LayerNorm + Softsign", ["S4D-LN-Softsign", "S5D-full"]),
]
DEFAULT_TRAJECTORY_DAYS = 730
DEFAULT_TREND_WINDOW_DAYS = 365


def _set_wrr_style() -> None:
    sns.set_theme(
        context="paper",
        style="ticks",
        font="serif",
        rc={
            "axes.edgecolor": "0.15",
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "figure.dpi": 120,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "grid.color": "0.88",
            "legend.frameon": False,
            "legend.fontsize": 8,
            "savefig.bbox": "tight",
            "xtick.direction": "in",
            "xtick.labelsize": 8,
            "ytick.direction": "in",
            "ytick.labelsize": 8,
        },
    )


def _finish_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, linewidth=0.5, alpha=0.6)
    ax.tick_params(which="both", direction="in")
    sns.despine(ax=ax, trim=True)


def _runtime_args(seed: int, mode: str, test_epoch: int) -> SimpleNamespace:
    return SimpleNamespace(
        mode=mode,
        seed=seed,
        gpu_id=None,
        test_epoch=test_epoch,
        start_epoch=None,
        epochs=None,
        loss=None,
        verbose=False,
    )


def load_variant_config(variant: Variant, *, seed: int, mode: str, test_epoch: int) -> dict[str, Any]:
    config = load_config(str(PROJECT_DIR / variant.config))
    apply_runtime_overrides(config, _runtime_args(seed, mode, test_epoch))
    return config


def _checkpoint_path(config: dict[str, Any], epoch: int) -> Path:
    phy_name = config["model"]["phy"]["name"][0].lower()
    return Path(config["model_dir"]) / f"{phy_name}_ep{epoch}.pt"


def _metrics_path(config: dict[str, Any]) -> Path:
    test_start = config["test"].get("start_time", "1995/01/01")
    test_end = config["test"].get("end_time", "2010/12/31")
    test_epoch = config["test"].get("test_epoch", "")
    test_folder = f"test{test_start.split('/')[0]}-{test_end.split('/')[0]}_Ep{test_epoch}"
    return Path(config["out_path"]).parent / test_folder / "metrics.json"


def _metrics_agg_path(config: dict[str, Any]) -> Path:
    return _metrics_path(config).with_name("metrics_agg.json")


def _read_json_maybe_encoded(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if isinstance(data, str):
        data = json.loads(data)
    return data


def _load_basin_ids(config: dict[str, Any], expected: int | None = None) -> list[str]:
    subset = config["observations"].get("subset_path")
    if subset:
        raw = Path(subset).read_text(encoding="utf-8").strip()
        try:
            basin_ids = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            basin_ids = [line.strip() for line in raw.splitlines() if line.strip()]
    else:
        basin_ids = np.load(config["observations"]["gage_info"], allow_pickle=True).tolist()
    basin_ids = [str(int(item)) if str(item).isdigit() else str(item) for item in basin_ids]
    if expected is not None and len(basin_ids) != expected:
        return [str(i) for i in range(expected)]
    return basin_ids


def _available_variants(seed: int, test_epoch: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    available = []
    missing = []
    for variant in VARIANTS:
        config = load_variant_config(variant, seed=seed, mode="test", test_epoch=test_epoch)
        metrics = _metrics_path(config)
        checkpoint = _checkpoint_path(config, test_epoch)
        row = {
            "variant": variant,
            "config": config,
            "metrics_path": metrics,
            "metrics_agg_path": _metrics_agg_path(config),
            "checkpoint_path": checkpoint,
        }
        if metrics.exists() and checkpoint.exists():
            available.append(row)
        else:
            row["missing_metrics"] = not metrics.exists()
            row["missing_checkpoint"] = not checkpoint.exists()
            missing.append(row)
    return available, missing


def copy_configs(results_dir: Path) -> None:
    config_dir = results_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        PROJECT_DIR / "conf" / "config_dhbv_s5d_ablation_base.yaml",
        config_dir / "config_dhbv_s5d_ablation_base.yaml",
    )
    for variant in VARIANTS:
        shutil.copy2(PROJECT_DIR / variant.config, config_dir / Path(variant.config).name)
    mirror_dir = config_dir / "ablation_mirror"
    mirror_dir.mkdir(exist_ok=True)
    for path in sorted((PROJECT_DIR / "conf" / "ablation").glob("config_dhbv*.yaml")):
        shutil.copy2(path, mirror_dir / path.name)


def run_experiments(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    copy_configs(results_dir)

    for variant in VARIANTS:
        config = load_variant_config(
            variant,
            seed=args.seed,
            mode="test",
            test_epoch=args.test_epoch,
        )
        metrics = _metrics_path(config)
        checkpoint = _checkpoint_path(config, args.test_epoch)
        if metrics.exists() and checkpoint.exists() and not args.force:
            print(f"skip existing: {variant.label} -> {metrics}")
            continue

        command = [
            sys.executable,
            str(PROJECT_DIR / "run_experiment.py"),
            "--config",
            str(PROJECT_DIR / variant.config),
            "--mode",
            "train_test",
            "--seed",
            str(args.seed),
            "--test-epoch",
            str(args.test_epoch),
            "--no-verbose",
        ]
        print(f"run missing ablation: {variant.label}")
        subprocess.run(command, cwd=str(PROJECT_DIR), check=True)


def collect_performance(seed: int, test_epoch: int, results_dir: Path) -> None:
    _set_wrr_style()
    results_dir.mkdir(parents=True, exist_ok=True)
    copy_configs(results_dir)
    basinwise_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        config = load_variant_config(variant, seed=seed, mode="test", test_epoch=test_epoch)
        metrics_path = _metrics_path(config)
        agg_path = _metrics_agg_path(config)
        if not metrics_path.exists():
            missing_rows.append(
                {
                    "variant": variant.label,
                    "missing": str(metrics_path),
                    "reason": "metrics.json not found",
                }
            )
            continue

        metrics = _read_json_maybe_encoded(metrics_path)
        nse = np.asarray(metrics["nse"], dtype=float)
        kge = np.asarray(metrics["kge"], dtype=float)
        basin_ids = _load_basin_ids(config, expected=len(nse))

        for basin_id, nse_value, kge_value in zip(basin_ids, nse, kge):
            basinwise_rows.append(
                {
                    "variant": variant.label,
                    "variant_key": variant.key,
                    "basin_id": basin_id,
                    "nse": nse_value,
                    "kge": kge_value,
                    "normalization": variant.normalization,
                    "activation": variant.activation,
                    "add_conv": variant.add_conv,
                }
            )

        if agg_path.exists():
            agg = _read_json_maybe_encoded(agg_path)
            median_nse = agg.get("nse", {}).get("median", float(np.nanmedian(nse)))
            mean_nse = agg.get("nse", {}).get("mean", float(np.nanmean(nse)))
            median_kge = agg.get("kge", {}).get("median", float(np.nanmedian(kge)))
            mean_kge = agg.get("kge", {}).get("mean", float(np.nanmean(kge)))
        else:
            median_nse = float(np.nanmedian(nse))
            mean_nse = float(np.nanmean(nse))
            median_kge = float(np.nanmedian(kge))
            mean_kge = float(np.nanmean(kge))

        summary_rows.append(
            {
                "variant": variant.label,
                "variant_key": variant.key,
                "seed": seed,
                "test_epoch": test_epoch,
                "n_basins": int(len(nse)),
                "median_nse": median_nse,
                "mean_nse": mean_nse,
                "median_kge": median_kge,
                "mean_kge": mean_kge,
                "normalization": variant.normalization,
                "activation": variant.activation,
                "add_conv": variant.add_conv,
                "metrics_path": str(metrics_path),
            }
        )

    basinwise_df = pd.DataFrame(basinwise_rows)
    summary_df = pd.DataFrame(summary_rows)
    basinwise_df.to_csv(results_dir / "ablation_basinwise_metrics.csv", index=False)
    summary_df.to_csv(results_dir / "ablation_performance_summary.csv", index=False)
    pd.DataFrame(missing_rows, columns=["variant", "missing", "reason"]).to_csv(
        results_dir / "missing_outputs.csv",
        index=False,
    )

    if not basinwise_df.empty:
        long_df = basinwise_df.melt(
            id_vars=["variant", "variant_key", "basin_id"],
            value_vars=["nse", "kge"],
            var_name="metric",
            value_name="value",
        )
        fig, ax = plt.subplots(figsize=(7.4, 3.2))
        sns.boxplot(
            data=long_df,
            x="variant",
            y="value",
            hue="metric",
            order=VARIANT_ORDER,
            hue_order=["nse", "kge"],
            palette=METRIC_PALETTE,
            width=0.68,
            linewidth=0.8,
            fliersize=0,
            dodge=True,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.set_title("Predictive performance")
        ax.set_ylim(bottom=min(-0.5, float(np.nanpercentile(long_df["value"], 1)) - 0.05), top=1.02)
        ax.tick_params(axis="x", rotation=28)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        _finish_axis(ax)
        ax.legend(title="", ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.01), borderaxespad=0)
        fig.tight_layout()
        fig.savefig(results_dir / "ablation_performance_boxplot.png", dpi=300)
        plt.close(fig)


def _build_eval_dataset(config: dict[str, Any]) -> dict[str, torch.Tensor]:
    loader_config = dict(config)
    loader_config["device"] = "cpu"
    data_loader_cls = import_data_loader(config["data_loader"])
    return data_loader_cls(loader_config, test_split=True, overwrite=False).eval_dataset


def _dynamic_param_names(config: dict[str, Any]) -> list[str]:
    phy_name = config["model"]["phy"]["name"][0]
    dynamic_params = config["model"]["phy"].get("dynamic_params", {})
    if isinstance(dynamic_params, dict):
        return list(dynamic_params.get(phy_name, []))
    return list(dynamic_params)


def _normalize_basin_key(value: str | int) -> str:
    text = str(value)
    return str(int(text)) if text.isdigit() else text


def _as_4d_parameters(params: torch.Tensor, n_params: int) -> torch.Tensor:
    if params.ndim == 4:
        return params
    if params.ndim == 3 and n_params > 0 and params.shape[-1] % n_params == 0:
        nmul = params.shape[-1] // n_params
        return params.reshape(params.shape[0], params.shape[1], n_params, nmul)
    raise ValueError(f"Expected dynamic parameter tensor with 3 or 4 dims, got {tuple(params.shape)}")


def _predict_dynamic_parameter_tensor(
    nn_model: torch.nn.Module,
    z1: torch.Tensor,
    *,
    n_params: int,
) -> torch.Tensor:
    """Return normalized dynamic parameters as (time, basin, parameter, nmul)."""
    params = nn_model.predict_timevar_parameters(z1)
    if params.ndim == 4 and params.shape[0] == z1.shape[0] and params.shape[1] == z1.shape[1]:
        return params
    if params.ndim == 3 and params.shape[0] == z1.shape[0] and params.shape[1] == z1.shape[1]:
        return _as_4d_parameters(params, n_params)

    if hasattr(nn_model, "hope_layer"):
        hidden = nn_model.hope_layer(torch.permute(z1, (1, 0, 2))).permute(1, 0, 2)
        if hasattr(nn_model, "fc"):
            raw = nn_model.fc(hidden)
            normalized = 0.5 * (F.softsign(raw) + 1.0)
        else:
            normalized = torch.sigmoid(hidden)
        return _as_4d_parameters(normalized, n_params)

    return _as_4d_parameters(params, n_params)


def _select_median_component_trajectory(param_tensor: np.ndarray) -> np.ndarray:
    """Match plot_parameters.py: choose the single component whose time-mean is median."""
    if param_tensor.ndim != 3:
        raise ValueError(f"Expected (time, parameter, component), got {param_tensor.shape}")

    time_steps, n_params, _ = param_tensor.shape
    selected = np.empty((time_steps, n_params), dtype=param_tensor.dtype)
    for p_idx in range(n_params):
        series = param_tensor[:, p_idx, :]
        component_means = np.nanmean(series, axis=0)
        median_mean = np.nanmedian(component_means)
        median_component_index = int(np.nanargmin(np.abs(component_means - median_mean)))
        selected[:, p_idx] = series[:, median_component_index]
    return selected


def _trend_window_length(n_time: int, *, preferred: int = DEFAULT_TREND_WINDOW_DAYS) -> int:
    if n_time <= 1:
        return 1
    return max(1, min(preferred, n_time // 4 if n_time >= 4 else n_time // 2 or 1))


def _build_parameter_reliability_summary(
    variability_df: pd.DataFrame,
    roughness_df: pd.DataFrame,
    shift_df: pd.DataFrame,
    trend_noise_df: pd.DataFrame,
    saturation_df: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "variant",
        "variant_key",
        "basin_id",
        "parameter",
        "normalization",
        "activation",
        "add_conv",
    ]
    merged = variability_df.merge(roughness_df, on=keys, how="inner")
    merged = merged.merge(shift_df, on=keys, how="inner")
    merged = merged.merge(trend_noise_df, on=keys, how="inner")
    merged = merged.merge(saturation_df, on=keys, how="inner")

    summary = (
        merged.groupby(
            ["variant", "variant_key", "parameter", "normalization", "activation", "add_conv"],
            as_index=False,
        )
        .agg(
            n_basins=("basin_id", "nunique"),
            mean_variability=("variability", "mean"),
            median_variability=("variability", "median"),
            mean_roughness=("roughness", "mean"),
            median_roughness=("roughness", "median"),
            mean_long_term_shift=("long_term_shift", "mean"),
            median_long_term_shift=("long_term_shift", "median"),
            mean_trend_to_noise_ratio=("trend_to_noise_ratio", "mean"),
            median_trend_to_noise_ratio=("trend_to_noise_ratio", "median"),
            mean_boundary_saturation_ratio=("boundary_saturation_ratio", "mean"),
            median_boundary_saturation_ratio=("boundary_saturation_ratio", "median"),
        )
        .sort_values(["parameter", "variant"], key=lambda col: col.map({v: i for i, v in enumerate(VARIANT_ORDER)}) if col.name == "variant" else col)
    )
    return summary


def _plot_parameter_metric_heatmap(
    summary_df: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    cbar_label: str,
    output_path: Path,
) -> None:
    heatmap_df = summary_df.pivot_table(
        index="variant",
        columns="parameter",
        values=value_col,
        aggfunc="mean",
    )
    if heatmap_df.empty:
        return

    fig, ax = plt.subplots(figsize=(5.8, max(2.8, 0.42 * len(heatmap_df))))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".4f",
        cmap=sns.light_palette("#4C78A8", as_cmap=True),
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": cbar_label, "shrink": 0.78},
        ax=ax,
    )
    ax.set_xlabel("HBV dynamic parameter")
    ax.set_ylabel("")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def export_parameter_diagnostics(
    seed: int,
    test_epoch: int,
    results_dir: Path,
    basin_batch_size: int,
    representative_basins: list[str],
    trajectory_days: int = DEFAULT_TRAJECTORY_DAYS,
) -> None:
    _set_wrr_style()
    results_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = results_dir / "parameter_trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    for old_figure in trajectories_dir.glob("basin_*_dynamic_parameter_trajectories.png"):
        old_figure.unlink()
    npz_dir = results_dir / "dynamic_parameter_samples"
    npz_dir.mkdir(parents=True, exist_ok=True)

    variability_rows: list[dict[str, Any]] = []
    roughness_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    trend_noise_rows: list[dict[str, Any]] = []
    saturation_rows: list[dict[str, Any]] = []
    selected_trajectories: dict[str, dict[str, Any]] = {}
    missing_rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        config = load_variant_config(variant, seed=seed, mode="test", test_epoch=test_epoch)
        checkpoint = _checkpoint_path(config, test_epoch)
        if not checkpoint.exists():
            missing_rows.append(
                {
                    "variant": variant.label,
                    "missing": str(checkpoint),
                    "reason": "checkpoint not found",
                }
            )
            continue

        print(f"export dynamic-parameter diagnostics: {variant.label}")
        set_randomseed(config["random_seed"])
        eval_dataset = _build_eval_dataset(config)
        basin_ids = _load_basin_ids(config, expected=eval_dataset["xc_nn_norm"].shape[1])
        basin_index = {_normalize_basin_key(basin_id): i for i, basin_id in enumerate(basin_ids)}
        selected_indices = [
            basin_index[_normalize_basin_key(basin_id)]
            for basin_id in representative_basins
            if _normalize_basin_key(basin_id) in basin_index
        ]

        handler = LocalModelHandler(config, verbose=False)
        phy_name = config["model"]["phy"]["name"][0]
        nn_model = handler.model_dict[phy_name].nn_model
        nn_model.eval()
        param_names = _dynamic_param_names(config)
        variant_selected: dict[str, np.ndarray] = {}

        with torch.no_grad():
            xc = eval_dataset["xc_nn_norm"]
            for start in range(0, xc.shape[1], basin_batch_size):
                end = min(start + basin_batch_size, xc.shape[1])
                z1 = xc[:, start:end].to(config["device"])
                params = _predict_dynamic_parameter_tensor(
                    nn_model,
                    z1,
                    n_params=len(param_names),
                ).detach().cpu().numpy()
                diffs = np.diff(params, axis=0)
                trend_window = _trend_window_length(params.shape[0])

                variability = np.nanmedian(np.abs(diffs), axis=(0, 3))
                roughness = np.nanmean(np.square(diffs), axis=(0, 3))
                early = np.nanmean(params[:trend_window], axis=0)
                late = np.nanmean(params[-trend_window:], axis=0)
                long_term_shift = np.nanmedian(np.abs(late - early), axis=2)
                trend_to_noise = long_term_shift / np.maximum(variability, 1e-6)
                saturation = np.nanmean((params < 0.02) | (params > 0.98), axis=(0, 3))

                for local_basin, basin_id in enumerate(basin_ids[start:end]):
                    for p_idx, param_name in enumerate(param_names):
                        base = {
                            "variant": variant.label,
                            "variant_key": variant.key,
                            "basin_id": basin_id,
                            "parameter": param_name,
                            "normalization": variant.normalization,
                            "activation": variant.activation,
                            "add_conv": variant.add_conv,
                        }
                        variability_rows.append({**base, "variability": variability[local_basin, p_idx]})
                        roughness_rows.append({**base, "roughness": roughness[local_basin, p_idx]})
                        shift_rows.append({**base, "long_term_shift": long_term_shift[local_basin, p_idx]})
                        trend_noise_rows.append(
                            {**base, "trend_to_noise_ratio": trend_to_noise[local_basin, p_idx]}
                        )
                        saturation_rows.append({**base, "boundary_saturation_ratio": saturation[local_basin, p_idx]})

                for absolute_idx in selected_indices:
                    if start <= absolute_idx < end:
                        local_idx = absolute_idx - start
                        basin_id = basin_ids[absolute_idx]
                        variant_selected[basin_id] = _select_median_component_trajectory(
                            params[:, local_idx, :, :]
                        )

        selected_trajectories[variant.label] = {
            "variant_key": variant.key,
            "parameter_names": param_names,
            "trajectories": variant_selected,
        }
        np.savez_compressed(
            npz_dir / f"{variant.key}_selected_dynamic_parameter_trajectories.npz",
            parameter_names=np.array(param_names, dtype=object),
            basin_ids=np.array(list(variant_selected), dtype=object),
            **{f"basin_{basin_id}": data for basin_id, data in variant_selected.items()},
        )

    variability_df = pd.DataFrame(variability_rows)
    roughness_df = pd.DataFrame(roughness_rows)
    shift_df = pd.DataFrame(shift_rows)
    trend_noise_df = pd.DataFrame(trend_noise_rows)
    saturation_df = pd.DataFrame(saturation_rows)
    variability_df.to_csv(results_dir / "ablation_parameter_variability.csv", index=False)
    roughness_df.to_csv(results_dir / "ablation_parameter_roughness.csv", index=False)
    shift_df.to_csv(results_dir / "ablation_parameter_long_term_shift.csv", index=False)
    trend_noise_df.to_csv(results_dir / "ablation_parameter_trend_to_noise_ratio.csv", index=False)
    saturation_df.to_csv(results_dir / "ablation_boundary_saturation.csv", index=False)
    pd.DataFrame(missing_rows, columns=["variant", "missing", "reason"]).to_csv(
        results_dir / "missing_parameter_outputs.csv",
        index=False,
    )

    reliability_summary_df = _build_parameter_reliability_summary(
        variability_df,
        roughness_df,
        shift_df,
        trend_noise_df,
        saturation_df,
    )
    reliability_summary_df.to_csv(results_dir / "ablation_parameter_reliability_summary.csv", index=False)

    if not variability_df.empty:
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        sns.boxplot(
            data=variability_df,
            x="variant",
            y="variability",
            hue="variant",
            order=VARIANT_ORDER,
            hue_order=VARIANT_ORDER,
            palette=VARIANT_PALETTE,
            width=0.58,
            linewidth=0.8,
            fliersize=0,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Median |p[t+1] - p[t]|")
        ax.set_title("Dynamic-parameter temporal variability")
        ax.tick_params(axis="x", rotation=28)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        _finish_axis(ax)
        fig.tight_layout()
        fig.savefig(results_dir / "ablation_parameter_variability_boxplot.png", dpi=300)
        plt.close(fig)
        _plot_parameter_metric_heatmap(
            reliability_summary_df,
            value_col="mean_variability",
            title="Mean parameter variability",
            cbar_label="Mean variability",
            output_path=results_dir / "ablation_parameter_variability_heatmap.png",
        )
        _plot_parameter_metric_heatmap(
            reliability_summary_df,
            value_col="mean_roughness",
            title="Mean parameter roughness",
            cbar_label="Mean roughness",
            output_path=results_dir / "ablation_parameter_roughness_heatmap.png",
        )
        _plot_parameter_metric_heatmap(
            reliability_summary_df,
            value_col="mean_long_term_shift",
            title="Mean long-term parameter shift",
            cbar_label="Mean |late - early|",
            output_path=results_dir / "ablation_parameter_long_term_shift_heatmap.png",
        )
        _plot_parameter_metric_heatmap(
            reliability_summary_df,
            value_col="mean_trend_to_noise_ratio",
            title="Mean trend-to-noise ratio",
            cbar_label="Long-term shift / short-term variability",
            output_path=results_dir / "ablation_parameter_trend_to_noise_ratio_heatmap.png",
        )
        _plot_parameter_metric_heatmap(
            reliability_summary_df,
            value_col="mean_boundary_saturation_ratio",
            title="Mean boundary saturation ratio",
            cbar_label="Boundary saturation ratio",
            output_path=results_dir / "ablation_boundary_saturation_heatmap.png",
        )

    if not saturation_df.empty:
        _plot_boundary_saturation_boxplot(saturation_df, results_dir)

    _plot_representative_trajectories(selected_trajectories, trajectories_dir, trajectory_days)


def _plot_boundary_saturation_boxplot(saturation_df: pd.DataFrame, results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    group_centers = {"Sigmoid": 0.0, "Softsign": 1.0}
    group_offsets = {
        "Sigmoid": {
            "S4D-baseline": -0.27,
            "S4D-LN": -0.09,
            "S5D-ConvOnly": 0.09,
            "S5D-ConvLN-Sigmoid": 0.27,
        },
        "Softsign": {
            "S4D-Softsign": -0.27,
            "S4D-LN-Softsign": -0.09,
            "S5D-ConvBN-Softsign": 0.09,
            "S5D-full": 0.27,
        },
    }
    handles = []
    labels = []
    for activation, offsets in group_offsets.items():
        for variant, offset in offsets.items():
            values = saturation_df.loc[
                saturation_df["variant"].eq(variant),
                "boundary_saturation_ratio",
            ].dropna()
            if values.empty:
                continue
            position = group_centers[activation] + offset
            parts = ax.boxplot(
                values,
                positions=[position],
                widths=0.13,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "0.1", "linewidth": 0.9},
                whiskerprops={"color": "0.25", "linewidth": 0.7},
                capprops={"color": "0.25", "linewidth": 0.7},
                boxprops={"edgecolor": "0.25", "linewidth": 0.7},
            )
            color = VARIANT_PALETTE[variant]
            parts["boxes"][0].set_facecolor(color)
            parts["boxes"][0].set_alpha(0.78)
            handles.append(parts["boxes"][0])
            labels.append(variant)

    ax.set_xticks([group_centers["Sigmoid"], group_centers["Softsign"]])
    ax.set_xticklabels(["Sigmoid-based", "Softsign-based"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_xlabel("")
    ax.set_ylabel("Fraction p < 0.02 or p > 0.98")
    ax.set_title("Boundary saturation by activation family")
    _finish_axis(ax)
    ax.legend(handles, labels, title="", ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(results_dir / "ablation_boundary_saturation_boxplot.png", dpi=300)
    plt.close(fig)


def _plot_representative_trajectories(
    selected_trajectories: dict[str, dict[str, Any]],
    trajectories_dir: Path,
    trajectory_days: int,
) -> None:
    basin_ids = sorted(
        {
            basin_id
            for payload in selected_trajectories.values()
            for basin_id in payload["trajectories"]
        },
        key=lambda item: int(item) if str(item).isdigit() else str(item),
    )
    if not basin_ids:
        return

    first_payload = next(iter(selected_trajectories.values()))
    param_names = list(first_payload["parameter_names"])

    for basin_id in basin_ids:
        fig, axes = plt.subplots(
            len(param_names),
            len(TRAJECTORY_GROUPS),
            figsize=(max(9.2, 2.6 * len(TRAJECTORY_GROUPS)), 6.6),
            sharex=True,
            sharey="row",
        )
        if len(param_names) == 1:
            axes = np.expand_dims(axes, axis=0)
        for p_idx, param_name in enumerate(param_names):
            row_values = []
            for payload in selected_trajectories.values():
                trajectory = payload["trajectories"].get(basin_id)
                if trajectory is not None:
                    row_values.append(trajectory[trajectory_days:2*trajectory_days, p_idx])

            ymin = min((float(np.nanmin(values)) for values in row_values), default=0.0)
            ymax = max((float(np.nanmax(values)) for values in row_values), default=1.0)
            margin = max(0.02, 0.05 * (ymax - ymin))

            for group_idx, (group_name, group_variants) in enumerate(TRAJECTORY_GROUPS):
                ax = axes[p_idx, group_idx]
                for variant_label in group_variants:
                    payload = selected_trajectories.get(variant_label)
                    if payload is None:
                        continue
                    trajectory = payload["trajectories"].get(basin_id)
                    if trajectory is None:
                        continue
                    values = trajectory[trajectory_days:2*trajectory_days, p_idx]
                    ax.plot(
                        np.arange(len(values)),
                        values,
                        linewidth=0.9,
                        color=VARIANT_PALETTE[variant_label],
                        label=variant_label,
                    )
                if p_idx == 0:
                    ax.set_title(group_name)
                if group_idx == 0:
                    ax.set_ylabel(param_name)
                ax.set_ylim(max(0.0, ymin - margin), min(1.0, ymax + margin))
                ax.set_xlim(0, max(1, trajectory_days - 1))
                _finish_axis(ax)
        for ax in axes[-1, :]:
            ax.set_xlabel("Day")

        handles = []
        labels = []
        for ax in axes.ravel():
            axis_handles, axis_labels = ax.get_legend_handles_labels()
            for handle, label in zip(axis_handles, axis_labels):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.985))
        fig.suptitle(f"Basin {basin_id}: dynamic HBV parameters, first {trajectory_days} days", y=1.035)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(trajectories_dir / f"basin_{basin_id}_dynamic_parameter_trajectories.png", dpi=300)
        plt.close(fig)


def _select_representative_basins(results_dir: Path, requested: list[str]) -> list[str]:
    selected = [_normalize_basin_key(item) for item in requested]
    variability_path = results_dir / "ablation_parameter_variability.csv"
    if variability_path.exists():
        df = pd.read_csv(variability_path)
        if not df.empty:
            preferred = df[df["variant_key"].eq("s4d_baseline")]
            if preferred.empty:
                preferred = df
            basin_scores = preferred.groupby("basin_id")["variability"].mean().sort_values(ascending=False)
            if not basin_scores.empty:
                oscillatory = _normalize_basin_key(basin_scores.index[0])
                if oscillatory not in selected:
                    selected.append(oscillatory)
    return selected


def write_readme(seed: int, test_epoch: int, results_dir: Path) -> None:
    available, missing = _available_variants(seed, test_epoch)
    lines = [
        "# S5D Component-Wise Ablation Results",
        "",
        f"Seed: `{seed}`",
        f"Test epoch: `{test_epoch}`",
        "",
        "This is a controlled single-seed diagnostic ablation for the reviewer response.",
        "It is intended to compare component behavior, not to claim a robust accuracy gain.",
        "",
        "## Variants",
        "",
        "| Variant | Config | Conv | Norm | Activation | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    available_keys = {item["variant"].key for item in available}
    missing_by_key = {item["variant"].key: item for item in missing}
    for variant in VARIANTS:
        status = "available" if variant.key in available_keys else "missing"
        if variant.key in missing_by_key:
            flags = []
            if missing_by_key[variant.key]["missing_metrics"]:
                flags.append("metrics")
            if missing_by_key[variant.key]["missing_checkpoint"]:
                flags.append("checkpoint")
            status += f" ({', '.join(flags)})"
        lines.append(
            f"| {variant.label} | `{variant.config}` | {variant.add_conv} | "
            f"{variant.normalization} | {variant.activation} | {status} |"
        )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `ablation_performance_summary.csv`",
            "- `ablation_basinwise_metrics.csv`",
            "- `ablation_performance_boxplot.png`",
            "- `ablation_parameter_variability.csv`",
            "- `ablation_parameter_roughness.csv`",
            "- `ablation_parameter_long_term_shift.csv`",
            "- `ablation_parameter_trend_to_noise_ratio.csv`",
            "- `ablation_parameter_reliability_summary.csv`",
            "- `ablation_boundary_saturation.csv`",
            "- `ablation_parameter_variability_boxplot.png`",
            "- `ablation_parameter_variability_heatmap.png`",
            "- `ablation_parameter_roughness_heatmap.png`",
            "- `ablation_parameter_long_term_shift_heatmap.png`",
            "- `ablation_parameter_trend_to_noise_ratio_heatmap.png`",
            "- `ablation_boundary_saturation_heatmap.png`",
            "- `ablation_boundary_saturation_boxplot.png`",
            "- `parameter_trajectories/`",
            "- `configs/`",
            "",
            "## Interpretation Guardrail",
            "",
            "Small NSE/KGE differences should be reported as maintaining, reducing, or modestly improving skill.",
            "Primary interpretation should focus on parameter reliability by parameter: coherent long-term shift, low short-term noise, and low boundary saturation.",
            "Predictive accuracy is secondary evidence rather than the sole ranking criterion in this controlled ablation.",
        ]
    )
    if missing:
        lines.extend(
            [
                "",
                "## Missing Controlled Outputs",
                "",
                "The CSV/figure builders skip variants whose controlled ablation metrics or checkpoints are absent.",
                "Run missing variants with:",
                "",
                "```bash",
                "python project/bettermodel/ablation/s5d_ablation_pipeline.py run-experiments --seed 111 --test-epoch 100",
                "```",
            ]
        )
    (results_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_all(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    if args.run_missing:
        run_args = argparse.Namespace(
            seed=args.seed,
            test_epoch=args.test_epoch,
            results_dir=args.results_dir,
            force=False,
        )
        run_experiments(run_args)

    collect_performance(args.seed, args.test_epoch, results_dir)
    representative_basins = _select_representative_basins(results_dir, args.representative_basin)
    export_parameter_diagnostics(
        args.seed,
        args.test_epoch,
        results_dir,
        args.basin_batch_size,
        representative_basins,
        args.trajectory_days,
    )
    write_readme(args.seed, args.test_epoch, results_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and summarize S5D component-wise ablations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--seed", type=int, default=111)
        subparser.add_argument("--test-epoch", type=int, default=100)
        subparser.add_argument("--results-dir", default=str(RESULTS_DIR))

    run_parser = subparsers.add_parser("run-experiments", help="Run missing controlled ablation experiments.")
    add_common(run_parser)
    run_parser.add_argument("--force", action="store_true", help="Rerun even when metrics and checkpoint exist.")

    performance_parser = subparsers.add_parser("collect-performance")
    add_common(performance_parser)

    parameter_parser = subparsers.add_parser("export-parameters")
    add_common(parameter_parser)
    parameter_parser.add_argument("--basin-batch-size", type=int, default=25)
    parameter_parser.add_argument("--trajectory-days", type=int, default=DEFAULT_TRAJECTORY_DAYS)
    parameter_parser.add_argument(
        "--representative-basin",
        action="append",
        default=["1466500", "4105700", "6431500"],
        help="Representative basin id. Repeat to add more.",
    )

    readme_parser = subparsers.add_parser("write-readme")
    add_common(readme_parser)

    all_parser = subparsers.add_parser("all", help="Build all summaries/figures from available outputs.")
    add_common(all_parser)
    all_parser.add_argument("--run-missing", action="store_true")
    all_parser.add_argument("--basin-batch-size", type=int, default=25)
    all_parser.add_argument("--trajectory-days", type=int, default=DEFAULT_TRAJECTORY_DAYS)
    all_parser.add_argument(
        "--representative-basin",
        action="append",
        default=["1466500", "4105700", "6431500"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "run-experiments":
        run_experiments(args)
    elif args.command == "collect-performance":
        collect_performance(args.seed, args.test_epoch, Path(args.results_dir))
    elif args.command == "export-parameters":
        representative_basins = _select_representative_basins(
            Path(args.results_dir),
            args.representative_basin,
        )
        export_parameter_diagnostics(
            args.seed,
            args.test_epoch,
            Path(args.results_dir),
            args.basin_batch_size,
            representative_basins,
            args.trajectory_days,
        )
    elif args.command == "write-readme":
        write_readme(args.seed, args.test_epoch, Path(args.results_dir))
    elif args.command == "all":
        build_all(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
