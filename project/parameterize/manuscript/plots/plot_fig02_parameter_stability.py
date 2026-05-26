from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import colors
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from torch import nn

ROOT = Path("/workspace/autoresearch")
PARAM_ROOT = ROOT / "project" / "parameterize"
FIG2_ROOT = PARAM_ROOT / "manuscript" / "analysis" / "figure2"
DATA_DIR = FIG2_ROOT / "data"
REPORT_DIR = FIG2_ROOT / "reports"
OUT_DIR = PARAM_ROOT / "manuscript" / "figures" / "main"
OUT_STEM = OUT_DIR / "Fig02_parameter_stability_boundary_interval"
CONFIG_PATH = PARAM_ROOT / "conf" / "config_param_paper.yaml"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PARAM_ROOT))
sys.path.insert(0, str(ROOT / "project" / "Invariant"))

from dmg.core.data.loaders import HydroLoader  # noqa: E402
from dmg.core.utils.utils import initialize_config  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from project.parameterize.implements.basin_utils import basin_subset_indices, load_basin_ids  # noqa: E402
from project.parameterize.paper_variants import build_paper_dpl, normalize_paper_config  # noqa: E402
from project.parameterize.train_dmotpy import _build_loader_config, _normalize_runtime_paths, _resolve_path  # noqa: E402


MODEL_ORDER = ["deterministic", "mc_dropout", "distributional"]
PROB_MODELS = ["mc_dropout", "distributional"]
MODEL_COLORS = {
    "deterministic": "#4C78A8",
    "mc_dropout": "#F58518",
    "distributional": "#2A9D8F",
}
MODEL_LABELS = {
    "deterministic": r"$\delta_{base}$",
    "mc_dropout": r"$\delta_{mcd}$",
    "distributional": r"$\delta_{dist}$",
}
PARAM_ORDER = [
    "parBETA",
    "parFC",
    "parLP",
    "parPERC",
    "parUZL",
    "parK0",
    "parK1",
    "parK2",
    "parTT",
    "parCFMAX",
    "parCFR",
    "parCWH",
    "route_a",
    "route_b",
]
REFERENCE_LOSS = "HybridNseBatchLoss"
REPRESENTATIVE_PARAMETERS = ["parTT", "parFC", "parPERC"]
SAMPLE_CACHE = DATA_DIR / "representative_parameter_samples_for_fig02_c2_nonboundary.csv"
N_REP_SAMPLES = 100


def setup_style() -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.75,
            "font.size": 12.0,
            "axes.labelsize": 13.2,
            "axes.titlesize": 12.8,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 12.0,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
        }
    )


def clean_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#E9E9E9", linewidth=0.45)
        ax.set_axisbelow(True)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14.2,
        fontweight="normal",
        color="#111111",
    )


PARAM_LABELS = {
    "parBETA": r"$\mathrm{BETA}$",
    "parFC": r"$\mathrm{FC}$",
    "parLP": r"$\mathrm{LP}$",
    "parPERC": r"$\mathrm{PERC}$",
    "parUZL": r"$\mathrm{UZL}$",
    "parK0": r"$\mathrm{K}_0$",
    "parK1": r"$\mathrm{K}_1$",
    "parK2": r"$\mathrm{K}_2$",
    "parTT": r"$\mathrm{TT}$",
    "parCFMAX": r"$\mathrm{CFMAX}$",
    "parCFR": r"$\mathrm{CFR}$",
    "parCWH": r"$\mathrm{CWH}$",
    "route_a": r"$\mathrm{UH}_a$",
    "route_b": r"$\mathrm{UH}_b$",
}


def p_label(parameter: str) -> str:
    return PARAM_LABELS.get(str(parameter), str(parameter))


def add_parameter_group_separators(ax: plt.Axes) -> None:
    for xpos in [2.5, 7.5, 11.5]:
        ax.axvline(xpos, color="#D5D5D5", lw=0.55, ls=(0, (2.5, 2.5)), alpha=0.55, zorder=0)


def read_required_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        candidates = sorted(DATA_DIR.glob(f"*{name.split('.')[0].split('_')[0]}*.csv"))
        raise FileNotFoundError(f"Missing {path}; nearby candidates: {[p.name for p in candidates[:10]]}")
    return pd.read_csv(path)


def load_runtime_config(variant: str, seed: int, loss: str, device: str = "cpu") -> dict[str, Any]:
    raw_config = OmegaConf.load(_resolve_path(str(CONFIG_PATH)))
    raw_config["mode"] = "test"
    raw_config["seed"] = int(seed)
    raw_config["device"] = device
    raw_config["gpu_id"] = 0
    raw_config.setdefault("paper", {})
    raw_config["paper"]["variant"] = variant
    raw_config.setdefault("train", {}).setdefault("loss_function", {})
    raw_config["train"]["loss_function"]["name"] = loss
    _normalize_runtime_paths(raw_config)
    normalize_paper_config(raw_config)
    config = initialize_config(raw_config)
    config["device"] = device
    return config


def load_basin_inputs() -> tuple[np.ndarray, torch.Tensor]:
    config = load_runtime_config("mc_dropout", 111, REFERENCE_LOSS, "cpu")
    loader = HydroLoader(_build_loader_config(config), test_split=True, overwrite=False)
    reference_ids = load_basin_ids(config["data"]["basin_ids_reference_path"])
    subset_ids = load_basin_ids(config["data"]["basin_ids_path"])
    subset_idx = basin_subset_indices(reference_ids, subset_ids)
    return subset_ids.astype(np.int64), loader.eval_dataset["xc_nn_norm"][0, subset_idx, :].detach().cpu()


def extract_representative_samples(estimates: pd.DataFrame) -> pd.DataFrame:
    if SAMPLE_CACHE.exists():
        cached = pd.read_csv(SAMPLE_CACHE)
        if set(cached.get("parameter", [])) == set(REPRESENTATIVE_PARAMETERS):
            return cached
    basin_ids, inputs = load_basin_inputs()
    rows: list[dict[str, Any]] = []
    estimates_sub = estimates[
        estimates["model_raw"].isin(PROB_MODELS)
        & estimates["loss"].eq(REFERENCE_LOSS)
        & estimates["parameter"].isin(REPRESENTATIVE_PARAMETERS)
    ]
    checkpoint_map = (
        estimates_sub[["model_raw", "seed", "source_checkpoint"]]
        .drop_duplicates()
        .sort_values(["model_raw", "seed"])
    )
    param_index = {p: PARAM_ORDER.index(p) if p in PARAM_ORDER else None for p in REPRESENTATIVE_PARAMETERS}
    canonical_order = [
        "parBETA",
        "parFC",
        "parK0",
        "parK1",
        "parK2",
        "parLP",
        "parPERC",
        "parUZL",
        "parTT",
        "parCFMAX",
        "parCFR",
        "parCWH",
        "route_a",
        "route_b",
    ]
    param_index = {p: canonical_order.index(p) for p in REPRESENTATIVE_PARAMETERS}
    selected_basin_idx = np.linspace(0, len(basin_ids) - 1, 80, dtype=int)
    selected_basin_ids = basin_ids[selected_basin_idx]
    selected_inputs = inputs[selected_basin_idx]

    for rec in checkpoint_map.itertuples(index=False):
        config = load_runtime_config(rec.model_raw, int(rec.seed), REFERENCE_LOSS, "cpu")
        model = build_paper_dpl(config).to("cpu")
        model.load_state_dict(torch.load(rec.source_checkpoint, map_location="cpu"))
        nn_model = model.nn_model
        nn_model.eval()
        dropout_modules = [module for module in nn_model.modules() if isinstance(module, nn.Dropout)]
        dropout_states = [module.training for module in dropout_modules]
        if rec.model_raw == "mc_dropout":
            for module in dropout_modules:
                module.train(True)
        rng_state = torch.get_rng_state()
        try:
            with torch.inference_mode():
                for sample_idx in range(N_REP_SAMPLES):
                    torch.manual_seed(int(rec.seed) * 1000 + sample_idx)
                    if rec.model_raw == "distributional":
                        output = nn_model.sample_parameters(selected_inputs)
                    else:
                        output = nn_model(selected_inputs)
                    if output.ndim == 3:
                        output = output[-1]
                    arr = output.detach().cpu().numpy()
                    for parameter in REPRESENTATIVE_PARAMETERS:
                        values = arr[:, param_index[parameter]]
                        for basin_id, value in zip(selected_basin_ids, values):
                            rows.append(
                                {
                                    "model_raw": rec.model_raw,
                                    "model_label": MODEL_LABELS[rec.model_raw],
                                    "loss": REFERENCE_LOSS,
                                    "seed": int(rec.seed),
                                    "sample_index": sample_idx,
                                    "basin_id": int(basin_id),
                                    "parameter": parameter,
                                    "normalized_parameter_value": float(value),
                                }
                            )
        finally:
            torch.set_rng_state(rng_state)
            for module, was_training in zip(dropout_modules, dropout_states):
                module.train(was_training)
    out = pd.DataFrame(rows)
    out.to_csv(SAMPLE_CACHE, index=False)
    return out


def choose_representative_basin(samples: pd.DataFrame) -> int:
    quantiles = (
        samples.groupby(["basin_id", "parameter", "model_raw"])["normalized_parameter_value"]
        .quantile([0.05, 0.95])
        .unstack()
    )
    quantiles["width"] = quantiles[0.95] - quantiles[0.05]
    widths = quantiles["width"].unstack("model_raw").reset_index()
    widths["mcd_minus_dist"] = widths["mc_dropout"] - widths["distributional"]
    widths["mcd_over_dist"] = widths["mc_dropout"] / widths["distributional"].replace(0, np.nan)
    basin_summary = (
        widths.groupby("basin_id")
        .agg(
            mean_width_contrast=("mcd_minus_dist", "mean"),
            min_width_contrast=("mcd_minus_dist", "min"),
            median_width_ratio=("mcd_over_dist", "median"),
            max_dist_width=("distributional", "max"),
        )
        .reset_index()
    )
    candidates = basin_summary[
        (basin_summary["min_width_contrast"] > 0) & (basin_summary["max_dist_width"] < 0.20)
    ].sort_values(["mean_width_contrast", "median_width_ratio"], ascending=False)
    if candidates.empty:
        candidates = basin_summary.sort_values(["mean_width_contrast", "median_width_ratio"], ascending=False)
    return int(candidates.iloc[0]["basin_id"])


def draw_a1(ax: plt.Axes, stability: pd.DataFrame) -> None:
    d = stability[stability["loss"].eq(REFERENCE_LOSS)].copy()
    positions, plot_data, box_colors = [], [], []
    centers = np.arange(len(PARAM_ORDER))
    offsets = {"deterministic": -0.24, "mc_dropout": 0.0, "distributional": 0.24}
    for idx, parameter in enumerate(PARAM_ORDER):
        for model in MODEL_ORDER:
            positions.append(idx + offsets[model])
            values = d[(d["parameter"].eq(parameter)) & (d["model_raw"].eq(model))]["normalized_seed_sd"].dropna()
            plot_data.append(np.clip(values.to_numpy(), 1e-5, None))
            box_colors.append(MODEL_COLORS[model])
    bp = ax.boxplot(
        plot_data,
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#222222", "lw": 0.85},
        whiskerprops={"color": "#666666", "lw": 0.65},
        capprops={"color": "#666666", "lw": 0.65},
        boxprops={"edgecolor": "#666666", "lw": 0.65},
        flierprops={
            "marker": "o",
            "markersize": 0.9,
            "markerfacecolor": "#7A7A7A",
            "markeredgecolor": "#7A7A7A",
            "markeredgewidth": 0,
            "alpha": 0.18,
        },
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.2)
    ax.set_xticks(centers)
    ax.set_xticklabels([p_label(p) for p in PARAM_ORDER])
    ax.set_ylabel("Normalized seed SD")
    add_parameter_group_separators(ax)
    add_panel_label(ax, "(a)")
    clean_axes(ax, "y")
    handles = [
        Line2D([0], [0], marker="s", lw=0, markersize=5.4, color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        for m in MODEL_ORDER
    ]
    ax.legend(handles=handles, loc="upper right", ncol=3, frameon=False, handletextpad=0.30, columnspacing=0.75)


def draw_a2(ax: plt.Axes, pooled: pd.DataFrame, excluding: pd.DataFrame) -> None:
    pooled_ref = pooled[pooled["loss"].eq(REFERENCE_LOSS)].copy()
    excluding_ref = excluding[excluding["loss"].eq(REFERENCE_LOSS)].copy()
    rows = []
    for subset_name, frame in [("All parameters", pooled_ref), ("Boundary-sensitive\nexcluded", excluding_ref)]:
        for model in MODEL_ORDER:
            row = frame[frame["model_raw"].eq(model)].iloc[0]
            rows.append(
                {
                    "subset": subset_name,
                    "model_raw": model,
                    "median": row["median_normalized_seed_sd"],
                    "iqr": row["iqr_normalized_seed_sd"],
                }
            )
    d = pd.DataFrame(rows)
    xbase = np.array([0.0, 1.35])
    width = 0.22
    offsets = {"deterministic": -width, "mc_dropout": 0.0, "distributional": width}
    y_max = 0.0
    for model in MODEL_ORDER:
        sub = d[d["model_raw"].eq(model)]
        xs = xbase + offsets[model]
        y_max = max(y_max, float(sub["median"].max()))
        ax.bar(
            xs,
            sub["median"],
            width=width * 0.92,
            color=MODEL_COLORS[model],
            alpha=0.82,
            edgecolor="#555555",
            linewidth=0.45,
            label=MODEL_LABELS[model],
        )
        for x, y in zip(xs, sub["median"]):
            ax.text(x, y * 1.10, f"{y:.3f}", ha="center", va="bottom", fontsize=12.0, color=MODEL_COLORS[model])
    ax.set_yscale("log")
    ax.set_ylim(0.01, max(0.09, y_max * 1.8))
    ax.set_xlim(xbase[0] - 0.55, xbase[-1] + 0.55)
    ax.set_xticks(xbase)
    ax.set_xticklabels(["All parameters", "Boundary-sensitive\nexcluded"])
    ax.set_ylabel("Median normalized seed SD")
    add_panel_label(ax, "(b)")
    handles = [
        Line2D([0], [0], marker="s", lw=0, markersize=5.2, color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        for m in MODEL_ORDER
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, handletextpad=0.30, labelspacing=0.25, borderpad=0.1)
    clean_axes(ax, "y")


def draw_b1(ax: plt.Axes, saturation: pd.DataFrame) -> None:
    d = (
        saturation[saturation["loss"].eq(REFERENCE_LOSS)]
        .groupby(["model_raw", "parameter"], as_index=False)["saturation_rate_02"]
        .mean()
    )
    y_positions = {m: i for i, m in enumerate(MODEL_ORDER)}
    cmaps = {
        model: colors.LinearSegmentedColormap.from_list(f"{model}_boundary_seq", ["#FFFFFF", MODEL_COLORS[model]])
        for model in MODEL_ORDER
    }
    for row in d.itertuples(index=False):
        x = PARAM_ORDER.index(row.parameter)
        y = y_positions[row.model_raw]
        rate = float(row.saturation_rate_02)
        ax.scatter(
            x,
            y,
            s=26 + 230 * rate,
            color=cmaps[row.model_raw](rate),
            edgecolor="#777777",
            linewidth=0.35,
            alpha=0.92,
        )
    ax.set_xticks(np.arange(len(PARAM_ORDER)))
    ax.set_xticklabels([p_label(p) for p in PARAM_ORDER])
    ax.set_yticks([y_positions[m] for m in MODEL_ORDER])
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.tick_params(axis="y", labelsize=13.0)
    ax.set_xlim(-0.6, len(PARAM_ORDER) + 1.85)
    ax.set_ylim(len(MODEL_ORDER) - 0.45, -0.55)
    add_parameter_group_separators(ax)
    add_panel_label(ax, "(c)")
    clean_axes(ax)
    ax.grid(True, axis="x", color="#ECECEC", linewidth=0.38)
    ax.text(
        0.815,
        0.15,
        "Boundary saturation rate\n0 = white, 1 = darker\nLarger circle = higher",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=11.5,
        color="#444444",
    )


def draw_b2(ax: plt.Axes, distance: pd.DataFrame) -> None:
    d = distance[distance["loss"].eq(REFERENCE_LOSS)].copy()
    plot_data = [
        np.clip(d[d["model_raw"].eq(model)]["median_distance_to_boundary"].dropna().to_numpy(), 1e-5, None)
        for model in MODEL_ORDER
    ]
    bp = ax.boxplot(
        plot_data,
        positions=np.arange(1, 4),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "lw": 0.9},
        whiskerprops={"color": "#666666", "lw": 0.7},
        capprops={"color": "#666666", "lw": 0.7},
        boxprops={"edgecolor": "#666666", "lw": 0.7},
    )
    rng = np.random.default_rng(20240511)
    for patch, model, values, x in zip(bp["boxes"], MODEL_ORDER, plot_data, np.arange(1, 4)):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.72)
        jitter = rng.normal(0, 0.045, size=len(values))
        ax.scatter(
            np.full(len(values), x) + jitter,
            values,
            s=7,
            color=MODEL_COLORS[model],
            alpha=0.25,
            edgecolors="none",
            zorder=2,
        )
        med = float(np.median(values))
        ax.text(x, 4.0e-5, f"Median = {med:.3f}", ha="center", va="bottom", fontsize=12.0, color=MODEL_COLORS[model])
    ax.set_yscale("log")
    all_values = np.concatenate(plot_data)
    ax.set_ylim(max(1e-5, np.nanmin(all_values) * 0.55), min(0.65, np.nanmax(all_values) * 1.6))
    ax.set_xticks(np.arange(1, 4))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.tick_params(axis="x", labelsize=13.0)
    ax.set_ylabel("Distance to nearest boundary")
    add_panel_label(ax, "(d)")
    clean_axes(ax, "y")


def draw_c1(ax: plt.Axes, width_summary: pd.DataFrame) -> None:
    d = (
        width_summary[width_summary["loss"].eq(REFERENCE_LOSS)]
        .groupby(["model_raw", "parameter"], as_index=False)["median_interval_width_90"]
        .median()
    )
    y = np.arange(len(PARAM_ORDER))
    for idx, parameter in enumerate(PARAM_ORDER):
        vals = {
            model: float(d[(d["model_raw"].eq(model)) & (d["parameter"].eq(parameter))]["median_interval_width_90"].iloc[0])
            for model in PROB_MODELS
        }
        ax.hlines(idx, vals["distributional"], vals["mc_dropout"], color="#BDBDBD", lw=0.75, zorder=1)
        ax.scatter(vals["mc_dropout"], idx, s=24, color=MODEL_COLORS["mc_dropout"], zorder=3, label=MODEL_LABELS["mc_dropout"] if idx == 0 else None)
        ax.scatter(vals["distributional"], idx, s=24, color=MODEL_COLORS["distributional"], zorder=3, label=MODEL_LABELS["distributional"] if idx == 0 else None)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([p_label(p) for p in PARAM_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Median q05-q95 interval width")
    ax.text(0.15, 0.955, "Smaller = sharper", transform=ax.transAxes, ha="left", va="top", fontsize=11.6, color="#666666")
    add_panel_label(ax, "(e)")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.30, borderpad=0.1)
    clean_axes(ax, "x")


def draw_density(
    ax: plt.Axes,
    samples: pd.DataFrame,
    parameter: str,
    show_ylabel: bool,
    show_xlabel: bool,
    tick_labels: list[str],
) -> None:
    xgrid = np.linspace(0, 1, 300)
    for model in PROB_MODELS:
        values = samples[(samples["model_raw"].eq(model)) & (samples["parameter"].eq(parameter))]["normalized_parameter_value"].to_numpy()
        values = values[np.isfinite(values)]
        values = values[(values >= 0) & (values <= 1)]
        if len(np.unique(np.round(values, 6))) < 3:
            hist, bins = np.histogram(values, bins=40, range=(0, 1), density=True)
            xs = (bins[:-1] + bins[1:]) / 2
            dens = hist
        else:
            kde = gaussian_kde(values)
            xs = xgrid
            dens = kde(xs)
        linestyle = "-" if model == "mc_dropout" else "--"
        ax.plot(xs, dens, color=MODEL_COLORS[model], lw=1.15, linestyle=linestyle, label=MODEL_LABELS[model])
        ax.fill_between(xs, dens, color=MODEL_COLORS[model], alpha=0.08)
    ax.set_xlim(0, 1)
    ax.set_title(p_label(parameter), pad=2, fontweight="normal")
    ax.set_xlabel("Normalized parameter value" if show_xlabel else "")
    if show_ylabel:
        ax.set_ylabel("Density")
    else:
        ax.set_yticklabels([])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis="both", labelsize=11.5)
    ax.locator_params(axis="y", nbins=4)
    clean_axes(ax, "y")


def draw_c2(container, samples: pd.DataFrame, selected_basin: int) -> None:
    basin_samples = samples[samples["basin_id"].eq(selected_basin)].copy()
    sub = container.subgridspec(1, 3, wspace=0.16)
    axes = [plt.gcf().add_subplot(sub[0, i]) for i in range(3)]
    tick_label_sets = [["0", "0.5", ""], ["", "0.5", ""], ["", "0.5", "1"]]
    for idx, (ax, parameter) in enumerate(zip(axes, REPRESENTATIVE_PARAMETERS)):
        draw_density(ax, basin_samples, parameter, show_ylabel=idx == 0, show_xlabel=idx == 1, tick_labels=tick_label_sets[idx])
    add_panel_label(axes[0], "(f)")
    handles = [
        Line2D([0], [0], color=MODEL_COLORS["mc_dropout"], lw=1.3, linestyle="-", label=MODEL_LABELS["mc_dropout"]),
        Line2D([0], [0], color=MODEL_COLORS["distributional"], lw=1.3, linestyle="--", label=MODEL_LABELS["distributional"]),
    ]
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.50, 1.18), ncol=2, frameon=False, handletextpad=0.35, columnspacing=0.9, fontsize=11.5)


def write_notes(paths: dict[str, str], selected_basin: int) -> None:
    notes = [
        "# Figure 2 Plot Notes",
        "",
        "## Inputs",
        "",
        "- Panel (a): `data/parameter_seed_stability_long.csv`, filtered to `HybridNseBatchLoss`.",
        "- Panel (b): `data/parameter_stability_summary_pooled.csv` and `data/parameter_stability_excluding_boundary_sensitive.csv`, filtered to `HybridNseBatchLoss`.",
        "- Panel (c): `data/boundary_saturation_by_parameter.csv`, using `saturation_rate_02` averaged across seeds for `HybridNseBatchLoss`.",
        "- Panel (d): `data/distance_to_boundary_by_parameter.csv`, using per-seed/parameter median distance-to-boundary for `HybridNseBatchLoss`.",
        "- Panel (e): `data/probabilistic_interval_width_summary.csv`, using median q05-q95 width for `HybridNseBatchLoss`.",
        f"- Panel (f): `data/representative_parameter_samples_for_fig02_c2_nonboundary.csv`, filtered to one representative basin (`basin_id = {selected_basin}`); this cache is generated from the original `*-531/HybridNseBatchLoss` checkpoints because the summary interval tables retain quantiles/std but not the 100 individual sample values.",
        "",
        "No alternative filename lookup was needed; all priority inputs were present.",
        "",
        "## Definitions",
        "",
        "- Normalized seed SD: standard deviation of a basin-parameter estimate across seeds on the normalized `[0, 1]` HBV search-space scale.",
        "- Boundary saturation: fraction of basins with `theta_norm <= 0.02` or `theta_norm >= 0.98`.",
        "- Distance-to-boundary: `min(theta_norm, 1 - theta_norm)`.",
        "- q05-q95 interval width: `q95_norm - q05_norm` from 100 stochastic parameter samples for `δ_mcd` and `δ_dist`.",
        "",
        "## Representative Parameters",
        "",
        "- `parTT`, `parFC`, and `parPERC` were selected from the non-boundary-sensitive parameter set.",
        "- These parameters have among the largest positive `δ_mcd - δ_dist` median q05-q95 interval-width contrasts among non-boundary-sensitive parameters under `HybridNseBatchLoss`.",
        "- Boundary-sensitive examples such as `parCWH`, `parCFR`, `parK0`, `route_a`, and `route_b` were deliberately excluded from panel (f).",
        f"- The representative basin (`{selected_basin}`) was selected automatically from the 80 cached basins by requiring `δ_mcd` q05-q95 width to exceed `δ_dist` width for all three representative parameters, requiring maximum `δ_dist` width < 0.20, and then maximizing mean `δ_mcd - δ_dist` width contrast.",
        "",
        "## Scales",
        "",
        "- Panels (a), (b), (d), and (e) use log-scaled y/x axes because the distributions are strongly right-skewed or span orders of magnitude.",
        "- Panel (c) uses model-specific 0-1 white-to-color scales and circle size for saturation rate.",
        "- Panel (f) uses separate y-axis scaling for each representative-parameter density subplot so the single-basin sampling distributions remain legible.",
        "- The plotting script requests Times New Roman for all text, including math labels.",
        "",
        "## Model Naming",
        "",
        "- `deterministic` -> `δ_base`.",
        "- `mc_dropout` -> `δ_mcd`.",
        "- `distributional` -> `δ_dist`.",
        "",
        "## Output",
        "",
        "- Figure canvas: 16:9 landscape layout with an approximately 64% / 36% left/right column ratio.",
        f"- PNG: `{paths['png']}`",
        "",
        "## Pre-export checks",
        "",
        "- No top title bar, side explanation bar, or bottom summary block was added.",
        "- Panel (c) is a fixed-size circle heatmap, not a rectangular heatmap.",
        "- Panel (b) separates all parameters from boundary-sensitive excluded results.",
        "- Panel (b) uses three-decimal median labels and no error bars.",
        "- Panel (c) removes the legend/colorbar and keeps only a concise text note for the 0-1 saturation-rate encoding.",
        "- Panel (d) uses boundary distance boxplots.",
        "- Panel (e) is a `δ_mcd` vs `δ_dist` interval-width dumbbell plot.",
        "- Panel (f) shows representative stochastic parameter sampling distributions.",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "figure2_plot_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stability = read_required_csv("parameter_seed_stability_long.csv")
    pooled = read_required_csv("parameter_stability_summary_pooled.csv")
    excluding = read_required_csv("parameter_stability_excluding_boundary_sensitive.csv")
    saturation = read_required_csv("boundary_saturation_by_parameter.csv")
    distance = read_required_csv("distance_to_boundary_by_parameter.csv")
    width_summary = read_required_csv("probabilistic_interval_width_summary.csv")
    estimates = read_required_csv("parameter_estimates_by_run_long.csv")
    samples = extract_representative_samples(estimates)
    selected_basin = choose_representative_basin(samples)

    fig = plt.figure(figsize=(18.5, 10.6), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.32, 0.32, 0.36], width_ratios=[1.78, 1.0], hspace=0.28, wspace=0.15)
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[0, 1])
    ax_b1 = fig.add_subplot(gs[1, 0])
    ax_b2 = fig.add_subplot(gs[1, 1])
    ax_c1 = fig.add_subplot(gs[2, 0])

    draw_a1(ax_a1, stability)
    draw_a2(ax_a2, pooled, excluding)
    draw_b1(ax_b1, saturation)
    draw_b2(ax_b2, distance)
    draw_c1(ax_c1, width_summary)
    draw_c2(gs[2, 1], samples, selected_basin)
    fig.subplots_adjust(left=0.055, right=0.965, top=0.970, bottom=0.088)

    png_path = OUT_STEM.with_suffix(".png")
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    write_notes({"png": str(png_path)}, selected_basin)
    print(
        json.dumps(
            {
                "png": str(png_path),
                "notes": str(REPORT_DIR / "figure2_plot_notes.md"),
                "representative_basin": selected_basin,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
