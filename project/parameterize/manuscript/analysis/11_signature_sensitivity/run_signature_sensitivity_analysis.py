from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dmg.core.data.loaders import HydroLoader
from dmg.core.utils.utils import initialize_config
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[5]
PARAM_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.parameterize.implements import build_paper_dpl
from project.parameterize.implements.basin_utils import (
    basin_subset_indices,
    load_basin_ids,
    subset_dataset_by_indices,
)
from project.parameterize.implements.differentiable_signatures import (
    baseflow_index,
    calibrate_mean_annual_peak_tau,
    mean_annual_peak,
    recession_constant,
    total_runoff_volume,
    water_year_ids_from_dates,
)
from project.parameterize.manuscript.plots.common import (
    FigureSpec,
    PARAM_LABELS,
    PARAM_ORDER,
    clean_axes,
    figure,
    muted_diverging,
    p_label,
    save,
    setup_style,
)
from project.parameterize.paper_variants import normalize_paper_config
from project.parameterize.train_dmotpy import (
    _build_loader_config,
    _normalize_runtime_paths,
    _resolve_path,
)


CONFIG_PATH = PARAM_ROOT / "conf" / "config_param_paper.yaml"
OUTPUT_ROOT = PARAM_ROOT / "outputs" / "distributional-531" / "HybridNseBatchLoss"
ANALYSIS_ROOT = PARAM_ROOT / "manuscript" / "analysis" / "11_signature_sensitivity"
DATA_DIR = ANALYSIS_ROOT / "data"
FIG_DIR = ANALYSIS_ROOT / "figures"
REPORT_DIR = ANALYSIS_ROOT / "reports"
METHODS_DIR = ANALYSIS_ROOT / "methods"
LOG_DIR = ANALYSIS_ROOT / "logs"
SEEDS = (111, 222, 333, 444, 555)
SIGNATURE_ORDER = (
    "total_runoff_volume",
    "mean_annual_peak",
    "recession_constant",
    "baseflow_index",
)
SIGNATURE_LABELS = {
    "total_runoff_volume": "Total runoff",
    "mean_annual_peak": "Mean annual peak",
    "recession_constant": "Recession const.",
    "baseflow_index": "BFI",
}
PLOT_SIGNATURE_LABELS = {
    "total_runoff_volume": "Total runoff",
    "mean_annual_peak": "Annual peak",
    "recession_constant": "Recession const.",
    "baseflow_index": "BFI",
}
PRIMARY_SIGNATURE = "mean_annual_peak"
FOCUS_PARAMETERS = ("parPERC", "parTT")
EPS = 1.0e-12


@dataclass(frozen=True)
class LoadedData:
    dataset: dict[str, torch.Tensor]
    basin_ids: np.ndarray
    eval_dates: tuple[datetime, ...]
    config: dict


def _ensure_dirs() -> None:
    for path in (DATA_DIR, FIG_DIR, REPORT_DIR, METHODS_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _load_runtime_config(seed: int) -> dict:
    raw_config = OmegaConf.load(_resolve_path(str(CONFIG_PATH)))
    raw_config["mode"] = "test"
    raw_config["seed"] = int(seed)
    raw_config["device"] = "cpu"
    raw_config["gpu_id"] = 0
    raw_config.setdefault("paper", {})
    raw_config["paper"]["variant"] = "distributional"
    raw_config.setdefault("train", {}).setdefault("loss_function", {})
    raw_config["train"]["loss_function"]["name"] = "HybridNseBatchLoss"
    _normalize_runtime_paths(raw_config)
    normalize_paper_config(raw_config)
    return initialize_config(raw_config)


@lru_cache(maxsize=1)
def _loaded_data() -> LoadedData:
    config = _load_runtime_config(SEEDS[0])
    loader = HydroLoader(_build_loader_config(config), test_split=True, overwrite=False)
    reference_ids = load_basin_ids(config["data"]["basin_ids_reference_path"])
    basin_ids = load_basin_ids(config["data"]["basin_ids_path"])
    subset_idx = basin_subset_indices(reference_ids, basin_ids)
    dataset = subset_dataset_by_indices(loader.eval_dataset, subset_idx)
    start = datetime.strptime(config["test_time"][0], "%Y/%m/%d")
    total_steps = int(dataset["x_phy"].shape[0])
    eval_dates = tuple(start + timedelta(days=index) for index in range(total_steps))
    return LoadedData(dataset=dataset, basin_ids=basin_ids, eval_dates=eval_dates, config=config)


def _checkpoint_path(seed: int) -> Path:
    return OUTPUT_ROOT / f"seed_{seed}" / "model" / "model_epoch100.pt"


def _load_model(seed: int) -> torch.nn.Module:
    config = _load_runtime_config(seed)
    model = build_paper_dpl(config).to("cpu")
    state_dict = torch.load(_checkpoint_path(seed), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _last_parameter_slice(parameters: torch.Tensor) -> torch.Tensor:
    return parameters[-1] if parameters.ndim == 3 else parameters


def _water_year_ids() -> torch.Tensor:
    loaded = _loaded_data()
    warm_up = int(loaded.config["model"]["phy"].get("warm_up", 365))
    effective_dates = loaded.eval_dates[warm_up:]
    return water_year_ids_from_dates(effective_dates)


def _build_signature_map(
    q: torch.Tensor,
    water_year_ids: torch.Tensor,
    tau: float,
) -> dict[str, torch.Tensor]:
    return {
        "total_runoff_volume": total_runoff_volume(q),
        "mean_annual_peak": mean_annual_peak(q, water_year_ids=water_year_ids.to(q.device), tau=tau),
        "recession_constant": recession_constant(q),
        "baseflow_index": baseflow_index(q),
    }


def _seed_working_point(seed: int) -> dict[str, object]:
    loaded = _loaded_data()
    model = _load_model(seed)
    water_year_ids = _water_year_ids()

    rng_state = torch.get_rng_state()
    try:
        with torch.no_grad():
            torch.manual_seed(int(seed) * 1000)
            parameters = model.nn_model(loaded.dataset["xc_nn_norm"])
            q_ref = model.phy_model(loaded.dataset, parameters)["streamflow"]
            physical_theta = model.phy_model.physical_parameters_from_normalized(parameters)
    finally:
        torch.set_rng_state(rng_state)

    normalized_theta = _last_parameter_slice(parameters).detach().cpu()
    tau_info = calibrate_mean_annual_peak_tau(q_ref, water_year_ids=water_year_ids)
    tau = float(tau_info["selected_tau"])

    theta = physical_theta.detach().clone().requires_grad_(True)
    q = model.phy_model.forward_from_physical(loaded.dataset, theta)["streamflow"]
    signatures = _build_signature_map(q, water_year_ids, tau=tau)
    signature_tensor = torch.stack([signatures[name] for name in SIGNATURE_ORDER], dim=-1)

    raw_grads = []
    for index, name in enumerate(SIGNATURE_ORDER):
        grad = torch.autograd.grad(
            signatures[name].sum(),
            theta,
            retain_graph=index < len(SIGNATURE_ORDER) - 1,
        )[0]
        raw_grads.append(grad)
    raw_grad_tensor = torch.stack(raw_grads, dim=-1)

    sigma_theta = theta.detach().std(dim=0, unbiased=False).clamp_min(EPS)
    sigma_signature = signature_tensor.detach().std(dim=0, unbiased=False).clamp_min(EPS)
    standardized = raw_grad_tensor * sigma_theta.view(1, -1, 1) / sigma_signature.view(1, 1, -1)

    near_boundary_02 = (normalized_theta <= 0.02) | (normalized_theta >= 0.98)
    near_boundary_05 = (normalized_theta <= 0.05) | (normalized_theta >= 0.95)
    distance = torch.minimum(normalized_theta, 1.0 - normalized_theta)

    return {
        "seed": seed,
        "normalized_theta": normalized_theta.numpy(),
        "physical_theta": theta.detach().cpu().numpy(),
        "q_ref": q_ref.detach().cpu().numpy(),
        "signature_values": signature_tensor.detach().cpu().numpy(),
        "raw_gradients": raw_grad_tensor.detach().cpu().numpy(),
        "standardized_sensitivities": standardized.detach().cpu().numpy(),
        "sigma_theta": sigma_theta.detach().cpu().numpy(),
        "sigma_signature": sigma_signature.detach().cpu().numpy(),
        "near_boundary_02": near_boundary_02.detach().cpu().numpy(),
        "near_boundary_05": near_boundary_05.detach().cpu().numpy(),
        "distance_to_boundary": distance.detach().cpu().numpy(),
        "tau_info": tau_info,
    }


def _seed_basin_long(seed_result: dict[str, object], basin_ids: np.ndarray) -> pd.DataFrame:
    seed = int(seed_result["seed"])
    physical_theta = np.asarray(seed_result["physical_theta"], dtype=np.float64)
    signature_values = np.asarray(seed_result["signature_values"], dtype=np.float64)
    raw_gradients = np.asarray(seed_result["raw_gradients"], dtype=np.float64)
    standardized = np.asarray(seed_result["standardized_sensitivities"], dtype=np.float64)
    sigma_theta = np.asarray(seed_result["sigma_theta"], dtype=np.float64)
    sigma_signature = np.asarray(seed_result["sigma_signature"], dtype=np.float64)

    frames: list[pd.DataFrame] = []
    for sig_idx, signature in enumerate(SIGNATURE_ORDER):
        for param_idx, parameter in enumerate(PARAM_ORDER):
            frames.append(
                pd.DataFrame(
                    {
                        "seed": seed,
                        "basin_id": basin_ids,
                        "parameter": parameter,
                        "parameter_label": PARAM_LABELS[parameter],
                        "signature": signature,
                        "signature_label": SIGNATURE_LABELS[signature],
                        "physical_theta": physical_theta[:, param_idx],
                        "signature_value": signature_values[:, sig_idx],
                        "raw_gradient": raw_gradients[:, param_idx, sig_idx],
                        "standardized_sensitivity": standardized[:, param_idx, sig_idx],
                        "sigma_theta": sigma_theta[param_idx],
                        "sigma_signature": sigma_signature[sig_idx],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _seed_parameter_summary(seed_result: dict[str, object]) -> pd.DataFrame:
    seed = int(seed_result["seed"])
    standardized = np.asarray(seed_result["standardized_sensitivities"], dtype=np.float64)
    rows: list[dict[str, object]] = []
    for param_idx, parameter in enumerate(PARAM_ORDER):
        for sig_idx, signature in enumerate(SIGNATURE_ORDER):
            values = standardized[:, param_idx, sig_idx]
            rows.append(
                {
                    "seed": seed,
                    "parameter": parameter,
                    "signature": signature,
                    "median_standardized_sensitivity": float(np.median(values)),
                    "median_abs_standardized_sensitivity": float(np.median(np.abs(values))),
                    "mean_standardized_sensitivity": float(np.mean(values)),
                    "mean_abs_standardized_sensitivity": float(np.mean(np.abs(values))),
                }
            )
    return pd.DataFrame(rows)


def _boundary_summary(seed_results: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in seed_results:
        near02 = np.asarray(result["near_boundary_02"], dtype=bool)
        near05 = np.asarray(result["near_boundary_05"], dtype=bool)
        distance = np.asarray(result["distance_to_boundary"], dtype=np.float64)
        for param_idx, parameter in enumerate(PARAM_ORDER):
            rows.append(
                {
                    "seed": int(result["seed"]),
                    "parameter": parameter,
                    "near_boundary_share_02": float(near02[:, param_idx].mean()),
                    "near_boundary_share_05": float(near05[:, param_idx].mean()),
                    "mean_distance_to_boundary": float(distance[:, param_idx].mean()),
                    "median_distance_to_boundary": float(np.median(distance[:, param_idx])),
                }
            )
    seed_boundary = pd.DataFrame(rows)
    summary = (
        seed_boundary.groupby("parameter", as_index=False)
        .agg(
            mean_near_boundary_share_02=("near_boundary_share_02", "mean"),
            sd_near_boundary_share_02=("near_boundary_share_02", "std"),
            mean_near_boundary_share_05=("near_boundary_share_05", "mean"),
            sd_near_boundary_share_05=("near_boundary_share_05", "std"),
            mean_distance_to_boundary=("mean_distance_to_boundary", "mean"),
            median_distance_to_boundary=("median_distance_to_boundary", "median"),
        )
    )
    summary["boundary_caution_flag"] = (
        (summary["mean_near_boundary_share_02"] > 0.30)
        | (summary["mean_near_boundary_share_05"] > 0.50)
    )
    summary["tt_partial_estimability_flag"] = summary["parameter"].eq("parTT")
    return seed_boundary.merge(summary, on="parameter", how="left")


def _seed_tau_summary(seed_results: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for result in seed_results:
        tau_info = dict(result["tau_info"])
        rows.append(
            {
                "seed": int(result["seed"]),
                "selected_tau": float(tau_info["selected_tau"]),
                "selected_max_relative_error": float(tau_info["selected_max_relative_error"]),
                "selected_mean_relative_error": float(tau_info["selected_mean_relative_error"]),
                "selected_median_relative_error": float(tau_info["selected_median_relative_error"]),
            }
        )
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def _aggregate_cross_seed(seed_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregated = (
        seed_summary.groupby(["parameter", "signature"], as_index=False)
        .agg(
            across_seed_median=("median_standardized_sensitivity", "median"),
            across_seed_sd=("median_standardized_sensitivity", "std"),
            across_seed_mean_abs=("median_abs_standardized_sensitivity", "mean"),
        )
    )
    median_matrix = (
        aggregated.pivot(index="parameter", columns="signature", values="across_seed_median")
        .reindex(index=PARAM_ORDER, columns=SIGNATURE_ORDER)
    )
    sd_matrix = (
        aggregated.pivot(index="parameter", columns="signature", values="across_seed_sd")
        .reindex(index=PARAM_ORDER, columns=SIGNATURE_ORDER)
    )
    return aggregated, median_matrix, sd_matrix


def _matrix_sanity_assertions(median_matrix: pd.DataFrame, sd_matrix: pd.DataFrame) -> None:
    if median_matrix.shape != (len(PARAM_ORDER), len(SIGNATURE_ORDER)):
        raise AssertionError(
            f"Expected standardized sensitivity median matrix shape {(len(PARAM_ORDER), len(SIGNATURE_ORDER))}, "
            f"got {median_matrix.shape}."
        )
    if sd_matrix.shape != (len(PARAM_ORDER), len(SIGNATURE_ORDER)):
        raise AssertionError(
            f"Expected across-seed SD matrix shape {(len(PARAM_ORDER), len(SIGNATURE_ORDER))}, got {sd_matrix.shape}."
        )
    if median_matrix.isna().any().any():
        raise AssertionError("Standardized sensitivity median matrix contains NaN values.")
    if sd_matrix.isna().any().any():
        raise AssertionError("Across-seed SD matrix contains NaN values.")
    if (sd_matrix.to_numpy(dtype=float) < 0.0).any():
        raise AssertionError("Across-seed SD matrix contains negative values.")


def _sigma_theta_summary(seed_results: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in seed_results:
        sigma_theta = np.asarray(result["sigma_theta"], dtype=np.float64)
        seed = int(result["seed"])
        seed_median = float(np.median(sigma_theta))
        seed_min = float(np.min(sigma_theta))
        seed_max = float(np.max(sigma_theta))
        descending_rank = pd.Series(sigma_theta, index=PARAM_ORDER).rank(method="dense", ascending=False)
        for param_idx, parameter in enumerate(PARAM_ORDER):
            rows.append(
                {
                    "seed": seed,
                    "parameter": parameter,
                    "sigma_theta": float(sigma_theta[param_idx]),
                    "sigma_theta_rank_desc": int(descending_rank.loc[parameter]),
                    "sigma_theta_seed_median_all_params": seed_median,
                    "sigma_theta_seed_min_all_params": seed_min,
                    "sigma_theta_seed_max_all_params": seed_max,
                    "sigma_theta_over_seed_median": float(sigma_theta[param_idx] / max(seed_median, EPS)),
                }
            )
    return pd.DataFrame(rows)


def _focused_zero_sensitivity_diagnostics(
    basin_long: pd.DataFrame,
    sigma_theta_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    focused = basin_long.loc[basin_long["parameter"].isin(FOCUS_PARAMETERS)].copy()
    safe_signature = focused["signature_value"].where(focused["signature_value"].abs() >= EPS, np.nan)
    focused["half_elasticity"] = focused["raw_gradient"] * focused["physical_theta"] / safe_signature

    diagnostics = (
        focused.groupby(["seed", "parameter", "signature"], as_index=False)
        .agg(
            raw_gradient_median=("raw_gradient", "median"),
            raw_gradient_abs_median=("raw_gradient", lambda s: float(np.median(np.abs(s)))),
            half_elasticity_median=("half_elasticity", "median"),
            half_elasticity_abs_median=("half_elasticity", lambda s: float(np.nanmedian(np.abs(s)))),
            physical_theta_median=("physical_theta", "median"),
            signature_value_median=("signature_value", "median"),
        )
        .merge(sigma_theta_summary, on=["seed", "parameter"], how="left")
        .sort_values(["parameter", "seed", "signature"])
        .reset_index(drop=True)
    )

    verdict_rows: list[dict[str, object]] = []
    for parameter, sub in diagnostics.groupby("parameter", sort=True):
        max_abs_raw = float(sub["raw_gradient_abs_median"].max())
        max_abs_half = float(sub["half_elasticity_abs_median"].max())
        sigma_ratio = float(sub["sigma_theta_over_seed_median"].median())
        sigma_rank = float(sub["sigma_theta_rank_desc"].median())
        sigma_small = sigma_ratio < 0.25
        if max_abs_half < 1.0e-2 and max_abs_raw < 1.0e-2:
            verdict = "real_small_in_raw_and_half_elasticity"
            reason = (
                f"max |raw median|={max_abs_raw:.3e}, max |half-elasticity median|={max_abs_half:.3e}; "
                f"sigma_theta / seed-median={sigma_ratio:.2f}"
            )
        elif sigma_small and max_abs_half >= 1.0e-2:
            verdict = "normalization_artifact_candidate"
            reason = (
                f"sigma_theta is compressed relative to other parameters (ratio={sigma_ratio:.2f}, rank={sigma_rank:.1f}), "
                f"but half-elasticity remains non-trivial (max={max_abs_half:.3e})"
            )
        else:
            verdict = "real_small_not_sigma_theta_artifact"
            reason = (
                f"sigma_theta is not exceptionally tiny (ratio={sigma_ratio:.2f}, rank={sigma_rank:.1f}); "
                f"max |half-elasticity median|={max_abs_half:.3e}"
            )
        verdict_rows.append(
            {
                "parameter": parameter,
                "verdict": verdict,
                "reason": reason,
                "max_abs_raw_gradient_median": max_abs_raw,
                "max_abs_half_elasticity_median": max_abs_half,
                "median_sigma_theta_over_seed_median": sigma_ratio,
                "median_sigma_theta_rank_desc": sigma_rank,
            }
        )
    verdicts = pd.DataFrame(verdict_rows).sort_values("parameter").reset_index(drop=True)
    return diagnostics, verdicts


def _control_class_crosswalk(
    aggregated: pd.DataFrame,
    boundary_summary: pd.DataFrame,
) -> pd.DataFrame:
    dominant = pd.read_csv(
        PARAM_ROOT / "manuscript" / "analysis" / "01_model_consistency" / "data" / "model_dominant_consistency_summary.csv"
    )
    peak = aggregated.loc[aggregated["signature"].eq(PRIMARY_SIGNATURE), ["parameter", "across_seed_median", "across_seed_sd"]].rename(
        columns={
            "across_seed_median": "peak_signature_sensitivity_median",
            "across_seed_sd": "peak_signature_sensitivity_sd",
        }
    )
    max_abs_rows = []
    for parameter, sub in aggregated.groupby("parameter", sort=False):
        idx = sub["across_seed_median"].abs().idxmax()
        row = sub.loc[idx]
        max_abs_rows.append(
            {
                "parameter": parameter,
                "max_abs_signature": row["signature"],
                "max_abs_sensitivity_median": float(abs(row["across_seed_median"])),
                "max_abs_sensitivity_signed_median": float(row["across_seed_median"]),
                "max_abs_sensitivity_sd": float(row["across_seed_sd"]),
            }
        )
    max_abs = pd.DataFrame(max_abs_rows)
    boundary_flags = (
        boundary_summary.groupby("parameter", as_index=False)
        .agg(
            mean_near_boundary_share_02=("mean_near_boundary_share_02", "first"),
            mean_near_boundary_share_05=("mean_near_boundary_share_05", "first"),
            boundary_caution_flag=("boundary_caution_flag", "first"),
            tt_partial_estimability_flag=("tt_partial_estimability_flag", "first"),
        )
    )
    crosswalk = dominant.merge(peak, on="parameter", how="left").merge(max_abs, on="parameter", how="left").merge(
        boundary_flags, on="parameter", how="left"
    )
    crosswalk["parameter_label"] = crosswalk["parameter"].map(lambda item: p_label(item))
    crosswalk["peak_signature_abs_median"] = crosswalk["peak_signature_sensitivity_median"].abs()
    class_rank = {
        "shared dominant controls": 0,
        "partially shared controls": 1,
        "model-sensitive controls": 2,
    }
    crosswalk["relationship_class_rank"] = crosswalk["relationship_class"].map(class_rank).fillna(99).astype(int)
    return crosswalk.sort_values(["relationship_class_rank", "parameter"]).reset_index(drop=True)


def _write_methods() -> None:
    lines = [
        "# Signature Sensitivity Methods",
        "",
        "- Formulation: distributional (`delta_dist`) with `HybridNseBatchLoss`.",
        "- Seeds: 111, 222, 333, 444, 555.",
        "- Working point per seed: one sampled normalized parameter field from `model.nn_model(xc_nn_norm)` after `torch.manual_seed(seed * 1000)`, converted to physical scale with `physical_parameters_from_normalized`.",
        "- Forward path for gradients: `HbvStatic.forward_from_physical`, preserving the existing hard rain/snow partition and post-warm-up sensitivity definition.",
        "- Signatures: total runoff volume, calibrated soft mean annual peak, soft recession constant, and Lyne-Hollick baseflow index.",
        "- Jacobian extraction: `autograd.grad(signature.sum(), theta)` for each signature, relying on basinwise independence of the routed HBV forward so the summed gradient returns the per-basin diagonal Jacobian rows.",
        "- Standardized sensitivity: `(dS/dtheta_j) * (sigma_theta_j / sigma_S)` using cross-basin standard deviations within each seed.",
        "- Cross-seed summary: per-seed cross-basin median standardized sensitivity, then across-seed median and standard deviation.",
        "- Boundary fusion: near-boundary shares at 0.02/0.98 and 0.05/0.95 thresholds, with caution flags when mean share exceeds 0.30 or 0.50 respectively.",
        "- `parTT` is explicitly marked partially estimable because the hard rain/snow phase partition remains non-differentiable.",
    ]
    (METHODS_DIR / "method_definitions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_heatmap(
    median_matrix: pd.DataFrame,
    sd_matrix: pd.DataFrame,
    boundary_flags: pd.DataFrame,
) -> Path:
    setup_style()
    spec = FigureSpec(
        stem="distributional_hybrid_signature_sensitivity_heatmap",
        directory=FIG_DIR,
        width_mm=188.0,
        height_mm=158.0,
    )
    fig, axes = figure(spec, rows=1, cols=2, gridspec_kw={"width_ratios": [5.6, 1.2]})
    ax_heat, ax_bound = axes

    values = median_matrix.to_numpy(dtype=float)
    bound = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    im = ax_heat.imshow(values, aspect="auto", cmap=muted_diverging(), vmin=-bound, vmax=bound)

    flag_lookup = (
        boundary_flags.groupby("parameter", as_index=False)
        .agg(
            boundary_caution_flag=("boundary_caution_flag", "first"),
            mean_near_boundary_share_02=("mean_near_boundary_share_02", "first"),
        )
        .set_index("parameter")
    )
    for row_idx, parameter in enumerate(median_matrix.index):
        if bool(flag_lookup.loc[parameter, "boundary_caution_flag"]):
            ax_heat.axhspan(row_idx - 0.5, row_idx + 0.5, color="#D7DADC", alpha=0.22, zorder=0)

    for row_idx, parameter in enumerate(median_matrix.index):
        for col_idx, signature in enumerate(median_matrix.columns):
            cell_value = float(median_matrix.loc[parameter, signature])
            cell_sd = float(sd_matrix.loc[parameter, signature])
            if not np.isfinite(cell_value):
                continue
            text_color = "white" if abs(cell_value) >= 0.55 * bound and bound > 0 else "#222222"
            ax_heat.text(
                col_idx,
                row_idx,
                f"{cell_value:.2f}\n±{0.0 if np.isnan(cell_sd) else cell_sd:.2f}",
                ha="center",
                va="center",
                fontsize=8.3,
                color=text_color,
            )

    ylabels = []
    for parameter in median_matrix.index:
        label = p_label(parameter)
        if parameter == "parTT":
            label = label + r"$^{\dagger}$"
        ylabels.append(label)
    ax_heat.set_xticks(np.arange(len(SIGNATURE_ORDER)))
    ax_heat.set_xticklabels([PLOT_SIGNATURE_LABELS[name] for name in SIGNATURE_ORDER], rotation=20, ha="right")
    ax_heat.set_yticks(np.arange(len(PARAM_ORDER)))
    ax_heat.set_yticklabels(ylabels)
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.set_title("Standardized response sensitivity")

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.02)
    cbar.set_label("Across-seed median standardized sensitivity")
    cbar.ax.tick_params(length=2)

    boundary_values = flag_lookup.loc[median_matrix.index, "mean_near_boundary_share_02"].to_numpy(dtype=float)
    boundary_flags_arr = flag_lookup.loc[median_matrix.index, "boundary_caution_flag"].to_numpy(dtype=bool)
    ax_bound.barh(
        np.arange(len(boundary_values)),
        boundary_values,
        color=np.where(boundary_flags_arr, "#8F969C", "#D8DDDF"),
        edgecolor="#5B646B",
        height=0.7,
        linewidth=0.6,
    )
    ax_bound.set_xlim(0.0, max(0.35, float(np.nanmax(boundary_values)) * 1.12))
    ax_bound.set_ylim(len(boundary_values) - 0.5, -0.5)
    ax_bound.set_yticks(np.arange(len(boundary_values)))
    ax_bound.set_yticklabels([])
    ax_bound.set_xlabel("Mean near-boundary share")
    ax_bound.set_title("0.02 / 0.98")
    clean_axes(ax_bound, "x")
    ax_bound.grid(True, axis="x", color="#E6E6E6", linewidth=0.55)

    save(fig, spec, formats=("png",))
    return FIG_DIR / f"{spec.stem}.png"


def _summary_report(
    crosswalk: pd.DataFrame,
    median_matrix: pd.DataFrame,
    sd_matrix: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    tau_summary: pd.DataFrame,
    figure_path: Path,
) -> str:
    class_groups = crosswalk.groupby("relationship_class")
    shared = crosswalk.loc[crosswalk["relationship_class"].eq("shared dominant controls")]
    partial = crosswalk.loc[crosswalk["relationship_class"].eq("partially shared controls")]
    model_sensitive = crosswalk.loc[crosswalk["relationship_class"].eq("model-sensitive controls")]

    shared_core = ["parBETA", "parFC", "parPERC", "parUZL", "parK2", "parTT", "route_a"]
    shared_core_rows = crosswalk.set_index("parameter").loc[shared_core].reset_index()
    shared_core_peak_abs = float(shared_core_rows["peak_signature_abs_median"].median())
    partial_peak_abs = float(partial["peak_signature_abs_median"].median()) if not partial.empty else float("nan")
    lp_peak_abs = float(model_sensitive.loc[model_sensitive["parameter"].eq("parLP"), "peak_signature_abs_median"].iloc[0])
    shared_core_max_abs = float(shared_core_rows["max_abs_sensitivity_median"].median())
    partial_max_abs = float(partial["max_abs_sensitivity_median"].median()) if not partial.empty else float("nan")
    lp_max_abs = float(model_sensitive.loc[model_sensitive["parameter"].eq("parLP"), "max_abs_sensitivity_median"].iloc[0])

    top_rank = crosswalk.sort_values("max_abs_sensitivity_median", ascending=False).head(5)
    low_rank = crosswalk.sort_values("max_abs_sensitivity_median", ascending=True).head(5)
    flagged = boundary_summary.groupby("parameter", as_index=False).agg(
        boundary_caution_flag=("boundary_caution_flag", "first"),
        mean_near_boundary_share_02=("mean_near_boundary_share_02", "first"),
        mean_near_boundary_share_05=("mean_near_boundary_share_05", "first"),
    )
    flagged = flagged.loc[flagged["boundary_caution_flag"]]

    exceed_statement = (
        "Shared-core peak sensitivities exceed LP and the partially shared set on the median summary."
        if shared_core_peak_abs > lp_peak_abs and (np.isnan(partial_peak_abs) or shared_core_peak_abs > partial_peak_abs)
        else "Shared-core peak sensitivities do not uniformly exceed LP and the partially shared set."
    )
    exceed_max_statement = (
        "The same ordering holds when each parameter is ranked by its max-over-signatures sensitivity."
        if shared_core_max_abs > lp_max_abs and (np.isnan(partial_max_abs) or shared_core_max_abs > partial_max_abs)
        else "That ordering weakens when ranking by each parameter's max-over-signatures sensitivity."
    )
    if not flagged.empty:
        flagged_text = ", ".join(
            f"{row.parameter} (share02={row.mean_near_boundary_share_02:.2f}, share05={row.mean_near_boundary_share_05:.2f})"
            for row in flagged.itertuples(index=False)
        )
        boundary_line = f"- Boundary caution parameters: {flagged_text}."
    else:
        boundary_line = "- No parameter crossed the manuscript boundary-caution thresholds."

    lines = [
        "# Cycle B Signature Sensitivity Summary",
        "",
        "## Scope",
        "",
        "- Formulation: distributional under `HybridNseBatchLoss`.",
        f"- Seeds: {', '.join(str(seed) for seed in SEEDS)}.",
        "- Basins: 531 CAMELS-US basins.",
        f"- Heatmap figure: `{figure_path}`.",
        "",
        "## High-level findings",
        "",
        f"- Highest max-over-signatures sensitivities: {', '.join(f'{row.parameter} ({row.max_abs_signature}, {row.max_abs_sensitivity_median:.2f})' for row in top_rank.itertuples(index=False))}.",
        f"- Lowest max-over-signatures sensitivities: {', '.join(f'{row.parameter} ({row.max_abs_signature}, {row.max_abs_sensitivity_median:.2f})' for row in low_rank.itertuples(index=False))}.",
        f"- Shared-core median peak sensitivity = {shared_core_peak_abs:.2f}; partially shared median peak sensitivity = {partial_peak_abs:.2f}; LP peak sensitivity = {lp_peak_abs:.2f}.",
        f"- Shared-core median max-over-signatures sensitivity = {shared_core_max_abs:.2f}; partially shared median max-over-signatures sensitivity = {partial_max_abs:.2f}; LP max-over-signatures sensitivity = {lp_max_abs:.2f}.",
        f"- {exceed_statement}",
        f"- {exceed_max_statement}",
        "",
        "## Reliability flags",
        "",
        boundary_line,
        "- `parTT` is marked partially estimable: its reported gradient includes melt/refreezing pathways but omits the hard rain/snow phase-partition pathway.",
        f"- Mean annual peak soft-max calibration selected taus {tau_summary['selected_tau'].tolist()} across the five seeds.",
        "",
        "## Class cross-reference",
        "",
        f"- Shared dominant controls: {', '.join(shared['parameter'])}.",
        f"- Partially shared controls: {', '.join(partial['parameter'])}.",
        f"- Model-sensitive controls: {', '.join(model_sensitive['parameter'])}.",
        "",
        "## Notable mismatches",
        "",
    ]

    mismatch_rows = crosswalk.sort_values(["relationship_class_rank", "max_abs_sensitivity_median"], ascending=[True, False])
    shared_low = mismatch_rows.loc[mismatch_rows["relationship_class"].eq("shared dominant controls")].sort_values(
        "max_abs_sensitivity_median"
    ).head(2)
    partial_high = mismatch_rows.loc[mismatch_rows["relationship_class"].eq("partially shared controls")].sort_values(
        "max_abs_sensitivity_median", ascending=False
    ).head(2)
    for row in shared_low.itertuples(index=False):
        lines.append(
            f"- Shared-control parameter {row.parameter} is relatively weak in response sensitivity ({row.max_abs_signature}, {row.max_abs_sensitivity_median:.2f})."
        )
    for row in partial_high.itertuples(index=False):
        lines.append(
            f"- Partially shared parameter {row.parameter} is comparatively strong in response sensitivity ({row.max_abs_signature}, {row.max_abs_sensitivity_median:.2f})."
        )

    lines.extend(
        [
            "",
            "## Output files",
            "",
            f"- `data/standardized_sensitivity_median_matrix.csv`",
            f"- `data/standardized_sensitivity_sd_matrix.csv`",
            f"- `data/control_class_sensitivity_crosswalk.csv`",
            f"- `data/seed_basin_signature_jacobians_long.csv`",
            f"- `data/seed_parameter_signature_summary.csv`",
            f"- `data/boundary_reliability_flags.csv`",
            f"- `data/mean_annual_peak_tau_by_seed.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def _focused_diagnostic_report(
    diagnostics: pd.DataFrame,
    verdicts: pd.DataFrame,
) -> str:
    lines = [
        "# PERC/TT Zero-Sensitivity Diagnostic",
        "",
        "## Scope",
        "",
        "- Parameters checked: `parPERC`, `parTT`.",
        "- Diagnostics: raw basin-median gradient, physical-parameter cross-basin standard deviation, and half-elasticity median.",
        "- Goal: distinguish true low response sensitivity from a `sigma_theta` standardization artifact.",
        "",
        "## Seed-level diagnostics",
        "",
    ]
    for parameter in FOCUS_PARAMETERS:
        lines.append(f"### {parameter}")
        sub = diagnostics.loc[diagnostics["parameter"].eq(parameter)].copy()
        for row in sub.itertuples(index=False):
            lines.append(
                f"- Seed {row.seed}, {row.signature}: raw median={row.raw_gradient_median:.6e}, "
                f"half-elasticity median={row.half_elasticity_median:.6e}, sigma_theta={row.sigma_theta:.6e} "
                f"(rank={row.sigma_theta_rank_desc}, ratio_to_seed_median={row.sigma_theta_over_seed_median:.2f})."
            )
        verdict_row = verdicts.loc[verdicts["parameter"].eq(parameter)].iloc[0]
        lines.append(f"- Verdict: `{verdict_row['verdict']}`.")
        lines.append(f"- Reason: {verdict_row['reason']}.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run() -> None:
    _ensure_dirs()
    _write_methods()
    loaded = _loaded_data()

    print("Computing seed-level signature sensitivities...")
    seed_results = [_seed_working_point(seed) for seed in SEEDS]

    print("Writing basin-level and seed-level tables...")
    basin_long = pd.concat([_seed_basin_long(result, loaded.basin_ids) for result in seed_results], ignore_index=True)
    seed_summary = pd.concat([_seed_parameter_summary(result) for result in seed_results], ignore_index=True)
    boundary_summary = _boundary_summary(seed_results)
    tau_summary = _seed_tau_summary(seed_results)
    aggregated, median_matrix, sd_matrix = _aggregate_cross_seed(seed_summary)
    _matrix_sanity_assertions(median_matrix, sd_matrix)
    sigma_theta_summary = _sigma_theta_summary(seed_results)
    focused_diagnostics, focused_verdicts = _focused_zero_sensitivity_diagnostics(basin_long, sigma_theta_summary)
    crosswalk = _control_class_crosswalk(aggregated, boundary_summary)

    basin_long.to_csv(DATA_DIR / "seed_basin_signature_jacobians_long.csv", index=False)
    seed_summary.to_csv(DATA_DIR / "seed_parameter_signature_summary.csv", index=False)
    boundary_summary.to_csv(DATA_DIR / "boundary_reliability_flags.csv", index=False)
    sigma_theta_summary.to_csv(DATA_DIR / "sigma_theta_by_seed.csv", index=False)
    focused_diagnostics.to_csv(DATA_DIR / "perc_tt_zero_sensitivity_diagnostics.csv", index=False)
    focused_verdicts.to_csv(DATA_DIR / "perc_tt_zero_sensitivity_verdicts.csv", index=False)
    tau_summary.to_csv(DATA_DIR / "mean_annual_peak_tau_by_seed.csv", index=False)
    aggregated.to_csv(DATA_DIR / "standardized_sensitivity_aggregated_long.csv", index=False)
    median_matrix.reset_index().rename(columns={"index": "parameter"}).to_csv(
        DATA_DIR / "standardized_sensitivity_median_matrix.csv", index=False
    )
    sd_matrix.reset_index().rename(columns={"index": "parameter"}).to_csv(
        DATA_DIR / "standardized_sensitivity_sd_matrix.csv", index=False
    )
    crosswalk.to_csv(DATA_DIR / "control_class_sensitivity_crosswalk.csv", index=False)

    print("Rendering manuscript-style heatmap...")
    figure_path = _plot_heatmap(median_matrix, sd_matrix, boundary_summary)

    print("Writing summary report...")
    summary = _summary_report(crosswalk, median_matrix, sd_matrix, boundary_summary, tau_summary, figure_path)
    (REPORT_DIR / "signature_sensitivity_summary.md").write_text(summary, encoding="utf-8")
    focused_report = _focused_diagnostic_report(focused_diagnostics, focused_verdicts)
    (REPORT_DIR / "perc_tt_zero_sensitivity_diagnostic.md").write_text(focused_report, encoding="utf-8")

    log_lines = [
        "# Signature Sensitivity Log",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Seeds: {', '.join(str(seed) for seed in SEEDS)}",
        f"- Basin count: {len(loaded.basin_ids)}",
        f"- Outputs written under: `{ANALYSIS_ROOT}`",
        "- Sanity assertions passed: matrix shape 14x4, no NaN in standardized median matrix, non-negative across-seed SD.",
    ]
    (LOG_DIR / "signature_sensitivity_log.md").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    top = crosswalk.sort_values("max_abs_sensitivity_median", ascending=False).head(5)
    print("Top response-sensitive parameters:")
    for row in top.itertuples(index=False):
        print(
            f"  {row.parameter}: {row.max_abs_signature} median={row.max_abs_sensitivity_median:.3f} "
            f"class={row.relationship_class}"
        )


if __name__ == "__main__":
    run()
