"""Figure 13: uncertainty calibration scatter plots."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

from project.parameterize.figures.common import pretty_parameter_name, reference_loss_only, save_figure, apply_wrr_style


def generate_figure(data_dict: dict, output_dir, formats=("png", "pdf")):
    apply_wrr_style()
    params = reference_loss_only(data_dict["params_long"], data_dict)
    params = params.loc[params["model"] == "distributional"]
    metric_frame = reference_loss_only(data_dict["metrics_long"], data_dict)
    metric_frame = metric_frame.loc[metric_frame["model"] == "distributional"]

    basin_std = params.groupby(["basin_id", "parameter"], as_index=False)["std"].mean()
    basin_std = basin_std.pivot(index="basin_id", columns="parameter", values="std").reset_index()
    basin_nse = metric_frame.groupby("basin_id", as_index=False)["nse"].mean()
    basin_attrs = data_dict["attributes"][["basin_id", "aridity", "frac_snow"]]
    merged = basin_std.merge(basin_nse, on="basin_id", how="inner").merge(basin_attrs, on="basin_id", how="inner")

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), sharex=False, sharey=False)
    focus_params = data_dict["climate_parameters"]
    color_values = merged["frac_snow"].to_numpy()

    for col_idx, parameter in enumerate(focus_params):
        x_nse = merged[parameter].to_numpy()
        y_nse = merged["nse"].to_numpy()
        ax = axes[0, col_idx]
        scatter = ax.scatter(x_nse, y_nse, c=color_values, cmap="viridis", alpha=0.7, s=18)
        smoothed = lowess(y_nse, x_nse, frac=0.35, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="black", linewidth=1.3)
        rho, _ = spearmanr(x_nse, y_nse, nan_policy="omit")
        ax.set_title(f"{pretty_parameter_name(parameter)} std vs NSE")
        ax.set_xlabel("Parameter std")
        ax.set_ylabel("Test NSE")
        ax.text(0.03, 0.95, f"ρ={rho:.2f}", transform=ax.transAxes, va="top")

        x_attr = merged[parameter].to_numpy()
        y_attr = merged["aridity"].to_numpy()
        ax = axes[1, col_idx]
        ax.scatter(x_attr, y_attr, c=color_values, cmap="viridis", alpha=0.7, s=18)
        smoothed = lowess(y_attr, x_attr, frac=0.35, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="black", linewidth=1.3)
        rho, _ = spearmanr(x_attr, y_attr, nan_policy="omit")
        ax.set_title(f"{pretty_parameter_name(parameter)} std vs aridity")
        ax.set_xlabel("Parameter std")
        ax.set_ylabel("Aridity")
        ax.text(0.03, 0.95, f"ρ={rho:.2f}", transform=ax.transAxes, va="top")

    fig.colorbar(scatter, ax=axes, fraction=0.025, pad=0.02, label="frac_snow")
    return save_figure(fig, "fig13_uncertainty_calibration_scatter", output_dir, formats=formats)

