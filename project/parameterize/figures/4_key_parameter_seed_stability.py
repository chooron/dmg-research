"""Figure 4: detailed cross-seed stability for key parameters."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from project.parameterize.figures.common import (
    COLORS,
    add_panel_labels,
    apply_wrr_style,
    pretty_model_name,
    pretty_parameter_name,
    reference_loss_only,
    save_figure,
    seed_error_summary,
)


def generate_figure(data_dict: dict, output_dir, formats=("png", "pdf")):
    apply_wrr_style()
    params = reference_loss_only(data_dict["params_long"], data_dict)
    attrs = data_dict["attributes"][["basin_id", "aridity"]].copy()
    summary = seed_error_summary(params).merge(attrs, on="basin_id", how="left")
    summary = summary.loc[summary["parameter"].isin(data_dict["focus_parameters"])]

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.2), sharex=False)
    offsets = {"deterministic": -0.22, "mc_dropout": 0.0, "distributional": 0.22}

    for ax, parameter in zip(axes.flat, data_dict["focus_parameters"]):
        subset = summary.loc[summary["parameter"] == parameter].sort_values("aridity")
        x_base = np.arange(len(subset["basin_id"].unique()))
        basin_order = subset[["basin_id", "aridity"]].drop_duplicates().sort_values("aridity")
        basin_positions = {basin_id: idx for idx, basin_id in enumerate(basin_order["basin_id"])}

        for model in data_dict["model_order"]:
            model_subset = subset.loc[subset["model"] == model].copy()
            x = model_subset["basin_id"].map(basin_positions).to_numpy(dtype=float) + offsets[model]
            ax.errorbar(
                x,
                model_subset["mean_seed"],
                yerr=model_subset["std_seed"].fillna(0.0),
                fmt="o",
                markersize=2.2,
                linewidth=0.5,
                elinewidth=0.5,
                alpha=0.55,
                color=COLORS[model],
                label=pretty_model_name(model),
            )

        ax.set_title(pretty_parameter_name(parameter))
        ax.set_xlabel("Basins sorted by aridity")
        ax.set_ylabel("Parameter mean ± seed std")
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    axes.flat[0].legend(frameon=False, ncol=3, loc="upper left")
    add_panel_labels(axes)
    return save_figure(fig, "fig04_key_parameter_seed_stability", output_dir, formats=formats)

