"""Figure 11: parameter distributions along climate gradients."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from project.parameterize.figures.common import (
    pretty_parameter_name,
    reference_loss_only,
    save_figure,
    apply_wrr_style,
)


def generate_figure(data_dict: dict, output_dir, formats=("png", "pdf")):
    apply_wrr_style()
    params = reference_loss_only(data_dict["params_long"], data_dict)
    params = params.loc[params["model"] == "distributional"]
    basin_params = params.groupby(["basin_id", "parameter"], as_index=False)["mean"].mean()
    basin_params = basin_params.pivot(index="basin_id", columns="parameter", values="mean").reset_index()
    merged = basin_params.merge(
        data_dict["attributes"][["basin_id", "aridity", "frac_snow", "slope_mean"]],
        on="basin_id",
        how="inner",
    )

    gradients = ["aridity", "frac_snow", "slope_mean"]
    parameters = data_dict["climate_parameters"]
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 10.0), sharex=False)
    for row_idx, gradient in enumerate(gradients):
        merged[f"{gradient}_group"] = pd.qcut(merged[gradient], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
        for col_idx, parameter in enumerate(parameters):
            ax = axes[row_idx, col_idx]
            sns.boxplot(
                data=merged,
                x=f"{gradient}_group",
                y=parameter,
                color="#B8C0CC",
                showfliers=False,
                ax=ax,
            )
            ax.set_title(f"{gradient} vs {pretty_parameter_name(parameter)}")
            ax.set_xlabel("")
            ax.set_ylabel("Parameter mean")
            ax.grid(axis="y", linestyle="--", alpha=0.25)

    return save_figure(fig, "fig11_parameter_climate_gradients", output_dir, formats=formats)

