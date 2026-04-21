"""Figure 9: learned parameters versus prior ranges."""

from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from project.parameterize.figures.common import (
    COLORS,
    add_panel_labels,
    apply_wrr_style,
    pretty_model_name,
    pretty_parameter_name,
    reference_loss_only,
    save_figure,
)


def generate_figure(data_dict: dict, output_dir, formats=("png", "pdf")):
    apply_wrr_style()
    params = reference_loss_only(data_dict["params_long"], data_dict)
    parameters = data_dict["param_names"]
    ncols = min(5, max(2, math.ceil(math.sqrt(len(parameters)))))
    nrows = math.ceil(len(parameters) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.7 * nrows), squeeze=False)

    for ax, parameter in zip(axes.flat, parameters):
        subset = params.loc[params["parameter"] == parameter]
        sns.violinplot(
            data=subset,
            x="model",
            y="mean",
            order=data_dict["model_order"],
            palette=COLORS,
            inner="quartile",
            cut=0,
            linewidth=0.8,
            ax=ax,
        )
        label = pretty_parameter_name(parameter)
        prior = data_dict["hbv_priors"].get(label)
        if prior is not None:
            ax.axhspan(prior[0], prior[1], color="grey", alpha=0.15, zorder=0)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("Parameter mean")
        ax.set_xticklabels([pretty_model_name(model) for model in data_dict["model_order"]], rotation=18)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    for ax in axes.flat[len(parameters) :]:
        ax.axis("off")

    add_panel_labels([ax for ax in axes.flat[: len(parameters)]])
    return save_figure(fig, "fig09_parameter_prior_validity", output_dir, formats=formats)

