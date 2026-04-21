"""Figure 14: conceptual positioning of the three parameterization methods."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from project.parameterize.figures.common import COLORS, CONCEPT_POINTS, apply_wrr_style, pretty_model_name, save_figure


def generate_figure(data_dict: dict, output_dir, formats=("png", "pdf")):
    apply_wrr_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for model in data_dict["model_order"]:
        spec = CONCEPT_POINTS[model]
        x_coord, y_coord = spec["xy"]
        ellipse = Ellipse(
            xy=spec["xy"],
            width=spec["width"],
            height=spec["height"],
            facecolor=COLORS[model],
            alpha=0.18,
            edgecolor=COLORS[model],
            linewidth=1.2,
        )
        ax.add_patch(ellipse)
        ax.scatter([x_coord], [y_coord], s=90, color=COLORS[model], zorder=3)
        ax.text(x_coord + 0.02, y_coord + 0.015, pretty_model_name(model), fontsize=9)

    ax.annotate(
        "Ideal for parameter interpretation",
        xy=(0.90, 0.92),
        xytext=(0.48, 0.60),
        arrowprops={"arrowstyle": "->", "lw": 1.2},
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predictive emphasis")
    ax.set_ylabel("Relationship reliability")
    ax.set_title("Conceptual positioning of parameterization strategies")
    ax.grid(linestyle="--", alpha=0.25)
    return save_figure(fig, "fig14_synthesis_positioning", output_dir, formats=formats)

