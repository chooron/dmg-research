"""Fig. S15 - Absolute uncertainty coupling and boundary sensitivity."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common_appendix import (  # noqa: E402
    APP_FIG_DIR,
    MM,
    add_panel_label,
    clean_axes,
    save_fig,
    setup_style,
)

OUT_STEM = "figS15_uncertainty_coupling_boundary_abs"
DATA_STEM = "figS15_data_used"
CAPTION_NOTE = "figS15_caption_note.md"

MANUSCRIPT_ROOT = Path(__file__).resolve().parents[1]
EXTENDS_DIR = MANUSCRIPT_ROOT / "extends"
DIAGNOSTIC_FILE = EXTENDS_DIR / "uncertainty_diagnostic_classification.csv"

DIAGNOSTIC_ORDER = [
    "less-confounded",
    "mean-coupled",
    "boundary-sensitive",
    "mean-coupled and boundary-sensitive",
]

DIAGNOSTIC_COLORS = {
    "less-confounded": "#8AA6B8",
    "mean-coupled": "#E69F00",
    "boundary-sensitive": "#6A3D9A",
    "mean-coupled and boundary-sensitive": "#009E73",
}

PARAMETER_LABELS = {
    "parBETA": "BETA",
    "parFC": "FC",
    "parLP": "LP",
    "parPERC": "PERC",
    "parUZL": "UZL",
    "parK0": "K0",
    "parK1": "K1",
    "parK2": "K2",
    "parTT": "TT",
    "parCFMAX": "CFMAX",
    "parCFR": "CFR",
    "parCWH": "CWH",
    "route_a": r"$\mathrm{UH}_a$",
    "route_b": r"$\mathrm{UH}_b$",
}

TEXT_POSITIONS = {
    "parBETA": (0.43, 0.84),
    "parCFMAX": (0.11, 0.11),
    "parCFR": (0.36, 0.53),
    "parCWH": (0.92, 0.98),
    "parFC": (0.58, 0.76),
    "parK0": (0.32, 0.93),
    "parK1": (0.84, 0.16),
    "parK2": (0.78, 0.84),
    "parLP": (0.19, 0.73),
    "parPERC": (0.75, 0.94),
    "parTT": (0.23, 0.23),
    "route_a": (0.36, 0.67),
    "route_b": (0.90, 0.91),
    "parUZL": (0.90, 0.76),
}


def label_parameter(parameter: str) -> str:
    """Return compact parameter label for point annotations."""
    return PARAMETER_LABELS.get(str(parameter), str(parameter).replace("par", ""))


def point_size(near_boundary_share: float) -> float:
    """Map near-boundary share to marker area."""
    return 34.0 + 130.0 * float(near_boundary_share)


def load_uncertainty_diagnostics() -> pd.DataFrame:
    """Load diagnostics and add absolute plotting coordinates."""
    data = pd.read_csv(DIAGNOSTIC_FILE)
    data["parameter_label_plot"] = data["parameter"].map(label_parameter)
    data["abs_mean_std_spearman"] = data["mean_std_spearman"].abs()
    data["abs_boundary_distance_std_spearman"] = data["boundary_distance_std_spearman"].abs()
    data["diagnostic_class"] = pd.Categorical(
        data["diagnostic_class"],
        categories=DIAGNOSTIC_ORDER,
        ordered=True,
    )
    data = data.sort_values(["diagnostic_class", "parameter"]).reset_index(drop=True)
    data.to_csv(APP_FIG_DIR / f"{DATA_STEM}.csv", index=False)
    return data


def draw_quadrant_labels(ax: plt.Axes) -> None:
    """Add low-contrast quadrant labels."""
    quadrant_style = {
        "fontsize": 6.6,
        "color": "#9A9A9A",
        "ha": "center",
        "va": "center",
        "zorder": 0,
    }
    ax.text(0.25, 0.25, "less-confounded", **quadrant_style)
    ax.text(0.75, 0.25, "mean-coupled", **quadrant_style)
    ax.text(0.25, 0.75, "boundary-sensitive", **quadrant_style)
    ax.text(0.75, 0.75, "mean-coupled +\nboundary-sensitive", **quadrant_style)


def draw_reference_lines(ax: plt.Axes) -> None:
    """Add threshold guides for the diagnostic regions."""
    ax.axvline(0.5, color="#BDBDBD", linewidth=0.85, linestyle=(0, (3, 3)), zorder=1)
    ax.axhline(0.5, color="#BDBDBD", linewidth=0.85, linestyle=(0, (3, 3)), zorder=1)
    ax.text(0.505, 0.02, "|rho| = 0.5", fontsize=6.4, color="#777777", ha="left", va="bottom")
    ax.text(0.02, 0.505, "|rho| = 0.5", fontsize=6.4, color="#777777", ha="left", va="bottom")


def draw_uncertainty_scatter(ax: plt.Axes, data: pd.DataFrame) -> None:
    """Draw absolute-correlation diagnostic scatter."""
    draw_quadrant_labels(ax)
    draw_reference_lines(ax)

    for diagnostic_class in DIAGNOSTIC_ORDER:
        panel = data.loc[data["diagnostic_class"].astype(str) == diagnostic_class]
        if panel.empty:
            continue
        ax.scatter(
            panel["abs_mean_std_spearman"],
            panel["abs_boundary_distance_std_spearman"],
            s=[point_size(value) for value in panel["near_boundary_share"]],
            color=DIAGNOSTIC_COLORS[diagnostic_class],
            edgecolor="white",
            linewidth=0.75,
            alpha=0.93,
            zorder=3,
        )

    for _, row in data.iterrows():
        xy = (row["abs_mean_std_spearman"], row["abs_boundary_distance_std_spearman"])
        text_xy = TEXT_POSITIONS.get(row["parameter"], xy)
        ax.annotate(
            label_parameter(row["parameter"]),
            xy=xy,
            xytext=text_xy,
            textcoords="data",
            fontsize=6.0,
            color="#222222",
            ha="left" if text_xy[0] >= xy[0] else "right",
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": "#777777",
                "linewidth": 0.45,
                "shrinkA": 2,
                "shrinkB": 4,
            },
            clip_on=False,
            zorder=4,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$|\rho(\mathrm{mean},\ \mathrm{distributional\ std.})|$", fontsize=8.6)
    ax.set_ylabel(r"$|\rho(\mathrm{boundary\ distance},\ \mathrm{distributional\ std.})|$", fontsize=8.6)
    ax.tick_params(labelsize=7.5, length=2.5, pad=2)
    clean_axes(ax, grid_axis="both")


def add_legends(fig: plt.Figure) -> None:
    """Add diagnostic class and marker-size legends."""
    class_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor=DIAGNOSTIC_COLORS[diagnostic_class],
            markeredgecolor="white",
            markersize=6.2,
            label=diagnostic_class,
        )
        for diagnostic_class in DIAGNOSTIC_ORDER
    ]
    size_values = [0.15, 0.40, 0.65]
    size_handles = [
        plt.scatter(
            [],
            [],
            s=point_size(value),
            color="#B8B8B8",
            edgecolor="white",
            linewidth=0.75,
            label=f"{value:.2f}",
        )
        for value in size_values
    ]

    leg1 = fig.legend(
        handles=class_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=7.0,
        bbox_to_anchor=(0.50, 0.075),
        handletextpad=0.5,
        columnspacing=1.4,
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=size_handles,
        title="Near-boundary fraction",
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=6.8,
        title_fontsize=7.0,
        bbox_to_anchor=(0.50, 0.012),
        handletextpad=0.4,
        columnspacing=1.2,
    )


def write_caption_note() -> None:
    """Write the requested caption-ready note."""
    note = (
        "Figure S15 summarizes diagnostic checks for distributional parameter standard "
        "deviations. Lower values of |rho(mean, std)| and |rho(boundary distance, std)| "
        "indicate uncertainty gradients that are less affected by mean-std coupling or "
        "boundary proximity. CFMAX and TT appear in the lower-left region, whereas CWH, "
        "PERC, UZL, and UH_b are closer to the coupled or boundary-sensitive regions. "
        "The plotted coordinates are absolute Spearman rho magnitudes; signed values "
        "are retained in figS15_data_used.csv.\n"
    )
    (APP_FIG_DIR / CAPTION_NOTE).write_text(note, encoding="utf-8")


def main() -> None:
    setup_style()
    data = load_uncertainty_diagnostics()
    write_caption_note()

    fig = plt.figure(figsize=(126 * MM, 126 * MM))
    ax = fig.add_axes([0.15, 0.24, 0.78, 0.68])
    draw_uncertainty_scatter(ax, data)
    add_legends(fig)

    save_fig(fig, OUT_STEM)
    print(f"Saved {APP_FIG_DIR / OUT_STEM}.png / .pdf")
    print(f"Saved {APP_FIG_DIR / DATA_STEM}.csv")
    print(f"Saved {APP_FIG_DIR / CAPTION_NOTE}")


if __name__ == "__main__":
    main()
