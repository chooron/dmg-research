"""Fig. S13 - Attribute collinearity heatmap for selected basin descriptors."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from common_appendix import (  # noqa: E402
    APP_FIG_DIR,
    MM,
    add_panel_label,
    save_fig,
    setup_style,
)

OUT_STEM = "figS13_attribute_collinearity_heatmap"

MANUSCRIPT_ROOT = Path(__file__).resolve().parents[1]
EXTENDS_DIR = MANUSCRIPT_ROOT / "extends"
MATRIX_FILE = EXTENDS_DIR / "attribute_collinearity_matrix.csv"
PAIRS_FILE = EXTENDS_DIR / "attribute_collinearity_pairs.csv"

ATTRIBUTE_LABELS = {
    "slope_mean": "Mean slope",
    "elev_mean": "Elevation",
    "frac_snow": "Snow fraction",
    "aridity": "Aridity",
    "pet_mean": "PET",
    "p_mean": "Precip.",
    "p_seasonality": "Precip. seasonality",
    "soil_conductivity": "Soil cond.",
    "soil_depth": "Soil depth",
    "forest_frac": "Forest",
    "lai_diff": "LAI diff.",
    "high_prec_freq": "High-precip. freq.",
    "high_prec_dur": "High-precip. dur.",
    "low_prec_freq": "Low-precip. freq.",
}


def label_attribute(attribute: str) -> str:
    """Return the manuscript label for an attribute key."""
    return ATTRIBUTE_LABELS.get(str(attribute), str(attribute).replace("_", " "))


def load_collinearity_matrix() -> pd.DataFrame:
    """Load and align the square Spearman correlation matrix."""
    matrix = pd.read_csv(MATRIX_FILE, index_col=0)
    matrix = matrix.loc[matrix.columns, matrix.columns]
    return matrix.astype(float)


def load_high_collinearity_pairs() -> pd.DataFrame:
    """Load high-correlation pairs for the compact note below the heatmap."""
    pairs = pd.read_csv(PAIRS_FILE)
    return pairs.loc[pairs["abs_ge_0_8"].astype(bool)].copy()


def format_pair_note(pairs: pd.DataFrame) -> str:
    """Format the strongest pair list without crowding the heatmap cells."""
    if pairs.empty:
        return "No descriptor pairs exceeded |rho| >= 0.8."
    fragments = []
    for _, row in pairs.sort_values("abs_rho", ascending=False).iterrows():
        left = label_attribute(row["attribute_a"])
        right = label_attribute(row["attribute_b"])
        fragments.append(f"{left} vs {right}: rho = {row['spearman_rho']:.2f}")
    return "Strong pairs (|rho| >= 0.8): " + "; ".join(fragments)


def draw_collinearity_heatmap(ax: plt.Axes, matrix: pd.DataFrame) -> None:
    """Draw the matrix using the appendix diverging palette and font scale."""
    values = matrix.to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list(
        "s13_purple_white_green",
        ["#6A3D9A", "#F7F7F7", "#1B9E77"],
        N=256,
    )
    im = ax.imshow(
        values,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
        aspect="equal",
    )

    tick_positions = np.arange(matrix.shape[0])
    tick_labels = [label_attribute(attr) for attr in matrix.columns]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=7.2)
    ax.set_yticklabels(tick_labels, fontsize=7.2)
    ax.tick_params(length=0, pad=2)

    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            rho = values[row_idx, col_idx]
            if row_idx == col_idx or abs(rho) < 0.6:
                continue
            text_color = "white" if abs(rho) >= 0.78 else "#222222"
            ax.text(
                col_idx,
                row_idx,
                f"{rho:.2f}",
                ha="center",
                va="center",
                fontsize=6.1,
                color=text_color,
            )

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.036, pad=0.025)
    cbar.set_label("Spearman rho", fontsize=8.2)
    cbar.ax.tick_params(labelsize=7.3, length=2.0)


def main() -> None:
    setup_style()
    matrix = load_collinearity_matrix()
    strong_pairs = load_high_collinearity_pairs()

    fig = plt.figure(figsize=(170 * MM, 162 * MM))
    ax = fig.add_axes([0.19, 0.20, 0.69, 0.70])
    draw_collinearity_heatmap(ax, matrix)
    add_panel_label(ax, "S13", x=-0.20, y=1.04, fontsize=11.5)

    fig.text(
        0.52,
        0.055,
        format_pair_note(strong_pairs),
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#555555",
    )

    save_fig(fig, OUT_STEM)
    print(f"Saved {APP_FIG_DIR / OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
