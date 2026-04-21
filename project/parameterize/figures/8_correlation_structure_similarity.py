"""Figure 8: correlation matrix structural similarity."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

from project.parameterize.figures.common import (
    COLORS,
    LOSS_MARKERS,
    add_panel_labels,
    apply_wrr_style,
    correlation_vector_table,
    pretty_loss_name,
    pretty_model_name,
    save_figure,
)


def generate_figure(data_dict: dict, output_dir, formats=("png", "pdf")):
    apply_wrr_style()
    vectors = correlation_vector_table(data_dict["corr_long"]).fillna(0.0)
    cosine_distance = pairwise_distances(vectors.to_numpy(), metric="cosine")
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords = mds.fit_transform(cosine_distance)
    similarity = 1.0 - cosine_distance

    labels = [
        f"{pretty_model_name(model)}\n{pretty_loss_name(loss)}\nseed {seed}"
        for model, loss, seed in vectors.index
    ]
    if len(labels) > 2:
        order = leaves_list(linkage(squareform(cosine_distance, checks=False), method="average"))
    else:
        order = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0))
    for (model, loss, _seed), (x_coord, y_coord) in zip(vectors.index, coords):
        axes[0].scatter(
            x_coord,
            y_coord,
            c=COLORS[model],
            marker=LOSS_MARKERS.get(loss, "o"),
            s=65,
            alpha=0.9,
        )
    axes[0].set_xlabel("MDS-1")
    axes[0].set_ylabel("MDS-2")
    axes[0].set_title("MDS embedding of correlation matrices")
    axes[0].grid(linestyle="--", alpha=0.25)

    ordered_similarity = similarity[np.ix_(order, order)]
    image = axes[1].imshow(ordered_similarity, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    axes[1].set_xticks(np.arange(len(order)))
    axes[1].set_xticklabels([labels[idx] for idx in order], rotation=90)
    axes[1].set_yticks(np.arange(len(order)))
    axes[1].set_yticklabels([labels[idx] for idx in order])
    axes[1].set_title("Pairwise cosine similarity")
    fig.colorbar(image, ax=axes[1], fraction=0.04, pad=0.02, label="Cosine similarity")

    add_panel_labels(axes)
    return save_figure(fig, "fig08_correlation_structure_similarity", output_dir, formats=formats)

