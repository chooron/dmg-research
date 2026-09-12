#!/usr/bin/env python3
"""Coordinator: figure-ready CSVs and scientific QC plots."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import CAMELS_35_ATTRIBUTES, RESULTS, write_csv  # noqa: E402

FIG = RESULTS / "figures"


def save(name, frame, plotter):
    FIG.mkdir(parents=True, exist_ok=True)
    write_csv(FIG / f"{name}_data.csv", frame)
    fig, ax = plt.subplots(figsize=(8, 5))
    plotter(ax, frame)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    a01 = pd.read_csv(RESULTS / "agent_A/A01_MODEL_GAP_HETEROGENEITY.csv")
    a02 = pd.read_csv(RESULTS / "agent_A/A02_BASIN_GAP_LONG.csv")
    a03 = pd.read_csv(RESULTS / "agent_A/A03_BASIN_SHARED_MAPPING_SUSCEPTIBILITY.csv")
    a04 = pd.read_csv(RESULTS / "agent_A/A04_PREDICTIVELY_ADMISSIBLE_BASIN_LEVEL.csv")
    b01 = pd.read_csv(RESULTS / "agent_B/B01_MODEL_ATTRIBUTE_GAP_ASSOCIATION.csv")
    c01 = pd.read_csv(RESULTS / "agent_C/C01_PARAMETER_REALIZATION_DISTANCE.csv")
    d00 = pd.read_csv(RESULTS / "agent_D/D00_MODEL_BRIDGE_TABLE.csv")
    c07 = pd.read_csv(RESULTS / "agent_C/C07_MODEL_RESTART_SUMMARY.csv")

    f = a01.sort_values("G_seen_median")
    save("F01_model_gap_distribution", f, lambda ax, d: (ax.errorbar(np.arange(len(d)), d.G_seen_median, yerr=[d.G_seen_median-d.G_seen_bootstrap_ci95_low, d.G_seen_bootstrap_ci95_high-d.G_seen_median], fmt="o", ms=3), ax.axhline(0, color="black", lw=.8), ax.set_xticks(np.arange(len(d))), ax.set_xticklabels(d.model, rotation=90, fontsize=6), ax.set_ylabel("G_seen = IC KGE - dPL KGE"), ax.set_title("Model-level seen-basin gap with bootstrap CI")))
    f = a03[["basin_id", "median_G_seen", "G_seen_spread_max_minus_min", "fraction_models_G_positive"]].copy()
    save("F02_basin_gap_heterogeneity", f, lambda ax, d: (ax.scatter(d.median_G_seen, d.G_seen_spread_max_minus_min, c=d.fraction_models_G_positive, s=9, cmap="coolwarm", alpha=.7), ax.axvline(0, color="black", lw=.7), ax.set_xlabel("Basin median G_seen"), ax.set_ylabel("Across-model G_seen spread"), ax.set_title("Basin-level heterogeneity")))
    f = a04[a04.tau == .05][["basin_id", "admissible_model_count", "admissible_G_seen_spread", "best_worst_G_seen_difference", "admissible_lowest_G_seen", "admissible_highest_G_seen"]]
    save("F03_predictively_admissible_gap", f, lambda ax, d: (ax.scatter(d.admissible_model_count, d.best_worst_G_seen_difference, c=d.admissible_G_seen_spread, s=10, cmap="viridis"), ax.set_xlabel("Admissible model count (tau=0.05)"), ax.set_ylabel("Best-worst G_seen difference"), ax.set_title("Predictively admissible gap separation")))
    pivot = b01.pivot(index="model", columns="attribute", values="rho"); top = pivot.abs().median().sort_values(ascending=False).head(20).index; f = pivot[top]
    write_csv(FIG / "F04_gap_attribute_heatmap_data.csv", f.reset_index())
    fig, ax = plt.subplots(figsize=(11, 7)); im = ax.imshow(f.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1); ax.set_yticks(np.arange(len(f))); ax.set_yticklabels(f.index, fontsize=6); ax.set_xticks(np.arange(len(f.columns))); ax.set_xticklabels(f.columns, rotation=75, ha="right", fontsize=7); ax.set_title("G_seen–attribute Spearman rho (20 largest median |rho|)"); fig.colorbar(im, ax=ax, label="rho"); fig.tight_layout(); fig.savefig(FIG / "F04_gap_attribute_heatmap.png", dpi=160); plt.close(fig)
    f = a02.merge(c01[["model", "basin_id", "D_theta_rms"]], on=["model", "basin_id"], validate="one_to_one")[["model", "basin_id", "G_seen", "D_theta_rms"]]
    save("F05_gap_vs_parameter_distance", f, lambda ax, d: (ax.scatter(d.G_seen, d.D_theta_rms, s=2, alpha=.12), ax.axvline(0, color="black", lw=.7), ax.set_xlabel("G_seen"), ax.set_ylabel("D_theta RMS"), ax.set_title("Gap versus parameter realization distance")))
    f = d00[["model", "G_seen_median", "reproducibility_median", "dominant_control_agreement_model"]]
    save("F06_reproducibility_vs_gap", f, lambda ax, d: (ax.scatter(d.G_seen_median, d.reproducibility_median, s=24), ax.axvline(0, color="black", lw=.7), ax.set_xlabel("Model median G_seen"), ax.set_ylabel("Median relationship reproducibility"), ax.set_title("Reproducibility versus model gap")))
    f = d00[["model", "D_theta_median", "reproducibility_median"]]
    save("F07_reproducibility_vs_parameter_distance", f, lambda ax, d: (ax.scatter(d.D_theta_median, d.reproducibility_median, s=24), ax.set_xlabel("Model median D_theta"), ax.set_ylabel("Median relationship reproducibility"), ax.set_title("Reproducibility versus parameter distance")))
    f = c07.merge(d00[["model", "D_theta_median", "reproducibility_median"]], on="model")
    save("F08_restart_identifiability_links", f, lambda ax, d: (ax.scatter(d.restart_mean_u_sd, d.D_theta_median, c=d.reproducibility_median, s=26, cmap="viridis"), ax.set_xlabel("Median restart mean u SD"), ax.set_ylabel("Median D_theta"), ax.set_title("Restart uncertainty, distance, and reproducibility")))
    print("Figures complete", len(list(FIG.glob("F*.png"))))


if __name__ == "__main__":
    main()
