"""Fig. S10 — Supplementary explainability check: captum IntegratedGradients beeswarm.

Layout: 4 rows × 9 columns.
  Rows (a–d): base mean / mcd mean / dist mean / dist std
  Cols: FOCUS_PARAMS (one parameter per column)
Each cell: beeswarm of per-basin signed IG attributions for that specific
parameter output, coloured by relative feature value.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common_appendix import (
    MM, APP_FIG_DIR, clean_axes, save_fig, setup_style,
)

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[4] / "project" / "parameterize" / "implements"),
)
from mc_mlp import McMlpModel
from param_models import DeterministicParamModel, DistributionalParamModel

OUT_STEM = "figS10_explainability_check"

OUTPUTS_ROOT = (
    Path(__file__).resolve().parents[4] / "project" / "parameterize" / "outputs"
)
ATTRS_CSV = (
    OUTPUTS_ROOT / "analysis" / "stability_stats" / "tables" / "basin_attributes.csv"
)

LOSS = "HybridNseBatchLoss"
SEED = "seed_111"
N_IG_STEPS = 50

# Full parameter order as trained
ALL_PARAMS = [
    "parBETA", "parFC", "parK0", "parK1", "parK2", "parLP",
    "parPERC", "parUZL", "parTT", "parCFMAX", "parCFR", "parCWH",
    "route_a", "route_b",
]

FOCUS_PARAMS = [
    "parFC", "parPERC", "parUZL", "parCFR", "parCWH",
    "parBETA", "parK1", "parK2", "route_b",
]

ROW_TITLES = [
    r"$\delta_{\mathrm{base}}$ — mean",
    r"$\delta_{\mathrm{mcd}}$ — mean",
    r"$\delta_{\mathrm{dist}}$ — mean",
    r"$\delta_{\mathrm{dist}}$ — std",
]

TOP_FEATS = 8  # features shown per cell

PARAM_LABELS = {
    "parBETA": r"$\mathrm{BETA}$",
    "parFC":   r"$\mathrm{FC}$",
    "parLP":   r"$\mathrm{LP}$",
    "parPERC": r"$\mathrm{PERC}$",
    "parUZL":  r"$\mathrm{UZL}$",
    "parK0":   r"$\mathrm{K}_0$",
    "parK1":   r"$\mathrm{K}_1$",
    "parK2":   r"$\mathrm{K}_2$",
    "parTT":   r"$\mathrm{TT}$",
    "parCFMAX":r"$\mathrm{CFMAX}$",
    "parCFR":  r"$\mathrm{CFR}$",
    "parCWH":  r"$\mathrm{CWH}$",
    "route_a": r"$\mathrm{UH}_a$",
    "route_b": r"$\mathrm{UH}_b$",
}

ATTR_SHORT = {
    "frac_snow":            "Snow frac.",
    "elev_mean":            "Elevation",
    "pet_mean":             "PET",
    "aridity":              "Aridity",
    "p_seasonality":        "P season.",
    "p_mean":               "Precip.",
    "slope_mean":           "Slope",
    "soil_conductivity":    "Soil cond.",
    "clay_frac":            "Clay frac.",
    "frac_forest":          "Forest frac.",
    "soil_depth_pelletier": "Soil depth",
    "lai_diff":             "LAI season.",
    "high_prec_dur":        "Hi-prec dur.",
    "high_prec_freq":       "Hi-prec freq.",
    "low_prec_dur":         "Lo-prec dur.",
    "low_prec_freq":        "Lo-prec freq.",
    "area_gages2":          "Drain. area",
    "carbonate_rocks_frac": "Carbonate",
    "sand_frac":            "Sand frac.",
    "silt_frac":            "Silt frac.",
    "geol_permeability":    "Geol. perm.",
    "geol_porosity":        "Geol. por.",
    "soil_porosity":        "Soil por.",
    "max_water_content":    "Max water",
    "lai_max":              "LAI max",
}


def _feat_label(name: str) -> str:
    return ATTR_SHORT.get(name, name.replace("_", " "))


def _param_label(name: str) -> str:
    return PARAM_LABELS.get(name, name)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_norm_and_features(variant: str) -> tuple[list[str], dict]:
    model_dir = OUTPUTS_ROOT / variant / LOSS / SEED / "model"
    norm = json.load(open(model_dir / "normalization_statistics.json"))
    feats = [k for k in norm.keys() if k != "streamflow"]
    return feats, norm


def _load_basin_X(
    feats: list[str], norm: dict
) -> tuple[np.ndarray, np.ndarray]:
    attrs = pd.read_csv(ATTRS_CSV, dtype={"basin_id": str})
    X_raw = attrs[feats].values.astype(np.float32)
    Xn = X_raw.copy()
    for i, f in enumerate(feats):
        _mn, _mx, mu, sd = norm[f]
        Xn[:, i] = (Xn[:, i] - mu) / (sd + 1e-8)
    return X_raw, Xn


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class _MeanWrapper(torch.nn.Module):
    def __init__(self, model: DistributionalParamModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, _, _ = self.model._distribution_stats(x)
        return torch.sigmoid(mu)


class _StdWrapper(torch.nn.Module):
    def __init__(
        self, model: DistributionalParamModel, n_samples: int = 30
    ) -> None:
        super().__init__()
        self.model = model
        self.n_samples = n_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        samples = torch.stack(
            [self.model.sample_parameters(x) for _ in range(self.n_samples)],
            dim=0,
        )
        return samples.std(dim=0)


def _load_deterministic(feats: list[str]) -> torch.nn.Module:
    model_dir = OUTPUTS_ROOT / "deterministic" / LOSS / SEED / "model"
    state = torch.load(
        model_dir / "model_epoch100.pt", map_location="cpu", weights_only=False
    )
    new_state = {k.replace("nn_model.", ""): v for k, v in state.items()}
    ny = new_state["layers.4.weight"].shape[0]
    cfg = {"hidden_size": 128, "output_activation": "sigmoid"}
    model = DeterministicParamModel(cfg, len(feats), ny)
    model.load_state_dict(new_state)
    model.eval()
    return model


def _load_mc_dropout(feats: list[str]) -> torch.nn.Module:
    model_dir = OUTPUTS_ROOT / "mc_dropout" / LOSS / SEED / "model"
    state = torch.load(
        model_dir / "model_epoch100.pt", map_location="cpu", weights_only=False
    )
    new_state = {k.replace("nn_model.", ""): v for k, v in state.items()}
    ny = new_state["layers.6.weight"].shape[0]
    cfg = {"hidden_size": 128, "output_activation": "sigmoid", "dropout": 0.1}
    model = McMlpModel(cfg, len(feats), ny)
    model.load_state_dict(new_state)
    model.eval()
    return model


def _load_distributional(feats: list[str]) -> DistributionalParamModel:
    model_dir = OUTPUTS_ROOT / "distributional" / LOSS / SEED / "model"
    state = torch.load(
        model_dir / "model_epoch100.pt", map_location="cpu", weights_only=False
    )
    new_state = {k.replace("nn_model.", ""): v for k, v in state.items()}
    ny = new_state["latent_mu_head.weight"].shape[0]
    cfg = {"hidden_size": 128, "output_activation": "sigmoid"}
    model = DistributionalParamModel(cfg, len(feats), ny)
    model.load_state_dict(new_state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# IG computation — per parameter output dimension
# ---------------------------------------------------------------------------

def _compute_ig_for_param(
    wrapper: torch.nn.Module,
    X_norm: np.ndarray,
    param_idx: int,
    n_steps: int = N_IG_STEPS,
) -> np.ndarray:
    """Return signed IG attributions for one output dimension.

    Returns array of shape (n_basins, n_features).
    """
    from captum.attr import IntegratedGradients

    xt = torch.tensor(X_norm, dtype=torch.float32)
    baseline = torch.zeros_like(xt)
    ig = IntegratedGradients(wrapper)
    attrs = ig.attribute(xt, baseline, target=param_idx, n_steps=n_steps)
    return attrs.detach().numpy()


# ---------------------------------------------------------------------------
# Beeswarm cell
# ---------------------------------------------------------------------------

def _beeswarm_cell(
    ax: plt.Axes,
    ig_vals: np.ndarray,
    X_raw: np.ndarray,
    feat_names: list[str],
    top_k: int = TOP_FEATS,
    show_yticks: bool = True,
) -> None:
    """Draw one beeswarm cell.

    ig_vals : (n_basins, n_features) signed IG for one parameter
    X_raw   : (n_basins, n_features) raw feature values for colouring
    """
    cmap = plt.cm.viridis

    # Rank by mean |IG|, show top_k
    mean_abs = np.abs(ig_vals).mean(axis=0)
    ranked = np.argsort(mean_abs)[::-1][:top_k]
    ranked = ranked[::-1]  # bottom = most important

    for row_idx, feat_idx in enumerate(ranked):
        vals = ig_vals[:, feat_idx]
        raw = X_raw[:, feat_idx]

        rmin = np.percentile(raw, 2)
        rmax = np.percentile(raw, 98)
        raw_norm = np.clip((raw - rmin) / (rmax - rmin + 1e-8), 0, 1)

        # Bin-based y-jitter
        y_jitter = np.zeros(len(vals))
        bin_edges = np.linspace(vals.min(), vals.max() + 1e-8, 30)
        for b in range(len(bin_edges) - 1):
            mask = (vals >= bin_edges[b]) & (vals < bin_edges[b + 1])
            n = mask.sum()
            if n > 1:
                spread = min(0.38, n * 0.045)
                y_jitter[mask] = np.linspace(-spread, spread, n)

        ax.scatter(
            vals,
            row_idx + y_jitter,
            c=cmap(raw_norm),
            s=4,
            alpha=0.55,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    if show_yticks:
        ax.set_yticks(range(len(ranked)))
        ax.set_yticklabels(
            [_feat_label(feat_names[i]) for i in ranked], fontsize=8.0
        )
    else:
        ax.set_yticks(range(len(ranked)))
        ax.set_yticklabels([])

    ax.axvline(0, color="#AAAAAA", linewidth=0.6, zorder=1)
    ax.tick_params(axis="x", labelsize=8.0, length=2, pad=1)
    clean_axes(ax, grid_axis="x")
    return ranked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_style()

    feats, norm = _load_norm_and_features("distributional")
    X_raw, X_norm = _load_basin_X(feats, norm)

    det_model = _load_deterministic(feats)
    mcd_model = _load_mc_dropout(feats)
    dist_model = _load_distributional(feats)

    wrappers = [
        det_model,
        mcd_model,
        _MeanWrapper(dist_model),
        _StdWrapper(dist_model, n_samples=30),
    ]

    n_rows = len(wrappers)
    n_cols = len(FOCUS_PARAMS)

    fig_w = 320 * MM   # 16:9
    fig_h = 180 * MM
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows, n_cols,
        left=0.10, right=0.98, top=0.93, bottom=0.10,
        hspace=0.35, wspace=0.12,
    )

    panel_idx = 0
    for row, (wrapper, row_title) in enumerate(zip(wrappers, ROW_TITLES)):
        for col, param in enumerate(FOCUS_PARAMS):
            param_idx = ALL_PARAMS.index(param)
            print(f"  IG row={row} col={col} ({param})…")
            ig_vals = _compute_ig_for_param(wrapper, X_norm, param_idx)

            ax = fig.add_subplot(gs[row, col])
            show_y = (col == 0)
            _beeswarm_cell(ax, ig_vals, X_raw, feats, show_yticks=show_y)

            if row == 0:
                ax.set_title(_param_label(param), fontsize=10.5, pad=3)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=9.0, labelpad=4)
            else:
                ax.set_ylabel("")

            panel_idx += 1

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=mcolors.Normalize(0, 1)
    )
    sm.set_array([])
    cbar_ax = fig.add_axes([0.35, 0.025, 0.30, 0.016])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Low", "Mid", "High"], fontsize=8.5)
    cb.set_label("Feature value (relative)", fontsize=9.0, labelpad=1)
    cb.outline.set_linewidth(0.4)

    save_fig(fig, OUT_STEM)
    print(f"Saved {APP_FIG_DIR / OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
