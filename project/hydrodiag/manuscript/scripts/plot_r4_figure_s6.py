"""Generate Figure S6: Multi-seed replicability of soil-water state consistency (R4 SI).

Demonstrates that the two primary Figure 7 signatures:
  (a) Snow-burden decile dependence (upper SWE tail separation)
  (b) Hydroclimatic process-phase fingerprint (Active Melt concentration)
replicate across distinct random seeds of dPL (canonical seed 42 and replicate seed 123),
as well as the multi-start fused IC optimization regime.

Outputs:
  manuscript/supplement/figures/Fig_S6_r4_multiseed_replication.png (300 DPI, PNG only)
  manuscript/figures/Fig_S6_r4_multiseed_replication.png (300 DPI, PNG only)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript.scripts.r1_plot_style import (  # noqa: E402
    apply_clean_spines,
    setup_publication_style,
)
from r4.common import default_results_root  # noqa: E402
from r4.robustness_analysis import bootstrap_median_ci  # noqa: E402

FIGURES_DIR = HERE.parents[0] / "supplement" / "figures"
ALT_FIGURES_DIR = HERE.parents[0] / "figures"

REGIME_CFG = {
    "dPL_seed42": {
        "label": "dPL (seed 42; canonical)",
        "color": "#882E72",
        "marker": "o",
        "ls": "-",
        "lw": 1.4,
    },
    "dPL_seed123": {
        "label": "dPL (seed 123; replicate)",
        "color": "#117733",
        "marker": "s",
        "ls": "--",
        "lw": 1.3,
    },
    "IC_fused": {
        "label": "IC (fused; sensitivity)",
        "color": "#333333",
        "marker": "D",
        "ls": "-.",
        "lw": 1.3,
    },
}
REGIMES = ["dPL_seed42", "dPL_seed123", "IC_fused"]

PHASE_ORDER = [
    "Phase_1_Snow_Accumulation",
    "Phase_2_Active_Melt_Recharge",
    "Phase_3_Post_Melt_Transition",
    "Phase_4_Summer_Dry_Down",
]
PHASE_LABELS = [
    "Accumulation\n(Oct–Feb)",
    "Active melt\n(Mar–May)",
    "Post-melt\n(Jun–Jul)",
    "Dry-down\n(Aug–Sep)",
]


def generate_figure_s6(results_root: Path, out_dir: Path) -> Path:
    setup_publication_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    ALT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    r4_dir = results_root / "r4_phase1_soil_official"
    df_swe = pd.read_csv(r4_dir / "robustness_swe_decile_shape.csv")
    df_phase = pd.read_csv(r4_dir / "robustness_process_phase_consistency.csv")

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.4),
        gridspec_kw={
            "wspace": 0.28,
            "left": 0.08,
            "right": 0.96,
            "top": 0.90,
            "bottom": 0.16,
        },
    )

    # -----------------------------------------------------------------------
    # (a) Snow-burden decile dependence replication
    # -----------------------------------------------------------------------
    apply_clean_spines(ax_a)
    ax_a.axhline(0, color="#999999", ls="--", lw=0.8, zorder=1)
    ax_a.axvspan(7.5, 9.5, color="#EAF1F8", alpha=0.55, zorder=0)
    ax_a.text(
        8.5,
        0.22,
        "Upper SWE tail",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#4A6FA5",
        zorder=5,
    )

    x_dec = np.arange(10)
    for reg in REGIMES:
        cfg = REGIME_CFG[reg]
        sub = df_swe[df_swe["regime"] == reg].sort_values("decile")
        ym = sub["delta_anomaly_corr_median"].to_numpy()
        ylo = sub["delta_anomaly_corr_ci_lower"].to_numpy()
        yhi = sub["delta_anomaly_corr_ci_upper"].to_numpy()
        ax_a.errorbar(
            x_dec,
            ym,
            yerr=[ym - ylo, yhi - ym],
            fmt="none",
            ecolor=cfg["color"],
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.8,
            alpha=0.85,
            zorder=3,
        )
        ax_a.plot(
            x_dec,
            ym,
            color=cfg["color"],
            marker=cfg["marker"],
            markersize=4.5,
            lw=cfg["lw"],
            ls=cfg["ls"],
            label=cfg["label"],
            zorder=4,
        )

    ax_a.set_xticks(x_dec)
    ax_a.set_xticklabels([f"D{i + 1:02d}" for i in range(10)], fontsize=7.2)
    ax_a.set_xlabel("Snow-17 SWE burden decile", fontsize=7.6)
    ax_a.set_ylabel("$\Delta$ anomaly correlation (CN − Base)", fontsize=7.6)
    ax_a.set_title(
        "(a) Snow-burden dependence replication",
        loc="left",
        fontweight="bold",
        fontsize=8.2,
    )
    ax_a.set_ylim(-0.04, 0.25)
    ax_a.legend(
        loc="upper left",
        fontsize=5.8,
        frameon=True,
        facecolor="#FFFFFF",
        framealpha=0.92,
    )

    # -----------------------------------------------------------------------
    # (b) Process phase fingerprints replication
    # -----------------------------------------------------------------------
    apply_clean_spines(ax_b)
    ax_b.axhline(0, color="#999999", ls="--", lw=0.8, zorder=1)
    ax_b.axvspan(0.5, 1.5, color="#EAF1F8", alpha=0.55, zorder=0)

    x_phase = np.arange(4)
    offsets = [-0.15, 0.0, 0.15]
    for idx, reg in enumerate(REGIMES):
        cfg = REGIME_CFG[reg]
        df_r = df_phase[df_phase["regime"] == reg]
        ym, ylo, yhi = [], [], []
        for p in PHASE_ORDER:
            vals = df_r.loc[df_r["phase_name"] == p, "delta_anomaly_corr"].to_numpy(
                dtype=float
            )
            m, l, h = bootstrap_median_ci(vals)
            ym.append(m)
            ylo.append(l)
            yhi.append(h)
        ym, ylo, yhi = np.array(ym), np.array(ylo), np.array(yhi)
        xp = x_phase + offsets[idx]
        ax_b.errorbar(
            xp,
            ym,
            yerr=[ym - ylo, yhi - ym],
            fmt="none",
            ecolor=cfg["color"],
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.8,
            alpha=0.85,
            zorder=3,
        )
        ax_b.plot(
            xp,
            ym,
            color=cfg["color"],
            marker=cfg["marker"],
            markersize=4.5,
            ls="none",
            label=cfg["label"],
            zorder=4,
        )

    ax_b.set_xticks(x_phase)
    ax_b.set_xticklabels(PHASE_LABELS, fontsize=6.8)
    ax_b.set_xlabel("Hydroclimatic process phase", fontsize=7.6)
    ax_b.set_ylabel("$\Delta$ anomaly correlation (CN − Base)", fontsize=7.6)
    ax_b.set_title(
        "(b) Phase localization replication",
        loc="left",
        fontweight="bold",
        fontsize=8.2,
    )
    ax_b.set_ylim(-0.06, 0.35)
    ax_b.legend(
        loc="upper left",
        fontsize=5.8,
        frameon=True,
        facecolor="#FFFFFF",
        framealpha=0.92,
    )

    out_png = out_dir / "Fig_S6_r4_multiseed_replication.png"
    fig.savefig(out_png, dpi=300)
    fig.savefig(ALT_FIGURES_DIR / "Fig_S6_r4_multiseed_replication.png", dpi=300)
    plt.close(fig)

    print(f"Generated Figure S6 (PNG only, 300 dpi):\n  {out_png}")
    return out_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    generate_figure_s6(args.results_root or default_results_root(), args.out_dir)
