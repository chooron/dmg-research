#!/usr/bin/env python3
"""
Rebuild the all36 IC-dPL gap diagnosis from the train-loss-selected dPL
checkpoints (see reselect_dpl_trainloss_eval.py).

Replicates the exact methodology of the original all36 diagnosis
(results/all36_dpl_gap_diagnosis_20260812/):
  * per-basin gap = kge_ic - kge_dpl on 1995-10-01..2010-09-30 (365d warmup)
  * spearman rho/p via scipy.stats.spearmanr
  * rho CI: paired bootstrap, seed=42, n_boot=500, 2.5/97.5 percentiles
  * median-gap CI: paired bootstrap, seed=42, n_boot=1000
  * quantile groups: pd.qcut(attr, 5, duplicates='drop'); median-gap CI
    within group: bootstrap seed=42, n_boot=1000
  * FDR: Benjamini-Hochberg (statsmodels fdr_bh) over ALL 36x32=1152 tests
  * 32 continuous Caravan attributes (categoricals excluded)

Inputs:
  - old aligned CSV (kge_ic, provenance, status, caveats)
  - new by_basin kge_dpl CSVs (train-loss selection)
  - data/caravan_671_attributes.csv
Outputs written under results/all36_dpl_gap_diagnosis_20260812_trainloss/
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
OLD_DIR = BENCHMARK_ROOT / "results/all36_dpl_gap_diagnosis_20260812"
NEW_DIR = BENCHMARK_ROOT / "results/all36_dpl_gap_diagnosis_20260812_trainloss"
DATA = REPO_ROOT / "data"

RNG_SEED = 42
RHO_NBOOT = 500
GAP_NBOOT = 1000
ALPHA = 0.05
EVAL_START, EVAL_END = "1995-10-01", "2010-09-30"
CATEGORICAL = {"lc_dom", "g_1st_cls", "g_2nd_cls"}


def load_attributes() -> pd.DataFrame:
    caravan = pd.read_csv(DATA / "caravan_671_attributes.csv")
    ids = np.asarray(ast.literal_eval((DATA / "531sub_id.txt").read_text()), dtype=np.int64)
    ids8 = pd.Index([f"{b:08d}" for b in ids], name="basin_id")
    caravan["gid8"] = caravan.gauge_id.astype(str).str.strip().map(
        lambda s: f"{int(s):08d}" if s.isdigit() else s
    )
    keep = [c for c in caravan.columns if c not in {"gauge_id", "gid8", *CATEGORICAL}]
    attrs = caravan[caravan.gid8.isin(set(ids8))].set_index("gid8").reindex(ids8)[keep]
    attrs.index = pd.Index([int(b) for b in ids], name="basin_id")
    return attrs


def bootstrap_ci(values: np.ndarray, n_boot: int, stat: str = "median") -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    n = len(values)
    if stat == "median":
        draws = np.array([np.median(rng.choice(values, n, replace=True)) for _ in range(n_boot)])
    else:
        draws = np.array([np.median(rng.choice(values, n, replace=True)) for _ in range(n_boot)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def rho_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    n = len(x)
    draws = []
    for _ in range(RHO_NBOOT):
        idx = rng.choice(n, n, replace=True)
        draws.append(stats.spearmanr(x[idx], y[idx])[0])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    NEW_DIR.mkdir(parents=True, exist_ok=True)
    (NEW_DIR / "by_basin").mkdir(exist_ok=True)
    old = pd.read_csv(OLD_DIR / "all36_aligned_basin_level.csv")
    sel = pd.read_csv(NEW_DIR / "dpl_epoch_selection_trainloss.csv").set_index("model")
    attrs = load_attributes()

    # ---------- 1. aligned basin-level CSV ----------
    frames = []
    for model in sorted(old.model.unique()):
        o = old[old.model == model][
            ["basin_id", "kge_ic", "ic_provenance", "ic_valid", "model_status", "technical_caveat"]
        ].copy()
        dpl = pd.read_csv(NEW_DIR / "by_basin" / f"{model}.csv")
        o = o.merge(dpl, on="basin_id", how="left")
        o["model"] = model
        o["gap_ic_minus_dpl"] = o.kge_ic - o.kge_dpl
        o["evaluation_start"] = EVAL_START
        o["evaluation_end"] = EVAL_END
        o["dpl_provenance"] = f"results/dpl_round13_20260805/auto100/checkpoints/{model}/epoch_{int(sel.loc[model,'selected_epoch']):03d}.pt"
        o["dpl_epoch"] = int(sel.loc[model, "selected_epoch"])
        o["dpl_selection_rule"] = "train_loss_argmin_over_saved_checkpoints"
        o["dpl_valid"] = True
        frames.append(o)
    aligned = pd.concat(frames, ignore_index=True)
    aligned = aligned[
        ["basin_id", "model", "kge_ic", "kge_dpl", "gap_ic_minus_dpl", "evaluation_start",
         "evaluation_end", "ic_provenance", "dpl_provenance", "ic_valid", "dpl_valid",
         "model_status", "technical_caveat", "dpl_epoch", "dpl_selection_rule"]
    ]
    aligned.to_csv(NEW_DIR / "all36_aligned_basin_level.csv", index=False)
    print(f"aligned rows: {len(aligned)}")

    # ---------- 2. model gap distribution summary ----------
    rows = []
    for model in sorted(aligned.model.unique()):
        g = aligned[aligned.model == model].gap_ic_minus_dpl.dropna()
        ic = aligned[aligned.model == model].kge_ic.dropna()
        dpl = aligned[aligned.model == model].kge_dpl.dropna()
        lo, hi = bootstrap_ci(g.values, GAP_NBOOT)
        rows.append(
            {
                "model": model,
                "n_ic": len(ic), "n_dpl": len(dpl), "n_paired": len(g),
                "ic_median_kge": float(ic.median()), "dpl_median_kge": float(dpl.median()),
                "difference_of_medians": float(ic.median() - dpl.median()),
                "paired_median_gap": float(g.median()), "paired_mean_gap": float(g.mean()),
                "gap_std": float(g.std()), "gap_iqr": float(np.percentile(g, 75) - np.percentile(g, 25)),
                "gap_p10": float(np.percentile(g, 10)), "gap_p25": float(np.percentile(g, 25)),
                "gap_p75": float(np.percentile(g, 75)), "gap_p90": float(np.percentile(g, 90)),
                "fraction_ic_gt_dpl": float((g > 0).mean()),
                "fraction_dpl_gt_ic": float((g < 0).mean()),
                "fraction_abs_gap_gt_005": float((g.abs() > 0.05).mean()),
                "fraction_abs_gap_gt_010": float((g.abs() > 0.10).mean()),
                "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
                "model_status": aligned[aligned.model == model].model_status.iloc[0],
                "technical_caveat": aligned[aligned.model == model].technical_caveat.iloc[0],
            }
        )
    summary = pd.DataFrame(rows).sort_values("paired_median_gap", ascending=False).reset_index(drop=True)
    summary.to_csv(NEW_DIR / "model_gap_distribution_summary.csv", index=False)
    print("model summary written")

    # ---------- 3. basin x model gap matrix ----------
    matrix = aligned.pivot(index="basin_id", columns="model", values="gap_ic_minus_dpl")[sorted(aligned.model.unique())]
    matrix.to_csv(NEW_DIR / "basin_model_gap_matrix.csv", index=False)

    # ---------- 4. basin cross-model gap summary ----------
    g = matrix.values
    cross = pd.DataFrame(
        {
            "basin_id": matrix.index,
            "cross_model_median_gap": np.nanmedian(g, axis=1),
            "cross_model_mean_gap": np.nanmean(g, axis=1),
            "cross_model_gap_std": np.nanstd(g, axis=1),
            "cross_model_gap_iqr": np.nanpercentile(g, 75, axis=1) - np.nanpercentile(g, 25, axis=1),
            "fraction_models_ic_gt_dpl": np.nanmean(g > 0, axis=1),
            "fraction_models_gap_gt_005": np.nanmean(g > 0.05, axis=1),
            "fraction_models_gap_lt_minus_005": np.nanmean(g < -0.05, axis=1),
            "valid_model_count": int(matrix.shape[1]),
        }
    )
    cross.to_csv(NEW_DIR / "basin_cross_model_gap_summary.csv", index=False)

    # ---------- 5. analysis basin attributes ----------
    attrs_out = attrs.reset_index()
    attrs_out.to_csv(NEW_DIR / "analysis_basin_attributes.csv", index=False)

    # ---------- 6. spearman per (model, attr) ----------
    spearman_rows = []
    gap_by = aligned.set_index(["basin_id", "model"]).gap_ic_minus_dpl.unstack()
    for model in sorted(aligned.model.unique()):
        g = gap_by[model].reindex(attrs.index)
        for attr in attrs.columns:
            x = attrs[attr].astype(float)
            mask = x.notna() & g.notna()
            rho, p = stats.spearmanr(x[mask], g[mask])
            lo, hi = rho_ci(x[mask].values, g[mask].values)
            spearman_rows.append(
                {"model": model, "attribute": attr, "rho": float(rho), "p": float(p),
                 "n": int(mask.sum()), "ci_low": lo, "ci_high": hi}
            )
    spearman = pd.DataFrame(spearman_rows)
    _, qvals, _, _ = multipletests(spearman.p.values, method="fdr_bh")
    spearman["q_fdr"] = qvals
    spearman.to_csv(NEW_DIR / "model_attribute_gap_spearman.csv", index=False)
    print("spearman written")

    # ---------- 7. quantile gradients ----------
    qrows = []
    for model in sorted(aligned.model.unique()):
        g = gap_by[model].reindex(attrs.index)
        for attr in attrs.columns:
            x = attrs[attr].astype(float)
            mask = x.notna() & g.notna()
            x_ser = x[mask]
            try:
                qq = pd.qcut(x_ser, 5, labels=False, duplicates="drop")
                if int(qq.nunique()) != 5:
                    raise ValueError("fewer than 5 bins")
            except Exception:
                # Tie-heavy attributes: fall back to quantile edges over the
                # unique values so the Q1..Q5 schema is preserved.
                edges = np.quantile(np.unique(x_ser), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                qq = pd.cut(x_ser, np.unique(edges), labels=False, include_lowest=True)
            for lab in range(int(qq.max()) + 1):
                quintile_name = f"Q{lab + 1}"
                sub = g[mask][qq == lab]
                if len(sub) == 0:
                    continue
                lo, hi = bootstrap_ci(sub.values, GAP_NBOOT)
                qrows.append(
                    {"model": model, "attribute": attr, "quintile": quintile_name,
                     "median_gap": float(sub.median()), "mean_gap": float(sub.mean()),
                     "ci_low": lo, "ci_high": hi, "basin_count": int(len(sub))}
                )
    quantiles = pd.DataFrame(qrows)
    quantiles.to_csv(NEW_DIR / "model_attribute_gap_quantiles.csv", index=False)
    print("quantiles written")

    # ---------- 8. attribute cross-model consistency ----------
    cons_rows = []
    for attr in attrs.columns:
        sub = spearman[spearman.attribute == attr]
        pos = int(((sub.q_fdr < ALPHA) & (sub.rho > 0)).sum())
        neg = int(((sub.q_fdr < ALPHA) & (sub.rho < 0)).sum())
        cons_rows.append(
            {
                "attribute": attr,
                "median_rho": float(sub.rho.median()),
                "iqr_rho": float(np.percentile(sub.rho, 75) - np.percentile(sub.rho, 25)),
                "fraction_positive_rho": float((sub.rho > 0).mean()),
                "fraction_negative_rho": float((sub.rho < 0).mean()),
                "fdr_significant_positive_count": pos,
                "fdr_significant_negative_count": neg,
                "total_fdr_significant_count": pos + neg,
            }
        )
    consistency = pd.DataFrame(cons_rows).sort_values("total_fdr_significant_count", ascending=False)
    consistency.to_csv(NEW_DIR / "attribute_cross_model_consistency.csv", index=False)
    print("consistency written")

    # ---------- 9. figures ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 9a gap distribution boxplot
    fig, ax = plt.subplots(figsize=(16, 7))
    order = summary.sort_values("paired_median_gap")["model"].tolist()
    data = [matrix[m].dropna().values for m in order]
    bp = ax.boxplot(data, labels=order, showfliers=False, patch_artist=True, widths=0.7)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Paired Gap (KGE_IC - KGE_dPL)")
    ax.set_title("Basin-level IC-dPL gap distribution across 36 models (train-loss dPL selection)")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(NEW_DIR / "all36_gap_distribution.png", dpi=150)
    plt.close(fig)

    # 9b basin x model heatmap
    fig, ax = plt.subplots(figsize=(16, 10))
    vmax = np.nanpercentile(np.abs(g), 95)
    im = ax.imshow(matrix.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Model")
    ax.set_ylabel("531 basins (CAMELS id order)")
    ax.set_title("Basin x model paired gap matrix (train-loss dPL selection)")
    fig.colorbar(im, label="KGE_IC - KGE_dPL")
    fig.tight_layout()
    fig.savefig(NEW_DIR / "basin_model_gap_heatmap.png", dpi=150)
    plt.close(fig)

    # 9c rho heatmap
    rho_piv = spearman.pivot(index="model", columns="attribute", values="rho")
    rho_piv = rho_piv.reindex(index=order, columns=consistency.attribute.tolist())
    fig, ax = plt.subplots(figsize=(16, 9))
    im = ax.imshow(rho_piv.values, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(len(rho_piv.columns)))
    ax.set_xticklabels(rho_piv.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(rho_piv.index)))
    ax.set_yticklabels(rho_piv.index, fontsize=8)
    ax.set_title("Spearman rho: model gap vs catchment attribute (train-loss dPL selection)")
    fig.colorbar(im, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(NEW_DIR / "model_attribute_rho_heatmap.png", dpi=150)
    plt.close(fig)

    # 9d consistency bars
    c = consistency.sort_values("median_rho")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.barh(c.attribute, c.median_rho, color=["#d7191c" if v > 0 else "#2c7bb6" for v in c.median_rho])
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_title("Cross-model median Spearman rho")
    ax2.barh(c.attribute, c.total_fdr_significant_count, color="#636363")
    ax2.set_title(f"FDR-significant model count (q<{ALPHA})")
    for a in (ax1, ax2):
        a.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(NEW_DIR / "attribute_cross_model_consistency.png", dpi=150)
    plt.close(fig)
    print("figures written")

    print("Done.")


if __name__ == "__main__":
    main()
