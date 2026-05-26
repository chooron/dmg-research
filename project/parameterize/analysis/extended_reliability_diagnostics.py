from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("/workspace/autoresearch/project/parameterize")
ANALYSIS = ROOT / "manuscript" / "analysis"
OUT = ROOT / "manuscript" / "extends"

PARAMETER_LABELS = {
    "parBETA": "BETA",
    "parFC": "FC",
    "parK0": "K0",
    "parK1": "K1",
    "parK2": "K2",
    "parLP": "LP",
    "parPERC": "PERC",
    "parUZL": "UZL",
    "parTT": "TT",
    "parCFMAX": "CFMAX",
    "parCFR": "CFR",
    "parCWH": "CWH",
    "route_a": "UH_a",
    "route_b": "UH_b",
}

ATTRIBUTE_MAP = {
    "slope_mean": "slope_mean",
    "elev_mean": "elev_mean",
    "frac_snow": "frac_snow",
    "aridity": "aridity",
    "pet_mean": "pet_mean",
    "p_mean": "p_mean",
    "p_seasonality": "p_seasonality",
    "soil_conductivity": "soil_conductivity",
    "soil_depth": "soil_depth_pelletier",
    "forest_frac": "frac_forest",
    "lai_diff": "lai_diff",
    "high_prec_freq": "high_prec_freq",
    "high_prec_dur": "high_prec_dur",
    "low_prec_freq": "low_prec_freq",
}

KEY_ATTRIBUTES = list(ATTRIBUTE_MAP)

MAIN_RELATIONSHIPS = [
    ("parBETA", "slope_mean", "mean"),
    ("parFC", "pet_mean", "mean"),
    ("parPERC", "aridity", "mean"),
    ("parUZL", "soil_conductivity", "mean"),
    ("parCWH", "frac_snow", "mean"),
    ("parCFR", "frac_snow", "mean"),
]

PARTIAL_RELATIONSHIPS = [
    ("parBETA", "slope_mean", ["elev_mean", "frac_snow"]),
    ("parFC", "pet_mean", ["aridity", "p_mean"]),
    ("parPERC", "aridity", ["pet_mean", "p_mean"]),
    ("parUZL", "soil_conductivity", ["soil_depth", "slope_mean", "forest_frac"]),
]

EVIDENCE_ROWS = [
    ("parBETA", "slope_mean", "mean"),
    ("parFC", "pet_mean", "mean"),
    ("parPERC", "aridity", "mean"),
    ("parUZL", "soil_conductivity", "mean"),
    ("parCWH", "frac_snow", "mean"),
    ("parCFR", "frac_snow", "mean"),
    ("parCFMAX", "frac_snow", "std"),
    ("parTT", "frac_snow", "std"),
    ("parCWH", "frac_snow", "std"),
    ("parPERC", "aridity", "std"),
    ("parUZL", "soil_conductivity", "std"),
]


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def norm_basin_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.lstrip("0").replace("", "0")


def load_attributes() -> pd.DataFrame:
    attrs = read_csv(ROOT / "outputs" / "analysis" / "stability_stats" / "tables" / "basin_attributes.csv")
    attrs["basin_key"] = norm_basin_id(attrs["basin_id"])
    out = attrs[["basin_id", "basin_key"]].copy()
    missing = []
    for requested, actual in ATTRIBUTE_MAP.items():
        if actual not in attrs.columns:
            missing.append(f"{requested}->{actual}")
            out[requested] = np.nan
        else:
            out[requested] = pd.to_numeric(attrs[actual], errors="coerce")
    if missing:
        print(f"Missing mapped attributes: {missing}")
    return out


def load_parameters() -> pd.DataFrame:
    params = read_csv(ANALYSIS / "figure2" / "data" / "parameter_estimates_by_run_long.csv")
    params["basin_key"] = norm_basin_id(params["basin_id"])
    params["parameter_label"] = params["parameter"].map(PARAMETER_LABELS).fillna(params["parameter"])
    params["mean_response"] = pd.to_numeric(params["estimate_norm"], errors="coerce")
    params["std_response"] = pd.to_numeric(params["sample_std_norm"], errors="coerce")
    return params


def load_groups() -> pd.DataFrame:
    groups = read_csv(ANALYSIS / "figure4" / "data" / "basin_group_assignment_531.csv")
    groups["basin_key"] = norm_basin_id(groups["basin_id"])
    return groups[["basin_key", "group_id", "group_name"]].drop_duplicates("basin_key")


def rho(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan, np.nan, len(frame)
    stat, p = spearmanr(frame["x"], frame["y"])
    return float(stat), float(p), len(frame)


def sign_label(value: float) -> str:
    if pd.isna(value) or abs(value) < 1e-12:
        return "near_zero"
    return "positive" if value > 0 else "negative"


def sign_consistency(values: pd.Series) -> str:
    vals = values.dropna()
    vals = vals.loc[vals.abs() > 1e-12]
    if vals.empty:
        return "no_signal"
    signs = set(np.sign(vals).astype(int))
    if len(signs) == 1:
        return "consistent_positive" if 1 in signs else "consistent_negative"
    return "sign_flip_present"


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "No rows available."
    show = df if max_rows is None else df.head(max_rows)
    show = show.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda v: "NA" if pd.isna(v) else f"{v:.4g}")
        else:
            show[col] = show[col].map(lambda v: "NA" if pd.isna(v) else str(v))
    columns = list(show.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("\n", " ") for col in columns) + " |")
    return "\n".join(lines)


def write_md(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    lines = [f"# {title}", ""]
    for heading, content in sections:
        lines.extend([f"## {heading}", "", content.strip() or "None", ""])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def save_fig(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(matrix: pd.DataFrame, stem: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman rho")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6)
    save_fig(fig, stem)


def residualize_rank(target: pd.Series, controls: pd.DataFrame) -> pd.Series:
    frame = pd.concat([target.rename("target"), controls], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 5:
        return pd.Series(dtype=float)
    y = frame["target"].rank(method="average").to_numpy(dtype=float)
    x_cols = [frame[c].rank(method="average").to_numpy(dtype=float) for c in controls.columns]
    x = np.column_stack([np.ones(len(frame)), *x_cols])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    return pd.Series(residual, index=frame.index)


def classify_change(original: float, partial: float) -> str:
    if pd.isna(original) or pd.isna(partial):
        return "NA"
    if np.sign(original) != np.sign(partial) and abs(partial) > 0.05:
        return "reversed"
    ratio = abs(partial) / abs(original) if abs(original) > 1e-12 else np.nan
    if pd.notna(ratio) and ratio < 0.5:
        return "same sign, weakened"
    if pd.notna(ratio) and ratio > 1.25:
        return "same sign, stronger"
    return "same sign, similar magnitude"


def response_column(response_type: str) -> str:
    return "std_response" if response_type == "std" else "mean_response"


def run_level_relationships(params: pd.DataFrame, attrs: pd.DataFrame, rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    merged = params.merge(attrs, on="basin_key", how="inner")
    out = []
    for parameter, attribute, resp_type in rows:
        col = response_column(resp_type)
        sub = merged.loc[(merged["model_raw"] == "distributional") & (merged["parameter"] == parameter)]
        for (loss, seed), run in sub.groupby(["loss", "seed"]):
            stat, p, n = rho(run[col], run[attribute])
            out.append(
                {
                    "parameter": parameter,
                    "attribute": attribute,
                    "response_type": resp_type,
                    "loss": loss,
                    "seed": seed,
                    "spearman_rho": stat,
                    "p_value": p,
                    "n_basins": n,
                }
            )
    return pd.DataFrame(out)


def summarize_run_rhos(run_rhos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, sub in run_rhos.groupby(["parameter", "attribute", "response_type"]):
        by_loss = sub.groupby("loss")["spearman_rho"]
        loss_means = by_loss.mean()
        seed_sds = by_loss.std(ddof=1)
        rows.append(
            {
                "parameter": key[0],
                "attribute": key[1],
                "response_type": key[2],
                "run_mean_rho": sub["spearman_rho"].mean(),
                "run_sd_rho": sub["spearman_rho"].std(ddof=1),
                "cross_seed_sd": seed_sds.mean(),
                "cross_loss_sd": loss_means.std(ddof=1),
                "n_runs": len(sub),
                "run_sign_consistency": sign_consistency(sub["spearman_rho"]),
            }
        )
    return pd.DataFrame(rows)


def tercile_difference(merged: pd.DataFrame, parameter: str, attribute: str, response_type: str) -> float:
    col = response_column(response_type)
    sub = merged.loc[merged["parameter"] == parameter, ["basin_key", col, attribute]].dropna()
    if sub.empty:
        return np.nan
    sub = sub.groupby("basin_key", as_index=False).agg({col: "mean", attribute: "first"})
    ranks = sub[attribute].rank(method="first")
    try:
        sub["tercile"] = pd.qcut(ranks, 3, labels=["low", "middle", "high"])
    except ValueError:
        sub["tercile"] = pd.cut(ranks, 3, labels=["low", "middle", "high"])
    med = sub.groupby("tercile", observed=True)[col].median()
    if "high" not in med or "low" not in med:
        return np.nan
    return float(med.loc["high"] - med.loc["low"])


def make_collinearity(attrs: pd.DataFrame) -> pd.DataFrame:
    matrix = attrs[KEY_ATTRIBUTES].corr(method="spearman")
    matrix.to_csv(OUT / "attribute_collinearity_matrix.csv")
    plot_heatmap(matrix, "attribute_collinearity_heatmap", "Attribute collinearity among interpreted basin attributes")

    pairs = []
    for i, a in enumerate(KEY_ATTRIBUTES):
        for b in KEY_ATTRIBUTES[i + 1 :]:
            val = matrix.loc[a, b]
            pairs.append(
                {
                    "attribute_a": a,
                    "attribute_b": b,
                    "spearman_rho": val,
                    "abs_rho": abs(val),
                    "abs_ge_0_6": bool(abs(val) >= 0.6),
                    "abs_ge_0_8": bool(abs(val) >= 0.8),
                }
            )
    pair_df = pd.DataFrame(pairs).sort_values("abs_rho", ascending=False)
    pair_df.to_csv(OUT / "attribute_collinearity_pairs.csv", index=False)

    confound = []
    for parameter, attribute, resp_type in EVIDENCE_ROWS:
        correlated_06 = pair_df.loc[
            ((pair_df["attribute_a"] == attribute) | (pair_df["attribute_b"] == attribute)) & pair_df["abs_ge_0_6"]
        ].copy()
        correlated_08 = correlated_06.loc[correlated_06["abs_ge_0_8"]]
        confound.append(
            {
                "parameter": PARAMETER_LABELS.get(parameter, parameter),
                "attribute": attribute,
                "response_type": resp_type,
                "correlated_attributes_abs_ge_0_6": "; ".join(
                    sorted(
                        {
                            r["attribute_b"] if r["attribute_a"] == attribute else r["attribute_a"]
                            for _, r in correlated_06.iterrows()
                        }
                    )
                )
                or "none",
                "correlated_attributes_abs_ge_0_8": "; ".join(
                    sorted(
                        {
                            r["attribute_b"] if r["attribute_a"] == attribute else r["attribute_a"]
                            for _, r in correlated_08.iterrows()
                        }
                    )
                )
                or "none",
            }
        )
    confound_df = pd.DataFrame(confound)
    strong06 = pair_df.loc[pair_df["abs_ge_0_6"], ["attribute_a", "attribute_b", "spearman_rho", "abs_rho"]]
    strong08 = pair_df.loc[pair_df["abs_ge_0_8"], ["attribute_a", "attribute_b", "spearman_rho", "abs_rho"]]
    write_md(
        OUT / "collinearity_summary.md",
        "Attribute Collinearity Summary",
        [
            ("Input", f"Computed from {len(attrs)} CAMELS-US basins using Spearman correlations among requested attributes."),
            ("Pairs with |rho| >= 0.8", md_table(strong08)),
            ("Pairs with |rho| >= 0.6", md_table(strong06)),
            ("Relationship-level correlated attributes", md_table(confound_df)),
            (
                "Interpretation note",
                "Rows with correlated attributes should be treated as environmental-gradient summaries rather than isolated single-attribute controls. No causal interpretation is made here.",
            ),
        ],
    )
    return matrix


def make_partial_sensitivity(params: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    merged = params.merge(attrs, on="basin_key", how="inner")
    rows = []
    for parameter, attribute, controls in PARTIAL_RELATIONSHIPS:
        sub = merged.loc[(merged["model_raw"] == "distributional") & (merged["parameter"] == parameter)].copy()
        for (loss, seed), run in sub.groupby(["loss", "seed"]):
            needed = ["mean_response", attribute, *controls]
            run = run[needed].replace([np.inf, -np.inf], np.nan).dropna()
            original, original_p, n = rho(run["mean_response"], run[attribute])
            y_res = residualize_rank(run["mean_response"], run[controls])
            x_res = residualize_rank(run[attribute], run[controls])
            common_idx = y_res.index.intersection(x_res.index)
            partial, partial_p, partial_n = rho(y_res.loc[common_idx], x_res.loc[common_idx])
            rows.append(
                {
                    "parameter": parameter,
                    "parameter_label": PARAMETER_LABELS.get(parameter, parameter),
                    "attribute": attribute,
                    "controls": "; ".join(controls),
                    "loss": loss,
                    "seed": seed,
                    "original_spearman_rho": original,
                    "partial_residual_spearman_rho": partial,
                    "n_basins": n,
                    "partial_n_basins": partial_n,
                    "change_class": classify_change(original, partial),
                }
            )
    run_df = pd.DataFrame(rows)
    summary = (
        run_df.groupby(["parameter", "parameter_label", "attribute", "controls"], as_index=False)
        .agg(
            original_rho_mean=("original_spearman_rho", "mean"),
            original_rho_sd=("original_spearman_rho", "std"),
            partial_rho_mean=("partial_residual_spearman_rho", "mean"),
            partial_rho_sd=("partial_residual_spearman_rho", "std"),
            min_partial_rho=("partial_residual_spearman_rho", "min"),
            max_partial_rho=("partial_residual_spearman_rho", "max"),
            n_runs=("partial_residual_spearman_rho", "count"),
        )
    )
    sign_summary = (
        run_df.groupby(["parameter", "attribute"])["partial_residual_spearman_rho"]
        .apply(sign_consistency)
        .rename("sign_stability_after_controls")
        .reset_index()
    )
    summary = summary.merge(sign_summary, on=["parameter", "attribute"], how="left")
    summary["mean_change_class"] = summary.apply(
        lambda r: classify_change(r["original_rho_mean"], r["partial_rho_mean"]), axis=1
    )
    summary.to_csv(OUT / "partial_relationship_sensitivity.csv", index=False)
    run_df.to_csv(OUT / "partial_relationship_sensitivity_by_run.csv", index=False)
    write_md(
        OUT / "partial_relationship_sensitivity.md",
        "Partial Relationship Sensitivity",
        [
            (
                "Method",
                "For each distributional run, parameter means and target attributes were rank-residualized against the requested controls, then Spearman rho was computed between residuals.",
            ),
            ("Summary", md_table(summary)),
            (
                "Interpretation note",
                "The table reports sign retention, weakening, strengthening, or reversal after control adjustment. These checks address sensitivity to correlated attributes and do not isolate causal controls.",
            ),
        ],
    )
    return summary


def make_group_sensitivity(params: pd.DataFrame, attrs: pd.DataFrame, groups: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dist = params.loc[params["model_raw"] == "distributional"].copy()
    avg = (
        dist.groupby(["basin_key", "parameter"], as_index=False)
        .agg(mean_response=("mean_response", "mean"), std_response=("std_response", "mean"))
        .merge(attrs, on="basin_key", how="inner")
        .merge(groups, on="basin_key", how="inner")
    )
    leave_rows = []
    group_rows = []
    for parameter, attribute, _ in MAIN_RELATIONSHIPS:
        sub = avg.loc[avg["parameter"] == parameter].copy()
        all_rho, all_p, all_n = rho(sub["mean_response"], sub[attribute])
        for (gid, gname), gsub in sub.groupby(["group_id", "group_name"]):
            leave = sub.loc[sub["group_id"] != gid]
            leave_rho, leave_p, leave_n = rho(leave["mean_response"], leave[attribute])
            group_rho, group_p, group_n = rho(gsub["mean_response"], gsub[attribute])
            leave_rows.append(
                {
                    "parameter": parameter,
                    "parameter_label": PARAMETER_LABELS.get(parameter, parameter),
                    "attribute": attribute,
                    "all_basins_rho": all_rho,
                    "all_basins_n": all_n,
                    "excluded_group_id": gid,
                    "excluded_group_name": gname,
                    "leave_group_out_rho": leave_rho,
                    "leave_group_out_n": leave_n,
                    "delta_vs_all": leave_rho - all_rho if pd.notna(leave_rho) and pd.notna(all_rho) else np.nan,
                    "sign_matches_all": sign_label(leave_rho) == sign_label(all_rho),
                }
            )
            group_rows.append(
                {
                    "parameter": parameter,
                    "parameter_label": PARAMETER_LABELS.get(parameter, parameter),
                    "attribute": attribute,
                    "all_basins_rho": all_rho,
                    "group_id": gid,
                    "group_name": gname,
                    "within_group_rho": group_rho,
                    "within_group_p_value": group_p,
                    "within_group_n": group_n,
                    "delta_vs_all": group_rho - all_rho if pd.notna(group_rho) and pd.notna(all_rho) else np.nan,
                    "sign_matches_all": sign_label(group_rho) == sign_label(all_rho),
                }
            )
    leave_df = pd.DataFrame(leave_rows)
    group_df = pd.DataFrame(group_rows)
    leave_df.to_csv(OUT / "leave_group_out_relationships.csv", index=False)
    group_df.to_csv(OUT / "groupwise_relationships.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = []
    x = np.arange(len(MAIN_RELATIONSHIPS))
    width = 0.1
    group_ids = sorted(leave_df["excluded_group_id"].unique())
    for idx, gid in enumerate(group_ids):
        vals = []
        for parameter, attribute, _ in MAIN_RELATIONSHIPS:
            row = leave_df.loc[
                (leave_df["parameter"] == parameter) & (leave_df["attribute"] == attribute) & (leave_df["excluded_group_id"] == gid)
            ]
            vals.append(row["leave_group_out_rho"].iloc[0] if not row.empty else np.nan)
        ax.plot(x + (idx - len(group_ids) / 2) * width, vals, marker="o", linestyle="", label=gid, alpha=0.8)
    all_vals = [
        leave_df.loc[(leave_df["parameter"] == p) & (leave_df["attribute"] == a), "all_basins_rho"].iloc[0]
        for p, a, _ in MAIN_RELATIONSHIPS
    ]
    ax.plot(x, all_vals, color="black", marker="_", linestyle="", markersize=16, label="all")
    for p, a, _ in MAIN_RELATIONSHIPS:
        labels.append(f"{PARAMETER_LABELS[p]}-{a}")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Leave-group-out relationship sensitivity")
    ax.legend(ncol=4, fontsize=7)
    save_fig(fig, "leave_group_out_core_relationships")

    unstable_leave = leave_df.loc[leave_df["delta_vs_all"].abs() >= 0.15]
    unstable_group = group_df.loc[(group_df["within_group_n"] >= 20) & ((~group_df["sign_matches_all"]) | (group_df["delta_vs_all"].abs() >= 0.25))]
    write_md(
        OUT / "basin_group_sensitivity_summary.md",
        "Basin Group Sensitivity Summary",
        [
            ("Method", "Distributional parameter means were averaged across available distributional runs, then evaluated across all basins, leave-one-hydroclimatic-group-out subsets, and within each group."),
            ("Leave-group-out changes with |delta rho| >= 0.15", md_table(unstable_leave)),
            ("Within-group sign or magnitude changes", md_table(unstable_group)),
            ("Interpretation note", "This is a hydroclimatic-strata robustness check, not ungauged validation."),
        ],
    )
    return leave_df, group_df


def make_uncertainty_classification(std_corr: pd.DataFrame, mean_std: pd.DataFrame, boundary: pd.DataFrame) -> pd.DataFrame:
    drop_diag_cols = [
        "mean_std_spearman",
        "mean_std_p_value",
        "boundary_distance_std_spearman",
        "boundary_distance_std_p_value",
        "near_boundary_share",
        "interpretation_flag",
    ]
    dom = (
        std_corr.drop(columns=[c for c in drop_diag_cols if c in std_corr.columns])
        .sort_values(["parameter", "abs_rho"], ascending=[True, False])
        .drop_duplicates("parameter")
    )
    diag = dom.merge(mean_std, on="parameter", how="left").merge(boundary, on="parameter", how="left", suffixes=("", "_boundary"))
    diag["mean_std_coupling_flag"] = diag["mean_std_spearman"].abs() >= 0.5
    diag["boundary_sensitivity_flag"] = (diag["boundary_distance_std_spearman"].abs() >= 0.4) | (diag["near_boundary_share"] >= 0.25)

    def cls(row: pd.Series) -> str:
        mean_flag = bool(row["mean_std_coupling_flag"])
        boundary_flag = bool(row["boundary_sensitivity_flag"])
        if mean_flag and boundary_flag:
            return "mean-coupled and boundary-sensitive"
        if mean_flag:
            return "mean-coupled"
        if boundary_flag:
            return "boundary-sensitive"
        return "less-confounded"

    diag["diagnostic_class"] = diag.apply(cls, axis=1)
    out = diag[
        [
            "parameter",
            "attribute",
            "spearman_rho",
            "mean_std_spearman",
            "boundary_distance_std_spearman",
            "near_boundary_share",
            "mean_std_coupling_flag",
            "boundary_sensitivity_flag",
            "diagnostic_class",
        ]
    ].copy()
    out["parameter_label"] = out["parameter"].map(PARAMETER_LABELS).fillna(out["parameter"])
    out = out[
        [
            "parameter",
            "parameter_label",
            "attribute",
            "spearman_rho",
            "mean_std_spearman",
            "boundary_distance_std_spearman",
            "near_boundary_share",
            "mean_std_coupling_flag",
            "boundary_sensitivity_flag",
            "diagnostic_class",
        ]
    ].sort_values("parameter_label")
    out.to_csv(OUT / "uncertainty_diagnostic_classification.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    classes = sorted(out["diagnostic_class"].unique())
    for klass in classes:
        sub = out.loc[out["diagnostic_class"] == klass]
        ax.scatter(
            sub["mean_std_spearman"].abs(),
            sub["boundary_distance_std_spearman"].abs(),
            s=40 + 220 * sub["near_boundary_share"].fillna(0),
            alpha=0.75,
            label=klass,
        )
        for _, row in sub.iterrows():
            ax.text(abs(row["mean_std_spearman"]), abs(row["boundary_distance_std_spearman"]), row["parameter_label"], fontsize=7)
    ax.axvline(0.5, color="0.4", lw=0.8, linestyle="--")
    ax.axhline(0.4, color="0.4", lw=0.8, linestyle="--")
    ax.set_xlabel("|mean-std Spearman rho|")
    ax.set_ylabel("|boundary-distance/std Spearman rho|")
    ax.set_title("Uncertainty coupling and boundary diagnostics")
    ax.legend(fontsize=7)
    save_fig(fig, "uncertainty_coupling_boundary_scatter")

    write_md(
        OUT / "uncertainty_diagnostic_classification.md",
        "Uncertainty Diagnostic Classification",
        [
            ("Method", "For each parameter, the strongest distributional std-attribute relationship was combined with mean-std coupling and boundary-distance diagnostics."),
            ("Classification table", md_table(out)),
            ("Interpretation note", "Less-confounded rows are cleaner diagnostic uncertainty structures; mean-coupled or boundary-sensitive rows should remain cautionary diagnostic patterns."),
        ],
    )
    return out


def make_evidence_table(
    params: pd.DataFrame,
    attrs: pd.DataFrame,
    mean_corr: pd.DataFrame,
    std_corr: pd.DataFrame,
    consistency: pd.DataFrame,
    run_summary: pd.DataFrame,
    uncertainty_class: pd.DataFrame,
) -> pd.DataFrame:
    route_b_dom = uncertainty_class.loc[uncertainty_class["parameter"] == "route_b", "attribute"]
    rows = EVIDENCE_ROWS.copy()
    if not route_b_dom.empty:
        rows.append(("route_b", route_b_dom.iloc[0], "std"))
    dist_avg = (
        params.loc[params["model_raw"] == "distributional"]
        .groupby(["basin_key", "parameter"], as_index=False)
        .agg(mean_response=("mean_response", "mean"), std_response=("std_response", "mean"))
        .merge(attrs, on="basin_key", how="inner")
    )
    out = []
    for parameter, attribute, resp_type in rows:
        source = std_corr if resp_type == "std" else mean_corr
        corr_row = source.loc[(source["parameter"] == parameter) & (source["attribute"] == attribute)]
        run_row = run_summary.loc[
            (run_summary["parameter"] == parameter)
            & (run_summary["attribute"] == attribute)
            & (run_summary["response_type"] == resp_type)
        ]
        cons_row = consistency.loc[consistency["parameter"] == parameter]
        unc_row = uncertainty_class.loc[uncertainty_class["parameter"] == parameter]
        rho_value = corr_row["spearman_rho"].iloc[0] if not corr_row.empty else np.nan
        cross_seed_sd = run_row["cross_seed_sd"].iloc[0] if not run_row.empty else np.nan
        cross_loss_sd = run_row["cross_loss_sd"].iloc[0] if not run_row.empty else np.nan
        dominant_class_raw = cons_row["relationship_class"].iloc[0] if not cons_row.empty and resp_type == "mean" else "uncertainty diagnostic"
        if dominant_class_raw == "shared dominant controls":
            dominant_class = "shared"
        elif dominant_class_raw == "partially shared controls":
            dominant_class = "partially shared"
        elif dominant_class_raw == "model-sensitive controls":
            dominant_class = "model-sensitive"
        else:
            dominant_class = "uncertainty diagnostic"
        boundary_flag = bool(unc_row["boundary_sensitivity_flag"].iloc[0]) if (resp_type == "std" and not unc_row.empty) else np.nan
        mean_std_flag = bool(unc_row["mean_std_coupling_flag"].iloc[0]) if (resp_type == "std" and not unc_row.empty) else np.nan
        if resp_type == "std" and (boundary_flag or mean_std_flag):
            evidence_category = "caution: boundary or coupling affected"
        elif resp_type == "std":
            evidence_category = "structured uncertainty diagnostic"
        elif dominant_class == "shared" and pd.notna(cross_seed_sd) and pd.notna(cross_loss_sd) and cross_seed_sd <= 0.05 and cross_loss_sd <= 0.08:
            evidence_category = "stable behavioral gradient"
        elif dominant_class == "shared":
            evidence_category = "shared but weaker gradient"
        elif dominant_class == "partially shared":
            evidence_category = "partially shared diagnostic gradient"
        else:
            evidence_category = "shared but weaker gradient"
        out.append(
            {
                "parameter": parameter,
                "parameter_label": PARAMETER_LABELS.get(parameter, parameter),
                "attribute": attribute,
                "response_type": resp_type,
                "formulation": "distributional main result",
                "spearman_rho": rho_value,
                "cross_seed_sd": cross_seed_sd,
                "cross_loss_sd": cross_loss_sd,
                "high_minus_low_tercile_difference": tercile_difference(dist_avg, parameter, attribute, resp_type),
                "dominant_control_class": dominant_class,
                "boundary_sensitivity_flag": boundary_flag,
                "mean_std_coupling_flag": mean_std_flag,
                "suggested_evidence_category": evidence_category,
            }
        )
    table = pd.DataFrame(out)
    table.to_csv(OUT / "evidence_hierarchy_table.csv", index=False)
    write_md(
        OUT / "evidence_hierarchy_table.md",
        "Evidence Hierarchy Table",
        [
            ("Table", md_table(table)),
            ("Metric note", "Cross-seed SD is the mean within-loss SD across distributional runs; cross-loss SD is the SD of loss-wise mean rho. NA indicates the metric was not available for that row."),
        ],
    )
    return table


def make_value_vs_relationship_stability(params: pd.DataFrame) -> pd.DataFrame:
    value = read_csv(ANALYSIS / "figure2" / "data" / "parameter_stability_summary_by_parameter.csv")
    boundary = read_csv(ANALYSIS / "figure2" / "data" / "boundary_saturation_by_parameter.csv")
    rel = read_csv(ANALYSIS / "figure3" / "data" / "parameter_level_correlation_stability.csv")
    dominant = read_csv(ANALYSIS / "01_model_consistency" / "data" / "dominant_attribute_by_run.csv")

    value_summary = (
        value.groupby(["model_raw", "model_label", "parameter"], as_index=False)
        .agg(
            raw_parameter_cross_seed_sd=("median_normalized_seed_sd", "mean"),
            raw_parameter_cross_seed_sd_iqr=("iqr_normalized_seed_sd", "mean"),
        )
    )
    boundary_summary = (
        boundary.groupby(["model_raw", "model_label", "parameter"], as_index=False)
        .agg(boundary_saturation_fraction=("saturation_rate_05", "mean"))
    )
    rel_summary = (
        rel.groupby(["model", "parameter"], as_index=False)
        .agg(
            relationship_cross_seed_sd=("median_seed_sd_rho", "mean"),
            top3_overlap=("mean_top3_overlap", "mean"),
            top5_overlap=("mean_top5_overlap", "mean"),
        )
        .rename(columns={"model": "model_raw"})
    )
    dom_rows = []
    for (model, parameter), sub in dominant.groupby(["model_raw", "parameter"]):
        counts = sub["dominant_attribute"].value_counts()
        modal = counts.index[0]
        dom_rows.append(
            {
                "model_raw": model,
                "parameter": parameter,
                "modal_dominant_attribute": modal,
                "dominant_control_consistency": counts.iloc[0] / len(sub),
                "dominant_n_runs": len(sub),
            }
        )
    dom_summary = pd.DataFrame(dom_rows)
    table = (
        value_summary.merge(boundary_summary, on=["model_raw", "model_label", "parameter"], how="outer")
        .merge(rel_summary, on=["model_raw", "parameter"], how="outer")
        .merge(dom_summary, on=["model_raw", "parameter"], how="outer")
    )
    table["parameter_label"] = table["parameter"].map(PARAMETER_LABELS).fillna(table["parameter"])
    stable_cut = table["raw_parameter_cross_seed_sd"].quantile(0.25)
    variable_cut = table["raw_parameter_cross_seed_sd"].quantile(0.75)
    rel_stable_cut = table["relationship_cross_seed_sd"].quantile(0.25)
    rel_unstable_cut = table["relationship_cross_seed_sd"].quantile(0.75)

    def mismatch(row: pd.Series) -> str:
        flags = []
        if row["raw_parameter_cross_seed_sd"] <= stable_cut and (
            row["relationship_cross_seed_sd"] >= rel_unstable_cut or row["dominant_control_consistency"] < 0.5
        ):
            flags.append("stable values but unstable relationships")
        if row["raw_parameter_cross_seed_sd"] >= variable_cut and (
            row["relationship_cross_seed_sd"] <= rel_stable_cut and row["dominant_control_consistency"] >= 0.7
        ):
            flags.append("variable values but stable relationships")
        if row["raw_parameter_cross_seed_sd"] <= stable_cut and row["boundary_saturation_fraction"] >= 0.25:
            flags.append("low variability with boundary saturation")
        return "; ".join(flags) if flags else "aligned or unflagged"

    table["stability_mismatch_flag"] = table.apply(mismatch, axis=1)
    table = table[
        [
            "model_raw",
            "model_label",
            "parameter",
            "parameter_label",
            "raw_parameter_cross_seed_sd",
            "boundary_saturation_fraction",
            "relationship_cross_seed_sd",
            "dominant_control_consistency",
            "modal_dominant_attribute",
            "top3_overlap",
            "top5_overlap",
            "stability_mismatch_flag",
        ]
    ].sort_values(["model_raw", "parameter_label"])
    table.to_csv(OUT / "parameter_value_vs_relationship_stability.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for model, sub in table.groupby("model_raw"):
        ax.scatter(
            sub["raw_parameter_cross_seed_sd"],
            sub["relationship_cross_seed_sd"],
            s=40 + 220 * sub["boundary_saturation_fraction"].fillna(0),
            alpha=0.7,
            label=model,
        )
    ax.set_xlabel("Raw parameter cross-seed SD (normalized)")
    ax.set_ylabel("Relationship cross-seed SD")
    ax.set_title("Parameter-value stability vs relationship stability")
    ax.legend(fontsize=8)
    save_fig(fig, "parameter_value_vs_relationship_stability")

    flagged = table.loc[table["stability_mismatch_flag"] != "aligned or unflagged"]
    write_md(
        OUT / "stability_mismatch_summary.md",
        "Stability Mismatch Summary",
        [
            ("Thresholds", f"Stable-value cutoff={stable_cut:.4f}; variable-value cutoff={variable_cut:.4f}; stable-relationship cutoff={rel_stable_cut:.4f}; unstable-relationship cutoff={rel_unstable_cut:.4f}."),
            ("Flagged rows", md_table(flagged)),
            ("Interpretation note", "These mismatches show why raw parameter stability and relationship stability should be evaluated separately."),
        ],
    )
    return table


def final_report(generated: list[str], counts: dict[str, int], missing: list[str]) -> None:
    write_md(
        OUT / "analysis_extension_report.md",
        "Extended Reliability Diagnostics Report",
        [
            (
                "Input files used",
                "\n".join(
                    [
                        "- outputs/analysis/stability_stats/tables/basin_attributes.csv",
                        "- manuscript/analysis/figure2/data/parameter_estimates_by_run_long.csv",
                        "- manuscript/analysis/figure2/data/parameter_stability_summary_by_parameter.csv",
                        "- manuscript/analysis/figure2/data/boundary_saturation_by_parameter.csv",
                        "- manuscript/analysis/figure3/data/parameter_level_correlation_stability.csv",
                        "- manuscript/analysis/figure4/data/basin_group_assignment_531.csv",
                        "- manuscript/analysis/04_mean_attribute_relationships/data/distributional_mean_attribute_correlations.csv",
                        "- manuscript/analysis/06_uncertainty_spatial_data/data/mean_std_coupling_diagnostics.csv",
                        "- manuscript/analysis/06_uncertainty_spatial_data/data/boundary_uncertainty_diagnostics.csv",
                        "- manuscript/analysis/07_uncertainty_attribute_relationships/data/distributional_std_attribute_correlations.csv",
                        "- manuscript/analysis/01_model_consistency/data/model_dominant_consistency_summary.csv",
                        "- manuscript/analysis/01_model_consistency/data/dominant_attribute_by_run.csv",
                    ]
                ),
            ),
            (
                "Data included",
                f"Basins={counts['basins']}; distributional runs={counts['distributional_runs']}; all runs={counts['all_runs']}; parameters={counts['parameters']}; interpreted attributes={counts['attributes']}.",
            ),
            (
                "Analyses completed",
                "\n".join(
                    [
                        "- Attribute collinearity matrix, heatmap, high-correlation pairs, and relationship-level confounding list.",
                        "- Partial rank-residual relationship sensitivity for four shared mean relationships.",
                        "- Leave-hydroclimatic-group-out and within-group robustness checks for six core relationships.",
                        "- Compact evidence hierarchy for mean and uncertainty-gradient rows.",
                        "- Parameterwise uncertainty diagnostic classes using mean-std coupling and boundary diagnostics.",
                        "- Parameter-value stability versus relationship-stability mismatch table and scatter plot.",
                    ]
                ),
            ),
            ("Generated files", "\n".join(f"- {name}" for name in generated)),
            ("Missing data or unavailable metrics", "\n".join(f"- {item}" for item in missing) if missing else "- None"),
            (
                "Recommended implications for manuscript interpretation",
                "\n".join(
                    [
                        "- Treat correlated-attribute rows as behavioral gradients over environmental covariation, not isolated controls.",
                        "- Keep uncertainty rows with mean-std coupling or boundary sensitivity in a diagnostic/cautionary category.",
                        "- Use hydroclimatic-group robustness as a sensitivity check across strata, not as ungauged validation.",
                        "- Evaluate relationship stability separately from raw parameter-value stability because flagged mismatches occur.",
                    ]
                ),
            ),
        ],
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    attrs = load_attributes()
    params = load_parameters()
    groups = load_groups()
    make_collinearity(attrs)
    make_partial_sensitivity(params, attrs)
    make_group_sensitivity(params, attrs, groups)

    mean_corr = read_csv(ANALYSIS / "04_mean_attribute_relationships" / "data" / "distributional_mean_attribute_correlations.csv")
    std_corr = read_csv(ANALYSIS / "07_uncertainty_attribute_relationships" / "data" / "distributional_std_attribute_correlations.csv")
    mean_std = read_csv(ANALYSIS / "06_uncertainty_spatial_data" / "data" / "mean_std_coupling_diagnostics.csv")
    boundary = read_csv(ANALYSIS / "06_uncertainty_spatial_data" / "data" / "boundary_uncertainty_diagnostics.csv")
    consistency = read_csv(ANALYSIS / "01_model_consistency" / "data" / "model_dominant_consistency_summary.csv")

    uncertainty_class = make_uncertainty_classification(std_corr, mean_std, boundary)
    evidence_pairs = EVIDENCE_ROWS.copy()
    route_b_dom = uncertainty_class.loc[uncertainty_class["parameter"] == "route_b", "attribute"]
    if not route_b_dom.empty:
        evidence_pairs.append(("route_b", route_b_dom.iloc[0], "std"))
    run_rhos = run_level_relationships(params, attrs, evidence_pairs)
    run_rhos.to_csv(OUT / "evidence_relationships_by_distributional_run.csv", index=False)
    run_summary = summarize_run_rhos(run_rhos)
    make_evidence_table(params, attrs, mean_corr, std_corr, consistency, run_summary, uncertainty_class)
    make_value_vs_relationship_stability(params)

    generated = sorted(p.name for p in OUT.iterdir() if p.is_file())
    counts = {
        "basins": int(attrs["basin_key"].nunique()),
        "distributional_runs": int(params.loc[params["model_raw"] == "distributional", ["loss", "seed"]].drop_duplicates().shape[0]),
        "all_runs": int(params[["model_raw", "loss", "seed"]].drop_duplicates().shape[0]),
        "parameters": int(params["parameter"].nunique()),
        "attributes": len(KEY_ATTRIBUTES),
    }
    missing = []
    if "soil_depth_pelletier" not in read_csv(ROOT / "outputs" / "analysis" / "stability_stats" / "tables" / "basin_attributes.csv", nrows=1).columns:
        missing.append("soil_depth requested; soil_depth_pelletier mapping unavailable")
    if "frac_forest" not in read_csv(ROOT / "outputs" / "analysis" / "stability_stats" / "tables" / "basin_attributes.csv", nrows=1).columns:
        missing.append("forest_frac requested; frac_forest mapping unavailable")
    final_report(generated, counts, missing)
    print(f"Wrote {len(generated)} files to {OUT}")


if __name__ == "__main__":
    main()
