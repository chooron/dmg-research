from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path("/workspace/autoresearch")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PREV_DIR = ROOT / "project/flexmopex/analysis/flex_mopex_v2_parameter_decoupling"
ATTR_DIR = ROOT / "project/flexmopex/analysis/flex_mopex_v2_attribute_controls"
BLOCK_DIR = ROOT / "project/flexmopex/analysis/flex_mopex_v2_spatial_block_robustness"
OUT_DIR = ROOT / "project/flexmopex/analysis/flex_mopex_v2_parameter_decoupling_revised"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_ALPHA = 0.01
ALPHAS = [0.005, 0.01, 0.03]
RANDOM_STATE = 20260527

TARGETS = ["sum_weight", "share_snow", "share_int", "share_phen", "share_sub"]
SHARES = ["share_snow", "share_int", "share_phen", "share_sub"]
STRUCTURAL_SUMMARY = SHARES + ["sum_weight"]
PROCESS_GROUP_FOR_TARGET = {
    "share_snow": "snow_temperature",
    "share_sub": "routing_groundwater",
    "share_phen": "et_vegetation",
    "share_int": "et_vegetation",
    "sum_weight": "other_or_uncertain",
}
KEY_ATTRS = {
    "share_snow": ["frac_snow", "frac_snow_months_era5", "sd_seasonality_mm", "elev_mean", "tmean", "p_mean"],
    "share_phen": ["e_seasonality", "gvf_diff", "lai_diff", "p_seasonality", "seasonality_index"],
    "share_sub": [
        "baseflow_index",
        "runoff_ratio",
        "slope_mean",
        "soil_conductivity",
        "soil_porosity",
        "budyko_residual",
    ],
    "share_int": ["frac_forest", "lai_max", "lai_diff", "gvf_max", "gvf_diff", "swvl1_mean", "soil_porosity"],
    "sum_weight": ["aridity_index", "runoff_ratio", "budyko_residual", "low_prec_dur", "p_seasonality", "e_seasonality"],
}
COMPRESSION_TARGETS = [
    "frac_snow",
    "frac_snow_months_era5",
    "sd_seasonality_mm",
    "e_seasonality",
    "baseflow_index",
    "runoff_ratio",
    "budyko_residual",
    "aridity_index",
    "p_seasonality",
    "elev_mean",
    "slope_mean",
]
PCA_CONTROL_SPECS = [
    ("pc1", 1),
    ("pc2", 2),
    ("pc5", 5),
    ("pc10", 10),
    ("var25", 0.25),
    ("var50", 0.50),
    ("var70", 0.70),
    ("var80", 0.80),
    ("var90", 0.90),
    ("var95", 0.95),
]


def fdr_bh(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out.tolist()
    vals = p[ok]
    order = np.argsort(vals)
    ranks = np.arange(1, len(vals) + 1)
    adjusted = vals[order] * len(vals) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    tmp = np.empty_like(vals)
    tmp[order] = adjusted
    out[ok] = tmp
    return out.tolist()


def parameter_columns(df: pd.DataFrame) -> list[str]:
    excluded_tokens = ("reconstructed_w_", "share", "sum_weight")
    cols = []
    for col in df.columns:
        if not col.startswith("param_"):
            continue
        if any(token in col for token in excluded_tokens):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def assign_parameter_group(param: str) -> tuple[str, str, str]:
    name = param.removeprefix("param_")
    stem = name.split("_m")[0]
    low = stem.lower()
    if any(token in low for token in ["ddf", "tcrit", "tmin", "tmax", "snow", "melt", "temp"]):
        return "snow_temperature", "high", "temperature/snow threshold parameter name"
    if any(token in low for token in ["sb1", "sb2", "se", "storage", "soil", "bucket", "capacity"]):
        return "storage_soil", "high", "storage or soil-capacity parameter name"
    if any(token in low for token in ["tu", "tc", "rout", "baseflow", "recession", "groundwater", "subsurface"]):
        return "routing_groundwater", "high", "routing or groundwater timescale parameter name"
    if any(token in low for token in ["tw", "is_time", "et", "evap", "pet", "veg", "phen"]):
        return "et_vegetation", "medium", "ET/seasonality/phenology-related parameter name"
    return "other_or_uncertain", "low", "no transparent process keyword match"


def load_inputs() -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    merged = pd.read_csv(PREV_DIR / "merged_structure_parameters_attributes.csv")
    if (BLOCK_DIR / "block_assignments.csv").is_file():
        blocks = pd.read_csv(BLOCK_DIR / "block_assignments.csv")
        blocks["basin_id"] = blocks["basin_id"].astype(int)
        merged["basin_id"] = merged["basin_id"].astype(int)
        add_cols = [c for c in ["basin_id", "spatial_block_k5", "spatial_block_k8", "hydroclimatic_block"] if c in blocks.columns]
        for col in add_cols:
            if col != "basin_id" and col in merged.columns:
                merged = merged.drop(columns=[col])
        merged = merged.merge(blocks[add_cols], on="basin_id", how="left")
    param_cols = parameter_columns(merged)
    group_map_rows = []
    for param in param_cols:
        group, confidence, rule = assign_parameter_group(param)
        group_map_rows.append({"parameter": param, "group": group, "confidence": confidence, "rule": rule})
    group_map = pd.DataFrame(group_map_rows)
    group_map.to_csv(OUT_DIR / "parameter_group_map.csv", index=False)
    summary = (
        group_map.groupby(["group", "confidence"], dropna=False)
        .size()
        .reset_index(name="n_parameters")
        .sort_values(["group", "confidence"])
    )
    summary.to_csv(OUT_DIR / "parameter_group_summary.csv", index=False)
    groups = {g: rows["parameter"].tolist() for g, rows in group_map.groupby("group")}
    return merged, param_cols, groups


def model_factory(name: str):
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 25)))
    if name == "rf":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=40,
                max_depth=10,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )
    raise ValueError(name)


def cv_for_scheme(scheme: str):
    if scheme == "random5":
        return KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return LeaveOneGroupOut()


def valid_model_frame(df: pd.DataFrame, xcols: list[str], ycol: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
    x = df[xcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) if xcols else np.empty((len(df), 0))
    mask = np.isfinite(y)
    if xcols:
        mask &= np.isfinite(x).sum(axis=1) > 0
    return df.loc[mask].copy(), x[mask], y[mask]


def cv_predict(model_name: str, x: np.ndarray, y: np.ndarray, scheme: str, groups: np.ndarray | None = None) -> np.ndarray:
    return cross_val_predict(model_factory(model_name), x, y, cv=cv_for_scheme(scheme), groups=groups)


def cv_metrics(model_name: str, df: pd.DataFrame, xcols: list[str], ycol: str, scheme: str, group_col: str | None = None) -> dict[str, float]:
    sub, x, y = valid_model_frame(df, xcols, ycol)
    if len(y) < 30 or len(xcols) == 0:
        return {"n": len(y), "r2": np.nan, "rmse": np.nan}
    groups = None
    if group_col is not None:
        groups = sub[group_col].to_numpy()
        if len(pd.Series(groups).dropna().unique()) < 2:
            return {"n": len(y), "r2": np.nan, "rmse": np.nan}
    pred = cv_predict(model_name, x, y, scheme, groups)
    return {"n": len(y), "r2": float(r2_score(y, pred)), "rmse": float(math.sqrt(mean_squared_error(y, pred)))}


def residualize_oof(df: pd.DataFrame, cols: list[str], ycol: str) -> tuple[pd.Series, pd.Series]:
    y = pd.to_numeric(df[ycol], errors="coerce")
    if not cols:
        return y, pd.Series(True, index=df.index)
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    mask = y.notna() & (x.notna().sum(axis=1) > 0)
    if mask.sum() < 30:
        return pd.Series(np.nan, index=df.index), mask
    pred = cross_val_predict(
        model_factory("ridge"),
        x.loc[mask].to_numpy(dtype=float),
        y.loc[mask].to_numpy(dtype=float),
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    )
    resid = pd.Series(np.nan, index=df.index, dtype=float)
    resid.loc[mask] = y.loc[mask].to_numpy(dtype=float) - pred
    return resid, mask


def spearman_pair(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return np.nan, np.nan, int(mask.sum())
    result = stats.spearmanr(x[mask], y[mask])
    return float(result.statistic), float(result.pvalue), int(mask.sum())


def standardized_pcs(df: pd.DataFrame, param_cols: list[str]) -> tuple[np.ndarray, PCA, pd.DataFrame]:
    x = df[param_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x_imp = SimpleImputer(strategy="median").fit_transform(x)
    xz = StandardScaler().fit_transform(x_imp)
    pca = PCA(random_state=RANDOM_STATE).fit(xz)
    scores = pca.transform(xz)
    explained = pd.DataFrame(
        {
            "pc": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    return scores, pca, explained


def n_pcs_for_spec(explained: pd.DataFrame, spec: int | float) -> int:
    if isinstance(spec, int):
        return min(spec, len(explained))
    return int(np.searchsorted(explained["cumulative_variance_ratio"].to_numpy(), spec) + 1)


def run_grouped_residuals(df: pd.DataFrame, param_cols: list[str], groups: dict[str, list[str]], scores: np.ndarray, explained: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alpha in ALPHAS:
        sub = df[np.isclose(df["alpha"], alpha)].copy().reset_index(drop=True)
        score_sub = scores[df.index[np.isclose(df["alpha"], alpha)], :]
        pc_df = pd.DataFrame(score_sub, columns=[f"PC{i}" for i in range(1, score_sub.shape[1] + 1)])
        sub_pc = pd.concat([sub.reset_index(drop=True), pc_df], axis=1)
        for target, attrs in KEY_ATTRS.items():
            corresponding = PROCESS_GROUP_FOR_TARGET[target]
            non_corr_cols = [p for g, ps in groups.items() if g != corresponding for p in ps]
            corr_cols = groups.get(corresponding, [])
            control_sets = {
                "A_no_parameter_control": [],
                "B_non_corresponding_groups_only": non_corr_cols,
                "C_corresponding_group_only": corr_cols,
                "D_all_parameter_groups": param_cols,
                "E_first_5_parameter_pcs": [f"PC{i}" for i in range(1, min(5, score_sub.shape[1]) + 1)],
            }
            for label, threshold in [("F_pcs_90pct_variance", 0.90), ("F_pcs_95pct_variance", 0.95)]:
                n_pc = n_pcs_for_spec(explained, threshold)
                control_sets[label] = [f"PC{i}" for i in range(1, n_pc + 1)]
            for attr in attrs:
                if attr not in sub_pc.columns:
                    continue
                for control_name, controls in control_sets.items():
                    target_resid, _ = residualize_oof(sub_pc, controls, target)
                    attr_resid, _ = residualize_oof(sub_pc, controls, attr)
                    rho, pval, n = spearman_pair(target_resid, attr_resid)
                    rows.append(
                        {
                            "alpha": alpha,
                            "target": target,
                            "attribute": attr,
                            "corresponding_group": corresponding,
                            "control": control_name,
                            "n_control_features": len(controls),
                            "spearman_rho": rho,
                            "p_value": pval,
                            "n": n,
                        }
                    )
    out = pd.DataFrame(rows)
    out["fdr_p"] = fdr_bh(out["p_value"].tolist())
    return out


def summarize_grouped_residuals(corr: pd.DataFrame) -> pd.DataFrame:
    primary = corr[np.isclose(corr["alpha"], PRIMARY_ALPHA)].copy()
    rows = []
    for (target, control), group in primary.groupby(["target", "control"]):
        rows.append(
            {
                "alpha": PRIMARY_ALPHA,
                "target": target,
                "control": control,
                "median_abs_rho": float(group["spearman_rho"].abs().median()),
                "max_abs_rho": float(group["spearman_rho"].abs().max()),
                "n_fdr10_absrho15": int(((group["fdr_p"] < 0.1) & (group["spearman_rho"].abs() >= 0.15)).sum()),
                "n_pairs": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def run_pca_control_curve(df: pd.DataFrame, param_cols: list[str], scores: np.ndarray, explained: pd.DataFrame) -> pd.DataFrame:
    sub = df[np.isclose(df["alpha"], PRIMARY_ALPHA)].copy().reset_index(drop=True)
    score_sub = scores[df.index[np.isclose(df["alpha"], PRIMARY_ALPHA)], :]
    pc_df = pd.DataFrame(score_sub, columns=[f"PC{i}" for i in range(1, score_sub.shape[1] + 1)])
    sub = pd.concat([sub, pc_df], axis=1)
    rows = []
    for target, attrs in KEY_ATTRS.items():
        for attr in attrs:
            if attr not in sub.columns:
                continue
            rho0, p0, n0 = spearman_pair(sub[target], sub[attr])
            rows.append(
                {
                    "alpha": PRIMARY_ALPHA,
                    "target": target,
                    "attribute": attr,
                    "control_spec": "none",
                    "n_pcs": 0,
                    "controlled_variance_ratio": 0.0,
                    "spearman_rho": rho0,
                    "p_value": p0,
                    "n": n0,
                }
            )
            for label, spec in PCA_CONTROL_SPECS:
                n_pc = n_pcs_for_spec(explained, spec)
                controls = [f"PC{i}" for i in range(1, n_pc + 1)]
                target_resid, _ = residualize_oof(sub, controls, target)
                attr_resid, _ = residualize_oof(sub, controls, attr)
                rho, pval, n = spearman_pair(target_resid, attr_resid)
                rows.append(
                    {
                        "alpha": PRIMARY_ALPHA,
                        "target": target,
                        "attribute": attr,
                        "control_spec": label,
                        "n_pcs": n_pc,
                        "controlled_variance_ratio": float(explained.loc[n_pc - 1, "cumulative_variance_ratio"]),
                        "spearman_rho": rho,
                        "p_value": pval,
                        "n": n,
                    }
                )
    out = pd.DataFrame(rows)
    out["fdr_p"] = fdr_bh(out["p_value"].tolist())
    return out


def cv_schemes(df: pd.DataFrame) -> list[tuple[str, str | None]]:
    schemes = [("random5", None)]
    if "spatial_block_k5" in df.columns:
        schemes.append(("spatial_block_k5", "spatial_block_k5"))
    if "hydroclimatic_block" in df.columns:
        schemes.append(("hydroclimatic_block", "hydroclimatic_block"))
    return schemes


def run_group_predictability(df: pd.DataFrame, param_cols: list[str], groups: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    primary = df[np.isclose(df["alpha"], PRIMARY_ALPHA)].copy()
    feature_sets: dict[str, list[str]] = {"all_groups": param_cols}
    for group, cols in groups.items():
        feature_sets[f"group_only__{group}"] = cols
        feature_sets[f"all_except__{group}"] = [c for c in param_cols if c not in cols]
    for target in TARGETS:
        for fs_name, cols in feature_sets.items():
            if not cols:
                continue
            for model_name in ["ridge", "rf"]:
                for scheme, group_col in cv_schemes(primary):
                    metric = cv_metrics(model_name, primary, cols, target, scheme, group_col)
                    rows.append(
                        {
                            "alpha": PRIMARY_ALPHA,
                            "target": target,
                            "feature_set": fs_name,
                            "model": model_name,
                            "cv_scheme": scheme,
                            "n_features": len(cols),
                            **metric,
                        }
                    )
    pred = pd.DataFrame(rows)
    imp_rows = []
    for (target, model_name, scheme), group in pred.groupby(["target", "model", "cv_scheme"]):
        all_r2 = group.loc[group["feature_set"] == "all_groups", "r2"]
        if all_r2.empty:
            continue
        all_val = float(all_r2.iloc[0])
        for g in groups:
            without = group.loc[group["feature_set"] == f"all_except__{g}", "r2"]
            only = group.loc[group["feature_set"] == f"group_only__{g}", "r2"]
            imp_rows.append(
                {
                    "alpha": PRIMARY_ALPHA,
                    "target": target,
                    "group": g,
                    "model": model_name,
                    "cv_scheme": scheme,
                    "all_groups_r2": all_val,
                    "group_only_r2": float(only.iloc[0]) if len(only) else np.nan,
                    "all_except_group_r2": float(without.iloc[0]) if len(without) else np.nan,
                    "delta_r2_when_removed": all_val - float(without.iloc[0]) if len(without) else np.nan,
                }
            )
    return pred, pd.DataFrame(imp_rows)


def run_compression(df: pd.DataFrame, param_cols: list[str], scores: np.ndarray, explained: pd.DataFrame) -> pd.DataFrame:
    primary = df[np.isclose(df["alpha"], PRIMARY_ALPHA)].copy().reset_index(drop=True)
    if len(scores) == len(primary):
        score_sub = scores
    else:
        score_sub = scores[np.where(np.isclose(df["alpha"], PRIMARY_ALPHA))[0], :]
    n95 = n_pcs_for_spec(explained, 0.95)
    pc_cols = [f"PC{i}" for i in range(1, score_sub.shape[1] + 1)]
    primary = pd.concat([primary, pd.DataFrame(score_sub, columns=pc_cols)], axis=1)
    feature_sets = {
        "A_all_parameters": param_cols,
        "B_parameter_pcs_95pct": pc_cols[:n95],
        "C_first_5_parameter_pcs": pc_cols[:5],
        "D_structural_summary_5d": STRUCTURAL_SUMMARY,
        "E_structural_shares_4d": SHARES,
    }
    rows = []
    for target in COMPRESSION_TARGETS:
        if target not in primary.columns:
            continue
        for model_name in ["ridge", "rf"]:
            for scheme, group_col in cv_schemes(primary):
                scores_by_set = {}
                rmses_by_set = {}
                ns_by_set = {}
                for fs_name, cols in feature_sets.items():
                    metric = cv_metrics(model_name, primary, cols, target, scheme, group_col)
                    scores_by_set[fs_name] = metric["r2"]
                    rmses_by_set[fs_name] = metric["rmse"]
                    ns_by_set[fs_name] = metric["n"]
                all_r2 = scores_by_set["A_all_parameters"]
                pc5_r2 = scores_by_set["C_first_5_parameter_pcs"]
                for fs_name, cols in feature_sets.items():
                    r2 = scores_by_set[fs_name]
                    rows.append(
                        {
                            "alpha": PRIMARY_ALPHA,
                            "attribute_target": target,
                            "model": model_name,
                            "cv_scheme": scheme,
                            "feature_set": fs_name,
                            "n_features": len(cols),
                            "r2": r2,
                            "rmse": rmses_by_set[fs_name],
                            "n": ns_by_set[fs_name],
                            "retention_ratio_vs_all_parameters": r2 / all_r2 if np.isfinite(r2) and np.isfinite(all_r2) and all_r2 > 0 else np.nan,
                            "structural_vs_pc5_delta": scores_by_set["D_structural_summary_5d"] - pc5_r2,
                            "pc95_n_features": n95,
                        }
                    )
    return pd.DataFrame(rows)


def classify_revised(
    grouped_summary: pd.DataFrame,
    group_pred: pd.DataFrame,
    logo: pd.DataFrame,
    compression: pd.DataFrame,
) -> pd.DataFrame:
    comp_primary = compression[
        (compression["model"] == "ridge")
        & (compression["cv_scheme"] == "random5")
        & (compression["feature_set"] == "D_structural_summary_5d")
    ]
    global_median_retention = float(comp_primary["retention_ratio_vs_all_parameters"].replace([np.inf, -np.inf], np.nan).median())
    global_median_struct_vs_pc5 = float(comp_primary["structural_vs_pc5_delta"].median())
    rows = []
    for target in TARGETS:
        corr_group = PROCESS_GROUP_FOR_TARGET[target]
        pred_sub = group_pred[
            (group_pred["target"] == target)
            & (group_pred["model"] == "ridge")
            & (group_pred["cv_scheme"] == "random5")
            & group_pred["feature_set"].str.startswith("group_only__")
        ].copy()
        pred_sub["group"] = pred_sub["feature_set"].str.replace("group_only__", "", regex=False)
        best_group = pred_sub.sort_values("r2", ascending=False).head(1)
        best_group_name = best_group["group"].iloc[0] if len(best_group) else ""
        best_group_r2 = float(best_group["r2"].iloc[0]) if len(best_group) else np.nan
        corr_group_r2 = pred_sub.loc[pred_sub["group"] == corr_group, "r2"]
        corr_group_r2_val = float(corr_group_r2.iloc[0]) if len(corr_group_r2) else np.nan
        summ = grouped_summary[grouped_summary["target"] == target]
        no_control = summ.loc[summ["control"] == "A_no_parameter_control", "median_abs_rho"]
        noncorr = summ.loc[summ["control"] == "B_non_corresponding_groups_only", "median_abs_rho"]
        corr = summ.loc[summ["control"] == "C_corresponding_group_only", "median_abs_rho"]
        all_ctrl = summ.loc[summ["control"] == "D_all_parameter_groups", "median_abs_rho"]
        no_val = float(no_control.iloc[0]) if len(no_control) else np.nan
        noncorr_val = float(noncorr.iloc[0]) if len(noncorr) else np.nan
        corr_val = float(corr.iloc[0]) if len(corr) else np.nan
        all_val = float(all_ctrl.iloc[0]) if len(all_ctrl) else np.nan
        relevant_attrs = [a for a in KEY_ATTRS[target] if a in comp_primary["attribute_target"].unique()]
        comp_relevant = comp_primary[comp_primary["attribute_target"].isin(relevant_attrs)]
        median_retention = (
            float(comp_relevant["retention_ratio_vs_all_parameters"].replace([np.inf, -np.inf], np.nan).median())
            if not comp_relevant.empty
            else global_median_retention
        )
        median_struct_vs_pc5 = (
            float(comp_relevant["structural_vs_pc5_delta"].median())
            if not comp_relevant.empty
            else global_median_struct_vs_pc5
        )
        process_specific = (
            best_group_name == corr_group
            and np.isfinite(noncorr_val)
            and np.isfinite(corr_val)
            and noncorr_val > corr_val + 0.05
        )
        compact_summary = np.isfinite(median_retention) and median_retention >= 0.7
        if process_specific:
            cls = "A_process_specific_parameter_coupling"
        elif compact_summary and all_val < max(no_val * 0.5, 0.15):
            cls = "B_low_dimensional_summary"
        else:
            cls = "C_weak_or_ambiguous"
        rows.append(
            {
                "alpha": PRIMARY_ALPHA,
                "structural_variable": target,
                "corresponding_group": corr_group,
                "best_group": best_group_name,
                "best_group_r2": best_group_r2,
                "corresponding_group_r2": corr_group_r2_val,
                "median_abs_rho_no_control": no_val,
                "median_abs_rho_noncorresponding_control": noncorr_val,
                "median_abs_rho_corresponding_control": corr_val,
                "median_abs_rho_all_parameter_control": all_val,
                "relevant_compression_attributes": ",".join(relevant_attrs),
                "median_relevant_structural_summary_retention_ratio": median_retention,
                "median_relevant_structural_vs_pc5_delta": median_struct_vs_pc5,
                "classification": cls,
            }
        )
    return pd.DataFrame(rows)


def plot_grouped_decay(summary: pd.DataFrame) -> None:
    order = [
        "A_no_parameter_control",
        "B_non_corresponding_groups_only",
        "C_corresponding_group_only",
        "D_all_parameter_groups",
        "E_first_5_parameter_pcs",
        "F_pcs_90pct_variance",
        "F_pcs_95pct_variance",
    ]
    sub = summary[np.isclose(summary["alpha"], PRIMARY_ALPHA)]
    fig, ax = plt.subplots(figsize=(10, 5))
    for target in TARGETS:
        vals = []
        for control in order:
            s = sub[(sub["target"] == target) & (sub["control"] == control)]["median_abs_rho"]
            vals.append(float(s.iloc[0]) if len(s) else np.nan)
        ax.plot(range(len(order)), vals, marker="o", label=target)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([x.split("_", 1)[1].replace("_", " ") for x in order], rotation=35, ha="right")
    ax.set_ylabel("median absolute residual Spearman rho")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_grouped_residual_corr_decay_alpha_0.01.png", dpi=180)
    plt.close(fig)


def plot_process_examples(corr: pd.DataFrame) -> None:
    pairs = [
        ("share_snow", "frac_snow"),
        ("share_phen", "e_seasonality"),
        ("share_sub", "baseflow_index"),
        ("share_int", "frac_forest"),
    ]
    controls = ["A_no_parameter_control", "B_non_corresponding_groups_only", "C_corresponding_group_only", "D_all_parameter_groups"]
    sub = corr[np.isclose(corr["alpha"], PRIMARY_ALPHA)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for ax, (target, attr) in zip(axes.ravel(), pairs):
        vals = []
        for c in controls:
            r = sub[(sub["target"] == target) & (sub["attribute"] == attr) & (sub["control"] == c)]["spearman_rho"]
            vals.append(float(r.iloc[0]) if len(r) else np.nan)
        ax.bar(range(len(controls)), vals)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{target} vs {attr}")
        ax.set_xticks(range(len(controls)))
        ax.set_xticklabels(["none", "noncorr", "corr", "all"], rotation=20)
        ax.set_ylabel("residual rho")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_process_specific_control_examples.png", dpi=180)
    plt.close(fig)


def plot_pca_curves(curve: pd.DataFrame) -> None:
    for target in ["share_snow", "share_phen", "share_sub", "sum_weight"]:
        sub = curve[(curve["target"] == target) & (curve["control_spec"] != "none")].copy()
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for attr, group in sub.groupby("attribute"):
            group = group.sort_values("n_pcs")
            ax.plot(group["n_pcs"], group["spearman_rho"], marker="o", linewidth=1.2, label=attr)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("number of controlled parameter PCs")
        ax.set_ylabel("residual Spearman rho")
        ax.set_title(target)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"fig_pca_control_curve_{target}.png", dpi=180)
        plt.close(fig)


def plot_group_contribution(pred: pd.DataFrame) -> None:
    sub = pred[
        (pred["model"] == "ridge")
        & (pred["cv_scheme"] == "random5")
        & pred["feature_set"].str.startswith("group_only__")
    ].copy()
    sub["group"] = sub["feature_set"].str.replace("group_only__", "", regex=False)
    mat = sub.pivot(index="target", columns="group", values="r2").reindex(TARGETS)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=35, ha="right")
    fig.colorbar(im, ax=ax, label="random 5-fold Ridge R2")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_group_contribution_to_shares.png", dpi=180)
    plt.close(fig)


def plot_compression(comp: pd.DataFrame) -> None:
    sub = comp[(comp["model"] == "ridge") & (comp["cv_scheme"] == "random5")].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    targets = [t for t in COMPRESSION_TARGETS if t in sub["attribute_target"].unique()]
    feature_sets = ["A_all_parameters", "C_first_5_parameter_pcs", "D_structural_summary_5d", "E_structural_shares_4d"]
    x = np.arange(len(targets))
    width = 0.18
    for i, fs in enumerate(feature_sets):
        vals = [sub[(sub["attribute_target"] == t) & (sub["feature_set"] == fs)]["r2"].mean() for t in targets]
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=fs.replace("_", " "))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=35, ha="right")
    ax.set_ylabel("CV R2")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_compression_efficiency_r2.png", dpi=180)
    plt.close(fig)

    struct = sub[sub["feature_set"] == "D_structural_summary_5d"].copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(struct))
    ax.bar(x, struct["retention_ratio_vs_all_parameters"])
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("R2 retention vs all parameters")
    ax.set_xticks(x)
    ax.set_xticklabels(struct["attribute_target"], rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_retention_ratio_by_attribute.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(struct))
    ax.bar(x, struct["structural_vs_pc5_delta"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("R2(structural 5D) - R2(first 5 PCs)")
    ax.set_xticks(x)
    ax.set_xticklabels(struct["attribute_target"], rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_structure_vs_pc5_summary.png", dpi=180)
    plt.close(fig)


def table_text(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


def write_report(
    group_map: pd.DataFrame,
    group_summary: pd.DataFrame,
    grouped_summary: pd.DataFrame,
    pca_curve: pd.DataFrame,
    group_pred: pd.DataFrame,
    compression: pd.DataFrame,
    classification: pd.DataFrame,
    explained: pd.DataFrame,
) -> None:
    primary_group = grouped_summary[np.isclose(grouped_summary["alpha"], PRIMARY_ALPHA)]
    comp_ridge = compression[
        (compression["model"] == "ridge")
        & (compression["cv_scheme"] == "random5")
        & compression["feature_set"].isin(["A_all_parameters", "C_first_5_parameter_pcs", "D_structural_summary_5d", "E_structural_shares_4d"])
    ]
    comp_pivot = comp_ridge.pivot_table(index="attribute_target", columns="feature_set", values="r2").round(3)
    retention = comp_ridge[comp_ridge["feature_set"] == "D_structural_summary_5d"][
        ["attribute_target", "r2", "retention_ratio_vs_all_parameters", "structural_vs_pc5_delta"]
    ].round(3)
    low_order = (
        pca_curve[(pca_curve["control_spec"].isin(["pc1", "pc2", "pc5"])) & (pca_curve["target"].isin(TARGETS))]
        .groupby(["target", "control_spec"])["spearman_rho"]
        .apply(lambda s: float(np.nanmedian(np.abs(s))))
        .reset_index(name="median_abs_rho")
        .pivot(index="target", columns="control_spec", values="median_abs_rho")
        .round(3)
    )
    pc95_n = int(explained.loc[explained["cumulative_variance_ratio"] >= 0.95, "pc"].iloc[0])
    lines = [
        "# Revised Parameter Decoupling Report",
        "",
        "## A. Why the full-parameter residual test was too strict",
        "",
        "The previous full-control test asked whether structural shares contain information that cannot be recovered from all 50 learned continuous parameters. That is a useful upper-bound redundancy test, but it is too strict as the only criterion because the structural weights and continuous parameters are jointly generated by the same differentiable mapper from the same basin attributes. Co-encoding is therefore expected. The revised question is whether the shares provide compact, process-interpretable summaries of the parameter manifold.",
        "",
        "## B. Parameter group map",
        "",
        table_text(group_summary),
        "",
        "Uncertain parameters for manual review:",
        "",
        table_text(group_map[group_map["group"] == "other_or_uncertain"]),
        "",
        "## C. Process-specific coupling",
        "",
        "Median absolute target-attribute residual correlations by control at alpha=0.01:",
        "",
        table_text(primary_group.pivot_table(index="target", columns="control", values="median_abs_rho").round(3).reset_index()),
        "",
        "A process-specific pattern is strongest when non-corresponding parameter controls preserve a target-attribute relationship but the corresponding parameter group removes it. The current evidence is mixed: some shares are organized by meaningful groups, but not all corresponding groups dominate cleanly.",
        "",
        "## D. Low-order PCA controls",
        "",
        f"The first {pc95_n} parameter PCs explain at least 95% of parameter variance.",
        "",
        "Median absolute residual correlations after controlling low-order PCs:",
        "",
        low_order.to_string(),
        "",
        "Signals that survive PC1/PC2 but decay only after broader PC control are better interpreted as higher-order parameter-manifold organization rather than a single dominant projection.",
        "",
        "## E. Compression efficiency",
        "",
        "Random 5-fold Ridge R2 for hydrological attributes:",
        "",
        comp_pivot.to_string(),
        "",
        "Structural summary retention and comparison with first 5 PCs:",
        "",
        table_text(retention),
        "",
        "## F. Revised structural interpretation classification",
        "",
        table_text(classification.round(3)),
        "",
        "## G. Best compact process summaries",
        "",
        "`share_snow` is the clearest compact process summary: it is strongly encoded by snow-temperature parameters and the 5D structural summary retains high predictive information for snow attributes, although it does not outperform the first five parameter PCs. `share_phen` retains useful information for evaporation seasonality but is not cleanly tied to the ET/vegetation parameter group. `share_sub`, `sum_weight`, and `share_int` remain weak or ambiguous under the revised criteria because group-specific residual behavior and compression retention are not strong enough to support a process-summary claim.",
        "",
        "## H. Recommended Section 3.4 claim",
        "",
        "Title: Structural weights as compact process-level summaries of the learned parameter manifold.",
        "",
        "Recommended wording: learned structural weights are not independent information channels beyond the full learned parameter manifold. Instead, they provide compact, process-interpretable summaries of hydrological organization embedded in differentiable parameter learning. Process-specific coupling and compression efficiency are stronger evidence than full independence. Avoid claims of definitive identifiability, true mechanism discovery, or causality.",
        "",
        "Caveat: because both parameters and structural weights are generated by the mapper from basin attributes, reverse prediction of attributes is representation analysis, not a causal test.",
    ]
    (OUT_DIR / "parameter_decoupling_revised_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df, param_cols, groups = load_inputs()
    primary = df[np.isclose(df["alpha"], PRIMARY_ALPHA)].copy()
    scores, _, explained = standardized_pcs(primary, param_cols)
    explained.to_csv(OUT_DIR / "parameter_pca_explained_variance.csv", index=False)

    # For multi-alpha grouped residuals, compute alpha-specific PCs and concatenate in row order.
    score_all = np.full((len(df), len(param_cols)), np.nan)
    explained_primary = explained
    for alpha in ALPHAS:
        mask = np.isclose(df["alpha"], alpha)
        alpha_scores, _, _ = standardized_pcs(df.loc[mask].reset_index(drop=True), param_cols)
        score_all[np.where(mask)[0], : alpha_scores.shape[1]] = alpha_scores

    grouped_corr = run_grouped_residuals(df.reset_index(drop=True), param_cols, groups, score_all, explained_primary)
    grouped_corr.to_csv(OUT_DIR / "grouped_residual_attribute_correlations.csv", index=False)
    grouped_summary = summarize_grouped_residuals(grouped_corr)
    grouped_summary.to_csv(OUT_DIR / "grouped_residual_attribute_summary.csv", index=False)
    plot_grouped_decay(grouped_summary)
    plot_process_examples(grouped_corr)

    pca_curve = run_pca_control_curve(primary.reset_index(drop=True), param_cols, scores, explained)
    pca_curve.to_csv(OUT_DIR / "pca_control_curve_correlations.csv", index=False)
    plot_pca_curves(pca_curve)

    group_pred, logo = run_group_predictability(primary, param_cols, groups)
    group_pred.to_csv(OUT_DIR / "parameter_group_to_structure_predictability.csv", index=False)
    logo.to_csv(OUT_DIR / "leave_one_group_out_importance.csv", index=False)
    plot_group_contribution(group_pred)

    compression = run_compression(primary, param_cols, scores, explained)
    compression.to_csv(OUT_DIR / "information_compression_efficiency.csv", index=False)
    plot_compression(compression)

    group_map = pd.read_csv(OUT_DIR / "parameter_group_map.csv")
    group_summary = pd.read_csv(OUT_DIR / "parameter_group_summary.csv")
    classification = classify_revised(grouped_summary, group_pred, logo, compression)
    classification.to_csv(OUT_DIR / "revised_structure_interpretation_classification.csv", index=False)
    write_report(group_map, group_summary, grouped_summary, pca_curve, group_pred, compression, classification, explained)
    print(f"Wrote revised parameter decoupling outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
