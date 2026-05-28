from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path("/workspace/autoresearch")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.base import clone
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

from project.flexmopex.models.base_mopex import (
    MOPEX_PARAM_NAMES,
    MOPEX_PARAMS_BOUNDS,
    ROUTING_BOUNDS,
    ROUTING_PARAM_NAMES,
)
from project.flexmopex.models.parameter_nets import LearnedStructureNet
STRUCTURE_PATH = ROOT / "project/flexmopex/analysis/flex_mopex_v2_attribute_controls/merged_full_v2.csv"
BLOCK_PATH = ROOT / "project/flexmopex/analysis/flex_mopex_v2_spatial_block_robustness/block_assignments.csv"
FLEX_OUTPUT_ROOT = ROOT / "project/flexmopex/outputs/flex_mopex_v1"
OUT_DIR = ROOT / "project/flexmopex/analysis/flex_mopex_v2_parameter_decoupling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_ALPHA = 0.01
ALPHAS = [0.005, 0.01, 0.03]
TARGETS = ["sum_weight", "share_snow", "share_int", "share_phen", "share_sub"]
SHARES = ["share_snow", "share_int", "share_phen", "share_sub"]
WEIGHT_NAMES = ["w_phen", "w_int", "w_snow", "w_sub"]
RANDOM_STATE = 20260527
N_PERM = 100


class CheckpointLearnedStructureNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        head_hidden_dim: int,
        dropout: float,
        output_sizes: dict[str, int],
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_dim, head_hidden_dim),
                    nn.Tanh(),
                    nn.Linear(head_hidden_dim, size),
                )
                for name, size in output_sizes.items()
            }
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        shared = self.backbone(x["c_nn_norm"])
        return {name: head(shared) for name, head in self.heads.items()}

NN_ATTRS = [
    "p_mean",
    "pet_mean",
    "p_seasonality",
    "frac_snow",
    "aridity",
    "high_prec_freq",
    "high_prec_dur",
    "low_prec_freq",
    "low_prec_dur",
    "elev_mean",
    "slope_mean",
    "area_gages2",
    "frac_forest",
    "lai_max",
    "lai_diff",
    "gvf_max",
    "gvf_diff",
    "dom_land_cover_frac",
    "dom_land_cover",
    "root_depth_50",
    "soil_depth_pelletier",
    "soil_depth_statsgo",
    "soil_porosity",
    "soil_conductivity",
    "max_water_content",
    "sand_frac",
    "silt_frac",
    "clay_frac",
    "geol_1st_class",
    "glim_1st_class_frac",
    "geol_2nd_class",
    "glim_2nd_class_frac",
    "carbonate_rocks_frac",
    "geol_porosity",
    "geol_permeability",
]

RESIDUAL_ATTRS = {
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
INCREMENTAL_TARGETS = [
    "frac_snow",
    "frac_snow_months_era5",
    "e_seasonality",
    "baseflow_index",
    "runoff_ratio",
    "budyko_residual",
    "aridity_index",
    "p_seasonality",
    "elev_mean",
]


def format_alpha(alpha: float) -> str:
    text = f"{alpha:g}"
    return text


def parse_alpha(path: Path) -> float | None:
    for part in path.parts:
        if part.startswith("alpha_"):
            value = part.removeprefix("alpha_").replace("_", ".")
            try:
                return float(value)
            except ValueError:
                continue
    return None


def parse_seed(path: Path) -> int | None:
    match = re.search(r"(?:seed_|_)(\d+)(?:/|$)", str(path))
    if match:
        return int(match.group(1))
    # Flex-MOPEX run names include the training seed as the final token before the model name.
    match = re.search(r"_noWU_(\d+)/FlexMopex", str(path))
    return int(match.group(1)) if match else None


def find_checkpoint(alpha: float) -> Path | None:
    alpha_dir = f"alpha_{format_alpha(alpha)}"
    pattern = f"{alpha_dir}/**/dFlexMopexV1_Ep50.pt"
    matches = sorted(FLEX_OUTPUT_ROOT.glob(pattern))
    return matches[0] if matches else None


def checkpoint_sibling(path: Path, name: str) -> Path:
    return path.parent / name


def normalize_attrs(df: pd.DataFrame, stats_path: Path) -> np.ndarray:
    with stats_path.open("r", encoding="utf-8") as f:
        norm = json.load(f)
    cols = []
    for attr in NN_ATTRS:
        if attr not in df.columns:
            raise KeyError(f"Missing mapper attribute column: {attr}")
        values = pd.to_numeric(df[attr], errors="coerce").to_numpy(dtype=float)
        if attr in norm and len(norm[attr]) >= 4:
            mean, std = float(norm[attr][2]), float(norm[attr][3])
            if std == 0 or not np.isfinite(std):
                std = 1.0
            values = (values - mean) / std
        cols.append(values)
    x = np.column_stack(cols).astype(np.float32)
    med = np.nanmedian(x, axis=0)
    inds = np.where(~np.isfinite(x))
    if len(inds[0]):
        x[inds] = med[inds[1]]
    return x


def descale_params(raw: np.ndarray, nmul: int) -> pd.DataFrame:
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    raw_params = raw[:, : len(MOPEX_PARAM_NAMES) * nmul].reshape(len(raw), len(MOPEX_PARAM_NAMES), nmul)
    out = {}
    for i, name in enumerate(MOPEX_PARAM_NAMES):
        lo, hi = MOPEX_PARAMS_BOUNDS[name]
        vals = lo + sigmoid(raw_params[:, i, :]) * (hi - lo)
        for m in range(nmul):
            out[f"param_{name}_m{m + 1}"] = vals[:, m]
    return pd.DataFrame(out)


def descale_routing(raw_gamma: np.ndarray) -> pd.DataFrame:
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    out = {}
    for i, name in enumerate(ROUTING_PARAM_NAMES):
        lo, hi = ROUTING_BOUNDS[name]
        out[f"param_{name}"] = lo + sigmoid(raw_gamma[:, i]) * (hi - lo)
    return pd.DataFrame(out)


def load_reconstructed_params(structure: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory_rows = []
    merge_rows = []
    param_frames = []

    for alpha in ALPHAS:
        ckpt = find_checkpoint(alpha)
        sub = structure[np.isclose(structure["alpha"], alpha)].copy().reset_index(drop=True)
        if ckpt is None or sub.empty:
            inventory_rows.append(
                {
                    "file_path": str(ckpt) if ckpt else "",
                    "parameter_variable_names": "",
                    "shape": "",
                    "dtype": "",
                    "number_of_basins": 0,
                    "alpha": alpha,
                    "seed": np.nan,
                    "basin_id_can_be_matched": False,
                    "source_type": "missing_checkpoint",
                }
            )
            continue

        config_path = checkpoint_sibling(ckpt, "config.json")
        stats_path = checkpoint_sibling(ckpt, "normalization_statistics.json")
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        nmul = int(cfg["delta_model"]["phy_model"].get("nmul", 4))
        hidden_size = int(cfg["delta_model"]["nn_model"].get("hidden_size", 128))
        dropout = float(cfg["delta_model"]["nn_model"].get("dr", 0.0))

        x = normalize_attrs(sub, stats_path)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        stripped = {k.removeprefix("nn_model."): v for k, v in state.items() if k.startswith("nn_model.")}
        if any(k.startswith("heads.params.2.") for k in stripped):
            head_hidden_dim = int(stripped["heads.params.0.weight"].shape[0])
            output_sizes = {
                "params": int(stripped["heads.params.2.weight"].shape[0]),
                "weights": int(stripped["heads.weights.2.weight"].shape[0]),
                "gamma_uh": int(stripped["heads.gamma_uh.2.weight"].shape[0]),
            }
            model = CheckpointLearnedStructureNet(
                input_dim=len(NN_ATTRS),
                hidden_dim=hidden_size,
                head_hidden_dim=head_hidden_dim,
                dropout=dropout,
                output_sizes=output_sizes,
            )
        else:
            model = LearnedStructureNet(
                input_dim=len(NN_ATTRS),
                hidden_dim=hidden_size,
                dropout=dropout,
                nmul=nmul,
                device="cpu",
            )
        model.load_state_dict(stripped)
        model.eval()
        with torch.no_grad():
            outputs = model({"c_nn_norm": torch.from_numpy(x)})

        raw_params = outputs["params"].numpy()
        raw_gamma = outputs["gamma_uh"].numpy()
        raw_weights = outputs["weights"].numpy()
        param_df = pd.concat([descale_params(raw_params, nmul), descale_routing(raw_gamma)], axis=1)
        weight_prob = torch.softmax(torch.from_numpy(raw_weights).view(len(sub), 4, 2), dim=-1)[..., 1].numpy()
        for idx, name in enumerate(WEIGHT_NAMES):
            param_df[f"reconstructed_{name}"] = weight_prob[:, idx]
        param_df.insert(0, "alpha", alpha)
        param_df.insert(1, "station_index", sub["station_index"].to_numpy())
        param_df.insert(2, "basin_id", sub["basin_id"].astype(int).to_numpy())
        param_df.insert(3, "gauge_id", sub["gauge_id"].astype(int).to_numpy())
        param_df.insert(4, "parameter_source_file", str(ckpt))
        param_frames.append(param_df)

        param_cols = [c for c in param_df.columns if c.startswith("param_")]
        max_weight_diff = {
            f"max_abs_diff_{name}": float(np.nanmax(np.abs(sub[name].to_numpy(dtype=float) - weight_prob[:, idx])))
            for idx, name in enumerate(WEIGHT_NAMES)
        }
        inventory_rows.append(
            {
                "file_path": str(ckpt),
                "parameter_variable_names": ",".join(param_cols),
                "shape": f"{len(sub)}x{len(param_cols)}",
                "dtype": "float64",
                "number_of_basins": int(len(sub)),
                "alpha": alpha,
                "seed": parse_seed(ckpt),
                "basin_id_can_be_matched": True,
                "source_type": "reconstructed_from_trained_mapper_checkpoint",
            }
        )
        merge_rows.append(
            {
                "alpha": alpha,
                "checkpoint_path": str(ckpt),
                "config_path": str(config_path),
                "normalization_statistics_path": str(stats_path),
                "n_structure_rows": int(len(sub)),
                "n_parameter_rows": int(len(param_df)),
                "n_matched_rows": int(len(param_df)),
                "n_parameter_columns": int(len(param_cols)),
                **max_weight_diff,
            }
        )

    inventory = pd.DataFrame(inventory_rows)
    diagnostics = pd.DataFrame(merge_rows)
    params = pd.concat(param_frames, ignore_index=True) if param_frames else pd.DataFrame()
    return params, inventory, diagnostics


def cv_predictions(model, x: np.ndarray, y: np.ndarray, scheme: str, groups: np.ndarray | None = None) -> np.ndarray:
    if scheme == "random5":
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = LeaveOneGroupOut()
    return cross_val_predict(model, x, y, cv=cv, groups=groups, n_jobs=None)


def model_factory(name: str, fast_null: bool = False):
    if name == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-4, 4, 25)),
        )
    if name == "rf":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=8 if fast_null else 40,
                max_depth=4 if fast_null else 10,
                min_samples_leaf=8 if fast_null else 5,
                max_features="sqrt",
                bootstrap=True,
                max_samples=0.75 if fast_null else None,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )
    raise ValueError(name)


def cv_score(model_name: str, x: np.ndarray, y: np.ndarray, scheme: str, groups: np.ndarray | None = None) -> dict[str, float]:
    pred = cv_predictions(model_factory(model_name), x, y, scheme, groups)
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
    }


def null_r2_p95(model_name: str, x: np.ndarray, y: np.ndarray, scheme: str, groups: np.ndarray | None = None) -> float:
    rng = np.random.default_rng(RANDOM_STATE)
    scores = []
    for _ in range(N_PERM):
        y_perm = rng.permutation(y)
        pred = cv_predictions(model_factory(model_name, fast_null=True), x, y_perm, scheme, groups)
        scores.append(r2_score(y_perm, pred))
    return float(np.nanpercentile(scores, 95))


def add_block_assignments(df: pd.DataFrame) -> pd.DataFrame:
    if not BLOCK_PATH.is_file():
        return df
    blocks = pd.read_csv(BLOCK_PATH)
    blocks["basin_id"] = blocks["basin_id"].astype(int)
    return df.merge(blocks.drop(columns=["lat", "lon"], errors="ignore"), on="basin_id", how="left")


def parameter_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("param_") and pd.api.types.is_numeric_dtype(df[c])]


def finite_xy(df: pd.DataFrame, xcols: list[str], ycol: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cols = xcols + [ycol]
    sub = df[cols].copy()
    y = pd.to_numeric(sub[ycol], errors="coerce").to_numpy(dtype=float)
    x = sub[xcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y) & (np.isfinite(x).sum(axis=1) > 0)
    return df.loc[mask].copy(), x[mask], y[mask]


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


def table_text(df: pd.DataFrame, **kwargs) -> str:
    return df.to_string(index=False, **kwargs)


def run_correlations(merged: pd.DataFrame, param_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for alpha in ALPHAS:
        sub = merged[np.isclose(merged["alpha"], alpha)]
        for target in TARGETS:
            for param in param_cols:
                x = pd.to_numeric(sub[param], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(sub[target], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 30 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
                    continue
                spear = stats.spearmanr(x[mask], y[mask])
                pear = stats.pearsonr(x[mask], y[mask])
                rows.append(
                    {
                        "alpha": alpha,
                        "structural_variable": target,
                        "parameter": param,
                        "spearman_rho": float(spear.statistic),
                        "spearman_p": float(spear.pvalue),
                        "pearson_r": float(pear.statistic),
                        "pearson_p": float(pear.pvalue),
                        "n": int(mask.sum()),
                    }
                )
    corr = pd.DataFrame(rows)
    summary_rows = []
    for (alpha, target), group in corr.groupby(["alpha", "structural_variable"]):
        absrho = group["spearman_rho"].abs()
        summary_rows.append(
            {
                "alpha": alpha,
                "structural_variable": target,
                "max_abs_spearman": float(absrho.max()),
                "median_abs_spearman": float(absrho.median()),
                "n_params_abs_spearman_gt_0p5": int((absrho > 0.5).sum()),
                "n_params_abs_spearman_gt_0p7": int((absrho > 0.7).sum()),
                "top_parameter": group.loc[absrho.idxmax(), "parameter"],
            }
        )
    return corr, pd.DataFrame(summary_rows)


def run_predictability(merged: pd.DataFrame, param_cols: list[str]) -> pd.DataFrame:
    rows = []
    cv_specs = [("random5", None)]
    if "spatial_block_k5" in merged.columns:
        cv_specs.append(("spatial_block_k5", "spatial_block_k5"))
    if "hydroclimatic_block" in merged.columns:
        cv_specs.append(("hydroclimatic_block", "hydroclimatic_block"))
    for alpha in ALPHAS:
        sub_alpha = merged[np.isclose(merged["alpha"], alpha)].copy()
        for target in TARGETS:
            print(f"parameter-to-structure alpha={alpha:g} target={target}", flush=True)
            sub, x, y = finite_xy(sub_alpha, param_cols, target)
            for model_name in ["ridge", "rf"]:
                for scheme, group_col in cv_specs:
                    groups = None if group_col is None else sub[group_col].to_numpy()
                    if groups is not None and len(np.unique(groups[~pd.isna(groups)])) < 2:
                        continue
                    score = cv_score(model_name, x, y, scheme, groups)
                    null_p95 = null_r2_p95(model_name, x, y, scheme, groups) if scheme == "random5" else np.nan
                    rows.append(
                        {
                            "alpha": alpha,
                            "target": target,
                            "model": model_name,
                            "cv_scheme": scheme,
                            "n": len(y),
                            "cv_r2": score["r2"],
                            "rmse": score["rmse"],
                            "null_shuffled_target_r2_95th": null_p95,
                        }
                    )
    return pd.DataFrame(rows)


def run_pca(merged: pd.DataFrame, param_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub = merged[np.isclose(merged["alpha"], PRIMARY_ALPHA)].copy()
    _, x, _ = finite_xy(sub, param_cols, TARGETS[0])
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    xz = pipe.fit_transform(x)
    pca = PCA(random_state=RANDOM_STATE)
    scores = pca.fit_transform(xz)
    explained = pd.DataFrame(
        {
            "pc": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    rows = []
    for target in TARGETS:
        _, _, y = finite_xy(sub, param_cols, target)
        for threshold in [0.8, 0.9, 0.95]:
            n_pc = int(np.searchsorted(explained["cumulative_variance_ratio"].to_numpy(), threshold) + 1)
            pred = cv_predictions(model_factory("ridge"), scores[:, :n_pc], y, "random5")
            rows.append(
                {
                    "alpha": PRIMARY_ALPHA,
                    "target": target,
                    "variance_threshold": threshold,
                    "n_pcs": n_pc,
                    "cv_r2": float(r2_score(y, pred)),
                    "rmse": float(math.sqrt(mean_squared_error(y, pred))),
                }
            )
    loadings = pd.DataFrame(pca.components_.T[:, : min(10, len(param_cols))], index=param_cols)
    loadings.columns = [f"PC{i}" for i in range(1, loadings.shape[1] + 1)]
    score_df = sub[["alpha", "basin_id", "station_index", *TARGETS]].copy()
    score_df["PC1"] = scores[:, 0]
    score_df["PC2"] = scores[:, 1]
    return explained, pd.DataFrame(rows), score_df, loadings


def residual_attribute_correlations(merged: pd.DataFrame, param_cols: list[str]) -> pd.DataFrame:
    sub_alpha = merged[np.isclose(merged["alpha"], PRIMARY_ALPHA)].copy()
    rows = []
    for target, attrs in RESIDUAL_ATTRS.items():
        sub, x, y = finite_xy(sub_alpha, param_cols, target)
        pred = cv_predictions(model_factory("ridge"), x, y, "random5")
        residual = y - pred
        sub = sub.copy()
        sub["residual_share"] = residual
        for attr in attrs:
            if attr not in sub.columns:
                continue
            a = pd.to_numeric(sub[attr], errors="coerce").to_numpy(dtype=float)
            mask_orig = np.isfinite(a) & np.isfinite(y)
            mask_resid = np.isfinite(a) & np.isfinite(residual)
            if mask_orig.sum() < 30 or np.nanstd(a[mask_orig]) == 0:
                continue
            orig = stats.spearmanr(a[mask_orig], y[mask_orig])
            resid = stats.spearmanr(a[mask_resid], residual[mask_resid])
            rows.append(
                {
                    "alpha": PRIMARY_ALPHA,
                    "target": target,
                    "attribute": attr,
                    "original_spearman_rho": float(orig.statistic),
                    "original_p": float(orig.pvalue),
                    "residual_spearman_rho": float(resid.statistic),
                    "residual_p": float(resid.pvalue),
                    "n": int(mask_resid.sum()),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["original_fdr_p"] = fdr_bh(out["original_p"].tolist())
        out["residual_fdr_p"] = fdr_bh(out["residual_p"].tolist())
    return out


def incremental_information(merged: pd.DataFrame, param_cols: list[str]) -> pd.DataFrame:
    sub_alpha = merged[np.isclose(merged["alpha"], PRIMARY_ALPHA)].copy()
    rows = []
    feature_sets = {
        "A_parameters_only": param_cols,
        "B_structural_shares_only": SHARES,
        "C_parameters_plus_shares": param_cols + SHARES,
        "D_parameters_plus_shares_plus_sum_weight": param_cols + SHARES + ["sum_weight"],
    }
    rng = np.random.default_rng(RANDOM_STATE)
    for target in INCREMENTAL_TARGETS:
        if target not in sub_alpha.columns:
            continue
        print(f"incremental target={target}", flush=True)
        real_scores: dict[tuple[str, str], float] = {}
        for model_name in ["ridge", "rf"]:
            for set_name, cols in feature_sets.items():
                sub, x, y = finite_xy(sub_alpha, cols, target)
                score = cv_score(model_name, x, y, "random5")
                real_scores[(model_name, set_name)] = score["r2"]
                rows.append(
                    {
                        "alpha": PRIMARY_ALPHA,
                        "attribute_target": target,
                        "model": model_name,
                        "feature_set": set_name,
                        "cv_scheme": "random5",
                        "r2": score["r2"],
                        "rmse": score["rmse"],
                        "delta_r2_vs_parameters_only": score["r2"] - real_scores.get((model_name, "A_parameters_only"), np.nan),
                        "shuffled_share_delta_r2_95th": np.nan,
                        "n": len(y),
                    }
                )
            base_r2 = real_scores[(model_name, "A_parameters_only")]
            for set_name, cols in {
                "C_parameters_plus_shares": param_cols + SHARES,
                "D_parameters_plus_shares_plus_sum_weight": param_cols + SHARES + ["sum_weight"],
            }.items():
                null_delta = []
                for _ in range(N_PERM):
                    shuffled = sub_alpha.copy()
                    for col in SHARES + (["sum_weight"] if "sum_weight" in cols else []):
                        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
                    sub, x, y = finite_xy(shuffled, cols, target)
                    pred = cv_predictions(model_factory(model_name, fast_null=True), x, y, "random5")
                    null_delta.append(r2_score(y, pred) - base_r2)
                rows.append(
                    {
                        "alpha": PRIMARY_ALPHA,
                        "attribute_target": target,
                        "model": model_name,
                        "feature_set": set_name + "_shuffled_share_null",
                        "cv_scheme": "random5",
                        "r2": np.nan,
                        "rmse": np.nan,
                        "delta_r2_vs_parameters_only": np.nan,
                        "shuffled_share_delta_r2_95th": float(np.nanpercentile(null_delta, 95)),
                        "n": int(len(sub_alpha)),
                    }
                )
    return pd.DataFrame(rows)


def classify_redundancy(
    corr_summary: pd.DataFrame,
    pred: pd.DataFrame,
    residual: pd.DataFrame,
    incremental: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    pred_primary = pred[
        np.isclose(pred["alpha"], PRIMARY_ALPHA)
        & (pred["cv_scheme"] == "random5")
        & (pred["model"] == "ridge")
    ]
    corr_primary = corr_summary[np.isclose(corr_summary["alpha"], PRIMARY_ALPHA)]
    inc_real = incremental[
        incremental["feature_set"].isin(["C_parameters_plus_shares", "D_parameters_plus_shares_plus_sum_weight"])
        & (incremental["model"] == "ridge")
    ]
    max_delta = float(inc_real["delta_r2_vs_parameters_only"].max()) if not inc_real.empty else np.nan
    for target in TARGETS:
        r2 = pred_primary.loc[pred_primary["target"] == target, "cv_r2"]
        r2_val = float(r2.iloc[0]) if len(r2) else np.nan
        maxrho = corr_primary.loc[corr_primary["structural_variable"] == target, "max_abs_spearman"]
        maxrho_val = float(maxrho.iloc[0]) if len(maxrho) else np.nan
        resid_sig = residual[
            (residual["target"] == target) & (residual["residual_fdr_p"] < 0.1) & (residual["residual_spearman_rho"].abs() >= 0.15)
        ]
        if r2_val > 0.75 and maxrho_val > 0.8 and resid_sig.empty:
            cls = "strongly_redundant"
        elif 0.4 <= r2_val <= 0.75:
            cls = "partly_redundant"
        elif r2_val < 0.4 and (not resid_sig.empty or (np.isfinite(max_delta) and max_delta > 0.05)):
            cls = "non_redundant_added_structural_information"
        elif r2_val < 0.4:
            cls = "weakly_coupled_but_added_information_not_established"
        else:
            cls = "mixed_or_inconclusive"
        rows.append(
            {
                "alpha": PRIMARY_ALPHA,
                "structural_variable": target,
                "ridge_random5_parameter_to_structure_r2": r2_val,
                "max_abs_spearman_with_parameter": maxrho_val,
                "n_residual_attribute_signals_fdr10_absrho15": int(len(resid_sig)),
                "max_incremental_delta_r2_any_attribute_ridge": max_delta,
                "classification": cls,
            }
        )
    return pd.DataFrame(rows)


def plot_corr_heatmap(corr: pd.DataFrame, alpha: float, path: Path) -> None:
    sub = corr[np.isclose(corr["alpha"], alpha)]
    mat = sub.pivot(index="structural_variable", columns="parameter", values="spearman_rho").reindex(TARGETS)
    fig, ax = plt.subplots(figsize=(14, 4.8))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=6)
    ax.set_title(f"Structure-parameter Spearman correlations, alpha={alpha:g}")
    fig.colorbar(im, ax=ax, label="rho")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_maxcorr(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for target in TARGETS:
        sub = summary[summary["structural_variable"] == target].sort_values("alpha")
        ax.plot(sub["alpha"], sub["max_abs_spearman"], marker="o", label=target)
    ax.set_xlabel("alpha")
    ax.set_ylabel("max absolute Spearman rho")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_predictability(pred: pd.DataFrame, path: Path) -> None:
    sub = pred[np.isclose(pred["alpha"], PRIMARY_ALPHA) & (pred["cv_scheme"] == "random5")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(TARGETS))
    width = 0.35
    for i, model_name in enumerate(["ridge", "rf"]):
        vals = [sub[(sub["target"] == t) & (sub["model"] == model_name)]["cv_r2"].mean() for t in TARGETS]
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=model_name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGETS, rotation=25, ha="right")
    ax.set_ylabel("CV R2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pca_scores(score_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, target in zip(axes, TARGETS):
        sc = ax.scatter(score_df["PC1"], score_df["PC2"], c=score_df[target], s=18, cmap="viridis")
        ax.set_title(target)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    axes[-1].axis("off")
    for ax in axes[:5]:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_loadings(loadings: pd.DataFrame, path: Path) -> None:
    pcs = [c for c in ["PC1", "PC2", "PC3"] if c in loadings.columns]
    records = []
    for pc in pcs:
        vals = loadings[pc].abs().sort_values(ascending=False).head(10)
        for param, val in vals.items():
            records.append({"pc": pc, "parameter": param, "abs_loading": val})
    df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, len(pcs), figsize=(5 * len(pcs), 4.5))
    if len(pcs) == 1:
        axes = [axes]
    for ax, pc in zip(axes, pcs):
        sub = df[df["pc"] == pc].sort_values("abs_loading")
        ax.barh(sub["parameter"], sub["abs_loading"])
        ax.set_title(pc)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_residuals(resid: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(resid["original_spearman_rho"], resid["residual_spearman_rho"], s=35)
    lim = max(0.1, float(np.nanmax(np.abs(resid[["original_spearman_rho", "residual_spearman_rho"]].to_numpy()))))
    ax.plot([-lim, lim], [-lim, lim], color="gray", linestyle="--", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Original Spearman rho")
    ax.set_ylabel("Residual Spearman rho")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_original_vs_residual_attribute_corr.png", dpi=180)
    plt.close(fig)

    mat = resid.pivot(index="target", columns="attribute", values="residual_spearman_rho")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-0.6, vmax=0.6)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, label="residual rho")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_residual_share_attribute_heatmap_alpha_0.01.png", dpi=180)
    plt.close(fig)


def plot_incremental(inc: pd.DataFrame) -> None:
    real = inc[inc["feature_set"].isin(["A_parameters_only", "C_parameters_plus_shares", "D_parameters_plus_shares_plus_sum_weight"])]
    sub = real[real["model"] == "ridge"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(INCREMENTAL_TARGETS))
    width = 0.25
    for i, fs in enumerate(["A_parameters_only", "C_parameters_plus_shares", "D_parameters_plus_shares_plus_sum_weight"]):
        vals = [
            sub[(sub["attribute_target"] == t) & (sub["feature_set"] == fs)]["r2"].mean()
            for t in INCREMENTAL_TARGETS
        ]
        ax.bar(x + (i - 1) * width, vals, width=width, label=fs.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(INCREMENTAL_TARGETS, rotation=35, ha="right")
    ax.set_ylabel("CV R2")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_incremental_r2_by_attribute.png", dpi=180)
    plt.close(fig)

    real_delta = inc[inc["feature_set"] == "C_parameters_plus_shares"].copy()
    null = inc[inc["feature_set"] == "C_parameters_plus_shares_shuffled_share_null"][
        ["attribute_target", "model", "shuffled_share_delta_r2_95th"]
    ]
    comp = real_delta.merge(
        null,
        on=["attribute_target", "model"],
        how="left",
        suffixes=("", "_null"),
    )
    if "shuffled_share_delta_r2_95th" not in comp and "shuffled_share_delta_r2_95th_null" in comp:
        comp["shuffled_share_delta_r2_95th"] = comp["shuffled_share_delta_r2_95th_null"]
    comp = comp[comp["model"] == "ridge"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(comp))
    ax.bar(x - 0.18, comp["delta_r2_vs_parameters_only"], width=0.36, label="real shares")
    ax.bar(x + 0.18, comp["shuffled_share_delta_r2_95th"], width=0.36, label="shuffled 95th")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(comp["attribute_target"], rotation=35, ha="right")
    ax.set_ylabel("Delta R2 vs parameters only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_real_vs_shuffled_incremental_r2.png", dpi=180)
    plt.close(fig)


def write_missing_report(inventory: pd.DataFrame) -> None:
    lines = [
        "# Parameter Missing Report",
        "",
        "No basin-level conventional learned hydrological parameter outputs could be found or reconstructed for the requested alphas.",
        "",
        "Section 3.4 requires one row per basin and alpha with the continuous Flex-MOPEX parameter outputs from the mapper:",
        "- 12 MOPEX parameters for each multiplicative component (`Sb1`, `tw`, `tu`, `Se`, `tc`, `ddf`, `tcrit`, `Sb2`, `alpha`, `is_time`, `tmin`, `tmax`);",
        "- routing parameters (`rout_a`, `rout_b`);",
        "- basin identifier matching `gauge_id`, `basin_id`, or `station_index`;",
        "- alpha and seed/run metadata.",
        "",
        "`model_outputs.npz` files containing only streamflow and structural weights are not sufficient for this analysis.",
    ]
    if not inventory.empty:
        lines.extend(["", "## Inventory", "", table_text(inventory)])
    (OUT_DIR / "parameter_missing_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(
    inventory: pd.DataFrame,
    diagnostics: pd.DataFrame,
    corr_summary: pd.DataFrame,
    pred: pd.DataFrame,
    resid: pd.DataFrame,
    inc: pd.DataFrame,
    classification: pd.DataFrame,
) -> None:
    primary_corr = corr_summary[np.isclose(corr_summary["alpha"], PRIMARY_ALPHA)]
    primary_pred = pred[np.isclose(pred["alpha"], PRIMARY_ALPHA) & (pred["cv_scheme"] == "random5")]
    residual_hits = resid[(resid["residual_fdr_p"] < 0.1) & (resid["residual_spearman_rho"].abs() >= 0.15)]
    inc_real = inc[inc["feature_set"].isin(["C_parameters_plus_shares", "D_parameters_plus_shares_plus_sum_weight"])]

    lines = [
        "# Parameter Decoupling Report",
        "",
        "## A. Parameter discovery and merge",
        "",
        "Conventional Flex-MOPEX continuous parameter outputs were reconstructed from the trained V1 mapper checkpoints, using each run's saved normalization statistics and the v2 merged basin attributes. The reconstruction is explicitly tracked in `parameter_file_inventory.csv` as `reconstructed_from_trained_mapper_checkpoint`.",
        "",
        f"- Alphas analyzed: {', '.join(format_alpha(a) for a in sorted(inventory['alpha'].dropna().unique()))}",
        f"- Parameter columns used: {int(diagnostics['n_parameter_columns'].max()) if not diagnostics.empty else 0}",
        f"- Maximum reconstructed-vs-saved structural weight difference across diagnostics: {diagnostics.filter(like='max_abs_diff_').max().max():.3g}",
        "",
        "## B. Structure-parameter correlations",
        "",
        "At alpha=0.01, maximum absolute Spearman correlations with individual continuous parameters were:",
        "",
        table_text(
            primary_corr[["structural_variable", "max_abs_spearman", "median_abs_spearman", "top_parameter"]]
            .sort_values("structural_variable")
            .round(3)
        ),
        "",
        "## C. Parameter-to-structure predictability",
        "",
        "Random 5-fold CV R2 at alpha=0.01:",
        "",
        primary_pred.pivot_table(index="target", columns="model", values="cv_r2").round(3).to_string(),
        "",
        "High values indicate coupling with the conventional parameter manifold; low or unstable values indicate decoupling.",
        "",
        "## D. Residual hydrological coherence",
        "",
        f"After removing Ridge-predicted parameter effects, {len(residual_hits)} target-attribute residual correlations remain at FDR < 0.1 and |rho| >= 0.15.",
    ]
    if not residual_hits.empty:
        lines.extend(["", table_text(residual_hits[["target", "attribute", "residual_spearman_rho", "residual_fdr_p"]].round(3))])
    lines.extend(
        [
            "",
            "## E. Incremental basin-attribute information",
            "",
            "Ridge incremental R2 from adding structural shares to parameters at alpha=0.01:",
            "",
            table_text(
                inc_real[inc_real["model"] == "ridge"][["attribute_target", "feature_set", "delta_r2_vs_parameters_only"]]
                .round(3)
            ),
            "",
            "## F. Redundancy classification",
            "",
            table_text(classification.round(3)),
            "",
            "## G. Section 3.4 interpretation",
            "",
            "The evidence supports mostly parameter redundancy for the strongest structural quantities, with limited added-information evidence. `share_snow`, `share_sub`, and `sum_weight` are highly predictable from conventional parameters and their physically relevant attribute correlations largely disappear after removing parameter effects. `share_phen` is partly redundant. `share_int` is less predictable from parameters, but because its residual attribute coherence and incremental predictive value are weak, it should be treated as weakly coupled rather than as strong added structural information.",
            "",
            "## H. Recommended paper claim",
            "",
            "Use cautious wording: in this Flex-MOPEX run, learned structural shares are substantially coupled with the conventional parameter manifold. The analysis does not support a strong identifiability claim. A defensible paper claim is that structural weights remain useful as interpretable summaries of learned model organization, but the strongest hydrologically coherent shares are largely encoded by continuous parameters in the current mapper. Any claim of complementary information should be limited to weak or partial evidence, especially for `share_phen` and `share_int`, and should avoid saying the structural shares are fully independent of ordinary parameters.",
        ]
    )
    (OUT_DIR / "parameter_decoupling_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    structure = pd.read_csv(STRUCTURE_PATH)
    structure["basin_id"] = structure["basin_id"].astype(int)
    structure["gauge_id"] = structure["gauge_id"].astype(int)
    structure = add_block_assignments(structure)

    params, inventory, diagnostics = load_reconstructed_params(structure)
    inventory.to_csv(OUT_DIR / "parameter_file_inventory.csv", index=False)
    diagnostics.to_csv(OUT_DIR / "parameter_merge_diagnostics.csv", index=False)

    if params.empty:
        write_missing_report(inventory)
        print("No conventional parameter data found. Wrote missing report.")
        return

    drop_recon = [c for c in params.columns if c.startswith("reconstructed_w_")]
    param_cols = [c for c in params.columns if c.startswith("param_")]
    merged = structure.merge(
        params.drop(columns=drop_recon),
        on=["alpha", "station_index", "basin_id", "gauge_id"],
        how="inner",
        validate="one_to_one",
    )
    merged.to_csv(OUT_DIR / "merged_structure_parameters_attributes.csv", index=False)

    corr, corr_summary = run_correlations(merged, param_cols)
    corr.to_csv(OUT_DIR / "structure_parameter_correlations.csv", index=False)
    corr_summary.to_csv(OUT_DIR / "structure_parameter_correlation_summary.csv", index=False)
    plot_corr_heatmap(corr, PRIMARY_ALPHA, OUT_DIR / "fig_structure_parameter_corr_heatmap_alpha_0.01.png")
    plot_maxcorr(corr_summary, OUT_DIR / "fig_structure_parameter_maxcorr_by_alpha.png")

    pred = run_predictability(merged, param_cols)
    pred.to_csv(OUT_DIR / "parameter_to_structure_predictability.csv", index=False)
    plot_predictability(pred, OUT_DIR / "fig_parameter_to_structure_r2_alpha_0.01.png")

    explained, pc_r2, score_df, loadings = run_pca(merged, param_cols)
    explained.to_csv(OUT_DIR / "parameter_pca_explained_variance.csv", index=False)
    pc_r2.to_csv(OUT_DIR / "structure_from_parameter_pcs.csv", index=False)
    plot_pca_scores(score_df, OUT_DIR / "fig_parameter_pca_colored_by_shares.png")
    plot_loadings(loadings, OUT_DIR / "fig_parameter_pca_loadings_top.png")

    resid = residual_attribute_correlations(merged, param_cols)
    resid.to_csv(OUT_DIR / "residual_share_attribute_correlations.csv", index=False)
    plot_residuals(resid)

    inc = incremental_information(merged, param_cols)
    inc.to_csv(OUT_DIR / "incremental_information_test.csv", index=False)
    plot_incremental(inc)

    classification = classify_redundancy(corr_summary, pred, resid, inc)
    classification.to_csv(OUT_DIR / "structure_redundancy_classification.csv", index=False)

    write_report(inventory, diagnostics, corr_summary, pred, resid, inc, classification)
    print(f"Wrote Section 3.4 parameter decoupling outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
