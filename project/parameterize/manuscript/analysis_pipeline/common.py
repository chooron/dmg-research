from __future__ import annotations

import itertools
import ast
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, mannwhitneyu, pearsonr, spearmanr, wilcoxon


ROOT = Path("/workspace/autoresearch")
PARAM_ROOT = ROOT / "project" / "parameterize"
OUTPUTS_ROOT = PARAM_ROOT / "outputs"
ANALYSIS_ROOT = PARAM_ROOT / "manuscript" / "analysis"
STABILITY_ROOT = OUTPUTS_ROOT / "analysis" / "stability_stats"
DATA_ROOT = ROOT / "data"

BLOCKS = [
    "00_data_inventory",
    "01_model_consistency",
    "02_seed_loss_sensitivity",
    "03_distributional_parameter_spatial_data",
    "04_mean_attribute_relationships",
    "05_environmental_gradient_groups",
    "06_uncertainty_spatial_data",
    "07_uncertainty_attribute_relationships",
    "08_representative_basin_groups",
    "09_integrated_summary",
]

MODEL_LABELS = {
    "deterministic": "delta_base",
    "mc_dropout": "delta_mcd",
    "distributional": "delta_dist",
}
MODEL_LATEX = {
    "deterministic": r"$\delta_{base}$",
    "mc_dropout": r"$\delta_{mcd}$",
    "distributional": r"$\delta_{dist}$",
}
PRIMARY_LOSS = "HybridNseBatchLoss"
PARAMETERS = [
    "parBETA",
    "parFC",
    "parK0",
    "parK1",
    "parK2",
    "parLP",
    "parPERC",
    "parUZL",
    "parTT",
    "parCFMAX",
    "parCFR",
    "parCWH",
    "route_a",
    "route_b",
]
PARAMETER_FAMILY = {
    "parBETA": "soil_moisture",
    "parFC": "soil_moisture",
    "parLP": "soil_moisture",
    "parPERC": "groundwater_exchange",
    "parUZL": "runoff_generation",
    "parK0": "runoff_recession",
    "parK1": "runoff_recession",
    "parK2": "runoff_recession",
    "parTT": "snow",
    "parCFMAX": "snow",
    "parCFR": "snow",
    "parCWH": "snow",
    "route_a": "routing",
    "route_b": "routing",
}
PARAMETER_BOUNDS = {
    "parBETA": (1.0, 6.0),
    "parFC": (50.0, 1000.0),
    "parK0": (0.05, 0.9),
    "parK1": (0.01, 0.5),
    "parK2": (0.001, 0.2),
    "parLP": (0.2, 1.0),
    "parPERC": (0.0, 10.0),
    "parUZL": (0.0, 100.0),
    "parTT": (-2.5, 2.5),
    "parCFMAX": (0.5, 10.0),
    "parCFR": (0.0, 0.1),
    "parCWH": (0.0, 0.2),
    "route_a": (0.0, 2.9),
    "route_b": (0.0, 6.5),
}
REQUIRED_ATTRIBUTES = [
    "aridity",
    "frac_snow",
    "p_seasonality",
    "slope_mean",
    "elev_mean",
    "pet_mean",
    "p_mean",
    "soil_conductivity",
    "soil_depth_pelletier",
    "clay_frac",
    "sand_frac",
    "geol_porosity",
    "lai_diff",
    "gvf_diff",
    "high_prec_dur",
    "low_prec_dur",
    "high_prec_freq",
    "low_prec_freq",
]
CORE_GRADIENTS = ["aridity", "frac_snow", "slope_mean", "pet_mean", "soil_conductivity", "p_seasonality"]
REPRESENTATIVE_GRADIENTS = ["aridity", "frac_snow", "slope_mean", "soil_conductivity", "pet_mean"]
FOCUSED_PAIRS = [
    ("parBETA", "slope_mean"),
    ("parFC", "pet_mean"),
    ("parPERC", "aridity"),
    ("parUZL", "soil_conductivity"),
    ("parCFR", "frac_snow"),
    ("parCFR", "elev_mean"),
    ("parCWH", "frac_snow"),
    ("route_a", "slope_mean"),
    ("parK1", "lai_diff"),
]
KEY_MEAN_PAIRS = [
    ("parBETA", "slope_mean"),
    ("parBETA", "soil_depth_pelletier"),
    ("parFC", "pet_mean"),
    ("parPERC", "aridity"),
    ("parPERC", "slope_mean"),
    ("parUZL", "soil_conductivity"),
    ("parUZL", "slope_mean"),
    ("parCFR", "frac_snow"),
    ("parCWH", "frac_snow"),
    ("route_a", "slope_mean"),
]
KEY_GRADIENT_PAIRS = [
    ("aridity", "parPERC"),
    ("aridity", "parFC"),
    ("frac_snow", "parCWH"),
    ("frac_snow", "parCFR"),
    ("slope_mean", "parBETA"),
    ("slope_mean", "parUZL"),
    ("pet_mean", "parFC"),
    ("soil_conductivity", "parUZL"),
]
KEY_STD_PAIRS = [
    ("parCWH", "frac_snow"),
    ("parPERC", "aridity"),
    ("parUZL", "soil_conductivity"),
    ("parUZL", "slope_mean"),
    ("parBETA", "slope_mean"),
]


@dataclass
class PipelineLog:
    path: Path
    start_time: datetime = field(default_factory=datetime.now)
    entries: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.entries.append(f"[{datetime.now().isoformat(timespec='seconds')}] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        self.add(f"WARNING: {message}")

    def record_output(self, path: Path) -> None:
        self.outputs.append(path)

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "Back-half manuscript analysis pipeline log",
            f"Started: {self.start_time.isoformat(timespec='seconds')}",
            "",
            "Entries:",
            *self.entries,
            "",
            "Warnings:",
            *(self.warnings or ["None"]),
            "",
            "Output files:",
            *[str(path) for path in self.outputs],
            "",
            f"Completed: {datetime.now().isoformat(timespec='seconds')}",
        ]
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_block_dirs() -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    for block in BLOCKS:
        result[block] = {}
        for sub in ("data", "reports", "methods", "logs"):
            path = ANALYSIS_ROOT / block / sub
            path.mkdir(parents=True, exist_ok=True)
            result[block][sub] = path
    (ANALYSIS_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    return result


def save_csv(df: pd.DataFrame, path: Path, log: PipelineLog | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if log:
        log.record_output(path)
    return path


def write_md(path: Path, title: str, sections: dict[str, str | list[str]], log: PipelineLog | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for heading, content in sections.items():
        lines.extend([f"## {heading}", ""])
        if isinstance(content, list):
            if content:
                lines.extend([f"- {item}" for item in content])
            else:
                lines.append("- None")
        else:
            lines.append(content if content else "None")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    if log:
        log.record_output(path)
    return path


def discover_runs() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_dir in sorted(OUTPUTS_ROOT.iterdir()):
        if not model_dir.is_dir() or "531" not in model_dir.name or model_dir.name == "analysis":
            continue
        model_raw = model_dir.name.replace("-531", "")
        for loss_dir in sorted(model_dir.iterdir()):
            if not loss_dir.is_dir():
                continue
            for seed_dir in sorted(loss_dir.glob("seed_*")):
                if not seed_dir.is_dir():
                    continue
                seed_match = re.search(r"seed_(\d+)", seed_dir.name)
                if not seed_match:
                    continue
                seed = int(seed_match.group(1))
                result_path = seed_dir / f"results_seed{seed}.csv"
                model_files = sorted((seed_dir / "model").glob("model_epoch*.pt"))
                latest_checkpoint = model_files[-1] if model_files else None
                rows.append(
                    {
                        "run_id": f"{model_raw}__{loss_dir.name}__seed_{seed}",
                        "model_raw": model_raw,
                        "model_label": MODEL_LABELS.get(model_raw, model_raw),
                        "loss": loss_dir.name,
                        "seed": seed,
                        "run_dir": str(seed_dir),
                        "results_file": str(result_path) if result_path.exists() else "",
                        "metrics_file": str(seed_dir / "metrics_avg.json") if (seed_dir / "metrics_avg.json").exists() else "",
                        "checkpoint_file": str(latest_checkpoint) if latest_checkpoint else "",
                        "has_results": result_path.exists(),
                        "has_checkpoint": latest_checkpoint is not None,
                    }
                )
    return pd.DataFrame(rows)


def read_basin_ids() -> pd.Series:
    path = DATA_ROOT / "531sub_id.txt"
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        values = ast.literal_eval(raw)
        return pd.Series(values, dtype="object").astype(str).str.strip()
    frame = pd.read_csv(path, header=None, dtype=str)
    return pd.Series(frame.to_numpy().ravel()).dropna().astype(str).str.strip()


def modelize(df: pd.DataFrame, model_col: str = "model_raw") -> pd.DataFrame:
    out = df.copy()
    if model_col not in out.columns and "model" in out.columns:
        out = out.rename(columns={"model": model_col})
    out["model_label"] = out[model_col].map(MODEL_LABELS).fillna(out[model_col])
    return out


def load_attributes(log: PipelineLog | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = STABILITY_ROOT / "tables" / "basin_attributes.csv"
    attrs = pd.read_csv(path, dtype={"basin_id": str})
    attrs["basin_id"] = attrs["basin_id"].astype(str).str.strip()
    mapping_rows = []
    available = set(attrs.columns)
    for requested in REQUIRED_ATTRIBUTES:
        if requested in available:
            mapped = requested
            status = "exact"
        else:
            candidates = sorted(available, key=lambda col: _simple_similarity(requested, col), reverse=True)
            mapped = candidates[0] if candidates else ""
            status = "fuzzy" if mapped else "missing"
            if log:
                log.warn(f"Attribute {requested} mapped by fuzzy match to {mapped}")
        mapping_rows.append({"requested_attribute": requested, "mapped_attribute": mapped, "mapping_status": status})
    return attrs, pd.DataFrame(mapping_rows)


def _simple_similarity(left: str, right: str) -> float:
    left_parts = set(left.lower().split("_"))
    right_parts = set(right.lower().split("_"))
    overlap = len(left_parts & right_parts)
    return overlap / max(len(left_parts | right_parts), 1)


def load_coordinates(log: PipelineLog | None = None) -> pd.DataFrame:
    try:
        import geopandas as gpd

        shp = DATA_ROOT / "camels_loc" / "camels_671_loc.shp"
        coords = gpd.read_file(shp)[["gage_id", "lat", "lon"]].copy()
        coords["basin_id"] = coords["gage_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        coords = coords.rename(columns={"lon": "longitude", "lat": "latitude"})
        return coords[["basin_id", "longitude", "latitude"]]
    except Exception as exc:  # pragma: no cover - environment fallback
        if log:
            log.warn(f"Could not read coordinates shapefile: {type(exc).__name__}: {exc}")
        return pd.DataFrame(columns=["basin_id", "longitude", "latitude"])


def load_params(run_inventory: pd.DataFrame, log: PipelineLog | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = STABILITY_ROOT / "tables" / "params_long.csv"
    params = pd.read_csv(path, dtype={"basin_id": str})
    params = params.rename(columns={"model": "model_raw"})
    params["basin_id"] = params["basin_id"].astype(str).str.strip()
    params["run_key"] = (
        params["model_raw"].astype(str)
        + "__"
        + params["loss"].astype(str)
        + "__seed_"
        + params["seed"].astype(int).astype(str)
    )
    valid_keys = set(run_inventory["run_id"])
    params = params.loc[params["run_key"].isin(valid_keys)].copy()
    params = params.loc[params["parameter"].isin(PARAMETERS)].copy()

    key_cols = ["model_raw", "loss", "seed", "basin_id", "parameter"]
    dup = (
        params.groupby(key_cols, as_index=False)
        .agg(
            row_count=("mean", "size"),
            mean_min=("mean", "min"),
            mean_max=("mean", "max"),
            std_min=("std", "min"),
            std_max=("std", "max"),
        )
    )
    dup["mean_conflict"] = (dup["mean_max"] - dup["mean_min"]).abs() > 1e-10
    dup["std_conflict"] = (dup["std_max"] - dup["std_min"]).abs() > 1e-10
    duplicate_diagnostics = dup.loc[dup["row_count"] > 1].copy()
    conflict_count = int((duplicate_diagnostics["mean_conflict"] | duplicate_diagnostics["std_conflict"]).sum())
    if log and not duplicate_diagnostics.empty:
        log.warn(
            f"Collapsed {len(duplicate_diagnostics)} duplicate parameter keys; "
            f"{conflict_count} had non-identical mean/std values."
        )

    collapsed = (
        params.groupby(key_cols + ["parameter_label"], as_index=False, dropna=False)
        .agg(mean=("mean", "mean"), std=("std", "mean"), sample_count=("sample_count", "max"))
    )
    collapsed = modelize(collapsed)
    collapsed["parameter_family"] = collapsed["parameter"].map(PARAMETER_FAMILY)
    collapsed = normalize_parameter_frame(collapsed)
    return collapsed, duplicate_diagnostics


def normalize_parameter_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lows = out["parameter"].map(lambda p: PARAMETER_BOUNDS[p][0])
    highs = out["parameter"].map(lambda p: PARAMETER_BOUNDS[p][1])
    width = highs - lows
    out["parameter_mean_unit"] = (out["mean"] - lows) / width
    out["parameter_std_unit"] = out["std"] / width
    out["parameter_mean_unit"] = out["parameter_mean_unit"].clip(0, 1)
    out["distance_to_boundary"] = np.minimum(out["parameter_mean_unit"], 1.0 - out["parameter_mean_unit"])
    out["near_boundary_flag"] = out["distance_to_boundary"] <= 0.05
    return out


def correlation_value(x: Iterable[float], y: Iterable[float], method: str = "spearman") -> tuple[float, float, int]:
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(frame)
    if n < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan, np.nan, n
    if method == "pearson":
        stat, p = pearsonr(frame["x"], frame["y"])
    elif method == "kendall":
        stat, p = kendalltau(frame["x"], frame["y"])
    else:
        stat, p = spearmanr(frame["x"], frame["y"])
    return float(stat), float(p), n


def fdr_bh(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q
    valid_idx = np.where(valid)[0]
    order = valid_idx[np.argsort(p[valid])]
    ranked = p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0, 1)
    return q


def pairwise_mean_abs_diff(values: Iterable[float]) -> float:
    arr = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if len(arr) < 2:
        return np.nan
    return float(np.mean([abs(a - b) for a, b in itertools.combinations(arr, 2)]))


def sign_consistency(values: Iterable[float]) -> str:
    arr = np.asarray([v for v in values if pd.notna(v) and abs(v) > 1e-12], dtype=float)
    if len(arr) == 0:
        return "no_signal"
    signs = set(np.sign(arr).astype(int).tolist())
    if len(signs) == 1:
        return "consistent_positive" if 1 in signs else "consistent_negative"
    return "sign_flip_present"


def sign_label(value: float) -> str:
    if pd.isna(value) or abs(value) <= 1e-12:
        return "near_zero"
    return "positive" if value > 0 else "negative"


def mann_whitney_test(low: Iterable[float], high: Iterable[float]) -> dict[str, float | str]:
    low_arr = pd.Series(low).dropna().astype(float)
    high_arr = pd.Series(high).dropna().astype(float)
    if len(low_arr) < 3 or len(high_arr) < 3:
        return {"test": "mann_whitney_u", "statistic": np.nan, "p_value": np.nan, "effect_size_rank_biserial": np.nan}
    stat, p = mannwhitneyu(high_arr, low_arr, alternative="two-sided")
    n1, n2 = len(high_arr), len(low_arr)
    rank_biserial = 2 * stat / (n1 * n2) - 1
    return {
        "test": "mann_whitney_u",
        "statistic": float(stat),
        "p_value": float(p),
        "effect_size_rank_biserial": float(rank_biserial),
    }


def wilcoxon_test(x: Iterable[float], y: Iterable[float]) -> dict[str, float | str]:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3:
        return {"test": "wilcoxon_signed_rank", "statistic": np.nan, "p_value": np.nan}
    try:
        stat, p = wilcoxon(frame["x"], frame["y"])
    except ValueError:
        stat, p = np.nan, np.nan
    return {"test": "wilcoxon_signed_rank", "statistic": float(stat) if pd.notna(stat) else np.nan, "p_value": float(p) if pd.notna(p) else np.nan}


def make_tercile_assignments(df: pd.DataFrame, attribute: str) -> pd.DataFrame:
    values = df[["basin_id", attribute]].dropna().drop_duplicates("basin_id").copy()
    values["tercile_rank"] = values[attribute].rank(method="first")
    try:
        values["gradient_group"] = pd.qcut(values["tercile_rank"], q=3, labels=["low", "middle", "high"])
    except ValueError:
        values["gradient_group"] = pd.cut(values["tercile_rank"], bins=3, labels=["low", "middle", "high"])
    values["gradient_attribute"] = attribute
    return values[["basin_id", "gradient_attribute", attribute, "gradient_group"]]


def report_common_sections(objective: str, input_files: list[str], data_filters: list[str], metric_definitions: list[str], main_results: list[str], tables: list[str], caveats: list[str], wording: list[str], figure_usage: list[str]) -> dict[str, list[str]]:
    return {
        "Objective": [objective],
        "Input files": input_files,
        "Data filters": data_filters,
        "Metric definitions": metric_definitions,
        "Main results": main_results,
        "Tables saved": tables,
        "Caveats": caveats,
        "Recommended wording": wording,
        "Suggested later figure usage": figure_usage,
    }


def concise_top(df: pd.DataFrame, columns: list[str], n: int = 8) -> list[str]:
    if df.empty:
        return ["No rows available."]
    rows = []
    for _, row in df.head(n).iterrows():
        rows.append(", ".join(f"{col}={row[col]}" for col in columns if col in row))
    return rows
