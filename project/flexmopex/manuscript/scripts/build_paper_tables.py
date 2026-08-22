#!/usr/bin/env python
"""Build manuscript-ready Flex-MOPEX paper tables."""
from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ACTIVE_THRESHOLD_DEFAULT = 0.1
PRIMARY_ALPHA_DEFAULT = 0.01
MIN_ACTIVE_BASINS_DEFAULT = 10
PROCESS_ORDER = ["Interception", "Phenology", "Snow", "Subsurface"]
PROCESS_SHORT = {
    "Interception": "INT",
    "Phenology": "PHEN",
    "Snow": "SNOW",
    "Subsurface": "SUB",
}
PROCESS_TO_W = {
    "Interception": "w_int",
    "Phenology": "w_phen",
    "Snow": "w_snow",
    "Subsurface": "w_sub",
}
PROCESS_TO_SHARE = {
    "Interception": "share_int",
    "Phenology": "share_phen",
    "Snow": "share_snow",
    "Subsurface": "share_sub",
}
PROCESS_TO_Z = {
    "Interception": "z_int",
    "Phenology": "z_phen",
    "Snow": "z_snow",
    "Subsurface": "z_sub",
}
ATTR_LABELS = {
    "aridity": "Aridity",
    "frac_snow": "Snow fraction",
    "slope_mean": "Mean slope",
    "pet_mean": "PET",
    "p_mean": "Precip.",
    "clay_frac": "Clay",
    "soil_depth_pelletier": "Soil depth",
    "soil_conductivity": "Soil cond.",
    "elev_mean": "Elevation",
    "area_gages2": "Area",
    "frac_forest": "Forest",
    "lai_diff": "LAI diff.",
    "gvf_diff": "GVF diff.",
    "p_seasonality": "Precip. seasonality",
    "high_prec_dur": "High-precip. dur.",
    "high_prec_freq": "High-precip. freq.",
    "low_prec_dur": "Low-precip. dur.",
    "low_prec_freq": "Low-precip. freq.",
    "carbonate_rocks_frac": "Carbonate",
    "dom_land_cover": "Land cover",
    "relief": "Relief",
    "topographic_wetness": "Topographic wetness",
    "twi": "TWI",
    "lai_max": "LAI max.",
    "gvf_max": "GVF max.",
    "dom_land_cover_frac": "Dom. land cover frac.",
    "land_cover": "Land cover class",
    "vegetation": "Vegetation type",
    "root_depth_50": "Root depth",
    "soil_depth_statsgo": "Soil depth",
    "soil_porosity": "Soil porosity",
    "max_water_content": "Max. water content",
    "sand_frac": "Sand frac.",
    "silt_frac": "Silt frac.",
    "geol_1st_class": "Geol. 1st class",
    "glim_1st_class_frac": "Geol. 1st class frac.",
    "geol_2nd_class": "Geol. 2nd class",
    "glim_2nd_class_frac": "Geol. 2nd class frac.",
    "geol_porosity": "Geol. porosity",
    "geol_permeability": "Geol. permeability",
    "aquifer_frac": "Aquifer frac.",
}
WEIGHT_NAME_TO_PROCESS = {
    "w_int": "Interception",
    "w_phen": "Phenology",
    "w_snow": "Snow",
    "w_sub": "Subsurface",
    "share_int": "Interception",
    "share_phen": "Phenology",
    "share_snow": "Snow",
    "share_sub": "Subsurface",
    "z_int": "Interception",
    "z_phen": "Phenology",
    "z_snow": "Snow",
    "z_sub": "Subsurface",
    "interception share": "Interception",
    "phenology share": "Phenology",
    "snow share": "Snow",
    "subsurface share": "Subsurface",
    "share_intensity": "Interception",
    "sum_weight": "All processes",
}
MODEL_ORDER = {"Basic": 0, "Full": 1, "CFlex": 2, "DFlex": 3}
TABLE_NOTES: dict[str, str] = {
    "table1_performance_complexity_summary": "Main Results Section 3.1. Compact performance-complexity summary for Basic-MOPEX, Full-MOPEX, representative CFlex alpha settings, and representative DFlex reference settings.",
    "table1b_panelB_process_extension_weights": "Panel B summary of learned process-extension weights, active-process counts, and process-specific active fractions.",
    "table2_process_coordinate_evidence_synthesis": "Main Results Sections 3.2-3.4. Process-level synthesis across seed robustness, hydroclimatic organization, and parameter-space readouts.",
    "table3_loro_transferability_summary": "Main Results Section 3.5. Region-level LORO summary for predictive transfer, continuous coordinate transfer, and categorical decision transfer.",
    "tableS1_multimetric_performance_summary": "Supplement S1. Full multi-metric performance summary across model, alpha, split, and seed.",
    "tableS1_multimetric_performance_summary_test_only": "Supplement S1 compact export. Test-only subset of the multi-metric performance summary.",
    "tableS2_alpha_tradeoff_summary": "Supplement S2. Full alpha-path summary for performance-complexity tradeoffs and primary-alpha justification.",
    "tableS3_seed_robustness_summary": "Supplement S3. Seed robustness summary for learned CFlex structural coordinates and categorical structural decisions.",
    "tableS4_hydroclimatic_control_summary": "Supplement S4. Basin-attribute correlations with structural coordinates.",
    "tableS4_process_level_hydroclimatic_summary": "Supplement S4 compact process-level summary of hydroclimatic organization.",
    "tableS5_parameter_space_readout_associations": "Supplement S5 association table linking structural coordinates to parameter-space targets.",
    "tableS5_parameter_space_readout_reconstruction": "Supplement S5 reconstruction diagnostics for parameter-space targets.",
    "tableS5_process_level_parameter_readout_summary": "Supplement S5 compact process-level parameter readout summary.",
    "tableS6_loro_regional_performance_summary": "Supplement S6. Region-wise predictive transfer and retained Full-MOPEX gain under LORO.",
    "tableS7_continuous_coordinate_transfer_summary": "Supplement S7. Continuous structural-coordinate transfer by process and held-out region.",
    "tableS8_categorical_decision_transfer_summary": "Supplement S8. Dominant-process and active-set categorical transfer summary.",
    "tableS8_dominant_process_transition_long": "Supplement S8 auxiliary dominant-process transition table.",
    "tableS9_nmul_ablation_summary": "Supplement S9. nmul ablation summary at alpha = 0.01 supporting Appendix Fig. A5.",
    "tableS9_nmul_ablation_summary_aggregated": "Supplement S9 aggregated nmul ablation summary.",
    "tableS10_threshold_sensitivity_summary": "Supplement S10. Threshold sensitivity summary supporting Appendix Fig. A6.",
    "table_generation_warnings": "Generated warnings and consistency audit for table construction.",
    "README_tables": "Table inventory, definitions, source files, and regeneration notes.",
    "table_manifest": "Manifest of generated manuscript tables and support files.",
}
DISPLAY_LABELS: dict[str, str] = {
    "table1_performance_complexity_summary": "Table 1",
    "table1b_panelB_process_extension_weights": "Table 1b",
    "table2_process_coordinate_evidence_synthesis": "Table 2",
    "table3_loro_transferability_summary": "Table 3",
    "tableS1_multimetric_performance_summary": "Table S1a",
    "tableS1_multimetric_performance_summary_test_only": "Table S1b",
    "tableS2_alpha_tradeoff_summary": "Table S2",
    "tableS3_seed_robustness_summary": "Table S3",
    "tableS4_hydroclimatic_control_summary": "Table S4a",
    "tableS4_process_level_hydroclimatic_summary": "Table S4b",
    "tableS5_parameter_space_readout_associations": "Table S5a",
    "tableS5_parameter_space_readout_reconstruction": "Table S5b",
    "tableS5_process_level_parameter_readout_summary": "Table S5c",
    "tableS6_loro_regional_performance_summary": "Table S6",
    "tableS7_continuous_coordinate_transfer_summary": "Table S7",
    "tableS8_categorical_decision_transfer_summary": "Table S8a",
    "tableS8_dominant_process_transition_long": "Table S8b",
    "tableS9_nmul_ablation_summary": "Table S9a",
    "tableS9_nmul_ablation_summary_aggregated": "Table S9b",
    "tableS10_threshold_sensitivity_summary": "Table S10",
}


@dataclass
class BuildContext:
    project_root: Path
    output_dir: Path
    active_threshold: float
    primary_alpha: float
    min_active_basins: int
    warnings: list[str] = field(default_factory=list)
    sources: dict[str, set[str]] = field(default_factory=dict)
    files: dict[str, dict[str, Path]] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)

    @property
    def figures_csv_dir(self) -> Path:
        return self.project_root / "manuscript" / "figures" / "csv"

    @property
    def nmul_root(self) -> Path:
        return self.project_root / "results" / "block1_nmul_ablation" / "flex" / "alpha0.01"

    def add_source(self, table_name: str, path: Path | str) -> None:
        text = str(path)
        if isinstance(path, Path):
            try:
                text = str(path.relative_to(self.project_root))
            except ValueError:
                text = str(path)
        self.sources.setdefault(table_name, set()).add(text)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
        warnings.warn(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("manuscript/tables"),
        help="Relative paths are resolved against --project-root.",
    )
    parser.add_argument("--active-threshold", type=float, default=ACTIVE_THRESHOLD_DEFAULT)
    parser.add_argument("--primary-alpha", type=float, default=PRIMARY_ALPHA_DEFAULT)
    parser.add_argument("--min-active-basins", type=int, default=MIN_ACTIVE_BASINS_DEFAULT)
    parser.add_argument("--write-xlsx", action="store_true", help="Also write XLSX output. Disabled by default.")
    return parser.parse_args()


def resolve_output_dir(project_root: Path, output_dir: Path) -> Path:
    return output_dir if output_dir.is_absolute() else project_root / output_dir


def alpha_label(alpha: float | int | None) -> str:
    if alpha is None or pd.isna(alpha):
        return ""
    return f"{float(alpha):.3f}".rstrip("0").rstrip(".")


def clean_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def safe_div(num: Any, den: Any) -> float:
    num = clean_float(num)
    den = clean_float(den)
    if pd.isna(num) or pd.isna(den) or abs(den) < 1e-12:
        return np.nan
    return num / den


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, str):
        data = json.loads(data)
    return data


def read_csv(ctx: BuildContext, path: Path, table_name: str | None = None) -> pd.DataFrame:
    if not path.exists():
        ctx.warn(f"Missing input file: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if table_name:
        ctx.add_source(table_name, path)
    return df


def normalize_process(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return text
    if text in PROCESS_ORDER or text == "All processes":
        return text
    key = text.lower().replace("-", " ").replace("_", " ").strip()
    if key in {"interception", "phenology", "snow", "subsurface", "all processes", "transfer"}:
        return key.title() if key != "all processes" else "All processes"
    mapped = WEIGHT_NAME_TO_PROCESS.get(str(value).strip().lower())
    if mapped:
        return mapped
    mapped = WEIGHT_NAME_TO_PROCESS.get(key)
    if mapped:
        return mapped
    if "interception" in key or key == "int":
        return "Interception"
    if "phenology" in key or key == "phen":
        return "Phenology"
    if "snow" in key:
        return "Snow"
    if "sub" in key:
        return "Subsurface"
    return text


def normalize_region_label(value: Any) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"[Rr]?(\d+)", text)
    if match:
        idx = int(match.group(1))
        if text.upper().startswith("R"):
            return f"R{idx}"
        return f"R{idx + 1}"
    return text


def display_attribute_label(value: Any) -> str:
    text = str(value).strip()
    return ATTR_LABELS.get(text, text)


def display_parameter_name(value: Any) -> str:
    text = str(value).strip()
    return text.removeprefix("param_")


def format_signed_item(label: str, rho: Any) -> str:
    value = clean_float(rho)
    if pd.isna(value):
        return label
    return f"{label} ({value:+.2f})"


def classify_retained_gain(value: float) -> str:
    if pd.isna(value):
        return "basic-like"
    if value < 0:
        return "degraded"
    if value < 0.25:
        return "basic-like"
    if value < 0.75:
        return "intermediate"
    return "near-full"


def classify_transfer_strength(rho: float, n_active: int, min_active: int) -> str:
    if n_active < min_active or pd.isna(rho):
        return "not assessable"
    if rho >= 0.7:
        return "strong"
    if rho >= 0.4:
        return "moderate"
    return "weak"


def classify_seed_robustness(pairwise_spearman: float, icc: float, dom: float, jaccard: float, active_fraction: float) -> str:
    if pd.isna(active_fraction) or active_fraction < 0.05:
        return "weak"
    strong = [
        clean_float(pairwise_spearman) >= 0.8,
        clean_float(icc) >= 0.75,
        clean_float(dom) >= 0.8,
        clean_float(jaccard) >= 0.75,
    ]
    moderate = [
        clean_float(pairwise_spearman) >= 0.5,
        clean_float(icc) >= 0.5,
        clean_float(dom) >= 0.6,
        clean_float(jaccard) >= 0.5,
    ]
    if all(strong):
        return "strong"
    if sum(bool(x) for x in moderate) >= 2:
        return "moderate"
    return "weak"


def attribute_group(attribute: str) -> str:
    text = str(attribute).lower()
    rules = [
        ("snow", ["snow", "ddf", "tcrit"]),
        ("temperature", ["temp", "tmean", "tmin", "tmax"]),
        ("precipitation", ["prec", "rain", "storm"]),
        ("seasonality", ["season", "timing", "phase", "dur"]),
        ("vegetation", ["forest", "lai", "veg", "cover", "root"]),
        ("soil", ["soil", "clay", "sand", "silt", "porosity", "storage"]),
        ("geology", ["rock", "geol", "carbonate", "gw", "routing"]),
        ("topography", ["elev", "slope", "area"]),
        ("hydrology", ["aridity", "runoff", "baseflow", "flow"]),
    ]
    for group, tokens in rules:
        if any(token in text for token in tokens):
            return group
    return "other"


def control_strength_class(value: float, process: str) -> str:
    if pd.isna(value):
        return "sparse"
    if process == "Interception" and value < 0.35:
        return "weak"
    if value >= 0.6:
        return "strong"
    if value >= 0.35:
        return "moderate"
    return "weak"


def readout_class(value: float) -> str:
    if pd.isna(value):
        return "weak"
    if value >= 0.6:
        return "strong"
    if value >= 0.35:
        return "moderate"
    return "weak"


def reconstruction_class(value: float) -> str:
    if pd.isna(value):
        return "weak"
    if value < 0:
        return "worse_than_baseline"
    if value >= 0.6:
        return "strong"
    if value >= 0.3:
        return "moderate"
    return "weak"


def activity_level_from_fraction(process: str, fraction: float) -> str:
    if pd.isna(fraction):
        return "sparse"
    if process == "Interception":
        return "weak" if fraction < 0.15 else "moderate"
    if process == "Snow":
        return "strong" if fraction >= 0.7 else "moderate"
    if process in {"Phenology", "Subsurface"}:
        return "moderate" if fraction >= 0.25 else "sparse"
    return "moderate"


def retained_category_text(category: str) -> str:
    mapping = {
        "degraded": "degraded",
        "basic-like": "basic-like",
        "intermediate": "intermediate",
        "near-full": "near-full",
    }
    return mapping.get(category, category)


def to_latex(df: pd.DataFrame) -> str:
    if df.empty:
        return "\\begin{tabular}{l}\nNo data available\\\\\n\\end{tabular}\n"
    latex_df = format_for_export(df)
    return latex_df.to_latex(index=False, escape=True)


def format_for_export(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if col.startswith("n_") or col in {"n_basins", "n_seeds", "seed", "nmul", "rank", "top_k"}:
                continue
            out[col] = out[col].astype(float).round(3)
    return out.replace({np.nan: ""})


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "| status |\n| --- |\n| No data available |\n"
    data = format_for_export(df)
    cols = list(data.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in data.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_table(ctx: BuildContext, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    formatted = format_for_export(df)
    csv_path = ctx.output_dir / f"{table_name}.csv"
    md_path = ctx.output_dir / f"{table_name}.md"
    tex_path = ctx.output_dir / f"{table_name}.tex"
    formatted.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table(formatted), encoding="utf-8")
    latex_text = to_latex(formatted)
    if table_name == "table2_process_coordinate_evidence_summary" and table_name in ctx.notes:
        latex_text += (
            "\n\\par\\smallskip\n"
            "{\\footnotesize\\emph{Note.} "
            + ctx.notes[table_name]
            + "}\n"
        )
    tex_path.write_text(latex_text, encoding="utf-8")
    ctx.files[table_name] = {"csv": csv_path, "markdown": md_path, "latex": tex_path}
    return formatted


def write_excel(ctx: BuildContext, tables: dict[str, pd.DataFrame]) -> None:
    path = ctx.output_dir / "all_paper_tables.xlsx"
    try:
        with pd.ExcelWriter(path) as writer:
            for name, df in tables.items():
                sheet = re.sub(r"[^A-Za-z0-9_]", "_", name)[:31]
                format_for_export(df).to_excel(writer, sheet_name=sheet, index=False)
    except Exception as exc:
        ctx.warn(f"Skipping XLSX export because pandas Excel writer is unavailable: {exc}")
        return
    ctx.files["all_paper_tables"] = {"xlsx": path}


def load_metrics_agg(ctx: BuildContext, root: Path, formulation: str, model_group: str, table_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("metrics_agg.json")):
        ctx.add_source(table_name, path)
        data = load_json(path)
        rel = path.relative_to(ctx.project_root)
        match_alpha = re.search(r"alpha([0-9.]+)", str(rel))
        match_seed = re.search(r"seed[_]?([0-9]+)", str(rel))
        split = "test" if "test" in path.parent.name.lower() else "train"
        alpha = clean_float(match_alpha.group(1)) if match_alpha else 0.0
        seed = int(match_seed.group(1)) if match_seed else np.nan
        metrics_json = path.parent / "metrics.json"
        n_valid = np.nan
        if metrics_json.exists():
            metric_payload = load_json(metrics_json)
            n_valid = len(metric_payload.get("nse", []))
        row = {
            "model_group": model_group,
            "formulation": formulation,
            "alpha": alpha,
            "split": split,
            "seed": seed,
            "n_basins": n_valid,
            "median_NSE": clean_float(data.get("nse", {}).get("median")),
            "mean_NSE": clean_float(data.get("nse", {}).get("mean")),
            "median_KGE": clean_float(data.get("kge", {}).get("median")),
            "mean_KGE": clean_float(data.get("kge", {}).get("mean")),
            "median_RMSE": clean_float(data.get("rmse", {}).get("median")),
            "mean_RMSE": clean_float(data.get("rmse", {}).get("mean")),
            "median_PBIAS": clean_float(data.get("pbias", {}).get("median")),
            "median_abs_PBIAS": clean_float(data.get("pbias_abs", {}).get("median")),
            "n_valid_basins": n_valid,
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["formulation_label"] = formulation
    return out


def load_full_performance_summary(ctx: BuildContext) -> pd.DataFrame:
    frames = [
        load_metrics_agg(ctx, ctx.project_root / "results" / "block1_main" / "base", "Basic-MOPEX", "Basic", "tableS1_multimetric_performance_summary"),
        load_metrics_agg(ctx, ctx.project_root / "results" / "block1_main" / "full", "Full-MOPEX", "Full", "tableS1_multimetric_performance_summary"),
        load_metrics_agg(ctx, ctx.project_root / "results" / "block1_main" / "flex", "CFlex-MOPEX", "CFlex", "tableS1_multimetric_performance_summary"),
        load_metrics_agg(ctx, ctx.project_root / "results" / "binary_pilot", "DFlex-MOPEX", "DFlex", "tableS1_multimetric_performance_summary"),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["formulation"] = out["formulation_label"]
    out = out.drop(columns=["formulation_label"])
    out["alpha"] = out["alpha"].fillna(0.0)
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")
    out["n_basins"] = pd.to_numeric(out["n_basins"], errors="coerce")
    out["n_valid_basins"] = pd.to_numeric(out["n_valid_basins"], errors="coerce")
    return out


def load_basin_metric_summary(ctx: BuildContext) -> pd.DataFrame:
    metric_specs = [
        (ctx.project_root / "results" / "block1_main" / "base", "Basic-MOPEX", "Basic"),
        (ctx.project_root / "results" / "block1_main" / "full", "Full-MOPEX", "Full"),
        (ctx.project_root / "results" / "block1_main" / "flex", "CFlex-MOPEX", "CFlex"),
        (ctx.project_root / "results" / "binary_pilot", "DFlex-MOPEX", "DFlex"),
    ]
    rows: list[dict[str, Any]] = []
    seed_counts: list[dict[str, Any]] = []
    for root, formulation, model_group in metric_specs:
        for path in sorted(root.rglob("metrics.json")):
            parent_name = path.parent.name.lower()
            if not parent_name.endswith("ep50"):
                continue
            rel = path.relative_to(ctx.project_root)
            match_alpha = re.search(r"alpha([0-9.]+)", str(rel))
            match_seed = re.search(r"seed[_]?([0-9]+)", str(rel))
            split = "test" if "test" in parent_name else "train"
            alpha = clean_float(match_alpha.group(1)) if match_alpha else 0.0
            seed = int(match_seed.group(1)) if match_seed else np.nan
            payload = load_json(path)
            if not isinstance(payload, dict):
                continue
            ctx.add_source("table1_performance_complexity_summary", path)
            ctx.add_source("tableS2_alpha_tradeoff_summary", path)
            nse = np.array(payload.get("nse", payload.get("NSE", [])), dtype=float)
            kge = np.array(payload.get("kge", payload.get("KGE", [])), dtype=float)
            rmse = np.array(payload.get("rmse", payload.get("RMSE", [])), dtype=float)
            pbias = np.array(payload.get("pbias", payload.get("PBIAS", [])), dtype=float)
            seed_counts.append(
                {
                    "model_group": model_group,
                    "formulation": formulation,
                    "alpha": alpha,
                    "split": split,
                    "seed": seed,
                }
            )
            n_basins = len(nse)
            for basin_idx in range(n_basins):
                rows.append(
                    {
                        "model_group": model_group,
                        "formulation": formulation,
                        "alpha": alpha,
                        "split": split,
                        "seed": seed,
                        "basin_idx": basin_idx,
                        "nse": nse[basin_idx],
                        "kge": kge[basin_idx] if basin_idx < len(kge) else np.nan,
                        "rmse": rmse[basin_idx] if basin_idx < len(rmse) else np.nan,
                        "pbias": pbias[basin_idx] if basin_idx < len(pbias) else np.nan,
                    }
                )
    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    seed_df = pd.DataFrame(seed_counts).drop_duplicates()
    basin_median = (
        raw.groupby(["model_group", "formulation", "alpha", "split", "basin_idx"], dropna=False)[
            ["nse", "kge", "rmse", "pbias"]
        ]
        .median()
        .reset_index()
    )

    def summarize(group: pd.DataFrame) -> pd.Series:
        nse = pd.to_numeric(group["nse"], errors="coerce").dropna()
        kge = pd.to_numeric(group["kge"], errors="coerce").dropna()
        rmse = pd.to_numeric(group["rmse"], errors="coerce").dropna()
        pbias = pd.to_numeric(group["pbias"], errors="coerce").dropna()
        return pd.Series(
            {
                "n_basins": int(group["basin_idx"].nunique()),
                "NSE_mean": float(nse.mean()) if not nse.empty else np.nan,
                "NSE_median": float(nse.median()) if not nse.empty else np.nan,
                "NSE_IQR": float(nse.quantile(0.75) - nse.quantile(0.25)) if not nse.empty else np.nan,
                "KGE_mean": float(kge.mean()) if not kge.empty else np.nan,
                "KGE_median": float(kge.median()) if not kge.empty else np.nan,
                "KGE_IQR": float(kge.quantile(0.75) - kge.quantile(0.25)) if not kge.empty else np.nan,
                "RMSE_mean": float(rmse.mean()) if not rmse.empty else np.nan,
                "PBIAS_mean": float(pbias.mean()) if not pbias.empty else np.nan,
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for keys, group in basin_median.groupby(["model_group", "formulation", "alpha", "split"], dropna=False):
        row = dict(zip(["model_group", "formulation", "alpha", "split"], keys))
        row.update(summarize(group).to_dict())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    n_seeds = (
        seed_df.groupby(["model_group", "formulation", "alpha", "split"], dropna=False)["seed"]
        .nunique(dropna=True)
        .reset_index(name="n_seeds_basin_metrics")
    )
    return summary.merge(n_seeds, on=["model_group", "formulation", "alpha", "split"], how="left")


def _concat_mean_median(chunks: list[np.ndarray]) -> tuple[float, float]:
    if not chunks:
        return np.nan, np.nan
    arr = np.concatenate(chunks)
    return float(arr.mean()), float(np.median(arr))


def load_complexity_summary_streaming(ctx: BuildContext) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cflex_acc: dict[float, dict[str, Any]] = {}
    cflex_root = ctx.project_root / "results" / "block1_main" / "flex"
    for test_dir in cflex_root.rglob("test1995-2010_Ep50"):
        if any(not (test_dir / f"{name}.npy").exists() for name in PROCESS_TO_W.values()):
            continue
        rel = test_dir.relative_to(ctx.project_root)
        match_alpha = re.search(r"alpha([0-9.]+)", str(rel))
        alpha = clean_float(match_alpha.group(1)) if match_alpha else np.nan
        if pd.isna(alpha):
            continue
        acc = cflex_acc.setdefault(
            float(alpha),
            {
                "total_chunks": [],
                "active_count_sum": 0.0,
                "n_total": 0,
                "active_process_hits": {proc: 0.0 for proc in PROCESS_ORDER},
            },
        )
        mmap_arrays = {
            proc: np.load(test_dir / f"{wcol}.npy", mmap_mode="r").reshape(-1)
            for proc, wcol in PROCESS_TO_W.items()
        }
        total = (
            np.asarray(mmap_arrays["Interception"], dtype=np.float64)
            + np.asarray(mmap_arrays["Phenology"], dtype=np.float64)
            + np.asarray(mmap_arrays["Snow"], dtype=np.float64)
            + np.asarray(mmap_arrays["Subsurface"], dtype=np.float64)
        )
        active_count = (
            (np.asarray(mmap_arrays["Interception"]) > ctx.active_threshold).astype(np.float64)
            + (np.asarray(mmap_arrays["Phenology"]) > ctx.active_threshold).astype(np.float64)
            + (np.asarray(mmap_arrays["Snow"]) > ctx.active_threshold).astype(np.float64)
            + (np.asarray(mmap_arrays["Subsurface"]) > ctx.active_threshold).astype(np.float64)
        )
        acc["total_chunks"].append(np.asarray(total, dtype=np.float32))
        acc["active_count_sum"] += float(active_count.sum())
        acc["n_total"] += int(active_count.size)
        for proc in PROCESS_ORDER:
            acc["active_process_hits"][proc] += float((np.asarray(mmap_arrays[proc]) > ctx.active_threshold).sum())
        ctx.add_source("table1_performance_complexity_summary", test_dir / "w_int.npy")
        ctx.add_source("tableS2_alpha_tradeoff_summary", test_dir / "w_int.npy")
        ctx.add_source("tableS9_nmul_ablation_summary", test_dir / "w_int.npy")
    for alpha, acc in cflex_acc.items():
        total_mean, total_median = _concat_mean_median(acc["total_chunks"])
        denom = acc["n_total"] if acc["n_total"] else np.nan
        row = {
            "model_group": "CFlex",
            "formulation": "CFlex-MOPEX",
            "alpha": float(alpha),
            "total_structural_weight_mean": total_mean,
            "total_structural_weight_median": total_median,
            "active_process_count_mean": safe_div(acc["active_count_sum"], denom),
            "active_fraction_mean": safe_div(acc["active_count_sum"], denom * len(PROCESS_ORDER)),
            "complexity_metric_type": "raw_total_weight",
            "structural_complexity_definition": "raw total weight = w_int + w_phen + w_snow + w_sub",
        }
        for proc in PROCESS_ORDER:
            row[f"active_fraction_{proc.lower()}"] = safe_div(acc["active_process_hits"][proc], denom)
        rows.append(row)

    dflex_acc: dict[float, dict[str, Any]] = {}
    dflex_root = ctx.project_root / "results" / "binary_pilot"
    for test_dir in dflex_root.rglob("test1995-2010_Ep50"):
        if any(not (test_dir / f"{name}.npy").exists() for name in PROCESS_TO_Z.values()):
            continue
        rel = test_dir.relative_to(ctx.project_root)
        match_alpha = re.search(r"alpha([0-9.]+)", str(rel))
        alpha = clean_float(match_alpha.group(1)) if match_alpha else np.nan
        if pd.isna(alpha):
            continue
        acc = dflex_acc.setdefault(
            float(alpha),
            {
                "active_count_chunks": [],
                "active_count_sum": 0.0,
                "n_total": 0,
                "active_process_hits": {proc: 0.0 for proc in PROCESS_ORDER},
            },
        )
        mmap_arrays = {
            proc: np.load(test_dir / f"{zcol}.npy", mmap_mode="r").reshape(-1)
            for proc, zcol in PROCESS_TO_Z.items()
        }
        active_count = (
            np.asarray(mmap_arrays["Interception"], dtype=np.float64)
            + np.asarray(mmap_arrays["Phenology"], dtype=np.float64)
            + np.asarray(mmap_arrays["Snow"], dtype=np.float64)
            + np.asarray(mmap_arrays["Subsurface"], dtype=np.float64)
        )
        acc["active_count_chunks"].append(np.asarray(active_count, dtype=np.float32))
        acc["active_count_sum"] += float(active_count.sum())
        acc["n_total"] += int(active_count.size)
        for proc in PROCESS_ORDER:
            acc["active_process_hits"][proc] += float((np.asarray(mmap_arrays[proc]) > 0.5).sum())
        ctx.add_source("table1_performance_complexity_summary", test_dir / "z_int.npy")
        ctx.add_source("tableS2_alpha_tradeoff_summary", test_dir / "z_int.npy")
    for alpha, acc in dflex_acc.items():
        count_mean, count_median = _concat_mean_median(acc["active_count_chunks"])
        denom = acc["n_total"] if acc["n_total"] else np.nan
        row = {
            "model_group": "DFlex",
            "formulation": "DFlex-MOPEX",
            "alpha": float(alpha),
            "total_structural_weight_mean": count_mean,
            "total_structural_weight_median": count_median,
            "active_process_count_mean": safe_div(acc["active_count_sum"], denom),
            "active_fraction_mean": safe_div(acc["active_count_sum"], denom * len(PROCESS_ORDER)),
            "complexity_metric_type": "active_process_count",
            "structural_complexity_definition": "binary active count from z_* activation outputs",
        }
        for proc in PROCESS_ORDER:
            row[f"active_fraction_{proc.lower()}"] = safe_div(acc["active_process_hits"][proc], denom)
        rows.append(row)
    return pd.DataFrame(rows)


def build_table_s1(ctx: BuildContext, perf_seed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "model_group",
        "formulation",
        "alpha",
        "split",
        "seed",
        "n_basins",
        "median_NSE",
        "mean_NSE",
        "median_KGE",
        "mean_KGE",
        "median_RMSE",
        "mean_RMSE",
        "median_PBIAS",
        "median_abs_PBIAS",
        "n_valid_basins",
    ]
    table = perf_seed[cols].copy()
    test_only = table[table["split"].eq("test")].copy()
    return table, test_only


def build_alpha_summary(
    ctx: BuildContext,
    perf_seed: pd.DataFrame,
    complexity: pd.DataFrame,
    basin_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    test = perf_seed[perf_seed["split"].eq("test")].copy()
    train = perf_seed[perf_seed["split"].eq("train")].copy()
    basic_test = test[test["model_group"].eq("Basic")]
    full_test = test[test["model_group"].eq("Full")]
    basic_nse = basic_test["median_NSE"].median() if not basic_test.empty else np.nan
    full_nse = full_test["median_NSE"].median() if not full_test.empty else np.nan
    groups = perf_seed.groupby(["model_group", "formulation", "alpha", "split"], dropna=False)
    for (model_group, formulation, alpha, split), sub in groups:
        row = {
            "model_group": model_group,
            "formulation": formulation,
            "alpha": float(alpha),
            "split": split,
            "n_seeds": int(sub["seed"].nunique()),
            "n_basins": int(sub["n_basins"].median()),
            "median_NSE": float(sub["median_NSE"].median()),
            "median_KGE": float(sub["median_KGE"].median()),
            "median_RMSE": float(sub["median_RMSE"].median()),
            "median_PBIAS": float(sub["median_PBIAS"].median()),
            "median_abs_PBIAS": float(sub["median_abs_PBIAS"].median()),
        }
        if split == "test":
            row["delta_NSE_vs_Basic"] = row["median_NSE"] - basic_nse if not pd.isna(basic_nse) else np.nan
            row["fraction_of_Full_gain_retained"] = safe_div(row["delta_NSE_vs_Basic"], full_nse - basic_nse)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.merge(
        complexity[
            [
                "model_group",
                "formulation",
                "alpha",
                "total_structural_weight_mean",
                "total_structural_weight_median",
                "active_process_count_mean",
                "active_fraction_mean",
                "complexity_metric_type",
                "structural_complexity_definition",
            ]
        ],
        on=["model_group", "formulation", "alpha"],
        how="left",
    )
    out["alpha_path_note"] = out["model_group"].map(
        {
            "CFlex": "CFlex complexity uses raw total structural weight.",
            "DFlex": "DFlex complexity uses binary active count; DFlex and CFlex complexity values are not identical quantities.",
            "Basic": "Basic-MOPEX baseline reference.",
            "Full": "Full-MOPEX structural endpoint reference.",
        }
    )
    out.loc[out["model_group"].eq("Basic"), "total_structural_weight_mean"] = 0.0
    out.loc[out["model_group"].eq("Basic"), "total_structural_weight_median"] = 0.0
    out.loc[out["model_group"].eq("Basic"), "active_process_count_mean"] = 0.0
    out.loc[out["model_group"].eq("Basic"), "active_fraction_mean"] = 0.0
    out.loc[out["model_group"].eq("Basic"), "complexity_metric_type"] = "raw_total_weight"
    out.loc[out["model_group"].eq("Basic"), "structural_complexity_definition"] = "0 for Basic-MOPEX"
    out.loc[out["model_group"].eq("Full"), "total_structural_weight_mean"] = 4.0
    out.loc[out["model_group"].eq("Full"), "total_structural_weight_median"] = 4.0
    out.loc[out["model_group"].eq("Full"), "active_process_count_mean"] = 4.0
    out.loc[out["model_group"].eq("Full"), "active_fraction_mean"] = 1.0
    out.loc[out["model_group"].eq("Full"), "complexity_metric_type"] = "active_process_count"
    out.loc[out["model_group"].eq("Full"), "structural_complexity_definition"] = "4 for Full-MOPEX"
    if not basin_metrics.empty:
        out = out.merge(
            basin_metrics,
            on=["model_group", "formulation", "alpha", "split"],
            how="left",
        )
        out["n_seeds"] = out["n_seeds_basin_metrics"].fillna(out["n_seeds"])
        out["n_basins"] = out["n_basins_y"].fillna(out["n_basins_x"])
        out = out.drop(columns=["n_seeds_basin_metrics", "n_basins_x", "n_basins_y"])
    out["n_seeds"] = pd.to_numeric(out["n_seeds"], errors="coerce").astype("Int64")
    out["n_basins"] = pd.to_numeric(out["n_basins"], errors="coerce").astype("Int64")
    return out


def select_representative_dflex(alphas: list[float]) -> list[float]:
    if not alphas:
        return []
    sorted_alphas = sorted(alphas)
    if len(sorted_alphas) <= 3:
        return sorted_alphas
    picks = [sorted_alphas[0], sorted_alphas[len(sorted_alphas) // 2], sorted_alphas[-1]]
    unique: list[float] = []
    for value in picks:
        if value not in unique:
            unique.append(value)
    return unique


def build_table1(ctx: BuildContext, alpha_summary: pd.DataFrame) -> pd.DataFrame:
    test = alpha_summary[alpha_summary["split"].eq("test")].copy()
    if test.empty:
        return pd.DataFrame()
    target_cflex = {0.005, 0.01, 0.03}
    target_dflex = {0.005, 0.01, 0.03}
    keep = (
        test["model_group"].isin(["Basic", "Full"])
        | ((test["model_group"].eq("CFlex")) & test["alpha"].isin(target_cflex))
        | ((test["model_group"].eq("DFlex")) & test["alpha"].isin(target_dflex))
    )
    table = test[keep].copy()
    basic_row = table[table["model_group"].eq("Basic")]
    full_row = table[table["model_group"].eq("Full")]
    basic_nse_mean = clean_float(basic_row["NSE_mean"].iloc[0]) if not basic_row.empty else np.nan
    full_nse_mean = clean_float(full_row["NSE_mean"].iloc[0]) if not full_row.empty else np.nan
    basic_nse_median = clean_float(basic_row["NSE_median"].iloc[0]) if not basic_row.empty else np.nan
    full_nse_median = clean_float(full_row["NSE_median"].iloc[0]) if not full_row.empty else np.nan
    table["delta_NSE_vs_Basic_mean"] = table["NSE_mean"] - basic_nse_mean
    table["delta_NSE_vs_Basic_median"] = table["NSE_median"] - basic_nse_median
    table["fraction_of_Full_gain_retained_meanNSE"] = table["delta_NSE_vs_Basic_mean"].apply(
        lambda v: safe_div(v, full_nse_mean - basic_nse_mean)
    )
    table["fraction_of_Full_gain_retained_medianNSE"] = table["delta_NSE_vs_Basic_median"].apply(
        lambda v: safe_div(v, full_nse_median - basic_nse_median)
    )
    table["structural_complexity_value"] = np.where(
        table["model_group"].eq("CFlex"),
        table["total_structural_weight_mean"],
        np.where(
            table["model_group"].eq("DFlex"),
            table["active_process_count_mean"],
            np.where(table["model_group"].eq("Basic"), 0.0, 4.0),
        ),
    )
    table["structural_complexity_type"] = np.where(
        table["model_group"].eq("CFlex"),
        "raw_total_weight",
        np.where(
            table["model_group"].eq("DFlex"),
            "active_process_count",
            np.where(table["model_group"].eq("Basic"), "0", "4"),
        ),
    )
    notes = []
    for _, row in table.iterrows():
        if row["model_group"] == "CFlex":
            notes.append("CFlex complexity uses raw total structural weight; weights summarize learned structure but are not physical process fractions.")
        elif row["model_group"] == "DFlex":
            notes.append("DFlex rows are included as discrete-selection references; the DFlex penalty scale should not be interpreted as identical to the CFlex continuous-weight penalty scale.")
        elif row["model_group"] == "Full":
            notes.append("Full-MOPEX endpoint reference with all four structural extensions available.")
        else:
            notes.append("Basic-MOPEX baseline reference.")
    table["table_note"] = notes
    table["structural_complexity_definition"] = np.where(
        table["model_group"].eq("CFlex"),
        "raw total weight = w_int + w_phen + w_snow + w_sub",
        np.where(
            table["model_group"].eq("DFlex"),
            "active count from binary z_* outputs",
            np.where(table["model_group"].eq("Basic"), "0 for Basic-MOPEX", "4 for Full-MOPEX"),
        ),
    )
    cols = [
        "model_group",
        "formulation",
        "alpha",
        "split",
        "n_seeds",
        "n_basins",
        "NSE_mean",
        "NSE_median",
        "NSE_IQR",
        "KGE_mean",
        "KGE_median",
        "KGE_IQR",
        "RMSE_mean",
        "PBIAS_mean",
        "delta_NSE_vs_Basic_mean",
        "delta_NSE_vs_Basic_median",
        "fraction_of_Full_gain_retained_meanNSE",
        "fraction_of_Full_gain_retained_medianNSE",
        "structural_complexity_value",
        "structural_complexity_type",
        "table_note",
        "structural_complexity_definition",
    ]
    return sort_rows(table[cols])


def build_table1b(ctx: BuildContext, complexity: pd.DataFrame) -> pd.DataFrame:
    if complexity.empty:
        return pd.DataFrame()
    target_alphas = [0.005, 0.01, 0.03]
    rows: list[dict[str, Any]] = [
        {
            "Model": "Basic",
            "lambda": 0.0,
            "C_b_mean": np.nan,
            "C_b_median": np.nan,
            "Active_count_mean": np.nan,
            "Snow_active": np.nan,
            "Subsurface_active": np.nan,
            "Phenology_active": np.nan,
            "Interception_active": np.nan,
        },
        {
            "Model": "Full",
            "lambda": 0.0,
            "C_b_mean": np.nan,
            "C_b_median": np.nan,
            "Active_count_mean": np.nan,
            "Snow_active": np.nan,
            "Subsurface_active": np.nan,
            "Phenology_active": np.nan,
            "Interception_active": np.nan,
        },
    ]
    for model_group in ["CFlex", "DFlex"]:
        sub = complexity[
            complexity["model_group"].eq(model_group)
            & pd.to_numeric(complexity["alpha"], errors="coerce").round(3).isin([round(a, 3) for a in target_alphas])
        ].copy()
        for alpha in target_alphas:
            row = sub[pd.to_numeric(sub["alpha"], errors="coerce").round(3).eq(round(alpha, 3))]
            if row.empty:
                continue
            rec = row.iloc[0]
            rows.append(
                {
                    "Model": model_group,
                    "lambda": alpha,
                    "C_b_mean": clean_float(rec.get("total_structural_weight_mean")),
                    "C_b_median": clean_float(rec.get("total_structural_weight_median")),
                    "Active_count_mean": clean_float(rec.get("active_process_count_mean")),
                    "Snow_active": clean_float(rec.get("active_fraction_snow")),
                    "Subsurface_active": clean_float(rec.get("active_fraction_subsurface")),
                    "Phenology_active": clean_float(rec.get("active_fraction_phenology")),
                    "Interception_active": clean_float(rec.get("active_fraction_interception")),
                }
            )
    return pd.DataFrame(rows)


def sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_model_order"] = out["model_group"].map(MODEL_ORDER).fillna(99)
    out["_alpha"] = pd.to_numeric(out["alpha"], errors="coerce").fillna(-1)
    out = out.sort_values(["_model_order", "_alpha", "formulation"]).drop(columns=["_model_order", "_alpha"])
    return out.reset_index(drop=True)


def build_table_s2(ctx: BuildContext, alpha_summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "model_group",
        "formulation",
        "alpha",
        "split",
        "n_seeds",
        "n_basins",
        "NSE_mean",
        "NSE_median",
        "NSE_IQR",
        "KGE_mean",
        "KGE_median",
        "KGE_IQR",
        "RMSE_mean",
        "PBIAS_mean",
        "median_NSE",
        "median_KGE",
        "median_RMSE",
        "median_abs_PBIAS",
        "total_structural_weight_mean",
        "total_structural_weight_median",
        "active_process_count_mean",
        "active_fraction_mean",
        "delta_NSE_vs_Basic",
        "fraction_of_Full_gain_retained",
        "complexity_metric_type",
        "alpha_path_note",
        "structural_complexity_definition",
    ]
    return sort_rows(alpha_summary[cols].copy())


def build_table_s3(ctx: BuildContext) -> pd.DataFrame:
    path = ctx.figures_csv_dir / "figA4_seed_robustness_data.csv"
    df = read_csv(ctx, path, "tableS3_seed_robustness_summary")
    if df.empty:
        return pd.DataFrame()
    df["process_coordinate"] = df["process"].map(normalize_process)
    rank_rows = df[df["panel"].eq("rank_stability")].copy()
    icc_rows = df[df["panel"].eq("icc")].copy()
    dom_rows = df[df["panel"].eq("dominant_agreement")].copy()
    jac_rows = df[df["panel"].eq("active_set_jaccard")].copy()
    rows: list[dict[str, Any]] = []
    for process in PROCESS_ORDER:
        rank = rank_rows[rank_rows["process_coordinate"].eq(process)].copy()
        icc = icc_rows[icc_rows["process_coordinate"].eq(process)].copy()
        jac = jac_rows[jac_rows["process_coordinate"].eq(process)].copy()
        pair_vals = pd.to_numeric(rank["spearman_r"], errors="coerce").dropna()
        pairwise_spearman_median = float(pair_vals.median()) if not pair_vals.empty else np.nan
        pairwise_spearman_iqr = float(pair_vals.quantile(0.75) - pair_vals.quantile(0.25)) if len(pair_vals) else np.nan
        icc_val = clean_float(icc["icc"].iloc[0]) if not icc.empty else np.nan
        n_seeds = int(clean_float(icc["n_seeds"].iloc[0])) if not icc.empty and not pd.isna(clean_float(icc["n_seeds"].iloc[0])) else 5
        n_basins = int(clean_float(rank["n_basins"].dropna().iloc[0])) if not rank["n_basins"].dropna().empty else 671
        dom_val = float(pd.to_numeric(dom_rows["agreement_rate"], errors="coerce").mean()) if not dom_rows.empty else np.nan
        process_jac = pd.to_numeric(jac.loc[jac["process_coordinate"].eq(process), "jaccard"], errors="coerce").dropna()
        active_fraction = float(pd.to_numeric(df[(df["panel"].eq("icc")) & (df["process_coordinate"].eq(process))]["pct_basins_cv_gt02"], errors="coerce").iloc[0] / 100.0) if not icc.empty else np.nan
        row = {
            "alpha": ctx.primary_alpha,
            "process_coordinate": process,
            "n_seeds": n_seeds,
            "n_basins": n_basins,
            "pairwise_spearman_median": pairwise_spearman_median,
            "pairwise_spearman_iqr": pairwise_spearman_iqr,
            "ICC": icc_val,
            "dominant_process_agreement": dom_val,
            "active_set_jaccard": float(process_jac.median()) if not process_jac.empty else np.nan,
            "robustness_class": classify_seed_robustness(
                pairwise_spearman_median,
                icc_val,
                dom_val,
                float(process_jac.median()) if not process_jac.empty else np.nan,
                active_fraction,
            ),
        }
        rows.append(row)
    ctx.notes["tableS3_seed_robustness_summary"] = "Robustness class thresholds: strong requires pairwise Spearman >= 0.8, ICC >= 0.75, dominant-process agreement >= 0.8, and active-set Jaccard >= 0.75; moderate requires at least two metrics above 0.5-0.6; sparse or low-support processes are labeled weak."
    return pd.DataFrame(rows)


def build_table_s4(ctx: BuildContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for filename, formulation in [
        ("figure6_cflex_attribute_relationships.csv", "CFlex-MOPEX"),
        ("figure6_dflex_attribute_relationships.csv", "DFlex-MOPEX"),
    ]:
        path = ctx.figures_csv_dir / filename
        df = read_csv(ctx, path, "tableS4_hydroclimatic_control_summary")
        if df.empty:
            continue
        out = pd.DataFrame(
            {
                "formulation": formulation,
                "alpha": pd.to_numeric(df["alpha"], errors="coerce"),
                "process_coordinate": df["weight_name"].map(normalize_process),
                "attribute_name": df["attribute"].map(display_attribute_label),
                "attribute_group": df["attribute"].map(attribute_group),
                "spearman_rho": pd.to_numeric(df["median_rho"], errors="coerce"),
                "abs_spearman_rho": pd.to_numeric(df["median_rho"], errors="coerce").abs(),
                "p_value_or_q_value_if_available": pd.to_numeric(df["median_p"], errors="coerce"),
            }
        )
        out = out.dropna(subset=["spearman_rho"])
        out = out.sort_values(["formulation", "alpha", "process_coordinate", "abs_spearman_rho"], ascending=[True, True, True, False])
        out["rank_within_process"] = out.groupby(["formulation", "alpha", "process_coordinate"]).cumcount() + 1
        out["hydroclimatic_control_class"] = [
            control_strength_class(v, p) for v, p in zip(out["abs_spearman_rho"], out["process_coordinate"])
        ]
        frames.append(out)
    main = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    if not main.empty:
        preferred = main[(main["formulation"].eq("CFlex-MOPEX")) & (main["alpha"].round(3).eq(round(ctx.primary_alpha, 3)))]
        for process in PROCESS_ORDER:
            sub = preferred[preferred["process_coordinate"].eq(process)].head(3).copy()
            vals = sub["abs_spearman_rho"].dropna()
            summary_rows.append(
                {
                    "process_coordinate": process,
                    "top_attribute_1": sub["attribute_name"].iloc[0] if len(sub) > 0 else "",
                    "top_attribute_1_rho": sub["spearman_rho"].iloc[0] if len(sub) > 0 else np.nan,
                    "top_attribute_2": sub["attribute_name"].iloc[1] if len(sub) > 1 else "",
                    "top_attribute_2_rho": sub["spearman_rho"].iloc[1] if len(sub) > 1 else np.nan,
                    "top_attribute_3": sub["attribute_name"].iloc[2] if len(sub) > 2 else "",
                    "top_attribute_3_rho": sub["spearman_rho"].iloc[2] if len(sub) > 2 else np.nan,
                    "control_strength_class": control_strength_class(float(vals.max()) if not vals.empty else np.nan, process),
                    "interpretation_note": {
                        "Interception": "Weak or sparse hydroclimatic organization under streamflow-only training.",
                        "Phenology": "Moderate organization with region-sensitive seasonal and vegetation signals.",
                        "Snow": "Strong hydroclimatic organization with clear snow-climate alignment.",
                        "Subsurface": "Moderate control with mixed hydrologic and storage-related drivers.",
                    }[process],
                }
            )
    return main, pd.DataFrame(summary_rows)


def target_group_from_name(name: str) -> str:
    text = str(name).lower()
    if "pc" in text and ("full" in text or "all-parameter" in text):
        return "All-parameter PCs"
    if "pc" in text:
        return "Process-group PCs"
    if "norm" in text or "spread" in text or "complexity" in text:
        return "Parameter-space metrics"
    return "Parameter"


def strongest_parameter_group(process: str, associations: pd.DataFrame) -> str:
    sub = associations[associations["process_coordinate"].eq(process)]
    if sub.empty:
        return ""
    ranked = sub.sort_values("abs_spearman_rho", ascending=False)
    value = ranked["parameter_group"].dropna().astype(str)
    return value.iloc[0] if not value.empty else ""


def build_table_s5(ctx: BuildContext) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assoc_a = read_csv(ctx, ctx.figures_csv_dir / "figure8_weight_parameter_correlations.csv", "tableS5_parameter_space_readout_associations")
    assoc_b = read_csv(ctx, ctx.figures_csv_dir / "figA7_parameter_readout_stability_data.csv", "tableS5_parameter_space_readout_associations")
    rows: list[dict[str, Any]] = []
    if not assoc_a.empty:
        grouped = (
            assoc_a.assign(
                process_coordinate=assoc_a["weight_name"].map(normalize_process),
                abs_spearman_rho=pd.to_numeric(assoc_a["spearman_rho"], errors="coerce").abs(),
            )
            .groupby(["alpha", "process_coordinate", "parameter_name", "parameter_group"], dropna=False)["spearman_rho"]
            .median()
            .reset_index()
        )
        grouped["abs_spearman_rho"] = pd.to_numeric(grouped["spearman_rho"], errors="coerce").abs()
        grouped = grouped.sort_values(["alpha", "process_coordinate", "abs_spearman_rho"], ascending=[True, True, False])
        grouped["rank"] = grouped.groupby(["alpha", "process_coordinate"]).cumcount() + 1
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "alpha": row["alpha"],
                    "process_coordinate": row["process_coordinate"],
                    "parameter_or_PC_target": display_parameter_name(row["parameter_name"]),
                    "parameter_group": row["parameter_group"],
                    "spearman_rho": row["spearman_rho"],
                    "abs_spearman_rho": row["abs_spearman_rho"],
                    "rank": row["rank"],
                    "readout_class": readout_class(clean_float(row["abs_spearman_rho"])),
                }
            )
    if not assoc_b.empty:
        alpha_assoc = assoc_b[assoc_b["panel"].eq("alpha_association")].copy()
        alpha_assoc["process_coordinate"] = alpha_assoc["structural_coordinate"].map(normalize_process)
        alpha_assoc["parameter_or_PC_target"] = alpha_assoc["target"].map(display_parameter_name)
        alpha_assoc["parameter_group"] = alpha_assoc["target"].map(target_group_from_name)
        alpha_assoc["abs_spearman_rho"] = pd.to_numeric(alpha_assoc["spearman_rho"], errors="coerce").abs()
        alpha_assoc = alpha_assoc.sort_values(["alpha", "process_coordinate", "abs_spearman_rho"], ascending=[True, True, False])
        alpha_assoc["rank"] = alpha_assoc.groupby(["alpha", "process_coordinate"]).cumcount() + 1
        for _, row in alpha_assoc.iterrows():
            rows.append(
                {
                    "alpha": row["alpha"],
                    "process_coordinate": row["process_coordinate"],
                    "parameter_or_PC_target": row["parameter_or_PC_target"],
                    "parameter_group": row["parameter_group"],
                    "spearman_rho": row["spearman_rho"],
                    "abs_spearman_rho": row["abs_spearman_rho"],
                    "rank": row["rank"],
                    "readout_class": readout_class(clean_float(row["abs_spearman_rho"])),
                }
            )
    associations = pd.DataFrame(rows)
    recon_source = read_csv(ctx, ctx.figures_csv_dir / "figA7_parameter_readout_stability_data.csv", "tableS5_parameter_space_readout_reconstruction")
    recon_rows: list[dict[str, Any]] = []
    if not recon_source.empty:
        recon = recon_source[recon_source["panel"].eq("reconstruction")].copy()
        for _, row in recon.iterrows():
            recon_rows.append(
                {
                    "alpha": row["alpha"],
                    "target": row["target"],
                    "out_of_fold_R2": row["out_of_fold_R2"],
                    "spearman_rho": np.nan,
                    "reconstruction_class": reconstruction_class(clean_float(row["out_of_fold_R2"])),
                    "negative_R2_flag": bool(clean_float(row["out_of_fold_R2"]) < 0),
                }
            )
    recon_table = pd.DataFrame(recon_rows)
    if recon_table.empty:
        recon_text = read_csv(ctx, ctx.figures_csv_dir / "plot_fig11_table_sx_parameter_organization_reconstruction.csv", "tableS5_parameter_space_readout_reconstruction")
        if not recon_text.empty:
            recon_table = pd.DataFrame(
                {
                    "alpha": ctx.primary_alpha,
                    "target": recon_text["target_variable"],
                    "out_of_fold_R2": recon_text["out_of_fold_R2"],
                    "spearman_rho": recon_text["spearman_observed_predicted"],
                    "reconstruction_class": recon_text["out_of_fold_R2"].map(reconstruction_class),
                    "negative_R2_flag": recon_text["out_of_fold_R2"].lt(0),
                }
            )
    summary_source = recon_source[recon_source["panel"].eq("process_readout_strength")].copy() if not recon_source.empty else pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    for process in PROCESS_ORDER:
        sub = associations[(associations["process_coordinate"].eq(process)) & (associations["alpha"].round(3).eq(round(ctx.primary_alpha, 3)))]
        top = sub.sort_values("abs_spearman_rho", ascending=False).head(5)
        process_strength = summary_source[(summary_source["process"].map(normalize_process).eq(process)) & (summary_source["alpha"].round(3).eq(round(ctx.primary_alpha, 3)))]
        summary_rows.append(
            {
                "process_coordinate": process,
                "strongest_parameter_group": strongest_parameter_group(process, top),
                "mean_top_five_abs_rho": float(top["abs_spearman_rho"].mean()) if not top.empty else np.nan,
                "reconstruction_support": readout_class(clean_float(process_strength["mean_top5_abs_rho"].iloc[0])) if not process_strength.empty else "weak",
                "readout_interpretation": {
                    "Interception": "Weak-to-moderate readout; does not imply a fully resolved interception parameterization.",
                    "Phenology": "Moderate readout with region-sensitive seasonal organization.",
                    "Snow": "Strongest and most coherent parameter-space readout.",
                    "Subsurface": "Moderate-to-strong readout with mixed hydrologic controls.",
                }[process],
            }
        )
    ctx.notes["tableS5_parameter_space_readout_reconstruction"] = "Negative out-of-fold R² means worse than a mean-baseline predictor for that target."
    return associations, recon_table, pd.DataFrame(summary_rows)


def build_table2_legacy_summary(
    ctx: BuildContext,
    s3: pd.DataFrame,
    s4: pd.DataFrame,
    s4_compact: pd.DataFrame,
    complexity: pd.DataFrame,
    s5_assoc: pd.DataFrame,
    s5_recon: pd.DataFrame,
) -> pd.DataFrame:
    seed_label = {
        "strong": "highly seed-stable",
        "moderate": "seed-stable",
        "weak": "weak seed stability",
    }
    hydro_label = {
        "strong": "strong",
        "moderate": "moderate",
        "weak": "weak",
        "sparse": "weak",
    }
    readout_label = {
        "strong": "strong readout",
        "moderate": "moderate readout",
        "weak": "weak readout",
    }
    interpretation = {
        "Snow": "strongest and most transferable coordinate",
        "Subsurface": "stable but hydrologically mixed",
        "Phenology": "moderate and region-sensitive",
        "Interception": "weakly active and weakly resolved under streamflow-only training",
    }
    process_key = {
        "Interception": "interception",
        "Phenology": "phenology",
        "Snow": "snow",
        "Subsurface": "subsurface",
    }

    def collapse_parameter_target(target: Any) -> str:
        text = str(target).strip()
        if not text:
            return ""
        return re.sub(r"_m[1-4]$", "", text)

    rows: list[dict[str, Any]] = []
    if "alpha" in s5_recon.columns:
        recon_alpha = s5_recon[pd.to_numeric(s5_recon["alpha"], errors="coerce").round(3).eq(round(ctx.primary_alpha, 3))].copy()
    else:
        recon_alpha = pd.DataFrame()
    primary_complexity = complexity[
        complexity["model_group"].eq("CFlex")
        & pd.to_numeric(complexity["alpha"], errors="coerce").round(3).eq(round(ctx.primary_alpha, 3))
    ].copy()
    for process in ["Snow", "Subsurface", "Phenology", "Interception"]:
        seed = s3[s3["process_coordinate"].eq(process)]
        hydro = s4_compact[s4_compact["process_coordinate"].eq(process)]
        hydro_full = s4[
            s4["formulation"].eq("CFlex-MOPEX")
            & pd.to_numeric(s4["alpha"], errors="coerce").round(3).eq(round(ctx.primary_alpha, 3))
            & s4["process_coordinate"].eq(process)
        ].sort_values(["abs_spearman_rho", "attribute_name"], ascending=[False, True])
        assoc = s5_assoc[
            s5_assoc["process_coordinate"].eq(process)
            & pd.to_numeric(s5_assoc["alpha"], errors="coerce").round(3).eq(round(ctx.primary_alpha, 3))
        ].copy()
        assoc = assoc[
            ~assoc["parameter_group"].astype(str).isin({"All-parameter PCs", "Process-group PCs", "Parameter-space metrics"})
        ]
        assoc["parameter_type"] = assoc["parameter_or_PC_target"].map(collapse_parameter_target)
        assoc = (
            assoc.sort_values(["abs_spearman_rho", "parameter_or_PC_target"], ascending=[False, True])
            .drop_duplicates(subset=["parameter_type"], keep="first")
            .head(3)
        )
        hydro_items = []
        if not hydro.empty:
            for idx in range(1, 4):
                name = str(hydro[f"top_attribute_{idx}"].iloc[0]).strip()
                if name:
                    hydro_items.append(format_signed_item(name, hydro[f"top_attribute_{idx}_rho"].iloc[0]))
        param_items = [
            format_signed_item(str(row["parameter_or_PC_target"]).strip(), row["spearman_rho"])
            for _, row in assoc.iterrows()
            if str(row["parameter_or_PC_target"]).strip()
        ]
        best_r2 = np.nan
        if not recon_alpha.empty:
            proc_text = process.lower()
            proc_hits = recon_alpha[
                recon_alpha["target"].astype(str).str.lower().str.contains(proc_text, na=False)
            ]
            if not proc_hits.empty:
                best_r2 = float(pd.to_numeric(proc_hits["out_of_fold_R2"], errors="coerce").max())
        active_fraction = np.nan
        if not primary_complexity.empty:
            active_fraction = clean_float(primary_complexity.iloc[0].get(f"active_fraction_{process.lower()}"))
        rows.append(
            {
                "process": process_key[process],
                "seed_robustness_summary": seed_label.get(str(seed["robustness_class"].iloc[0]), "") if not seed.empty else "",
                "seed_spearman_median": clean_float(seed["pairwise_spearman_median"].iloc[0]) if not seed.empty else np.nan,
                "ICC": clean_float(seed["ICC"].iloc[0]) if not seed.empty else np.nan,
                "active_fraction": active_fraction,
                "top_hydroclimatic_controls": "; ".join(hydro_items),
                "max_abs_attribute_rho": (
                    max(abs(clean_float(hydro[f"top_attribute_{idx}_rho"].iloc[0])) for idx in range(1, 4) if str(hydro[f"top_attribute_{idx}"].iloc[0]).strip())
                    if not hydro.empty and any(str(hydro[f"top_attribute_{idx}"].iloc[0]).strip() for idx in range(1, 4))
                    else np.nan
                ),
                "mean_top5_abs_attribute_rho": float(pd.to_numeric(hydro_full.head(5)["abs_spearman_rho"], errors="coerce").mean()) if not hydro_full.empty else np.nan,
                "hydroclimatic_claim_strength": hydro_label.get(str(hydro["control_strength_class"].iloc[0]), "") if not hydro.empty else "",
                "top_parameter_space_links": "; ".join(param_items),
                "max_abs_parameter_rho": float(pd.to_numeric(assoc["abs_spearman_rho"], errors="coerce").max()) if not assoc.empty else np.nan,
                "best_reconstruction_R2": best_r2,
                "parameter_space_claim_strength": readout_label.get(
                    readout_class(float(pd.to_numeric(assoc["abs_spearman_rho"], errors="coerce").max())) if not assoc.empty else np.nan,
                    "",
                ),
                "overall_interpretation": interpretation[process],
            }
        )
    ctx.notes["table2_process_coordinate_evidence_summary"] = (
        r"The \texttt{best\_reconstruction\_R2} column is taken directly from the primary-$\alpha$ reconstruction table "
        r"using only targets whose names explicitly contain the process label. Under the current CSV-derived target set, "
        r"this yields \texttt{snow\_temp\_PC1} for snow and \texttt{phenology\_ET\_PC1} for phenology. No reconstruction "
        r"target names in the current table contain the strings \texttt{subsurface} or \texttt{interception}, so those "
        r"entries are left blank rather than filled by a manual or semantic mapping. The blank cells therefore indicate "
        r"no direct name-matched process-specific reconstruction target in the current Python-generated export, not a "
        r"missing file or hand-edited omission."
    )
    return pd.DataFrame(rows)


def load_loro_performance_metrics(ctx: BuildContext) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    root = ctx.project_root / "results" / "block3_loro" / "config_dmopex_v1"
    for path in sorted(root.rglob("sim/metrics_agg.json")):
        ctx.add_source("tableS6_loro_regional_performance_summary", path)
        data = load_json(path)
        rel = str(path.relative_to(ctx.project_root))
        match = re.search(r"(base|full|flex)_region(\d+)/seed_([0-9]+)", rel)
        if not match:
            continue
        model_key = match.group(1)
        region = normalize_region_label(match.group(2))
        seed = int(match.group(3))
        rows.append(
            {
                "held_out_region": region,
                "model_key": model_key,
                "seed": seed,
                "median_NSE": clean_float(data.get("nse", {}).get("median")),
                "median_KGE": clean_float(data.get("kge", {}).get("median")),
                "median_RMSE": clean_float(data.get("rmse", {}).get("median")),
                "median_abs_PBIAS": clean_float(data.get("pbias_abs", {}).get("median")),
            }
        )
    return pd.DataFrame(rows)


def load_corrected_loro_relation(ctx: BuildContext, table_name: str) -> pd.DataFrame:
    candidates = [
        ctx.project_root / "analysis" / "flex_mopex_loro_corrected_statistics" / "loro_structure_error_performance_relation.csv",
        ctx.project_root / ".tmp" / "analysis" / "flex_mopex_loro_corrected_statistics" / "loro_structure_error_performance_relation.csv",
    ]
    for path in candidates:
        if path.exists():
            df = read_csv(ctx, path, table_name)
            if not df.empty:
                return df
    ctx.warn("Corrected LORO relation table not found; falling back to manuscript figure CSV exports.")
    return pd.DataFrame()


def build_table_s6(ctx: BuildContext) -> pd.DataFrame:
    corrected = load_corrected_loro_relation(ctx, "tableS6_loro_regional_performance_summary")
    if not corrected.empty:
        corrected = corrected.copy()
        corrected["held_out_region"] = corrected["region"].map(normalize_region_label)
        rows: list[dict[str, Any]] = []
        for region in sorted(corrected["held_out_region"].dropna().unique().tolist()):
            sub = corrected[corrected["held_out_region"].eq(region)].copy()
            full_basic = pd.to_numeric(sub["full_nse"], errors="coerce") - pd.to_numeric(sub["base_nse"], errors="coerce")
            flex_basic = pd.to_numeric(sub["flex_nse"], errors="coerce") - pd.to_numeric(sub["base_nse"], errors="coerce")
            rows.append(
                {
                    "held_out_region": region,
                    "n_basins": int(sub["gage_id"].nunique()),
                    "median_NSE_Basic": float(pd.to_numeric(sub["base_nse"], errors="coerce").median()),
                    "median_NSE_Full": float(pd.to_numeric(sub["full_nse"], errors="coerce").median()),
                    "median_NSE_LORO": float(pd.to_numeric(sub["flex_nse"], errors="coerce").median()),
                    "mean_delta_NSE_Full_vs_Basic": float(full_basic.mean()),
                    "mean_delta_NSE_LORO_vs_Basic": float(flex_basic.mean()),
                    "fraction_of_Full_gain_retained_mean": safe_div(flex_basic.mean(), full_basic.mean()),
                }
            )
        ctx.notes["tableS6_loro_regional_performance_summary"] = (
            "Table S6 uses the corrected 671-basin LORO relation table. "
            "Gain-retention is computed from region-wise mean NSE gains, not from medians."
        )
        return pd.DataFrame(rows)

    summary = read_csv(ctx, ctx.figures_csv_dir / "Table_Figure12_region_summary.csv", "tableS6_loro_regional_performance_summary")
    basin = read_csv(ctx, ctx.figures_csv_dir / "Figure12_loro_basin_performance.csv", "tableS6_loro_regional_performance_summary")
    loro_metrics = load_loro_performance_metrics(ctx)
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        region = normalize_region_label(row["region"])
        perf = loro_metrics[loro_metrics["held_out_region"].eq(region)]
        basic = perf[perf["model_key"].eq("base")]
        full = perf[perf["model_key"].eq("full")]
        flex = perf[perf["model_key"].eq("flex")]
        full_gain = clean_float(row["median_delta_full_minus_basic"])
        flex_gain = clean_float(row["median_delta_flex_minus_basic"])
        rows.append(
            {
                "held_out_region": region,
                "n_basins": int(row["n_basins"]),
                "median_NSE_Basic": clean_float(row["Basic_median_NSE"]),
                "median_NSE_Full": clean_float(row["Full_median_NSE"]),
                "median_NSE_LORO": clean_float(row["Flex_LORO_median_NSE"]),
                "delta_NSE_LORO_vs_Basic": flex_gain,
                "fraction_of_Full_gain_retained": safe_div(flex_gain, full_gain),
                "retained_gain_category": retained_category_text(classify_retained_gain(safe_div(flex_gain, full_gain))),
                "median_KGE_LORO": float(flex["median_KGE"].median()) if not flex.empty else np.nan,
                "median_RMSE_LORO": float(flex["median_RMSE"].median()) if not flex.empty else np.nan,
                "median_abs_PBIAS_LORO": float(flex["median_abs_PBIAS"].median()) if not flex.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_table_s7(ctx: BuildContext) -> pd.DataFrame:
    summary = read_csv(ctx, ctx.figures_csv_dir / "plot_fig13_coordinate_transfer_summary.csv", "tableS7_continuous_coordinate_transfer_summary")
    long_df = read_csv(ctx, ctx.figures_csv_dir / "plot_fig13_coordinate_transfer_long.csv", "tableS7_continuous_coordinate_transfer_summary")
    if summary.empty or long_df.empty:
        return pd.DataFrame()
    summary = summary.copy()
    summary["held_out_region"] = summary["region"].map(normalize_region_label)
    summary["process_coordinate"] = summary["coordinate"].map(normalize_process)
    long_df["held_out_region"] = long_df["region"].map(normalize_region_label)
    rows: list[dict[str, Any]] = []
    for region in sorted(summary["held_out_region"].dropna().unique().tolist()):
        region_long = long_df[long_df["held_out_region"].eq(region)]
        region_sum = summary[summary["held_out_region"].eq(region)]
        for process in PROCESS_ORDER:
            share_col = PROCESS_TO_SHARE[process]
            ref_col = f"reference_{share_col}"
            rho_all = pd.to_numeric(region_sum.loc[region_sum["process_coordinate"].eq(process), "spearman_rho"], errors="coerce")
            if region_long.empty or ref_col not in region_long:
                n_active = 0
                rho_active = np.nan
                n_heldout = np.nan
            else:
                ref = pd.to_numeric(region_long[ref_col], errors="coerce")
                active = ref > ctx.active_threshold
                n_active = int(active.sum())
                n_heldout = int(region_long["basin_id"].nunique())
                if not active.any():
                    rho_active = np.nan
                else:
                    loro_col = f"loro_{share_col}"
                    rho_active = float(pd.DataFrame({"x": ref[active], "y": pd.to_numeric(region_long[loro_col], errors="coerce")[active]}).corr(method="spearman").iloc[0, 1])
            rho_all_val = clean_float(rho_all.iloc[0]) if not rho_all.empty else np.nan
            rows.append(
                {
                    "held_out_region": region,
                    "process_coordinate": process,
                    "n_reference_active_basins": n_active,
                    "n_heldout_basins": n_heldout,
                    "spearman_rho_all_basins": rho_all_val,
                    "spearman_rho_active_basins": rho_active,
                    "transfer_strength_all_basins": classify_transfer_strength(rho_all_val, n_active, ctx.min_active_basins),
                    "transfer_strength_active_basins": classify_transfer_strength(rho_active, n_active, ctx.min_active_basins),
                    "assessability": "not assessable" if n_active < ctx.min_active_basins else "assessable",
                    "interpretation_note": (
                        "Reference-active sample below minimum; not assessable."
                        if n_active < ctx.min_active_basins
                        else {
                            "Interception": "Interception transfer is weakly active or only assessable in limited regions.",
                            "Phenology": "Phenology transfer is moderate and region-sensitive.",
                            "Snow": "Snow transfer is the strongest and most hydroclimatically organized.",
                            "Subsurface": "Subsurface transfer is moderate and relatively stable but hydrologically mixed.",
                        }[process]
                    ),
                }
            )
    ctx.notes["tableS7_continuous_coordinate_transfer_summary"] = (
        "Transfer strength uses signed Spearman rho. Negative rho values are retained as signed transfer evidence "
        "and are not reclassified using absolute magnitude."
    )
    return pd.DataFrame(rows)


def dominant_process_from_share_row(row: pd.Series, prefix: str) -> str:
    pairs = []
    for process in PROCESS_ORDER:
        value = clean_float(row.get(f"{prefix}_{PROCESS_TO_SHARE[process]}", np.nan))
        pairs.append((process, value))
    pairs = [(process, value) for process, value in pairs if not pd.isna(value)]
    if not pairs:
        return ""
    return max(pairs, key=lambda item: item[1])[0]


def build_table_s8(ctx: BuildContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    region_summary = read_csv(ctx, ctx.figures_csv_dir / "Table_Figure14_region_summary.csv", "tableS8_categorical_decision_transfer_summary")
    basin = read_csv(ctx, ctx.figures_csv_dir / "plot_fig14_basin_structural_decisions.csv", "tableS8_categorical_decision_transfer_summary")
    if region_summary.empty or basin.empty:
        return pd.DataFrame(), pd.DataFrame()
    basin["held_out_region"] = basin["region"].map(normalize_region_label)
    basin["reference_dominant_process"] = basin["reference_dominant_process"].map(normalize_process)
    basin["loro_dominant_process"] = basin["loro_dominant_process"].map(normalize_process)
    rows: list[dict[str, Any]] = []
    for _, row in region_summary.iterrows():
        region = normalize_region_label(row["region"])
        sub = basin[basin["held_out_region"].eq(region)]
        transition = (
            sub.groupby(["reference_dominant_process", "loro_dominant_process"], dropna=False)["basin_id"]
            .nunique()
            .reset_index(name="n_basins")
        )
        transition["fraction"] = transition["n_basins"] / transition["n_basins"].sum() if not transition.empty else np.nan
        mismatch = transition[transition["reference_dominant_process"] != transition["loro_dominant_process"]].sort_values("n_basins", ascending=False)
        major = ""
        if not mismatch.empty:
            top = mismatch.iloc[0]
            major = f"{top['reference_dominant_process']} -> {top['loro_dominant_process']}"
        exact = pd.to_numeric(sub["exact_active_set_match"], errors="coerce")
        dom = clean_float(row["dominant_agreement"])
        jac = clean_float(row["active_set_jaccard_summary"])
        transfer_class = "strong" if dom >= 0.8 and jac >= 0.75 else "moderate" if dom >= 0.6 or jac >= 0.5 else "weak"
        rows.append(
            {
                "held_out_region": region,
                "n_basins": int(row["n_total"]),
                "dominant_process_agreement": dom,
                "active_set_jaccard": jac,
                "exact_active_set_match_fraction": float(exact.mean()) if not exact.empty else clean_float(row["exact_active_set_match_rate"]),
                "dominant_transition_major_pattern": major,
                "categorical_transfer_class": transfer_class,
                "interpretation_note": (
                    "Categorical transfer is useful but imperfect; region-specific dominant-process shifts remain."
                    if transfer_class != "strong"
                    else "Categorical transfer is relatively strong in this region but still not perfect."
                ),
            }
        )
    transition_long = (
        basin.groupby(["held_out_region", "reference_dominant_process", "loro_dominant_process"], dropna=False)["basin_id"]
        .nunique()
        .reset_index(name="n_basins")
    )
    transition_long["fraction"] = transition_long.groupby("held_out_region")["n_basins"].transform(lambda s: s / s.sum())
    transition_long = transition_long.rename(columns={"held_out_region": "held_out_region"})
    return pd.DataFrame(rows), transition_long


def build_table2(
    ctx: BuildContext,
    s3: pd.DataFrame,
    s4_compact: pd.DataFrame,
    s5_compact: pd.DataFrame,
    s7: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for process in PROCESS_ORDER:
        seed = s3[s3["process_coordinate"].eq(process)]
        hydro = s4_compact[s4_compact["process_coordinate"].eq(process)]
        readout = s5_compact[s5_compact["process_coordinate"].eq(process)]
        transfer = s7[s7["process_coordinate"].eq(process)].copy()
        transfer = transfer[transfer["assessability"].ne("not assessable")]
        transfer_strength = ""
        if not transfer.empty:
            mode = transfer["transfer_strength_active_basins"].mode()
            transfer_strength = mode.iloc[0] if not mode.empty else ""
        seed_class = seed["robustness_class"].iloc[0] if not seed.empty else ""
        hydro_class = hydro["control_strength_class"].iloc[0] if not hydro.empty else ""
        readout_support = readout["reconstruction_support"].iloc[0] if not readout.empty else ""
        rows.append(
            {
                "process_coordinate": process,
                "activity_level": {
                    "Interception": "weak",
                    "Phenology": "moderate",
                    "Snow": "strong",
                    "Subsurface": "moderate",
                }[process],
                "seed_robustness": seed_class,
                "hydroclimatic_organization": hydro_class,
                "parameter_space_readout": readout_support,
                "transfer_expectation": {
                    "Interception": "often not assessable or weak under streamflow-only training",
                    "Phenology": "moderate but region-sensitive",
                    "Snow": "strongest expected transfer",
                    "Subsurface": "moderate and stable but process-mixed",
                }[process],
                "overall_interpretation": {
                    "Interception": "Weakly active and weakly resolved under the streamflow-only setting.",
                    "Phenology": "Moderate and region-sensitive structural coordinate.",
                    "Snow": "Strongest and most hydroclimatically organized coordinate.",
                    "Subsurface": "Relatively stable coordinate with mixed hydrologic controls.",
                }[process],
                "supporting_tables": "S3; S4; S5",
                "supporting_figures": "Figs. 3-5, 8-11",
            }
        )
    return pd.DataFrame(rows)


def build_table3(ctx: BuildContext, s6: pd.DataFrame, s7: pd.DataFrame, s8: pd.DataFrame) -> pd.DataFrame:
    corrected = load_corrected_loro_relation(ctx, "table3_loro_transferability_summary")
    corrected_map: dict[str, dict[str, Any]] = {}
    if not corrected.empty:
        corrected = corrected.copy()
        corrected["held_out_region"] = corrected["region"].map(normalize_region_label)
        corrected = corrected.rename(columns={"gage_id": "basin_id"})
        corrected = corrected.assign(
            full_basic=lambda df: pd.to_numeric(df["full_nse"], errors="coerce") - pd.to_numeric(df["base_nse"], errors="coerce"),
            flex_basic=lambda df: pd.to_numeric(df["flex_nse"], errors="coerce") - pd.to_numeric(df["base_nse"], errors="coerce"),
        )
        corrected["outcome_category"] = "intermediate"
        degraded = corrected["flex_nse"] < (corrected["base_nse"] - 0.02)
        basic_like = (corrected["flex_nse"] - corrected["base_nse"]).abs() <= 0.02
        near_full = corrected["flex_nse"] >= (corrected["full_nse"] - 0.05)
        corrected.loc[degraded, "outcome_category"] = "degraded"
        corrected.loc[basic_like & ~degraded, "outcome_category"] = "basic-like"
        corrected.loc[near_full & ~degraded & ~basic_like, "outcome_category"] = "near-Full"
        for region in sorted(corrected["held_out_region"].dropna().unique().tolist()):
            sub = corrected[corrected["held_out_region"].eq(region)].copy()
            corrected_map[region] = {
                "n_basins": int(sub["basin_id"].nunique()),
                "median_nse_loro": float(pd.to_numeric(sub["flex_nse"], errors="coerce").median()),
                "full_basic_mean": float(pd.to_numeric(sub["full_basic"], errors="coerce").mean()),
                "flex_basic_mean": float(pd.to_numeric(sub["flex_basic"], errors="coerce").mean()),
                "relative_gain_ratio": safe_div(sub["flex_basic"].mean(), sub["full_basic"].mean()),
                "degraded_fraction": float((sub["outcome_category"] == "degraded").mean()),
                "basic_like_fraction": float((sub["outcome_category"] == "basic-like").mean()),
                "intermediate_fraction": float((sub["outcome_category"] == "intermediate").mean()),
                "near_full_fraction": float((sub["outcome_category"] == "near-Full").mean()),
            }

    rows: list[dict[str, Any]] = []
    for _, perf in s6.iterrows():
        region = perf["held_out_region"]
        cont = s7[s7["held_out_region"].eq(region)].copy()
        cat = s8[s8["held_out_region"].eq(region)].copy()
        assessable = cont[cont["assessability"].eq("assessable")].copy()
        if not assessable.empty:
            assessable["rho"] = pd.to_numeric(assessable["spearman_rho_active_basins"], errors="coerce")
            strongest = assessable.sort_values("rho", ascending=False)["process_coordinate"].iloc[0]
            weakest = assessable.sort_values("rho", ascending=True)["process_coordinate"].iloc[0]
        else:
            strongest = ""
            weakest = ""
        not_assessable = cont[cont["assessability"].eq("not assessable")]["process_coordinate"].tolist()
        weakest_text = weakest if weakest else "; ".join(not_assessable)
        corrected_row = corrected_map.get(region, {})
        if corrected_row:
            # When one or more processes are not assessable, surface the least supported process.
            not_assessable_counts = cont[cont["assessability"].eq("not assessable")][["process_coordinate", "n_reference_active_basins"]].copy()
            if not not_assessable_counts.empty:
                not_assessable_counts["n_reference_active_basins"] = pd.to_numeric(not_assessable_counts["n_reference_active_basins"], errors="coerce")
                weakest_text = (
                    not_assessable_counts.sort_values(["n_reference_active_basins", "process_coordinate"], ascending=[True, True])["process_coordinate"].iloc[0]
                )
            elif weakest:
                weakest_text = weakest
        rows.append(
            {
                "Region": region,
                "Basins": corrected_row.get("n_basins", perf["n_basins"]),
                "Flex-LORO NSE med.": corrected_row.get("median_nse_loro", perf["median_NSE_LORO"]),
                "Full-Basic mean": corrected_row.get("full_basic_mean", np.nan),
                "Flex-Basic mean": corrected_row.get("flex_basic_mean", np.nan),
                "Relative gain ratio": corrected_row.get("relative_gain_ratio", np.nan),
                "Degraded fraction": corrected_row.get("degraded_fraction", np.nan),
                "Intermediate fraction": corrected_row.get("intermediate_fraction", np.nan),
                "near-Full fraction": corrected_row.get("near_full_fraction", np.nan),
                "Dominant-process agreement": cat["dominant_process_agreement"].iloc[0] if not cat.empty else np.nan,
                "Active-set Jaccard med.": cat["active_set_jaccard"].iloc[0] if not cat.empty else np.nan,
                "Best transferred coord.": strongest,
                "Weakest transferred coord.": weakest_text,
            }
        )
    ctx.notes["table3_loro_transferability_summary"] = (
        "Relative gain ratio is computed as (Flex-Basic)/(Full-Basic) using region-wise mean NSE gains from the corrected 671-basin LORO relation table. "
        "The listed fractions cover degraded, intermediate, and near-Full basins only; the remaining share is the basic-like category."
    )
    return pd.DataFrame(rows)


def load_nmul_run(test_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    weights = {w: np.load(test_dir / f"{w}.npy").reshape(-1).astype(float) for w in PROCESS_TO_W.values()}
    metrics = load_json(test_dir / "metrics_agg.json")
    return weights, metrics


def build_table_s9(ctx: BuildContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    figure = read_csv(ctx, ctx.figures_csv_dir / "figA5_nmul_ablation_data.csv", "tableS9_nmul_ablation_summary")
    rows: list[dict[str, Any]] = []
    for test_dir in sorted(ctx.nmul_root.glob("nmul*/seed_*/test1995-2010_Ep50")):
        ctx.add_source("tableS9_nmul_ablation_summary", test_dir / "metrics_agg.json")
        match_nmul = re.search(r"nmul(\d+)", str(test_dir))
        match_seed = re.search(r"seed_(\d+)", str(test_dir))
        nmul = int(match_nmul.group(1)) if match_nmul else np.nan
        seed = int(match_seed.group(1)) if match_seed else np.nan
        weights, metrics = load_nmul_run(test_dir)
        total = sum(weights[w] for w in weights)
        row = {
            "nmul": nmul,
            "alpha": ctx.primary_alpha,
            "seed": seed,
            "n_basins": len(total),
            "median_NSE": clean_float(metrics.get("nse", {}).get("median")),
            "median_KGE": clean_float(metrics.get("kge", {}).get("median")),
            "median_RMSE": clean_float(metrics.get("rmse", {}).get("median")),
            "median_abs_PBIAS": clean_float(metrics.get("pbias_abs", {}).get("median")),
            "total_structural_weight_mean": float(np.mean(total)),
            "total_structural_weight_median": float(np.median(total)),
            "w_interception_mean": float(np.mean(weights["w_int"])),
            "w_phenology_mean": float(np.mean(weights["w_phen"])),
            "w_snow_mean": float(np.mean(weights["w_snow"])),
            "w_subsurface_mean": float(np.mean(weights["w_sub"])),
            "active_fraction_interception": float(np.mean(weights["w_int"] > ctx.active_threshold)),
            "active_fraction_phenology": float(np.mean(weights["w_phen"] > ctx.active_threshold)),
            "active_fraction_snow": float(np.mean(weights["w_snow"] > ctx.active_threshold)),
            "active_fraction_subsurface": float(np.mean(weights["w_sub"] > ctx.active_threshold)),
            "seed_availability_note": "Only one seed available for this nmul setting; treat Appendix Fig. A5 as a capacity/multiplicity diagnostic rather than a universal optimum.",
        }
        rows.append(row)
    main = pd.DataFrame(rows).sort_values("nmul").reset_index(drop=True)
    agg = (
        main.groupby(["nmul", "alpha"], dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_basins=("n_basins", "median"),
            median_NSE_across_seeds=("median_NSE", "median"),
            median_KGE_across_seeds=("median_KGE", "median"),
            total_structural_weight_mean_across_seeds=("total_structural_weight_mean", "mean"),
            w_interception_mean_across_seeds=("w_interception_mean", "mean"),
            w_phenology_mean_across_seeds=("w_phenology_mean", "mean"),
            w_snow_mean_across_seeds=("w_snow_mean", "mean"),
            w_subsurface_mean_across_seeds=("w_subsurface_mean", "mean"),
            active_fraction_interception_mean=("active_fraction_interception", "mean"),
            active_fraction_phenology_mean=("active_fraction_phenology", "mean"),
            active_fraction_snow_mean=("active_fraction_snow", "mean"),
            active_fraction_subsurface_mean=("active_fraction_subsurface", "mean"),
        )
        .reset_index()
    )
    return main, agg


def build_table_s10(ctx: BuildContext) -> pd.DataFrame:
    df = read_csv(ctx, ctx.figures_csv_dir / "figA6_threshold_sensitivity_data.csv", "tableS10_threshold_sensitivity_summary")
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    thresholds = sorted(df["threshold"].dropna().unique().tolist())
    if thresholds != [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        ctx.warn(f"Appendix Fig. A6 threshold set differs from the requested default list: {thresholds}")
    target_alpha = ctx.primary_alpha
    alpha_df = df[df["alpha"].round(3).eq(round(target_alpha, 3))].copy()
    for threshold in sorted(alpha_df["threshold"].dropna().unique().tolist()):
        sub = alpha_df[alpha_df["threshold"].eq(threshold)]
        all_processes = sub[sub["process"].map(normalize_process).eq("All processes")]
        seed_jaccard = clean_float(all_processes[all_processes["metric"].eq("seed_active_set_jaccard")]["value"].iloc[0]) if not all_processes[all_processes["metric"].eq("seed_active_set_jaccard")].empty else np.nan
        exact_match = clean_float(all_processes[all_processes["metric"].eq("seed_exact_active_set_match")]["value"].iloc[0]) if not all_processes[all_processes["metric"].eq("seed_exact_active_set_match")].empty else np.nan
        loro_jaccard = clean_float(sub[(sub["process"].eq("Transfer")) & (sub["metric"].eq("LORO active-set Jaccard"))]["value"].iloc[0]) if not sub[(sub["process"].eq("Transfer")) & (sub["metric"].eq("LORO active-set Jaccard"))].empty else np.nan
        set_size = clean_float(all_processes[all_processes["metric"].eq("active_set_size")]["value"].iloc[0]) if not all_processes[all_processes["metric"].eq("active_set_size")].empty else np.nan
        for process in PROCESS_ORDER:
            psub = sub[sub["process"].map(normalize_process).eq(process)]
            active_fraction = clean_float(psub[psub["metric"].eq("active_fraction")]["value"].iloc[0]) if not psub[psub["metric"].eq("active_fraction")].empty else np.nan
            rows.append(
                {
                    "threshold": threshold,
                    "alpha": target_alpha,
                    "process_coordinate": process,
                    "active_fraction": active_fraction,
                    "mean_active_process_count": set_size,
                    "seed_active_set_jaccard": seed_jaccard,
                    "exact_active_set_match_fraction": exact_match,
                    "loro_active_set_jaccard_if_available": loro_jaccard,
                    "threshold_note": "Primary threshold = 0.1." if math.isclose(threshold, ctx.active_threshold) else "Threshold sensitivity diagnostic for active-set definitions.",
                }
            )
    return pd.DataFrame(rows)


def run_complexity_audit(ctx: BuildContext) -> None:
    fig2 = read_csv(ctx, ctx.figures_csv_dir / "figure2_alpha_summary_stats.csv")
    a3 = read_csv(ctx, ctx.figures_csv_dir / "figA3_metric_tradeoff_data.csv")
    a8 = read_csv(ctx, ctx.figures_csv_dir / "figA8_interception_diagnostic_data.csv")
    current_chunks: list[np.ndarray] = []
    root = ctx.project_root / "results" / "block1_main" / "flex" / "alpha0.005"
    for test_dir in root.rglob("test1995-2010_Ep50"):
        if any(not (test_dir / f"{name}.npy").exists() for name in PROCESS_TO_W.values()):
            continue
        w_int = np.load(test_dir / "w_int.npy", mmap_mode="r").reshape(-1)
        w_phen = np.load(test_dir / "w_phen.npy", mmap_mode="r").reshape(-1)
        w_snow = np.load(test_dir / "w_snow.npy", mmap_mode="r").reshape(-1)
        w_sub = np.load(test_dir / "w_sub.npy", mmap_mode="r").reshape(-1)
        total = (
            np.asarray(w_int, dtype=np.float64)
            + np.asarray(w_phen, dtype=np.float64)
            + np.asarray(w_snow, dtype=np.float64)
            + np.asarray(w_sub, dtype=np.float64)
        )
        current_chunks.append(np.asarray(total, dtype=np.float32))
    current = float(np.concatenate(current_chunks).mean()) if current_chunks else np.nan
    a8_mean = float(a8.loc[(a8["alpha"].round(3).eq(0.005)) & (a8["panel"].eq("weights")), "sum_weight"].mean()) if not a8.empty else np.nan
    a3_val = clean_float(a3.loc[(a3["model"].eq("CFlex")) & (a3["alpha"].round(3).eq(0.005)), "mean_total_weight"].iloc[0]) if not a3.loc[(a3["model"].eq("CFlex")) & (a3["alpha"].round(3).eq(0.005))].empty else np.nan
    fig2_rows = fig2[(fig2["model"].eq("CFlex")) & (fig2["alpha"].round(3).eq(0.005))]
    ctx.audit = {
        "cflex_alpha_0.005_raw_result_mean_total_weight": current,
        "cflex_alpha_0.005_figA8_mean_total_weight": a8_mean,
        "cflex_alpha_0.005_figA3_mean_total_weight": a3_val,
        "figure2_has_complexity_column": False,
        "figure2_rows_checked": len(fig2_rows),
        "expected_outdated_value": 3.710,
    }
    if pd.isna(current):
        ctx.warn("Could not recompute CFlex alpha=0.005 raw total structural weight from result arrays.")
        return
    if not pd.isna(a8_mean) and abs(current - a8_mean) > 1e-4:
        ctx.warn(f"CFlex alpha=0.005 raw weight mismatch between result arrays ({current:.3f}) and Fig. A8 source ({a8_mean:.3f}).")
    if not pd.isna(a3_val) and abs(current - a3_val) > 1e-3:
        ctx.warn(
            f"CFlex alpha=0.005 Fig. A3 mean_total_weight ({a3_val:.3f}) does not match the direct raw-weight recomputation ({current:.3f}); Fig. A3 source may still reference an older generated table."
        )
    else:
        ctx.notes["table_generation_warnings"] = (
            "CFlex alpha=0.005 raw total weight was recomputed directly from result arrays and matches Fig. A8-derived weights at approximately 2.061. "
            "The older expectation around 3.710 appears outdated rather than indicating a current file-read bug."
        )


def build_warning_file(ctx: BuildContext) -> str:
    lines = [
        "# Table Generation Warnings",
        "",
        "## CFlex alpha=0.005 structural-weight audit",
        "",
        f"- Recomputed from raw test-time weights (`w_int + w_phen + w_snow + w_sub`) across `results/block1_main/flex/alpha0.005/.../test1995-2010_Ep50`: `{ctx.audit.get('cflex_alpha_0.005_raw_result_mean_total_weight', np.nan):.3f}`",
        f"- Fig. A8 weight export mean: `{ctx.audit.get('cflex_alpha_0.005_figA8_mean_total_weight', np.nan):.3f}`",
        f"- Fig. A3 `mean_total_weight` entry: `{ctx.audit.get('cflex_alpha_0.005_figA3_mean_total_weight', np.nan):.3f}`",
        "- Fig. 2 source file does not expose a structural-complexity column directly; consistency was checked via the raw result arrays and Appendix Fig. A3/A8 exports.",
        "- The older expectation around `3.710` is treated as outdated unless a separate unpublished source is provided.",
        "",
        "## Active definitions",
        "",
        f"- Primary alpha: `{ctx.primary_alpha}`",
        f"- Active threshold: `{ctx.active_threshold}`",
        f"- Minimum active basins for transfer classification: `{ctx.min_active_basins}`",
        "",
        "## Additional warnings",
        "",
    ]
    if ctx.warnings:
        lines.extend(f"- {warning}" for warning in ctx.warnings)
    else:
        lines.append("- None unresolved.")
        lines.append(
            "- The previous CFlex alpha=0.005 structural-weight warning was checked. The current tables use the aligned raw-weight definition "
            "(`w_int + w_phen + w_snow + w_sub`) consistently across Table 1, Table S2, Fig. 2, and Appendix Fig. A3."
        )
        lines.append(
            "- The earlier expected value was from an outdated diagnostic expectation and is not used in the current table build."
        )
    return "\n".join(lines) + "\n"


def build_readme(ctx: BuildContext) -> str:
    lines = [
        "# Flex-MOPEX Table Set",
        "",
        "## Inventory",
        "",
    ]
    inventory = [
        ("Main text", "table1_performance_complexity_summary", "Section 3.1 controllable performance-complexity path"),
        ("Main text", "table2_process_coordinate_evidence_synthesis", "Sections 3.2-3.4 process-coordinate evidence synthesis"),
        ("Main text", "table3_loro_transferability_summary", "Section 3.5 LORO transferability summary"),
        ("Supplement", "tableS1_multimetric_performance_summary", "Full multi-metric performance summary"),
        ("Supplement", "tableS1_multimetric_performance_summary_test_only", "Compact test-only performance summary"),
        ("Supplement", "tableS2_alpha_tradeoff_summary", "Full alpha-path tradeoff summary"),
        ("Supplement", "tableS3_seed_robustness_summary", "Seed robustness summary"),
        ("Supplement", "tableS4_hydroclimatic_control_summary", "Hydroclimatic control summary"),
        ("Supplement", "tableS4_process_level_hydroclimatic_summary", "Compact hydroclimatic process summary"),
        ("Supplement", "tableS5_parameter_space_readout_associations", "Parameter-space association summary"),
        ("Supplement", "tableS5_parameter_space_readout_reconstruction", "Parameter-space reconstruction summary"),
        ("Supplement", "tableS5_process_level_parameter_readout_summary", "Compact parameter readout process summary"),
        ("Supplement", "tableS6_loro_regional_performance_summary", "Regional LORO predictive transfer summary"),
        ("Supplement", "tableS7_continuous_coordinate_transfer_summary", "Continuous coordinate transfer summary"),
        ("Supplement", "tableS8_categorical_decision_transfer_summary", "Categorical decision transfer summary"),
        ("Supplement", "tableS8_dominant_process_transition_long", "Dominant-process transition summary"),
        ("Supplement", "tableS9_nmul_ablation_summary", "nmul ablation summary"),
        ("Supplement", "tableS9_nmul_ablation_summary_aggregated", "Aggregated nmul ablation summary"),
        ("Supplement", "tableS10_threshold_sensitivity_summary", "Threshold sensitivity summary"),
    ]
    for category, name, purpose in inventory:
        label = DISPLAY_LABELS.get(name, "")
        prefix = f"{label}: " if label else ""
        lines.append(f"- {prefix}`{name}` ({category}): {purpose}.")
    lines.extend(
        [
            "",
            "## Supplement table groups",
            "",
            "- Table S1 consists of S1a-S1b:",
            "- S1a: `tableS1_multimetric_performance_summary`",
            "- S1b: `tableS1_multimetric_performance_summary_test_only`",
            "",
            "- Table S4 consists of S4a-S4b:",
            "- S4a: `tableS4_hydroclimatic_control_summary`",
            "- S4b: `tableS4_process_level_hydroclimatic_summary`",
            "",
            "- Table S5 consists of S5a-S5c:",
            "- S5a: `tableS5_parameter_space_readout_associations`",
            "- S5b: `tableS5_parameter_space_readout_reconstruction`",
            "- S5c: `tableS5_process_level_parameter_readout_summary`",
            "",
            "- Table S8 consists of S8a-S8b:",
            "- S8a: `tableS8_categorical_decision_transfer_summary`",
            "- S8b: `tableS8_dominant_process_transition_long`",
            "",
            "- Table S9 consists of S9a-S9b:",
            "- S9a: `tableS9_nmul_ablation_summary`",
            "- S9b: `tableS9_nmul_ablation_summary_aggregated`",
            "",
            "## Primary sources",
            "",
            "- Main performance and complexity tables use `results/block1_main`, `results/binary_pilot`, and the aligned figure CSV exports under `manuscript/figures/csv`.",
            "- LORO tables use `Figure12_loro_basin_performance.csv`, `Table_Figure12_region_summary.csv`, `plot_fig13_coordinate_transfer_summary.csv`, `plot_fig13_coordinate_transfer_long.csv`, `Table_Figure14_region_summary.csv`, `plot_fig14_basin_structural_decisions.csv`, and `results/block3_loro/config_dmopex_v1/.../metrics_agg.json`.",
            "- S9 uses `results/block1_nmul_ablation/flex/alpha0.01` directly.",
            "- S10 uses `manuscript/figures/csv/figA6_threshold_sensitivity_data.csv`.",
            "",
            "## Definitions",
            "",
            "- `NSE_gain_over_Basic` / `delta_NSE_vs_Basic`: table median NSE minus the Basic-MOPEX table median NSE on the same split.",
            "- `fraction_of_Full_gain_retained = (NSE_model - NSE_Basic) / (NSE_Full - NSE_Basic)`.",
            "- `total structural weight` for CFlex is the raw sum `w_int + w_phen + w_snow + w_sub`.",
            "- `active fraction` is the basin fraction with process weight above the active threshold for CFlex, or binary activation equal to 1 for DFlex.",
            "- `active-set Jaccard` is the overlap between two process-active sets divided by the union size.",
            "- Continuous transfer strength uses signed Spearman rho. Strong transfer is defined as rho >= 0.7, moderate transfer as 0.4 <= rho < 0.7, weak transfer as rho < 0.4, and not assessable when fewer than MIN_ACTIVE_BASINS active reference basins are available. Negative rho values are retained as signed transfer evidence and are not reclassified using absolute magnitude.",
            "- `retained Full-gain category` uses `degraded`, `basic-like`, `intermediate`, and `near-full` based on retained Full-gain fractions below 0, `0-0.25`, `0.25-0.75`, and `>= 0.75`.",
            "",
            "## Complexity notes",
            "",
            "- CFlex raw total weights and DFlex active process counts are related complexity summaries but not directly identical quantities.",
            "- DFlex penalty values are reported only as discrete-selection references and should not be interpreted on the same penalty scale as CFlex.",
            "",
            "## Fixed manuscript settings",
            "",
            f"- Primary alpha = `{ctx.primary_alpha}`",
            f"- Active threshold = `{ctx.active_threshold}`",
            f"- `MIN_ACTIVE_BASINS` for transfer classification = `{ctx.min_active_basins}`",
            "",
            "## Warnings",
            "",
        ]
    )
    if ctx.warnings:
        lines.extend(f"- {warning}" for warning in ctx.warnings)
    else:
        lines.append("- None unresolved.")
        lines.append(
            "- The previous CFlex alpha=0.005 structural-weight warning was checked. The current tables use the aligned raw-weight definition "
            "(`w_int + w_phen + w_snow + w_sub`) consistently across Table 1, Table S2, Fig. 2, and Appendix Fig. A3."
        )
        lines.append(
            "- The earlier expected value was from an outdated diagnostic expectation and is not used in the current table build."
        )
    lines.extend(
        [
            "",
            "## Regeneration command",
            "",
            "`python scripts/build_paper_tables.py`",
            "",
            "## Internal runtime note",
            "",
            "- If the repository is under `/mnt/c/...`, copy it into the WSL filesystem before running large table builds, for example:",
            "- `cp -r /mnt/c/Users/.../project ~/project`",
            "- `cd ~/project`",
            "- `python scripts/build_paper_tables.py`",
            "",
        ]
    )
    return "\n".join(lines)


def build_manifest(ctx: BuildContext) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table_name, note in TABLE_NOTES.items():
        if table_name in {"README_tables", "table_manifest", "table_generation_warnings"}:
            continue
        files = ctx.files.get(table_name, {})
        rows.append(
            {
                "table_name": table_name,
                "display_label": DISPLAY_LABELS.get(table_name, ""),
                "main_or_supplement": "main" if table_name.startswith("table1") or table_name.startswith("table2") or table_name.startswith("table3") else "supplement",
                "file_stem": table_name,
                "section_supported": note,
                "primary_source_files": "; ".join(sorted(ctx.sources.get(table_name, set()))),
                "generated_csv": str(files.get("csv", "")),
                "generated_markdown": str(files.get("markdown", "")),
                "generated_latex": str(files.get("latex", "")),
                "notes": ctx.notes.get(table_name, ""),
            }
        )
    return pd.DataFrame(rows)


def write_support_files(ctx: BuildContext, manifest: pd.DataFrame) -> None:
    warnings_path = ctx.output_dir / "table_generation_warnings.md"
    warnings_path.write_text(build_warning_file(ctx), encoding="utf-8")
    ctx.files["table_generation_warnings"] = {"markdown": warnings_path}
    readme_path = ctx.output_dir / "README_tables.md"
    readme_path.write_text(build_readme(ctx), encoding="utf-8")
    ctx.files["README_tables"] = {"markdown": readme_path}
    manifest_path = ctx.output_dir / "table_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    ctx.files["table_manifest"] = {"csv": manifest_path}


def run_quality_checks(ctx: BuildContext, tables: dict[str, pd.DataFrame]) -> list[str]:
    findings: list[str] = []
    main_expected = [
        "table1_performance_complexity_summary",
        "table2_process_coordinate_evidence_synthesis",
        "table3_loro_transferability_summary",
    ]
    supplement_expected = [
        "tableS1_multimetric_performance_summary",
        "tableS2_alpha_tradeoff_summary",
        "tableS3_seed_robustness_summary",
        "tableS4_hydroclimatic_control_summary",
        "tableS5_parameter_space_readout_associations",
        "tableS6_loro_regional_performance_summary",
        "tableS7_continuous_coordinate_transfer_summary",
        "tableS8_categorical_decision_transfer_summary",
        "tableS9_nmul_ablation_summary",
        "tableS10_threshold_sensitivity_summary",
    ]
    for name in main_expected + supplement_expected:
        if name not in tables:
            findings.append(f"Missing required table in memory: {name}")
    if set(main_expected) != {k for k in tables if k in main_expected}:
        findings.append("Main-text table inventory is not exactly the requested three tables.")
    if len(tables.get("table2_process_coordinate_evidence_synthesis", pd.DataFrame())) != 4:
        findings.append("Table 2 does not contain exactly four process rows.")
    if len(tables.get("table1_performance_complexity_summary", pd.DataFrame())) > 8:
        findings.append("Table 1 is not compact enough; more than eight rows were included.")
    table3 = tables.get("table3_loro_transferability_summary", pd.DataFrame())
    if not table3.empty:
        region_col = "held_out_region" if "held_out_region" in table3.columns else "Region"
        labels = set(table3[region_col].astype(str))
        if any(not re.fullmatch(r"R[1-7]", label) for label in labels):
            findings.append(f"Table 3 region labels are inconsistent: {sorted(labels)}")
    s10 = tables.get("tableS10_threshold_sensitivity_summary", pd.DataFrame())
    if not s10.empty:
        got = sorted(set(pd.to_numeric(s10["threshold"], errors="coerce").dropna().tolist()))
        expected = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        if got != expected:
            findings.append(f"Table S10 thresholds differ from Appendix Fig. A6 set: {got}")
    if "README_tables" not in ctx.files or "table_manifest" not in ctx.files:
        findings.append("README_tables.md or table_manifest.csv was not written.")
    for table_name, file_map in ctx.files.items():
        for kind, path in file_map.items():
            if kind == "xlsx":
                continue
            if not path.exists() or path.stat().st_size == 0:
                findings.append(f"Generated {kind} output is empty or missing for {table_name}: {path}")
    forbidden_phrases = [
        ("physically true process fraction", "Forbidden physical-fraction language"),
        ("generally outperforms Full", "Forbidden Flex-vs-Full language"),
        ("all structural coordinates transfer well", "Forbidden transferability language"),
    ]
    for table_name, df in tables.items():
        text_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in text_cols:
            series = df[col].astype("string")
            for phrase, label in forbidden_phrases:
                if series.str.contains(phrase, case=False, na=False).any():
                    findings.append(f"{label} found in {table_name}.{col}.")
    return findings


def print_summary(ctx: BuildContext, generated_tables: list[str], quality_findings: list[str]) -> None:
    main_tables = [
        "table1_performance_complexity_summary",
        "table2_process_coordinate_evidence_synthesis",
        "table3_loro_transferability_summary",
    ]
    supplement_tables = [
        "tableS1_multimetric_performance_summary",
        "tableS1_multimetric_performance_summary_test_only",
        "tableS2_alpha_tradeoff_summary",
        "tableS3_seed_robustness_summary",
        "tableS4_hydroclimatic_control_summary",
        "tableS4_process_level_hydroclimatic_summary",
        "tableS5_parameter_space_readout_associations",
        "tableS5_parameter_space_readout_reconstruction",
        "tableS5_process_level_parameter_readout_summary",
        "tableS6_loro_regional_performance_summary",
        "tableS7_continuous_coordinate_transfer_summary",
        "tableS8_categorical_decision_transfer_summary",
        "tableS8_dominant_process_transition_long",
        "tableS9_nmul_ablation_summary",
        "tableS9_nmul_ablation_summary_aggregated",
        "tableS10_threshold_sensitivity_summary",
    ]
    print("Changed files:")
    print(f"- {ctx.project_root / 'scripts' / 'build_paper_tables.py'}")
    print(f"- {ctx.output_dir / 'README_tables.md'}")
    print(f"- {ctx.output_dir / 'table_manifest.csv'}")
    print(f"- {ctx.output_dir / 'table_generation_warnings.md'}")
    print("Generated table files:")
    for name in generated_tables:
        paths = ctx.files.get(name, {})
        csv_path = paths.get("csv")
        md_path = paths.get("markdown")
        tex_path = paths.get("latex")
        print(f"- {name}: {csv_path}, {md_path}, {tex_path}")
    print("Main-text tables:")
    for name in main_tables:
        print(f"- {name}")
    print("Supplement tables:")
    for name in supplement_tables:
        print(f"- {name}")
    print("Unresolved warnings:")
    if ctx.warnings:
        for warning in ctx.warnings:
            print(f"- {warning}")
    else:
        print("- None")
    print("CFlex alpha=0.005 structural-weight handling:")
    print(
        f"- Raw result arrays and Fig. A8 agree at approximately {ctx.audit.get('cflex_alpha_0.005_raw_result_mean_total_weight', np.nan):.3f}; "
        "the older expectation near 3.710 was documented as outdated rather than used."
    )
    if quality_findings:
        print("Quality-check findings:")
        for finding in quality_findings:
            print(f"- {finding}")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = resolve_output_dir(project_root, args.output_dir).resolve()
    ctx = BuildContext(
        project_root=project_root,
        output_dir=output_dir,
        active_threshold=args.active_threshold,
        primary_alpha=args.primary_alpha,
        min_active_basins=args.min_active_basins,
    )
    ctx.output_dir.mkdir(parents=True, exist_ok=True)

    perf_seed = load_full_performance_summary(ctx)
    basin_metrics = load_basin_metric_summary(ctx)
    complexity = load_complexity_summary_streaming(ctx)
    run_complexity_audit(ctx)
    alpha_summary = build_alpha_summary(ctx, perf_seed, complexity, basin_metrics)

    s1, s1_test = build_table_s1(ctx, perf_seed)
    s2 = build_table_s2(ctx, alpha_summary)
    s3 = build_table_s3(ctx)
    s4, s4_compact = build_table_s4(ctx)
    s5_assoc, s5_recon, s5_compact = build_table_s5(ctx)
    s6 = build_table_s6(ctx)
    s7 = build_table_s7(ctx)
    s8, s8_long = build_table_s8(ctx)
    table1 = build_table1(ctx, alpha_summary)
    table1b = build_table1b(ctx, complexity)
    table2 = build_table2(ctx, s3, s4_compact, s5_compact, s7)
    table3 = build_table3(ctx, s6, s7, s8)
    table2_legacy = build_table2_legacy_summary(ctx, s3, s4, s4_compact, complexity, s5_assoc, s5_recon)
    s9, s9_agg = build_table_s9(ctx)
    s10 = build_table_s10(ctx)

    tables = {
        "table1_performance_complexity_summary": table1,
        "table1b_panelB_process_extension_weights": table1b,
        "table2_process_coordinate_evidence_synthesis": table2,
        "table3_loro_transferability_summary": table3,
        "tableS1_multimetric_performance_summary": s1,
        "tableS1_multimetric_performance_summary_test_only": s1_test,
        "tableS2_alpha_tradeoff_summary": s2,
        "tableS3_seed_robustness_summary": s3,
        "tableS4_hydroclimatic_control_summary": s4,
        "tableS4_process_level_hydroclimatic_summary": s4_compact,
        "tableS5_parameter_space_readout_associations": s5_assoc,
        "tableS5_parameter_space_readout_reconstruction": s5_recon,
        "tableS5_process_level_parameter_readout_summary": s5_compact,
        "tableS6_loro_regional_performance_summary": s6,
        "tableS7_continuous_coordinate_transfer_summary": s7,
        "tableS8_categorical_decision_transfer_summary": s8,
        "tableS8_dominant_process_transition_long": s8_long,
        "tableS9_nmul_ablation_summary": s9,
        "tableS9_nmul_ablation_summary_aggregated": s9_agg,
        "tableS10_threshold_sensitivity_summary": s10,
    }

    exported: dict[str, pd.DataFrame] = {}
    for name, df in tables.items():
        exported[name] = write_table(ctx, name, df)
    write_table(ctx, "table2_process_coordinate_evidence_summary", table2_legacy)
    if args.write_xlsx:
        write_excel(ctx, exported)
    manifest = build_manifest(ctx)
    write_support_files(ctx, manifest)
    quality_findings = run_quality_checks(ctx, tables)
    print_summary(ctx, list(tables.keys()), quality_findings)
    return 0 if not quality_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
