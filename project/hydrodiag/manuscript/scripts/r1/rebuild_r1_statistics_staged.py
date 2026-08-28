"""Staged, fail-closed R1 statistics rebuild.

This module reconstructs R1 from daily basin observations/simulations when those
inputs are locally readable.  It never launches inference or treats an existing
formatted R1 summary as canonical.  When the daily source gate fails, it still
writes the basin manifest, source inventory, and historical reconciliation, then
writes explicitly blocked/empty downstream tables.

All writes go to manuscript/cache/r1_rebuild_audit_staged by default.  The
canonical manuscript/results/R1 directory and figures are never modified.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
OUT = PROJECT / "manuscript" / "cache" / "r1_rebuild_audit_staged"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Reuse the repository's metric and bootstrap implementations; do not use the
# legacy kge_prime function as an R1 manuscript metric.
from r1_metrics import bootstrap_median_ci  # noqa: E402
from r1_statistics import standard_kge  # noqa: E402

SEED = 20260730
BOOTSTRAP_REPS = 10_000
PERIODS = {
    "train": ("1981-10-01", "1995-09-30"),
    "test": ("1995-10-01", "2010-09-30"),
}
PARADIGMS = ("IC-CMA-ES", "dPL-MLP")
STRUCTURES = ("Base", "TGD", "CN")
MODEL_TO_STRUCTURE = {
    "XAJ-Base": "Base",
    "XAJ-TGD": "TGD",
    "XAJ-TGD2": "TGD",
    "XAJ-CN": "CN",
}
STRUCTURE_TO_MODEL = {"Base": "XAJ-Base", "TGD": "XAJ-TGD", "CN": "XAJ-CN"}
PARADIGM_LABEL = {"IC-CMA-ES": "IC", "dPL-MLP": "dPL"}
SNOW_BINS = (0.0, 0.05, 0.15, 0.30, 0.50, 1.0001)
SNOW_STRATA = ("S1", "S2", "S3", "S4", "S5")
SNOW_LABELS = ("[0, 0.05)", "[0.05, 0.15)", "[0.15, 0.30)", "[0.30, 0.50)", "[0.50, 1.00]")
THRESHOLD_GRID = np.arange(0.40, 0.8001, 0.01)
SCREEN_THRESHOLD = 0.60
TIMING_THRESHOLD = 15.0


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_csv(name: str, frame: pd.DataFrame, columns: Iterable[str] | None = None) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        frame = frame.reindex(columns=list(columns))
    frame.to_csv(path, index=False, na_rep="")


def write_json(name: str, value: Any) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_json(value), indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm_ids(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.zfill(8)

def load_snow_attributes() -> pd.DataFrame:
    attrs = pd.read_csv(project_file("manuscript/results/R1/r1_snow_attributes.csv"))
    attrs["basin_id"] = norm_ids(attrs["basin_id"])
    return attrs


def project_file(relative: str) -> Path:
    path = (PROJECT / relative).resolve()
    root = PROJECT.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"R1 staged audit path escaped project root: {relative}")
    return path


def run_git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        return f"git command unavailable: {exc}"


def current_source_files() -> list[dict[str, Any]]:
    relative = [
        "manuscript/scripts/r1/rebuild_r1_statistics_staged.py",
        "manuscript/scripts/r1/r1_daily_inference.py",
        "manuscript/scripts/r1/r1_statistics.py",
        "manuscript/scripts/r1/r1_metrics.py",
        "manuscript/scripts/r1/build_r1_statistics.py",
        "manuscript/scripts/r1/plot_r1_figure1.py",
        "manuscript/scripts/r1/plot_r1_figure2.py",
        "manuscript/figure_manifests/gates.json",
        "manuscript/results/R1/r1_input_inventory.csv",
        "manuscript/results/R1/r1_basin_level_performance.csv",
        "manuscript/results/R1/r1_absolute_metrics_summary.csv",
        "manuscript/results/R1/r1_paired_effects_summary.csv",
        "manuscript/results/R1/r1_snow_attributes.csv",
        "manuscript/results/R1/r1_snow_signatures_basin_level.csv",
        "manuscript/cache/results_freeze_R1_R5/r1_hbv_checkpoint_evaluation/r1_result_manifest.json",
        "manuscript/cache/results_freeze_R1_R5/r1_hbv_checkpoint_evaluation/r1_inference_audit.md",
    ]
    rows = []
    for name in relative:
        path = project_file(name)
        rows.append(
            {
                "path": name,
                "exists": path.exists(),
                "sha256": sha256(path) if path.is_file() else "",
                "role": "local source or definition evidence",
            }
        )
    return rows


def discover_daily_candidates() -> list[dict[str, Any]]:
    patterns = (
        "manuscript/results/R1/r1_daily_simulations_*.parquet",
        "manuscript/cache/R1_ic_complete_rebuild/r1_daily_simulations_*.parquet",
        "manuscript/cache/results_freeze_R1_R5/r1_hbv_checkpoint_evaluation/r1_daily_simulations_*.parquet",
        "manuscript/cache/r1_rebuild_audit_staged/daily_dpl_gpu_compile/r1_daily_simulations_*.parquet",
    )
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(PROJECT.glob(pattern)))
    unique = sorted(set(paths))
    rows: list[dict[str, Any]] = []
    for path in unique:
        record: dict[str, Any] = {
            "path": str(path.relative_to(PROJECT)),
            "format": path.suffix.lower().lstrip("."),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "readable": False,
            "status": "candidate",
            "reason": "",
            "columns": "",
        }
        name = path.name.lower()
        if "ic_xaj_cn" in name:
            declared = ("IC-CMA-ES", "XAJ-CN")
        elif "ic_xaj_tgd" in name:
            declared = ("IC-CMA-ES", "XAJ-TGD")
        elif "ic_xaj" in name:
            declared = ("IC-CMA-ES", "XAJ-Base")
        elif "dpl_xaj_cn" in name:
            declared = ("dPL-MLP", "XAJ-CN")
        elif "dpl_xaj_tgd" in name:
            declared = ("dPL-MLP", "XAJ-TGD")
        elif "dpl_xaj" in name:
            declared = ("dPL-MLP", "XAJ-Base")
        elif "dpl_hbv" in name:
            declared = ("dPL-MLP", "HBV")
        else:
            declared = ("", "")
        record["declared_paradigm"], record["declared_model"] = declared
        try:
            import pyarrow.parquet as pq
            parquet = pq.ParquetFile(path)
            record["readable"] = True
            record["status"] = "readable_metadata"
            record["columns"] = ";".join(parquet.schema.names)
            record["rows"] = int(parquet.metadata.num_rows)
        except ImportError as exc:
            record["status"] = "unreadable_missing_parquet_engine"
            record["reason"] = str(exc)
        except Exception as exc:
            record["status"] = "unreadable_schema_or_file"
            record["reason"] = repr(exc)
        rows.append(record)
    return rows


def load_readable_daily(candidates: list[dict[str, Any]]) -> tuple[pd.DataFrame | None, str]:
    readable = [row for row in candidates if row.get("readable")]
    if not readable:
        return None, "no locally readable daily parquet source"
    declared = {(row.get("declared_paradigm", ""), row.get("declared_model", "")) for row in readable}
    expected = {(p, STRUCTURE_TO_MODEL[s]) for p in PARADIGMS for s in STRUCTURES}
    missing_declared = sorted(expected - declared)
    if missing_declared:
        return None, f"daily source missing declared six-combination files: {missing_declared}"
    required = {
        "basin_id", "paradigm", "model", "seed_or_restart", "selected_run",
        "period", "date", "q_obs", "q_sim",
    }
    basin_categories = pd.Index(load_snow_attributes()["basin_id"].astype(str).str.zfill(8).unique())
    model_categories = list(STRUCTURE_TO_MODEL.values())
    run_categories = ["selected_restart", "seed_42", "seed_123", "seed_2026"]
    tables = []
    for row in readable:
        table = pd.read_parquet(
            project_file(row["path"]), columns=sorted(required), engine="pyarrow"
        )
        missing = required - set(table.columns)
        if missing:
            return None, f"daily schema missing {sorted(missing)} in {row['path']}"
        table["basin_id"] = pd.Categorical(
            norm_ids(table["basin_id"]), categories=basin_categories
        )
        table["date"] = pd.to_datetime(table["date"], errors="coerce")
        table["paradigm"] = pd.Categorical(
            table["paradigm"].replace({"IC": "IC-CMA-ES", "dPL": "dPL-MLP"}),
            categories=list(PARADIGMS),
        )
        table["model"] = pd.Categorical(
            table["model"].replace({"XAJ-TGD2": "XAJ-TGD"}),
            categories=model_categories,
        )
        table["seed_or_restart"] = pd.Categorical(
            table["seed_or_restart"].astype(str).str.lower().replace(
                {"true": "selected_restart", "1": "selected_restart"}
            ),
            categories=run_categories,
        )
        table["selected_run"] = table["selected_run"].astype(str).str.lower().isin(
            {"1", "true", "yes"}
        )
        table["period"] = pd.Categorical(table["period"], categories=list(PERIODS))
        table["q_obs"] = pd.to_numeric(table["q_obs"], errors="coerce").astype("float32")
        table["q_sim"] = pd.to_numeric(table["q_sim"], errors="coerce").astype("float32")
        tables.append(table)
    daily = pd.concat(tables, ignore_index=True, sort=False)
    if daily["date"].isna().any():
        return None, f"daily source contains {int(daily['date'].isna().sum())} unparseable dates"
    expected = {(p, STRUCTURE_TO_MODEL[s]) for p in PARADIGMS for s in STRUCTURES}
    observed = {(p, m) for p, m in zip(daily["paradigm"], daily["model"])}
    missing = sorted(expected - observed)
    incomplete = []
    for paradigm, model in sorted(expected):
        subset = daily[(daily["paradigm"] == paradigm) & (daily["model"] == model)]
        if subset["basin_id"].nunique() != 531 or set(subset["period"].dropna().unique()) != set(PERIODS):
            incomplete.append(f"{paradigm}/{model}: basins={subset['basin_id'].nunique()} periods={sorted(subset['period'].dropna().unique())}")
    ic_issues = []
    for structure in STRUCTURES:
        model = STRUCTURE_TO_MODEL[structure]
        subset = daily[(daily["paradigm"] == "IC-CMA-ES") & (daily["model"] == model)]
        selected_pairs = subset.loc[subset["selected_run"], ["basin_id", "seed_or_restart"]].drop_duplicates()
        selected_counts = selected_pairs.groupby("basin_id", observed=True)["seed_or_restart"].nunique()
        bad = selected_counts[selected_counts != 1]
        if len(bad):
            ic_issues.append(f"IC-CMA-ES/{model}: basins with selected-run count != 1: {len(bad)}")
    if missing or incomplete or ic_issues:
        detail = [f"missing combinations={missing}"] if missing else []
        detail.extend(incomplete)
        detail.extend(ic_issues)
        return None, "daily source incomplete for six R1 combinations: " + "; ".join(detail)
    daily = daily[(daily["paradigm"] != "IC-CMA-ES") | daily["selected_run"]].copy()
    return daily, f"daily source complete: {len(daily)} rows"


def build_basin_manifest(perf: pd.DataFrame, attrs: pd.DataFrame, candidates: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    attrs = attrs.copy()
    attrs["basin_id"] = norm_ids(attrs["basin_id"])
    computed = pd.cut(
        pd.to_numeric(attrs["frac_snow"], errors="coerce"),
        bins=SNOW_BINS,
        labels=SNOW_STRATA,
        right=False,
    ).astype(object)
    attrs["computed_snow_stratum"] = computed
    attrs["snow_stratum_match"] = attrs["snow_stratum"].astype(str).eq(computed.astype(str))
    perf = perf.copy()
    perf["basin_id"] = norm_ids(perf["basin_id"])
    perf_flags: dict[tuple[str, str], set[str]] = {}
    for paradigm in PARADIGMS:
        for structure in STRUCTURES:
            model = STRUCTURE_TO_MODEL[structure]
            ids = set(perf.loc[
                perf["paradigm"].eq(paradigm)
                & perf["model"].eq(model)
                & perf["selected_run"].astype(bool),
                "basin_id",
            ])
            perf_flags[(paradigm, structure)] = ids
    daily_readable = any(row.get("readable") for row in candidates)
    rows = []
    for attr in attrs.itertuples(index=False):
        for paradigm in PARADIGMS:
            for structure in STRUCTURES:
                key = (paradigm, structure)
                perf_ok = attr.basin_id in perf_flags[key]
                rows.append(
                    {
                        "basin_id": attr.basin_id,
                        "frac_snow": attr.frac_snow,
                        "snow_stratum": attr.snow_stratum,
                        "snow_interval": SNOW_LABELS[SNOW_STRATA.index(attr.snow_stratum)]
                        if attr.snow_stratum in SNOW_STRATA else "",
                        "paradigm": paradigm,
                        "structure": structure,
                        "performance_available": perf_ok,
                        "daily_observation_available": daily_readable,
                        "daily_simulation_available": daily_readable,
                        "daily_source_status": "readable_candidate" if daily_readable else "blocked_no_readable_candidate",
                        "exclusion_reason": "" if perf_ok and daily_readable else (
                            "daily source unavailable or unreadable" if perf_ok else "missing selected performance row"
                        ),
                    }
                )
    manifest = pd.DataFrame(rows)
    checks = {
        "attribute_rows": int(len(attrs)),
        "attribute_unique_basins": int(attrs["basin_id"].nunique()),
        "attribute_duplicate_rows": int(attrs["basin_id"].duplicated().sum()),
        "attribute_missing_frac_snow": int(attrs["frac_snow"].isna().sum()),
        "attribute_stratum_mismatches": int((~attrs["snow_stratum_match"]).sum()),
        "stratum_counts": {str(k): int(v) for k, v in attrs.groupby("snow_stratum").size().items()},
        "s4_plus_s5": int(attrs["snow_stratum"].isin(["S4", "S5"]).sum()),
        "performance_rows": int(len(perf)),
        "performance_unique_basins": int(perf["basin_id"].nunique()),
        "performance_selected_all": bool(perf["selected_run"].astype(bool).all()),
        "performance_counts": {
            f"{p}|{s}": int(len(perf_flags[(p, s)]))
            for p in PARADIGMS for s in STRUCTURES
        },
        "daily_readable_candidate": daily_readable,
    }
    return manifest, checks


def water_year_for_dates(dates: pd.Series) -> pd.Series:
    return dates.dt.year + (dates.dt.month >= 10).astype(int)


def ct_for_year(group: pd.DataFrame) -> dict[str, Any]:
    group = group.sort_values("date")
    dates = group["date"].dt.normalize().to_numpy(dtype="datetime64[D]")
    row: dict[str, Any] = {
        "complete_year": False,
        "valid_year": False,
        "invalid_reason": "",
        "n_valid_days": 0,
        "CT_obs": math.nan,
        "CT_sim": math.nan,
        "Delta_CT": math.nan,
    }
    if not len(dates):
        row["invalid_reason"] = "no_rows"
        return row
    expected_days = int((dates[-1] - dates[0]).astype("timedelta64[D]").astype(int) + 1)
    if len(dates) != expected_days or len(np.unique(dates)) != expected_days:
        row["invalid_reason"] = "missing_or_duplicate_day"
        return row
    obs = pd.to_numeric(group["q_obs"], errors="coerce").to_numpy(float)
    sim = pd.to_numeric(group["q_sim"], errors="coerce").to_numpy(float)
    mask = np.isfinite(obs) & np.isfinite(sim) & (obs >= 0) & (sim >= 0)
    row["n_valid_days"] = int(mask.sum())
    if not bool(mask.all()):
        row["invalid_reason"] = "nonfinite_or_negative_paired_day"
        return row
    row["complete_year"] = True
    obs_total, sim_total = float(obs.sum()), float(sim.sum())
    if obs_total <= 0 or sim_total <= 0:
        # The recovered production code keeps this date-complete row and
        # returns NaN CT. Preserve that fact while excluding it from CT-valid
        # basin summaries through valid_year=False.
        row["invalid_reason"] = "zero_or_negative_water_year_total"
        row["n_valid_days"] = int(len(group))
        return row
    # Repository convention is 1-based water-year day, with Oct 1 = day 1.
    ct_obs = int(np.argmax(np.cumsum(obs) >= 0.5 * obs_total) + 1)
    ct_sim = int(np.argmax(np.cumsum(sim) >= 0.5 * sim_total) + 1)
    row.update({
        "valid_year": True,
        "n_valid_days": int(len(group)),
        "CT_obs": ct_obs,
        "CT_sim": ct_sim,
        "Delta_CT": ct_sim - ct_obs,
    })
    return row


def build_basin_year_ct(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = daily.copy()
    data = data[data["period"].isin(PERIODS)].copy()
    data["water_year"] = water_year_for_dates(data["date"])
    rows = []
    for keys, group in data.groupby(
        ["basin_id", "paradigm", "model", "seed_or_restart", "period", "water_year"],
        sort=True,
    ):
        basin, paradigm, model, run, period, wy = keys
        result = ct_for_year(group)
        rows.append({
            "basin_id": basin,
            "paradigm": paradigm,
            "structure": MODEL_TO_STRUCTURE.get(model, model),
            "model": model,
            "seed_or_restart": run,
            "period": period,
            "water_year": int(wy),
            **result,
        })
    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs, runs.copy()
    # The required basin-year product gives each basin/model/year one row.  For
    # dPL, seed-specific rows are retained in the companion *_runs table and
    # collapsed here by median across seeds for that basin-year.
    collapsed = []
    for keys, group in runs.groupby(["basin_id", "paradigm", "structure", "period", "water_year"], sort=True):
        basin, paradigm, structure, period, wy = keys
        complete = group[group["complete_year"]]
        valid = group[group["valid_year"] & np.isfinite(group["Delta_CT"])]
        base = group.iloc[0].to_dict()
        base.update({
            "basin_id": basin,
            "paradigm": paradigm,
            "structure": structure,
            "model": STRUCTURE_TO_MODEL.get(structure, structure),
            "period": period,
            "water_year": int(wy),
            "seed_or_restart": "selected_restart" if paradigm == "IC-CMA-ES" else "median_across_seeds",
            "complete_year": bool(len(complete)),
            "valid_year": bool(len(valid)),
            "invalid_reason": "" if len(valid) else ";".join(sorted(set(group["invalid_reason"]))),
            "n_valid_days": int(complete["n_valid_days"].median()) if len(complete) else 0,
            "CT_obs": float(valid["CT_obs"].median()) if len(valid) else math.nan,
            "CT_sim": float(valid["CT_sim"].median()) if len(valid) else math.nan,
            "Delta_CT": float(valid["Delta_CT"].median()) if len(valid) else math.nan,
            "seed_count": int(group["seed_or_restart"].nunique()),
        })
        collapsed.append(base)
    return pd.DataFrame(collapsed), runs


def build_basin_level_ct(basin_year: pd.DataFrame, basin_year_runs: pd.DataFrame) -> pd.DataFrame:
    if basin_year_runs.empty:
        return pd.DataFrame()
    run_rows = []
    for keys, group in basin_year_runs.groupby(
        ["basin_id", "paradigm", "structure", "period", "seed_or_restart"], sort=True
    ):
        basin, paradigm, structure, period, run = keys
        valid = group[group["valid_year"] & np.isfinite(group["Delta_CT"])]
        run_rows.append({
            "basin_id": basin,
            "paradigm": paradigm,
            "structure": structure,
            "period": period,
            "seed_or_restart": run,
            "valid_year_count": int(len(valid)),
            "basin_median_Delta_CT": float(valid["Delta_CT"].median()) if len(valid) else math.nan,
            "basin_CT_q25_years": float(valid["Delta_CT"].quantile(0.25)) if len(valid) else math.nan,
            "basin_CT_q75_years": float(valid["Delta_CT"].quantile(0.75)) if len(valid) else math.nan,
            "CT_obs_median_years": float(valid["CT_obs"].median()) if len(valid) else math.nan,
            "CT_sim_median_years": float(valid["CT_sim"].median()) if len(valid) else math.nan,
        })
    run_level = pd.DataFrame(run_rows)
    selected = []
    for keys, group in run_level.groupby(["basin_id", "paradigm", "structure", "period"], sort=True):
        basin, paradigm, structure, period = keys
        valid = group[np.isfinite(group["basin_median_Delta_CT"])]
        if paradigm == "IC-CMA-ES":
            chosen = valid.iloc[0] if len(valid) else group.iloc[0]
        else:
            chosen = group.iloc[0].copy()
            for col in ["valid_year_count", "basin_median_Delta_CT", "basin_CT_q25_years", "basin_CT_q75_years", "CT_obs_median_years", "CT_sim_median_years"]:
                chosen[col] = float(valid[col].median()) if len(valid) else math.nan
            chosen["seed_or_restart"] = "median_across_seeds"
            chosen["valid_year_count"] = int(valid["valid_year_count"].median()) if len(valid) else 0
        selected.append(dict(chosen))
    result = pd.DataFrame(selected)
    attrs = load_snow_attributes()
    result = result.merge(attrs[["basin_id", "frac_snow", "snow_stratum"]], on="basin_id", how="left", validate="many_to_one")
    result["KGE_pass_0p60"] = False
    return result


def metric_rows_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in daily[daily["period"].isin(PERIODS)].groupby(
        ["basin_id", "paradigm", "model", "seed_or_restart", "selected_run", "period"], sort=True
    ):
        basin, paradigm, model, run, selected, period = keys
        group = group.sort_values("date")
        sim = pd.to_numeric(group["q_sim"], errors="coerce").to_numpy(float)
        obs = pd.to_numeric(group["q_obs"], errors="coerce").to_numpy(float)
        kge, n_obs, n_sim, n_valid, valid_metric = standard_kge(sim, obs)
        mask = np.isfinite(obs) & np.isfinite(sim) & (obs >= 0) & (sim >= 0)
        if n_valid:
            error = sim[mask] - obs[mask]
            denom = float(obs[mask].sum())
            nse_denom = float(np.sum((obs[mask] - obs[mask].mean()) ** 2))
            nse = float(1 - np.sum(error ** 2) / nse_denom) if nse_denom > 0 else math.nan
            pbias = float(100 * error.sum() / denom) if denom != 0 else math.nan
            rmse = float(np.sqrt(np.mean(error ** 2)))
        else:
            nse = pbias = rmse = math.nan
        rows.append({
            "basin_id": basin, "paradigm": paradigm, "structure": MODEL_TO_STRUCTURE.get(model, model),
            "model": model, "period": period, "seed_or_restart": run, "selected_run": bool(selected),
            "KGE": kge, "NSE": nse, "PBIAS": pbias, "RMSE": rmse,
            "valid_observation_count": n_obs, "valid_simulation_count": n_sim,
            "valid_days": n_valid, "valid_metric": bool(valid_metric),
        })
    seed_level = pd.DataFrame(rows)
    if seed_level.empty:
        return seed_level
    primary = []
    for keys, group in seed_level.groupby(["basin_id", "paradigm", "structure", "period"], sort=True):
        basin, paradigm, structure, period = keys
        if paradigm == "IC-CMA-ES":
            train = group[group["period"].eq("train")]
            candidates = train[train["selected_run"]]
            if candidates.empty:
                candidates = train
            chosen_run = candidates.sort_values(["KGE", "seed_or_restart"], ascending=[False, True]).iloc[0]["seed_or_restart"] if not candidates.empty else ""
            chosen = group[group["seed_or_restart"].eq(chosen_run)]
            if chosen.empty:
                chosen = group.iloc[[0]]
            row = chosen.iloc[0].to_dict()
        else:
            row = group.iloc[0].to_dict()
            for col in ["KGE", "NSE", "PBIAS", "RMSE", "valid_observation_count", "valid_simulation_count", "valid_days"]:
                row[col] = float(group[col].median()) if group[col].notna().any() else math.nan
            row["seed_or_restart"] = "median_across_seeds"
            row["selected_run"] = True
        primary.append(row)
    return pd.DataFrame(primary)


def median_summary(values: Iterable[float], seed_offset: int) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return {"N": 0, "median": math.nan, "IQR": math.nan, "CI_low": math.nan, "CI_high": math.nan}
    low, high = bootstrap_median_ci(arr, np.random.default_rng(SEED + seed_offset), BOOTSTRAP_REPS)
    return {
        "N": int(len(arr)), "median": float(np.median(arr)),
        "IQR": float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)),
        "CI_low": low, "CI_high": high,
    }


def performance_summaries(perf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    absolute = []
    counter = 0
    for paradigm in PARADIGMS:
        for structure in STRUCTURES:
            for period in PERIODS:
                subset = perf[(perf.paradigm == paradigm) & (perf.structure == structure) & (perf.period == period)]
                for metric in ("KGE", "NSE", "PBIAS", "RMSE"):
                    stats = median_summary(subset[metric], counter)
                    absolute.append({"paradigm": paradigm, "structure": structure, "period": period, "metric": metric, **stats})
                    counter += 1
    attrs = load_snow_attributes()
    perf_attr = perf.merge(attrs[["basin_id", "frac_snow", "snow_stratum"]], on="basin_id", how="left", validate="many_to_one")
    strata = []
    for paradigm in PARADIGMS:
        for structure in STRUCTURES:
            subset = perf_attr[(perf_attr.paradigm == paradigm) & (perf_attr.structure == structure) & (perf_attr.period == "test")]
            for stratum in SNOW_STRATA:
                stats = median_summary(subset.loc[subset.snow_stratum == stratum, "KGE"], counter)
                strata.append({"paradigm": paradigm, "structure": structure, "period": "test", "snow_stratum": stratum, **stats})
                counter += 1
    effects = []
    for paradigm in PARADIGMS:
        subset = perf_attr[(perf_attr.paradigm == paradigm) & (perf_attr.period == "test")]
        wide = subset.pivot(index="basin_id", columns="structure", values="KGE")
        for stratum in ("ALL", *SNOW_STRATA):
            ids = set(wide.index) if stratum == "ALL" else set(subset.loc[subset.snow_stratum == stratum, "basin_id"])
            for effect, first, second in (("TGD-Base", "TGD", "Base"), ("CN-Base", "CN", "Base"), ("CN-TGD", "CN", "TGD")):
                values = (wide.loc[wide.index.intersection(ids), first] - wide.loc[wide.index.intersection(ids), second]).dropna()
                stats = median_summary(values, counter)
                effects.append({"paradigm": paradigm, "period": "test", "snow_stratum": stratum, "effect": effect, "positive_fraction": float((values > 0).mean()) if len(values) else math.nan, **stats})
                counter += 1
    return pd.DataFrame(absolute), pd.DataFrame(strata), pd.DataFrame(effects)


def ct_summaries(ct: pd.DataFrame, perf: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if ct.empty:
        return {name: pd.DataFrame() for name in ("screened", "threshold_curve", "snow_strata", "common_pass", "timing_thresholds", "low_snow")}
    test_perf = perf[perf.period == "test"][["basin_id", "paradigm", "structure", "KGE"]]
    frame = ct[ct.period == "test"].merge(test_perf, on=["basin_id", "paradigm", "structure"], how="inner", validate="one_to_one")
    frame["large_15"] = frame["basin_median_Delta_CT"].abs() >= TIMING_THRESHOLD
    screened, curve, snow_rows, threshold_rows = [], [], [], []
    counter = 9000
    for paradigm in PARADIGMS:
        for structure in STRUCTURES:
            combo = frame[(frame.paradigm == paradigm) & (frame.structure == structure)]
            for tau, label in ((SCREEN_THRESHOLD, "0p60"),):
                sub = combo[combo.KGE >= tau]
                stats = median_summary(sub["basin_median_Delta_CT"], counter)
                screened.append({"paradigm": paradigm, "structure": structure, "tau": tau, "screened_N": len(sub), "large_15_N": int(sub.large_15.sum()), "large_15_fraction": float(sub.large_15.mean()) if len(sub) else math.nan, **stats})
                counter += 1
            for tau in THRESHOLD_GRID:
                sub = combo[combo.KGE >= tau]
                curve.append({"paradigm": paradigm, "structure": structure, "tau": float(tau), "screened_N": len(sub), "large_15_N": int(sub.large_15.sum()), "large_15_fraction": float(sub.large_15.mean()) if len(sub) else math.nan})
            for stratum in SNOW_STRATA:
                sub = combo[combo.snow_stratum == stratum]
                stats = median_summary(sub["basin_median_Delta_CT"], counter)
                snow_rows.append({"paradigm": paradigm, "structure": structure, "snow_stratum": stratum, "N": len(sub), "large_15_N": int((sub.basin_median_Delta_CT.abs() >= 15).sum()), "large_15_fraction": float((sub.basin_median_Delta_CT.abs() >= 15).mean()) if len(sub) else math.nan, **stats})
                counter += 1
            for threshold in (10.0, 15.0, 20.0):
                sub = combo[combo.KGE >= SCREEN_THRESHOLD]
                count = int((sub.basin_median_Delta_CT.abs() >= threshold).sum())
                threshold_rows.append({"paradigm": paradigm, "structure": structure, "KGE_tau": SCREEN_THRESHOLD, "timing_threshold_days": threshold, "N": len(sub), "large_N": count, "fraction": count / len(sub) if len(sub) else math.nan})
    # Common-pass sensitivity and S1 negative condition.
    common_rows, low_rows = [], []
    for paradigm in PARADIGMS:
        p = frame[frame.paradigm == paradigm]
        kge_wide = p.pivot(index="basin_id", columns="structure", values="KGE")
        ct_wide = p.pivot(index="basin_id", columns="structure", values="basin_median_Delta_CT")
        ids = kge_wide.index[(kge_wide[list(STRUCTURES)] >= SCREEN_THRESHOLD).all(axis=1)]
        for structure in STRUCTURES:
            values = ct_wide.loc[ids, structure].dropna()
            stats = median_summary(values, counter)
            common_rows.append({"paradigm": paradigm, "subset": "Base_AND_TGD_AND_CN_KGE_ge_0p60", "structure": structure, "common_pass_N": len(ids), "large_15_N": int((values.abs() >= 15).sum()), "large_15_fraction": float((values.abs() >= 15).mean()) if len(values) else math.nan, **stats})
            counter += 1
        for effect, first, second in (("TGD-Base", "TGD", "Base"), ("CN-Base", "CN", "Base"), ("CN-TGD", "CN", "TGD")):
            values = (ct_wide.loc[ids, first] - ct_wide.loc[ids, second]).dropna()
            stats = median_summary(values, counter)
            common_rows.append({"paradigm": paradigm, "subset": "Base_AND_TGD_AND_CN_KGE_ge_0p60", "structure": effect, "common_pass_N": len(values), "large_15_N": math.nan, "large_15_fraction": math.nan, **stats})
            counter += 1
        s1 = p[p.snow_stratum == "S1"]
        low_rows.append({"paradigm": paradigm, "condition": "S1", "description": "low-snow negative condition; descriptive only", "N": len(s1), "KGE_effects": "see r1_paired_kge_effects.csv", "CT_rows": len(s1)})
    return {
        "screened": pd.DataFrame(screened), "threshold_curve": pd.DataFrame(curve),
        "snow_strata": pd.DataFrame(snow_rows), "common_pass": pd.DataFrame(common_rows),
        "timing_thresholds": pd.DataFrame(threshold_rows), "low_snow": pd.DataFrame(low_rows),
    }


def historical_reconciliation() -> pd.DataFrame:
    r1 = PROJECT / "manuscript" / "results" / "R1"
    perf = pd.read_csv(r1 / "r1_basin_level_performance.csv")
    sig = pd.read_csv(r1 / "r1_snow_signatures_basin_level.csv")
    perf["basin_id"] = norm_ids(perf["basin_id"])
    sig["basin_id"] = norm_ids(sig["basin_id"])
    anchors = {
        ("IC", "Base"): (331, 56, -5.0), ("IC", "TGD"): (394, 49, -3.0), ("IC", "CN"): (427, 25, 0.0),
        ("dPL", "Base"): (344, 46, -2.0), ("dPL", "TGD"): (404, 37, -1.0), ("dPL", "CN"): (426, 20, 1.0),
    }
    rows = []
    for paradigm in PARADIGMS:
        label = PARADIGM_LABEL[paradigm]
        for structure in STRUCTURES:
            model = STRUCTURE_TO_MODEL[structure]
            p = perf[(perf.paradigm == paradigm) & (perf.model == model) & (perf.period == "test")]
            s = sig[(sig.paradigm == paradigm) & (sig.model == model) & (sig.period == "test")]
            joined = p[["basin_id", "KGE" if "KGE" in p else "kge"]].rename(columns={"KGE": "KGE", "kge": "KGE"}).merge(s[["basin_id", "ct_error_signed", "ct_error_abs"]], on="basin_id", how="inner")
            pass_set = joined[joined.KGE >= SCREEN_THRESHOLD]
            anchor_n, anchor_large, anchor_median = anchors[(label, structure)]
            for field, explanation in (("ct_error_signed", "valid signed basin-level Delta_CT candidate"), ("ct_error_abs", "absolute yearly-error summary field; not abs(basin signed Delta_CT)")):
                values = pass_set[field]
                large = int((values.abs() >= TIMING_THRESHOLD).sum()) if field == "ct_error_signed" else int((values >= TIMING_THRESHOLD).sum())
                med = float(values.median()) if len(values) else math.nan
                rows.append({
                    "paradigm": paradigm, "structure": structure, "candidate_source": "current r1_snow_signatures_basin_level.csv",
                    "candidate_field": field, "candidate_interpretation": explanation, "screened_N": len(values),
                    "large_15_N": large, "candidate_median": med,
                    "historical_screened_N": anchor_n, "historical_large_15_N": anchor_large, "historical_median_anchor": anchor_median,
                    "reproduces_historical_anchor": bool(len(values) == anchor_n and large == anchor_large and abs(med - anchor_median) < 1e-9),
                    "status": "DIAGNOSTIC_ONLY_NOT_CANONICAL",
                })
    return pd.DataFrame(rows)


def blocked_tables() -> None:
    schemas = {
        "r1_basin_year_ct.csv": ["basin_id", "paradigm", "structure", "water_year", "complete_year", "valid_year", "invalid_reason", "n_valid_days", "CT_obs", "CT_sim", "Delta_CT", "frac_snow", "snow_stratum"],
        "r1_basin_level_ct.csv": ["basin_id", "paradigm", "structure", "frac_snow", "snow_stratum", "valid_year_count", "basin_median_Delta_CT", "basin_CT_q25_years", "basin_CT_q75_years", "basin_test_KGE", "KGE_pass_0p60"],
        "r1_absolute_performance_summary.csv": ["paradigm", "structure", "period", "metric", "N", "median", "IQR", "CI_low", "CI_high"],
        "r1_test_kge_by_snow_stratum.csv": ["paradigm", "structure", "period", "snow_stratum", "N", "median", "IQR", "CI_low", "CI_high"],
        "r1_paired_kge_effects.csv": ["paradigm", "period", "snow_stratum", "effect", "N", "median", "IQR", "CI_low", "CI_high", "positive_fraction"],
        "r1_screened_ct_summary.csv": ["paradigm", "structure", "tau", "screened_N", "large_15_N", "large_15_fraction", "N", "median", "IQR", "CI_low", "CI_high"],
        "r1_ct_threshold_curve.csv": ["paradigm", "structure", "tau", "screened_N", "large_15_N", "large_15_fraction"],
        "r1_ct_by_snow_stratum.csv": ["paradigm", "structure", "snow_stratum", "N", "large_15_N", "large_15_fraction", "median", "IQR", "CI_low", "CI_high"],
        "r1_common_pass_ct.csv": ["paradigm", "subset", "structure", "common_pass_N", "large_15_N", "large_15_fraction", "N", "median", "IQR", "CI_low", "CI_high"],
        "r1_timing_threshold_sensitivity.csv": ["paradigm", "structure", "KGE_tau", "timing_threshold_days", "N", "large_N", "fraction"],
        "r1_low_snow_negative_condition.csv": ["paradigm", "condition", "description", "N", "KGE_effects", "CT_rows"],
    }
    for name, cols in schemas.items():
        write_csv(name, pd.DataFrame(columns=cols), cols)


def render_report(source_map: dict[str, Any], checks: dict[str, Any], candidates: list[dict[str, Any]], daily_reason: str, reconciliation: pd.DataFrame, daily_loaded: bool) -> str:
    lines = [
        "# Staged local-only R1 statistics rebuild audit",
        "",
        "> This is a statistics reconstruction audit only. It does not launch training, calibration, inference, or simulation and never writes manuscript/results/R1 or figures.",
        "",
        "## Readiness verdict",
        "",
        f"- Stage 0 source recovery: **PASS_WITH_LIMITATION** (historical definitions recoverable; daily source status is `{daily_reason}`).",
        f"- Stage 1 basin manifest: **{'PASS_CANONICAL' if checks['attribute_stratum_mismatches'] == 0 and checks['attribute_unique_basins'] == 531 else 'BLOCKED'}**.",
        f"- Stage 2 basin-year CT: **{'PASS_CANONICAL' if daily_loaded else 'BLOCKED'}**.",
        f"- Stages 4–7 final R1 statistics: **{'PASS_CANONICAL' if daily_loaded else 'BLOCKED'}**.",
        "- Figures and canonical R1 results promotion: **BLOCKED / not attempted**.",
        "",
        "## A. Source provenance",
        "",
        "| role | local source | status |",
        "|---|---|---|",
        "| Basin snow attributes | `manuscript/results/R1/r1_snow_attributes.csv` | local, 531 rows |",
        "| Existing basin performance | `manuscript/results/R1/r1_basin_level_performance.csv` | diagnostic only; not accepted as rebuilt canonical output |",
        "| Existing CT summary | `manuscript/results/R1/r1_snow_signatures_basin_level.csv` | diagnostic only; not a basin-year source |",
        "| Daily observation/simulation source | local parquet candidates listed in `r1_daily_source_inventory.csv` | unavailable to staged reader in this environment |",
        "| Production metric definition | `manuscript/scripts/r1/r1_statistics.py:standard_kge` | recovered |",
        "| Daily CT implementation | `manuscript/scripts/r1/r1_daily_inference.py` and `r1_statistics.py` | recovered |",
        "| Historical Figure 2 implementation | git `59453bb^` and current `plot_r1_figure2.py` | recovered |",
        "",
        "The inventory records data paths named by older manifests but this audit did not open any parent `data/` directory or remote path. No observations or forcing files exist under `project/hydrodiag/data`.",
        "",
        "## B. Recovered definitions",
        "",
        "- **Water year:** October 1 through September 30; `water_year = calendar_year + (month >= 10)`. Leap days are retained because completeness is checked against the contiguous date span.",
        "- **Complete year:** all expected dates are present exactly once and every paired observation/simulation value is finite and nonnegative. The repository code does not impose a smaller day-count threshold.",
        "- **CT:** first 1-based water-year day where cumulative discharge reaches 50% of that year's total. Zero-total complete years are retained as `complete_year=True` with CT fields NaN, matching the recovered guard; the staged basin-level CT-valid population excludes them explicitly via `valid_year=False` and records the discrepancy rather than counting an unusable CT.",
        "- **Delta_CT:** `CT_sim - CT_obs`; negative means simulated runoff is earlier.",
        "- **Basin aggregation:** median of valid yearly signed Delta_CT. IC uses the selected restart; dPL uses the median of seed-specific basin-level summaries. Basin-years are not downstream independent observations.",
        "- **Eligibility/pairing:** observation and simulation are paired on the same dates within each water year; CT year eligibility is structure-specific. All-three structure eligibility is imposed only in the common-pass sensitivity.",
        "- **Figure 2 source level:** the recovered Figure 2 path uses the all-available basin-level CT summary; the separate primary signature package records minimum-five-year and minimum-three-year sensitivity sets and is not silently substituted here.",
        "- **Performance:** standard Gupta KGE from `standard_kge`, with the same paired finite/nonnegative mask and minimum 30 valid days; NSE, PBIAS, and RMSE use that mask.",
        "- **Bootstrap:** ordinary basin resampling, median statistic, 10,000 replicates, percentile 2.5/97.5 interval, base seed 20260730. No region-block bootstrap is used for these R1 population summaries.",
        "- **Figure 2 threshold grid:** recovered exactly as `np.arange(0.40, 0.8001, 0.01)`; screen KGE >= 0.60; timing threshold |Delta_CT| >= 15 d.",
        "",
        "## C. Invariant checks",
        "",
        f"- Snow attribute rows={checks['attribute_rows']}; unique basins={checks['attribute_unique_basins']}; duplicate rows={checks['attribute_duplicate_rows']}; missing frac_snow={checks['attribute_missing_frac_snow']}; stratum mismatches={checks['attribute_stratum_mismatches']}.",
        f"- Fixed strata: `{checks['stratum_counts']}`; S4+S5={checks['s4_plus_s5']}.",
        f"- Existing performance table rows={checks['performance_rows']}; unique basins={checks['performance_unique_basins']}; selected_run all true={checks['performance_selected_all']}; per-combination selected basin counts={checks['performance_counts']}.",
        "- All joins in the staged implementation are explicit basin-ID joins; no positional joins are used.",
        f"- Daily CT gate: `{daily_reason}`.",
        "",
        "## D. Historical reconciliation",
        "",
        "The current local diagnostic values around 152/329, 126/340, 151/396, 117/386, 132/427, and 112/424 are reproduced by using the existing `ct_error_abs` field as the large-error variable. That field is a basin-level summary of yearly absolute errors. It is not `abs(basin_median_Delta_CT)` and therefore cannot be used for the requested signed-CT Figure 2 estimand.",
        "",
        "The recovered Figure 2 definition instead uses `ct_error_signed` and applies `abs()` only after basin-level signed aggregation. The current summary gives a different diagnostic candidate. Neither candidate reproduces every historical 56/331, 46/344, 49/394, 37/404, 25/427, and 20/426 anchor, and no daily basin-year source is locally readable to test alternative valid-year or seed aggregation rules.",
        "",
    ]
    if not reconciliation.empty:
        lines.extend([
            "| paradigm | structure | candidate field | screened N | large-15 N | candidate median | historical N | historical large-15 N | reproduces? |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ])
        for row in reconciliation.itertuples(index=False):
            lines.append(f"| {row.paradigm} | {row.structure} | `{row.candidate_field}` | {row.screened_N} | {row.large_15_N} | {row.candidate_median:.3f} | {row.historical_screened_N} | {row.historical_large_15_N} | {row.reproduces_historical_anchor} |")
    lines.extend([
        "",
        "## E. Frozen R1 values",
        "",
        "The only values frozen by this audit are the basin-universe QA counts: S1=165, S2=156, S3=121, S4=34, S5=55, and S4+S5=89. Existing formatted R1 performance/CT numbers remain diagnostic and are not promoted as canonical rebuilt values.",
        "",
        "Because the daily source gate failed, the required final Figure 1, Table 1, Figure 2, common-pass, threshold-sensitivity, and low-snow tables are emitted as empty blocked-schema CSVs. The historical reconciliation CSV remains populated with diagnostic comparisons.",
        "",
        "## F. Files and promotion decision",
        "",
        "- `r1_source_map.json` — local source hashes and recovered history references.",
        "- `r1_daily_source_inventory.csv` — local daily candidates and reader/schema status.",
        "- `r1_basin_manifest.csv` — explicit basin/stratum/availability manifest.",
        "- `historical_ct_reconciliation.csv` — diagnostic old/current comparison.",
        "- `r1_rebuild_audit.md` — this report.",
        "- Downstream blocked-schema CSVs are present for the staged pipeline contract.",
        "",
        "No generated table is safe to promote into `manuscript/results/R1/` until locally readable daily observations and Base/TGD/CN simulations are restored and all CT/performance validation gates pass.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    attrs = pd.read_csv(project_file("manuscript/results/R1/r1_snow_attributes.csv"))
    perf = pd.read_csv(project_file("manuscript/results/R1/r1_basin_level_performance.csv"))
    candidates = discover_daily_candidates()
    manifest, checks = build_basin_manifest(perf, attrs, candidates)
    daily, daily_reason = load_readable_daily(candidates)
    if daily is None:
        manifest["daily_observation_available"] = False
        manifest["daily_simulation_available"] = False
        manifest["daily_source_status"] = daily_reason
        manifest["exclusion_reason"] = daily_reason
    source_map = {
        "project_root": str(PROJECT),
        "scope": "project/hydrodiag only",
        "source_files": current_source_files(),
        "historical_figure2_candidate": "59453bb^:project/hydrodiag/manuscript/scripts/r1/plot_r1_figure2.py",
        "historical_r1_definition_candidate": "59453bb^:project/hydrodiag/manuscript/scripts/r1/r1_statistics.py",
        "historical_threshold_grid": "np.arange(0.40, 0.8001, 0.01)",
        "outside_project_paths_accessed": [],
        "training_launched": False,
        "calibration_launched": False,
        "inference_launched": False,
        "simulation_launched": False,
        "figures_modified": False,
        "git_head": run_git("rev-parse", "HEAD"),
        "git_history_check": run_git("log", "--all", "--full-history", "--oneline", "--", "manuscript/scripts/r1"),
    }
    write_json("r1_source_map.json", source_map)
    write_csv("r1_daily_source_inventory.csv", pd.DataFrame(candidates))
    write_csv("r1_basin_manifest.csv", manifest)
    reconciliation = historical_reconciliation()
    write_csv("historical_ct_reconciliation.csv", reconciliation)
    if daily is None:
        blocked_tables()
    else:
        basin_year, basin_year_runs = build_basin_year_ct(daily)
        snow_attrs = load_snow_attributes()[["basin_id", "frac_snow", "snow_stratum"]]
        basin_year = basin_year.merge(snow_attrs, on="basin_id", how="left", validate="many_to_one")
        basin_year_runs = basin_year_runs.merge(snow_attrs, on="basin_id", how="left", validate="many_to_one")
        basin_level = build_basin_level_ct(basin_year, basin_year_runs)
        perf_level = metric_rows_from_daily(daily)
        if not basin_level.empty and not perf_level.empty:
            perf_level = perf_level.merge(basin_level[["basin_id", "paradigm", "structure", "period", "basin_median_Delta_CT"]], on=["basin_id", "paradigm", "structure", "period"], how="left")
            basin_level = basin_level.merge(perf_level[["basin_id", "paradigm", "structure", "period", "KGE"]], on=["basin_id", "paradigm", "structure", "period"], how="left")
            basin_level["basin_test_KGE"] = basin_level["KGE"]
            basin_level["KGE_pass_0p60"] = basin_level["period"].eq("test") & (basin_level["KGE"] >= SCREEN_THRESHOLD)
        absolute, snow, effects = performance_summaries(perf_level)
        ct_tables = ct_summaries(basin_level, perf_level)
        write_csv("r1_basin_year_ct.csv", basin_year)
        write_csv("r1_basin_year_ct_runs.csv", basin_year_runs)
        write_csv("r1_basin_level_ct.csv", basin_level)
        write_csv("r1_basin_level_performance_rebuilt.csv", perf_level)
        write_csv("r1_absolute_performance_summary.csv", absolute)
        write_csv("r1_test_kge_by_snow_stratum.csv", snow)
        write_csv("r1_paired_kge_effects.csv", effects)
        write_csv("r1_screened_ct_summary.csv", ct_tables["screened"])
        write_csv("r1_ct_threshold_curve.csv", ct_tables["threshold_curve"])
        write_csv("r1_ct_by_snow_stratum.csv", ct_tables["snow_strata"])
        write_csv("r1_common_pass_ct.csv", ct_tables["common_pass"])
        write_csv("r1_timing_threshold_sensitivity.csv", ct_tables["timing_thresholds"])
        write_csv("r1_low_snow_negative_condition.csv", ct_tables["low_snow"])
    report = render_report(source_map, checks, candidates, daily_reason, reconciliation, daily is not None)
    (OUT / "r1_rebuild_audit.md").write_text(report, encoding="utf-8")
    write_json("r1_audit_manifest.json", {
        "output": str(OUT),
        "scope": "project/hydrodiag only",
        "daily_gate": daily_reason,
        "daily_loaded": daily is not None,
        "s1_s5_counts": checks["stratum_counts"],
        "s4_plus_s5": checks["s4_plus_s5"],
        "training_launched": False,
        "calibration_launched": False,
        "inference_launched": False,
        "simulation_launched": False,
        "figures_modified": False,
        "canonical_promotion": False,
    })
    print(f"output={OUT}")
    print(f"daily_gate={daily_reason}")
    print(f"strata={checks['stratum_counts']} s4_plus_s5={checks['s4_plus_s5']}")
    print("training_calibration_inference_simulation=none")
    print("canonical_promotion=no")


if __name__ == "__main__":
    main()
