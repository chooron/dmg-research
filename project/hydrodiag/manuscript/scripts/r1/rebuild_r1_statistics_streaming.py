"""Memory-bounded R1 statistics rebuild from daily Parquet row groups.

The forward model outputs are already produced from existing checkpoints.  This
script performs only the statistics stage.  It reads one Parquet row group at a
time, keeps compact basin-year/metric summaries, and uses CUDA tensors for the
CT and daily metric reductions.  It never materializes the multi-million-row
daily table in pandas and never writes canonical R1 results.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
OUT = PROJECT / "manuscript" / "cache" / "r1_rebuild_audit_staged"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from rebuild_r1_statistics_staged import (  # noqa: E402
    BOOTSTRAP_REPS,
    MODEL_TO_STRUCTURE,
    PARADIGMS,
    PERIODS,
    SCREEN_THRESHOLD,
    STRUCTURES,
    STRUCTURE_TO_MODEL,
    TIMING_THRESHOLD,
    build_basin_level_ct,
    build_basin_manifest,
    current_source_files,
    ct_summaries,
    discover_daily_candidates,
    historical_reconciliation,
    load_snow_attributes,
    median_summary,
    performance_summaries,
    project_file,
    render_report,
    run_git,
    safe_json,
    sha256,
    write_csv,
    write_json,
)
from r1_metrics import bootstrap_median_ci  # noqa: E402

DEVICE = torch.device("cuda")
REQUIRED_COLUMNS = [
    "basin_id",
    "paradigm",
    "model",
    "seed_or_restart",
    "selected_run",
    "period",
    "date",
    "q_obs",
    "q_sim",
]
EXPECTED_ROWS_PER_FILE = 5_624_352


def _as_numpy(column: Any) -> np.ndarray:
    return column.combine_chunks().to_numpy(zero_copy_only=False)


def _date_days(values: np.ndarray) -> np.ndarray:
    return pd.to_datetime(values, errors="coerce").to_numpy(dtype="datetime64[D]")


def _water_years(dates: np.ndarray) -> np.ndarray:
    years = dates.astype("datetime64[Y]").astype(np.int32) + 1970
    months = dates.astype("datetime64[M]").astype(np.int32) % 12 + 1
    return years + (months >= 10).astype(np.int32)


def _basin_spans(basin_ids: np.ndarray) -> list[tuple[int, int]]:
    if len(basin_ids) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(basin_ids[1:] != basin_ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(basin_ids)]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _segment_spans(water_year: np.ndarray) -> list[tuple[int, int, int]]:
    if len(water_year) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(water_year[1:] != water_year[:-1]) + 1]
    ends = np.r_[starts[1:], len(water_year)]
    return [(int(a), int(b), int(water_year[a])) for a, b in zip(starts, ends)]


def _pad_rows(rows: list[np.ndarray], fill: float = np.nan) -> np.ndarray:
    width = max((len(row) for row in rows), default=0)
    result = np.full((len(rows), width), fill, dtype=np.float64)
    for index, row in enumerate(rows):
        result[index, : len(row)] = row.astype(np.float64, copy=False)
    return result


def _gpu_daily_metrics(obs_rows: list[np.ndarray], sim_rows: list[np.ndarray]) -> list[dict[str, Any]]:
    obs = torch.from_numpy(_pad_rows(obs_rows)).to(DEVICE, dtype=torch.float64)
    sim = torch.from_numpy(_pad_rows(sim_rows)).to(DEVICE, dtype=torch.float64)
    valid_obs = torch.isfinite(obs) & (obs >= 0)
    valid_sim = torch.isfinite(sim) & (sim >= 0)
    valid = valid_obs & valid_sim
    obs_safe = torch.where(valid, obs, torch.zeros_like(obs))
    sim_safe = torch.where(valid, sim, torch.zeros_like(sim))
    n_obs = valid_obs.sum(dim=1)
    n_sim = valid_sim.sum(dim=1)
    n_valid = valid.sum(dim=1)
    denom = torch.clamp(n_valid, min=1).to(torch.float64)
    obs_mean = obs_safe.sum(dim=1) / denom
    sim_mean = sim_safe.sum(dim=1) / denom
    obs_centered = torch.where(valid, obs - obs_mean[:, None], torch.zeros_like(obs))
    sim_centered = torch.where(valid, sim - sim_mean[:, None], torch.zeros_like(sim))
    obs_std = torch.sqrt((obs_centered * obs_centered).sum(dim=1) / denom)
    sim_std = torch.sqrt((sim_centered * sim_centered).sum(dim=1) / denom)
    covariance = (obs_centered * sim_centered).sum(dim=1) / denom
    corr_denom = obs_std * sim_std
    corr = covariance / corr_denom
    alpha = sim_std / obs_std
    beta = sim_mean / obs_mean
    kge = 1.0 - torch.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    error = sim_safe - obs_safe
    nse_denom = (obs_centered * obs_centered).sum(dim=1)
    nse = 1.0 - (error * error).sum(dim=1) / nse_denom
    obs_total = obs_safe.sum(dim=1)
    pbias = 100.0 * error.sum(dim=1) / obs_total
    rmse = torch.sqrt((error * error).sum(dim=1) / denom)
    valid_metric = (n_valid >= 30) & (obs_std >= 1e-10) & (obs_mean != 0)
    values = {
        "kge": kge.detach().cpu().numpy(),
        "nse": nse.detach().cpu().numpy(),
        "pbias": pbias.detach().cpu().numpy(),
        "rmse": rmse.detach().cpu().numpy(),
        "n_obs": n_obs.detach().cpu().numpy(),
        "n_sim": n_sim.detach().cpu().numpy(),
        "n_valid": n_valid.detach().cpu().numpy(),
        "valid_metric": valid_metric.detach().cpu().numpy(),
    }
    rows = []
    for i in range(len(obs_rows)):
        ok = bool(values["valid_metric"][i])
        rows.append(
            {
                "KGE": float(values["kge"][i]) if ok else math.nan,
                "NSE": float(values["nse"][i]) if ok else math.nan,
                "PBIAS": float(values["pbias"][i]) if int(values["n_valid"][i]) else math.nan,
                "RMSE": float(values["rmse"][i]) if int(values["n_valid"][i]) else math.nan,
                "valid_observation_count": int(values["n_obs"][i]),
                "valid_simulation_count": int(values["n_sim"][i]),
                "valid_days": int(values["n_valid"][i]),
                "valid_metric": ok,
            }
        )
    return rows


def _gpu_ct_segments(
    obs_rows: list[np.ndarray], sim_rows: list[np.ndarray], metadata: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    lengths = [len(row) for row in obs_rows]
    width = max(lengths, default=0)
    obs = torch.full((len(obs_rows), width), float("nan"), device=DEVICE, dtype=torch.float64)
    sim = torch.full_like(obs, float("nan"))
    for i, (obs_row, sim_row) in enumerate(zip(obs_rows, sim_rows)):
        obs[i, : len(obs_row)] = torch.from_numpy(
            np.array(obs_row, dtype=np.float64, copy=True)
        ).to(DEVICE, dtype=torch.float64)
        sim[i, : len(sim_row)] = torch.from_numpy(
            np.array(sim_row, dtype=np.float64, copy=True)
        ).to(DEVICE, dtype=torch.float64)
    length_tensor = torch.tensor(lengths, device=DEVICE, dtype=torch.int64)
    active = torch.arange(width, device=DEVICE)[None, :] < length_tensor[:, None]
    valid = active & torch.isfinite(obs) & torch.isfinite(sim) & (obs >= 0) & (sim >= 0)
    obs_safe = torch.where(valid, obs, torch.zeros_like(obs))
    sim_safe = torch.where(valid, sim, torch.zeros_like(sim))
    counts = valid.sum(dim=1)
    obs_total = obs_safe.sum(dim=1)
    sim_total = sim_safe.sum(dim=1)
    obs_cum = torch.cumsum(obs_safe, dim=1)
    sim_cum = torch.cumsum(sim_safe, dim=1)
    obs_hit = obs_cum >= (obs_total[:, None] * 0.5)
    sim_hit = sim_cum >= (sim_total[:, None] * 0.5)
    obs_ct = torch.argmax(obs_hit.to(torch.int8), dim=1) + 1
    sim_ct = torch.argmax(sim_hit.to(torch.int8), dim=1) + 1
    # GPU reductions are primary; near-threshold rows are recomputed with the
    # repository's NumPy float64 convention to avoid one-day tie drift.
    near_obs = torch.min(torch.abs(obs_cum - obs_total[:, None] * 0.5), dim=1).values < 1e-10
    near_sim = torch.min(torch.abs(sim_cum - sim_total[:, None] * 0.5), dim=1).values < 1e-10
    near_threshold = (near_obs | near_sim).detach().cpu().numpy()
    complete = ((~active) | valid).all(dim=1)
    positive = (obs_total > 0) & (sim_total > 0)
    valid_year = complete & positive
    out = []
    for i, meta in enumerate(metadata):
        if not bool(complete[i].item()):
            reason = "nonfinite_or_negative_paired_day"
        elif not bool(positive[i].item()):
            reason = "zero_or_negative_water_year_total"
        else:
            reason = ""
        obs_ct_value = int(obs_ct[i].item())
        sim_ct_value = int(sim_ct[i].item())
        if bool(valid_year[i].item()) and bool(near_threshold[i]):
            obs_cumulative = np.cumsum(obs_rows[i], dtype=np.float64)
            sim_cumulative = np.cumsum(sim_rows[i], dtype=np.float64)
            obs_ct_value = int(np.argmax(obs_cumulative >= 0.5 * float(np.sum(obs_rows[i], dtype=np.float64))) + 1)
            sim_ct_value = int(np.argmax(sim_cumulative >= 0.5 * float(np.sum(sim_rows[i], dtype=np.float64))) + 1)
        out.append(
            {
                **meta,
                "complete_year": bool(complete[i].item()),
                "valid_year": bool(valid_year[i].item()),
                "invalid_reason": reason,
                "n_valid_days": int(counts[i].item()),
                "CT_obs": obs_ct_value if bool(valid_year[i].item()) else math.nan,
                "CT_sim": sim_ct_value if bool(valid_year[i].item()) else math.nan,
                "Delta_CT": sim_ct_value - obs_ct_value if bool(valid_year[i].item()) else math.nan,
            }
        )
    return out


def _normalize_label(value: Any) -> str:
    return str(value).strip()


def _read_file_summaries(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pf = pq.ParquetFile(path)
    if pf.metadata.num_rows != EXPECTED_ROWS_PER_FILE:
        raise ValueError(f"{path}: expected {EXPECTED_ROWS_PER_FILE} rows, got {pf.metadata.num_rows}")
    metric_rows: list[dict[str, Any]] = []
    ct_rows: list[dict[str, Any]] = []
    previous_last_basin: str | None = None
    for rg_index in range(pf.num_row_groups):
        table = pf.read_row_group(rg_index, columns=REQUIRED_COLUMNS, use_threads=False)
        basin_ids = np.asarray(_as_numpy(table["basin_id"]), dtype=object)
        basin_ids = np.asarray([_normalize_label(v).zfill(8) for v in basin_ids], dtype=object)
        paradigm_values = np.asarray(_as_numpy(table["paradigm"]), dtype=object)
        model_values = np.asarray(_as_numpy(table["model"]), dtype=object)
        run_values = np.asarray(_as_numpy(table["seed_or_restart"]), dtype=object)
        period_values = np.asarray(_as_numpy(table["period"]), dtype=object)
        selected_values = np.asarray(_as_numpy(table["selected_run"]), dtype=object)
        dates = _date_days(np.asarray(_as_numpy(table["date"])))
        if np.isnat(dates).any():
            raise ValueError(f"{path}: unparseable date in row group {rg_index}")
        obs = np.asarray(_as_numpy(table["q_obs"]), dtype=np.float64)
        sim = np.asarray(_as_numpy(table["q_sim"]), dtype=np.float64)
        paradigm = _normalize_label(paradigm_values[0])
        if paradigm == "IC":
            paradigm = "IC-CMA-ES"
        elif paradigm == "dPL":
            paradigm = "dPL-MLP"
        model = _normalize_label(model_values[0]).replace("XAJ-TGD2", "XAJ-TGD")
        run = _normalize_label(run_values[0])
        period = _normalize_label(period_values[0])
        selected = _normalize_label(selected_values[0]).lower() in {"1", "true", "yes"}
        if previous_last_basin is not None and previous_last_basin == basin_ids[0]:
            raise ValueError(f"{path}: basin split across row groups at {basin_ids[0]}")
        spans = _basin_spans(basin_ids)
        if len(np.unique(basin_ids)) != len(spans):
            raise ValueError(f"{path}: non-contiguous basin rows in row group {rg_index}")
        previous_last_basin = basin_ids[-1]
        obs_rows = [obs[a:b] for a, b in spans]
        sim_rows = [sim[a:b] for a, b in spans]
        basin_row_ids = [basin_ids[a] for a, _ in spans]
        metric_values = _gpu_daily_metrics(obs_rows, sim_rows)
        for basin, values in zip(basin_row_ids, metric_values):
            metric_rows.append(
                {
                    "basin_id": basin,
                    "paradigm": paradigm,
                    "structure": MODEL_TO_STRUCTURE[model],
                    "model": model,
                    "period": period,
                    "seed_or_restart": run,
                    "selected_run": selected,
                    **values,
                }
            )
        segment_obs: list[np.ndarray] = []
        segment_sim: list[np.ndarray] = []
        segment_meta: list[dict[str, Any]] = []
        for (a, b), basin in zip(spans, basin_row_ids):
            basin_dates = dates[a:b]
            basin_obs = obs[a:b]
            basin_sim = sim[a:b]
            if len(basin_dates):
                water_year = _water_years(basin_dates)
                for ys, ye, wy in _segment_spans(water_year):
                    expected_days = int((basin_dates[ye - 1] - basin_dates[ys]).astype(int) + 1)
                    date_complete = len(basin_dates[ys:ye]) == expected_days and len(np.unique(basin_dates[ys:ye])) == expected_days
                    segment_obs.append(basin_obs[ys:ye])
                    segment_sim.append(basin_sim[ys:ye])
                    segment_meta.append(
                        {
                            "basin_id": basin,
                            "paradigm": paradigm,
                            "structure": MODEL_TO_STRUCTURE[model],
                            "model": model,
                            "period": period,
                            "seed_or_restart": run,
                            "water_year": wy,
                            "date_complete": date_complete,
                        }
                    )
        ct_values = _gpu_ct_segments(segment_obs, segment_sim, segment_meta)
        for row in ct_values:
            date_complete = bool(row.pop("date_complete"))
            if not date_complete:
                row["complete_year"] = False
                row["valid_year"] = False
                row["invalid_reason"] = "missing_or_duplicate_day"
            ct_rows.append(row)
    return metric_rows, ct_rows

def _collapse_metric_rows(seed_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in seed_level.groupby(
        ["basin_id", "paradigm", "structure", "period"],
        sort=True,
        observed=True,
    ):
        basin, paradigm, structure, period = keys
        if paradigm == "IC-CMA-ES":
            chosen = group[group["selected_run"]]
            if chosen.empty:
                chosen = group
            row = chosen.iloc[0].to_dict()
        else:
            row = group.iloc[0].to_dict()
            for col in ["KGE", "NSE", "PBIAS", "RMSE", "valid_observation_count", "valid_simulation_count", "valid_days"]:
                row[col] = float(group[col].median()) if group[col].notna().any() else math.nan
            row["seed_or_restart"] = "median_across_seeds"
            row["selected_run"] = True
        row.update({"basin_id": basin, "paradigm": paradigm, "structure": structure, "period": period})
        rows.append(row)
    return pd.DataFrame(rows)


def _collapse_basin_year(runs: pd.DataFrame) -> pd.DataFrame:
    collapsed: list[dict[str, Any]] = []
    for keys, group in runs.groupby(
        ["basin_id", "paradigm", "structure", "period", "water_year"],
        sort=True,
        observed=True,
    ):
        basin, paradigm, structure, period, wy = keys
        complete = group[group["complete_year"]]
        valid = group[group["valid_year"] & np.isfinite(group["Delta_CT"])]
        base = group.iloc[0].to_dict()
        base.update(
            {
                "basin_id": basin,
                "paradigm": paradigm,
                "structure": structure,
                "model": STRUCTURE_TO_MODEL[structure],
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
            }
        )
        collapsed.append(base)
    return pd.DataFrame(collapsed)


def _select_sources(candidates: list[dict[str, Any]]) -> list[Path]:
    selected = []
    for row in candidates:
        path = row["path"]
        if row.get("declared_paradigm") == "IC-CMA-ES" and row.get("declared_model") in {"XAJ-Base", "XAJ-TGD", "XAJ-CN"}:
            selected.append(project_file(path))
        elif row.get("declared_paradigm") == "dPL-MLP" and row.get("declared_model") in {"XAJ-Base", "XAJ-TGD", "XAJ-CN"} and "daily_dpl_gpu_compile" in path:
            selected.append(project_file(path))
    return sorted(selected)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for streaming R1 reductions; refusing CPU fallback")
    OUT.mkdir(parents=True, exist_ok=True)
    attrs = pd.read_csv(project_file("manuscript/results/R1/r1_snow_attributes.csv"))
    perf_existing = pd.read_csv(project_file("manuscript/results/R1/r1_basin_level_performance.csv"))
    candidates = discover_daily_candidates()
    manifest, checks = build_basin_manifest(perf_existing, attrs, candidates)
    write_csv("r1_daily_source_inventory.csv", pd.DataFrame(candidates))
    write_csv("r1_basin_manifest.csv", manifest)
    write_json(
        "r1_source_map.json",
        {
            "project_root": str(PROJECT),
            "scope": "project/hydrodiag statistics stage",
            "daily_sources": [str(p.relative_to(PROJECT)) for p in _select_sources(candidates)],
            "daily_reader": "pyarrow row groups; one row group resident at a time",
            "gpu_reduction_device": torch.cuda.get_device_name(0),
            "training_launched": False,
            "calibration_launched": False,
            "inference_launched": False,
            "simulation_launched": False,
            "figures_modified": False,
            "canonical_promotion": False,
            "git_head": run_git("rev-parse", "HEAD"),
            "source_files": current_source_files(),
        },
    )
    source_paths = _select_sources(candidates)
    expected_pairs = {(p, STRUCTURE_TO_MODEL[s]) for p in PARADIGMS for s in STRUCTURES}
    observed_pairs = set()
    metric_rows: list[dict[str, Any]] = []
    ct_rows: list[dict[str, Any]] = []
    for path in source_paths:
        name = path.name.lower()
        if "dpl_xaj_cn" in name:
            pair = ("dPL-MLP", "XAJ-CN")
        elif "dpl_xaj_tgd" in name:
            pair = ("dPL-MLP", "XAJ-TGD")
        elif "dpl_xaj" in name:
            pair = ("dPL-MLP", "XAJ-Base")
        elif "ic_xaj_cn" in name:
            pair = ("IC-CMA-ES", "XAJ-CN")
        elif "ic_xaj_tgd" in name:
            pair = ("IC-CMA-ES", "XAJ-TGD")
        else:
            pair = ("IC-CMA-ES", "XAJ-Base")
        observed_pairs.add(pair)
        file_metrics, file_ct = _read_file_summaries(path)
        metric_rows.extend(file_metrics)
        ct_rows.extend(file_ct)
    missing = sorted(expected_pairs - observed_pairs)
    if missing:
        raise RuntimeError(f"streaming daily gate missing combinations: {missing}")
    perf_seed = pd.DataFrame(metric_rows)
    runs = pd.DataFrame(ct_rows)
    runs = runs.drop(columns=["date_complete"], errors="ignore")
    if perf_seed.empty or runs.empty:
        raise RuntimeError("streaming reductions produced no rows")
    # IC source files are selected restarts; validate before dPL seed collapse.
    ic_selected = perf_seed[perf_seed["paradigm"] == "IC-CMA-ES"].groupby(
        ["basin_id", "structure", "period"], observed=True
    )["seed_or_restart"].nunique()
    if (ic_selected != 1).any():
        raise RuntimeError("IC selected-restart uniqueness failed in streaming source")
    perf_level = _collapse_metric_rows(perf_seed)
    basin_year = _collapse_basin_year(runs)
    snow_attrs = load_snow_attributes()[["basin_id", "frac_snow", "snow_stratum"]]
    basin_year = basin_year.merge(snow_attrs, on="basin_id", how="left", validate="many_to_one")
    runs = runs.merge(snow_attrs, on="basin_id", how="left", validate="many_to_one")
    basin_level = build_basin_level_ct(basin_year, runs)
    perf_level = perf_level.merge(
        basin_level[["basin_id", "paradigm", "structure", "period", "basin_median_Delta_CT"]],
        on=["basin_id", "paradigm", "structure", "period"],
        how="left",
    )
    basin_level = basin_level.merge(
        perf_level[["basin_id", "paradigm", "structure", "period", "KGE"]],
        on=["basin_id", "paradigm", "structure", "period"],
        how="left",
    )
    basin_level["basin_test_KGE"] = basin_level["KGE"]
    basin_level["KGE_pass_0p60"] = basin_level["period"].eq("test") & (basin_level["KGE"] >= SCREEN_THRESHOLD)
    absolute, snow, effects = performance_summaries(perf_level)
    ct_tables = ct_summaries(basin_level.drop(columns=["KGE"], errors="ignore"), perf_level)
    write_csv("r1_basin_year_ct.csv", basin_year)
    write_csv("r1_basin_year_ct_runs.csv", runs)
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
    reconciliation = historical_reconciliation()
    write_csv("historical_ct_reconciliation.csv", reconciliation)
    daily_reason = f"PASS: streaming daily source complete; {len(source_paths)} files; {len(perf_level)} metric rows; {len(runs)} basin-year run rows"
    report = "\n".join(
        [
            "# Staged R1 statistics rebuild audit",
            "",
            "## Verdict",
            "",
            f"- Daily source gate: **PASS** — {daily_reason}.",
            "- Basin-year CT and basin-level CT: **PASS_WITH_GPU_REDUCTIONS**.",
            "- Performance, snow-stratum, threshold, common-pass, and timing tables: **STAGED**.",
            "- Canonical promotion: **BLOCKED / not attempted**.",
            "",
            "## Execution",
            "",
            f"- CUDA device: `{torch.cuda.get_device_name(0)}`.",
            "- CT and daily metric reductions used CUDA tensors; Parquet was read row-group by row-group.",
            "- Each row group passed basin-contiguity checks; near-threshold CT ties use a tiny NumPy float64 correction to match the canonical reducer.",
            "- No full multi-million-row daily DataFrame was materialized.",
            "- Training, calibration, inference, and simulation were not launched in this statistics stage.",
            "",
            "## Invariants",
            "",
            f"- Fixed snow strata: `{checks['stratum_counts']}`; S4+S5={checks['s4_plus_s5']}.",
            f"- Basin-year run rows={len(runs)}; basin-year collapsed rows={len(basin_year)}; basin-level CT rows={len(basin_level)}.",
            f"- Performance rows={len(perf_level)}; unique metric basins={perf_level['basin_id'].nunique()}.",
            "- All outputs remain under `manuscript/cache/r1_rebuild_audit_staged/`.",
            "",
        ]
    )
    (OUT / "r1_rebuild_audit.md").write_text(report, encoding="utf-8")
    write_json(
        "r1_audit_manifest.json",
        {
            "output": str(OUT),
            "daily_gate": daily_reason,
            "daily_loaded": True,
            "daily_files": [str(p.relative_to(PROJECT)) for p in source_paths],
            "gpu_device": torch.cuda.get_device_name(0),
            "basin_year_run_rows": len(runs),
            "basin_year_rows": len(basin_year),
            "basin_level_rows": len(basin_level),
            "performance_rows": len(perf_level),
            "training_calibration_inference_simulation": "none",
            "canonical_promotion": False,
        },
    )
    print(f"output={OUT}")
    print(f"daily_gate={daily_reason}")
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print(f"basin_year_run_rows={len(runs)} basin_year_rows={len(basin_year)} basin_level_rows={len(basin_level)}")
    print(f"performance_rows={len(perf_level)}")
    print("canonical_promotion=no")


if __name__ == "__main__":
    main()
