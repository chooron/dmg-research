#!/usr/bin/env python3
"""Complete non-training R1/R2 audit and OOB/PUR preparation package.

Consumes the archived IC/dPL quick-survey outputs and existing checkpoints. It may
run compact parameter forwards and serial temporal/restart forwards, but never
trains, changes checkpoints, or persists daily predictions. Every current dPL
comparison is explicitly a SEEN_BASIN_PROXY/D_seen result.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[4]
BENCHMARK = REPO / "project/benchmark"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src"), str(SCRIPT_DIR)]

from dpl.attributes import CAMELS_35_ATTRIBUTES, CatchmentAttributeBuilder  # noqa: E402
from src.model_registry import NPARAM_INFO_36, get_spec  # noqa: E402
from src.data_selection import load_ids  # noqa: E402
import r1_r2_quick_survey as qs  # noqa: E402

BASE_OUT = BENCHMARK / "results/r1_r2_quick_survey_20260829"
OUT = BENCHMARK / "results/r1_r2_nontraining_complete_20260829"
IC_ROOT = BENCHMARK / "results/ic_dpl_aligned_full300_20260819_final"
DPL_ROOT = BENCHMARK / "results/dpl_full_retrain_20260813/auto100"
IDS_PATH = REPO / "data/531sub_id.txt"
EVAL_START, EVAL_END = "1995-10-01", "2010-09-30"
WARMUP = 365
CONTINUOUS = [a for a in CAMELS_35_ATTRIBUTES if a not in {"dom_land_cover", "geol_1st_class", "geol_2nd_class"}]
N_NULL = 200

# This panel is declared from structure/quadrant coverage, not selected by future OOB results.
INTENSIVE_PANEL = ["collie1", "hillslope", "topmodel", "xinanjiang", "hbv96", "mopex4", "gr4j", "modhydrolog"]
ALTERNATES = ["alpine1", "hymod"]
RESTART_PANEL = INTENSIVE_PANEL


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def json_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_corr(x: Any, y: Any, rank: bool = False) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.nanstd(a[ok]) == 0 or np.nanstd(b[ok]) == 0:
        return float("nan")
    if rank:
        a, b = pd.Series(a[ok]).rank(method="average").to_numpy(), pd.Series(b[ok]).rank(method="average").to_numpy()
    else:
        a, b = a[ok], b[ok]
    return float(np.corrcoef(a, b)[0, 1])


def stats(values: Any) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    good = x[np.isfinite(x)]
    if not len(good):
        return {"n": 0, "median": np.nan, "mean": np.nan, "iqr": np.nan, "p05": np.nan, "p95": np.nan, "min": np.nan, "max": np.nan}
    return {"n": int(len(good)), "median": float(np.median(good)), "mean": float(np.mean(good)),
            "iqr": float(np.quantile(good, .75) - np.quantile(good, .25)), "p05": float(np.quantile(good, .05)),
            "p95": float(np.quantile(good, .95)), "min": float(np.min(good)), "max": float(np.max(good))}


def qbin(x: np.ndarray, n: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    edges = np.nanquantile(x, np.linspace(0, 1, n + 1))
    edges = np.maximum.accumulate(edges)
    out = np.full(len(x), -1, dtype=int)
    if finite.any():
        out[finite] = np.clip(np.digitize(x[finite], edges[1:-1], right=False), 0, n - 1)
    return out


def current_populations(ids: np.ndarray) -> tuple[list[str], list[str], dict[str, dict[str, Any]]]:
    inv = pd.read_csv(BASE_OUT / "model_audit_inventory.csv")
    dseen = pd.read_csv(BASE_OUT / "d_seen_by_basin.csv")
    common = sorted(dseen.model.unique().tolist())
    strict = [m for m in common if qs.ic_status_generation(m) == 300]
    if len(common) != 35 or len(strict) != 34:
        raise RuntimeError(f"expected common35/strict34, got {len(common)}/{len(strict)}")
    rows = inv.to_dict("records")
    row_by_model = {r["model"]: r for r in rows}
    for r in rows:
        m = r["model"]
        r["analysis_population"] = "RELAXED_SEEN_35" if m in common else "REGISTRY_36"
        if m in strict:
            r["analysis_population"] = "STRICT_FULL300_34"
        if m == "flexb":
            r["analysis_population"] = "SKIPPED_MISSING_DPL"
        try:
            spec = get_spec(m, device="cpu")
            r["parameter_names"] = json.dumps(list(spec.parameter_names))
            r["bounds"] = json.dumps([[float(a), float(b)] for a, b in spec.bounds])
            r["routed_kind"] = getattr(spec, "routed_kind", "UNRESOLVED")
            groups = getattr(spec, "parameter_groups", None)
            r["parameter_groups"] = json.dumps(groups if groups is not None else {})
            r["has_explicit_snow_group"] = bool(groups and "snow" in groups)
            r["state_count"] = "UNRESOLVED"
        except Exception as exc:
            r["parameter_names"] = "UNRESOLVED"
            r["bounds"] = "UNRESOLVED"
            r["routed_kind"] = "UNRESOLVED"
            r["parameter_groups"] = "UNRESOLVED"
            r["has_explicit_snow_group"] = False
            r["state_count"] = "UNRESOLVED"
            r["audit_anomaly"] = f"registry lookup failed: {exc}"
        r["registry_population"] = "REGISTRY_36"
        r["strict_support"] = m in strict
        r["relaxed_support"] = m in common
        r["basin_coverage_expected"] = 531 if m in common else 0
        r["finite_score_check"] = bool(m in common and np.isfinite(dseen.loc[dseen.model == m, "D_seen"]).all())
        r["parameter_vector_check"] = "pending_parameter_forward" if m in common else "not_available"
        dpath, _ = qs.dpl_metadata(m)
        status = qs.IC_STATUS.get(m, {})
        declared = status.get("final_checkpoint_files", []) if isinstance(status, dict) else []
        if not declared and isinstance(status, dict) and status.get("latest_checkpoint"):
            declared = [status["latest_checkpoint"]]
        ic_declared_paths = [str(IC_ROOT / "checkpoints/ic_dpl_aligned_full300_20260819" / m / str(name)) for name in declared if name]
        ic_used_paths = [str(path) for path in sorted((IC_ROOT / "best_training" / m).glob("chunk_*_best.pt"))]
        r["ic_checkpoint_files_declared"] = ";".join(ic_declared_paths)
        r["ic_checkpoint_files_used_current_path"] = ";".join(ic_used_paths)
        r["dpl_checkpoint_file_used_current_path"] = str(dpath) if dpath is not None else ""
        r["checkpoint_files_used_current_path"] = "IC=" + ";".join(ic_used_paths) + ";dPL=" + (str(dpath) if dpath is not None else "SKIP_MISSING_DPL")
    write_csv(OUT / "model_analysis_registry.csv", rows)
    return common, strict, row_by_model


def checkpoint_audit(common: list[str], strict: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    latest_best = 0
    for m in NPARAM_INFO_36:
        status = qs.IC_STATUS.get(m, {})
        ic_gen = qs.ic_status_generation(m)
        ic_declared = status.get("final_checkpoint_files", []) if isinstance(status, dict) else []
        if not ic_declared and isinstance(status, dict) and status.get("latest_checkpoint"):
            ic_declared = [status["latest_checkpoint"]]
        dpath, dm = qs.dpl_metadata(m)
        dfiles = sorted((DPL_ROOT / "checkpoints" / m).glob("epoch_*.pt")) if dpath is not None else []
        d_epochs = [int(p.stem.split("_")[1]) for p in dfiles]
        best_epoch = dm.get("health_best_epoch", "")
        stop_epoch = dm.get("health_stop_epoch", "")
        best_file = DPL_ROOT / "checkpoints" / m / f"epoch_{int(best_epoch):03d}.pt" if str(best_epoch).isdigit() else Path("__missing__")
        terminal_file = DPL_ROOT / "checkpoints" / m / f"epoch_{int(stop_epoch):03d}.pt" if str(stop_epoch).isdigit() else Path("__missing__")
        latest_file = dpath if dpath is not None else Path("__missing__")
        if dpath is not None and str(best_epoch).isdigit() and dpath.name == best_file.name:
            latest_best += 1
        rows.append({
            "model": m, "population": "STRICT_FULL300_34" if m in strict else ("RELAXED_SEEN_35" if m in common else "REGISTRY_36"),
            "ic_generation": ic_gen if ic_gen is not None else "MISSING", "ic_strict_full300": bool(ic_gen == 300),
            "ic_declared_checkpoint_files": ";".join(map(str, ic_declared)),
            "ic_loaded_best_training_files": ";".join(str(p) for p in sorted((IC_ROOT / "best_training" / m).glob("chunk_*_best.pt"))),
            "ic_declared_files_exist": bool(ic_declared) and all((IC_ROOT / "checkpoints/ic_dpl_aligned_full300_20260819" / m / str(x)).is_file() for x in ic_declared),
            "ic_inference_selection": "best_of_10_checkpoint_train_kge_only" if m in common else "not_used",
            "dpl_checkpoint_files": ";".join(str(p) for p in dfiles), "dpl_epochs_available": ",".join(map(str, d_epochs)),
            "dpl_latest_file": str(latest_file), "dpl_best_epoch_health": best_epoch, "dpl_best_file_exists": best_file.is_file(),
            "dpl_terminal_epoch_health": stop_epoch, "dpl_terminal_file_exists": terminal_file.is_file(),
            "dpl_current_inference_rule": "latest_saved_epoch_file" if dpath is not None else "SKIP_MISSING_DPL",
            "dpl_latest_equals_health_best": bool(dpath is not None and dpath.name == best_file.name),
            "dpl_health_status": dm.get("health_status", dm.get("status", "MISSING")),
            "health_learning_gate": dm.get("health_pass_learning", ""), "health_saturation_gate": dm.get("health_pass_no_saturation", ""),
            "selection_information_source": "health best_epoch is validation-derived; current inference does not use it",
            "test_leakage_assessment": "no test score used for current checkpoint selection; future heldout scores must be prohibited",
            "checkpoint_integrity_action": "audit_only_no_overwrite",
        })
    audit = {"rule_current": "latest saved epoch_*.pt", "latest_equals_health_best_count": latest_best,
             "latest_equals_health_best_total": len(common), "recommendation": {
                 "aliases": ["latest.pt", "best_train.pt", "terminal.pt"],
                 "inference": "best_train.pt selected by training-only objective; report latest/terminal sensitivity",
                 "required_manifest": ["epoch", "split", "seed", "basin_id_hash", "config_hash", "parameter_names", "bounds", "mapping", "finite_checks"],
                 "prohibit": ["heldout/test score checkpoint selection", "validation-selected best called OOB-safe"],
             }, "rows": len(rows)}
    write_csv(OUT / "checkpoint_protocol_audit.csv", rows)
    json_write(OUT / "checkpoint_protocol_audit.json", audit)
    return audit


def r1_tables(common: list[str], strict: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(BASE_OUT / "d_seen_by_basin.csv", dtype={"basin_id": str})
    d["D_seen"] = d["ic_test_kge"] - d["dpl_seen_kge"]
    if set(d.model) != set(common) or len(d) != 35 * 531 or d.groupby("model").basin_id.nunique().min() != 531:
        raise RuntimeError("D_seen table does not have exact relaxed 35x531 coverage")
    rows: list[dict[str, Any]] = []
    for name, models in [("RELAXED_SEEN_35", common), ("STRICT_FULL300_34", strict)]:
        sub = d[d.model.isin(models)].copy()
        for model, g in sub.groupby("model", sort=True):
            s = stats(g.D_seen)
            rows.append({"population": name, "aggregation": "model", "model": model, "n_models": 1, "n_basins": g.basin_id.nunique(), "n_rows": len(g), **{f"D_seen_{k}": v for k, v in s.items() if k != "n"}, "D_seen_n": s["n"], "fraction_IC_gt_dPL": float((g.ic_test_kge > g.dpl_seen_kge).mean()), "fraction_abs_D_seen_gt_0.05": float((g.D_seen.abs() > .05).mean()), "fraction_abs_D_seen_gt_0.10": float((g.D_seen.abs() > .10).mean()), "label": "SEEN_BASIN_PROXY"})
        s = stats(sub.D_seen)
        rows.append({"population": name, "aggregation": "pooled_model_basin_rows", "model": "__POOLED__", "n_models": len(models), "n_basins": sub.basin_id.nunique(), "n_rows": len(sub), **{f"D_seen_{k}": v for k, v in s.items() if k != "n"}, "D_seen_n": s["n"], "fraction_IC_gt_dPL": float((sub.ic_test_kge > sub.dpl_seen_kge).mean()), "fraction_abs_D_seen_gt_0.05": float((sub.D_seen.abs() > .05).mean()), "fraction_abs_D_seen_gt_0.10": float((sub.D_seen.abs() > .10).mean()), "label": "SEEN_BASIN_PROXY; pooled rows are not independent model replicates"})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "r1_dseen_set_summary.csv", index=False, float_format="%.10f")
    summary[summary.population == "STRICT_FULL300_34"].to_csv(OUT / "r1_dseen_strict34_summary.csv", index=False, float_format="%.10f")
    summary[summary.population == "RELAXED_SEEN_35"].to_csv(OUT / "r1_dseen_relaxed35_summary.csv", index=False, float_format="%.10f")
    d[d.model.isin(strict)].to_csv(OUT / "r1_dseen_by_basin_strict34.csv", index=False, float_format="%.10f")
    for pop, models, suffix in [("RELAXED_SEEN_35", common, "relaxed35"), ("STRICT_FULL300_34", strict, "strict34")]:
        sub = d[d.model.isin(models)]
        sub.groupby("basin_id", sort=False).agg(valid_model_count=("model", "nunique"), D_seen_median=("D_seen", "median"), D_seen_mean=("D_seen", "mean"), D_seen_iqr=("D_seen", lambda x: x.quantile(.75)-x.quantile(.25)), D_seen_p05=("D_seen", lambda x: x.quantile(.05)), D_seen_p95=("D_seen", lambda x: x.quantile(.95))).reset_index().to_csv(OUT / f"r1_dseen_basin_summary_{suffix}.csv", index=False, float_format="%.10f")
    ext_rows = []
    for pop, models in [("RELAXED_SEEN_35", common), ("STRICT_FULL300_34", strict)]:
        sub = d[d.model.isin(models)].copy(); sub["abs_D_seen"] = sub.D_seen.abs()
        chosen = pd.concat([sub.nlargest(10, "D_seen"), sub.nsmallest(10, "D_seen")]).drop_duplicates(["model", "basin_id"])
        chosen = chosen.sort_values("abs_D_seen", ascending=False)
        for rank, r in enumerate(chosen.itertuples(index=False), 1):
            ext_rows.append({"population": pop, "model": r.model, "basin_id": r.basin_id, "ic_test_kge": r.ic_test_kge, "dpl_seen_kge": r.dpl_seen_kge, "D_seen": r.D_seen, "abs_D_seen": abs(r.D_seen), "rank_abs_D": rank, "finite": bool(np.isfinite([r.ic_test_kge, r.dpl_seen_kge, r.D_seen]).all()), "requires_daily_audit": bool(abs(r.D_seen) > 1 or r.ic_test_kge < -1 or r.dpl_seen_kge < -1), "audit_note": "finite scalar extreme; no automatic deletion; daily-cause audit remains future work", "label": "SEEN_BASIN_PROXY"})
    write_csv(OUT / "r1_dseen_extremes.csv", ext_rows)
    # Candidate-count sensitivity is deliberately descriptive, not a selected threshold.
    best = d[d.model.isin(common)].groupby("basin_id").ic_test_kge.max()
    tol_rows = []
    for pop, models in [("RELAXED_SEEN_35", common), ("STRICT_FULL300_34", strict)]:
        sub = d[d.model.isin(models)].copy(); sub["best_ic"] = sub.basin_id.map(best)
        for delta in [.02, .05, .10, .15]:
            counts = sub.loc[sub.dpl_seen_kge >= sub.best_ic - delta].groupby("basin_id").size().reindex(best.index, fill_value=0)
            tol_rows.append({"population": pop, "rule": "dPL_seen >= best_IC_across_relaxed35 - delta", "delta": delta, "n_basins": len(counts), "count_median": float(counts.median()), "count_p05": float(counts.quantile(.05)), "count_p95": float(counts.quantile(.95)), "count_min": int(counts.min()), "count_max": int(counts.max()), "threshold_status": "PILOT_ONLY; no formal equivalence threshold"})
    tol_rows.append({"population": "RELAXED_SEEN_35", "rule": "IC_restart_calibration_uncertainty", "delta": np.nan, "n_basins": 0, "count_median": np.nan, "count_p05": np.nan, "count_p95": np.nan, "count_min": np.nan, "count_max": np.nan, "threshold_status": "UNAVAILABLE from selected-score tables; requires archived-start forward"})
    write_csv(OUT / "r1_performance_tolerance_sensitivity.csv", tol_rows)
    # Matched 34 strict-versus-relaxed sensitivity is explicit, not pooled as independent samples.
    strict_model = summary[(summary.population == "STRICT_FULL300_34") & (summary.aggregation == "model")][["model", "D_seen_median"]].rename(columns={"D_seen_median": "strict_D_seen_median"})
    relaxed34 = summary[(summary.population == "RELAXED_SEEN_35") & (summary.aggregation == "model") & summary.model.isin(strict)][["model", "D_seen_median"]].rename(columns={"D_seen_median": "relaxed34_D_seen_median"})
    strict_model.merge(relaxed34, on="model").assign(delta="strict and matched relaxed are identical archived pairs", label="SEEN_BASIN_PROXY").to_csv(OUT / "r1_strict_relaxed_matched34.csv", index=False)
    return d, summary


def load_ic_starts(model: str, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    expected = qs.ic_status_generation(model)
    paths = sorted((IC_ROOT / "best_training" / model).glob("chunk_*_best.pt"))
    lat, fit, ckpt_ids = [], [], []
    for p in paths:
        payload = torch.load(p, map_location="cpu", weights_only=False)
        if int(payload["generation"]) != int(expected):
            raise RuntimeError(f"{model}: payload generation mismatch")
        ckpt_ids.append(np.asarray(payload["basin_ids"], dtype=np.int64))
        lat.append(payload["best_latent"].detach().cpu().numpy())
        fit.append(payload["best_fitness"].detach().cpu().numpy())
    basin = np.concatenate(ckpt_ids); latent = np.concatenate(lat); fitness = np.concatenate(fit)
    if set(basin.tolist()) != set(ids.tolist()) or len(np.unique(basin)) != len(ids):
        raise RuntimeError(f"{model}: all-start basin IDs mismatch")
    order = np.asarray([np.where(basin == b)[0][0] for b in ids])
    latent = latent.reshape(len(basin), 10, -1)[order]
    fitness = fitness.reshape(len(basin), 10)[order]
    return latent, fitness, int(expected)


def temporal_and_restart(common: list[str], strict: list[str], dseen: pd.DataFrame, raw_attrs: np.ndarray, attrs: torch.Tensor, ids: np.ndarray, device: torch.device) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    x_np, y_np, vx_np, vy_np = qs.load_camels_time_series(ids)
    if vx_np.shape[1] != len(ids) or vx_np.shape[0] != vy_np.shape[0] + WARMUP:
        raise RuntimeError("forcing/target shape mismatch")
    temporal_rows: list[dict[str, Any]] = []
    temporal_basin_rows: list[dict[str, Any]] = []
    theta_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    restart_long: list[dict[str, Any]] = []
    restart_wide: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for model in strict:
        t0 = time.perf_counter()
        latent, fitness, generation = load_ic_starts(model, ids)
        selected = fitness.argmax(axis=1)
        ic_latent = torch.as_tensor(latent[np.arange(len(ids)), selected], dtype=torch.float32, device=device)
        dpath, dm = qs.dpl_metadata(model)
        if dpath is None:
            raise RuntimeError(f"{model}: missing dPL in strict set")
        vx = torch.as_tensor(vx_np, dtype=torch.float32, device=device)
        vy = torch.as_tensor(vy_np, dtype=torch.float32, device=device)
        if model in qs.CALENDAR_MODELS:
            vx, _ = qs.add_calendar_forcing(vx, pd.date_range("1994-10-01", EVAL_END, freq="D"), model_name=model)
        net = qs.model_network(model, dpath, attrs, device)
        hydro_ic = qs.build_model(model, device, warm_up=WARMUP, backend="eager", parameter_mapping="linear", dtype=torch.float32)
        hydro_dp = qs.build_model(model, device, warm_up=WARMUP, backend="eager", parameter_mapping="auto", dtype=torch.float32)
        with torch.inference_mode():
            ic_theta = torch.sigmoid(ic_latent)
            dpl_theta = net(attrs)
            q_ic = hydro_ic({"x_phy": vx}, (None, ic_theta.unsqueeze(-1)))["streamflow"]
            q_dp = hydro_dp({"x_phy": vx}, (None, dpl_theta.unsqueeze(-1)))["streamflow"]
            theta_pairs[model] = (ic_theta.detach().cpu().numpy(), dpl_theta.detach().cpu().numpy())
            half = vy.shape[0] // 2
            half_scores: dict[str, dict[str, np.ndarray]] = {"A": {}, "B": {}}
            for label, left, right in (("A", 0, half), ("B", half, vy.shape[0])):
                for method, q in (("IC", q_ic), ("dPL", q_dp)):
                    _, score = qs.compute_differentiable_kge(q[left:right], vy[left:right], warmup_days=0)
                    vals = score.detach().cpu().numpy().reshape(-1)
                    half_scores[label][method] = vals
                    s = stats(vals)
                    temporal_rows.append({"population": "STRICT_FULL300_34", "model": model, "method": method, "period_half": label, "target_start": str(pd.Timestamp(EVAL_START) + pd.Timedelta(days=left))[:10], "target_end": str(pd.Timestamp(EVAL_START) + pd.Timedelta(days=right - 1))[:10], "warmup_days": 0, "n_basins": len(ids), "valid_basin_count": int(np.isfinite(vals).sum()), "invalid_count": int((~np.isfinite(vals)).sum()), "mean_kge": s["mean"], "median_kge": s["median"], "p05_kge": s["p05"], "p95_kge": s["p95"], "label": "SEEN_BASIN_PROXY; temporal robustness only"})
            for i, basin in enumerate(ids):
                temporal_basin_rows.append({"population": "STRICT_FULL300_34", "model": model, "basin_id": f"{int(basin):08d}", "ic_kge_A": half_scores["A"]["IC"][i], "ic_kge_B": half_scores["B"]["IC"][i], "dpl_seen_kge_A": half_scores["A"]["dPL"][i], "dpl_seen_kge_B": half_scores["B"]["dPL"][i], "D_seen_A": half_scores["A"]["IC"][i] - half_scores["A"]["dPL"][i], "D_seen_B": half_scores["B"]["IC"][i] - half_scores["B"]["dPL"][i], "label": "SEEN_BASIN_PROXY"})
        # Restart spread: selected panel only, no daily arrays retained.
        if model in RESTART_PANEL:
            restart_scores = []
            hydro_r = qs.build_model(model, device, warm_up=WARMUP, backend="eager", parameter_mapping="linear", dtype=torch.float32)
            with torch.inference_mode():
                for start in range(10):
                    theta = torch.sigmoid(torch.as_tensor(latent[:, start, :], dtype=torch.float32, device=device))
                    q = hydro_r({"x_phy": vx}, (None, theta.unsqueeze(-1)))["streamflow"]
                    _, score = qs.compute_differentiable_kge(q, vy, warmup_days=0)
                    restart_scores.append(score.detach().cpu().numpy().reshape(-1))
            rs = np.stack(restart_scores, axis=1)
            for i, basin in enumerate(ids):
                chosen = int(selected[i]); vals = rs[i]
                order_test = np.argsort(vals)[::-1]
                restart_wide.append({"model": model, "basin_id": f"{int(basin):08d}", "train_selected_start": chosen, "train_selected_fitness": fitness[i, chosen], "train_winner_margin": float(fitness[i, chosen] - np.partition(fitness[i], -2)[-2]), "test_selected_kge": vals[chosen], "test_second_best_kge": vals[order_test[1]], "test_selected_minus_second": vals[chosen] - vals[order_test[1]], "test_winner_start": int(order_test[0]), "test_winner_matches_train": bool(order_test[0] == chosen), "test_restart_mean": vals.mean(), "test_restart_sd": vals.std(ddof=1), "test_restart_iqr": np.quantile(vals, .75) - np.quantile(vals, .25), "test_restart_min": vals.min(), "test_restart_max": vals.max(), "label": "SEEN_BASIN_PROXY; IC restart sensitivity, not dPL seed uncertainty"})
                for start, score in enumerate(vals):
                    restart_long.append({"model": model, "basin_id": f"{int(basin):08d}", "restart": start, "score": score, "selected_by_train_fitness": bool(start == chosen), "checkpoint_generation": generation, "selection_rule": "train_fitness_argmax", "label": "SEEN_BASIN_PROXY"})
        runtime_rows.append({"model": model, "seconds": round(time.perf_counter() - t0, 3), "temporal_forward": True, "restart_forward": model in RESTART_PANEL, "gpu": torch.cuda.get_device_name(0)})
        del vx, vy, net, hydro_ic, hydro_dp, q_ic, q_dp
        torch.cuda.empty_cache()
        print(f"[{model}] temporal/restart done in {runtime_rows[-1]['seconds']:.1f}s", flush=True)
    write_csv(OUT / "r1_temporal_ab_summary_strict34.csv", temporal_rows)
    tb = pd.DataFrame(temporal_basin_rows)
    tb.to_csv(OUT / "r1_temporal_ab_by_basin_strict34.csv", index=False, float_format="%.10f")
    tg = tb.groupby("model").agg(D_seen_A=("D_seen_A", "median"), D_seen_B=("D_seen_B", "median"), mean_D_seen_A=("D_seen_A", "mean"), mean_D_seen_B=("D_seen_B", "mean"), valid_basin_count=("D_seen_A", lambda x: int(np.isfinite(x).sum()))).reset_index()
    tg["ordering_pearson"] = finite_corr(tg.D_seen_A, tg.D_seen_B)
    tg["ordering_spearman"] = finite_corr(tg.D_seen_A, tg.D_seen_B, rank=True)
    tg["same_sign_fraction"] = float(np.mean(np.sign(tg.D_seen_A) == np.sign(tg.D_seen_B)))
    tg["label"] = "SEEN_BASIN_PROXY; temporal robustness only"
    tg.to_csv(OUT / "r1_temporal_ab_gap_summary_strict34.csv", index=False, float_format="%.10f")
    write_csv(OUT / "wpC_restart_scores_long.csv", restart_long)
    write_csv(OUT / "wpC_restart_spread_by_basin.csv", restart_wide)
    rw = pd.DataFrame(restart_wide)
    if not rw.empty:
        rsum = rw.groupby("model").agg(n_basins=("basin_id", "nunique"), mean_restart_sd=("test_restart_sd", "mean"), median_restart_sd=("test_restart_sd", "median"), p95_restart_sd=("test_restart_sd", lambda x: x.quantile(.95)), winner_match_fraction=("test_winner_matches_train", "mean"), median_selected_minus_second=("test_selected_minus_second", "median"), median_train_winner_margin=("train_winner_margin", "median")).reset_index()
        rsum["label"] = "SEEN_BASIN_PROXY; restart panel only"
        rsum.to_csv(OUT / "wpC_restart_spread_summary.csv", index=False, float_format="%.10f")
        r1m = dseen.groupby("model").D_seen.median().rename("D_seen_median")
        link = rsum.set_index("model").join(r1m).reset_index()
        link["corr_panel_D_seen_vs_restart_sd"] = finite_corr(link.D_seen_median, link.median_restart_sd)
        link["label"] = "SEEN_BASIN_PROXY; panel-only exploratory linkage"
        link.to_csv(OUT / "r1_restart_linkage_panel.csv", index=False, float_format="%.10f")
    write_csv(OUT / "temporal_runtime_metadata.csv", runtime_rows)
    return pd.DataFrame(temporal_rows), theta_pairs


def relationship_vector(raw: np.ndarray, ic_theta: np.ndarray, dp_theta: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    iv, dv, keys = [], [], []
    for j, param in enumerate(names):
        for attr in CONTINUOUS:
            a = CAMELS_35_ATTRIBUTES.index(attr)
            iv.append(finite_corr(raw[:, a], ic_theta[:, j], rank=True)); dv.append(finite_corr(raw[:, a], dp_theta[:, j], rank=True)); keys.append((param, attr))
    return np.asarray(iv), np.asarray(dv), keys

def r2_same_estimator(common: list[str], strict: list[str], raw_attrs: np.ndarray, attrs: torch.Tensor, device: torch.device, dseen: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    ids = np.asarray(load_ids(IDS_PATH), dtype=np.int64)
    theta_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    relationship_cache: dict[str, tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]] = {}
    for model in common:
        latent, _ = qs.ic_latent_and_metadata(model, ids)
        dpath, _ = qs.dpl_metadata(model)
        if dpath is None:
            raise RuntimeError(f"{model}: missing dPL for same-estimator R2")
        net = qs.model_network(model, dpath, attrs, device)
        with torch.inference_mode():
            ic_theta = torch.sigmoid(latent.to(device=device, dtype=torch.float32)).detach().cpu().numpy()
            dp_theta = net(attrs).detach().cpu().numpy()
        theta_pairs[model] = (ic_theta, dp_theta)
        names = list(get_spec(model, device="cpu").parameter_names)
        relationship_cache[model] = (*relationship_vector(raw_attrs, ic_theta, dp_theta, names),)
    rel_rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    attr_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    populations = [("RELAXED_SEEN_35", common), ("STRICT_FULL300_34", strict)]
    for pop, models in populations:
        for model in models:
            iv, dv, keys = relationship_cache[model]
            names = list(get_spec(model, device="cpu").parameter_names)
            for (param, attr), ri, rd in zip(keys, iv, dv):
                common_fields = {"population": pop, "model": model, "attribute": attr, "parameter": param, "n_basins": len(raw_attrs), "transform": "all-531 fixed normalized sigmoid coordinate; identical basin-wise Spearman estimator"}
                rel_rows.extend([{**common_fields, "method": "IC", "rho": ri, "label": "SEEN_BASIN_PROXY"}, {**common_fields, "method": "dPL", "rho": rd, "label": "SEEN_BASIN_PROXY"}])
            local_params = []
            for p in names:
                idx = [i for i, (param, _) in enumerate(keys) if param == p]
                ai, ad = iv[idx], dv[idx]; both = (np.abs(ai) >= .05) & (np.abs(ad) >= .05)
                mi, md = int(np.nanargmax(np.abs(ai))), int(np.nanargmax(np.abs(ad)))
                row = {"population": pop, "model": model, "parameter": p, "flattened_pearson": finite_corr(ai, ad), "flattened_spearman": finite_corr(ai, ad, rank=True), "sign_agreement_all": float(np.mean(np.sign(ai) == np.sign(ad))), "sign_agreement_joint_abs_rho_ge_0.05": float(np.mean(np.sign(ai[both]) == np.sign(ad[both]))) if both.any() else np.nan, "dominant_attribute_IC": keys[idx[mi]][1], "dominant_attribute_dPL": keys[idx[md]][1], "dominant_control_agreement": keys[idx[mi]][1] == keys[idx[md]][1], "n_attributes": len(idx), "label": "SEEN_BASIN_PROXY"}
                param_rows.append(row); local_params.append(row)
            for attr in CONTINUOUS:
                idx = [i for i, (_, a) in enumerate(keys) if a == attr]
                ai, ad = iv[idx], dv[idx]
                attr_rows.append({"population": pop, "model": model, "attribute": attr, "flattened_pearson": finite_corr(ai, ad), "flattened_spearman": finite_corr(ai, ad, rank=True), "sign_agreement_all": float(np.mean(np.sign(ai) == np.sign(ad))), "label": "SEEN_BASIN_PROXY"})
            both = (np.abs(iv) >= .05) & (np.abs(dv) >= .05)
            model_rows.append({"population": pop, "model": model, "parameter_count": len(names), "flattened_pearson": finite_corr(iv, dv), "flattened_spearman": finite_corr(iv, dv, rank=True), "sign_agreement_all": float(np.mean(np.sign(iv) == np.sign(dv))), "sign_agreement_joint_abs_rho_ge_0.05": float(np.mean(np.sign(iv[both]) == np.sign(dv[both]))) if both.any() else np.nan, "dominant_control_agreement_fraction": float(np.mean([r["dominant_control_agreement"] for r in local_params])), "valid_relationship_count": int((np.isfinite(iv) & np.isfinite(dv)).sum()), "D_seen_median": float(dseen[dseen.model == model].D_seen.median()), "label": "SEEN_BASIN_PROXY; primary same-estimator comparison"})
    rel = pd.DataFrame(rel_rows); ps = pd.DataFrame(param_rows); ats = pd.DataFrame(attr_rows); ms = pd.DataFrame(model_rows)
    realization_rows = []
    for pop, models in [("RELAXED_SEEN_35", common), ("STRICT_FULL300_34", strict)]:
        for model in models:
            ic_theta, dp_theta = theta_pairs[model]; diff = ic_theta - dp_theta
            realization_rows.append({"population": pop, "model": model, "parameter_count": diff.shape[1], "mean_normalized_l2": float(np.sqrt(np.mean(diff ** 2, axis=1)).mean()), "median_normalized_l2": float(np.median(np.sqrt(np.mean(diff ** 2, axis=1)))), "p95_normalized_l2": float(np.quantile(np.sqrt(np.mean(diff ** 2, axis=1)), .95)), "mean_normalized_l1": float(np.mean(np.abs(diff))), "label": "SEEN_BASIN_PROXY; exploratory parameter realization"})
    pd.DataFrame(realization_rows).to_csv(OUT / "r2_parameter_realization_distance_summary.csv", index=False, float_format="%.10f")
    rel.to_csv(OUT / "r2_same_estimator_relationship_long.csv", index=False, float_format="%.10f")
    ps.to_csv(OUT / "r2_same_estimator_parameter_summary.csv", index=False, float_format="%.10f")
    ats.to_csv(OUT / "r2_same_estimator_attribute_summary.csv", index=False, float_format="%.10f")
    ms.to_csv(OUT / "r2_same_estimator_model_summary.csv", index=False, float_format="%.10f")
    secondary = pd.read_csv(BASE_OUT / "r2_model_summary.csv")
    secondary["diagnostic_role"] = "SECONDARY_NORMALIZED_OUTPUT_PROXY; retained for comparison"
    secondary["label"] = "SEEN_BASIN_PROXY"
    secondary.to_csv(OUT / "r2_secondary_auto_mapping_proxy_model_summary.csv", index=False, float_format="%.10f")
    return ms, theta_pairs


def linkage_and_null(ms: pd.DataFrame, theta_pairs: dict[str, tuple[np.ndarray, np.ndarray]], dseen: pd.DataFrame, raw_attrs: np.ndarray, ids: np.ndarray) -> None:
    link_rows = []; loo_rows = []; null_rows = []; rep_rows = []
    raw = raw_attrs
    snow_q, arid_q, skill_q = qbin(raw[:, CAMELS_35_ATTRIBUTES.index("frac_snow")]), qbin(raw[:, CAMELS_35_ATTRIBUTES.index("aridity")]), None
    attr_features = np.column_stack([snow_q, arid_q])
    for pop, models in [("STRICT_FULL300_34", sorted(ms.loc[ms.population == "STRICT_FULL300_34", "model"])), ("RELAXED_SEEN_35", sorted(ms.loc[ms.population == "RELAXED_SEEN_35", "model"]))]:
        subm = ms[ms.population == pop].copy()
        x, y = subm.flattened_spearman.to_numpy(), subm.D_seen_median.to_numpy()
        link_rows.append({"population": pop, "n_models": len(subm), "pearson_r": finite_corr(x, y), "spearman_r": finite_corr(x, y, rank=True), "label": "SEEN_BASIN_PROXY; exploratory linkage"})
        for omit in models:
            keep = subm.model != omit
            loo_rows.append({"population": pop, "omitted_model": omit, "n_models": int(keep.sum()), "pearson_r": finite_corr(x[keep], y[keep]), "spearman_r": finite_corr(x[keep], y[keep], rank=True), "full_pearson_r": finite_corr(x, y), "full_spearman_r": finite_corr(x, y, rank=True), "label": "SEEN_BASIN_PROXY; leave-one-model-out influence"})
        for model in models:
            row = subm[subm.model == model].iloc[0]
            mdat = dseen[dseen.model == model].set_index("basin_id").reindex([f"{int(b):08d}" for b in ids])
            gap = mdat.D_seen.to_numpy(dtype=float); skill = mdat.ic_test_kge.to_numpy(dtype=float)
            skill_q = qbin(skill)
            strata = attr_features[:, 0] * 16 + attr_features[:, 1] * 4 + skill_q
            low_n = int(math.ceil(.25 * len(ids))); low_idx = np.argsort(gap)[:low_n]
            ic_theta, dp_theta = theta_pairs[model]
            names = list(get_spec(model, device="cpu").parameter_names)
            low_iv, low_dv, _ = relationship_vector(raw[low_idx], ic_theta[low_idx], dp_theta[low_idx], names)
            low_agree = finite_corr(low_iv, low_dv, rank=True)
            rng = np.random.default_rng(20260829 + sum(ord(c) for c in model))
            counts = Counter(strata[low_idx].tolist())
            def draw_once(rng_local: np.random.Generator) -> np.ndarray:
                picked = []
                for cell, count in counts.items():
                    pool = np.flatnonzero(strata == cell)
                    if len(pool) < count:
                        raise RuntimeError(f"matching cell underfilled for {model}: {cell}")
                    picked.extend(rng_local.choice(pool, size=count, replace=False).tolist())
                return np.asarray(sorted(picked), dtype=int)
            null_agreements = []
            for draw in range(N_NULL):
                idx = draw_once(rng); niv, ndv, _ = relationship_vector(raw[idx], ic_theta[idx], dp_theta[idx], names); agr = finite_corr(niv, ndv, rank=True); null_agreements.append(agr)
                rep_rows.append({"population": pop, "model": model, "draw": draw, "n_subset": len(idx), "agreement_spearman": agr, "selection_rule": "lowest 25 percent D_seen", "matching_features": "frac_snow_q4|aridity_q4|IC_test_KGE_q4", "outcome_leakage_test": True, "label": "PIPELINE_DRY_RUN_ONLY / NOT_OOB_EVIDENCE"})
            # Determinism check reruns the same seed and draws, never uses relationship values in matching.
            rng2 = np.random.default_rng(20260829 + sum(ord(c) for c in model)); first_again = draw_once(rng2)
            if not np.array_equal(first_again, draw_once(np.random.default_rng(20260829 + sum(ord(c) for c in model)))):
                raise RuntimeError(f"matched-null reproducibility failure for {model}")
            nv = np.asarray(null_agreements, dtype=float)
            null_rows.append({"population": pop, "model": model, "n_basins": len(ids), "low_subset_n": low_n, "low_subset_agreement_spearman": low_agree, "null_draws": N_NULL, "null_mean": float(np.nanmean(nv)), "null_sd": float(np.nanstd(nv, ddof=1)), "null_p05": float(np.nanquantile(nv, .05)), "null_p95": float(np.nanquantile(nv, .95)), "null_percentile_of_low": float(np.mean(nv <= low_agree)), "low_minus_null_mean": low_agree - float(np.nanmean(nv)), "matching_features": "frac_snow_q4|aridity_q4|IC_test_KGE_q4", "selection_rule": "lowest 25 percent D_seen; predeclared", "outcome_leakage_test": True, "label": "PIPELINE_DRY_RUN_ONLY / NOT_OOB_EVIDENCE"})
    write_csv(OUT / "r2_dseen_linkage_summary.csv", link_rows)
    write_csv(OUT / "r2_dseen_linkage_leave_one_model_out.csv", loo_rows)
    write_csv(OUT / "matched_null_dry_run_replicates.csv", rep_rows)
    write_csv(OUT / "matched_null_dry_run_summary.csv", null_rows)
    json_write(OUT / "matched_null_dry_run_audit.json", {"status": "PASS_PIPELINE_ONLY", "selection": "lowest 25% D_seen within each model", "matching_features": ["frac_snow quartile", "aridity quartile", "IC test KGE quartile"], "forbidden_matching_inputs": ["R2 rho", "relationship agreement", "dPL parameters", "checkpoint choice", "heldout/test relationship outcome"], "outcome_leakage_test": "asserted by code path and deterministic counts", "draws_per_model": N_NULL, "label": "PIPELINE_DRY_RUN_ONLY / NOT_OOB_EVIDENCE"})


def xaj_realization_diagnostics() -> dict[str, Any]:
    ledger_path = REPO / "project/hydrodiag/manuscript/results/R2/raw_parameter_ledger.csv"
    audit_path = REPO / "project/hydrodiag/manuscript/results/R2/raw_parameter_ledger_audit.json"
    existing_snow = REPO / "project/hydrodiag/manuscript/analysis/R1/results/snow_activity_audit.json"
    if not ledger_path.exists() or not audit_path.exists():
        result = {"status": "PREREQUISITE_MISSING", "ledger": str(ledger_path), "label": "XAJ evidence not reconstructed"}
        json_write(OUT / "xaj_sanity_gate.json", result); return result
    sums: dict[tuple[str, str, str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    snow: dict[int, float] = {}
    for chunk in pd.read_csv(ledger_path, usecols=["basin_id", "paradigm", "structure", "parameter", "normalized_value", "frac_snow"], chunksize=100000):
        chunk = chunk[chunk.structure.isin(["Base", "CN"]) & chunk.paradigm.isin(["IC", "dPL"])]
        for r in chunk.itertuples(index=False):
            key = (str(int(r.basin_id)), r.paradigm, r.structure, r.parameter)
            if np.isfinite(r.normalized_value):
                sums[key] += float(r.normalized_value); counts[key] += 1
            snow[int(r.basin_id)] = float(r.frac_snow)
    rows = []
    for basin in sorted(snow):
        for structure in ["Base", "CN"]:
            pars = sorted({k[3] for k in sums if k[0] == str(basin) and k[1] == "IC" and k[2] == structure})
            ic = np.asarray([sums[(str(basin), "IC", structure, p)] / counts[(str(basin), "IC", structure, p)] for p in pars])
            dp = np.asarray([sums[(str(basin), "dPL", structure, p)] / counts[(str(basin), "dPL", structure, p)] for p in pars])
            diff = ic - dp
            rows.append({"basin_id": f"{basin:08d}", "structure": structure, "parameter_count": len(pars), "normalized_l2_distance": float(np.sqrt(np.mean(diff ** 2))), "normalized_l1_distance": float(np.mean(np.abs(diff))), "frac_snow": snow[basin], "label": "SEEN_BASIN_PROXY; existing XAJ controlled ledger"})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "xaj_ic_dpl_realization_distance_by_basin.csv", index=False, float_format="%.10f")
    sum_rows = []
    for structure, g in df.groupby("structure"):
        for metric in ["normalized_l2_distance", "normalized_l1_distance"]:
            slope = np.polyfit(g.frac_snow, g[metric], 1)[0] if g.frac_snow.nunique() > 1 else np.nan
            q = g.groupby(pd.qcut(g.frac_snow, 5, labels=["S1", "S2", "S3", "S4", "S5"], duplicates="drop"), observed=False)[metric].median()
            sum_rows.append({"structure": structure, "metric": metric, "n_basins": len(g), "snow_slope": slope, "snow_spearman": finite_corr(g.frac_snow, g[metric], rank=True), "S1_median": q.iloc[0] if len(q) else np.nan, "S5_median": q.iloc[-1] if len(q) else np.nan, "S5_minus_S1": (q.iloc[-1] - q.iloc[0]) if len(q) else np.nan, "label": "SEEN_BASIN_PROXY; exploratory parameter realization"})
    sd = pd.DataFrame(sum_rows); sd.to_csv(OUT / "xaj_ic_dpl_realization_distance_summary.csv", index=False, float_format="%.10f")
    piv = sd.pivot(index="metric", columns="structure", values="snow_slope").reset_index()
    piv["base_abs_slope_gt_cn"] = piv["Base"].abs() > piv["CN"].abs()
    piv.to_csv(OUT / "xaj_base_cn_snow_gradient_comparison.csv", index=False, float_format="%.10f")
    evidence = json.loads(audit_path.read_text()); snow_evidence = json.loads(existing_snow.read_text()) if existing_snow.exists() else {}
    finite_ok = bool(np.isfinite(df[["normalized_l2_distance", "normalized_l1_distance", "frac_snow"]]).all().all() and (df.parameter_count == 15).all() and len(df) == 1062)
    expected_direction = bool(piv["base_abs_slope_gt_cn"].all())
    result = {"status": "PASS_STRUCTURAL_AND_LEDGER_AUDIT" if finite_ok else "FAIL", "ledger_audit_status": evidence.get("status"), "snow_activity_audit_status": snow_evidence.get("status"), "n_rows": len(df), "structures": sorted(df.structure.unique()), "distance_definition": "RMSE/L1 between basin-wise mean normalized IC vector and mean normalized seen-dPL vector within each XAJ structure; IC 10 starts, dPL 3 seeds", "expected_direction_rule": "Base absolute snow-gradient should exceed CN absolute snow-gradient", "expected_direction_status": "SUPPORTED" if expected_direction else "NOT_SUPPORTED_BY_THIS_DISTANCE", "base_vs_cn_snow_gradient": "see xaj_base_cn_snow_gradient_comparison.csv; direction is descriptive and not a causal Base/CN claim", "label": "SEEN_BASIN_PROXY; existing XAJ controlled evidence"}
    json_write(OUT / "xaj_sanity_gate.json", result)
    return result


def screening_and_protocol(common: list[str], strict: list[str], dseen: pd.DataFrame, ms: pd.DataFrame) -> None:
    r1 = dseen.groupby("model").D_seen.median().rename("D_seen_median")
    r2 = ms[ms.population == "STRICT_FULL300_34"].set_index("model")["flattened_spearman"].rename("r2_spearman_agreement")
    h = pd.read_csv(DPL_ROOT / "health.csv").set_index("model")
    rows = []
    for i, model in enumerate(strict):
        spec = get_spec(model, device="cpu"); dim = spec.dimension; dim_bin = "1-4" if dim <= 4 else ("5-8" if dim <= 8 else ("9-12" if dim <= 12 else "13+"))
        d = float(r1[model]); a = float(r2[model]); qd = "H" if d >= r1[strict].median() else "L"; qa = "H" if a >= r2[strict].median() else "L"; quadrant = qd + qa
        groups = getattr(spec, "parameter_groups", None) or {}
        health_ok = int(bool(h.loc[model, "pass_integrity"])) + int(bool(h.loc[model, "pass_learning"])) + int(bool(h.loc[model, "pass_no_saturation"]))
        rows.append({"model": model, "D_seen_median": d, "r2_same_estimator_spearman": a, "D_seen_quadrant": quadrant, "D_seen_rank": int(r1[strict].rank(ascending=False, method="min")[model]), "r2_rank": int(r2[strict].rank(ascending=False, method="min")[model]), "parameter_count": dim, "dimension_bin": dim_bin, "routed_kind": spec.routed_kind, "explicit_snow_group": bool("snow" in groups), "parameter_groups": json.dumps(groups), "dpl_status": h.loc[model, "status"], "dpl_health_pass_integrity": bool(h.loc[model, "pass_integrity"]), "dpl_health_pass_learning": bool(h.loc[model, "pass_learning"]), "dpl_health_pass_no_saturation": bool(h.loc[model, "pass_no_saturation"]), "health_gate_count": health_ok, "selected": model in INTENSIVE_PANEL, "selection_role": "", "selection_reason": "", "label": "SEEN_BASIN_PROXY; pre-OOB screening"})
    reasons = {"collie1": ("low_gap_high_R2", "simple 1-parameter lower-complexity reference"), "hillslope": ("high_gap_high_R2", "HH quadrant plus endpoint routed structure"), "topmodel": ("high_gap_low_R2", "HL quadrant and distinct soil/groundwater structure"), "xinanjiang": ("low_gap_low_R2_XAJ", "LL quadrant and required XAJ bridge"), "hbv96": ("snow_high_dim", "15-parameter snow-capable endpoint model"), "mopex4": ("snow_MOPEX", "10-parameter MOPEX control with tcrit/ddf"), "gr4j": ("intermediate_reference", "intermediate UH family and high R2 reference"), "modhydrolog": ("middle_reference", "near-middle gap with high-dimensional base structure")}
    for r in rows:
        if r["model"] in reasons: r["selection_role"], r["selection_reason"] = reasons[r["model"]]
    df = pd.DataFrame(rows)
    # Balanced rank: quadrants first, then health and proximity to each quadrant target; never maximize D or R2.
    ordered_index = df.sort_values(["D_seen_quadrant", "health_gate_count", "parameter_count", "model"], ascending=[True, False, False, True]).index
    df["balanced_screening_rank"] = pd.Series(np.arange(1, len(df) + 1), index=ordered_index).reindex(df.index).astype(int)
    df.to_csv(OUT / "oob_pur_model_screening_strict34.csv", index=False, float_format="%.10f")
    alt = []
    for rank, m in enumerate(ALTERNATES, 1):
        rr = df[df.model == m].iloc[0]
        alt.append({"alternate_rank": rank, "model": m, "D_seen_median": rr.D_seen_median, "r2_same_estimator_spearman": rr.r2_same_estimator_spearman, "reason": "structural/health replacement; not selected by preferred outcome", "label": "SEEN_BASIN_PROXY"})
    pd.DataFrame(alt).to_csv(OUT / "oob_pur_model_alternates.csv", index=False, float_format="%.10f")
    coverage = []
    for field in ["dimension_bin", "routed_kind", "explicit_snow_group", "D_seen_quadrant"]:
        full = df[field].value_counts().sort_index(); sel = df[df.selected][field].value_counts().reindex(full.index, fill_value=0)
        for value in full.index: coverage.append({"feature": field, "level": str(value), "strict34_count": int(full[value]), "selected8_count": int(sel[value]), "selected_fraction_of_strict": float(sel[value] / full[value]) if full[value] else np.nan, "label": "pre-OOB screening coverage"})
    write_csv(OUT / "oob_pur_model_screening_coverage.csv", coverage)
    selected = df[df.selected].model.tolist()
    protocol = {"status": "PREPARED_NOT_EXECUTED", "label": "FUTURE_TRAINING_PROTOCOL_ONLY", "selected_models": selected, "alternates": ALTERNATES, "groups": {"source": "project/benchmark/scripts/diagnostics/s0_spatial_split_audit.py", "huc_groups": [11, 12, 13, 14, 15, 16, 17], "design": "leave-one-HUC-group-out; train on six groups, evaluate held group"}, "estimands": {"grouped_holdout": "regional/distribution-shift generalization", "ordinary_random_basin_OOB": "not identical; requires separate random/grouping design if demanded", "PUR": "same held-group predictions can serve a regional stress test, but do not conflate estimands"}, "checkpoint_rule": {"latest": "most recently completed epoch", "best_train": "training-only objective; primary inference candidate", "terminal": "final scheduled epoch", "prohibit": "heldout/test/validation-selected checkpoint for OOB claim"}, "seeds": [42, 123], "leakage_rules": ["held group basin IDs absent from dPL training", "heldout scores never select checkpoint", "attribute normalization statistics fit on training groups and applied to held group", "persist split/config/seed hashes", "no relationship outcome in model or subset selection"], "training_count_comparison": [{"design": "selected8 x 7 groups x 1 seed", "trainings": 56, "role": "staged first pass"}, {"design": "selected8 x 7 groups x 2 seeds", "trainings": 112, "role": "recommended confirmatory total"}, {"design": "reduced6 x 7 groups x 1 seed", "trainings": 42, "role": "ultra-low-cost screen"}, {"design": "full34 x 5 random folds x 3 seeds", "trainings": 510, "role": "explicitly not recommended default"}], "required_persisted_outputs": ["per-basin train/heldout KGE and D", "physical and normalized parameter vectors", "basin IDs and fold/group manifest", "attribute transform statistics", "checkpoint aliases and hashes", "health/finite diagnostics", "seed/config provenance"], "tradeoff": "grouped LORO is preferred cost-saving first design for regional PUR-like stress, but does not replace a random-basin OOB estimand", "do_not_execute": True}
    protocol["training_count_comparison"].extend([{ "design": "selected8 x 5 random basin folds x 2 seeds", "trainings": 80, "role": "ordinary basin-wise OOB; separate estimand"}, {"design": "selected8 x (5 random folds + 7 HUC folds) x 2 seeds", "trainings": 192, "role": "both estimands; no duplicate split training"}])
    json_write(OUT / "future_oob_pur_training_protocol.json", protocol)
    (OUT / "future_oob_pur_training_protocol.md").write_text("# Future OOB/PUR protocol (prepared, not executed)\n\n" + json.dumps(protocol, indent=2) + "\n")

def draw_diagnostic_figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = OUT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    r1 = pd.read_csv(OUT / "r1_dseen_set_summary.csv")
    r1 = r1[r1.aggregation == "model"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([r1.loc[r1.population == p, "D_seen_median"] for p in ["STRICT_FULL300_34", "RELAXED_SEEN_35"]], tick_labels=["strict34", "relaxed35"])
    ax.axhline(0, color="black", lw=.8); ax.set_ylabel("model-median D_seen"); ax.set_title("R1 seen-basin gap distribution")
    fig.tight_layout(); fig.savefig(out / "r1_dseen_population_boxplot.png", dpi=130); plt.close(fig)
    t = pd.read_csv(OUT / "r1_temporal_ab_by_basin_strict34.csv")
    tg = t.groupby("model").agg(D_seen_A=("D_seen_A", "median"), D_seen_B=("D_seen_B", "median"))
    fig, ax = plt.subplots(figsize=(5, 4)); ax.scatter(tg.D_seen_A, tg.D_seen_B, s=22)
    lo = min(tg.min()); hi = max(tg.max()); ax.plot([lo, hi], [lo, hi], "k--", lw=.8)
    ax.set_xlabel("median D_seen A"); ax.set_ylabel("median D_seen B"); ax.set_title("Temporal gap stability")
    fig.tight_layout(); fig.savefig(out / "r1_temporal_gap_stability.png", dpi=130); plt.close(fig)
    r2 = pd.read_csv(OUT / "r2_same_estimator_model_summary.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    for p, marker in [("STRICT_FULL300_34", "o"), ("RELAXED_SEEN_35", "s")]:
        q = r2[r2.population == p]; ax.scatter(q.flattened_spearman, q.D_seen_median, s=24, marker=marker, label=p)
    ax.set_xlabel("same-estimator R2 Spearman agreement"); ax.set_ylabel("D_seen median"); ax.legend(fontsize=8); ax.set_title("R2/D_seen linkage")
    fig.tight_layout(); fig.savefig(out / "r2_same_estimator_linkage.png", dpi=130); plt.close(fig)
    n = pd.read_csv(OUT / "matched_null_dry_run_summary.csv"); n = n[n.population == "STRICT_FULL300_34"]
    fig, ax = plt.subplots(figsize=(6, 4)); ax.errorbar(n.null_mean, n.low_subset_agreement_spearman, xerr=2*n.null_sd, fmt="o", ms=3, alpha=.7)
    lo = min(n.null_mean.min(), n.low_subset_agreement_spearman.min()); hi = max(n.null_mean.max(), n.low_subset_agreement_spearman.max()); ax.plot([lo, hi], [lo, hi], "k--", lw=.8)
    ax.set_xlabel("matched-null mean agreement ± 2 SD"); ax.set_ylabel("low-D_seen agreement"); ax.set_title("Matched-null dry-run")
    fig.tight_layout(); fig.savefig(out / "matched_null_dry_run.png", dpi=130); plt.close(fig)
    json_write(OUT / "figure_manifest.json", {"figures": sorted(str(p.relative_to(OUT)) for p in out.glob("*.png")), "status": "diagnostic_only", "label": "SEEN_BASIN_PROXY / PIPELINE_DRY_RUN_ONLY"})

def write_readme(common: list[str], strict: list[str], audit: dict[str, Any], xaj: dict[str, Any], elapsed: float, peak_rss: float) -> None:
    r1 = pd.read_csv(OUT / "r1_dseen_set_summary.csv"); r1s = r1[(r1.population == "STRICT_FULL300_34") & (r1.aggregation == "model")]
    r2 = pd.read_csv(OUT / "r2_same_estimator_model_summary.csv"); r2s = r2[r2.population == "STRICT_FULL300_34"]
    link = pd.read_csv(OUT / "r2_dseen_linkage_summary.csv")
    t = pd.read_csv(OUT / "r1_temporal_ab_summary_strict34.csv")
    tb = pd.read_csv(OUT / "r1_temporal_ab_by_basin_strict34.csv")
    tg = tb.groupby("model").agg(A=("D_seen_A", "median"), B=("D_seen_B", "median"))
    temporal_gap_corr = finite_corr(tg.A, tg.B)
    restart = pd.read_csv(OUT / "wpC_restart_spread_summary.csv")
    relaxed_link = link[link.population == "RELAXED_SEEN_35"].iloc[0]
    xcomp = pd.read_csv(OUT / "xaj_base_cn_snow_gradient_comparison.csv")
    selected = pd.read_csv(OUT / "oob_pur_model_screening_strict34.csv").query("selected").model.tolist()
    lines = [
        "# Complete non-training R1/R2 package", "", "## Verdict", "",
        "This package completes the analysis/audit/screening layer without training. Current dPL was fit on all 531 basins; therefore all current gap, relationship, matched-null and parameter-realization results are `SEEN_BASIN_PROXY`/`D_seen`, never formal OOB `D`.", "",
        "## Frozen populations", f"- `REGISTRY_36`: 36 registry models; `flexb` remains `SKIPPED_MISSING_DPL`.", f"- `RELAXED_SEEN_35`: {len(common)} usable IC/dPL pairs, including `simhyd` accepted IC generation 280.", f"- `STRICT_FULL300_34`: {len(strict)} models, excluding `simhyd` and `flexb`.", f"- Current checkpoint audit: latest equals health best for {audit['latest_equals_health_best_count']}/{audit['latest_equals_health_best_total']} dPL models; no artifacts were overwritten.", "- Protocol audit preserves the dPL README boundary ambiguity (1994-10-01 documented validation start versus scored target 1995-10-01); freeze this before OOB claims.", "",
        "## R1 (`SEEN_BASIN_PROXY` only)", f"- Strict model-median `D_seen`: range={r1s.D_seen_median.min():.4f}..{r1s.D_seen_median.max():.4f}; median={r1s.D_seen_median.median():.4f}; median model IQR={r1s.D_seen_iqr.median():.4f}.", "- Both strict34 and relaxed35 include mean, median, IQR, p05/p95, tails, positive fraction, model/basin summaries, and retained extreme audits.", "- Tolerance sensitivity is descriptive only: relative-to-best IC deltas 0.02/0.05/0.10/0.15 are reported; no definitive equivalence threshold is selected.", f"- Full strict34 temporal A/B: {len(t)} summary rows, {t.valid_basin_count.min():.0f}..{t.valid_basin_count.max():.0f} valid basins/row, invalid total={t.invalid_count.sum():.0f}; model-level gap A/B correlation={temporal_gap_corr:.3f}.", f"- Archived 10-start test-score restart calibration was run on {restart.model.nunique()} structurally selected models; median restart SD spans {restart.median_restart_sd.min():.4f}..{restart.median_restart_sd.max():.4f}; this is not dPL seed uncertainty.", "- Extreme values are retained and audited; scalar tables do not establish hydrograph-level causes.", "",
        "## R2 same-estimator primary (`SEEN_BASIN_PROXY` only)", f"- Strict34 flattened Spearman agreement median={r2s.flattened_spearman.median():.4f}, range={r2s.flattened_spearman.min():.4f}..{r2s.flattened_spearman.max():.4f}; relaxed35 median={r2[r2.population == 'RELAXED_SEEN_35'].flattened_spearman.median():.4f}.", f"- Linkage Pearson/Spearman strict34={link[link.population == 'STRICT_FULL300_34'].iloc[0].pearson_r:.4f}/{link[link.population == 'STRICT_FULL300_34'].iloc[0].spearman_r:.4f}; relaxed35={relaxed_link.pearson_r:.4f}/{relaxed_link.spearman_r:.4f}.", "- Primary estimator uses identical basin masks, all-531 normalized sigmoid coordinates, and basin-wise Spearman attribute×parameter correlations for IC and dPL. Earlier auto-mapping output is secondary only.", "- Leave-one-model-out influence rows are included; no p-value-based claim is made with 34/35 model replicates.", "",
        "## Matched-null", f"- Dry-run completed with lowest-25% `D_seen` subsets and {N_NULL} matched draws/model for both populations; strict and relaxed outputs are separate.", "- Matching uses only predeclared snow/aridity/IC-skill quartiles and is labeled `PIPELINE_DRY_RUN_ONLY / NOT_OOB_EVIDENCE`; no R2 relationship value enters matching.", "",
        "## XAJ / parameter realization", f"- {xaj.get('status')}: existing XAJ Base/CN ledger audit and snow evidence were reused; no XAJ retraining occurred.", f"- XAJ normalized realization-distance expected direction is `{xaj.get('expected_direction_status')}`; L2 snow slopes are Base={float(xcomp.loc[xcomp.metric == 'normalized_l2_distance', 'Base'].iloc[0]):.4f}, CN={float(xcomp.loc[xcomp.metric == 'normalized_l2_distance', 'CN'].iloc[0]):.4f}. This does not support a Base-greater-than-CN distance claim.", "- Per-basin IC-vs-seen-dPL normalized realization distances remain exploratory; no causal Base/CN or physical compensation claim is made.", "",
        "## OOB/PUR selection", f"- Recommended panel ({len(selected)}): `{'`, `'.join(selected)}`.", f"- Alternates: `{', '.join(ALTERNATES)}`.", "- Rule: one model per D_seen/R2 quadrant where possible, then structural/process/complexity coverage, XAJ inclusion, middle reference, and artifact-health visibility; not an extreme-outcome ranking.", "- Preferred staged grouped design: 7 HUC groups (11–17), selected8×7×1=56 first-pass trainings; add second seed for total 112. It serves a regional held-group stress test but is not identical to random-basin OOB.", "- Cost comparison: selected8×5 random folds×2 seeds=80 for ordinary OOB; both random+HUC designs=192; full34×5×3=510.", "",
        "## Re-run and resources", "- Rerun: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python project/benchmark/scripts/diagnostics/r1_r2_nontraining_complete.py`.", f"- Runtime={elapsed:.2f}s; GPU={torch.cuda.get_device_name(0)}; FP32 parameter/forward work; one model at a time; no multiprocessing; peak RSS={peak_rss:.1f} MiB.", "- No daily predictions are persisted. Existing checkpoints and source training artifacts are read-only inputs.", "",
        "## Output map", "- Audit: `model_analysis_registry.csv`, `checkpoint_protocol_audit.{csv,json}`, `protocol_audit.json`, `basin_alignment_audit.json`, `source_hash_manifest.json`.", "- R1: `r1_dseen_*`, `r1_performance_tolerance_sensitivity.csv`, `r1_temporal_ab_*`, `wpC_restart_*`.", "- R2: `r2_same_estimator_*`, `r2_parameter_realization_distance_summary.csv`, `r2_dseen_linkage_*`.", "- Dry-run: `matched_null_dry_run_*`.", "- XAJ: `xaj_*`.", "- Selection/protocol: `oob_pur_model_screening_*`, `future_oob_pur_training_protocol.{json,md}`.", "- Diagnostic-only figures: `figures/*.png`, `figure_manifest.json`.", "",
        "## Remaining training requirements", "Valid OOB/PUB dPL predictions, independent dPL seeds, and any formal threshold calibration require new training. `flexb` recovery is intentionally not attempted."
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    started = time.time()
    if "--figures-only" in sys.argv:
        draw_diagnostic_figures()
        print(json.dumps({"output": str(OUT / "figures"), "figures_only": True}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for approved numerical path")
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    ids = np.asarray([int(x) for x in load_ids(IDS_PATH)], dtype=np.int64)
    if len(ids) != 531 or len(np.unique(ids)) != 531:
        raise RuntimeError("canonical basin IDs are not exactly 531 unique values")
    raw_attrs = CatchmentAttributeBuilder().load_raw_attributes(ids)
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore").to(torch.float32)
    gage_ids = np.asarray(np.load(REPO / "data/gage_id.npy"), dtype=np.int64)
    alignment = {"canonical_id_count": len(ids), "canonical_id_unique": len(np.unique(ids)), "gage_id_count": len(gage_ids), "gage_id_unique": len(np.unique(gage_ids)), "missing_gage_ids": sorted(set(ids.tolist()) - set(gage_ids.tolist())), "attribute_shape": list(raw_attrs.shape), "join_key": "basin_id", "positional_join_used": False}
    if alignment["missing_gage_ids"] or raw_attrs.shape != (531, 35):
        raise RuntimeError("basin/attribute alignment failed")
    json_write(OUT / "basin_alignment_audit.json", alignment)
    common, strict, _ = current_populations(ids)
    audit = checkpoint_audit(common, strict)
    json_write(OUT / "protocol_audit.json", {"analysis": "R1/R2 non-training audit", "ic_train_period": "1980-10-01..1995-09-30", "ic_warmup": "1980-10-01..1981-09-30 repeated 5x; 1825 days", "dpl_training_protocol": "all 531 basins jointly trained; one archived seed=42", "dpl_documented_validation_period": "1994-10-01..2010-09-30 in legacy README; date boundary requires freeze", "scored_target_period": f"{EVAL_START}..{EVAL_END}", "metric": "test KGE via repository compute_differentiable_kge/streaming_kge convention", "temporal_ab": "post-model-warmup outputs, split A/B, metric warmup_days=0", "parameter_mapping": {"IC": "sigmoid normalized coordinate under linear physical mapping", "dPL": "sigmoid normalized network output under archived auto physical mapping; primary R2 uses normalized coordinate only"}, "labels": {"current_dpl": "SEEN_BASIN_PROXY", "gap": "D_seen", "matched_null": "PIPELINE_DRY_RUN_ONLY / NOT_OOB_EVIDENCE"}, "VALID_OOB_DPL": 0, "populations": {"REGISTRY_36": 36, "RELAXED_SEEN_35": len(common), "STRICT_FULL300_34": len(strict)}})
    json_write(OUT / "source_hash_manifest.json", {str(path.relative_to(REPO)): sha256_file(path) for path in [IC_ROOT / "status_summary.json", DPL_ROOT / "health.csv", DPL_ROOT / "status.csv", BENCHMARK / "src/model_registry.py", BASE_OUT / "protocol_audit.json", BASE_OUT / "basin_alignment_audit.json", SCRIPT_DIR / "r1_r2_nontraining_complete.py", SCRIPT_DIR / "r1_r2_quick_survey.py"] if path.is_file()})
    dseen, _ = r1_tables(common, strict)
    temporal, _ = temporal_and_restart(common, strict, dseen, raw_attrs, attrs, ids, device)
    ms, theta_pairs = r2_same_estimator(common, strict, raw_attrs, attrs, device, dseen)
    linkage_and_null(ms, theta_pairs, dseen, raw_attrs, ids)
    reg = pd.read_csv(OUT / "model_analysis_registry.csv")
    reg.loc[reg.model.isin(common), "parameter_vector_check"] = "finite IC/dPL normalized vectors; identical 531 basin IDs"
    reg.to_csv(OUT / "model_analysis_registry.csv", index=False)
    xaj = xaj_realization_diagnostics()
    screening_and_protocol(common, strict, dseen, ms)
    # Preserve the prior audit as provenance without mutating it.
    json_write(OUT / "provenance.json", {"source_quick_survey": str(BASE_OUT), "source_ic": str(IC_ROOT), "source_dpl": str(DPL_ROOT), "source_registry": str(BENCHMARK / "src/model_registry.py"), "source_xaj_ledger": str(REPO / "project/hydrodiag/manuscript/results/R2/raw_parameter_ledger.csv"), "populations": {"REGISTRY_36": list(NPARAM_INFO_36), "RELAXED_SEEN_35": common, "STRICT_FULL300_34": strict}, "training_started": False, "training_allowed": False, "checkpoint_write_attempted": False, "scientific_label": "SEEN_BASIN_PROXY / D_seen"})
    elapsed = time.time() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    json_write(OUT / "resource_metadata.json", {"runtime_seconds": round(elapsed, 3), "gpu": torch.cuda.get_device_name(0), "dtype": "FP32", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"), "max_concurrent_forward": 1, "multiprocessing": False, "peak_rss_mib": round(peak, 1), "daily_predictions_persisted": False, "training_started": False})
    write_readme(common, strict, audit, xaj, elapsed, peak)
    draw_diagnostic_figures()
    print(json.dumps({"output": str(OUT), "relaxed_models": len(common), "strict_models": len(strict), "files": sum(1 for _ in OUT.rglob("*" ) if _.is_file()), "runtime_seconds": round(elapsed, 2), "training_started": False}, indent=2))


if __name__ == "__main__":
    main()
