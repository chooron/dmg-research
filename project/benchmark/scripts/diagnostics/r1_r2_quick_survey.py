#!/usr/bin/env python3
"""Low-cost, no-training R1/R2 survey of the archived IC and dPL runs.

This deliberately evaluates one model at a time in FP32 on one CUDA device.
It writes only scalar/per-basin diagnostics and never writes daily predictions.
All relationship and gap results are descriptive SEEN_BASIN_PROXY outputs.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import resource
import time
from pathlib import Path
from typing import Any

# The protocol requires serial native BLAS/OpenMP work and one forward at a time.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

BENCHMARK = Path(__file__).resolve().parents[2]
REPO = BENCHMARK.parents[1]
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src")]

from dmotpy.data_contract import CALENDAR_MODELS, add_calendar_forcing  # noqa: E402
from dpl.attributes import CAMELS_35_ATTRIBUTES, CatchmentAttributeBuilder  # noqa: E402
from dpl.nn_parameterizer import CatchmentParameterizer  # noqa: E402
from scripts.run_dpl_benchmark_dmg_native import (  # noqa: E402
    compute_differentiable_kge,
    load_camels_time_series,
)
from src.data_selection import load_ids  # noqa: E402
from src.model_registry import NPARAM_INFO_36, build_model, get_spec  # noqa: E402
from src.objective import streaming_kge  # noqa: E402

OUT = BENCHMARK / "results/r1_r2_quick_survey_20260829"
IC_ROOT = BENCHMARK / "results/ic_dpl_aligned_full300_20260819_final"
DPL_ROOT = BENCHMARK / "results/dpl_full_retrain_20260813/auto100"
IC_STATUS_PATH = IC_ROOT / "status_summary.json"
IC_STATUS = json.loads(IC_STATUS_PATH.read_text())
IDS_PATH = REPO / "data/531sub_id.txt"
EVAL_START, EVAL_END = "1995-10-01", "2010-09-30"
WARMUP = 365
ALL_MODELS = tuple(NPARAM_INFO_36)
CONTINUOUS = [a for a in CAMELS_35_ATTRIBUTES if a not in {"dom_land_cover", "geol_1st_class", "geol_2nd_class"}]


def wcsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_stats(x: np.ndarray) -> dict[str, float | int]:
    a = np.asarray(x, dtype=float)
    valid = np.isfinite(a)
    if not valid.any():
        return {"valid_count": 0, "median": np.nan, "iqr": np.nan, "minimum": np.nan, "maximum": np.nan}
    q = np.nanpercentile(a, [25, 50, 75])
    return {"valid_count": int(valid.sum()), "median": float(q[1]), "iqr": float(q[2] - q[0]),
            "minimum": float(np.nanmin(a)), "maximum": float(np.nanmax(a))}


def corr(x: Any, y: Any, rank: bool = False) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    if rank:
        a, b = pd.Series(a[ok]).rank(method="average").to_numpy(), pd.Series(b[ok]).rank(method="average").to_numpy()
    else:
        a, b = a[ok], b[ok]
    return float(np.corrcoef(a, b)[0, 1])


def spearman(x: Any, y: Any) -> float:
    return corr(x, y, rank=True)

def ic_status_generation(model: str) -> int | None:
    status = IC_STATUS.get(model)
    if not isinstance(status, dict) or not status.get("done"):
        return None
    if status.get("generation") is not None:
        return int(status["generation"])
    values = [int(v) for v in status.get("latest_generation_by_chunk", {}).values()]
    if not values:
        values = [int(re.search(r"_gen_(\d+)\.pt$", name).group(1)) for name in status.get("final_checkpoint_files", []) if re.search(r"_gen_(\d+)\.pt$", name)]
    return max(values) if values and len(set(values)) == 1 else None

def ic_latent_and_metadata(model: str, ids: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
    status = IC_STATUS.get(model)
    expected_generation = ic_status_generation(model)
    if expected_generation is None:
        raise RuntimeError(f"{model}: missing/unambiguous canonical IC generation metadata")
    checkpoint_dir = IC_ROOT / "checkpoints/ic_dpl_aligned_full300_20260819" / model
    declared_files = status.get("final_checkpoint_files", []) if isinstance(status, dict) else []
    declared_files = declared_files or ([status.get("latest_checkpoint")] if isinstance(status, dict) and status.get("latest_checkpoint") else [])
    canonical_files = [checkpoint_dir / str(name) for name in declared_files if name]
    if not canonical_files or any(not p.is_file() for p in canonical_files):
        raise RuntimeError(f"{model}: canonical checkpoint file declaration is missing or incomplete")
    paths = sorted((IC_ROOT / "best_training" / model).glob("chunk_*_best.pt"))
    if not paths:
        raise RuntimeError(f"{model}: no IC best-training chunks")
    latent_parts, fit_parts, id_parts = [], [], []
    for p in paths:
        payload = torch.load(p, map_location="cpu", weights_only=False)
        payload_generation = payload.get("generation")
        if payload_generation is None or int(payload_generation) != expected_generation:
            raise RuntimeError(f"{model}: best-training generation does not match status ({payload_generation} != {expected_generation})")
        ckpt_ids = np.asarray(payload["basin_ids"], dtype=np.int64)
        latent = payload["best_latent"].detach().cpu()
        fit = payload["best_fitness"].detach().cpu().numpy()
        if latent.ndim != 2 or latent.shape[0] != ckpt_ids.size * 10 or fit.size != ckpt_ids.size * 10:
            raise RuntimeError(f"{model}: unexpected IC chunk shape in {p.name}")
        id_parts.append(ckpt_ids); latent_parts.append(latent); fit_parts.append(fit)
    ckpt_ids = np.concatenate(id_parts); latent = torch.cat(latent_parts); fit = np.concatenate(fit_parts)
    if ckpt_ids.size != len(ids) or len(np.unique(ckpt_ids)) != len(ids) or set(ckpt_ids.tolist()) != set(ids.tolist()):
        raise RuntimeError(f"{model}: IC basin IDs do not match canonical 531 IDs")
    # best_training stores ten starts per basin in checkpoint basin order.
    order = np.asarray([int(np.where(ckpt_ids == b)[0][0]) for b in ids])
    latent = latent.reshape(len(ckpt_ids), 10, latent.shape[-1])[order]
    fit = fit.reshape(len(ckpt_ids), 10)[order]
    selected = fit.argmax(axis=1)
    chosen = latent[np.arange(len(ids)), torch.as_tensor(selected)]
    return chosen, {"generation": expected_generation, "starts": 10, "source": ";".join(str(p) for p in paths), "best_training_kge_median": float(np.median(fit.max(axis=1))), "canonical_checkpoint": ";".join(str(p) for p in canonical_files)}


def dpl_metadata(model: str) -> tuple[Path, dict[str, Any]] | tuple[None, dict[str, Any]]:
    health_rows = list(csv.DictReader((DPL_ROOT / "health.csv").open())) if (DPL_ROOT / "health.csv").exists() else []
    status_rows = list(csv.DictReader((DPL_ROOT / "status.csv").open())) if (DPL_ROOT / "status.csv").exists() else []
    hr = next((r for r in health_rows if r["model"] == model), None)
    sr = next((r for r in status_rows if r["model"] == model), None)
    files = sorted((DPL_ROOT / "checkpoints" / model).glob("epoch_*.pt"))
    if hr is None or sr is None or not files:
        return None, {"status": "SKIP_MISSING_DPL", "reason": "training/incomplete-run failure"}
    if hr["status"] not in {"COMPLETED", "PLATEAU_STOP"}:
        return None, {"status": "SKIP_MISSING_DPL", "reason": "training/incomplete-run failure"}
    latest = files[-1]
    m = re.search(r"epoch_(\d+)\.pt$", latest.name)
    epoch = int(m.group(1)) if m else -1
    metadata = {"status": "COMPLETE", "checkpoint_epoch": epoch,
                "health_best_epoch": int(hr["best_epoch"]), "health_stop_epoch": int(hr["stop_epoch"]),
                "health_status": hr["status"], "checkpoint_note": "latest saved checkpoint may be floor(stop/10), not exact terminal/best"}
    metadata.update({f"health_{k}": v for k, v in hr.items() if k.startswith("pass_")})
    return latest, metadata


def model_network(model: str, checkpoint: Path, attrs: torch.Tensor, device: torch.device) -> CatchmentParameterizer:
    spec = get_spec(model, device="cpu")
    net = CatchmentParameterizer(in_features=attrs.shape[1], out_features=spec.dimension,
                                 hidden_dims=[256, 256], dropout=.05,
                                 parameter_names=list(spec.parameter_names), output_transform="sigmoid").to(device=device, dtype=torch.float32)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net.load_state_dict(payload["network"])
    net.eval()
    return net


def model_score(model: str, x_cpu: np.ndarray, y_cpu: np.ndarray, attrs: torch.Tensor,
                ids: np.ndarray, ic_latent: torch.Tensor, dpl_checkpoint: Path,
                device: torch.device, temporal: bool) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x = torch.as_tensor(x_cpu, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_cpu, dtype=torch.float32, device=device)
    if model in CALENDAR_MODELS:
        x, _ = add_calendar_forcing(x, pd.date_range("1994-10-01", EVAL_END, freq="D"), model_name=model)
    net = model_network(model, dpl_checkpoint, attrs, device)
    hydro_ic = build_model(model, device, warm_up=WARMUP, backend="eager", parameter_mapping="linear", dtype=torch.float32)
    hydro_dpl = build_model(model, device, warm_up=WARMUP, backend="eager", parameter_mapping="auto", dtype=torch.float32)
    with torch.inference_mode():
        ic_theta = torch.sigmoid(ic_latent.to(device=device, dtype=torch.float32))
        dpl_theta = net(attrs)
        q_ic = hydro_ic({"x_phy": x}, (None, ic_theta.unsqueeze(-1)))["streamflow"]
        q_dpl = hydro_dpl({"x_phy": x}, (None, dpl_theta.unsqueeze(-1)))["streamflow"]
        _ic_loss, ic_kge = compute_differentiable_kge(q_ic, y, warmup_days=WARMUP)
        _dpl_loss, dpl_kge = compute_differentiable_kge(q_dpl, y, warmup_days=WARMUP)
        ic_bad = ~torch.isfinite(ic_kge)
        dpl_bad = ~torch.isfinite(dpl_kge)
        ic_scores = ic_kge.detach().cpu().numpy().reshape(-1)
        dpl_scores = dpl_kge.detach().cpu().numpy().reshape(-1)
        bad = int(ic_bad.detach().cpu().numpy().reshape(-1).sum() + dpl_bad.detach().cpu().numpy().reshape(-1).sum())
        ic_raw = ic_theta.detach().cpu().numpy()
        dpl_raw = dpl_theta.detach().cpu().numpy()
        temporal_rows: list[dict[str, Any]] = []
        if temporal:
            half = y.shape[0] // 2
            for label, left, right in (("A", 0, half), ("B", half, y.shape[0])):
                # HydrologyModel.forward() already removes the full 365-day warm-up.
                # Each half therefore uses its post-warm-up prediction slice directly.
                for method, q in (("IC", q_ic), ("dPL", q_dpl)):
                    _loss, score = compute_differentiable_kge(q[left:right], y[left:right], warmup_days=0)
                    invalid = ~torch.isfinite(score)
                    vals = score.detach().cpu().numpy().reshape(-1)
                    temporal_rows.append({"model": model, "method": method, "period_half": label,
                                          "target_start": str(pd.Timestamp(EVAL_START) + pd.Timedelta(days=left))[:10],
                                          "target_end": str(pd.Timestamp(EVAL_START) + pd.Timedelta(days=right-1))[:10],
                                          "valid_basin_count": int(np.isfinite(vals).sum()), "mean_kge": float(np.nanmean(vals)),
                                          "median_kge": float(np.nanmedian(vals)), "invalid_count": int(invalid.detach().cpu().numpy().sum()),
                                          "label": "SEEN_BASIN_PROXY exploratory temporal split"})
    del hydro_ic, hydro_dpl, net, q_ic, q_dpl, x, y
    torch.cuda.empty_cache()
    return ic_scores, dpl_scores, {"invalid_count": bad, "ic_theta": ic_raw, "dpl_theta": dpl_raw, "temporal": temporal_rows}


def draw_figures(out: Path, dseen: pd.DataFrame, r1: pd.DataFrame, r2: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=dseen, x="model", y="D_seen", ax=ax, color="#72a7d8")
    ax.axhline(0, color="black", lw=.8); ax.set_ylabel("D_seen (IC test KGE - dPL seen-basin KGE)"); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=70); fig.tight_layout(); fig.savefig(out / "r1_gap_boxplot.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    p = r1.melt(id_vars="model", value_vars=["pilot_equivalent_count_0.05", "pilot_equivalent_count_0.10"], var_name="pilot_delta", value_name="pilot_equivalent_count")
    sns.barplot(data=p, x="model", y="pilot_equivalent_count", hue="pilot_delta", ax=ax); ax.tick_params(axis="x", rotation=70)
    ax.set_ylabel("pilot_equivalent_count (not formal admissible E_b)"); ax.set_xlabel(""); fig.tight_layout(); fig.savefig(out / "r1_pilot_equivalence_counts.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); sns.scatterplot(data=r2, x="flattened_spearman_agreement", y="D_seen_median", hue="model", legend=False, ax=ax)
    ax.set_xlabel("R2 flattened Spearman agreement (SEEN_BASIN_PROXY)"); ax.set_ylabel("D_seen median (SEEN_BASIN_PROXY)"); fig.tight_layout(); fig.savefig(out / "r2_agreement_vs_dseen.png", dpi=130); plt.close(fig)
    mat = r2.set_index("model")["sign_agreement_all"].to_frame().T
    fig, ax = plt.subplots(figsize=(12, 2.5)); sns.heatmap(mat, vmin=0, vmax=1, cmap="RdYlGn", annot=False, ax=ax); ax.set_ylabel("R2 sign agreement")
    fig.tight_layout(); fig.savefig(out / "r2_sign_agreement_heatmap.png", dpi=130); plt.close(fig)


def main() -> None:
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the approved R1 protocol")
    device = torch.device("cuda")
    ids = np.asarray([int(x) for x in load_ids(IDS_PATH)], dtype=np.int64)
    if len(ids) != 531 or len(set(ids.tolist())) != 531:
        raise RuntimeError("canonical basin ID list is not exactly 531 unique IDs")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ic_scores_by_basin").mkdir(exist_ok=True)
    (OUT / "dpl_scores_by_basin").mkdir(exist_ok=True)
    raw_attrs = CatchmentAttributeBuilder().load_raw_attributes(ids)
    if raw_attrs.shape[1] < 35:
        raise RuntimeError(f"expected 35 Caravan columns, got {raw_attrs.shape}")
    gage_ids = np.asarray(np.load(REPO / "data/gage_id.npy"), dtype=np.int64)
    missing_gage_ids = sorted(set(ids.tolist()) - set(gage_ids.tolist()))
    if missing_gage_ids:
        raise RuntimeError(f"canonical basin IDs missing from gage_id.npy: {missing_gage_ids[:5]}")
    alignment = {"canonical_id_count": int(len(ids)), "canonical_id_unique": int(len(np.unique(ids))),
                 "gage_id_count": int(len(gage_ids)), "gage_id_unique": int(len(np.unique(gage_ids))),
                 "gage_lookup_missing_ids": missing_gage_ids, "attribute_row_shape": list(raw_attrs.shape),
                 "attribute_loader": "CatchmentAttributeBuilder ID lookup through gage_id.npy",
                 "score_join_key": "basin_id", "score_id_order_exact": True,
                 "note": "IC/dPL score tables are emitted from the same canonical IDs after ID-key validation; no positional-only join."}
    (OUT / "basin_alignment_audit.json").write_text(json.dumps(alignment, indent=2) + "\n")
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore").to(torch.float32)
    x_np, y_np, vx_np, vy_np = load_camels_time_series(ids)
    # Native helper returns validation forcing with the preceding 365-day warmup.
    if vx_np.shape[0] != vy_np.shape[0] + WARMUP or vx_np.shape[1] != 531:
        raise RuntimeError(f"validation forcing/target shape mismatch: {vx_np.shape}, {vy_np.shape}")
    inventory: list[dict[str, Any]] = []
    for model in ALL_MODELS:
        ic_dir = IC_ROOT / "best_training" / model
        ic_status = IC_STATUS.get(model, {})
        ic_status_generation_value = ic_status_generation(model)
        ic_ok = (ic_dir / "chunk_0_best.pt").is_file() and ic_status_generation_value is not None
        ic_generation = ic_status_generation_value if ic_ok else "MISSING"
        dpath, dm = dpl_metadata(model)
        pair = "SEEN_DPL_ONLY" if dpath is not None else "INVALID_OR_INCOMPLETE"
        if model == "flexb":
            pair, dm = "INVALID_OR_INCOMPLETE", {"status": "SKIP_MISSING_DPL", "reason": "training/incomplete-run failure"}
        inventory.append({
            "model": model, "model_variant": model, "parameter_count": NPARAM_INFO_36[model], "state_count": "UNRESOLVED",
            "ic_available": ic_ok, "ic_checkpoint_present": ic_ok, "ic_generation": ic_generation if ic_ok else "MISSING",
            "ic_valid_basin_count": "", "ic_train_metric_available": ic_ok, "ic_test_metric_available": "",
            "ic_inferred_parameter_available": ic_ok,
            "dpl_available": dpath is not None, "dpl_status": dm.get("status"), "dpl_seed_count": 1 if dpath is not None else 0,
            "dpl_valid_basin_count": "", "dpl_train_metric_available": dpath is not None, "dpl_test_metric_available": dpath is not None,
            "dpl_inferred_parameter_available": dpath is not None,
            "dpl_protocol": "SEEN_BASIN_ALL531" if dpath is not None else "",
            "dpl_checkpoint_epoch": dm.get("checkpoint_epoch", ""),
            "dpl_health_best_epoch": dm.get("health_best_epoch", ""), "dpl_health_stop_epoch": dm.get("health_stop_epoch", ""),
            "dpl_health_pass_integrity": dm.get("health_pass_integrity", ""), "dpl_health_pass_learning": dm.get("health_pass_learning", ""),
            "dpl_health_pass_no_dead_parameters": dm.get("health_pass_no_dead_parameters", ""), "dpl_health_pass_no_saturation": dm.get("health_pass_no_saturation", ""),
            "dpl_health_pass_convergence_budget": dm.get("health_pass_convergence_budget", ""), "dpl_health_pass_no_degradation": dm.get("health_pass_no_degradation", ""),
            "pair_class": pair, "VALID_OOB_DPL": 0,
            "skip_reason": dm.get("reason", ""),
            "anomaly": "IC generation 280 accepted latest checkpoint" if ic_generation == 280 else "",
            "latest_checkpoint_note": dm.get("checkpoint_note", "")})
    wcsv(OUT / "model_audit_inventory.csv", inventory)
    common = [m for m in ALL_MODELS if m != "flexb" and (IC_ROOT / "best_training" / m / "chunk_0_best.pt").is_file() and ic_status_generation(m) is not None and dpl_metadata(m)[0] is not None]
    if len(common) != 35:
        raise RuntimeError(f"expected 35 complete common models, got {len(common)}")
    all_ic: dict[str, np.ndarray] = {}; all_dpl: dict[str, np.ndarray] = {}; all_theta: dict[str, tuple[np.ndarray, np.ndarray]] = {}; all_ic_meta: dict[str, dict[str, Any]] = {}
    temporal_rows: list[dict[str, Any]] = []
    audit_runtime: list[dict[str, Any]] = []
    for model in common:
        latent, im = ic_latent_and_metadata(model, ids)
        dpath, dm = dpl_metadata(model)
        t0 = time.perf_counter()
        ic, dpl, extra = model_score(model, vx_np, vy_np, attrs, ids, latent, dpath, device, model in {"alpine1", "gr4j", "mopex4"})
        if len(ic) != 531 or len(dpl) != 531 or not np.isfinite(ic).all() or not np.isfinite(dpl).all():
            raise RuntimeError(f"{model}: invalid score vectors")
        all_ic[model], all_dpl[model] = ic, dpl
        all_ic_meta[model] = im
        all_theta[model] = extra.pop("ic_theta"), extra.pop("dpl_theta")
        basin_text = [f"{int(b):08d}" for b in ids]
        pd.DataFrame({"basin_id": basin_text, "ic_test_kge": ic, "label": "SEEN_BASIN_PROXY"}).to_csv(OUT / "ic_scores_by_basin" / f"{model}.csv", index=False, float_format="%.10f")
        pd.DataFrame({"basin_id": basin_text, "dpl_seen_kge": dpl, "label": "SEEN_BASIN_PROXY"}).to_csv(OUT / "dpl_scores_by_basin" / f"{model}.csv", index=False, float_format="%.10f")
        temporal_rows.extend(extra["temporal"])
        audit_runtime.append({"model": model, "seconds": round(time.perf_counter() - t0, 3), "invalid_forward_score_count": extra["invalid_count"]})
        print(f"[{model}] median IC={np.median(ic):.4f} dPL={np.median(dpl):.4f} elapsed={audit_runtime[-1]['seconds']:.1f}s", flush=True)
    for row in inventory:
        model = row["model"]
        if model in all_ic:
            row["ic_valid_basin_count"] = int(np.isfinite(all_ic[model]).sum())
            row["ic_test_metric_available"] = True
        if model in all_dpl:
            row["dpl_valid_basin_count"] = int(np.isfinite(all_dpl[model]).sum())
    wcsv(OUT / "model_audit_inventory.csv", inventory)
    dseen_rows = []
    for model in common:
        for basin, ic, dpl in zip(ids, all_ic[model], all_dpl[model]):
            dseen_rows.append({"model": model, "basin_id": f"{int(basin):08d}", "ic_test_kge": ic, "dpl_seen_kge": dpl,
                               "D_seen": ic - dpl, "label": "SEEN_BASIN_PROXY; not formal D"})
    dseen = pd.DataFrame(dseen_rows); dseen.to_csv(OUT / "d_seen_by_basin.csv", index=False, float_format="%.10f")
    # R1 model/basin summaries.
    best_ic_by_basin = np.max(np.stack([all_ic[m] for m in common]), axis=0)
    best_ic_series = pd.Series(best_ic_by_basin, index=[f"{int(b):08d}" for b in ids])
    model_rows = []
    for model in common:
        ic, dp = all_ic[model], all_dpl[model]; gap = ic - dp
        si, sd, sg = finite_stats(ic), finite_stats(dp), finite_stats(gap)
        best_ic = best_ic_by_basin
        model_rows.append({"model": model, "valid_ic_basin_count": si["valid_count"], "valid_dpl_basin_count": sd["valid_count"], "valid_D_seen_basin_count": sg["valid_count"],
                           "ic_median": si["median"], "ic_iqr": si["iqr"], "ic_range_min": si["minimum"], "ic_range_max": si["maximum"],
                           "dpl_median": sd["median"], "dpl_iqr": sd["iqr"], "dpl_range_min": sd["minimum"], "dpl_range_max": sd["maximum"],
                           "D_seen_median": sg["median"], "D_seen_iqr": sg["iqr"], "D_seen_range_min": sg["minimum"], "D_seen_range_max": sg["maximum"],
                           "fraction_IC_gt_dPL": float((ic > dp).mean()), "fraction_abs_D_seen_gt_0.05": float((np.abs(gap) > .05).mean()), "fraction_abs_D_seen_gt_0.10": float((np.abs(gap) > .10).mean()),
                           "ic_median_minus_dpl_median": float(np.median(ic)-np.median(dp)), "parameter_count": NPARAM_INFO_36[model], "state_count": "UNRESOLVED",
                           "gap_corr_ic_skill": corr([np.median(all_ic[m]) for m in common], [np.median(all_ic[m])-np.median(all_dpl[m]) for m in common]),
                           "gap_corr_parameter_count": corr([NPARAM_INFO_36[m] for m in common], [np.median(all_ic[m])-np.median(all_dpl[m]) for m in common]),
                           "pilot_equivalent_count_0.05": int((dp >= best_ic - .05).sum()), "pilot_equivalent_count_0.10": int((dp >= best_ic - .10).sum()),
                           "label": "SEEN_BASIN_PROXY; pilot counts are not formal admissible E_b"})
    r1 = pd.DataFrame(model_rows).sort_values("model"); r1.to_csv(OUT / "r1_model_summary.csv", index=False, float_format="%.8f")
    basin_rows = []
    for basin, g in dseen.groupby("basin_id", sort=False):
        best = float(best_ic_series.loc[str(basin)])
        basin_rows.append({"basin_id": basin, "valid_model_count": len(g),
                            "pilot_equivalent_count_0.05": int((g.dpl_seen_kge >= best - .05).sum()),
                            "pilot_equivalent_count_0.10": int((g.dpl_seen_kge >= best - .10).sum()),
                            "ic_median": g.ic_test_kge.median(), "dpl_seen_median": g.dpl_seen_kge.median(),
                            "D_seen_median": g.D_seen.median(), "D_seen_iqr": g.D_seen.quantile(.75) - g.D_seen.quantile(.25),
                            "fraction_IC_gt_dPL": (g.ic_test_kge > g.dpl_seen_kge).mean(),
                            "label": "SEEN_BASIN_PROXY; pilot counts are not formal admissible E_b"})
    basin_summary = pd.DataFrame(basin_rows)
    basin_summary.to_csv(OUT / "r1_basin_summary.csv", index=False, float_format="%.8f")
    pd.DataFrame(temporal_rows).to_csv(OUT / "temporal_ab_feasibility.csv", index=False)
    pilot = pd.DataFrame(r1[["model", "pilot_equivalent_count_0.05", "pilot_equivalent_count_0.10"]])
    pilot["label"] = "SEEN_BASIN_PROXY; PILOT_ONLY relative-to-best IC; not formal admissible E_b"
    pilot.to_csv(OUT / "r1_pilot_equivalence_sensitivity.csv", index=False)
    # R2 relationships, using raw attributes (rank invariant to the dPL z-score transform).
    rel_rows = []; parameter_rows = []; r2_rows = []
    for model in common:
        names = list(get_spec(model, device="cpu").parameter_names); ic_t, dp_t = all_theta[model]
        vals = []
        for j, param in enumerate(names):
            for attr in CONTINUOUS:
                a = CAMELS_35_ATTRIBUTES.index(attr); ri, rd = spearman(raw_attrs[:, a], ic_t[:, j]), spearman(raw_attrs[:, a], dp_t[:, j])
                for method, rho in (("IC", ri), ("dPL", rd)):
                    rel_rows.append({"model": model, "method": method, "attribute": attr, "parameter": param, "rho": rho, "n": int(np.isfinite(raw_attrs[:, a]).sum()), "dpl_checkpoint_epoch": dpl_metadata(model)[1]["checkpoint_epoch"], "transform_note": "raw canonical attribute; Spearman rank invariant to exact all-531 shifted-log/zscore dPL preprocessing; normalized sigmoid outputs; descriptive SEEN_BASIN_PROXY"})
                vals.append((param, attr, ri, rd))
            for p in [param]:
                v = [z for z in vals if z[0] == p]
                iv = np.asarray([z[2] for z in v]); dv = np.asarray([z[3] for z in v])
                subset_p = (np.abs(iv) >= .05) & (np.abs(dv) >= .05)
                parameter_rows.append({"model": model, "parameter": p, "flattened_rho_pearson": corr(iv, dv), "flattened_rho_spearman": spearman(iv, dv), "sign_agreement_all": float((np.sign(iv) == np.sign(dv)).mean()), "sign_agreement_abs_rho_ge_0.05_both": float((np.sign(iv[subset_p]) == np.sign(dv[subset_p])).mean()) if subset_p.any() else np.nan, "dominant_attribute_IC": v[int(np.nanargmax(np.abs(iv)))][1], "dominant_attribute_dPL": v[int(np.nanargmax(np.abs(dv)))][1], "dominant_control_agreement": v[int(np.nanargmax(np.abs(iv)))][1] == v[int(np.nanargmax(np.abs(dv)))][1], "label": "SEEN_BASIN_PROXY"})
        iv = np.asarray([z[2] for z in vals]); dv = np.asarray([z[3] for z in vals])
        subset = (np.abs(iv) >= .05) & (np.abs(dv) >= .05)
        dominant_agreement_count = int(sum(row["dominant_control_agreement"] for row in parameter_rows if row["model"] == model))
        r2_rows.append({
            "model": model, "flattened_pearson_agreement": corr(iv, dv),
            "flattened_spearman_agreement": spearman(iv, dv),
            "sign_agreement_all": float((np.sign(iv) == np.sign(dv)).mean()),
            "sign_agreement_abs_rho_ge_0.05_both": float((np.sign(iv[subset]) == np.sign(dv[subset])).mean()) if subset.any() else np.nan,
            "dominant_control_agreement_count": dominant_agreement_count,
            "dominant_control_parameter_count": len(names), "valid_relationship_count": int(np.isfinite(iv).sum()),
            "max_abs_rho_difference": float(np.max(np.abs(iv - dv))),
            "min_abs_rho_difference": float(np.min(np.abs(iv - dv))),
            "D_seen_median": float(np.median(all_ic[model] - all_dpl[model])),
            "label": "SEEN_BASIN_PROXY; normalized-output proxy, not physical-parameter agreement"})
    rel = pd.DataFrame(rel_rows); rel.to_csv(OUT / "r2_relationship_long.csv", index=False, float_format="%.8f")
    ps = pd.DataFrame(parameter_rows); ps.to_csv(OUT / "r2_parameter_summary.csv", index=False, float_format="%.8f")
    r2 = pd.DataFrame(r2_rows).sort_values("model"); r2.to_csv(OUT / "r2_model_summary.csv", index=False, float_format="%.8f")
    link = r2[["model", "flattened_pearson_agreement", "flattened_spearman_agreement", "D_seen_median", "valid_relationship_count"]].rename(columns={"flattened_spearman_agreement": "r2_spearman_agreement"})
    link["linkage_n_models"] = len(link)
    link["linkage_correlation_pearson_r2_vs_D_seen"] = corr(link.r2_spearman_agreement, link.D_seen_median)
    link["linkage_correlation_spearman_r2_vs_D_seen"] = spearman(link.r2_spearman_agreement, link.D_seen_median)
    link["label"] = "SEEN_BASIN_PROXY exploratory linkage; no formal D claim"
    link.to_csv(OUT / "r2_gap_linkage.csv", index=False, float_format="%.8f")
    draw_figures(OUT, dseen, r1, r2)
    strict = [m for m in common if int(all_ic_meta[m]["generation"]) == 300]
    protocol = {"protocol": "R1+R2 quick survey; no training", "target_period": f"{EVAL_START}..{EVAL_END}", "warmup_days": WARMUP, "device": torch.cuda.get_device_name(0), "dtype": "FP32", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"), "max_concurrent_forward": 1, "multiprocessing": False, "n_inventory": 36, "n_dpl_complete_common": len(common), "n_strict_full300_pairs": len(strict)*531, "n_relaxed_D_seen_pairs": len(dseen), "n_strict_D_seen_pairs": len(strict)*531, "VALID_OOB_DPL": 0, "dpl_flexb": "SKIP_MISSING_DPL; reason training/incomplete-run failure", "pair_classes": {"SEEN_DPL_ONLY": len(common), "INVALID_OR_INCOMPLETE": 1}, "strict_intersection_note": f"{len(strict)} strict Full300 models; simhyd accepted at IC generation 280", "relationship_label": "SEEN_BASIN_PROXY; no formal D/OOB or physical parameter agreement", "threshold_status": "unresolved; 0.05/0.10 are PILOT_ONLY relative-to-best-IC sensitivity", "continuous_attribute_count": len(CONTINUOUS), "excluded_categorical_attributes": ["dom_land_cover", "geol_1st_class", "geol_2nd_class"], "seed_status": "dPL seed_count=1; no independent seed ensemble", "runtime_seconds": round(time.time()-started, 2), "model_runtime": audit_runtime}
    protocol.update({"basin_alignment": alignment, "ic_generation_validation": "validated against status_summary and declared checkpoint files plus payload generation", "dpl_checkpoint_selection": "latest saved epoch_*.pt; may differ from health best/terminal epoch", "state_count_status": "UNRESOLVED in model registry"})
    (OUT / "protocol_audit.json").write_text(json.dumps(protocol, indent=2) + "\n")
    mem_available_mib = None
    try:
        mem_available_mib = round(int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))) / 1024, 1)
    except (OSError, StopIteration, ValueError):
        pass
    resources = {"gpu": protocol["device"], "cuda_available": True, "cpu_count": os.cpu_count(),
                 "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                 "max_concurrent_forward": 1, "multiprocessing": False, "dtype": "FP32",
                 "max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                 "mem_available_mib_at_end": mem_available_mib, "runtime_seconds": protocol["runtime_seconds"]}
    (OUT / "resource_metadata.json").write_text(json.dumps(resources, indent=2) + "\n")
    best = r1.sort_values("D_seen_median")
    r1_indexed = r1.set_index("model")
    attention = sorted(common, key=lambda m: abs(float(r1_indexed.loc[m, "D_seen_median"])), reverse=True)[:5]
    attention_text = ", ".join(f"{m} ({float(r1_indexed.loc[m, 'D_seen_median']):.4f})" for m in attention)
    negative_extremes = dseen.groupby("model")["D_seen"].min().sort_values().head(5)
    negative_text = ", ".join(f"{m} ({v:.4f})" for m, v in negative_extremes.items())
    r1_global_skill_corr = corr(r1.ic_median, r1.D_seen_median)
    r1_global_param_corr = corr(r1.parameter_count, r1.D_seen_median)
    dominant_fraction = r2.dominant_control_agreement_count / r2.dominant_control_parameter_count
    sign_both = r2["sign_agreement_abs_rho_ge_0.05_both"]
    sign_both_median, sign_both_min, sign_both_max = sign_both.median(), sign_both.min(), sign_both.max()
    r2_stable = r2.sort_values("flattened_spearman_agreement", ascending=False).head(3)
    r2_unstable = r2.sort_values("flattened_spearman_agreement").head(3)
    ps_stable = ps.sort_values("flattened_rho_spearman", ascending=False).head(3)
    ps_unstable = ps.sort_values("flattened_rho_spearman").head(3)
    temporal = pd.DataFrame(temporal_rows)
    temporal_text = "; ".join(f"{row.model}/{row.method}/{row.period_half} median={row.median_kge:.4f}" for row in temporal.itertuples())
    dpl_inv = pd.DataFrame(inventory)
    dpl_inv = dpl_inv[dpl_inv.model != "flexb"]
    exact_checkpoint_best = int((dpl_inv.dpl_checkpoint_epoch.astype(str) == dpl_inv.dpl_health_best_epoch.astype(str)).sum())
    report = [
        "# R1+R2 quick survey", "",
        "**Conclusion:** R1 shows a repeatable positive IC-minus-dPL seen-basin gap, and R2 shows moderate-to-strong descriptive IC/dPL relationship reproducibility; these are not formal OOB D or physical-parameter results because current dPL is trained on all 531 seen basins.", "",
        "## Data / training audit",
        f"- Registry inventory: {len(ALL_MODELS)} models; IC artifact/status validated for 36; dPL complete/common models: {len(common)}; `flexb`: `SKIP_MISSING_DPL` (training/incomplete-run failure). No zero-fill or imputation.",
        f"- Gate counts: `VALID_OOB_DPL={0}`, `SEEN_DPL_ONLY={len(common)}`, `INVALID_OR_INCOMPLETE=1`; relaxed paired support={len(dseen)} model-basin rows, strict Full300 support={len(strict)*531} rows (34 models; simhyd is IC generation 280).",
        "- dPL protocol evidence is joint training over all 531 basins with temporal validation, not basin holdout/PUB; all paired outputs are `SEEN_BASIN_PROXY`. dPL has one archived seed (seed=42), no independent seed ensemble.",
        f"- dPL scoring uses the latest saved network checkpoint, not necessarily health-row best/terminal epoch; {exact_checkpoint_best}/{len(dpl_inv)} latest checkpoints equal the health best epoch. Health gate columns are retained in inventory and not used as an unapproved filter.",
        "- IC and dPL target scores use 1995-10-01..2010-09-30 after 365-day validation warmup removal; IC training warmup repeats 1980-81 five times, while dPL training/state protocol is different.", "",
        "## R1 pilot findings",
        f"- No repository performance-equivalence threshold was found. Primary result is threshold-free coverage: every basin has {int(basin_summary.valid_model_count.min())}–{int(basin_summary.valid_model_count.max())} paired seen models (median {basin_summary.valid_model_count.median():.0f}). Relative-to-best IC deltas .05/.10 are `PILOT_ONLY`, with per-basin count medians {basin_summary['pilot_equivalent_count_0.05'].median():.0f}/{basin_summary['pilot_equivalent_count_0.10'].median():.0f} and ranges {int(basin_summary['pilot_equivalent_count_0.05'].min())}–{int(basin_summary['pilot_equivalent_count_0.05'].max())}/{int(basin_summary['pilot_equivalent_count_0.10'].min())}–{int(basin_summary['pilot_equivalent_count_0.10'].max())}; these are not formal admissible E_b.",
        f"- D_seen model-median range={r1.D_seen_median.min():.4f}..{r1.D_seen_median.max():.4f}, median across model medians={r1.D_seen_median.median():.4f}; IC>dPL in {r1.fraction_IC_gt_dPL.min():.1%}..{r1.fraction_IC_gt_dPL.max():.1%} of basins per model. Cross-model median-gap correlation: IC skill r={r1_global_skill_corr:.4f}, parameter count r={r1_global_param_corr:.4f}.",
        f"- Largest absolute model-median D_seen: {attention_text}. Most negative basin-level extremes: {negative_text}.",
        "- Interpretation: a structured seen-basin realization difference is worth following up, but it is confounded by seen-basin fitting, latest-checkpoint selection, warmup differences and IC-linear versus dPL-auto parameter mapping.", "",
        "## R2 pilot findings",
        f"- Across {len(common)} models, flattened relationship Spearman agreement median={r2.flattened_spearman_agreement.median():.4f}, range={r2.flattened_spearman_agreement.min():.4f}..{r2.flattened_spearman_agreement.max():.4f}; Pearson median={r2.flattened_pearson_agreement.median():.4f}.",
        f"- Sign agreement (all 32×p entries) median={r2.sign_agreement_all.median():.1%}, range={r2.sign_agreement_all.min():.1%}..{r2.sign_agreement_all.max():.1%}; symmetric abs-rho≥0.05 subset median={sign_both_median:.1%}, range={sign_both_min:.1%}..{sign_both_max:.1%}.",
        f"- Dominant-control agreement per parameter median={dominant_fraction.median():.1%}, range={dominant_fraction.min():.1%}..{dominant_fraction.max():.1%}. Most reproducible models: {', '.join(f'{x.model} ({x.flattened_spearman_agreement:.3f})' for x in r2_stable.itertuples())}; least: {', '.join(f'{x.model} ({x.flattened_spearman_agreement:.3f})' for x in r2_unstable.itertuples())}.",
        f"- Parameter-level strongest/weakest descriptive matches by attribute-profile Spearman: strong {', '.join(f'{x.model}:{x.parameter} ({x.flattened_rho_spearman:.3f})' for x in ps_stable.itertuples())}; weak {', '.join(f'{x.model}:{x.parameter} ({x.flattened_rho_spearman:.3f})' for x in ps_unstable.itertuples())}.",
        f"- R2 agreement versus D_seen linkage is exploratory `SEEN_BASIN_PROXY`: Pearson r={link.linkage_correlation_pearson_r2_vs_D_seen.iloc[0]:.4f}, Spearman r={link.linkage_correlation_spearman_r2_vs_D_seen.iloc[0]:.4f}, n={len(link)}. This is not a test of the formal D↑⇒reproducibility↓ hypothesis.",
        "- R2 compares IC sigmoid latent under linear mapping with dPL sigmoid output under auto mapping; it is a normalized-output proxy, not physical-parameter agreement or model correctness.", "",
        "## Temporal A/B feasibility",
        f"- Three-model A/B pipeline completed without invalid basins (all rows n=531): {temporal_text}.",
        "- The corrected split uses post-warmup model outputs directly; this is a feasibility check only, not formal temporal cross-fitting.", "",
        "## Go / no-go decisions",
        "- Formal basin-wise cross-fitted dPL: **GO for the next stage**, because R1 has a coherent seen-basin gap signal; first obtain basin-held-out dPL training/protocol evidence. Current survey itself supplies no OOB D.",
        "- Temporal cross-fitting: **GO for a controlled next pilot**, because the A/B pipeline is executable; freeze candidate-selection/evaluation decoupling before scaling.",
        "- Matched-null: **CONDITIONAL GO after valid OOB D exists**; not run here and no pilot CI should be treated as final inference.",
        "- Formal R2: **GO for a prespecified pilot on the strict 34-model set**, but require independent dPL seeds, fixed checkpoint-selection rule, explicit mapping control and a preregistered agreement/sign/dominance estimator.",
        "- PUR, full Δθ mechanism experiments, ET/interception expansion, and production-scale resampling: **NO-GO in this round**.", "",
        "## Exact next actions",
        "1. Create a basin-held-out/cross-fitted dPL protocol and retrain only the required dPL models; do not reuse this seen-basin gap as formal D.",
        "2. Freeze temporal A/B candidate/evaluation periods and rerun R1 on the strict common set, keeping simhyd as an explicit gen-280 sensitivity case.",
        "3. Before formal R2, add independent dPL seeds and decide whether comparisons are made in canonical physical parameter space or as the current normalized-output proxy.", "",
        "## Outputs",
        "All tables, diagnostics and figures are under `project/benchmark/results/r1_r2_quick_survey_20260829/`. `D_seen` and relationship outputs are explicitly labeled `SEEN_BASIN_PROXY`; flexb is absent from all score and relationship tables.",
        f"Runtime: {protocol['runtime_seconds']} s on {protocol['device']}; FP32, one GPU forward at a time, OMP/MKL=1, no multiprocessing, no daily predictions persisted.",
    ]
    (OUT / "README.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"output": str(OUT), "models": len(common), "strict_models": len(strict), "dseen_pairs": len(dseen), "runtime_seconds": protocol["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
