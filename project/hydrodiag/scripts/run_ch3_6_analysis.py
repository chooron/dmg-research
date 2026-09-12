#!/usr/bin/env python3
"""Compute the legal Chapter 3.6 summaries from local replay artifacts.

This analysis intentionally does not invent the unavailable ET dry-spell,
recession, low-flow/FDC, P4, or corrected-P5 protocols. It computes the
outlet, basic ET partition, common-parameter, and existing Lyne--Hollick BFI
layers that are directly supported by the repository and writes explicit
PENDING status artifacts for the blocked layers.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
RESULTS = PROJECT / "results"
OUT = RESULTS / "ch3_6_cross_process"
REPLAY = OUT / "02_replay"
IC_ROOT = RESULTS / "ic_phase0_controlled_531_v1"
DPL_ROOT = RESULTS / "dpl_controlled_531_v1"
DPL_N_ROOT = OUT / "dpl_controlled_n_531_v1"
DATA_ROOT = (PROJECT.parents[1] / "data").resolve()

sys.path.insert(0, str(PROJECT))
from ablation.ic_core.parameter_adapter import get_parameter_spec  # noqa: E402
from manuscript.scripts.r4.common import load_bundle  # noqa: E402
from scripts.phase0_sampling import lyne_hollick_bfi  # noqa: E402

IC_LABELS = ("N", "D_E", "G_E", "D_R", "G_R")
DPL_LABELS = (
    "XAJ_CONTROLLED_N_CN",
    "XAJ_D_E_CN",
    "XAJ_G_E_CN",
    "XAJ_D_R_CN",
    "XAJ_G_R_CN",
)
DPL_TO_STRUCTURE = {
    "XAJ_CONTROLLED_N_CN": "N",
    "XAJ_D_E_CN": "D_E",
    "XAJ_G_E_CN": "G_E",
    "XAJ_D_R_CN": "D_R",
    "XAJ_G_R_CN": "G_R",
}
PERIODS = {"train": slice(365, 5478), "test": slice(5478, 10957)}
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260730


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def bootstrap_median(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_DRAWS, values.size))
    medians = np.median(values[indices], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def summary_row(regime: str, contrast: str, values: np.ndarray, valid_n: int | None = None) -> dict:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    lo, hi = bootstrap_median(finite)
    return {
        "regime": regime,
        "contrast": contrast,
        "valid_n": int(finite.size if valid_n is None else valid_n),
        "median": float(np.median(finite)) if finite.size else math.nan,
        "q25": float(np.percentile(finite, 25)) if finite.size else math.nan,
        "q75": float(np.percentile(finite, 75)) if finite.size else math.nan,
        "ci95_low": lo,
        "ci95_high": hi,
        "positive_fraction": float(np.mean(finite > 0)) if finite.size else math.nan,
        "negative_fraction": float(np.mean(finite < 0)) if finite.size else math.nan,
    }


def replay_path(regime: str, label: str) -> Path:
    return REPLAY / ("IC" if regime == "IC" else "dPL") / f"{label}_full_replay.npz"

def load_replay(regime: str, label: str):
    path = replay_path(regime, label)
    if not path.exists():
        raise FileNotFoundError(f"missing replay artifact: {path}")
    # Keep the compressed NPZ lazy; each analysis layer accesses only the
    # arrays it needs, which avoids holding all ten full replays in RAM.
    return np.load(path, allow_pickle=False)


def load_all_replays() -> dict[str, dict[str, dict[str, np.ndarray]]]:
    out: dict[str, dict[str, dict[str, np.ndarray]]] = {"IC": {}, "dPL": {}}
    for label in IC_LABELS:
        out["IC"][label] = load_replay("IC", label)
    for label in DPL_LABELS:
        out["dPL"][label] = load_replay("dPL", label)
    return out


def outlet_results(replays: dict[str, dict[str, dict[str, np.ndarray]]], basin_ids: np.ndarray) -> tuple[list[dict], list[dict]]:
    basin_rows: list[dict] = []
    summary: list[dict] = []
    for regime, variants, baseline in (
        ("IC", ("D_E", "G_E", "D_R", "G_R"), "N"),
        ("dPL", ("XAJ_D_E_CN", "XAJ_G_E_CN", "XAJ_D_R_CN", "XAJ_G_R_CN"), "XAJ_CONTROLLED_N_CN"),
    ):
        base = replays[regime][baseline]["split_replay_eval_kge"].astype(np.float64)
        for variant in variants:
            value = replays[regime][variant]["split_replay_eval_kge"].astype(np.float64)
            delta = value - base
            contrast = f"{variant}-{baseline}"
            for basin, b, v, d in zip(basin_ids, base, value, delta):
                basin_rows.append({
                    "regime": regime, "contrast": contrast, "basin_id": basin,
                    "kge_baseline": b, "kge_variant": v, "delta_kge": d,
                })
            summary.append(summary_row(regime, contrast, delta))
    fields = ["regime", "contrast", "basin_id", "kge_baseline", "kge_variant", "delta_kge"]
    write_csv(OUT / "03_et" / "et_outlet_paired.csv", [r for r in basin_rows if r["contrast"].startswith(("D_E", "G_E", "XAJ_D_E", "XAJ_G_E"))], fields)
    write_csv(OUT / "04_response" / "response_outlet_paired.csv", [r for r in basin_rows if "D_R" in r["contrast"] or "G_R" in r["contrast"]], fields)
    write_csv(OUT / "03_et" / "et_outlet_summary.csv", [r for r in summary if "D_E" in r["contrast"] or "G_E" in r["contrast"]])
    write_csv(OUT / "04_response" / "response_outlet_summary.csv", [r for r in summary if "D_R" in r["contrast"] or "G_R" in r["contrast"]])
    return basin_rows, summary


def et_partition(replays: dict[str, dict[str, dict[str, np.ndarray]]], forcing: np.ndarray, dates: np.ndarray, basin_ids: np.ndarray) -> list[dict]:
    p = forcing[:, PERIODS["test"], 0].astype(np.float64)
    month = dates[PERIODS["test"]].astype("datetime64[M]").astype(int) % 12
    rows: list[dict] = []
    for regime, labels in (("IC", IC_LABELS), ("dPL", DPL_LABELS)):
        for label in labels:
            archive = replays[regime][label]
            et = archive["evap"][:, PERIODS["test"]].astype(np.float64)
            valid = np.isfinite(et) & np.isfinite(p) & (et >= 0) & (p >= 0)
            et_sum = np.where(valid, et, 0.0).sum(axis=1)
            p_sum = np.where(valid, p, 0.0).sum(axis=1)
            ratio = np.divide(et_sum, p_sum, out=np.full(len(basin_ids), np.nan), where=p_sum > 0)
            monthly = np.full((len(basin_ids), 12), np.nan)
            for m in range(12):
                mask = month == m
                with np.errstate(all="ignore"):
                    monthly[:, m] = np.nanmean(np.where(valid[:, mask], et[:, mask], np.nan), axis=1)
            peak = np.full(len(basin_ids), np.nan)
            peak_val = np.full(len(basin_ids), np.nan)
            for index, row_monthly in enumerate(monthly):
                finite_months = np.isfinite(row_monthly)
                if finite_months.any():
                    peak[index] = int(np.argmax(np.where(finite_months, row_monthly, -np.inf)) + 1)
                    peak_val[index] = float(np.max(row_monthly[finite_months]))
            for basin, r, pm, pv, nvalid in zip(basin_ids, ratio, peak, peak_val, valid.sum(axis=1)):
                rows.append({
                    "regime": regime, "model": label, "basin_id": basin,
                    "et_over_p_test": r, "monthly_et_peak_month": int(pm) if np.isfinite(pm) else "",
                    "monthly_et_peak_mean": pv, "valid_et_p_days": int(nvalid),
                })
    write_csv(OUT / "03_et" / "et_partition_basin.csv", rows)
    summary = []
    for regime, base, variants in (("IC", "N", ("D_E", "G_E")), ("dPL", "XAJ_CONTROLLED_N_CN", ("XAJ_D_E_CN", "XAJ_G_E_CN"))):
        b = {r["basin_id"]: r for r in rows if r["regime"] == regime and r["model"] == base}
        for variant in variants:
            v = {r["basin_id"]: r for r in rows if r["regime"] == regime and r["model"] == variant}
            ratio_delta = np.asarray([v[k]["et_over_p_test"] - b[k]["et_over_p_test"] for k in basin_ids], dtype=np.float64)
            peak_delta = np.asarray([((v[k]["monthly_et_peak_month"] - b[k]["monthly_et_peak_month"] + 6) % 12) - 6 if v[k]["monthly_et_peak_month"] and b[k]["monthly_et_peak_month"] else np.nan for k in basin_ids], dtype=np.float64)
            row = summary_row(regime, f"{variant}-{base}", ratio_delta)
            row["metric"] = "et_over_p_test_delta"
            summary.append(row)
            row = summary_row(regime, f"{variant}-{base}", peak_delta)
            row["metric"] = "monthly_peak_month_delta"
            summary.append(row)
    write_csv(OUT / "03_et" / "et_partition_summary.csv", summary)
    return rows


def read_parameter_rows(regime: str, model: str) -> dict[str, dict[str, float]]:
    path = OUT / "01_parameter_tables" / ("ic_canonical_parameters.csv" if regime == "IC" else "dpl_seed42_parameters.csv")
    out: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("regime") != regime or row.get("model") != model:
                continue
            basin = str(row["basin_id"]).zfill(8)
            out[basin] = {
                key.removeprefix("param_"): float(value)
                for key, value in row.items()
                if key.startswith("param_") and value not in ("", None)
            }
    return out


def parameter_shifts(basin_ids: np.ndarray) -> tuple[list[dict], list[dict]]:
    shifts: list[dict] = []
    boundary: list[dict] = []
    for regime, base, variants in (("IC", "N", ("D_E", "G_E", "D_R", "G_R")), ("dPL", "XAJ_CONTROLLED_N_CN", ("XAJ_D_E_CN", "XAJ_G_E_CN", "XAJ_D_R_CN", "XAJ_G_R_CN"))):
        base_rows = read_parameter_rows(regime, base)
        variant_rows = {model: read_parameter_rows(regime, model) for model in variants}
        for model, values in variant_rows.items():
            common = sorted(set(base_rows[next(iter(base_rows))]) & set(values[next(iter(values))]))
            for basin in basin_ids:
                for parameter in common:
                    delta = values[str(basin)][parameter] - base_rows[str(basin)][parameter]
                    shifts.append({
                        "regime": regime, "contrast": f"{model}-{base}", "basin_id": basin,
                        "parameter": parameter, "signed_delta": delta, "absolute_delta": abs(delta),
                    })
        models = (base,) + variants
        for model in models:
            values = read_parameter_rows(regime, model)
            if not values:
                continue
            for parameter in sorted(set().union(*(set(v) for v in values.values()))):
                if regime == "IC":
                    specs = get_parameter_spec(model if model in IC_LABELS else "N")
                    spec = specs.get(parameter)
                else:
                    config_path = DPL_ROOT / model / "seed_42" / "config.json" if model != "XAJ_CONTROLLED_N_CN" else DPL_N_ROOT / model / "seed_42" / "config.json"
                    spec = json.loads(config_path.read_text(encoding="utf-8")).get("parameter_specs", {}).get(parameter)
                if not spec:
                    continue
                lower, upper = float(spec["lower"]), float(spec["upper"])
                span = max(upper - lower, 1e-30)
                vals = np.asarray([values[str(b)][parameter] for b in basin_ids if parameter in values[str(b)]], dtype=np.float64)
                exact = np.isclose(vals, lower, atol=1e-6, rtol=1e-7) | np.isclose(vals, upper, atol=1e-6, rtol=1e-7)
                near = (np.minimum(np.abs(vals - lower), np.abs(vals - upper)) <= 0.01 * span)
                boundary.append({
                    "regime": regime, "model": model, "parameter": parameter, "n": len(vals),
                    "lower": lower, "upper": upper, "exact_boundary_n": int(exact.sum()),
                    "exact_boundary_fraction": float(exact.mean()) if len(vals) else math.nan,
                    "near_boundary_n_1pct": int(near.sum()),
                    "near_boundary_fraction_1pct": float(near.mean()) if len(vals) else math.nan,
                })
    write_csv(OUT / "03_et" / "et_parameter_shift_basin.csv", [r for r in shifts if "D_E" in r["contrast"] or "G_E" in r["contrast"]])
    write_csv(OUT / "04_response" / "response_parameter_shift_basin.csv", [r for r in shifts if "D_R" in r["contrast"] or "G_R" in r["contrast"]])
    shift_summary = []
    for (regime, contrast, parameter), group in _group_rows(shifts, ("regime", "contrast", "parameter")):
        values = np.asarray([r["signed_delta"] for r in group], dtype=np.float64)
        row = summary_row(regime, contrast, values)
        row["parameter"] = parameter
        row["metric"] = "signed_parameter_delta"
        shift_summary.append(row)
    write_csv(OUT / "03_et" / "et_parameter_shift_summary.csv", [r for r in shift_summary if "D_E" in r["contrast"] or "G_E" in r["contrast"]])
    write_csv(OUT / "04_response" / "response_parameter_shift_summary.csv", [r for r in shift_summary if "D_R" in r["contrast"] or "G_R" in r["contrast"]])
    write_csv(OUT / "03_et" / "et_parameter_boundary_audit.csv", [r for r in boundary if "D_E" in r["model"] or "G_E" in r["model"] or r["model"] in ("N", "XAJ_CONTROLLED_N_CN")])
    write_csv(OUT / "04_response" / "response_parameter_boundary_audit.csv", [r for r in boundary if "D_R" in r["model"] or "G_R" in r["model"]])
    return shifts, boundary


def _group_rows(rows: list[dict], keys: tuple[str, ...]):
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    return groups.items()


def response_bfi(replays: dict[str, dict[str, dict[str, np.ndarray]]], basin_ids: np.ndarray) -> list[dict]:
    rows = []
    for regime, labels in (("IC", IC_LABELS), ("dPL", DPL_LABELS)):
        for label in labels:
            q = replays[regime][label]["qsim"][:, PERIODS["test"]].astype(np.float64)
            values = np.asarray([lyne_hollick_bfi(series) for series in q], dtype=np.float64)
            for basin, value in zip(basin_ids, values):
                rows.append({"regime": regime, "model": label, "basin_id": basin, "bfi": value})
    write_csv(OUT / "04_response" / "response_bfi_basin.csv", rows)
    summary = []
    for regime, base, variants in (("IC", "N", ("D_R", "G_R")), ("dPL", "XAJ_CONTROLLED_N_CN", ("XAJ_D_R_CN", "XAJ_G_R_CN"))):
        b = {r["basin_id"]: r["bfi"] for r in rows if r["regime"] == regime and r["model"] == base}
        for variant in variants:
            v = {r["basin_id"]: r["bfi"] for r in rows if r["regime"] == regime and r["model"] == variant}
            delta = np.asarray([v[k] - b[k] for k in basin_ids], dtype=np.float64)
            summary.append(summary_row(regime, f"{variant}-{base}", delta))
    write_csv(OUT / "04_response" / "response_signatures_summary.csv", summary)
    return rows


def tau_bfi_association(basin_ids: np.ndarray, bfi_rows: list[dict]) -> list[dict]:
    out = []
    for regime, base, variants in (("IC", "N", ("D_R", "G_R")), ("dPL", "XAJ_CONTROLLED_N_CN", ("XAJ_D_R_CN", "XAJ_G_R_CN"))):
        bfi_base = {r["basin_id"]: r["bfi"] for r in bfi_rows if r["regime"] == regime and r["model"] == base}
        for variant in variants:
            param = read_parameter_rows(regime, variant)
            bfi_variant = {r["basin_id"]: r["bfi"] for r in bfi_rows if r["regime"] == regime and r["model"] == variant}
            tau = np.asarray([param[str(b)]["xaj_tau0"] for b in basin_ids], dtype=np.float64)
            delta = np.asarray([bfi_variant[str(b)] - bfi_base[str(b)] for b in basin_ids], dtype=np.float64)
            valid = np.isfinite(tau) & np.isfinite(delta)
            rho = float(spearmanr(tau[valid], delta[valid]).statistic) if valid.sum() >= 3 else math.nan
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            boot = []
            indices = np.flatnonzero(valid)
            for _ in range(BOOTSTRAP_DRAWS):
                sample = rng.choice(indices, size=len(indices), replace=True)
                boot.append(float(spearmanr(tau[sample], delta[sample]).statistic))
            out.append({
                "regime": regime, "contrast": f"{variant}-{base}", "parameter": "xaj_tau0",
                "associated_response_metric": "delta_BFI",
                "valid_n": int(valid.sum()), "tau_median": float(np.median(tau[valid])) if valid.any() else math.nan,
                "rho_spearman": rho,
                "ci95_low": float(np.percentile(boot, 2.5)) if boot else math.nan,
                "ci95_high": float(np.percentile(boot, 97.5)) if boot else math.nan,
                "interpretation": "parameter footprint associated with BFI change; no causal claim",
            })
    write_csv(OUT / "04_response" / "gr_parameter_bfi_association.csv", out)
    return out


def write_status_artifacts() -> None:
    write_json(OUT / "03_et" / "drydown_status.json", {
        "status": "PENDING",
        "reason": "Exact frozen ET dry-spell/AUC/decay protocol was not recoverable from repository artifacts; no event definition was invented.",
        "threshold_sensitivity": "PENDING",
        "main_threshold": "PENDING",
        "sensitivity_threshold": "PENDING",
    })
    (OUT / "03_et" / "drydown_status.md").write_text(
        "# ET dry-down status\n\n**PENDING** — exact frozen dry-spell event, AUC, decay, and separation definitions are not present in the repository. No new event protocol was invented. Therefore the `P < 0.5 mm/day` closure remains pending.\n",
        encoding="utf-8",
    )
    (OUT / "04_response" / "response_signature_definitions.md").write_text(
        "# Response signature definitions\n\n"
        "## BFI (computed)\n\n"
        "Existing `scripts/phase0_sampling.py:lyne_hollick_bfi` is reused exactly: longest finite nonnegative segment, minimum 365 days, three Lyne--Hollick passes, alpha=0.925, baseflow sum divided by discharge sum. The test-period Q from the full-axis replay is used.\n\n"
        "## Recession / low-flow-FDC (PENDING)\n\n"
        "No repository implementation or frozen definition was found for the planned recession or low-flow/FDC primary signatures. No substitute definition was introduced.\n",
        encoding="utf-8",
    )
    write_json(OUT / "04_response" / "response_signature_status.json", {
        "BFI": "COMPUTED_REUSING_PHASE0_LYNE_HOLLICK",
        "recession": "PENDING_NO_FROZEN_DEFINITION",
        "low_flow_FDC": "PENDING_NO_FROZEN_DEFINITION",
        "storage_release": "PENDING_P4_GATE_NOT_FOUND",
    })
    write_json(OUT / "04_response" / "storage_release_status.json", {
        "status": "PENDING",
        "reason": "No COMPARABLE_AFTER_EXPLICIT_AGGREGATION P4 gate or equivalent frozen metric was found.",
    })
    write_json(OUT / "05_p5" / "process_only_distance_status.json", {
        "status": "PENDING",
        "reason": "Required ET dry-down and response recession/low-flow/FDC components are pending; no process-only distance was fabricated.",
        "delta_kge_in_vector": False,
        "old_distance_status": "No local machine-readable old P5 artifact found",
    })
    (OUT / "05_p5" / "process_only_distance_status.md").write_text(
        "# Corrected P5 status\n\n**PENDING**. The process-only vector would exclude `ΔKGE`, but the required frozen ET dry-down and response recession/low-flow/FDC components are not available. No distance or decoupling claim is produced.\n",
        encoding="utf-8",
    )


def cross_process_matrix() -> list[dict]:
    rows = [
        {"process": "Snow", "structural_contrast": "prior frozen Snow R1-R5 contrasts", "outlet_visibility": "frozen prior audit", "parameter_footprint": "frozen prior audit", "targeted_process_visibility": "frozen prior audit", "internal_evidence": "frozen prior audit", "ic_dpl_dependence": "reported in prior audit", "evidence_strength": "prior frozen reference; not recalculated"},
        {"process": "ET", "structural_contrast": "D_E/G_E vs N", "outlet_visibility": "COMPUTED", "parameter_footprint": "COMPUTED common shifts/boundaries", "targeted_process_visibility": "PENDING exact dry-down protocol", "internal_evidence": "not primary", "ic_dpl_dependence": "dPL seed42 only", "evidence_strength": "outlet/parameter available; process layer pending"},
        {"process": "Response", "structural_contrast": "D_R/G_R vs N", "outlet_visibility": "COMPUTED", "parameter_footprint": "COMPUTED tau distribution and tau–ΔBFI association", "targeted_process_visibility": "BFI computed; recession/FDC pending", "internal_evidence": "PENDING P4 gate", "ic_dpl_dependence": "dPL seed42 only", "evidence_strength": "moderate/qualified"},
    ]
    write_csv(OUT / "06_cross_process" / "cross_process_evidence_matrix.csv", rows)
    write_csv(OUT / "figure_ready" / "Table3_X_cross_process.csv", rows)
    return rows


def figure_ready(outlet_rows: list[dict], et_rows: list[dict], bfi_rows: list[dict], tau_rows: list[dict]) -> None:
    write_csv(OUT / "figure_ready" / "Fig3_13_outlet_deltaKGE.csv", [r for r in outlet_rows if r["contrast"].startswith(("D_E", "G_E", "XAJ_D_E", "XAJ_G_E"))])
    write_csv(OUT / "figure_ready" / "Fig3_13_et_partition.csv", et_rows)
    write_csv(OUT / "figure_ready" / "Fig3_14_response_BFI.csv", [r for r in bfi_rows if r["model"] in ("N", "D_R", "G_R", "XAJ_CONTROLLED_N_CN", "XAJ_D_R_CN", "XAJ_G_R_CN")])
    write_csv(OUT / "figure_ready" / "Fig3_14_tau_BFI_association.csv", tau_rows)
    (OUT / "figure_ready" / "figure_panel_status.md").write_text(
        "# Figure-ready panel status\n\n"
        "- Fig3-13(a) outlet ΔKGE: available.\n- Fig3-13(b) basic ET/P and monthly ET: available.\n- Fig3-13(c) dry-down AUC/decay: PENDING protocol.\n- Fig3-13(d) parameter–dry-down: PENDING protocol.\n"
        "- Fig3-14(a) outlet + BFI: available, recession/FDC pending.\n- Fig3-14(b) tau footprint: available as tau distribution/association with ΔBFI, not Δtau against native N.\n- Fig3-14(c) tau–recession: PENDING recession definition.\n- Fig3-14(d) IC–dPL: available only as single-seed qualified comparison.\n",
        encoding="utf-8",
    )


def make_report(outlet_summary: list[dict], tau_rows: list[dict], matrix: list[dict], replay_validation: list[dict]) -> None:
    report = OUT / "CH3_3_6_FINAL_ANALYSIS_REPORT.md"
    def fmt(value):
        return "N/A" if value is None or not np.isfinite(value) else f"{float(value):.6g}"
    lines = [
        "# 3.6 其他水文过程结构差异扩展诊断——最终计算与数据核准报告",
        "",
        "## 1. Executive verdict",
        "",
        "**Readiness: `NOT_READY`**。受控 IC 五模型和 dPL 四变体已完成且 replay 资产已生成；但 dPL controlled-N 仅补 seed42，ET dry-down exact protocol、Response recession/low-flow/FDC 定义、P4 与 corrected P5 仍未核准。因此当前可写方法与 outlet/有限 signature 结果，不能完成 3.6 的 process-level 主结论。",
        "",
        "## 2. Training provenance",
        "",
        "训练源分别为 `results/ic_phase0_controlled_531_v1/` 与 `results/dpl_controlled_531_v1/`。IC: N/D_E/G_E/D_R/G_R，各 531 basin × 10 starts × 100 generations。dPL: D_E/G_E/D_R/G_R，seed42、100 epochs、531 basin。controlled-N dPL 因 legacy XAJ_CN 的 CI/CG domain 不等价，在本轮新增 `dpl_controlled_n_531_v1/XAJ_CONTROLLED_N_CN/seed_42`。",
        "",
        "## 3. Controlled-N dPL baseline verdict",
        "",
        "`controlled_n_equivalence.json` 判定 legacy `XAJ_CN` 与 controlled-N **NOT_EQUIVALENT**：数据、日期、参数名称顺序和 native XAJ dPL 实际映射一致；实质差异为 legacy 的 `ci∈[0.1,1.0]`、`cg∈[0.9,1.0]`，而 controlled domain 为 `ci∈[0.1,0.9]`、`cg∈[0.9,0.998]`。因此未复用 legacy N。",
        "",
        "## 4. tau mapping / bounds verdict",
        "",
        "`tau_mapping_audit.md` 由 stored normalized/physical parameters 反推：IC D_R/G_R 与 dPL D_R/G_R 均为 **`LINEAR_MAPPING_CONFIRMED`**。dPL controlled tau bounds 为 `[0.4342944819, 499.4998332]`。这冻结实际训练口径，不把旧 helper 的 log 声明当作实际训练实现。",
        "",
        "## 5. Canonical parameter inventory",
        "",
        """Canonical IC 选择规则为每 basin/start raw record 中 `max best_train_objective`，同值时 `min start`；dPL 使用 `best_parameters_physical.npz` 与 `basin_final_summary.csv` 的 canonical row order。参数位移与参数–过程 correspondence 分开保存。""",
        "",
        "## 6. Replay validation",
        "",
        "Evaluation-style replay 使用 test forcing 的 365-day preceding warm-up；full replay 使用完整 12,418-day axis、zero initial states，过程指标使用 test slice。无 frozen numerical tolerance，故报告 mismatch 分布而不静默 pass/fail。",
        "",
        "| Regime | Model | N | abs mismatch median | abs mismatch P95 | abs mismatch max | nonfinite Q |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in replay_validation:
        lines.append(f"| {row['regime']} | {row['model']} | {row['basin_n']} | {fmt(row['mismatch_abs_median'])} | {fmt(row['mismatch_abs_p95'])} | {fmt(row['mismatch_abs_max'])} | {row['qsim_nonfinite']} |")
    lines += [
        "",
        "## 7. ET outlet results",
        "",
        "ET D_E/G_E outlet paired ΔKGE 已计算，详见 `03_et/et_outlet_summary.csv`。不预设 near-equivalence；正文应仅引用带 CI 的实际结果。",
        "",
        "| Regime | Contrast | Median ΔKGE | 95% CI | Positive fraction |",
        "|---|---|---:|---|---:|",
    ]
    for row in outlet_summary:
        if "D_E" in row["contrast"] or "G_E" in row["contrast"]:
            lines.append(f"| {row['regime']} | {row['contrast']} | {fmt(row['median'])} | [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | {fmt(row['positive_fraction'])} |")
    lines += [
        "",
        "## 8. ET long-term / seasonal partition",
        "",
        "已计算 evaluation `ET/P` 与 monthly climatological ET peak month，作为基础 ET partition 层。它不是 dry-down 主证据。",
        "",
        "## 9. ET dry-down results",
        "",
        "**`PENDING`**：exact frozen dry-spell、AUC、decay 与 event separation 定义无法从当前 repository artifacts 恢复；没有自行发明 protocol。",
        "",
        "## 10. ET 0.5 mm/day sensitivity",
        "",
        "**`PENDING`**。由于主 `P<1.0 mm/day` 事件定义本身未核准，不能执行 `1.0→0.5` 的唯一阈值 closure。",
        "",
        "## 11. ET parameter displacement",
        "",
        "已输出 D_E/G_E 相对 N 的 common-parameter signed/absolute shifts 与 exact/1%-near boundary audit。不能把位移大小称为“强参数补偿”。",
        "",
        "## 12. ET parameter–process correspondence",
        "",
        "**`PENDING`**：predeclared `xaj_k`–dry-down correspondence 依赖缺失的 dry-down metric；本轮不计算 substitute correlation。",
        "",
        "## 13. Response outlet results",
        "",
        "D_R/G_R 相对 N 的 paired ΔKGE 已计算，详见 `04_response/response_outlet_summary.csv`。",
        "",
        "## 14. Response targeted signatures",
        "",
        "现有仓库唯一可复用的 primary signature 是三遍 Lyne–Hollick BFI，已计算并保存。recession 与 low-flow/FDC 没有 frozen implementation/definition，因此保持 `PENDING`。",
        "",
        "## 15. Response parameter footprint",
        "",
        "tau0 实际为 linear mapping。由于没有 frozen native-N effective timescale，不构造 `Δtau` against N；仅报告 controlled D_R/G_R tau distribution 及 tau–paired ΔBFI association。",
        "",
        "| Regime | Contrast | Tau median | Spearman tau–ΔBFI | 95% CI |",
        "|---|---|---:|---:|---|",
    ]
    for row in tau_rows:
        lines.append(f"| {row['regime']} | {row['contrast']} | {fmt(row['tau_median'])} | {fmt(row['rho_spearman'])} | [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] |")
    lines += [
        "",
        "## 16. IC vs dPL Response expression",
        "",
        "dPL 仅有 seed42；IC–dPL 只能作为 single-seed directional/expression comparison，不能比较 seed stability、variance superiority 或 compensation strength。",
        "",
        "## 17. Response storage–release evidence",
        "",
        "**`PENDING`**：未找到 `COMPARABLE_AFTER_EXPLICIT_AGGREGATION` P4 gate 或等价冻结指标；不把 `z`/native `qi,qg` 的未经转换绝对量写成内部证据。",
        "",
        "## 18. Corrected P5",
        "",
        "**`PENDING`**。process-only vector 明确排除 `ΔKGE`，但 ET dry-down 和 Response recession/low-flow/FDC 组成项缺失；不生成 decoupling claim。",
        "",
        "## 19. Snow / ET / Response evidence matrix",
        "",
        "见 `06_cross_process/cross_process_evidence_matrix.csv`。Snow 仅读取 prior frozen audit，不重新计算；ET/Response 按实际完成层级记录。",
        "",
        "## 20. Figure 3-13 final panel recommendation",
        "",
        "保留 outlet ΔKGE 与基础 ET partition；dry-down AUC/decay 和 parameter–dry-down panel 标为 PENDING，不为填图强行生成。",
        "",
        "## 21. Figure 3-14 final panel recommendation",
        "",
        "保留 outlet + BFI、tau footprint（不作 native-N Δtau）和 qualified IC–dPL panel；recession/FDC panel 暂缓。",
        "",
        "## 22. Table 3-X recommendation",
        "",
        "使用 `figure_ready/Table3_X_cross_process.csv`，明确 Snow 为 prior frozen reference，ET/Response 为分层完成状态，不做跨过程 effect-size 排名。",
        "",
        "## 23. 建议进入 3.6 正文的 12–18 个关键数字",
        "",
        "当前可安全使用的数字包括：531 basin；IC 10 starts；IC 100 generations；dPL seed42；dPL 100 epochs；full axis 12,418 d；test 5,479 d；warm-up 365 d；train 5,113 d；参数维度 17/16/17/15/16；controlled CI `[0.1,0.9]`；controlled CG `[0.9,0.998]`；tau `[0.4342944819,499.4998332]`；以及 `03_et/04_response` 中带 CI 的实际 ΔKGE/BFI/tau association 数字。dry-down/P5 数字不可进入正文。",
        "",
        "## 24. 建议进入脚注的数据",
        "",
        "dPL single-seed；dPL root manifest 中历史 `started` 行；remote launcher 的 pid-file warning；IC manifest `git_commit=UNVERIFIED`；实际 tau linear mapping 与旧文档/helper 冲突；BFI valid N；所有 boundary/finite-N 规则。",
        "",
        "## 25. 建议进入附录的数据",
        "",
        "全部 canonical parameter tables、restart table、replay NPZ/JSON、boundary audit、BFI basin table、tau association bootstrap、PENDING status files 与 checksums。",
        "`checksums.sha256` 覆盖本输出根目录的结果、状态、日志和图件文件；排除 `logs/torch_inductor_*` 临时编译缓存，不替代源码或原始训练资产的 checksum。",
        "",
        "## 26. Null / opposite / anomalous results",
        "",
        "所有正负方向、null、非单调和 boundary-sensitive 结果按 CSV 原值保留；本轮不 winsorize、不静默删除。",
        "下游 BFI/partition/parameter summaries 使用 replay 中 531 个 finite basin；不使用异常 mismatch 作为事后排除条件，异常计数保留在 replay validation 中，属于残余 validation risk。",
        "",
        "## 27. Remaining limitations",
        "",
        "ET dry-down protocol、Response recession/low-flow/FDC protocol、P4 storage-release gate、corrected P5、dPL multi-seed 与 runtime code snapshot provenance 仍有限制。",
        "",
        "## 28. 是否需要额外 dPL seeds",
        "",
        "`RECOMMENDED BUT NOT REQUIRED`：seed42 足以生成限定性的 dPL expression evidence，但不足以声明 seed stability 或普遍性。",
        "",
        "## 29. 3.6 最终三级标题",
        "",
        "1. 3.6.1 蒸散发过程结构差异的诊断\n2. 3.6.2 地下水响应路径结构差异的诊断\n3. 3.6.3 不同水文过程结构差异的综合比较",
        "",
        "## 30. Final claim hierarchy",
        "",
        "| Claim | Strength | Exact evidence | Limitation | Recommended thesis wording |",
        "|---|---|---|---|---|",
        "| 受控 IC 五模型训练完成 | Strong | IC DONE/raw/checkpoint/manifest | `git_commit=UNVERIFIED` | “五个受控结构均完成 Phase-0 IC 训练协议。” |",
        "| 受控 dPL 四变体 seed42 完成 | Moderate | COMPLETE、epoch100、531-row summaries | single seed、manifest registry drift | “在 seed42 dPL 约束下……” |",
        "| ET/Response outlet contrast | Moderate | paired replay KGE CSV + bootstrap | process-level layers incomplete | “出口层差异按实际 ΔKGE 与 CI 报告。” |",
        "| BFI response evidence | Moderate | response BFI basin/summary | recession/FDC 未定义 | “BFI 层显示的 response signature 差异……” |",
        "| tau footprint | Moderate | linear mapping audit + tau–ΔBFI association | no native-N Δtau / causal claim | “tau parameter footprint is associated with BFI change.” |",
        "| ET dry-down non-equivalence | Unsupported | no legal event protocol/result | exact definition missing | 不写。 |",
        "| performance–process decoupling | Unsupported | corrected P5 pending | process-only vector incomplete | 不写。 |",
        "",
        "# Final status: `NOT_READY`",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    bundle = load_bundle(args.data_root)
    basin_ids = np.asarray([str(value).zfill(8) for value in bundle.basin_ids])
    replays = load_all_replays()
    # The loading itself is the final hard prerequisite: no local calculation
    # proceeds with a missing controlled-N or missing full replay.
    replay_validation_path = REPLAY / "replay_validation.csv"
    if not replay_validation_path.exists():
        raise RuntimeError(f"Missing required replay validation: {replay_validation_path}")
    with replay_validation_path.open(newline="", encoding="utf-8") as handle:
        replay_validation = [
            {k: (float(v) if k not in ("regime", "model", "structure", "artifact") else v) for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]
    expected_validation_models = {"N", "D_E", "G_E", "D_R", "G_R", "XAJ_CONTROLLED_N_CN", "XAJ_D_E_CN", "XAJ_G_E_CN", "XAJ_D_R_CN", "XAJ_G_R_CN"}
    if len(replay_validation) != 10 or {row["model"] for row in replay_validation} != expected_validation_models:
        raise RuntimeError("Replay validation must contain exactly one complete row for each of the ten required models")
    outlet_rows, outlet_summary = outlet_results(replays, basin_ids)
    et_rows = et_partition(replays, np.asarray(bundle.forcing), np.asarray(bundle.dates), basin_ids)
    parameter_shifts(basin_ids)
    bfi_rows = response_bfi(replays, basin_ids)
    tau_rows = tau_bfi_association(basin_ids, bfi_rows)
    write_status_artifacts()
    matrix = cross_process_matrix()
    figure_ready(outlet_rows, et_rows, bfi_rows, tau_rows)
    make_report(outlet_summary, tau_rows, matrix, replay_validation)
    for group in replays.values():
        for archive in group.values():
            archive.close()
    print(f"Wrote Chapter 3.6 analysis outputs under {OUT}")


if __name__ == "__main__":
    main()
