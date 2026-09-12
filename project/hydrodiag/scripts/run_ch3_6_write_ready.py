#!/usr/bin/env python3
"""Assemble Chapter 3.6 write-ready summaries from canonical artifacts.

No model is trained or replayed here. Existing basin-level CSVs and stored full
replay state arrays are read, summarized with basin bootstrap, and written only
under results/ch3_6_cross_process/08_write_ready.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
ROOT = PROJECT / "results" / "ch3_6_cross_process"
OUT = ROOT / "08_write_ready"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260730
EVAL = slice(5478, 10957)
MODEL_GROUPS = {
    "IC": {"N": "N", "D_E": "D_E", "G_E": "G_E"},
    "dPL": {"N": "XAJ_CONTROLLED_N_CN", "D_E": "XAJ_D_E_CN", "G_E": "XAJ_G_E_CN"},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str | float | int | None) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def fmt(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.10g}"


def finite_values(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = finite_values(values)
    if not values.size:
        return float("nan"), float("nan")
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_DRAWS, values.size))
    medians = np.median(values[indices], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def summary(values: list[float] | np.ndarray, rng: np.random.Generator,
            include_abs: bool = False) -> dict[str, str]:
    values = finite_values(values)
    empty = {"valid_n": "0", "median": "", "q25": "", "q75": "", "ci95_low": "", "ci95_high": "", "positive_fraction": "", "negative_fraction": ""}
    if not values.size:
        if include_abs:
            empty["median_abs"] = ""
        return empty
    low, high = bootstrap_ci(values, rng)
    result = {
        "valid_n": str(values.size),
        "median": fmt(float(np.median(values))),
        "q25": fmt(float(np.percentile(values, 25))),
        "q75": fmt(float(np.percentile(values, 75))),
        "ci95_low": fmt(low),
        "ci95_high": fmt(high),
        "positive_fraction": fmt(float(np.mean(values > 0))),
        "negative_fraction": fmt(float(np.mean(values < 0))),
    }
    if include_abs:
        result["median_abs"] = fmt(float(np.median(np.abs(values))))
    return result


def group_by(rows: list[dict[str, str]], keys: tuple[str, ...]):
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups.items()


def model_from_role(regime: str, role: str, groups: dict[str, str]) -> str:
    return groups[role]


def contrast_parts(contrast: str) -> tuple[str, str]:
    """Canonical contrast strings are variant-baseline; return baseline, variant."""
    variant, baseline = contrast.split("-", 1)
    return baseline, variant


def basin_maps(rows: list[dict[str, str]], regime: str, model: str, key: str) -> dict[str, float]:
    return {r["basin_id"]: as_float(r[key]) for r in rows if r["regime"] == regime and r["model"] == model}


def write_et_outlet(rng: np.random.Generator) -> None:
    source = read_csv(ROOT / "03_et" / "et_outlet_paired.csv")
    fields = ["regime", "contrast", "baseline_model", "variant_model", "valid_n", "median_delta_kge", "q25_delta_kge", "q75_delta_kge", "ci95_low", "ci95_high", "positive_fraction", "negative_fraction", "median_abs_delta_kge"]
    rows: list[dict] = []
    for (regime, contrast), group in group_by(source, ("regime", "contrast")):
        stats = summary([as_float(r["delta_kge"]) for r in group], rng, include_abs=True)
        base, variant = contrast_parts(contrast)
        rows.append({"regime": regime, "contrast": contrast, "baseline_model": base, "variant_model": variant,
                     "valid_n": stats["valid_n"], "median_delta_kge": stats["median"], "q25_delta_kge": stats["q25"], "q75_delta_kge": stats["q75"],
                     "ci95_low": stats["ci95_low"], "ci95_high": stats["ci95_high"], "positive_fraction": stats["positive_fraction"],
                     "negative_fraction": stats["negative_fraction"], "median_abs_delta_kge": stats["median_abs"]})
    write_csv(OUT / "et_outlet_summary.csv", rows, fields)


def write_et_partition(rng: np.random.Generator) -> None:
    source = read_csv(ROOT / "03_et" / "et_partition_basin.csv")
    fields = ["record_type", "regime", "model", "contrast", "baseline_model", "variant_model", "metric", "valid_n", "median", "q25", "q75", "ci95_low", "ci95_high", "positive_fraction", "negative_fraction"]
    rows: list[dict] = []
    for regime, groups in MODEL_GROUPS.items():
        maps = {}
        for role, model in groups.items():
            maps[role] = basin_maps(source, regime, model, "et_over_p_test")
            stats = summary(list(maps[role].values()), rng)
            stats["positive_fraction"] = ""
            stats["negative_fraction"] = ""
            rows.append({"record_type": "structure", "regime": regime, "model": model, "contrast": "", "baseline_model": "", "variant_model": "", "metric": "et_over_p_test", **stats})
        for role in ("D_E", "G_E"):
            common = sorted(set(maps["N"]) & set(maps[role]))
            values = np.asarray([maps[role][basin] - maps["N"][basin] for basin in common], dtype=np.float64)
            stats = summary(values, rng)
            rows.append({"record_type": "contrast", "regime": regime, "model": "", "contrast": f"{groups[role]}-{groups['N']}", "baseline_model": groups["N"], "variant_model": groups[role], "metric": "delta_et_over_p_test", **stats})
    write_csv(OUT / "et_partition_summary.csv", rows, fields)


def load_monthly_et(regime: str, model: str, expected_ids: tuple[str, ...], expected_dates: np.ndarray) -> dict[int, np.ndarray]:
    folder = "IC" if regime == "IC" else "dPL"
    path = ROOT / "02_replay" / folder / f"{model}_full_replay.npz"
    with np.load(path, allow_pickle=False) as archive:
        ids = tuple(str(x).zfill(8) for x in archive["basin_ids"])
        if ids != expected_ids:
            raise RuntimeError(f"{regime}/{model}: basin order differs from canonical bundle")
        dates_full = np.asarray(archive["dates"], dtype="datetime64[D]")
        qsim_shape = tuple(archive["qsim"].shape)
        evap_shape = tuple(archive["evap"].shape)
        if dates_full.shape != (12418,) or qsim_shape != (531, 12418) or evap_shape != (531, 12418):
            raise RuntimeError(f"{regime}/{model}: unexpected full replay shapes dates={dates_full.shape} qsim={qsim_shape} evap={evap_shape}")
        if not np.array_equal(dates_full, expected_dates):
            raise RuntimeError(f"{regime}/{model}: full replay dates differ from canonical reference")
        if not np.all(np.diff(dates_full).astype("timedelta64[D]") == np.timedelta64(1, "D")):
            raise RuntimeError(f"{regime}/{model}: dates are not consecutive daily records")
        dates = dates_full[EVAL]
        evap = np.asarray(archive["evap"][:, EVAL], dtype=np.float64)
    month_numbers = (dates.astype("datetime64[M]").astype(np.int64) % 12) + 1
    output = {}
    for month in range(1, 13):
        day_mask = month_numbers == month
        output[month] = np.nanmean(np.where(np.isfinite(evap[:, day_mask]), evap[:, day_mask], np.nan), axis=1) if np.any(day_mask) else np.full(evap.shape[0], np.nan)
    return output


def write_monthly_climatology(rng: np.random.Generator) -> None:
    partition = read_csv(ROOT / "03_et" / "et_partition_basin.csv")
    expected = tuple(r["basin_id"] for r in partition if r["regime"] == "IC" and r["model"] == "N")
    with np.load(ROOT / "02_replay" / "IC" / "N_full_replay.npz", allow_pickle=False) as archive:
        expected_dates = np.asarray(archive["dates"], dtype="datetime64[D]")
    if len(expected) != 531 or len(set(expected)) != 531:
        raise RuntimeError("canonical IC N basin IDs are not an ordered 531-basin set")
    fields = ["record_type", "regime", "model", "contrast", "baseline_model", "variant_model", "month", "month_name", "metric", "valid_n", "median", "q25", "q75", "ci95_low", "ci95_high", "positive_fraction", "negative_fraction"]
    rows: list[dict] = []
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for regime, groups in MODEL_GROUPS.items():
        values_by_role = {role: load_monthly_et(regime, model, expected, expected_dates) for role, model in groups.items()}
        for month in range(1, 13):
            for role, model in groups.items():
                stats = summary(values_by_role[role][month], rng)
                stats["positive_fraction"] = ""
                stats["negative_fraction"] = ""
                rows.append({"record_type": "structure", "regime": regime, "model": model, "contrast": "", "baseline_model": "", "variant_model": "", "month": month, "month_name": month_names[month - 1], "metric": "monthly_mean_daily_et_mm_day", **stats})
            for role in ("D_E", "G_E"):
                values = values_by_role[role][month] - values_by_role["N"][month]
                stats = summary(values, rng)
                rows.append({"record_type": "contrast", "regime": regime, "model": "", "contrast": f"{groups[role]}-{groups['N']}", "baseline_model": groups["N"], "variant_model": groups[role], "month": month, "month_name": month_names[month - 1], "metric": "monthly_delta_et_mm_day", **stats})
    write_csv(OUT / "et_monthly_climatology.csv", rows, fields)


def parameter_bounds() -> dict[tuple[str, str, str], dict[str, str]]:
    return {(r["regime"], r["model"], r["parameter"]): r for r in read_csv(ROOT / "00_provenance" / "parameter_bounds.csv")}


def et_boundary_map() -> dict[tuple[str, str, str], dict[str, str]]:
    return {(r["regime"], r["model"], r["parameter"]): r for r in read_csv(ROOT / "03_et" / "et_parameter_boundary_audit.csv")}


def write_parameter_shifts(source_rel: str, boundary: bool, output: str, rng: np.random.Generator, allowed: set[str]) -> None:
    source = read_csv(ROOT / source_rel)
    bounds = parameter_bounds()
    bmap = et_boundary_map() if boundary else {}
    fields = ["regime", "contrast", "baseline_model", "variant_model", "parameter", "valid_n", "median_signed_shift_physical", "normalized_signed_median_shift", "median_absolute_shift", "normalized_median_absolute_shift", "q25_signed_shift", "q75_signed_shift", "ci95_low_signed", "ci95_high_signed", "positive_fraction", "negative_fraction", "baseline_exact_boundary_fraction", "variant_exact_boundary_fraction", "baseline_near_boundary_fraction_1pct", "variant_near_boundary_fraction_1pct"]
    rows: list[dict] = []
    for (regime, contrast, parameter), group in group_by(source, ("regime", "contrast", "parameter")):
        if contrast not in allowed:
            continue
        baseline, variant = contrast_parts(contrast)
        physical = np.asarray([as_float(r["signed_delta"]) for r in group], dtype=np.float64)
        absolute = np.asarray([as_float(r["absolute_delta"]) for r in group], dtype=np.float64)
        spec = bounds.get((regime, variant, parameter)) or bounds.get((regime, baseline, parameter))
        if not spec:
            raise RuntimeError(f"missing authoritative bounds for {regime}/{contrast}/{parameter}")
        span = as_float(spec["upper"]) - as_float(spec["lower"])
        normalized = physical / span if span > 0 else np.full_like(physical, np.nan)
        pstats = summary(physical, rng)
        nstats = summary(normalized, rng)
        abs_values = finite_values(absolute)
        normalized_abs_values = finite_values(np.abs(normalized))
        base_row = bmap.get((regime, baseline, parameter), {})
        variant_row = bmap.get((regime, variant, parameter), {})
        rows.append({
            "regime": regime, "contrast": contrast, "baseline_model": baseline, "variant_model": variant, "parameter": parameter, "valid_n": pstats["valid_n"],
            "median_signed_shift_physical": pstats["median"], "normalized_signed_median_shift": nstats["median"], "median_absolute_shift": fmt(float(np.median(abs_values))) if abs_values.size else "", "normalized_median_absolute_shift": fmt(float(np.median(normalized_abs_values))) if normalized_abs_values.size else "",
            "q25_signed_shift": pstats["q25"], "q75_signed_shift": pstats["q75"], "ci95_low_signed": pstats["ci95_low"], "ci95_high_signed": pstats["ci95_high"],
            "positive_fraction": pstats["positive_fraction"], "negative_fraction": pstats["negative_fraction"],
            "baseline_exact_boundary_fraction": base_row.get("exact_boundary_fraction", "") if boundary else "", "variant_exact_boundary_fraction": variant_row.get("exact_boundary_fraction", "") if boundary else "",
            "baseline_near_boundary_fraction_1pct": base_row.get("near_boundary_fraction_1pct", "") if boundary else "", "variant_near_boundary_fraction_1pct": variant_row.get("near_boundary_fraction_1pct", "") if boundary else "",
        })
    write_csv(OUT / output, rows, fields)


def response_boundary_map() -> dict[tuple[str, str, str], dict[str, str]]:
    return {(r["regime"], r["model"], r["parameter"]): r for r in read_csv(ROOT / "04_response" / "response_parameter_boundary_audit.csv")}


def write_response_outlet_bfi(rng: np.random.Generator) -> None:
    outlet = read_csv(ROOT / "04_response" / "response_outlet_paired.csv")
    bfi = read_csv(ROOT / "04_response" / "response_bfi_basin.csv")
    bfi_maps = {(regime, model): basin_maps(bfi, regime, model, "bfi") for regime in ("IC", "dPL") for model in set(r["model"] for r in bfi if r["regime"] == regime)}
    fields = ["regime", "contrast", "baseline_model", "variant_model", "outlet_valid_n", "outlet_median_delta_kge", "outlet_ci95_low", "outlet_ci95_high", "outlet_positive_fraction", "outlet_median_abs_delta_kge", "bfi_valid_n", "bfi_median_delta", "bfi_q25", "bfi_q75", "bfi_ci95_low", "bfi_ci95_high", "bfi_positive_fraction", "bfi_negative_fraction"]
    rows = []
    for (regime, contrast), group in group_by(outlet, ("regime", "contrast")):
        baseline, variant = contrast_parts(contrast)
        out_stats = summary([as_float(r["delta_kge"]) for r in group], rng, include_abs=True)
        bbase = bfi_maps[(regime, baseline)]
        bvariant = bfi_maps[(regime, variant)]
        bvalues = [bvariant[basin] - bbase[basin] for basin in sorted(set(bbase) & set(bvariant))]
        bstats = summary(bvalues, rng)
        rows.append({"regime": regime, "contrast": contrast, "baseline_model": baseline, "variant_model": variant,
                     "outlet_valid_n": out_stats["valid_n"], "outlet_median_delta_kge": out_stats["median"], "outlet_ci95_low": out_stats["ci95_low"], "outlet_ci95_high": out_stats["ci95_high"], "outlet_positive_fraction": out_stats["positive_fraction"], "outlet_median_abs_delta_kge": out_stats["median_abs"],
                     "bfi_valid_n": bstats["valid_n"], "bfi_median_delta": bstats["median"], "bfi_q25": bstats["q25"], "bfi_q75": bstats["q75"], "bfi_ci95_low": bstats["ci95_low"], "bfi_ci95_high": bstats["ci95_high"], "bfi_positive_fraction": bstats["positive_fraction"], "bfi_negative_fraction": bstats["negative_fraction"]})
    write_csv(OUT / "response_outlet_bfi_summary.csv", rows, fields)


def response_tau_values(regime: str, model: str) -> dict[str, float]:
    filename = "ic_canonical_parameters.csv" if regime == "IC" else "dpl_seed42_parameters.csv"
    rows = read_csv(ROOT / "01_parameter_tables" / filename)
    return {r["basin_id"]: as_float(r["param_xaj_tau0"]) for r in rows if r["regime"] == regime and r["model"] == model}


def write_tau_bfi() -> None:
    source = read_csv(ROOT / "04_response" / "gr_parameter_bfi_association.csv")
    bfi = read_csv(ROOT / "04_response" / "response_bfi_basin.csv")
    boundary = response_boundary_map()
    bfi_maps = {(regime, model): basin_maps(bfi, regime, model, "bfi") for regime in ("IC", "dPL") for model in set(r["model"] for r in bfi if r["regime"] == regime)}
    fields = ["regime", "contrast", "baseline_model", "variant_model", "parameter", "associated_response_metric", "valid_n", "tau_median", "tau_q25", "tau_q75", "rho_spearman", "ci95_low", "ci95_high", "rho_sign", "variant_exact_boundary_fraction", "variant_near_boundary_fraction_1pct", "ci_method", "bootstrap_draws", "bootstrap_seed", "interpretation"]
    rows = []
    for r in source:
        baseline, variant = contrast_parts(r["contrast"])
        rho = as_float(r["rho_spearman"])
        base_bfi = bfi_maps[(r["regime"], baseline)]
        variant_bfi = bfi_maps[(r["regime"], variant)]
        tau = response_tau_values(r["regime"], variant)
        valid_basins = sorted(set(base_bfi) & set(variant_bfi) & set(tau))
        tau_values = finite_values([tau[basin] for basin in valid_basins if math.isfinite(variant_bfi[basin] - base_bfi[basin])])
        if tau_values.size != int(r["valid_n"]):
            raise RuntimeError(f"{r['regime']}/{r['contrast']}: tau valid N differs from canonical association")
        tau_row = boundary.get((r["regime"], variant, "xaj_tau0"), {})
        rows.append({**r, "baseline_model": baseline, "variant_model": variant, "tau_median": fmt(float(np.median(tau_values))) if tau_values.size else "", "tau_q25": fmt(float(np.percentile(tau_values, 25))) if tau_values.size else "", "tau_q75": fmt(float(np.percentile(tau_values, 75))) if tau_values.size else "", "rho_sign": "positive" if rho > 0 else "negative" if rho < 0 else "zero", "variant_exact_boundary_fraction": tau_row.get("exact_boundary_fraction", ""), "variant_near_boundary_fraction_1pct": tau_row.get("near_boundary_fraction_1pct", ""), "ci_method": "canonical association bootstrap over finite basin pairs", "bootstrap_draws": 2000, "bootstrap_seed": 20260730, "interpretation": "weak association; no causal claim"})
    write_csv(OUT / "response_tau_bfi_summary.csv", rows, fields)


def write_selection() -> None:
    (OUT / "et_representative_parameter_selection.md").write_text("""# ET representative parameter selection

**Selection: no representative parameter is promoted to the main-text panel.**

The rule prioritized direct ET meaning, a strict common parameter, broadly consistent IC/dPL direction, limited boundary concentration, and no dependence on unrecovered dry-down correspondence. `xaj_k` is the only common parameter with an explicit potential-ET/reference-evaporation interpretation. Its signed shift is negative in both IC contrasts and both dPL contrasts, but exact-boundary prevalence is approximately 18.1–18.5% in IC; it is therefore not a sufficiently clean main-text representative under the boundary rule.

The remaining common parameters are storage, routing, or snow parameters rather than direct ET controls; several have higher boundary prevalence or inconsistent IC/dPL directions. They are retained in `et_parameter_shift_all.csv` for complete evidence and are not screened by effect size. No parameter is selected by maximum magnitude, statistical significance, or unavailable dry-down correspondence.

The main ET figure consequently uses outlet, ET/P, and full annual-cycle climatology panels only.
""", encoding="utf-8")


def write_matrix() -> None:
    fields = ["Process", "Structural intervention", "Outlet evidence", "Parameter evidence", "Targeted process/internal evidence", "IC/dPL consistency", "Evidence completeness", "Main observable layer"]
    rows = [
        {"Process": "Snow", "Structural intervention": "Prior frozen R1–R5 structural contrasts", "Outlet evidence": "Prior frozen outlet timing/skill evidence", "Parameter evidence": "Prior frozen parameter compensation", "Targeted process/internal evidence": "Prior frozen active-melt timing and internal-state evidence", "IC/dPL consistency": "As reported in frozen Snow audits", "Evidence completeness": "Frozen and complete for prior scope", "Main observable layer": "High-process-activity outlet plus parameter/internal evidence"},
        {"Process": "ET", "Structural intervention": "D_E/G_E versus N", "Outlet evidence": "Small paired ΔKGE; some CIs span zero", "Parameter evidence": "All common shifts and boundary audit; no clean representative", "Targeted process/internal evidence": "ET/P and monthly ET climatology available; dry-down unavailable", "IC/dPL consistency": "Directions partly agree; dPL seed42 only", "Evidence completeness": "Write-ready for available layers; historical dry-down plan not implemented", "Main observable layer": "Outlet/partition weak-to-limited; boundary-qualified parameter evidence"},
        {"Process": "Response", "Structural intervention": "D_R/G_R versus N", "Outlet evidence": "Small paired ΔKGE", "Parameter evidence": "Shared-parameter shifts and tau–BFI association", "Targeted process/internal evidence": "BFI available; recession/FDC and P4 not implemented", "IC/dPL consistency": "BFI directions differ by contrast; dPL seed42 only", "Evidence completeness": "Write-ready for available layers; targeted protocol boundary explicit", "Main observable layer": "BFI/tau–BFI targeted evidence with qualified parameter footprint"},
    ]
    write_csv(OUT / "cross_process_evidence_matrix_final.csv", rows, fields)


def write_figure_ready() -> None:
    fig = OUT / "figure_ready"
    fig.mkdir(parents=True, exist_ok=True)
    for source, output in [
        ("et_outlet_summary.csv", "Fig3_13_ET_outlet.csv"),
        ("et_partition_summary.csv", "Fig3_13_ET_partition.csv"),
        ("et_monthly_climatology.csv", "Fig3_13_ET_monthly.csv"),
        ("response_outlet_bfi_summary.csv", "Fig3_14_Response_outlet_BFI.csv"),
        ("response_tau_bfi_summary.csv", "Fig3_14_Response_tau_BFI.csv"),
        ("cross_process_evidence_matrix_final.csv", "Table3_X_cross_process_final.csv"),
    ]:
        rows = read_csv(OUT / source)
        if rows:
            write_csv(fig / output, rows, list(rows[0]))
    (fig / "FIGURE_PANEL_RECOMMENDATION.md").write_text("""# Figure panel recommendation

## Figure 3-13 — ET

Use three panels: paired outlet ΔKGE, paired ET/P, and the full 12-month ET climatology. Do not add dry-down or a single-parameter panel; the former protocol is unavailable and no representative parameter passed the boundary/interpretation rule.

## Figure 3-14 — Response

Use paired outlet ΔKGE, paired BFI, and a tau–BFI association summary. The tau association is weak and should be labelled as such. Do not add recession, low-flow/FDC, tau–recession, or corrected-P5 panels.
""", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    write_et_outlet(rng)
    write_et_partition(rng)
    write_monthly_climatology(rng)
    write_parameter_shifts("03_et/et_parameter_shift_basin.csv", True, "et_parameter_shift_all.csv", rng, {"D_E-N", "G_E-N", "XAJ_D_E_CN-XAJ_CONTROLLED_N_CN", "XAJ_G_E_CN-XAJ_CONTROLLED_N_CN"})
    write_response_outlet_bfi(rng)
    write_tau_bfi()
    write_parameter_shifts("04_response/response_parameter_shift_basin.csv", False, "response_shared_parameter_shift_all.csv", rng, {"D_R-N", "G_R-N", "XAJ_D_R_CN-XAJ_CONTROLLED_N_CN", "XAJ_G_R_CN-XAJ_CONTROLLED_N_CN"})
    write_selection()
    write_matrix()
    write_figure_ready()
    (OUT / "write_ready_metadata.json").write_text(json.dumps({
        "source_root": str(ROOT), "output_root": str(OUT), "no_training_or_replay": True,
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED, "basin_as_inferential_unit": True,
        "monthly_source": "existing full replay evap arrays, evaluation slice 5478:10957; no forward rerun",
        "parameter_normalization": "existing physical signed_delta divided by authoritative variant-model parameter span; physical value retained separately",
        "excluded_unrecoverable_metrics": ["ET dry-down", "recession", "low-flow/FDC", "tau-recession", "corrected P5"],
        "lite_full_gate": "07_gap_fill/01_lite_full_gate/lite_full_summary.csv",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
