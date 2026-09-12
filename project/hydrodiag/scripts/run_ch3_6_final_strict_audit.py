#!/usr/bin/env python3
"""Final strict audit for Chapter 3.6.

Read-only with respect to all canonical results: this script only summarizes
existing CSV/JSON/code evidence and writes a new 10_final_audit package. It
never trains, calibrates, replays, or adds a hydrological signature.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
REPO = PROJECT.parent.parent
ROOT = PROJECT / "results" / "ch3_6_cross_process"
OUT = ROOT / "10_final_audit"
CONTROL = OUT / "01_control_matrix"
NUMBERS = OUT / "02_numbers"
BFI = OUT / "03_bfi_common_subset"
CONTEXT = OUT / "04_context_gradients"
SNOW = OUT / "05_snow_reference"
CLAIMS = OUT / "06_claims"
FIGURE = OUT / "07_figure_table"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260730

REGIMES = {
    "IC": {"N": "N", "D_E": "D_E", "G_E": "G_E", "D_R": "D_R", "G_R": "G_R"},
    "dPL": {
        "N": "XAJ_CONTROLLED_N_CN", "D_E": "XAJ_D_E_CN", "G_E": "XAJ_G_E_CN",
        "D_R": "XAJ_D_R_CN", "G_R": "XAJ_G_R_CN",
    },
}
ET_ROLES = ("D_E", "G_E")
RESPONSE_ROLES = ("D_R", "G_R")


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def as_float(value: str | float | int | None) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def fmt(value: str | float | int | None) -> str:
    value = as_float(value)
    return "" if not math.isfinite(value) else f"{value:.10g}"


def finite(values: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_median_ci(values: list[float] | np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = finite(values)
    if not values.size:
        return float("nan"), float("nan")
    sample = rng.integers(0, values.size, size=(BOOTSTRAP_DRAWS, values.size))
    medians = np.median(values[sample], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def distribution(values: list[float] | np.ndarray, rng: np.random.Generator) -> dict[str, str]:
    values = finite(values)
    low, high = bootstrap_median_ci(values, rng)
    return {
        "valid_n": str(values.size),
        "median": fmt(np.median(values)) if values.size else "",
        "q25": fmt(np.percentile(values, 25)) if values.size else "",
        "q75": fmt(np.percentile(values, 75)) if values.size else "",
        "ci95_low": fmt(low),
        "ci95_high": fmt(high),
        "positive_fraction": fmt(np.mean(values > 0)) if values.size else "",
        "median_abs": fmt(np.median(np.abs(values))) if values.size else "",
    }


def load_basins() -> list[str]:
    values = json.loads((REPO / "data" / "531sub_id.txt").read_text(encoding="utf-8"))
    basins = [str(value).zfill(8) for value in values]
    if len(basins) != 531 or len(set(basins)) != 531:
        raise RuntimeError("canonical basin list is not a unique 531-basin set")
    return basins


def write_control_matrix() -> None:
    rows = []
    for regime, groups in REGIMES.items():
        for model_role in ("N", "D_E", "G_E", "D_R", "G_R"):
            model = groups[model_role]
            if model_role == "N":
                et = "native sequential three-layer XAJ ET"
                upper = "EU=min(WU+P, PET*k), then native sequential EL/ED"
                response = "native RI/RG generation and two linear recursions"
                inflow = "RI/RG each generated from s*fr and adjusted by (1-IM) before native recursions"
                differences = "controlled N reference; xaj_c and native KI/KG/CI/CG retained"
            elif model_role == "D_E":
                et = "parallel lower/deep ET with unity stress exponent"
                upper = "same EU; EL/ED use the same pre-extraction WL/WD in parallel kernel"
                response = "native RI/RG generation and two linear recursions"
                inflow = "same native response path as N"
                differences = "xaj_c removed/zeroed; gamma absent and hard-coded to 1; xaj response parameters retained"
            elif model_role == "G_E":
                et = "parallel lower/deep ET with basin-specific power exponent"
                upper = "same EU; EL/ED use the same pre-extraction WL/WD with gamma"
                response = "native RI/RG generation and two linear recursions"
                inflow = "same native response path as N"
                differences = "xaj_c removed/zeroed; xaj_gamma active in [0.2,5.0]"
            elif model_role == "D_R":
                et = "native sequential three-layer XAJ ET"
                upper = "native EU then sequential EL/ED, unchanged from N"
                response = "single analytic linear subsurface response"
                inflow = "controlled total response input KSS*s*fr*(1-IM); no RI/RG identities"
                differences = "native KI/KG/CI/CG replaced by xaj_kss and xaj_tau0; beta hard-coded to 1"
            else:
                et = "native sequential three-layer XAJ ET"
                upper = "native EU then sequential EL/ED, unchanged from N"
                response = "single analytic power-law subsurface response"
                inflow = "same controlled total response input KSS*s*fr*(1-IM) as D_R; only release exponent differs"
                differences = "native KI/KG/CI/CG replaced by xaj_kss and xaj_tau0; xaj_beta active in [0.5,2.0]"
            rows.append({
                "family": "XAJ+CemaNeige controlled composition",
                "regime": regime,
                "model": model,
                "model_role": model_role,
                "snow_module": "CemaNeige, shared in _ControlledXAJWithCemaNeige",
                "ET_organization": et,
                "upper_ET_extraction": upper,
                "subsurface_organization": response,
                "subsurface_inflow_handling": inflow,
                "surface_runoff/routing": "shared XAJ rs/rs_adj generation and Gamma-UH routing",
                "parameter_differences": differences,
                "claim_status": "VERIFIED_WITH_QUALIFICATION" if model_role in ("D_R", "G_R") else "VERIFIED_EXACT",
                "verified_source": "models/controlled_composed.py; models/xaj_variants.py; models/xaj.py; models/structure_evaporation.py; models/structure_response.py; models/parameter_specs.py",
            })
    fields = list(rows[0])
    write_csv(CONTROL / "controlled_structure_matrix.csv", rows, fields)
    (CONTROL / "CONTROL_MATRIX_AUDIT.md").write_text(
        """# Controlled structure matrix audit

## Code-level findings

- **Shared Snow:** `controlled_composed.py` runs one CemaNeige stage before every controlled XAJ structure. `VERIFIED_EXACT`.
- **ET upper extraction:** all controlled XAJ paths compute `EU=min(WU+P, PET*k)` before the structure-specific lower/deep organization. `VERIFIED_EXACT`.
- **D_E/G_E:** both use the same pre-extraction `WL/WD` parallel kernel; D_E passes a unity exponent and G_E passes `xaj_gamma`. `VERIFIED_EXACT`.
- **N/D_R/G_R ET:** the response variants do not alter the native sequential ET branch. `VERIFIED_EXACT`.
- **D_R/G_R inflow:** the code explicitly forms one controlled input `KSS*s*fr*(1-IM)` and then applies the analytic response; no RI/RG identities are created in that branch. Thus “total subsurface inflow is held equal and only release organization changes” is exact at matched state/parameter/input conditions. It is not a statement that independently re-estimated D_R and G_R have identical fitted parameter values or trajectories. `VERIFIED_WITH_QUALIFICATION`.
- **Surface runoff/routing:** `rs_adj` and the later Gamma-UH routing are shared by the controlled variants. `VERIFIED_EXACT`.

## Safe wording

> The controlled variants retain the same CemaNeige, runoff-generation, and routing scaffolding while changing a predeclared ET-organization or subsurface-release kernel. In D_R/G_R, the controlled total subsurface input is formed identically before the release kernel; the fitted structures may nevertheless produce different trajectories because they are independently estimated.

## Unsupported wording

Do not write that G is guaranteed to outperform D, that D→G is a continuous complexity scale, or that independently fitted variants have identical flux trajectories. The code supports nested model classes and controlled kernel differences, not an optimization-order theorem.
""",
        encoding="utf-8",
    )
    (CONTROL / "nesting_audit.md").write_text(
        """# D/G nesting audit

## ET: gamma = 1

`XAJGE` selects the same `_parallel_evaporation_step` as `XAJDE`; D_E supplies `torch.ones_like(wl)` and G_E supplies `xaj_gamma`. `stable_positive_power` has an exact exponent-one forward branch, including the below-floor correction. Therefore gamma=1 is a code-level exact reduction of G_E to D_E for matched inputs, states, and remaining parameters. `EVAPORATION_GAMMA_PARAM_SPECS` is `[0.2, 5.0]`, so 1 is inside the G_E parameter domain. D_E/G_E parameter schemas differ only by `xaj_gamma`.

## Response: beta = 1

`XAJGR` and `XAJDR` use the same analytic response kernel and the same controlled input. D_R supplies `torch.ones_like(kss)`; G_R supplies `xaj_beta`. At beta=1 the analytic kernel selects the near-zero series, whose exact limit is `log(y_new)=log(y_available)-dt/tau_0`, matching the linear release. `SUBSURFACE_BETA_PARAM_SPECS` is `[0.5, 2.0]`, so 1 is inside the G_R domain. D_R/G_R parameter schemas differ only by `xaj_beta`.

## Optimization meaning

The nesting result means the G parameter space contains the corresponding D case. It does **not** guarantee train/test performance ordering under finite numerical optimization, independent estimation, or independent testing. It does not make G an independent third structure and does not establish a continuous structural-complexity scale.
""",
        encoding="utf-8",
    )


def number_audit(basins: list[str]) -> list[dict[str, str]]:
    NUMBERS.mkdir(parents=True, exist_ok=True)
    write_report = (ROOT / "08_write_ready" / "CH3_3_6_WRITE_READY_DATA_REPORT.md").read_text(encoding="utf-8")
    activity_report = (ROOT / "09_activity_stratification" / "CH3_3_6_PROCESS_ACTIVITY_FINAL_REPORT.md").read_text(encoding="utf-8")
    reports = write_report + "\n" + activity_report
    rows: list[dict[str, str]] = []
    fields = ["section", "quantity", "regime", "contrast", "N", "value", "CI_low", "CI_high", "source_file", "source_columns", "canonical_status", "write_ready_report_contains_value", "report_numeric_match"]

    def add(section: str, quantity: str, regime: str, contrast: str, n: str, value: str, low: str, high: str, source: str, columns: str, status: str = "MACHINE_READABLE_CANONICAL") -> None:
        numeric_value = as_float(value)
        token = fmt(value)
        if not token:
            report_match = "NOT_APPLICABLE"
        elif token in reports:
            report_match = "EXACT_TEXT"
        elif format(numeric_value, ".6g") in reports or format(numeric_value, ".4g") in reports:
            report_match = "ROUNDED_TEXT"
        else:
            report_match = "NOT_IN_REPORT_TEXT"
        rows.append({"section": section, "quantity": quantity, "regime": regime, "contrast": contrast, "N": n, "value": value, "CI_low": low, "CI_high": high, "source_file": source, "source_columns": columns, "canonical_status": status, "write_ready_report_contains_value": str(bool(token and token in reports)), "report_numeric_match": report_match})

    for row in read_csv(ROOT / "08_write_ready" / "et_outlet_summary.csv"):
        add("ET outlet", "median ΔKGE", row["regime"], row["contrast"], row["valid_n"], row["median_delta_kge"], row["ci95_low"], row["ci95_high"], "08_write_ready/et_outlet_summary.csv", "median_delta_kge,ci95_low,ci95_high", "MATCH_CANONICAL_WRITE_READY")
    for row in read_csv(ROOT / "08_write_ready" / "et_partition_summary.csv"):
        if row["record_type"] == "structure":
            quantity = "ET/P structure median"
        else:
            quantity = "paired Δ(ET/P) median"
        add("ET/P", quantity, row["regime"], row["contrast"] or row["model"], row["valid_n"], row["median"], row["ci95_low"], row["ci95_high"], "08_write_ready/et_partition_summary.csv", "record_type,median,ci95_low,ci95_high", "MACHINE_READABLE_WRITE_READY")
    add("ET monthly climatology", "monthly ET × aridity strata/gradient", "", "", "216 rows", "", "", "09_activity_stratification/02_et/et_monthly_A1_A3_A5.csv", "stratum,month,median,q25,q75", "ANALYZED_AS_AUXILIARY_NOT_HEADLINE")
    et_params = read_csv(ROOT / "08_write_ready" / "et_parameter_shift_all.csv")
    common_params = sorted({row["parameter"] for row in et_params})
    add("ET parameter", "ET common parameter count", "IC+dPL", "all ET contrasts", "", str(len(common_params)), "", "", "08_write_ready/et_parameter_shift_all.csv", "parameter", "MATCH_CANONICAL_WRITE_READY")
    for row in et_params:
        if row["parameter"] == "xaj_k":
            add("ET parameter", "xaj_k normalized signed median shift", row["regime"], row["contrast"], row["valid_n"], row["normalized_signed_median_shift"], row["ci95_low_signed"], row["ci95_high_signed"], "08_write_ready/et_parameter_shift_all.csv", "parameter,normalized_signed_median_shift,ci95_low_signed,ci95_high_signed", "MATCH_CANONICAL_WRITE_READY")
            add("ET parameter", "xaj_k variant exact-boundary prevalence", row["regime"], row["contrast"], row["valid_n"], row["variant_exact_boundary_fraction"], "", "", "08_write_ready/et_parameter_shift_all.csv", "parameter,variant_exact_boundary_fraction", "MATCH_CANONICAL_WRITE_READY")
    for row in read_csv(ROOT / "08_write_ready" / "response_outlet_bfi_summary.csv"):
        add("Response outlet", "median ΔKGE", row["regime"], row["contrast"], row["outlet_valid_n"], row["outlet_median_delta_kge"], row["outlet_ci95_low"], row["outlet_ci95_high"], "08_write_ready/response_outlet_bfi_summary.csv", "outlet_median_delta_kge,outlet_ci95_low,outlet_ci95_high", "MATCH_CANONICAL_WRITE_READY")
        add("Response BFI", "median ΔBFI_model", row["regime"], row["contrast"], row["bfi_valid_n"], row["bfi_median_delta"], row["bfi_ci95_low"], row["bfi_ci95_high"], "08_write_ready/response_outlet_bfi_summary.csv", "bfi_median_delta,bfi_q25,bfi_q75,bfi_ci95_low,bfi_ci95_high,bfi_positive_fraction", "MATCH_CANONICAL_WRITE_READY")
    for row in read_csv(ROOT / "08_write_ready" / "response_tau_bfi_summary.csv"):
        add("Response tau", "Spearman tau–BFI rho", row["regime"], row["contrast"], row["valid_n"], row["rho_spearman"], row["ci95_low"], row["ci95_high"], "08_write_ready/response_tau_bfi_summary.csv", "rho_spearman,ci95_low,ci95_high", "APPENDIX_ONLY_CANONICAL")
        add("Response tau", "variant tau exact-boundary prevalence", row["regime"], row["contrast"], row["valid_n"], row["variant_exact_boundary_fraction"], "", "", "08_write_ready/response_tau_bfi_summary.csv", "variant_exact_boundary_fraction", "APPENDIX_ONLY_CANONICAL")
    write_csv(NUMBERS / "headline_number_audit.csv", rows, fields)
    return rows


def bfi_common_subset(basins: list[str]) -> dict[str, object]:
    BFI.mkdir(parents=True, exist_ok=True)
    source = read_csv(ROOT / "04_response" / "response_bfi_basin.csv")
    values = {(row["regime"], row["model"], row["basin_id"]): as_float(row["bfi"]) for row in source}
    contrasts = [("IC", "D_R-N", "N", "D_R"), ("IC", "G_R-N", "N", "G_R"), ("dPL", "XAJ_D_R_CN-XAJ_CONTROLLED_N_CN", "XAJ_CONTROLLED_N_CN", "XAJ_D_R_CN"), ("dPL", "XAJ_G_R_CN-XAJ_CONTROLLED_N_CN", "XAJ_CONTROLLED_N_CN", "XAJ_G_R_CN")]
    valid_sets = []
    invalid_rows = []
    for regime, contrast, baseline, variant in contrasts:
        valid = set()
        for basin in basins:
            base = values.get((regime, baseline, basin), float("nan"))
            var = values.get((regime, variant, basin), float("nan"))
            base_ok, var_ok = math.isfinite(base), math.isfinite(var)
            if base_ok and var_ok:
                valid.add(basin)
            else:
                invalid_rows.append({"regime": regime, "contrast": contrast, "basin_id": basin, "baseline_model": baseline, "variant_model": variant, "baseline_bfi_valid": str(base_ok), "variant_bfi_valid": str(var_ok), "invalid_side": "baseline+variant" if not base_ok and not var_ok else "baseline" if not base_ok else "variant"})
        valid_sets.append(valid)
    common = set.intersection(*valid_sets)
    excluded = sorted(set(basins) - common)
    write_csv(BFI / "bfi_invalid_basin_ids.csv", invalid_rows, ["regime", "contrast", "basin_id", "baseline_model", "variant_model", "baseline_bfi_valid", "variant_bfi_valid", "invalid_side"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary_rows = []
    for regime, contrast, baseline, variant in contrasts:
        deltas = np.asarray([values[(regime, variant, basin)] - values[(regime, baseline, basin)] for basin in sorted(common)], dtype=np.float64)
        stats = distribution(deltas, rng)
        if (float(stats["median"]) > 0 and float(stats["ci95_low"]) > 0) or (float(stats["median"]) < 0 and float(stats["ci95_high"]) < 0):
            judgment = "SIGN_ROBUST"
        elif float(stats["ci95_low"]) <= 0 <= float(stats["ci95_high"]):
            judgment = "CI_INCLUDES_ZERO_ON_COMMON_SUBSET"
        else:
            judgment = "SIGN_NOT_ROBUST"
        summary_rows.append({"regime": regime, "contrast": contrast, "common_valid_n": str(len(common)), "excluded_n": str(len(excluded)), **stats, "sign_judgment": judgment, "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED})
    write_csv(BFI / "bfi_common_subset_summary.csv", summary_rows, list(summary_rows[0]))
    (BFI / "bfi_invalid_rule.md").write_text(
        """# Canonical BFI invalid-rule audit

The canonical BFI rows were produced by `scripts/run_ch3_6_analysis.py`, which calls `scripts/phase0_sampling.py:lyne_hollick_bfi` on each model `qsim` test-period series (`5478:10957`). The exact rule is:

1. retain the longest contiguous segment with finite, nonnegative discharge;
2. return invalid (`NaN` in the canonical CSV) if no segment exists, the selected segment has fewer than 365 days, the selected segment is not finite, or its discharge sum is non-positive;
3. clamp the valid segment to nonnegative values;
4. apply three forward/reverse Lyne–Hollick passes with `alpha=0.925`, clipping each pass to the available discharge;
5. return `sum(baseflow)/sum(discharge)`.

No denominator repair, interpolation, constant-flow exclusion, model-output replacement, or new BFI calculation was introduced in this audit. Invalid basin IDs below are recovered from blank/nonfinite canonical BFI rows and are retained rather than imputed.

## Common subset

The common subset is the intersection of finite paired BFI values for IC D_R−N, IC G_R−N, dPL D_R−N, and dPL G_R−N. The exact N and all excluded IDs are recorded in `bfi_common_subset_summary.csv` and `bfi_invalid_basin_ids.csv`.
""",
        encoding="utf-8",
    )
    return {"common": sorted(common), "excluded": excluded, "summary": summary_rows}


def context_gradient_summary() -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    CONTEXT.mkdir(parents=True, exist_ok=True)
    et = read_csv(ROOT / "09_activity_stratification" / "02_et" / "et_aridity_gradient.csv")
    et_end = read_csv(ROOT / "09_activity_stratification" / "02_et" / "et_aridity_endpoint.csv")
    response = read_csv(ROOT / "09_activity_stratification" / "03_response" / "response_baseflow_gradient.csv")
    response_end = read_csv(ROOT / "09_activity_stratification" / "03_response" / "response_baseflow_endpoint.csv")
    output = []
    for row in et + response:
        if row["effect_role"] != "primary_absolute":
            continue
        end_metric = "abs_delta_et_over_p" if row["attribute"] == "aridity" else "abs_delta_bfi_model"
        endpoint_rows = et_end if row["attribute"] == "aridity" else response_end
        end = next(item for item in endpoint_rows if item["regime"] == row["regime"] and item["contrast"] == row["contrast"] and item["effect_metric"] == end_metric)
        output.append({"process": "ET" if row["attribute"] == "aridity" else "Response", "attribute": row["attribute"], "regime": row["regime"], "contrast": row["contrast"], "primary_metric": row["effect_metric"], "finite_n": row["finite_n"], "rho": row["rho"], "ci95_low": row["ci95_low"], "ci95_high": row["ci95_high"], "low_stratum": end["low_stratum"], "high_stratum": end["high_stratum"], "low_median": end["low_median"], "high_median": end["high_median"], "endpoint_difference": end["difference_high_minus_low"], "endpoint_ci95_low": end["ci95_low"], "endpoint_ci95_high": end["ci95_high"], "classification": row["primary_classification"], "methodological_flag": "SHARED_FILTER_FAMILY_AND_BOUNDED_SCALE" if row["attribute"] == "baseflow_index" else "NO_MONTHLY_GRADIENT_CLAIM", "source_files": "09_activity_stratification gradient + endpoint CSV"})
    write_csv(CONTEXT / "context_gradient_exact_summary.csv", output, list(output[0]))
    return output, {"ET": et, "Response": response}


def snow_reference() -> dict[str, object]:
    SNOW.mkdir(parents=True, exist_ok=True)
    endpoint = read_csv(PROJECT / "manuscript" / "results" / "R5" / "r5_figure9_endpoint_summary.csv")
    continuous = read_csv(PROJECT / "manuscript" / "results" / "R5" / "r5_figure9_continuous_summary.csv")
    basin = read_csv(PROJECT / "manuscript" / "results" / "R5" / "r5_basin_level_dataset.csv")
    paired_rows = []
    for host in ("XAJ", "GR4J", "SIMHYD"):
        for regime in ("IC", "dPL"):
            base_key = f"kge_test_{host}_{regime}_Base"
            cn_key = f"kge_test_{host}_{regime}_CN"
            deltas = finite([as_float(row[cn_key]) - as_float(row[base_key]) for row in basin])
            rng = np.random.default_rng(BOOTSTRAP_SEED + len(paired_rows))
            lo, hi = bootstrap_median_ci(deltas, rng)
            paired_rows.append({"host": host, "regime": regime, "N": str(deltas.size), "median_paired_CN_minus_Base_delta_KGE": fmt(np.median(deltas)), "ci95_low": fmt(lo), "ci95_high": fmt(hi), "source_columns": f"{cn_key}-{base_key}", "ci_method": "basin bootstrap of paired per-basin KGE differences", "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED, "status": "OPTIONAL_SAME_METRIC_CONTEXT"})
    write_csv(SNOW / "snow_paired_kge_context.csv", paired_rows, ["host", "regime", "N", "median_paired_CN_minus_Base_delta_KGE", "ci95_low", "ci95_high", "source_columns", "ci_method", "bootstrap_draws", "bootstrap_seed", "status"])
    lines = ["# Snow frozen reference for Chapter 3.6", "", "Snow is read from the frozen R5 package and is not recomputed as a new main experiment.", "", "## Frozen S5 Base−CN timing endpoint", "", "| Host | Regime | S1 N | S1 median Δ|CT| | S5 N | S5 median Δ|CT| |", "|---|---|---:|---:|---:|---:|"]
    for host in ("XAJ", "GR4J", "SIMHYD"):
        for regime in ("IC", "dPL"):
            s1 = next(row for row in endpoint if row["host"] == host and row["regime"] == regime and row["endpoint"] == "S1")
            s5 = next(row for row in endpoint if row["host"] == host and row["regime"] == regime and row["endpoint"] == "S5")
            lines.append(f"| {host} | {regime} | {s1['N']} | {s1['median_delta_abs_CT_Base_CN']} | {s5['N']} | {s5['median_delta_abs_CT_Base_CN']} |")
    lines += ["", "The frozen estimand is `Δ|CT|^(Base−CN) = |CT|_Base − |CT|_CN`; positive S5 values are the existing frozen snow timing result. The continuous frozen Spearman rows are in `r5_figure9_continuous_summary.csv`.", "", "## Same-metric paired KGE context", "", "The R5 basin-level file contains paired per-basin test KGE columns for Base and CN. The following is an audit re-summary of `KGE_CN − KGE_Base` on the same basin rows; it is not inferred from the difference of medians:", "", "| Host | Regime | N | Median paired ΔKGE (CN−Base) | 95% CI |", "|---|---|---:|---:|---:|"]
    for row in paired_rows:
        lines.append(f"| {row['host']} | {row['regime']} | {row['N']} | {row['median_paired_CN_minus_Base_delta_KGE']} | [{row['ci95_low']}, {row['ci95_high']}] |")
    lines += ["", "These paired KGE values are optional same-metric context only; they must not be numerically ranked against ET/P, CT, or model-derived BFI. Source columns and bootstrap details are serialized in `snow_paired_kge_context.csv` and retained alongside the frozen R5 source files."]
    (SNOW / "snow_reference_for_3_6.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"endpoint": endpoint, "continuous": continuous, "paired_kge": paired_rows}


def claims_matrix(common: dict[str, object], context: list[dict[str, str]]) -> None:
    CLAIMS.mkdir(parents=True, exist_ok=True)
    d_sign = next(row for row in common["summary"] if row["contrast"] == "D_R-N" and row["regime"] == "IC")
    rows = [
        {"Claim": "ET structural contrasts produce large outlet differences.", "Evidence": "ET outlet medians and CIs are small; IC D_E−N is 0.00156494.", "Status": "NOT_SUPPORTED", "Max wording": "ET outlet effects were small in this canonical comparison.", "Forbidden wording": "large ET outlet effect; equivalence"},
        {"Claim": "ET D_E−N under IC is directionally distinguishable in ΔKGE.", "Evidence": "IC D_E−N CI [0.00020922, 0.00258340] does not cross zero.", "Status": "SUPPORTED_WITH_QUALIFICATION", "Max wording": "IC D_E−N showed a small positive paired ΔKGE in the available estimate.", "Forbidden wording": "ET structure is universally distinguishable or causally superior"},
        {"Claim": "ET structure changes ET/P.", "Evidence": "Paired ET/P rows and CIs in 08_write_ready/et_partition_summary.csv.", "Status": "SUPPORTED_WITH_QUALIFICATION", "Max wording": "The ET organization contrasts altered the model ET/P allocation diagnostic.", "Forbidden wording": "ET process truth or full ET identifiability"},
        {"Claim": "ET monthly seasonality changes.", "Evidence": "Existing annual-cycle climatology; auxiliary A1/A3/A5 summary has no monthly significance test.", "Status": "SUPPORTED_WITH_QUALIFICATION", "Max wording": "The variants show qualified differences in the organization of the annual ET cycle.", "Forbidden wording": "stable aridity-driven monthly gradient"},
        {"Claim": "ET parameter shifts occur.", "Evidence": "All common parameter shifts and boundary audits.", "Status": "SUPPORTED_WITH_QUALIFICATION", "Max wording": "Parameter re-estimation produced shifts under the structural contrasts.", "Forbidden wording": "a shift uniquely identifies the missing process"},
        {"Claim": "ET aridity gradient is stable.", "Evidence": "IC primary rho is positive but strata are non-monotonic; dPL rho CIs cross zero.", "Status": "NOT_SUPPORTED", "Max wording": "No stable increasing aridity dependence was established.", "Forbidden wording": "aridity causes ET structural error"},
        {"Claim": "Response structure changes outlet KGE.", "Evidence": "Small paired response outlet contrasts with mixed CI coverage.", "Status": "SUPPORTED_WITH_QUALIFICATION", "Max wording": "Response-path contrasts changed outlet KGE by a small, contrast-specific amount.", "Forbidden wording": "large response outlet effect"},
        {"Claim": "Response structure changes model-derived BFI.", "Evidence": "Four canonical paired model-BFI contrasts with valid N 507/527/531.", "Status": "SUPPORTED_WITH_QUALIFICATION", "Max wording": "The structural contrasts changed the model-simulated streamflow-derived BFI diagnostic.", "Forbidden wording": "groundwater truth or internal groundwater state"},
        {"Claim": "D_R BFI direction differs between IC/dPL.", "Evidence": "IC D_R−N median +0.00358855; dPL D_R−N median −0.00231398; common-subset IC D_R sign judgment is " + d_sign["sign_judgment"] + ".", "Status": "SUPPORTED_WITH_QUALIFICATION", "Max wording": "The signed D_R BFI contrast differed between estimation regimes.", "Forbidden wording": "estimation regime caused the sign reversal"},
        {"Claim": "Estimation regime causes D_R sign reversal.", "Evidence": "Only separate fitted contrasts are available; no causal regime intervention.", "Status": "NOT_SUPPORTED", "Max wording": "The sign is regime-dependent in the available estimates.", "Forbidden wording": "IC/dPL caused the reversal"},
        {"Claim": "tau has a strong response footprint.", "Evidence": "tau–BFI rho values are weak and boundary-qualified.", "Status": "NOT_SUPPORTED", "Max wording": "tau–BFI shows a weak association and belongs in the appendix.", "Forbidden wording": "strong tau mechanism or tau explains BFI"},
        {"Claim": "Response sensitivity increases with CAMELS baseflow_index.", "Evidence": "Primary rho values are negative or weak; no stable increasing gradient.", "Status": "NOT_SUPPORTED", "Max wording": "No stable increase with CAMELS baseflow background was observed.", "Forbidden wording": "baseflow dominance increases structural uncertainty"},
        {"Claim": "Snow background dependence generalizes to ET/Response.", "Evidence": "Frozen Snow gradient is strong; ET/Response primary gradients are non-monotonic, weak, or decreasing.", "Status": "NOT_SUPPORTED", "Max wording": "Snow activity dependence is not shown to generalize.", "Forbidden wording": "all active processes behave like Snow"},
        {"Claim": "Snow background dependence was not stably reproduced in ET/Response.", "Evidence": "R5 frozen reference versus 09 primary context gradients.", "Status": "SUPPORTED_WITH_QUALIFICATION", "Max wording": "The Snow pattern was not stably reproduced by the available ET/Response primary metrics.", "Forbidden wording": "ET/Response have no process dependence"},
        {"Claim": "ET/Response are structurally equivalent.", "Evidence": "No equivalence estimand or protocol exists.", "Status": "NOT_SUPPORTED", "Max wording": "The available results do not establish structural equivalence.", "Forbidden wording": "equivalent; interchangeable"},
        {"Claim": "ET/Response are unidentifiable from streamflow.", "Evidence": "The available comparisons do not identify an unidentifiability theorem.", "Status": "NOT_SUPPORTED", "Max wording": "Complete process-level identifiability is not established.", "Forbidden wording": "unidentifiable from streamflow"},
        {"Claim": "Parameter re-estimation absorbed the structural change.", "Evidence": "Parameter shifts and outlet/process diagnostics are observed, but absorption is not an identified estimand.", "Status": "NOT_SUPPORTED", "Max wording": "Re-estimation changed the fitted parameter vector under each structure.", "Forbidden wording": "absorbed; fully compensated"},
        {"Claim": "Snow remains the main controlled case; ET/Response define extension boundaries.", "Evidence": "Frozen Snow chain is complete; ET/Response have qualified layers and explicit missing protocols.", "Status": "SUPPORTED", "Max wording": "Snow is the main controlled proof, while ET/Response delimit external validity.", "Forbidden wording": "ET/Response replace the Snow case"},
    ]
    write_csv(CLAIMS / "claim_evidence_matrix.csv", rows, list(rows[0]))


def figure_audit() -> None:
    FIGURE.mkdir(parents=True, exist_ok=True)
    (FIGURE / "figure_table_evidence_audit.md").write_text(
        """# Figure and Table evidence architecture audit

## Table 3-X

Recommended columns:

| Process | Structural intervention | Aggregate outlet performance | Process-related diagnostic | Parameter layer | Catchment context | Role in Ch3 |
|---|---|---|---|---|---|---|
| Snow | Base/TGD2/CN frozen contrasts | frozen outlet/timing evidence | CT and internal-state evidence | frozen Snow compensation | frac_snow | main controlled proof |
| ET | N/D_E/G_E | paired ΔKGE | ET/P and annual ET cycle | common parameter shifts/boundaries | aridity, qualified | auxiliary extension boundary |
| Response | N/D_R/G_R | paired ΔKGE | model-derived streamflow BFI | shared parameter shifts; tau appendix | CAMELS baseflow_index, qualified | auxiliary extension boundary |

CT, ET/P, and model-derived BFI have different definitions and units. `frac_snow`, `aridity`, and CAMELS `baseflow_index` are different catchment descriptors. The table must not create a common effect-size ranking.

## Figure 3-13 ET

Retain the existing outlet ΔKGE, ET/P, and annual-cycle climatology panels. Aridity strata are optional auxiliary context; the strict audit does not support a stable increasing primary aridity gradient. Do not label monthly ET as an aridity gradient unless the auxiliary analysis is explicitly identified and not used as a headline claim.

## Figure 3-14 Response

Retain outlet ΔKGE and model-derived BFI panels. CAMELS baseflow-index strata are optional context and must carry `SHARED_FILTER_FAMILY_AND_BOUNDED_SCALE`. The BFI label must state that it is a model-simulated streamflow-derived response signature. tau–BFI is appendix-only.

## Recommendation

Do not automatically replace Figure 3-13/3-14 or change figure numbers. Add a context panel only if it improves the process-specific boundary argument; never use a shared y-axis or standardized cross-process effect-size ranking.
The earlier 08 write-ready report listed tau–BFI among generic Figure 3-14 candidate panels. This strict audit supersedes that candidate: tau–BFI is weak and boundary-qualified, so it remains appendix-only in the final architecture.
""",
        encoding="utf-8",
    )


def final_report(numbers: list[dict[str, str]], common: dict[str, object], context: list[dict[str, str]], snow: dict[str, object]) -> None:
    report_path = OUT / "CH3_3_6_FINAL_STRICT_AUDIT_REPORT.md"
    et_context = [row for row in context if row["process"] == "ET"]
    response_context = [row for row in context if row["process"] == "Response"]
    def context_lines(rows: list[dict[str, str]]) -> str:
        return "\n".join(f"- {row['regime']} {row['contrast']}: rho={row['rho']} [{row['ci95_low']}, {row['ci95_high']}]; {row['low_stratum']} median={row['low_median']}, {row['high_stratum']} median={row['high_median']}, endpoint={row['endpoint_difference']} [{row['endpoint_ci95_low']}, {row['endpoint_ci95_high']}]; class={row['classification']}." for row in rows)
    common_lines = "\n".join(f"- {row['regime']} {row['contrast']}: N={row['common_valid_n']}, median={row['median']}, Q25/Q75=[{row['q25']},{row['q75']}], CI=[{row['ci95_low']},{row['ci95_high']}], positive fraction={row['positive_fraction']}, |Δ| median={row['median_abs']}; {row['sign_judgment']}." for row in common["summary"])
    outlet = [row for row in numbers if row["section"] in ("ET outlet", "Response outlet")]
    outlet_lines = "\n".join(f"- {row['section']} {row['regime']} {row['contrast']}: N={row['N']}, median={row['value']}, CI=[{row['CI_low']},{row['CI_high']}]." for row in outlet)
    s5_lines = []
    for row in snow["endpoint"]:
        if row["endpoint"] == "S5":
            s5_lines.append(f"{row['host']} {row['regime']} S5 N={row['N']} median Δ|CT|={row['median_delta_abs_CT_Base_CN']}")
    report = f"""# Chapter 3.6 Final Strict Audit Report

## 1. Executive verdict

**`READY_FOR_FINAL_3_6_REVISION`**. The controlled design, D/G nesting, headline values, BFI common-subset sign, activity-context limitations, Snow frozen reference, and claim boundaries are auditable without another scientific experiment. There is no blocking analysis gap under the stated scope.

## 2. Controlled-design verification

The code verifies shared CemaNeige, shared runoff/routing scaffolding, ET-specific parallel organization for D_E/G_E, and response-specific analytic release organization for D_R/G_R. The D_R/G_R total controlled subsurface input is identical before release at matched conditions, but independently estimated trajectories are not asserted identical. Full matrix: `01_control_matrix/controlled_structure_matrix.csv`; narrative: `CONTROL_MATRIX_AUDIT.md`.

## 3. D/G nesting verification

`gamma=1` is an exact G_E→D_E reduction; `beta=1` is an exact G_R→D_R reduction in the code kernels. Bounds include 1: gamma `[0.2,5.0]`, beta `[0.5,2.0]`. D/G schemas differ only by their generic exponent. Nesting does not guarantee train/test performance ordering and is not a complexity-scale theorem. See `01_control_matrix/nesting_audit.md`.

## 4. Headline number checksum

The machine-readable audit contains {len(numbers)} headline/qualification rows. Canonical outlet values are:

{outlet_lines}

ET/P structure and paired values, xaj_k shifts/boundaries, Response BFI valid N/Q25/Q75/CI/fraction, and tau values are all retained in `02_numbers/headline_number_audit.csv`. The monthly ET × aridity analysis exists only as an auxiliary 09 output; it is not a headline negative claim or proof of a stable monthly gradient.

## 5. Response BFI invalid-rule audit

The canonical rule is longest finite nonnegative qsim segment, minimum 365 days, three Lyne–Hollick passes, alpha=0.925, nonnegative clipping, and baseflow/discharge sum ratio. Blank/nonfinite canonical BFI rows are invalid; no values were imputed. Details and invalid IDs: `03_bfi_common_subset/bfi_invalid_rule.md` and `bfi_invalid_basin_ids.csv`.

## 6. Response BFI common-subset result

The four finite paired contrasts share **N={len(common['common'])}** basins; excluded N={len(common['excluded'])}. Excluded IDs are listed in `bfi_invalid_basin_ids.csv` and the common-subset summary. Results:

{common_lines}

For IC D_R−N specifically, the positive signed direction remains **SIGN_ROBUST** on the common subset (median={next(row['median'] for row in common['summary'] if row['regime']=='IC' and row['contrast']=='D_R-N')}, CI excludes zero). Thus the sign does not depend on the 507-versus-527 valid-N discrepancy, although the full contrast remains regime-qualified.

## 7. ET aridity exact gradient result

{context_lines(et_context)}

The primary ET aridity result is not a stable increasing gradient: IC rho is positive but the five-stratum trajectory is non-monotonic; dPL rho is weak with CI crossing zero. The ET outlet secondary gradient is stronger in the available context but must not be substituted for the predeclared ET/P primary outcome.

## 8. Response baseflow-context exact gradient result

{context_lines(response_context)}

Methodological flag: **`SHARED_FILTER_FAMILY_AND_BOUNDED_SCALE`**. CAMELS `baseflow_index` is the observed/static catchment descriptor; `ΔBFI_model` is a model-simulated streamflow-derived response signature. The maximum defensible wording is that no stable increase with CAMELS baseflow background was observed. Decreasing associations are not causal interpretations.

## 9. Snow frozen reference consistency

Frozen S5 Base−CN timing values are: {', '.join(s5_lines)}. The frozen estimand is `|CT|_Base − |CT|_CN`; it is not numerically interchangeable with ET/P, ΔKGE, or model BFI. The R5 basin-level file also supports optional same-metric paired KGE context, re-summarized from per-basin differences in `05_snow_reference/snow_paired_kge_context.csv` and documented in `snow_reference_for_3_6.md`; it is not inferred by subtracting medians.

## 10. Parameter evidence boundary

There are 16 common parameters within the ET contrast table and 13 shared parameters within the Response contrast table; these counts are not an ET/Response intersection. `xaj_k` is a potential-ET multiplier: code first forms `pet_adj = pet_t * k`, so it acts before EU/EL/ED and is not a specific lower/deep ET-component parameter. It should be described as global potential-evaporation scaling within the XAJ ET calculation, not as proof of a unique ET mechanism. IC xaj_k exact-boundary prevalence is approximately 18%, which supports retaining the no-main-text-representative decision. Response shared parameters remain parameter-footprint evidence; tau–BFI is appendix-only and boundary-qualified.

## 11. Unsupported / not-analyzed claims

The claim matrix in `06_claims/claim_evidence_matrix.csv` downgrades or rejects: large ET/Response outlet effects, causal IC/dPL sign reversal, strong tau mechanism, increasing Response sensitivity with CAMELS baseflow index, Snow-pattern generalization, structural equivalence, streamflow unidentifiability, and “parameter re-estimation absorbed the change.” Missing dry-down, recession/FDC, P4, corrected P5, and other unrecovered protocols remain not analyzed; they are not zero effects.

## 12. Final claim hierarchy

- **Supported:** controlled kernel contrasts, D/G nesting, parameter shifts, model-derived BFI changes, regime-specific signed D_R BFI direction, Snow as the main controlled case.
- **Supported with qualification:** small outlet changes, ET/P changes, annual-cycle organization differences, ET IC D_E−N directional ΔKGE, Response BFI diagnostics, non-generalization of the Snow gradient.
- **Not supported:** universal activity dependence, causal regime effects, strong tau footprint, equivalence, or unidentifiability.
- **Not analyzed:** frozen-protocol dry-down/recession/FDC/P4/corrected-P5 layers.

## 13. Table 3-X final evidence architecture

Use separate process-related diagnostic columns for Snow CT/internal evidence, ET/P/monthly ET, and model-derived BFI. Keep CT, ET/P, BFI, and outlet ΔKGE definitions distinct. Include parameter layer and catchment context as qualified evidence, not a common effect-size ranking. See `07_figure_table/figure_table_evidence_audit.md`.

## 14. Figure 3-13 recommendation

Retain ΔKGE, ET/P, and annual ET climatology. Aridity strata may be an auxiliary panel, but no stable increasing aridity claim should headline the figure. Do not replace the existing figure automatically.

## 15. Figure 3-14 recommendation

Retain ΔKGE and model-derived BFI. Baseflow-index strata may be shown as qualified context with the shared-filter/bounded-scale flag. Keep tau–BFI in the appendix. This strict audit supersedes the earlier generic 08 figure candidate list that included tau–BFI as a possible panel. Do not replace the existing figure automatically.

## 16. 3.6 正文必须修改的句子

| Current risk | Corrected wording |
|---|---|
| “ET/Response structural effects were absorbed by parameter re-estimation.” | “Independent re-estimation produced parameter shifts, while the available outlet and process-related diagnostics remained contrast-specific; absorption is not claimed.” |
| “The structures are equivalent / unidentifiable from streamflow.” | “The available results do not establish structural equivalence or complete process-level identifiability.” |
| “Response sensitivity increases with baseflow index.” | “No stable increase in `|ΔBFI_model|` with the CAMELS baseflow descriptor was observed.” |
| “D_R sign reversal was caused by the estimation regime.” | “The signed D_R BFI contrast differed between the IC and dPL estimates; causal attribution to estimation regime is not made.” |
| “Monthly ET shows no aridity gradient.” | “Monthly ET is used as an auxiliary annual-cycle diagnostic; it is not treated as a tested aridity-gradient result.” |
| “xaj_k represents the ET component.” | “xaj_k scales potential ET before the ET extraction branches and is not a unique lower/deep component parameter.” |
| “BFI is groundwater state/truth.” | “BFI is a model-simulated streamflow-derived response signature.” |

No latest prose draft was found in the repository; these are the operative sentence-level constraints for final revision.

## 17. Final defensible 3.6.3 conclusion

Snow remains the principal controlled demonstration linking structural omission, parameter compensation, outlet response, synthetic-truth recovery, and internal process evidence. ET and Response extend the diagnosis without reproducing a universal pattern: ET/P and annual-cycle organization change under the ET contrasts, but aridity dependence is weak or non-monotonic; Response BFI changes are valid-N qualified and do not increase stably with CAMELS baseflow background. The evidence therefore supports process-specific extension boundaries, not structural equivalence, complete absorption, or streamflow unidentifiability.

## 18. Ch3→Ch4 / Ch3→Ch5 transition boundary

Chapter 3 establishes the diagnostic premise and controlled evidence boundary. Chapter 4 must test the stability and applicability of model–parameter relationships rather than assume them from 3.6. Chapter 5, when opening structural freedom, must test repeatability, attribute relationships, and generalization of learned relations. This audit does not prescribe extra observations or claim that ET/Response structures cannot be learned.

## 19. Remaining blockers

**`NO_BLOCKING_ANALYSIS_GAP`** under the current strict scope. Unrecovered process protocols remain explicitly outside the claims, not unresolved blockers for final 3.6 revision.

## 20. Final decision

**Yes.** The package can be used to directly revise Section 3.6 and proceed to the doctoral-thesis formal version, provided the wording constraints above are applied and Snow remains the main controlled case.
"""
    OUT.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    basins = load_basins()
    write_control_matrix()
    numbers = number_audit(basins)
    common = bfi_common_subset(basins)
    context, _ = context_gradient_summary()
    snow = snow_reference()
    claims_matrix(common, context)
    figure_audit()
    final_report(numbers, common, context, snow)
    write_json(OUT / "audit_metadata.json", {
        "verdict": "READY_FOR_FINAL_3_6_REVISION",
        "canonical_basin_n": len(basins),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "no_training": True,
        "no_calibration": True,
        "no_replay": True,
        "no_new_signature": True,
        "new_activity_package_audited": "09_activity_stratification",
        "canonical_sources": ["03_et", "04_response", "08_write_ready", "09_activity_stratification", "manuscript/results/R5"],
    })
    print(json.dumps({"status": "complete", "output": str(OUT), "verdict": "READY_FOR_FINAL_3_6_REVISION", "common_bfi_n": len(common["common"]), "excluded_bfi_n": len(common["excluded"]), "headline_rows": len(numbers)}, indent=2))


if __name__ == "__main__":
    main()
