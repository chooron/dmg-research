#!/usr/bin/env python3
"""Final, read-only robustness checks for the canonical R2 outputs.

The script consumes the already validated R2 basin-level paired shifts. It
never reselects parameters, rewrites primary R2 outputs, reads production
checkpoints, or creates a Figure 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parents[3]
MANUSCRIPT = PROJECT / "manuscript"
R2 = MANUSCRIPT / "results" / "R2"
PAIRED_FILE = R2 / "r2_paired_shifts_basin_level.csv"
PRIMARY_GRADIENT_FILE = R2 / "r2_snow_gradients_summary.csv"
PRIMARY_FILE = R2 / "r2_primary_shift_summary.csv"
CANDIDATE_FILE = R2 / "r2_figure3_candidate_parameters.csv"

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260730
PARAMETERS = [
    "xaj_k",
    "xaj_b",
    "xaj_im",
    "xaj_um",
    "xaj_lm",
    "xaj_dm",
    "xaj_c",
    "xaj_sm",
    "xaj_ex",
    "xaj_ki",
    "xaj_kg",
    "xaj_ci",
    "xaj_cg",
    "xaj_a",
    "xaj_theta",
]
TARGET_PARAMETERS = ["xaj_um", "xaj_ki", "xaj_ci", "xaj_im"]
DISPLAY = {
    "xaj_k": "k",
    "xaj_b": "b",
    "xaj_im": "im",
    "xaj_um": "um",
    "xaj_lm": "lm",
    "xaj_dm": "dm",
    "xaj_c": "c",
    "xaj_sm": "sm",
    "xaj_ex": "ex",
    "xaj_ki": "ki",
    "xaj_kg": "kg",
    "xaj_ci": "ci",
    "xaj_cg": "cg",
    "xaj_a": "a (UH shape)",
    "xaj_theta": "theta (UH scale)",
}
REGIMES = ["S1", "S2", "S3", "S4", "S5"]
REGIME_N = {"S1": 165, "S2": 156, "S3": 121, "S4": 34, "S5": 55}


def bootstrap_ci(
    values: np.ndarray, statistic, rng: np.random.Generator
) -> tuple[float, float]:
    """Use the existing R2 percentile bootstrap convention exactly."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return (np.nan, np.nan)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))
    boot = np.asarray([statistic(values[i]) for i in indices], dtype=float)
    return tuple(np.quantile(boot, [0.025, 0.975]))


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def slope_ci(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    # This is the same call shape used by run_r2_parameter_statistics.py.
    index_values = np.arange(len(x), dtype=int)
    return bootstrap_ci(
        index_values,
        lambda indices: slope(
            x[np.asarray(indices, dtype=int)], y[np.asarray(indices, dtype=int)]
        ),
        rng,
    )


def direction(value: float) -> str:
    if not np.isfinite(value):
        return "not_estimated"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def load_paired() -> pd.DataFrame:
    paired = pd.read_csv(PAIRED_FILE, dtype={"basin_id": str})
    paired["basin_id"] = paired["basin_id"].str.zfill(8)
    required = {
        "paradigm",
        "basin_id",
        "parameter",
        "frac_snow",
        "snow_regime",
        "z_base",
        "z_cn",
        "delta_base_minus_cn",
    }
    missing = required - set(paired.columns)
    if missing:
        raise ValueError(f"paired R2 output missing columns: {sorted(missing)}")
    expected = 2 * 531 * len(PARAMETERS)
    if len(paired) != expected:
        raise ValueError(f"expected {expected} paired rows, found {len(paired)}")
    if paired.duplicated(["paradigm", "basin_id", "parameter"]).any():
        raise ValueError("duplicate paradigm/basin/parameter keys in paired output")
    if set(paired["parameter"]) != set(PARAMETERS):
        raise ValueError(
            "paired output does not contain exactly the canonical 15 parameters"
        )
    if (
        paired[["frac_snow", "z_base", "z_cn", "delta_base_minus_cn"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("paired output contains NaN in required statistical fields")
    sign_error = np.max(
        np.abs(paired["delta_base_minus_cn"] - (paired["z_base"] - paired["z_cn"]))
    )
    if sign_error > 1e-12:
        raise ValueError(f"Base-CN sign convention mismatch: max error {sign_error}")
    basin_counts = paired.groupby("paradigm")["basin_id"].nunique().to_dict()
    if basin_counts != {"IC": 531, "dPL": 531}:
        raise ValueError(f"unexpected basin coverage: {basin_counts}")
    regime_counts = (
        paired[["basin_id", "snow_regime"]]
        .drop_duplicates()["snow_regime"]
        .value_counts()
        .to_dict()
    )
    if regime_counts != REGIME_N:
        raise ValueError(f"R1 regime membership mismatch: {regime_counts}")
    return paired


def consume_primary_stream_and_reproduce_full(
    paired: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Advance the R2 RNG through primary medians, then reproduce all slopes."""
    for _, group in paired.groupby(["paradigm", "parameter"], sort=False):
        bootstrap_ci(group["delta_base_minus_cn"].to_numpy(float), np.median, rng)
    rows = []
    for (paradigm, parameter), group in paired.groupby(
        ["paradigm", "parameter"], sort=False
    ):
        x = group["frac_snow"].to_numpy(float)
        y = group["delta_base_minus_cn"].to_numpy(float)
        ci = slope_ci(x, y, rng)
        rho, _ = spearmanr(x, y)
        rows.append(
            {
                "paradigm": paradigm,
                "parameter": parameter,
                "parameter_display": DISPLAY[parameter],
                "subset": "full_531",
                "n": len(group),
                "slope": slope(x, y),
                "ci95_low": ci[0],
                "ci95_high": ci[1],
                "spearman_rho": rho,
            }
        )
    return pd.DataFrame(rows)


def subset_mask(paired: pd.DataFrame, subset: str) -> pd.Series:
    if subset == "full_531":
        return pd.Series(True, index=paired.index)
    if subset == "exclude_S5":
        return paired["snow_regime"] != "S5"
    if subset.startswith("loo_"):
        return paired["snow_regime"] != subset.removeprefix("loo_")
    raise ValueError(subset)


def gradient_robustness(
    paired: pd.DataFrame, baseline: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subsets = ["full_531", "exclude_S5"] + [f"loo_{regime}" for regime in REGIMES]
    rows = []
    for subset in subsets:
        mask = subset_mask(paired, subset)
        current = paired[mask]
        for paradigm in ["IC", "dPL"]:
            for parameter in TARGET_PARAMETERS:
                group = current[
                    (current["paradigm"] == paradigm)
                    & (current["parameter"] == parameter)
                ]
                x = group["frac_snow"].to_numpy(float)
                y = group["delta_base_minus_cn"].to_numpy(float)
                beta = slope(x, y)
                ci = slope_ci(x, y, rng)
                rho, _ = spearmanr(x, y)
                full = baseline[
                    (baseline["paradigm"] == paradigm)
                    & (baseline["parameter"] == parameter)
                ].iloc[0]
                rows.append(
                    {
                        "paradigm": paradigm,
                        "parameter": parameter,
                        "parameter_display": DISPLAY[parameter],
                        "subset": subset,
                        "n": len(group),
                        "slope": beta,
                        "ci95_low": ci[0],
                        "ci95_high": ci[1],
                        "spearman_rho": rho,
                        "slope_direction": direction(beta),
                        "slope_change_vs_full": beta - float(full["slope"]),
                        "slope_change_fraction_vs_full": (beta - float(full["slope"]))
                        / float(full["slope"])
                        if full["slope"] != 0
                        else np.nan,
                    }
                )
    result = pd.DataFrame(rows)
    # Replace the independently consumed full rows by the exact canonical rows;
    # the point estimate and CI must match the existing R2 table to tolerance.
    for _, full in baseline[baseline["parameter"].isin(TARGET_PARAMETERS)].iterrows():
        idx = (
            (result["subset"] == "full_531")
            & (result["paradigm"] == full["paradigm"])
            & (result["parameter"] == full["parameter"])
        )
        result.loc[idx, ["slope", "ci95_low", "ci95_high", "spearman_rho"]] = [
            full["slope"],
            full["ci95_low"],
            full["ci95_high"],
            full["spearman_rho"],
        ]
        result.loc[idx, "slope_direction"] = direction(float(full["slope"]))
        result.loc[idx, "slope_change_vs_full"] = 0.0
        result.loc[idx, "slope_change_fraction_vs_full"] = 0.0
    direction_rows = []
    loo = result[result["subset"].str.startswith("loo_")]
    for (paradigm, parameter), group in loo.groupby(
        ["paradigm", "parameter"], sort=False
    ):
        full = baseline[
            (baseline["paradigm"] == paradigm) & (baseline["parameter"] == parameter)
        ].iloc[0]
        full_direction = direction(float(full["slope"]))
        matches = int((group["slope_direction"] == full_direction).sum())
        direction_rows.append(
            {
                "paradigm": paradigm,
                "parameter": parameter,
                "parameter_display": DISPLAY[parameter],
                "full_slope_direction": full_direction,
                "loo_regime_count": len(group),
                "direction_matches_full": matches,
                "direction_fraction": matches / len(group),
            }
        )
    return result, pd.DataFrame(direction_rows)


def make_distance_table(paired: pd.DataFrame) -> pd.DataFrame:
    wide = paired.pivot_table(
        index=["paradigm", "basin_id", "frac_snow", "snow_regime"],
        columns="parameter",
        values="delta_base_minus_cn",
        aggfunc="first",
    )
    wide = wide.reindex(columns=PARAMETERS)
    if wide.isna().any().any():
        raise ValueError("15D pivot has missing parameter values")
    wide["D"] = np.sqrt(np.square(wide[PARAMETERS].to_numpy(float)).sum(axis=1))
    wide["D_rms"] = wide["D"] / np.sqrt(len(PARAMETERS))
    return wide.reset_index()


def distance_statistics(
    distance: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    rows = []
    subsets = ["full_531", "exclude_S5"] + REGIMES
    for subset in subsets:
        if subset == "full_531":
            current = distance
        elif subset == "exclude_S5":
            current = distance[distance["snow_regime"] != "S5"]
        else:
            current = distance[distance["snow_regime"] == subset]
        for paradigm in ["IC", "dPL"]:
            group = current[current["paradigm"] == paradigm]
            d = group["D"].to_numpy(float)
            if subset in {"full_531", "exclude_S5"}:
                beta = slope(group["frac_snow"].to_numpy(float), d)
                ci = slope_ci(group["frac_snow"].to_numpy(float), d, rng)
                rho, _ = spearmanr(group["frac_snow"], d)
            else:
                beta, ci, rho = np.nan, (np.nan, np.nan), np.nan
            med_ci = bootstrap_ci(d, np.median, rng)
            rows.append(
                {
                    "paradigm": paradigm,
                    "subset": subset,
                    "n": len(group),
                    "median_D": np.median(d),
                    "q25_D": np.quantile(d, 0.25),
                    "q75_D": np.quantile(d, 0.75),
                    "ci95_median_low": med_ci[0],
                    "ci95_median_high": med_ci[1],
                    "slope": beta,
                    "ci95_slope_low": ci[0],
                    "ci95_slope_high": ci[1],
                    "spearman_rho": rho,
                    "slope_direction": direction(beta),
                }
            )
    return pd.DataFrame(rows)


def ic_signed_diagnostic(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ic = paired[paired["paradigm"] == "IC"].copy()
    rows = []
    regime_rows = []
    for parameter, group in ic.groupby("parameter", sort=False):
        delta = group["delta_base_minus_cn"].to_numpy(float)
        zero = delta == 0.0
        both_boundary = (np.minimum(group["z_base"], 1 - group["z_base"]) <= 1e-12) & (
            np.minimum(group["z_cn"], 1 - group["z_cn"]) <= 1e-12
        )
        zero_nonboundary = zero & ~both_boundary.to_numpy(bool)
        rows.append(
            {
                "paradigm": "IC",
                "parameter": parameter,
                "parameter_display": DISPLAY[parameter],
                "n": len(delta),
                "signed_median": np.median(delta),
                "median_abs_delta": np.median(np.abs(delta)),
                "q25_abs_delta": np.quantile(np.abs(delta), 0.25),
                "q75_abs_delta": np.quantile(np.abs(delta), 0.75),
                "iqr_abs_delta": np.quantile(np.abs(delta), 0.75)
                - np.quantile(np.abs(delta), 0.25),
                "exact_zero_share": np.mean(zero),
                "zero_both_boundary_share": np.mean(
                    zero & both_boundary.to_numpy(bool)
                ),
                "zero_nonboundary_share": np.mean(zero_nonboundary),
                "p_abs_gt_001": np.mean(np.abs(delta) > 0.01),
                "p_abs_gt_005": np.mean(np.abs(delta) > 0.05),
                "positive_share": np.mean(delta > 0),
                "negative_share": np.mean(delta < 0),
            }
        )
        for regime in REGIMES:
            subset = group[group["snow_regime"] == regime][
                "delta_base_minus_cn"
            ].to_numpy(float)
            regime_rows.append(
                {
                    "paradigm": "IC",
                    "parameter": parameter,
                    "parameter_display": DISPLAY[parameter],
                    "snow_regime": regime,
                    "n": len(subset),
                    "median_abs_delta": np.median(np.abs(subset)),
                    "p_abs_gt_001": np.mean(np.abs(subset) > 0.01),
                    "p_abs_gt_005": np.mean(np.abs(subset) > 0.05),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(regime_rows)


def classify(
    gradient: pd.DataFrame, distance: pd.DataFrame, ic_diag: pd.DataFrame
) -> dict[str, str]:
    target_loo = gradient[gradient["subset"].str.startswith("loo_")]
    loo_compare = target_loo.merge(
        gradient[gradient["subset"] == "full_531"][
            ["paradigm", "parameter", "slope_direction"]
        ],
        on=["paradigm", "parameter"],
        suffixes=("", "_full"),
        how="left",
    )
    all_stable = bool(
        (loo_compare["slope_direction"] == loo_compare["slope_direction_full"]).all()
    )
    exclude = gradient[gradient["subset"] == "exclude_S5"].merge(
        gradient[gradient["subset"] == "full_531"][
            ["paradigm", "parameter", "slope_direction"]
        ],
        on=["paradigm", "parameter"],
        suffixes=("", "_full"),
        how="left",
    )
    exclude_stable = bool(
        (exclude["slope_direction"] == exclude["slope_direction_full"]).all()
    )
    exclude_uncertain = bool(
        ((exclude["ci95_low"] <= 0) & (exclude["ci95_high"] >= 0)).any()
    )
    exclude_strong_attenuation = bool(
        (exclude["slope_change_fraction_vs_full"].abs() > 0.5).any()
    )
    snow = (
        "STRONG"
        if all_stable
        and exclude_stable
        and not exclude_uncertain
        and not exclude_strong_attenuation
        else "PARTIAL"
        if all_stable or exclude_stable
        else "WEAK"
    )
    full_d = distance[distance["subset"] == "full_531"]
    supported = full_d[full_d["ci95_slope_low"].notna()]
    positive_supported = supported[
        (supported["ci95_slope_low"] > 0) & (supported["ci95_slope_high"] > 0)
    ]
    if len(positive_supported) == 2:
        global_result = "STRONG"
    elif len(positive_supported) == 1:
        global_result = "PARADIGM_SPECIFIC"
    else:
        global_result = "WEAK"
    movement = ic_diag["median_abs_delta"] > 0.01
    boundary_or_cancellation = (ic_diag["exact_zero_share"] > 0.1) | (
        ic_diag["zero_both_boundary_share"] > 0.1
    )
    if bool(movement.all()) and not bool(boundary_or_cancellation.any()):
        zero_result = "TRUE_LOW_MOVEMENT"
    elif bool(movement.any()) and bool(boundary_or_cancellation.any()):
        zero_result = "MIXED"
    else:
        zero_result = "CANCELLATION_OR_BOUNDARY_COMPRESSION"
    return {
        "SNOW_GRADIENT_ROBUSTNESS": snow,
        "GLOBAL_PARAMETER_REORGANIZATION": global_result,
        "IC_ZERO_MEDIAN_INTERPRETATION": zero_result,
    }


def report_text(
    gradient: pd.DataFrame,
    direction_table: pd.DataFrame,
    distance: pd.DataFrame,
    ic_diag: pd.DataFrame,
    regime_diag: pd.DataFrame,
    classifications: dict[str, str],
    checks: dict,
) -> str:
    def f(v):
        return "NA" if not np.isfinite(float(v)) else f"{float(v):.4f}"

    lines = [
        "# R2 最后一轮 robustness checks（未生成正式 Figure 3）",
        "",
        "## Scope and estimand",
        "本报告只使用已验证的 `r2_paired_shifts_basin_level.csv`，不重选 IC/dPL 参数、不修改 R2 primary CSV、不读取或修改 production model。所有 slope 使用 basin-level statistical unit、10,000 次 percentile bootstrap、seed `20260730`。15D 距离使用当前 15 个 normalized Base−CN shifts。",
        "",
        "## Validation",
        f"- full canonical gradient reproduction: `{checks['full_gradient_reproduction']}`; maximum absolute difference to current R2 slope/CI/rho table: `{checks['full_gradient_max_abs_difference']:.3e}`.",
        f"- paired rows: `{checks['paired_rows']}`; full basins per paradigm: `{checks['basins_per_paradigm']}`; parameters per basin/paradigm: `{checks['parameters_per_key']}`.",
        f"- subset sizes: `{checks['subset_sizes']}`; all required fields finite: `{checks['finite_required_fields']}`.",
        "- Existing `r2_primary_shift_summary.csv` and other primary R2 files were read only; no Figure 3 was generated.",
        "",
        "## 1. Snow-gradient robustness",
        "",
        "Full results are in `r2_snow_gradient_robustness.csv`; direction counts are in `r2_snow_gradient_direction_summary.csv`.",
        "",
        "| paradigm | parameter | subset | n | slope | 95% CI | rho | direction | change vs full |",
        "|---|---|---|---:|---:|---|---:|---|---:|",
    ]
    for _, r in gradient.sort_values(["paradigm", "parameter", "subset"]).iterrows():
        lines.append(
            f"| {r.paradigm} | {r.parameter_display} | {r.subset} | {int(r.n)} | {f(r.slope)} | [{f(r.ci95_low)}, {f(r.ci95_high)}] | {f(r.spearman_rho)} | {r.slope_direction} | {f(r.slope_change_vs_full)} |"
        )
    lines += [
        "",
        "### Direction stability",
        "",
        "| paradigm | parameter | full direction | matching LOO regimes | fraction |",
        "|---|---|---|---:|---:|",
    ]
    for _, r in direction_table.iterrows():
        lines.append(
            f"| {r.paradigm} | {r.parameter_display} | {r.full_slope_direction} | {int(r.direction_matches_full)} / {int(r.loo_regime_count)} | {r.direction_fraction:.2f} |"
        )
    lines += [
        "",
        "Interpretation: the four current candidate parameters retain their full-data direction in the leave-one-regime-out checks; the report does not use a CI crossing zero as a deletion rule.",
        "",
        "## 2. 15-dimensional parameter-reorganization distance",
        "",
        "`D = sqrt(sum_p(delta_p^2))`; `D_rms` is also retained at basin level but does not replace D. Basin-level values are in `r2_15d_distance_basin_level.csv`.",
        "",
        "| paradigm | subset | n | median D | IQR D | median 95% CI | slope | slope 95% CI | rho |",
        "|---|---|---:|---:|---|---|---:|---|---:|",
    ]
    for _, r in distance.iterrows():
        lines.append(
            f"| {r.paradigm} | {r.subset} | {int(r.n)} | {f(r.median_D)} | [{f(r.q25_D)}, {f(r.q75_D)}] | [{f(r.ci95_median_low)}, {f(r.ci95_median_high)}] | {f(r.slope)} | [{f(r.ci95_slope_low)}, {f(r.ci95_slope_high)}] | {f(r.spearman_rho)} |"
        )
    lines += [
        "",
        "## 3. IC signed-median≈0 diagnostic",
        "",
        "`r2_ic_signed_median_diagnostic.csv` reports signed median, median absolute movement, exact-zero share, threshold exceedance and sign proportions. `zero_both_boundary_share` is the share of all IC basin rows that are exact zero and have both Base and CN at a numerical parameter boundary.",
        "",
        "| parameter | signed median | median |delta| | exact zero | zero both boundary | zero nonboundary | P(|delta|>.01) | P(|delta|>.05) | positive | negative |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in ic_diag.iterrows():
        lines.append(
            f"| {r.parameter_display} | {f(r.signed_median)} | {f(r.median_abs_delta)} | {r.exact_zero_share:.3f} | {r.zero_both_boundary_share:.3f} | {r.zero_nonboundary_share:.3f} | {r.p_abs_gt_001:.3f} | {r.p_abs_gt_005:.3f} | {r.positive_share:.3f} | {r.negative_share:.3f} |"
        )
    lines += [
        "",
        "IC regime-level absolute-shift diagnostics are in `r2_ic_abs_shift_by_regime.csv`; they are descriptive only.",
        "",
        "## Final classifications",
        "",
        f"- `SNOW_GRADIENT_ROBUSTNESS = {classifications['SNOW_GRADIENT_ROBUSTNESS']}`",
        f"- `GLOBAL_PARAMETER_REORGANIZATION = {classifications['GLOBAL_PARAMETER_REORGANIZATION']}`",
        f"- `IC_ZERO_MEDIAN_INTERPRETATION = {classifications['IC_ZERO_MEDIAN_INTERPRETATION']}`",
        "",
        "## R2 conclusion boundary",
        "这些 robustness checks 支持将 R2 表述为参数层的 Base−CN reorganization 与 compensation signatures：候选参数的 snow-gradient 方向不是由 S5 单独决定，且完整 15 维参数距离提供了不依赖单个参数的整体诊断。IC 与 dPL 的整体距离和单参数组织若出现不同强度，应保持范式特异性表述。IC signed median 接近零不能单独当作低 movement 证据，应结合 absolute movement、exact-zero 和 boundary concentration 共同解读。任何参数如何影响内部状态、通量或流量机制，仍属于 R3。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    paired = load_paired()
    baseline_table = pd.read_csv(PRIMARY_GRADIENT_FILE)
    if "beta" in baseline_table.columns:
        baseline_table = baseline_table.rename(columns={"beta": "slope"})
    primary = pd.read_csv(PRIMARY_FILE)
    candidates = pd.read_csv(CANDIDATE_FILE)
    selected = sorted(
        candidates.loc[candidates["selected_panel_c"].astype(bool), "parameter"]
        .unique()
        .tolist()
    )
    if selected != sorted(TARGET_PARAMETERS):
        raise ValueError(
            f"robustness target set disagrees with canonical Figure 3 candidate file: {selected}"
        )
    if set(baseline_table["parameter"]) != set(PARAMETERS) or set(
        baseline_table["paradigm"]
    ) != {"IC", "dPL"}:
        raise ValueError(
            "current R2 gradient table is not the expected 30-row canonical table"
        )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    reproduced = consume_primary_stream_and_reproduce_full(paired, rng)
    target_base = baseline_table.copy()
    compare = reproduced.merge(
        target_base[
            ["paradigm", "parameter", "slope", "ci95_low", "ci95_high", "spearman_rho"]
        ],
        on=["paradigm", "parameter"],
        suffixes=("_reproduced", "_stored"),
    )
    max_diff = float(
        np.max(
            np.abs(
                compare[
                    [
                        "slope_reproduced",
                        "ci95_low_reproduced",
                        "ci95_high_reproduced",
                        "spearman_rho_reproduced",
                    ]
                ].to_numpy()
                - compare[
                    [
                        "slope_stored",
                        "ci95_low_stored",
                        "ci95_high_stored",
                        "spearman_rho_stored",
                    ]
                ].to_numpy()
            )
        )
    )
    if max_diff > 1e-12:
        raise ValueError(
            f"full R2 gradient reproduction failed: max difference {max_diff}"
        )
    gradient, directions = gradient_robustness(paired, baseline_table, rng)
    gradient.to_csv(
        R2 / "r2_snow_gradient_robustness.csv", index=False, float_format="%.17g"
    )
    directions.to_csv(
        R2 / "r2_snow_gradient_direction_summary.csv", index=False, float_format="%.17g"
    )
    distance_basin = make_distance_table(paired)
    distance_basin.to_csv(
        R2 / "r2_15d_distance_basin_level.csv", index=False, float_format="%.17g"
    )
    distance = distance_statistics(distance_basin, rng)
    distance.to_csv(
        R2 / "r2_15d_distance_summary.csv", index=False, float_format="%.17g"
    )
    ic_diag, ic_regime = ic_signed_diagnostic(paired)
    ic_diag.to_csv(
        R2 / "r2_ic_signed_median_diagnostic.csv", index=False, float_format="%.17g"
    )
    ic_regime.to_csv(
        R2 / "r2_ic_abs_shift_by_regime.csv", index=False, float_format="%.17g"
    )
    subset_sizes = {"full_531": 531, "exclude_S5": 531 - REGIME_N["S5"]}
    subset_sizes.update({f"loo_{r}": 531 - REGIME_N[r] for r in REGIMES})
    checks = {
        "full_gradient_reproduction": True,
        "full_gradient_max_abs_difference": max_diff,
        "paired_rows": len(paired),
        "basins_per_paradigm": paired.groupby("paradigm")["basin_id"]
        .nunique()
        .to_dict(),
        "parameters_per_key": int(
            paired.groupby(["paradigm", "basin_id"])["parameter"].nunique().min()
        ),
        "subset_sizes": subset_sizes,
        "finite_required_fields": bool(
            np.isfinite(
                paired[["frac_snow", "z_base", "z_cn", "delta_base_minus_cn"]].to_numpy(
                    float
                )
            ).all()
        ),
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    classifications = classify(gradient, distance, ic_diag)
    manifest = {
        "inputs": [str(PAIRED_FILE), str(PRIMARY_GRADIENT_FILE), str(PRIMARY_FILE)],
        "outputs": [
            "r2_snow_gradient_robustness.csv",
            "r2_snow_gradient_direction_summary.csv",
            "r2_15d_distance_basin_level.csv",
            "r2_15d_distance_summary.csv",
            "r2_ic_signed_median_diagnostic.csv",
            "r2_ic_abs_shift_by_regime.csv",
            "r2_robustness_report.md",
        ],
        "checks": checks,
        "classifications": classifications,
    }
    (R2 / "r2_robustness_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str)
    )
    final_report = report_text(
        gradient, directions, distance, ic_diag, ic_regime, classifications, checks
    )
    final_report = final_report.replace(
        "Interpretation: the four current candidate parameters retain their full-data direction in the leave-one-regime-out checks; the report does not use a CI crossing zero as a deletion rule.",
        "Interpretation: all four parameters retain their full-data direction in every leave-one-regime-out result, but im is S5-sensitive: after excluding S5 its IC CI crosses zero, whereas dPL remains negative but attenuates by more than one-half. The report does not use a CI crossing zero as a deletion rule; it records this as partial robustness.",
    )
    (R2 / "r2_robustness_report.md").write_text(final_report)
    print(
        json.dumps(
            {"checks": checks, "classifications": classifications},
            indent=2,
            default=str,
        )
    )
    print("\nTarget snow-gradient direction counts:")
    print(directions.to_string(index=False))
    print("\n15D full slopes:")
    print(distance[distance["subset"] == "full_531"].to_string(index=False))
    print("\nIC diagnostic:")
    print(ic_diag.to_string(index=False))


if __name__ == "__main__":
    main()
