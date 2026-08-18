"""R1 table construction from existing KGE records and metadata only."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from r1_metrics import (
    bh_adjust,
    block_bootstrap_mean_ci,
    block_bootstrap_median_ci,
    bootstrap_median_ci,
    rank_relationship,
    summary,
    support_status,
)

PERIODS = {
    "train": ("1981-10-01", "1995-09-30"),
    "test": ("1995-10-01", "2010-09-30"),
}
MODEL_ALIASES = {
    "XAJ": "XAJ-Base",
    "XAJ_CN": "XAJ-CN",
    "XAJ_TGD2": "XAJ-TGD",
    "XAJ_TGD": "XAJ-TGD",
    "HBV": "HBV",
}
STRUCTURES = ["XAJ-Base", "XAJ-TGD", "XAJ-CN"]
STRUCTURAL_EFFECT_SPECS = (
    ("CN-Base", "XAJ-CN", "XAJ-Base"),
    ("TGD-Base", "XAJ-TGD", "XAJ-Base"),
    ("CN-TGD", "XAJ-CN", "XAJ-TGD"),
)
FULL_SAMPLE = "all_531_basins"
SNOW_STRATA = (
    ("S1", "[0, 0.05)", 0.0, 0.05),
    ("S2", "[0.05, 0.15)", 0.05, 0.15),
    ("S3", "[0.15, 0.30)", 0.15, 0.30),
    ("S4", "[0.30, 0.50)", 0.30, 0.50),
    ("S5", "[0.50, 1.00]", 0.50, 1.00),
)


def _key_seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _nan() -> float:
    return math.nan


def _empty_metric_row(
    basin_id: str,
    paradigm: str,
    model: str,
    period: str,
    run: str,
    source: str,
    status: str,
    stored_kge: float = math.nan,
) -> dict[str, Any]:
    start, end = PERIODS.get(period, ("", ""))
    return {
        "basin_id": basin_id,
        "paradigm": paradigm,
        "model": model,
        "period": period,
        "seed_or_restart": run,
        "selected_run": False,
        "kge_prime": _nan(),
        "stored_original_kge": stored_kge,
        "nse": _nan(),
        "pbias": _nan(),
        "rmse": _nan(),
        "valid_observation_count": _nan(),
        "valid_simulation_count": _nan(),
        "valid_days": _nan(),
        "period_start": start,
        "period_end": end,
        "discharge_unit": "mm/day",
        "status": status,
        "source_file": source,
    }


def load_ic(
    root: Path, model_key: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_dir = (
        root
        / "raw"
        / {"XAJ": "xaj", "XAJ_CN": "xaj_cn", "XAJ_TGD2": "xaj_tgd2"}[model_key]
    )
    records: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.json")):
        data = json.loads(path.read_text())
        train = data.get("train_metrics", {}).get("kge", math.nan)
        test = data.get("test_metrics", {}).get("kge", math.nan)
        records.append(
            {
                "basin_id": str(data.get("basin_id", "")).zfill(8),
                "start": int(data.get("start", -1)),
                "seed": data.get("seed", math.nan),
                "status": data.get("status", "missing"),
                "train_kge": float(train) if np.isfinite(train) else math.nan,
                "test_kge": float(test) if np.isfinite(test) else math.nan,
                "source_file": str(path),
                "generations": data.get("generations", math.nan),
            }
        )
    level = pd.DataFrame(records)
    if level.empty:
        raise ValueError(f"No IC raw records found in {raw_dir}")
    level["selected_run"] = False
    valid = level["train_kge"].notna() & np.isfinite(level["train_kge"])
    for basin, group in level[valid].groupby("basin_id", sort=False):
        best = group.sort_values(["train_kge", "start"], ascending=[False, True]).index[
            0
        ]
        level.loc[best, "selected_run"] = True
    rows: list[dict[str, Any]] = []
    model = MODEL_ALIASES[model_key]
    for _, record in level[level["selected_run"]].iterrows():
        for period, field in (("train", "train_kge"), ("test", "test_kge")):
            rows.append(
                _empty_metric_row(
                    record.basin_id,
                    "IC-CMA-ES",
                    model,
                    period,
                    f"restart_{int(record.start):02d}",
                    record.source_file,
                    "stored_original_kge_only",
                    record[field],
                )
            )
            rows[-1]["selected_run"] = True
    level["model"] = model
    level["paradigm"] = "IC-CMA-ES"
    level["stored_metric_definition"] = "original KGE alpha=std_sim/std_obs"
    level["selection_rule"] = (
        "maximum train-period stored original KGE; tie=min restart"
    )
    return (
        pd.DataFrame(rows),
        level,
        {
            "raw_files": len(level),
            "selected_basins": int(level["selected_run"].sum()),
            "expected_restarts": 10,
        },
    )


def load_dpl(
    root: Path, model_key: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    model_dir = root / model_key
    records: list[dict[str, Any]] = []
    for seed_dir in sorted(model_dir.glob("seed_*")):
        path = seed_dir / "train_test_kge_by_basin.csv"
        if not path.exists():
            continue
        seed = seed_dir.name.removeprefix("seed_")
        table = pd.read_csv(path)
        table["basin_id"] = table["basin_id"].astype(str).str.zfill(8)
        for _, row in table.iterrows():
            records.append(
                {
                    "basin_id": row["basin_id"],
                    "seed": seed,
                    "train_kge": row["train_kge"],
                    "test_kge": row["test_kge"],
                    "source_file": str(path),
                }
            )
    level = pd.DataFrame(records)
    if level.empty:
        raise ValueError(f"No dPL KGE files found in {model_dir}")
    model = MODEL_ALIASES[model_key]
    rows: list[dict[str, Any]] = []
    for _, record in level.iterrows():
        for period, field in (("train", "train_kge"), ("test", "test_kge")):
            value = float(record[field]) if np.isfinite(record[field]) else math.nan
            rows.append(
                _empty_metric_row(
                    record.basin_id,
                    "dPL-MLP",
                    model,
                    period,
                    f"seed_{record.seed}",
                    record.source_file,
                    "stored_original_kge_only",
                    value,
                )
            )
    level["model"] = model
    level["paradigm"] = "dPL-MLP"
    level["stored_metric_definition"] = "original KGE alpha=std_sim/std_obs"
    level["valid_record"] = np.isfinite(level["train_kge"]) & np.isfinite(
        level["test_kge"]
    )
    return (
        pd.DataFrame(rows),
        level,
        {
            "seed_count": int(level["seed"].nunique()),
            "basin_count_per_seed": int(level["basin_id"].nunique()),
        },
    )


def median_dpl_rows(seed_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, basin), group in seed_level.groupby(["model", "basin_id"], sort=True):
        for period in ("train", "test"):
            subset = group[group["period"] == period]
            values = subset["stored_original_kge"].to_numpy(dtype=float)
            stored = (
                float(np.nanmedian(values)) if np.isfinite(values).any() else math.nan
            )
            row = _empty_metric_row(
                basin,
                "dPL-MLP",
                model,
                period,
                "median_across_seeds",
                "multiple seed files",
                "median_seed_original_kge_only",
                stored,
            )
            row["selected_run"] = True
            row["seed_count"] = int(np.isfinite(values).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def pivot_effects(
    performance: pd.DataFrame, paradigm: str, metric: str = "kge"
) -> pd.DataFrame:
    subset = performance[
        (performance["paradigm"] == paradigm) & (performance["selected_run"])
    ]
    run_columns = ["basin_id", "period"] + (
        ["seed_or_restart"] if paradigm == "dPL-MLP" else []
    )
    pivot = subset[run_columns + ["model", metric]].drop_duplicates(
        run_columns + ["model"]
    )
    wide = pivot.pivot(index=run_columns, columns="model", values=metric).reset_index()
    wide.columns.name = None
    for column in STRUCTURES:
        if column not in wide:
            wide[column] = math.nan
    rows = []
    for _, row in wide.iterrows():
        base, tgd, cn = row["XAJ-Base"], row["XAJ-TGD"], row["XAJ-CN"]
        for name, value in (
            ("CN-Base", cn - base),
            ("TGD-Base", tgd - base),
            ("CN-TGD", cn - tgd),
        ):
            result = {
                "basin_id": row["basin_id"],
                "paradigm": paradigm,
                "period": row["period"],
                "effect": name,
                "effect_value": value,
                "status": "valid" if np.isfinite(value) else "missing_required_member",
                "metric": metric,
                "effect_family": "structural",
                "estimand": f"KGE_{name.replace('-', '_')}",
            }
            if paradigm == "dPL-MLP":
                result["seed_or_restart"] = row["seed_or_restart"]
            rows.append(result)
    return pd.DataFrame(rows)


def median_seed_effects(seed_effects: pd.DataFrame) -> pd.DataFrame:
    """Collapse dPL effects only after calculating each seed's basin effect."""
    if seed_effects.empty:
        return seed_effects.copy()
    rows = []
    group_columns = ["basin_id", "paradigm"]
    for column in (
        "model",
        "period",
        "effect",
        "metric",
        "effect_family",
        "analysis_set",
    ):
        if column in seed_effects.columns:
            group_columns.append(column)
    for keys, group in seed_effects.groupby(group_columns, sort=True):
        key_values = dict(
            zip(group_columns, keys if isinstance(keys, tuple) else (keys,))
        )
        values = group["effect_value"].to_numpy(dtype=float)
        valid = values[np.isfinite(values)]
        rows.append(
            {
                **key_values,
                "effect_value": float(np.median(valid)) if len(valid) else math.nan,
                "status": "median_across_seeds"
                if len(valid)
                else "missing_required_member",
                "seed_count": int(len(valid)),
                "seed_or_restart": "median_across_seeds",
            }
        )
    return pd.DataFrame(rows)


def generalization_effects(
    performance: pd.DataFrame, paradigm: str, metric: str = "kge"
) -> pd.DataFrame:
    subset = performance[
        (performance["paradigm"] == paradigm)
        & (performance["selected_run"])
        & (performance["model"].isin(STRUCTURES))
    ]
    run_columns = ["basin_id", "model"] + (
        ["seed_or_restart"] if paradigm == "dPL-MLP" else []
    )
    wide = (
        subset[run_columns + ["period", metric]]
        .drop_duplicates(run_columns + ["period"])
        .pivot(index=run_columns, columns="period", values=metric)
        .reset_index()
    )
    rows = []
    for _, row in wide.iterrows():
        value = row.get("train", math.nan) - row.get("test", math.nan)
        result = {
            "basin_id": row["basin_id"],
            "paradigm": paradigm,
            "model": row["model"],
            "effect": "train_minus_test",
            "effect_value": value,
            "status": "valid" if np.isfinite(value) else "missing_required_member",
            "metric": metric,
            "effect_family": "train_test_gap",
            "estimand": "KGE_train - KGE_test",
        }
        if paradigm == "dPL-MLP":
            result["seed_or_restart"] = row["seed_or_restart"]
        rows.append(result)
    exposure_columns = ["basin_id"] + (
        ["seed_or_restart"] if paradigm == "dPL-MLP" else []
    )
    exposure = subset[exposure_columns + ["model", "period", metric]].drop_duplicates(
        exposure_columns + ["model", "period"]
    )
    exposure = exposure.pivot(
        index=exposure_columns, columns=["model", "period"], values=metric
    ).reset_index()
    for effect_name, enhanced, base in STRUCTURAL_EFFECT_SPECS:
        for period in ("test", "train"):
            left = exposure.get(
                (enhanced, period), pd.Series(index=exposure.index, dtype=float)
            )
            right = exposure.get(
                (base, period), pd.Series(index=exposure.index, dtype=float)
            )
            for index, value in (left - right).items():
                result = {
                    "basin_id": exposure.iloc[index][("basin_id", "")],
                    "paradigm": paradigm,
                    "model": enhanced,
                    "period": period,
                    "effect": f"{effect_name}_{period}",
                    "effect_value": value,
                    "status": "valid"
                    if np.isfinite(value)
                    else "missing_required_member",
                    "metric": metric,
                    "effect_family": "generalization_exposure",
                    "estimand": f"KGE_{enhanced}_{period} - KGE_{base}_{period}",
                }
                if paradigm == "dPL-MLP":
                    result["seed_or_restart"] = exposure.iloc[index][
                        ("seed_or_restart", "")
                    ]
                rows.append(result)
        test_delta = exposure.get(
            (enhanced, "test"), pd.Series(index=exposure.index, dtype=float)
        ) - exposure.get((base, "test"), pd.Series(index=exposure.index, dtype=float))
        train_delta = exposure.get(
            (enhanced, "train"), pd.Series(index=exposure.index, dtype=float)
        ) - exposure.get((base, "train"), pd.Series(index=exposure.index, dtype=float))
        for index, value in (test_delta - train_delta).items():
            result = {
                "basin_id": exposure.iloc[index][("basin_id", "")],
                "paradigm": paradigm,
                "model": enhanced,
                "period": "test_minus_train",
                "effect": f"E_{effect_name}",
                "effect_value": value,
                "status": "valid" if np.isfinite(value) else "missing_required_member",
                "metric": metric,
                "effect_family": "generalization_exposure",
                "estimand": f"(KGE_{enhanced}_test - KGE_{base}_test) - (KGE_{enhanced}_train - KGE_{base}_train)",
            }
            if paradigm == "dPL-MLP":
                result["seed_or_restart"] = exposure.iloc[index][
                    ("seed_or_restart", "")
                ]
            rows.append(result)
    return pd.DataFrame(rows)


def ic_dpl_effects(
    ic_performance: pd.DataFrame, dpl_performance: pd.DataFrame, metric: str = "kge"
) -> pd.DataFrame:
    ic = ic_performance[
        ic_performance["selected_run"] & ic_performance["model"].isin(STRUCTURES)
    ]
    dpl = dpl_performance[
        dpl_performance["selected_run"] & dpl_performance["model"].isin(STRUCTURES)
    ]
    ic = ic[["basin_id", "model", "period", metric]].drop_duplicates(
        ["basin_id", "model", "period"]
    )
    dpl = dpl[
        ["basin_id", "model", "period", "seed_or_restart", metric]
    ].drop_duplicates(["basin_id", "model", "period", "seed_or_restart"])
    wide = dpl.merge(
        ic, on=["basin_id", "model", "period"], how="left", suffixes=("_dpl", "_ic")
    )
    rows = []
    for keys, group in wide.groupby(["basin_id", "model", "period"], sort=True):
        basin, model, period = keys
        values = group[f"{metric}_ic"].to_numpy(dtype=float) - group[
            f"{metric}_dpl"
        ].to_numpy(dtype=float)
        valid = values[np.isfinite(values)]
        rows.append(
            {
                "basin_id": basin,
                "paradigm": "IC-dPL",
                "model": model,
                "period": period,
                "effect": "IC_minus_dPL",
                "effect_value": float(np.median(valid)) if len(valid) else math.nan,
                "status": "median_across_dpl_seeds"
                if len(valid)
                else "missing_required_member",
                "metric": metric,
                "seed_count": int(len(valid)),
                "effect_family": "ic_dpl_transfer",
                "estimand": "KGE_IC - median_seed(KGE_dPL)",
            }
        )
    return pd.DataFrame(rows)


def snow_attributes(
    data_dir: Path, basin_list: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import pickle

    ids = [str(x).zfill(8) for x in json.loads(basin_list.read_text())]
    with (data_dir / "camels_dataset").open("rb") as handle:
        _forcing, _target, attributes = pickle.load(handle)
    source_ids = [str(int(x)).zfill(8) for x in np.load(data_dir / "gage_id.npy")]
    positions = {basin: i for i, basin in enumerate(source_ids)}
    rows = []
    for basin in ids:
        source_index = positions.get(basin)
        value = (
            float(np.asarray(attributes)[source_index, 3])
            if source_index is not None
            else math.nan
        )
        rows.append(
            {
                "basin_id": basin,
                "frac_snow": value,
                "attribute_name": "frac_snow",
                "attribute_index": 3,
                "unit": "stored CAMELS fraction",
                "source_file": str(data_dir / "camels_dataset"),
                "match_status": "matched" if source_index is not None else "missing",
            }
        )
    return pd.DataFrame(rows), {
        "source_file": str(data_dir / "camels_dataset"),
        "field_name": "attributes[:,3] / frac_snow",
        "units": "stored CAMELS fraction",
        "missing_value_handling": "no imputation; nonfinite excluded",
        "basin_matching": "531 IDs matched by forcing metadata basin IDs",
    }


def _ols_slope(
    x: pd.Series, y: pd.Series, groups: pd.Series | None = None
) -> dict[str, float]:
    import statsmodels.api as sm

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.unique(x[mask]).size < 2:
        return {
            "slope": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "std_error": math.nan,
            "p_value": math.nan,
        }
    design = sm.add_constant(x[mask], has_constant="add")
    fit = sm.OLS(y[mask], design).fit()
    ci = fit.conf_int(alpha=0.05)[1]
    return {
        "slope": float(fit.params[1]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "std_error": float(fit.bse[1]),
        "p_value": float(fit.pvalues[1]),
    }


def _relationship_row(
    paradigm: str,
    effect: str,
    period: str,
    joined: pd.DataFrame,
    *,
    analysis_type: str = "within_paradigm",
    **extra: Any,
) -> dict[str, Any]:
    if len(joined) < 3:
        return {
            "paradigm": paradigm,
            "effect": effect,
            "period": period,
            "paired_basin_count": int(len(joined)),
            "analysis_type": analysis_type,
            "status": "unresolved_insufficient_pairs",
            **extra,
        }
    rho, p = rank_relationship(joined["frac_snow"], joined["effect_value"])
    from scipy.stats import theilslopes

    robust = theilslopes(joined["effect_value"], joined["frac_snow"], alpha=0.95)
    ols = _ols_slope(joined["frac_snow"], joined["effect_value"])
    return {
        "paradigm": paradigm,
        "effect": effect,
        "period": period,
        "paired_basin_count": int(len(joined)),
        "median": float(joined.effect_value.median()),
        "iqr": float(
            joined.effect_value.quantile(0.75) - joined.effect_value.quantile(0.25)
        ),
        "mean": float(joined.effect_value.mean()),
        "fraction_greater_zero": float((joined.effect_value > 0).mean()),
        "spearman_rho": rho,
        "spearman_p": p,
        "robust_slope": float(robust.slope),
        "robust_ci_low": float(robust.low_slope),
        "robust_ci_high": float(robust.high_slope),
        "ols_slope": ols["slope"],
        "ols_ci_low": ols["ci_low"],
        "ols_ci_high": ols["ci_high"],
        "analysis_type": analysis_type,
        "status": "valid_continuous_attribute",
        **extra,
    }


def _interaction_relationship(
    effects: pd.DataFrame, attributes: pd.DataFrame
) -> list[dict[str, Any]]:
    subset = effects[
        (effects["effect"].eq("CN-TGD"))
        & (effects["period"].eq("test"))
        & effects["paradigm"].isin(["IC-CMA-ES", "dPL-MLP"])
    ].copy()
    joined = subset.merge(
        attributes[["basin_id", "frac_snow"]], on="basin_id", how="inner"
    )
    joined = joined[
        np.isfinite(joined["effect_value"]) & np.isfinite(joined["frac_snow"])
    ]
    if len(joined) < 6 or joined["basin_id"].nunique() < 3:
        return [
            {
                "paradigm": "combined",
                "effect": "CN-TGD",
                "period": "test",
                "analysis_type": "cluster_robust_interaction",
                "status": "unresolved_insufficient_pairs",
                "matched_basin_count": int(joined["basin_id"].nunique()),
            }
        ]
    import statsmodels.api as sm

    joined["paradigm_dPL"] = (joined["paradigm"] == "dPL-MLP").astype(float)
    joined["snow_x_dPL"] = joined["frac_snow"] * joined["paradigm_dPL"]
    design = sm.add_constant(
        joined[["frac_snow", "paradigm_dPL", "snow_x_dPL"]], has_constant="add"
    )
    fit = sm.OLS(joined["effect_value"].to_numpy(float), design).fit(
        cov_type="cluster",
        cov_kwds={"groups": joined["basin_id"].astype(str).to_numpy()},
    )
    labels = {
        "const": "eta_0",
        "frac_snow": "eta_1_frac_snow",
        "paradigm_dPL": "eta_2_paradigm_dPL",
        "snow_x_dPL": "eta_3_frac_snow_x_paradigm_dPL",
    }
    rows = []
    for column, term in labels.items():
        ci = (
            fit.conf_int(alpha=0.05).loc[column]
            if hasattr(fit.conf_int(), "loc")
            else fit.conf_int(alpha=0.05)[list(design.columns).index(column)]
        )
        rows.append(
            {
                "paradigm": "combined",
                "effect": "CN-TGD",
                "period": "test",
                "analysis_type": "cluster_robust_interaction",
                "term": term,
                "estimate": float(fit.params[column]),
                "std_error": float(fit.bse[column]),
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
                "p_value": float(fit.pvalues[column]),
                "matched_basin_count": int(joined["basin_id"].nunique()),
                "reference_category": "IC-CMA-ES",
                "model_specification": "effect_value ~ frac_snow + paradigm_dPL + frac_snow:paradigm_dPL; cluster=basin_id",
                "status": "valid_cluster_robust",
                "support_status": support_status(float(ci[0]), float(ci[1])),
            }
        )
    return rows


def _association_row(
    x: pd.Series,
    y: pd.Series,
    *,
    paradigm: str,
    effect: str,
    period: str,
    analysis_type: str,
    x_metric: str,
    y_metric: str,
) -> dict[str, Any]:
    joined = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(joined) < 3:
        return {
            "paradigm": paradigm,
            "effect": effect,
            "period": period,
            "analysis_type": analysis_type,
            "paired_basin_count": int(len(joined)),
            "x_metric": x_metric,
            "y_metric": y_metric,
            "status": "unresolved_insufficient_pairs",
        }
    rho, p = rank_relationship(joined["x"], joined["y"])
    from scipy.stats import theilslopes

    robust = theilslopes(joined["y"], joined["x"], alpha=0.95)
    ols = _ols_slope(joined["x"], joined["y"])
    return {
        "paradigm": paradigm,
        "effect": effect,
        "period": period,
        "analysis_type": analysis_type,
        "paired_basin_count": int(len(joined)),
        "spearman_rho": rho,
        "spearman_p": p,
        "robust_slope": float(robust.slope),
        "robust_ci_low": float(robust.low_slope),
        "robust_ci_high": float(robust.high_slope),
        "ols_slope": ols["slope"],
        "ols_ci_low": ols["ci_low"],
        "ols_ci_high": ols["ci_high"],
        "x_metric": x_metric,
        "y_metric": y_metric,
        "status": "valid_association",
    }


def snow_relationships(
    effects: pd.DataFrame,
    attributes: pd.DataFrame,
    signature_effects: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (paradigm, effect, period), group in effects.groupby(
        ["paradigm", "effect", "period"], dropna=False
    ):
        joined = group.merge(
            attributes[["basin_id", "frac_snow"]], on="basin_id", how="inner"
        )
        joined = joined[
            np.isfinite(joined["effect_value"]) & np.isfinite(joined["frac_snow"])
        ]
        rows.append(_relationship_row(paradigm, effect, period, joined))
    rows.extend(_interaction_relationship(effects, attributes))
    if signature_effects is not None and not signature_effects.empty:
        kge = effects[
            (effects["effect"].eq("CN-TGD"))
            & (effects["period"].eq("test"))
            & effects["paradigm"].isin(["IC-CMA-ES", "dPL-MLP"])
        ][["basin_id", "paradigm", "effect_value"]]
        sig = signature_effects[
            (signature_effects["signature"].isin(["CT", "AMJJ"]))
            & signature_effects["first_model"].eq("XAJ-TGD")
            & signature_effects["second_model"].eq("XAJ-CN")
            & signature_effects.get(
                "analysis_set", pd.Series("primary_min5", index=signature_effects.index)
            ).eq("primary_min5")
        ]
        for paradigm, kge_group in kge.groupby("paradigm", sort=True):
            for signature, group in sig[sig["paradigm"].eq(paradigm)].groupby(
                "signature", sort=True
            ):
                group = group.drop_duplicates(
                    ["basin_id", "paradigm", "signature", "first_model", "second_model"]
                )
                kge_group = kge_group.drop_duplicates(["basin_id", "paradigm"])
                joined = kge_group.merge(
                    group[["basin_id", "paradigm", "effect_value"]],
                    on=["basin_id", "paradigm"],
                    suffixes=("_kge", "_signature"),
                )
                rows.append(
                    _association_row(
                        joined["effect_value_kge"],
                        joined["effect_value_signature"],
                        paradigm=paradigm,
                        effect=f"KGE_CN-TGD_vs_{signature}_TGD-CN",
                        period="test",
                        analysis_type="kge_signature_association",
                        x_metric="KGE_CN-TGD_test",
                        y_metric=f"{signature}_TGD-CN_error_reduction",
                    )
                )
    return pd.DataFrame(rows)


def write_summary_tables(
    performance: pd.DataFrame,
    effects: pd.DataFrame,
    output: Path,
    seed: int,
    metric_col: str = "kge",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    absolute_rows, paired_rows, bootstrap_rows, test_rows = [], [], [], []
    for keys, group in performance[performance["selected_run"]].groupby(
        ["paradigm", "model", "period"]
    ):
        key_values = dict(zip(["paradigm", "model", "period"], keys))
        stats = summary(
            group[metric_col],
            np.random.default_rng(
                _key_seed(seed, "absolute_summary", *key_values.values(), metric_col)
            ),
        )
        absolute_rows.append(
            {
                **dict(zip(["paradigm", "model", "period"], keys)),
                "metric": metric_col,
                "status": "valid_daily_recomputation",
                **stats,
            }
        )
        bootstrap_rows.append(
            {
                "record_type": "bootstrap",
                "family": "absolute_metric",
                **dict(zip(["paradigm", "model", "period"], keys)),
                "metric": metric_col,
                "n": stats["valid_basin_count"],
                "ci_low": stats["bootstrap_ci_low"],
                "ci_high": stats["bootstrap_ci_high"],
                "resamples": 10_000,
                "seed": seed,
                "method": "ordinary_basin_bootstrap",
                "status": "valid_daily_recomputation",
            }
        )
    effect_tables = [effects]
    if "effect" in effects:
        effect_group_columns = [
            c
            for c in ["paradigm", "model", "period", "effect", "metric", "analysis_set"]
            if c in effects
        ]
        for keys, group in effects.groupby(effect_group_columns, dropna=False):
            key_values = dict(
                zip(effect_group_columns, keys if isinstance(keys, tuple) else (keys,))
            )
            stats = summary(
                group["effect_value"],
                np.random.default_rng(
                    _key_seed(seed, "effect_summary", *key_values.values())
                ),
            )
            row = {**key_values, "status": "valid_daily_recomputation", **stats}
            paired_rows.append(row)
            finite = group[np.isfinite(group["effect_value"])]
            bootstrap_rows.append(
                {
                    "record_type": "bootstrap",
                    "family": "paired_effect",
                    **{
                        k: row.get(k, "")
                        for k in (
                            "paradigm",
                            "model",
                            "period",
                            "effect",
                            "metric",
                            "analysis_set",
                        )
                    },
                    "n": len(finite),
                    "ci_low": stats["bootstrap_ci_low"],
                    "ci_high": stats["bootstrap_ci_high"],
                    "resamples": 10_000,
                    "seed": seed,
                    "method": "ordinary_basin_bootstrap",
                    "status": row["status"],
                }
            )
            test_row = test_result(
                finite["effect_value"].to_numpy(),
                {
                    k: row.get(k, "")
                    for k in (
                        "paradigm",
                        "model",
                        "period",
                        "effect",
                        "metric",
                        "analysis_set",
                    )
                },
                "paired_effect",
            )
            test_row["record_type"] = "test"
            test_rows.append(test_row)
    return (
        pd.DataFrame(absolute_rows),
        pd.DataFrame(paired_rows),
        pd.DataFrame(bootstrap_rows + test_rows),
    )


def test_result(
    values: np.ndarray, keys: dict[str, Any], family: str
) -> dict[str, Any]:
    from scipy.stats import binomtest, wilcoxon

    values = values[np.isfinite(values)]
    result: dict[str, Any] = {
        "family": family,
        **keys,
        "n": int(len(values)),
        "wilcoxon_p": math.nan,
        "sign_test_p": math.nan,
        "status": "computed",
    }
    if len(values):
        try:
            result["wilcoxon_p"] = float(
                wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue
            )
        except ValueError:
            pass
        nonzero = values[values != 0]
        if len(nonzero):
            result["sign_test_p"] = float(
                binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5).pvalue
            )
    return result


def standard_kge(
    sim: np.ndarray, obs: np.ndarray, min_valid: int = 30
) -> tuple[float, int, int, int, int]:
    """Reproduce the repository KGE(Q) evaluator from one paired daily mask."""
    sim, obs = np.asarray(sim, dtype=float), np.asarray(obs, dtype=float)
    valid_obs = np.isfinite(obs) & (obs >= 0)
    valid_sim = np.isfinite(sim) & (sim >= 0)
    mask = valid_obs & valid_sim
    n_obs, n_sim, n = int(valid_obs.sum()), int(valid_sim.sum()), int(mask.sum())
    if n < min_valid:
        return math.nan, n_obs, n_sim, n, 0
    s, o = sim[mask].astype(np.float64), obs[mask].astype(np.float64)
    obs_std = float(o.std())
    if obs_std < 1e-10 or float(o.mean()) == 0.0:
        return math.nan, n_obs, n_sim, n, 0
    r = float(np.corrcoef(s, o)[0, 1])
    alpha = float(s.std() / obs_std)
    beta = float(s.mean() / o.mean())
    value = float(
        1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    )
    return value, n_obs, n_sim, n, 1


def daily_metric_rows(output: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    files = sorted(output.glob("r1_daily_simulations_*.parquet"))
    if not files:
        raise FileNotFoundError(
            "statistics mode requires r1_daily_simulations_*.parquet"
        )
    for path in files:
        table = pd.read_parquet(
            path,
            columns=[
                "basin_id",
                "paradigm",
                "model",
                "seed_or_restart",
                "selected_run",
                "period",
                "date",
                "q_obs",
                "q_sim",
                "discharge_unit",
                "source_checkpoint_or_parameter_file",
                "source_configuration",
            ],
        )
        required = {
            "basin_id",
            "paradigm",
            "model",
            "seed_or_restart",
            "selected_run",
            "period",
            "date",
            "q_obs",
            "q_sim",
            "discharge_unit",
        }
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"daily schema mismatch in {path}: {sorted(missing)}")
        for keys, group in table.groupby(
            [
                "basin_id",
                "paradigm",
                "model",
                "seed_or_restart",
                "selected_run",
                "period",
            ],
            sort=True,
        ):
            basin, paradigm, model, run, selected, period = keys
            dates = pd.to_datetime(group["date"])
            obs = group["q_obs"].to_numpy(dtype=float)
            sim = group["q_sim"].to_numpy(dtype=float)
            kge, n_obs, n_sim, valid_days, valid_metric = standard_kge(sim, obs)
            mask = np.isfinite(obs) & np.isfinite(sim) & (obs >= 0) & (sim >= 0)
            if valid_days:
                error = sim[mask] - obs[mask]
                denominator = float(np.sum(obs[mask]))
                nse_denominator = float(np.sum((obs[mask] - np.mean(obs[mask])) ** 2))
                nse = (
                    float(1.0 - np.sum(error**2) / nse_denominator)
                    if nse_denominator > 0
                    else math.nan
                )
                pbias = (
                    float(100.0 * np.sum(error) / denominator)
                    if denominator != 0
                    else math.nan
                )
                rmse = float(np.sqrt(np.mean(error**2)))
            else:
                nse = pbias = rmse = math.nan
            source_checkpoint = str(
                group["source_checkpoint_or_parameter_file"].iloc[0]
            )
            source_config = str(group["source_configuration"].iloc[0])
            restart = run
            if paradigm == "IC-CMA-ES":
                match = re.search(r"_start(\d+)\.json", source_checkpoint)
                restart = f"restart_{int(match.group(1)):02d}" if match else run
            rows.append(
                {
                    "basin_id": str(basin).zfill(8),
                    "paradigm": paradigm,
                    "model": model,
                    "period": period,
                    "seed_or_restart": restart,
                    "selected_run": bool(selected),
                    "kge": kge,
                    "kge_prime": math.nan,
                    "nse": nse,
                    "pbias": pbias,
                    "rmse": rmse,
                    "valid_observation_count": n_obs,
                    "valid_simulation_count": n_sim,
                    "valid_days": valid_days,
                    "period_start": dates.min().date().isoformat(),
                    "period_end": dates.max().date().isoformat(),
                    "discharge_unit": str(group["discharge_unit"].iloc[0]),
                    "status": "valid" if valid_metric else "invalid_metric",
                    "source_file": str(path),
                    "source_checkpoint_or_parameter_file": source_checkpoint,
                    "source_configuration": source_config,
                }
            )
    return pd.DataFrame(rows)


def median_seed_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric = [
        "kge",
        "kge_prime",
        "nse",
        "pbias",
        "rmse",
        "valid_observation_count",
        "valid_simulation_count",
        "valid_days",
    ]
    for keys, group in seed_metrics.groupby(
        ["basin_id", "paradigm", "model", "period"], sort=True
    ):
        row = dict(zip(["basin_id", "paradigm", "model", "period"], keys))
        row.update(
            {
                column: float(np.nanmedian(group[column]))
                if group[column].notna().any()
                else math.nan
                for column in numeric
            }
        )
        row.update(
            {
                "seed_or_restart": "median_across_seeds",
                "selected_run": True,
                "period_start": group.period_start.iloc[0],
                "period_end": group.period_end.iloc[0],
                "discharge_unit": group.discharge_unit.iloc[0],
                "status": "median_across_seed_daily_recomputation",
                "source_file": ";".join(sorted(group.source_file.unique())),
                "source_checkpoint_or_parameter_file": ";".join(
                    sorted(group.source_checkpoint_or_parameter_file.unique())
                ),
                "source_configuration": ";".join(
                    sorted(group.source_configuration.unique())
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _complete_water_year(group: pd.DataFrame) -> bool:
    dates = (
        pd.to_datetime(group["date"])
        .dt.normalize()
        .sort_values()
        .to_numpy(dtype="datetime64[D]")
    )
    if not len(dates):
        return False
    start = dates[0]
    end = dates[-1]
    expected = int((end - start).astype("timedelta64[D]").astype(int) + 1)
    return (
        len(dates) == expected
        and len(np.unique(dates)) == expected
        and bool(
            (
                group["q_obs"].notna()
                & group["q_sim"].notna()
                & (group["q_obs"] >= 0)
                & (group["q_sim"] >= 0)
            ).all()
        )
    )


def signature_tables_from_years(
    years: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, group in years.groupby(
        ["basin_id", "paradigm", "model", "seed_or_restart", "period"], sort=True
    ):
        basin, paradigm, model, run, period = keys
        valid = group[group.status.eq("valid_ct_amjj_spo_unresolved")]
        row = {
            "basin_id": basin,
            "paradigm": paradigm,
            "model": model,
            "seed_or_restart": run,
            "period": period,
            "valid_years": int(len(valid)),
            "analysis_set": "all_available",
            "status": "valid_ct_amjj_spo_unresolved"
            if len(valid)
            else "no_complete_water_year",
        }
        for column in (
            "ct_obs",
            "ct_sim",
            "ct_error_signed",
            "ct_error_absolute",
            "amjj_obs",
            "amjj_sim",
            "amjj_error_signed",
            "amjj_error_absolute",
        ):
            row[column] = float(valid[column].median()) if len(valid) else math.nan
        for column in ("spo_obs", "spo_sim", "spo_error_signed", "spo_error_absolute"):
            row[column] = math.nan
        rows.append(row)
    basin_level = pd.DataFrame(rows)
    effect_rows = []
    for (paradigm, period, run), group in basin_level.groupby(
        ["paradigm", "period", "seed_or_restart"], sort=True
    ):
        wide = group.pivot(
            index="basin_id",
            columns="model",
            values=["ct_error_absolute", "amjj_error_absolute"],
        )
        for signature, error_column in (
            ("CT", "ct_error_absolute"),
            ("AMJJ", "amjj_error_absolute"),
        ):
            for first, second in (
                ("XAJ-Base", "XAJ-CN"),
                ("XAJ-Base", "XAJ-TGD"),
                ("XAJ-TGD", "XAJ-CN"),
            ):
                if (error_column, first) not in wide or (
                    error_column,
                    second,
                ) not in wide:
                    continue
                values = wide[(error_column, first)] - wide[(error_column, second)]
                for basin, value in values.dropna().items():
                    effect_rows.append(
                        {
                            "basin_id": basin,
                            "paradigm": paradigm,
                            "period": period,
                            "seed_or_restart": run,
                            "signature": signature,
                            "first_model": first,
                            "second_model": second,
                            "effect_value": value,
                            "status": "valid",
                            "analysis_set": "all_available",
                        }
                    )
    return basin_level, pd.DataFrame(effect_rows)


def signature_rows(output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    year_rows: list[dict[str, Any]] = []
    for path in sorted(output.glob("r1_daily_simulations_*.parquet")):
        table = pd.read_parquet(
            path,
            columns=[
                "basin_id",
                "paradigm",
                "model",
                "seed_or_restart",
                "period",
                "date",
                "q_obs",
                "q_sim",
            ],
            filters=[("period", "in", ["train", "test"])],
        )
        table["date"] = pd.to_datetime(table["date"])
        table["water_year"] = table["date"].dt.year + (
            table["date"].dt.month >= 10
        ).astype(int)
        for keys, group in table.groupby(
            [
                "basin_id",
                "paradigm",
                "model",
                "seed_or_restart",
                "period",
                "water_year",
            ],
            sort=True,
        ):
            basin, paradigm, model, run, period, water_year = keys
            row = {
                "basin_id": str(basin).zfill(8),
                "paradigm": paradigm,
                "model": model,
                "seed_or_restart": run,
                "period": period,
                "water_year": int(water_year),
                "status": "incomplete_water_year",
            }
            if _complete_water_year(group):
                group = group.sort_values("date")
                obs, sim = group.q_obs.to_numpy(float), group.q_sim.to_numpy(float)
                obs_total, sim_total = float(obs.sum()), float(sim.sum())
                ct_obs = (
                    int(np.argmax(np.cumsum(obs) >= 0.5 * obs_total) + 1)
                    if obs_total > 0
                    else math.nan
                )
                ct_sim = (
                    int(np.argmax(np.cumsum(sim) >= 0.5 * sim_total) + 1)
                    if sim_total > 0
                    else math.nan
                )
                month = group.date.dt.month.to_numpy()
                amjj = (month >= 4) & (month <= 7)
                amjj_obs = (
                    float(obs[amjj].sum() / obs_total) if obs_total > 0 else math.nan
                )
                amjj_sim = (
                    float(sim[amjj].sum() / sim_total) if sim_total > 0 else math.nan
                )
                row.update(
                    {
                        "ct_obs": ct_obs,
                        "ct_sim": ct_sim,
                        "ct_error_signed": ct_sim - ct_obs,
                        "ct_error_absolute": abs(ct_sim - ct_obs),
                        "spo_obs": math.nan,
                        "spo_sim": math.nan,
                        "spo_error_signed": math.nan,
                        "spo_error_absolute": math.nan,
                        "amjj_obs": amjj_obs,
                        "amjj_sim": amjj_sim,
                        "amjj_error_signed": amjj_sim - amjj_obs,
                        "amjj_error_absolute": abs(amjj_sim - amjj_obs),
                        "status": "valid_ct_amjj_spo_unresolved",
                        "spo_status": "unresolved_definition_search_window",
                    }
                )
            else:
                row.update(
                    {
                        column: math.nan
                        for column in (
                            "ct_obs",
                            "ct_sim",
                            "ct_error_signed",
                            "ct_error_absolute",
                            "spo_obs",
                            "spo_sim",
                            "spo_error_signed",
                            "spo_error_absolute",
                            "amjj_obs",
                            "amjj_sim",
                            "amjj_error_signed",
                            "amjj_error_absolute",
                            "spo_status",
                        )
                    }
                )
            year_rows.append(row)
    years = pd.DataFrame(year_rows)
    basin_level, effects = signature_tables_from_years(years)
    return years, basin_level, effects


def primary_signature_rows(
    basin_level: pd.DataFrame, minimum_years: int = 5
) -> pd.DataFrame:
    """Make one basin record after dPL seed-wise signature calculation."""
    valid = basin_level[basin_level["valid_years"] >= minimum_years].copy()
    numeric = [
        "valid_years",
        "ct_obs",
        "ct_sim",
        "ct_error_signed",
        "ct_error_absolute",
        "amjj_obs",
        "amjj_sim",
        "amjj_error_signed",
        "amjj_error_absolute",
    ]
    rows = []
    for keys, group in valid.groupby(
        ["basin_id", "paradigm", "model", "period"], sort=True
    ):
        basin, paradigm, model, period = keys
        row = {
            "basin_id": basin,
            "paradigm": paradigm,
            "model": model,
            "period": period,
            "seed_or_restart": "median_across_seeds"
            if paradigm == "dPL-MLP"
            else "selected_restart",
            "status": "primary_minimum_years",
            "analysis_set": f"minimum_{minimum_years}_years",
        }
        for column in numeric:
            row[column] = (
                float(np.median(group[column].dropna()))
                if group[column].notna().any()
                else math.nan
            )
        for column in ("spo_obs", "spo_sim", "spo_error_signed", "spo_error_absolute"):
            row[column] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def signature_effects_for_minimum(
    years: pd.DataFrame, minimum_years: int, analysis_set: str
) -> pd.DataFrame:
    valid = years[years["status"].eq("valid_ct_amjj_spo_unresolved")].copy()
    model_rows = []
    for keys, group in valid.groupby(
        ["basin_id", "paradigm", "model", "seed_or_restart", "period"], sort=True
    ):
        basin, paradigm, model, run, period = keys
        if len(group) < minimum_years:
            continue
        model_rows.append(
            {
                "basin_id": basin,
                "paradigm": paradigm,
                "model": model,
                "seed_or_restart": run,
                "period": period,
                "valid_years": int(len(group)),
                "ct_error_absolute": float(group["ct_error_absolute"].median()),
                "amjj_error_absolute": float(group["amjj_error_absolute"].median()),
                "analysis_set": analysis_set,
            }
        )
    if not model_rows:
        return pd.DataFrame(
            columns=[
                "basin_id",
                "paradigm",
                "period",
                "seed_or_restart",
                "signature",
                "first_model",
                "second_model",
                "effect_value",
                "status",
                "analysis_set",
                "valid_years",
            ]
        )
    model_level = pd.DataFrame(model_rows)
    rows = []
    for keys, group in model_level.groupby(
        ["basin_id", "paradigm", "period", "seed_or_restart"], sort=True
    ):
        basin, paradigm, period, run = keys
        for signature, error_column in (
            ("CT", "ct_error_absolute"),
            ("AMJJ", "amjj_error_absolute"),
        ):
            wide = group.set_index("model")[error_column]
            years_by_model = group.set_index("model")["valid_years"]
            for first, second in (
                ("XAJ-Base", "XAJ-CN"),
                ("XAJ-Base", "XAJ-TGD"),
                ("XAJ-TGD", "XAJ-CN"),
            ):
                if first not in wide or second not in wide:
                    continue
                value = float(wide[first] - wide[second])
                rows.append(
                    {
                        "basin_id": basin,
                        "paradigm": paradigm,
                        "period": period,
                        "seed_or_restart": run,
                        "signature": signature,
                        "first_model": first,
                        "second_model": second,
                        "effect_value": value,
                        "status": "valid",
                        "analysis_set": analysis_set,
                        "valid_years": int(
                            min(years_by_model[first], years_by_model[second])
                        ),
                    }
                )
    return pd.DataFrame(rows)


def primary_signature_effects(
    effects: pd.DataFrame, minimum_years: int = 5
) -> pd.DataFrame:
    """Collapse dPL signature reductions across seeds after basin matching."""
    if effects.empty:
        return effects.copy()
    rows = []
    for keys, group in effects.groupby(
        ["basin_id", "paradigm", "period", "signature", "first_model", "second_model"],
        sort=True,
    ):
        basin, paradigm, period, signature, first_model, second_model = keys
        values = group["effect_value"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        incoming_set = (
            str(group.get("analysis_set", pd.Series(["primary_min5"])).iloc[0])
            if len(group)
            else "primary_min5"
        )
        output_set = incoming_set.replace("_seed", "")
        rows.append(
            {
                "basin_id": basin,
                "paradigm": paradigm,
                "period": period,
                "seed_or_restart": "median_across_seeds"
                if paradigm == "dPL-MLP"
                else "selected_restart",
                "signature": signature,
                "first_model": first_model,
                "second_model": second_model,
                "effect_value": float(np.median(values)) if len(values) else math.nan,
                "status": "primary_minimum_years"
                if len(values)
                else "missing_required_member",
                "analysis_set": output_set,
                "valid_years": int(np.nanmin(group["valid_years"]))
                if "valid_years" in group and group["valid_years"].notna().any()
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def signature_summary_tables(
    primary_signatures: pd.DataFrame,
    primary_effects: pd.DataFrame,
    seed: int,
    analysis_set: str = "primary_min5",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize CT and AMJJ errors; retain unresolved SPO status rows."""
    rng = np.random.default_rng(seed)
    absolute, paired, bootstrap, tests = [], [], [], []
    for keys, group in primary_signatures.groupby(
        ["paradigm", "model", "period"], sort=True
    ):
        paradigm, model, period = keys
        for signature, column in (
            ("CT", "ct_error_absolute"),
            ("AMJJ", "amjj_error_absolute"),
        ):
            stats = summary(
                group[column],
                np.random.default_rng(
                    _key_seed(
                        seed, "signature_summary", paradigm, model, period, signature
                    )
                ),
            )
            absolute.append(
                {
                    "paradigm": paradigm,
                    "model": model,
                    "period": period,
                    "metric": f"{signature}_error_absolute",
                    "analysis_set": analysis_set,
                    "status": f"valid_complete_water_years_{analysis_set}",
                    **stats,
                }
            )
            bootstrap.append(
                {
                    "record_type": "bootstrap",
                    "family": "absolute_signature_error",
                    "paradigm": paradigm,
                    "model": model,
                    "period": period,
                    "metric": f"{signature}_error_absolute",
                    "analysis_set": analysis_set,
                    "n": stats["valid_basin_count"],
                    "ci_low": stats["bootstrap_ci_low"],
                    "ci_high": stats["bootstrap_ci_high"],
                    "resamples": 10000,
                    "seed": seed,
                    "method": "ordinary_basin_bootstrap",
                    "status": f"valid_complete_water_years_{analysis_set}",
                }
            )
        for signature in ("SPO",):
            absolute.append(
                {
                    "paradigm": paradigm,
                    "model": model,
                    "period": period,
                    "metric": "SPO_error_absolute",
                    "analysis_set": analysis_set,
                    "status": "unresolved_definition",
                    **summary([], rng),
                }
            )
    if not primary_effects.empty:
        for keys, group in primary_effects.groupby(
            ["paradigm", "period", "signature", "first_model", "second_model"],
            sort=True,
        ):
            paradigm, period, signature, first_model, second_model = keys
            stats = summary(
                group["effect_value"],
                np.random.default_rng(
                    _key_seed(
                        seed,
                        "signature_effect_summary",
                        paradigm,
                        period,
                        signature,
                        first_model,
                        second_model,
                    )
                ),
            )
            metric = f"{signature}_error_reduction"
            effect = f"{first_model}_minus_{second_model}"
            common = {
                "paradigm": paradigm,
                "model": f"{first_model}_vs_{second_model}",
                "period": period,
                "effect": effect,
                "metric": metric,
                "analysis_set": analysis_set,
            }
            paired.append(
                {
                    **common,
                    "status": f"valid_complete_water_years_{analysis_set}",
                    **stats,
                }
            )
            bootstrap.append(
                {
                    "record_type": "bootstrap",
                    "family": "paired_signature_effect",
                    **common,
                    "n": stats["valid_basin_count"],
                    "ci_low": stats["bootstrap_ci_low"],
                    "ci_high": stats["bootstrap_ci_high"],
                    "resamples": 10000,
                    "seed": seed,
                    "method": "ordinary_basin_bootstrap",
                    "status": f"valid_complete_water_years_{analysis_set}",
                }
            )
            result = test_result(
                group["effect_value"].to_numpy(dtype=float),
                common,
                "paired_signature_effect",
            )
            result["record_type"] = "test"
            tests.append(result)
    return (
        pd.DataFrame(absolute),
        pd.DataFrame(paired),
        pd.DataFrame(bootstrap),
        pd.DataFrame(tests),
    )


def _summary_effect_rows(
    effects: pd.DataFrame,
    seed: int,
    family: str = "paired_effect",
    method: str = "ordinary_basin_bootstrap",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if effects.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    group_columns = [
        column
        for column in (
            "paradigm",
            "model",
            "period",
            "effect",
            "metric",
            "analysis_set",
            "robustness_type",
            "seed_or_restart",
        )
        if column in effects.columns
    ]
    paired, bootstrap, tests = [], [], []
    for keys, group in effects.groupby(group_columns, dropna=False, sort=True):
        key_values = dict(
            zip(group_columns, keys if isinstance(keys, tuple) else (keys,))
        )
        stats = summary(
            group["effect_value"],
            np.random.default_rng(
                _key_seed(
                    seed,
                    "summary_effect_rows",
                    *[key_values.get(column, "") for column in group_columns],
                )
            ),
        )
        row = {
            **key_values,
            "status": "valid_robustness_summary"
            if "robustness_type" in key_values
            else "valid_daily_recomputation",
            **stats,
        }
        paired.append(row)
        common = {
            **key_values,
            "record_type": "bootstrap",
            "family": family,
            "n": stats["valid_basin_count"],
            "ci_low": stats["bootstrap_ci_low"],
            "ci_high": stats["bootstrap_ci_high"],
            "resamples": 10000,
            "seed": seed,
            "method": method,
            "status": row["status"],
        }
        bootstrap.append(common)
        test = test_result(
            group["effect_value"].to_numpy(float),
            {
                k: v
                for k, v in key_values.items()
                if k
                in (
                    "paradigm",
                    "model",
                    "period",
                    "effect",
                    "metric",
                    "analysis_set",
                    "robustness_type",
                    "seed_or_restart",
                )
            },
            family,
        )
        test["record_type"] = "test"
        test["method"] = "wilcoxon_signed_rank_and_sign_test"
        tests.append(test)
    return pd.DataFrame(paired), pd.DataFrame(bootstrap), pd.DataFrame(tests)


def _claim_id(row: pd.Series) -> str:
    if row.get("robustness_type") == "IC_restart_median":
        return "restart_robustness"
    if row.get("robustness_type") == "dPL_seed":
        return "seed_robustness"
    effect, metric, period, model = (
        str(row.get("effect", "")),
        str(row.get("metric", "")),
        str(row.get("period", "")),
        str(row.get("model", "")),
    )
    if metric == "kge" and effect == "CN-Base" and period == "train":
        return "calibration_masking_cn_base"
    if metric == "kge" and effect == "CN-TGD" and period == "train":
        return "calibration_masking_cn_tgd"
    if effect == "E_CN-Base":
        return "generalization_exposure_cn_base"
    if effect == "E_CN-TGD":
        return "generalization_exposure_cn_tgd"
    if effect == "IC_minus_dPL" and model == "XAJ-Base":
        return "ic_dpl_transfer_base"
    if effect == "IC_minus_dPL" and model == "XAJ-TGD":
        return "ic_dpl_transfer_tgd"
    if effect == "IC_minus_dPL" and model == "XAJ-CN":
        return "ic_dpl_transfer_cn"
    if metric == "kge" and effect == "CN-TGD" and period == "test":
        return "snow_specificity_cn_tgd"
    if metric == "CT_error_reduction" and effect == "XAJ-TGD_minus_XAJ-CN":
        return "ct_repair_cn_tgd"
    if metric == "AMJJ_error_reduction" and effect == "XAJ-TGD_minus_XAJ-CN":
        return "amjj_repair_cn_tgd"
    return ""


def add_claim_columns(summary_table: pd.DataFrame) -> pd.DataFrame:
    if summary_table.empty:
        return summary_table
    table = summary_table.copy()
    table["claim_id"] = table.apply(_claim_id, axis=1)
    table["estimand"] = table.get("estimand", "")
    table["estimate"] = table.get("median", np.nan)
    table["ci_lower_bound"] = table.get("bootstrap_ci_low", np.nan)
    table["ci_upper_bound"] = table.get("bootstrap_ci_high", np.nan)
    table["positive_fraction"] = table.get("fraction_positive", np.nan)
    table["statistical_method"] = np.where(
        table.get("record_type", "").eq("test")
        if hasattr(table.get("record_type", ""), "eq")
        else False,
        "Wilcoxon signed-rank and sign test",
        "paired basin bootstrap mean CI",
    )
    table["support_status"] = [
        support_status(low, high)
        for low, high in zip(table["ci_lower_bound"], table["ci_upper_bound"])
    ]
    return table


def region_membership(
    data_root: Path, basin_ids: Iterable[str]
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np

    requested = {str(value).zfill(8) for value in basin_ids}
    mapping: dict[str, str] = {}
    source = []
    for region in range(7):
        path = data_root / "basin_groups" / f"group_{11 + region}.npy"
        if not path.exists():
            return {}, {
                "status": "unavailable",
                "reason": f"missing authoritative region file: {path}",
            }
        source.append(str(path))
        for basin in np.load(path).astype(int):
            basin_id = str(basin).zfill(8)
            if basin_id in requested:
                mapping[basin_id] = f"region_{region}"
    if set(mapping) != requested:
        return mapping, {
            "status": "incomplete",
            "reason": "not all R1 basins map to the seven documented LORO regions",
            "matched_basins": len(mapping),
            "source_files": source,
        }
    return mapping, {
        "status": "available",
        "source_files": source,
        "definition": "group_11..group_17 are the seven LORO regions documented by project/flexmopex/run_model.py",
        "region_count": len(set(mapping.values())),
    }


def region_block_bootstrap_rows(
    effect_frames: list[tuple[pd.DataFrame, str]],
    basin_regions: dict[str, str],
    seed: int,
) -> pd.DataFrame:
    rows = []
    if not basin_regions:
        return pd.DataFrame()
    for frame, family in effect_frames:
        if frame.empty:
            continue
        current = frame.copy()
        current["region"] = (
            current["basin_id"].astype(str).str.zfill(8).map(basin_regions)
        )
        current = current[
            current["region"].notna() & np.isfinite(current["effect_value"])
        ]
        group_columns = [
            column
            for column in (
                "paradigm",
                "model",
                "period",
                "effect",
                "metric",
                "analysis_set",
            )
            if column in current.columns
        ]
        for keys, group in current.groupby(group_columns, dropna=False, sort=True):
            key_values = dict(
                zip(group_columns, keys if isinstance(keys, tuple) else (keys,))
            )
            rng = np.random.default_rng(seed + len(rows))
            ci_low, ci_high = block_bootstrap_mean_ci(
                group["effect_value"], group["region"], rng
            )
            rows.append(
                {
                    "record_type": "bootstrap",
                    "family": family,
                    **key_values,
                    "n": int(group["basin_id"].nunique()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "resamples": 10000,
                    "seed": seed,
                    "method": "region_block_bootstrap",
                    "block_count": int(group["region"].nunique()),
                    "block_source": "data/basin_groups/group_11..group_17.npy",
                    "status": "valid_region_block_bootstrap",
                }
            )
    return pd.DataFrame(rows)


def restart_robustness_effects(restart_table: pd.DataFrame) -> pd.DataFrame:
    if restart_table.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in restart_table.groupby(["basin_id", "start"], sort=True):
        basin, start = keys
        wide = group.set_index("model")
        for effect_name, enhanced, base in STRUCTURAL_EFFECT_SPECS:
            if enhanced not in wide.index or base not in wide.index:
                continue
            test_value = float(
                wide.loc[enhanced, "test_kge"] - wide.loc[base, "test_kge"]
            )
            rows.append(
                {
                    "basin_id": basin,
                    "start": start,
                    "period": "test",
                    "effect": effect_name,
                    "effect_value": test_value,
                    "metric": "kge",
                    "analysis_set": "median_across_restarts",
                    "robustness_type": "IC_restart_median",
                }
            )
            train_value = float(
                wide.loc[enhanced, "train_kge"] - wide.loc[base, "train_kge"]
            )
            rows.append(
                {
                    "basin_id": basin,
                    "start": start,
                    "period": "train",
                    "effect": effect_name,
                    "effect_value": train_value,
                    "metric": "kge",
                    "analysis_set": "median_across_restarts",
                    "robustness_type": "IC_restart_median",
                }
            )
            rows.append(
                {
                    "basin_id": basin,
                    "start": start,
                    "period": "test_minus_train",
                    "effect": f"E_{effect_name}",
                    "effect_value": test_value - train_value,
                    "metric": "kge",
                    "analysis_set": "median_across_restarts",
                    "robustness_type": "IC_restart_median",
                }
            )
    return pd.DataFrame(rows)


def seed_robustness_effects(
    effects: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if effects.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rows = []
    for keys, group in effects.groupby(
        [
            column
            for column in ("paradigm", "model", "period", "effect", "metric")
            if column in effects.columns and column != "seed_or_restart"
        ],
        sort=True,
    ):
        key_values = dict(
            zip(
                [
                    column
                    for column in ("paradigm", "model", "period", "effect", "metric")
                    if column in effects.columns and column != "seed_or_restart"
                ],
                keys if isinstance(keys, tuple) else (keys,),
            )
        )
        seed_estimates = (
            group.groupby("seed_or_restart")["effect_value"].median().dropna()
        )
        if seed_estimates.empty:
            continue
        rng = np.random.default_rng(seed)
        for run, estimate in seed_estimates.items():
            values = group[group["seed_or_restart"].eq(run)]["effect_value"]
            stats = summary(values, rng)
            rows.append(
                {
                    **key_values,
                    "seed_or_restart": run,
                    "robustness_type": "dPL_seed",
                    "analysis_set": "individual_seed",
                    "effect_value": float(estimate),
                    "estimate": float(estimate),
                    "seed_min": float(seed_estimates.min()),
                    "seed_max": float(seed_estimates.max()),
                    "same_sign_count": math.nan,
                    "all_seeds_agree": math.nan,
                    "status": "seed_specific",
                    **stats,
                }
            )
        primary_sign = np.sign(np.nanmedian(seed_estimates.to_numpy(float)))
        rows.append(
            {
                **key_values,
                "seed_or_restart": "all_seeds",
                "robustness_type": "dPL_seed",
                "analysis_set": "across_seed_summary",
                "effect_value": float(seed_estimates.median()),
                "estimate": float(seed_estimates.median()),
                "seed_min": float(seed_estimates.min()),
                "seed_max": float(seed_estimates.max()),
                "same_sign_count": int(
                    (np.sign(seed_estimates.to_numpy(float)) == primary_sign).sum()
                ),
                "all_seeds_agree": bool(
                    (np.sign(seed_estimates.to_numpy(float)) == primary_sign).all()
                ),
                "valid_basin_count": int(group["basin_id"].nunique()),
                "bootstrap_ci_low": math.nan,
                "bootstrap_ci_high": math.nan,
                "status": "across_seed_descriptive",
                "median": float(seed_estimates.median()),
                "p25": float(seed_estimates.quantile(0.25)),
                "p75": float(seed_estimates.quantile(0.75)),
                "mean": float(seed_estimates.mean()),
                "sd": float(seed_estimates.std(ddof=1))
                if len(seed_estimates) > 1
                else math.nan,
                "minimum": float(seed_estimates.min()),
                "maximum": float(seed_estimates.max()),
                "fraction_positive": float((seed_estimates > 0).mean()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame()


def build_statistics_from_daily(
    output: Path,
    data_root: Path,
    results_root: Path,
    random_seed: int = 20260730,
    precomputed: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    if precomputed is None:
        metrics = daily_metric_rows(output)
    else:
        metrics = precomputed["metrics"].copy()
    ic = metrics[metrics.paradigm.eq("IC-CMA-ES")].copy()
    dpl_seed = metrics[metrics.paradigm.eq("dPL-MLP")].copy()
    dpl_primary = median_seed_metrics(dpl_seed)
    primary = pd.concat([ic, dpl_primary], ignore_index=True, sort=False)
    seed_level = dpl_seed
    ic_structural = pivot_effects(ic, "IC-CMA-ES")
    dpl_structural_seed = pivot_effects(dpl_seed, "dPL-MLP")
    structural = pd.concat(
        [ic_structural, median_seed_effects(dpl_structural_seed)],
        ignore_index=True,
        sort=False,
    )
    ic_generalization = generalization_effects(ic, "IC-CMA-ES")
    dpl_generalization_seed = generalization_effects(dpl_seed, "dPL-MLP")
    generalization = pd.concat(
        [ic_generalization, median_seed_effects(dpl_generalization_seed)],
        ignore_index=True,
        sort=False,
    )
    transfer = ic_dpl_effects(ic, dpl_seed)
    attributes, _attribute_meta = snow_attributes(
        data_root, data_root / "531sub_id.txt"
    )
    if precomputed is None:
        years, signatures, signature_effects = signature_rows(output)
    else:
        years = precomputed["signature_years"].copy()
        signatures = precomputed["signature_basin_level"].copy()
        signature_effects = precomputed["signature_effects"].copy()
    signature_effects_seed_min5 = signature_effects_for_minimum(
        years, 5, "primary_min5_seed"
    )
    signature_effects_primary = primary_signature_effects(signature_effects_seed_min5)
    signature_effects_seed_min3 = signature_effects_for_minimum(
        years, 3, "sensitivity_min3_seed"
    )
    signature_effects_min3 = primary_signature_effects(signature_effects_seed_min3)
    relationships = snow_relationships(
        structural, attributes, signature_effects_primary
    )
    restart_tables = []
    ic_roots = {
        "XAJ": "xaj_base_cmaes_531_batched_paired_v2",
        "XAJ_TGD2": "xaj_tgd2_cmaes_531_batched_v1",
        "XAJ_CN": "xaj_cn_cmaes_531_batched_paired_v2",
    }
    for model_key, directory in ic_roots.items():
        _selected, restart_level, _meta = load_ic(results_root / directory, model_key)
        model = MODEL_ALIASES[model_key]
        selected_daily = ic[(ic.model == model) & ic.selected_run]
        for period in ("train", "test"):
            values = selected_daily[selected_daily.period.eq(period)].set_index(
                "basin_id"
            )["kge"]
            restart_level[f"recomputed_{period}_kge"] = restart_level.basin_id.map(
                values
            )
            restart_level[f"median_restart_{period}_kge"] = restart_level.groupby(
                "basin_id"
            )[f"{period}_kge"].transform("median")
        restart_level["recomputed_metric"] = "repository_standard_KGE_from_daily"
        restart_tables.append(restart_level)
    all_effects = pd.concat(
        [structural, generalization, transfer], ignore_index=True, sort=False
    )
    absolute, paired, combined = write_summary_tables(
        primary, all_effects, output, random_seed, metric_col="kge"
    )
    primary_signatures = primary_signature_rows(signatures, minimum_years=5)
    sensitivity_signatures = primary_signature_rows(signatures, minimum_years=3)
    signature_absolute, signature_paired, signature_bootstrap, signature_tests = (
        signature_summary_tables(
            primary_signatures, signature_effects_primary, random_seed, "primary_min5"
        )
    )
    (
        sensitivity_absolute,
        sensitivity_paired,
        sensitivity_bootstrap,
        sensitivity_tests,
    ) = signature_summary_tables(
        sensitivity_signatures, signature_effects_min3, random_seed, "sensitivity_min3"
    )
    absolute = pd.concat(
        [absolute, signature_absolute, sensitivity_absolute],
        ignore_index=True,
        sort=False,
    )
    paired = pd.concat(
        [paired, signature_paired, sensitivity_paired], ignore_index=True, sort=False
    )

    restart_table = pd.concat(restart_tables, ignore_index=True, sort=False)
    restart_effects = restart_robustness_effects(restart_table)
    restart_effects = restart_effects.groupby(
        ["basin_id", "period", "effect", "metric", "analysis_set", "robustness_type"],
        sort=True,
        as_index=False,
    )["effect_value"].median()
    restart_paired, restart_bootstrap, restart_tests = _summary_effect_rows(
        restart_effects,
        random_seed,
        family="restart_robustness",
        method="ordinary_basin_bootstrap",
    )
    unavailable_restart = pd.DataFrame(
        [
            {
                "paradigm": "IC-CMA-ES",
                "model": "XAJ-TGD_vs_XAJ-CN",
                "period": "test",
                "effect": "XAJ-TGD_minus_XAJ-CN",
                "metric": metric,
                "analysis_set": "median_across_restarts",
                "robustness_type": "IC_restart_median",
                "status": "unavailable_no_restart_daily_signatures",
                "valid_basin_count": 0,
                "median": math.nan,
                "p25": math.nan,
                "p75": math.nan,
                "mean": math.nan,
                "sd": math.nan,
                "minimum": math.nan,
                "maximum": math.nan,
                "fraction_positive": math.nan,
                "bootstrap_ci_low": math.nan,
                "bootstrap_ci_high": math.nan,
            }
            for metric in ("CT_error_reduction", "AMJJ_error_reduction")
        ]
    )
    seed_signature = signature_effects_seed_min5[
        signature_effects_seed_min5["paradigm"].eq("dPL-MLP")
        & signature_effects_seed_min5["first_model"].eq("XAJ-TGD")
        & signature_effects_seed_min5["second_model"].eq("XAJ-CN")
    ].copy()
    if not seed_signature.empty:
        seed_signature["model"] = "XAJ-TGD_vs_XAJ-CN"
        seed_signature["effect"] = seed_signature["signature"].map(
            lambda value: "XAJ-TGD_minus_XAJ-CN"
        )
        seed_signature["metric"] = seed_signature["signature"].map(
            lambda value: f"{value}_error_reduction"
        )
        seed_signature = seed_signature.rename(
            columns={"analysis_set": "analysis_set_seed"}
        )
        seed_signature["analysis_set"] = "primary_min5"
    seed_input = pd.concat(
        [
            dpl_structural_seed[
                (dpl_structural_seed["period"].eq("test"))
                & dpl_structural_seed["effect"].isin(["CN-Base", "TGD-Base", "CN-TGD"])
            ],
            dpl_generalization_seed[
                dpl_generalization_seed["effect"].isin(["E_CN-Base", "E_CN-TGD"])
            ],
            seed_signature,
        ],
        ignore_index=True,
        sort=False,
    )
    seed_robustness, _, _ = seed_robustness_effects(seed_input, random_seed)

    basin_regions, region_meta = region_membership(
        data_root, primary["basin_id"].astype(str).tolist()
    )
    signature_region = signature_effects_primary[
        (signature_effects_primary["first_model"].eq("XAJ-TGD"))
        & (signature_effects_primary["second_model"].eq("XAJ-CN"))
        & (signature_effects_primary["period"].eq("test"))
    ].copy()
    signature_region["model"] = "XAJ-TGD_vs_XAJ-CN"
    signature_region["effect"] = "XAJ-TGD_minus_XAJ-CN"
    signature_region["metric"] = signature_region["signature"].map(
        lambda value: f"{value}_error_reduction"
    )
    region_bootstrap = region_block_bootstrap_rows(
        [
            (
                structural[
                    (structural["period"].eq("test"))
                    & structural["effect"].isin(["CN-Base", "CN-TGD"])
                ],
                "region_block_paired_effect",
            ),
            (
                generalization[
                    generalization["effect"].isin(["E_CN-Base", "E_CN-TGD"])
                ],
                "region_block_generalization_effect",
            ),
            (signature_region, "region_block_signature_effect"),
        ],
        basin_regions,
        random_seed,
    )

    restart_paired["primary_estimate"] = math.nan
    restart_paired["difference_from_primary"] = math.nan
    restart_paired["same_direction_as_primary"] = pd.Series(
        index=restart_paired.index, dtype=object
    )
    seed_robustness["primary_estimate"] = math.nan
    seed_robustness["difference_from_primary"] = math.nan
    seed_robustness["same_direction_as_primary"] = pd.Series(
        index=seed_robustness.index, dtype=object
    )
    paired = pd.concat(
        [paired, restart_paired, unavailable_restart, seed_robustness],
        ignore_index=True,
        sort=False,
    )
    baseline = paired[
        paired.get("robustness_type", pd.Series(index=paired.index)).isna()
    ].copy()
    for index, row in paired[
        paired.get("robustness_type", pd.Series(index=paired.index)).notna()
    ].iterrows():
        if row.get("status") == "unavailable_no_restart_daily_signatures":
            continue
        candidates = baseline[
            (
                baseline["paradigm"]
                == (
                    "IC-CMA-ES"
                    if row["robustness_type"] == "IC_restart_median"
                    else "dPL-MLP"
                )
            )
            & (baseline["period"].astype(str) == str(row.get("period")))
            & (baseline["effect"].astype(str) == str(row.get("effect")))
            & (baseline["metric"].astype(str) == str(row.get("metric")))
        ]
        candidates = candidates[
            ~candidates.get("analysis_set", pd.Series(index=candidates.index)).isin(
                [
                    "sensitivity_min3",
                    "individual_seed",
                    "across_seed_summary",
                    "median_across_restarts",
                ]
            )
        ]
        if len(candidates):
            primary_estimate = float(candidates.iloc[0]["median"])
            estimate_value = row.get("estimate", math.nan)
            estimate = (
                float(estimate_value)
                if np.isfinite(estimate_value)
                else float(row.get("median", math.nan))
            )
            paired.loc[index, "primary_estimate"] = primary_estimate
            paired.loc[index, "difference_from_primary"] = estimate - primary_estimate
            paired.loc[index, "same_direction_as_primary"] = (
                bool(np.sign(estimate) == np.sign(primary_estimate))
                if np.isfinite(estimate) and np.isfinite(primary_estimate)
                else math.nan
            )
    paired = add_claim_columns(paired)
    paired["primary_estimate"] = paired.get("primary_estimate", math.nan)
    paired["difference_from_primary"] = paired.get("difference_from_primary", math.nan)
    paired["same_direction_as_primary"] = paired.get(
        "same_direction_as_primary", math.nan
    )
    for name, table in {
        "r1_basin_level_performance.csv": primary,
        "r1_ic_restart_level_performance.csv": restart_table,
        "r1_dpl_seed_level_performance.csv": seed_level,
        "r1_structural_effects_basin_level.csv": structural,
        "r1_generalization_effects_basin_level.csv": pd.concat(
            [structural, generalization, transfer], ignore_index=True, sort=False
        ),
        "r1_snow_attributes.csv": attributes,
        "r1_snow_relationships_summary.csv": relationships,
        "r1_snow_signatures_basin_year.csv": years,
        "r1_snow_signatures_basin_level.csv": signatures,
        "r1_signature_effects_basin_level.csv": pd.concat(
            [
                signature_effects,
                signature_effects_seed_min5,
                signature_effects_primary,
                signature_effects_seed_min3,
                signature_effects_min3,
            ],
            ignore_index=True,
            sort=False,
        ),
        "r1_absolute_metrics_summary.csv": absolute,
        "r1_paired_effects_summary.csv": paired,
    }.items():
        table.to_csv(output / name, index=False, na_rep="")
    bootstrap = pd.concat(
        [
            combined[combined["record_type"].eq("bootstrap")],
            signature_bootstrap,
            sensitivity_bootstrap,
            restart_bootstrap,
            region_bootstrap,
        ],
        ignore_index=True,
        sort=False,
    )
    tests = pd.concat(
        [
            combined[combined["record_type"].eq("test")],
            signature_tests,
            sensitivity_tests,
            restart_tests,
        ],
        ignore_index=True,
        sort=False,
    )
    tests["bh_family"] = np.where(
        tests["effect"].astype(str).str.startswith("E_"),
        "generalization_effect",
        tests.get("family", "paired_effect"),
    )
    tests["wilcoxon_bh_p"] = math.nan
    tests["sign_test_bh_p"] = math.nan
    for family, indexes in tests.groupby("bh_family", dropna=False).groups.items():
        tests.loc[indexes, "wilcoxon_bh_p"] = bh_adjust(
            tests.loc[indexes, "wilcoxon_p"].tolist()
        )
        tests.loc[indexes, "sign_test_bh_p"] = bh_adjust(
            tests.loc[indexes, "sign_test_p"].tolist()
        )
    tests["claim_id"] = tests.apply(_claim_id, axis=1)
    tests["support_status"] = "descriptive_only"
    bootstrap["method"] = bootstrap.get("method", "ordinary_basin_bootstrap")
    bootstrap["claim_id"] = (
        bootstrap.apply(_claim_id, axis=1)
        if not bootstrap.empty
        else pd.Series(dtype=object)
    )
    bootstrap.to_csv(output / "r1_bootstrap_intervals.csv", index=False, na_rep="")
    tests.to_csv(output / "r1_statistical_tests.csv", index=False, na_rep="")
    exclusions = pd.DataFrame(
        [
            {
                "item": "SPO",
                "status": "unresolved",
                "reason": "R1 plan defines cumulative-departure SPO but does not fix t0, search window, or no-pulse handling; no guess was made.",
            },
            {
                "item": "AMJJ primary threshold",
                "status": "documented",
                "reason": "Primary basin-level AMJJ requires at least five complete water years; three-year sensitivity is retained in valid_years metadata.",
            },
            {
                "item": "daily inference",
                "status": "complete",
                "reason": "Existing daily simulations and online partition summaries were reused; this extension launched no inference.",
            },
            {
                "item": "region block bootstrap",
                "status": region_meta.get("status"),
                "reason": region_meta.get(
                    "definition", region_meta.get("reason", "unavailable")
                ),
            },
            {
                "item": "IC restart signature robustness",
                "status": "unavailable",
                "reason": "Restart-level tables contain KGE records but no non-selected-restart daily CT/AMJJ series; no signature restart estimate was fabricated.",
            },
        ]
    )
    exclusions.to_csv(output / "r1_exclusion_log.csv", index=False)
    return {
        "metrics": metrics,
        "primary": primary,
        "seed_level": seed_level,
        "signatures": signatures,
        "signature_effects": signature_effects,
        "exclusions": exclusions,
        "attribute_meta": _attribute_meta,
        "region_meta": region_meta,
        "relationships": relationships,
    }


def _normalise_basin_ids(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "basin_id" in frame:
        frame["basin_id"] = (
            frame["basin_id"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(8)
        )
    return frame


def _median_statistics(
    values: Iterable[float], rng: np.random.Generator
) -> dict[str, float]:
    values = np.asarray(list(values), dtype=float)
    values = np.sort(values[np.isfinite(values)])
    low, high = bootstrap_median_ci(values, rng)
    return {
        "valid_basin_count": int(values.size),
        "median": float(np.median(values)) if values.size else math.nan,
        "p25": float(np.percentile(values, 25)) if values.size else math.nan,
        "p75": float(np.percentile(values, 75)) if values.size else math.nan,
        "mean": float(values.mean()) if values.size else math.nan,
        "sd": float(values.std(ddof=1)) if values.size > 1 else math.nan,
        "minimum": float(values.min()) if values.size else math.nan,
        "maximum": float(values.max()) if values.size else math.nan,
        "fraction_positive": float((values > 0).mean()) if values.size else math.nan,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
    }


def _set_summary_metadata(
    frame: pd.DataFrame, *, role: str = "sensitivity"
) -> pd.DataFrame:
    frame = frame.copy()
    frame["summary_level"] = frame.get("summary_level", "full_sample")
    frame["result_role"] = frame.get("result_role", role)
    return frame


def _primary_effect_source(
    frame: pd.DataFrame, paradigm: str, effect: str, period: str
) -> pd.DataFrame:
    subset = frame[(frame["paradigm"] == paradigm) & (frame["effect"] == effect)]
    if "period" in subset:
        subset = subset[subset["period"].astype(str) == str(period)]
    if "seed_or_restart" in subset and paradigm == "dPL-MLP":
        median_rows = subset[
            subset["seed_or_restart"].astype(str).eq("median_across_seeds")
        ]
        if not median_rows.empty:
            subset = median_rows
    subset = subset[np.isfinite(pd.to_numeric(subset["effect_value"], errors="coerce"))]
    return subset.drop_duplicates(["basin_id"])


def _transfer_difference_frame(
    generalization: pd.DataFrame, model: str
) -> pd.DataFrame:
    subset = generalization[
        generalization["paradigm"].eq("IC-dPL")
        & generalization["effect"].eq("IC_minus_dPL")
        & generalization["model"].eq(model)
    ].copy()
    subset["effect_value"] = pd.to_numeric(subset["effect_value"], errors="coerce")
    wide = subset.pivot_table(
        index="basin_id", columns="period", values="effect_value", aggfunc="first"
    )
    if not {"train", "test"}.issubset(wide.columns):
        return pd.DataFrame(columns=["basin_id", "effect_value"])
    result = wide.assign(effect_value=wide["test"] - wide["train"]).reset_index()[
        ["basin_id", "effect_value"]
    ]
    return result[np.isfinite(result["effect_value"])]


def _claim_for_effect(
    effect: str, metric: str, period: str, paradigm: str, model: str
) -> str:
    if metric == "kge" and effect == "CN-Base" and period == "train":
        return "calibration_masking_cn_base"
    if metric == "kge" and effect == "CN-TGD" and period == "train":
        return "calibration_masking_cn_tgd"
    if metric == "kge" and effect == "CN-TGD" and period == "test":
        return "snow_specificity_cn_tgd"
    if effect == "E_CN-Base":
        return "generalization_exposure_cn_base"
    if effect == "E_CN-TGD":
        return "generalization_exposure_cn_tgd"
    if effect == "E_TGD-Base":
        return "generalization_exposure_tgd_base"
    if effect.startswith("D_"):
        return f"ic_dpl_transfer_{effect.removeprefix('D_').lower()}"
    if metric == "CT_error_reduction" and effect == "R_CN-TGD":
        return "ct_repair_cn_tgd"
    if metric == "AMJJ_error_reduction" and effect == "R_CN-TGD":
        return "amjj_repair_cn_tgd"
    if metric == "CT_error_reduction":
        return f"ct_repair_{effect.removeprefix('R_').lower().replace('-', '_')}"
    if metric == "AMJJ_error_reduction":
        return f"amjj_repair_{effect.removeprefix('R_').lower().replace('-', '_')}"
    if metric == "kge" and effect == "train_minus_test":
        return "train_test_gap"
    if metric == "kge" and effect == "CN-Base":
        return "structural_test_cn_base"
    if metric == "kge" and effect == "TGD-Base":
        return "structural_test_tgd_base"
    return f"{paradigm}_{model}_{effect}_{period}".lower().replace(" ", "_")


def _effect_bundle(
    frame: pd.DataFrame,
    *,
    paradigm: str,
    model: str,
    period: str,
    effect: str,
    metric: str,
    estimand: str,
    family: str,
    claim_id: str,
    rng: np.random.Generator,
    basin_regions: dict[str, str],
    aggregation_rule: str,
    analysis_set: str = FULL_SAMPLE,
    summary_level: str = "full_sample",
    result_role: str = "primary",
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    extra_metadata = extra_metadata or {}
    values = pd.to_numeric(frame["effect_value"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    stable_rng = np.random.default_rng(
        _key_seed(
            20260730,
            analysis_set,
            extra_metadata.get("snow_stratum", ""),
            paradigm,
            model,
            period,
            effect,
            metric,
            claim_id,
        )
    )
    stats = _median_statistics(values, stable_rng)
    tests = test_result(
        values,
        {
            "paradigm": paradigm,
            "model": model,
            "period": period,
            "effect": effect,
            "metric": metric,
            "analysis_set": FULL_SAMPLE,
        },
        family,
    )
    key = f"{analysis_set}|{extra_metadata.get('snow_stratum', '')}|{paradigm}|{model}|{period}|{effect}|{metric}|{claim_id}"
    region_low = region_high = math.nan
    region_frame = frame.copy()
    region_frame["region"] = region_frame["basin_id"].astype(str).map(basin_regions)
    region_frame = region_frame[region_frame["region"].notna()]
    if not region_frame.empty:
        region_low, region_high = block_bootstrap_median_ci(
            region_frame["effect_value"],
            region_frame["region"],
            np.random.default_rng(
                _key_seed(
                    20260730,
                    "region",
                    analysis_set,
                    extra_metadata.get("snow_stratum", ""),
                    paradigm,
                    model,
                    period,
                    effect,
                    metric,
                    claim_id,
                )
            ),
        )
    row = {
        "paradigm": paradigm,
        "model": model,
        "period": period,
        "effect": effect,
        "metric": metric,
        "status": "valid_full_sample",
        "analysis_set": analysis_set,
        "summary_level": summary_level,
        "result_role": result_role,
        "estimand": estimand,
        "claim_id": claim_id,
        "aggregation_rule": aggregation_rule,
        "bootstrap_statistic": "median",
        "regional_bootstrap_ci_low": region_low,
        "regional_bootstrap_ci_high": region_high,
        "wilcoxon_p": tests.get("wilcoxon_p", math.nan),
        "sign_test_p": tests.get("sign_test_p", math.nan),
        "support_status": support_status(
            stats["bootstrap_ci_low"], stats["bootstrap_ci_high"]
        ),
        "ci_lower_bound": stats["bootstrap_ci_low"],
        "ci_upper_bound": stats["bootstrap_ci_high"],
        "positive_fraction": stats["fraction_positive"],
        "statistical_method": "paired basin median bootstrap; Wilcoxon signed-rank and sign test",
        "_summary_key": key,
        **stats,
        **extra_metadata,
    }
    bootstrap = [
        {
            "record_type": "bootstrap",
            "family": family,
            "paradigm": paradigm,
            "model": model,
            "period": period,
            "metric": metric,
            "effect": effect,
            "analysis_set": analysis_set,
            "summary_level": summary_level,
            "result_role": result_role,
            "n": stats["valid_basin_count"],
            "ci_low": stats["bootstrap_ci_low"],
            "ci_high": stats["bootstrap_ci_high"],
            "resamples": 10000,
            "seed": 20260730,
            "method": "ordinary_basin_bootstrap_median",
            "bootstrap_statistic": "median",
            "status": "valid_full_sample",
            "claim_id": claim_id,
            **extra_metadata,
        },
    ]
    if basin_regions and not region_frame.empty:
        bootstrap.append(
            {
                "record_type": "bootstrap",
                "family": family,
                "paradigm": paradigm,
                "model": model,
                "period": period,
                "metric": metric,
                "effect": effect,
                "analysis_set": analysis_set,
                "summary_level": summary_level,
                "result_role": result_role,
                "n": int(region_frame["basin_id"].nunique()),
                "ci_low": region_low,
                "ci_high": region_high,
                "resamples": 10000,
                "seed": 20260730,
                "method": "region_block_bootstrap_median",
                "bootstrap_statistic": "median",
                "block_count": int(region_frame["region"].nunique()),
                "block_source": "data/basin_groups/group_11..group_17.npy",
                "status": "valid_region_block_bootstrap",
                "claim_id": claim_id,
                **extra_metadata,
            }
        )
    tests.update(
        {
            "record_type": "test",
            "family": family,
            "ci_low": stats["bootstrap_ci_low"],
            "ci_high": stats["bootstrap_ci_high"],
            "method": "Wilcoxon signed-rank and sign test; paired median bootstrap CI",
            "analysis_set": analysis_set,
            "summary_level": summary_level,
            "result_role": result_role,
            "claim_id": claim_id,
            "support_status": row["support_status"],
            "_summary_key": key,
            **extra_metadata,
        }
    )
    return row, bootstrap, tests


def _apply_primary_bh(tests: pd.DataFrame) -> pd.DataFrame:
    tests = tests.copy()
    if tests.empty:
        return tests
    tests["bh_family"] = tests["family"].astype(str)
    tests["wilcoxon_bh_p"] = math.nan
    tests["sign_test_bh_p"] = math.nan
    for family, indexes in tests.groupby("bh_family", dropna=False).groups.items():
        tests.loc[indexes, "wilcoxon_bh_p"] = bh_adjust(
            tests.loc[indexes, "wilcoxon_p"].tolist()
        )
        tests.loc[indexes, "sign_test_bh_p"] = bh_adjust(
            tests.loc[indexes, "sign_test_p"].tolist()
        )
    return tests


def _dpl_seed_robustness_rows(
    output: Path, primary_effects: pd.DataFrame, signature_effects: pd.DataFrame
) -> pd.DataFrame:
    """Return compact dPL seed estimates without pooling seed-by-basin records."""
    seed_performance = _normalise_basin_ids(
        pd.read_csv(output / "r1_dpl_seed_level_performance.csv")
    )
    rows: list[dict[str, Any]] = []
    specs = [
        ("CN-Base", "kge", "test", "XAJ-CN", "XAJ-Base"),
        ("TGD-Base", "kge", "test", "XAJ-TGD", "XAJ-Base"),
        ("CN-TGD", "kge", "test", "XAJ-CN", "XAJ-TGD"),
    ]
    for effect, metric, period, enhanced, base in specs:
        values_by_seed: dict[str, np.ndarray] = {}
        for seed in ("seed_42", "seed_123", "seed_2026"):
            subset = seed_performance[
                (seed_performance["seed_or_restart"] == seed)
                & (seed_performance["period"] == period)
                & seed_performance["model"].isin([enhanced, base])
            ]
            wide = subset.pivot_table(
                index="basin_id", columns="model", values="kge", aggfunc="first"
            )
            values = (wide[enhanced] - wide[base]).to_numpy(dtype=float)
            values_by_seed[seed] = values[np.isfinite(values)]
        _append_seed_rows(
            rows, values_by_seed, primary_effects, "dPL-MLP", effect, metric, period
        )
    for effect, metric, period in (
        ("E_CN-Base", "kge", "test_minus_train"),
        ("E_CN-TGD", "kge", "test_minus_train"),
    ):
        values_by_seed = {}
        for seed in ("seed_42", "seed_123", "seed_2026"):
            subset = seed_performance[
                (seed_performance["seed_or_restart"] == seed)
                & seed_performance["model"].isin(["XAJ-Base", "XAJ-CN", "XAJ-TGD"])
            ]
            wide = subset.pivot_table(
                index="basin_id",
                columns=["model", "period"],
                values="kge",
                aggfunc="first",
            )
            enhanced = "XAJ-CN" if effect == "E_CN-Base" else "XAJ-CN"
            base = "XAJ-Base" if effect == "E_CN-Base" else "XAJ-TGD"
            values = (wide[(enhanced, "test")] - wide[(base, "test")]) - (
                wide[(enhanced, "train")] - wide[(base, "train")]
            )
            values_by_seed[seed] = values.to_numpy(dtype=float)
        _append_seed_rows(
            rows, values_by_seed, primary_effects, "dPL-MLP", effect, metric, period
        )
    for signature in ("CT", "AMJJ"):
        effect = "R_CN-TGD"
        metric = f"{signature}_error_reduction"
        values_by_seed = {}
        subset = signature_effects[
            (signature_effects["paradigm"] == "dPL-MLP")
            & (signature_effects["analysis_set"] == "primary_min5_seed")
            & (signature_effects["period"] == "test")
            & (signature_effects["signature"] == signature)
            & (signature_effects["first_model"] == "XAJ-TGD")
            & (signature_effects["second_model"] == "XAJ-CN")
        ]
        for seed in ("seed_42", "seed_123", "seed_2026"):
            values_by_seed[seed] = (
                pd.to_numeric(
                    subset[subset["seed_or_restart"] == seed]["effect_value"],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(dtype=float)
            )
        _append_seed_rows(
            rows, values_by_seed, primary_effects, "dPL-MLP", effect, metric, "test"
        )
    return pd.DataFrame(rows)


def _append_seed_rows(
    rows: list[dict[str, Any]],
    values_by_seed: dict[str, np.ndarray],
    primary_effects: pd.DataFrame,
    paradigm: str,
    effect: str,
    metric: str,
    period: str,
) -> None:
    estimates = {
        seed: float(np.median(values)) if len(values) else math.nan
        for seed, values in values_by_seed.items()
    }
    finite = np.asarray(
        [value for value in estimates.values() if np.isfinite(value)], dtype=float
    )
    if not len(finite):
        return
    primary = primary_effects[
        (primary_effects["paradigm"] == paradigm)
        & (primary_effects["effect"] == effect)
        & (primary_effects["metric"] == metric)
        & (primary_effects["period"] == period)
    ]
    primary_estimate = (
        float(primary.iloc[0]["median"])
        if not primary.empty
        else float(np.median(finite))
    )
    primary_sign = np.sign(primary_estimate)
    same_count = int(np.sum(np.sign(finite) == primary_sign))
    all_agree = bool(np.all(np.sign(finite) == primary_sign))
    for seed, estimate in estimates.items():
        values = values_by_seed[seed]
        rows.append(
            {
                "paradigm": paradigm,
                "model": "seed_robustness",
                "period": period,
                "effect": effect,
                "metric": metric,
                "status": "seed_specific",
                "analysis_set": "individual_seed",
                "summary_level": "full_sample",
                "result_role": "sensitivity",
                "robustness_type": "dPL_seed",
                "seed_or_restart": seed,
                "estimate": estimate,
                "median": estimate,
                "valid_basin_count": int(len(values)),
                "seed_min": float(np.min(finite)),
                "seed_max": float(np.max(finite)),
                "same_sign_count": math.nan,
                "all_seeds_agree": math.nan,
                "primary_estimate": primary_estimate,
                "difference_from_primary": estimate - primary_estimate,
                "same_direction_as_primary": bool(np.sign(estimate) == primary_sign),
                "claim_id": "seed_robustness",
                "aggregation_rule": "basin effect within seed; no seed-by-basin pooling",
            }
        )
    rows.append(
        {
            "paradigm": paradigm,
            "model": "seed_robustness",
            "period": period,
            "effect": effect,
            "metric": metric,
            "status": "across_seed_descriptive",
            "analysis_set": "across_seed_summary",
            "summary_level": "full_sample",
            "result_role": "sensitivity",
            "robustness_type": "dPL_seed",
            "seed_or_restart": "all_seeds",
            "estimate": float(np.median(finite)),
            "median": float(np.median(finite)),
            "valid_basin_count": int(
                min(len(values) for values in values_by_seed.values())
            ),
            "seed_min": float(np.min(finite)),
            "seed_max": float(np.max(finite)),
            "same_sign_count": same_count,
            "all_seeds_agree": all_agree,
            "primary_estimate": primary_estimate,
            "difference_from_primary": float(np.median(finite) - primary_estimate),
            "same_direction_as_primary": bool(
                np.sign(np.median(finite)) == primary_sign
            ),
            "claim_id": "seed_robustness",
            "aggregation_rule": "median of three seed-specific basin estimates",
        }
    )


def _primary_absolute_table(
    performance: pd.DataFrame, existing: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    performance = _normalise_basin_ids(performance)
    existing = _set_summary_metadata(existing)
    existing = existing[
        ~(
            existing["analysis_set"].eq(FULL_SAMPLE)
            & existing["result_role"].eq("primary")
        )
    ]
    metric_columns = {"kge", "nse", "pbias", "rmse"}
    signature_metrics = {"CT_error_absolute", "AMJJ_error_absolute"}
    existing = existing[
        ~existing["metric"].astype(str).str.contains("SPO", case=False, na=False)
    ]
    existing = existing[~existing["metric"].isin(metric_columns)]
    existing = existing[
        ~(
            existing["metric"].isin(signature_metrics)
            & existing["analysis_set"].eq("primary_min5")
        )
    ]
    rows = []
    bootstrap = []
    rng = np.random.default_rng(seed)
    valid_models = {
        "IC-CMA-ES": ["XAJ-Base", "XAJ-TGD", "XAJ-CN"],
        "dPL-MLP": ["XAJ-Base", "XAJ-TGD", "XAJ-CN", "HBV"],
    }
    rules = {
        "IC-CMA-ES": "selected restart per basin by maximum train-period KGE",
        "dPL-MLP": "within-seed basin metric followed by median across seeds",
    }
    for paradigm, models in valid_models.items():
        for model in models:
            for period in ("train", "test"):
                group = performance[
                    (performance["paradigm"] == paradigm)
                    & (performance["model"] == model)
                    & (performance["period"] == period)
                ].drop_duplicates("basin_id")
                for metric in metric_columns:
                    if metric not in group:
                        continue
                    stats = _median_statistics(
                        pd.to_numeric(group[metric], errors="coerce"),
                        np.random.default_rng(
                            _key_seed(seed, "absolute", paradigm, model, period, metric)
                        ),
                    )
                    row = {
                        "paradigm": paradigm,
                        "model": model,
                        "period": period,
                        "metric": metric,
                        "status": "valid_full_sample",
                        "analysis_set": FULL_SAMPLE,
                        "summary_level": "full_sample",
                        "result_role": "primary",
                        "aggregation_rule": rules[paradigm],
                        "bootstrap_statistic": "median",
                        **stats,
                    }
                    rows.append(row)
                    bootstrap.append(
                        {
                            "record_type": "bootstrap",
                            "family": "absolute_metric",
                            "paradigm": paradigm,
                            "model": model,
                            "period": period,
                            "metric": metric,
                            "analysis_set": FULL_SAMPLE,
                            "summary_level": "full_sample",
                            "result_role": "primary",
                            "n": stats["valid_basin_count"],
                            "ci_low": stats["bootstrap_ci_low"],
                            "ci_high": stats["bootstrap_ci_high"],
                            "resamples": 10000,
                            "seed": seed,
                            "method": "ordinary_basin_bootstrap_median",
                            "bootstrap_statistic": "median",
                            "status": "valid_full_sample",
                        }
                    )
    return pd.concat(
        [existing, pd.DataFrame(rows)], ignore_index=True, sort=False
    ), bootstrap


def _signature_absolute_rows(
    signature_basin: pd.DataFrame, existing: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    existing = existing[
        ~existing["metric"].astype(str).str.contains("SPO", case=False, na=False)
    ].copy()
    existing = existing[
        ~(
            existing["analysis_set"].eq(FULL_SAMPLE)
            & existing["result_role"].eq("primary")
            & existing["metric"].isin(["CT_error_absolute", "AMJJ_error_absolute"])
        )
    ]
    existing = existing[
        ~(
            existing["metric"].isin(["CT_error_absolute", "AMJJ_error_absolute"])
            & existing["analysis_set"].eq("primary_min5")
        )
    ]
    rows, bootstrap = [], []
    rng = np.random.default_rng(seed + 1000)
    primary = signature_basin[signature_basin["analysis_set"].eq("primary_min5")]
    for (paradigm, model, period), group in primary.groupby(
        ["paradigm", "model", "period"], sort=True
    ):
        for signature, column in (
            ("CT", "ct_error_absolute"),
            ("AMJJ", "amjj_error_absolute"),
        ):
            stats = _median_statistics(
                group[column],
                np.random.default_rng(
                    _key_seed(
                        seed, "signature_absolute", paradigm, model, period, signature
                    )
                ),
            )
            metric = f"{signature}_error_absolute"
            rows.append(
                {
                    "paradigm": paradigm,
                    "model": model,
                    "period": period,
                    "metric": metric,
                    "status": "valid_full_sample_min5_years",
                    "analysis_set": FULL_SAMPLE,
                    "summary_level": "full_sample",
                    "result_role": "primary",
                    "valid_year_requirement": "minimum_5_complete_water_years",
                    "aggregation_rule": "median across complete-water-year basin summaries; dPL median across seeds",
                    "bootstrap_statistic": "median",
                    **stats,
                }
            )
            bootstrap.append(
                {
                    "record_type": "bootstrap",
                    "family": "absolute_signature_error",
                    "paradigm": paradigm,
                    "model": model,
                    "period": period,
                    "metric": metric,
                    "analysis_set": FULL_SAMPLE,
                    "summary_level": "full_sample",
                    "result_role": "primary",
                    "n": stats["valid_basin_count"],
                    "ci_low": stats["bootstrap_ci_low"],
                    "ci_high": stats["bootstrap_ci_high"],
                    "resamples": 10000,
                    "seed": seed,
                    "method": "ordinary_basin_bootstrap_median",
                    "bootstrap_statistic": "median",
                    "status": "valid_full_sample_min5_years",
                }
            )
    return pd.concat(
        [_set_summary_metadata(existing), pd.DataFrame(rows)],
        ignore_index=True,
        sort=False,
    ), bootstrap


def aggregate_full_sample(
    output: Path, data_root: Path, random_seed: int = 20260730
) -> dict[str, Any]:
    """Aggregate completed R1 basin tables without reading or writing daily simulations."""
    performance = _normalise_basin_ids(
        pd.read_csv(output / "r1_basin_level_performance.csv")
    )
    structural = _normalise_basin_ids(
        pd.read_csv(output / "r1_structural_effects_basin_level.csv")
    )
    generalization = _normalise_basin_ids(
        pd.read_csv(output / "r1_generalization_effects_basin_level.csv")
    )
    signatures = _normalise_basin_ids(
        pd.read_csv(output / "r1_snow_signatures_basin_level.csv")
    )
    signature_effects_source = _normalise_basin_ids(
        pd.read_csv(output / "r1_signature_effects_basin_level.csv")
    )
    existing_absolute = pd.read_csv(output / "r1_absolute_metrics_summary.csv")
    absolute, absolute_bootstrap = _primary_absolute_table(
        performance, existing_absolute, random_seed
    )
    absolute, signature_bootstrap = _signature_absolute_rows(
        signatures, absolute, random_seed
    )

    basin_regions, region_meta = region_membership(
        data_root, performance["basin_id"].unique()
    )
    rng = np.random.default_rng(random_seed + 2000)
    effect_rows, bootstrap_rows, test_rows = [], [], []
    structural_specs = [
        ("CN-Base", "KGE_CN - KGE_Base"),
        ("TGD-Base", "KGE_TGD - KGE_Base"),
        ("CN-TGD", "KGE_CN - KGE_TGD"),
    ]
    for paradigm in ("IC-CMA-ES", "dPL-MLP"):
        for effect, estimand in structural_specs:
            for period in ("train", "test"):
                frame = _primary_effect_source(structural, paradigm, effect, period)
                row, boot, test = _effect_bundle(
                    frame,
                    paradigm=paradigm,
                    model=effect,
                    period=period,
                    effect=effect,
                    metric="kge",
                    estimand=estimand,
                    family="structural_effect",
                    claim_id=_claim_for_effect(effect, "kge", period, paradigm, effect),
                    rng=rng,
                    basin_regions=basin_regions,
                    aggregation_rule="basin-wise paired effect; dPL median across seed-specific basin effects",
                )
                effect_rows.append(row)
                bootstrap_rows.extend(boot)
                test_rows.append(test)
        for effect, estimand in (
            ("E_CN-Base", "(KGE_CN,test-KGE_Base,test)-(KGE_CN,train-KGE_Base,train)"),
            (
                "E_TGD-Base",
                "(KGE_TGD,test-KGE_Base,test)-(KGE_TGD,train-KGE_Base,train)",
            ),
            ("E_CN-TGD", "(KGE_CN,test-KGE_TGD,test)-(KGE_CN,train-KGE_TGD,train)"),
        ):
            frame = _primary_effect_source(
                generalization, paradigm, effect, "test_minus_train"
            )
            row, boot, test = _effect_bundle(
                frame,
                paradigm=paradigm,
                model=effect,
                period="test_minus_train",
                effect=effect,
                metric="kge",
                estimand=estimand,
                family="generalization_effect",
                claim_id=_claim_for_effect(
                    effect, "kge", "test_minus_train", paradigm, effect
                ),
                rng=rng,
                basin_regions=basin_regions,
                aggregation_rule="basin-wise structural effect difference-in-differences",
            )
            effect_rows.append(row)
            bootstrap_rows.extend(boot)
            test_rows.append(test)
        for model in ("XAJ-Base", "XAJ-TGD", "XAJ-CN"):
            frame = (
                performance[
                    (performance["paradigm"] == paradigm)
                    & (performance["model"] == model)
                ]
                .pivot_table(
                    index="basin_id", columns="period", values="kge", aggfunc="first"
                )
                .reset_index()
            )
            frame["effect_value"] = frame.get("train", np.nan) - frame.get(
                "test", np.nan
            )
            row, boot, test = _effect_bundle(
                frame,
                paradigm=paradigm,
                model=model,
                period="train_minus_test",
                effect="train_minus_test",
                metric="kge",
                estimand="KGE_train-KGE_test",
                family="train_test_gap",
                claim_id=_claim_for_effect(
                    "train_minus_test", "kge", "train_minus_test", paradigm, model
                ),
                rng=rng,
                basin_regions=basin_regions,
                aggregation_rule="basin-wise train-minus-test gap",
            )
            effect_rows.append(row)
            bootstrap_rows.extend(boot)
            test_rows.append(test)

    for model in ("XAJ-Base", "XAJ-TGD", "XAJ-CN"):
        frame = _transfer_difference_frame(generalization, model)
        effect = f"D_{model.removeprefix('XAJ-')}"
        row, boot, test = _effect_bundle(
            frame,
            paradigm="IC-dPL",
            model=model,
            period="test_minus_train",
            effect=effect,
            metric="kge",
            estimand="(KGE_IC,test-KGE_dPL,test)-(KGE_IC,train-KGE_dPL,train)",
            family="ic_dpl_transfer",
            claim_id=_claim_for_effect(
                effect, "kge", "test_minus_train", "IC-dPL", model
            ),
            rng=rng,
            basin_regions=basin_regions,
            aggregation_rule="basin-wise IC-dPL transfer difference-in-differences",
        )
        effect_rows.append(row)
        bootstrap_rows.extend(boot)
        test_rows.append(test)

    signature_specs = {
        ("XAJ-Base", "XAJ-CN"): ("R_CN-Base", "|E_Base|-|E_CN|"),
        ("XAJ-Base", "XAJ-TGD"): ("R_TGD-Base", "|E_Base|-|E_TGD|"),
        ("XAJ-TGD", "XAJ-CN"): ("R_CN-TGD", "|E_TGD|-|E_CN|"),
    }
    signature_primary = signature_effects_source[
        (signature_effects_source["analysis_set"] == "primary_min5")
        & signature_effects_source["period"].eq("test")
    ]
    signature_primary = signature_primary[
        signature_primary["signature"].isin(["CT", "AMJJ"])
    ]
    for paradigm in ("IC-CMA-ES", "dPL-MLP"):
        for signature in ("CT", "AMJJ"):
            for (first, second), (effect, estimand) in signature_specs.items():
                left = signature_primary[
                    (signature_primary["paradigm"] == paradigm)
                    & (signature_primary["signature"] == signature)
                    & (signature_primary["first_model"] == first)
                    & (signature_primary["second_model"] == second)
                ]
                left = left.rename(columns={"effect_value": "effect_value"})
                metric = f"{signature}_error_reduction"
                row, boot, test = _effect_bundle(
                    left,
                    paradigm=paradigm,
                    model=f"{first}_vs_{second}",
                    period="test",
                    effect=effect,
                    metric=metric,
                    estimand=estimand,
                    family="signature_effect",
                    claim_id=_claim_for_effect(
                        effect, metric, "test", paradigm, f"{first}_vs_{second}"
                    ),
                    rng=rng,
                    basin_regions=basin_regions,
                    aggregation_rule="basin-wise paired error reduction from five-year signature summaries",
                )
                row["valid_year_requirement"] = "minimum_5_complete_water_years"
                effect_rows.append(row)
                bootstrap_rows.extend(boot)
                test_rows.append(test)

    effect_summary = pd.DataFrame(effect_rows)
    new_tests = _apply_primary_bh(pd.DataFrame(test_rows))
    test_p = (
        new_tests.drop_duplicates("_summary_key", keep="last").set_index("_summary_key")
        if not new_tests.empty
        else pd.DataFrame()
    )
    if not effect_summary.empty and not test_p.empty:
        effect_summary["wilcoxon_bh_p"] = effect_summary["_summary_key"].map(
            test_p["wilcoxon_bh_p"]
        )
        effect_summary["sign_test_bh_p"] = effect_summary["_summary_key"].map(
            test_p["sign_test_bh_p"]
        )
    dpl_seed_rows = _dpl_seed_robustness_rows(
        output, effect_summary, signature_effects_source
    )
    old_paired = _set_summary_metadata(
        pd.read_csv(output / "r1_paired_effects_summary.csv")
    )
    old_paired = old_paired[
        ~old_paired.astype(str)
        .apply(lambda col: col.str.contains("SPO", case=False, na=False))
        .any(axis=1)
    ]
    old_paired = old_paired[
        ~(
            old_paired["analysis_set"].eq(FULL_SAMPLE)
            & old_paired["result_role"].eq("primary")
        )
    ]
    dpl_robust_mask = old_paired.get(
        "robustness_type", pd.Series(index=old_paired.index)
    ).astype(str).eq("dPL_seed") & (
        (
            old_paired.get("metric", pd.Series(index=old_paired.index))
            .astype(str)
            .eq("kge")
            & old_paired.get("period", pd.Series(index=old_paired.index))
            .astype(str)
            .isin(["test", "test_minus_train"])
            & old_paired.get("effect", pd.Series(index=old_paired.index))
            .astype(str)
            .isin(["CN-Base", "TGD-Base", "CN-TGD", "E_CN-Base", "E_CN-TGD"])
        )
        | (
            old_paired.get("metric", pd.Series(index=old_paired.index))
            .astype(str)
            .isin(["CT_error_reduction", "AMJJ_error_reduction"])
            & old_paired.get("period", pd.Series(index=old_paired.index))
            .astype(str)
            .eq("test")
            & old_paired.get("effect", pd.Series(index=old_paired.index))
            .astype(str)
            .eq("XAJ-TGD_minus_XAJ-CN")
        )
    )
    old_paired = old_paired[~dpl_robust_mask]
    old_paired["result_role"] = old_paired["result_role"].where(
        old_paired["result_role"].notna(),
        np.where(old_paired["analysis_set"].eq(FULL_SAMPLE), "primary", "sensitivity"),
    )
    paired = pd.concat(
        [
            old_paired,
            effect_summary.drop(columns=["_summary_key"], errors="ignore"),
            dpl_seed_rows,
        ],
        ignore_index=True,
        sort=False,
    )

    old_bootstrap = pd.read_csv(output / "r1_bootstrap_intervals.csv")
    old_bootstrap = old_bootstrap[
        ~old_bootstrap.astype(str)
        .apply(lambda col: col.str.contains("SPO", case=False, na=False))
        .any(axis=1)
    ]
    old_bootstrap = old_bootstrap[
        ~(
            old_bootstrap.get("analysis_set", pd.Series(index=old_bootstrap.index))
            .astype(str)
            .eq(FULL_SAMPLE)
            & old_bootstrap.get("result_role", pd.Series(index=old_bootstrap.index))
            .astype(str)
            .eq("primary")
        )
    ]
    bootstrap = pd.concat(
        [
            old_bootstrap,
            pd.DataFrame(absolute_bootstrap + signature_bootstrap + bootstrap_rows),
        ],
        ignore_index=True,
        sort=False,
    )
    old_tests = pd.read_csv(output / "r1_statistical_tests.csv")
    old_tests = old_tests[
        ~old_tests.astype(str)
        .apply(lambda col: col.str.contains("SPO", case=False, na=False))
        .any(axis=1)
    ]
    old_tests = old_tests[
        ~(
            old_tests.get("analysis_set", pd.Series(index=old_tests.index))
            .astype(str)
            .eq(FULL_SAMPLE)
            & old_tests.get("result_role", pd.Series(index=old_tests.index))
            .astype(str)
            .eq("primary")
        )
    ]
    tests = pd.concat(
        [old_tests, new_tests.drop(columns=["_summary_key"], errors="ignore")],
        ignore_index=True,
        sort=False,
    )

    relationships = pd.read_csv(output / "r1_snow_relationships_summary.csv")
    relationships = relationships[
        ~relationships.astype(str)
        .apply(lambda col: col.str.contains("SPO", case=False, na=False))
        .any(axis=1)
    ].copy()
    relationships = relationships[
        ~(
            relationships.get("analysis_set", pd.Series(index=relationships.index))
            .astype(str)
            .eq(FULL_SAMPLE)
            & relationships.get("result_role", pd.Series(index=relationships.index))
            .astype(str)
            .eq("primary")
            & relationships.get("analysis_type", pd.Series(index=relationships.index))
            .astype(str)
            .eq("full_sample_attribute")
        )
    ]
    relationships["analysis_set"] = relationships.get("analysis_set", FULL_SAMPLE)
    relationships["summary_level"] = relationships.get("summary_level", "detailed")
    relationships["result_role"] = relationships.get("result_role", "secondary")
    attributes = _normalise_basin_ids(pd.read_csv(output / "r1_snow_attributes.csv"))
    snow_rows = []
    snow_rng = np.random.default_rng(random_seed + 4000)
    for paradigm in ("IC-CMA-ES", "dPL-MLP"):
        frame = _primary_effect_source(structural, paradigm, "CN-TGD", "test")
        joined = frame.merge(
            attributes[["basin_id", "frac_snow"]], on="basin_id", how="inner"
        )
        joined = joined[
            np.isfinite(joined["effect_value"]) & np.isfinite(joined["frac_snow"])
        ]
        row = _relationship_row(
            paradigm,
            "CN-TGD",
            "test",
            joined,
            analysis_type="full_sample_attribute",
            x_metric="frac_snow",
            y_metric="KGE_CN-TGD_test",
        )
        low, high = bootstrap_median_ci(joined["effect_value"], snow_rng)
        row.update(
            {
                "analysis_set": FULL_SAMPLE,
                "summary_level": "full_sample",
                "result_role": "primary",
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_statistic": "median",
                "aggregation_rule": "basin-wise CN-TGD test effect with continuous frac_snow",
            }
        )
        snow_rows.append(row)
    interaction_mask = relationships["analysis_type"].eq("cluster_robust_interaction")
    relationships.loc[interaction_mask, "analysis_set"] = FULL_SAMPLE
    relationships.loc[interaction_mask, "summary_level"] = "full_sample"
    relationships.loc[interaction_mask, "result_role"] = "primary"
    relationships = pd.concat(
        [relationships, pd.DataFrame(snow_rows)], ignore_index=True, sort=False
    )

    return {
        "absolute": absolute,
        "paired": paired,
        "relationships": relationships,
        "bootstrap": bootstrap,
        "tests": tests,
        "effect_summary": effect_summary,
        "new_tests": new_tests,
        "region_meta": region_meta,
        "primary_effects": effect_summary[effect_summary["result_role"].eq("primary")]
        if not effect_summary.empty
        else effect_summary,
    }


def _snow_strata(attributes: pd.DataFrame) -> pd.DataFrame:
    attributes = _normalise_basin_ids(attributes)
    values = pd.to_numeric(attributes["frac_snow"], errors="coerce")
    rows = []
    for basin_id, value in zip(attributes["basin_id"], values):
        label = interval = math.nan
        if np.isfinite(value):
            for candidate, candidate_interval, lower, upper in SNOW_STRATA:
                if (value >= lower and value < upper) or (
                    candidate == "S5" and value <= upper
                ):
                    label, interval = candidate, candidate_interval
                    break
        rows.append(
            {"basin_id": basin_id, "snow_stratum": label, "snow_interval": interval}
        )
    strata = pd.DataFrame(rows)
    counts = strata["snow_stratum"].value_counts().to_dict()
    strata["stratum_n"] = strata["snow_stratum"].map(counts)
    return strata


def append_snow_stratified(
    output: Path,
    data_root: Path,
    aggregate: dict[str, Any],
    random_seed: int = 20260730,
) -> dict[str, Any]:
    """Append fixed, project-documented frac_snow stratum summaries in place."""
    attributes = _normalise_basin_ids(pd.read_csv(output / "r1_snow_attributes.csv"))
    strata = _snow_strata(attributes)
    strata_counts = (
        strata.dropna(subset=["snow_stratum"])
        .groupby(["snow_stratum", "snow_interval"], as_index=False)["basin_id"]
        .nunique()
        .rename(columns={"basin_id": "stratum_n"})
    )
    performance = _normalise_basin_ids(
        pd.read_csv(output / "r1_basin_level_performance.csv")
    ).merge(strata, on="basin_id", how="left")
    structural = _normalise_basin_ids(
        pd.read_csv(output / "r1_structural_effects_basin_level.csv")
    ).merge(strata, on="basin_id", how="left")
    generalization = _normalise_basin_ids(
        pd.read_csv(output / "r1_generalization_effects_basin_level.csv")
    ).merge(strata, on="basin_id", how="left")
    signatures = _normalise_basin_ids(
        pd.read_csv(output / "r1_snow_signatures_basin_level.csv")
    ).merge(strata, on="basin_id", how="left")
    signature_effects = _normalise_basin_ids(
        pd.read_csv(output / "r1_signature_effects_basin_level.csv")
    )
    signature_effects = signature_effects.drop(
        columns=["snow_stratum", "snow_interval", "stratum_n"], errors="ignore"
    ).merge(strata, on="basin_id", how="left")
    rng = np.random.default_rng(random_seed + 7000)
    strat_abs, strat_boot, strat_effect_rows, strat_tests = [], [], [], []
    valid_models = {
        "IC-CMA-ES": ["XAJ-Base", "XAJ-TGD", "XAJ-CN"],
        "dPL-MLP": ["XAJ-Base", "XAJ-TGD", "XAJ-CN", "HBV"],
    }
    for (stratum, interval), stratum_group in performance.dropna(
        subset=["snow_stratum"]
    ).groupby(["snow_stratum", "snow_interval"], sort=True):
        n = int(stratum_group["basin_id"].nunique())
        for paradigm, models in valid_models.items():
            for model in models:
                for period in ("train", "test"):
                    group = stratum_group[
                        (stratum_group["paradigm"] == paradigm)
                        & (stratum_group["model"] == model)
                        & (stratum_group["period"] == period)
                    ].drop_duplicates("basin_id")
                    for metric in ("kge", "nse", "pbias", "rmse"):
                        stats = _median_statistics(
                            pd.to_numeric(group[metric], errors="coerce"),
                            np.random.default_rng(
                                _key_seed(
                                    random_seed,
                                    "stratum_absolute",
                                    stratum,
                                    paradigm,
                                    model,
                                    period,
                                    metric,
                                )
                            ),
                        )
                        strat_abs.append(
                            {
                                "paradigm": paradigm,
                                "model": model,
                                "period": period,
                                "metric": metric,
                                "status": "valid_snow_stratum",
                                "analysis_set": "snow_fixed_strata",
                                "summary_level": "snow_stratum",
                                "result_role": "stratified_primary",
                                "snow_stratum": stratum,
                                "snow_interval": interval,
                                "stratum_n": n,
                                "aggregation_rule": "basin-level summary within project-fixed frac_snow stratum",
                                "bootstrap_statistic": "median",
                                **stats,
                            }
                        )
                        strat_boot.append(
                            {
                                "record_type": "bootstrap",
                                "family": "absolute_metric",
                                "paradigm": paradigm,
                                "model": model,
                                "period": period,
                                "metric": metric,
                                "analysis_set": "snow_fixed_strata",
                                "summary_level": "snow_stratum",
                                "result_role": "stratified_primary",
                                "snow_stratum": stratum,
                                "snow_interval": interval,
                                "stratum_n": n,
                                "n": stats["valid_basin_count"],
                                "ci_low": stats["bootstrap_ci_low"],
                                "ci_high": stats["bootstrap_ci_high"],
                                "resamples": 10000,
                                "seed": random_seed,
                                "method": "ordinary_basin_bootstrap_median",
                                "bootstrap_statistic": "median",
                                "status": "valid_snow_stratum",
                            }
                        )
    primary_signature = signatures[signatures["analysis_set"].eq("primary_min5")]
    for stratum, interval, _ in strata_counts.itertuples(index=False, name=None):
        for (paradigm, model, period), group in primary_signature[
            primary_signature["snow_stratum"].eq(stratum)
        ].groupby(["paradigm", "model", "period"], sort=True):
            for signature, column in (
                ("CT", "ct_error_absolute"),
                ("AMJJ", "amjj_error_absolute"),
            ):
                stats = _median_statistics(
                    group[column],
                    np.random.default_rng(
                        _key_seed(
                            random_seed,
                            "stratum_signature_absolute",
                            stratum,
                            paradigm,
                            model,
                            period,
                            signature,
                        )
                    ),
                )
                metric = f"{signature}_error_absolute"
                strat_abs.append(
                    {
                        "paradigm": paradigm,
                        "model": model,
                        "period": period,
                        "metric": metric,
                        "status": "valid_snow_stratum_min5_years",
                        "analysis_set": "snow_fixed_strata",
                        "summary_level": "snow_stratum",
                        "result_role": "stratified_primary",
                        "snow_stratum": stratum,
                        "snow_interval": interval,
                        "stratum_n": int(group["basin_id"].nunique()),
                        "valid_year_requirement": "minimum_5_complete_water_years",
                        "aggregation_rule": "basin-level median over complete water years within stratum",
                        "bootstrap_statistic": "median",
                        **stats,
                    }
                )
                strat_boot.append(
                    {
                        "record_type": "bootstrap",
                        "family": "absolute_signature_error",
                        "paradigm": paradigm,
                        "model": model,
                        "period": period,
                        "metric": metric,
                        "analysis_set": "snow_fixed_strata",
                        "summary_level": "snow_stratum",
                        "result_role": "stratified_primary",
                        "snow_stratum": stratum,
                        "snow_interval": interval,
                        "stratum_n": int(group["basin_id"].nunique()),
                        "n": stats["valid_basin_count"],
                        "ci_low": stats["bootstrap_ci_low"],
                        "ci_high": stats["bootstrap_ci_high"],
                        "resamples": 10000,
                        "seed": random_seed,
                        "method": "ordinary_basin_bootstrap_median",
                        "bootstrap_statistic": "median",
                        "status": "valid_snow_stratum_min5_years",
                    }
                )

    structural_specs = [
        ("CN-Base", "KGE_CN - KGE_Base"),
        ("TGD-Base", "KGE_TGD - KGE_Base"),
        ("CN-TGD", "KGE_CN - KGE_TGD"),
    ]
    signature_specs = {
        ("XAJ-Base", "XAJ-CN"): ("R_CN-Base", "|E_Base|-|E_CN|"),
        ("XAJ-Base", "XAJ-TGD"): ("R_TGD-Base", "|E_Base|-|E_TGD|"),
        ("XAJ-TGD", "XAJ-CN"): ("R_CN-TGD", "|E_TGD|-|E_CN|"),
    }
    for stratum, interval, _ in strata_counts.itertuples(index=False, name=None):
        extra = {
            "snow_stratum": stratum,
            "snow_interval": interval,
            "stratum_n": int(
                strata_counts.loc[
                    strata_counts.snow_stratum.eq(stratum), "stratum_n"
                ].iloc[0]
            ),
        }
        for paradigm in ("IC-CMA-ES", "dPL-MLP"):
            for effect, estimand in structural_specs:
                for period in ("train", "test"):
                    frame = _primary_effect_source(structural, paradigm, effect, period)
                    frame = frame[frame["snow_stratum"].eq(stratum)]
                    row, boot, test = _effect_bundle(
                        frame,
                        paradigm=paradigm,
                        model=effect,
                        period=period,
                        effect=effect,
                        metric="kge",
                        estimand=estimand,
                        family="structural_effect",
                        claim_id=_claim_for_effect(
                            effect, "kge", period, paradigm, effect
                        ),
                        rng=rng,
                        basin_regions={},
                        aggregation_rule="basin-wise paired effect within fixed frac_snow stratum",
                        analysis_set="snow_fixed_strata",
                        summary_level="snow_stratum",
                        result_role="stratified_primary",
                        extra_metadata=extra,
                    )
                    strat_effect_rows.append(row)
                    strat_boot.extend(boot)
                    strat_tests.append(test)
            for effect, estimand in (
                (
                    "E_CN-Base",
                    "(KGE_CN,test-KGE_Base,test)-(KGE_CN,train-KGE_Base,train)",
                ),
                (
                    "E_TGD-Base",
                    "(KGE_TGD,test-KGE_Base,test)-(KGE_TGD,train-KGE_Base,train)",
                ),
                ("E_CN-TGD", "(KGE_CN,test-KGE_TGD,test)-(KGE_CN,train-KGE_TGD,train)"),
            ):
                frame = _primary_effect_source(
                    generalization, paradigm, effect, "test_minus_train"
                )
                frame = frame[frame["snow_stratum"].eq(stratum)]
                row, boot, test = _effect_bundle(
                    frame,
                    paradigm=paradigm,
                    model=effect,
                    period="test_minus_train",
                    effect=effect,
                    metric="kge",
                    estimand=estimand,
                    family="generalization_effect",
                    claim_id=_claim_for_effect(
                        effect, "kge", "test_minus_train", paradigm, effect
                    ),
                    rng=rng,
                    basin_regions={},
                    aggregation_rule="basin-wise exposure effect within fixed frac_snow stratum",
                    analysis_set="snow_fixed_strata",
                    summary_level="snow_stratum",
                    result_role="stratified_primary",
                    extra_metadata=extra,
                )
                strat_effect_rows.append(row)
                strat_boot.extend(boot)
                strat_tests.append(test)
            for model in ("XAJ-Base", "XAJ-TGD", "XAJ-CN"):
                frame = (
                    performance[
                        (performance["paradigm"] == paradigm)
                        & (performance["model"] == model)
                        & performance["snow_stratum"].eq(stratum)
                    ]
                    .pivot_table(
                        index="basin_id",
                        columns="period",
                        values="kge",
                        aggfunc="first",
                    )
                    .reset_index()
                )
                frame["effect_value"] = frame.get("train", np.nan) - frame.get(
                    "test", np.nan
                )
                row, boot, test = _effect_bundle(
                    frame,
                    paradigm=paradigm,
                    model=model,
                    period="train_minus_test",
                    effect="train_minus_test",
                    metric="kge",
                    estimand="KGE_train-KGE_test",
                    family="train_test_gap",
                    claim_id=_claim_for_effect(
                        "train_minus_test", "kge", "train_minus_test", paradigm, model
                    ),
                    rng=rng,
                    basin_regions={},
                    aggregation_rule="basin-wise train-minus-test gap within fixed frac_snow stratum",
                    analysis_set="snow_fixed_strata",
                    summary_level="snow_stratum",
                    result_role="stratified_primary",
                    extra_metadata=extra,
                )
                strat_effect_rows.append(row)
                strat_boot.extend(boot)
                strat_tests.append(test)
        for model in ("XAJ-Base", "XAJ-TGD", "XAJ-CN"):
            frame = _transfer_difference_frame(generalization, model)
            frame = frame.merge(
                strata[["basin_id", "snow_stratum"]], on="basin_id", how="left"
            )
            frame = frame[frame["snow_stratum"].eq(stratum)]
            effect = f"D_{model.removeprefix('XAJ-')}"
            row, boot, test = _effect_bundle(
                frame,
                paradigm="IC-dPL",
                model=model,
                period="test_minus_train",
                effect=effect,
                metric="kge",
                estimand="(KGE_IC,test-KGE_dPL,test)-(KGE_IC,train-KGE_dPL,train)",
                family="ic_dpl_transfer",
                claim_id=_claim_for_effect(
                    effect, "kge", "test_minus_train", "IC-dPL", model
                ),
                rng=rng,
                basin_regions={},
                aggregation_rule="basin-wise IC-dPL transfer difference within fixed frac_snow stratum",
                analysis_set="snow_fixed_strata",
                summary_level="snow_stratum",
                result_role="stratified_primary",
                extra_metadata=extra,
            )
            strat_effect_rows.append(row)
            strat_boot.extend(boot)
            strat_tests.append(test)
        primary_effects = signature_effects[
            (signature_effects["analysis_set"] == "primary_min5")
            & signature_effects["period"].eq("test")
            & signature_effects["snow_stratum"].eq(stratum)
        ]
        for paradigm in ("IC-CMA-ES", "dPL-MLP"):
            for signature in ("CT", "AMJJ"):
                for (first, second), (effect, estimand) in signature_specs.items():
                    frame = primary_effects[
                        (primary_effects["paradigm"] == paradigm)
                        & (primary_effects["signature"] == signature)
                        & (primary_effects["first_model"] == first)
                        & (primary_effects["second_model"] == second)
                    ]
                    metric = f"{signature}_error_reduction"
                    row, boot, test = _effect_bundle(
                        frame,
                        paradigm=paradigm,
                        model=f"{first}_vs_{second}",
                        period="test",
                        effect=effect,
                        metric=metric,
                        estimand=estimand,
                        family="signature_effect",
                        claim_id=_claim_for_effect(
                            effect, metric, "test", paradigm, f"{first}_vs_{second}"
                        ),
                        rng=rng,
                        basin_regions={},
                        aggregation_rule="basin-wise signature error reduction within fixed frac_snow stratum",
                        analysis_set="snow_fixed_strata",
                        summary_level="snow_stratum",
                        result_role="stratified_primary",
                        extra_metadata={
                            **extra,
                            "valid_year_requirement": "minimum_5_complete_water_years",
                        },
                    )
                    strat_effect_rows.append(row)
                    strat_boot.extend(boot)
                    strat_tests.append(test)

    strat_effects = pd.DataFrame(strat_effect_rows)
    strat_tests_frame = _apply_primary_bh(pd.DataFrame(strat_tests))
    if not strat_effects.empty and not strat_tests_frame.empty:
        test_map = strat_tests_frame.drop_duplicates("_summary_key").set_index(
            "_summary_key"
        )
        strat_effects["wilcoxon_bh_p"] = strat_effects["_summary_key"].map(
            test_map["wilcoxon_bh_p"]
        )
        strat_effects["sign_test_bh_p"] = strat_effects["_summary_key"].map(
            test_map["sign_test_bh_p"]
        )
    old_abs = aggregate["absolute"]
    old_abs = old_abs[
        ~old_abs.get("analysis_set", pd.Series(index=old_abs.index))
        .astype(str)
        .eq("snow_fixed_strata")
    ]
    absolute = pd.concat(
        [old_abs, pd.DataFrame(strat_abs)], ignore_index=True, sort=False
    )
    old_pair = aggregate["paired"]
    old_pair = old_pair[
        ~old_pair.get("analysis_set", pd.Series(index=old_pair.index))
        .astype(str)
        .eq("snow_fixed_strata")
    ]
    paired = pd.concat(
        [old_pair, strat_effects.drop(columns=["_summary_key"], errors="ignore")],
        ignore_index=True,
        sort=False,
    )
    old_boot = aggregate["bootstrap"]
    old_boot = old_boot[
        ~old_boot.get("analysis_set", pd.Series(index=old_boot.index))
        .astype(str)
        .eq("snow_fixed_strata")
    ]
    bootstrap = pd.concat(
        [old_boot, pd.DataFrame(strat_boot)], ignore_index=True, sort=False
    )
    old_tests = aggregate["tests"]
    old_tests = old_tests[
        ~old_tests.get("analysis_set", pd.Series(index=old_tests.index))
        .astype(str)
        .eq("snow_fixed_strata")
    ]
    tests = pd.concat(
        [old_tests, strat_tests_frame.drop(columns=["_summary_key"], errors="ignore")],
        ignore_index=True,
        sort=False,
    )

    relationships = aggregate["relationships"]
    relationships = relationships[
        ~relationships.get("analysis_set", pd.Series(index=relationships.index))
        .astype(str)
        .eq("snow_fixed_strata")
    ]
    relationship_rows = []
    for stratum, interval, _ in strata_counts.itertuples(index=False, name=None):
        for paradigm in ("IC-CMA-ES", "dPL-MLP"):
            frame = _primary_effect_source(structural, paradigm, "CN-TGD", "test")
            joined = frame[
                frame["basin_id"].isin(
                    strata[strata["snow_stratum"].eq(stratum)]["basin_id"]
                )
            ].merge(attributes[["basin_id", "frac_snow"]], on="basin_id", how="inner")
            row = _relationship_row(
                paradigm,
                "CN-TGD",
                "test",
                joined,
                analysis_type="snow_stratum_summary",
                x_metric="frac_snow",
                y_metric="KGE_CN-TGD_test",
            )
            low, high = bootstrap_median_ci(
                joined["effect_value"],
                np.random.default_rng(
                    _key_seed(
                        random_seed,
                        "snow_stratum_relationship",
                        stratum,
                        paradigm,
                        "CN-TGD",
                        "test",
                    )
                ),
            )
            row.update(
                {
                    "analysis_set": "snow_fixed_strata",
                    "summary_level": "snow_stratum",
                    "result_role": "stratified_primary",
                    "snow_stratum": stratum,
                    "snow_interval": interval,
                    "stratum_n": int(
                        strata_counts.loc[
                            strata_counts.snow_stratum.eq(stratum), "stratum_n"
                        ].iloc[0]
                    ),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "bootstrap_statistic": "median",
                }
            )
            relationship_rows.append(row)
    relationships = pd.concat(
        [relationships, pd.DataFrame(relationship_rows)], ignore_index=True, sort=False
    )
    aggregate.update(
        {
            "absolute": absolute,
            "paired": paired,
            "relationships": relationships,
            "bootstrap": bootstrap,
            "tests": tests,
            "snow_strata": strata_counts,
        }
    )
    return aggregate


REMAINING_ANALYSIS_PREFIX = "r1_remaining_"


def _drop_remaining(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "analysis_set" not in table:
        return table.copy()
    mask = table["analysis_set"].astype(str).str.startswith(REMAINING_ANALYSIS_PREFIX)
    return table.loc[~mask].copy()


def _paired_tests(values: np.ndarray) -> tuple[float, float]:
    from scipy.stats import binomtest, wilcoxon

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    try:
        wilcoxon_p = float(
            wilcoxon(
                values, zero_method="wilcox", alternative="two-sided", method="auto"
            ).pvalue
        )
    except ValueError:
        wilcoxon_p = 1.0
    nonzero = values[values != 0]
    sign_p = (
        float(binomtest(int((nonzero > 0).sum()), int(len(nonzero)), 0.5).pvalue)
        if len(nonzero)
        else 1.0
    )
    return wilcoxon_p, sign_p


def _remaining_key(row: dict[str, Any]) -> str:
    return "|".join(
        str(row.get(column, ""))
        for column in (
            "paradigm",
            "model",
            "period",
            "effect",
            "metric",
            "analysis_set",
            "snow_stratum",
        )
    )


def _remaining_effect_bundle(
    frame: pd.DataFrame,
    *,
    paradigm: str,
    model: str,
    period: str,
    effect: str,
    metric: str,
    family: str,
    estimand: str,
    analysis_set: str,
    regions: dict[str, str],
    seed: int,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    extra = extra or {}
    current = frame.copy()
    current["basin_id"] = current["basin_id"].astype(str).str.zfill(8)
    current = current.drop_duplicates("basin_id")
    values = pd.to_numeric(current.get("effect_value"), errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    stats = _median_statistics(values, np.random.default_rng(seed))
    ci_low, ci_high = stats["bootstrap_ci_low"], stats["bootstrap_ci_high"]
    wilcoxon_p, sign_p = _paired_tests(values)
    row = {
        "paradigm": paradigm,
        "model": model,
        "period": period,
        "effect": effect,
        "metric": metric,
        "effect_family": family,
        "estimand": estimand,
        "analysis_set": analysis_set,
        "summary_level": "full_sample"
        if "snow_stratum" not in extra
        else "snow_stratum",
        "result_role": "primary",
        "aggregation_rule": "basin-wise paired effect before aggregation; dPL primary is within-seed then median across seeds",
        "bootstrap_statistic": "median",
        "statistical_method": "paired basin bootstrap median CI",
        "support_status": support_status(ci_low, ci_high),
        "positive_fraction": stats["fraction_positive"],
        "wilcoxon_p": wilcoxon_p,
        "sign_test_p": sign_p,
        "valid_year_requirement": extra.get("valid_year_requirement", ""),
        **stats,
        **extra,
    }
    row["claim_id"] = ""
    row["_summary_key"] = _remaining_key(row)
    bootstrap = [
        {
            "record_type": "bootstrap",
            "family": family,
            "paradigm": paradigm,
            "model": model,
            "period": period,
            "metric": metric,
            "effect": effect,
            "analysis_set": analysis_set,
            "summary_level": row["summary_level"],
            "result_role": "primary",
            "n": stats["valid_basin_count"],
            "ci_low": ci_low,
            "ci_high": ci_high,
            "resamples": 10000,
            "seed": seed,
            "method": "ordinary_basin_bootstrap_median",
            "bootstrap_statistic": "median",
            "status": "valid_remaining_check",
            "claim_id": "",
            **extra,
        }
    ]
    if regions:
        region_frame = current.copy()
        region_frame["region"] = region_frame["basin_id"].map(regions)
        region_frame = region_frame[
            region_frame["region"].notna()
            & np.isfinite(pd.to_numeric(region_frame["effect_value"], errors="coerce"))
        ]
        if not region_frame.empty:
            low, high = block_bootstrap_median_ci(
                region_frame["effect_value"],
                region_frame["region"],
                np.random.default_rng(seed + 1),
            )
            row.update(
                {"regional_bootstrap_ci_low": low, "regional_bootstrap_ci_high": high}
            )
            bootstrap.append(
                {
                    "record_type": "bootstrap",
                    "family": family,
                    "paradigm": paradigm,
                    "model": model,
                    "period": period,
                    "metric": metric,
                    "effect": effect,
                    "analysis_set": analysis_set,
                    "summary_level": row["summary_level"],
                    "result_role": "primary",
                    "n": int(region_frame["basin_id"].nunique()),
                    "ci_low": low,
                    "ci_high": high,
                    "resamples": 10000,
                    "seed": seed,
                    "method": "region_block_bootstrap_median",
                    "bootstrap_statistic": "median",
                    "block_count": int(region_frame["region"].nunique()),
                    "block_source": "data/basin_groups/group_11..group_17.npy",
                    "status": "valid_region_block_bootstrap",
                    "claim_id": "",
                    **extra,
                }
            )
    test = {
        "record_type": "test",
        "family": family,
        "paradigm": paradigm,
        "model": model,
        "period": period,
        "metric": metric,
        "effect": effect,
        "analysis_set": analysis_set,
        "summary_level": row["summary_level"],
        "result_role": "primary",
        "n": stats["valid_basin_count"],
        "ci_low": ci_low,
        "ci_high": ci_high,
        "resamples": 10000,
        "seed": seed,
        "method": "Wilcoxon signed-rank and sign test; paired median bootstrap CI",
        "wilcoxon_p": wilcoxon_p,
        "sign_test_p": sign_p,
        "support_status": row["support_status"],
        "claim_id": "",
        "_summary_key": row["_summary_key"],
        **extra,
    }
    return row, bootstrap, test


def _finish_remaining_tests(
    rows: list[dict[str, Any]], tests: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame(rows)
    test_frame = _apply_primary_bh(pd.DataFrame(tests)) if tests else pd.DataFrame()
    if not summary.empty and not test_frame.empty:
        lookup = test_frame.drop_duplicates("_summary_key").set_index("_summary_key")
        summary["wilcoxon_bh_p"] = summary["_summary_key"].map(lookup["wilcoxon_bh_p"])
        summary["sign_test_bh_p"] = summary["_summary_key"].map(
            lookup["sign_test_bh_p"]
        )
    return summary, test_frame


def _primary_signature_models(
    years: pd.DataFrame, minimum_years: int = 5
) -> pd.DataFrame:
    """Collapse complete water-year errors to the primary basin/model unit."""
    valid = years[years["status"].astype(str).eq("valid_ct_amjj_spo_unresolved")].copy()
    rows = []
    for keys, group in valid.groupby(
        ["basin_id", "paradigm", "model", "seed_or_restart", "period"], sort=True
    ):
        basin, paradigm, model, run, period = keys
        if len(group) < minimum_years:
            continue
        rows.append(
            {
                "basin_id": str(basin).zfill(8),
                "paradigm": paradigm,
                "model": model,
                "period": period,
                "seed_or_restart": run,
                "valid_years": int(len(group)),
                "CT_error_absolute": float(group["ct_error_absolute"].median()),
                "AMJJ_error_absolute": float(group["amjj_error_absolute"].median()),
            }
        )
    model_seed = pd.DataFrame(rows)
    if model_seed.empty:
        return model_seed
    primary = []
    for keys, group in model_seed.groupby(
        ["basin_id", "paradigm", "model", "period"], sort=True
    ):
        basin, paradigm, model, period = keys
        primary.append(
            {
                "basin_id": basin,
                "paradigm": paradigm,
                "model": model,
                "period": period,
                "seed_or_restart": "median_across_seeds"
                if paradigm == "dPL-MLP"
                else "selected_restart",
                "valid_years": int(group["valid_years"].min()),
                "CT_error_absolute": float(group["CT_error_absolute"].median()),
                "AMJJ_error_absolute": float(group["AMJJ_error_absolute"].median()),
            }
        )
    return pd.DataFrame(primary)


def _block_slope_ci(frame: pd.DataFrame, seed: int) -> tuple[float, float]:
    current = frame[
        np.isfinite(frame["frac_snow"])
        & np.isfinite(frame["effect_value"])
        & frame["region"].notna()
    ].copy()
    if current.empty or current["region"].nunique() < 2:
        return math.nan, math.nan
    blocks = sorted(current["region"].unique())
    rng = np.random.default_rng(seed)
    block_stats = []
    for block in blocks:
        group = current[current["region"].eq(block)]
        x = group["frac_snow"].to_numpy(float)
        y = group["effect_value"].to_numpy(float)
        block_stats.append((len(x), x.sum(), y.sum(), np.dot(x, x), np.dot(x, y)))
    stats = np.asarray(block_stats, dtype=float)
    counts = rng.multinomial(
        len(blocks), np.full(len(blocks), 1.0 / len(blocks)), size=10000
    )
    n = counts @ stats[:, 0]
    sx = counts @ stats[:, 1]
    sy = counts @ stats[:, 2]
    sxx = counts @ stats[:, 3]
    sxy = counts @ stats[:, 4]
    denominator = sxx - sx * sx / n
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = (sxy - sx * sy / n) / denominator
    slopes = slopes[np.isfinite(slopes)]
    return (
        tuple(float(x) for x in np.percentile(slopes, [2.5, 97.5]))
        if slopes.size
        else (math.nan, math.nan)
    )


def _remaining_relationship(
    frame: pd.DataFrame,
    *,
    paradigm: str,
    effect: str,
    period: str,
    analysis_type: str,
    x_metric: str,
    y_metric: str,
    analysis_set: str,
    seed: int,
    regions: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra = extra or {}
    current = frame.copy()
    current["frac_snow"] = pd.to_numeric(current["frac_snow"], errors="coerce")
    current["effect_value"] = pd.to_numeric(current["effect_value"], errors="coerce")
    current = current[
        np.isfinite(current["frac_snow"]) & np.isfinite(current["effect_value"])
    ].drop_duplicates("basin_id")
    row = _relationship_row(
        paradigm,
        effect,
        period,
        current,
        analysis_type=analysis_type,
        x_metric=x_metric,
        y_metric=y_metric,
    )
    row.update(
        {
            "analysis_set": analysis_set,
            "summary_level": "full_sample",
            "result_role": "primary",
            "reference_category": extra.get("reference_category", ""),
            **extra,
        }
    )
    current["region"] = current["basin_id"].astype(str).str.zfill(8).map(regions)
    low, high = _block_slope_ci(current, seed) if regions else (math.nan, math.nan)
    row.update(
        {
            "regional_slope_ci_low": low,
            "regional_slope_ci_high": high,
            "bootstrap_method": "region block bootstrap OLS slope sensitivity"
            if regions
            else "unavailable",
        }
    )
    return row


def _interaction_transfer_rows(
    long_frame: pd.DataFrame, analysis_set: str
) -> list[dict[str, Any]]:
    import statsmodels.api as sm
    from scipy.stats import norm

    frame = long_frame.dropna(subset=["frac_snow", "effect_value"]).copy()
    if frame.empty or frame["basin_id"].nunique() < 3:
        return []
    frame["structure_Base"] = (frame["structure"] == "XAJ-Base").astype(float)
    frame["structure_CN"] = (frame["structure"] == "XAJ-CN").astype(float)
    frame["frac_snow_x_Base"] = frame["frac_snow"] * frame["structure_Base"]
    frame["frac_snow_x_CN"] = frame["frac_snow"] * frame["structure_CN"]
    columns = [
        "frac_snow",
        "structure_Base",
        "structure_CN",
        "frac_snow_x_Base",
        "frac_snow_x_CN",
    ]
    design = sm.add_constant(frame[columns], has_constant="add")
    fit = sm.OLS(frame["effect_value"].to_numpy(float), design).fit(
        cov_type="cluster",
        cov_kwds={"groups": frame["basin_id"].astype(str).to_numpy()},
    )
    cov = np.asarray(fit.cov_params(), dtype=float)
    names = list(design.columns)
    terms = [
        ("beta0_intercept", ["const"], "intercept"),
        ("beta1_frac_snow_TGD", ["frac_snow"], "TGD"),
        ("beta2_structure_Base", ["structure_Base"], "Base"),
        ("beta2_structure_CN", ["structure_CN"], "CN"),
        ("beta3_frac_snow_x_Base", ["frac_snow_x_Base"], "Base"),
        ("beta3_frac_snow_x_CN", ["frac_snow_x_CN"], "CN"),
    ]
    rows = []
    for term, components, structure in terms:
        weights = np.zeros(len(names))
        if term == "beta1_frac_snow_TGD":
            weights[names.index("frac_snow")] = 1.0
        else:
            for component in components:
                weights[names.index(component)] = 1.0
        estimate = float(weights @ np.asarray(fit.params, dtype=float))
        se = float(np.sqrt(max(0.0, weights @ cov @ weights)))
        z = estimate / se if se else math.nan
        p = float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else math.nan
        low, high = estimate - 1.96 * se, estimate + 1.96 * se
        rows.append(
            {
                "paradigm": "combined",
                "effect": "D_transfer_gradient",
                "period": "test_minus_train",
                "analysis_type": "transfer_gradient_interaction"
                if term.startswith("beta")
                else "transfer_gradient_interaction",
                "term": term,
                "structure": structure,
                "estimate": estimate,
                "std_error": se,
                "ci_low": low,
                "ci_high": high,
                "p_value": p,
                "matched_basin_count": int(frame["basin_id"].nunique()),
                "reference_category": "XAJ-TGD",
                "model_specification": "D ~ frac_snow + structure + frac_snow:structure; cluster=basin_id; TGD reference",
                "support_status": support_status(low, high),
                "analysis_set": analysis_set,
                "summary_level": "full_sample",
                "result_role": "primary",
            }
        )
    slopes = [
        ("XAJ-TGD", {"frac_snow": 1.0}),
        ("XAJ-Base", {"frac_snow": 1.0, "frac_snow_x_Base": 1.0}),
        ("XAJ-CN", {"frac_snow": 1.0, "frac_snow_x_CN": 1.0}),
    ]
    for structure, component_weights in slopes:
        weights = np.zeros(len(names))
        for component, value in component_weights.items():
            weights[names.index(component)] = value
        estimate = float(weights @ np.asarray(fit.params, dtype=float))
        se = float(np.sqrt(max(0.0, weights @ cov @ weights)))
        z = estimate / se if se else math.nan
        p = float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else math.nan
        low, high = estimate - 1.96 * se, estimate + 1.96 * se
        rows.append(
            {
                "paradigm": "combined",
                "effect": "D_transfer_gradient",
                "period": "test_minus_train",
                "analysis_type": "transfer_gradient_structure_slope",
                "term": f"slope_{structure}",
                "structure": structure,
                "estimate": estimate,
                "std_error": se,
                "ci_low": low,
                "ci_high": high,
                "p_value": p,
                "matched_basin_count": int(frame["basin_id"].nunique()),
                "reference_category": "XAJ-TGD",
                "model_specification": "D ~ frac_snow + structure + frac_snow:structure; cluster=basin_id; TGD reference",
                "support_status": support_status(low, high),
                "analysis_set": analysis_set,
                "summary_level": "full_sample",
                "result_role": "primary",
            }
        )
    return rows


def _process_effect_frame(
    primary_models: pd.DataFrame,
    paradigm: str,
    period: str,
    signature: str,
    first: str,
    second: str,
) -> pd.DataFrame:
    subset = primary_models[
        (primary_models["paradigm"] == paradigm) & (primary_models["period"] == period)
    ]
    wide = subset.pivot_table(
        index="basin_id",
        columns="model",
        values=f"{signature}_error_absolute",
        aggfunc="first",
    ).reset_index()
    if first not in wide or second not in wide:
        return pd.DataFrame(columns=["basin_id", "effect_value"])
    return pd.DataFrame(
        {"basin_id": wide["basin_id"], "effect_value": wide[first] - wide[second]}
    ).dropna()


def append_remaining_checks(
    output: Path,
    data_root: Path,
    aggregate: dict[str, Any],
    random_seed: int = 20260730,
) -> dict[str, Any]:
    """Complete transfer, exposure, and process checks from existing basin tables."""
    performance = _normalise_basin_ids(
        pd.read_csv(output / "r1_basin_level_performance.csv")
    )
    attributes = _normalise_basin_ids(pd.read_csv(output / "r1_snow_attributes.csv"))
    attributes["frac_snow"] = pd.to_numeric(attributes["frac_snow"], errors="coerce")
    strata = _snow_strata(attributes)
    regions, region_meta = region_membership(
        data_root, performance["basin_id"].unique()
    )
    years = pd.read_csv(output / "r1_snow_signatures_basin_year.csv")
    years["basin_id"] = years["basin_id"].astype(str).str.zfill(8)
    primary_signatures = _primary_signature_models(years, 5)
    rows, bootstrap, tests, relationships = [], [], [], []
    transfer_basin = []
    for structure in STRUCTURES:
        wide = (
            performance[
                performance["model"].eq(structure) & performance["selected_run"]
            ]
            .pivot_table(
                index=["basin_id", "paradigm"],
                columns="period",
                values="kge",
                aggfunc="first",
            )
            .reset_index()
        )
        ic = wide[wide["paradigm"].eq("IC-CMA-ES")].set_index("basin_id")
        dpl = wide[wide["paradigm"].eq("dPL-MLP")].set_index("basin_id")
        joined = (
            ic.join(dpl, lsuffix="_ic", rsuffix="_dpl", how="inner")
            .reset_index()
            .merge(attributes[["basin_id", "frac_snow"]], on="basin_id", how="inner")
        )
        joined["A"] = joined["train_ic"] - joined["train_dpl"]
        joined["B"] = joined["test_ic"] - joined["test_dpl"]
        joined["D"] = joined["B"] - joined["A"]
        joined["G_IC"] = joined["train_ic"] - joined["test_ic"]
        joined["G_dPL"] = joined["train_dpl"] - joined["test_dpl"]
        joined["G_IC_minus_G_dPL"] = joined["G_IC"] - joined["G_dPL"]
        joined["structure"] = structure
        transfer_basin.append(joined)
        for effect, column, period, estimand in (
            ("A_IC_minus_dPL", "A", "train", "KGE_IC,train-KGE_dPL,train"),
            ("B_IC_minus_dPL", "B", "test", "KGE_IC,test-KGE_dPL,test"),
            ("D_IC_minus_dPL", "D", "test_minus_train", "B-A"),
            ("G_IC", "G_IC", "train_minus_test", "KGE_IC,train-KGE_IC,test"),
            ("G_dPL", "G_dPL", "train_minus_test", "KGE_dPL,train-KGE_dPL,test"),
            (
                "G_IC_minus_G_dPL",
                "G_IC_minus_G_dPL",
                "train_minus_test",
                "G_IC-G_dPL=-D",
            ),
        ):
            frame = joined[["basin_id", column]].rename(
                columns={column: "effect_value"}
            )
            row, boot, test = _remaining_effect_bundle(
                frame,
                paradigm="IC-dPL",
                model=structure,
                period=period,
                effect=effect,
                metric="kge",
                family="tgd_transfer",
                estimand=estimand,
                analysis_set="r1_remaining_transfer_full_sample",
                regions=regions,
                seed=random_seed + len(rows),
            )
            rows.append(row)
            bootstrap.extend(boot)
            tests.append(test)
        relation_frame = joined[["basin_id", "frac_snow", "D"]].rename(
            columns={"D": "effect_value"}
        )
        relationships.append(
            _remaining_relationship(
                relation_frame,
                paradigm="IC-dPL",
                effect=f"D_{structure.removeprefix('XAJ-')}",
                period="test_minus_train",
                analysis_type="transfer_gradient",
                x_metric="frac_snow",
                y_metric="D_IC_minus_dPL",
                analysis_set="r1_remaining_transfer_gradient",
                seed=random_seed + len(relationships),
                regions=regions,
                extra={"structure": structure, "reference_category": ""},
            )
        )
    transfer = (
        pd.concat(transfer_basin, ignore_index=True)
        if transfer_basin
        else pd.DataFrame()
    )
    for structure in STRUCTURES:
        frame = transfer[transfer["structure"].eq(structure)].copy()
        for effect, column, period in (
            ("A_IC_minus_dPL", "A", "train"),
            ("B_IC_minus_dPL", "B", "test"),
            ("D_IC_minus_dPL", "D", "test_minus_train"),
            ("G_IC", "G_IC", "train_minus_test"),
            ("G_dPL", "G_dPL", "train_minus_test"),
            ("G_IC_minus_G_dPL", "G_IC_minus_G_dPL", "train_minus_test"),
        ):
            basin_rows = (
                frame[["basin_id", column]]
                .rename(columns={column: "effect_value"})
                .copy()
            )
            basin_rows["frac_snow"] = frame["frac_snow"].to_numpy()
            basin_rows["structure"] = structure
            for stratum, interval, n in (
                strata.dropna(subset=["snow_stratum"])
                .drop_duplicates(["snow_stratum", "snow_interval", "stratum_n"])[
                    ["snow_stratum", "snow_interval", "stratum_n"]
                ]
                .itertuples(index=False, name=None)
            ):
                selected = basin_rows.merge(
                    strata[["basin_id", "snow_stratum"]], on="basin_id", how="left"
                )
                selected = selected[selected["snow_stratum"].eq(stratum)]
                row, boot, test = _remaining_effect_bundle(
                    selected,
                    paradigm="IC-dPL",
                    model=structure,
                    period=period,
                    effect=effect,
                    metric="kge",
                    family="tgd_transfer",
                    estimand=effect,
                    analysis_set="r1_remaining_transfer_snow_strata",
                    regions=regions,
                    seed=random_seed + len(rows),
                    extra={
                        "snow_stratum": stratum,
                        "snow_interval": interval,
                        "stratum_n": int(n),
                    },
                )
                rows.append(row)
                bootstrap.extend(boot)
                tests.append(test)

    transfer_long = transfer[["basin_id", "structure", "frac_snow", "D"]].rename(
        columns={"D": "effect_value"}
    )
    interaction_rows = _interaction_transfer_rows(
        transfer_long, "r1_remaining_transfer_gradient_interaction"
    )
    relationships.extend(interaction_rows)

    for paradigm in ("IC-CMA-ES", "dPL-MLP"):
        for effect in ("E_CN-Base", "E_TGD-Base", "E_CN-TGD"):
            frame = _primary_effect_source(
                _normalise_basin_ids(
                    pd.read_csv(output / "r1_generalization_effects_basin_level.csv")
                ),
                paradigm,
                effect,
                "test_minus_train",
            )
            frame = frame.merge(
                attributes[["basin_id", "frac_snow"]], on="basin_id", how="inner"
            )
            relationships.append(
                _remaining_relationship(
                    frame,
                    paradigm=paradigm,
                    effect=effect,
                    period="test_minus_train",
                    analysis_type="snow_exposure_continuous",
                    x_metric="frac_snow",
                    y_metric=effect,
                    analysis_set="r1_remaining_exposure_gradient",
                    seed=random_seed + len(relationships),
                    regions=regions,
                    extra={"effect_family": "generalization_exposure"},
                )
            )

    # Reconstruct primary five-year signature records from the existing basin-year table.
    primary_signature_effects = []
    for paradigm in ("IC-CMA-ES", "dPL-MLP"):
        models = ["XAJ-Base", "XAJ-TGD", "XAJ-CN"] + (
            ["HBV"] if paradigm == "dPL-MLP" else []
        )
        for period in ("train", "test"):
            for signature in ("CT", "AMJJ"):
                for model in models:
                    frame = primary_signatures[
                        (primary_signatures["paradigm"].eq(paradigm))
                        & (primary_signatures["model"].eq(model))
                        & (primary_signatures["period"].eq(period))
                    ]
                    values = frame[["basin_id", f"{signature}_error_absolute"]].rename(
                        columns={f"{signature}_error_absolute": "effect_value"}
                    )
                    stats = _median_statistics(
                        values["effect_value"],
                        np.random.default_rng(random_seed + len(rows)),
                    )
                    abs_row = {
                        "paradigm": paradigm,
                        "model": model,
                        "period": period,
                        "metric": f"{signature}_error_absolute",
                        "analysis_set": "r1_remaining_primary_min5",
                        "summary_level": "full_sample",
                        "result_role": "primary",
                        "status": "valid_complete_water_years_primary_min5",
                        "valid_year_requirement": "minimum_5_complete_water_years",
                        "metric_unit": "days"
                        if signature == "CT"
                        else "fractional_flow_error",
                        "aggregation_rule": "complete water-year median within basin; dPL median across seeds",
                        "bootstrap_statistic": "median",
                        **stats,
                    }
                    aggregate.setdefault("_remaining_absolute", []).append(abs_row)
                    abs_frame = values.merge(
                        attributes[["basin_id", "frac_snow"]],
                        on="basin_id",
                        how="inner",
                    )
                    abs_frame["region"] = abs_frame["basin_id"].map(regions)
                    low, high = (
                        block_bootstrap_median_ci(
                            abs_frame["effect_value"],
                            abs_frame["region"],
                            np.random.default_rng(random_seed + len(rows) + 1),
                        )
                        if regions and not abs_frame.empty
                        else (math.nan, math.nan)
                    )
                    bootstrap.append(
                        {
                            "record_type": "bootstrap",
                            "family": "absolute_signature_error",
                            "paradigm": paradigm,
                            "model": model,
                            "period": period,
                            "metric": f"{signature}_error_absolute",
                            "analysis_set": "r1_remaining_primary_min5",
                            "summary_level": "full_sample",
                            "result_role": "primary",
                            "n": stats["valid_basin_count"],
                            "ci_low": stats["bootstrap_ci_low"],
                            "ci_high": stats["bootstrap_ci_high"],
                            "resamples": 10000,
                            "seed": random_seed,
                            "method": "ordinary_basin_bootstrap_median",
                            "bootstrap_statistic": "median",
                            "status": "valid_complete_water_years_primary_min5",
                        }
                    )
                    if regions and not abs_frame.empty:
                        bootstrap.append(
                            {
                                "record_type": "bootstrap",
                                "family": "absolute_signature_error",
                                "paradigm": paradigm,
                                "model": model,
                                "period": period,
                                "metric": f"{signature}_error_absolute",
                                "analysis_set": "r1_remaining_primary_min5",
                                "summary_level": "full_sample",
                                "result_role": "primary",
                                "n": int(abs_frame["basin_id"].nunique()),
                                "ci_low": low,
                                "ci_high": high,
                                "resamples": 10000,
                                "seed": random_seed,
                                "method": "region_block_bootstrap_median",
                                "bootstrap_statistic": "median",
                                "block_count": int(abs_frame["region"].nunique()),
                                "block_source": "data/basin_groups/group_11..group_17.npy",
                                "status": "valid_region_block_bootstrap",
                            }
                        )
    for paradigm in ("IC-CMA-ES", "dPL-MLP"):
        for period in ("train", "test"):
            for signature in ("CT", "AMJJ"):
                for first, second, effect in (
                    ("XAJ-Base", "XAJ-CN", "R_CN-Base"),
                    ("XAJ-Base", "XAJ-TGD", "R_TGD-Base"),
                    ("XAJ-TGD", "XAJ-CN", "R_CN-TGD"),
                ):
                    frame = _process_effect_frame(
                        primary_signatures, paradigm, period, signature, first, second
                    )
                    if frame.empty:
                        continue
                    frame = frame.merge(
                        attributes[["basin_id", "frac_snow"]],
                        on="basin_id",
                        how="inner",
                    )
                    row, boot, test = _remaining_effect_bundle(
                        frame,
                        paradigm=paradigm,
                        model=f"{first}_vs_{second}",
                        period=period,
                        effect=effect,
                        metric=f"{signature}_error_reduction",
                        family="signature_effect",
                        estimand=f"|E_{first}|-|E_{second}|",
                        analysis_set="r1_remaining_signature_effect",
                        regions=regions,
                        seed=random_seed + len(rows),
                        extra={
                            "valid_year_requirement": "minimum_5_complete_water_years",
                            "signature": signature,
                            "comparison": effect,
                            "metric_unit": "days"
                            if signature == "CT"
                            else "fractional_flow_error",
                        },
                    )
                    rows.append(row)
                    bootstrap.extend(boot)
                    tests.append(test)
                    for stratum, interval, n in (
                        strata.dropna(subset=["snow_stratum"])
                        .drop_duplicates(
                            ["snow_stratum", "snow_interval", "stratum_n"]
                        )[["snow_stratum", "snow_interval", "stratum_n"]]
                        .itertuples(index=False, name=None)
                    ):
                        selected = frame.merge(
                            strata[["basin_id", "snow_stratum"]],
                            on="basin_id",
                            how="left",
                        )
                        selected = selected[selected["snow_stratum"].eq(stratum)]
                        srow, sboot, stest = _remaining_effect_bundle(
                            selected,
                            paradigm=paradigm,
                            model=f"{first}_vs_{second}",
                            period=period,
                            effect=effect,
                            metric=f"{signature}_error_reduction",
                            family="signature_effect",
                            estimand=f"|E_{first}|-|E_{second}|",
                            analysis_set="r1_remaining_signature_snow_strata",
                            regions=regions,
                            seed=random_seed + len(rows),
                            extra={
                                "valid_year_requirement": "minimum_5_complete_water_years",
                                "signature": signature,
                                "comparison": effect,
                                "metric_unit": "days"
                                if signature == "CT"
                                else "fractional_flow_error",
                                "snow_stratum": stratum,
                                "snow_interval": interval,
                                "stratum_n": int(n),
                            },
                        )
                        rows.append(srow)
                        bootstrap.extend(sboot)
                        tests.append(stest)
                    relationships.append(
                        _remaining_relationship(
                            frame,
                            paradigm=paradigm,
                            effect=effect,
                            period=period,
                            analysis_type="process_gradient",
                            x_metric="frac_snow",
                            y_metric=f"{signature}_{effect}",
                            analysis_set="r1_remaining_process_gradient",
                            seed=random_seed + len(relationships),
                            regions=regions,
                            extra={
                                "signature": signature,
                                "comparison": effect,
                                "metric_unit": "days"
                                if signature == "CT"
                                else "fractional_flow_error",
                            },
                        )
                    )
                    for stratum, interval, n in (
                        strata.dropna(subset=["snow_stratum"])
                        .drop_duplicates(
                            ["snow_stratum", "snow_interval", "stratum_n"]
                        )[["snow_stratum", "snow_interval", "stratum_n"]]
                        .itertuples(index=False, name=None)
                    ):
                        selected = frame.merge(
                            strata[["basin_id", "snow_stratum"]],
                            on="basin_id",
                            how="left",
                        )
                        selected = selected[selected["snow_stratum"].eq(stratum)]
                        relationships.append(
                            _remaining_relationship(
                                selected,
                                paradigm=paradigm,
                                effect=effect,
                                period=period,
                                analysis_type="process_gradient_snow_stratum",
                                x_metric="frac_snow",
                                y_metric=f"{signature}_{effect}",
                                analysis_set="r1_remaining_process_gradient_snow_strata",
                                seed=random_seed + len(relationships),
                                regions=regions,
                                extra={
                                    "signature": signature,
                                    "comparison": effect,
                                    "metric_unit": "days"
                                    if signature == "CT"
                                    else "fractional_flow_error",
                                    "snow_stratum": stratum,
                                    "snow_interval": interval,
                                    "stratum_n": int(n),
                                },
                            )
                        )
                    if effect == "R_CN-TGD" and period == "test":
                        kge = _primary_effect_source(
                            _normalise_basin_ids(
                                pd.read_csv(
                                    output / "r1_structural_effects_basin_level.csv"
                                )
                            ),
                            paradigm,
                            "CN-TGD",
                            period,
                        )
                        kge = kge.merge(
                            frame[["basin_id", "effect_value"]],
                            on="basin_id",
                            how="inner",
                            suffixes=("_kge", "_signature"),
                        )
                        relationships.append(
                            _remaining_relationship(
                                kge.rename(
                                    columns={
                                        "effect_value_kge": "kge_value",
                                        "effect_value_signature": "effect_value",
                                    }
                                ).assign(
                                    effect_value=lambda x: x["effect_value"],
                                    frac_snow=kge["basin_id"].map(
                                        attributes.set_index("basin_id")["frac_snow"]
                                    ),
                                ),
                                paradigm=paradigm,
                                effect=f"KGE_CN-TGD_vs_{signature}",
                                period=period,
                                analysis_type="kge_signature_association",
                                x_metric="KGE_CN-TGD",
                                y_metric=f"{signature}_R_CN-TGD",
                                analysis_set="r1_remaining_kge_signature_association",
                                seed=random_seed + len(relationships),
                                regions=regions,
                                extra={
                                    "signature": signature,
                                    "comparison": effect,
                                    "association_x": "KGE_CN-TGD_test",
                                },
                            )
                        )

    remaining_rows, remaining_tests = _finish_remaining_tests(rows, tests)
    if not remaining_rows.empty:
        test_lookup = (
            remaining_tests.drop_duplicates("_summary_key").set_index("_summary_key")
            if not remaining_tests.empty
            else pd.DataFrame()
        )
        if not test_lookup.empty:
            remaining_rows["wilcoxon_bh_p"] = remaining_rows["_summary_key"].map(
                test_lookup["wilcoxon_bh_p"]
            )
            remaining_rows["sign_test_bh_p"] = remaining_rows["_summary_key"].map(
                test_lookup["sign_test_bh_p"]
            )
    absolute = _drop_remaining(aggregate["absolute"])
    absolute = pd.concat(
        [absolute, pd.DataFrame(aggregate.get("_remaining_absolute", []))],
        ignore_index=True,
        sort=False,
    )
    paired = pd.concat(
        [
            _drop_remaining(aggregate["paired"]),
            remaining_rows.drop(columns=["_summary_key"], errors="ignore"),
        ],
        ignore_index=True,
        sort=False,
    )
    relationships_frame = pd.concat(
        [_drop_remaining(aggregate["relationships"]), pd.DataFrame(relationships)],
        ignore_index=True,
        sort=False,
    )
    bootstrap_frame = pd.concat(
        [_drop_remaining(aggregate["bootstrap"]), pd.DataFrame(bootstrap)],
        ignore_index=True,
        sort=False,
    )
    tests_frame = pd.concat(
        [
            _drop_remaining(aggregate["tests"]),
            remaining_tests.drop(columns=["_summary_key"], errors="ignore"),
        ],
        ignore_index=True,
        sort=False,
    )
    generalization_path = output / "r1_generalization_effects_basin_level.csv"
    generalization = _normalise_basin_ids(pd.read_csv(generalization_path))
    existing_transfer = generalization[
        generalization.get("effect_family", pd.Series(index=generalization.index)).eq(
            "tgd_transfer"
        )
    ]
    generalization = generalization[
        ~generalization.get("effect_family", pd.Series(index=generalization.index)).eq(
            "tgd_transfer"
        )
    ]
    transfer_rows = []
    for _, item in transfer.iterrows():
        for effect, column, period in (
            ("A_IC_minus_dPL", "A", "train"),
            ("B_IC_minus_dPL", "B", "test"),
            ("D_IC_minus_dPL", "D", "test_minus_train"),
            ("G_IC", "G_IC", "train_minus_test"),
            ("G_dPL", "G_dPL", "train_minus_test"),
            ("G_IC_minus_G_dPL", "G_IC_minus_G_dPL", "train_minus_test"),
        ):
            transfer_rows.append(
                {
                    "basin_id": item["basin_id"],
                    "paradigm": "IC-dPL",
                    "model": item["structure"],
                    "period": period,
                    "effect": effect,
                    "effect_value": item[column],
                    "status": "valid",
                    "metric": "kge",
                    "effect_family": "tgd_transfer",
                    "estimand": effect,
                    "seed_or_restart": "matched_primary",
                    "structure": item["structure"],
                    "analysis_set": "r1_remaining_transfer_full_sample",
                }
            )
    generalization = pd.concat(
        [generalization, pd.DataFrame(transfer_rows)], ignore_index=True, sort=False
    )
    signature_path = output / "r1_signature_effects_basin_level.csv"
    signature_table = _normalise_basin_ids(pd.read_csv(signature_path)).drop(
        columns=["snow_stratum", "snow_interval", "stratum_n"], errors="ignore"
    )
    signature_table = signature_table.merge(
        strata[["basin_id", "snow_stratum", "snow_interval", "stratum_n"]],
        on="basin_id",
        how="left",
    )
    signature_table["effect_family"] = "signature_error_reduction"
    aggregate.update(
        {
            "absolute": absolute,
            "paired": paired,
            "relationships": relationships_frame,
            "bootstrap": bootstrap_frame,
            "tests": tests_frame,
            "generalization_basin": generalization,
            "signature_basin": signature_table,
            "performance": performance,
            "attributes": attributes,
            "snow_basin_strata": strata,
            "remaining": {
                "transfer": transfer,
                "interaction": pd.DataFrame(interaction_rows),
                "process_models": primary_signatures,
                "region_meta": region_meta,
            },
        }
    )
    return aggregate
