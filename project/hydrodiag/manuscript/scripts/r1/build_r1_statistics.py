"""Build the read-only R1 statistical package from existing result metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from r1_daily_inference import run_daily_export  # noqa: E402
from r1_metrics import support_status  # noqa: E402
from r1_statistics import (  # noqa: E402
    FULL_SAMPLE,
    MODEL_ALIASES,
    STRUCTURES,
    aggregate_full_sample,
    append_remaining_checks,
    append_snow_stratified,
    build_statistics_from_daily,
    generalization_effects,
    ic_dpl_effects,
    load_dpl,
    load_ic,
    median_dpl_rows,
    pivot_effects,
    signature_tables_from_years,
    snow_attributes,
    snow_relationships,
    write_summary_tables,
)

SEED = 20260730
PERIOD_TEXT = "warmup 1980-10-01..1981-09-30; train 1981-10-01..1995-09-30; test 1995-10-01..2010-09-30"
PERFORMANCE_COLUMNS = [
    "basin_id",
    "paradigm",
    "model",
    "period",
    "seed_or_restart",
    "selected_run",
    "kge_prime",
    "stored_original_kge",
    "nse",
    "pbias",
    "rmse",
    "valid_observation_count",
    "valid_simulation_count",
    "valid_days",
    "period_start",
    "period_end",
    "discharge_unit",
    "status",
    "source_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get(
            "HYDRODIAG_DATA_ROOT",
            str(Path(__file__).resolve().parents[3] / "data"),
        )),
    )
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=(
            "audit",
            "summary",
            "daily-inference",
            "statistics",
            "full",
            "merge-partitions",
        ),
        default="full",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("XAJ", "XAJ_TGD2", "XAJ_CN", "HBV"),
        default=None,
    )
    parser.add_argument("--tgd2-epoch", type=int, default=None)
    parser.add_argument("--paradigm", choices=("all", "ic", "dpl"), default="all")
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--partition-suffix", default="")
    parser.add_argument("--partition-root", type=Path, default=None)
    return parser.parse_args()


def git_status(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True
        ).strip()
    except Exception as exc:
        return f"git status unavailable: {exc}"


def script_hashes() -> dict[str, str]:
    hashes = {}
    for name in (
        "build_r1_statistics.py",
        "r1_daily_inference.py",
        "r1_metrics.py",
        "r1_statistics.py",
    ):
        path = HERE / name
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    return hashes


def canonical_summary_command(project: Path, data: Path, results: Path) -> str:
    """Return the stable reproduction command stored in deterministic metadata."""
    output = project / "manuscript" / "results" / "R1"
    return (
        "PYTHONDONTWRITEBYTECODE=1 python manuscript/scripts/r1/build_r1_statistics.py "
        f"--mode summary --project-root {project} --results-root {results} "
        f"--data-root {data} --output-root {output}"
    )


def output_schemas(output: Path) -> dict[str, list[str]]:
    schemas = {}
    for path in sorted(output.glob("*.csv")):
        schemas[path.name] = list(pd.read_csv(path, nrows=0).columns)
    return schemas


def deterministic_artifact_records(output: Path) -> list[dict[str, Any]]:
    """Describe summary artifacts without hashing the manifest itself."""
    logical_keys = {
        "r1_absolute_metrics_summary.csv": "paradigm|model|period|metric|analysis_set|snow_stratum|result_role|valid_year_requirement",
        "r1_paired_effects_summary.csv": "paradigm|model|period|effect|metric|analysis_set|robustness_type|seed_or_restart|snow_stratum|signature|comparison|result_role",
        "r1_bootstrap_intervals.csv": "record_type|family|paradigm|model|period|metric|effect|analysis_set|method|snow_stratum|signature|comparison|claim_id|robustness_type",
        "r1_statistical_tests.csv": "record_type|family|paradigm|model|period|metric|effect|analysis_set|method|snow_stratum|signature|comparison|claim_id|robustness_type",
        "r1_generalization_effects_basin_level.csv": "basin_id|paradigm|model|period|effect|metric|effect_family|seed_or_restart|analysis_set",
        "r1_structural_effects_basin_level.csv": "basin_id|paradigm|period|effect|metric|seed_or_restart|analysis_set",
        "r1_signature_effects_basin_level.csv": "basin_id|paradigm|period|seed_or_restart|signature|first_model|second_model|analysis_set|snow_stratum",
    }
    records: list[dict[str, Any]] = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.name in {"r1_result_manifest.json", "r1_execution.log"}:
            continue
        if path.suffix not in {".csv", ".json", ".md"}:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        record: dict[str, Any] = {
            "relative_path": path.name,
            "sha256": digest,
            "classification": "deterministic_scientific_artifact"
            if path.suffix == ".csv"
            else "deterministic_documentation_or_metadata",
        }
        if path.suffix == ".csv":
            record["row_count"] = int(pd.read_csv(path).shape[0])
            record["logical_key"] = logical_keys.get(path.name, "not_applicable")
        records.append(record)
    return records


def _number(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def _ci(
    row: pd.Series,
    *,
    median: str = "median",
    low: str = "bootstrap_ci_low",
    high: str = "bootstrap_ci_high",
) -> str:
    return f"{_number(row.get(median))} [{_number(row.get(low))}, {_number(row.get(high))}]"


def _iqr(row: pd.Series) -> str:
    return f"{_number(row.get('median'))} [{_number(row.get('p25'))}, {_number(row.get('p75'))}]"


def _lookup(frame: pd.DataFrame, **keys: Any) -> pd.Series | None:
    if frame.empty:
        return None
    mask = pd.Series(True, index=frame.index)
    for key, value in keys.items():
        if key in frame:
            mask &= frame[key].astype(str).eq(str(value))
    subset = frame[mask]
    return subset.iloc[0] if not subset.empty else None


def render_full_sample_notes(aggregate: dict[str, Any], existing_notes: str) -> str:
    """Render the compact full-sample report without basin-level records."""
    marker = "## R1 full-basin statistical summary"
    base = existing_notes.split(marker, 1)[0].rstrip()
    absolute = aggregate["absolute"]
    paired = aggregate["paired"]
    effects = aggregate["effect_summary"]
    bootstrap = aggregate["bootstrap"]
    relationships = aggregate["relationships"]

    lines = [
        marker,
        "",
        "All rows below are full-sample aggregates; the inferential unit is the basin and `n` is the valid matched basin count.",
        "",
        "### Absolute performance",
        "",
        "| paradigm | model | train KGE median [IQR] | test KGE median [IQR] | train-test gap median [95% CI] | n |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for paradigm in ("IC-CMA-ES", "dPL-MLP"):
        models = ["XAJ-Base", "XAJ-TGD", "XAJ-CN"] + (
            ["HBV"] if paradigm == "dPL-MLP" else []
        )
        for model in models:
            train = _lookup(
                absolute,
                paradigm=paradigm,
                model=model,
                period="train",
                metric="kge",
                analysis_set=FULL_SAMPLE,
            )
            test = _lookup(
                absolute,
                paradigm=paradigm,
                model=model,
                period="test",
                metric="kge",
                analysis_set=FULL_SAMPLE,
            )
            gap = _lookup(
                paired,
                paradigm=paradigm,
                model=model,
                period="train_minus_test",
                effect="train_minus_test",
                metric="kge",
                analysis_set=FULL_SAMPLE,
            )
            n = train.get("valid_basin_count", "NA") if train is not None else "NA"
            lines.append(
                f"| {paradigm} | {model} | {_iqr(train) if train is not None else 'NA'} | {_iqr(test) if test is not None else 'NA'} | {_ci(gap) if gap is not None else 'NA'} | {int(n) if pd.notna(n) else 'NA'} |"
            )

    lines += [
        "",
        "### Structural effects",
        "",
        "| paradigm | estimand | period | median effect [95% CI] | positive fraction | n | support status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for paradigm in ("IC-CMA-ES", "dPL-MLP"):
        for effect in ("CN-Base", "TGD-Base", "CN-TGD"):
            for period in ("train", "test"):
                row = _lookup(
                    effects,
                    paradigm=paradigm,
                    effect=effect,
                    period=period,
                    metric="kge",
                    analysis_set=FULL_SAMPLE,
                )
                if row is not None:
                    lines.append(
                        f"| {paradigm} | {row.get('estimand', effect)} | {period} | {_ci(row)} | {_number(row.get('fraction_positive'))} | {int(row['valid_basin_count'])} | {row.get('support_status', 'NA')} |"
                    )

    lines += [
        "",
        "### Generalization and paradigm transfer",
        "",
        "| estimand | paradigm or model | median effect [95% CI] | positive fraction | n | support status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for effect in ("E_CN-Base", "E_TGD-Base", "E_CN-TGD"):
        for paradigm in ("IC-CMA-ES", "dPL-MLP"):
            row = _lookup(
                effects,
                paradigm=paradigm,
                effect=effect,
                period="test_minus_train",
                metric="kge",
                analysis_set=FULL_SAMPLE,
            )
            if row is not None:
                lines.append(
                    f"| {effect} | {paradigm} | {_ci(row)} | {_number(row.get('fraction_positive'))} | {int(row['valid_basin_count'])} | {row.get('support_status', 'NA')} |"
                )
    for model in ("XAJ-Base", "XAJ-TGD", "XAJ-CN"):
        effect = f"D_{model.removeprefix('XAJ-')}"
        row = _lookup(
            effects,
            paradigm="IC-dPL",
            effect=effect,
            period="test_minus_train",
            metric="kge",
            analysis_set=FULL_SAMPLE,
        )
        if row is not None:
            lines.append(
                f"| {row.get('estimand', effect)} | IC-dPL / {model} | {_ci(row)} | {_number(row.get('fraction_positive'))} | {int(row['valid_basin_count'])} | {row.get('support_status', 'NA')} |"
            )

    lines += [
        "",
        "### Snow specificity",
        "",
        "| paradigm | CN-TGD test median [95% CI] | positive fraction | Spearman rho | robust slope [95% CI] | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for paradigm in ("IC-CMA-ES", "dPL-MLP"):
        row = _lookup(
            relationships,
            paradigm=paradigm,
            effect="CN-TGD",
            period="test",
            analysis_type="full_sample_attribute",
            analysis_set=FULL_SAMPLE,
        )
        if row is not None:
            lines.append(
                f"| {paradigm} | {_ci(row)} | {_number(row.get('fraction_greater_zero'))} | {_number(row.get('spearman_rho'))} | {_number(row.get('robust_slope'))} [{_number(row.get('robust_ci_low'))}, {_number(row.get('robust_ci_high'))}] | {int(row['paired_basin_count'])} |"
            )
    lines += [
        "",
        "Interaction model coefficients:",
        "",
        "| coefficient | estimate | standard error | 95% CI | p-value | reference | n |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    interaction = relationships[
        relationships["analysis_type"].eq("cluster_robust_interaction")
    ]
    for _, row in interaction.iterrows():
        lines.append(
            f"| {row.get('term', 'NA')} | {_number(row.get('estimate'))} | {_number(row.get('std_error'))} | [{_number(row.get('ci_low'))}, {_number(row.get('ci_high'))}] | {_number(row.get('p_value'))} | {row.get('reference_category', 'NA')} | {int(row['matched_basin_count']) if pd.notna(row.get('matched_basin_count')) else 'NA'} |"
        )

    lines += [
        "",
        "### Snowmelt signatures",
        "",
        "| paradigm | signature | comparison | median error reduction [95% CI] | positive fraction | n | support status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    sig_effects = effects[
        effects["metric"].isin(["CT_error_reduction", "AMJJ_error_reduction"])
    ]
    for _, row in sig_effects.sort_values(["paradigm", "metric", "effect"]).iterrows():
        lines.append(
            f"| {row.get('paradigm')} | {str(row.get('metric')).split('_')[0]} | {row.get('effect')} | {_ci(row)} | {_number(row.get('fraction_positive'))} | {int(row['valid_basin_count'])} | {row.get('support_status', 'NA')} |"
        )

    lines += [
        "",
        "### frac_snow fixed strata",
        "",
        "Project-fixed strata: S1 `[0, 0.05)`, S2 `[0.05, 0.15)`, S3 `[0.15, 0.30)`, S4 `[0.30, 0.50)`, S5 `[0.50, 1.00]`.",
        "",
        "| stratum | n | IC Base test KGE | IC TGD test KGE | IC CN test KGE | dPL Base test KGE | dPL TGD test KGE | dPL CN test KGE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    strata_table = aggregate.get("snow_strata", pd.DataFrame())
    for _, stratum_row in strata_table.iterrows():
        stratum = stratum_row["snow_stratum"]
        interval = stratum_row["snow_interval"]
        cells = []
        for paradigm in ("IC-CMA-ES", "dPL-MLP"):
            for model in ("XAJ-Base", "XAJ-TGD", "XAJ-CN"):
                row = _lookup(
                    absolute,
                    paradigm=paradigm,
                    model=model,
                    period="test",
                    metric="kge",
                    analysis_set="snow_fixed_strata",
                    snow_stratum=stratum,
                )
                cells.append(_ci(row) if row is not None else "NA")
        lines.append(
            f"| {stratum} {interval} | {int(stratum_row['stratum_n'])} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "| stratum | paradigm | CN-Base test effect | TGD-Base test effect | CN-TGD test effect |",
        "|---|---|---:|---:|---:|",
    ]
    for _, stratum_row in strata_table.iterrows():
        stratum = stratum_row["snow_stratum"]
        for paradigm in ("IC-CMA-ES", "dPL-MLP"):
            values = []
            for effect in ("CN-Base", "TGD-Base", "CN-TGD"):
                row = _lookup(
                    paired,
                    paradigm=paradigm,
                    effect=effect,
                    period="test",
                    metric="kge",
                    analysis_set="snow_fixed_strata",
                    snow_stratum=stratum,
                )
                values.append(_ci(row) if row is not None else "NA")
            lines.append(f"| {stratum} | {paradigm} | " + " | ".join(values) + " |")

    lines += [
        "",
        "### Robustness",
        "",
        "| effect | primary direction | IC restart agreement | dPL seed agreement | basin-bootstrap conclusion | region-block-bootstrap conclusion |",
        "|---|---|---|---|---|---|",
    ]
    robustness_specs = [
        ("test CN-Base", "CN-Base", "kge", "test"),
        ("test CN-TGD", "CN-TGD", "kge", "test"),
        ("E_CN-Base", "E_CN-Base", "kge", "test_minus_train"),
        ("E_CN-TGD", "E_CN-TGD", "kge", "test_minus_train"),
        ("CT CN-TGD", "R_CN-TGD", "CT_error_reduction", "test"),
        ("AMJJ CN-TGD", "R_CN-TGD", "AMJJ_error_reduction", "test"),
    ]
    old_robust = paired[
        paired.get("robustness_type", pd.Series(index=paired.index)).notna()
    ]
    for label, effect, metric, period in robustness_specs:
        primary = effects[
            (effects["effect"] == effect)
            & (effects["metric"] == metric)
            & (effects["period"] == period)
        ]
        direction = (
            "; ".join(sorted(primary["support_status"].dropna().astype(str).unique()))
            or "not available"
        )
        ic_effect = effect
        if metric.endswith("error_reduction"):
            ic_effect = "XAJ-TGD_minus_XAJ-CN"
        ic_rows = old_robust[
            (old_robust["robustness_type"] == "IC_restart_median")
            & (old_robust["effect"] == ic_effect)
            & (old_robust["metric"] == metric)
        ]
        ic_flags = (
            ic_rows["same_direction_as_primary"].dropna().astype(bool)
            if not ic_rows.empty
            else pd.Series(dtype=bool)
        )
        ic_agreement = (
            "yes"
            if not ic_rows.empty and len(ic_flags) == len(ic_rows) and ic_flags.all()
            else ("not available" if ic_rows.empty else "no")
        )
        dpl_rows = old_robust[
            (old_robust["robustness_type"] == "dPL_seed")
            & (old_robust["effect"] == ic_effect)
            & (old_robust["metric"] == metric)
            & old_robust["analysis_set"].eq("across_seed_summary")
        ]
        dpl_flags = (
            dpl_rows["all_seeds_agree"].dropna().astype(bool)
            if not dpl_rows.empty
            else pd.Series(dtype=bool)
        )
        dpl_agreement = (
            "yes"
            if not dpl_rows.empty
            and len(dpl_flags) == len(dpl_rows)
            and dpl_flags.all()
            else ("not available" if dpl_rows.empty else "no")
        )
        region = bootstrap[
            (bootstrap["method"].astype(str) == "region_block_bootstrap_median")
            & (bootstrap["effect"] == effect)
            & (bootstrap["metric"] == metric)
        ]
        region_status = (
            support_status(
                float(region.iloc[0]["ci_low"]), float(region.iloc[0]["ci_high"])
            )
            if not region.empty
            else "not available"
        )
        lines.append(
            f"| {label} | {direction} | {ic_agreement} | {dpl_agreement} | {direction} | {region_status} |"
        )

    lines += [
        "",
        "### Machine-generated claim status",
        "",
        "| claim ID | statistical status | primary estimate | 95% CI | valid basin count |",
        "|---|---|---:|---:|---:|",
    ]
    claims = effects[effects["claim_id"].astype(str).ne("")].sort_values(
        ["claim_id", "paradigm", "effect", "period"]
    )
    for _, row in claims.iterrows():
        lines.append(
            f"| {row.get('claim_id')} | {row.get('support_status', 'NA')} | {_number(row.get('median'))} | [{_number(row.get('bootstrap_ci_low'))}, {_number(row.get('bootstrap_ci_high'))}] | {int(row['valid_basin_count'])} |"
        )
    return base + "\n\n" + "\n".join(lines).rstrip() + "\n"


def render_remaining_notes(aggregate: dict[str, Any], existing_notes: str) -> str:
    """Append compact transfer, exposure, and process-check tables."""
    marker = "## TGD transfer and snow-stratified exposure checks"
    base = existing_notes.split(marker, 1)[0].rstrip()
    paired = aggregate["paired"]
    absolute = aggregate["absolute"]
    relationships = aggregate["relationships"]
    performance = aggregate["performance"]
    strata = aggregate["snow_strata"]
    basin_strata = aggregate["snow_basin_strata"]

    def number(value: Any) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "NA"
        return "NA" if not np.isfinite(value) else f"{value:.3f}"

    def ci(row: pd.Series | None) -> str:
        if row is None:
            return "NA"
        return f"{number(row.get('median'))} [{number(row.get('bootstrap_ci_low'))}, {number(row.get('bootstrap_ci_high'))}]"

    def lookup(frame: pd.DataFrame, **keys: Any) -> pd.Series | None:
        if frame.empty:
            return None
        mask = pd.Series(True, index=frame.index)
        for key, value in keys.items():
            if key in frame:
                mask &= frame[key].astype(str).eq(str(value))
        found = frame[mask]
        return found.iloc[0] if not found.empty else None

    lines = [
        marker,
        "",
        "All quantities are basin-wise; no individual basins are listed.",
        "",
        "### TGD IC versus dPL by snow stratum",
        "",
        "| snow stratum | n | IC train KGE | dPL train KGE | train advantage A [95% CI] | IC test KGE | dPL test KGE | test advantage B [95% CI] | transfer D [95% CI] | support status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, item in strata.iterrows():
        stratum = item["snow_stratum"]
        basin_ids = basin_strata.loc[
            basin_strata["snow_stratum"].eq(stratum), "basin_id"
        ]
        train_ic = performance[
            (performance["paradigm"] == "IC-CMA-ES")
            & (performance["model"] == "XAJ-TGD")
            & (performance["period"] == "train")
            & performance["basin_id"].isin(basin_ids)
        ]["kge"]
        train_dpl = performance[
            (performance["paradigm"] == "dPL-MLP")
            & (performance["model"] == "XAJ-TGD")
            & (performance["period"] == "train")
            & performance["basin_id"].isin(basin_ids)
        ]["kge"]
        test_ic = performance[
            (performance["paradigm"] == "IC-CMA-ES")
            & (performance["model"] == "XAJ-TGD")
            & (performance["period"] == "test")
            & performance["basin_id"].isin(basin_ids)
        ]["kge"]
        test_dpl = performance[
            (performance["paradigm"] == "dPL-MLP")
            & (performance["model"] == "XAJ-TGD")
            & (performance["period"] == "test")
            & performance["basin_id"].isin(basin_ids)
        ]["kge"]
        a = lookup(
            paired,
            analysis_set="r1_remaining_transfer_snow_strata",
            snow_stratum=stratum,
            model="XAJ-TGD",
            effect="A_IC_minus_dPL",
            period="train",
        )
        b = lookup(
            paired,
            analysis_set="r1_remaining_transfer_snow_strata",
            snow_stratum=stratum,
            model="XAJ-TGD",
            effect="B_IC_minus_dPL",
            period="test",
        )
        d = lookup(
            paired,
            analysis_set="r1_remaining_transfer_snow_strata",
            snow_stratum=stratum,
            model="XAJ-TGD",
            effect="D_IC_minus_dPL",
            period="test_minus_train",
        )
        lines.append(
            f"| {stratum} | {int(item['stratum_n'])} | {number(train_ic.median())} | {number(train_dpl.median())} | {ci(a)} | {number(test_ic.median())} | {number(test_dpl.median())} | {ci(b)} | {ci(d)} | {d.get('support_status', 'NA') if d is not None else 'NA'} |"
        )

    lines += [
        "",
        "### Transfer-loss snow gradient",
        "",
        "| structure | Spearman rho | robust slope [95% CI] | OLS slope [95% CI] | n |",
        "|---|---:|---:|---:|---:|",
    ]
    gradients = relationships[
        relationships.get("analysis_set", pd.Series(index=relationships.index)).eq(
            "r1_remaining_transfer_gradient"
        )
        & relationships.get("analysis_type", pd.Series(index=relationships.index)).eq(
            "transfer_gradient"
        )
    ]
    for _, row in gradients.sort_values("structure").iterrows():
        lines.append(
            f"| {row.get('structure', 'NA')} | {number(row.get('spearman_rho'))} | {number(row.get('robust_slope'))} [{number(row.get('regional_slope_ci_low', row.get('robust_ci_low')))}, {number(row.get('regional_slope_ci_high', row.get('robust_ci_high')))}] | {number(row.get('ols_slope'))} [{number(row.get('ols_ci_low'))}, {number(row.get('ols_ci_high'))}] | {int(row.get('paired_basin_count'))} |"
        )
    interaction = relationships[
        relationships.get("analysis_set", pd.Series(index=relationships.index)).eq(
            "r1_remaining_transfer_gradient_interaction"
        )
    ]
    lines += [
        "",
        "Transfer-gradient interaction coefficients (TGD reference; basin-clustered standard errors):",
        "",
        "| coefficient | estimate | SE | 95% CI | p-value | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in interaction[
        interaction.get("analysis_type", pd.Series(index=interaction.index)).eq(
            "transfer_gradient_interaction"
        )
    ].iterrows():
        lines.append(
            f"| {row.get('term', 'NA')} | {number(row.get('estimate'))} | {number(row.get('std_error'))} | [{number(row.get('ci_low'))}, {number(row.get('ci_high'))}] | {number(row.get('p_value'))} | {int(row.get('matched_basin_count'))} |"
        )

    lines += [
        "",
        "### Snow-stratified structural effects",
        "",
        "| paradigm | stratum | period | Base / TGD / CN KGE | CN-Base / TGD-Base / CN-TGD effects |",
        "|---|---|---|---:|---:|",
    ]
    for stratum in strata["snow_stratum"]:
        for paradigm in ("IC-CMA-ES", "dPL-MLP"):
            for period in ("train", "test"):
                cells = []
                for model in ("XAJ-Base", "XAJ-TGD", "XAJ-CN"):
                    row = lookup(
                        absolute,
                        analysis_set="snow_fixed_strata",
                        snow_stratum=stratum,
                        paradigm=paradigm,
                        model=model,
                        period=period,
                        metric="kge",
                    )
                    cells.append(number(row.get("median")) if row is not None else "NA")
                effects = []
                for effect in ("CN-Base", "TGD-Base", "CN-TGD"):
                    row = lookup(
                        paired,
                        analysis_set="snow_fixed_strata",
                        snow_stratum=stratum,
                        paradigm=paradigm,
                        effect=effect,
                        period=period,
                        metric="kge",
                    )
                    effects.append(ci(row))
                lines.append(
                    f"| {paradigm} | {stratum} | {period} | {' / '.join(cells)} | {' / '.join(effects)} |"
                )

    lines += [
        "",
        "### Snow-stratified exposure",
        "",
        "| paradigm | stratum | estimand | median [95% CI] | positive fraction | n | support status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    exposure = paired[
        (
            paired.get("analysis_set", pd.Series(index=paired.index)).eq(
                "snow_fixed_strata"
            )
        )
        & paired.get("effect", pd.Series(index=paired.index)).isin(
            ["E_CN-Base", "E_TGD-Base", "E_CN-TGD"]
        )
    ]
    for _, row in exposure.sort_values(
        ["paradigm", "snow_stratum", "effect"]
    ).iterrows():
        lines.append(
            f"| {row.get('paradigm')} | {row.get('snow_stratum')} | {row.get('effect')} | {ci(row)} | {number(row.get('fraction_positive'))} | {int(row.get('valid_basin_count'))} | {row.get('support_status')} |"
        )

    lines += [
        "",
        "### CT and AMJJ",
        "",
        "| paradigm | signature | analysis set | comparison | median [95% CI] | region-block interval | positive fraction | n | support status |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    process = paired[
        paired.get("analysis_set", pd.Series(index=paired.index)).isin(
            ["r1_remaining_signature_effect", "r1_remaining_signature_snow_strata"]
        )
    ]
    for _, row in (
        process[(process.get("snow_stratum", pd.Series(index=process.index)).isna())]
        .sort_values(["paradigm", "metric", "effect"])
        .iterrows()
    ):
        region = f"[{number(row.get('regional_bootstrap_ci_low'))}, {number(row.get('regional_bootstrap_ci_high'))}]"
        lines.append(
            f"| {row.get('paradigm')} | {row.get('signature', row.get('metric', '').split('_')[0])} | {row.get('analysis_set')} | {row.get('effect')} | {ci(row)} | {region} | {number(row.get('fraction_positive'))} | {int(row.get('valid_basin_count'))} | {row.get('support_status')} |"
        )

    lines += [
        "",
        "### Process-gradient evidence",
        "",
        "| paradigm | signature | comparison | Spearman rho | robust slope [95% CI] | n |",
        "|---|---|---|---:|---:|---:|",
    ]
    process_rel = relationships[
        relationships.get("analysis_type", pd.Series(index=relationships.index)).eq(
            "process_gradient"
        )
        & relationships.get("analysis_set", pd.Series(index=relationships.index)).eq(
            "r1_remaining_process_gradient"
        )
    ]
    for _, row in process_rel.sort_values(
        ["paradigm", "signature", "comparison"]
    ).iterrows():
        lines.append(
            f"| {row.get('paradigm')} | {row.get('signature')} | {row.get('comparison')} | {number(row.get('spearman_rho'))} | {number(row.get('robust_slope'))} [{number(row.get('regional_slope_ci_low', row.get('robust_ci_low')))}, {number(row.get('regional_slope_ci_high', row.get('robust_ci_high')))}] | {int(row.get('paired_basin_count'))} |"
        )
    return base + "\n\n" + "\n".join(lines).rstrip() + "\n"


def setup_logging(output: Path, *, append: bool = False) -> logging.Logger:
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("r1")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(
        output / "r1_execution.log", mode="a" if append else "w", encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    return logger


def write_csv(
    table: pd.DataFrame, path: Path, columns: list[str] | None = None
) -> None:
    if columns is not None:
        for column in columns:
            if column not in table:
                table[column] = np.nan
        table = table[columns]
    table.to_csv(
        path,
        index=False,
        na_rep="",
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.17g",
    )


def stable_output_table(table: pd.DataFrame) -> pd.DataFrame:
    """Make task-owned CSV row order independent of concatenation order."""
    if table.empty:
        return table
    columns = list(table.columns)
    return table.sort_values(
        columns,
        key=lambda values: values.astype(str),
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def write_stable_output_table(table: pd.DataFrame, path: Path) -> None:
    """Write a sorted summary table with explicit, round-trip-safe serialization."""
    stable_output_table(table).to_csv(
        path,
        index=False,
        na_rep="",
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.17g",
    )


def inventory_row(
    name: str,
    paradigm: str,
    model: str,
    path: Path,
    *,
    source: str,
    basin_ids: list[str],
    seeds: list[str],
    missing: str,
    status: str,
    interpretation: str,
    train: str = "1981-10-01..1995-09-30",
    test: str = "1995-10-01..2010-09-30",
) -> dict[str, Any]:
    return {
        "result_set": name,
        "model": model,
        "calibration_paradigm": paradigm,
        "train_period": train,
        "test_period": test,
        "basin_count": len(set(basin_ids)),
        "basin_ids": json.dumps(sorted(set(basin_ids))),
        "seed_or_restart_count": len(seeds),
        "seed_or_restart_ids": json.dumps(seeds),
        "prediction_file_path": "",
        "observation_file_path": str(
            Path("data/camels_dataset")
        ),
        "forcing_or_attribute_file_path": str(
            Path("data/camels_dataset")
        ),
        "date_coverage": "1980-10-01..2014-09-30",
        "discharge_units": "raw ft3/s; IC/dPL target conversion mm/day",
        "missing_files": missing,
        "duplicated_basins": "none observed",
        "incomplete_runs": status,
        "non_finite_values": "not observed in stored KGE records"
        if status == "complete"
        else "not assessed",
        "failed_or_unusable_runs": "none observed" if status == "complete" else status,
        "interpretation": interpretation,
        "source_path": str(path),
        "source_evidence": source,
    }


def build_inventory(project: Path, data: Path) -> tuple[pd.DataFrame, list[str]]:
    results = project / "results"
    inventory: list[dict[str, Any]] = []
    notes: list[str] = []
    ids = [str(x).zfill(8) for x in json.loads((data / "531sub_id.txt").read_text())]
    ic_specs = [
        (
            "xaj_base_cmaes_531_batched_paired_v2",
            "XAJ",
            "XAJ-Base",
            "formal manifest + 5,310 raw JSON records",
        ),
        (
            "xaj_cn_cmaes_531_batched_paired_v2",
            "XAJ_CN",
            "XAJ-CN",
            "formal manifest + 5,310 raw JSON records",
        ),
        (
            "xaj_tgd2_cmaes_531_batched_v1",
            "XAJ_TGD2",
            "XAJ-TGD",
            "formal manifest + 5,310 raw JSON records; TGD2 is the current repository key",
        ),
    ]
    for directory, key, model, evidence in ic_specs:
        root = results / directory
        raw = (
            root
            / "raw"
            / {"XAJ": "xaj", "XAJ_CN": "xaj_cn", "XAJ_TGD2": "xaj_tgd2"}[key]
        )
        files = sorted(raw.glob("*.json"))
        observed = []
        for file in files:
            try:
                observed.append(
                    str(json.loads(file.read_text()).get("basin_id", "")).zfill(8)
                )
            except Exception:
                notes.append(f"unreadable IC file: {file}")
        starts = sorted({file.stem.rsplit("_start", 1)[-1] for file in files})
        inventory.append(
            inventory_row(
                directory,
                "IC-CMA-ES",
                model,
                root,
                source=evidence,
                basin_ids=observed,
                seeds=[f"restart_{x}" for x in starts],
                missing="none"
                if len(files) == 5310
                else f"expected 5310 raw JSON, found {len(files)}",
                status="complete"
                if len(files) == 5310 and len(set(observed)) == 531
                else "incomplete",
                interpretation=f"{key} -> {model}; stored metric is original KGE, not R1 KGE'",
            )
        )
    dpl_specs = [("XAJ", "XAJ-Base"), ("XAJ_CN", "XAJ-CN"), ("HBV", "HBV")]
    dpl_root = results / "dpl_camels_531_lite_v2"
    for key, model in dpl_specs:
        root = dpl_root / key
        paths = sorted(root.glob("seed_*/train_test_kge_by_basin.csv"))
        observed = []
        for path in paths:
            observed.extend(
                pd.read_csv(path)["basin_id"].astype(str).str.zfill(8).tolist()
            )
        seeds = [path.parent.name for path in paths]
        inventory.append(
            inventory_row(
                f"dpl_camels_531_lite_v2/{key}",
                "dPL-MLP",
                model,
                root,
                source="three completed seed CSV files",
                basin_ids=observed,
                seeds=seeds,
                missing="none"
                if len(paths) == 3
                else "one or more seed CSV files missing",
                status="complete"
                if len(paths) == 3
                and all(observed.count(x) == 3 for x in set(observed))
                else "incomplete",
                interpretation=f"{key} -> {model}; completed historical Lite-v2 tree; not substituted for current XAJ_TGD2",
            )
        )
    active = results / "dpl_camels_531_lite_v3_tgd2_dpl_audited"
    active_paths = sorted((active / "XAJ_TGD2").glob("seed_*/checkpoint_epoch_*.pt"))
    active_ids = [str(x).zfill(8) for x in ids]
    inventory.append(
        inventory_row(
            "dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2",
            "dPL-MLP",
            "XAJ-TGD",
            active,
            source="active TGD2 checkpoint metadata and epoch histories",
            basin_ids=active_ids,
            seeds=sorted({p.parent.name for p in active_paths}),
            missing="no COMPLETE marker; provisional periodic checkpoints are available",
            status="provisional current artifact included in R1",
            interpretation="XAJ_TGD2 -> XAJ-TGD; latest common valid periodic checkpoint selected by metadata",
        )
    )
    for path in sorted((results / "archive").glob("**/*xnes*"))[:20]:
        notes.append(
            f"historical XNES path discovered and excluded from main R1: {path}"
        )
    notes.append(
        "obsolete GD design, historical XAJ_TGD, and XNES results are excluded from main outputs"
    )
    return pd.DataFrame(inventory), notes


def unresolved_signature_table(kind: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paradigm": "",
                "model": "",
                "basin_id": "",
                "water_year": "",
                "signature": signature,
                "obs_value": np.nan,
                "sim_value": np.nan,
                "error_signed": np.nan,
                "error_absolute": np.nan,
                "valid_years": np.nan,
                "status": "unresolved_no_saved_daily_simulation",
                "notes": "No complete daily simulated discharge; no inference launched.",
            }
            for signature in ("CT", "SPO", "AMJJ")
        ]
    )


def merge_partition_outputs(
    output: Path, partition_root: Path, partition_count: int = 5, epoch: int = 100
) -> dict[str, pd.DataFrame]:
    import pyarrow.parquet as pq

    parts = [partition_root / f"part_{index}" for index in range(partition_count)]
    if not all(path.is_dir() for path in parts):
        raise FileNotFoundError(f"missing partition directories under {partition_root}")
    merged_metrics = []
    merged_years = []
    basin_sets = []
    for part in parts:
        metrics_path = part / "r1_online_performance.csv"
        years_path = part / "r1_online_signature_basin_year.csv"
        if not metrics_path.exists() or not years_path.exists():
            raise FileNotFoundError(f"incomplete partition: {part}")
        metrics = pd.read_csv(metrics_path)
        years = pd.read_csv(years_path)
        merged_metrics.append(metrics)
        merged_years.append(years)
        basin_sets.append(set(metrics["basin_id"].astype(str).str.zfill(8)))
    if (
        set.union(*basin_sets)
        != set(
            pd.concat(merged_metrics, ignore_index=True)["basin_id"]
            .astype(str)
            .str.zfill(8)
            .unique()
        )
        or sum(map(len, basin_sets)) != 531
    ):
        raise ValueError("partition basin coverage is not exactly 531 unique basins")
    if set.intersection(*basin_sets):
        raise ValueError("partition basin sets overlap")
    online_metrics = (
        pd.concat(merged_metrics, ignore_index=True)
        .sort_values(["basin_id", "seed_or_restart", "period"])
        .reset_index(drop=True)
    )
    online_years = (
        pd.concat(merged_years, ignore_index=True)
        .sort_values(["basin_id", "seed_or_restart", "period", "water_year"])
        .reset_index(drop=True)
    )
    online_metrics["basin_id"] = (
        online_metrics["basin_id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )
    online_years["basin_id"] = (
        online_years["basin_id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )
    if len(online_metrics) != 531 * 3 * 2:
        raise ValueError(f"unexpected merged metric rows: {len(online_metrics)}")
    output.mkdir(parents=True, exist_ok=True)
    online_metrics.to_csv(output / "r1_online_performance.csv", index=False)
    online_years.to_csv(output / "r1_online_signature_basin_year.csv", index=False)
    for seed in ("42", "123", "2026"):
        paths = [
            part / f"r1_daily_simulations_dpl_xaj_tgd2_seed_{seed}_part_{index}.parquet"
            for index, part in enumerate(parts)
        ]
        if not all(path.exists() for path in paths):
            raise FileNotFoundError(f"missing daily partition for seed {seed}")
        target = output / f"r1_daily_simulations_dpl_xaj_tgd2_seed_{seed}.parquet"
        if not target.exists():
            raise FileNotFoundError(
                f"existing merged daily file is required and cannot be created in extension mode: {target}"
            )
        expected_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
        actual_rows = pq.ParquetFile(target).metadata.num_rows
        if actual_rows != expected_rows:
            raise ValueError(
                f"existing merged daily file row mismatch for {seed}: {actual_rows} != {expected_rows}"
            )
    previous_metrics = pd.read_csv(output / "r1_dpl_seed_level_performance.csv")
    previous_primary = pd.read_csv(output / "r1_basin_level_performance.csv")
    for table in (previous_metrics, previous_primary):
        table["basin_id"] = (
            table["basin_id"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(8)
        )
    full_metrics = pd.concat(
        [
            previous_primary[previous_primary["paradigm"].eq("IC-CMA-ES")],
            previous_metrics[
                ~(
                    (previous_metrics["paradigm"].eq("dPL-MLP"))
                    & (previous_metrics["model"].eq("XAJ-TGD"))
                )
            ],
            online_metrics,
        ],
        ignore_index=True,
        sort=False,
    )
    previous_years = pd.read_csv(output / "r1_snow_signatures_basin_year.csv")
    previous_years["basin_id"] = (
        previous_years["basin_id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )
    previous_years = previous_years[
        ~(
            (previous_years["paradigm"].eq("dPL-MLP"))
            & (previous_years["model"].eq("XAJ-TGD"))
        )
    ]
    all_years = pd.concat([previous_years, online_years], ignore_index=True, sort=False)
    signatures, signature_effects = signature_tables_from_years(all_years)
    (output / "r1_epoch100_partition_manifest.json").write_text(
        json.dumps(
            {
                "epoch": epoch,
                "partition_count": partition_count,
                "partition_root": str(partition_root),
                "basin_count": 531,
                "metric_rows": len(online_metrics),
                "signature_year_rows": len(online_years),
                "merged_daily_files": [
                    str(
                        output
                        / f"r1_daily_simulations_dpl_xaj_tgd2_seed_{seed}.parquet"
                    )
                    for seed in ("42", "123", "2026")
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "metrics": full_metrics,
        "signature_years": all_years,
        "signature_basin_level": signatures,
        "signature_effects": signature_effects,
    }


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve()
    data = args.data_root.resolve()
    results = (args.results_root or project / "results").resolve()
    output = (args.output_root or project / "manuscript" / "results" / "R1").resolve()
    logger = setup_logging(output, append=args.mode == "summary")
    before = git_status(project)
    logger.info("command=%s", " ".join(sys.argv))
    logger.info("git_status_before=%s", before.replace("\n", " | "))
    logger.info(
        "mode=%s project_root=%s results_root=%s output_root=%s",
        args.mode,
        project,
        results,
        output,
    )
    if args.mode == "summary":
        logger.info(
            "summary mode reads existing CSV tables only; no training, calibration, inference, or daily export launched"
        )
        aggregate = aggregate_full_sample(output, data, SEED)
        aggregate = append_snow_stratified(output, data, aggregate, SEED)
        aggregate = append_remaining_checks(output, data, aggregate, SEED)
        for name, table in {
            "r1_absolute_metrics_summary.csv": aggregate["absolute"],
            "r1_paired_effects_summary.csv": aggregate["paired"],
            "r1_snow_relationships_summary.csv": aggregate["relationships"],
            "r1_bootstrap_intervals.csv": aggregate["bootstrap"],
            "r1_statistical_tests.csv": aggregate["tests"],
            "r1_generalization_effects_basin_level.csv": aggregate[
                "generalization_basin"
            ],
            "r1_signature_effects_basin_level.csv": aggregate["signature_basin"],
        }.items():
            write_stable_output_table(table, output / name)

        exclusions = pd.read_csv(output / "r1_exclusion_log.csv")
        exclusions = exclusions[~exclusions["item"].astype(str).eq("SPO")]
        exclusions = pd.concat(
            [
                exclusions,
                pd.DataFrame(
                    [
                        {
                            "item": "SPO",
                            "status": "excluded_from_R1_incomplete_prespecified_definition",
                            "reason": "SPO is excluded from active R1 because the project does not prespecify cumulative start date, search window, reference discharge, no-pulse handling, incomplete-year handling, tied-minimum handling, or date encoding. Daily simulation data are available; the limitation is definition and reproducibility.",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        write_stable_output_table(exclusions, output / "r1_exclusion_log.csv")

        notes_path = output / "r1_statistics_notes.md"
        notes = notes_path.read_text(encoding="utf-8")
        notes = notes.replace(
            "- SPO remains unresolved. CT and AMJJ are exported when complete water years exist.",
            "- SPO is excluded from active R1 under `excluded_from_R1_incomplete_prespecified_definition`; daily simulation data exist, but the prespecified definition is incomplete. CT is the primary timing signature and AMJJ is the secondary seasonal-volume signature.",
        )
        notes = render_full_sample_notes(aggregate, notes)
        notes = render_remaining_notes(aggregate, notes)
        notes += "\n## Reproducibility\n\nSummary mode is deterministic across independent Python processes and `PYTHONHASHSEED` values. Deterministic CSV artifacts use stable row ordering, explicit UTF-8/LF serialization, and round-trip-safe `%.17g` floats. The manifest stores the canonical reproduction command and excludes itself from artifact hashes; `r1_execution.log` remains volatile because it contains timestamps, runtime paths, and git-status snapshots.\n"
        notes_path.write_text(notes, encoding="utf-8")

        audit_path = output / "r1_data_audit.md"
        audit = audit_path.read_text(encoding="utf-8")
        audit = audit.replace(
            "SPO is unresolved because the project plan does not fix its calculation start, search window, or no-pulse rule.",
            "SPO is excluded from active R1 under `excluded_from_R1_incomplete_prespecified_definition`. Daily simulation data are available; the limitation is definition and reproducibility because cumulative start date, search window, reference discharge, no-pulse handling, incomplete-year handling, tied-minimum handling, and date encoding are not prespecified.",
        )
        audit_marker = "## Full-sample aggregation pass"
        audit = (
            audit.split(audit_marker, 1)[0].rstrip()
            + f"""

{audit_marker}

The aggregation pass reads the existing basin-level performance, structural-effect, generalization-effect, signature, attribute, and snow-relationship CSV files. It does not read or rewrite daily Parquet files and launches no inference. Primary absolute metrics use `{FULL_SAMPLE}`. dPL primary metrics use within-seed basin metrics followed by the median across seeds; IC primary metrics use the selected restart chosen by train-period KGE. Primary signature summaries require five complete water years; the three-year result remains sensitivity only. Bootstrap intervals target the basin-level median with 10,000 resamples and seed `{SEED}`. Regional sensitivity uses the existing seven LORO region files. Snow-stratified rows use the existing fixed S1-S5 frac_snow boundaries and are descriptive stratified summaries within the same R1 tables. Remaining checks add basin-wise IC-dPL transfer A/B/D, transfer-gradient interaction with TGD as the reference structure and basin-clustered standard errors, continuous snow-gradient exposure, and primary five-water-year CT/AMJJ absolute and paired reduction summaries. Regional slope intervals are seven-region block-bootstrap OLS slope sensitivities; robust slopes retain ordinary Theil-Sen confidence intervals.
"""
        )
        audit += """
## Reproducibility audit

The pre-fix comparison classified all CSV and Markdown outputs as byte-identical and all parsed scientific JSON values as identical. The only pre-fix JSON difference was the invocation-specific temporary `--output-root` embedded in `r1_result_manifest.json`; `r1_execution.log` differed only in timestamps, runtime paths, and git-status snapshots. The fix stores the canonical reproduction command in the manifest, uses explicit UTF-8/LF/`%.17g` summary serialization, and records deterministic artifact hashes without hashing the manifest itself. Bootstrap master seed is `20260730`; keyed estimands use the existing SHA-256-derived seed helper. The three post-fix processes with `PYTHONHASHSEED=1`, `777`, and `20260731` produced identical inventories, schemas, exact parsed scientific values, and deterministic artifact bytes. The remaining volatile artifact is `r1_execution.log`.
"""
        audit_path.write_text(audit, encoding="utf-8")

        manifest_path = output / "r1_result_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        historical_created_at = manifest.pop("created_at_utc", None)
        manifest.update(
            {
                "status": "complete_spo_excluded_incomplete_definition",
                "command": canonical_summary_command(project, data, results),
                "aggregation_pass": {
                    "analysis_set": FULL_SAMPLE,
                    "summary_level": "full_sample",
                    "result_role": "primary",
                    "input_tables": [
                        "r1_basin_level_performance.csv",
                        "r1_structural_effects_basin_level.csv",
                        "r1_generalization_effects_basin_level.csv",
                        "r1_snow_signatures_basin_level.csv",
                        "r1_snow_attributes.csv",
                        "r1_snow_relationships_summary.csv",
                    ],
                    "bootstrap_statistic": "median",
                    "aggregation_rule_dpl": "within-seed basin metrics followed by median across seeds",
                    "aggregation_rule_ic": "selected best-train-KGE restart per basin",
                    "primary_signature_requirement": "minimum five complete water years",
                    "sensitivity_signature_requirement": "minimum three complete water years",
                    "remaining_checks": {
                        "tgd_transfer": "A=KGE_IC,train-KGE_dPL,train; B=KGE_IC,test-KGE_dPL,test; D=B-A; G_IC-G_dPL=-D",
                        "transfer_gradient": "D ~ frac_snow + structure + frac_snow:structure; TGD reference; cluster=basin_id",
                        "regional_slope_bootstrap": "seven-region block bootstrap of OLS slope sensitivity; robust slope retains Theil-Sen 95% CI",
                        "signature_units": {
                            "CT": "days",
                            "AMJJ": "fractional-flow error",
                        },
                        "signature_requirement": "minimum five complete water years; three-year sensitivity retained",
                        "analysis_sets": sorted(
                            set(
                                aggregate["paired"]
                                .get("analysis_set", pd.Series(dtype=str))
                                .dropna()
                                .astype(str)
                                .loc[
                                    lambda values: values.str.startswith(
                                        "r1_remaining_"
                                    )
                                ]
                                .tolist()
                            )
                        ),
                    },
                },
                "snow_stratification": {
                    "field": "frac_snow",
                    "source": "data/camels_dataset attributes[:,3]",
                    "scheme": "project-fixed S1-S5 boundaries from manuscript/supplement/S1_Data_and_study_catchments_verified.md",
                    "strata": aggregate["snow_strata"].to_dict(orient="records"),
                    "analysis_set": "snow_fixed_strata",
                    "summary_level": "snow_stratum",
                    "result_role": "stratified_primary",
                    "interpretation": "descriptive stratified accuracy differences; no new threshold introduced",
                },
                "spo": {
                    "status": "excluded_from_R1_incomplete_prespecified_definition",
                    "reason": "Daily simulation data are available; the complete prespecified SPO definition is not.",
                    "missing_definition_fields": [
                        "cumulative start date",
                        "search window",
                        "reference discharge",
                        "no-pulse handling",
                        "incomplete-year handling",
                        "tied-minimum handling",
                        "date encoding",
                    ],
                },
                "region_block_bootstrap": aggregate["region_meta"],
                "script_sha256": script_hashes(),
                "output_schemas": output_schemas(output),
                "volatile_metadata": {"created_at_utc": historical_created_at},
                "reproducibility": {
                    "deterministic_artifacts": deterministic_artifact_records(output),
                    "volatile_artifacts": [
                        {
                            "relative_path": "r1_execution.log",
                            "volatile_fields": [
                                "timestamps",
                                "runtime paths",
                                "git-status snapshots",
                            ],
                        }
                    ],
                    "serialization": {
                        "encoding": "UTF-8",
                        "newline": "\\n",
                        "float_format": "%.17g",
                        "csv_index": False,
                        "json": "sort_keys=True, indent=2, one trailing newline",
                    },
                    "master_seed": SEED,
                    "stable_seed_rule": "SHA-256-derived keyed seeds for semantic estimands; fixed ordered loops for composite aggregation",
                    "manifest_hash_excluded": True,
                },
            }
        )
        issues = manifest.get("unresolved_issues", [])
        issues = [issue for issue in issues if issue.get("item") != "SPO"]
        issues.append(
            {
                "item": "SPO",
                "status": "excluded_from_R1_incomplete_prespecified_definition",
                "reason": "Daily simulation data are available; the complete prespecified SPO definition is not.",
            }
        )
        manifest["unresolved_issues"] = issues
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        after = git_status(project)
        logger.info("git_status_after=%s", after.replace("\n", " | "))
        logger.info(
            "summary_outputs=%s",
            ",".join(
                [
                    "r1_absolute_metrics_summary.csv",
                    "r1_paired_effects_summary.csv",
                    "r1_snow_relationships_summary.csv",
                    "r1_bootstrap_intervals.csv",
                    "r1_statistical_tests.csv",
                    "r1_result_manifest.json",
                    "r1_data_audit.md",
                    "r1_statistics_notes.md",
                    "r1_exclusion_log.csv",
                    "r1_execution.log",
                ]
            ),
        )
        return 0
    inventory, inventory_notes = build_inventory(project, data)
    write_csv(inventory, output / "r1_input_inventory.csv")
    precomputed = None
    merged_epoch_mode = args.mode == "merge-partitions"
    if args.mode == "merge-partitions":
        partition_root = (
            args.partition_root or output / "epoch100_partitions"
        ).resolve()
        precomputed = merge_partition_outputs(
            output, partition_root, args.partition_count, args.tgd2_epoch or 100
        )
        args.mode = "statistics"
    if args.mode == "audit":
        logger.info("audit-only completed; no inference or statistics launched")
        (output / "r1_data_audit.md").write_text(
            "# R1 Data Audit\n\nAudit-only mode completed. See `r1_input_inventory.csv`.\n",
            encoding="utf-8",
        )
    else:
        inference = None
        if args.mode in ("daily-inference", "full"):
            prior_daily_inventory = (
                pd.read_csv(output / "r1_daily_simulation_inventory.csv")
                if (output / "r1_daily_simulation_inventory.csv").exists()
                else pd.DataFrame()
            )
            inference = run_daily_export(
                project,
                results,
                data,
                output,
                device=args.device,
                batch_size=args.batch_size,
                model_keys=tuple(args.models) if args.models else None,
                tgd2_epoch=args.tgd2_epoch,
                paradigm=args.paradigm,
                partition_count=args.partition_count,
                partition_index=args.partition_index,
                partition_suffix=args.partition_suffix,
            )
            daily_inventory = pd.DataFrame(inference["files"])
            if not prior_daily_inventory.empty:
                selected_pairs = set(
                    zip(
                        daily_inventory.get("paradigm", []),
                        daily_inventory.get("model", []),
                    )
                )
                prior_daily_inventory = prior_daily_inventory[
                    ~prior_daily_inventory.apply(
                        lambda row: (
                            (row.get("paradigm"), row.get("model")) in selected_pairs
                        ),
                        axis=1,
                    )
                ]
                daily_inventory = pd.concat(
                    [prior_daily_inventory, daily_inventory],
                    ignore_index=True,
                    sort=False,
                )
            daily_inventory.to_csv(
                output / "r1_daily_simulation_inventory.csv", index=False
            )
            for _, row in daily_inventory.iterrows():
                mask = (inventory["model"] == row["model"]) & (
                    inventory["calibration_paradigm"] == row["paradigm"]
                )
                inventory.loc[mask, "prediction_file_path"] = ";".join(
                    inventory.loc[mask, "prediction_file_path"]
                    .replace("", np.nan)
                    .dropna()
                    .tolist()
                    + [row["file"]]
                )
                inventory.loc[mask, "incomplete_runs"] = "complete daily export"
            write_csv(inventory, output / "r1_input_inventory.csv")
        if args.mode in ("statistics", "full"):
            if not list(output.glob("r1_daily_simulations_*.parquet")):
                raise FileNotFoundError(
                    "No daily Parquet exports found; run --mode daily-inference or --mode full first"
                )
            result = build_statistics_from_daily(
                output,
                data,
                results,
                SEED,
                precomputed=inference.get("precomputed")
                if inference is not None
                else precomputed,
            )
            if merged_epoch_mode:
                absolute = pd.read_csv(output / "r1_absolute_metrics_summary.csv")
                paired = pd.read_csv(output / "r1_paired_effects_summary.csv")
                absolute.insert(0, "record_type", "absolute")
                paired.insert(0, "record_type", "paired_effect")
                comparison = pd.concat(
                    [absolute, paired], ignore_index=True, sort=False
                )
                comparison.insert(1, "checkpoint_epoch", args.tgd2_epoch or 100)
                comparison.to_csv(
                    output / "r1_epoch100_comparison.csv", index=False, na_rep=""
                )
            exclusions = pd.concat(
                [
                    result["exclusions"],
                    pd.DataFrame(
                        [
                            {
                                "item": "inventory_note",
                                "status": "audit_note",
                                "reason": note,
                            }
                            for note in inventory_notes
                        ]
                    ),
                ],
                ignore_index=True,
            )
            exclusions.to_csv(output / "r1_exclusion_log.csv", index=False)
            (output / "r1_data_audit.md").write_text(
                f"""# R1 Data Audit

## Scope

Current R1 uses IC-CMA-ES XAJ-Base, XAJ-TGD, XAJ-CN and dPL-MLP XAJ-Base, XAJ-TGD, XAJ-CN, and HBV. The period contract is {PERIOD_TEXT}. The CAMELS-531 list is `{data / "531sub_id.txt"}`.

## Inference provenance

Existing daily simulation exports and partition summaries are reused in this extension. No training, resume, recalibration, preprocessing, or inference job was launched. The current provisional dPL XAJ-TGD artifact is selected from the common valid periodic checkpoint epoch recorded in `r1_inference_audit.md`; the current exported result is epoch 100. Existing merged Parquet files are verified for row-count integrity and are not rewritten.

Forcing and observations come from `{data / "camels_dataset"}` with forcing order `P,T,PET`. Observed discharge is raw ft3/s and is converted to mm/day using the repository's `area_gages2` index-11 conversion. Nonfinite and negative flows are masked; zero is valid. dPL uses the repository robust median/IQR normalization and sigmoid physical mapping, including inverse-log mapping for TGD2 residence times. IC uses the repository Lite `ModelAdapter` with selected restarts based only on stored train-period KGE.

## Metric definition

In `full` mode, statistics are computed immediately from the same aligned target and predictions used to write the daily Parquet exports; `statistics` mode recomputes them from those daily exports. The authoritative repository evaluator is standard KGE(Q): `1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)`, with `alpha=std_sim/std_obs`, `beta=mean_sim/mean_obs`, finite nonnegative paired mask, minimum 30 paired days, and invalid zero-variance observations excluded. It is not CV-ratio KGE-prime.

NSE, PBIAS, and RMSE use the same paired valid-day mask. PBIAS is `100*sum(sim-obs)/sum(obs)`; positive values indicate simulated excess.

## Signatures

Water years begin October 1. CT is the first day cumulative flow reaches 50% of annual flow. AMJJ is April-July flow divided by water-year flow. Complete water years are calculated first and basin summaries retain `valid_years`; primary CT and AMJJ effects require at least five complete water years and a three-year minimum sensitivity is retained. SPO is unresolved because the project plan does not fix its calculation start, search window, or no-pulse rule.

## Extension estimands

Generalization exposure is `E_enhanced-base = (KGE_enhanced,test - KGE_base,test) - (KGE_enhanced,train - KGE_base,train)` for CN-Base, TGD-Base, and CN-TGD, calculated basin by basin within each paradigm. IC-dPL transfer is `KGE_IC - median_seed(KGE_dPL)` for each XAJ structure and period. All paired summaries use matched basin IDs.

The snow-specific CN-TGD relationship uses `KGE_CN,test - KGE_TGD,test`. The combined interaction model is `effect_value ~ frac_snow + paradigm_dPL + frac_snow:paradigm_dPL`, with IC-CMA-ES as the reference category and cluster-robust standard errors clustered by `basin_id`.

CT and AMJJ error reductions are `|E_Base| - |E_CN|`, `|E_Base| - |E_TGD|`, and `|E_TGD| - |E_CN|`; positive values indicate lower error for the second model in the stored first-minus-second convention. Effects are calculated from basin-level medians over complete water years. The primary set requires five valid years and the sensitivity set requires three.

Ordinary paired basin bootstrap uses 10,000 resamples and seed `20260730`. The spatial sensitivity uses the authoritative `data/basin_groups/group_11.npy` through `group_17.npy` seven-region LORO grouping documented in `project/flexmopex/run_model.py`; region blocks are sampled with replacement. Restart signature robustness is unavailable because non-selected restart daily CT/AMJJ series do not exist. dPL seed robustness calculates basin effects within each seed (`42`, `123`, `2026`) before reporting seed estimates and the across-seed median.

See `r1_inference_audit.md`, `r1_daily_simulation_inventory.csv`, `r1_exclusion_log.csv`, and `r1_result_manifest.json`.
""",
                encoding="utf-8",
            )
            (output / "r1_statistics_notes.md").write_text(
                """# R1 Statistics Notes

- This extension reuses existing daily exports and existing partition summaries; it launches no inference, training, calibration, preprocessing, or plotting.
- The main metric is the repository's standard KGE(Q), using the standard-deviation ratio `alpha=std_sim/std_obs`; no KGE-prime relabeling is applied.
- Full mode computes metrics from aligned daily observed/simulated arrays during inference; statistics-only mode recomputes them from daily exports with matched valid-day masks.
- dPL seed-by-basin records remain separate; primary dPL effects summarize seed-specific basin effects by median.
- Generalization exposure is `(enhanced_test - base_test) - (enhanced_train - base_train)` for CN-Base, TGD-Base, and CN-TGD. IC-dPL transfer is `IC - median_seed(dPL)` for Base, TGD, and CN.
- Snow interaction uses `effect ~ frac_snow + paradigm_dPL + frac_snow:paradigm_dPL`, with IC-CMA-ES as reference and basin-clustered standard errors.
- CT and AMJJ effects use `|E_first| - |E_second|`, with primary five-year complete-water-year matching and three-year sensitivity.
- Bootstrap uses 10,000 basin resamples and fixed seed `20260730`; region block bootstrap uses seven authoritative LORO regions from `data/basin_groups/group_11.npy` through `group_17.npy`.
- dPL robustness is evaluated independently within seeds `42`, `123`, and `2026`; IC restart robustness uses stored restart KGE and does not select by test performance. IC signature restart sensitivity is unavailable without non-selected-restart daily series.
- SPO remains unresolved. CT and AMJJ are exported when complete water years exist.
""",
                encoding="utf-8",
            )
    statistics_meta = result if args.mode in ("statistics", "full") else {}
    region_meta = (
        statistics_meta.get("region_meta", {})
        if isinstance(statistics_meta, dict)
        else {}
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_with_documented_spo_limitation"
        if args.mode != "audit"
        else "audit_only",
        "inference": {
            "device": "cuda",
            "batch_size": 64,
            "partition_count": 5,
            "partition_execution": "sequential",
        },
        "reproduction_command": (
            "for i in 0 1 2 3 4; do python "
            "manuscript/scripts/r1/build_r1_statistics.py "
            "--mode daily-inference --models XAJ_TGD2 --paradigm dpl "
            "--tgd2-epoch 100 --device cuda --batch-size 64 "
            "--partition-count 5 --partition-index $i "
            "--partition-suffix _part_$i "
            "--project-root project/hydrodiag --results-root project/hydrodiag/results "
            "--data-root data "
            "--output-root project/hydrodiag/manuscript/cache/R1/part_$i "
            "|| exit $?; done",
        ),
        "scope": [
            "XAJ-Base",
            "XAJ-TGD",
            "XAJ-CN",
            "HBV benchmark",
            "IC-CMA-ES",
            "dPL-MLP",
        ],
        "selected_artifacts": {
            "dPL-MLP/XAJ-TGD": {
                "root": str(
                    results / "dpl_camels_531_lite_v3_tgd2_dpl_audited" / "XAJ_TGD2"
                ),
                "checkpoint_epoch": args.tgd2_epoch or 100,
                "seeds": ["42", "123", "2026"],
                "selection_rule": "maximum checkpoint epoch common to all three seeds and present in epoch_history",
            }
        },
        "metric": {
            "name": "KGE(Q)",
            "formula": "1-sqrt((r-1)^2+(alpha-1)^2+(beta-1)^2)",
            "alpha": "std_sim/std_obs",
            "beta": "mean_sim/mean_obs",
            "source": "repository production evaluators",
            "invalid_rules": "shared finite nonnegative paired mask; minimum 30 paired days; invalid zero-variance observations excluded",
        },
        "extension_estimands": {
            "generalization_exposure": "(KGE_enhanced,test-KGE_base,test)-(KGE_enhanced,train-KGE_base,train) for CN-Base, TGD-Base, and CN-TGD",
            "ic_dpl_transfer": "KGE_IC - median_seed(KGE_dPL), by structure and period",
            "signature_error_reduction": "abs(error_first)-abs(error_second) for CT and AMJJ; primary minimum five complete water years; sensitivity minimum three",
            "snow_specificity": "KGE_CN,test-KGE_TGD,test",
            "interaction_model": "effect_value ~ frac_snow + paradigm_dPL + frac_snow:paradigm_dPL; reference=IC-CMA-ES; cluster=basin_id",
            "support_status": "supported_positive if CI is entirely above zero; supported_negative if entirely below zero; inconclusive if CI crosses zero; otherwise descriptive_only",
        },
        "bootstrap": {
            "resamples": 10000,
            "seed": SEED,
            "ordinary_method": "paired basin resampling",
            "region_method": "region block resampling when authoritative grouping is available",
        },
        "region_block_bootstrap": region_meta,
        "script_sha256": script_hashes(),
        "output_schemas": output_schemas(output),
        "statistics_input_mode": "inference_arrays_in_full_mode; daily_parquet_recomputation_in_statistics_mode",
        "random_seed": SEED,
        "bootstrap_resamples": 10000,
        "command": " ".join(sys.argv),
        "scripts": [
            str(HERE / name)
            for name in (
                "build_r1_statistics.py",
                "r1_daily_inference.py",
                "r1_metrics.py",
                "r1_statistics.py",
            )
        ],
        "outputs": sorted(
            set(
                [path.name for path in output.iterdir() if path.is_file()]
                + ["r1_result_manifest.json"]
            )
        ),
        "source_roots": {
            "project": str(project),
            "results": str(results),
            "data": str(data),
        },
        "unresolved_issues": pd.read_csv(output / "r1_exclusion_log.csv").to_dict(
            orient="records"
        )
        if (output / "r1_exclusion_log.csv").exists()
        else [],
    }
    (output / "r1_result_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    after = git_status(project)
    logger.info("git_status_after=%s", after.replace("\n", " | "))
    logger.info("outputs=%s", ",".join(manifest["outputs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
