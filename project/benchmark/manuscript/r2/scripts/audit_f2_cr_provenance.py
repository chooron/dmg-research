"""Audit provenance and coverage for the R2 contraction-ratio panel."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

R2 = Path(__file__).resolve().parents[1]
BENCHMARK = R2.parents[1]
TABLES = R2 / "tables"
CACHE = R2 / "cache"
RESULTS = BENCHMARK / "results/joh_direct_parameter_change_diagnostic_20260905"
AGENT = RESULTS / "r2_final_robustness_20260906/agent_B_contraction_robustness"
CORRECTED = AGENT / "corrected_primary_23models"
SHARED = RESULTS / "r2_final_robustness_20260906/shared"
CR_CACHE = CACHE / "fig3d_contraction_reference.csv"
FROZEN = {"canonical": 0.6140173622, "consensus": 0.9794559561, "IC-self": 1.0043244360}
CR_COLUMNS = {"canonical": "CR_canonical", "consensus": "CR_consensus", "IC-self": "CR_ICself"}
FULL_COLUMNS = {"canonical": "canonical_CR_median", "consensus": "consensus_CR_median", "IC-self": "self_reference_CR_median"}


def rel(path: Path) -> str:
    return str(path.relative_to(BENCHMARK))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    cr = pd.read_csv(CR_CACHE)
    qc = pd.read_csv(AGENT / "qc.csv")
    model_summary = pd.read_csv(AGENT / "model_summary.csv")
    strict_audit = pd.read_csv(TABLES / "R2_STRICT_SUBSET_AUDIT.csv")
    full_ids = sorted(pd.read_parquet(CACHE / "fig2a_paired_displacement_plane.parquet").model_id.astype(str).unique())
    current_ids = sorted(cr.model_id.astype(str).tolist())
    eligible = set(qc.loc[qc.primary_coverage_status == "PASS", "model"].astype(str))
    insufficient = set(qc.loc[qc.primary_coverage_status == "INSUFFICIENT_REFERENCE", "model"].astype(str))
    rank_ids = set(strict_audit.loc[strict_audit.rank_strict_eligible, "model_id"].astype(str))
    localization_ids = set(strict_audit.loc[strict_audit.localization_strict_eligible, "model_id"].astype(str))

    if len(full_ids) != 36 or len(current_ids) != 23:
        raise RuntimeError(f"unexpected coverage: full={len(full_ids)} current={len(current_ids)}")
    if eligible != set(current_ids) or eligible != rank_ids or eligible != localization_ids:
        raise RuntimeError("CR 23-model set does not match primary/rank/localization eligibility")
    if len(insufficient) != 13 or set(full_ids) != eligible | insufficient:
        raise RuntimeError("36-model eligibility partition is not exhaustive")

    source_files = ";".join(sorted(cr.source_file.dropna().astype(str).unique()))
    current_rows = []
    for model in current_ids:
        q = qc.loc[qc.model.astype(str) == model].iloc[0]
        current_rows.append({
            "model_id": model,
            "present_canonical": True,
            "present_consensus": True,
            "present_icself": True,
            "source_file": source_files,
            "eligibility_reason": f"primary coverage {float(q.primary_coverage):.6f} >= 0.90 under frozen within-0.01 non-canonical restart rule",
        })
    pd.DataFrame(current_rows).to_csv(TABLES / "F2_CR_CURRENT23_MODEL_LIST.csv", index=False)

    availability_rows = []
    for model in full_ids:
        q = qc.loc[qc.model.astype(str) == model].iloc[0]
        is_strict = model in eligible
        reason = "" if is_strict else (
            f"primary coverage {float(q.primary_coverage):.6f} < 0.90; incomplete eligible non-canonical IC reference; "
            "canonical fallback is prohibited by the frozen contract"
        )
        status = "AVAILABLE" if is_strict else "STRICT_FILTER_ONLY"
        availability_rows.append({
            "model_id": model,
            "canonical_available": "AVAILABLE",
            "consensus_available": status,
            "icself_available": status,
            "canonical_source": rel(AGENT / "canonical_coordinate_summary.csv") + ";" + rel(AGENT / "model_summary.csv"),
            "consensus_source": rel(CORRECTED / "matched_model_comparison_23.csv") if is_strict else rel(AGENT / "model_summary.csv"),
            "icself_source": rel(CORRECTED / "matched_model_comparison_23.csv") if is_strict else rel(AGENT / "raw_replicate_arrays.npz") + ";" + rel(AGENT / "model_summary.csv"),
            "missing_reason": reason,
            "restart_coverage": float(q.primary_coverage),
            "strict_eligibility": bool(is_strict),
        })
    pd.DataFrame(availability_rows).to_csv(TABLES / "F2_CR_AVAILABILITY_MATRIX.csv", index=False)

    headline_rows = []
    for reference, col in CR_COLUMNS.items():
        value = float(cr[col].median())
        headline_rows.append({
            "reference": reference,
            "dataset_variant": "strict23_primary",
            "n_models": 23,
            "model_equal_median": value,
            "source": rel(CR_CACHE),
            "reproduces_frozen_value": bool(np.isclose(value, FROZEN[reference], rtol=0, atol=1e-8)),
            "subset_definition": "PRIMARY_MATCHED_23; primary coverage >= 0.90",
        })
    full_primary = model_summary[(model_summary.prefix == 5000) & (model_summary.pool == "primary")]
    if len(full_primary) != 36:
        raise RuntimeError("expected 36 primary model summary rows for diagnostic comparison")
    source_join = cr.set_index("model_id").join(full_primary.set_index("model"), how="left")
    source_backcheck = {}
    for reference, cache_col in CR_COLUMNS.items():
        diff = (source_join[cache_col] - source_join[FULL_COLUMNS[reference]]).abs()
        source_backcheck[reference] = float(diff.max())
    max_source_backcheck = max(source_backcheck.values())
    if max_source_backcheck > 1e-9:
        raise RuntimeError(f"current 23-model CR cache does not reproduce upstream model_summary: {source_backcheck}")
    for reference, col in FULL_COLUMNS.items():
        value = float(full_primary[col].median())
        headline_rows.append({
            "reference": reference,
            "dataset_variant": "coverage_incomplete_full36_diagnostic",
            "n_models": 36,
            "model_equal_median": value,
            "source": rel(AGENT / "model_summary.csv"),
            "reproduces_frozen_value": bool(np.isclose(value, FROZEN[reference], rtol=0, atol=1e-8)),
            "subset_definition": "all36 rows retained by upstream diagnostics, including 13 INSUFFICIENT_REFERENCE models; not primary-valid",
        })
    pd.DataFrame(headline_rows).to_csv(TABLES / "F2_CR_HEADLINE_REPRODUCTION.csv", index=False)

    stats_rows = []
    for variant, frame, columns in [
        ("strict23_primary", cr, CR_COLUMNS),
        ("coverage_incomplete_full36_diagnostic", full_primary, FULL_COLUMNS),
    ]:
        for reference, col in columns.items():
            values = frame[col].to_numpy(float)
            stats_rows.append({
                "dataset_variant": variant,
                "reference": reference,
                "n_models": int(len(values)),
                "median": float(np.median(values)),
                "q05": float(np.quantile(values, .05)),
                "q25": float(np.quantile(values, .25)),
                "q75": float(np.quantile(values, .75)),
                "q95": float(np.quantile(values, .95)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median_minus_strict23": 0.0 if variant == "strict23_primary" else float(np.median(values) - cr[CR_COLUMNS[reference]].median()),
            })
    stats = pd.DataFrame(stats_rows)
    strict_stats = stats[stats.dataset_variant == "strict23_primary"].set_index("reference")
    full_stats = stats[stats.dataset_variant == "coverage_incomplete_full36_diagnostic"].set_index("reference")

    source_manifest = [
        AGENT / "qc.csv", AGENT / "model_summary.csv", AGENT / "raw_replicate_arrays.npz",
        AGENT / "canonical_coordinate_summary.csv", AGENT / "consensus_coordinate_summary.csv",
        SHARED / "IC_RESTART_REFERENCE_CONTRACT.md", SHARED / "ic_restart_eligibility.csv", SHARED / "ic_self_draw_plan.npz",
        CORRECTED / "corrected_primary_23models.py", CORRECTED / "matched_model_comparison_23.csv",
    ]
    source_lines = "\n".join(f"- `{rel(p)}` — SHA256 `{sha256(p)}`" for p in source_manifest if p.exists())
    missing_text = ", ".join(sorted(insufficient))
    summary_lines = []
    for reference in ["canonical", "consensus", "IC-self"]:
        s = strict_stats.loc[reference]
        f = full_stats.loc[reference]
        summary_lines.append(
            f"- **{reference}:** strict23 median `{s['median']:.10f}`; coverage-incomplete full36 diagnostic median `{f['median']:.10f}`; difference `{f['median_minus_strict23']:+.10f}`."
        )

    definitions = f"""# F2 CR reference definitions

## Scope

The canonical contraction-robustness pipeline evaluates the same coordinate-wise diagnostic for three alternative IC reference constructions. It uses 271 parameter coordinates per model and 531 basins per model. The frozen primary restart rule is archived non-canonical IC training KGE within `0.01` of the basin best/canonical IC restart; the canonical restart itself is excluded and no canonical fallback is allowed.

## Exact CR calculation

For a model `m` and parameter coordinate `j`, define the IC interquartile range:

```text
IQR_IC(m,j) = Q75_b(IC[m,b,j]) - Q25_b(IC[m,b,j])
```

For any reference field `F[m,b,j]`, the coordinate-level contraction ratio is:

```text
CR(m,j) = IQR_b(F[m,b,j]) / IQR_IC(m,j)
```

There is no epsilon added to the denominator. If the IC IQR is zero or non-finite, that coordinate's CR is unavailable. The model-level CR is the median across the model's parameter coordinates of the coordinate-level CR values (for IC-self, the coordinate-level CR is first summarized by the median across the prescribed synthetic draws, then the model-coordinate median is taken).

### canonical

`F` is the canonical dPL parameter field, using the dPL seed-42 normalized parameter matrix. Thus `CR_canonical` is `IQR_b(dPL[m,b,j]) / IQR_b(IC[m,b,j])`, followed by the median across coordinates. It is computable from the canonical dPL and IC matrices for all 36 models and does not require an IC restart pool.

### consensus

`F` is the basin-wise median across eligible non-canonical primary IC restarts for each model, basin, and coordinate. Basins with no eligible non-canonical restart remain missing. The resulting field's basin IQR is divided by the canonical IC basin IQR, then the median across coordinates is taken. The primary model set requires at least 90% of the 531 basins to have an eligible non-canonical primary restart.

### IC-self

`F` is each fixed synthetic IC-self field from `primary_draw_indices`: one eligible non-canonical restart is selected per basin and the same restart realization supplies all coordinates, preserving the within-basin joint structure. For each of 5,000 draws, coordinate-wise basin IQR/IC-IQR produces a CR field; the median across draws is then taken per coordinate, followed by the median across coordinates for the model-level CR. The same primary eligibility and no-fallback rule applies.

## Why the current table has 23 rows

The corrected primary analysis intentionally filters the 36 upstream model diagnostics to the 23 models with primary non-canonical restart coverage at least 0.90. The 13 excluded models are: `{missing_text}`. Numeric rows retained in the upstream `model_summary.csv` for those models are marked `INSUFFICIENT_REFERENCE`; using them would mix incomplete primary reference fields into the formal model-equal estimand. The all-valid restart pool is a sensitivity product and is not substituted for the frozen primary pool.
"""
    (TABLES / "F2_CR_REFERENCE_DEFINITIONS.md").write_text(definitions)

    report = f"""# F2 CR provenance audit report

## CR DATA VERDICT = STRICT23 ONLY

The provenance-valid primary CR dataset contains **23 models × 3 references = 69 values**. A full-36 primary ridgeline is not valid because 13 models fail the frozen primary IC restart-reference coverage gate. Their upstream diagnostics retain numeric coverage-incomplete rows, but those rows are explicitly marked `INSUFFICIENT_REFERENCE`; treating them as primary-valid would change the estimand. Therefore Phase B, if performed, must use the same strict 23-model set for all three ridgelines and must disclose `23/36`.

## Current 23-model table

- Cache: `{rel(CR_CACHE)}`
- Rows: `{len(cr)}`
- Current IDs match the primary eligible set: `{current_ids == sorted(eligible)}`
- Current IDs match strict rank set: `{set(current_ids) == rank_ids}`
- Current IDs match strict localization set: `{set(current_ids) == localization_ids}`
- Source script: `{'; '.join(sorted(cr.source_script.dropna().astype(str).unique()))}`
- Source files: `{source_files}`
- Current cache vs upstream `model_summary.csv` maximum absolute difference across the 23 models and 3 references: `{max_source_backcheck:.3e}`.

## Full-36 availability interpretation

- Canonical inputs and canonical CR calculations are available for all 36 models.
- Consensus and IC-self primary fields are formally available for the 23 models passing the >=0.90 primary coverage gate.
- For the 13 excluded models, upstream `model_summary.csv` contains coverage-incomplete diagnostic values, but the formal status is `INSUFFICIENT_REFERENCE`; these cells are classified `STRICT_FILTER_ONLY`, not promoted to primary-valid values.
- The shared contract explicitly retains unavailable primary basins and prohibits canonical fallback.

Excluded 13 models: `{missing_text}`.

## Headline reproduction

The frozen headline values are strict23 quantities:

- canonical: `{FROZEN['canonical']:.10f}`
- consensus: `{FROZEN['consensus']:.10f}`
- IC-self: `{FROZEN['IC-self']:.10f}`

The headline table records both these accepted strict23 values and the non-primary full36 diagnostic medians. The latter are not used for the final panel.

{chr(10).join(summary_lines)}

## Reconstruction decision

No new full36 CR file was created. Although raw 36-model artifacts exist (`raw_replicate_arrays.npz` has 36 model axes and primary/all-valid fields), the 13 insufficient-reference models do not satisfy the frozen primary eligibility condition. The all-valid sensitivity pool is a different reference construction and cannot be silently substituted. The provenance-valid action is therefore a strict23 ridgeline, not a fabricated full36 ridgeline.

## Source artifacts and hashes

{source_lines}

## Required Phase B disclosure

If redrawn, panel (c) must use the identical 23-model set in all three rows and include either the title suffix `(strict subset)` or an explicit `strict subset: 23/36 models` annotation/caption. It must use the three 23-value distributions, not the coverage-incomplete full36 diagnostic rows.
"""
    (TABLES / "F2_CR_PROVENANCE_AUDIT_REPORT.md").write_text(report)
    print({
        "status": "COMPLETE",
        "verdict": "STRICT23 ONLY",
        "current_rows": len(cr),
        "full_models": len(full_ids),
        "excluded_models": len(insufficient),
        "headline_values": {k: float(cr[v].median()) for k, v in CR_COLUMNS.items()},
        "outputs": [
            "tables/F2_CR_CURRENT23_MODEL_LIST.csv",
            "tables/F2_CR_AVAILABILITY_MATRIX.csv",
            "tables/F2_CR_HEADLINE_REPRODUCTION.csv",
            "tables/F2_CR_REFERENCE_DEFINITIONS.md",
            "tables/F2_CR_PROVENANCE_AUDIT_REPORT.md",
        ],
    })


if __name__ == "__main__":
    main()
