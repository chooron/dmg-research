#!/usr/bin/env python3
"""Rule-based representative-model selection for a future OOB experiment.

This script only reads frozen seen-basin summaries.  It does not access OOB/PUB
outputs and never trains, optimizes, or updates a checkpoint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from selection_common import (  # noqa: E402
    CONTRAST_RAW,
    MODELS,
    OUT,
    SELECTION_RAW,
    SELECTION_RANK,
    euclidean,
    load_features,
    write_csv,
    write_json,
)

QUADRANTS = ["Q1_LOW_G_HIGH_R", "Q2_LOW_G_LOW_R", "Q3_HIGH_G_HIGH_R", "Q4_HIGH_G_LOW_R"]
QUAD_LABELS = {"Q1_LOW_G_HIGH_R": "low G / high R", "Q2_LOW_G_LOW_R": "low G / low R", "Q3_HIGH_G_HIGH_R": "high G / high R", "Q4_HIGH_G_LOW_R": "high G / low R"}
COVERAGE_TARGET = 2


def initial_selection(features: pd.DataFrame, thresholds: dict[str, float]):
    selected = {}
    records = []
    ranking_frames = []
    pool_by_quad = {}
    slot_scores = {}
    quadrant_rows = []
    performance_cut = thresholds["K_joint_Q25"]
    for quadrant in QUADRANTS:
        group = features[features.quadrant == quadrant].copy()
        gated = group[group.performance_gate_pass].copy()
        pool = gated if len(gated) >= 2 else group
        relaxed = len(gated) < 2
        pool_by_quad[quadrant] = set(pool.model)
        center = pool[SELECTION_RANK].mean().to_numpy(float)
        group["representative_distance"] = euclidean(group, SELECTION_RANK, center)
        rep_pool = group[group.model.isin(pool.model)].sort_values(["representative_distance", "model"])
        rep = rep_pool.iloc[0]
        group["contrast_distance"] = np.sqrt(np.square(group[[f"{x}_pct" for x in CONTRAST_RAW]].to_numpy(float) - rep[[f"{x}_pct" for x in CONTRAST_RAW]].to_numpy(float)).sum(axis=1))
        contrast_pool = group[(group.model.isin(pool.model)) & (group.model != rep.model)].sort_values(["contrast_distance", "model"], ascending=[False, True])
        contrast = contrast_pool.iloc[0]
        selected[f"{quadrant}:REP"] = rep.model
        selected[f"{quadrant}:CONTRAST"] = contrast.model
        for _, row in group.iterrows():
            slot_scores[(quadrant, row.model)] = {"representative_distance": float(row.representative_distance), "contrast_distance": float(row.contrast_distance)}
        rank = group.copy()
        rank["quadrant_pool"] = rank.model.isin(pool.model)
        rank["performance_gate_pass"] = rank.K_joint >= performance_cut
        rank["performance_gate_relaxed_for_quadrant"] = relaxed
        rank["representative_rank_in_pool"] = rank[rank.quadrant_pool].representative_distance.rank(method="min", ascending=True)
        rank["contrast_rank_in_pool"] = rank[rank.quadrant_pool].contrast_distance.rank(method="min", ascending=False)
        ranking_frames.append(rank)
        quadrant_rows.append({"quadrant": quadrant, "quadrant_label": QUAD_LABELS[quadrant], "candidate_count": len(group), "performance_gate_pass_count": len(gated), "selection_pool_count": len(pool), "performance_gate_relaxed": relaxed, "performance_gate_threshold_K_joint_Q25": performance_cut, "centroid_definition": "mean of seven percentile-rank features within performance selection pool", "representative_model_initial": rep.model, "contrast_model_initial": contrast.model})
    return selected, pd.DataFrame(quadrant_rows), pd.concat(ranking_frames, ignore_index=True), pool_by_quad, slot_scores


def coverage_counts(selected: dict[str, str], features: pd.DataFrame) -> dict[str, int]:
    rows = features[features.model.isin(selected.values())]
    return {
        "A_low": int((rows.A_tercile == "low").sum()),
        "A_high": int((rows.A_tercile == "high").sum()),
        "P_low": int((rows.P_tercile == "low").sum()),
        "P_medium": int((rows.P_tercile == "medium").sum()),
        "P_high": int((rows.P_tercile == "high").sum()),
    }


def coverage_deficit(counts: dict[str, int]) -> int:
    return sum(max(0, COVERAGE_TARGET - value) for value in counts.values())


def apply_coverage_repairs(selected, features, pool_by_quad, slot_scores):
    adjustments = []
    requirements = [("A_low", "A_tercile", "low"), ("A_high", "A_tercile", "high"), ("P_low", "P_tercile", "low"), ("P_medium", "P_tercile", "medium"), ("P_high", "P_tercile", "high")]
    for requirement, column, value in requirements:
        while coverage_counts(selected, features)[requirement] < COVERAGE_TARGET:
            current_deficit = coverage_deficit(coverage_counts(selected, features))
            proposals = []
            selected_models = set(selected.values())
            for slot, current_model in sorted(selected.items()):
                quadrant = slot.split(":")[0]
                role = slot.split(":")[1]
                alternatives = features[(features.quadrant == quadrant) & (features.model.isin(pool_by_quad[quadrant])) & (features[column] == value) & (~features.model.isin(selected_models))]
                for _, candidate in alternatives.iterrows():
                    trial = dict(selected); trial[slot] = candidate.model
                    trial_counts = coverage_counts(trial, features)
                    trial_deficit = coverage_deficit(trial_counts)
                    if trial_deficit >= current_deficit:
                        continue
                    old_score = slot_scores[(quadrant, current_model)]["representative_distance" if role == "REP" else "contrast_distance"]
                    new_score = slot_scores[(quadrant, candidate.model)]["representative_distance" if role == "REP" else "contrast_distance"]
                    loss = new_score - old_score if role == "REP" else old_score - new_score
                    proposals.append((trial_deficit, loss, quadrant, role, candidate.model, current_model, trial_counts))
            if not proposals:
                break
            proposals.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
            _, loss, quadrant, role, candidate_model, old_model, trial_counts = proposals[0]
            slot = f"{quadrant}:{role}"
            selected[slot] = candidate_model
            adjustments.append({"requirement": requirement, "quadrant": quadrant, "role": role, "old_model": old_model, "new_model": candidate_model, "selection_score_loss": loss, "resulting_coverage_counts": trial_counts})
    return selected, adjustments


def fallback_selection(selected, features):
    reps = [selected[f"{quadrant}:REP"] for quadrant in QUADRANTS]
    contrasts = [selected[f"{quadrant}:CONTRAST"] for quadrant in QUADRANTS]
    chosen = list(reps)
    vectors = features.set_index("model")[[f"{x}_pct" for x in CONTRAST_RAW]]
    while len(chosen) < 6:
        candidates = [model for model in contrasts if model not in chosen]
        scored = []
        for model in candidates:
            distances = [float(np.linalg.norm(vectors.loc[model].to_numpy() - vectors.loc[other].to_numpy())) for other in chosen]
            scored.append((min(distances), model))
        scored.sort(key=lambda x: (-x[0], x[1]))
        chosen.append(scored[0][1])
    return chosen


def main() -> None:
    features, thresholds = load_features()
    selected, quadrant_table, ranking, pool_by_quad, slot_scores = initial_selection(features, thresholds)
    initial_selected = dict(selected)
    selected, adjustments = apply_coverage_repairs(selected, features, pool_by_quad, slot_scores)
    primary_models = list(selected.values())
    fallback_models = fallback_selection(selected, features)
    role_labels = {model: f"{key.split(':')[0].split('_')[0]}_{key.split(':')[1]}" for key, model in selected.items()}
    fallback_roles = {model: ";".join(sorted(role_labels[candidate] for candidate in selected.values() if candidate == model)) for model in fallback_models}

    counts = coverage_counts(selected, features)
    coverage_pass = all(counts[key] >= COVERAGE_TARGET for key in counts)
    ranking["primary_role"] = ranking.apply(lambda row: role_labels.get(row.model, ""), axis=1)
    ranking["selected_primary_8"] = ranking.model.isin(primary_models)
    ranking["selected_fallback_6"] = ranking.model.isin(fallback_models)
    ranking["tercile_extreme_status"] = ranking.apply(lambda row: f"G_{row.G_extreme}_R_{row.R_extreme}", axis=1)
    ranking["selection_adjusted_after_coverage_gate"] = ranking.model.isin([x["new_model"] for x in adjustments])
    ranking["selection_score_note"] = ranking.apply(lambda row: "rep centroid distance / contrast 5D distance computed within its quadrant pool" if row.quadrant_pool else "not in quadrant selection pool", axis=1)
    write_csv(OUT / "OOB_MODEL_SELECTION_RANKING.csv", ranking.sort_values(["quadrant", "selected_primary_8", "primary_role", "model"], ascending=[True, False, True, True]))

    feature_out = features.copy()
    feature_out["primary_role"] = feature_out.model.map(role_labels).fillna("")
    feature_out["selected_primary_8"] = feature_out.model.isin(primary_models)
    feature_out["selected_fallback_6"] = feature_out.model.isin(fallback_models)
    feature_out["selection_pool"] = feature_out.apply(lambda row: row.model in pool_by_quad[row.quadrant], axis=1)
    write_csv(OUT / "OOB_MODEL_SELECTION_FEATURES.csv", feature_out)

    quadrant_table["primary_models_final"] = quadrant_table.quadrant.map(lambda q: ";".join(selected[f"{q}:{role}"] for role in ["REP", "CONTRAST"]))
    quadrant_table["coverage_gate_adjustment_count"] = quadrant_table.quadrant.map(lambda q: sum(x["quadrant"] == q for x in adjustments))
    write_csv(OUT / "OOB_MODEL_SELECTION_QUADRANTS.csv", quadrant_table)

    final_rows = []
    for role, model in selected.items():
        row = features[features.model == model].iloc[0].to_dict()
        row.update({"role": role_labels[model], "primary_8": True, "fallback_6": model in fallback_models, "selection_reason": "quadrant-centroid representative" if role.endswith("REP") else "same-quadrant contrast maximizing D/U/A/P/K_joint distance from representative"})
        final_rows.append(row)
    final_table = pd.DataFrame(final_rows).sort_values("role")
    write_csv(OUT / "OOB_PRIMARY_8_MODEL_TABLE.csv", final_table)
    (OUT / "OOB_PRIMARY_8_MODELS.txt").write_text("\n".join(f"{row.role}: {row.model}" for row in final_table.itertuples()) + "\n")
    (OUT / "OOB_FALLBACK_6_MODELS.txt").write_text("\n".join(f"{model}: {fallback_roles[model]}" for model in fallback_models) + "\n")

    features["primary_role"] = features.model.map(role_labels).fillna("")
    tercile = features[features.model.isin(primary_models)][["model", "quadrant", "G_extreme", "R_extreme", "A_tercile", "P_tercile", "primary_role"]].copy()
    tercile["quadrant_extreme_match"] = tercile.apply(lambda row: (row.G_extreme == ("low" if "LOW_G" in row.quadrant else "high")) and (row.R_extreme == ("high" if "HIGH_R" in row.quadrant else "low")), axis=1)
    write_csv(OUT / "OOB_SELECTION_TERCILE_SENSITIVITY.csv", tercile.sort_values("primary_role"))

    exclusions = pd.DataFrame([{"candidate_pool": "36 canonical models", "excluded_model": "", "excluded": False, "reason": "none; all models have frozen IC/dPL, atlas, and restart diagnostics"}])
    write_csv(OUT / "OOB_MODEL_SELECTION_EXCLUSION_GATE.csv", exclusions)
    write_json(OUT / "OOB_SELECTION_PROVENANCE.json", {"candidate_pool_n": 36, "candidate_models": MODELS, "G_seen_definition": "median basin-level (KGE_IC-KGE_dPL)", "features": SELECTION_RAW, "percentile_rank_definition": "(average rank - 1)/(N - 1), N=36", "G_median": thresholds["G_median"], "R_median": thresholds["R_median"], "K_joint_Q25": thresholds["K_joint_Q25"], "tercile_thresholds": {key: thresholds[key] for key in thresholds if "Q33" in key or "Q67" in key}, "performance_gate_relaxed_quadrants": quadrant_table[quadrant_table.performance_gate_relaxed].quadrant.tolist(), "coverage_counts_final": counts, "coverage_gate_pass": coverage_pass, "coverage_gate_rule": "at least two A-low and A-high plus two P-low, P-medium, P-high; repairs remain within quadrant pool", "family_metadata_found": False, "family_metadata_note": "No trusted frozen family metadata found; no name-based family was invented.", "manual_override": False, "oob_information_used": False, "training_started": False, "simhyd_included": "simhyd" in MODELS, "vic_dynamic_doy_included": True, "initial_selection": initial_selected, "coverage_adjustments": adjustments, "primary_8": primary_models, "fallback_6": fallback_models})

    report = render_report(features, thresholds, quadrant_table, final_table, fallback_models, tercile, adjustments, coverage_pass, counts, ranking)
    (OUT / "OOB_MODEL_SELECTION_REPORT.md").write_text(report)
    print("Selection complete", "primary8", primary_models, "fallback6", fallback_models, "coverage", counts, "pass", coverage_pass)


def render_report(features, thresholds, quadrant_table, final_table, fallback_models, tercile, adjustments, coverage_pass, counts, ranking) -> str:
    selected8 = set(final_table.model)
    performance_relaxed = quadrant_table[quadrant_table.performance_gate_relaxed].quadrant.tolist()
    model_lines = []
    for row in final_table.itertuples():
        model_lines.append(f"| {row.model} | {row.role} | {row.G:.6f} | {row.R:.6f} | {row.D:.6f} | {row.U:.6f} | {row.K_IC:.6f} | {row.K_dPL:.6f} | {int(row.A)} | {int(row.P)} | {row.selection_reason} |")
    tercile_counts = tercile.groupby("quadrant_extreme_match").size().to_dict()
    g_extreme_count = int((tercile.G_extreme != "middle").sum())
    r_extreme_count = int((tercile.R_extreme != "middle").sum())
    return f"""# OOB representative-model selection

## Decision

**Rule-based selection only; no OOB/PUB run and no training were started.** The candidate pool is the complete frozen canonical 36-model set. `G = median_b(KGE_IC - KGE_dPL)` is the seen-basin shared-mapping flexibility gap; `R` is frozen parameter–attribute reproducibility, `D` is frozen bounds-normalized RMS parameter realization distance, and `U` is frozen median IC restart uncertainty.

## Frozen thresholds and gates

- G median split threshold: **{thresholds['G_median']:.10f}**.
- R median split threshold: **{thresholds['R_median']:.10f}**.
- Performance viability: `K_joint=min(K_IC,K_dPL) >= Q25`, with Q25 **{thresholds['K_joint_Q25']:.10f}**.
- Percentile ranks use `(average rank−1)/(36−1)` and all seven selection dimensions are equally weighted.
- Primary selection: each G×R quadrant gets a centroid-nearest representative and a same-quadrant 5D contrast; coverage repairs, if any, remain within the quadrant selection pool.
- A signal: count of BH-FDR significant (`q<0.05`) G-vs-35-attribute associations; max and median absolute rho are retained in the feature CSV.
- No trusted frozen model-family metadata was found; no family was invented from model names.

## Primary 8

| model | role | G_seen | R | D_theta | restart U | K_IC | K_dPL | A | param count | reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(model_lines)}

## Fallback 6

`{' '.join(fallback_models)}`. This retains one representative per quadrant and selects two contrast models by sequential maximin distance in the percentile-ranked **D/U/A/P/K_joint** space.

## Coverage and sensitivity

- Final A/P coverage counts: `{counts}`; gate pass: **{coverage_pass}**.
- Performance-gate-relaxed quadrants: `{performance_relaxed if performance_relaxed else 'none'}`.
- Coverage adjustments: **{len(adjustments)}**; exact replacements are recorded in `OOB_SELECTION_PROVENANCE.json`.
- Tercile-extreme check: both-axis quadrant match `{tercile_counts}`; G is in a low/high extreme for **{g_extreme_count}/8**, and R is in a low/high extreme for **{r_extreme_count}/8**. Full model-level rows are in `OOB_SELECTION_TERCILE_SENSITIVITY.csv`.
- Manual override: **NO**. Exclusions: **none**.

## Quadrant interpretation

{chr(10).join(f"- **{row.quadrant} ({row.quadrant_label})**: final `{row.primary_models_final}`; performance gate pass `{int(row.performance_gate_pass_count)}/{int(row.candidate_count)}`, relaxed `{bool(row.performance_gate_relaxed)}`." for row in quadrant_table.itertuples())}

## Future hypotheses (not answered here)

- H1: whether low-G/high-R models retain better OOB parameter realization.
- H2: whether high-G/low-R models show greater OOB degradation.
- H3: whether Q2/Q3 demonstrate that performance flexibility and parameter reliability are distinct dimensions.
- H4: whether model–place signal strength predicts OOB behavior.

These remain preregisterable OOB hypotheses. This selection cannot answer them because no OOB information was used.

## Required outputs

- `OOB_MODEL_SELECTION_FEATURES.csv`
- `OOB_MODEL_SELECTION_QUADRANTS.csv`
- `OOB_MODEL_SELECTION_RANKING.csv`
- `OOB_PRIMARY_8_MODELS.txt`, `OOB_FALLBACK_6_MODELS.txt`
- `qc_G_vs_reproducibility_selected.png`, `qc_selected_feature_heatmap.png`
- `OOB_SELECTION_TERCILE_SENSITIVITY.csv`, `OOB_SELECTION_PROVENANCE.json`

No OOB/PUB, multi-seed, PUR, H1, or training operation was started.
"""


if __name__ == "__main__":
    main()
