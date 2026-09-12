#!/usr/bin/env python3
"""Re-freeze the OOB representative list under the MOPEX redundancy gate.

The original selector remains untouched.  This script evaluates Scenario A and
B using the same quadrant/performance/percentile rules, then writes new files.
It performs no OOB/PUB analysis, training, or checkpoint operation.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from select_models import QUADRANTS, fallback_selection, initial_selection  # noqa: E402
from selection_common import CONTRAST_RAW, MODELS, OUT, SELECTION_RANK, load_features, write_csv, write_json  # noqa: E402
from structure_qc import coverage_summary, descriptors  # noqa: E402

ORIGINAL_RESULT = OUT / "OOB_PRIMARY_8_MODEL_TABLE.csv"


def row_for(ranking: pd.DataFrame, model: str) -> pd.Series:
    return ranking[ranking.model == model].iloc[0]


def role_score(ranking: pd.DataFrame, model: str, role: str) -> float:
    row = row_for(ranking, model)
    return float(row.representative_distance if role == "REP" else row.contrast_distance)


def pairwise_stats(features: pd.DataFrame, models: list[str]) -> tuple[float, float]:
    x = features.set_index("model").loc[models, SELECTION_RANK].to_numpy(float)
    distances = [float(np.linalg.norm(x[i] - x[j])) for i in range(len(x)) for j in range(i + 1, len(x))]
    return min(distances), float(np.mean(distances))


def scenario_selection(base, ranking, features, pool_by_quad, banned):
    affected = [(slot, model) for slot, model in base.items() if model in banned]
    fixed = {model for slot, model in base.items() if model not in banned}
    retained_mopex = {model for model in fixed if model.startswith("mopex")}
    option_map = {}
    for slot, old_model in affected:
        quadrant, role = slot.split(":")
        candidates = features[(features.quadrant == quadrant) & (features.model.isin(pool_by_quad[quadrant])) & (~features.model.isin(banned | fixed))].copy()
        candidates = candidates[(~candidates.model.str.startswith("mopex")) | candidates.model.isin(retained_mopex)]
        candidates = candidates[~candidates.model.isin({old for other, old in affected if other != slot})]
        score_col = "representative_distance" if role == "REP" else "contrast_distance"
        candidates["replacement_score"] = candidates.model.map(lambda model: role_score(ranking, model, role))
        candidates = candidates.sort_values(["replacement_score", "model"], ascending=[role == "REP", True])
        option_map[slot] = candidates.model.tolist()

    performance_valid, all_valid = [], []
    slots = [slot for slot, _ in affected]
    for choices in itertools.product(*(option_map[slot] for slot in slots)):
        trial = dict(base)
        for slot, model in zip(slots, choices):
            trial[slot] = model
        names = list(trial.values())
        if len(set(names)) != 8 or any(model in banned for model in names) or sum(model.startswith("mopex") for model in names) != 1:
            continue
        chosen = features[features.model.isin(names)]
        counts = {"A_low": int((chosen.A_tercile == "low").sum()), "A_high": int((chosen.A_tercile == "high").sum()), "P_low": int((chosen.P_tercile == "low").sum()), "P_medium": int((chosen.P_tercile == "medium").sum()), "P_high": int((chosen.P_tercile == "high").sum())}
        if not bool(chosen.performance_gate_pass.all()):
            continue
        losses = []
        for slot, old_model in affected:
            role = slot.split(":")[1]
            old_score = role_score(ranking, old_model, role)
            new_score = role_score(ranking, trial[slot], role)
            losses.append(new_score - old_score if role == "REP" else old_score - new_score)
        item = (float(sum(losses)), tuple(sorted(names)), trial, counts)
        performance_valid.append(item)
        if counts["A_low"] >= 2 and counts["A_high"] >= 2 and min(counts.values()) >= 2:
            all_valid.append(item)
    pool = all_valid if all_valid else performance_valid
    if not pool:
        return {"status": "BLOCKED", "reason": "No same-quadrant legal replacement preserves the performance gate", "affected": [{"slot": slot, "old_model": old} for slot, old in affected], "candidate_option_counts": {slot: len(options) for slot, options in option_map.items()}}
    pool.sort(key=lambda item: (item[0], item[1]))
    loss, _, selected, counts = pool[0]
    failed_gates = []
    if not all_valid:
        failed_gates.append("complexity_tertile_coverage")
    return {"status": "PASS" if all_valid else "BLOCKED", "reason": "all hard gates pass" if all_valid else "No legal same-quadrant replacement satisfies all hard gates; best performance-valid candidate retained for comparison", "failed_gates": failed_gates, "selected": selected, "counts": counts, "objective_loss": loss, "affected": [{"slot": slot, "old_model": old, "new_model": selected[slot]} for slot, old in affected], "candidate_option_counts": {slot: len(options) for slot, options in option_map.items()}, "candidate_combinations_passing_performance": len(performance_valid), "candidate_combinations_passing_all_gates": len(all_valid)}


def quality(scenario_name, selected, features, thresholds, objective_loss, counts, original_models):
    models = list(selected.values())
    frame = features[features.model.isin(models)].copy()
    min_pair, mean_pair = pairwise_stats(features, models)
    g_extreme = int((frame.G_extreme != "middle").sum())
    r_extreme = int((frame.R_extreme != "middle").sum())
    joint_extreme = int(((frame.G_extreme != "middle") & (frame.R_extreme != "middle")).sum())
    return {
        "scenario": scenario_name,
        "model_count": len(models),
        "quadrant_complete": len(models) == 8 and len(set(frame.quadrant)) == 4 and bool((frame.groupby("quadrant").size() == 2).all()),
        "performance_gate_all_pass": bool(frame.performance_gate_pass.all()),
        "A_low_count": counts["A_low"], "A_high_count": counts["A_high"], "A_coverage_pass": counts["A_low"] >= 2 and counts["A_high"] >= 2,
        "P_low_count": counts["P_low"], "P_medium_count": counts["P_medium"], "P_high_count": counts["P_high"], "complexity_coverage_pass": min(counts.values()) >= 2,
        "mopex_count": int(frame.model.str.startswith("mopex").sum()), "mopex_gate_pass": int(frame.model.str.startswith("mopex").sum()) == 1 and "mopex2" not in set(frame.model),
        "min_pairwise_feature_distance_7D": min_pair, "mean_pairwise_feature_distance_7D": mean_pair,
        "selection_objective_loss_vs_original": objective_loss, "models_changed_from_original": len(set(original_models) - set(models)),
        "G_extreme_count": g_extreme, "R_extreme_count": r_extreme, "G_R_joint_extreme_count": joint_extreme,
        "G_extreme_coverage": g_extreme / 8.0, "R_extreme_coverage": r_extreme / 8.0, "G_R_joint_extreme_coverage": joint_extreme / 8.0,
        "all_hard_gates_pass": len(models) == 8 and len(set(frame.quadrant)) == 4 and bool((frame.groupby("quadrant").size() == 2).all()) and bool(frame.performance_gate_pass.all()) and counts["A_low"] >= 2 and counts["A_high"] >= 2 and min(counts.values()) >= 2 and int(frame.model.str.startswith("mopex").sum()) == 1 and "mopex2" not in set(frame.model),
    }


def render_report(scenarios, quality_table, final_name, final_selected, fallback, thresholds, structural):
    lines = []
    for row in final_selected.itertuples():
        lines.append(f"| {row.model} | {row.role} | {row.G:.6f} | {row.R:.6f} | {row.D:.6f} | {row.U:.6f} | {row.K_IC:.6f} | {row.K_dPL:.6f} | {int(row.A)} | {int(row.P)} |")
    qlines = []
    for row in quality_table.itertuples():
        qlines.append(f"| {row.scenario} | {bool(row.all_hard_gates_pass)} | {row.min_pairwise_feature_distance_7D:.6f} | {row.mean_pairwise_feature_distance_7D:.6f} | {row.min_pairwise_delta_vs_original:.6f} | {row.mean_pairwise_delta_vs_original:.6f} | {row.selection_objective_loss_vs_original:.6f} | {row.G_extreme_count}/8 | {row.R_extreme_count}/8 | {row.G_R_joint_extreme_count}/8 | {row.A_low_count}/{row.A_high_count} | {row.P_low_count}/{row.P_medium_count}/{row.P_high_count} |")
    adjustment_text = []
    for name, data in scenarios.items():
        if data["status"] == "BLOCKED":
            replacement = "; ".join(f"{item['slot']} {item['old_model']}→{item.get('new_model', 'none')}" for item in data["affected"])
            adjustment_text.append(f"- {name}: BLOCKED — {data['reason']}; candidate replacement {replacement}; option counts {data['candidate_option_counts']}")
        else:
            adjustment_text.append(f"- {name}: " + "; ".join(f"{item['slot']} {item['old_model']}→{item['new_model']}" for item in data["affected"]))
    return f"""# OOB model selection — MOPEX redundancy revision

## Decision

**Final scenario: {final_name}**. This revision keeps the original G_seen×reproducibility quadrants, percentile-rank feature definitions, performance gate, A gate, and complexity gate. It applies only the hard redundancy rule: `mopex2` absent and exactly one of `mopex4`/`mopex5` retained. No OOB/PUB information, training, multi-seed, or PUR operation was used.
No revised PRIMARY 8 or FALLBACK 6 is adopted because both legal scenarios fail the unchanged parameter-complexity tertile gate under same-quadrant replacements. Scenario candidate tables and the blocker evidence are retained; the original primary list remains the only non-blocked list.

The two scenarios were evaluated before choosing the final list. Among scenarios passing all hard gates, the deterministic quality precedence was: higher 7D percentile feature-space minimum pairwise distance, then higher mean pairwise distance, then greater joint/G/R extreme coverage, then lower role-selection objective loss relative to the original primary set. Replacement candidates were restricted to the original quadrant performance pool.

## Thresholds

- G median split: **{thresholds['G_median']:.10f}**.
- R median split: **{thresholds['R_median']:.10f}**.
- K_joint Q25 performance threshold: **{thresholds['K_joint_Q25']:.10f}**.
- A/P coverage repairs remain within quadrants; manual override: **NO**.

## Scenario comparison

| scenario | all hard gates | min pairwise 7D | mean pairwise 7D | min Δ vs orig | mean Δ vs orig | objective loss | G extreme | R extreme | joint extreme | A low/high | P low/medium/high |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(qlines)}

Scenario replacements (no manual choices):
{chr(10).join(adjustment_text)}

## Final PRIMARY 8

| model | role | G_seen | R | D_theta | restart U | K_IC | K_dPL | A | P |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

## Fallback 6

{("`" + " ".join(fallback) + "`. Four quadrant representatives are retained; the remaining two are selected from the four contrasts by the original sequential maximin rule in D/U/A/P/K_joint percentile space. MOPEX count in fallback is **" + str(sum(model.startswith("mopex") for model in fallback)) + "**.") if fallback else "No FALLBACK 6 is frozen because no scenario passed all hard gates."}

## Structural coverage QC (auxiliary only)

Registry/core descriptors were extracted without simulation and were not used as selection criteria. Auxiliary MOPEX counts by population are **{'; '.join(f'{row.population}={int(row.mopex_count)}' for row in structural.itertuples())}**. See `OOB_STRUCTURAL_MODEL_DESCRIPTORS.csv` and `OOB_STRUCTURAL_COVERAGE_QC.csv` for parameter count, state/storage count, snow markers, routing form, and core runoff-definition coverage.

## Future hypotheses (not answered)

H1–H4 remain untested OOB hypotheses: low-G/high-R OOB realization, high-G/low-R degradation, Q2/Q3 separation of performance flexibility and reliability, and model–place signal prediction. This file only freezes a rule-based candidate list.

## Output/provenance

- Original selection files are preserved unchanged.
- Revised features: `OOB_REVISED_SELECTION_FEATURES.csv`.
- Scenario tables: `OOB_SCENARIO_A_KEEP_MOPEX5.csv`, `OOB_SCENARIO_B_KEEP_MOPEX4.csv`.
- Revised lists: `OOB_PRIMARY_8_MODELS_REVISED.txt`, `OOB_FALLBACK_6_MODELS_REVISED.txt`.
- Scenario quality: `OOB_SCENARIO_SELECTION_QUALITY.csv`.
- Structural QC: `OOB_STRUCTURAL_COVERAGE_QC.csv`.

`simhyd` participated in the original pool but was not removed by the MOPEX gate. VIC remained eligible under its dynamic-DOY canonical result. No OOB/PUB or training operation was started.
"""


def main() -> None:
    features, thresholds = load_features()
    base, _, ranking, pool_by_quad, _ = initial_selection(features, thresholds)
    # Reproduce the already-frozen original primary list before applying the new gate.
    original = pd.read_csv(ORIGINAL_RESULT)
    original_models = original.model.tolist()
    original_roles = {f"{row.quadrant}:{'REP' if row.role.endswith('_REP') else 'CONTRAST'}": row.model for row in original.itertuples()}
    if set(original_roles.values()) != set(original_models):
        raise RuntimeError("original primary table role/model mismatch")
    base = original_roles
    scenarios = {}
    quality_rows = []
    for name, banned in [("SCENARIO_A_KEEP_MOPEX5", {"mopex2", "mopex4"}), ("SCENARIO_B_KEEP_MOPEX4", {"mopex2", "mopex5"})]:
        result = scenario_selection(base, ranking, features, pool_by_quad, banned)
        scenarios[name] = result
        if result.get("selected") is not None:
            qrow = quality(name, result["selected"], features, thresholds, result["objective_loss"], result["counts"], original_models)
            qrow["selection_status"] = result["status"]
            qrow["failed_gates"] = ";".join(result.get("failed_gates", []))
            if result["status"] != "PASS":
                qrow["all_hard_gates_pass"] = False
            quality_rows.append(qrow)
        else:
            quality_rows.append({"scenario": name, "selection_status": "BLOCKED", "failed_gates": "performance_gate", "model_count": 0, "quadrant_complete": False, "performance_gate_all_pass": False, "A_low_count": 0, "A_high_count": 0, "A_coverage_pass": False, "P_low_count": 0, "P_medium_count": 0, "P_high_count": 0, "complexity_coverage_pass": False, "mopex_count": None, "mopex_gate_pass": False, "min_pairwise_feature_distance_7D": np.nan, "mean_pairwise_feature_distance_7D": np.nan, "selection_objective_loss_vs_original": np.nan, "models_changed_from_original": None, "G_extreme_count": None, "R_extreme_count": None, "G_R_joint_extreme_count": None, "G_extreme_coverage": np.nan, "R_extreme_coverage": np.nan, "G_R_joint_extreme_coverage": np.nan, "all_hard_gates_pass": False})
    quality_table = pd.DataFrame(quality_rows)
    original_min_pair, original_mean_pair = pairwise_stats(features, original_models)
    quality_table["original_min_pairwise_feature_distance_7D"] = original_min_pair
    quality_table["original_mean_pairwise_feature_distance_7D"] = original_mean_pair
    quality_table["min_pairwise_delta_vs_original"] = quality_table.min_pairwise_feature_distance_7D - original_min_pair
    quality_table["mean_pairwise_delta_vs_original"] = quality_table.mean_pairwise_feature_distance_7D - original_mean_pair
    passing = quality_table[quality_table.all_hard_gates_pass]
    if len(passing) == 0:
        final_name = "BLOCKED_NO_FINAL_PRIMARY_8"
        final_selected_map = {}
    else:
        passing = passing.sort_values(["min_pairwise_feature_distance_7D", "mean_pairwise_feature_distance_7D", "G_R_joint_extreme_count", "G_extreme_count", "R_extreme_count", "selection_objective_loss_vs_original"], ascending=[False, False, False, False, False, True])
        final_name = passing.iloc[0].scenario
        final_selected_map = scenarios[final_name]["selected"]
    final_models = list(final_selected_map.values())
    final_roles = {model: f"{slot.split(':')[0].split('_')[0]}_{slot.split(':')[1]}" for slot, model in final_selected_map.items()}
    revised_features = features.copy()
    revised_features["revised_role"] = revised_features.model.map(final_roles).fillna("")
    revised_features["selected_revised_primary_8"] = revised_features.model.isin(final_models)
    write_csv(OUT / "OOB_REVISED_SELECTION_FEATURES.csv", revised_features)
    for name, result in scenarios.items():
        output_name = OUT / ("OOB_SCENARIO_A_KEEP_MOPEX5.csv" if name.startswith("SCENARIO_A") else "OOB_SCENARIO_B_KEEP_MOPEX4.csv")
        if result.get("selected") is None:
            blocked = pd.DataFrame([{ "scenario": name, "status": "BLOCKED", "reason": result["reason"], "affected_slots": ";".join(item["slot"] for item in result["affected"]), "candidate_option_counts": str(result["candidate_option_counts"]) }])
            write_csv(output_name, blocked)
            continue
        rows = []
        for slot, model in result["selected"].items():
            row = features[features.model == model].iloc[0].to_dict()
            row.update({"scenario": name, "status": result["status"], "role": f"{slot.split(':')[0].split('_')[0]}_{slot.split(':')[1]}", "selected": True, "scenario_objective_loss": result["objective_loss"], "failed_gates": ";".join(result.get("failed_gates", []))})
            rows.append(row)
        write_csv(output_name, pd.DataFrame(rows).sort_values("role"))
    write_csv(OUT / "OOB_SCENARIO_SELECTION_QUALITY.csv", quality_table)
    if final_models:
        final_table = pd.DataFrame([dict(features[features.model == model].iloc[0], role=final_roles[model], scenario=final_name) for model in final_models]).sort_values("role")
        fallback = fallback_selection(final_selected_map, features)
        if sum(model.startswith("mopex") for model in fallback) > 1:
            raise RuntimeError("fallback MOPEX redundancy gate failed")
        primary_text = "\n".join(f"{row.role}: {row.model}" for row in final_table.itertuples()) + "\n"
        fallback_text = "\n".join(fallback) + "\n"
    else:
        final_table = pd.DataFrame()
        fallback = []
        primary_text = "BLOCKED_NO_FINAL_PRIMARY_8\n"
        fallback_text = "BLOCKED_NO_FINAL_FALLBACK_6\n"
    write_csv(OUT / "OOB_PRIMARY_8_MODEL_TABLE_REVISED.csv", final_table)
    (OUT / "OOB_PRIMARY_8_MODELS_REVISED.txt").write_text(primary_text)
    (OUT / "OOB_FALLBACK_6_MODELS_REVISED.txt").write_text(fallback_text)

    original_models_struct = pd.read_csv(ORIGINAL_RESULT).model.tolist()
    structural_frames = [descriptors(original_models_struct, "ORIGINAL_PRIMARY_8")]
    for name, result in scenarios.items():
        if result.get("selected") is not None:
            structural_frames.append(descriptors(list(result["selected"].values()), f"{name}_LEGAL_CANDIDATE"))
    structural = pd.concat(structural_frames, ignore_index=True)
    write_csv(OUT / "OOB_STRUCTURAL_MODEL_DESCRIPTORS.csv", structural)
    structural_summary = coverage_summary(structural)
    write_csv(OUT / "OOB_STRUCTURAL_COVERAGE_QC.csv", structural_summary)
    (OUT / "OOB_STRUCTURAL_COVERAGE_QC.md").write_text("# Structural coverage QC\n\nAuxiliary registry/core descriptors only; not selection criteria.\n")
    write_json(OUT / "OOB_MOPEX_REVISION_PROVENANCE.json", {"candidate_pool_n": 36, "banned_scenario_A": ["mopex2", "mopex4"], "banned_scenario_B": ["mopex2", "mopex5"], "scenarios": {name: result for name, result in scenarios.items()}, "quality": quality_table.to_dict(orient="records"), "final_scenario": final_name, "primary_8": final_models, "fallback_6": fallback, "mopex_primary_count": sum(model.startswith("mopex") for model in final_models), "mopex_fallback_count": sum(model.startswith("mopex") for model in fallback), "manual_override": False, "oob_information_used": False, "training_started": False, "family_metadata_used": False, "G_median": thresholds["G_median"], "R_median": thresholds["R_median"], "K_joint_Q25": thresholds["K_joint_Q25"]})
    (OUT / "OOB_MODEL_SELECTION_MOPEX_REDUNDANCY_REVISION.md").write_text(render_report(scenarios, quality_table, final_name, final_table, fallback, thresholds, structural_summary))
    print("MOPEX revision complete", final_name, final_models, fallback)


if __name__ == "__main__":
    main()
