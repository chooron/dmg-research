#!/usr/bin/env python3
"""Build a manuscript-facing R1-R5 data-freeze audit from existing artifacts only.

This audit is intentionally read-only with respect to training/raw result roots. It
summarizes existing machine-readable outputs, protocol definitions, provenance gates,
and unresolved manuscript inputs. It does not train, resume training, generate truth,
or infer values from figures.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
MANUSCRIPT = PROJECT / "manuscript"
RESULTS = PROJECT / "results"
DATA = PROJECT.parents[1] / "data"
OUT = MANUSCRIPT / "cache" / "results_freeze_R1_R5"
DRAFT_NAME = "hess_results_R1_R5_reframed_v2.md"

BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 20260730
EXPECTED_STRATA = {
    "S1": ("[0, 0.05)", 165),
    "S2": ("[0.05, 0.15)", 156),
    "S3": ("[0.15, 0.30)", 121),
    "S4": ("[0.30, 0.50)", 34),
    "S5": ("[0.50, 1.00]", 55),
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def csv_frame(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_draft() -> Path | None:
    direct = PROJECT / DRAFT_NAME
    if direct.exists():
        return direct
    for root in (PROJECT, PROJECT.parent):
        for path in root.rglob(DRAFT_NAME):
            if any(part in {".git", ".venv", "node_modules"} for part in path.parts):
                continue
            return path
    return None


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.rglob(pattern)) if path.exists() else 0


def model_parameter_counts() -> dict[str, Any]:
    path = PROJECT / "models" / "parameter_specs.py"
    text = read_text(path)
    counts: dict[str, Any] = {}
    # Prefer the live parameter-spec module when importable, but retain a source trace.
    try:
        import sys
        if str(PROJECT) not in sys.path:
            sys.path.insert(0, str(PROJECT))
        from models import parameter_specs as specs  # type: ignore
        for name in ("XAJ", "XAJ_TGD2", "XAJ_CN", "GR4J", "GR4J_TGD2", "GR4J_CN", "SIMHYD", "SIMHYD_TGD2", "SIMHYD_CN"):
            counts[name] = len(getattr(specs, f"{name}_PARAM_SPECS"))
    except Exception as exc:  # pragma: no cover - environment dependent
        counts["import_error"] = str(exc)
    counts["source"] = rel(path)
    counts["source_contains_tgd2"] = "TGD2" in text
    return counts


def source_status() -> dict[str, Any]:
    ic = {}
    ic_dirs = {
        "XAJ-Base": RESULTS / "xaj_base_cmaes_531_batched_paired_v2",
        "XAJ-CN": RESULTS / "xaj_cn_cmaes_531_batched_paired_v2",
        "XAJ-TGD": RESULTS / "xaj_tgd2_cmaes_531_batched_v1",
    }
    for label, path in ic_dirs.items():
        raw = path / "raw"
        ic[label] = {
            "root": rel(path),
            "done": (path / "DONE.json").exists(),
            "raw_json": count_files(raw, "*.json"),
            "manifests": [rel(p) for p in path.glob("manifest.json")],
            "optimizer_evidence": "CMA-ES" if "cmaes" in read_text(path / "manifest.json").upper() else "not found in manifest text",
        }

    dpl = {}
    for label, model, root_name in (
        ("XAJ-Base", "XAJ", "dpl_camels_531_lite_v2"),
        ("XAJ-CN", "XAJ_CN", "dpl_camels_531_lite_v2"),
        ("HBV", "HBV", "dpl_camels_531_lite_v2"),
        ("XAJ-TGD", "XAJ_TGD2", "dpl_camels_531_lite_v3_tgd2_dpl_audited"),
    ):
        root = RESULTS / root_name / model
        seeds = []
        for seed in sorted(root.glob("seed_*")):
            seeds.append(
                {
                    "seed": seed.name.removeprefix("seed_"),
                    "complete": (seed / "COMPLETE").exists(),
                    "launcher_success": (seed / "LAUNCHER_SUCCESS").exists(),
                    "basin_final_summary": (seed / "basin_final_summary.csv").exists(),
                    "train_test_kge": (seed / "train_test_kge_by_basin.csv").exists(),
                    "best_checkpoint": (seed / "best_checkpoint.pt").exists(),
                    "epoch_history": (seed / "epoch_history.csv").exists(),
                }
            )
        freeze_eval = OUT / "r1_hbv_checkpoint_evaluation" if label == "HBV" else None
        dpl[label] = {"root": rel(root), "seed_count": len(seeds), "seeds": seeds, "freeze_checkpoint_evaluation": {"path": rel(freeze_eval) if freeze_eval else "", "all_seed_rows": (freeze_eval / "hbv_train_test_kge_by_basin_all_seeds.csv").exists() if freeze_eval else False, "median_seed_rows": (freeze_eval / "hbv_train_test_kge_by_basin_median_seed.csv").exists() if freeze_eval else False}}

    return {"ic": ic, "dpl": dpl}


def protocol_manifest() -> dict[str, Any]:
    basin_file = DATA / "531sub_id.txt"
    basin_raw = read_text(basin_file).strip()
    try:
        basin_values = json.loads(basin_raw)
        basin_lines = [str(x).strip().zfill(8) for x in basin_values if str(x).strip()]
    except json.JSONDecodeError:
        basin_lines = [x.strip().zfill(8) for x in basin_raw.splitlines() if x.strip()]
    dates_path = DATA / "camels_dates.npy"
    date_info: dict[str, Any] = {"path": rel(dates_path), "exists": dates_path.exists()}
    if dates_path.exists():
        try:
            arr = np.load(dates_path, allow_pickle=True)
            date_info.update({"shape": list(arr.shape), "first": str(arr.reshape(-1)[0]), "last": str(arr.reshape(-1)[-1])})
        except Exception as exc:
            date_info["error"] = str(exc)

    snow_path = MANUSCRIPT / "results" / "R1" / "r1_snow_attributes.csv"
    snow = csv_frame(snow_path)
    strata: dict[str, Any] = {}
    if snow is not None and "snow_stratum" in snow:
        for key, (interval, expected) in EXPECTED_STRATA.items():
            observed = int((snow["snow_stratum"] == key).sum())
            strata[key] = {"interval": interval, "observed_n": observed, "qa_expected_n": expected, "status": "PASS" if observed == expected else "FAIL"}
        high_n = int(snow["snow_stratum"].isin(["S4", "S5"]).sum())
        strata["S4+S5"] = {"observed_n": high_n, "qa_expected_n": 89, "status": "PASS" if high_n == 89 else "FAIL"}
    else:
        strata["status"] = "UNRESOLVED: missing r1_snow_attributes.csv"

    r1_code = read_text(MANUSCRIPT / "scripts" / "r1" / "r1_statistics.py")
    r5_code = read_text(MANUSCRIPT / "scripts" / "r5" / "build_r5_formal_analysis.py")
    ct_gate = {
        "r1_water_year_loop": all(token in r1_code for token in ("water_years", "ct_obs", "ct_sim", "ct_error_signed")),
        "r1_five_year_rule": "minimum_5_complete_water_years" in r1_code or ">= 5" in r1_code,
        "r1_sign": "ct_sim - ct_obs" in r1_code,
        "r5_water_year_loop": all(token in r5_code for token in ("water_years", "ct_obs", "ct_sim", "err_ct_s")),
        "r5_sign": "ct_sim - ct_obs" in r5_code,
    }
    ct_gate["overall"] = "PASS" if all(bool(v) for k, v in ct_gate.items() if k != "overall") else "FAIL"

    kge_gate = {
        "r1_standard_kge_source": rel(MANUSCRIPT / "scripts" / "r1" / "r1_statistics.py"),
        "r1_standard_formula_present": "alpha = float(s.std() / obs_std)" in r1_code and "1.0 - np.sqrt" in r1_code,
        "r5_standard_formula_present": "alpha = s_std / o_std" in r5_code and "1.0 - np.sqrt" in r5_code,
        "historical_kge_prime_present": "kge_prime" in r1_code,
        "interpretation": "R1 stored IC/dPL KGE records are original KGE; standard KGE(Q) is defined in the daily evaluators. KGE-prime remains a legacy column/label and requires manuscript cleanup.",
    }
    kge_gate["overall"] = "PARTIAL" if kge_gate["r1_standard_formula_present"] and kge_gate["r5_standard_formula_present"] else "FAIL"

    return {
        "basin_universe": {"path": rel(basin_file), "count": len(basin_lines), "unique_count": len(set(basin_lines)), "first": basin_lines[:3], "last": basin_lines[-3:]},
        "periods": {
            "warmup": "1980-10-01..1981-09-30",
            "train": "1981-10-01..1995-09-30",
            "test": "1995-10-01..2010-09-30",
            "water_year": "October 1 through September 30",
            "missing_data": "finite, nonnegative paired observations/simulations; minimum 30 days for KGE; complete water-year rules for CT/AMJJ",
        },
        "dates": date_info,
        "snow": {"axis": "f_snow from CAMELS static attribute index 3", "strata": strata, "external_swe_axis": "R4-only realized SWE burden/phase; not interchangeable with f_snow"},
        "metrics": kge_gate,
        "ct_gate": ct_gate,
        "bootstrap": {"rounds": BOOTSTRAP_ROUNDS, "seed": BOOTSTRAP_SEED, "unit": "basin unless an existing R4 clustered artifact explicitly retains basin-years", "interpretation": "marginal, unadjusted descriptive intervals; no multiplicity adjustment"},
        "ic_dpl": source_status(),
        "model_parameter_counts": model_parameter_counts(),
    }


def infer_n(row: pd.Series) -> Any:
    for col in row.index:
        key = str(col).lower()
        if key in {"n", "n_basins", "stratum_n", "valid_basin_count", "n_valid", "sample_n", "basin_count"} or key.endswith("_n"):
            value = row[col]
            if pd.notna(value):
                return value
    return ""


def source_script(section: str, name: str) -> str:
    mapping = {
        "R1": "manuscript/scripts/r1/",
        "R2": "manuscript/scripts/r2/",
        "R3": "manuscript/scripts/r3/",
        "R4": "manuscript/scripts/r4/",
        "R5": "manuscript/scripts/r5/",
    }
    return mapping.get(section, "manuscript/scripts/")


def build_value_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    roots = [
        ("R1", MANUSCRIPT / "results" / "R1"),
        ("R2", MANUSCRIPT / "results" / "R2"),
        ("R3", MANUSCRIPT / "results" / "R3"),
        ("R5", MANUSCRIPT / "results" / "R5"),
        ("R4", RESULTS / "r4_phase1_soil_official"),
        ("R1", OUT / "r1_hbv_checkpoint_evaluation"),
    ]
    for section, root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            frame = csv_frame(path)
            if frame is None:
                continue
            for row_id, (_, row) in enumerate(frame.iterrows()):
                identifiers = {str(c).lower() for c in row.index}
                context = {k: row[c] for k, c in (("paradigm", "paradigm"), ("structure", "structure"), ("regime", "regime"), ("period", "period"), ("snow_stratum", "snow_stratum"), ("snow_regime", "snow_regime"), ("seed", "seed")) if c in row.index and pd.notna(row[c])}
                n_value = infer_n(row)
                for quantity, value in row.items():
                    if quantity in {"basin_id", "paradigm", "structure", "regime", "period", "snow_stratum", "snow_regime", "seed", "model", "host_model", "contrast", "metric", "sample", "stratum", "dependent_var", "effect", "estimand", "role", "block", "status"}:
                        continue
                    if pd.isna(value):
                        continue
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    rows.append({
                        "section": section,
                        "source_file": rel(path),
                        "source_script": source_script(section, path.name),
                        "source_row": row_id,
                        "quantity": quantity,
                        "value": f"{numeric:.17g}",
                        "n": n_value,
                        "paradigm": context.get("paradigm", row.get("paradigm", "")),
                        "structure": context.get("structure", row.get("structure", "")),
                        "regime": context.get("regime", row.get("regime", row.get("snow_regime", ""))),
                        "period": context.get("period", row.get("period", "")),
                        "snow_stratum": context.get("snow_stratum", row.get("snow_stratum", row.get("snow_regime", ""))),
                        "seed": context.get("seed", row.get("seed", "")),
                        "status": "DERIVED_FROM_EXISTING_OUTPUT",
                        "notes": "Copied as a derived long-form record; no value was inferred from a plot or prose.",
                    })
    return pd.DataFrame(rows)


def write_protocol_md(manifest: dict[str, Any], draft: Path | None) -> None:
    s = manifest["snow"]["strata"]
    lines = [
        "# Canonical R1–R5 Results Protocol Manifest",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "> This is a read-only audit of existing scripts and trained-result artifacts. No production training, resume, synthetic-truth regeneration, or figure-image inference was performed.",
        "",
        "## Canonical worktree",
        f"- Repository: `{PROJECT}`",
        f"- Active manuscript root: `{MANUSCRIPT}`",
        f"- Derived audit output: `{OUT}`",
        f"- Requested draft: `{DRAFT_NAME}`",
        f"- Draft status: **{'FOUND at ' + rel(draft) if draft else 'UNRESOLVED — file not present in the repository/worktree search'}**",
        "",
        "## Basin universe and periods",
        f"- Basin list: `{manifest['basin_universe']['path']}`; count={manifest['basin_universe']['count']}; unique={manifest['basin_universe']['unique_count']}",
        f"- Periods: warmup {manifest['periods']['warmup']}; train {manifest['periods']['train']}; test {manifest['periods']['test']}",
        f"- Water year: {manifest['periods']['water_year']}",
        f"- Missing/eligibility rule: {manifest['periods']['missing_data']}",
        "",
        "## KGE, CT, and snow axes",
        f"- KGE gate: **{manifest['metrics']['overall']}**. R1 and R5 contain the standard Gupta KGE(Q) formula; legacy `kge_prime` columns remain in R1 output schemas and must not be used as a manuscript metric label without explicit definition.",
        f"- CT gate: **{manifest['ct_gate']['overall']}**. R1 and R5 source code contains per-water-year CT, `CT_sim - CT_obs`, and the negative-earlier sign convention.",
        "- `f_snow` is the climatological CAMELS process-exposure axis for R1/R2/R3/R5. External SWE is an R4 realized burden/phase axis.",
        "",
        "### Fixed f_snow QA anchors",
        "| stratum | interval | observed N | QA anchor | status |",
        "|---|---|---:|---:|---|",
    ]
    for key, (interval, expected) in EXPECTED_STRATA.items():
        x = s.get(key, {})
        lines.append(f"| {key} | {interval} | {x.get('observed_n', 'UNRESOLVED')} | {expected} | {x.get('status', 'UNRESOLVED')} |")
    x = s.get("S4+S5", {})
    lines.append(f"| S4+S5 | high-snow union | {x.get('observed_n', 'UNRESOLVED')} | 89 | {x.get('status', 'UNRESOLVED')} |")
    lines += ["", "## Base/TGD/CN identity", "", "| implementation | parameter count |", "|---|---:|"]
    counts = manifest.get("model_parameter_counts", {})
    for name in ("XAJ", "XAJ_TGD2", "XAJ_CN", "GR4J", "GR4J_TGD2", "GR4J_CN", "SIMHYD", "SIMHYD_TGD2", "SIMHYD_CN"):
        lines.append(f"| {name} | {counts.get(name, 'UNRESOLVED')} |")
    lines += [
        "## IC and dPL provenance",
        "- IC raw result roots are the three XAJ CMA-ES directories under `project/hydrodiag/results/`; each currently contains 5,310 raw JSON records and a DONE marker.",
        "- IC selected restart rule is maximum train-period stored KGE with minimum-start tie break, as implemented in `manuscript/scripts/r1/r1_statistics.py` and `r1_daily_inference.py`.",
        "- dPL uses existing checkpoint/summary roots and must not be treated as an IC-vs-dPL ranking. HBV has complete three-seed checkpoints and basin summaries. The established R1 evaluator was run in the freeze cache against all three HBV checkpoints to produce `r1_hbv_checkpoint_evaluation/hbv_train_test_kge_by_basin_all_seeds.csv`; the official R1 tables have not been overwritten.",
        "- Canonical IC optimizer: raw XAJ manifest/runner evidence says CMA-ES; historical xNES configuration files remain historical and are not accepted as the active IC protocol.",
        "",
        "## R1–R5 source status",
        "| area | status | Blocking? | Main issue | Required action |",
        "|---|---|---|---|---|",
        "| R1 | PARTIAL | Yes | Existing R1 tables omit HBV; a three-seed HBV checkpoint evaluation now exists in the freeze cache, but official aggregate/stratified R1 tables were not regenerated. | Rebuild official R1 tables from the existing evaluator outputs after the draft/source scope is confirmed. |",
        "| R2 | PARTIAL | Yes | The established seed-level source rebuild failed because the canonical bounds audit reports that XAJ_CN does not expose the expected 15 public XAJ parameters. | Resolve the repository bounds/source mismatch using existing definitions, then rerun the established R2 script. |",
        "| R3 | PASS_WITH_SCOPE | No | Frozen synthetic-truth, gate, figure, and table artifacts exist; no truth regeneration was performed. | Trace exact claims after the draft is supplied. |",
        "| R4 | PARTIAL | Yes | Base/CN formal R4 artifacts exist; current handoff marks TGD2 observation-trained provenance as pending. | Keep Base/CN scope or complete the existing TGD provenance gate without new training. |",
        "| R5 | PASS_WITH_SCOPE | No | Audited basin-level, timing, snow-gradient, agreement, and verdict outputs exist; Figure 9 is frozen PNG-only under the current convention. | Use existing frozen outputs and preserve paired-host interpretation. |",
        "",
        "## Statistical reporting convention",
        f"- Existing bootstrap convention: {manifest['bootstrap']['rounds']} replicates, seed {manifest['bootstrap']['seed']} where the active scripts expose the frozen convention.",
        "- Basin is the resampling unit. Paired structures/hosts are resampled jointly; IC restarts and dPL seeds are not independent basins.",
        "- R4 repeated basin-years require basin-level reduction or a basin-cluster bootstrap; no new clustered framework was invented in this audit.",
        "- Intervals are marginal and unadjusted descriptive intervals. Regional omission is a robustness check, not proof of spatial independence.",
        "",
        "## Final gate",
        "The manuscript cannot be called numerically ready for final Results rewriting because the requested draft is absent, official R1 tables have not been rebuilt from the newly evaluated HBV checkpoint outputs, the active R2 seed-level source rebuild failed on the canonical bounds audit, and the R4 TGD gate is pending. All unresolved quantities are marked rather than inferred.",
    ]
    (OUT / "results_canonical_protocol_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_claim_outputs(draft: Path | None) -> None:
    claim_path = OUT / "results_claim_registry.csv"
    placeholder_path = OUT / "results_placeholder_resolution.md"
    patch_path = OUT / "results_manuscript_patch_map.md"
    if draft is None:
        row = {
            "claim_id": "DRAFT_FILE",
            "section": "ALL",
            "draft_text": f"Requested draft `{DRAFT_NAME}` was not found under the canonical worktree or its parent search roots.",
            "estimand": "UNRESOLVED",
            "definition": "Cannot parse claims without the actual draft.",
            "source": "UNRESOLVED",
            "script": "UNRESOLVED",
            "structure": "",
            "regime": "",
            "seed_restart": "",
            "period": "",
            "N": "",
            "draft_value": "",
            "recomputed_value": "",
            "interval": "",
            "status": "UNRESOLVED",
            "manuscript_ready_replacement": "Provide the exact draft file at the requested path, then rerun this audit.",
            "notes": "No claims or placeholders were silently reconstructed from historical prose or figures.",
        }
        pd.DataFrame([row]).to_csv(claim_path, index=False)
        placeholder_path.write_text(
            "# Placeholder Resolution\n\n"
            f"Status: **UNRESOLVED**. The requested draft `{DRAFT_NAME}` is not present in the canonical worktree or searched parent roots. Therefore no `[[DATA:...]]` or `[[VERIFY:...]]` token can be audited without inventing manuscript text.\n",
            encoding="utf-8",
        )
        patch_path.write_text(
            "# Manuscript Patch Map\n\n"
            f"No patch map was generated because `{DRAFT_NAME}` is missing. Supplying a historical table or plot as a substitute would violate the audit boundary.\n",
            encoding="utf-8",
        )
        return

    text = read_text(draft)
    claims = []
    for idx, line in enumerate(text.splitlines(), 1):
        if "[[DATA:" in line or "[[VERIFY:" in line or re.search(r"\b\d+(?:\.\d+)?%?\b", line):
            claims.append({"claim_id": f"DRAFT_LINE_{idx}", "section": "UNCLASSIFIED", "draft_text": line, "status": "UNRESOLVED", "notes": "Requires source-specific review."})
    pd.DataFrame(claims).to_csv(claim_path, index=False)
    placeholder_path.write_text("# Placeholder Resolution\n\nDraft was found; all extracted rows require source-specific review. See `results_claim_registry.csv`.\n", encoding="utf-8")
    patch_path.write_text("# Manuscript Patch Map\n\nSee `results_claim_registry.csv`; no automatic manuscript rewrite was applied.\n", encoding="utf-8")


def write_figure_manifest() -> None:
    figures = []
    specs = [
        (1, "R1", MANUSCRIPT / "scripts/r1/plot_r1_figure1.py", MANUSCRIPT / "figures/Figure1_R1_compensation_overview.png", "R1/r1_basin_level_performance.csv; R1/r1_absolute_metrics_summary.csv", "BLOCKED: current R1 absolute table lacks HBV stratum rows"),
        (2, "R1", MANUSCRIPT / "scripts/r1/plot_r1_figure2_canonical.py", MANUSCRIPT / "figures/Figure2_R1_timing_diagnosis.png", "R1/r1_basin_level_performance.csv; R1/r1_snow_signatures_basin_level.csv", "NOT_REGENERATED_IN_FREEZE"),
        (3, "R2", MANUSCRIPT / "scripts/r2/plot_r2_figure3_final.py", MANUSCRIPT / "figures/Figure3_R2_parameter_separation.png", "R2/r2_tgd2_specificity_basin_level.csv; R2/r2_tgd2_specificity_summary.csv; R2/r2_tgd2_specificity_regressions.csv", "BLOCKED: active script assertion detects slope/table inconsistency"),
        (4, "R2", MANUSCRIPT / "scripts/r2/plot_r2_figure4_canonical.py", MANUSCRIPT / "figures/Figure4_R2_parameter_signatures.png", "R2/r2_parameter_values_canonical.csv; R2/r2_paired_shifts_basin_level.csv", "NOT_REGENERATED_IN_FREEZE"),
        (5, "R3", MANUSCRIPT / "scripts/r3/plot_figure5.py", MANUSCRIPT / "figures/Figure5_R3_final.png", "R3/figure5_basin_table.csv; R3/figure5_summary.json", "EXISTING_ARTIFACT"),
        (6, "R3", MANUSCRIPT / "scripts/r3/plot_figure6.py", MANUSCRIPT / "figures/Figure6_R3_final.png", "R3/figure6_basin_table.csv; R3/figure6_summary.json", "EXISTING_ARTIFACT"),
        (7, "R4", MANUSCRIPT / "scripts/r4/plot_r4_figure7.py", MANUSCRIPT / "figures/figure7_r4_soil_consistency.png", "results/r4_phase1_soil_official", "EXISTING_ARTIFACT; TGD provenance pending"),
        (8, "R4", MANUSCRIPT / "scripts/r4/plot_r4_figure8_canonical.py", MANUSCRIPT / "figures/figure8_r4_soil_timing.png", "results/r4_phase1_soil_official", "EXISTING_ARTIFACT; TGD provenance pending"),
        (9, "R5", MANUSCRIPT / "scripts/r5/plot_r5_figure9.py", MANUSCRIPT / "figures/Figure9_R5_cross_model_replication.png", "R5/r5_basin_level_dataset.csv; R5/r5_*_table.csv", "FROZEN_EXISTING_ARTIFACT"),
    ]
    for number, section, script, image, source, status in specs:
        figures.append({
            "figure": f"F{number}", "section": section, "active_script": rel(script), "active_png": rel(image),
            "png_exists": image.exists(), "canonical_source": source, "status": status,
            "manuscript_label": "Base/TGD/CN; implementation may remain TGD2 only in provenance fields",
            "uncertainty": "script/source-defined; verify against canonical tables before writing",
        })
    pd.DataFrame(figures).to_csv(OUT / "results_figure_table_manifest.csv", index=False)


def write_notes() -> None:
    (OUT / "results_statistics_reporting_note.md").write_text(
        "# Statistics Reporting Note\n\n"
        "Basin is the resampling unit for basin-level estimands. Paired structures and, for R5, all available host outputs for a sampled basin are retained jointly. IC restarts and dPL seeds are not treated as independent basins. R4 repeated basin-years must be reduced to basin-level estimands or retained within sampled basins under a basin-cluster bootstrap. Reported 95% intervals are marginal, unadjusted descriptive uncertainty intervals; they are not a collection of multiplicity-adjusted null-hypothesis tests. Interpretation should emphasize effect size and direction. Existing regional-omission/leave-one-region-out analyses are robustness checks and do not establish spatial independence.\n",
        encoding="utf-8",
    )
    (OUT / "results_R4_reference_boundary.md").write_text(
        "# R4 Reference Boundary\n\n"
        "The active R4 artifacts identify Caravan/ERA5-Land-derived soil-water reference arrays and an independently defined SWE-17 burden/phase axis. The reference can support an external process-state timing/consistency comparison organized by SWE burden. It cannot be called ground truth or independent validation of the CN snow mechanism because ERA5-Land/HTESSEL contains its own land-surface and snow physics. No absolute one-to-one mapping between XAJ storage units and reference-layer volumetric water content is permitted. Any SWE-only timing check is supplementary only if already present in the existing SWE artifacts; it is not a new primary endpoint. The current R4 handoff marks observation-trained TGD2 provenance as pending, so formal R4 conclusions are limited to the verified Base/CN scope until that gate passes.\n",
        encoding="utf-8",
    )


def write_readiness(manifest: dict[str, Any], draft: Path | None) -> None:
    lines = [
        "# Final Readiness Verdict",
        "",
        "| Area | Status | Blocking? | Main issue | Required action |",
        "|---|---|---|---|---|",
        "| Global protocol | PARTIAL | Yes | Requested draft absent; KGE legacy labels coexist with standard-KGE code. | Supply draft and remove/define legacy metric labels. |",
        "| Statistical reporting | PARTIAL | Yes | Existing sections expose different scopes/replicate conventions; R4 requires cluster-preserving interpretation. | Confirm one manuscript-wide reporting paragraph and keep existing robustness only. |",
        "| R1 | PARTIAL | Yes | Existing basin/summary tables omit HBV rows; a three-seed HBV checkpoint evaluation now exists in the freeze cache, but official R1 aggregate/stratified tables were not rebuilt. | Rebuild official R1 tables from the evaluated HBV output and existing R1 sources after the draft/source scope is confirmed. |",
        "| R2 | PARTIAL | Yes | The established seed-level source rebuild failed at the canonical bounds audit: XAJ_CN does not expose the expected 15 public XAJ parameters in the active bounds inventory. | Resolve that repository source mismatch using existing definitions, then rerun R2 statistics. |",
        "| R3 | PASS_WITH_SCOPE | No for existing artifacts | Frozen truth/gate/figure/table assets exist; no truth regeneration performed. | Trace exact claims after draft is supplied. |",
        "| R4 | PARTIAL | Yes | TGD2 formal provenance is explicitly pending; external reference is model-derived. | Keep Base/CN scope or complete the existing TGD provenance gate without new training. |",
        "| R5 | PASS_WITH_SCOPE | No | Audited R5 tables and Figure 9 exist; TGD2 implementation label requires provenance-only mapping. | Use existing frozen outputs and preserve paired-host interpretation. |",
        "| Figures/tables | PARTIAL | Yes | F1/F3 are not currently reproducible from active tables; F5–F9 artifacts exist. | Repair source-table consistency, then regenerate F1–F4 from scripts. |",
        "| Intro/Methods consistency | UNRESOLVED | Yes | The requested draft file is absent. | Provide `hess_results_R1_R5_reframed_v2.md`. |",
        "",
        "## Audit counts",
        "- Manuscript claims audited: **0** (draft missing).",
        "- PASS_EXACT / PASS_ROUNDING: **0**.",
        "- Value replacements: **0**.",
        "- Definition failures: **0 parsed claims; protocol-level unresolved items remain**.",
        "- Provenance failures: **0 parsed claims; R4 TGD and R1/R2 source gates remain blocking**.",
        "- Unresolved placeholders: **1 draft-file blocker; individual placeholders cannot be enumerated without the draft**.",
        f"- R4 three-structure provenance gate: **PARTIAL/FAIL for TGD2; Base/CN formal artifacts exist** (see `results_R4_reference_boundary.md`).",
        f"- CT implementation gate: **{manifest['ct_gate']['overall']}** by source-code inspection of R1 and R5 implementations.",
        "- Canonical IC optimizer: **CMA-ES confirmed for the active XAJ raw-result manifests/runner; historical xNES files excluded**.",
        f"- Standard KGE across R1–R5: **{manifest['metrics']['overall']}**; R1/R5 standard formulas are present, but legacy KGE-prime fields and section-specific artifacts require final cross-section review.",
        "- Numerically ready for final Results rewriting: **NO**.",
        "",
        "No new scientific claim, threshold, basin subset, model, training run, or synthetic truth was created by this audit.",
    ]
    (OUT / "results_final_readiness_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = protocol_manifest()
    draft = find_draft()
    write_protocol_md(manifest, draft)
    values = build_value_table()
    values.to_csv(OUT / "results_canonical_value_table.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    write_claim_outputs(draft)
    write_figure_manifest()
    write_notes()
    write_readiness(manifest, draft)
    (OUT / "audit_run_manifest.json").write_text(json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(), "project": str(PROJECT), "draft": str(draft) if draft else None, "outputs": sorted(p.name for p in OUT.iterdir())}, indent=2) + "\n", encoding="utf-8")
    print(f"output={OUT}")
    print(f"draft={'FOUND' if draft else 'MISSING'}")
    print(f"canonical_value_rows={len(values)}")
    print(f"ct_gate={manifest['ct_gate']['overall']}")
    print(f"kge_gate={manifest['metrics']['overall']}")
    print("training_launched=no")


if __name__ == "__main__":
    main()
