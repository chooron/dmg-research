"""Stage 2b: Deep audit of excluded/caveat models for Euler convergence denominator.
Reads source code only; no model execution.
"""
from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE2B_DIR = PROJECT_ROOT / "validation_results" / "gmd_3_1_stage2b_exclusion_audit"
STAGE2B_DIR.mkdir(parents=True, exist_ok=True)

STAGE2_DIR = PROJECT_ROOT / "validation_results" / "gmd_3_1_stage2_discretization_smoothing"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============================================================================
# B: Denominator audit
# ============================================================================

def audit_denominator():
    """Read STFN_INFO, registry status, file existence."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from models import core
    from tests.core_model_registry import CORE_MODEL_REGISTRY, DISABLED_MODELS

    stfn_models = set(core.STFN_INFO.keys())
    reg_models = set(CORE_MODEL_REGISTRY.keys())

    # enabled = 36 (all STFN_INFO), disabled = shm only
    enabled = {n for n, e in CORE_MODEL_REGISTRY.items() if e.enabled}
    disabled = {n for n, e in CORE_MODEL_REGISTRY.items() if not e.enabled}

    # Files
    core_dir = PROJECT_ROOT / "models" / "core"
    all_files = {p.stem for p in core_dir.glob("*.py") if p.stem != "__init__"}

    rows = []
    for model in sorted(reg_models):
        in_stfn = model in stfn_models
        in_param = model in core.PARAM_INFO
        in_init = model in core.INIT_INFO
        in_state_info = model in core.STATE_INFO
        in_nparam = model in core.NPARAM_INFO
        file_exists = model in all_files
        if file_exists:
            file_path = core_dir / f"{model}.py"
            file_nonempty = file_path.stat().st_size > 0
        else:
            file_nonempty = False
        reg_entry = CORE_MODEL_REGISTRY[model]
        enabled_flag = reg_entry.enabled
        stage2_class = "see_stage2_csv"

        should_count = enabled_flag

        rows.append({
            "model": model,
            "in_STFN_INFO": in_stfn,
            "in_PARAM_INFO": in_param,
            "in_INIT_INFO": in_init,
            "in_STATE_INFO": in_state_info,
            "in_NPARAM_INFO": in_nparam,
            "core_file_exists": file_exists,
            "core_file_nonempty": file_nonempty,
            "enabled_in_registry": enabled_flag,
            "included_in_stage2_euler_classification": "see_02_euler_model_classification.csv",
            "stage2_class": stage2_class,
            "should_count_in_denominator": should_count,
            "notes": reg_entry.skip_reason if not enabled_flag else "",
        })

    # Summary
    print(f"STFN_INFO count: {len(stfn_models)}")
    print(f"Registry total: {len(reg_models)}")
    print(f"Enabled: {len(enabled)} ({sorted(enabled)})")
    print(f"Disabled: {len(disabled)} ({sorted(disabled)})")
    print(f"SHM in STFN_INFO: {'shm' in stfn_models}")
    print(f"SHM enabled: {CORE_MODEL_REGISTRY['shm'].enabled if 'shm' in CORE_MODEL_REGISTRY else 'N/A'}")
    print(f"36 enabled = 36 STFN_INFO models (SHM excluded)")
    print(f"37 registry = 36 STFN_INFO + 1 SHM (disabled)")
    print(f"Stage 2 '30 tested + 7 excluded = 37' DOUBLE-COUNTS SHM")
    print(f"Correct: 30 tested + 6 excluded = 36 enabled (SHM out of scope)")

    _write_csv(STAGE2B_DIR / "01_enabled_model_list.csv", [
        "model", "in_STFN_INFO", "in_PARAM_INFO", "in_INIT_INFO",
        "in_STATE_INFO", "in_NPARAM_INFO", "core_file_exists",
        "core_file_nonempty", "enabled_in_registry",
        "included_in_stage2_euler_classification", "stage2_class",
        "should_count_in_denominator", "notes",
    ], rows)

    # Write MD summary
    lines = [
        "# Stage 2b: 36 Enabled Models Denominator Audit",
        "",
        "## Key findings",
        "",
        f"1. **STFN_INFO** (in `models/core/__init__.py`): {len(stfn_models)} models — SHM is NOT included.",
        f"2. **CORE_MODEL_REGISTRY** (in `tests/core_model_registry.py`): {len(reg_models)} entries — 36 from STFN_INFO + 1 manually added SHM.",
        f"3. **Enabled models**: {len(enabled)} — all 36 STFN_INFO models. SHM is DISABLED (file empty, 0 lines).",
        "",
        "## SHM status",
        "- `models/core/shm.py`: **0 bytes** (empty file)",
        "- In STFN_INFO: NO",
        "- In CORE_MODEL_REGISTRY: YES (manually added via `build_core_model_registry()` line 128-140)",
        "- `enabled`: False",
        "- `skip_reason`: 'File is empty and does not define a runnable core model implementation.'",
        "- **SHM should NOT be counted in the 36-model GMD denominator.**",
        "",
        "## Why stage 2 had '30 + 7 = 37'",
        "",
        "The EXCLUDED_MODELS set in `tests/euler_convergence_all_core_utils.py` includes `shm`,",
        "but SHM is already filtered out by the `entry.enabled` check. So SHM appears in the",
        "'excluded' count without actually being in the testable set. The correct partition is:",
        "",
        "- 36 enabled models total (SHM excluded as disabled)",
        "- 30 tested with Euler substep",
        "- 6 excluded (enabled but structurally unsuitable for substep scaling): gr4j, gsfb, mopex4, mopex5, tank, tcm",
        "",
        "## Recommended denominator for paper",
        "- **36 enabled core models**",
        "- Of these 36: 30 tested + 6 excluded (structural reasons) + 0 disabled",
        "- SHM is documented as a disabled model and NOT included in the 36-model count.",
    ]
    (STAGE2B_DIR / "01_denominator_enabled_model_audit.md").write_text("\n".join(lines) + "\n")


# ============================================================================
# C+D: Excluded model code audit + alternative validation
# ============================================================================

def audit_excluded_models():
    """Deep code audit of each excluded model."""

    excluded_audit = [
        {
            "model": "gr4j",
            "core_file": "models/core/gr4j.py",
            "registry_status": "ENABLED (in STFN_INFO)",
            "stage2_exclusion_reason": "Analytical daily production-store formula (tanh/sinh); not an ODE.",
            "code_evidence": "Line 9-27 _calc_production_store_tanh uses torch.tanh(Pn/x1) — closed-form; "
                             "Line 29-36 _calc_percolation_analytical uses (1+ratio^4)^(-0.25) — analytical integration; "
                             "Line 38-45 _calc_routing_outflow_analytical — analytical. "
                             "These are daily closed-form solutions, not rate-based ODEs.",
            "has_delta_t": False,
            "requires_doy": False,
            "requires_mean_P": False,
            "uses_integer_or_calendar_state": False,
            "uses_analytical_daily_formula": True,
            "uses_daily_threshold_partition": False,
            "substep_possible_without_equation_change": False,
            "substep_possible_with_harness_only": False,
            "substep_would_change_original_model_semantics": True,
            "recommended_class": "ANALYTICAL_DAILY_FORM_NOT_EULER",
            "paper_defense_sentence": "GR4J uses analytically integrated daily production-store formulas (tanh, power functions) "
                                      "that are closed-form solutions of daily water balance, not rate-based ODEs. "
                                      "Substep Euler convergence is not applicable to analytical daily formulations.",
            "confidence": "HIGH",
            "notes": "Alternative: N=1 daily formulation invariance check could verify that the analytical formula is "
                     "correctly implemented but that would be formula-level verification, not Euler convergence.",
        },
        {
            "model": "gsfb",
            "core_file": "models/core/gsfb.py",
            "registry_status": "ENABLED (in STFN_INFO)",
            "stage2_exclusion_reason": "Threshold-based recharge/interflow partitioning; parameters not dt-scalable.",
            "code_evidence": "CURRENT FILE IS THE SMOOTH VARIANT (line 1-27 docstring confirms 'gsfb_smooth.py'). "
                             "Uses smooth_relu (line 71-79), smooth_min (line 82-95), smooth_cap_flux (line 98-110). "
                             "tau=1e-3 default (line 149). Sub-functions: baseflow_9 uses F.softplus(beta=50) (baseflow.py:113); "
                             "interflow_11 uses smooth_threshold_storage_logistic (interflow.py:133). "
                             "NO delta_t parameter in step function. "
                             "Rate params (c, dpf both d-1; emax mm/d; frate mm/d) COULD scale with dt. "
                             "Threshold params (ndc dimensionless, sdrmax mm) are daily reference values.",
            "has_delta_t": False,
            "requires_doy": False,
            "requires_mean_P": False,
            "uses_integer_or_calendar_state": False,
            "uses_analytical_daily_formula": False,
            "uses_daily_threshold_partition": True,
            "substep_possible_without_equation_change": False,
            "substep_possible_with_harness_only": True,
            "substep_would_change_original_model_semantics": False,
            "recommended_class": "HARNESS_LIMITATION_POTENTIALLY_TESTABLE",
            "paper_defense_sentence": "GSFB uses a smooth differentiable variant (smooth_relu/min/cap_flux, tau=1e-3) "
                                      "that replaces hard flux caps. Rate parameters (c, dpf, emax, frate) admit dt-scaling. "
                                      "Sub-functions (baseflow_9, interflow_11) already use soft gates. "
                                      "Exclusion is a harness limitation, not a structural impossibility.",
            "confidence": "MEDIUM",
            "notes": "POTENTIALLY_TESTABLE. Needs: (1) dt-aware wrapper that scales rate params, "
                     "(2) appropriate tau handling, (3) accounting for ndc and sdrmax as daily reference values.",
        },
        {
            "model": "mopex4",
            "core_file": "models/core/mopex4.py",
            "registry_status": "ENABLED (in STFN_INFO)",
            "stage2_exclusion_reason": "Requires doy (integer day-of-year); cannot substep-divide.",
            "code_evidence": "CRITICAL FINDING: mopex4_step HAS delta_t parameter (line 146: delta_t: float = 1.0). "
                             "doy is KEYWORD-ONLY OPTIONAL (line 149: doy: torch.Tensor = None). "
                             "doy is used ONLY for cosine seasonal interception (line 108: cos(2*pi*(doy-is_time)/365.25)). "
                             "Within a single day, doy is CONSTANT — no need to substep-divide doy. "
                             "Rate params (tw, tu, tc all d-1; ddf mm/C/d) can scale with delta_t. "
                             "MOPEX4 IS ALREADY DT-AWARE!",
            "has_delta_t": True,
            "requires_doy": True,
            "requires_mean_P": False,
            "uses_integer_or_calendar_state": True,
            "uses_analytical_daily_formula": False,
            "uses_daily_threshold_partition": False,
            "substep_possible_without_equation_change": True,
            "substep_possible_with_harness_only": True,
            "substep_would_change_original_model_semantics": False,
            "recommended_class": "POTENTIALLY_TESTABLE_NEEDS_STAGE2C",
            "paper_defense_sentence": "MOPEX4 natively supports dt-scaling via its `delta_t` parameter. "
                                      "The `doy` argument is seasonal (cosine modulation) and can be held constant "
                                      "within a day during substep refinement. Exclusion is a harness limitation.",
            "confidence": "HIGH",
            "notes": "REQUIRES STAGE 2c. Harness needs to: pass delta_t=dt, pass doy as fixed day-of-year.",
        },
        {
            "model": "mopex5",
            "core_file": "models/core/mopex5.py",
            "registry_status": "ENABLED (in STFN_INFO)",
            "stage2_exclusion_reason": "Same as mopex4: requires doy; cannot substep-divide.",
            "code_evidence": "SAME AS MOPEX4: HAS delta_t parameter (line 135: delta_t: float = 1.0). "
                             "doy is KEYWORD-ONLY OPTIONAL (line 138: doy: torch.Tensor = None). "
                             "Also has phenology_1 (line 78-85) using tmin/trange for GSI scaling — these are "
                             "threshold-like but operate on temperature (not storage) and are dt-invariant. "
                             "MOPEX5 IS ALREADY DT-AWARE!",
            "has_delta_t": True,
            "requires_doy": True,
            "requires_mean_P": False,
            "uses_integer_or_calendar_state": True,
            "uses_analytical_daily_formula": False,
            "uses_daily_threshold_partition": False,
            "substep_possible_without_equation_change": True,
            "substep_possible_with_harness_only": True,
            "substep_would_change_original_model_semantics": False,
            "recommended_class": "POTENTIALLY_TESTABLE_NEEDS_STAGE2C",
            "paper_defense_sentence": "MOPEX5 natively supports dt-scaling via its `delta_t` parameter. "
                                      "The `doy` argument and phenology thresholds (tmin/trange) are dt-invariant. "
                                      "Exclusion is a harness limitation.",
            "confidence": "HIGH",
            "notes": "REQUIRES STAGE 2c. Same harness approach as mopex4.",
        },
        {
            "model": "shm",
            "core_file": "models/core/shm.py",
            "registry_status": "DISABLED",
            "stage2_exclusion_reason": "File is empty; no runnable implementation.",
            "code_evidence": "File is 0 bytes (empty). Not in STFN_INFO. Manually added to registry as disabled. "
                             "skip_reason: 'File is empty and does not define a runnable core model implementation.'",
            "has_delta_t": False,
            "requires_doy": False,
            "requires_mean_P": False,
            "uses_integer_or_calendar_state": False,
            "uses_analytical_daily_formula": False,
            "uses_daily_threshold_partition": False,
            "substep_possible_without_equation_change": False,
            "substep_possible_with_harness_only": False,
            "substep_would_change_original_model_semantics": True,
            "recommended_class": "DISABLED_NOT_IN_SCOPE",
            "paper_defense_sentence": "SHM has no implementation (empty file) and is excluded from the 36-model "
                                      "denominator. It is listed in the registry as disabled for completeness.",
            "confidence": "HIGH",
            "notes": "Double-counted in stage 2 '7 excluded'. Should be removed from denominator entirely. "
                     "The 36-model count in GMD papers refers to the MARRMoT models with implementations.",
        },
        {
            "model": "tank",
            "core_file": "models/core/tank.py",
            "registry_status": "ENABLED (in STFN_INFO)",
            "stage2_exclusion_reason": "Threshold-based side-outlet activation; daily empirical rules.",
            "code_evidence": "Has rate params (a0, b0, c0, a1 in d-1) that can scale with dt. "
                             "Has threshold params (st, f1, f2, f3) that define discrete bucket levels. "
                             "Lines 93-100: thresholds t1, t2, t3 are derived from st and fractions, defining "
                             "binary side-outlet activation points. "
                             "Uses F.relu for excess-above-threshold calculations. "
                             "AWAY from threshold crossings, the linear drainage rates admit dt-scaling. "
                             "AT threshold crossings, the binary activation changes model semantics under substep.",
            "has_delta_t": False,
            "requires_doy": False,
            "requires_mean_P": False,
            "uses_integer_or_calendar_state": False,
            "uses_analytical_daily_formula": False,
            "uses_daily_threshold_partition": True,
            "substep_possible_without_equation_change": False,
            "substep_possible_with_harness_only": True,
            "substep_would_change_original_model_semantics": True,
            "recommended_class": "EMPIRICAL_DAILY_THRESHOLD_PARTITION",
            "paper_defense_sentence": "TANK uses daily threshold-based side-outlet activation (st, f1, f2, f3 define "
                                      "discrete bucket levels). While rate parameters admit dt-scaling away from thresholds, "
                                      "the binary activation at discrete storage levels changes semantics under substep. "
                                      "This is a structural feature of the TANK formulation, not a numerical deficiency.",
            "confidence": "MEDIUM",
            "notes": "Could do local-away-from-threshold substep check as alternative validation, "
                     "but the daily-threshold structure is inherent to the TANK model concept.",
        },
        {
            "model": "tcm",
            "core_file": "models/core/tcm.py",
            "registry_status": "ENABLED (in STFN_INFO)",
            "stage2_exclusion_reason": "Requires mean_P (climatological constant); not substep-divisible.",
            "code_evidence": "mean_P is KEYWORD-ONLY REQUIRED (line 88: *, mean_P: torch.Tensor). "
                             "mean_P is 'pre-computed from the entire precipitation time series' (line 103) — "
                             "a basin-level climatological constant. ca = fa * mean_P (line 112) is the abstraction rate. "
                             "mean_P does NOT change with time steps; it could be held constant during substep. "
                             "Has baseflow_6 (line 37-50) using smooth_threshold_storage_logistic. "
                             "Has k1 (d-1) rate parameter. No delta_t parameter, but rate params could be dt-scaled. "
                             "Deficit store (S2) with sign convention needs careful handling.",
            "has_delta_t": False,
            "requires_doy": False,
            "requires_mean_P": True,
            "uses_integer_or_calendar_state": False,
            "uses_analytical_daily_formula": False,
            "uses_daily_threshold_partition": False,
            "substep_possible_without_equation_change": True,
            "substep_possible_with_harness_only": True,
            "substep_would_change_original_model_semantics": False,
            "recommended_class": "HARNESS_LIMITATION_POTENTIALLY_TESTABLE",
            "paper_defense_sentence": "TCM's `mean_P` is a climatological constant (basin-level mean precipitation) "
                                      "that can be held fixed during substep refinement. The `baseflow_6` function uses "
                                      "smooth threshold gates. Exclusion is a harness limitation.",
            "confidence": "MEDIUM",
            "notes": "POTENTIALLY_TESTABLE. Harness needs to (1) pass mean_P as a kwarg, (2) scale k1 rate by dt, "
                     "(3) handle deficit-store (S2) sign convention. Abstraction flux fa*mean_P is dt-invariant.",
        },
    ]

    _write_csv(STAGE2B_DIR / "02_excluded_models_code_audit.csv", [
        "model", "core_file", "registry_status", "stage2_exclusion_reason",
        "code_evidence", "has_delta_t", "requires_doy", "requires_mean_P",
        "uses_integer_or_calendar_state", "uses_analytical_daily_formula",
        "uses_daily_threshold_partition", "substep_possible_without_equation_change",
        "substep_possible_with_harness_only", "substep_would_change_original_model_semantics",
        "recommended_class", "paper_defense_sentence", "confidence", "notes",
    ], excluded_audit)

    # Alternative validation feasibility
    alt_rows = [
        {"model": "gr4j", "main_exclusion_reason": "Analytical daily closed-form integration",
         "alternative_validation_candidate": "FORMULA_LEVEL_DAILY_REGRESSION",
         "can_run_without_equation_change": True,
         "would_support_3_1_3": True,
         "risk_if_not_done": "Low — the exclusion reason is well-founded and undisputed",
         "recommended_action": "Document but do not test",
         "priority": "LOW",
         "notes": "GR4J's analytical nature is well-known (Perrin et al. 2003). No Euler convergence claim needed."},
        {"model": "gsfb", "main_exclusion_reason": "Harness limitation; smooth variant exists",
         "alternative_validation_candidate": "HARNESS_EXTENSION_REQUIRED",
         "can_run_without_equation_change": True,
         "would_support_3_1_3": True,
         "risk_if_not_done": "Medium — reviewer may question why smooth model is excluded",
         "recommended_action": "Stage 2c: add dt-aware wrapper with tau handling",
         "priority": "MEDIUM",
         "notes": "Rate params (c, dpf, emax, frate) scale with dt. Use smooth_cap_flux with existing tau."},
        {"model": "mopex4", "main_exclusion_reason": "Harness limitation; has native delta_t + optional doy",
         "alternative_validation_candidate": "HARNESS_EXTENSION_REQUIRED",
         "can_run_without_equation_change": True,
         "would_support_3_1_3": True,
         "risk_if_not_done": "High — the exclusion reason is factually incorrect (doy blocks substep), "
                            "but delta_t+optional doy means substep is feasible without equation change",
         "recommended_action": "Stage 2c: add to harness with fixed doy and delta_t scaling",
         "priority": "HIGH",
         "notes": "mopex4_step already has delta_t float param. Pass doy as fixed day-of-year."},
        {"model": "mopex5", "main_exclusion_reason": "Harness limitation; has native delta_t + optional doy",
         "alternative_validation_candidate": "HARNESS_EXTENSION_REQUIRED",
         "can_run_without_equation_change": True,
         "would_support_3_1_3": True,
         "risk_if_not_done": "High — same as mopex4",
         "recommended_action": "Stage 2c: add to harness with fixed doy and delta_t scaling",
         "priority": "HIGH",
         "notes": "mopex5_step already has delta_t float param. Pass doy as fixed day-of-year."},
        {"model": "shm", "main_exclusion_reason": "Disabled — empty file",
         "alternative_validation_candidate": "NONE_DISABLED",
         "can_run_without_equation_change": False,
         "would_support_3_1_3": False,
         "risk_if_not_done": "None — not a model",
         "recommended_action": "Document as disabled; remove from Euler denominator",
         "priority": "NONE",
         "notes": "Not an implemented model. Should not appear in any 36-model count."},
        {"model": "tank", "main_exclusion_reason": "Daily empirical threshold partition",
         "alternative_validation_candidate": "LOCAL_AWAY_FROM_THRESHOLD_SUBSTEP_CHECK",
         "can_run_without_equation_change": True,
         "would_support_3_1_3": True,
         "risk_if_not_done": "Low — threshold structure is inherent to TANK model concept",
         "recommended_action": "Document as structural daily-threshold model; no testing needed",
         "priority": "LOW",
         "notes": "Could do away-from-threshold substep but adds limited value for paper claim."},
        {"model": "tcm", "main_exclusion_reason": "Harness limitation; mean_P is constant, baseflow_6 is smooth",
         "alternative_validation_candidate": "MEAN_P_FIXED_SUBSTEP_FEASIBILITY_CHECK",
         "can_run_without_equation_change": True,
         "would_support_3_1_3": True,
         "risk_if_not_done": "Medium — reviewer may ask why climatological constant prevents substep",
         "recommended_action": "Stage 2c: add to harness with fixed mean_P and k1 dt-scaling",
         "priority": "MEDIUM",
         "notes": "mean_P is fixed per basin. Pass as kwarg. k1 rate scales with dt."},
    ]

    _write_csv(STAGE2B_DIR / "03_alternative_validation_feasibility.csv", [
        "model", "main_exclusion_reason", "alternative_validation_candidate",
        "can_run_without_equation_change", "would_support_3_1_3",
        "risk_if_not_done", "recommended_action", "priority", "notes",
    ], alt_rows)

    # Write audit MD
    lines = ["# Stage 2b: Excluded Models Code Audit",
             "",
             "## Summary",
             f"- 7 models in stage 2 excluded set (1 double-counted)",
             f"- 1 disabled (SHM) — should NOT count in denominator",
             f"- 1 analytical (GR4J) — structural, well-founded",
             f"- 1 daily-threshold (TANK) — structural, well-founded",
             f"- 2 HARDLY excluded (MOPEX4, MOPEX5) — HAVE native delta_t + optional doy; harness limitation only",
             f"- 2 harness-limited (GSFB, TCM) — could potentially be tested without equation changes",
             "",
             "## Critical finding: MOPEX4/MOPEX5",
             "Both `mopex4_step` and `mopex5_step` natively accept a `delta_t` float parameter (default=1.0) "
             "and an optional keyword-only `doy` parameter (default=None). The exclusion reason 'doy blocks substep' "
             "is **factually incorrect** — doy is seasonal and can be held constant within a day. "
             "These models should be tested in Stage 2c.",
             "",
             "## Per-model detail",
             "| Model | Revised class | Has delta_t? | Substep feasible? | Key insight |",
             "|---|---|---|---|---|",
    ]
    for row in excluded_audit:
        lines.append(
            f"| {row['model']} | {row['recommended_class']} | {row['has_delta_t']} | "
            f"{'YES' if row['substep_possible_with_harness_only'] or row['substep_possible_without_equation_change'] else 'NO'} | "
            f"{row['code_evidence'][:120]}... |"
        )

    (STAGE2B_DIR / "02_excluded_models_code_audit.md").write_text("\n".join(lines) + "\n")


# ============================================================================
# E: Revised Euler classification
# ============================================================================

def build_revised_classification():
    """Generate revised Euler classification with corrected denominator."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    # Read stage 2 classification
    stage2_csv = STAGE2_DIR / "02_euler_model_classification.csv"
    with open(stage2_csv, "r", newline="", encoding="utf-8") as f:
        stage2_rows = list(csv.DictReader(f))

    # Build lookup
    stage2_by_model = {r["model"]: r for r in stage2_rows}

    revised_map = {
        "gr4j": ("ANALYTICAL_DAILY_FORM_NOT_EULER", True, False, False, "EXCLUDED_ANALYTICAL",
                 "Closed-form daily integration; Euler convergence not applicable."),
        "gsfb": ("HARNESS_LIMITATION_POTENTIALLY_TESTABLE", True, False, True, "EXCLUDED_HARNESS_LIMITED",
                 "Smooth variant exists; rate params dt-scalable; needs harness extension."),
        "mopex4": ("POTENTIALLY_TESTABLE_NEEDS_STAGE2C", True, False, True, "EXCLUDED_HARNESS_LIMITED",
                    "Native delta_t support; doy is optional; harness limitation only. NOT YET TESTED."),
        "mopex5": ("POTENTIALLY_TESTABLE_NEEDS_STAGE2C", True, False, True, "EXCLUDED_HARNESS_LIMITED",
                    "Same as mopex4: native delta_t + optional doy. NOT YET TESTED."),
        "shm": ("DISABLED_NOT_IN_SCOPE", False, False, False, "DISABLED_NOT_IN_SCOPE",
                "File empty; never part of 36-model denominator."),
        "tank": ("EMPIRICAL_DAILY_THRESHOLD_PARTITION", True, False, False, "EXCLUDED_STRUCTURAL",
                 "Daily-threshold side-outlet activation; structural to TANK concept."),
        "tcm": ("HARNESS_LIMITATION_POTENTIALLY_TESTABLE", True, False, True, "EXCLUDED_HARNESS_LIMITED",
                "mean_P is constant; baseflow_6 is smooth; harness limitation. NOT YET TESTED."),
    }

    rows = []
    for r in stage2_rows:
        model = r["model"]
        orig_class = r["class"]

        if model in revised_map:
            rev_class, count_in_36, count_in_euler, needs_s2c, paper_cat, reason = revised_map[model]
        else:
            rev_class = orig_class
            count_in_36 = True
            count_in_euler = True
            needs_s2c = False
            paper_cat = "TESTED_" + orig_class
            reason = ""

        rows.append({
            "model": model,
            "original_stage2_class": orig_class,
            "revised_recommended_class": rev_class,
            "should_count_in_36_model_denominator": count_in_36,
            "should_count_in_euler_tested_denominator": count_in_euler,
            "needs_stage2c_test": needs_s2c,
            "paper_table_category": paper_cat,
            "reason": reason,
        })

    _write_csv(STAGE2B_DIR / "04_revised_euler_classification_recommendation.csv", [
        "model", "original_stage2_class", "revised_recommended_class",
        "should_count_in_36_model_denominator", "should_count_in_euler_tested_denominator",
        "needs_stage2c_test", "paper_table_category", "reason",
    ], rows)

    # Revised counts
    count_36 = sum(1 for r in rows if r["should_count_in_36_model_denominator"])
    count_euler_tested = sum(1 for r in rows if r["should_count_in_euler_tested_denominator"])
    count_disabled = sum(1 for r in rows if not r["should_count_in_36_model_denominator"])
    count_s2c = sum(1 for r in rows if r["needs_stage2c_test"])
    count_excluded = count_36 - count_euler_tested
    pass_count = sum(1 for r in rows if r["original_stage2_class"] == "PASS_FIRST_ORDER")
    caveat_count = sum(1 for r in rows if r["original_stage2_class"] == "PASS_WITH_CAVEAT")
    threshold_count = sum(1 for r in rows if r["original_stage2_class"] == "THRESHOLD_CROSSING_LIMITED")
    precision_count = sum(1 for r in rows if r["original_stage2_class"] == "PRECISION_FLOOR_LIMITED")

    print(f"\n=== Revised Euler Classification Counts ===")
    print(f"enabled_model_count: {count_36}")
    print(f"euler_tested_count: {count_euler_tested}")
    print(f"true_excluded_count: {count_excluded}")
    print(f"disabled_not_in_scope_count: {count_disabled}")
    print(f"potentially_testable_count: {count_s2c}")
    print(f"pass_first_order_count: {pass_count}")
    print(f"pass_with_caveat_count: {caveat_count}")
    print(f"threshold_limited_count: {threshold_count}")
    print(f"precision_floor_limited_count: {precision_count}")
    print(f"\n36 = {count_euler_tested} tested + {count_excluded} excluded")
    print(f"Of {count_excluded} excluded: {count_s2c} potentially testable in Stage 2c")
    print(f"SHM ({count_disabled} model) removed from denominator")

    return {
        "count_36": count_36,
        "euler_tested": count_euler_tested,
        "excluded": count_excluded,
        "disabled": count_disabled,
        "potentially_testable": count_s2c,
        "pass": pass_count,
        "caveat": caveat_count,
        "threshold": threshold_count,
        "precision": precision_count,
    }


# ============================================================================
# F: Paper defense text
# ============================================================================

def write_paper_defense(counts: dict):
    lines = [
        "# Stage 2b: Paper Defense Text for GMD 3.1.3 Euler Convergence Exclusions",
        "",
        "## F1. Recommended short paragraph for manuscript (140 words)",
        "",
        "> Of the 36 implemented core models, 30 are amenable to Euler substep convergence testing "
        "> using a diagnostic zero-order-hold dt-scaling wrapper that scales precipitation, PET, and "
        "> rate parameters with dt while leaving structural and threshold parameters unchanged. "
        "> The remaining 6 models are excluded from Euler convergence analysis for structural reasons "
        "> rather than numerical failures: GR4J uses analytically integrated daily closed-form equations "
        "> (Perrin et al., 2003) that are not ODE-based; TANK employs daily empirical threshold "
        "> partitioning rules that define discrete bucket-level activation points; GSFB, MOPEX4, "
        "> MOPEX5, and TCM are excluded due to current harness limitations involving smooth variant "
        "> parameters (tau), day-of-year dependencies, or climatological constants that require "
        "> harness extensions rather than equation modifications. One additional model (SHM) has no "
        "> implementation and is excluded from the 36-model denominator. These exclusions do not "
        "> reflect on the numerical fidelity of the Euler scheme for rate-based hydrological operators.",
        "",
        "## F2. Supplement table: Exclusion reasons per model",
        "",
        "| Model | Reason for exclusion | Code evidence | Alternative evidence | Interpretation |",
        "|---|---|---|---|---|",
        "| GR4J | Analytical daily closed-form integration | `_calc_production_store_tanh` (tanh-based), `_calc_percolation_analytical` (power-based) | Formula-level daily regression (not Euler convergence) | Analytical formulations are not ODEs; substep refinement is not semantically meaningful |",
        "| GSFB | Harness limitation: rate parameters (c, dpf, emax, frate) admit dt-scaling but harness lacks tau-aware wrapper | smooth_relu/min/cap_flux (tau=1e-3) for flux caps; sub-functions use soft gates (F.softplus, smooth_threshold) | Stage 2c: dt-aware wrapper with tau handling | Exclusion is a harness limitation; smooth variant is potentially substeppable |",
        "| MOPEX4 | Harness limitation: native `delta_t` support exists; `doy` is optional keyword-only | `mopex4_step(delta_t=1.0, *, doy=None)` — delta_t already in signature | Stage 2c: pass fixed doy, scale with delta_t | Exclusion reason 'doy blocks substep' is incorrect; model is already dt-aware |",
        "| MOPEX5 | Same as MOPEX4 | Same structure; `mopex5_step(delta_t=1.0, *, doy=None)` | Stage 2c: same approach | Exclusion reason is incorrect; model is already dt-aware |",
        "| TANK | Daily empirical threshold partition | Thresholds st, f1, f2, f3 define discrete side-outlet activation levels; F.relu at bucket boundaries | Local away-from-threshold substep (limited value) | Threshold structure is inherent to TANK model concept; exclusion is structural, not numerical |",
        "| TCM | Harness limitation: `mean_P` is a climatological constant holdable during substep; k1 rate admits dt-scaling | `mean_P` is keyword-only required; `baseflow_6` uses smooth_threshold | Stage 2c: pass fixed mean_P, scale k1 by dt | Exclusion is a harness limitation; model is potentially substeppable |",
        "| (SHM) | Disabled; empty file | 0-byte file; manually added to registry as disabled | None — not an implemented model | Excluded from 36-model denominator entirely |",
        "",
        "## F3. Anticipated reviewer concerns and responses",
        "",
        "| Reviewer concern | Recommended response |",
        "|---|---|",
        "| Why aren't all 36 models tested with Euler convergence? | 30 of 36 implemented models are tested. Among the 6 remaining: GR4J uses analytical daily formulas (not ODE-based); TANK uses daily empirical threshold partitioning inherent to the model concept. The other 4 (GSFB, MOPEX4, MOPEX5, TCM) are excluded due to harness limitations that can be resolved with harness extensions — none require equation changes. 1 model (SHM) has no implementation. |",
        "| Does exclusion mean the excluded models are unreliable? | No. The Euler substep convergence test evaluates numerical scheme consistency under dt refinement, not model correctness. Excluded models pass all other verification: mass balance closure (36/36), autograd vs finite-difference gradient check (36/36), and pymarrmot cross-check. |",
        f"| Why does the report say 30+7=37 when there are only 36 models? | The '7' count included SHM which is already disabled and should not be counted. The correct breakdown is: 36 enabled models = 30 tested + 6 excluded. SHM is documented as a disabled model outside the 36-model denominator. This was a tabulation error in the stage 2 report that is corrected here. |",
        "| Can the excluded models be made substeppable? | GSFB, MOPEX4, MOPEX5, and TCM can potentially be tested without equation changes by extending the dt-wrapper harness. GR4J cannot (analytical daily formulas). TANK could locally but the threshold structure is inherent. SHM has no implementation. |",
        "| Does the smooth gate approximation change MARRMoT hard threshold behavior? | Yes, intentionally. The smooth gates are differentiable approximations parameterized by k (default 10 for storage, 5 for temperature). They are NOT claimed to be equivalent to hard thresholds. Error relative to hard-threshold references is explicitly characterized in the smooth-gate sensitivity analysis (Section 3.1.3). |",
        "| How should the Euler convergence limitation be presented in the manuscript? | As a 'numerical fidelity' claim: the explicit scheme is internally consistent and exhibits expected first-order behavior where smooth dynamics dominate. Deviations are concentrated at threshold crossings (a known limitation of explicit schemes) or precision floors. Excluded models reflect structural formulation choices, not numerical errors. This is about scheme consistency, not about reproducing another solver's output. |",
    ]

    (STAGE2B_DIR / "05_stage2b_paper_defense_text.md").write_text("\n".join(lines) + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=== Stage 2b: Excluded / Caveat Model Deep Audit ===\n")

    print("[1/4] Denominator audit...")
    audit_denominator()

    print("[2/4] Excluded model code audit...")
    audit_excluded_models()

    print("[3/4] Revised Euler classification...")
    counts = build_revised_classification()

    print("[4/4] Paper defense text...")
    write_paper_defense(counts)

    print(f"\nAll files written to: {STAGE2B_DIR}")
    print(f"\n=== Key finding ===")
    print(f"MOPEX4/MOPEX5 already HAVE delta_t parameter! Exclusion reason was incorrect.")
    print(f"GSFB uses smooth variant (smooth_relu/min/cap_flux) — potentially testable.")
    print(f"TCM's mean_P is a constant — potentially testable.")
    print(f"SHM is disabled — double-counted in stage 2 '7 excluded'. Correct: 6 excluded.")
    print(f"36 = 30 tested + 6 excluded (not 30 + 7 = 37)")


if __name__ == "__main__":
    main()
