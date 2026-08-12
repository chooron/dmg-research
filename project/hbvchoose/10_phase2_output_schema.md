# 10 — Phase 2 Stress Pilot: Output Schema

## Required Output Files

The Phase 2 pilot, when executed, must produce these files:

| # | File | Purpose |
|---|------|---------|
| 1 | gate_metrics_by_basin.csv | Per-basin gate impact metrics |
| 2 | gate_metrics_by_role.csv | Per-role aggregated gate metrics |
| 3 | event_retention_summary.csv | Pre/post gate event counts |
| 4 | boundary_case_diagnostics.csv | Diagnostic boundary basin details |
| 5 | suspected_B_handling_summary.csv | Suspected_B failure evidence |
| 6 | phase2_stress_pilot_report.md | Full pilot report |

## Field Definitions

### gate_metrics_by_basin.csv

```text
basin_id            — basin identifier
role                — basin role category
K                   — gate threshold used
min_event_length    — minimum event length used
n_events_before     — event count before gate application
n_events_after      — event count after gate application
event_removal_rate  — fraction of events removed (0–1)
point_removal_rate  — fraction of timesteps removed (0–1)
sample_collapse_flag — TRUE if removal_rate > 0.50
phase_space_R2      — R² of pre vs post phase-space projection
bootstrap_stability — bootstrap CI width of removal rate
IC_phase_conflict_flag — TRUE if basin identified as IC-phase conflict
strong_A_preserved  — TRUE if strong-A signal survives gate
suspected_B_flag    — TRUE if basin flagged as suspected_B
diagnostic_boundary_flag — TRUE if basin is a diagnostic boundary case
training_eligible   — TRUE if basin eligible for training
final_status        — PASS / WARNING / FAIL
```

### gate_metrics_by_role.csv

```text
role                — basin role category
n_basins            — number of basins in role
mean_event_removal_rate — mean removal rate across role
median_event_removal_rate — median removal rate
n_collapsed         — number of basins with sample_collapse_flag
mean_phase_space_R2 — mean R² across role
strong_A_retention  — fraction of strong-A events retained
stable_count        — number of basins with bootstrap CI < 0.3
IC_phase_conflicts  — number of IC-phase conflict flags
final_role_status   — PASS / WARNING / FAIL
```

### event_retention_summary.csv

```text
basin_id            — basin identifier
role                — basin role category
n_events_raw        — raw event count
n_events_after_K1p5_len5 — events after primary gate
n_events_after_K1p5_len7 — events after sensitivity gate
events_retained_primary — fraction retained under primary
events_retained_sensitivity — fraction retained under sensitivity
sensitivity_reverses_main — TRUE if sensitivity changes basin status
```

### boundary_case_diagnostics.csv

```text
basin_id            — basin identifier
role                — IC_phase_boundary
K                   — gate threshold
n_events_raw        — raw event count
n_events_after      — events after gate
IC_phase_proximity  — distance metric to IC-phase boundary
gate_impact_ratio   — gate impact relative to non-boundary basins
boundary_distinguishable — TRUE if boundary case is separable
requires_exclusion  — TRUE if boundary behavior warrants exclusion
```

### suspected_B_handling_summary.csv

```text
basin_id            — basin identifier
role                — suspected_B_failure_evidence
failure_mode        — description of suspected failure
evidence_strength   — LOW / MEDIUM / HIGH
separable_from_boundary — TRUE if distinct from boundary cases
systematic_risk     — TRUE if failure suggests systematic gate defect
recommended_action  — KEEP_AS_EVIDENCE / ESCALATE / DISMISS
```

## Configuration Columns

All output CSVs must include columns for:

```text
K                   — replicate gate threshold
min_event_length    — replicate minimum event length
config_type         — primary or sensitivity
```

This ensures each row is self-describing and auditable independent of file name.
