# Maintained project scripts

Place reusable analysis, validation, migration, and reporting tools here.
Production trainers belong under `training/`; one-off completed runners belong
under `archive/`.

Current TGD2 work:

- `analyze_tgd2_cmaes_vs_dpl_interim.py`: basin-paired, snow-stratified
  comparison between the formal TGD2 CMA-ES result and the paused dPL result.
- `run_model_test_suite.py`: canonical active model regression gate.
- `audit_state_and_water_balance.py`: shared state/water-balance diagnostics.

The remaining analysis scripts reproduce historical Base/CN/old-TGD, XNES,
and dPL tables. They read old result paths through compatibility symlinks and
must not be treated as active model implementations. Scripts importing
retired TGD classes were moved to
`archive/project_cleanup_20260730/legacy_tgd_analysis_scripts/`.

New scripts must write products beneath `results/<run_id>/analysis/` or
`results/archive/<category>/<run_id>/`, never to a new physical `outputs/`
tree.

