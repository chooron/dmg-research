# Fresh Validation Source Snapshot

**Execution Timestamp**: 2026-09-08T21:30:00Z  
**Repository Working Directory**: `/home/jingxin/code/dmg-research`

---

## 1. Git Repository State

- **Committed HEAD Commit**: `3caca37a4243ae0a95ebe9cc4f22998672ddf464`
- **Current Branch**: `master`
- **Working-Tree Status**: `DIRTY` (active development modifications tested)
- **Tracked Diff SHA-256**: `9148dbfa28bb9c90c1afd8e3b1a620f808b70026c807a80f7adfd79e2375e9d2`
- **Saved Diff Patch**: `fresh_source_diff.patch`

---

## 2. Tracked File Modifications (`git status --short`)

```text
M .gitignore
 M dmotpy/data_contract.py
 M dmotpy/models/core/vic.py
 M dmotpy/models/hydrology_model.py
 M project/benchmark/scripts/diagnostics/e2_boundary_kink.py
 M project/benchmark/scripts/diagnostics/full_model_fd_warmup_modes.py
 M project/benchmark/scripts/diagnostics/k_full_retrain.py
 M project/benchmark/scripts/diagnostics/round12_edge_probe.py
 M project/benchmark/scripts/diagnostics/warmup_gradient_contract.py
 M project/benchmark/src/model_registry.py
 M project/hydrodiag/manuscript/scripts/supplement/plot_huc2_loro_robustness.py
 M project/hydrodiag/manuscript/supplement/final_assets/figures/Figure_S1/Figure_S1.png
 M project/hydrodiag/manuscript/supplement/final_assets/figures/Figure_S1/caption_facts.md
 M project/hydrodiag/manuscript/supplement/final_assets/figures/Figure_S1/plot_Figure_S1.py
 M project/hydrodiag/manuscript/supplement/final_assets/tables/Table_S3/Table_S3.md
 M project/hydrodiag/manuscript/supplement/final_assets/tables/Table_S3/Table_S3_panelA.csv
 M project/hydrodiag/manuscript/supplement/final_assets/tables/Table_S3/Table_S3_panelB.csv
 M project/hydrodiag/training/dpl/run_dpl_model.py
?? caravan_671_attributes.npy:Zone.Identifier
?? dmotpy/DMOTPY_CURRENT_STATE_AUDIT_20260831.md
?? dmotpy/tests/test_vic_doy_remediation.py
?? project/benchmark/BENCHMARK_CURRENT_STATE_AUDIT_20260831.md
?? project/benchmark/CANONICAL_V2_POST_TRAINING_CONSISTENCY_AUDIT_20260901.md
?? project/benchmark/CANONICAL_V2_REMOTE_TO_LOCAL_REPLAY_FORENSICS_20260901.md
?? project/benchmark/CURRENT_PROJECT_DPL_AUDIT_SYNTHESIS_20260831.md
?? project/benchmark/DPL_NOGPU_PHASE_FINAL_REPORT_20260831.md
?? project/benchmark/PENMAN_TRUNCATE90_CLEANUP_AND_SYNC_REPORT_20260831.md
?? project/benchmark/PENMAN_TRUNCATE90_CORRECTION_NOTE_20260831.md
?? project/benchmark/PENMAN_TRUNCATE90_PROVENANCE_AUDIT_20260831.md
?? project/benchmark/analysis/
?? project/benchmark/configs/oob_primary8_5fold_20260902.yaml
?? project/benchmark/manuscript/
?? project/benchmark/scripts/ablation/
?? project/benchmark/scripts/canonical_v2/
?? project/benchmark/scripts/diagnostics/formal_seenbasin_atlas.py
?? project/benchmark/scripts/diagnostics/parameter_attribute_atlas.py
?? project/benchmark/scripts/diagnostics/parameter_attribute_atlas_followup.py
?? project/benchmark/scripts/diagnostics/run_agent2_replay.py
?? project/benchmark/scripts/diagnostics/run_agent3_audit.py
?? project/benchmark/scripts/diagnostics/run_agent4_decomposition.py
?? project/benchmark/scripts/diagnostics/run_task_s0_evaluation.py
?? project/benchmark/scripts/oob/
?? project/benchmark/tests/
?? project/flexmopex/manuscript/stats/
?? project/hydrodiag/CH3_3_3_PARAMETER_COMPENSATION_AUDIT.md
?? project/hydrodiag/CH3_3_4_CONTROLLED_RECOVERY_AUDIT.md
?? project/hydrodiag/CH3_3_5_INTERNAL_STATE_PROCESS_AUDIT.md
?? project/hydrodiag/CH3_3_6_CROSS_PROCESS_STATUS_AUDIT.md
?? project/hydrodiag/CH3_3_6_WRITE_READY_DATA_REPORT.md
?? project/hydrodiag/CH3_3_6_WRITE_READY_DATA_REPORT.md:Zone.Identifier
?? project/hydrodiag/camels_clim.txt
?? project/hydrodiag/camels_clim.txt:Zone.Identifier
?? project/hydrodiag/camels_hydro.txt
?? project/hydrodiag/camels_hydro.txt:Zone.Identifier
?? project/hydrodiag/manuscript/supplement/HESS_Supplement.docx:Zone.Identifier
?? project/hydrodiag/manuscript/supplement/final_assets.zip
?? project/hydrodiag/scripts/run_ch3_6_activity_stratification.py
?? project/hydrodiag/scripts/run_ch3_6_analysis.py
?? project/hydrodiag/scripts/run_ch3_6_final_strict_audit.py
?? project/hydrodiag/scripts/run_ch3_6_gap_fill.py
?? project/hydrodiag/scripts/run_ch3_6_provenance.py
?? project/hydrodiag/scripts/run_ch3_6_replay.py
?? project/hydrodiag/scripts/run_ch3_6_write_ready.py
```

---

## 3. Working-Tree Changes Summary Relevant to Tested Code

1. `dmotpy/data_contract.py`:
   - Updated `CALENDAR_MODELS = frozenset({"mopex4", "mopex5", "vic"})` to include `vic`.
2. `dmotpy/models/core/vic.py`:
   - Updated `vic_step` signature and implementation to accept optional keyword argument `doy: torch.Tensor = None`.
   - Updated `t_idx = doy if doy is not None else torch.ones_like(P)` for dynamic Day of Year phenology.
3. `dmotpy/models/hydrology_model.py`:
   - Added support for extracting and unbinding `doy_seq` from 4-channel forcing or `x_dict["doy"]`, forwarding `doy` to `vic_step` during both warm-up and simulation loops.
4. `project/benchmark/src/model_registry.py`:
   - Added `_validate_warmup_grad_mode` to reject unimplemented modes and enforce `warmup_grad_mode == "detach"`.

---

## 4. Combined Source State Attestation

The validation suite in this run tests the exact working tree defined by commit `3caca37a4243ae0a95ebe9cc4f22998672ddf464` plus the uncommitted diff saved in `fresh_source_diff.patch` (SHA-256: `9148dbfa28bb9c90c1afd8e3b1a620f808b70026c807a80f7adfd79e2375e9d2`). No production model code was patched or modified during the execution of this test suite.
