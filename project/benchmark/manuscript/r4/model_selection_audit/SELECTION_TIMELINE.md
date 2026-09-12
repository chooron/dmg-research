# R4 Model Subset Selection Timeline and Provenance Audit

## 1. Executive Summary and Git Baseline

- **Repository Root:** `/home/jingxin/code/dmg-research`
- **Current Git HEAD:** `3caca37a4243ae0a95ebe9cc4f22998672ddf464`
- **Audit Target:** Provenance and timing of the eight-model subset (`alpine2`, `hbv96`, `xinanjiang`, `newzealand2`, `ihacres`, `us1`, `mopex4`, `hillslope`) used in the R4 held-out-basin/out-of-bag (OOB) experiment.
- **Key Finding:** The exact eight-model list was deterministically generated, revised for structural anti-redundancy, and formally frozen in `project/benchmark/configs/oob_primary8_5fold_20260902.yaml` on **2026-09-02 09:23:18 +0800**, and deployed to the remote execution cluster on **2026-09-02 10:58:35 +0800**. The first OOB job (`alpine2_fold0`) started on **2026-09-02 11:03:47 +0800** and completed on **2026-09-02 12:08:08 +0800**. 
- **Prespecification Verdict:** **CONFIRMED**. The exact eight-model list was fixed, configured, and deployed prior to the availability of any OOB outcome.
- **Non-Test-Informed Verdict:** **CONFIRMED**. Selection criteria were strictly derived from pre-existing seen-basin R1–R3 metrics ($G_{\text{seen}}$, $R$, $D$, $U$, $K_{\text{joint}}$, $P$, $A$), with zero access to held-out OOB outcomes.

---

## 2. Chronological Timeline of Selection Events

### Event 1: Early Exploratory Protocol Concept
- **Timestamp:** 2026-08-29 12:58:21 +0800 (04:58:21 UTC)
- **Artifacts:**
  - `project/benchmark/results/r1_r2_nontraining_complete_20260829/future_oob_pur_training_protocol.md` (SHA256: `4bb74fd4...`)
  - `project/benchmark/results/r1_r2_nontraining_complete_20260829/future_oob_pur_training_protocol.json` (SHA256: `3a9f5f99...`)
  - `project/benchmark/results/r1_r2_nontraining_complete_20260829/oob_pur_model_screening_strict34.csv` (SHA256: `94a23f34...`)
- **Action/Content:**
  - An exploratory screening protocol for regional stress testing (leave-one-HUC-group-out / PUR) was drafted.
  - Proposed an initial 8-model screening set: `collie1`, `gr4j`, `hbv96`, `hillslope`, `modhydrolog`, `mopex4`, `topmodel`, `xinanjiang` (with alternates `alpine1`, `hymod`).
  - Marked explicitly: `status: PREPARED_NOT_EXECUTED`, `do_not_execute: true`.
  - **Relevance:** Demonstrates that no OOB training had occurred, and establishes the project's strict anti-leakage rule ("no relationship outcome in model or subset selection").

---

### Event 2: Formal Rule-Based Quadrant Model Selection
- **Timestamp:** 2026-09-01 22:29:34 – 22:36:56 +0800 (14:29:34 – 14:36:56 UTC)
- **Artifacts:**
  - `project/benchmark/analysis/oob_model_selection_20260901/selection_common.py` (SHA256: `58879291...`)
  - `project/benchmark/analysis/oob_model_selection_20260901/select_models.py` (SHA256: `780ec46f...`)
  - `project/benchmark/analysis/oob_model_selection_20260901/plot_selection_qc.py` (SHA256: `bb1a8f18...`)
  - `project/benchmark/results/oob_model_selection_20260901/OOB_MODEL_SELECTION_REPORT.md` (SHA256: `b6d2fe40...`)
  - `project/benchmark/results/oob_model_selection_20260901/OOB_SELECTION_PROVENANCE.json` (SHA256: `ebad026f...`)
  - `project/benchmark/results/oob_model_selection_20260901/OOB_PRIMARY_8_MODELS.txt` (SHA256: `56edfd29...`)
  - `project/benchmark/results/oob_model_selection_20260901/OOB_PRIMARY_8_MODEL_TABLE.csv` (SHA256: `620132d9...`)
  - `project/benchmark/results/oob_model_selection_20260901/OOB_MODEL_SELECTION_QUADRANTS.csv` (SHA256: `73d3d60e...`)
  - `project/benchmark/results/oob_model_selection_20260901/OOB_MODEL_SELECTION_FEATURES.csv` (SHA256: `b6befb2e...`)
- **Action/Content:**
  - Complete candidate pool: all 36 canonical benchmark models.
  - Formulated a deterministic 7D feature space using 0–1 percentile ranks:
    1. $G_{\text{seen}} = \text{median}_b(\text{KGE}_{\text{IC}} - \text{KGE}_{\text{dPL}})$ (seen flexibility gap);
    2. $R$ (parameter–attribute reproducibility);
    3. $D_{\theta}$ (RMS parameter displacement between IC and dPL);
    4. $U$ (IC multi-start parameter uncertainty);
    5. $K_{\text{joint}} = \min(\text{KGE}_{\text{IC}}, \text{KGE}_{\text{dPL}})$;
    6. $P$ (parameter count);
    7. $A$ (count of FDR-significant $G$–attribute associations).
  - Four quadrants defined by median splits on $G$ (0.01041) and $R$ (0.73263).
  - Performance viability gate: $K_{\text{joint}} \ge Q25 = 0.56099$.
  - Selected 2 models per quadrant: centroid-nearest representative + 5D maximum-contrast model.
  - Selected Initial Primary 8:
    - Q1 (Low G / High R): `alpine2` (REP), `hbv96` (CONTRAST) [via coverage repair: `simhyd` replaced by `alpine2` to ensure $P_{\text{low}}$ coverage]
    - Q2 (Low G / Low R): `mopex5` (REP), `xinanjiang` (CONTRAST)
    - Q3 (High G / High R): `ihacres` (REP), `mopex2` (CONTRAST)
    - Q4 (High G / Low R): `hillslope` (REP), `mopex4` (CONTRAST)

---

### Event 3: MOPEX Structural Anti-Redundancy Revision
- **Timestamp:** 2026-09-01 22:50:36 – 23:09:20 +0800 (14:50:36 – 15:09:20 UTC)
- **Artifacts:**
  - `project/benchmark/analysis/oob_model_selection_20260901/revise_mopex_redundancy.py` (SHA256: `a08be09d...`)
  - `project/benchmark/analysis/oob_model_selection_20260901/structure_qc.py` (SHA256: `6fba9c51...`)
  - `project/benchmark/analysis/oob_model_selection_20260901/README.md` (SHA256: `31c7d038...`)
  - `project/benchmark/analysis/oob_model_selection_20260901/HANDOFF.md` (SHA256: `499e721c...`)
  - `project/benchmark/results/oob_model_selection_20260901/OOB_MODEL_SELECTION_MOPEX_REDUNDANCY_REVISION.md` (SHA256: `f3b1d3f9...`)
  - `project/benchmark/results/oob_model_selection_20260901/OOB_MOPEX_REVISION_PROVENANCE.json` (SHA256: `92ed0152...`)
  - `project/benchmark/results/oob_model_selection_20260901/OOB_SCENARIO_A_KEEP_MOPEX5.csv` (SHA256: `c19b6a7c...`)
  - `project/benchmark/results/oob_model_selection_20260901/OOB_SCENARIO_B_KEEP_MOPEX4.csv` (SHA256: `3c425be0...`)
  - `project/benchmark/results/oob_model_selection_20260901/OOB_SCENARIO_SELECTION_QUALITY.csv`
  - `project/benchmark/results/oob_model_selection_20260901/OOB_STRUCTURAL_MODEL_DESCRIPTORS.csv` (SHA256: `6078dfa3...`)
  - `project/benchmark/results/oob_model_selection_20260901/OOB_STRUCTURAL_COVERAGE_QC.csv` (SHA256: `fb2c59fe...`)
- **Action/Content:**
  - Recognition that the initial selection had 3 MOPEX models (`mopex2`, `mopex4`, `mopex5`), creating structural over-representation.
  - Imposed a hard structural rule: eliminate `mopex2` and retain exactly one of `mopex4` or `mopex5`.
  - Evaluated two deterministic same-quadrant replacement scenarios:
    - **Scenario A (Keep `mopex5`):** `mopex2` $\rightarrow$ `us1` (Q3), `mopex4` $\rightarrow$ `tank` (Q4).
    - **Scenario B (Keep `mopex4`):** `mopex5` $\rightarrow$ `newzealand2` (Q2), `mopex2` $\rightarrow$ `us1` (Q3).
  - Deterministic comparison: Scenario B had significantly higher minimum pairwise 7D distance (0.7986 vs 0.5087) and mean pairwise distance (1.0971 vs 1.0807), higher $R$-extreme coverage (7/8 vs 6/8) and joint extreme coverage (4/8 vs 3/8), and lower objective loss (0.3472 vs 0.5937).
  - Scenario B yielded the exact model list:
    `hbv96`, `alpine2`, `xinanjiang`, `newzealand2`, `us1`, `ihacres`, `mopex4`, `hillslope`.
  - Both scenarios were flagged by the automated script as `BLOCKED_NO_FINAL_PRIMARY_8` solely because only 1 medium-$P$ model was present ($P_{\text{low}}: 3, P_{\text{medium}}: 1, P_{\text{high}}: 4$, instead of $\ge 2$). The candidate tables, quality metrics, and blocker evidence were preserved for decision-making before execution.

---

### Event 4: Formal Freezing and Partition Assignment for Execution
- **Timestamp:** 2026-09-02 09:11:40 – 09:23:18 +0800 (01:11:40 – 01:23:18 UTC)
- **Artifacts:**
  - `project/benchmark/results/oob_primary8_5fold_20260902/OOB_FOLD_ASSIGNMENT.csv` (SHA256: `fe9013bcd7155ea99ac85597e59ea5e417a13897dc08aaa69f2c63ddb6780085`)
  - `project/benchmark/configs/oob_primary8_5fold_20260902.yaml` (SHA256: `1e9e4e03e43d002aef723cb18dc9a331a6042f2a067cbe80bb7efbf3ecf50c05`)
- **Action/Content:**
  - 5-fold cross-validation split generated across 531 CAMELS-US basins using seed `20260902` (fold sizes: 107, 106, 106, 106, 106).
  - `oob_primary8_5fold_20260902.yaml` formally adopted Scenario B's eight models:
    ```yaml
    models:
      - alpine2
      - hbv96
      - xinanjiang
      - newzealand2
      - ihacres
      - us1
      - mopex4
      - hillslope
    ```
  - Fixed exact training contract: 5 folds, seed 42, AdamW, train loss checkpoint selection (`best.pt`), train-only attribute normalization.

---

### Event 5: Deployment Package Assembly and Remote Dispatch
- **Timestamp:** 2026-09-02 10:58:02 – 10:58:35 +0800 (02:58:02 – 02:58:35 UTC)
- **Artifacts:**
  - `project/benchmark/scripts/oob/prepare_oob_deployment.py` (SHA256: `bfeed651...`)
  - `project/benchmark/results/oob_deploy_20260902/OOB_PRIMARY8_MODELS.txt` (SHA256: `d67c57a7...`)
  - `project/benchmark/results/oob_deploy_20260902/OOB_DEPLOY_METADATA.json` (SHA256: `2be926c8...`)
  - `project/benchmark/results/oob_deploy_20260902/OOB_DEPLOY_SHA256.txt` (SHA256: `ad069fe7...`)
  - `project/benchmark/results/oob_deploy_20260902/oob_deploy_20260902.tar.gz` (SHA256: `70afc9f9...`)
- **Action/Content:**
  - Deployment bundle created and packaged for remote GPU cluster execution.

---

### Event 6: Remote Job Execution and Completion
- **Execution Interval:** 2026-09-02 11:03:47 +0800 to 2026-09-03 05:04:29 +0800
- **Sync/Summary Timestamp:** 2026-09-03 08:55:14 – 08:55:52 +0800
- **Artifacts:**
  - `project/benchmark/results/oob_primary8_5fold_20260902/remote_queue.log` (SHA256: `e109393a...`)
  - `project/benchmark/results/oob_primary8_5fold_20260902/OOB_PRIMARY8_5FOLD_SUMMARY.csv` (SHA256: `3b71c75f...`)
  - `project/benchmark/results/oob_primary8_5fold_20260902/OOB_PRIMARY8_5FOLD_LEDGER.md` (SHA256: `f22e3cd0...`)
  - `project/benchmark/results/oob_primary8_5fold_20260902/OOB_PRIMARY8_5FOLD_REPORT.md` (SHA256: `d33ed7cc...`)
- **Key Milestones:**
  - **First job started:** `alpine2_fold0` at 11:03:47 (2026-09-02).
  - **First job completed (first OOB result available):** `alpine2_fold0` at 12:08:08 (2026-09-02).
  - **All 40 jobs completed:** `hillslope_fold4` at 05:04:29 (2026-09-03) (40/40 completed, 0 failures, 0 retries).
  - **Results compiled:** 2026-09-03 08:55:52.

---

### Event 7: Manuscript R4 Analyses and Freeze
- **Timestamp:** 2026-09-03 09:24:08 – 10:11:24 +0800
- **Artifacts:**
  - `project/benchmark/manuscript/r4/R4_OOB_PROVENANCE_AND_COMPLETENESS_AUDIT.md` (SHA256: `2d32a8fa...`)
  - `project/benchmark/manuscript/r4/R4_RELATIONSHIP_CASE_SELECTION_RULE.md` (SHA256: `32c0ce5b...`)
  - `project/benchmark/manuscript/r4/R4_RELATIONSHIP_CASES_FROZEN.csv` (SHA256: `44dc1426...`)
  - `project/benchmark/manuscript/r4/R4_ESTIMAND_DICTIONARY.md` (SHA256: `52f69f3f...`)
  - `project/benchmark/manuscript/r4/R4_STATISTICAL_ANALYSIS_REPORT.md` (SHA256: `a425571c...`)
  - `project/benchmark/manuscript/r4/R4_HANDOFF.md` (SHA256: `6a1d9365...`)
- **Action/Content:**
  - Conducted post-hoc statistical analysis on the completed 40 OOB runs.
  - Froze four relationship cases from seen-basin data before evaluating OOB relationship persistence.

---

## 3. Provenance and Temporal Separation Matrix

| Phase | Milestone | Timestamp (Local) | Delta vs. First OOB Result |
|---|---|---|---|
| Exploratory Protocol | Regional PUR protocol draft | 2026-08-29 12:58:21 | -95 hours 10 min |
| Formal Selection | 4-quadrant rule & initial 8 | 2026-09-01 22:36:56 | -13 hours 31 min |
| Structural Revision | MOPEX anti-redundancy (Scenario B 8-model list) | 2026-09-01 23:09:19 | -12 hours 59 min |
| Config Freeze | `oob_primary8_5fold_20260902.yaml` created | 2026-09-02 09:23:18 | -2 hours 45 min |
| Remote Deploy | `oob_deploy_20260902.tar.gz` packaged | 2026-09-02 10:58:35 | -1 hour 10 min |
| Job Launch | First job (`alpine2_fold0`) launched | 2026-09-02 11:03:47 | -1 hour 04 min |
| **First Result** | **First job completed (`alpine2_fold0`)** | **2026-09-02 12:08:08** | **0 (T0)** |
| Final Job | Last job (`hillslope_fold4`) completed | 2026-09-03 05:04:29 | +16 hours 56 min |
| OOB Summary | All 40/40 runs downloaded & summarized | 2026-09-03 08:55:52 | +20 hours 47 min |
| R4 Manuscript | R4 audit, relationship cases & statistical report | 2026-09-03 09:24 – 10:11 | +21 to 22 hours |

---

## 4. Conclusion

1. **Temporal Precedence:** The eight models were fully determined and fixed on 2026-09-01 (23:09:19), frozen into config on 2026-09-02 (09:23:18), and deployed on 2026-09-02 (10:58:35). The first OOB result became available at 2026-09-02 12:08:08. Thus, model selection strictly preceded any OOB result by 2.75 to 13 hours.
2. **Zero OOB Feedback:** No model substitutions, additions, or exclusions occurred during or after the OOB execution. All 40/40 planned runs completed on the exact eight models specified in the pre-launch configuration.
3. **Audit Status:** Complete and verifiable.
