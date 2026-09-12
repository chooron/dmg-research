# R4 Held-Out Replay Audit Log: Does OOB Support Frozen R1–R3?

## Session Details
- **Repository Root:** `/home/jingxin/code/dmg-research`
- **Audit Directory:** `project/benchmark/manuscript/r4/r123_support_audit`
- **Git HEAD Baseline:** `3caca37a4243ae0a95ebe9cc4f22998672ddf464`
- **Tested Subset:** Prespecified 8-model stress-test panel (`alpine2`, `hbv96`, `xinanjiang`, `newzealand2`, `ihacres`, `us1`, `mopex4`, `hillslope`)
- **Dataset Contract:** 531 CAMELS-US basins, 5-fold cross-validation, 40/40 jobs completed
- **Execution Date:** 2026-09-06

---

## Audit Execution History

### Step 0: Workspace Setup & Source Inventory
- **Action:** Created audit workspace and inspected authoritative source tables, parquet files, and frozen R1–R3 definitions.
- **Verification:** Verified 40/40 completed OOB runs in `project/benchmark/results/oob_primary8_5fold_20260902/`, fold assignments, parameter masters, and performance masters.

### Step 1: Agent A — R1 Held-Out Replay
- **Action:** Replayed outlet performance comparison $\Delta\text{KGE}_{\text{OOB}}$ across 4,248 model–basin instances and executed two-way ANOVA variation decomposition.
- **Deliverable:** `AGENT_A_R1_OOB_SUPPORT.md`
- **Findings:** OOB-dPL median KGE is 0.581 (vs. IC 0.650), yielding a modest regionalization gap of $-0.065$ KGE. Basin-level heterogeneity remains extreme (OOB-dPL beats IC in 26.1% of basins; basin-additive variance accounts for 67.1% of total variance).
- **Verdict:** `R1 PARTIALLY SUPPORTED`

### Step 2: Agent B — R2 Held-Out Replay
- **Action:** Audited prespecification of R2 estimands and replayed normalized parameter displacement $D_{\text{RMS, OOB}}$ and the performance–parameter bridge.
- **Deliverable:** `AGENT_B_R2_OOB_SUPPORT.md`
- **Findings:** Parameter displacement remains large and stable ($D_{\text{RMS, OOB}} = 0.343$ vs. seen $0.314$), with minimal shift between seen and OOB dPL parameters ($0.100$). Performance bridge is positive across 8/8 models ($\rho = +0.242$). Un-replayed geometry estimands ($R_{\text{rank}}, C_{\text{eff}}$) are explicitly scoped as `NOT TESTED BY R4`.
- **Verdict:** `R2 PARTIALLY SUPPORTED`

### Step 3: Agent C — R3 Held-Out Replay
- **Action:** Replayed direct cross-paradigm profile correspondence $R_{\text{paired}}$ and same-coordinate specificity $A_{\text{diag}}$ with 1,000 parameter-label permutations.
- **Deliverable:** `AGENT_C_R3_OOB_SUPPORT.md`
- **Findings:** Direct IC vs. OOB-dPL profile correspondence is $R_{\text{paired, OOB}} = 0.739$ (matching seen $0.713$). Same-coordinate specificity is rigorously preserved ($A_{\text{diag, OOB}} = 0.718$ vs. off-diagonal $-0.011$; permutation $p = 0.000999$).
- **Verdict:** `R3 STRONGLY SUPPORTED`

### Step 4: Agent D — Hostile Reviewer Review
- **Action:** Evaluated objections D1–D8 regarding cherry-picking, performance degradation, noise, un-replayed claims, and constructive vs. empirical specificity.
- **Deliverable:** `AGENT_D_HOSTILE_R123_SUPPORT_REVIEW.md`
- **Findings:** Verified that no cherry-picking or data leakage occurred, confirmed empirical nature of coordinate specificity, and confirmed no blocker requiring a rerun exists.
- **Verdict:** `NO RERUN REQUIRED`

### Step 5: Parent Synthesis, Frozen Tables, and Final Verdict
- **Action:** Compiled final structured verdict, support matrix, numeric freeze table, handoff text, rejected interpretations, and source manifests.
- **Deliverables:**
  - `R4_R123_SUPPORT_FINAL_VERDICT.md`
  - `R4_R123_SUPPORT_MATRIX.csv`
  - `R4_NUMERIC_FREEZE_TABLE.csv`
  - `R4_METHODS_RESULTS_HANDOFF.md`
  - `R4_REJECTED_INTERPRETATIONS.md`
  - `R4_R123_SOURCE_MANIFEST.json`
  - `R4_R123_SOURCE_CHECKSUMS.sha256`
  - `R4_R123_AUDIT_LOG.md`
- **Final Selected Option:**
  `B. R4 PARTIALLY SUPPORTS R1–R3; CORE RETENTION PRESENT BUT SOME CLAIMS ARE NOT RETESTED`

---

## Final Status & Integrity Check
- **Zero Files Deleted**
- **Zero Existing Analysis Overwritten**
- **Zero Historical Sources Modified**
- **All 10 Audit Deliverables Generated in:** `/home/jingxin/code/dmg-research/project/benchmark/manuscript/r4/r123_support_audit/`
- **STOP Rule Applied**
