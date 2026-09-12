# R4 Model Selection Provenance Audit Log

## Session Details
- **Repository Root:** `/home/jingxin/code/dmg-research`
- **Audit Directory:** `project/benchmark/manuscript/r4/model_selection_audit`
- **Git HEAD Baseline:** `3caca37a4243ae0a95ebe9cc4f22998672ddf464`
- **Audit Execution Date:** 2026-09-06
- **Auditor Role:** Independent PiCode Provenance and Scientific Integrity Auditor

---

## Audit Execution History

### Phase A: Timeline and Evidence Manifest Reconstruction
- **Action:** Scanned repository, Git history, file timestamps, and SHA256 checksums across all candidate selection, deployment, execution, and manuscript analysis artifacts.
- **Artifacts Created:**
  - `SELECTION_TIMELINE.md`
  - `SELECTION_EVIDENCE_MANIFEST.tsv`
- **Key Discovery:** Model selection was finalized on 2026-09-01 23:09:19 +0800 and frozen into `oob_primary8_5fold_20260902.yaml` on 2026-09-02 09:23:18 +0800. The first OOB job completed on 2026-09-02 12:08:08 +0800. Prespecification is rigorously confirmed.

### Phase B: Historical Selection Rule Audit
- **Action:** Examined pre-OOB Python source code (`select_models.py`, `revise_mopex_redundancy.py`, `selection_common.py`) and JSON provenance files to classify selection criteria.
- **Artifacts Created:**
  - `ORIGINAL_SELECTION_RULE_AUDIT.md`
- **Key Discovery:** Selection was deterministic and rule-based across 7 normalized features ($G_{\text{seen}}, R_{\text{seen}}, D_{\theta}, U, K_{\text{joint}}, P, A$). A hard performance viability gate ($K_{\text{joint}} \ge 0.561$) and structural anti-redundancy rule (MOPEX capped at 1 member) were applied prior to OOB execution.

### Phase C: Pre-OOB Design Space Coverage Audit
- **Action:** Extracted structural descriptors and seen-basin empirical metrics for all 36 models and compared the 8 selected models against the full benchmark distribution.
- **Artifacts Created:**
  - `R4_ALL36_MODEL_COMPARISON_TABLE.csv`
  - `R4_MODEL_SUBSET_COVERAGE_TABLE.csv`
  - `R4_MODEL_SUBSET_COVERAGE_AUDIT.md`
- **Key Discovery:** The subset spans $P \in [5, 15]$ (71.4% range span), $S \in [1, 5]$ (100% range span), snow dynamics (37.5% vs 33.3%), 50/50 routing balance, and 4-quadrant $G \times R$ orthogonal regimes.

### Phase D: Hostile Selection-Bias Review
- **Action:** Evaluated potential reviewer objections regarding post-hoc cherry-picking, list revisions, and sample representativeness.
- **Artifacts Created:**
  - `HOSTILE_MODEL_SELECTION_REVIEW.md`
- **Key Discovery:** 0 job failures, 0 retries, and 0 post-hoc substitutions occurred across 40 completed OOB runs. However, the subset cannot be claimed as an unbiased statistical sample of the 36-model benchmark.

### Phase E: Manuscript Wording Decision
- **Action:** Selected appropriate wording tier from Tier 1–4.
- **Artifacts Created:**
  - `R4_SELECTION_WORDING_DECISION.md`
- **Decision:** **Tier 2 (Prespecified Spanning Subset)** adopted; "Representative" rejected for population inference.

### Phase F: Machine-Readable Manifest
- **Action:** Generated structured JSON manifest matching the required specification.
- **Artifacts Created:**
  - `R4_MODEL_SELECTION_MANIFEST.json`

### Phase G: Manuscript Handoff Text & Final Report
- **Action:** Drafted ready-to-use manuscript sentences and reviewer-response paragraph, plus final structured verdict report.
- **Artifacts Created:**
  - `R4_MODEL_SELECTION_MANUSCRIPT_HANDOFF.md`
  - `R4_MODEL_SUBSET_FINAL_VERDICT.md`

### Reproducibility Verification:
- **Action:** Created SHA256 checksums and source manifest for all 49 related artifacts.
- **Artifacts Created:**
  - `SOURCE_MANIFEST.json`
  - `SOURCE_CHECKSUMS.sha256`
  - `AUDIT_LOG.md`

---

## Final Status and Integrity Check
- **Files Deleted:** 0
- **Existing R4 Analysis Overwritten:** 0
- **Historical Source Modified:** 0
- **All Audit Artifacts Successfully Created:** 11 files in `project/benchmark/manuscript/r4/model_selection_audit/`
- **Final Verdict:** **`B. PRESPECIFIED SPANNING SUBSET SUPPORTED - NO RERUN`**
