# Original Selection Rule Audit and Criteria Classification

## 1. Overview and Purpose

This document audits and classifies the historical selection logic used to choose the eight models (`alpine2`, `hbv96`, `xinanjiang`, `newzealand2`, `ihacres`, `us1`, `mopex4`, `hillslope`) for the R4 held-out-basin/OOB experiment.

The audit reconstructs the historical selection rule from timestamped pre-OOB source code, configuration files, and generated JSON/Markdown artifacts, distinguishing explicit pre-OOB rules from retrospective properties.

---

## 2. Historical Selection Rule Architecture

The historical selection procedure operated in three sequential pre-OOB stages on 2026-09-01:

### Stage 1: Candidate Pool Definition
- **Candidate Pool:** All 36 canonical conceptual models from the benchmark study.
- **Exclusion:** Zero models excluded a priori (`manual_override: false`, `exclusions: none`).

### Stage 2: 7D Percentile Feature Space and 4-Quadrant Stratification
- Seven continuous features were normalized to $[0, 1]$ percentile ranks:
  $$\text{rank}_{01}(x_m) = \frac{\text{rank}(x_m) - 1}{36 - 1}$$
- **Core Stratification Axes:**
  1. $G_{\text{seen}} = \text{median}_b(\text{KGE}_{\text{IC}, b} - \text{KGE}_{\text{dPL}, b})$ (Seen-basin shared-mapping flexibility gap; median threshold = `0.01041477`)
  2. $R = \text{median}_p(\text{reproducibility})$ (Parameter–attribute association reproducibility; median threshold = `0.73263305`)
- **Auxiliary Contrast Features:**
  3. $D_{\theta}$ (Bounds-normalized RMS parameter displacement between IC and dPL)
  4. $U$ (IC multi-start restart uncertainty / parameter non-uniqueness)
  5. $K_{\text{joint}} = \min(\text{median}(\text{KGE}_{\text{IC}}), \text{median}(\text{KGE}_{\text{dPL}}))$ (Baseline performance viability)
  6. $P$ (Free parameter count / model complexity)
  7. $A$ (Count of FDR-significant $G$–attribute empirical associations)
- **Quadrant Structure:**
  - $Q_1$ (Low $G$ / High $R$): High performance fidelity under dPL, highly reproducible parameter–attribute links.
  - $Q_2$ (Low $G$ / Low $R$): High performance fidelity under dPL, but low parameter–attribute reproducibility.
  - $Q_3$ (High $G$ / High $R$): Significant performance gap (IC outperforms dPL), but high parameter–attribute reproducibility.
  - $Q_4$ (High $G$ / Low $R$): Significant performance gap and low parameter–attribute reproducibility.
- **Within-Quadrant Selection Rule:**
  - **Centroid Representative (REP):** Candidate with minimum Euclidean distance to the quadrant centroid in 7D percentile space.
  - **Contrast Model (CONTRAST):** Candidate maximizing Euclidean distance in 5D space ($D_{\theta}, U, A, P, K_{\text{joint}}$) from the centroid representative.
- **Hard Viability Gate:** $K_{\text{joint}} \ge Q25 = 0.56099$ to filter out degraded models.
- **Coverage Repair:** Enforced minimum representation across parameter complexity tertiles ($P_{\text{low}}, P_{\text{medium}}, P_{\text{high}} \ge 2$) and signal strength ($A_{\text{low}}, A_{\text{high}} \ge 2$), repairing slots within the same quadrant. This replaced `simhyd` with `alpine2` in $Q_1$.

### Stage 3: Structural Anti-Redundancy Revision (MOPEX Revision)
- The initial primary 8 selection contained three MOPEX formulations (`mopex2`, `mopex4`, `mopex5`), creating unwanted structural over-representation.
- A deterministic revision rule was applied: remove `mopex2` and retain exactly one of `mopex4` or `mopex5` via same-quadrant replacements.
- Evaluated two deterministic scenarios:
  - **Scenario A (Keep `mopex5`):** $Q_3$ contrast `mopex2` $\rightarrow$ `us1`; $Q_4$ contrast `mopex4` $\rightarrow$ `tank`.
  - **Scenario B (Keep `mopex4`):** $Q_2$ representative `mopex5` $\rightarrow$ `newzealand2`; $Q_3$ contrast `mopex2` $\rightarrow$ `us1`.
- **Selection Decision:** Scenario B was adopted because it yielded substantially higher 7D minimum pairwise distance ($0.7986$ vs $0.5087$), higher mean pairwise distance ($1.0971$ vs $1.0807$), higher $R$-extreme coverage ($7/8$ vs $6/8$), and lower objective loss ($0.3472$ vs $0.5937$).
- Final 8 models: `alpine2`, `hbv96`, `xinanjiang`, `newzealand2`, `ihacres`, `us1`, `mopex4`, `hillslope`.

---

## 3. Classification of Selection Criteria

| Candidate Criterion | Classification | Pre-OOB Evidence Source | Description & Role in Selection |
|---|---|---|---|
| **Flexibility Gap ($G_{\text{seen}}$) Orthogonal Split** | `DOCUMENTED_PRE_OOB` | `select_models.py`, `OOB_MODEL_SELECTION_REPORT.md` (2026-09-01) | Primary quadrant stratification axis separating models where dPL matches IC ($G \le 0.0104$) from models where IC substantially outperforms dPL ($G > 0.0104$). |
| **Parameter Reproducibility ($R$) Orthogonal Split** | `DOCUMENTED_PRE_OOB` | `select_models.py`, `OOB_MODEL_SELECTION_REPORT.md` (2026-09-01) | Primary quadrant stratification axis separating models with high parameter–attribute alignment ($R \ge 0.7326$) from unaligned/compensatory models ($R < 0.7326$). |
| **4-Quadrant Orthogonal Coverage ($G \times R$)** | `DOCUMENTED_PRE_OOB` | `select_models.py`, `OOB_MODEL_SELECTION_QUADRANTS.csv` (2026-09-01) | Exactly 2 models assigned to each of the 4 regimes ($Q_1, Q_2, Q_3, Q_4$). |
| **Within-Quadrant 5D Contrast Maximization** | `DOCUMENTED_PRE_OOB` | `select_models.py` lines 48–60 (2026-09-01) | Selection of a second model per quadrant maximizing dispersion in parameter displacement ($D$), IC uncertainty ($U$), parameter count ($P$), and signal count ($A$). |
| **Performance Viability Gate ($K_{\text{joint}} \ge Q25$)** | `DOCUMENTED_PRE_OOB` | `selection_common.py` line 67, `select_models.py` line 44 (2026-09-01) | Excluded bottom 25% poorly performing models ($\text{KGE} < 0.561$) to ensure baseline simulation validity. |
| **Parameter Complexity Stratification ($P$)** | `DOCUMENTED_PRE_OOB` | `select_models.py` lines 79–115, `OOB_SELECTION_PROVENANCE.json` (2026-09-01) | Enforced representation across low ($P \le 6$), medium ($7 \le P \le 10$), and high ($P \ge 11$) parameter counts. |
| **Structural Anti-Redundancy (MOPEX Capping)** | `DOCUMENTED_PRE_OOB` | `revise_mopex_redundancy.py`, `OOB_MODEL_SELECTION_MOPEX_REDUNDANCY_REVISION.md` (2026-09-01) | Explicit rule limiting MOPEX family members to exactly 1 model to avoid lineage redundancy. |
| **Model Family / Conceptual Architecture Diversity** | `SUPPORTED_BY_PRE_OOB_ARTIFACTS_BUT_NOT_EXPLICITLY_STATED` | `structure_qc.py`, `OOB_STRUCTURAL_MODEL_DESCRIPTORS.csv` (2026-09-01) | Structural properties (runoff mechanisms, bucket vs infiltration excess, store counts) were tracked and checked pre-OOB, though not an explicit quantitative loss term. |
| **Inclusion of Canonical Benchmark Models** | `SUPPORTED_BY_PRE_OOB_ARTIFACTS_BUT_NOT_EXPLICITLY_STATED` | `OOB_MODEL_SELECTION_FEATURES.csv` (2026-09-01) | Widely used benchmark formulations (HBV, Xin'anjiang, IHACRES) emerged naturally from the quantitative criteria without manual override. |
| **Computational Feasibility (40-Job Execution Budget)** | `DOCUMENTED_PRE_OOB` | `future_oob_pur_training_protocol.md` (2026-08-29), `oob_primary8_5fold_20260902.yaml` (2026-09-02) | 8 models $\times$ 5 folds = 40 training runs on 531 basins was chosen as the tractable budget for full 5-fold cross-validation. |
| **OOB Held-Out Performance / OOB KGE Outcomes** | `NOT_SUPPORTED` | Provenance audit; OOB runs executed *after* config freeze | OOB results did not exist and played zero role in model selection. |
| **OOB Parameter Displacement / Information Retention** | `NOT_SUPPORTED` | Provenance audit; R4 relationship case freeze | OOB parameter metrics were computed strictly post-hoc. |

---

## 4. Key Takeaways for Manuscript Wording

1. **Selection Logic was Seen-Basin Centered:** The eight models were chosen specifically to span the two empirical axes discovered in seen-basin analyses R1 and R2 ($G_{\text{seen}}$ and $R_{\text{seen}}$), while maintaining contrast across parameter complexity ($P$), realization displacement ($D_{\theta}$), and IC parameter non-uniqueness ($U$).
2. **Deterministic and Non-Cherry-Picked:** The selection was entirely rule-based and deterministic. No manual cherry-picking was involved (`manual_override: false`).
3. **Targeted Formulation Coverage:** The design was structured as a targeted challenge across contrasting conceptual regimes, not a formal probability sample of the 36-model benchmark.
