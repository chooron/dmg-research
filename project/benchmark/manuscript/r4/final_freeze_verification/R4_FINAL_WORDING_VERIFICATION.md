# R4 Final Manuscript Wording Verification & Language Freeze

## 1. Executive Summary

This document conducts the final language and phrasing verification across all R4 manuscript handoff products. It establishes strict, non-causal, epistemically modest wording standards that prevent over-claims, accurately declare representation scopes, and distinguish primary from sensitivity cohorts.

---

## 2. High-Risk Language Audit & Enforced Replacements

| Flagged / High-Risk Phrase | Source Context | Audit Reason | Approved Replacement Phrasing |
|---|---|---|---|
| **"representative sample of the 36 models"** | Methods / Abstract / Results | False implication of random probability sampling. | **"a prespecified eight-model subset spanning contrasting conceptual model formulations"** *(or "prespecified stress-test panel")* |
| **"proving that regional dPL occupies the exact same separated parameter regime"** | R2 Results | Causal / deterministic over-claim. | **"confirming that out-of-bag dPL remains substantially separated from IC in normalized parameter space, with a displacement magnitude comparable to seen-basin dPL"** |
| **"dPL incurs only a modest performance penalty"** | R1 Results | Subjective, unquantified adjective. | **"uncalibrated regional dPL incurs an expected aggregate performance gap of $-0.065$ KGE relative to local calibration (a regionalization shift of $-0.058$ KGE relative to seen dPL)"** |
| **"exact replay of the frozen 20-D R3 estimand"** | R3 Methods / Results | Inaccurate dimensional label; R4 evaluates continuous variables. | **"evaluation across the prespecified continuous information representation (13 orthogonal continuous clusters and the full 32 continuous attribute space)"** |
| **"OOB parameter displacement exceeds IC restart dispersion across all models"** | R2 Results | Conflates primary 5-model eligible cohort with all 8 models. | **"out-of-bag parameter displacement strictly exceeded the archived IC multi-start calibration dispersion across all five eligible tested models meeting the 90% restart coverage requirement"** |
| **"coordinate-specific information organization is an intrinsic property"** | Discussion | Over-interprets mathematical continuity as an unalterable natural property. | **"coordinate-specific information organization remained reproducible under held-out application of the shared parameter-learning framework"** |
| **"persisted in completely uncalibrated catchments"** | Discussion / Abstract | Ambiguous (IC comparator uses local calibration). | **"persisted for catchments held out from dPL training"** |
| **"high seen-to-OOB correlation proves external generalization"** | R3 Results | Conflates neural parameterizer continuity with cross-paradigm generalization. | **"parameter–attribute relationships in out-of-bag dPL were highly continuous with seen-basin dPL ($\rho = 0.962$), confirming the numerical stability of the shared parameterizer across data partitions"** |

---

## 3. Strict Section-by-Section Frozen Text

### 3.1 Methods Section
> "To evaluate out-of-bag regionalization under strict held-out-basin conditions without prohibitive computational cost, we prespecified an eight-model subset (`alpine2`, `hbv96`, `xinanjiang`, `newzealand2`, `ihacres`, `us1`, `mopex4`, and `hillslope`) prior to out-of-bag execution. This subset was selected via a deterministic 4-quadrant rule across seen-basin performance flexibility ($G_{\text{seen}}$) and parameter reproducibility ($R_{\text{seen}}$), subject to a baseline performance viability gate ($K_{\text{joint}} \ge 0.561$) and an explicit structural anti-redundancy constraint that limited the MOPEX lineage to a single member.
>
> A 5-fold cross-validation partition across all 531 CAMELS-US basins was evaluated with train-only attribute normalization (40 completed runs in total, yielding 531 unique held-out basin evaluations per model). Catchment-information organization was evaluated across the prespecified 13 continuous information clusters and the full 32 continuous attribute space. Benchmark comparison against IC multi-start calibration dispersion was evaluated across the five tested models satisfying the frozen 90% restart coverage requirement."

### 3.2 Results: R1 Outlet Performance
> "Under strict 5-fold held-out-basin evaluation across the eight prespecified models, uncalibrated regional dPL achieved an ensemble median KGE of 0.581 (compared to 0.650 for seen dPL and locally calibrated IC). Transferring shared parameter mappings to ungauged catchments incurred an expected aggregate performance gap of $-0.065$ KGE relative to local calibration (a regionalization shift of $-0.058$ KGE relative to seen dPL). Crucially, place-specific response heterogeneity remained pronounced: out-of-bag dPL matched or exceeded locally calibrated IC in 26.1% of individual basin instances (ranging from 18.3% in `us1` to 39.0% in `newzealand2`). Two-way variance decomposition revealed that catchment difficulty (basin-additive effects) accounted for 67.1% of total performance-gap variance, compared to just 0.5% for model formulation, confirming that aggregate outlet metrics obscure substantial place-specific variation."

### 3.3 Results: R2 Parameter Realization
> "Held-out basin evaluation confirms that regional dPL parameters remain distinctly separated from independent calibration realizations: bounds-normalized parameter displacement between IC and regional dPL remains substantial across all eight models (median $D_{\text{RMS, OOB}} = 0.343$, compared to $0.314$ on seen basins). In the five tested models meeting the strict 90% restart coverage threshold (`alpine2`, `hillslope`, `ihacres`, `us1`, `xinanjiang`), this out-of-bag displacement strictly exceeded the archived IC multi-start calibration dispersion ($D_{\text{cross, OOB}} - D_{\text{self}} = +0.227$ under model-level subtraction, and $+0.212$ under basin-matched subtraction; positive in 5/5 models). Furthermore, regionalized parameters transferred across held-out folds exhibited minimal shift from seen-dPL parameter vectors (median shift of $0.100$), and the performance–parameter bridge was preserved out-of-bag, with catchment-level $|\Delta\text{KGE}_{\text{OOB}}|$ positively correlated with parameter displacement across all eight models (median Spearman $\rho = +0.242$)."

### 3.4 Results: R3 Catchment Information Organization
> "When challenged under strict 5-fold held-out-basin conditions, conceptual parameter–catchment information profiles directly replayed against independent calibration exhibited strong same-parameter correspondence (model-equal median $R_{\text{paired, OOB}} = 0.739$ across the 13 continuous information clusters, and $0.762$ across all 32 continuous attributes, matching the seen-basin benchmark of $0.713$ / $0.745$). Same-coordinate specificity was rigorously preserved: diagonal profile alignment markedly exceeded within-model off-diagonal alternatives ($A_{\text{diag, OOB}} = 0.718$ vs. off-diagonal median $-0.011$; within-model parameter-label permutation $p = 0.000999$), confirming that coordinate-level information organization persists for catchments held out from dPL training. Parameter–attribute relationships in out-of-bag dPL were highly continuous with seen-basin dPL ($\rho = 0.962$, 99.9% non-trivial sign retention), reflecting the numerical stability of the shared parameterizer across data partitions."

### 3.5 Discussion: Scope and Epistemic Boundaries
> "Several methodological boundaries govern the interpretation of the held-out-basin findings. First, the tested eight-model subset is an intentionally stratified stress-test panel designed to span contrasting conceptual formulations ($P \in [5, 15]$, $S \in [1, 5]$, snow vs. rain-dominant) and seen-basin response regimes, rather than an unbiased random sample of the 36-model benchmark. Consequently, held-out conclusions demonstrate the architectural transferability of shared parameter learning across diverse model structures, but should not be interpreted as an exhaustive 36-model population census.
>
> Second, while held-out retention confirms that coordinate-specific information organization remained reproducible under held-out application of the shared parameter-learning framework, it does not establish physical parameter truth, identifiability, or causal process linkages. Third, secondary parameter-geometry properties not prespecified for held-out re-computation (such as cross-catchment rank reorganization and coordinate localization) remain strictly scoped to the 36-model seen-basin benchmark."

---

## 4. Final Language Approval

All high-risk expressions have been systematically corrected and replaced with exact, non-causal, mathematically verified formulations.
