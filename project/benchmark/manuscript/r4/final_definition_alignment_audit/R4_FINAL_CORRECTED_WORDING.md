# R4 Final Corrected Manuscript Text & Phrase Guidelines

This document provides the definitive, audit-corrected manuscript text for all sections of the paper touching R4, replacing any preliminary or imprecise statements.

---

## 1. Section 2: Methods (Section 2.X — Out-of-Bag Regionalization Protocol)

> **"To evaluate out-of-bag regionalization under strict held-out-basin conditions without prohibitive computational cost, we prespecified an eight-model subset (`alpine2`, `hbv96`, `xinanjiang`, `newzealand2`, `ihacres`, `us1`, `mopex4`, and `hillslope`) prior to out-of-bag execution. This subset was selected via a deterministic 4-quadrant rule across seen-basin performance flexibility ($G_{\text{seen}}$) and parameter reproducibility ($R_{\text{seen}}$), subject to a baseline performance viability gate ($K_{\text{joint}} \ge 0.561$) and an explicit structural anti-redundancy constraint that limited the MOPEX lineage to a single member.**
>
> **A 5-fold cross-validation partition across all 531 CAMELS-US basins was generated using a fixed random seed. In each fold, shared parameter learning (dPL) was trained exclusively on the 80% training basins (including train-only attribute normalization), and evaluated on the strictly held-out 20% test basins without site-specific calibration (40 completed runs in total, yielding 531 unique held-out basin evaluations per model). Catchment-information organization was evaluated across the prespecified 13 continuous information clusters and the full 32 continuous attribute space. Benchmark comparison against IC multi-start calibration dispersion was evaluated across the five tested models satisfying the frozen 90% restart coverage requirement."**

---

## 2. Section 3: Results — R1 Held-Out Replay (Section 3.X)

> **"Under strict 5-fold held-out-basin evaluation across the eight prespecified models, uncalibrated regional dPL achieved an ensemble median KGE of 0.581 (compared to 0.650 for seen dPL and locally calibrated IC). Transferring shared parameter mappings to ungauged catchments incurred an expected aggregate performance gap of $-0.065$ KGE relative to local calibration (a regionalization shift of $-0.058$ KGE relative to seen dPL). Crucially, place-specific response heterogeneity remained pronounced: out-of-bag dPL matched or exceeded locally calibrated IC in 26.1% of individual basin instances (ranging from 18.3% in `us1` to 39.0% in `newzealand2`). Two-way variance decomposition revealed that catchment difficulty (basin-additive effects) accounted for 67.1% of total performance-gap variance, compared to just 0.5% for model formulation, confirming that aggregate outlet metrics obscure substantial place-specific variation."**

---

## 3. Section 3: Results — R2 Held-Out Replay (Section 3.X)

> **"Held-out basin evaluation confirms that regional dPL parameters remain distinctly separated from independent calibration realizations: bounds-normalized parameter displacement between IC and regional dPL remains substantial across all eight models (median $D_{\text{RMS, OOB}} = 0.343$, compared to $0.314$ on seen basins). In the five tested models meeting the strict 90% restart coverage threshold (`alpine2`, `hillslope`, `ihacres`, `us1`, `xinanjiang`), this out-of-bag displacement strictly exceeded the archived IC multi-start calibration dispersion ($D_{\text{cross, OOB}} - D_{\text{self}} = +0.227$, positive in 5/5 models). Furthermore, regionalized parameters transferred across held-out folds exhibited minimal shift from seen-dPL parameter vectors (median shift of $0.100$), and the performance–parameter bridge was preserved out-of-bag, with catchment-level $|\Delta\text{KGE}_{\text{OOB}}|$ positively correlated with parameter displacement across all eight models (median Spearman $\rho = +0.242$)."**

---

## 4. Section 3: Results — R3 Held-Out Replay (Section 3.X)

> **"When challenged under strict 5-fold held-out-basin conditions, conceptual parameter–catchment information profiles directly replayed against independent calibration exhibited strong same-parameter correspondence (model-equal median $R_{\text{paired, OOB}} = 0.739$ across the 13 continuous information clusters, and $0.762$ across all 32 continuous attributes, matching the seen-basin benchmark of $0.713$ / $0.745$). Same-coordinate specificity was rigorously preserved: diagonal profile alignment markedly exceeded within-model off-diagonal alternatives ($A_{\text{diag, OOB}} = 0.718$ vs. off-diagonal median $-0.011$; within-model parameter-label permutation $p = 0.000999$), confirming that coordinate-level information organization persists in completely uncalibrated catchments. Parameter–attribute relationships in out-of-bag dPL were highly continuous with seen-basin dPL ($\rho = 0.962$, 99.9% non-trivial sign retention), reflecting the numerical stability of the shared parameterizer across data partitions."**

---

## 5. Section 4: Discussion (Scope and Limitations)

> **"Several methodological boundaries govern the interpretation of the held-out-basin findings. First, the tested eight-model subset is an intentionally stratified stress-test panel designed to span contrasting conceptual formulations ($P \in [5, 15]$, $S \in [1, 5]$, snow vs. rain-dominant) and seen-basin response regimes, rather than an unbiased random sample of the 36-model benchmark. Consequently, held-out conclusions demonstrate the architectural transferability of shared parameter learning across diverse model structures, but should not be interpreted as an exhaustive 36-model population census.**
>
> **Second, while held-out retention confirms that coordinate-specific information organization is an intrinsic property of shared parameter learning, it does not establish physical parameter truth, identifiability, or causal process linkages. Third, secondary parameter-geometry properties not prespecified for held-out re-computation (such as cross-catchment rank reorganization and coordinate localization) remain strictly scoped to the 36-model seen-basin benchmark."**

---

## 6. Prohibited vs. Approved Phrase Crosswalk

| Preliminary / Unsafe Phrasing | Corrected / Audit-Approved Phrasing | Scientific Rationale |
|---|---|---|
| *"a representative eight-model subset"* | *"a prespecified eight-model subset spanning contrasting conceptual formulations"* | Not an IID probability sample; stratified contrast panel. |
| *"proving that regional dPL occupies the exact same separated parameter regime"* | *"OOB-dPL remained substantially separated from IC in normalized parameter space, with a displacement magnitude comparable to seen-basin dPL"* | Epistemic modesty; avoids causal over-claim. |
| *"dPL incurs only a modest performance penalty"* | *"uncalibrated regional dPL incurs an aggregate performance gap of $-0.065$ KGE relative to local calibration"* | Quantitatively precise; replaces subjective adjective. |
| *"the 20-dimensional information space was replayed on OOB"* | *"information organization was evaluated across the prespecified 13 continuous information clusters and 32 continuous attributes"* | Exact dimensional crosswalk; avoids misrepresenting continuous representation as 20-D PC1. |
| *"OOB parameter displacement exceeds IC restart dispersion across all models"* | *"OOB parameter displacement strictly exceeds IC multi-start calibration dispersion across all five eligible tested models"* | Strictly respects the frozen 90% restart coverage threshold. |
| *"high seen-to-OOB correlation proves external generalization"* | *"parameter–attribute relationships in OOB-dPL were highly continuous with seen-basin dPL ($\rho = 0.962$), reflecting parameterizer stability"* | Distinguishes constructive neural continuity from the primary cross-paradigm IC $\leftrightarrow$ OOB challenge. |
