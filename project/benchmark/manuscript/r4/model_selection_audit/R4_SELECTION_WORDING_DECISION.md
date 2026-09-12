# R4 Model Selection Manuscript Wording Decision

## 1. Wording Tier Decision

### Selected Tier: **Tier 2 — Prespecified Spanning Subset**

### Wording Tier Assessment Matrix:

| Tier | Label | Evaluation Criteria | Audit Finding | Verdict |
|---|---|---|---|---|
| **Tier 1** | *Representative Subset* | Selection is prespecified AND provably representative (e.g. random probability sample or formal statistical surrogate of 36 models). | The subset is an intentionally stratified, contrast-maximizing panel with viability gating, not an IID random sample. | **REJECTED** |
| **Tier 2** | *Prespecified Spanning Subset* | Selection is demonstrably prespecified and non-test-informed, with verified coverage across contrasting conceptual formulations, but formal population representativeness is not claimed. | Fully proven by pre-OOB configs (`2026-09-02 09:23:18`), deterministic scripts, structural coverage ($P \in [5, 15], S \in [1, 5]$), and 4-quadrant $G \times R$ spanning. | **ADOPTED** |
| **Tier 3** | *Tested Subset* | Selection is non-test-informed, but prespecification timestamp cannot be verified. | Prespecification is rigorously verified with exact timestamps and checksums. A weaker claim is unnecessarily pessimistic. | **SUPERSEDED BY TIER 2** |
| **Tier 4** | *Selection Compromised* | Concrete evidence that OOB outcomes influenced model inclusion/exclusion. | 0 job failures, 0 retries, 0 post-hoc substitutions. Exact 40/40 run parity. | **CONTRADICTED / REFUTED** |

---

## 2. Canonical Manuscript Phrasing

### Recommended Primary Phrase:
> **"a prespecified eight-model subset spanning contrasting conceptual model formulations"**

### Permissible Contextual Variations:
- *"the prespecified eight-model panel"*
- *"an eight-model subset structured to span contrasting flexibility gaps and parameter reproducibility regimes"*
- *"the tested eight-model subset"*

### Strictly Forbidden Phrasing:
- ❌ *"a representative sample of the 36-model benchmark"* (Statistically misleading; implies unbiased population inference)
- ❌ *"the eight most representative models"* (Implies an optimal statistical surrogate)
- ❌ *"models selected based on their generalization capability"* (Factually false; violates non-test-informed protocol)

---

## 3. Justification for Tier 2 Adoption

1. **Prespecification is Cryptographically and Temporally Anchored:** The exact model list was frozen in `project/benchmark/configs/oob_primary8_5fold_20260902.yaml` (SHA256: `1e9e4e03...`) on 2026-09-02 09:23:18 +0800, prior to deployment (10:58:35 +0800) and before the first completed OOB run (12:08:08 +0800).
2. **Zero Information Leakage:** Selection rules relied solely on pre-existing seen-basin calibration metrics ($G_{\text{seen}}, R_{\text{seen}}, D_{\theta}, U, K_{\text{joint}}, P, A$). Zero OOB held-out metrics were available or used.
3. **Substantial Spanning Coverage:** The eight models provide rigorous 2-model orthogonal coverage across all four $G \times R$ quadrants, span parameter counts from 5 to 15, state dimensionality from 1 to 5, and incorporate contrasting process concepts (snow melt, saturation excess, unit hydrographs, multi-layer conceptual stores, and hillslope routing).
4. **Appropriate Epistemic Modesty:** Rejecting "representative" protects the manuscript against hostile reviewer critique regarding sample bias or over-generalization, while accurately reflecting the deliberate stress-test design of R4.
