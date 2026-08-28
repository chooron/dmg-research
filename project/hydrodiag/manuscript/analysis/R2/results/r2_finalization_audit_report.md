# Results 3.2 (R2) Finalization Audit Report

> **Audit Status:** FINALIZED  
> **Finalization Verdict:** `R2_FINALIZATION_STATUS = READY`  
> **Scope:** Shared calibrated parameter-space response to snow-process omission (Base vs CN vs TGD control across IC and dPL)

---

## 1. Prevalence Definition Conflict Provenance & Canonical Resolution

### 1.1 Root-Cause Provenance of Historical Discrepancies
- **Legacy IC values (S1 ≈ 49.7%, S5 ≈ 81.8%):** Generated in early exploratory scripts that evaluated single restart pairs or unpooled within-structure baselines rather than the full 10-restart combinatorial ensemble.
- **Legacy dPL values (S1 ≈ 80.0%, S5 = 100%):** Generated in early scripts where within-seed dispersion was ignored (assumed zero within baseline) or checked against a fixed arbitrary scalar ($D > 0.08$).
- **Canonical Estimand 4B (Strict & Frozen):**
  - **Inference / Reporting Unit:** Basin $b$ ($N=531$).
  - **Within-Structure Baseline:**
    - IC: $w_{\text{base}, b} = \text{median}_{45\text{ pairs}} \|z_i - z_j\|_{\text{RMS}}$, $w_{\text{cn}, b} = \text{median}_{45\text{ pairs}} \|z_i - z_j\|_{\text{RMS}}$, $w_{\text{pooled}, b} = (w_{\text{base}, b} + w_{\text{cn}, b})/2$.
    - dPL: $w_{\text{base}, b} = \text{median}_{3\text{ pairs}} \|z_i - z_j\|_{\text{RMS}}$, $w_{\text{cn}, b} = \text{median}_{3\text{ pairs}} \|z_i - z_j\|_{\text{RMS}}$, $w_{\text{pooled}, b} = (w_{\text{base}, b} + w_{\text{cn}, b})/2$.
  - **Between-Structure Separation:**
    - IC: $b_{\text{all}, b} = \text{median}_{100\text{ cross pairs}} \|z_{\text{base}, i} - z_{\text{cn}, j}\|_{\text{RMS}}$.
    - dPL: $b_{\text{all}, b} = \text{median}_{9\text{ cross pairs}} \|z_{\text{base}, i} - z_{\text{cn}, j}\|_{\text{RMS}}$.
  - **Prevalence Definition:** Proportion of basins where $b_{\text{all}, b} > w_{\text{pooled}, b} \iff \text{excess}_b > 0$.

### 1.2 Frozen Canonical Prevalence Table (Estimand 4B)

| Stratum / Subset | n | IC Base–CN Prevalence [95% CI] | IC Base–TGD Prevalence [95% CI] | dPL Base–CN Prevalence [95% CI] | dPL Base–TGD Prevalence [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full531** | 531 | **63.1%** [59.3%, 66.9%] | 62.5% [58.4%, 66.5%] | **83.8%** [81.2%, 86.8%] | 77.0% [73.4%, 80.6%] |
| **ExcludeS5** | 476 | **59.0%** [55.1%, 63.2%] | 60.1% [55.7%, 64.3%] | **83.4%** [80.3%, 86.6%] | 76.5% [72.5%, 80.0%] |
| **S1** ($f_{\text{snow}} < 0.05$) | 165 | **46.7%** [39.4%, 54.3%] | 47.9% [40.0%, 55.8%] | **70.3%** [63.6%, 77.0%] | 66.1% [58.8%, 73.3%] |
| **S2** ($f_{\text{snow}} \in [0.05, 0.15)$) | 156 | **52.6%** [44.2%, 59.6%] | 53.2% [45.5%, 60.9%] | **87.2%** [82.1%, 92.3%] | 78.2% [71.8%, 84.6%] |
| **S3** ($f_{\text{snow}} \in [0.15, 0.30)$) | 121 | **74.4%** [67.3%, 81.8%] | 76.0% [68.6%, 83.5%] | **92.6%** [88.0%, 96.7%] | 86.0% [79.3%, 91.7%] |
| **S4** ($f_{\text{snow}} \in [0.30, 0.50)$) | 34 | **94.1%** [85.3%, 100.0%] | 88.2% [76.5%, 97.1%] | **97.1%** [91.2%, 100.0%] | 85.3% [73.5%, 97.1%] |
| **S5** ($f_{\text{snow}} \ge 0.50$) | 55 | **98.2%** [94.5%, 100.0%] | 83.6% [74.5%, 92.7%] | **87.3%** [78.2%, 94.5%] | 81.8% [70.9%, 90.9%] |

---

## 2. Direct Basin-Paired Base–CN vs Base–TGD Macro Excess Contrast

For every basin $b \in \{1 \dots 531\}$:
$$\delta\text{\_excess}_b = \text{excess}(\text{Base-CN})_b - \text{excess}(\text{Base-TGD})_b$$

### 2.1 Paired Contrast Summary Table

| Stratum / Subset | n | IC Median $\delta\text{\_excess}$ [95% CI] | IC Prop($>0$) [95% CI] | dPL Median $\delta\text{\_excess}$ [95% CI] | dPL Prop($>0$) [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full531** | 531 | $-0.0014$ [$-0.0034$, $+0.0011$] | 48.2% [43.9%, 52.4%] | **$+0.0177$** [$+0.0123$, $+0.0233$] | **64.6%** [60.5%, 68.7%] |
| **ExcludeS5** | 476 | $-0.0014$ [$-0.0032$, $+0.0010$] | 47.9% [43.5%, 52.3%] | **$+0.0160$** [$+0.0103$, $+0.0219$] | **63.9%** [59.5%, 68.3%] |
| **S1** | 165 | $-0.0046$ [$-0.0091$, $+0.0005$] | 44.2% [37.0%, 51.5%] | $+0.0047$ [$-0.0008$, $+0.0091$] | 56.4% [49.1%, 63.6%] |
| **S2** | 156 | $-0.0009$ [$-0.0032$, $+0.0027$] | 48.7% [41.0%, 56.4%] | **$+0.0287$** [$+0.0153$, $+0.0369$] | **69.2%** [62.2%, 76.3%] |
| **S3** | 121 | $-0.0013$ [$-0.0046$, $+0.0020$] | 46.3% [37.2%, 55.4%] | **$+0.0239$** [$+0.0142$, $+0.0363$] | **65.3%** [57.0%, 73.6%] |
| **S4** | 34 | $+0.0155$ [$+0.0012$, $+0.0205$] | 67.6% [52.9%, 82.4%] | **$+0.0363$** [$+0.0072$, $+0.1025$] | **70.6%** [55.9%, 85.3%] |
| **S5** | 55 | $+0.0018$ [$-0.0243$, $+0.0219$] | 50.9% [38.2%, 63.6%] | **$+0.0317$** [$+0.0065$, $+0.0501$] | **70.9%** [58.2%, 81.8%] |

---

## 3. Reconciliation with Slope-Based $\Delta\beta$ & Scientific Verdict

### 3.1 Verdict Selection
**VERDICT: A — Differentiation mainly emerges at intermediate snow activity (S2/S3) and persists into high snow activity (S4/S5).**

### 3.2 Empirical Justification
1. **Intermediate Emergence (S2/S3):** Under dPL, paired $\delta\text{\_excess}$ is already clearly positive at S2 ($+0.0287$, 69.2% $>0$) and S3 ($+0.0239$, 65.3% $>0$), well before reaching the highest snow strata.
2. **Mutual Saturation Plateau at S5:** 
   - Base–CN median excess plateaus in S4/S5 (S4: 0.1322, S5: 0.1252).
   - Base–TGD median excess also plateaus in S4/S5 (S4: 0.1070, S5: 0.0851).
   - Because both responses saturate, S5 does not generate differentiation; it maintains the gap established across S2–S4.
3. **Slope Difference $\Delta\beta$ Reconciliation:**
   - Full531 paired $\Delta\beta = +0.0411$ [$+0.008$, $+0.077$].
   - ExcludeS5 paired $\Delta\beta = +0.0861$ [$+0.017$, $+0.157$].
   - Removing the leveraged S5 plateau increases the slope difference precisely because the linear response across the transitional range (S1 $\to$ S4) is steeper for Base–CN than Base–TGD.
4. **IC Contrast:** Under IC, Base–CN and Base–TGD exhibit virtually identical macro excess across S1–S3 ($\delta\text{\_excess} \approx 0$). Differentiation appears only in S4 ($+0.0155$) and widens in variance at S5, reflecting that unconstrained local optimization readily finds generic surrogate minima.

---

## 4. Frozen R2 Evidence Hierarchy

```text
[Tier 1: Primary Macro Evidence]
  │  Base–CN whole-parameter-space separation (Figure 3a, b)
  │  Organization along snow activity: IC monotonic, dPL ordered-nonlinear
  ▼
[Tier 2: Supporting Identification Evidence]
  │  Between-structure separation relative to regime-specific within-structure variability
  │  Prevalence P(between > within): IC 63.1%, dPL 83.8% (Figure 3c, d)
  ▼
[Tier 3: Attribution Control Layer]
  │  Base–CN vs Base–TGD macro response comparison
  │  Paired delta_excess and paired Delta_beta (+0.0411 Full / +0.0861 Excl-S5) (Figure 3e, f)
  ▼
[Tier 4: Primary Explanatory Layer]
  │  All-15 signed parameter shifts (Figure 4)
  │  Illustrative recurring directional shifts: um (+), ki (-), ci (-), im (secondary -)
  ▼
[Tier 5: Constraint-Regime Evidence]
     IC (per-basin CMA-ES) vs dPL (regionalized MLP) analyzed in parallel without ranking
```

---

## 5. Manuscript Claim Boundaries

### A. Strongly Supported Claims (Permitted)
- Structural omission of snow processes leaves systematic, structure-associated reorganization in the shared calibrated parameter space.
- The whole-parameter-space macro response is environmentally organized by snow activity in both parameter-estimation regimes.
- IC exhibits a progressive, near-monotonic response across snow strata S1 $\to$ S5 (excess: $-0.002 \to +0.083$).
- dPL exhibits an ordered, nonlinear response that rises from low to moderate snow activity and plateaus across S3–S5 (excess: $+0.019 \to +0.125$).
- Generic temperature-conditioned memory (TGD) reproduces part of the macro response, but under dPL, Base–CN shows additional specific separation emerging at intermediate snow activity (S2/S3).
- Parameter-estimation constraints govern how the structural response is expressed (local compensatory freedom vs regionalized regularization).

### B. Supported Only with Qualification (Requires Nuance)
- **CN–TGD differentiation under dPL:** Must note that differentiation emerges at intermediate snow activity (S2/S3) and that S5 represents a mutual saturation plateau.
- **Low-snow negative condition:** In S1, IC Base–CN excess is $-0.0018$ and prevalence is $46.7\%$ (baseline equivalence); in dPL, S1 excess is small ($+0.0186$) but non-zero, reflecting mild regionalized inductive bias.
- **Parameter shifts ($um, ki, ci, im$):** Discussed purely as illustrative recurring directional shifts, not as universal mechanistic substitutes.
- **Full531 vs ExcludeS5 $\Delta\beta$:** ExcludeS5 slope difference ($+0.0861$) unmasks the linear transition phase by removing the high-snow plateau.

### C. Prohibited / Unsupported Claims (Strictly Forbidden)
- **NO parameter truth or distortion claims:** R2 is real-catchment empirical calibration; there is no synthetic ground truth.
- **NO parameter-substitution claims:** Do not claim $um, ki, ci$ "directly replace snow storage or melt".
- **NO regime superiority claims:** Do not rank dPL over IC or vice versa.
- **NO complete attribution claims:** Do not claim the CN–TGD residual is the "exact irreducible contribution of snow physics".
- **NO structural gap closure quantification:** Gap closure metrics belong strictly to synthetic R3 experiments.

---

## 6. Stop-Rule Decision & Verdict

$$\mathbf{R2\_FINALIZATION\_STATUS = READY}$$

**Final Determination:**
> No additional R2 statistical analysis, parameter screening, regressions, or model simulations are required before figure finalization (Figures 3 & 4) and Results 3.2 drafting. All canonical tables, provenance chains, and gates are 100% frozen and verified.
