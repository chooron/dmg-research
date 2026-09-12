# R4 Out-of-Bag Parameter Separation vs. Archived IC-Self Reference Audit

## 1. Executive Summary & Verdict

### Final IC-Self Replay Verdict: **`A. SEPARATION EXCEEDS ARCHIVED IC-SELF REFERENCE IN ELIGIBLE MODELS`**

- **Audit Question:** Does the parameter realization displacement observed between independent calibration (IC) and regional out-of-bag parameter learning (OOB-dPL) exceed the multi-start calibration dispersion of IC ($D_{\text{self}}$) under the frozen R2 strict eligibility contract?
- **Key Findings:**
  1. **Strict Eligibility Audit (5 / 8 Models Pass):** Following the frozen R2 contract (requiring $\ge 90\%$ of catchments to possess alternative multi-start restarts within $\Delta\text{KGE} \le 0.01$ of the basin best without fallback), exactly 5 of the 8 tested models are strictly eligible: `alpine2` (97.7%), `hillslope` (95.7%), `ihacres` (95.1%), `us1` (96.2%), and `xinanjiang` (90.6%). Three models (`hbv96` 82.3%, `mopex4` 89.1%, `newzealand2` 67.2%) fall into the frozen `INSUFFICIENT_REFERENCE` tier.
  2. **Strict IC-Self Replay in Eligible Models:** Across the 5 eligible models, OOB parameter separation firmly exceeds the archived multi-start dispersion:
     - Median $D_{\text{cross, OOB}} = \mathbf{0.329326}$ vs. Median $D_{\text{self}} = \mathbf{0.032632}$
     - Model-equal median difference: **$D_{\text{cross, OOB}} - D_{\text{self}} = \mathbf{+0.227033}$**
     - Exceedance is strictly positive across **5 / 5** eligible models (100%).
  3. **Robustness Across All 8 Models:** Even when evaluated across all 8 models without coverage gating, $D_{\text{cross, OOB}} - D_{\text{self}}$ is strictly positive across **8 / 8** models (model-equal median difference **$+0.193436$**).
  4. **Manuscript Formulation:** The manuscript can legitimately claim that parameter realization separation under out-of-bag regionalization exceeds calibration restart uncertainty in the eligible tested models, while transparently scoping the 3 insufficient models.

---

## 2. Strict Eligibility Audit Table for 8 Tested Models

| Model | Free Parameters ($P$) | Valid Basins ($\Delta\text{KGE} \le 0.01$) | Primary Coverage (%) | R2 Strict Eligibility Status | Primary Reason |
|---|---:|---:|---:|---|---|
| **alpine2** | 6 | 519 / 531 | 97.74% | **ELIGIBLE (PASS)** | $\ge 90\%$ multi-start coverage |
| **hbv96** | 15 | 437 / 531 | 82.30% | **INSUFFICIENT_REFERENCE** | High dimension; $<90\%$ near-optimal restarts |
| **hillslope** | 7 | 508 / 531 | 95.67% | **ELIGIBLE (PASS)** | $\ge 90\%$ multi-start coverage |
| **ihacres** | 6 | 505 / 531 | 95.10% | **ELIGIBLE (PASS)** | $\ge 90\%$ multi-start coverage |
| **mopex4** | 10 | 473 / 531 | 89.08% | **INSUFFICIENT_REFERENCE** | Multi-store equifinality; 89.1% $< 90\%$ |
| **newzealand2** | 8 | 357 / 531 | 67.23% | **INSUFFICIENT_REFERENCE** | Multi-modal landscape; 67.2% $< 90\%$ |
| **us1** | 5 | 511 / 531 | 96.23% | **ELIGIBLE (PASS)** | $\ge 90\%$ multi-start coverage |
| **xinanjiang** | 12 | 481 / 531 | 90.58% | **ELIGIBLE (PASS)** | $\ge 90\%$ multi-start coverage |

---

## 3. Quantitative Replay Results: OOB Parameter Separation vs. IC-Self

### Table B1: Per-Model Separation and Restart Excess

| Model | Eligibility | $D_{\text{cross, seen}}$ (Median) | $D_{\text{cross, OOB}}$ (Median) | $D_{\text{self}}$ (Median) | Seen ($D_{\text{cross}} - D_{\text{self}}$) | OOB ($D_{\text{cross}} - D_{\text{self}}$) | Positive? |
|---|---|---:|---:|---:|---:|---:|---|
| **alpine2** | **Eligible** | 0.265767 | 0.209835 | 0.123901 | +0.150014 | **+0.085934** | YES |
| **hillslope** | **Eligible** | 0.403161 | 0.329326 | 0.032632 | +0.310600 | **+0.296693** | YES |
| **ihacres** | **Eligible** | 0.388615 | 0.343291 | 0.000407 | +0.331060 | **+0.342883** | YES |
| **us1** | **Eligible** | 0.174166 | 0.227033 | 0.000000 | +0.150719 | **+0.227033** | YES |
| **xinanjiang** | **Eligible** | 0.330847 | 0.342482 | 0.182642 | +0.153328 | **+0.159840** | YES |
| *hbv96* | *Insufficient* | 0.388408 | 0.361400 | 0.290424 | +0.103163 | *+0.070977* | YES |
| *mopex4* | *Insufficient* | 0.470367 | 0.451681 | 0.309482 | +0.167042 | *+0.142198* | YES |
| *newzealand2* | *Insufficient* | 0.372016 | 0.396316 | 0.044361 | +0.249574 | *+0.351955* | YES |

---

### Table B2: Summary by Eligibility Cohort

| Population | $N$ Models | $D_{\text{cross, OOB}}$ (Median) | $D_{\text{self}}$ (Median) | OOB ($D_{\text{cross}} - D_{\text{self}}$) | Positive Share | Full 36 Reference |
|---|---:|---:|---:|---:|---:|---|
| **Eligible Models (Strict R2)** | **5** | **0.329326** | **0.032632** | **+0.227033** | **5 / 5 (100%)** | $+0.218775$ (23 models) |
| **All Tested Models (Unfiltered)** | 8 | 0.342886 | 0.084131 | +0.193436 | 8 / 8 (100%) | $+0.193$ (36 models) |

---

## 4. Scientific Interpretation and Manuscript Bounds

1. **Parameter Separation is Real and Beyond Optimization Noise:** The parameter displacement between IC and regional dPL in held-out basins cannot be explained away as multi-start calibration jitter. In the eligible tested models, the median distance from IC canonical to OOB-dPL ($0.329$) is an order of magnitude larger than the internal restart spread of IC ($0.033$), yielding a net excess displacement of **$+0.227$** (closely matching the full 23-model seen benchmark of $+0.219$).
2. **Explicit Scope Qualification:** The manuscript must explicitly distinguish the 5 strictly eligible models (`alpine2`, `hillslope`, `ihacres`, `us1`, `xinanjiang`) from the 3 models with insufficient restart coverage (`hbv96`, `mopex4`, `newzealand2`), in exact alignment with the frozen R2 benchmark contract.

---

## 5. Manuscript-Ready Statements

- **Results R2 Statement:**
  > *"Out-of-bag evaluation confirms the core parameter separation discovered in R2: bounds-normalized parameter displacement between IC and regional dPL remains substantial across all eight models (median $D_{\text{RMS, OOB}} = 0.343$), and strictly exceeds the archived IC multi-start calibration dispersion across all five eligible tested models (median excess displacement $D_{\text{cross, OOB}} - D_{\text{self}} = +0.227$, positive in 5/5 models)."*
- **Scope Boundary:**
  > *"Comparison against IC multi-start dispersion is strictly restricted to the five models satisfying the frozen 90% restart coverage threshold, while the remaining three models exhibit substantial raw parameter separation."*
