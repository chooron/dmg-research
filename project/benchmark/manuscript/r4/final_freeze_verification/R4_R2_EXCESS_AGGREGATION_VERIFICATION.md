# R4 Out-of-Bag Parameter Excess: Aggregation Hierarchy & Numeric Verification

## 1. Executive Summary & Audit Verdict

### Final Verification Verdict: **`A. VERIFIED — +0.227033 IS A MODEL-MATCHED EXCESS ESTIMAND`**

- **The Mathematical Question:**
  In the R4 out-of-bag (OOB) parameter separation report, why does:
  $$\text{Median } D_{\text{cross, OOB}} = 0.329326, \quad \text{Median } D_{\text{self}} = 0.032632, \quad \text{Reported Excess} = \mathbf{+0.227033}$$
  when simple scalar subtraction yields $0.329326 - 0.032632 = 0.296693$?
- **The Finding:**
  `+0.227033` is **not** an arithmetic subtraction of grand medians. It is the **median across models of the model-specific differences** ($\text{median}_m(D_{\text{cross}, m} - D_{\text{self}, m})$) across the 5 strictly eligible models under the frozen R2 contract:
  - `alpine2`: $0.209835 - 0.123901 = +0.085934$
  - `xinanjiang`: $0.342482 - 0.182642 = +0.159840$
  - `us1`: $0.227033 - 0.000000 = \mathbf{+0.227033}$ *(median of the 5 values)*
  - `hillslope`: $0.329326 - 0.032632 = +0.296693$
  - `ihacres`: $0.343291 - 0.000407 = +0.342883$
  Because the median is a non-linear operator, $\text{median}_m(A_m - B_m) \neq \text{median}_m(A_m) - \text{median}_m(B_m)$.
- **Alternative Basin-Matched Hierarchy:**
  If excess displacement is computed at the basin level first ($E_{m, b} = D_{\text{cross}, m, b} - D_{\text{self}, m, b}$) before taking the basin median and then the model median, the result is **$+0.212135$** (positive in **5 / 5** eligible models, and positive in $71.9\%$ to $98.3\%$ of basins within each model).
- **Conclusion:** Both aggregation hierarchies confirm that out-of-bag parameter displacement strictly exceeds the archived IC multi-start calibration dispersion across **100% of eligible tested models** (excess of $\approx +0.21$ to $+0.23$).

---

## 2. Aggregation Hierarchy Taxonomy

To ensure complete transparency in the manuscript and avoid reader confusion, we define the three mathematical aggregation hierarchies:

### Case 1: Model-Summary Difference Median (Reported Value: `+0.227033`)
1. Compute the basin median parameter displacement for each model:
   $$D_{\text{cross}, m} = \text{median}_b\left(D_{\text{cross, OOB}}(m, b)\right)$$
2. Compute the basin median multi-start calibration dispersion for each model:
   $$D_{\text{self}, m} = \text{median}_b\left(D_{\text{self}}(m, b)\right)$$
3. Compute the model-specific difference:
   $$E_m = D_{\text{cross}, m} - D_{\text{self}, m}$$
4. Take the equal-weight median across the $M=5$ eligible models:
   $$E_{\text{Case 1}} = \text{median}_{m \in \text{Eligible}}(E_m) = \mathbf{+0.227033}$$

---

### Case 2: Basin-Matched Excess Median (Authoritative Basin Hierarchy: `+0.212135`)
1. For each basin $b$ within model $m$, subtract the local multi-start dispersion from the local displacement:
   $$E_{m, b} = D_{\text{cross, OOB}}(m, b) - D_{\text{self}}(m, b)$$
2. Compute the basin median excess for each model:
   $$E_{\text{basin-matched}, m} = \text{median}_b\left(E_{m, b}\right)$$
3. Take the equal-weight median across the $M=5$ eligible models:
   $$E_{\text{Case 2}} = \text{median}_{m \in \text{Eligible}}\left(E_{\text{basin-matched}, m}\right) = \mathbf{+0.212135}$$

---

### Case 3: Difference of Grand Model Medians (`+0.296693`)
1. Take the grand median of $D_{\text{cross}, m}$ across the 5 models: $\bar{D}_{\text{cross}} = 0.329326$ (from `hillslope`).
2. Take the grand median of $D_{\text{self}, m}$ across the 5 models: $\bar{D}_{\text{self}} = 0.032632$ (from `hillslope`).
3. Subtract the two grand medians:
   $$\Delta_{\text{Case 3}} = \bar{D}_{\text{cross}} - \bar{D}_{\text{self}} = 0.329326 - 0.032632 = \mathbf{+0.296693}$$

---

## 3. Independent Recomputation Table (5 Eligible Models)

| Model | Restart Coverage (%) | $D_{\text{cross, OOB}, m}$ | $D_{\text{self}, m}$ | Case 1: Model-Diff Excess ($E_m$) | Case 2: Basin-Matched Excess ($E_{\text{basin-matched}, m}$) | Basins with $E_{m, b} > 0$ (%) |
|---|---:|---:|---:|---:|---:|---:|
| **alpine2** | 97.74% | 0.209835 | 0.123901 | +0.085934 | +0.082201 | 71.94% |
| **xinanjiang** | 90.58% | 0.342482 | 0.182642 | +0.159840 | +0.147449 | 82.67% |
| **us1** | 96.23% | 0.227033 | 0.000000 | **+0.227033** | **+0.212135** | 98.31% |
| **hillslope** | 95.67% | 0.329326 | 0.032632 | +0.296693 | +0.227377 | 87.76% |
| **ihacres** | 95.10% | 0.343291 | 0.000407 | +0.342883 | +0.229192 | 90.21% |
| **5-Model Median** | **95.67%** | **0.329326** | **0.032632** | **+0.227033** | **+0.212135** | **87.76%** |

*All 5 eligible models exhibit strictly positive excess displacement under both Case 1 and Case 2.*

---

## 4. Sensitivity Cohort: All 8 Tested Models

For completeness, we evaluate the remaining 3 models pre-classified as `INSUFFICIENT_REFERENCE` in frozen R2 ($<90\%$ multi-start coverage):

| Model | Restart Coverage (%) | R2 Status | $D_{\text{cross, OOB}, m}$ | $D_{\text{self}, m}$ | Case 1: Model-Diff Excess | Case 2: Basin-Matched Excess | Basins with $E_{m, b} > 0$ (%) |
|---|---:|---|---:|---:|---:|---:|---:|
| *hbv96* | 82.30% | *Insufficient* | 0.361400 | 0.290424 | +0.070977 | +0.083581 | 70.62% |
| *mopex4* | 89.08% | *Insufficient* | 0.451681 | 0.309482 | +0.142198 | +0.132144 | 70.99% |
| *newzealand2* | 67.23% | *Insufficient* | 0.396316 | 0.044361 | +0.351955 | +0.276664 | 92.09% |
| **8-Model Median** | **90.83%** | **Unfiltered** | **0.342886** | **0.084131** | **+0.193436** | **+0.179792** | **77.30%** |

*Under all-8 unfiltered evaluation, excess displacement remains positive in 8/8 models (+0.1934 under Case 1, +0.1798 under Case 2). This confirms that the result is robust and not an artifact of the 90% eligibility threshold.*

---

## 5. Formal Verdict & Manuscript Recommendation

1. **Numeric Integrity:** `+0.227033` is **verified and mathematically exact** as the model-equal median of model-level differences ($E_{\text{Case 1}}$) across the 5 strictly eligible models.
2. **Manuscript Clarification:** To prevent any misunderstanding where a reader subtracts $0.329326 - 0.032632$, the manuscript text and tables should report:
   > *"In the five tested models meeting the strict 90% restart coverage requirement, out-of-bag parameter displacement strictly exceeded the archived IC multi-start calibration dispersion (model-equal median excess $D_{\text{cross, OOB}} - D_{\text{self}} = +0.227$ under model-level subtraction, and $+0.212$ under basin-matched subtraction; positive in 5/5 models)."*
