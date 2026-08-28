# R2 Legacy Definitions and Deprecation Note

This document formally records deprecated R2 metrics, obsolete estimates, and conflicting definitions that must NOT be used in the manuscript, figures, or supplement.

---

## 1. Deprecated Prevalence Definitions

| Deprecated Item | Obsolete Value / Definition | Canonical Replacement | Reason for Deprecation |
| :--- | :--- | :--- | :--- |
| **Legacy IC S1/S5 Prevalence** | IC S1 ≈ 49.7%, S5 ≈ 81.8% (single start or unpooled baseline) | **IC S1 = 46.7% [39.4%, 54.3%], S5 = 98.2% [94.5%, 100.0%]** | Early draft used single restart subset without 10-restart cross/within pooling. Canonical Estimand 4B pools all 45 within-Base, 45 within-CN, and 100 cross-pair distances per basin. |
| **Legacy dPL S1/S5 Prevalence** | dPL S1 ≈ 80.0%, S5 = 100% (zero/fixed within threshold) | **dPL S1 = 70.3% [63.6%, 77.0%], S5 = 87.3% [78.2%, 94.5%]** | Early draft evaluated against zero within baseline. Canonical Estimand 4B rigorously compares against each basin's specific 3-seed pooled within-dispersion. |
| **Fixed Scalar Baseline** | Evaluating $D > 0.08$ | **Basin-specific $w_{\text{pooled}, b} = (w_{\text{base}, b} + w_{\text{cn}, b})/2$** | Arbitrary fixed threshold distorts cross-catchment comparisons. |

---

## 2. Deprecated Interpretations

| Obsolete Interpretation | Correct Canonical Stance |
| :--- | :--- |
| **"CN–TGD differentiation is driven solely by S5 high-snow catchments"** | **DEPRECATED.** Paired $\delta\text{\_excess}$ proves differentiation emerges at intermediate snow activity (S2: $+0.0287$, S3: $+0.0239$) and persists into high snow activity. S5 represents an asymptotic saturation plateau, not the origin of differentiation. |
| **"dPL excess exhibits global linear growth across all frac_snow"** | **DEPRECATED.** dPL exhibits an ordered but nonlinear response that plateaus across S3–S5. Linear regression is descriptive only. |
| **"Parameters um, ki, ci replace snow processes"** | **DEPRECATED / PROHIBITED.** No parameter directly substitutes for snow physics. These are recurring directional shifts in the shared calibrated parameter space. |
| **"IC vs dPL comparison ranks method superiority"** | **DEPRECATED / PROHIBITED.** IC (per-basin CMA-ES) and dPL (regionalized MLP) represent two distinct parameter-estimation constraint regimes and are analyzed in parallel. |
