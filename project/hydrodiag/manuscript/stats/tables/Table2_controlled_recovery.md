# Table 2: Controlled Recovery of the Reference Outlet Gap

| Quantity | IC (Independent Calibration) | dPL (Differentiable Parameter Learning) |
| :--- | :---: | :---: |
| Denominator-valid catchments, N_valid (rate) | 427 (80.4%) | 460 (86.6%) |
| Reference outlet gap, D_b = KGE(CN) - KGE(Base_norefit) | +0.087 [+0.061, +0.106] | +0.091 [+0.070, +0.114] |
| Raw Base-refit gain, G_Base = KGE(Base_refit) - KGE(Base_norefit) | +0.0026 [+0.0008, +0.0077] | +0.0073 [+0.0039, +0.0099] |
| Raw TGD generic gain, G_TGD = KGE(TGD) - KGE(Base_norefit) | +0.0386 [+0.0254, +0.0498] | +0.0360 [+0.0268, +0.0446] |
| Recalibration gap-closure fraction, F_close = G_Base / D_b | 0.101 [0.080, 0.113] | 0.102 [0.095, 0.111] |
| Generic-control recovery fraction, F_TGD = G_TGD / D_b | 0.546 [0.519, 0.572] | 0.521 [0.505, 0.538] |
| Paired recovery-fraction difference, Δ F = F_TGD - F_close | +0.460 [+0.433, +0.498] | +0.441 [+0.412, +0.467] |
| Positive paired fraction, P(F_TGD > F_close) | 91.6% | 92.8% |

*Note*: Values report catchment-wise medians with marginal 95% bootstrap confidence intervals [2.5th, 97.5th percentiles] across denominator-valid catchments ($D_b = \mathrm{KGE}(\mathrm{CN}) - \mathrm{KGE}(\mathrm{Base}_{\mathrm{no\text{-}refit}}) > 10^{-6}$; 2,000 paired catchment resamples, seed 20260730). Here, $D_b$ represents the controlled reference outlet gap induced by the imposed snow-process omission. dPL values represent per-catchment seed medians across the three canonical training runs (seeds 42, 123, 2026). $N_{\mathrm{valid}}$ denotes catchments satisfying the R3 reference-outlet-gap denominator criterion ($D_b > 10^{-6}$) and is unrelated to the KGE-screened subsets in Sect. 3.1. $F_{\mathrm{close}} = G_{\mathrm{Base}} / D_b$ and $F_{\mathrm{TGD}} = G_{\mathrm{TGD}} / D_b$ are computed as catchment-wise ratios prior to population summarization and are not ratios of population medians. Similarly, $\Delta F$ is computed catchment-wise as $F_{\mathrm{TGD}} - F_{\mathrm{close}}$ prior to summarization; hence $\mathrm{median}(\Delta F) \neq \mathrm{median}(F_{\mathrm{TGD}}) - \mathrm{median}(F_{\mathrm{close}})$. $G_{\mathrm{Base}}$ and $G_{\mathrm{TGD}}$ denote primary raw paired KGE gains relative to uncalibrated structural knockout. For catchments where $D_b > 0$, the sign condition $F_{\mathrm{TGD}} > F_{\mathrm{close}}$ is algebraically equivalent to $G_{\mathrm{TGD}} > G_{\mathrm{Base}}$. Denominator sensitivity across alternative cutoffs is reported in Table S2 Panel B. IC and dPL represent parallel parameter-estimation regimes evaluated under identical sample selection.
