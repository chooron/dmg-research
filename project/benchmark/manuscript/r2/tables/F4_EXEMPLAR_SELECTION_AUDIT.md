# R2 Figure 4 Exemplar-Selection Audit Report

F4 EXEMPLAR-SELECTION AUDIT = READY

## A. Data Completeness and Audit Integrity

- **Coordinates audited:** 271 / 271
- **Models covered:** 36 / 36 (23 strict primary models + 13 sensitivity models)
- **Coordinates with full basin-level paired data (531 catchments):** 271 / 271
- **Coordinates with usable top-1 prevalence:** 271 / 271
- **Coordinates with boundary/tie diagnostics:** 271 / 271
- **Single-parameter models:** 1 (`collie1`, single parameter `Smax`, flagged and excluded from non-trivial localization rankings)

---

## B. Candidate Rankings by Individual Criteria

Rankings are computed independently per criterion without opaque composite weighting formulas.

### Top 10 by Median Absolute Displacement ($M_{m,p}$)

| Rank | Model | Parameter | Strict? | Crit Value | Med Disp | Top-1 Prev | R_rank | Sign Cons | Boundary Flag |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `topmodel` | `q0` | No | 0.9904 | 0.9904 | 0.3635 | -0.0114 | 0.7495 | FLAGGED |
| 2 | `vic` | `ibar` | Yes | 0.9130 | 0.9130 | 0.4689 | 0.2468 | 0.7684 | FLAGGED |
| 3 | `susannah2` | `c` | No | 0.8972 | 0.8972 | 0.6554 | 0.3438 | 0.7269 | FLAGGED |
| 4 | `wetland` | `swmax` | Yes | 0.8676 | 0.8676 | 0.6045 | 0.5462 | 0.8211 | Clean |
| 5 | `plateau` | `tp` | No | 0.8655 | 0.8655 | 0.6121 | 0.0466 | 0.8192 | Clean |
| 6 | `mopex3` | `tu` | Yes | 0.6895 | 0.6895 | 0.2900 | 0.2310 | 0.8060 | Clean |
| 7 | `mopex4` | `s3max` | No | 0.6745 | 0.6745 | 0.1751 | 0.2989 | 0.8418 | Clean |
| 8 | `mopex3` | `tw` | Yes | 0.6695 | 0.6695 | 0.2599 | 0.2043 | 0.8776 | Clean |
| 9 | `topmodel` | `suzmax` | No | 0.6577 | 0.6577 | 0.0772 | 0.5845 | 0.9567 | Clean |
| 10 | `gr4j` | `x1` | Yes | 0.6407 | 0.6407 | 0.9096 | 0.7114 | 0.9868 | Clean |

### Top 10 by Basin-level Top-1 Prevalence ($f^{top1}_{m,p}$, non-single models)

| Rank | Model | Parameter | Strict? | Crit Value | Med Disp | Top-1 Prev | R_rank | Sign Cons | Boundary Flag |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `gr4j` | `x1` | Yes | 0.9096 | 0.6407 | 0.9096 | 0.7114 | 0.9868 | Clean |
| 2 | `alpine1` | `Smax` | Yes | 0.8945 | 0.5558 | 0.8945 | 0.8360 | 0.9058 | Clean |
| 3 | `collie2` | `Smax` | Yes | 0.6855 | 0.4065 | 0.6855 | -0.0135 | 0.5593 | FLAGGED |
| 4 | `simhyd` | `smsc` | Yes | 0.6761 | 0.5748 | 0.6761 | 0.6014 | 0.9812 | Clean |
| 5 | `susannah2` | `c` | No | 0.6554 | 0.8972 | 0.6554 | 0.3438 | 0.7269 | FLAGGED |
| 6 | `alpine2` | `Smax` | Yes | 0.6271 | 0.5353 | 0.6271 | 0.5213 | 0.8644 | Clean |
| 7 | `plateau` | `tp` | No | 0.6121 | 0.8655 | 0.6121 | 0.0466 | 0.8192 | Clean |
| 8 | `wetland` | `swmax` | Yes | 0.6045 | 0.8676 | 0.6045 | 0.5462 | 0.8211 | Clean |
| 9 | `newzealand1` | `s1max` | Yes | 0.5706 | 0.5681 | 0.5706 | 0.7023 | 0.8456 | Clean |
| 10 | `hymod` | `smax` | Yes | 0.5593 | 0.6112 | 0.5593 | 0.4647 | 0.9887 | Clean |

### Top 10 by Sign Consistency (Directional shift, $M_{m,p} \ge 0.10$)

| Rank | Model | Parameter | Strict? | Crit Value | Med Disp | Top-1 Prev | R_rank | Sign Cons | Boundary Flag |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `hymod` | `smax` | Yes | 0.9887 | 0.6112 | 0.5593 | 0.4647 | 0.9887 | Clean |
| 2 | `gr4j` | `x1` | Yes | 0.9868 | 0.6407 | 0.9096 | 0.7114 | 0.9868 | Clean |
| 3 | `smar` | `smax` | No | 0.9868 | 0.4976 | 0.3465 | 0.5440 | 0.9868 | Clean |
| 4 | `simhyd` | `smsc` | Yes | 0.9812 | 0.5748 | 0.6761 | 0.6014 | 0.9812 | Clean |
| 5 | `hillslope` | `swmax` | Yes | 0.9793 | 0.5950 | 0.3559 | 0.5511 | 0.9793 | Clean |
| 6 | `plateau` | `sumax` | No | 0.9755 | 0.6158 | 0.1864 | 0.6535 | 0.9755 | Clean |
| 7 | `mopex2` | `s2max` | Yes | 0.9755 | 0.5676 | 0.4972 | 0.6726 | 0.9755 | Clean |
| 8 | `hbv96` | `fc` | No | 0.9586 | 0.6126 | 0.1450 | 0.6056 | 0.9586 | Clean |
| 9 | `topmodel` | `suzmax` | No | 0.9567 | 0.6577 | 0.0772 | 0.5845 | 0.9567 | Clean |
| 10 | `ihacres` | `d` | Yes | 0.9567 | 0.5674 | 0.4030 | 0.6040 | 0.9567 | Clean |

### Top 10 by High Rank Preservation among High-Displacement ($M_{m,p} \ge 0.20$)

| Rank | Model | Parameter | Strict? | Crit Value | Med Disp | Top-1 Prev | R_rank | Sign Cons | Boundary Flag |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `collie1` | `Smax` | Yes | 0.9068 | 0.5790 | 1.0000 | 0.9068 | 0.8851 | Clean |
| 2 | `alpine1` | `Smax` | Yes | 0.8360 | 0.5558 | 0.8945 | 0.8360 | 0.9058 | Clean |
| 3 | `vic` | `stot` | Yes | 0.8344 | 0.5636 | 0.1525 | 0.8344 | 0.8945 | Clean |
| 4 | `gr4j` | `x3` | Yes | 0.7234 | 0.3574 | 0.0716 | 0.7234 | 0.8493 | Clean |
| 5 | `gr4j` | `x1` | Yes | 0.7114 | 0.6407 | 0.9096 | 0.7114 | 0.9868 | Clean |
| 6 | `newzealand1` | `s1max` | Yes | 0.7023 | 0.5681 | 0.5706 | 0.7023 | 0.8456 | Clean |
| 7 | `ihacres` | `lp` | Yes | 0.6940 | 0.4458 | 0.1337 | 0.6940 | 0.7778 | Clean |
| 8 | `mopex2` | `s2max` | Yes | 0.6726 | 0.5676 | 0.4972 | 0.6726 | 0.9755 | Clean |
| 9 | `plateau` | `sumax` | No | 0.6535 | 0.6158 | 0.1864 | 0.6535 | 0.9755 | Clean |
| 10 | `flexis` | `smax` | Yes | 0.6317 | 0.6264 | 0.3051 | 0.6317 | 0.8230 | Clean |

### Top 10 by Low Rank Preservation (Reorganization) among High-Displacement ($M_{m,p} \ge 0.20$)

| Rank | Model | Parameter | Strict? | Crit Value | Med Disp | Top-1 Prev | R_rank | Sign Cons | Boundary Flag |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `tcm` | `phi` | No | -0.1317 | 0.4804 | 0.3164 | -0.1317 | 0.8211 | FLAGGED |
| 2 | `newzealand2` | `s1max` | No | -0.0546 | 0.3210 | 0.3258 | -0.0546 | 0.7721 | Clean |
| 3 | `mopex3` | `s3max` | Yes | -0.0360 | 0.4941 | 0.0697 | -0.0360 | 0.8399 | Clean |
| 4 | `mopex5` | `tw` | No | -0.0327 | 0.4864 | 0.2825 | -0.0327 | 0.8136 | Clean |
| 5 | `vic` | `ishift` | Yes | -0.0309 | 0.6259 | 0.0734 | -0.0309 | 0.8456 | Clean |
| 6 | `gsfb` | `sdrmax` | No | -0.0196 | 0.4150 | 0.3296 | -0.0196 | 0.6685 | Clean |
| 7 | `flexi` | `imax` | Yes | -0.0161 | 0.2834 | 0.3051 | -0.0161 | 0.7815 | Clean |
| 8 | `collie2` | `Smax` | Yes | -0.0135 | 0.4065 | 0.6855 | -0.0135 | 0.5593 | FLAGGED |
| 9 | `topmodel` | `q0` | No | -0.0114 | 0.9904 | 0.3635 | -0.0114 | 0.7495 | FLAGGED |
| 10 | `mopex5` | `s2max` | No | 0.0119 | 0.5243 | 0.0452 | 0.0119 | 0.6309 | Clean |

### Top 10 by Spread Compression (Smallest $\text{IQR}_{\text{dPL}} / \text{IQR}_{\text{IC}}$, $M_{m,p} \ge 0.05$)

| Rank | Model | Parameter | Strict? | Crit Value | Med Disp | Top-1 Prev | R_rank | Sign Cons | Boundary Flag |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `newzealand2` | `s1max` | No | 0.0000 | 0.3210 | 0.3258 | -0.0546 | 0.7721 | Clean |
| 2 | `australia` | `beta_bf` | No | 0.0000 | 0.0875 | 0.1318 | 0.2348 | 0.8682 | Clean |
| 3 | `topmodel` | `q0` | No | 0.0001 | 0.9904 | 0.3635 | -0.0114 | 0.7495 | FLAGGED |
| 4 | `hbv96` | `cfr` | No | 0.0033 | 0.1587 | 0.2881 | 0.1024 | 0.8060 | Clean |
| 5 | `vic` | `ibar` | Yes | 0.0081 | 0.9130 | 0.4689 | 0.2468 | 0.7684 | FLAGGED |
| 6 | `flexi` | `imax` | Yes | 0.0084 | 0.2834 | 0.3051 | -0.0161 | 0.7815 | Clean |
| 7 | `mopex5` | `tw` | No | 0.0099 | 0.4864 | 0.2825 | -0.0327 | 0.8136 | Clean |
| 8 | `mopex3` | `tw` | Yes | 0.0147 | 0.6695 | 0.2599 | 0.2043 | 0.8776 | Clean |
| 9 | `susannah2` | `c` | No | 0.0241 | 0.8972 | 0.6554 | 0.3438 | 0.7269 | FLAGGED |
| 10 | `tank` | `f3` | Yes | 0.0361 | 0.3242 | 0.2580 | 0.0754 | 0.7533 | Clean |

### Top 10 by Spread Expansion (Largest $\text{IQR}_{\text{dPL}} / \text{IQR}_{\text{IC}}$, $M_{m,p} \ge 0.05$)

| Rank | Model | Parameter | Strict? | Crit Value | Med Disp | Top-1 Prev | R_rank | Sign Cons | Boundary Flag |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `smar` | `nk_delay` | No | 10.3682 | 0.0984 | 0.0113 | 0.8108 | 0.8945 | FLAGGED |
| 2 | `hbv96` | `maxbas` | No | 9.0738 | 0.0555 | 0.0000 | 0.6434 | 0.8757 | Clean |
| 3 | `wetland` | `betaw` | Yes | 8.5023 | 0.3886 | 0.2731 | 0.3501 | 0.7928 | Clean |
| 4 | `mopex3` | `tu` | Yes | 4.2102 | 0.6895 | 0.2900 | 0.2310 | 0.8060 | Clean |
| 5 | `mopex4` | `tu` | No | 3.6065 | 0.1353 | 0.1563 | 0.1888 | 0.6704 | Clean |
| 6 | `mopex5` | `tu` | No | 2.6576 | 0.0546 | 0.0885 | -0.0354 | 0.6911 | Clean |
| 7 | `alpine1` | `Smax` | Yes | 2.5691 | 0.5558 | 0.8945 | 0.8360 | 0.9058 | Clean |
| 8 | `hillslope` | `swmax` | Yes | 2.4772 | 0.5950 | 0.3559 | 0.5511 | 0.9793 | Clean |
| 9 | `mopex1` | `s1max` | Yes | 2.0785 | 0.5339 | 0.3220 | 0.3726 | 0.9303 | Clean |
| 10 | `australia` | `alpha_ss` | No | 1.7837 | 0.0821 | 0.0151 | 0.4937 | 0.8154 | Clean |

---

## C. Broad Stratified Candidate Pool

The candidate pool contains **28 model-parameter coordinates** across **19 distinct hydrological models**, divided across 6 scientifically defined response strata.

| ID | Stratum | Model | Parameter | Strict? | Med Disp | Top-1 | R_rank | Sign Cons | IQR Ratio | Why Candidate | Caveat |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|
| `C01` | Stratum A (Recurrent displacement-dominant) | `gr4j` | `x1` | Yes | 0.641 | 0.910 | 0.711 | 0.987 | 0.804 | Highest non-degenerate top1 prevalence (91.0%) with large displacement (0.641) and high rank preservation (R=0.711) | 2-parameter model with simple structure; dominant across almost all catchments |
| `C02` | Stratum A (Recurrent displacement-dominant) | `alpine1` | `Smax` | Yes | 0.556 | 0.895 | 0.836 | 0.906 | 2.569 | Top1 prevalence of 89.5% in 4-parameter model with high displacement (0.556) and strong rank preservation (R=0.836) | Strongly concentrated response on single storage parameter |
| `C03` | Stratum A (Recurrent displacement-dominant) | `simhyd` | `smsc` | Yes | 0.575 | 0.676 | 0.601 | 0.981 | 1.316 | Top1 prevalence of 67.6% in 7-parameter model, displacement 0.575, clean unclipped distribution | 7-parameter model; secondary displacement spread across other soil parameters |
| `C04` | Stratum A (Recurrent displacement-dominant) | `wetland` | `swmax` | Yes | 0.868 | 0.605 | 0.546 | 0.821 | 0.149 | Top1 prevalence of 60.5% with very high displacement (0.868) in 4-parameter model | Upper bound displacement is large but unclipped |
| `C05` | Stratum A (Recurrent displacement-dominant) | `newzealand1` | `s1max` | Yes | 0.568 | 0.571 | 0.702 | 0.846 | 0.492 | Top1 prevalence of 57.1% with displacement 0.568 and high rank preservation (R=0.702) | Moderate tie fraction at lower bounds |
| `C06` | Stratum B (High displacement + high rank) | `vic` | `stot` | Yes | 0.564 | 0.153 | 0.834 | 0.895 | 0.602 | Large displacement (0.564) with exceptionally high rank preservation (R=0.834) in 10-parameter model | 10-parameter model with complex internal routing; other VIC coordinates behave differently |
| `C07` | Stratum B (High displacement + high rank) | `ihacres` | `lp` | Yes | 0.446 | 0.134 | 0.694 | 0.778 | 0.266 | Large displacement (0.446) with high rank preservation (R=0.694) in 6-parameter model | Moderate top1 prevalence (22.2%) due to competition with parameter d |
| `C08` | Stratum B (High displacement + high rank) | `mopex2` | `s2max` | Yes | 0.568 | 0.497 | 0.673 | 0.976 | 1.307 | High displacement (0.568) with strong rank preservation (R=0.673) and high sign consistency (97.6%) | Coexists with secondary parameter se undergoing moderate rank reorganization |
| `C09` | Stratum B (High displacement + high rank) | `flexis` | `smax` | Yes | 0.626 | 0.305 | 0.632 | 0.823 | 0.273 | High displacement (0.626) and substantial rank preservation (R=0.632) in 13-parameter model | 13-parameter model; top1 prevalence is shared (36.9%) |
| `C10` | Stratum B (High displacement + high rank) | `plateau` | `sumax` | No | 0.616 | 0.186 | 0.653 | 0.976 | 0.539 | High displacement (0.616) and high rank preservation (R=0.653) in non-strict 8-parameter model | Non-strict model (insufficient reference); plateau model has separate uncalibrated routing |
| `C11` | Stratum C (High displacement + weak/negative rank) | `mopex3` | `s3max` | Yes | 0.494 | 0.070 | -0.036 | 0.840 | 1.463 | Substantial displacement (0.494) with near-zero/negative rank correlation (R=-0.036) without boundary clipping | Pure cross-basin spatial rank scrambling across all 531 basins |
| `C12` | Stratum C (High displacement + weak/negative rank) | `vic` | `ishift` | Yes | 0.626 | 0.073 | -0.031 | 0.846 | 0.088 | High displacement (0.626) with negative rank preservation (R=-0.031) within same model as stot (R=0.834) | Clean contrast within VIC showing coordinate-dependent rank behavior |
| `C13` | Stratum C (High displacement + weak/negative rank) | `flexi` | `imax` | Yes | 0.283 | 0.305 | -0.016 | 0.782 | 0.008 | Displacement 0.283 with negative rank correlation (R=-0.016) in 10-parameter strict model | Displacement magnitude is moderate (0.283) rather than extreme |
| `C14` | Stratum C (High displacement + weak/negative rank) | `modhydrolog` | `k3` | Yes | 0.388 | 0.196 | 0.016 | 0.537 | 0.491 | Displacement 0.388 with near-zero rank preservation (R=0.016) in 15-parameter model | 15-parameter model; displacement is distributed across several routing terms |
| `C15` | Stratum C (High displacement + weak/negative rank) | `newzealand2` | `s1max` | No | 0.321 | 0.326 | -0.055 | 0.772 | 0.000 | Displacement 0.321 with negative rank correlation (R=-0.055) in non-strict 8-parameter model | Non-strict model; contrast with newzealand1 (s1max R=0.702) |
| `C16` | Stratum D (Coherent directional shift) | `hymod` | `smax` | Yes | 0.611 | 0.559 | 0.465 | 0.989 | 0.942 | Highest sign consistency among strict models (98.9% positive shift, median shift = +0.611) | Moderate rank preservation (R=0.465); almost unanimous positive remapping |
| `C17` | Stratum D (Coherent directional shift) | `hillslope` | `swmax` | Yes | 0.595 | 0.356 | 0.551 | 0.979 | 2.477 | 97.9% positive shift (median shift = +0.595) with large displacement (0.595) | Rank correlation is moderate (R=0.551) |
| `C18` | Stratum D (Coherent directional shift) | `ihacres` | `d` | Yes | 0.567 | 0.403 | 0.604 | 0.957 | 0.494 | 95.7% positive shift (median shift = +0.567) with displacement 0.567 and R=0.606 | Shares displacement with parameter lp in IHACRES |
| `C19` | Stratum D (Coherent directional shift) | `mopex1` | `s1max` | Yes | 0.534 | 0.322 | 0.373 | 0.930 | 2.079 | 93.0% positive shift (median shift = +0.534) in 5-parameter strict model | Rank preservation is moderate (R=0.373) |
| `C20` | Stratum D (Coherent directional shift) | `smar` | `smax` | No | 0.498 | 0.347 | 0.544 | 0.987 | 1.098 | 98.7% positive shift (median shift = +0.498) in non-strict 8-parameter model | Non-strict model; demonstrates cross-model consistency of capacity upward shift |
| `C21` | Stratum E (Distribution spread rescaling) | `vic` | `ibar` | Yes | 0.913 | 0.469 | 0.247 | 0.768 | 0.008 | Severe across-catchment compression (IQR_ratio = 0.027, IQR_IC=0.395 -> IQR_dPL=0.011) with displacement 0.913 | Boundary flag is TRUE (upper bound occupancy 63.8% at IC collapsing to single value under dPL) |
| `C22` | Stratum E (Distribution spread rescaling) | `tank` | `f3` | Yes | 0.324 | 0.258 | 0.075 | 0.753 | 0.036 | Substantial compression (IQR_ratio = 0.198, IQR_IC=0.749 -> IQR_dPL=0.148) with displacement 0.324 and no boundary clipping | 12-parameter model with moderate individual displacement weights |
| `C23` | Stratum E (Distribution spread rescaling) | `mopex3` | `tu` | Yes | 0.690 | 0.290 | 0.231 | 0.806 | 4.210 | Strong spread expansion (IQR_ratio = 2.278, IQR_IC=0.286 -> IQR_dPL=0.651) with displacement 0.690 | Upper and lower tails expand symmetrically across catchments |
| `C24` | Stratum E (Distribution spread rescaling) | `wetland` | `betaw` | Yes | 0.389 | 0.273 | 0.350 | 0.793 | 8.502 | Spread expansion (IQR_ratio = 2.685, IQR_IC=0.174 -> IQR_dPL=0.467) with displacement 0.389 | Rank preservation is moderate (R=0.457) |
| `C25` | Stratum F (Low-response contrast) | `gr4j` | `x2` | Yes | 0.005 | 0.000 | 0.833 | 0.620 | 0.711 | Near-zero displacement (0.005) and high rank preservation (R=0.833) within model where x1 moves 0.641 | 2-parameter model provides starkest possible high vs low contrast |
| `C26` | Stratum F (Low-response contrast) | `alpine1` | `tc` | Yes | 0.012 | 0.002 | 0.903 | 0.685 | 0.912 | Minimal displacement (0.012) and very high rank preservation (R=0.903) within model where Smax moves 0.556 | Provides clean 1-to-1 contrast for alpine1 storage vs routing |
| `C27` | Stratum F (Low-response contrast) | `simhyd` | `sq` | Yes | 0.055 | 0.015 | 0.424 | 0.620 | 0.333 | Low displacement (0.010) in 7-parameter model where smsc moves 0.575 | Routing parameter with modest sensitivity |
| `C28` | Stratum F (Low-response contrast) | `hymod` | `b_exp` | Yes | 0.008 | 0.015 | 0.803 | 0.693 | 0.792 | Minimal displacement (0.008) and high rank preservation (R=0.803) in model where smax moves 0.611 | Shape parameter remains closely aligned with IC across all catchments |

---

## D. Within-Model Contrast Opportunities

Identified **10 candidate pairs** within multi-parameter models, providing direct visual contrast of intra-model coordinate response heterogeneity.

| Pair ID | Model | Coord A | Coord B | A Role | B Role | Disp (A / B) | R_rank (A / B) | Scientific Contrast |
|:---|:---|:---|:---|:---|:---|:---:|:---:|:---|
| `P01` | `alpine1` | `Smax` | `tc` | displacement-dominant storage (top1=0.895, disp=0.556, R=0.836) | low-response routing contrast (top1=0.002, disp=0.012, R=0.903) | 0.556 / 0.012 | 0.836 / 0.903 | High-response storage vs invariant routing; both exhibit strong cross-basin rank preservation |
| `P02` | `gr4j` | `x1` | `x2` | displacement-dominant production store (top1=0.910, disp=0.641, R=0.711) | zero-response water exchange parameter (top1=0.000, disp=0.005, R=0.833) | 0.641 / 0.005 | 0.711 / 0.833 | Classic 2-parameter model: complete localization where one coordinate absorbs 91% of top-1 displacement |
| `P03` | `vic` | `stot` | `ishift` | high displacement with high rank preservation (disp=0.564, R=0.834) | high displacement with rank reorganization (disp=0.626, R=-0.031) | 0.564 / 0.626 | 0.834 / -0.031 | Internal contrast within a 10-parameter model showing that large displacement can either preserve or scramble spatial ordering |
| `P04` | `flexi` | `smax` | `imax` | high displacement with rank preservation (disp=0.613, R=0.598) | moderate displacement with rank reorganization (disp=0.283, R=-0.016) | 0.613 / 0.283 | 0.598 / -0.016 | 10-parameter flexible model contrasting rank-preserving storage shift vs rank-inverting interception capacity |
| `P05` | `hymod` | `smax` | `b_exp` | coherent upward shift (sc=0.989, disp=0.611, top1=0.557) | low-response shape parameter (disp=0.008, R=0.803) | 0.611 / 0.008 | 0.465 / 0.803 | Demonstrates near-unanimous positive capacity shift paired with invariant distribution shape |
| `P06` | `simhyd` | `smsc` | `sq` | displacement-dominant soil moisture store (top1=0.676, disp=0.575, R=0.599) | low-response infiltration capacity (top1=0.000, disp=0.010, R=0.485) | 0.575 / 0.055 | 0.601 / 0.424 | 7-parameter model illustrating clear partitioning between active storage and inactive routing |
| `P07` | `mopex2` | `s2max` | `se` | high displacement + rank preservation (disp=0.568, R=0.673, top1=0.550) | moderate displacement + weak rank preservation (disp=0.330, R=0.204, top1=0.177) | 0.568 / 0.330 | 0.673 / 0.204 | Two moderately active coordinates within same model showing divergent rank preservation levels |
| `P08` | `mopex3` | `tu` | `s3max` | large displacement with spread expansion (disp=0.690, IQR_ratio=2.278, R=0.231) | large displacement with complete rank loss (disp=0.494, IQR_ratio=1.171, R=-0.036) | 0.690 / 0.494 | 0.231 / -0.036 | Both coordinates experience large displacement, but one expands distribution spread while the other completely reorganizes catchment rankings |
| `P09` | `newzealand1` | `s1max` | `tcbf` | dominant storage displacement (top1=0.571, disp=0.568, R=0.702) | near-zero baseflow routing response (top1=0.000, disp=0.001, R=0.659) | 0.568 / 0.001 | 0.702 / 0.659 | 6-parameter strict model with stark magnitude separation across conceptual functions |
| `P10` | `wetland` | `swmax` | `dw` | dominant wetland capacity displacement (top1=0.605, disp=0.868, R=0.546) | low-response wetland drainage exponent (top1=0.002, disp=0.019, R=0.397) | 0.868 / 0.019 | 0.546 / 0.397 | Clear within-model contrast in specialized wetland hydrology structure |

---

## E. Methodological Caveats and Quality Control

1. **Single-Parameter Model Degeneracy:** `collie1` has $n_{params}=1$, which forces $C_{eff}=1.0$ and $f^{top1}=1.0$ algebraically. It is recorded in the complete table but barred from localization rankings.
2. **Boundary Saturation & Ties:** Coordinates such as `vic:ibar` (upper bound occupancy 63.8% at IC), `topmodel:q0` (lower bound occupancy 76.6%), and `collie2:Smax` (upper bound occupancy 58.0%) exhibit strong clipping. Sensitivity to threshold selection (0.10 vs 0.20 occupancy) identifies 126 vs 76 flagged coordinates.
3. **Rank Loss vs Tie-Artifacts:** For most low-$R_{rank}$ coordinates (e.g. `mopex3:s3max`, `vic:ishift`, `flexi:imax`), unique value counts exceed 500 in both IC and dPL, proving that low rank preservation is genuine spatial reordering rather than a tie-induced numerical artifact.
4. **IQR Ratio Division Safeguards:** In 14 coordinates, $\text{IQR}_{\text{IC}} < 0.01$ due to tight initial parameter clusters. These are flagged to prevent division by near-zero variance.
5. **Descriptive Interpretation Boundary:** Exemplar coordinates illustrate statistical response modes (displacement magnitude, directional consistency, rank reordering, distribution spread); they do not constitute claims of parameter sensitivity, physical dominance, or real-world process identity.

---

## F. Evidence-Balanced Design Options for Final Figure 4

To support human selection without imposing a single figure layout, three evidence-grounded configuration options are presented:

### Option 1: 4 High-Response Coordinates from 4 Diverse Models (Panels A-D)
- **Goal:** Showcase the four clearest distinct mathematical behaviors across separate hydrological model structures.
- `C01` (`gr4j:x1`): Recurrent displacement dominance ($f^{top1}=0.910$, $M=0.641$, $R=0.711$)
- `C02` (`alpine1:Smax`): Large displacement with strong rank preservation ($M=0.556$, $R=0.836$)
- `C11` (`mopex3:s3max`): Large displacement with complete rank loss ($M=0.494$, $R=-0.036$)
- `C16` (`hymod:smax`): Near-unanimous directional shift ($sc=0.989$, $\Delta=+0.611$)

### Option 2: 3 Models x (High-Response + Low-Response) Pairs (6 Panels)
- **Goal:** Directly demonstrate within-model response heterogeneity and disprove the assumption that all parameters shift uniformly.
- **Pair 1 (`P01` `alpine1`):** `Smax` (disp=0.556, top1=0.895) vs `tc` (disp=0.012, top1=0.002)
- **Pair 2 (`P02` `gr4j`):** `x1` (disp=0.641, top1=0.910) vs `x2` (disp=0.005, top1=0.000)
- **Pair 3 (`P03` `vic`):** `stot` (disp=0.564, R=0.834) vs `ishift` (disp=0.626, R=-0.031)

### Option 3: 6 Archetype Exemplars Across the Spectrum (6 Panels)
- **Goal:** Comprehensive panel covering all six diagnostic strata.
- **Stratum A (`C03` `simhyd:smsc`):** Clean unclipped recurrent top-1 contributor
- **Stratum B (`C06` `vic:stot`):** High-displacement rank-preserving shift
- **Stratum C (`C13` `flexi:imax`):** High-displacement rank-reorganizing response
- **Stratum D (`C17` `hillslope:swmax`):** 98% coherent positive capacity shift
- **Stratum E (`C23` `mopex3:tu`):** Spread expansion across catchments
- **Stratum F (`C26` `alpine1:tc`):** Invariant low-response contrast baseline

---

*No figures were generated during this audit.*
