# F6 Representative-Model and Topology Provenance Audit

**Scope:** representative-model selection for F6a only. No F5 result, dPL training, recalibration, new statistical experiment, or figure drawing was performed.

**Audit basis:** current working-tree `dmotpy`/benchmark code and current R3 artifacts, cross-checked against the official MARRMoT paper and the Knoben et al. (2020) 36-model paper. The code is the authority for the present store and parameter definitions. MARRMoT names identify the parent conceptual structures; they do not imply numerical or solver identity with the original MARRMoT implementation.

## 1. Executive verdict

- **Store-based selection is reasonable for F6a, but only as an illustration.** The number of dynamic core state tensors gives a transparent low-to-high structural axis, and the official MARRMoT Figure 2 also organizes model structures by store count. It is not a scalar measure of complexity, process richness, or model quality.
- **Use one representative from every observed 1S--6S stratum if the visual can support a compact 3-by-2 layout.** The current registry has a real 6S stratum, not merely a theoretical possibility. Omitting it is not required for the F6 claim, but including it makes the displayed structural coverage honest and removes an avoidable “why was the maximum omitted?” reviewer objection.
- **Recommended set:** `IHACRES` (1S), `TOPMODEL` (2S), `VIC` (3S), `TANK` (4S), `HBV96` (5S), and `SMAR` (6S). This is **Scheme C**, extended to six compact topology panels. The choice is frozen from code provenance, state/flux interpretability, source-model recognizability, and parameter dimensionality—not from `A_m`, diagonal appearance, Top-k, or any F6 outcome.
- **Do not claim a store-count effect.** One model per stratum confounds stratum with model identity. F6a may show where specificity sits inside selected real topologies; F6b and the 35-model estimand carry the generality claim.
- If page constraints force only five main panels, retain the same 1S--5S representatives and place `SMAR` in a clearly labeled supplementary compact panel. Then describe the main visual as **“1S--5S representatives”**, not “the full structural range.”

## 2. Canonical 36-model store inventory

### 2.1 Registry facts

The current implementation has **36** registered runnable models, **271** canonical parameters, and **110** dynamic core state tensors. The state-stratum counts are:

| Current core store count | Number of models | Models |
|---:|---:|---|
| 1S | 5 | `collie1`, `collie2`, `ihacres`, `newzealand1`, `wetland` |
| 2S | 12 | `alpine1`, `alpine2`, `collie3`, `gr4j`, `hillslope`, `newzealand2`, `plateau`, `simhyd`, `susannah1`, `susannah2`, `topmodel`, `us1` |
| 3S | 5 | `australia`, `flexb`, `gsfb`, `penman`, `vic` |
| 4S | 5 | `flexi`, `mopex1`, `tank`, `tcm`, `xinanjiang` |
| 5S | 8 | `flexis`, `hbv96`, `hymod`, `modhydrolog`, `mopex2`, `mopex3`, `mopex4`, `mopex5` |
| 6S | 1 | `smar` |
| 7S--8S | 0 | none in this benchmark subset |

`n_stores` below means the number of state tensors returned by the current model's `create_initial_state()` and declared in `STATE_INFO`. Endpoint or intermediate unit-hydrograph convolutions are routing operators and are **not silently counted as extra conceptual storage states**. This distinction is essential for IHACRES, HBV96, SMAR, GR4J, and the Flex models.

### 2.2 Complete canonical inventory

The table is intentionally code-oriented. `parameter_names` is the ordered key sequence of the current `*_PARAMS_BOUNDS` dictionary, which is also the canonical dPL/IC parameter coordinate order. `R3_eligible` means eligible for the F6 `adv` contrast (`P_m >= 2` and a complete current IC--dPL correspondence matrix), not the separate R2 IC-self restart-coverage gate.

| model_name | MARRMoT_ID | source_model_name | n_stores | n_parameters | parameter_names | major_stores | major_fluxes | snow_module | R3_eligible | exclusion_reason | topology_source | parameter_source |
|---|---|---|---:|---:|---|---|---|:---:|:---:|---|---|---|
| `collie1` | `m_01_collie1_1p_1s` | Collie River 1 | 1 | 1 | `Smax` | S1 soil-moisture bucket | saturation excess; ET | no | **no** | `P=1`; no `q != p` off-diagonal exists | `dmotpy/models/core/collie1.py::create_initial_state,collie1_step`; Jothityangkoon et al. (2001) | `COLLIE1_PARAMS_BOUNDS/DESC` |
| `wetland` | `m_02_wetland_4p_1s` | Wetland / FLEX-Topo | 1 | 4 | `dw, betaw, swmax, kw` | S1 wetland/soil store | interception; saturation/excess runoff; ET; baseflow | no | yes | — | `core/wetland.py`; Savenije (2010) | `WETLAND_PARAMS_BOUNDS/DESC` |
| `collie2` | `m_03_collie2_4p_1s` | Collie River 2 | 1 | 4 | `Smax, Sfc_frac, a, M` | S1 soil-moisture bucket | saturation excess; bare/vegetation ET; interflow | no | yes | — | `core/collie2.py`; Jothityangkoon et al. (2001) | `COLLIE2_PARAMS_BOUNDS/DESC` |
| `newzealand1` | `m_04_newzealand1_6p_1s` | New Zealand 1 | 1 | 6 | `s1max, sfc_frac, m, a, b, tcbf` | S1 soil-moisture store | saturation excess; bare/vegetation ET; interflow; baseflow | no | yes | — | `core/newzealand1.py`; Atkinson et al. (2002) | `NEWZEALAND1_PARAMS_BOUNDS/DESC` |
| `ihacres` | `m_05_ihacres_6p_1s` | IHACRES | 1 | 6 | `lp, d, p, alpha, tau_q, tau_s` | S1 moisture-deficit store | deficit ET; nonlinear effective rainfall; overflow; fast/slow split; two exponential endpoint routing branches | no | yes | — | `core/ihacres.py`; `endpoint_uh_model.py`; Croke & Jakeman (2004) | `IHACRES_PARAMS_BOUNDS/DESC` |
| `alpine1` | `m_06_alpine1_4p_2s` | Alpine 1 | 2 | 4 | `tt, ddf, Smax, tc` | S1 snow; S2 soil moisture | snowfall/rainfall partition; melt; saturation excess; ET; baseflow | yes | yes | — | `core/alpine1.py`; Eder et al. (2003) | `ALPINE1_PARAMS_BOUNDS/DESC` |
| `gr4j` | `m_07_gr4j_4p_2s` | GR4J | 2 | 4 | `x1, x2, x3, x4` | S1 production store; S2 routing store | net rainfall/ET; percolation; inter-store exchange; dual UH branches; routing outflow | no | yes | — | `core/gr4j.py`; `gr4j_uh_model.py`; Perrin et al. (2003) lineage | `GR4J_PARAMS_BOUNDS/DESC` |
| `us1` | `m_08_us1_5p_2s` | US1 | 2 | 5 | `alpha_ei, m, smax, fc, alpha_ss` | S1 unsaturated; S2 saturated | interception; infiltration/saturation excess; vegetation/bare ET; baseflow | no | yes | — | `core/us1.py` | `US1_PARAMS_BOUNDS/DESC` |
| `susannah1` | `m_09_susannah1_6p_2s` | Susannah Brook 1 | 2 | 6 | `sb, sfc_frac, m, a, b, r` | S1 soil moisture; S2 groundwater | saturation excess; vegetation/bare ET; nonlinear interflow split; baseflow | no | yes | — | `core/susannah1.py`; Son & Sivapalan (2007) | `SUSANNAH1_PARAMS_BOUNDS/DESC` |
| `susannah2` | `m_10_susannah2_6p_2s` | Susannah Brook 2 | 2 | 6 | `sb, phi, fc, r, c, d` | S1 unsaturated; S2 saturated | recharge/excess; ET from both stores; subsurface flow; groundwater sink; saturation runoff | no | yes | — | `core/susannah2.py`; Son & Sivapalan (2007) | `SUSANNAH2_PARAMS_BOUNDS/DESC` |
| `collie3` | `m_11_collie3_6p_2s` | Collie River 3 | 2 | 6 | `smax, fc, a, m, b, lambda_par` | S1 soil moisture; S2 groundwater/recharge | saturation excess; vegetation/bare ET; nonlinear interflow; interflow split; groundwater release | no | yes | — | `core/collie3.py`; Jothityangkoon et al. (2001) lineage | `COLLIE_PARAMS_BOUNDS/DESC` |
| `alpine2` | `m_12_alpine2_6p_2s` | Alpine 2 | 2 | 6 | `tt, ddf, Smax, Cfc, tcin, tcbf` | S1 snow; S2 soil moisture | snowfall/rainfall; melt; saturation excess; ET; interflow; baseflow | yes | yes | — | `core/alpine2.py`; Eder et al. (2003) | `ALPINE2_PARAMS_BOUNDS/DESC` |
| `hillslope` | `m_13_hillslope_7p_2s` | Hillslope / FLEX-Topo | 2 | 7 | `dw, betaw, swmax, a, th, c_rad, kh` | S1 soil moisture; S2 groundwater | interception; saturation excess; surface/groundwater split; ET; capillary rise; baseflow; raw core leaves `th` as passthrough while the endpoint wrapper applies tri3 routing | no | yes | — | `core/hillslope.py`; `endpoint_uh_model.py`; Savenije (2010) | `HILLSLOPE_PARAMS_BOUNDS/DESC` |
| `topmodel` | `m_14_topmodel_7p_2s` | TOPMODEL | 2 | 7 | `suzmax, st, kd, q0, f, chi, phi` | S1 unsaturated storage; S2 saturated-zone deficit | topographic saturation runoff; saturation excess; ET; interflow/recharge; deficit-controlled baseflow | no | yes | — | `core/topmodel.py`; Beven et al. (1995); Beven & Freer (2001) | `TOPMODEL_PARAMS_BOUNDS/DESC` |
| `plateau` | `m_15_plateau_8p_2s` | Plateau | 2 | 8 | `fmax, dp, sumax, lp, p_coeff, tp, c_rise, kp` | S1 unsaturated; S2 saturated | interception; infiltration; capillary rise; ET; saturation excess; baseflow; endpoint surface/baseflow routing | no | yes | — | `core/plateau.py`; `endpoint_uh_model.py` | `PLATEAU_PARAMS_BOUNDS/DESC` |
| `newzealand2` | `m_16_newzealand2_8p_2s` | New Zealand 2 | 2 | 8 | `s1max, s2max, sfc_frac, m, a, b, tcbf, d_delay` | S1 interception; S2 soil moisture | interception/ET; saturation excess; vegetation/bare ET; interflow/baseflow; total-flow endpoint routing | no | yes | — | `core/newzealand2.py`; `endpoint_uh_model.py`; Atkinson et al. (2002) lineage | `NEWZEALAND2_PARAMS_BOUNDS/DESC` |
| `penman` | `m_17_penman_4p_3s` | Penman | 3 | 4 | `smax, phi, gam, k1` | S1 upper soil; S2 lower-zone deficit; S3 groundwater/routing | saturation excess; split; ET; recharge; baseflow | no | yes | — | `core/penman.py` | `PENMAN_PARAMS_BOUNDS/DESC` |
| `simhyd` | `m_18_simhyd_7p_2s` | SIMHYD current reference variant | 2 | 7 | `insc, coeff, sq, smsc, sub, crak, k` | soil; groundwater | same-day interception/ET; infiltration capacity; direct runoff; interflow; recharge; baseflow; current variant has no UH | no | yes | — | `core/simhyd.py`; current source explicitly documents the no-UH variant | `SIMHYD_PARAMS_BOUNDS/DESC` |
| `australia` | `m_19_australia_8p_3s` | Australia | 3 | 8 | `sb, phi, fc_frac, alpha_ss, beta_ss, k_deep, alpha_bf, beta_bf` | S1 unsaturated; S2 saturated; S3 groundwater | saturation/excess runoff; ET; interflow; deep recharge; nonlinear groundwater flow | no | yes | — | `core/australia.py`; Farmer et al. (2003) | `AUSTRALIA_PARAMS_BOUNDS/DESC` |
| `gsfb` | `m_20_gsfb_8p_3s` | GSFB | 3 | 8 | `c, ndc, smax, emax, frate, b, dpf, sdrmax` | S1 soil moisture; S2 intermediate; S3 saturated zone | recharge; saturation runoff; ET; interflow; baseflow; deep-percolation/release | no | yes | — | `core/gsfb.py` | `GSFB_PARAMS_BOUNDS/DESC` |
| `flexb` | `m_21_flexb_9p_3s` | FLEX-B | 3 | 9 | `s1max, beta, d_split, percmax, lp, nlagf, nlags, kf, ks` | S1 unsaturated soil; S2 fast routing; S3 slow routing | rainfall partition; soil recharge/percolation/ET; fast/slow split; two intermediate UH routing branches | no | yes | — | `core/flexb.py`; `intermediate_uh_model.py`; Savenije (2010) lineage | `FLEXB_PARAMS_BOUNDS/DESC` |
| `vic` | `m_22_vic_10p_3s` | VIC | 3 | 10 | `ibar, idelta, ishift, stot, fsm, b, k1, c1, k2, c2` | S1 interception; S2 soil moisture; S3 groundwater | seasonal interception phenology; interception excess; infiltration excess; ET; percolation; groundwater saturation excess; baseflow | no | yes | — | `core/vic.py`; Liang et al. (1994); current dynamic-DOY wrapper path | `VIC_PARAMS_BOUNDS/DESC` |
| `mopex1` | `m_24_mopex1_5p_4s` | MOPEX 1 | 4 | 5 | `s1max, tw, tu, se, tc` | S1 surface/root soil; S2 subsurface; S3 fast route; S4 slow route | saturation excess; ET; leakage/recharge; two routing/baseflow branches | no | yes | — | `core/mopex1.py`; Ye et al. (2012) lineage | `MOPEX1_PARAMS_BOUNDS/DESC` |
| `tcm` | `m_25_tcm_6p_4s` | TCM | 4 | 6 | `phi, rc, gam, k1, fa, k2` | S1 upper soil; S2 soil-moisture deficit; S3 fast route; S4 slow route | effective rainfall; split; saturation excess; ET; fast/slow baseflow | no | yes | — | `core/tcm.py`; `tcm_model.py` | `TCM_PARAMS_BOUNDS/DESC` |
| `flexi` | `m_26_flexi_10p_4s` | FLEX-I | 4 | 10 | `smax, beta, d_split, percmax, lp, nlagf, nlags, kf, ks, imax` | S1 interception; S2 soil moisture; S3 fast route; S4 slow route | interception/ET; soil recharge/percolation/ET; fast/slow split; two intermediate UH branches | no | yes | — | `core/flexi.py`; `intermediate_uh_model.py`; Savenije (2010) lineage | `FLEXI_PARAMS_BOUNDS/DESC` |
| `tank` | `m_27_tank_12p_4s` | TANK | 4 | 12 | `a0, b0, c0, a1, fa, fb, fc, fd, st, f2, f1, f3` | S1 top; S2 second; S3 third; S4 bottom tank | thresholded runoff holes; inter-tank drainage; sequential ET; bottom-tank baseflow | no | yes | — | `core/tank.py`; Sugawara (1995) | `TANK_PARAMS_BOUNDS/DESC` |
| `xinanjiang` | `m_28_xinanjiang_12p_4s` | Xinanjiang | 4 | 12 | `aim, par_a, par_b, stot, fwm, flm, par_c, ex, ki, kg, ci, cg` | S1 tension-water store; S2 free-water store; S3 interflow route; S4 groundwater route | impervious/pervious split; tension/free-water runoff; ET; interflow; baseflow | no | yes | — | `core/xinanjiang.py`; Zhao (1992) | `XINANJIANG_PARAMS_BOUNDS/DESC` |
| `hymod` | `m_29_hymod_5p_5s` | HYMOD | 5 | 5 | `smax, b_exp, a_split, kf, ks` | S1 soil moisture; S2--S4 fast routing cascade; S5 slow routing | excess runoff; ET; fast/slow split; fast and slow linear release | no | yes | — | `core/hymod.py`; source-model topology | `HYMOD_PARAMS_BOUNDS/DESC` |
| `mopex2` | `m_30_mopex2_7p_5s` | MOPEX 2 | 5 | 7 | `tcrit, ddf, s2max, tw, tu, se, tc` | S1 snow; S2 soil; S3 subsurface; S4 fast route; S5 slow route | snowfall/rainfall; melt; ET; saturation excess; recharge; two route branches | yes | yes | — | `core/mopex2.py`; Ye et al. (2012) lineage | `MOPEX2_PARAMS_BOUNDS/DESC` |
| `mopex3` | `m_31_mopex3_8p_5s` | MOPEX 3 | 5 | 8 | `tcrit, ddf, s2max, tw, tu, se, s3max, tc` | S1 snow; S2 soil; S3 subsurface; S4 fast route; S5 slow route | snow/melt; ET; saturation excess; recharge; added subsurface capacity; two route branches | yes | yes | — | `core/mopex3.py`; Ye et al. (2012) lineage | `MOPEX3_PARAMS_BOUNDS/DESC` |
| `mopex4` | `m_32_mopex4_10p_5s` | MOPEX 4 | 5 | 10 | `tcrit, ddf, s2max, tw, alpha, is_time, tu, se, s3max, tc` | S1 snow; S2 soil; S3 subsurface; S4 fast route; S5 slow route | snow/melt; interception; seasonal phase; ET; saturation excess; recharge; route branches | yes | yes | — | `core/mopex4.py`; `mopex_doy_model.py`; Ye et al. (2012) lineage | `MOPEX4_PARAMS_BOUNDS/DESC` |
| `mopex5` | `m_35_mopex5_12p_5s` | MOPEX 5 | 5 | 12 | `tcrit, ddf, s2max, tw, alpha, is_time, tmin, trange, tu, se, s3max, tc` | S1 snow; S2 soil; S3 subsurface; S4 fast route; S5 slow route | snow/melt; seasonal phenology; interception; ET; saturation excess; recharge; route branches | yes | yes | — | `core/mopex5.py`; `mopex_doy_model.py`; Ye et al. (2012) lineage | `MOPEX5_PARAMS_BOUNDS/DESC` |
| `modhydrolog` | `m_36_modhydrolog_15p_5s` | MODHYDROLOG | 5 | 15 | `insc, coeff, sq, smsc, sub, crak, em, dsc, ads, md, vcond, dlev, k1, k2, k3` | S1 interception; S2 soil; S3 depression; S4 groundwater; S5 river/channel | interception/ET; infiltration/interflow/recharge; depression trapping/ET/delayed infiltration; seepage; groundwater-river exchange; channel outflow | no | yes | — | `core/modhydrolog.py` | `MODHYDROLOG_PARAMS_BOUNDS/DESC` |
| `hbv96` | `m_37_hbv_15p_5s` | HBV-96 | 5 | 15 | `tt, tti, ttm, cfr, cfmax, whc, cflux, fc, lp, beta, k0, alpha, perc, k1, maxbas` | S1 snow-water; S2 liquid snow; S3 soil; S4 upper response zone; S5 lower response zone | snowfall/rainfall spectrum; melt/refreeze; capillary rise; ET; recharge/percolation; interflow; baseflow; endpoint MAXBAS routing | yes | yes | — | `core/hbv96.py`; `endpoint_uh_model.py`; Lindström et al. (1997) | `HBV96_PARAMS_BOUNDS/DESC` |
| `flexis` | `m_34_flexis_12p_5s` | FLEX-IS | 5 | 12 | `smax, beta, d_split, percmax, lp, nlagf, nlags, kf, ks, imax, tt, ddf` | S1 snow; S2 interception; S3 soil; S4 fast route; S5 slow route | snow/melt; interception/ET; soil recharge/percolation/ET; fast/slow split; intermediate UH branches | yes | yes | — | `core/flexis.py`; `intermediate_uh_model.py`; Savenije (2010) lineage | `FLEXIS_PARAMS_BOUNDS/DESC` |
| `smar` | `m_40_smar_8p_6s` | SMAR | 6 | 8 | `h_runoff, y_inf, smax, c_evap, g_rech, kg, n_res, nk_delay` | S1--S5 five soil-moisture layers; S6 groundwater | effective rainfall; layered infiltration/ET; recharge; groundwater release; endpoint gamma routing | no | yes | — | `core/smar.py`; `endpoint_uh_model.py` | `SMAR_PARAMS_BOUNDS/DESC` |

**Minor source-level warning:** the current `gr4j.py::create_initial_state` docstring is stale and lists Flex-IS states, while the function annotation, returned tuple, `STATE_INFO`, and GR4J wrapper all implement two core stores. The inventory follows the executable contract (`2S`). This is exactly why F6 topology drawings must use the current wrapper and state contract, not a copied model-name diagram.

### 2.3 MARRMoT versus current dmotpy implementation

The MARRMoT IDs in the current fresh registry artifact encode the same `p` and `s` counts as `NPARAM_INFO` and `STATE_INFO` for all 36 rows. Thus, for this subset, the suffix agreement is:

- `MARRMoT_ID ... _{P}p_{S}s` = current `n_parameters=P`, `n_stores=S`: **36/36**;
- current calibrated parameter total: **271**;
- current core state total: **110**.

This agreement is a registry-contract fact, not evidence of equation-level identity. The official MARRMoT paper explicitly warns that toolbox formulations can differ from original publications because of common framework conventions, smoothing, spatial simplification, and numerical solution choices. The present dmotpy implementation additionally uses PyTorch code, daily explicit stepping, and model-specific wrappers. Therefore F6 must draw `dmotpy`'s current states and flux paths.

## 3. F6 eligibility audit

### 3.1 What “35 eligible models” means for F6

For the F6 estimand

\[
adv_{m,p}=C_{m,p,p}-\operatorname{median}_{q\ne p}C_{m,p,q},
\]

an off-diagonal comparator exists only when `P_m >= 2`. The current identity table contains complete matrices for all 36 registry models, but:

- `collie1` has `P=1`, so its diagonal and Top-1 information are defined, while its `adv` and off-diagonal median are undefined;
- the other **35 models** have at least one alternative parameter and are estimable for `adv`;
- current `R3_PARAMETER_CROSS_CORRESPONDENCE_LONG.csv` therefore has **270 eligible parameter rows per feature space** after excluding `collie1`.

Thus the correct F6 wording is:

> **35 F6-eligible multi-parameter models; the full registry contains 36 models, including the one-parameter `collie1` model.**

The F6 eligibility exclusion is **not** a training-failure or IC-restart-coverage exclusion.

### 3.2 Do not confuse F6 eligibility with other R3 populations

The existing source products contain three distinct notions that must not be mixed:

1. **All36:** all 36 current models, including accepted `simhyd` generation 280.
2. **F6 contrast-valid set:** all36 minus `collie1` = 35 models, because `q != p` is required.
3. **`exclude_simhyd` sensitivity:** all36 minus `simhyd` = 35 models, used as a separate sensitivity population in the current R3 scripts.

The supplied `A_m \approx 0.615` and `35/35` positive-contrast wording matches the **35 valid contributors within all36**, not the `exclude_simhyd` population. In the current frozen table, the all36 contrast denominator is `35/36`; the exclude-SIMHYD contrast denominator is `34/35` because `collie1` remains one-parameter.

The separate R2 primary restart-coverage set is `23` models and is unrelated to F6's off-diagonal definition. F6a should not silently inherit the R2 23-model gate.

### 3.3 Existing numeric-estimand reconciliation

The current frozen R3 table reports all36 information-cluster `A_diag = 0.6150375940`, with 35 valid model contributors, and the parameter-label permutation null has mean `0.0015360902` and empirical `p=0.0009990010`. These agree with the stated F6 context at the stated precision.

The current cross table also shows why Top-k labels require care. Excluding `collie1`, a read-only recomputation of diagonal ranks over the 270 eligible parameter rows gives approximately:

- Top-1: `57.41%`;
- Top-2: `71.11%`;
- Top-3: `84.07%`.


Using the current eligible parameter-count vector (270 coordinates), the corresponding parameter-count-matched random expectations are `12.96%`, `25.93%`, and `38.89%` for Top-1/2/3. The supplied `13.3%/26.2%/39.1%` values are close but are not reproduced exactly by this current registry vector; if those values are retained, their exact denominator or eligibility snapshot must be cited. This is an estimand/provenance reconciliation, not a reason to select or exclude a representative model.
These are **pooled eligible-coordinate fractions**. The current model-equal table reports a different quantity: information-cluster Top-1/Top-3 medians of `62.5%/87.5%`; it does not report the same Top-2 estimand. The report and figure must label the chosen aggregation explicitly. This reconciliation does not select representatives and does not constitute a new experiment.

## 4. Candidate models by store stratum

The following is the complete **F6-eligible** candidate pool. All rows are named current MARRMoT/dmotpy structures; “named source” means the model has a recognized source-model identity, while “MARRMoT-family variant” flags a named but closely related family or benchmark variant. `Clarity` is a pre-outcome code-readability assessment based on whether the current code gives semantic state labels and an unambiguous principal-flow path. It is not based on any F6 statistic.

### 4.1 Candidate comparison

| stratum | candidate | MARRMoT ID | stores / P | process inventory in current code | identity and recognizability | clarity | special arrangement or caveat | F6a suitability |
|---:|---|---|---|---|---|:---:|---|---|
| 1S | `collie2` | `m_03_collie2_4p_1s` | 1 / 4 | soil bucket; saturation excess; bare/vegetation ET; interflow | named source model; recognizable | medium | state is executable S1 but not semantically named in initializer | valid but low-dimensional and less clear than IHACRES |
| 1S | `ihacres` | `m_05_ihacres_6p_1s` | 1 / 6 | deficit store; deficit ET; nonlinear effective rainfall; overflow; fast/slow split; two exponential endpoint UHs | named source model; high recognizability | **high** | endpoint routing is outside core store count; current code explicitly says S1 is a deficit store | **strong representative** |
| 1S | `newzealand1` | `m_04_newzealand1_6p_1s` | 1 / 6 | soil bucket; saturation excess; forest/bare ET; interflow; baseflow | named source model; medium-high recognizability | medium | initializer does not semantically label S1; simultaneous flux evaluation is explicitly documented | good backup |
| 1S | `wetland` | `m_02_wetland_4p_1s` | 1 / 4 | interception; saturation/excess runoff; ET; baseflow | named FLEX-Topo structure | medium-high | P=4; single bucket represents wetland/soil behavior | valid but lower dimensional |
| 2S | `alpine1` | `m_06_alpine1_4p_2s` | 2 / 4 | snow; soil; snowfall/rainfall; melt; saturation excess; ET; baseflow | named Alpine structure | high | P=4; overlaps snow emphasis later represented by HBV96 | valid but low-dimensional |
| 2S | `alpine2` | `m_12_alpine2_6p_2s` | 2 / 6 | snow; soil; melt; saturation excess; ET; interflow; baseflow | named Alpine structure | high | snow-rich; less topographically distinctive than TOPMODEL | good backup |
| 2S | `collie3` | `m_11_collie3_6p_2s` | 2 / 6 | soil; groundwater; saturation excess; nonlinear interflow; split; groundwater release | named Collie structure | medium | current initializer uses generic S1/S2 labels | valid backup |
| 2S | `gr4j` | `m_07_gr4j_4p_2s` | 2 / 4 | production store; routing store; percolation; exchange; dual UH routes | named source model; very recognizable | high | P=4; raw core docstring is stale; `x4` is active in `GR4JUHModel`, not raw core step | do not prefer for this hero |
| 2S | `hillslope` | `m_13_hillslope_7p_2s` | 2 / 7 | soil; groundwater; interception; saturation-excess split; ET; capillary rise; endpoint tri3 routing | named FLEX-Topo structure | medium | raw core step passes surface routing through unchanged; the full endpoint wrapper applies `th` via tri3, so the diagram must be wrapper-aware | valid backup but less self-contained than TOPMODEL |
| 2S | `newzealand2` | `m_16_newzealand2_8p_2s` | 2 / 8 | interception; soil; ET; saturation excess; interflow/baseflow; total-flow endpoint UH | named source model | high | endpoint delay is a route, not an extra store | valid backup |
| 2S | `plateau` | `m_15_plateau_8p_2s` | 2 / 8 | unsaturated; saturated; interception; infiltration; capillary rise; ET; saturation excess; baseflow | named source structure | high | endpoint UH splits surface/baseflow | valid backup |
| 2S | `simhyd` | `m_18_simhyd_7p_2s` | 2 / 7 | soil; groundwater; interception/ET; infiltration; direct runoff; interflow; recharge; baseflow | named current adapted variant | medium | current source explicitly removes the Gamma UH; accepted IC generation 280 is a benchmark exception | avoid as representative unless variant is foregrounded |
| 2S | `susannah1` | `m_09_susannah1_6p_2s` | 2 / 6 | soil; groundwater; saturation excess; ET; nonlinear interflow split; baseflow | named source model | high | groundwater split is clear but model is less widely recognized | valid backup |
| 2S | `susannah2` | `m_10_susannah2_6p_2s` | 2 / 6 | unsaturated; saturated; recharge/excess; ET; subsurface flow; groundwater sink | named source model | high | `qr` is an external groundwater sink rather than plotted streamflow | valid backup |
| 2S | `topmodel` | `m_14_topmodel_7p_2s` | 2 / 7 | unsaturated store; saturated-zone deficit; topographic saturation runoff; ET; interflow; deficit-controlled baseflow | named source model; high recognizability | **high** | current code uses S2 as a deficit store; draw this implementation, not a generic TOPMODEL cartoon | **strong representative** |
| 2S | `us1` | `m_08_us1_5p_2s` | 2 / 5 | unsaturated; saturated; interception; infiltration/saturation excess; vegetation/bare ET; baseflow | named source structure | medium | P=5; state/process naming is less compact for a small panel | valid but lower dimensional |
| 3S | `australia` | `m_19_australia_8p_3s` | 3 / 8 | unsaturated; saturated; groundwater; saturation/excess; ET; interflow; deep recharge; nonlinear groundwater flow | named source model | high | current code is clear but less recognizable than VIC | valid backup |
| 3S | `flexb` | `m_21_flexb_9p_3s` | 3 / 9 | unsaturated soil; fast route; slow route; partition; percolation; ET; two UH routes | named FLEX family variant | high | intermediate UH and close relation to FLEX-I/FLEX-IS; family redundancy | avoid if preserving cross-stratum family diversity |
| 3S | `gsfb` | `m_20_gsfb_8p_3s` | 3 / 8 | soil; intermediate; saturated; recharge; saturation runoff; ET; interflow; baseflow | named MARRMoT structure; medium recognizability | medium-high | state roles are present in the initializer but process nomenclature is less familiar | valid backup |
| 3S | `penman` | `m_17_penman_4p_3s` | 3 / 4 | upper soil; lower deficit; groundwater/routing; split; ET; recharge; baseflow | named source model | high | P=4; current benchmark has a separately documented historical warmup issue, not a topology issue | valid but low-dimensional |
| 3S | `vic` | `m_22_vic_10p_3s` | 3 / 10 | interception; soil; groundwater; phenology; infiltration excess; ET; percolation; groundwater baseflow | named source model; very high recognizability | **high** | current working-tree code uses dynamic DOY; seasonal interception is an explicit process | **strong representative** |
| 4S | `flexi` | `m_26_flexi_10p_4s` | 4 / 10 | interception; soil; fast/slow routing; percolation; ET; intermediate UHs | named FLEX family variant | high | close variant of selected/available Flex-IS family; family redundancy | valid backup |
| 4S | `mopex1` | `m_24_mopex1_5p_4s` | 4 / 5 | surface/root soil; subsurface; fast route; slow route; saturation excess; ET; leakage/recharge | named MOPEX family structure | medium-high | P=5; first member of a highly related MOPEX sequence | valid but lower dimensional |
| 4S | `tank` | `m_27_tank_12p_4s` | 4 / 12 | four vertically connected tanks; thresholded runoff holes; inter-tank drainage; ET; bottom baseflow | named source model; high recognizability | **high** | parameter symbols are terse (`a0`, `b0`, etc.) but every symbol has a current code definition and a direct tank/threshold role | **strong representative** |
| 4S | `tcm` | `m_25_tcm_6p_4s` | 4 / 6 | upper soil; deficit soil; fast route; slow route; effective rainfall; split; ET; baseflow | named MARRMoT structure | high | specialized `TCMModel` wrapper | valid backup |
| 4S | `xinanjiang` | `m_28_xinanjiang_12p_4s` | 4 / 12 | tension water; free water; interflow route; groundwater route; impervious split; ET; interflow/baseflow | named source model; high recognizability | medium-high | current initializer leaves S1--S4 generic; semantic roles are in step-section comments; dense 12-symbol display | strong alternative, but less drawable than TANK |
| 5S | `flexis` | `m_34_flexis_12p_5s` | 5 / 12 | snow; interception; soil; fast route; slow route; percolation; ET; two UH branches | named FLEX family variant | high | family extension with 5 explicit stores; redundant with other Flex candidates | valid backup |
| 5S | `hbv96` | `m_37_hbv_15p_5s` | 5 / 15 | snow water; liquid snow; soil; upper response; lower response; melt/refreeze; ET; recharge; interflow/baseflow; MAXBAS route | named source model; very high recognizability | **high** | 15 symbols; current step sections map all five stores; endpoint MAXBAS is not a sixth store | **strong representative** |
| 5S | `hymod` | `m_29_hymod_5p_5s` | 5 / 5 | soil; three fast routing stores; one slow routing store; excess; ET; split; fast/slow release | named source model; high recognizability | high | P=5 makes random Top-k baseline comparatively high | valid but low-dimensional |
| 5S | `modhydrolog` | `m_36_modhydrolog_15p_5s` | 5 / 15 | interception; soil; depression; groundwater; river/channel; infiltration/interflow/recharge; exchange; seepage; outflow | named source structure; medium recognizability | medium | very long process chain and generic S1--S5 initializer labels | valid but visually dense |
| 5S | `mopex2` | `m_30_mopex2_7p_5s` | 5 / 7 | snow; soil; subsurface; fast route; slow route; melt; ET; saturation; recharge | named MOPEX family structure | medium-high | sequential MOPEX family; P=7 | valid backup |
| 5S | `mopex3` | `m_31_mopex3_8p_5s` | 5 / 8 | snow; soil; subsurface; fast route; slow route; added subsurface capacity | named MOPEX family structure | medium-high | family redundancy; current initializer uses generic state symbols | valid backup |
| 5S | `mopex4` | `m_32_mopex4_10p_5s` | 5 / 10 | snow; soil; subsurface; fast/slow routes; interception; seasonal phase; ET | named MOPEX family structure | medium | specialized DOY wrapper and seasonal phase parameter | avoid as generic hero |
| 5S | `mopex5` | `m_35_mopex5_12p_5s` | 5 / 12 | snow; soil; subsurface; fast/slow routes; seasonal phenology; interception; ET | named MOPEX family structure | medium | specialized DOY wrapper; family redundancy and dense phase labels | avoid as generic hero |
| 6S | `smar` | `m_40_smar_8p_6s` | 6 / 8 | five soil-moisture layers; groundwater; layered infiltration/ET; recharge; groundwater release; gamma endpoint route | named source model; high recognizability among hydrologists | **high** | singleton stratum; five-layer soil stack is visually dense; endpoint gamma route is not an extra store | **include as compact sixth panel** |

### 4.2 Candidate implications

- The pool is not a monotone complexity ladder. For example, `HYMOD` has 5 stores but only 5 parameters, while 1S `IHACRES` has 6 parameters. `TANK` and `Xinanjiang` both have 4 stores and 12 parameters but very different topologies.
- `GR4J` and `hillslope` are reminders that a nominal parameter can be a routing parameter implemented in a wrapper or an identity route. Parameter count alone cannot certify topology clarity.
- `SMAR` is not a hidden seventh or eighth store: the current code explicitly returns five soil layers plus one groundwater state, and then applies endpoint routing.

## 5. Audit of previously proposed five models

```text
IHACRES: KEEP
TOPMODEL: KEEP
VIC: KEEP
Xinanjiang: REPLACE
HBV96: KEEP
```

### IHACRES — KEEP

Current code confirms **1 core store and 6 parameters**, not a zero- or two-store interpretation. The state is explicitly a moisture-deficit store. `tau_q` and `tau_s` are active endpoint exponential routing coordinates, so the figure can show a compact one-store production block with two labeled routing branches. It is not too low-dimensional for the frozen hero rule (`P=6`), and the deficit-store semantics are clearer than the other 1S candidates.

### TOPMODEL — KEEP

Current code confirms **2 core stores and 7 parameters**. The second state is explicitly a saturated-zone deficit, and the step function separately identifies topographic saturation runoff, unsaturated-zone storage, interflow, and deficit-controlled baseflow. The plot must use this current implementation, including its deficit-store sign convention and safety limits; it must not use a generic TOPMODEL network copied from another implementation.

### VIC — KEEP

Current code confirms **3 core stores and 10 parameters**: interception, soil moisture, and groundwater. The present working tree adds a `doy` argument and uses dynamic day-of-year phenology, so the topology provenance is current `vic.py`, not an older constant-phenology snapshot. It offers an interpretable interception-to-soil-to-groundwater path with enough coordinates for a nontrivial same-coordinate/alternative-coordinate display.

### Xinanjiang — REPLACE

The proposed count and dimension are correct: **4 stores and 12 parameters**. It remains a scientifically valid backup, not a wrong model. The replacement is a visual-provenance decision: the current initializer returns generic `S1`--`S4` labels and the semantic mapping is distributed through the step-function comments, whereas current `TANK` gives four explicitly named tanks and a direct code-defined role for every parameter (`a0`--`f3`). TANK therefore offers a more auditable one-to-one placement of `adv_{m,p}` around a real topology in the same area. This replacement is frozen independently of F6 association results.

### HBV96 — KEEP

Current code confirms **5 stores and 15 parameters**, including the explicit snowpack, soil, upper-response, lower-response, and `maxbas` route. The current `EndpointUHModel` applies MAXBAS routing after the five-state core. It is highly recognizable and provides a genuinely different five-store organization from the selected lower-store models. The panel will be dense, but the code supplies unusually clear process and parameter descriptions.

## 6. Recommended pre-frozen selection rule

The selection rule is frozen before using any F6 outcome and is based only on registry and topology provenance.

### 6.1 Deterministic algorithm

For each target stratum `s` in ascending order:

1. **Stratum filter:** retain models with `STATE_INFO[model] == s`.
2. **F6 estimability filter:** retain complete current relationship-matrix models with `P_m >= 2`. This excludes only `collie1` from the contrast-valid pool.
3. **Dimensionality safeguard for the hero:** retain models with `P_m >= 6`. This is a display/interpretation safeguard, not a redefinition of the F6 inferential denominator. Lower-dimensional eligible models remain in the candidate table and the 35-model ensemble.
4. **Implementation-integrity filter:** reject a candidate as the primary illustration if the full current model path leaves a principal displayed route unimplemented or identity-only. Wrapper-aware cases such as `hillslope` (raw core passthrough plus endpoint tri3) are retained only with the wrapper shown; `simhyd` remains a special no-UH current variant.
5. **Source-model and topology filter:** prefer a named source model over a family-only or benchmark-adapted variant when both are otherwise comparable; require semantic state/process labels sufficient to map every displayed parameter to a store, flux, threshold, or routing branch.
6. **Topology-diversity rule:** when candidates remain tied, prefer the candidate whose current store/flux organization is not a direct Flex/MOPEX expansion of another selected representative. This is a predeclared illustration-diversity rule, not an F6-result rule.
7. **Fixed tie-breaker:** if the code-based flags still tie, choose the smallest numeric MARRMoT ID. No `A_m`, `C[p,q]`, Top-k, spatial pattern, KGE, or figure appearance may enter any step.
For machine auditability, the qualitative filters are treated as frozen binary/codebook fields, not free-form judgment: `P_floor` (P>=6), `route_complete` (no principal unused/TODO path), `source_model` (independently named source model rather than an adapted benchmark variant), `semantic_states` (all stores named in the initializer or unambiguously in step sections), `topology_drawable` (principal paths and every parameter role can be placed without inventing a state), and `recognizability` (landmark source model versus named but less familiar MARRMoT variant). The lexicographic priority is `(P_floor, route_complete, source_model, semantic_states, topology_drawable, recognizability, topology_diversity, fixed MARRMoT-ID tie-break)`. `topology_diversity` is a predeclared categorical check against the already selected process organizations, not a score from any F6 table.

The selected decisions are therefore reproducible as codebook decisions: IHACRES wins 1S on explicit deficit-store and dual-route semantics; TOPMODEL wins 2S on explicit deficit/topographic-flow semantics that are distinct from the selected 1S deficit-routing organization; VIC wins 3S on explicit interception-soil-groundwater semantics and a complete current route; TANK wins 4S on four explicitly named tanks and direct parameter-to-hole/threshold roles; HBV96 wins 5S on explicit snow/soil/upper/lower response semantics and source-model recognizability; SMAR is the unique 6S candidate. The rule is now frozen independently of F6 outcomes and should be used unchanged for plotting.

The final code-readable priority is therefore:

```text
state stratum
-> complete multi-parameter F6 matrix
-> P >= 6 for the hero (not for ensemble eligibility)
-> no principal unused/TODO route
-> named source model and semantic state/flux mapping
-> avoid direct family redundancy
-> smallest MARRMoT number if still tied
```

This rule selects `IHACRES`, `TOPMODEL`, `VIC`, `TANK`, `HBV96`, and the singleton `SMAR`. For `SMAR`, step 1 leaves exactly one candidate; it is not selected by an F6 outcome.

### 6.2 Why this rule is not outcome-driven

The rule can be applied to the current registry, source comments, and wrapper declarations without opening `R3_DIAGONAL_OFFDIAGONAL_SUMMARY.csv` or inspecting any model-level F6 value. The only use of R3 assets is to establish matrix completeness and the formal `q != p` estimability condition. The association result cannot change the representative set under this rule.

## 7. Final representative set

| display order | store count | model | MARRMoT ID | parameter count | topology characteristics | reason for inclusion |
|---:|---:|---|---|---:|---|---|
| 1 | 1S | `IHACRES` | `m_05_ihacres_6p_1s` | 6 | one moisture-deficit production store; fast/slow runoff split; two exponential endpoint routes | clearest 1S semantics; enough coordinates; compact but not degenerate |
| 2 | 2S | `TOPMODEL` | `m_14_topmodel_7p_2s` | 7 | unsaturated storage plus saturated-zone deficit; topographic saturation runoff; interflow and deficit baseflow | recognizable, explicit current state semantics, distinctive non-snow topology |
| 3 | 3S | `VIC` | `m_22_vic_10p_3s` | 10 | seasonal interception; soil store; groundwater store; infiltration/percolation/baseflow | recognizable and process-rich with explicit current dynamic-DOY path |
| 4 | 4S | `TANK` | `m_27_tank_12p_4s` | 12 | four vertically connected tanks with threshold holes, drainage, ET, bottom flow | most directly drawable 4S topology; every parameter has a current tank/threshold role |
| 5 | 5S | `HBV96` | `m_37_hbv_15p_5s` | 15 | two snow states; soil; upper response; lower response; MAXBAS endpoint route | highly recognizable, explicit five-state process decomposition |
| 6 | 6S | `SMAR` | `m_40_smar_8p_6s` | 8 | five soil layers plus groundwater; layered ET/recharge; gamma endpoint route | observed maximum stratum; singleton is deterministically included, not cherry-picked |

## 8. Figure-density audit

### 8.1 Parameter-label load

The recommended six-panel set has:

```text
IHACRES  6
TOPMODEL 7
VIC     10
TANK    12
HBV96   15
SMAR     8
----------------
TOTAL   58 parameter symbols
```

The 1S--5S subset has 50 symbols. The extra 6S panel adds only 8 symbols but prevents the main display from implying that 5S is the observed upper limit.

### 8.2 Readability decision

Six full-text hydrology diagrams would be too dense. Six **compact topology panels** are feasible in a 3-by-2 double-column layout if the display is restricted to:

- all stores as consistently styled nodes;
- principal flow paths only;
- all canonical parameter symbols, placed next to the store/flux/route they control;
- a short shared legend for runoff, ET, recharge/interflow, baseflow, snow, and routing;
- one uniform `adv_{m,p}` marker/colour encoding, applied to every displayed parameter;
- no repeated long process prose inside each panel.

Do **not** draw every secondary flux name from the code. For example, the HBV96 panel should show snowpack, soil, upper zone, lower zone, melt/refreeze, recharge, interflow/baseflow, and MAXBAS, but not repeat every internal helper-function label. The same applies to SMAR's five-layer stack and MODHYDROLOG-like process complexity, although those are not selected representatives.

A single horizontal row of five or six topologies is not readable. A 2-by-3 or 3-by-2 small-multiple layout is required. With a 3-by-2 layout, adding SMAR fills the sixth position rather than creating a seventh visual object; this is why the six-panel recommendation is preferable to a five-panel row plus an ambiguous omitted-stratum note.

### 8.3 Scheme comparison

| scheme | scientific advantage | main risk | parameter/topology load | reviewer risk | verdict |
|---|---|---|---|---|---|
| **A: one representative for 1S--5S** | simple structural narrative; 50 parameter symbols; easy to state the target range | omits the actual 6S maximum; one model identity remains confounded with each stratum | feasible only as compact 2-by-3 with one unused slot or 1-by-5 at poor readability | “Why is SMAR omitted?” and possible overstatement of range | acceptable only if explicitly limited to 1S--5S |
| **B: three-model low/intermediate/high span** | clearest panel-level readability; permits a high-end `SMAR` choice | less transparent coverage of intermediate strata; still one identity per selected region; high/low definitions need freezing | about 20--30 symbols, depending on choices | can look more cherry-picked and can be mistaken for a structural sample | useful fallback, not preferred for this F6a objective |
| **C: compact one-per-stratum panels** | retains real coordinates, exposes topology, and makes selection auditable; six panels cover every observed stratum | must suppress secondary prose; still cannot infer a store-count effect | 58 symbols in a 3-by-2 grid; feasible with shared legend | manageable if caption states illustration-only and F6b carries generality | **recommended; use six-panel extension C6** |

## 9. Reviewer attack test

### 1. Cherry-picking?

**Defensible, conditional on freezing the rule above.** The complete candidate pool is recorded, source paths are named, low-dimensional and special-route caveats are visible, and the final set is selected from implementation provenance. For the prospective F6 workflow, freeze this rule before plotting or any result-based selection. `SMAR` is included because it is the only current 6S candidate, not because of its `A_m`.

### 2. Store count and model identity confounded?

**Yes.** One model per stratum cannot estimate a store effect separately from model identity. F6a is an illustration of parameter-coordinate placement in selected structures. The 35-model F6b ensemble—not the six panels—supports the cross-model generality statement.

### 3. Untested store-count effect implied?

**It would be if the figure used a trend line, stratum-wise effect comparison, or language such as “specificity increases with stores.”** The figure may order panels by store count for navigation only. No `store_count -> A_m` hypothesis, regression, monotone trend, or stratum significance claim is authorized without a separate predeclared analysis.

### 4. Low-dimensional models inflate same-coordinate ranking?

**Potentially.** `collie1` is algebraically non-comparable for `adv`; `P=4` and `P=5` models also have higher random Top-k baselines than 10--15 parameter models. The hero rule uses `P>=6`, but the full 35-model result retains all estimable models and reports the appropriate parameter-count-aware null. F6a must not be presented as a random-baseline-free sample.

### 5. Topology inconsistent with running code?

**This is a real risk and must be actively prevented.** Draw from `dmotpy/models/core/<model>.py`, `create_initial_state`, the current step function, and applicable wrappers. In particular:

- IHACRES: endpoint exponential routes are wrappers, not additional core stores;
- VIC: use current dynamic-DOY `vic.py`;
- HBV96: MAXBAS is endpoint routing, not a sixth state;
- SMAR: five soil layers plus groundwater is six states;
- TOPMODEL: S2 is a saturated-zone deficit;
- TANK: use the four current code tanks and parameter-to-hole/threshold mapping;
- do not reproduce the stale GR4J initializer docstring or a generic original-model cartoon.

### 6. Representative set shows only models supporting the conclusion?

**Not if the candidate table and rule are published internally and the 35-model panel remains the inferential panel.** The selection was not conditioned on `A_m`, Top-k, diagonal appearance, KGE, or spatial maps. The report also records valid alternatives (`Xinanjiang`, `Flex-I`, `Flex-IS`, `Alpine2`, etc.) and why they were not selected for the compact topology display.

## 10. Final recommendation for F6a

```text
F6a representative-model strategy:
- Selection principle:
  Current registry store stratum -> complete multi-parameter F6 matrix ->
  P >= 6 hero safeguard -> no principal unused/TODO route ->
  named source model with semantic current-code topology -> avoid direct family
  redundancy -> smallest MARRMoT ID only for residual ties.

- Store strata:
  1S, 2S, 3S, 4S, 5S, 6S.

- Models:
  IHACRES, TOPMODEL, VIC, TANK, HBV96, SMAR.

- Parameter counts:
  6, 7, 10, 12, 15, 8 (total 58 symbols).

- Display order:
  ascending current core store count: 1S -> 2S -> 3S -> 4S -> 5S -> 6S.

- Topology detail level:
  stores + principal flow paths + all parameter symbols; shared legend;
  no repeated secondary flux prose; wrappers shown as routing annotations,
  not extra store nodes.

- Specificity encoding:
  one uniform marker/colour scale for adv_{m,p}, shown for every canonical
  parameter coordinate in every panel; no parameter is selected because its
  F6 value is attractive.

- What F6a may claim:
  It is a hydrological illustration of where same-coordinate specificity is
  located within six real current-code conceptual topologies spanning every
  observed 1S--6S store stratum.

- What F6a may NOT claim:
  It is not a statistical sample of store counts; it does not estimate a
  store-count effect; it does not show that specificity is universal by virtue
  of six positive panels; it does not establish physical parameter identity,
  causal mechanism, or equation-level equivalence to original MARRMoT code.
```

**Caption safeguard:** use “six implementation-selected representatives ordered by current core-store count” rather than “representative models across structural complexity.” If SMAR is moved to the supplement, write “1S--5S representatives; the current registry also contains one 6S model (SMAR), shown separately,” and do not write “full structural range.”

## Provenance and reproducibility record

### Current-code sources

- `dmotpy/models/registry.py`: `PARAM_INFO`, `NPARAM_INFO`, `STATE_INFO`, `NUMBER_INFO`, `STFN_INFO`.
- `dmotpy/models/core/__init__.py`: current imports, parameter dictionaries, step functions, and state initializers.
- `dmotpy/models/core/<model>.py`: current state/flux topology and `*_PARAMS_BOUNDS/DESC`.
- `dmotpy/models/hydrology_model.py`: wrapper dispatch and current state count contract.
- `dmotpy/models/endpoint_uh_model.py`: endpoint routing schemes.
- `dmotpy/models/intermediate_uh_model.py` and `dmotpy/models/gr4j_uh_model.py`: intermediate/GR4J routing schemes.
- `dmotpy/models/mopex_doy_model.py` and `dmotpy/models/tcm_model.py`: specialized current wrappers.
- `project/benchmark/src/model_registry.py`: canonical benchmark parameter order and model construction.
- `project/benchmark/manuscript/si/fresh_validation_20260908/fresh_36model_registry_results.csv`: fresh 36-row registry contract artifact and MARRMoT ID construction.

### Current R3 sources

- `project/benchmark/manuscript/r3/tables/R3_PARAMETER_CROSS_CORRESPONDENCE_LONG.csv`: frozen `C[p,q]` matrices.
- `project/benchmark/manuscript/r3/tables/R3_DIAGONAL_OFFDIAGONAL_SUMMARY.csv`: all36 and exclude-SIMHYD population summaries.
- `project/benchmark/manuscript/r3/tables/R3_PARAMETER_LABEL_PERMUTATION_NULL.csv`: identity-destroying parameter-label null.
- `project/benchmark/manuscript/r3/scripts/08_parameter_identity_correspondence.py`: exact diagonal/off-diagonal definitions and the one-parameter handling.
- `project/benchmark/manuscript/r3/agent_B_coordinate_specificity.md`: independent R3 specificity audit and interpretation boundary.

### External sources

1. Knoben, W. J. M., Freer, J. E., Fowler, K. J. A., Peel, M. C., & Woods, R. A. (2020). *A Brief Analysis of Conceptual Model Structure Uncertainty Using 36 Models and 559 Catchments*. **Water Resources Research, 56**(9). DOI: [10.1029/2019WR025975](https://doi.org/10.1029/2019WR025975). The paper defines model structure in terms of states, fluxes, and equations and cautions against treating parameter count as a complete complexity measure.
2. Knoben, W. J. M., Freer, J. E., Fowler, K. J. A., Peel, M. C., & Woods, R. A. (2019). *Modular Assessment of Rainfall--Runoff Models Toolbox (MARRMoT) v1.2*. **Geoscientific Model Development, 12**, 2463--2480. DOI: [10.5194/gmd-12-2463-2019](https://doi.org/10.5194/gmd-12-2463-2019). Official Figure 2 sorts MARRMoT structures by number of stores and distinguishes process/store categories; the paper also documents differences between toolbox formulations and source publications.
3. Official MARRMoT repository: [github.com/wknoben/MARRMoT](https://github.com/wknoben/MARRMoT).
4. Original/source references used by the current files include Croke & Jakeman (IHACRES), Beven et al. (TOPMODEL), Liang et al. (VIC), Zhao (Xinanjiang), Lindström et al. (HBV-96), and Sugawara (TANK). The current code files remain authoritative for the exact F6 drawing topology.

### Source hashes at audit time

```text
8fe6f03fd6a4bafcd65886f0728977e89001b50dd6ff5a37944f1ba230e11141  dmotpy/models/core/__init__.py
5d07f259c2d2f6081e7efe1d24a99ba657390624f8b32980da132288484fb8b3  dmotpy/models/registry.py
12d2dde3623261d12d9b213bc5619ce15c39949726254a340ebe5b53fd19b573  project/benchmark/src/model_registry.py
44c5bb17eabdab62296a654b991cce189acd0eacd04fa04e09463b9aa3ca5522  dmotpy/models/hydrology_model.py
f3a59ed205c733fca0aa550c45ef56c4d6b6134319af7159a56b1e7b35dc3b72  dmotpy/models/endpoint_uh_model.py
ca8679c83f07de5555edebf9c6b730f4385a500b1d92e511cdb07a26bc0aba76  dmotpy/models/intermediate_uh_model.py
852523eef7e3482a8fe705c8e0a85265cf0f4ed604fe5acad6e30f31dfd42219  project/benchmark/manuscript/si/fresh_validation_20260908/fresh_36model_registry_results.csv
04abe56b4ae5c480a2b1e343246cfc6d142e4643890983eb35934acbbbebfe2c  project/benchmark/manuscript/r3/tables/R3_PARAMETER_CROSS_CORRESPONDENCE_LONG.csv
3bd5dddf52acdef7b33717a69e7d6936b15a05c53af6a19fea78d2e03231a032  project/benchmark/manuscript/r3/tables/R3_DIAGONAL_OFFDIAGONAL_SUMMARY.csv
916b37b45ce61b65090bd2fc2ededa46eedbf1993ef20cd1937ea4eb706acffb  project/benchmark/manuscript/r3/tables/R3_PARAMETER_LABEL_PERMUTATION_NULL.csv
```

**Audit status:** COMPLETE for the requested representative-model provenance and selection audit. No figure was drawn.
